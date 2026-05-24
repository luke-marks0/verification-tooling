#!/usr/bin/env python3
"""Integration test for userspace TCP server.

Run on the DO droplet as root:
    python3 -m tests.integration.test_userspace_tcp

This script:
1. Starts the userspace TCP server in a background thread
2. Sends HTTP requests via curl to the server
3. Validates the response content
4. Captures packets to verify deterministic segmentation
5. Runs the same request twice and compares packet captures
"""
from __future__ import annotations

import json
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from modules.network.networkdet.userspace_tcp_server import (
    FULL_RESPONSE,
    UserspaceServer,
    get_gateway_mac,
    get_interface_info,
)


def get_public_ip() -> str:
    """Get the public IP of this machine."""
    result = subprocess.run(
        ["curl", "-s", "http://ifconfig.me"], capture_output=True, text=True, timeout=10
    )
    return result.stdout.strip()


def find_interface() -> str:
    """Find the primary network interface."""
    result = subprocess.run(
        ["ip", "route", "show", "default"], capture_output=True, text=True
    )
    parts = result.stdout.split()
    for i, p in enumerate(parts):
        if p == "dev" and i + 1 < len(parts):
            return parts[i + 1]
    return "eth0"


def start_tcpdump(interface: str, port: int, outfile: str) -> subprocess.Popen:
    """Start tcpdump capturing on the given port."""
    proc = subprocess.Popen(
        [
            "tcpdump", "-i", interface, "-nn", "-XX",
            f"tcp port {port}",
            "-w", outfile,
            "-c", "50",  # Capture at most 50 packets
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(1)  # Give tcpdump time to start
    return proc


def read_pcap_packets(pcapfile: str) -> list[bytes]:
    """Read raw packets from a pcap file."""
    packets = []
    try:
        with open(pcapfile, "rb") as f:
            # Read pcap global header (24 bytes)
            ghdr = f.read(24)
            if len(ghdr) < 24:
                return packets

            magic = struct.unpack("<I", ghdr[0:4])[0]
            if magic == 0xa1b2c3d4:
                endian = "<"
            elif magic == 0xd4c3b2a1:
                endian = ">"
            else:
                return packets

            while True:
                # Packet header: ts_sec(4) + ts_usec(4) + incl_len(4) + orig_len(4)
                phdr = f.read(16)
                if len(phdr) < 16:
                    break
                incl_len = struct.unpack(f"{endian}I", phdr[8:12])[0]
                data = f.read(incl_len)
                if len(data) < incl_len:
                    break
                packets.append(data)
    except FileNotFoundError:
        pass
    return packets


def filter_server_data_packets(packets: list[bytes], server_port: int) -> list[bytes]:
    """Filter for TCP data packets from the server (our userspace stack)."""
    data_packets = []
    for pkt in packets:
        if len(pkt) < 54:  # Eth(14) + IP(20) + TCP(20)
            continue
        ethertype = struct.unpack("!H", pkt[12:14])[0]
        if ethertype != 0x0800:
            continue
        ip_start = 14
        protocol = pkt[ip_start + 9]
        if protocol != 6:
            continue
        ip_header_len = (pkt[ip_start] & 0x0F) * 4
        tcp_start = ip_start + ip_header_len
        src_port = struct.unpack("!H", pkt[tcp_start:tcp_start + 2])[0]
        if src_port != server_port:
            continue
        tcp_data_offset = ((pkt[tcp_start + 12] >> 4) & 0x0F) * 4
        flags = pkt[tcp_start + 13]
        # Only data packets (ACK or PSH|ACK with payload)
        ip_total_len = struct.unpack("!H", pkt[ip_start + 2 : ip_start + 4])[0]
        payload_len = ip_total_len - ip_header_len - tcp_data_offset
        if payload_len > 0:
            data_packets.append(pkt)
    return data_packets


def test_basic_response(interface: str, port: int, local_ip: str) -> bool:
    """Test that curl gets the correct response."""
    print(f"\n{'='*60}")
    print("TEST 1: Basic HTTP response via userspace TCP")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            [
                "curl", "-s", "--max-time", "10",
                "--connect-timeout", "5",
                f"http://{local_ip}:{port}/deterministic",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        print(f"curl exit code: {result.returncode}")
        print(f"curl stdout: {result.stdout[:200]}")
        if result.stderr:
            print(f"curl stderr: {result.stderr[:200]}")

        if result.returncode != 0:
            print("FAIL: curl returned non-zero")
            return False

        # Parse the response body (curl gets just the body)
        try:
            body = json.loads(result.stdout)
            print(f"Parsed response: {json.dumps(body, indent=2)}")
            if body.get("server") == "userspace-tcp":
                print("PASS: Response received from userspace TCP server")
                return True
            else:
                print(f"FAIL: Unexpected response: {body}")
                return False
        except json.JSONDecodeError:
            print(f"FAIL: Response is not valid JSON: {result.stdout[:100]}")
            return False

    except subprocess.TimeoutExpired:
        print("FAIL: curl timed out")
        return False


def test_deterministic_segmentation(interface: str, port: int, local_ip: str, mss: int) -> bool:
    """Test that two requests produce identical segmentation."""
    print(f"\n{'='*60}")
    print("TEST 2: Deterministic segmentation (two runs)")
    print(f"{'='*60}")

    results = []
    for run in range(2):
        pcap_file = f"/tmp/userspace_tcp_run{run}.pcap"
        # Remove old pcap
        try:
            os.unlink(pcap_file)
        except FileNotFoundError:
            pass

        # Start tcpdump
        tcpdump = start_tcpdump(interface, port, pcap_file)

        # Make the request
        time.sleep(0.5)
        try:
            result = subprocess.run(
                [
                    "curl", "-s", "--max-time", "10",
                    "--connect-timeout", "5",
                    f"http://{local_ip}:{port}/deterministic",
                ],
                capture_output=True,
                timeout=15,
            )
        except subprocess.TimeoutExpired:
            print(f"FAIL: curl timed out on run {run}")
            tcpdump.terminate()
            return False

        # Wait for tcpdump to finish capturing
        time.sleep(2)
        tcpdump.terminate()
        tcpdump.wait()

        # Read the pcap
        packets = read_pcap_packets(pcap_file)
        data_pkts = filter_server_data_packets(packets, port)
        print(f"Run {run}: {len(packets)} total packets, {len(data_pkts)} server data packets")

        # Extract just the TCP payload lengths (segmentation pattern)
        seg_sizes = []
        for pkt in data_pkts:
            ip_start = 14
            ip_header_len = (pkt[ip_start] & 0x0F) * 4
            ip_total_len = struct.unpack("!H", pkt[ip_start + 2 : ip_start + 4])[0]
            tcp_start = ip_start + ip_header_len
            tcp_data_offset = ((pkt[tcp_start + 12] >> 4) & 0x0F) * 4
            payload_len = ip_total_len - ip_header_len - tcp_data_offset
            seg_sizes.append(payload_len)

        print(f"Run {run}: Segment sizes: {seg_sizes}")
        results.append(seg_sizes)

        time.sleep(1)  # Brief pause between runs

    if len(results) == 2 and results[0] == results[1]:
        print(f"PASS: Segmentation pattern is identical across runs: {results[0]}")
        # Verify MSS boundary
        for size in results[0][:-1]:  # All but last should be exactly MSS
            if size != mss and len(results[0]) > 1:
                print(f"  WARNING: Non-MSS segment size {size} (expected {mss})")
        return True
    else:
        print(f"FAIL: Segmentation differs between runs")
        print(f"  Run 0: {results[0] if results else 'empty'}")
        print(f"  Run 1: {results[1] if len(results) > 1 else 'empty'}")
        return False


def main():
    if os.geteuid() != 0:
        print("ERROR: Must run as root (need AF_PACKET)")
        sys.exit(1)

    PORT = 9999
    MSS = 1460
    RUN_ID = "userspace-poc-test"

    interface = find_interface()
    local_ip, local_mac = get_interface_info(interface)
    print(f"Interface: {interface}")
    print(f"Local IP: {local_ip}, MAC: {local_mac}")

    try:
        gateway_mac = get_gateway_mac(interface)
        print(f"Gateway MAC: {gateway_mac}")
    except RuntimeError as e:
        print(f"WARNING: {e}")
        print("Will use broadcast MAC for testing")
        gateway_mac = "ff:ff:ff:ff:ff:ff"

    # Start the server in a background thread
    server = UserspaceServer(
        interface=interface,
        port=PORT,
        local_ip=local_ip,
        local_mac=local_mac,
        gateway_mac=gateway_mac,
        mss=MSS,
        run_id=RUN_ID,
    )

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"\nServer started on {local_ip}:{PORT}")
    time.sleep(2)  # Let server initialize

    # Run tests
    passed = 0
    total = 0

    total += 1
    if test_basic_response(interface, PORT, local_ip):
        passed += 1

    time.sleep(2)  # Wait between tests

    total += 1
    if test_deterministic_segmentation(interface, PORT, local_ip, MSS):
        passed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} tests passed")
    print(f"Server stats: {server.stats}")
    print(f"{'='*60}")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
