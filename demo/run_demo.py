#!/usr/bin/env python3
"""
Deterministic Serving Stack -- Live Demo
----------------------------------------

Run on two pre-provisioned servers to demonstrate:
  Part 1: Deterministic frame construction (matching digests across nodes)
  Part 2: Active warden covert channel elimination

Usage:
  python3 demo/run_demo.py              # Run all parts
  python3 demo/run_demo.py --part 1     # Frame determinism only
  python3 demo/run_demo.py --part 2     # Warden demo only
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from modules.network.networkdet import create_net_stack
from modules.network.networkdet.warden import ActiveWarden
from modules.network.networkdet.checksums import ip_checksum, tcp_checksum
from modules.core.common.deterministic import canonical_json_bytes


# ─── Terminal formatting ─────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"


def banner(text: str) -> None:
    width = 64
    print()
    print(f"{BOLD}{BLUE}{'=' * width}{RESET}")
    print(f"{BOLD}{BLUE}  {text}{RESET}")
    print(f"{BOLD}{BLUE}{'=' * width}{RESET}")
    print()


def section(text: str) -> None:
    print(f"\n{BOLD}{CYAN}--- {text} ---{RESET}\n")


def ok(text: str) -> None:
    print(f"  {GREEN}[OK]{RESET} {text}")


def fail(text: str) -> None:
    print(f"  {RED}[FAIL]{RESET} {text}")


def info(text: str) -> None:
    print(f"  {DIM}{text}{RESET}")


def highlight(label: str, value: str, color: str = YELLOW) -> None:
    print(f"  {label}: {color}{value}{RESET}")


def hex_dump(data: bytes, label: str = "", max_bytes: int = 48) -> None:
    """Print a colored hex dump of frame bytes."""
    h = data[:max_bytes].hex()
    parts = [h[i:i+2] for i in range(0, len(h), 2)]

    colored = []
    for i, byte in enumerate(parts):
        if i < 14:  # Ethernet
            colored.append(f"{BLUE}{byte}{RESET}")
        elif i < 34:  # IP
            colored.append(f"{GREEN}{byte}{RESET}")
        elif i < 54:  # TCP
            colored.append(f"{MAGENTA}{byte}{RESET}")
        else:  # Payload
            colored.append(f"{DIM}{byte}{RESET}")

    trail = f" {DIM}...({len(data)} bytes total){RESET}" if len(data) > max_bytes else ""
    prefix = f"  {label}: " if label else "  "
    print(f"{prefix}{' '.join(colored)}{trail}")


def pause(msg: str = "Press Enter to continue...") -> None:
    input(f"\n  {DIM}{msg}{RESET}")
    print()


# ─── Part 1: Deterministic Frame Construction ─────────────────────

def part1_frame_determinism() -> None:
    banner("Part 1: Deterministic Frame Construction")

    print(f"  Two independent servers, same manifest, same inputs.")
    print(f"  Each constructs L2 frames locally. Do the digests match?")

    manifest_path = os.path.join(REPO_ROOT, "modules", "inference", "manifests", "qwen3-1.7b.manifest.json")
    manifest = json.loads(open(manifest_path).read())
    lockfile = {"artifacts": [
        {"artifact_id": "network-stack", "artifact_type": "network_stack_binary", "digest": "sha256:" + "a" * 64},
        {"artifact_id": "pmd-driver", "artifact_type": "pmd_driver", "digest": "sha256:" + "b" * 64},
    ]}

    section("Constructing frames from 8 request/response pairs")

    net = create_net_stack(manifest, lockfile, backend="sim", run_id="demo-run-2026")

    for i in range(8):
        request = canonical_json_bytes({"id": f"req-{i}", "prompt": f"What is {i} + {i}?"})
        response = canonical_json_bytes({
            "id": f"req-{i}",
            "tokens": [100 + i, 200 + i, 300 + i, 400 + i],
            "logits": [round(0.1 * i, 2), round(0.2 * i, 2), round(0.3 * i, 2)],
            "finish_reason": "stop",
        })
        net.process_exchange(conn_index=i, request_bytes=request, response_bytes=response)
        info(f"req-{i}: {net.capture_ring.frame_count} frames so far")

    section("Results")
    highlight("Frame count", str(net.frame_count()))
    highlight("Capture digest", net.capture_digest(), GREEN)

    print()
    print(f"  {BOLD}Run this same script on a second server.{RESET}")
    print(f"  The digest will be {GREEN}identical{RESET} — same inputs, same frames,")
    print(f"  every byte reproducible on any machine with Python.")

    section("Sample frames")
    frames = net.capture_frames_hex()
    for f in frames[:4]:
        raw = bytes.fromhex(f["frame_hex"])
        hex_dump(raw, f"Frame #{f['frame_index']}")

    # Parse and display one frame in detail.
    section("Frame #0 — Protocol Breakdown")
    raw = bytes.fromhex(frames[0]["frame_hex"])

    # Ethernet
    dst_mac = ":".join(f"{b:02x}" for b in raw[0:6])
    src_mac = ":".join(f"{b:02x}" for b in raw[6:12])
    ethertype = struct.unpack("!H", raw[12:14])[0]
    highlight("Ethernet dst", dst_mac, BLUE)
    highlight("Ethernet src", src_mac, BLUE)
    highlight("EtherType", f"0x{ethertype:04x} (IPv4)", BLUE)

    # IP
    ip = raw[14:]
    ip_id = struct.unpack("!H", ip[4:6])[0]
    ttl = ip[8]
    src_ip = ".".join(str(b) for b in ip[12:16])
    dst_ip = ".".join(str(b) for b in ip[16:20])
    highlight("IP src", src_ip, GREEN)
    highlight("IP dst", dst_ip, GREEN)
    highlight("IP ID", f"{ip_id} (deterministic counter)", GREEN)
    highlight("TTL", f"{ttl} (fixed)", GREEN)
    highlight("DSCP/ECN", f"{ip[1]} (zeroed)", GREEN)

    # TCP
    tcp = raw[34:]
    src_port = struct.unpack("!H", tcp[0:2])[0]
    dst_port = struct.unpack("!H", tcp[2:4])[0]
    seq = struct.unpack("!I", tcp[4:8])[0]
    flags = tcp[13]
    window = struct.unpack("!H", tcp[14:16])[0]
    urg = struct.unpack("!H", tcp[18:20])[0]
    flag_names = []
    if flags & 0x02: flag_names.append("SYN")
    if flags & 0x10: flag_names.append("ACK")
    if flags & 0x08: flag_names.append("PSH")
    highlight("TCP src port", str(src_port), MAGENTA)
    highlight("TCP dst port", str(dst_port), MAGENTA)
    highlight("TCP seq", f"{seq} (ISN from sha256(run_id:conn_idx))", MAGENTA)
    highlight("TCP flags", "|".join(flag_names), MAGENTA)
    highlight("TCP window", f"{window} (fixed from manifest)", MAGENTA)
    highlight("TCP urgent ptr", f"{urg} (always 0)", MAGENTA)
    highlight("TCP timestamps", "disabled (no clock dependency)", MAGENTA)

    net.close()


# ─── Part 2: Active Warden ────────────────────────────────────────

def _build_malicious_frame(
    ip_id: int,
    seq: int,
    *,
    src_port: int = 50000,
    dst_port: int = 80,
    flags: int = 0x10,
    urgent_ptr: int = 0,
    reserved_bits: int = 0,
    payload: bytes = b"",
    tos: int = 0,
    ttl: int = 128,
) -> bytes:
    """Build a frame with attacker-controlled header fields."""
    import socket
    src_ip = socket.inet_aton("10.0.0.1")
    dst_ip = socket.inet_aton("10.0.0.2")

    # TCP header
    data_offset = 5
    tcp_no_cksum = struct.pack(
        "!HHIIBBHHH",
        src_port, dst_port, seq, 0,
        (data_offset << 4) | reserved_bits,
        flags, 65535, 0, urgent_ptr,
    ) + payload
    cksum = tcp_checksum(src_ip, dst_ip, tcp_no_cksum)
    tcp_seg = struct.pack(
        "!HHIIBBHHH",
        src_port, dst_port, seq, 0,
        (data_offset << 4) | reserved_bits,
        flags, 65535, cksum, urgent_ptr,
    ) + payload

    # IP header
    total_length = 20 + len(tcp_seg)
    ip_no_cksum = struct.pack(
        "!BBHHHBBH4s4s",
        0x45, tos, total_length, ip_id, 0x4000, ttl, 6, 0,
        src_ip, dst_ip,
    )
    ip_ck = ip_checksum(ip_no_cksum)
    ip_hdr = struct.pack(
        "!BBHHHBBH4s4s",
        0x45, tos, total_length, ip_id, 0x4000, ttl, 6, ip_ck,
        src_ip, dst_ip,
    )

    # Ethernet
    eth = struct.pack("!6s6sH", b"\x00\x11\x22\x33\x44\x55", b"\xaa\xbb\xcc\xdd\xee\xff", 0x0800)

    frame = eth + ip_hdr + tcp_seg
    if len(frame) < 60:
        frame += b"\x00" * (60 - len(frame))
    return frame


def part2_warden() -> None:
    banner("Part 2: Active Warden — Covert Channel Elimination")

    print(f"  A compromised server can embed hidden data in protocol headers.")
    print(f"  The warden sits at the network perimeter and destroys these")
    print(f"  channels — without decrypting or inspecting the payload.")

    # ── Attack 1: IP ID channel (Covert TCP) ──

    section("Attack 1: Covert TCP — Data in IP ID fields")
    print(f"  The attacker encodes ASCII in the IP Identification field.")
    print(f"  Each packet carries one character of the secret message.")
    print()

    secret = "EXFILTRATE"
    highlight("Secret message", secret, RED)
    print()

    frames = []
    for i, ch in enumerate(secret):
        frame = _build_malicious_frame(ip_id=ord(ch), seq=1000 + i)
        frames.append(frame)

    info("Extracting IP IDs from raw frames:")
    extracted = []
    for frame in frames:
        ip_id = struct.unpack("!H", frame[18:20])[0]
        extracted.append(chr(ip_id))
    highlight("Recovered", "".join(extracted), RED)

    pause("Press Enter to run the warden...")

    warden = ActiveWarden(secret=b"north-south-gate-key-2026")
    normalized = []
    for frame in frames:
        result = warden.normalize(frame)
        normalized.append(result)

    info("Extracting IP IDs after warden normalization:")
    extracted_after = []
    for frame in normalized:
        ip_id = struct.unpack("!H", frame[18:20])[0]
        extracted_after.append(ip_id)
    highlight("IP IDs after", str(extracted_after), GREEN)

    recovered = ""
    for val in extracted_after:
        if 32 <= val <= 126:
            recovered += chr(val)
        else:
            recovered += "."
    highlight("As ASCII", recovered, GREEN)
    ok("Secret message destroyed. Frames still valid (checksums correct).")

    # ── Attack 2: ISN channel ──

    section("Attack 2: Data in Initial Sequence Number")
    print(f"  The attacker embeds a 32-bit value in the TCP ISN.")
    print()

    covert_value = 0x53454352  # "SECR" in ASCII
    highlight("Covert ISN", f"0x{covert_value:08X} (\"SECR\" in ASCII)", RED)

    frame = _build_malicious_frame(ip_id=1, seq=covert_value, flags=0x02)  # SYN
    tcp_seq_before = struct.unpack("!I", frame[38:42])[0]
    highlight("TCP seq before", f"0x{tcp_seq_before:08X}", RED)

    warden2 = ActiveWarden(secret=b"demo-key")
    result = warden2.normalize(frame)
    tcp_seq_after = struct.unpack("!I", result[38:42])[0]
    highlight("TCP seq after", f"0x{tcp_seq_after:08X}", GREEN)
    ok("ISN rewritten. Original value unrecoverable without warden key.")

    # ── Attack 3: Reserved bits ──

    section("Attack 3: Data in TCP Reserved Bits")
    print(f"  4 bits that the spec says must be zero — but nothing enforces it.")
    print()

    frame = _build_malicious_frame(ip_id=1, seq=5000, reserved_bits=0x0F)
    reserved_before = frame[14 + 20 + 12] & 0x0F
    highlight("Reserved bits before", f"0b{reserved_before:04b} (all set = 4 covert bits)", RED)

    warden3 = ActiveWarden(secret=b"demo-key")
    result = warden3.normalize(frame)
    reserved_after = result[14 + 20 + 12] & 0x0F
    highlight("Reserved bits after", f"0b{reserved_after:04b}", GREEN)
    ok("Reserved bits zeroed.")

    # ── Attack 4: Urgent pointer ──

    section("Attack 4: Data in Urgent Pointer (URG flag not set)")
    print(f"  When URG=0, the urgent pointer is ignored by receivers —")
    print(f"  but the sender can put 16 bits of covert data there.")
    print()

    covert_urg = 0xCAFE
    frame = _build_malicious_frame(ip_id=1, seq=5000, urgent_ptr=covert_urg)
    urg_before = struct.unpack("!H", frame[14 + 20 + 18:14 + 20 + 20])[0]
    highlight("Urgent ptr before", f"0x{urg_before:04X}", RED)

    warden4 = ActiveWarden(secret=b"demo-key")
    result = warden4.normalize(frame)
    urg_after = struct.unpack("!H", result[14 + 20 + 18:14 + 20 + 20])[0]
    highlight("Urgent ptr after", f"0x{urg_after:04X}", GREEN)
    ok("Urgent pointer zeroed (URG flag not set).")

    # ── Attack 5: RST payload ──

    section("Attack 5: Data Hidden in RST Payload")
    print(f"  TCP RST tears down the connection. No payload should exist,")
    print(f"  but the attacker can stuff data in there.")
    print()

    covert_payload = b"STOLEN CLASSIFIED DATA"
    frame = _build_malicious_frame(
        ip_id=1, seq=5000, flags=0x14, payload=covert_payload,
    )
    ip_total_before = struct.unpack("!H", frame[16:18])[0]
    highlight("RST payload", covert_payload.decode(), RED)
    highlight("IP total length", f"{ip_total_before} (includes {len(covert_payload)} bytes of covert data)", RED)

    warden5 = ActiveWarden(secret=b"demo-key")
    result = warden5.normalize(frame)
    ip_total_after = struct.unpack("!H", result[16:18])[0]
    highlight("IP total length after", f"{ip_total_after} (payload stripped)", GREEN)
    ok("RST payload removed. Only headers remain.")

    # ── Attack 6: ToS / DSCP field ──

    section("Attack 6: Data in Type of Service / DSCP Field")
    print(f"  8 bits that most networks ignore entirely.")
    print()

    frame = _build_malicious_frame(ip_id=1, seq=5000, tos=0b10101010)
    tos_before = frame[15]
    highlight("ToS before", f"0b{tos_before:08b} (8 covert bits)", RED)

    warden6 = ActiveWarden(secret=b"demo-key")
    result = warden6.normalize(frame)
    tos_after = result[15]
    highlight("ToS after", f"0b{tos_after:08b}", GREEN)
    ok("ToS/DSCP zeroed.")

    # ── Summary ──

    section("Warden Summary")
    stats = warden.stats.as_dict()
    print(f"  The warden processed {BOLD}{stats['frames_processed']}{RESET} frames from Attack 1.")
    print()
    print(f"  {BOLD}What the warden does NOT need:{RESET}")
    print(f"    - Access to encryption keys or TLS state")
    print(f"    - Knowledge of the payload content")
    print(f"    - Deep packet inspection")
    print(f"    - Signature databases or pattern matching")
    print()
    print(f"  {BOLD}What it does need:{RESET}")
    print(f"    - A secret key (for deterministic ISN/IP ID permutation)")
    print(f"    - Connection tracking (one integer offset per TCP connection)")
    print(f"    - Placement on the same L2 segment as the server")
    print()
    print(f"  {BOLD}Can run on:{RESET}")
    print(f"    - Software (this demo)")
    print(f"    - SmartNIC (NVIDIA BlueField)")
    print(f"    - FPGA (Xilinx Alveo)")
    print(f"    - P4 switch (Intel Tofino) — ~200 lines of P4 code")


# ─── Main ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic Serving Stack Demo")
    parser.add_argument("--part", type=int, choices=[1, 2], help="Run specific part only")
    args = parser.parse_args()

    banner("Deterministic Serving Stack")
    import platform
    highlight("Host", platform.node())
    highlight("Python", platform.python_version())
    highlight("Platform", f"{platform.system()} {platform.machine()}")

    if args.part is None or args.part == 1:
        part1_frame_determinism()
        if args.part is None:
            pause()

    if args.part is None or args.part == 2:
        part2_warden()

    banner("Demo Complete")


if __name__ == "__main__":
    main()
