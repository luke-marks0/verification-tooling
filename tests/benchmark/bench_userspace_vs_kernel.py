#!/usr/bin/env python3
"""Benchmark: Userspace TCP vs Kernel TCP path.

Measures throughput, latency, and connection rate for both:
  1. Kernel TCP: standard Python http.server (kernel handles TCP)
  2. Userspace TCP: our AF_PACKET-based DeterministicTCPConnection

Runs as two roles:
  - server: starts either userspace or kernel HTTP server
  - client: sends N requests to the server, measuring timing

Usage:
    # On server (droplet A):
    sudo python3 bench_userspace_vs_kernel.py server --mode userspace \
        --ip 143.198.114.248 --mac aa:bb:cc:dd:ee:ff --gw-mac ff:ee:dd:cc:bb:aa

    sudo python3 bench_userspace_vs_kernel.py server --mode kernel

    # On client (droplet B):
    python3 bench_userspace_vs_kernel.py client --server-ip 143.198.114.248 \
        --port 9999 --requests 500 --payload-size 40000
"""
from __future__ import annotations

import argparse
import http.server
import json
import hashlib
import socket
import struct
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
PAYLOAD_SIZES = [
    ("small", 200),           # ~1 segment, like a short JSON response
    ("medium", 5_000),        # ~4 segments, typical API response
    ("large", 40_000),        # ~28 segments, ~10k tokens of text
    ("xlarge", 150_000),      # ~103 segments, large model output
]

def generate_body(size: int) -> bytes:
    """Deterministic body of exactly `size` bytes."""
    block = hashlib.sha256(b"bench-payload-block").digest()
    reps = (size // len(block)) + 1
    return (block * reps)[:size]


def build_http_response(body: bytes) -> bytes:
    headers = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: application/octet-stream\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n"
        b"Connection: close\r\n"
        b"Server: Benchmark/1.0\r\n"
        b"Date: Thu, 01 Jan 2026 00:00:00 GMT\r\n"
        b"\r\n"
    )
    return headers + body


# ---------------------------------------------------------------------------
# Kernel TCP server (standard http.server)
# ---------------------------------------------------------------------------
class KernelHandler(http.server.BaseHTTPRequestHandler):
    responses_by_size: dict[int, bytes] = {}

    def version_string(self):
        return "Benchmark/1.0"

    def date_time_string(self, timestamp=None):
        return "Thu, 01 Jan 2026 00:00:00 GMT"

    def log_message(self, *args):
        pass

    def do_GET(self):
        # Parse ?size=N from path
        size = 5000  # default
        if "?" in self.path:
            for param in self.path.split("?")[1].split("&"):
                if param.startswith("size="):
                    size = int(param.split("=")[1])

        if size not in self.responses_by_size:
            self.responses_by_size[size] = generate_body(size)
        body = self.responses_by_size[size]

        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


def start_kernel_server(port: int):
    server = http.server.HTTPServer(("0.0.0.0", port), KernelHandler)
    print(f"Kernel TCP server listening on port {port}")
    server.serve_forever()


# ---------------------------------------------------------------------------
# Userspace TCP server wrapper
# ---------------------------------------------------------------------------
def start_userspace_server(port: int, ip: str, mac: str, gw_mac: str, interface: str):
    from modules.network.networkdet.userspace_tcp_server import UserspaceServer, RESPONSES

    # Pre-populate responses for all benchmark sizes
    for name, size in PAYLOAD_SIZES:
        body = generate_body(size)
        RESPONSES[f"/bench?size={size}"] = build_http_response(body)

    server = UserspaceServer(
        interface=interface,
        port=port,
        local_ip=ip,
        local_mac=mac,
        gateway_mac=gw_mac,
        mss=1460,
        run_id="bench-run",
    )
    print(f"Userspace TCP server listening on {ip}:{port} (interface={interface})")
    server.serve_forever()


# ---------------------------------------------------------------------------
# Client benchmark
# ---------------------------------------------------------------------------
def run_client_benchmark(
    server_ip: str,
    port: int,
    num_requests: int,
    payload_size: int,
    warmup: int = 5,
):
    """Send requests and measure latency + throughput."""
    import urllib.request

    url = f"http://{server_ip}:{port}/bench?size={payload_size}"
    expected_body = generate_body(payload_size)

    # Warmup
    print(f"Warming up with {warmup} requests...")
    for _ in range(warmup):
        try:
            resp = urllib.request.urlopen(url, timeout=30)
            resp.read()
        except Exception as e:
            print(f"  Warmup request failed: {e}")
            time.sleep(0.5)

    # Benchmark
    latencies = []
    bytes_received = 0
    errors = 0
    correct = 0

    print(f"Running {num_requests} requests to {url} ...")
    t_start = time.monotonic()

    for i in range(num_requests):
        t0 = time.monotonic()
        try:
            resp = urllib.request.urlopen(url, timeout=30)
            data = resp.read()
            t1 = time.monotonic()
            latencies.append(t1 - t0)
            bytes_received += len(data)
            if data == expected_body:
                correct += 1
            else:
                print(f"  Request {i}: body mismatch (got {len(data)} bytes, expected {payload_size})")
        except Exception as e:
            t1 = time.monotonic()
            latencies.append(t1 - t0)
            errors += 1
            if errors <= 3:
                print(f"  Request {i}: {e}")

    t_end = time.monotonic()
    wall_time = t_end - t_start

    return {
        "requests": num_requests,
        "payload_size": payload_size,
        "wall_time_s": round(wall_time, 3),
        "errors": errors,
        "correct_bodies": correct,
        "bytes_received": bytes_received,
        "throughput_mbps": round(bytes_received * 8 / wall_time / 1_000_000, 2) if wall_time > 0 else 0,
        "requests_per_sec": round(num_requests / wall_time, 1) if wall_time > 0 else 0,
        "latency_ms": {
            "min": round(min(latencies) * 1000, 2) if latencies else 0,
            "max": round(max(latencies) * 1000, 2) if latencies else 0,
            "mean": round(statistics.mean(latencies) * 1000, 2) if latencies else 0,
            "median": round(statistics.median(latencies) * 1000, 2) if latencies else 0,
            "p95": round(sorted(latencies)[int(len(latencies) * 0.95)] * 1000, 2) if latencies else 0,
            "p99": round(sorted(latencies)[int(len(latencies) * 0.99)] * 1000, 2) if latencies else 0,
            "stdev": round(statistics.stdev(latencies) * 1000, 2) if len(latencies) > 1 else 0,
        },
    }


def run_full_benchmark(server_ip: str, port: int, min_duration: float = 10.0):
    """Run the complete benchmark suite across all payload sizes."""
    results = {}

    # First, calibrate request counts with a short probe
    print("Calibrating request counts...")
    for name, size in PAYLOAD_SIZES:
        probe = run_client_benchmark(server_ip, port, 20, size, warmup=3)
        rate = probe["requests_per_sec"]
        n = max(50, int(rate * min_duration * 1.2))  # 20% margin
        print(f"  {name}: {rate:.0f} req/s -> {n} requests (~{n/rate:.0f}s)")
        results[f"_calibrate_{name}"] = {"rate": rate, "n": n}

    calibrated = {name: results[f"_calibrate_{name}"]["n"] for name, _ in PAYLOAD_SIZES}
    results = {}  # reset

    for name, size in PAYLOAD_SIZES:
        n = calibrated[name]

        print(f"\n{'='*60}")
        print(f"Payload: {name} ({size:,} bytes), {n} requests")
        print(f"{'='*60}")

        result = run_client_benchmark(server_ip, port, n, size)
        results[name] = result

        print(f"  Wall time: {result['wall_time_s']}s")
        print(f"  Throughput: {result['throughput_mbps']} Mbps")
        print(f"  Req/s: {result['requests_per_sec']}")
        print(f"  Latency (median): {result['latency_ms']['median']}ms")
        print(f"  Latency (p99): {result['latency_ms']['p99']}ms")
        print(f"  Errors: {result['errors']}")
        print(f"  Correct: {result['correct_bodies']}/{result['requests']}")

    return results


def print_summary(results: dict):
    """Print formatted summary."""
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"{'Payload':<10} {'Reqs':>6} {'Wall(s)':>8} {'Req/s':>8} {'Mbps':>8} "
          f"{'Med(ms)':>8} {'P99(ms)':>8} {'Err':>5}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<10} {r['requests']:>6} {r['wall_time_s']:>8.1f} "
              f"{r['requests_per_sec']:>8.1f} {r['throughput_mbps']:>8.2f} "
              f"{r['latency_ms']['median']:>8.2f} {r['latency_ms']['p99']:>8.2f} "
              f"{r['errors']:>5}")


def main():
    parser = argparse.ArgumentParser(description="Userspace vs Kernel TCP benchmark")
    sub = parser.add_subparsers(dest="role", required=True)

    # Server subcommand
    srv = sub.add_parser("server")
    srv.add_argument("--mode", choices=["userspace", "kernel"], required=True)
    srv.add_argument("--port", type=int, default=9999)
    srv.add_argument("--ip", help="Local IP (userspace mode)")
    srv.add_argument("--mac", help="Local MAC (userspace mode)")
    srv.add_argument("--gw-mac", help="Gateway MAC (userspace mode)")
    srv.add_argument("--interface", default="eth0")

    # Client subcommand
    cli = sub.add_parser("client")
    cli.add_argument("--server-ip", required=True)
    cli.add_argument("--port", type=int, default=9999)
    cli.add_argument("--payload-size", type=int, help="Single size to test")
    cli.add_argument("--requests", type=int, help="Number of requests (overrides auto)")
    cli.add_argument("--output", help="Save JSON results to file")

    args = parser.parse_args()

    if args.role == "server":
        if args.mode == "kernel":
            start_kernel_server(args.port)
        else:
            if not all([args.ip, args.mac, args.gw_mac]):
                parser.error("Userspace mode requires --ip, --mac, --gw-mac")
            start_userspace_server(args.port, args.ip, args.mac, args.gw_mac, args.interface)

    elif args.role == "client":
        if args.payload_size and args.requests:
            result = run_client_benchmark(args.server_ip, args.port, args.requests, args.payload_size)
            print(json.dumps(result, indent=2))
        else:
            results = run_full_benchmark(args.server_ip, args.port)
            print_summary(results)
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
