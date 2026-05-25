#!/usr/bin/env python3
"""Test: are userspace TCP server frames deterministic across connections?

Starts the userspace TCP server, waits for external clients to send
requests, captures outbound frames per connection, and compares
masked digests across trials.

Run on the server, send requests from another machine:
    curl http://<server-ip>:9999/deterministic

Usage:
    sudo python3 tests/benchmark/test_userspace_determinism.py \
        --ip 143.198.114.248 --mac 5e:70:d5:e1:3d:22 --gw-mac fe:00:00:00:01:01 \
        --trials 10
"""
from __future__ import annotations

import argparse
import hashlib
import struct
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.network.networkdet.userspace_tcp_server import UserspaceServer


def mask_frame(f: bytes, server_port: int) -> bytes:
    """Zero out client-derived fields."""
    buf = bytearray(f)
    buf[0:12] = b"\x00" * 12  # MACs
    ip_start = 14
    ihl = (buf[ip_start] & 0x0F) * 4
    tcp_start = ip_start + ihl
    sp = struct.unpack("!H", buf[tcp_start:tcp_start + 2])[0]
    if sp == server_port:
        buf[tcp_start + 2:tcp_start + 4] = b"\x00\x00"  # dst_port
    else:
        buf[tcp_start:tcp_start + 2] = b"\x00\x00"  # src_port
    buf[tcp_start + 4:tcp_start + 12] = b"\x00" * 8  # seq + ack
    buf[ip_start + 10:ip_start + 12] = b"\x00\x00"  # IP cksum
    buf[tcp_start + 16:tcp_start + 18] = b"\x00\x00"  # TCP cksum
    return bytes(buf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", required=True)
    parser.add_argument("--mac", required=True)
    parser.add_argument("--gw-mac", required=True)
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--run-id", default="determinism-proof")
    args = parser.parse_args()

    # Capture outbound frames
    current_frames: list[bytes] = []
    orig_send = UserspaceServer._send_frame

    def capturing_send(self, frame):
        current_frames.append(bytes(frame))
        orig_send(self, frame)

    UserspaceServer._send_frame = capturing_send

    server = UserspaceServer(
        args.interface, args.port, args.ip, args.mac, args.gw_mac,
        mss=1460, run_id=args.run_id,
    )
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(2)

    print("=" * 60)
    print("USERSPACE TCP DETERMINISM TEST")
    print(f"Server: {args.ip}:{args.port}")
    print(f"Waiting for {args.trials} requests from external client...")
    print(f"  curl http://{args.ip}:{args.port}/deterministic")
    print("=" * 60)
    print(flush=True)

    all_trials = []
    trial = 0
    last_count = 0

    while trial < args.trials:
        time.sleep(0.5)
        if len(current_frames) > last_count:
            last_count = len(current_frames)
            time.sleep(2)
            if len(current_frames) == last_count:
                frames = list(current_frames)

                raw_h = hashlib.sha256()
                mask_h = hashlib.sha256()
                for f in frames:
                    raw_h.update(f)
                    mask_h.update(mask_frame(f, args.port))

                # Get client port
                client_port = "?"
                for f in frames:
                    ihl = (f[14] & 0x0F) * 4
                    sp = struct.unpack("!H", f[14 + ihl:14 + ihl + 2])[0]
                    if sp == args.port:
                        client_port = struct.unpack("!H", f[14 + ihl + 2:14 + ihl + 4])[0]
                        break

                raw_d = raw_h.hexdigest()[:16]
                mask_d = mask_h.hexdigest()[:16]
                print(f"  Trial {trial}: {len(frames):3d} frames  "
                      f"client_port={client_port}  "
                      f"raw={raw_d}  masked={mask_d}",
                      flush=True)
                all_trials.append({
                    "frames": len(frames),
                    "raw": raw_d,
                    "masked": mask_d,
                    "client_port": client_port,
                })
                current_frames.clear()
                last_count = 0
                trial += 1

    print()
    raw_set = set(t["raw"] for t in all_trials)
    masked_set = set(t["masked"] for t in all_trials)
    counts = [t["frames"] for t in all_trials]
    ports = [t["client_port"] for t in all_trials]

    print(f"Frame counts: {counts}")
    print(f"Client ports: {ports}")
    print(f"Unique raw digests: {len(raw_set)}/{len(all_trials)}")
    print(f"Unique masked digests: {len(masked_set)}/{len(all_trials)}")
    print()

    if len(raw_set) == 1:
        print("FULLY DETERMINISTIC (raw): every frame byte-identical across all trials")
    elif len(masked_set) == 1:
        print("DETERMINISTIC (masked): identical after masking client port/ISN/checksums")
        print("Raw varies only because client ephemeral port and ISN differ (expected)")
    else:
        print("NON-DETERMINISTIC even after masking")
        by_d = {}
        for i, t in enumerate(all_trials):
            by_d.setdefault(t["masked"], []).append(i)
        for d, trials in by_d.items():
            print(f"  {d} -> trials {trials}")


if __name__ == "__main__":
    main()
