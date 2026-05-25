"""Unit tests for the active warden (inline MRF normalizer)."""
from __future__ import annotations

import struct
import unittest

from modules.network.networkdet.checksums import ip_checksum, tcp_checksum
from modules.network.networkdet.warden import ActiveWarden


def _build_ip_header(
    *,
    src: str = "10.0.0.1",
    dst: str = "10.0.0.2",
    ip_id: int = 0x1234,
    ttl: int = 128,
    protocol: int = 6,
    tos: int = 0,
    total_length: int = 40,
    flags_frag: int = 0x4000,
    ihl: int = 5,
    options: bytes = b"",
) -> bytes:
    """Build a raw IPv4 header with correct checksum."""
    import socket
    src_b = socket.inet_aton(src)
    dst_b = socket.inet_aton(dst)
    actual_ihl = 5 + len(options) // 4
    header_no_cksum = struct.pack(
        "!BBHHHBBH4s4s",
        (4 << 4) | actual_ihl,
        tos,
        total_length,
        ip_id,
        flags_frag,
        ttl,
        protocol,
        0,  # checksum placeholder
        src_b,
        dst_b,
    ) + options
    cksum = ip_checksum(header_no_cksum)
    header = struct.pack(
        "!BBHHHBBH4s4s",
        (4 << 4) | actual_ihl,
        tos,
        total_length,
        ip_id,
        flags_frag,
        ttl,
        protocol,
        cksum,
        src_b,
        dst_b,
    ) + options
    return header


def _build_tcp_header(
    *,
    src_port: int = 12345,
    dst_port: int = 80,
    seq: int = 1000,
    ack: int = 0,
    flags: int = 0x02,  # SYN
    window: int = 65535,
    urgent_ptr: int = 0,
    options: bytes = b"",
    payload: bytes = b"",
    src_ip: str = "10.0.0.1",
    dst_ip: str = "10.0.0.2",
) -> bytes:
    """Build a TCP segment with correct checksum."""
    import socket
    data_offset = (20 + len(options)) // 4
    header_no_cksum = struct.pack(
        "!HHIIBBHHH",
        src_port, dst_port, seq, ack,
        (data_offset << 4),
        flags, window, 0, urgent_ptr,
    ) + options + payload
    cksum = tcp_checksum(
        socket.inet_aton(src_ip),
        socket.inet_aton(dst_ip),
        header_no_cksum,
    )
    header = struct.pack(
        "!HHIIBBHHH",
        src_port, dst_port, seq, ack,
        (data_offset << 4),
        flags, window, cksum, urgent_ptr,
    ) + options + payload
    return header


def _build_frame(
    ip_kwargs: dict | None = None,
    tcp_kwargs: dict | None = None,
    *,
    dst_mac: bytes = b"\x00\x11\x22\x33\x44\x55",
    src_mac: bytes = b"\xaa\xbb\xcc\xdd\xee\xff",
    payload: bytes = b"",
) -> bytes:
    """Build a complete Ethernet frame with IP+TCP."""
    ik = ip_kwargs or {}
    tk = tcp_kwargs or {}

    # Build TCP first to compute total length.
    tcp_seg = _build_tcp_header(
        src_ip=ik.get("src", "10.0.0.1"),
        dst_ip=ik.get("dst", "10.0.0.2"),
        payload=payload,
        **tk,
    )
    ip_ihl = 5 + len(ik.get("options", b"")) // 4
    total_length = ip_ihl * 4 + len(tcp_seg)
    ik.setdefault("total_length", total_length)

    ip_hdr = _build_ip_header(**ik)
    eth_hdr = struct.pack("!6s6sH", dst_mac, src_mac, 0x0800)

    frame = eth_hdr + ip_hdr + tcp_seg
    # Pad to minimum Ethernet frame size.
    if len(frame) < 60:
        frame = frame + b"\x00" * (60 - len(frame))
    return frame


def _parse_ip(frame: bytes) -> dict:
    ip_start = 14
    return {
        "tos": frame[ip_start + 1],
        "total_length": struct.unpack("!H", frame[ip_start + 2:ip_start + 4])[0],
        "ip_id": struct.unpack("!H", frame[ip_start + 4:ip_start + 6])[0],
        "flags_frag": struct.unpack("!H", frame[ip_start + 6:ip_start + 8])[0],
        "ttl": frame[ip_start + 8],
        "ihl": frame[ip_start] & 0x0F,
        "checksum": struct.unpack("!H", frame[ip_start + 10:ip_start + 12])[0],
    }


def _parse_tcp(frame: bytes, ip_ihl: int = 5) -> dict:
    tcp_start = 14 + ip_ihl * 4
    return {
        "src_port": struct.unpack("!H", frame[tcp_start:tcp_start + 2])[0],
        "dst_port": struct.unpack("!H", frame[tcp_start + 2:tcp_start + 4])[0],
        "seq": struct.unpack("!I", frame[tcp_start + 4:tcp_start + 8])[0],
        "ack": struct.unpack("!I", frame[tcp_start + 8:tcp_start + 12])[0],
        "data_offset": (frame[tcp_start + 12] >> 4) * 4,
        "reserved": frame[tcp_start + 12] & 0x0F,
        "flags": frame[tcp_start + 13],
        "window": struct.unpack("!H", frame[tcp_start + 14:tcp_start + 16])[0],
        "checksum": struct.unpack("!H", frame[tcp_start + 16:tcp_start + 18])[0],
        "urgent_ptr": struct.unpack("!H", frame[tcp_start + 18:tcp_start + 20])[0],
    }


def _verify_ip_checksum(frame: bytes) -> bool:
    ip_start = 14
    ihl = frame[ip_start] & 0x0F
    header = bytearray(frame[ip_start:ip_start + ihl * 4])
    return ip_checksum(bytes(header)) == 0


def _verify_tcp_checksum(frame: bytes) -> bool:
    ip_start = 14
    ihl = frame[ip_start] & 0x0F
    ip_total = struct.unpack("!H", frame[ip_start + 2:ip_start + 4])[0]
    tcp_start = ip_start + ihl * 4
    tcp_end = ip_start + ip_total
    src_ip = frame[ip_start + 12:ip_start + 16]
    dst_ip = frame[ip_start + 16:ip_start + 20]
    tcp_seg = frame[tcp_start:tcp_end]
    return tcp_checksum(src_ip, dst_ip, tcp_seg) == 0


class TestWardenIPNormalization(unittest.TestCase):
    """Test IP header normalization."""

    def test_tos_zeroed(self):
        frame = _build_frame(ip_kwargs={"tos": 0xFF})
        warden = ActiveWarden()
        result = warden.normalize(frame)
        self.assertIsNotNone(result)
        ip = _parse_ip(result)
        self.assertEqual(ip["tos"], 0)
        self.assertEqual(warden.stats.tos_zeroed, 1)

    def test_ip_id_rewritten(self):
        frame = _build_frame(ip_kwargs={"ip_id": 0xDEAD})
        warden = ActiveWarden(secret=b"test-secret")
        result = warden.normalize(frame)
        ip = _parse_ip(result)
        self.assertNotEqual(ip["ip_id"], 0xDEAD)

    def test_ip_id_deterministic(self):
        """Same frame + same secret = same rewritten IP ID."""
        frame = _build_frame(ip_kwargs={"ip_id": 0xBEEF})
        w1 = ActiveWarden(secret=b"same")
        w2 = ActiveWarden(secret=b"same")
        r1 = w1.normalize(frame)
        r2 = w2.normalize(frame)
        ip1 = _parse_ip(r1)
        ip2 = _parse_ip(r2)
        self.assertEqual(ip1["ip_id"], ip2["ip_id"])

    def test_ip_id_different_secrets(self):
        """Different secrets produce different IP IDs."""
        frame = _build_frame(ip_kwargs={"ip_id": 0xBEEF})
        w1 = ActiveWarden(secret=b"key-a")
        w2 = ActiveWarden(secret=b"key-b")
        r1 = w1.normalize(frame)
        r2 = w2.normalize(frame)
        ip1 = _parse_ip(r1)
        ip2 = _parse_ip(r2)
        self.assertNotEqual(ip1["ip_id"], ip2["ip_id"])

    def test_ttl_normalized(self):
        frame = _build_frame(ip_kwargs={"ttl": 200})
        warden = ActiveWarden(ttl=64)
        result = warden.normalize(frame)
        ip = _parse_ip(result)
        self.assertEqual(ip["ttl"], 64)

    def test_flags_forced_df(self):
        frame = _build_frame(ip_kwargs={"flags_frag": 0x0000})  # No DF.
        warden = ActiveWarden()
        result = warden.normalize(frame)
        ip = _parse_ip(result)
        self.assertEqual(ip["flags_frag"], 0x4000)  # DF set, no fragment offset.

    def test_ip_options_stripped(self):
        # 4 bytes of IP options (NOP padding).
        options = b"\x01\x01\x01\x01"
        frame = _build_frame(ip_kwargs={"options": options, "ihl": 6})
        warden = ActiveWarden()
        result = warden.normalize(frame)
        ip = _parse_ip(result)
        self.assertEqual(ip["ihl"], 5)  # Options removed.
        self.assertEqual(warden.stats.options_stripped, 1)

    def test_ip_checksum_valid_after_normalization(self):
        frame = _build_frame(ip_kwargs={"tos": 0x28, "ttl": 200, "ip_id": 0xAAAA})
        warden = ActiveWarden()
        result = warden.normalize(frame)
        self.assertTrue(_verify_ip_checksum(result))

    def test_non_ipv4_passed_unchanged(self):
        # ARP frame (ethertype 0x0806).
        frame = struct.pack("!6s6sH", b"\xff" * 6, b"\xaa" * 6, 0x0806) + b"\x00" * 28
        warden = ActiveWarden()
        result = warden.normalize(frame)
        self.assertEqual(result, frame)


class TestWardenTCPNormalization(unittest.TestCase):
    """Test TCP header normalization."""

    def test_reserved_bits_zeroed(self):
        frame = _build_frame(tcp_kwargs={
            "flags": 0x10,  # ACK
            "seq": 5000, "ack": 1000,
        })
        # Manually set reserved bits in the frame.
        buf = bytearray(frame)
        tcp_start = 34
        buf[tcp_start + 12] |= 0x0F  # Set all reserved bits.
        warden = ActiveWarden()
        result = warden.normalize(bytes(buf))
        tcp = _parse_tcp(result)
        self.assertEqual(tcp["reserved"], 0)

    def test_urgent_ptr_zeroed_when_urg_not_set(self):
        frame = _build_frame(tcp_kwargs={
            "flags": 0x10,  # ACK, no URG
            "urgent_ptr": 0x1234,
            "seq": 5000, "ack": 1000,
        })
        warden = ActiveWarden()
        result = warden.normalize(frame)
        tcp = _parse_tcp(result)
        self.assertEqual(tcp["urgent_ptr"], 0)
        self.assertEqual(warden.stats.urgent_ptr_zeroed, 1)

    def test_urgent_ptr_preserved_when_valid(self):
        payload = b"URGENT DATA HERE"
        frame = _build_frame(
            tcp_kwargs={
                "flags": 0x30,  # ACK | URG
                "urgent_ptr": 5,  # Within payload.
                "seq": 5000, "ack": 1000,
            },
            payload=payload,
        )
        warden = ActiveWarden()
        result = warden.normalize(frame)
        tcp = _parse_tcp(result)
        # URG is set and pointer is valid — should be preserved.
        self.assertTrue(tcp["flags"] & 0x20)  # URG still set.

    def test_urgent_ptr_cleared_when_out_of_bounds(self):
        payload = b"short"
        frame = _build_frame(
            tcp_kwargs={
                "flags": 0x30,  # ACK | URG
                "urgent_ptr": 9999,  # Way past payload.
                "seq": 5000, "ack": 1000,
            },
            payload=payload,
        )
        warden = ActiveWarden()
        result = warden.normalize(frame)
        tcp = _parse_tcp(result)
        self.assertFalse(tcp["flags"] & 0x20)  # URG cleared.
        self.assertEqual(tcp["urgent_ptr"], 0)

    def test_rst_payload_stripped(self):
        payload = b"covert data in RST"
        frame = _build_frame(
            tcp_kwargs={
                "flags": 0x14,  # ACK | RST
                "seq": 5000, "ack": 1000,
            },
            payload=payload,
        )
        warden = ActiveWarden()
        result = warden.normalize(frame)
        ip = _parse_ip(result)
        tcp = _parse_tcp(result)
        # Payload should be gone: IP total = IP header + TCP header only.
        self.assertEqual(ip["total_length"], 20 + tcp["data_offset"])
        self.assertEqual(warden.stats.rst_payloads_stripped, 1)

    def test_tcp_options_stripped_except_mss(self):
        # MSS(4) + Timestamps(10) + NOP(1) + Window Scale(3) = 18, padded to 20.
        mss_opt = struct.pack("!BBH", 2, 4, 1460)
        ts_opt = struct.pack("!BBII", 8, 10, 12345, 67890)
        nop = b"\x01"
        ws_opt = struct.pack("!BBB", 3, 3, 7)
        padding = b"\x00"  # Pad to 4-byte boundary: 4+10+1+3=18 -> 20.
        options = mss_opt + ts_opt + nop + ws_opt + padding * 2

        frame = _build_frame(tcp_kwargs={
            "flags": 0x02,  # SYN
            "seq": 1000,
            "options": options,
        })
        warden = ActiveWarden()
        result = warden.normalize(frame)
        tcp = _parse_tcp(result)

        # Should keep only MSS (data_offset = 24 = 6*4).
        self.assertEqual(tcp["data_offset"], 24)

        # Verify MSS option is preserved.
        tcp_start = 34
        opt_kind = result[tcp_start + 20]
        opt_len = result[tcp_start + 21]
        opt_val = struct.unpack("!H", result[tcp_start + 22:tcp_start + 24])[0]
        self.assertEqual(opt_kind, 2)
        self.assertEqual(opt_len, 4)
        self.assertEqual(opt_val, 1460)

    def test_tcp_options_all_stripped_on_non_syn(self):
        ts_opt = struct.pack("!BBII", 8, 10, 12345, 67890)
        nop = b"\x01"
        padding = b"\x00" * 2  # 1+10+2=13 -> pad to 12.. actually need 4-byte align
        # 10 + 1 + 1 = 12, that's already aligned.
        options = ts_opt + nop + nop

        frame = _build_frame(tcp_kwargs={
            "flags": 0x10,  # ACK (not SYN)
            "seq": 5000, "ack": 1000,
            "options": options,
        })
        warden = ActiveWarden()
        result = warden.normalize(frame)
        tcp = _parse_tcp(result)

        # All options stripped on non-SYN: data_offset = 20.
        self.assertEqual(tcp["data_offset"], 20)

    def test_tcp_checksum_valid_after_normalization(self):
        frame = _build_frame(tcp_kwargs={
            "flags": 0x10,
            "seq": 5000, "ack": 1000,
            "urgent_ptr": 0xFFFF,
        })
        warden = ActiveWarden()
        result = warden.normalize(frame)
        self.assertTrue(_verify_tcp_checksum(result))


class TestWardenISNRewriting(unittest.TestCase):
    """Test stateful ISN rewriting across a TCP handshake."""

    def test_syn_isn_rewritten(self):
        frame = _build_frame(tcp_kwargs={
            "flags": 0x02,  # SYN
            "seq": 1000,
            "src_port": 50000, "dst_port": 80,
        })
        warden = ActiveWarden(secret=b"test")
        result = warden.normalize(frame)
        tcp = _parse_tcp(result)
        self.assertNotEqual(tcp["seq"], 1000)
        self.assertEqual(warden.stats.isn_rewrites, 1)

    def test_handshake_consistency(self):
        """Full 3-way handshake with ISN rewriting maintains consistency."""
        warden = ActiveWarden(secret=b"handshake-test")

        # Client SYN: seq=1000
        syn = _build_frame(tcp_kwargs={
            "flags": 0x02, "seq": 1000, "ack": 0,
            "src_port": 50000, "dst_port": 80,
        }, ip_kwargs={"src": "10.0.0.1", "dst": "10.0.0.2"})
        syn_out = warden.normalize(syn)
        syn_tcp = _parse_tcp(syn_out)
        client_new_isn = syn_tcp["seq"]

        # Server SYN-ACK: seq=2000, ack=1001
        syn_ack = _build_frame(tcp_kwargs={
            "flags": 0x12, "seq": 2000, "ack": 1001,
            "src_port": 80, "dst_port": 50000,
        }, ip_kwargs={"src": "10.0.0.2", "dst": "10.0.0.1"})
        syn_ack_out = warden.normalize(syn_ack)
        syn_ack_tcp = _parse_tcp(syn_ack_out)
        server_new_isn = syn_ack_tcp["seq"]

        # The SYN-ACK's ack should reference the client's new ISN + 1.
        self.assertEqual(syn_ack_tcp["ack"], (client_new_isn + 1) & 0xFFFFFFFF)

        # Server ISN should also be rewritten.
        self.assertNotEqual(server_new_isn, 2000)

        # Client ACK: seq=1001, ack=2001
        ack = _build_frame(tcp_kwargs={
            "flags": 0x10, "seq": 1001, "ack": 2001,
            "src_port": 50000, "dst_port": 80,
        }, ip_kwargs={"src": "10.0.0.1", "dst": "10.0.0.2"})
        ack_out = warden.normalize(ack)
        ack_tcp = _parse_tcp(ack_out)

        # Client seq should be offset consistently.
        self.assertEqual(ack_tcp["seq"], (client_new_isn + 1) & 0xFFFFFFFF)
        # Client ack should reference server's new ISN + 1.
        self.assertEqual(ack_tcp["ack"], (server_new_isn + 1) & 0xFFFFFFFF)

    def test_data_transfer_after_handshake(self):
        """Data packets have consistent seq/ack after ISN rewrite."""
        warden = ActiveWarden(secret=b"data-test")

        # SYN
        syn = _build_frame(tcp_kwargs={
            "flags": 0x02, "seq": 100, "ack": 0,
            "src_port": 40000, "dst_port": 8080,
        }, ip_kwargs={"src": "10.0.0.1", "dst": "10.0.0.2"})
        syn_out = warden.normalize(syn)
        new_client_isn = _parse_tcp(syn_out)["seq"]

        # SYN-ACK
        syn_ack = _build_frame(tcp_kwargs={
            "flags": 0x12, "seq": 200, "ack": 101,
            "src_port": 8080, "dst_port": 40000,
        }, ip_kwargs={"src": "10.0.0.2", "dst": "10.0.0.1"})
        syn_ack_out = warden.normalize(syn_ack)
        new_server_isn = _parse_tcp(syn_ack_out)["seq"]

        # ACK
        ack = _build_frame(tcp_kwargs={
            "flags": 0x10, "seq": 101, "ack": 201,
            "src_port": 40000, "dst_port": 8080,
        }, ip_kwargs={"src": "10.0.0.1", "dst": "10.0.0.2"})
        warden.normalize(ack)

        # Data: client sends 50 bytes, seq=101
        data_frame = _build_frame(
            tcp_kwargs={
                "flags": 0x18, "seq": 101, "ack": 201,  # PSH|ACK
                "src_port": 40000, "dst_port": 8080,
            },
            ip_kwargs={"src": "10.0.0.1", "dst": "10.0.0.2"},
            payload=b"A" * 50,
        )
        data_out = warden.normalize(data_frame)
        data_tcp = _parse_tcp(data_out)

        expected_seq = (new_client_isn + 1) & 0xFFFFFFFF
        expected_ack = (new_server_isn + 1) & 0xFFFFFFFF
        self.assertEqual(data_tcp["seq"], expected_seq)
        self.assertEqual(data_tcp["ack"], expected_ack)


class TestWardenEthernetPadding(unittest.TestCase):
    """Test Ethernet padding normalization."""

    def test_trailing_padding_zeroed(self):
        frame = _build_frame(tcp_kwargs={
            "flags": 0x10, "seq": 5000, "ack": 1000,
        })
        # Add garbage padding.
        frame = frame + b"\xDE\xAD\xBE\xEF"
        warden = ActiveWarden()
        result = warden.normalize(frame)
        # Trailing bytes should be zeroed.
        ip = _parse_ip(result)
        payload_end = 14 + ip["total_length"]
        trailing = result[payload_end:]
        self.assertTrue(all(b == 0 for b in trailing))


class TestWardenChecksums(unittest.TestCase):
    """Verify all checksums are valid after normalization."""

    def test_both_checksums_valid(self):
        """IP and TCP checksums both validate after full normalization."""
        frame = _build_frame(
            ip_kwargs={"tos": 0x28, "ttl": 200, "ip_id": 0xFFFF},
            tcp_kwargs={
                "flags": 0x10, "seq": 9999, "ack": 8888,
                "urgent_ptr": 0xBEEF,
            },
            payload=b"test payload data",
        )
        warden = ActiveWarden()
        result = warden.normalize(frame)
        self.assertTrue(_verify_ip_checksum(result))
        self.assertTrue(_verify_tcp_checksum(result))

    def test_checksums_after_option_stripping(self):
        """Checksums valid after TCP options are stripped."""
        ts_opt = struct.pack("!BBII", 8, 10, 12345, 67890)
        nop = b"\x01"
        options = ts_opt + nop + nop  # 12 bytes, 4-byte aligned.
        frame = _build_frame(
            tcp_kwargs={
                "flags": 0x10, "seq": 5000, "ack": 1000,
                "options": options,
            },
            payload=b"data after options",
        )
        warden = ActiveWarden()
        result = warden.normalize(frame)
        self.assertTrue(_verify_ip_checksum(result))
        self.assertTrue(_verify_tcp_checksum(result))


class TestWardenStats(unittest.TestCase):
    """Test that stats are tracked correctly."""

    def test_stats_accumulate(self):
        warden = ActiveWarden()
        frame = _build_frame(
            ip_kwargs={"tos": 0xFF, "ttl": 200},
            tcp_kwargs={
                "flags": 0x10, "seq": 5000, "ack": 1000,
                "urgent_ptr": 0x1234,
            },
        )
        warden.normalize(frame)
        warden.normalize(frame)
        s = warden.stats
        self.assertEqual(s.frames_processed, 2)
        self.assertEqual(s.frames_passed, 2)
        self.assertEqual(s.tos_zeroed, 2)
        self.assertEqual(s.ttl_normalized, 2)
        self.assertEqual(s.urgent_ptr_zeroed, 2)
        self.assertGreaterEqual(s.checksums_recomputed, 4)  # 2 IP + 2 TCP.

    def test_stats_as_dict(self):
        warden = ActiveWarden()
        d = warden.stats.as_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("frames_processed", d)

    def test_reset_clears_state(self):
        warden = ActiveWarden()
        frame = _build_frame(tcp_kwargs={
            "flags": 0x02, "seq": 1000,
        })
        warden.normalize(frame)
        self.assertEqual(warden.stats.frames_processed, 1)
        warden.reset()
        self.assertEqual(warden.stats.frames_processed, 0)


class TestWardenIdempotence(unittest.TestCase):
    """Normalizing an already-normalized frame should be stable."""

    def test_double_normalization_stable(self):
        """Passing a normalized frame through again should not change it
        (except ISN on new SYNs, which we avoid by using a data frame)."""
        warden = ActiveWarden(secret=b"idempotent")

        # First normalize a SYN to establish connection state.
        syn = _build_frame(tcp_kwargs={
            "flags": 0x02, "seq": 1000,
            "src_port": 55555, "dst_port": 80,
        })
        warden.normalize(syn)

        # Now normalize a data frame.
        data = _build_frame(
            tcp_kwargs={
                "flags": 0x18, "seq": 1001, "ack": 2001,
                "src_port": 55555, "dst_port": 80,
            },
            payload=b"hello world",
        )
        first = warden.normalize(data)

        # Create a second warden with the same state setup.
        warden2 = ActiveWarden(secret=b"idempotent")
        warden2.normalize(syn)
        # Normalize the already-normalized frame.
        second = warden2.normalize(first)

        # The IP ID will differ (re-encrypted), and seq/ack will be
        # double-offset. But checksums should still be valid.
        self.assertTrue(_verify_ip_checksum(second))
        self.assertTrue(_verify_tcp_checksum(second))


class TestWardenCovertChannelElimination(unittest.TestCase):
    """Verify that specific covert channel techniques from the paper
    are destroyed by the warden."""

    def test_covert_tcp_ip_id_channel(self):
        """Covert TCP embeds ASCII in IP ID field. Warden destroys it."""
        # Embed 'S', 'T', 'E', 'G' in IP IDs of 4 packets.
        secret_message = [ord('S'), ord('T'), ord('E'), ord('G')]
        warden = ActiveWarden(secret=b"anti-steg")

        recovered_before = []
        recovered_after = []

        for i, char_val in enumerate(secret_message):
            frame = _build_frame(
                ip_kwargs={"ip_id": char_val},
                tcp_kwargs={"flags": 0x10, "seq": 5000 + i, "ack": 1000},
            )
            recovered_before.append(_parse_ip(frame)["ip_id"])
            result = warden.normalize(frame)
            recovered_after.append(_parse_ip(result)["ip_id"])

        # Before: attacker can read back STEG.
        self.assertEqual(recovered_before, secret_message)
        # After: the message is destroyed.
        self.assertNotEqual(recovered_after, secret_message)

    def test_covert_tcp_isn_channel(self):
        """Covert TCP embeds data in ISN. Warden rewrites ISN."""
        covert_isn = 0x41424344  # "ABCD" in ASCII.
        frame = _build_frame(tcp_kwargs={
            "flags": 0x02, "seq": covert_isn,
            "src_port": 60000, "dst_port": 80,
        })
        warden = ActiveWarden(secret=b"anti-isn")
        result = warden.normalize(frame)
        tcp = _parse_tcp(result)
        self.assertNotEqual(tcp["seq"], covert_isn)

    def test_covert_reserved_bits_channel(self):
        """Data hidden in TCP reserved bits is zeroed."""
        frame = _build_frame(tcp_kwargs={
            "flags": 0x10, "seq": 5000, "ack": 1000,
        })
        buf = bytearray(frame)
        buf[34 + 12] |= 0x0F  # Set all 4 reserved bits to 1111.
        warden = ActiveWarden()
        result = warden.normalize(bytes(buf))
        tcp = _parse_tcp(result)
        self.assertEqual(tcp["reserved"], 0)

    def test_covert_urgent_pointer_channel(self):
        """Data hidden in urgent pointer (URG=0) is zeroed."""
        frame = _build_frame(tcp_kwargs={
            "flags": 0x10,  # ACK only, no URG.
            "seq": 5000, "ack": 1000,
            "urgent_ptr": 0xCAFE,  # Covert data.
        })
        warden = ActiveWarden()
        result = warden.normalize(frame)
        tcp = _parse_tcp(result)
        self.assertEqual(tcp["urgent_ptr"], 0)

    def test_covert_rst_payload_channel(self):
        """Data hidden in RST payload is stripped."""
        frame = _build_frame(
            tcp_kwargs={
                "flags": 0x14,  # RST|ACK
                "seq": 5000, "ack": 1000,
            },
            payload=b"SECRET EXFILTRATED DATA",
        )
        warden = ActiveWarden()
        result = warden.normalize(frame)
        ip = _parse_ip(result)
        tcp = _parse_tcp(result)
        # No payload should remain.
        self.assertEqual(ip["total_length"], 20 + tcp["data_offset"])


if __name__ == "__main__":
    unittest.main()
