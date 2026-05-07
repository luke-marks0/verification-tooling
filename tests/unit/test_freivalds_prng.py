"""Determinism and dtype-quantization tests for pkg.freivalds.prng.

The PRNG is the cross-implementation contract — any backend (stdlib,
torch, future custom) must produce identical canonical bytes for the same
``(seed, dtype, rows, cols)``. These tests pin the bytes for a fixed seed
and check the quantization invariants (e.g. fp64 magnitudes lie in
``[0.5, 1.0)``).
"""
from __future__ import annotations

import struct
import unittest

from pkg.freivalds import prng


class TestPRNGDeterminism(unittest.TestCase):
    def test_same_seed_same_bytes(self) -> None:
        a = prng.gen_matrix_bytes(seed=42, dtype="int8", rows=8, cols=16)
        b = prng.gen_matrix_bytes(seed=42, dtype="int8", rows=8, cols=16)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 8 * 16 * 1)

    def test_different_seed_different_bytes(self) -> None:
        a = prng.gen_matrix_bytes(seed=42, dtype="int8", rows=8, cols=16)
        b = prng.gen_matrix_bytes(seed=43, dtype="int8", rows=8, cols=16)
        self.assertNotEqual(a, b)

    def test_different_shape_different_bytes(self) -> None:
        a = prng.gen_matrix_bytes(seed=42, dtype="int8", rows=8, cols=16)
        b = prng.gen_matrix_bytes(seed=42, dtype="int8", rows=16, cols=8)
        # Same total bytes (128) but bytes themselves differ because the
        # shape is folded into the SHAKE label.
        self.assertEqual(len(a), len(b))
        self.assertNotEqual(a, b)

    def test_different_dtype_different_bytes_at_same_byte_count(self) -> None:
        a = prng.gen_matrix_bytes(seed=42, dtype="bf16", rows=4, cols=4)
        b = prng.gen_matrix_bytes(seed=42, dtype="fp16", rows=4, cols=4)
        self.assertEqual(len(a), len(b))  # both 2 bytes/elem * 16 elems
        self.assertNotEqual(a, b)

    def test_byte_lengths(self) -> None:
        cases = [
            ("int8", 1), ("int32", 4),
            ("fp16", 2), ("bf16", 2), ("fp32", 4), ("fp64", 8),
            ("fp8_e4m3", 1),
        ]
        for dtype, bpe in cases:
            with self.subTest(dtype=dtype):
                buf = prng.gen_matrix_bytes(seed=1, dtype=dtype, rows=3, cols=5)
                self.assertEqual(len(buf), 3 * 5 * bpe)


class TestFP64Quantization(unittest.TestCase):
    def test_fp64_values_in_half_to_one_magnitude(self) -> None:
        # All twiddled fp64 values: exponent biased to 1022 -> magnitude in [0.5, 1.0).
        m, _ = prng.gen_matrix_stdlib(seed=7, dtype="fp64", rows=16, cols=16)
        for row in m:
            for v in row:
                self.assertTrue(0.5 <= abs(v) < 1.0, f"value {v!r} out of expected band")

    def test_fp64_signs_are_mixed(self) -> None:
        # Sign bit is unmasked, so over many values we expect both signs.
        m, _ = prng.gen_matrix_stdlib(seed=11, dtype="fp64", rows=32, cols=32)
        flat = [v for row in m for v in row]
        positives = sum(1 for v in flat if v > 0)
        negatives = sum(1 for v in flat if v < 0)
        self.assertGreater(positives, 100)
        self.assertGreater(negatives, 100)


class TestFP32Quantization(unittest.TestCase):
    def test_fp32_bytes_decode_in_band(self) -> None:
        # We don't have stdlib reader for fp32 in the matrix sense, but we
        # can reinterpret bytes as fp32 directly.
        buf = prng.gen_matrix_bytes(seed=3, dtype="fp32", rows=8, cols=8)
        for i in range(0, len(buf), 4):
            (val,) = struct.unpack_from("<f", buf, i)
            self.assertTrue(0.5 <= abs(val) < 1.0, f"fp32 {val!r} out of band")


class TestBF16FP16FP8BitPattern(unittest.TestCase):
    """Verify the bit-twiddle pins the exponent for the small float dtypes."""

    def test_bf16_exponent_is_126(self) -> None:
        buf = prng.gen_matrix_bytes(seed=5, dtype="bf16", rows=4, cols=4)
        for i in range(0, len(buf), 2):
            (val,) = struct.unpack_from("<H", buf, i)
            exp = (val >> 7) & 0xFF
            self.assertEqual(exp, 126, f"bf16 exp at offset {i} is {exp}, expected 126")

    def test_fp16_exponent_is_14(self) -> None:
        buf = prng.gen_matrix_bytes(seed=5, dtype="fp16", rows=4, cols=4)
        for i in range(0, len(buf), 2):
            (val,) = struct.unpack_from("<H", buf, i)
            exp = (val >> 10) & 0x1F
            self.assertEqual(exp, 14, f"fp16 exp at offset {i} is {exp}, expected 14")

    def test_fp8_e4m3_exponent_is_6(self) -> None:
        buf = prng.gen_matrix_bytes(seed=5, dtype="fp8_e4m3", rows=4, cols=4)
        for b in buf:
            exp = (b >> 3) & 0x0F
            self.assertEqual(exp, 6, f"fp8 exp byte={b:#x} is {exp}, expected 6")


class TestStdlibReaders(unittest.TestCase):
    def test_int8_round_trip(self) -> None:
        m, buf = prng.gen_matrix_stdlib(seed=99, dtype="int8", rows=4, cols=5)
        self.assertEqual(len(m), 4)
        self.assertEqual(len(m[0]), 5)
        for row in m:
            for v in row:
                self.assertTrue(-128 <= v <= 127)
        # Round-trip via writer.
        buf2 = prng.write_matrix_bytes_stdlib(m, "int8")
        self.assertEqual(buf, buf2)

    def test_int32_round_trip(self) -> None:
        m, buf = prng.gen_matrix_stdlib(seed=99, dtype="int32", rows=3, cols=4)
        for row in m:
            for v in row:
                self.assertTrue(-(2**31) <= v < 2**31)
        buf2 = prng.write_matrix_bytes_stdlib(m, "int32")
        self.assertEqual(buf, buf2)

    def test_fp64_round_trip(self) -> None:
        m, buf = prng.gen_matrix_stdlib(seed=99, dtype="fp64", rows=3, cols=4)
        buf2 = prng.write_matrix_bytes_stdlib(m, "fp64")
        self.assertEqual(buf, buf2)

    def test_unsupported_dtype_raises(self) -> None:
        with self.assertRaises(ValueError):
            prng.read_matrix_stdlib(b"\x00" * 8, "bf16", 2, 2)


class TestDigest(unittest.TestCase):
    def test_digest_shape(self) -> None:
        d = prng.matrix_digest(b"\x00" * 32)
        self.assertTrue(d.startswith("sha256:"))
        self.assertEqual(len(d), len("sha256:") + 64)

    def test_digest_same_bytes_same_value(self) -> None:
        self.assertEqual(prng.matrix_digest(b"abc"), prng.matrix_digest(b"abc"))
        self.assertNotEqual(prng.matrix_digest(b"abc"), prng.matrix_digest(b"abd"))


if __name__ == "__main__":
    unittest.main()
