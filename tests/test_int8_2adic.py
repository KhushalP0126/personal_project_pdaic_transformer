from __future__ import annotations

import unittest

from padic_transformer.int8_2adic import (
    hensel_mul_truncated,
    matmul2x2_mod256,
    two_adic_distance_exponent,
    v2_saturated_array,
    v2_saturated,
    verify_numpy_int8,
    wrap_uint,
)


class Int82AdicTests(unittest.TestCase):
    def test_wrap_and_valuation(self) -> None:
        self.assertEqual(wrap_uint(-1, 8), 255)
        self.assertEqual(v2_saturated(0, 8), 8)
        self.assertEqual(v2_saturated(128, 8), 7)
        self.assertEqual(v2_saturated(3, 8), 0)
        self.assertEqual(v2_saturated_array([0, 1, 2, 4, 128], 8).tolist(), [8, 0, 1, 2, 7])

    def test_distance_exponent_uses_low_bits(self) -> None:
        self.assertEqual(two_adic_distance_exponent(0x00, 0x04, 8), 2)
        self.assertEqual(two_adic_distance_exponent(0x00, 0x01, 8), 0)

    def test_hensel_multiply_matches_modulo_int8(self) -> None:
        for left in [0, 1, 3, 7, 15, 31, 127, 128, 255]:
            for right in [0, 1, 5, 9, 17, 64, 129, 255]:
                self.assertEqual(hensel_mul_truncated(left, right, 8), (left * right) & 0xFF)

    def test_exhaustive_numpy_int8_matches_2adic_low_byte(self) -> None:
        report = verify_numpy_int8(8)
        self.assertEqual(report.unsigned_mismatches, 0)
        self.assertEqual(report.signed_low_byte_mismatches, 0)
        self.assertEqual(report.signed_wide_low_byte_mismatches, 0)
        self.assertGreater(report.signed_integer_semantic_mismatches, 0)
        self.assertEqual(report.valuation_mismatches, 0)
        self.assertEqual(report.hensel_mul_mismatches, 0)

    def test_2x2_matmul_wraps_mod256(self) -> None:
        left = [[0x03, 0x05], [0x80, 0xFF]]
        right = [[0x07, 0x02], [0x04, 0x09]]
        self.assertEqual(matmul2x2_mod256(left, right), [[41, 51], [124, 247]])


if __name__ == "__main__":
    unittest.main()
