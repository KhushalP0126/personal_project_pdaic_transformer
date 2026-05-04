"""INT8 hardware checks for truncated 2-adic arithmetic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Int8Verification:
    precision: int
    pair_count: int
    unsigned_mismatches: int
    signed_low_byte_mismatches: int
    signed_wide_low_byte_mismatches: int
    signed_integer_semantic_mismatches: int
    valuation_mismatches: int
    hensel_mul_mismatches: int


def modulus(r: int) -> int:
    if not 1 <= r <= 16:
        raise ValueError("r must be in [1, 16] for this hardware dry-lab check")
    return 1 << r


def wrap_uint(value: int, r: int = 8) -> int:
    return int(value) % modulus(r)


def v2_saturated(value: int, r: int = 8) -> int:
    """Return min(v_2(value mod 2**r), r)."""
    residue = wrap_uint(value, r)
    if residue == 0:
        return r

    valuation = 0
    while residue & 1 == 0:
        valuation += 1
        residue >>= 1
    return valuation


def v2_saturated_array(values: np.ndarray, r: int = 8) -> np.ndarray:
    """Vectorized saturated 2-adic valuation for bounded unsigned residues."""
    limit = modulus(r)
    residues = np.asarray(values, dtype=np.int64) % limit
    lookup = np.fromiter((v2_saturated(value, r) for value in range(limit)), dtype=np.int16)
    return lookup[residues]


def two_adic_distance_exponent(left: int, right: int, r: int = 8) -> int:
    """Return the saturated exponent k in distance 2**(-k)."""
    return v2_saturated(left - right, r)


def hensel_bits(value: int, r: int = 8) -> list[int]:
    residue = wrap_uint(value, r)
    return [(residue >> idx) & 1 for idx in range(r)]


def bits_to_uint(bits: list[int]) -> int:
    out = 0
    for idx, bit in enumerate(bits):
        out |= (bit & 1) << idx
    return out


def hensel_mul_truncated(left: int, right: int, r: int = 8) -> int:
    """Multiply two 2-adic residues and keep only the low r Hensel digits."""
    a = hensel_bits(left, r)
    b = hensel_bits(right, r)
    coeffs = [0 for _ in range(2 * r)]

    for idx, left_digit in enumerate(a):
        if left_digit == 0:
            continue
        for jdx, right_digit in enumerate(b):
            coeffs[idx + jdx] += left_digit * right_digit

    out: list[int] = []
    carry = 0
    for idx in range(r):
        total = coeffs[idx] + carry
        out.append(total & 1)
        carry = total >> 1
    return bits_to_uint(out)


def exhaustive_hensel_mul_mismatches(r: int = 8) -> int:
    limit = modulus(r)
    mismatches = 0
    for left in range(limit):
        for right in range(limit):
            expected = wrap_uint(left * right, r)
            if hensel_mul_truncated(left, right, r) != expected:
                mismatches += 1
    return mismatches


def verify_numpy_int8(r: int = 8) -> Int8Verification:
    """Compare NumPy INT8/UINT8 behavior to truncated 2-adic multiplication."""
    if r != 8:
        raise ValueError("NumPy INT8 verification requires r=8")

    values = np.arange(256, dtype=np.uint16)
    left, right = np.meshgrid(values, values, indexing="ij")
    theoretical = ((left * right) & 0xFF).astype(np.uint8)

    left_u8 = left.astype(np.uint8)
    right_u8 = right.astype(np.uint8)
    unsigned_tile = np.multiply(left_u8, right_u8, dtype=np.uint8)

    left_i8 = left_u8.view(np.int8)
    right_i8 = right_u8.view(np.int8)
    signed_low_byte = np.multiply(left_i8, right_i8, dtype=np.int8).view(np.uint8)
    signed_wide = left_i8.astype(np.int16) * right_i8.astype(np.int16)
    signed_wide_low_byte = (signed_wide & 0xFF).astype(np.uint8)
    unsigned_wide = left.astype(np.int32) * right.astype(np.int32)

    valuation_expected = v2_saturated_array(theoretical, 8)
    valuation_tile = v2_saturated_array(unsigned_tile, 8)

    return Int8Verification(
        precision=8,
        pair_count=256 * 256,
        unsigned_mismatches=int(np.count_nonzero(unsigned_tile != theoretical)),
        signed_low_byte_mismatches=int(np.count_nonzero(signed_low_byte != theoretical)),
        signed_wide_low_byte_mismatches=int(np.count_nonzero(signed_wide_low_byte != theoretical)),
        signed_integer_semantic_mismatches=int(np.count_nonzero(signed_wide != unsigned_wide)),
        valuation_mismatches=int(np.count_nonzero(valuation_tile != valuation_expected)),
        hensel_mul_mismatches=exhaustive_hensel_mul_mismatches(r),
    )


def matmul2x2_mod256(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    if len(left) != 2 or len(right) != 2:
        raise ValueError("left and right must be 2x2 matrices")
    if any(len(row) != 2 for row in left + right):
        raise ValueError("left and right must be 2x2 matrices")

    out = [[0, 0], [0, 0]]
    for row in range(2):
        for col in range(2):
            total = 0
            for inner in range(2):
                total += wrap_uint(left[row][inner], 8) * wrap_uint(right[inner][col], 8)
            out[row][col] = wrap_uint(total, 8)
    return out


DEFAULT_SYSCALL_MAP = {
    "read": 0b0000_0000,
    "write": 0b0000_0100,
    "open": 0b0000_1000,
    "close": 0b0000_1100,
    "mmap": 0b0000_0001,
    "brk": 0b0000_0101,
    "fork": 0b0000_1001,
    "execve": 0b0000_1101,
    "socket": 0b0000_0010,
    "connect": 0b0000_0110,
    "recvfrom": 0b0000_1010,
}


def syscall_distance_rows(syscall_map: dict[str, int] | None = None) -> list[tuple[str, str, int]]:
    mapping = syscall_map or DEFAULT_SYSCALL_MAP
    pairs = [
        ("read", "write"),
        ("read", "open"),
        ("mmap", "brk"),
        ("socket", "connect"),
        ("read", "socket"),
        ("mmap", "socket"),
    ]
    return [
        (left, right, two_adic_distance_exponent(mapping[left], mapping[right], 8))
        for left, right in pairs
    ]
