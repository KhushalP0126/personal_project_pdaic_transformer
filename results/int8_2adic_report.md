# INT8 2-adic Hardware Dry-Lab

- Generated UTC: `2026-05-04T19:11:52Z`
- Precision: `r=8`
- Modulus: `2^8 = 256`

## Exhaustive Multiply Check

| Check | Count |
|---|---:|
| Pair count | 65536 |
| Unsigned INT8 mismatches | 0 |
| Signed low-byte mismatches | 0 |
| Signed wide low-byte mismatches | 0 |
| Signed integer semantic mismatches | 48895 |
| 2-adic valuation mismatches | 0 |
| Hensel multiplication mismatches | 0 |

Unsigned INT8 wraparound matches truncated 2-adic multiplication when the low 8 bits are retained.
Signed low-byte products still match modulo 256, but signed integer semantics do not match unsigned 2-adic residues.
The tile must avoid signed interpretation in custom distance and threshold logic.

## Syscall Distance Sanity Map

| Left | Right | shared low-bit exponent | distance |
|---|---|---:|---|
| `read` | `write` | 2 | `2^-2` |
| `read` | `open` | 3 | `2^-3` |
| `mmap` | `brk` | 2 | `2^-2` |
| `socket` | `connect` | 2 | `2^-2` |
| `read` | `socket` | 1 | `2^-1` |
| `mmap` | `socket` | 0 | `2^-0` |

The sample mapping puts behaviorally related calls on the same low-order 2-adic branch.

## 2x2 Dot Product Mod 256

Left matrix: `[[3, 5], [128, 255]]`
Right matrix: `[[7, 2], [4, 9]]`

| Row | Col 0 | Col 1 |
|---:|---:|---:|
| 0 | 41 | 51 |
| 1 | 124 | 247 |
