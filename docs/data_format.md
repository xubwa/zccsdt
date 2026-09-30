# zccsdt HDF5 integral format (version 1)

One file per calculation. It contains the active-space (frozen orbitals
already removed) spin-orbital integrals of a single-determinant reference. The
file is read by `zccsdt.h5io.load` and written by `zccsdt.h5io.save`. Other
programs (e.g. a CFOUR converter) write the same layout.

## Attributes (root group)

| name | type | meaning |
|---|---|---|
| `format_version` | int | `1` |
| `dtype` | str | `float64` or `complex128` |
| `nocc` | int | number of active occupied spin orbitals |
| `e_hf` | float | total reference (SCF) energy, including frozen core and nuclear repulsion |
| `e_nuc` | float | nuclear repulsion energy |
| `source` | str | free text identifying the generator |

## Datasets

| name | shape | meaning |
|---|---|---|
| `fock` | (n, n) | MO Fock matrix of the active space, occupied first. It may be non-diagonal (non-canonical orbitals). |
| `mo_energy` | (n,) | diagonal of `fock` (real); used for denominators |
| `eri/<block>` | see below | antisymmetrized physicist integrals `<pq\|\|rs> = <pq\|rs> - <pq\|sr>` |

Blocks: `oooo, ooov, oovv, ovov, ovvo, ovvv, vvvv` (letters give the
spaces of p, q, r, s). This is the PySCF GCCSD convention. The other blocks
follow by antisymmetry and hermiticity, `<pq||rs> = <rs||pq>^*`.

Every block is stored as a 2-D matrix (bra pair, ket pair):
- A like-space pair (`oo` or `vv`) is packed over `p<q`, in row-major order
  (the order of `numpy.triu_indices(n, 1)`).
- A mixed pair (`ov`, `vo`) is stored in full, flattened row-major.

For example, `eri/vvvv` has shape `(v(v-1)/2, v(v-1)/2)` and `eri/ovvv` has
shape `(o*v, v(v-1)/2)`.

Complex numbers use the native HDF5 compound type written by h5py, which is
interleaved re/im and matches Fortran `complex(8)` and C99 `double complex`.
