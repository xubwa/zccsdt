# Benchmarks

Wall time per CCSDT iteration of zccsdt and CFOUR, and where the time goes.

## RCCSDT / UCCSDT on R/U-form integrals, 8 cores

These runs use the nonrelativistic code path: integrals in R (RHF) or U
(UHF) form, spin-blocked T3 (only Ms-conserving blocks) and, for closed
shells, the alpha <-> beta spin flip (only T3 blocks with 0 or 1 beta
indices are stored and computed; RCCSDT).

**Setup.**

- **Same allocation.** `bench/slurm_compare.sh` runs zccsdt and CFOUR back
  to back on the same 8 cores of one job (`--cpus-per-task=8 --mem=30G`;
  the partition allows 4000 MB per core, so a larger `--mem` silently adds
  cores, which inflated earlier "8-core" numbers).
- **Node load.** All six jobs ran at the same time on one node (c178, 2 x
  Xeon Gold 6248R), so the node was fully loaded and memory bandwidth was
  shared.
- **What is timed.** zccsdt: mean of 3 iterations after one warm-up
  iteration (version 2322512). CFOUR: the per-iteration wall times of
  iterations 2-5.
- **Calculations.** All-electron, C1 symmetry.

| System | o, v (spatial) | zccsdt | CFOUR | zccsdt / CFOUR | arithmetic ratio (model) |
|---|---|---|---|---|---|
| H2O/cc-pVTZ, RHF | 5, 53 | RCCSDT 1.27 s | `ncc` 1.03 s | 1.24 | 0.57-0.71 |
| N2/cc-pVTZ, RHF | 7, 53 | RCCSDT 2.80 s | `ncc` 3.97 s | 0.71 | 0.62-0.79 |
| H2S/cc-pVTZ, RHF | 9, 53 | RCCSDT 5.51 s | `ncc` 7.02 s | 0.78 | 0.64-0.85 |
| H2O/cc-pVQZ, RHF | 5, 110 | RCCSDT 10.9 s | `ncc` 16.1 s | 0.68 | 0.62-0.71 |
| NH2/cc-pVTZ, UHF (doublet) | 5+4, 53+54 | UCCSDT 1.51 s | `ecc` 9.4 s | 0.16 | -- |
| O2/cc-pVTZ, UHF (triplet) | 9+7, 51+53 | UCCSDT 5.70 s | `ecc` 56.1 s | 0.10 | -- |

The arithmetic ratio is the FMA count of the dominant terms (pp ladder and
ring), RCCSDT over the `ncc` model, from `bench/flop_counts.py`.

Same node, pinned cores (c571, otherwise idle), H2O/cc-pVQZ:

| Cores | zccsdt RCCSDT | `ncc` |
|---|---|---|
| 8 on one socket (`taskset -c 0-7`) | 11.1 s (b00ac29) | 17.0 s |
| 4 + 4 on two sockets | 16.2 s (2322512) | 20.9 s |

Peak memory for H2O/cc-pVQZ: zccsdt 9.7 GB, `ncc` 3.5 GB.

**Reading the numbers.**

- **zccsdt is 1.3-1.5 times faster than `ncc` from N2/cc-pVTZ upwards.**
  The measured time ratio sits inside the arithmetic ratio, so per unit of
  arithmetic the two codes are about equally efficient. At H2O/cc-pVTZ the
  dominant terms are small (about 90 GFLOP), and the rest of the iteration
  (intermediates, dressing, CCSD part, about 0.6 s, much of it Python-level
  work on small tensors) is half of the time.
- **RCCSDT does less arithmetic than the `ncc` model only for few occupied
  orbitals.** `ncc` stores T3 as i<=j<=k blocks, each a full abc cube, and
  the blocks with repeated indices are a large fraction when o is small.
  The ladder ratio RCCSDT / `ncc` is 0.72 at o = 5, about 1 at o = 10-15,
  and tends to 4/3 for large o (o^3 v^5 x 2/3 against o^3 v^5 / 2). The
  ring ratio stays at or below 1.
- **Against the spin-integrated UHF code** (`ecc`, REF=UHF), UCCSDT is 6-10
  times faster. `ecc` uses about 1.5 of the 8 cores.

### Where the time goes (H2O/cc-pVQZ RCCSDT, 8 cores, 11.1 s)

Timers of `bench/profile_iter.py` (c571, cores 0-7, b00ac29):

| Step | Time | Comment |
|---|---|---|
| pp ladder | 3.92 s | 2.08 TFLOP, 530 GFLOPS overall (the GEMM shape alone runs at about 575; 8-core AVX-512 peak about 650) |
| pass 2: ring + drive | 2.91 s | ring GEMM 1.56 s (675 GFLOP), reduction 0.45 s, gather 0.41 s, drive 0.24 s |
| pass 1: T3 -> T1/T2, W | 1.13 s | half of the spin combinations (flip); GEMM 0.49 s, gather 0.31 s |
| W_vvvo, W_vooo, effective Fock | 1.06 s | |
| CCSD part (T1-dressed) | 0.71 s | |
| T1 dressing (incl. ladder operator) | 0.69 s | one pass over the bare vvvv blocks |
| DIIS, T3 update, rest | 0.6 s | |

### Optimization history (H2O/cc-pVQZ RCCSDT, s/iteration)

Each step was measured on c571 with cores 0-7 pinned, unless marked
(slurm).

| Version | Time | Change |
|---|---|---|
| UCCSDT blocks on GHF-ordered integrals | 33.4 (slurm, 16 cores available) | |
| R-form integrals, RCCSDT | 19.9 | |
| W_vvvo pieces as spin-block GEMMs, Fock term folded into a GEMM, transpose-add kernel | 17.6 | c9c7a6a |
| one-pass T1/T2-dressed ladder operator, F_vv folding kernel, dressed vovv dropped | 15.5 | |
| pass 1 on half of the spin combinations (flip) | 13.8 | 2476c2b |
| pass-1 batches sized for spin blocks | 13.3 | 0d94abf |
| dressed ovvv by GEMMs, cached bare vvvo | 12.5 | a93a678 |
| engine workspace (no page faults), precombined gather tables | 11.7 | |
| loop-inverted pass-2 reduction | 11.4 | 2322512 |
| CCSD residual on packed pairs | 11.1 | b00ac29 |

## Spin-orbital (spinor-capable) code, 48 cores

The general spin-orbital path (no spin symmetry, as needed for spinors),
on GHF / X2C integrals.

### Setup

- **Hardware.** One exclusive node: 2 x Intel Xeon Gold 6248R, 48 cores,
  192 GB RAM, from the `parallel` partition.
- **Threads.** 48 for every program (`OMP_NUM_THREADS=MKL_NUM_THREADS=48`).
- **Calculations.** All-electron, no point-group symmetry (C1), cc-pVTZ.
- **What is timed.** Wall time per CC iteration, excluding integral
  generation and transformation. For CFOUR, the per-iteration lines of
  `xncc` / `xecc`, or the `xvcc` time divided by the number of iterations.
- **zccsdt.** `bench/profile_iter.py`, launched through
  `bench/slurm_profile.sh`. Uses the MKL build of the C kernels and
  `OMP_WAIT_POLICY=PASSIVE`. Every number is the mean of iterations 1-3.
  Run-to-run variation between nodes is about 20%.
- **CFOUR.** cfour-x2cso build: ifort with ILP64 threaded MKL.
  `bench/cfour/slurm_cfour.sh`, 6 iterations.
- **Measured GEMM peak** of one node (large square matrices): about
  2.4 TFLOPS (DGEMM, OpenBLAS), about 2.4-2.8 TFLOPS (ZGEMM, MKL).

Orbital counts: H2O/cc-pVTZ has 116 spin orbitals (o = 10, v = 106);
N2/cc-pVTZ has 120 (o = 14, v = 106).

### CCSDT iteration time

| Program (representation) | H2O/cc-pVTZ | N2/cc-pVTZ | H2O/cc-pVQZ |
|---|---|---|---|
| CFOUR `ncc`, REF=RHF (spin-adapted) | 0.60 s | 1.78 s | 11.7 s |
| CFOUR `ecc`, REF=RHF | 6.8 s | 19.8 s | -- |
| CFOUR `ecc`, REF=UHF (spin-integrated) | 18.7 s | 57 s | 173 s |
| CFOUR `vcc`, REF=UHF (spin-integrated) | 16 s | 56 s | -- |
| **zccsdt, real spin-orbital** | **3.2-3.6 s** | **6.7 s** | **34.6 s** |
| **zccsdt, complex spinor (X2C)** | **6.6 s** | **13.7-16.9 s** | **100.9 s** |

H2O/cc-pVQZ has 230 spin orbitals (o = 10, v = 220). It is run with
block-wise integrals and packed-only vvvv (`ZC_PACKED=1`). Peak memory is
36.5 GB (real) and 59.6 GB (complex); integral generation takes about
3 minutes and is not included.

Peak memory of zccsdt: H2O 6.2 GB (real) and 17.8 GB (complex); N2 8.3 GB
(real) and 20.3 GB (complex).

### Representation factor: theoretical vs measured

zccsdt works with spin orbitals and uses no spin symmetry, because spinors
have none. `bench/flop_counts.py` counts the minimal FMA of the two
dominant terms (pp ladder o^3 v^5, ring o^4 v^4) in three representations:

- **spin-orbital:** what zccsdt does.
- **Sz-blocked:** only Ms-conserving blocks, as in a spin-integrated UHF
  code (`ecc`/`vcc` with REF=UHF).
- **ncc model:** CFOUR `ncc` (RHF, spin-adapted), counted from its source
  code:
  - T3 is stored as i<=j<=k blocks (not all equal), each a full abc cube.
  - The pp ladder is 3 GEMMs of v^2 x v^2 x v per block (the MAGIC_CYCLE
    states with constraints 1, 4 and 7).
  - The ring terms are 9 (wmaei) plus 13-26 (wmaie) contractions of o v^4
    per block. Part of the wmaie work is skipped by an index-order test,
    hence the range.

Here o and v are spatial orbitals. Cost ratios (zccsdt / other):

| System | spin-orbital / Sz-blocked | spin-orbital / ncc (theory) | zccsdt / ncc (measured wall time) |
|---|---|---|---|
| H2O/cc-pVTZ | 7.5 | 8.5-10.6 | 5.3-6.0 |
| N2/cc-pVTZ | 7.6 | 9.4-12.1 | 3.8 |
| H2O/cc-pVQZ | 7.5 | 9.3-10.6 | 3.0 |

Asymptotically (large o) the ladder ratio spin-orbital / ncc approaches
256/12, about 21.

Achieved rate on the dominant terms (FMA x 2 / iteration time):

| System | ncc | zccsdt (real) |
|---|---|---|
| H2O/cc-pVTZ | 0.21-0.26 TFLOPS | 0.38-0.42 TFLOPS |
| N2/cc-pVTZ | 0.21-0.28 TFLOPS | 0.69 TFLOPS |
| H2O/cc-pVQZ | 0.33-0.38 TFLOPS | 1.19 TFLOPS |

Conclusions:

- **The measured slowdown against ncc (3-6x) is 2-3.5 times smaller than
  the arithmetic ratio (9-12x).** Per unit of work, zccsdt is 2-3.5 times
  more efficient than ncc at these sizes, and the advantage grows with the
  basis.
- **Against the spin-integrated codes** (`ecc`/`vcc`, REF=UHF), zccsdt is
  5-8 times faster in wall time, despite doing about 7.5 times more
  arithmetic.
- **The remaining gap to ncc is the price of the spin-orbital
  representation.** For nonrelativistic closed shells an RHF spin-adapted
  code will always win; for spinors (the target of zccsdt) there is no
  such reduction, and CFOUR has no spinor CCSDT.
- **Complex spinor iterations** cost about 2.5-3 times the real ones for 4
  times the arithmetic.

### Where the time goes (N2/cc-pVTZ, complex)

Per-iteration breakdown from the timers in `ccsdt_opt` and the C kernels
(node c054, 16.9 s per iteration in total):

| Step | Time | Comment |
|---|---|---|
| pass 2: ring + drive (C) | 6.2 s | ring GEMM 4.4 s (about 2.0 TFLOPS), antisymmetrizing reduction 0.8 s, gather 0.35 s, drive GEMM about 0.5 s |
| pp ladder (C) | 3.9 s | GEMM about 2.8 TFLOPS |
| pass 1: T3 -> T1/T2, W (C) | 2.4 s | GEMM 1.6 s |
| other intermediates (Python + MKL) | 1.5 s | packed W_vvvo 1.0 s |
| T1 dressing (Python + MKL) | 1.3 s | packed vvvv dressing 0.5 s |
| CCSD part, T1-dressed (Python + MKL) | 0.6 s | |
| DIIS (OpenMP) | 0.3 s | |

The T3 kernels (12.5 s) run their GEMMs at 2-2.8 TFLOPS. The Python-side
remainder is about 3.7 s.

### Optimization history (H2O/cc-pVTZ, real, s/iteration)

| Version | Time | Change |
|---|---|---|
| ported socutils code | does not fit in memory at TZ (full o^3 v^3 arrays) | -- |
| packed T3, numpy | 38 | no full T3 / n^4 tensors |
| [P,d,m,Q] layout, stacked GEMMs | 23 | |
| C gather/reduce kernels | 11.4 | |
| BLAS beta=1 accumulation | 10.2 | |
| T3 passes in C | 9.7 | |
| OMP_WAIT_POLICY=PASSIVE | 5.6 | spinning OpenMP threads halved BLAS throughput |
| MKL (GNU OpenMP layer) for the kernels | 4.4 | one thread pool for BLAS and loops |
| packed W_vvvo and CCSD ladder, MKL for Python GEMMs | 3.8 | |
| pair-antisymmetrize kernel, segment DIIS, merged drive GEMM, T1-dressed CCSD | 3.2-3.6 | N2 complex: 25 -> 16.9 s |

## Correctness references

| Reference | Agreement | Test |
|---|---|---|
| PySCF GCCSD (real and complex-rotated GHF) | 1e-10 | `tests/test_ccsd.py` |
| FCI, 3 electrons (real UHF; X2C spinor) | 1e-9 | `tests/test_ccsdt.py` |
| MRCC and CFOUR ncc/ecc/vcc CCSD and CCSDT, 5 small molecules | 1e-8 (observed about 5e-11) | `tests/test_references.py` |
| Determinant-space brute force, T3 residual element-wise | 1e-12 | `tests/test_ccsdt.py` |
| Invariance under complex occ-occ / vir-vir rotations (RHF -> GHF -> U) | 1e-10 | `tests/test_ccsdt.py` |
