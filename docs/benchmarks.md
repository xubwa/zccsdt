# Benchmarks

## Setup

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

## CCSDT iteration time

| Program (representation) | H2O/cc-pVTZ | N2/cc-pVTZ |
|---|---|---|
| CFOUR `ncc`, REF=RHF (spin-adapted) | 0.60 s | 1.78 s |
| CFOUR `ecc`, REF=RHF | 6.8 s | 19.8 s |
| CFOUR `ecc`, REF=UHF (spin-integrated) | 18.7 s | 57 s |
| CFOUR `vcc`, REF=UHF (spin-integrated) | 16 s | 56 s |
| **zccsdt, real spin-orbital** | **3.2-3.6 s** | **6.7 s** |
| **zccsdt, complex spinor (X2C)** | **6.6 s** | **16.9 s** |

Peak memory of zccsdt: H2O 6.2 GB (real) and 17.8 GB (complex); N2 8.3 GB
(real) and 20.3 GB (complex).

### Representation factor

zccsdt works with spin orbitals and uses no spin symmetry, because spinors
have none. A nonrelativistic spin-integrated code needs only the
Ms-conserving blocks.

The minimal FMA counts of the two dominant terms are computed by
`bench/flop_counts.py`:

| Term | H2O/cc-pVTZ: spin-orbital / Sz-blocked | N2/cc-pVTZ: spin-orbital / Sz-blocked |
|---|---|---|
| pp ladder (o^3 v^5) | 3.94e11 / 5.39e10 (7.3x) | 1.20e12 / 1.60e11 (7.5x) |
| ring (o^4 v^4) | 2.81e11 / 3.63e10 (7.8x) | 1.12e12 / 1.43e11 (7.8x) |

An RHF spin-adapted code such as `ncc` saves several times more again.

What this means for the comparison:

- **Against CFOUR's spin-integrated codes** (`ecc`/`vcc`, REF=UHF; the
  nonrelativistic analogue of a spinor calculation), zccsdt is **5-8 times
  faster in wall time**, while doing about 7.5 times more arithmetic.
- **Against `ncc`**, zccsdt is 4-6 times slower in wall time. The
  representation factor is at least 7.5 (more, because of the RHF spin
  adaptation), so per unit of work zccsdt is ahead.
- **Spinor case.** CFOUR has no spinor CCSDT, so there is no same-condition
  comparison. The complex zccsdt iteration costs about 2.5 times the real
  one for 4 times the arithmetic. The N2 complex iteration (16.9 s) is 1.9
  times the floor of dominant flops at 2.4 TFLOPS (about 9 s).

## Where the time goes (N2/cc-pVTZ, complex)

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

## Optimization history (H2O/cc-pVTZ, real, s/iteration)

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
