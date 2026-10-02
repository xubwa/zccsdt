# Benchmarks

Wall time per CCSDT iteration of zccsdt and CFOUR, and where the time goes.

## Closed-shell spin-adapted RCCSDT, 8 cores

Fully spin-adapted closed-shell CCSDT on spatial orbitals (`zccsdt.sa.rccsdt`,
design notes in `docs/spin_adapted.md`): RHF integrals in spatial form and
NOSA T3 with one v^3 cube per sorted occupied triple.  This is the RCCSDT path
for nonrelativistic closed shells.  CFOUR `ncc` is spin-adapted the same way
(NOSA), so the two codes solve the same equations in the same representation.

**Setup.**

- **Same cores.** `bench/slurm_compare.sh` with `ZC_SA=1` runs zccsdt and
  then CFOUR `ncc` on the same 8 cores.  Here: c571 (2 x Xeon Gold 6248R),
  cores 0-7 pinned, one comparison at a time, the rest of the node idle.
- **What is timed.** zccsdt (8963bdb; e9b32a7 has the same engine): mean of
  3 iterations after one warm-up iteration (`bench/profile_sa.py`; integral
  blocks transformed from the AO integrals, no n^4 array).  CFOUR: the
  per-iteration wall times of iterations 2-5.
- **Calculations.** All-electron, C1 symmetry.

| System | o, v (spatial) | zccsdt | CFOUR `ncc` | `ncc` / zccsdt | arithmetic ratio (model) |
|---|---|---|---|---|---|
| H2O/cc-pVTZ | 5, 53 | 0.47 s | 1.00 s | 2.1 | 2.6-3.2 |
| N2/cc-pVTZ | 7, 53 | 1.03 s | 3.81 s | 3.7 | 2.4-3.1 |
| H2S/cc-pVTZ | 9, 53 | 2.11 s | 7.14 s | 3.4 | 2.3-3.0 |
| CH2O/cc-pVTZ | 8, 80 | 6.43 s | 41.1 s | 6.4 | 2.4-3.0 |
| H2O/cc-pVQZ | 5, 110 | 5.11 s | 17.0 s | 3.3 | 2.8-3.1 |

The arithmetic ratio is the FMA count of the pp ladder and ring terms, `ncc`
model over the spin-adapted engine (`bench/flop_counts.py`).

Larger systems, same procedure on c233 (same CPUs; the node was shared, with
a 1-minute load of 4-24 during the runs), zccsdt 0154ff2 with the T3 DIIS
vectors in the scratch file:

| System | o, v | T3 vector | zccsdt | CFOUR `ncc` | `ncc` / zccsdt | arithmetic ratio (model) | Memory: zccsdt anonymous, `ncc` RSS |
|---|---|---|---|---|---|---|---|
| C2H4/cc-pVTZ | 8, 108 | 1.1 GB | 19.4 s | 135.3 s | 7.0 | 2.4-2.9 | 7.6 GB, 5.0 GB |
| C6H6/cc-pVDZ, frozen core | 15, 93 | 4.3 GB | 85.7 s | 455.7 s | 5.3 | 2.2-2.9 | 13.7 GB, 14.1 GB |

- `ncc` keeps 3.5 of the 8 cores busy for C2H4 (477 s CPU per iteration) and
  7.0 for benzene (3189 s); in core-seconds the ratios are 3.1 and 4.6.
- At CFOUR's geometry (see "Convergence and energies") the correlation
  energies of the first two iterations, before the two DIIS schemes diverge,
  agree with `ncc` to 4.4e-10 and 2.6e-10 Eh for benzene, and the SCF energy
  to 6e-12 Eh.
- Benzene, where the time goes (85.7 s): ring GEMMs 31.5 s and pp ladder GEMMs
  22.6 s (both about 515 GFLOP/s), T3 -> R1/R2/Wv/Wo 7.0 s, drive + hh ladder
  7.1 s, pair gather 4.3 s, Jacobi step and DIIS 4.3 s, ring scatter 2.3 s.

**Reading the numbers.**

- **zccsdt is 3.3-3.7 times faster than `ncc` from N2/cc-pVTZ upwards**,
  somewhat above the arithmetic ratio.  The engine does about a third of the
  arithmetic of `ncc` for the two dominant terms and runs it at a slightly
  higher rate.  The pp ladder uses
  (d,e)-symmetric / antisymmetric packed GEMMs, and blocks with a repeated
  occupied index take less work.  The ring is four GEMM terms per occupied
  pair.
- **CH2O/cc-pVTZ:** `ncc` uses 150 s of CPU time per 41 s iteration, so only
  3.7 of the 8 cores are busy (6.5-7.2 for the other systems).  In
  core-seconds the ratio is 2.9, as the model predicts.
- **H2O/cc-pVTZ** is small (the dominant terms are about 50 GFLOP), so the rest
  of the iteration (DIIS, intermediates, dressing, Python-level work) is a
  large share of its 0.47 s.
- **Against the spin-orbital RCCSDT** (next section), the spin-adapted engine
  is 2.2 times faster for H2O/cc-pVQZ (5.1 s against 11.1 s on the same
  cores).
- **Memory.** These runs kept the DIIS vectors in memory (peak RSS 6.1 GB for
  H2O/cc-pVQZ in this 4-iteration run; a full DIIS subspace of 8 holds 16
  T3-sized vectors).  `ncc` keeps T3 on disk (3.5 GB RSS).  The DIIS vectors
  can now live in a scratch file at almost no cost, see "DIIS storage" below.

**Convergence and energies.** H2O/cc-pVTZ converged tightly with both
programs: they need the same number of iterations.  |E - E_conv| < 1e-8 Eh is
reached after 11 (zccsdt) and 13 (`ncc`) iterations, and < 1e-10 after 22 and
19.  At the same geometry the SCF energies agree to 2e-12 Eh and the
converged CCSDT energies to 4e-11 Eh.

The same geometry needs care: CFOUR converts Angstrom to bohr with
0.5291771 and PySCF with 0.52917721092, so one Cartesian input in Angstrom
gives geometries that differ by 2e-7.  That shifts the H2O/cc-pVTZ SCF energy
by 1.4e-8 Eh (3.4e-8 Eh for benzene/cc-pVDZ) and the CCSDT energy by 4.5e-10
Eh.  `bench/profile_sa.py` now uses CFOUR's constant; the timings above were
taken with PySCF's.

### Where the time goes (H2O/cc-pVQZ, 8 cores, 5.1 s)

Timers of `bench/profile_sa.py` (c571, cores 0-7, 8963bdb):

| Step | Time | Comment |
|---|---|---|
| pp ladder | 2.02 s | GEMM 1.84 s (0.97 TFLOP, about 525 GFLOP/s; large-DGEMM rate on these cores about 583), pack / unpack 0.18 s |
| pair pass | 2.00 s | ring GEMMs 1.07 s (0.44 TFLOP), T3 -> R1/R2/Wv/Wo 0.48 s, gather 0.27 s, scatter 0.15 s |
| drive + hh ladder (k pass) | 0.41 s | scatter 0.15 s, gather 0.09 s, two GEMMs 0.16 s |
| Jacobi step, DIIS | 0.26 s | |
| intermediates | 0.19 s | |
| pp operator | 0.13 s | one pass over the packed bare vvvv |
| CCSD part, null projection, dressing | 0.15 s | |

### DIIS storage: memory or scratch file

As in PySCF, the large parts of the DIIS vectors (here T3) are stored in an
HDF5 file unless `incore_complete=True`: `diis_file` if given, else a
temporary file in the scratch directory (`ZCCSDT_TMPDIR`, else PySCF's
`lib.param.TMPDIR`); parts below 1e7 elements stay in memory.  The file's
datasets are accessed through `numpy.memmap`, so the DIIS kernels work on the
page cache without copies and the operating system pages the vectors out and
in when memory is short (`zccsdt/cc/diis.py`).

H2O/cc-pVQZ (T3 vector 320 MB), 12 iterations, DIIS subspace 8, c233 (same
CPUs as c571, scratch on a local NVMe, Intel P4510: 2.4 GB/s sequential read),
8 cores:

| DIIS vectors | Memory limit | s/iteration, full subspace | Anonymous memory | File-backed pages |
|---|---|---|---|---|
| in memory (`incore_complete=True`) | none | 5.4-5.5 | 9.7 GB | -- |
| scratch file (default) | none | 5.4 | 4.6 GB | 5.1 GB (reclaimable) |
| scratch file (default) | 6 GB | 8.2 | 4.6 GB | 1-2 GB resident |
| in memory | 6 GB | killed (out of memory) | | |

- **With enough memory the scratch file costs nothing measurable:** the dot
  products and the extrapolation take the same time as in memory (0.16 s and
  0.13 s per iteration with 8 vectors), and building the new vector in mapped
  file pages costs 0.06 s more.  The anonymous memory is halved; what is left
  (4.6 GB) is the engine's own working arrays.
- **Under a memory limit** (a job step with `--mem=6G`) the run completes with
  the kernel paging the vectors: 14 T3-sized reads and 2 writes per iteration
  (5.1 GB) at about 1.8 GB/s, near the speed of the device.  The same run with
  the vectors in memory is killed.
- An iteration with a full subspace of 8 takes about 0.3 s longer than the
  first iterations timed in the table above (5.4 s against 5.1 s): the DIIS
  step grows from 0.2 s to 0.4 s.
- Explicit reads and writes (what `h5py` does) would cost more here even
  from the page cache: 112 ms per 320 MB read and 129 ms per write, about
  1.8 s per iteration, against 21 ms for a pass over the mapped vector.

### Optimization history (H2O/cc-pVQZ spin-adapted RCCSDT, s/iteration)

c571, cores 0-7 pinned.

| Version | Time | Change |
|---|---|---|
| 1e56b2b | 7.4-7.6 | first engine with C kernels: fused pair gather, packed pp ladder, OpenBLAS kept single-threaded |
| 6e08bf9 | 6.43 | x-major tiled gather; T3 -> R1/R2/W as stacked GEMMs; F_vv in the ring operator; fused Jacobi / DIIS step |
| b49a7f0 | 5.45 | dressing from bare blocks; parallel permutations in `contract`; Wv in the drive layout; drive merged with the hh ladder; batched scatters |
| f731460 | 5.43 | integral blocks from AO integrals (peak memory 6.1 -> 4.8 GB) |
| 37e26ce | 5.40 | slab pair gather |
| 8963bdb | 5.12 | work arrays kept between iterations |

## CCSDT(Q), closed shell, 8 cores

The perturbative quadruples correction of `zccsdt.sa.rccsdt_q` (design notes
in `docs/spin_adapted.md`) against CFOUR `ncc` (CALC=CCSDT(Q)), which is
spin-adapted in the same way.

**Setup.**  c233 (2 x Xeon Gold 6248R), 8 cores of one socket, zccsdt 0212757.
zccsdt: wall time of the (Q) step (`bench/profile_sa.py` with `ZC_Q=1`).
`ncc`: its own "Non-iterative calculation of CCSDT(Q)" wall time.  The
cc-pVTZ runs of both programs used cores 0-7, one after the other; the
cc-pVQZ runs used cores 24-31.  The node was shared with other jobs (1-minute
load 3-20); `ncc` for H2O and N2 was run twice, at load 10-20 and at load
6-8, and its times agreed to 3%.

| System | o, v | Quadruples | zccsdt (Q) | `ncc` (Q) | `ncc` / zccsdt | zccsdt CCSDT iteration | Memory: zccsdt, `ncc` (RSS) |
|---|---|---|---|---|---|---|---|
| H2O/cc-pVTZ | 5, 53 | 45 | 5.2 s | 57.4 s | 11 | 0.45 s | 1.1 GB, 10.1 GB |
| N2/cc-pVTZ | 7, 53 | 161 | 18.5 s | 282.5 s | 15 | 1.06 s | 1.2 GB, 17.7 GB |
| H2S/cc-pVTZ | 9, 53 | 414 | 48.6 s | 613.1 s | 13 | 2.18 s | 1.3 GB, 23.2 GB |
| H2O/cc-pVQZ | 5, 110 | 45 | 103 s | 3837 s | 37 | 5.4 s | 12 GB, 22.0 GB |
| C6H6/cc-pVDZ, frozen core | 15, 93 | 2835 | 3223 s | not run | | 87.7 s | 6.5 GB + T3 (4.3 GB) |

The zccsdt memory is the peak anonymous memory of the (Q) step at cc-pVTZ and
the size of its work arrays (10 v^4 and smaller ones) for the two larger
cases.

- **Energies.**  With CCSDT converged in both programs (to different
  thresholds), the (Q) corrections agree to 3e-11 - 2e-10 Eh:

  | System | zccsdt | `ncc` |
  |---|---|---|
  | H2O/cc-pVTZ | -0.000367644522 | -0.000367644322 |
  | N2/cc-pVTZ | -0.001823317881 | -0.001823317657 |
  | H2S/cc-pVTZ | -0.000339324771 | -0.000339324630 |
  | H2O/cc-pVQZ | -0.000429291177 | -0.000429291148 |

  Against MRCC at identical geometries and tight convergence the agreement is
  1e-11 Eh (`bench/refs`).
- **Cost.**  The (Q) step takes as long as 12-22 CCSDT iterations of zccsdt
  for the small molecules and 37 for benzene; for `ncc` it is 57-86 of its
  own iterations at cc-pVTZ and 226 at cc-pVQZ, where it keeps only 3 of its
  8 cores busy (11522 s CPU in 3837 s).
- **Benzene.**  CCSDT converged in 14 iterations (correlation energy
  -0.859518675 Eh, energy threshold 1e-7); E[Q] = -0.000887989 Eh and
  E(Q) = -0.002741990 Eh in 54 minutes.  `ncc` was not run: scaled from its
  H2S time with o^4 v^5 it would take about a day.
- **Memory.**  The work arrays are 10 v^4 (0.6 GB at v = 53, 6.0 GB at v = 93,
  11.7 GB at v = 110), besides T3.  `ncc` needs 9-18 times more at cc-pVTZ
  and about twice as much at cc-pVQZ.

### Are both programs on 8 cores, and does the idle rest of the node matter?

- **Affinity.**  Every pinned run records its affinity list (0-7) in its log.
  `bench/audit_threads.sh` samples all threads of a run: for zccsdt
  (N2/cc-pVTZ, CCSDT + (Q)) and for `ncc` (H2O/cc-pVTZ) every thread is
  allowed on CPUs 0-7 only and was only seen running there.  CPU time over
  wall time is 6.4 cores for that zccsdt run (7.6 for the benzene run) and
  7.1 for `ncc`.  The node has 48 physical cores and no hyperthreading.
- **Fully loaded node.**  Six copies of the same job at once, one per group of
  8 cores (all 48 cores busy), N2/cc-pVTZ:

  | | alone on 8 cores | 6 jobs x 8 cores | slowdown |
  |---|---|---|---|
  | zccsdt CCSDT iteration | 1.06 s | 1.29-1.38 s | 1.25 |
  | `ncc` CCSDT iteration | 3.81 s | 4.94-5.00 s | 1.30 |
  | zccsdt (Q) | 18.5 s | 27.4-27.7 s | 1.5 |
  | `ncc` (Q) | 282.5 s | 522-526 s | 1.9 |

  Both programs lose speed when their neighbours compete for memory
  bandwidth and cache, `ncc` at least as much as zccsdt: the ratio
  `ncc` / zccsdt is 3.6 -> 3.7 for the CCSDT iteration and 15 -> 19 for (Q).
  The absolute times in the tables of this document are those of an
  otherwise lightly loaded node.

### Where the time goes

| Step | H2O/cc-pVQZ (103 s) | C6H6/cc-pVDZ (3223 s) | Comment |
|---|---|---|---|
| T3 x `<vv\|vo>` products | 23.3 s | 706 s | stacked GEMMs (K = 3v or 4v) at 540-570 GFLOP/s |
| I x T2 products | 21.8 s | 732 s | v^5 FMA each; bound by memory traffic (each reads one v^4 array and writes or updates another), about half the DGEMM rate |
| energy kernel | 22.1 s | 725 s | one pass over six v^4 accumulators per quadruple, 3.4 ns per element; the rows of a tile are scattered, so the pass is bound by memory latency |
| T3 x `<ov\|oo>`, T2 x G, gathers | 16.4 s | 899 s | small-K products (K = 2o .. 4o); each is a pass over a v^4 accumulator |
| intermediates I | 15.4 s | 129 s | 2 v^5 per build: once per pair (i, j) and twice per triple (i, j, k) |
| stacking | 0.6 s | 28 s | |

Only the T3 x `<vv|vo>` products run at the DGEMM rate; everything else is
passes over v^4 arrays.  For 15 occupied orbitals the o-contracted terms take
28% of the time.

### Optimization history (H2O/cc-pVTZ and H2O/cc-pVQZ, (Q) step, seconds)

| Version | cc-pVTZ | cc-pVQZ | Change |
|---|---|---|---|
| 37a52dc | 244 | | one product and one transposed v^4 add per term (numpy) |
| 47ecd0d | 6.9 | 132 | cyclic accumulator layouts, stacked GEMMs, symmetrized intermediates, tiled energy kernel (cc-pVQZ: with the factorized metric already) |
| 2591f48 | 6.8 | | base-ordered tile buffers in the kernel, intermediates without a transposed copy |
| 2c26c97 | 5.4 | | exchange symmetry of quadruples with a repeated index |
| e2314da | 5.2 | 103 | coinciding tiles copied in cache |

## RCCSDT / UCCSDT on R/U-form integrals, 8 cores

These runs use the nonrelativistic code path: integrals in R (RHF) or U
(UHF) form, spin-blocked T3 (only Ms-conserving blocks) and, for closed
shells, the alpha <-> beta spin flip (only T3 blocks with 0 or 1 beta
indices are stored and computed; RCCSDT).  For closed shells this
spin-orbital RCCSDT is superseded by the spin-adapted code above.

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
