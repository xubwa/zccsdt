# Benchmarks

Wall time per CCSDT iteration of zccsdt and CFOUR, and where the time goes.

## What is compared

Unless a table says otherwise, the numbers of this document are

- **wall time**, not CPU time: of one CCSDT iteration (residuals, update and
  DIIS) or of the (Q) step, excluding SCF, integral transformation and setup;
- for the **same calculation** in both programs: same geometry and basis,
  RHF reference, all electrons correlated unless frozen core is stated;
- with the **same threads on the same cores**: `OMP_NUM_THREADS` =
  `MKL_NUM_THREADS` = 8 and both programs pinned to the same 8 cores (checked
  per thread, see the (Q) section).  Eight threads are not always the best
  choice for `ncc`; the first subsection gives it its best number;
- **without point-group symmetry**: CFOUR is run with `SYMMETRY=OFF`.  zccsdt
  has no point-group symmetry, so this is the like-for-like comparison of
  the two implementations, and it is the relevant one for molecules without
  symmetry.  It is *not* what CFOUR does best for the small symmetric
  molecules used here: with symmetry `ncc` is several times faster (below).

CPU time is not a clean measure for either program on several threads
(idle OpenMP threads of `ncc` spin: its CCSDT(Q) step for H2O/cc-pVTZ takes
405 CPU seconds on 8 threads and 300 on one).  The single-thread runs are.

### The `ncc` reference: builds, threading, memory

A factor of several against `ncc` calls for checking the reference first.
The `ncc` source of the reference installation was rebuilt in a scratch copy
with other compilers and flags (`bench/cfour/rebuild_ncc.sh`; same `xvscf`,
`xvtran` and `xintprc`, same threaded MKL; the installation itself is
unchanged).  c233, no point-group symmetry, wall time in seconds, zccsdt
f7d3f51.

**Compiler and flags.**  The reference installation compiles `ncc` with g++
9.3 `-O2`, without `-march`.  H2O/cc-pVTZ; the 8-core jobs one at a time on
cores 0-7, the one-core jobs all at the same time on separate cores of the
other socket:

| `ncc` build | iteration, 8 cores | (Q), 8 cores | iteration, 1 core | (Q), 1 core |
|---|---|---|---|---|
| reference: g++ `-O2` | 0.92 | 57.7 | 6.40 | 308 |
| g++ `-O3 -march=native` | 0.75 | 55.9 | 4.95 | 296 |
| icpc `-O3 -xHost` | 0.78 | 56.8 | 5.11 | 295 |
| another installation on the cluster, icpc `-O3` | 0.81 | 57.2 | 5.37 | 301 |
| zccsdt | 0.48 | 5.4 | 2.26 | 34.9 |

- **Better flags make the CCSDT iteration of `ncc` 1.2-1.3 times faster and
  leave (Q) as it is.**  N2/cc-pVTZ: 3.59 -> 2.72 s per iteration on 8 cores
  and 20.4 -> 15.0 s on one; (Q) 283 -> 278 s and 1064 -> 1108 s.
- **Threading environment: no effect.**  In the g++ builds `ncc`'s OpenMP
  regions run on libgomp while MKL brings libiomp5, two runtimes in one
  process; the icpc builds have libiomp5 only and are not faster.
  `KMP_BLOCKTIME=0` (0.95 and 0.78 s per iteration for the two g++ builds) and
  `OMP_PROC_BIND=close OMP_PLACES=cores` (0.79 s, icpc) change nothing.
- **MRCC as a sanity check** (release of February 2018, a general-order
  program; same node, 8 threads, no symmetry, H2O/cc-pVTZ): 39.5 s per CCSDT
  iteration and 321 s for (Q).  `ncc` is 43-53 times faster per iteration and
  5.6 times faster in (Q), so the reference is not a broken build.

**Memory.**  `ncc` prints the memory it would like to have ("optimal memory
requirement").  For the CCSDT iterations that is small (3.0 GiB for
H2O/cc-pVQZ).  For (Q) it wants T4 and Z4 of all quadruples in memory: 9.0 GiB
for H2O/cc-pVTZ, 30 GiB for N2/cc-pVTZ, 76 GiB for H2S/cc-pVTZ and 167 GiB for
H2O/cc-pVQZ; with less it works through them in hunks.  The (Q) times of the
other sections were taken with `MEMORY_SIZE` = 24 GB, below that optimum for
the last three systems, so they are repeated here with more.

**Number of threads.**  `ncc` assigns the blocks of a contraction to its
threads by a fixed rule on the block indices (`core/contractions.cxx`): T3
block (i, j, k) goes to thread (i + o j + o^2 k) mod n.  If the number of
threads n divides o, the thread depends on one index alone and the work is
very uneven; other combinations are uneven to a lesser degree.  With o = 8
(CH2O, C2H4) and 8 threads one thread gets a third of the blocks, and `ncc`
is 1.7-1.8 times faster with 7 threads: CH2O/cc-pVTZ 17.9 s
per iteration against 30.9 s, C2H4/cc-pVTZ 57.6 s against 104.6 s.  For N2
(o = 7) 6 threads are a little faster than 8 (2.58 s against 2.72 s per
iteration, (Q) 241 s against 272 s).  For o = 5, 9 and 15 the rule is
nearly balanced with 8 threads (the largest share is 1.03-1.13 of the mean
for the T3 blocks); benzene (o = 15) is still a little faster with 7 threads
(275 s against 295 s).  The earlier ratios for CH2O and C2H4 (6.4 and 7.0,
"3.5 of the 8 cores busy") were taken in the unfavourable case.

**Best `ncc` against zccsdt on 8 cores.**  Best build: g++ `-O3
-march=native`.  `ncc` times per iteration are over the converged run (14-16
iterations; CH2O, C2H4, benzene: iterations 2-4).  zccsdt: 8 threads, mean
over a run of the same length (DIIS subspace 8), 2-8% more than the first
iterations timed elsewhere in this document.

| | zccsdt | `ncc`: reference build, 8 threads, 24 GB | best build, 8 threads | best build, best number of threads | best `ncc` / zccsdt |
|---|---|---|---|---|---|
| H2O/cc-pVTZ, iteration | 0.49 | 0.92 | 0.75 | 0.75 (8) | 1.5 |
| N2/cc-pVTZ, iteration | 1.13 | 3.59 | 2.72 | 2.58 (6) | 2.3 |
| H2S/cc-pVTZ, iteration | 2.38 | 6.75 | 4.79 | 4.79 (8) | 2.0 |
| CH2O/cc-pVTZ, iteration | 6.95 | 41.1 | 30.9 | 17.9 (7) | 2.6 |
| C2H4/cc-pVTZ, iteration | 20.6 | 135.3 | 104.6 | 57.6 (7) | 2.8 |
| H2O/cc-pVQZ, iteration | 5.54 | 16.8 | 13.9 | 13.9 (8) | 2.5 |
| C6H6/cc-pVDZ, frozen core, iteration | 87.7 | 455.7 | 295 | 275 (7) | 3.1 |
| H2O/cc-pVTZ, (Q) | 5.4 | 57.7 | 55.9 | 55.9 (8) | 10 |
| N2/cc-pVTZ, (Q) | 18.8 | 283 | 272 (40 GB) | 241 (6; 40 GB) | 13 |
| H2S/cc-pVTZ, (Q) | 49.9 | 613 | 575 (90 GB; 81 GB resident) | 575 (8) | 12 |
| H2O/cc-pVQZ, (Q) | 102 | 3837 | 2330 (100 GB; 80 GB resident) | 2330 (8) | 23 |

Numbers of threads other than 8 were tried where the rule above predicts an
imbalance (N2: 4, 6; CH2O, C2H4: 7) and for benzene (7).  The memory in
parentheses is `ncc`'s `MEMORY_SIZE`, at or above its optimum except for
H2O/cc-pVQZ, where the node cannot provide 167 GiB (3 hunks instead of 9).

One core, best build: H2O/cc-pVTZ 4.95 s against 2.26 s per iteration (2.2)
and 296 s against 34.9 s for (Q) (8.5); N2/cc-pVTZ 15.0 s against 6.27 s (2.4)
and 1108 s against 133 s (8.3).

- **Against the best `ncc` zccsdt is 1.5-3.1 times faster per CCSDT
  iteration and 10-13 times faster in (Q) at cc-pVTZ** (23 times at cc-pVQZ;
  no symmetry, 8 cores).  The ratios in the other sections of this document
  (2.1-7.0 per iteration, up to 37 in (Q)) are against the reference build
  with 8 threads and 24 GB.
- More memory changes the (Q) time of `ncc` by a few percent at cc-pVTZ (N2,
  same build: 278 -> 272 s) and by a factor of 1.6 at cc-pVQZ.  The (Q) step
  of zccsdt needs 1.1-1.3 GB at cc-pVTZ and 14 GB at cc-pVQZ.
- **With point-group symmetry the best `ncc` is ahead for these molecules**:
  0.25, 0.58 and 0.96 s per iteration and 6.6, 12.8 and 43.2 s for (Q) (H2O
  C2v, N2 D2h, H2S C2v; 8 cores).  That is 1.9-2.5 times faster than zccsdt
  per iteration, and in (Q) between 1.2 times slower and 1.5 times faster.

### `ncc`'s own profile

`ncc` has built-in timers (`-DPROFILE`; summed up by
`bench/cfour/ncc_profile.py`).  With them the icpc `-O3 -xHost` build runs at
the speed of the plain one (0.79 and 2.85 s per iteration for H2O and
N2/cc-pVTZ on 8 cores, 5.37 s for H2O on one, 299 s for benzene), and it
reports how much of the iteration is DGEMM.  Seconds per CCSDT iteration (thread time divided by the
number of threads), next to the GEMM timers of zccsdt in the same batch of
jobs (the one-core batch had seven jobs at once, that of the table above
five):

| | `ncc` iteration | in DGEMM | elsewhere | zccsdt iteration | in GEMM | elsewhere |
|---|---|---|---|---|---|---|
| H2O/cc-pVTZ, 1 core | 5.37 | 2.61 | 2.76 | 2.35 | 1.27 | 1.08 |
| H2O/cc-pVTZ, 8 cores | 0.79 | 0.31 | 0.48 | 0.48 | 0.19 | 0.29 |
| N2/cc-pVTZ, 8 cores | 2.85 | 0.89 | 1.96 | 1.06 | 0.57 | 0.49 |
| C6H6/cc-pVDZ, frozen core, 8 cores | 299 | 114 | 185 | 85.7 | 61 | 25 |

- **The time ratio has three modest factors.**  `ncc` does 1.2-1.3 times
  the GEMM arithmetic of zccsdt (7.9e10 against 6.0e10 flops for H2O, 2.33e11
  against 1.89e11 for N2, 3.9e13 against 3.0e13 for benzene; operation counts
  in "Where the difference comes from" below).  It executes it at 30-32
  GFLOP/s per thread, one block per thread (at most 250 with 8 threads; 43
  and 340 for benzene), where zccsdt reaches about 47 on one core and 320-330
  with 8 threads in larger stacked products (515 for benzene).  And it spends
  more outside the products: 51-69% of its iteration against 29-60% of a
  shorter one.
- **Outside DGEMM `ncc` is in BLAS 1 calls, spin adaptation and
  transposition** of T3 blocks: 21%, 14% and 16% of the thread time for H2O
  on 8 threads; 34%, 11% and 10% for N2, where the BLAS 1 calls run at 1.1-1.2
  GFLOP/s per thread, the speed of memory.  For N2 on 8 threads some of the
  rest is threads waiting for the one with the most blocks.  For benzene the
  BLAS 1 calls alone (1.1 GFLOP/s per thread) take at least as long as DGEMM
  (38%).  (The timers are nested, so the shares are approximate.)
- **What `ncc` could reach.**  Without any cost outside DGEMM, at its present
  DGEMM rate, an `ncc` iteration would take 0.31 s (H2O), 0.89 s (N2) and
  114 s (benzene) on 8 cores, where zccsdt takes 0.48 s, 1.06 s and 86 s.  For
  the small molecules zccsdt is not ahead of what `ncc`'s formulation
  allows, only closer to it; for benzene it is ahead, by the smaller
  operation count and the higher rate of large stacked products.
- **(Q) is a different case: `ncc` spends 3-5% of it in DGEMM.**  H2O/cc-pVTZ
  on one core (310 s): DGER 213 s, transposition 62 s, DGEMM 14 s.  On 8
  threads: DGER 58%, transposition 17%, DGEMM 3% of the thread time and 14%
  in no timer (N2: 43%, 12%, 2.5% and 37%; the last is mostly threads
  waiting, the distribution of the quadruples over 8 threads being uneven
  for o = 5 and more so for o = 7).
  One term takes 69-73% of the step, W(abc,ije) t(ed,kl) -> Z(abcd,ijkl), the
  product of the T2 x `<vv|vv>` intermediate with T2.  `ncc` stores W in
  blocks numbered by (i, j, e), so the sum over e is v rank-1 updates of a
  v^3 x v block of Z, each a pass over the whole block at 1.1-1.3 GFLOP/s per
  thread.  zccsdt keeps e as a matrix dimension of the intermediate and does
  one GEMM per term ("I x T2 products" in the (Q) section).  Compiler flags
  cannot change this, which is why all builds give the same (Q) time.

### Parallel scaling of the CCSDT iteration

Seconds per iteration and, in parentheses, the speedup over one thread.
cc-pVTZ, no symmetry, best `ncc` build; jobs with several threads one at a
time on one socket, the one-thread jobs at the same time on separate cores of
the other (zccsdt: first iterations; `ncc`: iterations 2-4):

| threads | zccsdt, H2S (o = 9) | `ncc`, H2S | zccsdt, CH2O (o = 8) | `ncc`, CH2O |
|---|---|---|---|---|
| 1 | 13.1 | 34.4 | 45.5 | 108.5 |
| 2 | 6.91 (1.9) | 17.2 (2.0) | 23.0 (2.0) | 60.9 (1.8) |
| 4 | 3.75 (3.5) | 9.09 (3.8) | 12.0 (3.8) | 39.4 (2.8) |
| 7 | | 6.07 (5.7) | 7.80 (5.8) | 17.9 (6.1) |
| 8 | 2.24 (5.9) | 5.06 (6.8) | 6.60 (6.9) | 30.9 (3.5) |

- **zccsdt does not scale better than `ncc` when `ncc`'s blocks are spread
  evenly**: on 8 threads 5.9 against 6.8 for H2S, 4.8 against 6.6 for H2O,
  5.9 against 5.5 for N2.  The time ratio is then decided on one core
  (`ncc` / zccsdt 2.2-2.6) and is the same or a little smaller on 8 (H2S:
  2.6 -> 2.3).
- **For o = 8 `ncc` loses whenever the number of threads divides 8**:
  speedups of 1.8, 2.8 and 3.5 with 2, 4 and 8 threads, 6.1 with 7.  The rule
  of "Number of threads" above gives 1.7, 2.6, 3.2 and 7.0.  The ratio of 4.7
  on 8 threads (6.4 with the reference build) comes from this and not from
  the scaling of zccsdt; on one thread it is 2.4, on 7 threads each 2.3.
- **zccsdt is threaded inside each operation**: MKL's threaded GEMM on the
  stacked matrices and OpenMP loops in the gather, scatter and packing
  kernels, so there are no blocks to hand out.  From 1 to 8 threads (H2O,
  N2) the GEMMs gain 6.1-7.2 and the gathers and scatters 6.5-9, but DIIS
  only 1.4-2.5 and the unpacking of the ladder 2.9, so small molecules scale
  worse (4.8 for H2O) than larger ones (6.9 for CH2O).

### One core, and `ncc` with point-group symmetry

cc-pVTZ, c233, one thread pinned to one core per job (all single-core jobs
ran at the same time on separate cores of one socket).  zccsdt 0212757;
`ncc`: reference build (its best build is 1.3 times faster per iteration, see
above); times in seconds.

| | zccsdt | `ncc`, no symmetry | `ncc` / zccsdt | `ncc`, with symmetry |
|---|---|---|---|---|
| H2O, CCSDT iteration | 2.34 | 6.98 | 3.0 | 0.97 (C2v) |
| H2O, (Q) | 36.4 | 301 | 8.3 | 29.3 |
| N2, CCSDT iteration | 6.27 | 21.7 | 3.5 | 1.59 (D2h) |
| N2, (Q) | 128 | 1064 | 8.3 | 53.8 |
| H2S, CCSDT iteration | 13.6 | 55.4 | 4.1 | 6.07 (C2v) |
| H2S, (Q) | 334 | not run | | 287 |

The same on 8 cores (wall time; CPU time of all threads in parentheses):

| | zccsdt | `ncc`, no symmetry | `ncc` / zccsdt | `ncc`, with symmetry |
|---|---|---|---|---|
| H2O, CCSDT iteration | 0.46 (2.9) | 0.97 (6.3-7.8) | 2.1 | 0.30 (2.4) |
| H2O, (Q) | 5.2 (36) | 57.4 (405) | 11 | 7.4 (59) |
| N2, CCSDT iteration | 1.05 (7.2) | 3.81 (20-30) | 3.6 | 0.64 (5.1) |
| N2, (Q) | 18.5 (128) | 283 (1364-1522) | 15 | 13.5 (107) |
| H2S, CCSDT iteration | 2.17 (15.5) | 7.14 (51.5) | 3.3 | 1.23 (9.8) |
| H2S, (Q) | 48.7 (337) | 613 (4269) | 13 | 48.1 (381) |

- **Like for like (no symmetry), on one core** zccsdt is 3.0-4.1 times
  faster per CCSDT iteration and 8.3 times faster in the (Q) step.  The
  8-core wall-time ratios of (Q) (11-15) are larger because the (Q) step of
  `ncc` gains only 3.8-5.2 from 8 threads, that of zccsdt 6.9-7.0.  For the
  CCSDT iteration zccsdt gains 5.1-6.3 and `ncc` 5.7-7.8.
- **With point-group symmetry `ncc` wins on these molecules.**  Its
  iteration becomes 7-14 times cheaper (C2v, D2h) and is then 2.2-3.9 times
  faster than zccsdt on one core and 1.5-1.8 times on 8 cores; its (Q) step
  is 1.2-2.4 times faster on one core and between 1.4 times slower and 1.4
  times faster on 8 cores.  zccsdt would need Abelian point-group symmetry
  to compete for symmetric molecules.
- The (Q) corrections of `ncc` with and without symmetry agree to 1e-13 Eh.

### Where the difference comes from: operations and rates

`ncc` counts its own BLAS operations (2 m n k per GEMM) and prints the rate.
Counting the GEMMs of zccsdt in the same way (the count does not depend on
the number of threads; no point-group symmetry in either program):

| | `ncc` flops | zccsdt flops | zccsdt / `ncc` | `ncc` GFLOP/s, 8 threads (1) | zccsdt GFLOP/s, 8 threads (1) |
|---|---|---|---|---|---|
| H2O/cc-pVTZ, CCSDT iteration | 7.9e10 | 6.0e10 | 0.76 | 86 (12) | 131 (26-29) |
| N2/cc-pVTZ, CCSDT iteration | 2.33e11 | 1.89e11 | 0.81 | 65 (11) | 180 (30-33) |
| H2S/cc-pVTZ, CCSDT iteration | 5.27e11 | 4.54e11 | 0.86 | 78 | 208 (33) |
| H2O/cc-pVQZ, CCSDT iteration | 2.53e12 | 1.60e12 | 0.63 | 151 | 310 |
| H2O/cc-pVTZ, (Q) | 5.98e11 | 6.91e11 | 1.16 | 10.4 (2.0) | 133 (19-21) |
| N2/cc-pVTZ, (Q) | 2.42e12 | 2.54e12 | 1.05 | 8.6 (2.3) | 137 (20-22) |
| H2S/cc-pVTZ, (Q) | 6.88e12 | 6.80e12 | 0.99 | 11.2 | 138 (20) |
| H2O/cc-pVQZ, (Q) | 2.19e13 | 2.53e13 | 1.16 | 5.7 | 251 |

DGEMM on these cores: 540-580 GFLOP/s with 8 threads; with one thread 71-73
for large square matrices and 31-41 for the thin shapes that occur here.

- **The two programs do about the same amount of arithmetic**: zccsdt 0.6-0.9
  of `ncc` in a CCSDT iteration and 1.0-1.2 in (Q).  The time difference
  is almost entirely the rate at which that arithmetic is executed: `ncc`
  reaches 12-28% of the DGEMM rate in the CCSDT iterations and 1-2% in (Q),
  zccsdt 24-55% and 25-45%.  (Rates of the reference build of `ncc`; its
  best build is 1.2-1.3 times faster in the iterations.)  The rest of
  `ncc`'s time is outside the multiplications, see its profile above; zccsdt
  is built around layouts in which the GEMM results need no further
  permutation, for this one case (closed shell, no symmetry).
- **Correction.**  Earlier versions of this document explained the speed by
  "about a third of the arithmetic of `ncc`", from a model of `ncc`'s
  algorithm in `bench/flop_counts.py`.  That model overestimates `ncc`'s own
  count about twofold (H2O/cc-pVTZ: 6.4-7.9e10 FMA in the model, 3.95e10
  counted by `ncc`).  The "arithmetic ratio (model)" columns and the
  efficiency statements derived from them in the sections below are wrong by
  that factor; the measured times are not affected.
- With point-group symmetry `ncc` does far less arithmetic (H2O, C2v:
  5.5e9 flops per iteration instead of 7.9e10), which is why it is faster
  than zccsdt for symmetric molecules although its rate drops further.

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
model over the spin-adapted engine (`bench/flop_counts.py`).  It overstates
the arithmetic of `ncc` about twofold; see "Where the difference comes from"
above for the operation counts of the two programs.

Larger systems, same procedure on c233 (same CPUs; the node was shared, with
a 1-minute load of 4-24 during the runs), zccsdt 0154ff2 with the T3 DIIS
vectors in the scratch file:

| System | o, v | T3 vector | zccsdt | CFOUR `ncc` | `ncc` / zccsdt | arithmetic ratio (model) | Memory: zccsdt anonymous, `ncc` RSS |
|---|---|---|---|---|---|---|---|
| C2H4/cc-pVTZ | 8, 108 | 1.1 GB | 19.4 s | 135.3 s | 7.0 | 2.4-2.9 | 7.6 GB, 5.0 GB |
| C6H6/cc-pVDZ, frozen core | 15, 93 | 4.3 GB | 85.7 s | 455.7 s | 5.3 | 2.2-2.9 | 13.7 GB, 14.1 GB |

- `ncc` keeps 3.5 of the 8 cores busy for C2H4 (477 s CPU per iteration):
  with o = 8 its distribution of blocks over 8 threads is uneven.  With 7
  threads and its best build it takes 57.6 s, 2.8 times the zccsdt iteration
  ("The `ncc` reference" at the top).  For benzene 7.0 cores are busy
  (3189 s); its best build takes 295 s with 8 threads and 275 s with 7, 3.1
  times the zccsdt iteration.
- At CFOUR's geometry (see "Convergence and energies") the correlation
  energies of the first two iterations, before the two DIIS schemes diverge,
  agree with `ncc` to 4.4e-10 and 2.6e-10 Eh for benzene, and the SCF energy
  to 6e-12 Eh.
- Benzene, where the time goes (85.7 s): ring GEMMs 31.5 s and pp ladder GEMMs
  22.6 s (both about 515 GFLOP/s), T3 -> R1/R2/Wv/Wo 7.0 s, drive + hh ladder
  7.1 s, pair gather 4.3 s, Jacobi step and DIIS 4.3 s, ring scatter 2.3 s.

**Reading the numbers.**

- **zccsdt is 3.3-3.7 times faster than the reference build of `ncc` from
  N2/cc-pVTZ upwards** (no point-group symmetry in either program), and
  2.0-3.1 times faster than `ncc` at its best: best build and number of
  threads, "The `ncc` reference" at the top.  By the programs' own operation
  counts zccsdt does 0.6-0.9 of the GEMM arithmetic of `ncc`.
- **CH2O/cc-pVTZ:** `ncc` uses 150 s of CPU time per 41 s iteration, so only
  3.7 of the 8 cores are busy (6.5-7.2 for the other systems): o = 8 on 8
  threads is the unfavourable case of its distribution of blocks over
  threads.  With 7 threads and its best build `ncc` takes 17.9 s, 2.6 times
  the zccsdt iteration.
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

### Against PySCF's RCCSDT

PySCF has had an RHF-CCSDT of its own since December 2025 (`pyscf.cc.RCCSDT`,
modules `rccsdt` and `rccsdt_highm`): T1-dressed, T3 stored either for
i <= j <= k (`compact_tamps=True`, the default) or in full ("if memory is
sufficient"), contractions through a pluggable einsum.  It is the closest
open-source counterpart of `zccsdt.sa.rccsdt`.

**Setup.**  One Slurm job on c157 (2 x Xeon Gold 6248R), 8 cores, each run a
step of its own, zccsdt and PySCF one after the other: `bench/profile_sa.py`
and `bench/pyscf/profile_rccsdt.py` (same geometries and length unit, all
electrons).  PySCF 2.13.1 (git afea3d3bd, June 2026) with the einsum backend
it recommends for this code, pytblis 0.0.16 (TBLIS, 8 threads); its default
backend (numpy) is in the last row.  Its C helpers (`libccsdt`) rebuilt with
`-O3 -march=native` changed nothing.  Time per iteration: the mean of the
iterations after the first (PySCF: between two calls of its callback).
Converged, the two programs agree: H2O/cc-pVTZ E_corr = -0.288726186090
(PySCF, 22 iterations) and -0.288726186079 (zccsdt, 16 iterations).

| System | o, v | zccsdt | PySCF, full T3 | ratio | PySCF, compact T3 | ratio |
|---|---|---|---|---|---|---|
| H2O/cc-pVTZ | 5, 53 | 0.47 s, 0.82 GB | 1.37 s, 1.13 GB | 2.9 | 3.81 s, 0.98 GB | 8.2 |
| N2/cc-pVTZ | 7, 53 | 1.05 s, 0.88 GB | 3.67 s, 1.95 GB | 3.5 | 9.42 s, 1.45 GB | 9.0 |
| H2O/cc-pVQZ | 5, 110 | 5.31 s, 4.57 GB | 21.6 s, 8.19 GB | 4.1 | 44.5 s, 6.64 GB | 8.4 |
| C2H4/cc-pVTZ | 8, 108 | 19.7 s, 7.48 GB | 70.7 s, 23.1 GB | 3.6 | 134.5 s, 16.6 GB | 6.8 |
| H2O/cc-pVTZ, PySCF defaults (compact, numpy) | 5, 53 | | | | 17.2 s, 1.67 GB | 37 |

(memory: peak anonymous memory of the process; the DIIS vectors of zccsdt
are in its scratch file, PySCF keeps its own on disk too unless
`incore_complete`: 1.51 s instead of 1.37 s for the first row.)

- **zccsdt is 3-4 times faster than PySCF's faster variant**, with half to a
  third of its memory, and 7-9 times faster than the default variant.  The
  ratio grows with v / o, as the ladder takes over.
- **Arithmetic.**  `rccsdt_highm` builds the full o^3 v^3 residual: its pp
  ladder is one o^3 v^5 contraction over the full `<ab|de>` and its ring
  three o^4 v^4 contractions.  The spin-adapted engine contracts only sorted
  occupied triples with the (d,e)-symmetric and -antisymmetric halves of the
  operator (about o^3 v^5 / 4) and needs two o^4 v^4 GEMMs for the ring
  (`docs/spin_adapted.md`).  The compact variant works on blocks of
  occupied triples i <= j <= k of its block size (8); with o <= 8, as here,
  that is one block with all triples, so it does the arithmetic of the full
  variant plus the unpacking and accumulation of the compact storage.
- **Memory.**  PySCF holds the full `pppp` integral array (n^4) and, in the
  full variant, several o^3 v^3 arrays (23 GB for C2H4/cc-pVTZ); zccsdt holds
  integral blocks, one packed copy of the pp operator and two T3-sized
  arrays.

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
cases.  `ncc`: reference build, 8 threads and `MEMORY_SIZE` = 24 GB, less
than it would like for all but H2O/cc-pVTZ.  With its best build, number of
threads and more memory its times are 3-15% shorter at cc-pVTZ and 39%
shorter at cc-pVQZ (ratios 10-13 and 23): see "The `ncc` reference" at the
top, which also shows where `ncc` spends the step.

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
  8 cores busy (11522 s CPU in 3837 s; that run had 22 GiB where `ncc` asks
  for 167).
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

## Not enough memory: T3, `<vv|vv>` and the (Q) arrays out of core

What happens when the arrays do not fit is described in
`docs/spin_adapted.md` ("Memory, and when it is short"; "When 10 v^4 does not
fit") and in Section 3 of `docs/factorization.md`; here is what it costs.

**Setup.**

- **Memory limits.** Each run is a step of a Slurm batch job with a memory
  limit of its own (`srun --mem`, `bench/slurm_steps.sh`), which zccsdt sees
  through its memory cgroup; the plan is the automatic one (budget: 80% of
  what the limit leaves).  "In memory" is the same calculation as a step
  whose limit does not matter.  The steps of a job run one after the other
  on the same cores and on the same node-local NVMe (mapped reads at
  1.75 GB/s, writes at 0.6-1.0 GB/s).
- **What is timed.** The mean of 2-3 iterations after a warm-up call.  The
  DIIS subspace holds few vectors in runs this short; with a full subspace
  the DIIS step reads 16 T3-sized vectors per iteration instead of 2-6 (see
  "DIIS storage" above for what that costs when they are not in the page
  cache).
- **Memory.** Peak anonymous memory of the process, that is, its arrays;
  mapped scratch files are page cache, which the limit takes back.
- c157 (2 x Xeon Gold 6248R; benzene: c395, the same CPUs), 8 cores unless
  stated.  Some steps shared the node with another benchmark job, which is
  why each ratio is taken against the run in memory of the same job.
- Versions: spin-orbital engines 20087bd (the X2C rows a4858a4: the same
  dense engine); spin-adapted engine 210f5b7 with the pair pass planned at
  15 instead of 16 o v^3 (benzene: a4858a4, see below).  Neither had the
  residual-in-memory placement of 210f5b7 / 664e367 yet: where T3 leaves
  memory in these tables, its residual leaves with it.

### Spin-adapted RCCSDT

| System | o, v | In memory | Limit | What leaves memory | Time | Ratio | Anonymous memory |
|---|---|---|---|---|---|---|---|
| H2O/cc-pVQZ | 5, 110 | 5.6 s, 4.5 GB | 3 GB | caches; bare pp operator, dressed rows in 3 slabs | 8.0 s | 1.4 | 2.6 GB |
| | | | 2 GB | and T3 with its residual, ladder batch-wise | 17.6 s | 3.2 | 1.5 GB |
| C2H4/cc-pVTZ | 8, 108 | 20.9 s, 7.5 GB | 6 GB | caches; bare pp operator | 27.6 s | 1.3 | 4.6 GB |
| | | | 4 GB | caches; T3 and its residual | 38.7 s | 1.9 | 2.9 GB |
| | | | 3 GB | caches; T3, its residual, bare pp operator | 44.4 s | 2.1 | 2.4 GB |
| H2O/cc-pV5Z, 16 cores | 5, 196 | 30.7 s, 27.9 GB | 16 GB | caches; bare pp operator, dressed rows in 5 slabs | 38.4 s | 1.25 | 11.7 GB |
| | | | 12 GB | and T3 with its residual, ladder in 3 batches | 77.6 s | 2.5 | 8.0 GB |
| | | | 9 GB | the same with the thinnest slabs, 5 batches | 131 s | 4.3 | 6.8 GB |
| C6H6/cc-pVDZ, frozen core (a4858a4) | 15, 93 | 75.5 s, 13.7 GB | 8 GB | T3 and its residual | 156 s | 2.1 | 4.5 GB |
| | | | 6 GB | T3 and its residual | 184 s | 2.4 | 4.4 GB |
| H2O/cc-pVQZ, (Q) | 5, 110 | 102 s, 13.5 GB | 6 GB | (v^4 arrays in slabs) | 185 s | 1.8 | 5.5 GB |

"Caches" are the second layouts of integral blocks and the work buffers
that a run in memory keeps between its stages (8 o v^3 together); a run
short of memory gives each back after the stage that uses it, which by
itself costs about a tenth of the iteration time (H2O/cc-pVQZ: 5.7-6.1 s
instead of 5.2-5.6 s with everything else in memory).  Before that change
(a4858a4) the plan counted 23 o v^3 as fixed, T3 left memory much earlier and
the ladder batches shrank: H2O/cc-pVQZ took 11.7 s under 3 GB, C2H4 46.5 s
under 4 GB, H2O/cc-pV5Z 89.7 s under 16 GB and 266 s under 12 GB (one block
per batch: the operator file was read 30 times per iteration).

### Spin-orbital and spinor engines

| System | o, v (spin orbitals) | In memory | Limit | What leaves memory | Time | Ratio | Anonymous memory |
|---|---|---|---|---|---|---|---|
| X2C H2O/cc-pVTZ (spinors, complex) | 10, 106 | 15.6 s, 5.1 GB | 4 GB | bare pp operator (dressed rows at once) | 16.5 s | 1.06 | 3.8 GB |
| | | | 3 GB | bare pp operator, dressed rows in 6 slabs | 21.2 s | 1.36 | 2.9 GB |
| | | | 2.5 GB | and T3 with its residual, ladder batch-wise | 21.7 s | 1.39 | 2.0 GB |
| | | | 2 GB | the same with thin slabs | 28.7 s | 1.84 | 1.8 GB |
| UHF NH2/cc-pVQZ (U form) | 9, 221 | 16.2 s, 8.9 GB | 6 GB | bare pp blocks, dressed rows in 4 slabs | 17.8 s | 1.10 | 4.9 GB |
| | | | 5 GB | the same in 15 slabs | 20.3 s | 1.26 | 4.1 GB |
| | | | 4 GB | and T3 with its residual, ladder batch-wise | 34.3 s | 2.1 | 3.0 GB |
| | | | 3.5 GB | the same with the smallest buffers | 55.7 s | 3.4 | 2.9 GB |
| RHF H2O/cc-pVQZ (R form, spin flip) | 10, 220 | 12.7 s, 7.8 GB | 5 GB | bare pp blocks, dressed rows in slabs | 17.9 s | 1.4 | 4.0 GB |
| | | | 4 GB | and T3 with its residual, ladder batch-wise | 27.1 s | 2.1 | 3.1 GB |

- **The runs complete with the energies of the runs in memory**: the
  correlation energy of every iteration agrees to 2e-14 (spin-orbital
  engines) and 1e-13 (spin-adapted, cc-pV5Z), the (Q) correction to all
  printed digits.
- **Below what must fit, the run stops before the iterations** with a
  `MemoryError` that says how much is needed: X2C H2O/cc-pVTZ under 1.8 GB
  (needs 1.5 GB beyond the integrals; 2 GB is the edge, where one of two
  runs was refused), UHF NH2/cc-pVQZ under 3 GB (2.8 GB), RHF H2O/cc-pVQZ in
  the R form under 3 GB (3.0 GB).  No run was killed by its limit.  One was
  before 20087bd: UHF NH2/cc-pVQZ under 5 GB, planned at 3.7 GB, because the
  spin-blocked engine kept a work array that the plan did not count (and a
  second one after a second plan); under 6 GB that version took 28.9 s.
- **The integrals stay within the limit as well**: built for the scratch
  file when the budget is short, the integral stage peaks at 1.3-1.6 GB for
  X2C H2O/cc-pVTZ (2.9 GB in memory; 14.5-15.2 s either way) and 1.6-1.8 GB
  for UHF NH2/cc-pVQZ (3.3 GB; 32-34 s).
- **What it costs depends on what has to leave.**
  1. *Caches only, or the bare pp operator*: 1.06-1.4 times the time in
     memory.  The operator file is read once per iteration (5.9 GB for
     cc-pV5Z: 3 s), and the dressed rows come in a few slabs.
  2. *T3 and its residual*: 1.4-2.5 times.  Every block of T3 is read by
     each pass that needs it (three pair positions, three positions of the
     index k, the ladder, the Jacobi step: about 9 times per iteration), and
     every block of the residual is read and written as often; what the
     page cache under the limit cannot hold goes through the disk.  Benzene,
     where two T3-sized arrays are 8.6 GB of the 13.7 GB, is the example:
     about 40 GB read and 40 GB written per iteration.  (The planner now
     keeps the residual in memory when it fits alone, so that only reads of
     T3 remain; not measured yet.)
  3. *Close to the least that works*: 3-4 times and more.  The buffers are
     at their smallest, the ladder runs in many batches, and each batch is a
     pass over the file with the rows of the ladder operator (cc-pV5Z under
     9 GB: 5 passes over 5.9 GB in 127 slabs each).
- **Telling the operating system does not help on a quiet node.**  With the
  hints of `ZCCSDT_IO=1` against without: benzene 183.9 s against 183.5 s
  (6 GB) and 159.4 s against 155.7 s (8 GB); C2H4 32.0 s against 29.9 s
  (6 GB) and 50.2 s against 49.1 s (4 GB); H2O/cc-pVQZ 13.0 s against
  12.6 s (3 GB); X2C H2O/cc-pVTZ 21.9 s against 22.5 s (3 GB) and 25.8 s
  both ways (2.5 GB).  They are an option, not the default.
- **What must still fit** (next to the integral blocks):
  spin-adapted, 16 o v^3 plus 2 n o v^2 + n^2 o v elements: 1.1 GB for
  H2O/cc-pVQZ, 1.7 GB for C2H4/cc-pVTZ, 2.0 GB for benzene/cc-pVDZ, 5.9 GB
  for H2O/cc-pV5Z, 38 GB for o = 30, v = 200;
  spinors and spin orbitals, 3.5 o v^3 of complex or real numbers plus the
  overhead: 1.5 GB for X2C H2O/cc-pVTZ, 7.3 GB for X2C H2O/cc-pVQZ;
  U / R form, 2.3 o v^3 of real numbers plus the overhead: 2.8 GB for
  NH2/cc-pVQZ.
  For (Q) in slabs: 6 (26 + o^2) v^3 elements (3.6 GB for H2O/cc-pVQZ, 60 GB
  for v = 282 and o = 5, where the block algorithm needs 506 GB).

## Spinor against spin-adapted CCSDT: time ratios and their theory, 8 cores

How much more does the relativistic (complex spinor, no spin symmetry)
iteration cost than the nonrelativistic closed-shell one, and is that the
ratio the operation counts predict?

c233, 8 threads on 8 pinned cores, wall time per CCSDT iteration (mean of 3,
or of 2 for the complex cc-pVQZ run), zccsdt 0212757.  Spin-orbital code:
`bench/profile_iter.py` with `ghf` (real, RHF orbitals as spin orbitals) and
`x2c` (complex two-component spinors); cc-pVQZ with `ZC_PACKED=1`.

| System | spin-adapted (real) | spin-orbital, real | spinor, complex |
|---|---|---|---|
| H2O/cc-pVTZ | 0.46 s | 5.28 s | 16.1 s |
| N2/cc-pVTZ | 1.05 s | 14.6 s | 47.6 s |
| H2O/cc-pVQZ | 5.1 s | 109 s | 367 s |

Theory: FMA of the pp ladder and ring terms from `bench/flop_counts.py`
(spin-orbital against spin-adapted), and a factor 4 for complex arithmetic
(one complex multiplication is four real ones).

| Ratio | Theory | H2O/cc-pVTZ | N2/cc-pVTZ | H2O/cc-pVQZ |
|---|---|---|---|---|
| spin-orbital real / spin-adapted | 28-30 | 11.5 | 13.9 | 21.4 |
| complex / real (spin-orbital code) | 4 | 3.05 | 3.25 | 3.36 |
| complex spinor / spin-adapted | 111-118 | 35 | 45 | 72 |

Rate on the dominant terms (flops of the ladder and ring terms over the
iteration time, GFLOP/s; the DGEMM rate of these 8 cores is 540-580):

| System | spin-adapted | spin-orbital, real | spinor, complex |
|---|---|---|---|
| H2O/cc-pVTZ | 106 | 256 | 336 |
| N2/cc-pVTZ | 149 | 316 | 388 |
| H2O/cc-pVQZ | 276 | 377 | 449 |

- **The measured ratios are below the theoretical ones and approach them as
  the calculation grows** (spinor / spin-adapted: 35 -> 72 from cc-pVTZ to
  cc-pVQZ, theory 117).  The operation counts only describe the GEMM work.
  The cheaper a method is, the larger the share of everything else (gathers,
  scatters, passes over T3, DIIS, small intermediates), so the spin-adapted
  code runs at 20-50% of the DGEMM rate and the spinor code at 60-80%.
- **Complex against real is 3.0-3.4, not 4**, for the same reason: the GEMMs
  cost four times more, the memory traffic only twice.
- The relativistic code is therefore not slower than it should be relative
  to the nonrelativistic one; the deviation from theory is overhead on the
  nonrelativistic side that matters less for larger basis sets.
- Peak memory, H2O/cc-pVQZ: 27 GB (real spin-orbital) and 42 GB (complex),
  against 5 GB for the spin-adapted iterations.

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
ring), RCCSDT over the `ncc` model, from `bench/flop_counts.py`.  (The `ncc`
model overstates `ncc`'s own flop count about twofold; see "Where the
difference comes from" at the top.)

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
  code.  *Correction:* `ncc`'s own flop counter gives about half of this
  model (see "Where the difference comes from" at the top), so the ratios
  "spin-orbital / ncc (theory)" and the `ncc` rates in the tables below are
  too high by about that factor.  The model:
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

## Two-component CCSD(T): CFOUR `xx2cccsd`, 8 cores

CFOUR's spinor CCSD(T) (`x2cccsd/`, described in `docs/cfour_notes.md`,
Section 2) is the reference for a spinor CCSD(T) in zccsdt, which does not
exist yet. zccsdt's CCSD (`solver.kernel(eris, 'ccsd')`, the CCSD part of
its CCSDT code) is measured next to it.

**Setup.**

- One Slurm job, `bench/cfour/slurm_x2ccc.sh` (job 31533171, 2026-10-02):
  8 cores of a 2 x Xeon Gold 6248R node, CFOUR first and then zccsdt, one
  molecule after the other.
- CFOUR (cfour-x2cso 35b1f40e3): `RELATIVISTIC=X2C1E`, `X2CKEY=SOATMX`,
  `ABCDTYPE=AOBASIS`, all electrons, C1, `CC_CONV=7` (largest element of
  the amplitude step). Times per residual evaluation from its own timers
  (`bench/cfour/x2ccc_summary.py`), the first evaluation left out; peak RSS
  from `/usr/bin/time`.
- zccsdt (bd8a2a8): X2C-1e of PySCF (`x2c.UHF`) -- the same orbital
  counts, not the same Hamiltonian, so the energies are not compared.
  `bench/profile_iter.py` with `ZC_METHOD=ccsd ZC_PACKED=1`, mean of 3
  iterations after a warm-up iteration.

Orbital counts (spinors): H2O/cc-pVTZ o = 10, v = 106 (N = 58 scalar AO
functions); H2O/cc-pVQZ o = 10, v = 220 (N = 115); benzene/cc-pVDZ o = 42,
v = 186 (N = 114).

### Where CFOUR's time goes (seconds per CCSD iteration)

| Term | H2O/cc-pVTZ | H2O/cc-pVQZ | C6H6/cc-pVDZ |
|---|---|---|---|
| W(mbej) T2 term: one (ov)^3 ZGEMM (I7) | 0.019 | 0.154 | 6.36 |
| ring: one (ov)^3 ZGEMM | 0.022 | 0.167 | 6.49 |
| pppp ladder in the AO basis | 0.032 | 0.357 | 3.40 |
| <ab\|\|ci> x T1 terms I8 + I22 (slab reads, small GEMMs) | 0.112 | 0.869 | 2.55 |
| other timed intermediates, hhhh ladder | 0.014 | 0.061 | 1.12 |
| not timed: other T1/T2 terms (three more passes over the slabs), DIIS, update | 0.127 | 1.155 | 7.01 |
| **iteration** | **0.33** | **2.76** | **26.9** |
| residual evaluations to `CC_CONV=7` | 11 | 11 | 13 |
| **(T)** | **2.1** | **23.7** | **1500** |
| peak RSS, xx2cccsd (xx2cvtran) | 0.70 (0.81) GB | 2.2 (3.7) GB | 11.1 (13.7) GB |

- The two (ov)^3 ZGEMMs run at 510-600 GFLOPS, close to the peak of the 8
  cores. They are half of the iteration for benzene but 12% for
  H2O/cc-pVQZ.
- With few occupied spinors the time goes to the terms that read the PPPH
  files, 5 o v^3 / 2 complex numbers per iteration: I22 (matrix-vector
  products) runs at 2 GFLOPS, I8 at 19-59 GFLOPS; together with the three
  untimed passes about 2 of the 2.76 s for H2O/cc-pVQZ.
- The AO ladder runs at 225 GFLOPS (H2O/cc-pVQZ) and 350 GFLOPS (benzene),
  its transformations included.
- **(T)**: its ZGEMMs alone, at 550-600 GFLOPS, would take 6 s
  (H2O/cc-pVQZ) and 280 s (benzene) of the 23.7 s and 1500 s. The rest is
  the PPPH slabs read for every triplet (31 GB and 1.8 TB in all) and the
  energy loop over a>b>c, whose W(ba, c) access has a stride of v(v+1)/2
  elements. For benzene (T) takes 4.3 times as long as the whole CCSD.

### zccsdt's CCSD on the same molecules

| | H2O/cc-pVTZ | H2O/cc-pVQZ | C6H6/cc-pVDZ |
|---|---|---|---|
| CFOUR iteration | 0.33 s | 2.76 s | 26.9 s |
| zccsdt iteration | 1.67 s | 23.1 s | -- |
| of which T1 dressing (`t1_dress`) | 1.16 s | 19.4 s | |
| of which residual (`ccsd`) | 0.41 s | 3.1 s | |
| zccsdt peak RSS | 3.2 GB | 27.3 GB | OOM-killed at the 30 GB limit while building the integrals |
| zccsdt integrals (SCF not included) | 14.6 s | 251 s | |

- zccsdt's CCSD is the CCSD part of its CCSDT code: every iteration
  rebuilds the T1-dressed packed <ab||cd> (`dressed_vvvv_pp`), a copy of
  the v^4/4 array plus an intermediate of v^4/2, limited by memory
  bandwidth. Next to the o^3 v^5 terms of CCSDT this is small; on its own
  it makes the CCSD iteration 5-8 times slower than CFOUR's. The residual
  alone is also slower than CFOUR's whole iteration.
- `solver.kernel(eris, 'ccsd')` has no memory plan (the bare and the
  dressed <ab||cd> are both kept). For benzene/cc-pVDZ the block-wise
  integrals (`eris._blocks_by_ao2mo`) went past the 30 GB limit although
  their floor check (the o v^3 block and its buffers, about 9 GB, against
  a 24 GB budget) passed: the memory model of the spinor integrals misses
  something when there are many occupied orbitals (not investigated yet).
- A spinor CCSD(T) in zccsdt therefore needs a CCSD iteration of its own
  (ladder on bare integrals or in the AO basis, no dressed <ab||cd>, a
  memory plan) and a (T) that does not read integral slabs per triplet.

### socutils' CCSD and (T), copied into zccsdt (`zccsdt.zccsd`)

The user's own spinor CCSD and (T) (`ZCCSD`, `DirectZCCSD`, `gccsd_t` from
socutils, copied unchanged apart from imports and the occ_loop fix), the
starting point of zccsdt's CCSD(T). One job, everything one after the other
on one node (c028, 2 x Xeon Gold 6248R, 8 cores; job 31541638): CFOUR as
above, then `bench/profile_zccsd.py` (`bench/slurm_zccsd.sh`), mean of
iterations 2-4 (2-3 for benzene). The copied code is not changed for the
timing: its functions are wrapped by `zccsdt.timing` timers. (T) runs the
occ_loop on the amplitudes of the last iteration; for benzene it was stopped
after 1200 s (645 of 11480 triples) and extrapolated (`ZC_T_BUDGET`). CFOUR's
benzene (T) is the 1500 s of job 31533171 (c146, whose CCSD iteration took
26.93 s against 26.90 s on c028).

| Seconds, peak memory | H2O/cc-pVTZ | H2O/cc-pVQZ | C6H6/cc-pVDZ |
|---|---|---|---|
| CFOUR iteration | 0.32 | 2.74 | 26.9 |
| `DirectZCCSD` iteration (AO ladder, <ma\|\|ef> terms AO-direct) | 2.40 (7.5x) | 23.5 (8.6x) | 130.7 (4.9x) |
| `ZCCSD` iteration (<ab\|\|cd> in a scratch file) | 1.99 (6.2x) | 17.9 (6.5x) | 106.9 (4.0x) |
| CFOUR (T) | 2.1 | 23.7 | 1500 |
| occ_loop (T), ovvv rebuilt from AO integrals first | 32.9 (16x) | 396 (17x) | about 21400 (14x) |
| integrals: CFOUR `xx2cvtran` / `DirectZCCSD` / `ZCCSD` | 0.5 / 1.1 / 13.5 | 3.4 / 10.1 / 253 | 11.9 / 29.7 / 230 |
| peak RSS: CFOUR / `DirectZCCSD` / `ZCCSD` (GB) | 0.8 / 1.4 / 3.5 | 3.7 / 9.4 / 16.1 | 13.7 / 22.8 / 16.5 |

- **`DirectZCCSD`**: the five <ma||ef> terms evaluated from the AO
  integrals (`_ovvv_fvv`, `_ovvv_ladder`, `_ovvv_t1`, `_ovvv_t2`,
  `_ovvv_wovvo`) take 19.1 of the 23.5 s for H2O/cc-pVQZ (81%) and 58 of
  the 131 s for benzene (44%); stored, <ma||ef> is only 1.7 GB and 4.3 GB.
  The AO ladder itself (`_ladder_lc`) takes 0.92 s and 12.6 s against
  CFOUR's 0.36 s and 3.4 s: it runs over all pairs (i, j), not i <= j.
- **`ZCCSD`**: `update_t2_vvvv_sym` (the packed MO ladder and its two tau
  terms) 6.6 s and 27.2 s (37% and 25%), the T1/T2 einsums inside
  `update_amps` 28-36%, W(mbej) 15-18%; `make_tau` is recomputed four times
  per iteration (9.3 s for benzene). Its integrals (all nmo^4 MO integrals
  through a scratch file of 43-45 GB) take 230-253 s.
- **Threads**: from 2 threads on a busy login node to 8 threads on c028
  (H2O/cc-pVTZ) CFOUR became 4.2 times faster, `DirectZCCSD` and `ZCCSD`
  1.7 times: much of the copied code is single-threaded numpy work
  (transposes and elementwise operations around the einsums), so with 2
  threads it looked only 2.5-3 times slower than CFOUR.
- **(T)**: the occ_loop spends 1.9 s (benzene) to 3.2 s (H2O/cc-pVQZ) per
  triple, most of it in single-threaded numpy passes over v^3 arrays; the
  vir_loop is slower still (136 s for H2O/cc-pVTZ on c543, 72 times CFOUR).
- On these molecules the copied code is not faster than zccsdt's own CCSD
  for H2O (1.67 s and 23.1 s on c146), but it runs benzene, where the
  integrals of zccsdt's CCSD did not fit 30 GB.

### Integrals of the copied CCSD: socutils' `nrr_fast` in slabs, within `max_memory`

The integral step of the copied code was its slowest part. `ZCCSD`
transformed all nmo^4 MO integrals with PySCF's `nrr_outcore` and read them
back to antisymmetrize and to pack <ab||ef>; `DirectZCCSD` used einsums on
the 4-index AO array. `zccsdt.zccsd.ao2mo` replaces both:

- **Kernels**: socutils' own optimized transformation `nrr_fast`
  (`lib/ao2mo` there; copied to `csrc/nrr_ao2mo_opt.c` and
  `zccsd/nrr_fast.py`): the bra pass on PySCF's s4 AO fill, the ket pass on
  the shell-block packing, the cheaper index order, all in C. (A first
  version, 41735e2, had kernels of its own -- real GEMMs on AO integrals
  packed in memory -- which were 5-20% slower and are gone.)
- **Organization**: only the integrals a CCSD needs, (ip|qr) with i
  occupied and, for `ZCCSD`, (ae|bf); the first MO index in batches; every
  slab antisymmetrized into its blocks (or packed <ab||ef> rows) at once.
  The half-transformed integrals stay in memory and nothing is read back.
- **Memory**: nothing that is computed is assumed to fit. `max_memory` (MB
  for the process, PySCF's meaning) sizes the batches of the first index and
  the ranges of AO shell pairs of the bra pass; the blocks stay in memory
  only up to a third of the budget, the smallest first, the others are
  mapped arrays of a scratch file filled slab by slab; below the working set
  of one orbital and one shell pair, `MemoryError`. Smaller batches compute
  the AO integrals more often; they do not write intermediates.

All numbers: 8 cores of one 2 x Xeon Gold 6248R node, everything of a table
one after the other in one job, CFOUR's chain up to `xx2cvtran` first
(`bench/slurm_ao2mo.sh`, `bench/profile_ao2mo.py`, `bench/profile_zccsd.py`).

**One block, four transformations** (job 31549591, c515; the in-memory run
of the stock kernels from job 31549669, c208, where the other three rows
were the same within 2%). (ip|qr) is what `xx2cvtran` makes:

| Seconds | H2O/cc-pVTZ | H2O/cc-pVQZ | C6H6/cc-pVDZ |
|---|---|---|---|
| CFOUR `xx2cvtran`: transformation (+ sorting) | 0.27 (+ 0.03) | 2.48 (+ 0.13) | 8.51 (+ 0.87) |
| PySCF `nrr_outcore` (no AO symmetry, HDF5) | 0.9 | 8.0 | 28.5 |
| its C kernels driven in memory (no AO symmetry) | 0.4 | 3.8 | 11.5 |
| socutils `nrr_fast` | 0.2 | 2.1 | 6.7 |
| zccsdt slabs (`nrr_fast` kernels) | 0.2 | 2.0 | 6.2 |

- `nrr_fast` is 1.2-1.3 times faster than CFOUR's transformation and about
  4 times faster than stock `nrr_outcore`: 2.1-2.5 times from not going
  through HDF5 and 1.75-2 times from the s4 symmetry of the AO integrals and
  the index order (first pass 2.4 times, second pass 1.4-1.6 times).
- For (ae|bf) the slabs bound the memory: H2O/cc-pVQZ 27.6 s and 13.3 GB
  with 24000 MB, 28.1 s and 6.6 GB with 8000 MB, against 28.2 s and 15.8 GB
  for one call of `nrr_fast`; benzene 15.9 s / 9.8 GB, 16.7 s / 5.6 GB,
  15.4 s / 11.2 GB.

**The integral step of the two classes** (the same job; AO integrals, the
Fock matrix by PySCF's direct spinor J/K (2 s), the antisymmetrized blocks
and the AO operators of the ladder or the packed <ab||ef> included):

| Seconds, peak anonymous memory | H2O/cc-pVQZ `DirectZCCSD` | H2O/cc-pVQZ `ZCCSD` | C6H6/cc-pVDZ `DirectZCCSD` | C6H6/cc-pVDZ `ZCCSD` |
|---|---|---|---|---|
| `max_memory` 24000 MB | 5.8, 5.5 GB | 34.9, 14.5 GB | 11.9, 10.8 GB | 26.6, 19.4 GB |
| 8000 MB | 6.0, 4.6 GB | 37.3, 7.6 GB | 13.9, 4.8 GB | 30.2, 7.5 GB |
| 4000 MB | 7.4, 2.1 GB | 39.6, 3.7 GB | 17.6, 2.6 GB | 36.7, 3.7 GB |
| `ZCCSD` through `nrr_outcore` (c195) | | 201 + 48 | | 195 + 16 |

- The peak stays below `max_memory`; with 4000 MB ovvv, ovov, ovvo and the
  packed <ab||ef> are in scratch files and a slab holds 2 to 8 orbitals.
- The old route of `ZCCSD` (the `ao2mo_outcore` switch; + the packing of
  <ab||ef> in its first iteration) spent 26 s of its 200 s in the C kernels:
  7 s + 18 s in the two passes, 13 s writing the half-transformed file, 58 s
  reading it and writing all MO integrals, 100-104 s reading them back into
  antisymmetrized blocks (job 31548804).
- Reserves: the Fock matrix could come from the AO integrals of the bra
  pass; a small budget repeats the AO integrals for every batch.

**The copied iteration on these integrals** (socutils' `update_amps`,
`max_memory` 24000 MB, c515): `DirectZCCSD` with ovvv stored 1.38 s, 9.66 s
and 94.3 s per iteration for the three molecules (with the ovvv terms from
the AO integrals, as copied: 2.3 s, 22.6 s, 127 s on c329); `ZCCSD` 1.48 s,
10.7 s and 92.9 s; CFOUR 0.32 s, 2.75 s, 26.7 s. Peak RSS for benzene 24-27
GB: the copied iteration is not planned (large einsum temporaries).

### The amplitude equations of `DirectZCCSD` on fixed layouts (`zccsd/amps.py`)

The same equations as socutils' `update_amps` (the two agree to 1e-16 per
element, `tests/test_zccsd.py`), evaluated differently:

- pairs packed, i > j and a > b; tau formed once;
- the two o^3 v^3 terms, W(mbej) += <mn||ef> t2 and the ring t2 W, as one
  ZGEMM each on matrices [(i a), (j b)];
- <ma||ef> stored packed over e > f (half of ovvv) and read once per
  iteration, one occupied index at a time, for its five terms;
- the AO ladder over the pairs i > j, the alpha-beta block of the
  back-transformed tau split into its symmetric and antisymmetric parts:
  two packed real operators of 2 nao^4 bytes each instead of the
  nao^2 x nao^2 matrix, and half the multiply-adds.

First version (numpy and one C kernel), 8 cores of c318, one job
(31550503): CFOUR first, then `DirectZCCSD` with these equations and with
socutils' `update_amps` on the same stored integrals. Seconds per iteration:

| | H2O/cc-pVTZ | H2O/cc-pVQZ | C6H6/cc-pVDZ |
|---|---|---|---|
| CFOUR `xx2cccsd` | 0.26 | 2.07 | 22.95 |
| socutils' `update_amps`, ovvv stored | | 7.03 | 75.0 |
| `zccsd/amps.py`, first version | 0.31 | 2.18 | 36.6 |
| peak memory of the iterations (GB): CFOUR / socutils / amps | 0.7 / / 0.7 | 2.2 / 7.9 / 2.9 | 11.0 / 23.7 / 18.6 |

Where the 36.6 s of benzene go: the two o^3 v^3 ZGEMMs 13.9 s (7.6e12
flops at 550 Gflop/s: the rate of the machine; CFOUR spends 6.0 s + 6.1 s
in the same two terms with real DGEMMs), the <ma||ef> terms 6.2 s (half of
it a transposed add in numpy), the remaining T2 terms 5.0 s and the ladder
4.3 s (CFOUR 3.1 s), both mostly index gathers on one thread, layouts of
t2 2.9 s, unpacking 1.1 s, DIIS, energy and vector conversions of PySCF's
driver 2.3 s. About 15 s are shuffles that a threaded kernel does in 1-2 s.

## Correctness references

| Reference | Agreement | Test |
|---|---|---|
| PySCF GCCSD (real and complex-rotated GHF) | 1e-10 | `tests/test_ccsd.py` |
| FCI, 3 electrons (real UHF; X2C spinor) | 1e-9 | `tests/test_ccsdt.py` |
| MRCC and CFOUR ncc/ecc/vcc CCSD and CCSDT, 5 small molecules | 1e-8 (observed about 5e-11) | `tests/test_references.py` |
| Determinant-space brute force, T3 residual element-wise | 1e-12 | `tests/test_ccsdt.py` |
| Invariance under complex occ-occ / vir-vir rotations (RHF -> GHF -> U) | 1e-10 | `tests/test_ccsdt.py` |
