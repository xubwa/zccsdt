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
fit"); here is what it costs.  Each run is a Slurm job step of 8 cores with a
memory limit (`srun --mem`), which zccsdt sees through its memory cgroup; the
plan is the automatic one (budget: 80% of what the limit leaves).  c233, the
node of the other measurements, scratch on its local NVMe (2.4 GB/s
sequential read), zccsdt ec877db - 75cc5c4.  "In memory" is the same
calculation without a limit; under every limit of the table it needs more
than the limit and is killed (the (Q) step refuses to start).

| System | In memory: needs, time | Limit | Goes to the scratch file | Time | Anonymous memory |
|---|---|---|---|---|---|
| H2O/cc-pVQZ, iteration | 4.5 GB, 5.15 s | 3 GB | T3, residual; pp operator in 6 slabs | 9.5-9.9 s | 2.4 GB |
| C2H4/cc-pVTZ, iteration | 7.6 GB, 19.4-20.1 s | 6 GB | T3, residual | 28.1 s | 4.3 GB |
| C2H4/cc-pVTZ, iteration | | 4 GB | T3, residual; pp operator in 6 slabs | 42.7 s | 3.3 GB |
| C6H6/cc-pVDZ, frozen core, iteration | 13.7 GB, 85.7 s | 8 GB | T3, residual | 155 s | 4.5 GB |
| H2O/cc-pVQZ, (Q) | 14 GB, 102 s | 6 GB | (v^4 arrays in slabs) | 194-225 s | 5.2 GB |

The DIIS vectors (2 x 8 T3-sized vectors) are in the scratch file in all
runs, also "in memory".  The node was shared with other jobs during these
runs (the benzene run at a 1-minute load of 15); the same set is queued as
separate Slurm jobs (`bench/slurm_memory.sh`).

- **The runs complete, with the energies of the runs in memory** (the (Q)
  correction to all printed digits), where the version without these
  changes was killed by the memory limit.
- **Cost: 1.4-2.2 times the time in memory** when a third to a half of the
  memory is missing; the sliced (Q) algorithm takes about twice the block
  algorithm.
- **Where the time goes: reading and writing T3.**  With T3 in the file an
  iteration reads it about 8 times and reads and writes the residual about
  8 times (three touches in the pair pass, three in the k pass, the ladder,
  the null projection, the Jacobi step), and DIIS reads its 16 vectors and
  writes 3: about 40 T3 volumes per iteration, 180 GB for benzene, at
  2 GB/s unless the page cache holds them.  The time above the run in memory
  is this.  A smaller `diis_space` reduces it (and the scratch space, 18 T3
  volumes by default); `diis_t3=False` (DIIS on T1 and T2 only) removes the
  DIIS part at the price of about a quarter more iterations.
- **The pp operator costs little to slice**: it is passed over once per
  iteration.  H2O/cc-pVQZ without a limit, but with an extreme plan (T3
  mapped, the operator in 10 slabs of 64 MB, one block per product): 7.9 s
  instead of 5.2 s per iteration, the difference being thin GEMMs and
  repeated packing.
- **What must still fit**: the ovvv-sized arrays, about 23 o v^3 elements
  (1.5 GB for H2O/cc-pVQZ, 2.7 GB for benzene/cc-pVDZ, 25 GB for
  H2O/cc-pV5Z, 46 GB for benzene/cc-pVTZ with frozen core); below that
  `MemoryError` is raised at the start.  For (Q) in slabs: 6 (26 + o^2) v^3
  elements (3.6 GB for H2O/cc-pVQZ, 60 GB for v = 282 and o = 5, where the
  block algorithm needs 506 GB).

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

## Correctness references

| Reference | Agreement | Test |
|---|---|---|
| PySCF GCCSD (real and complex-rotated GHF) | 1e-10 | `tests/test_ccsd.py` |
| FCI, 3 electrons (real UHF; X2C spinor) | 1e-9 | `tests/test_ccsdt.py` |
| MRCC and CFOUR ncc/ecc/vcc CCSD and CCSDT, 5 small molecules | 1e-8 (observed about 5e-11) | `tests/test_references.py` |
| Determinant-space brute force, T3 residual element-wise | 1e-12 | `tests/test_ccsdt.py` |
| Invariance under complex occ-occ / vir-vir rotations (RHF -> GHF -> U) | 1e-10 | `tests/test_ccsdt.py` |
