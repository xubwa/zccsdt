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
Section 2) is the reference for zccsdt's spinor CCSD(T),
`zccsdt.zccsd.DirectZCCSD`. The subsections follow its history: zccsdt's
CCSD as the CCSD part of its CCSDT code, socutils' CCSD and (T) copied into
zccsdt, their integrals, the amplitude equations and (T) rewritten, Kramers
pairs; the last subsection is where it stands.

**Setup.**

- One Slurm job, `bench/cfour/slurm_x2ccc.sh` (job 31533171, 2026-10-02):
  8 cores of a 2 x Xeon Gold 6248R node, CFOUR first and then zccsdt, one
  molecule after the other.
- CFOUR (cfour-x2cso 35b1f40e3): `RELATIVISTIC=X2C1E`, `X2CKEY=SOATMX`,
  `ABCDTYPE=AOBASIS`, all electrons, C1, `CC_CONV=7` (largest element of
  the amplitude step). Times per residual evaluation from its own timers
  (`bench/cfour/x2ccc_summary.py`), the first evaluation left out; peak RSS
  from `/usr/bin/time`.
- zccsdt (bd8a2a8): X2C-1e of PySCF (`x2c.UHF`). (The energies were not
  compared at first; converged, the two programs agree to 5e-8 Eh in SCF,
  CCSD and (T): the last subsection.)
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

### The amplitude equations of `DirectZCCSD` on packed amplitudes (`zccsd/amps.py`)

The same equations as socutils' `update_amps` (the two agree to 1e-12 per
element for random complex amplitudes, `tests/test_zccsd.py`), evaluated
differently:

- T2, tau and the residual are packed, [(i>j), (a>b)], from the first guess
  to the converged amplitudes: the iterations are PySCF's (Jacobi step, DIIS
  on [t1, packed t2], the same convergence test; the iterates agree to
  1e-13), without ever unpacking T2. Every layout a term needs comes from an
  OpenMP kernel (`csrc/zccsd_kernels.c`, numpy equivalents in
  `zccsd/kernels.py`), not from fancy indexing;
- the two o^3 v^3 terms, W(mbej) += <mn||ef> t2 and the ring t2 W, are one
  ZGEMM each on matrices [(i a), (j b)], in which t2 and <mn||ef> are
  symmetric: rows and columns of t2 are generated from the packed
  amplitudes, the two-index intermediates that contract <mn||ef> with t2
  are partial traces of the first product;
- <ma||ef> is stored packed over e > f (half of ovvv): one GEMM with tau,
  and the t1 terms of one occupied index at a time from rows unpacked in
  the cache;
- W(mnij) and the occupied-occupied Fock intermediate act on the packed
  pairs as one matrix;
- the AO ladder over the pairs i > j, the alpha-beta block of the
  back-transformed tau split into its symmetric and antisymmetric parts:
  two packed real operators of 2 nao^4 bytes each instead of the
  nao^2 x nao^2 matrix, and half the multiply-adds (kept, or made from the
  AO integrals in every iteration: the section on the AO operators below);
- the integrals arrive in the layouts of the terms (`zccsd/ao2mo.py`:
  <mn||ef> and <mb||je> as [m, e, n, f], <mn||ef> on packed pairs).

Seconds per iteration, 8 cores, each row block one job on one node with
CFOUR first (CFOUR varies by 15% between nodes of the same CPU):

| | H2O/cc-pVTZ | H2O/cc-pVQZ | C6H6/cc-pVDZ |
|---|---|---|---|
| job 31550503 (c318): CFOUR `xx2cccsd` | 0.26 | 2.07 | 22.95 |
| socutils' `update_amps`, ovvv stored | | 7.03 | 75.0 |
| fixed layouts, numpy gathers (57d39be) | 0.31 | 2.18 | 36.6 |
| job 31552287 (c098): CFOUR `xx2cccsd` | 0.32 | 2.72 | 26.76 |
| packed amplitudes, kernels, one pool of threads | 0.10 | 0.77 | 18.9 |
| peak memory of the iterations (GB): CFOUR / zccsdt | 0.7 / 0.6 | 2.2 / 2.5 | 11.0 / 10.4 |

Where the time goes (job 31552287, seconds per iteration):

| C6H6/cc-pVDZ | CFOUR | zccsdt |
|---|---|---|
| the two o^3 v^3 products (CFOUR: I7 and the ring) | 12.84 | 12.51 |
| other intermediates; the terms with <ma||ef> | 3.43 | 3.39 (GEMM with tau 1.62, t1 terms 1.19, small ring terms 0.58) |
| pp ladder in the AO basis | 3.39 | 1.82 (real GEMMs 1.02, transformations 0.66, packing 0.14) |
| hh ladder, T1/T2 terms, layouts, DIIS, update | 7.10 | 1.16 (t2 terms 0.46, intermediates 0.28, DIIS and energy 0.20) |
| sum | 26.76 | 18.88 |

- The two o^3 v^3 products run at the rate of the machine in both programs
  (7.6e12 flops in 12.5 s: 610 Gflop/s; CFOUR does four real DGEMMs for each
  at the same rate). They are two thirds of zccsdt's iteration.
- For H2O/cc-pVQZ (o = 10, v = 220) the terms with <ma||ef> (0.24 s), the
  ring products (0.29 s) and the ladder (0.14 s) are 0.67 of 0.77 s; CFOUR
  spends 1.06 s in its intermediates and 1.14 s outside its timers.

**One pool of threads.** The kernels are OpenMP; numpy's BLAS (OpenBLAS
with its own pthreads here) keeps its threads spinning for about 0.1 s
after every call, on the cores the next OpenMP region wants. With the
products of the step in numpy, H2O/cc-pVTZ took 0.20-0.37 s per iteration
and H2O/cc-pVQZ 1.06-1.48 s depending on which library did which product
and on `OMP_WAIT_POLICY` (job 31552066). All products of the step now go
through `zccsdt.blas`, i.e. the BLAS of the kernel library (MKL on GNU
OpenMP when `libzc_ccsdt_t3_mkl.so` is built): kernels and BLAS share one
pool, numpy's BLAS is not called with work for its threads, and the wait
policy no longer matters (0.104 / 0.771 / 18.9 s with `PASSIVE`, 0.103 /
0.760 / 18.8 s without).

**Memory.** Nothing but three packed T2 and smaller arrays is assumed to
fit. `amps.plan_step` slices the step from what `max_memory` leaves: pass 1
over slabs of the first occupied index (the rows of W(mbej): the integrals
<mn||ef>, <mb||je>, <ma||ef> are each read once, by rows, from memory or
from the scratch file of the integral step), pass 2 over blocks of the
columns; W(mbej) (o^2 v^2 elements) is in memory only if the smallest
blocks still fit next to it, else a mapped scratch array; the DIIS vectors
are in memory if they take no more than a third of what is left. Smaller
blocks mean that rows of t2 are generated more often. Benzene/cc-pVDZ, the
same job:

| `max_memory` | integrals in scratch files | plan of the step | s / iteration | peak anonymous memory |
|---|---|---|---|---|
| 24000 MB | none | one slab, one block; W and DIIS in memory | 18.9 | 10.4 GB |
| 8000 MB | ovvv_p | slabs of 11, blocks of 32 of 42 occupied; W in memory, DIIS in a file | 19.9 | 7.6 GB |
| 4000 MB | <mn||ef>, <mb||je>, ovvv_p | slabs of 2, blocks of 6; W and DIIS in files | 23.9 | 3.6 GB |

With 4000 MB the iteration is still faster than CFOUR's in 11 GB. H2O/
cc-pVQZ with 4000 MB runs as with 24000 MB (0.78 s, 2.7 GB). Below the
working set of one occupied index (three packed T2 plus about 5 o v^2
elements and one slab of <ma||ef>) `plan_step` raises `MemoryError`.

**Three real GEMMs for a complex product.** A complex product A B is
A_r B_r, A_i B_i and (A_r + A_i)(B_r + B_i): three real GEMMs instead of the
four a ZGEMM amounts to. For a 7812 x 7812 product on 8 cores MKL takes
5.97 s as ZGEMM and 1.52 s per DGEMM (OpenBLAS: 6.2 s and 1.93 s; its own
`zgemm3m` 8.2 s, MKL's is its ZGEMM; job 31552066). `amps` splits the
operands (t2 comes from the kernel in parts) and combines the three results
with threaded kernels. Job 31552593 (c182), seconds per iteration:

| | H2O/cc-pVTZ | H2O/cc-pVQZ | C6H6/cc-pVDZ |
|---|---|---|---|
| CFOUR | 0.32 | 2.74 | 26.82 |
| one ZGEMM per product | 0.102 | 0.767 | 18.72 (two products 12.52) |
| three real GEMMs | 0.106 | 0.785 | 16.54 (two products 10.05, splitting 0.31 more) |
| peak memory (GB), ZGEMM / three GEMMs | 0.60 / 0.64 | 2.5 / 2.7 | 10.4 / 12.8 |

The passes over the (o v)^2 arrays cost about as much as the GEMMs save
near o v = 2300, so the three real GEMMs are the default from o v = 3000
on (`amps.THREE_M_MIN`), and only while the blocks of columns that fit
with the real parts cover a quarter of the occupied space: with
`max_memory` 4000 MB for benzene (blocks of 3) they took 27.1 s per
iteration against 23.9 s for ZGEMM in blocks of 6.

The <ma||ef> t1 terms now come from the packed rows directly, like a packed
symmetric matrix-vector product (no row is unpacked).

With both, job 31553091 (c255): 0.085 s, 0.634 s and 15.2 s per iteration
against CFOUR's 0.26 s, 2.09 s and 23.12 s. Of the 15.2 s for benzene the
six real GEMMs of the o^3 v^3 terms are 9.4 s, the terms with <ma||ef> 2.4 s
(GEMM with tau 1.5 s, t1 terms 0.9 s), the ladder 1.7 s; of the 0.63 s for
H2O/cc-pVQZ the two ZGEMMs 0.28 s, <ma||ef> 0.16 s, the ladder 0.13 s.
Benzene with `max_memory` 8000 MB: 17.3 s per iteration, 7.6 GB.

### (T) in batches of virtual pairs (`zccsd/triples.py`)

The copied loops (`gccsd_t`) spend their time in numpy passes over v^3 or
o^3 elements on one thread: 33 s, 396 s and about 21000 s for the three
molecules against CFOUR's 2 s, 24 s and 1500 s. `triples.kernel` keeps
everything on packed pairs. With

    x_{a,(bc)}[i, (jk)] = sum_e <bc||ei> t2[j,k,a,e] - sum_m t2[i,m,b,c] <ma||jk>,

an o x o(o-1)/2 matrix for a virtual a and a pair b > c, the connected
triples of a > b > c are Z = x_{a,(bc)} - x_{b,(ac)} + x_{c,(ab)} and
w[i>j>k] = Z[i,(jk)] - Z[j,(ik)] + Z[k,(ij)]. The x of a batch (one a, a
range of b, all c < b) are GEMMs on contiguous blocks of <bc||ei> (the
stored <ie||bc>, transposed once), of t2 and of <ma||jk>; each x is made
once (o^3 v^4 / 4 multiply-adds for the particle term in all). An OpenMP
kernel assembles Z for one (b, c) per thread in its cache and adds up the
energy. Same energies as the copied loops and PySCF's `gccsd_t` to 1e-12,
also for a non-canonical reference.

First version (ZGEMM, separate products for the particle and the hole
term), job 31553091 (c255), CFOUR's CCSD(T) first on the same node:

| (T), seconds | H2O/cc-pVTZ | H2O/cc-pVQZ | C6H6/cc-pVDZ |
|---|---|---|---|
| CFOUR | 1.6 | 17.5 | 1154 |
| `triples.kernel` | 1.0 | 12.3 | 589 |
| of which GEMMs / kernel | 0.7 / 0.1 | 10.8 / 1.3 | 516 / 68 |

The flops of benzene (2.2e14) would take 355 s at the rate of the machine;
the rest is memory traffic: the x of all batches are 1.9 TB, written by the
particle product, read and written again by the hole product (a rank-42
update) and read by the kernel.

**One product per x, ranges of a, arrays that are reused.** What changed
after the first version:

- The particle and the hole term are one product over the joint index
  (e | m): with A[(b>c), i, (e|m)] = (<bc||ei> | t2[i,m,b,c]) and
  B[a, (j>k), (e|m)] = (t2[j,k,a,e] | -<ma||jk>) every x is one GEMM result,
  written once.
- A batch is a range of a with all b < a where that fits (two GEMMs: all
  pairs below the range against the a of the range, the pairs (a, x)
  against all y), else one a and a range of b (three GEMMs). H2O runs in
  ranges of a, benzene mostly in ranges of b.
- The x of all batches are written into the same arrays, allocated once
  for the largest batch. An array allocated per batch is mapped and zeroed
  by the system page by page while the GEMM writes it. For the shapes of
  one benzene batch (one a, 22 b; job 31554522, c152, 8 cores, seconds):

  | product, rows x columns (x 228) | ZGEMM, new array | ZGEMM, reused | 3 DGEMM, new arrays | 3 DGEMM, reused |
  |---|---|---|---|---|
  | x_{a,(bc)}: 101178 x 861 | 0.344 | 0.279 | 0.357 | 0.268 |
  | x_{y,(ax)}: 5082 x 18942 | 0.331 | 0.326 | 0.274 | 0.220 |
  | x_{y,(ax)}: 924 x 85239 | 0.285 | 0.232 | 0.244 | 0.197 |
  | sum | 0.960 | 0.837 | 0.875 | 0.685 |

  The reused real products run at 450-600 Gflop/s.
- The kernel keeps one o x o(o-1)/2 array per thread (Z; 0.58 MB for
  benzene, in the L2 cache), reads Z[k, (ij)] from it with a stride and
  forms the disconnected part inside the loop over i > j > k from three
  columns of t1 and three rows of <bc||jk>. A version that vectorized the
  loop over transposed copies of Z and of the disconnected part (four
  arrays, 2.3 MB per thread) was slower than the first one, which kept two:
  108 s for benzene with ZGEMM (job 31553298, c208) against 68 s (job
  31553091, c255, a node on which CFOUR is 1.3 times faster).
- The tasks of the threads are the pairs (b, c), not the b: a batch with
  one or two b (small batches under a memory limit) left most threads idle
  (job 31553298: 330 s in the kernel for benzene with `max_memory` 8000 MB),
  and with dynamic scheduling over b the long last b ended alone. On
  random arrays of benzene's dimensions one pair takes 0.4 ms of a thread
  (0.5 ms before): half of it to read the three x from memory (1.7 MB),
  half for the 11480 triples i > j > k.
- Three real GEMMs per product (`three_m`), the three real results going
  to the kernel as they are (it reads 24 instead of 16 bytes per element).
  With new arrays per batch they did not pay for benzene (job 31553298,
  c208: 695 s, of which GEMMs 537 s and kernel 146 s, against 660 s, 544 s
  and 108 s with ZGEMM; H2O/cc-pVQZ 13.3 s against 14.6 s).
  With the arrays reused the real GEMMs are a quarter faster where the
  batches are ranges of a (H2O/cc-pVQZ: 3.6 s instead of 4.9 s) and 4%
  faster for the batches of benzene (185 s instead of 193 s), where the
  kernel loses more than that (48.5 s instead of 34.6 s). Default: the real
  parts where both operands and a whole batch fit in memory with them (an
  operand in a scratch file or small batches cost more than the GEMMs
  save), else ZGEMM.
- Kramers pairs (next section): the triples (a, b, c) and (Ka, Kb, Kc)
  contribute complex-conjugate energies, so only the triples with b in the
  unbarred half are done and the sum is doubled. If the amplitudes do not
  have the symmetry (`kramers.symmetric`), all triples are done.

Seconds, 8 cores, CFOUR's CCSD(T) first on the same node; (T) on the
amplitudes of the fourth CCSD iteration, the same energy to 1e-12 in all
variants:

| (T), seconds | H2O/cc-pVTZ | H2O/cc-pVQZ | C6H6/cc-pVDZ |
|---|---|---|---|
| job 31554690 (c081): CFOUR | 2.1 | 23.9 | 1499 |
| all triples, three real GEMMs | 0.9 | 10.0 | 464.9 |
| all triples, ZGEMM | 1.3 | 12.5 | 483.2 |
| Kramers pairs, three real GEMMs | 0.6 | 6.3 | 235.0 |
| of which GEMMs / kernel | 0.4 / 0.1 | 4.8 / 1.2 | 185.4 / 48.5 |
| Kramers pairs, ZGEMM | 0.5 | 6.4 | 228.5 |
| of which GEMMs / kernel | 0.4 / 0.1 | 5.3 / 1.0 | 193.0 / 34.6 |
| peak memory (GB): CFOUR / three real GEMMs / ZGEMM | 0.7 / 2.2 / 1.7 | 2.2 / 5.7 / 5.2 | 11.1 / 14.0 / 12.5 |
| job 31556042 (c415): CFOUR | 2.0 | 23.6 | |
| Kramers pairs, the barred a in ranges, three real GEMMs | 0.5 | 5.1 | |
| the same with ZGEMM | 0.5 | 6.0 | |

- With Kramers pairs the ZGEMMs of benzene do 1.04e14 flops in 193 s: 540
  Gflop/s on a node whose CFOUR runs are the slower ones, the rate of the
  machine. What is left of (T) for benzene is the kernel (15-20%).
- For barred a the pairs (b, c) are the same for every a, those below the
  barred half: a range of a shares one product for its x_{a,(bc)}. With one
  a per batch the 5995 pairs of H2O/cc-pVQZ were read from A once for every
  a: 6.3 s, and 5.1 s with ranges (GEMMs 4.8 s and 3.6 s; the last rows of
  the table, both versions in the same job). The batches of benzene do not
  hold a range of a.
- zccsdt's (T) takes more memory than CFOUR's for small molecules: the
  batches are sized by `max_memory` (2 GB of x at most in these runs), not
  by the molecule. Benzene's (T) is as fast with `max_memory` 8000 MB (the
  larger operand in a scratch file) as with 24000 MB: the table of the last
  subsection.

### Kramers pairs (`zccsd/kramers.py`)

A closed-shell X2C reference without a magnetic field is symmetric under
time reversal K, K (a, b) = (-b*, a*) on the two components of a spinor: the
spinors come in degenerate pairs (p, Kp), and in a basis of such pairs every
tensor of the theory obeys

    X[Kp, Kq, ...] = s_p s_q ... conj(X[p, q, ...]),

s = +1 for the first member of a pair ("unbarred"), -1 for the second. The
spinors of an SCF are pairs only up to a phase, and not even that in a level
that is degenerate for another reason as well (a four-fold level of a
molecule with spatial symmetry). `kramers.pair` rotates the spinors within
each degenerate level into exact pairs and orders them [unbarred, barred]
among the occupied and among the virtual spinors, so that K is a shift by
half the dimension; the Fock matrix stays diagonal to the convergence of
the SCF (1e-7 here). `DirectZCCSD` does this when the reference allows it
(`kramers = None`; an open shell or a reference that is not symmetric is
detected level by level and runs without). The amplitudes then refer to the
paired spinors, `mycc.eris.mo_coeff`. CFOUR's `xx2cccsd` does not use the
symmetry.

Half of every expensive term is computed, the other half is its image:

- pass 1 computes the rows (m e) of W(mbej) with unbarred m (the product
  <mn||ef> t2, the t1 terms, the terms with <ma||ef>: the integrals with
  barred m are not read) and fills the rows of the partners in one pass
  over W;
- pass 2 computes the rows (i a) of t2 W with unbarred i; a block of
  columns is a range of unbarred j and their partners, so that the
  antisymmetrizer finds both;
- the AO ladder is done for one pair (i > j) of each two partners (ij),
  (Ki Kj);
- (T) does the virtual triples with b in the unbarred half.

The integral transformation is not halved. The step agrees with the general
one to 1e-12 per element for symmetric amplitudes (also in slabs and
blocks), and (T) on half of the triples with (T) on all of them to 1e-12
(`tests/test_zccsd.py`, on a molecule with a four-fold level).

Seconds per CCSD iteration, 8 cores, CFOUR first on the same node:

| | H2O/cc-pVTZ | H2O/cc-pVQZ | C6H6/cc-pVDZ |
|---|---|---|---|
| job 31554690 (c081): CFOUR `xx2cccsd` | 0.33 | 2.76 | 26.89 |
| the spinors of the SCF (`kramers = False`) | 0.099 | 0.710 | 16.56 |
| Kramers pairs | 0.067 | 0.428 | 9.95 |
| job 31554768 (c228): CFOUR `xx2cccsd` | 0.26 | 2.06 | 22.98 |
| Kramers pairs, intermediates completed by half-block copies | 0.055 | 0.368 | 8.36 |
| peak memory of the iterations (GB): CFOUR / zccsdt | 0.7 / 0.6 | 2.2 / 2.4 | 11.1 / 11.1 |

Where the time goes (job 31554768, seconds per iteration):

| C6H6/cc-pVDZ | CFOUR | zccsdt, Kramers pairs |
|---|---|---|
| the two o^3 v^3 products (CFOUR: I7 and the ring) | 12.16 | 4.69 |
| other intermediates; the terms with <ma\|\|ef> | 2.53 | 1.97 (GEMM with tau 0.76, t1 terms 0.48, small ring terms 0.24, layouts 0.25, W(mnij) 0.24) |
| pp ladder in the AO basis | 3.14 | 0.97 (real GEMMs 0.50, transformations 0.33, packing 0.10) |
| hh ladder, T1/T2 terms, DIIS, update | 5.16 | 0.73 (t2 terms 0.40, completion by K 0.12, DIIS and energy 0.10) |
| sum | 22.98 | 8.36 |

The six real GEMMs of the two o^3 v^3 terms (half of the rows each) run at
610 Gflop/s and are 56% of the step; for H2O/cc-pVQZ the two ZGEMMs
(0.14 s), the terms with <ma||ef> (0.09 s) and the ladder (0.08 s) are 0.30
of 0.37 s.

The three- and four-index intermediates that pass 1 makes for unbarred m
only (t1 <mb||je>, the <ma||ef> tau array) are completed by half-block
copies: K on a single spinor index is a swap of the two halves with a sign.
The first version did it with index arrays, one pass and one temporary per
axis, single threaded: 0.48 s of 8.74 s for benzene, outside every timer;
0.12 s now (`amps.kramers`; the same job: 8.74 s before, 8.36 s after).

What is left of a step is not in one place. A step of benzene allocates
about 10 GB of work arrays that the system maps and zeroes page by page;
with a malloc that keeps freed memory (`MALLOC_MMAP_MAX_=0`,
`MALLOC_TRIM_THRESHOLD_` large) the iteration took 8.20 s instead of 8.73 s
(14.50 s instead of 14.96 s without Kramers pairs; job 31554542, c094), at
1.8 GB more memory. zccsdt does not change the allocator of the process and
does not keep its work arrays between steps, which would have to come out of
the memory plan: these 3-6% are left.

### The AO operators of the ladder: kept or integral-direct (`zccsd/aoladder.py`)

The pp ladder contracts the AO amplitudes of every pair (i>j) with two real
matrices on pairs of AO functions, the antisymmetrized integrals W_same and
the symmetrized W_plus: 2 N^4 bytes each as square matrices for N scalar AO
functions (0.7 GB together at N = 115, 19 GB at N = 264, 250 GB at
N = 500). They were made from all N^4 integrals and kept in memory or, when
`max_memory` did not allow that, in a scratch file that was read by rows in
every iteration, once for every block of pairs. Two measurements decided
what to do when they do not fit:

- reading them from the node-local NVMe disk (a 63.7 GB operator, N = 423,
  30 GB of memory, 8 cores; job 31573334, c051): 1.42 GB/s as the code did
  it (GEMM on mapped rows), 1.76 GB/s with `MADV_SEQUENTIAL`, 1.86 GB/s
  with `read()`; written at 0.97 GB/s;
- computing them with libcint in blocks of the first index (job 31575083,
  c105, 8 cores): 1.0e8 integrals/s for C6H6/cc-pVDZ, 2.6e8 for
  C6H6/cc-pVTZ and H2O/unc-cc-pV5Z, 3.1e8 for H2O/cc-pVQZ, 4.5e8 and 5.6e8
  for HI in uncontracted ANO-RCC and cc-pVTZ-DK.

Both operators are symmetric matrices, so the triangles are enough: with
T = W[U, :c1] for rows U of a batch of shells (the rows c0 <= r < c1),
G[:, :c1] += A[:, U] T and G[:, U] += A[:, :c0] T[:, :c0]^T make A W once.
The rows of a batch need only the integrals (f x|t y) with f in the batch
and x, t, y up to its end, and libcint computes them with the symmetry in
(t, y): N^4/8 integrals for a pass over both operators (0.13-0.14 N^4 in
the runs below, for batches of several functions) instead of N^4. A kernel
turns a block of them into rows of both operators (`zc_ladder_rows`: with
M[x, y] = (f x|t y), M - M^T on x > y and M + M^T on x >= y). At the rates
above that is less time than reading even the triangles from the disk, it
scales with the cores, and nothing of size N^4 is stored. So
(`DirectZCCSD.ao_ladder`; `None` chooses between the first two):

- `'memory'`, if they fit into the share of `max_memory` for stored
  integrals (a third): the rows are kept in parts of about equal numbers of
  rows. One part is the square matrices, one GEMM each as before; more
  parts tend to the triangles and half the memory. The fewest parts that
  fit are used (1, 2, 4, ... up to 32). If the CCSD step then does not fit
  next to them, they are dropped and the ladder is direct (with
  `ao_ladder = 'memory'` that is a `MemoryError`).
- `'direct'`, if not even the triangles fit: the integrals are computed in
  every iteration, a batch of shells at a time, turned into rows about a
  thousand at a time (fewer for a small basis) and multiplied. A block of
  pairs of the ladder is then as large as the memory allows next to a unit
  of integrals, since the integrals are gone through once per block. A
  shell that is too much for the memory at once comes with ranges of the
  third index; below one shell against one shell, `MemoryError`.
- `'file'`: the triangles in a scratch file, on request.

All modes give the same step to 1e-12 per element (`tests/test_zccsd.py`,
also with the units at their smallest and with the numpy kernels).

Seconds, 8 cores, `max_memory` 24000 MB, one job (31577428, c282), the code
before last in the same job:

| | H2O/cc-pVQZ | C6H6/cc-pVDZ | H2O/cc-pV5Z | C6H6/cc-pVTZ |
|---|---|---|---|---|
| N; occupied, virtual spinors | 115; 10, 220 | 114; 42, 186 | 201; 10, 392 | 264; 42, 486 |
| the two square matrices | 0.70 GB | 0.68 GB | 6.53 GB | 19.43 GB |
| **before**: the operators | in memory | in memory | W_same in memory, W_plus in a file | in a file |
| integral step, of which the operators | 5.1, 0.94 | 10.0, 1.97 | 51.7, 11.60 | 378.0, 38.16 |
| iteration, of which the ladder | 0.365, 0.077 | 8.320, 0.968 | 2.117, 0.615 | 213.4, 45.2 |
| **now**, as `None` chooses | memory, 1 part | memory, 1 part | memory, 2 parts (4.91 GB) | direct |
| integral step, of which the operators | 4.6, 0.41 | 8.6, 0.55 | 43.7, 3.35 | 335.2, - |
| iteration, of which the ladder | 0.380, 0.080 | 8.390, 1.017 | 1.921, 0.589 | 185.7, 23.8 |
| **`'direct'`**: iteration, of which the ladder | 0.530, 0.236 | 8.823, 1.451 | 3.593, 2.260 | (the row above) |
| integrals per iteration, in units | 2.46e7 in 16 | 2.30e7 in 23 | 2.22e8 in 22 | 6.33e8 in 44 |
| of the ladder: libcint / rows / GEMMs | 0.110 / 0.028 / 0.055 | 0.276 / 0.030 / 0.629 | 1.370 / 0.239 / 0.478 | 2.50 / 0.71 / 15.8 |
| **more parts in memory**: iteration | 6 parts (0.41 GB): 0.382 | 6 parts (0.39 GB): 8.424 | 17 parts (3.48 GB): 2.004 | |
| **`'file'`**: iteration, of which the ladder | | 6 parts: 8.446, 1.042 | | 32 parts (10.04 GB): 188.2, 25.3 |

- **Benzene/cc-pVTZ**, where 19.4 GB of operators do not fit: 185.7 s per
  iteration instead of 213.4 s, and an integral step of 335 s instead of
  378 s. Before, the ladder read the 19.4 GB once for each of its four
  blocks of pairs (40.3 s in the products on mapped rows). Now one block
  holds all pairs and the integrals are gone through once: 2.5 s of libcint
  for 6.33e8 integrals (2.5e8 per second, the rate measured above), 0.7 s
  for the rows and 15.8 s in the products of 51 chunks of rows. With the
  triangles in a file instead (`'file'`: 10.04 GB in 32 parts, written in
  12 s) the products on mapped rows take 20.4 s where libcint, rows and
  products of the direct ladder take 19.0 s, and the iteration 188.2 s:
  computing is faster than reading, as the two rates said.
- **When they fit, keeping them is faster**, most of all for few electrons
  in a large basis: for H2O/cc-pV5Z the integrals of every iteration are
  1.4 s of libcint next to an iteration of 1.9 s. That is why `None` keeps
  them whenever it can, in parts if that is what fits: two parts (4.91 GB
  instead of 6.53 GB) keep both operators of H2O/cc-pV5Z in memory where
  one of the squares went to a file. For benzene/cc-pVDZ the direct ladder
  costs 0.43 s of 8.4 s (libcint 0.28 s; the products in 24 thin chunks of
  rows 0.63 s instead of 0.51 s).
- **Making the kept operators** takes 0.41, 0.55 and 3.35 s where the
  squares from all N^4 integrals took 0.94, 1.97 and 11.60 s.
- The rows in more parts than needed (the triangles, half the memory) cost
  0.4-4% of an iteration. In this job a single part still went through
  the rule for the parts, a GEMM with beta = 1 into a zeroed array: 0.380
  and 8.390 s per iteration against 0.365 and 8.320 s before, 0.05 s of
  the difference for benzene in the ladder. One part is now multiplied as
  it was before, one GEMM per square matrix into a new array, and the
  default is as fast as the code before (the next table).

The code as it is now against the code before, the same way (job 31582707,
c436, a node on which benzene/cc-pVDZ takes 9.4 s per iteration, not 8.3 s),
with the limits of the memory:

| seconds | H2O/cc-pVQZ | C6H6/cc-pVDZ | C6H6/cc-pVDZ, 8000 MB | C6H6/cc-pVDZ, 4000 MB |
|---|---|---|---|---|
| before: iteration, of which the ladder | 0.420, 0.085 | 9.368, 1.061 | 11.354, 1.059 | 14.064, 1.091 |
| now: iteration, of which the ladder | 0.426, 0.087 | 9.374, 1.063 | 11.070, 1.061 | 12.955, 1.105 |
| before: integral step | 5.6 | 11.7 | 13.4 | 17.0 |
| now: integral step | 5.1 | 10.2 | 11.8 | 14.4 |
| before: slabs and blocks of the step (of 42) | | one, one | 9, 13 | 2, 5 |
| now: slabs and blocks of the step | | one, one | 10, 15 | 2, 8 |
| peak memory of the iterations (GB), before / now | 2.39 / 2.41 | 11.11 / 11.50 | 7.34 / 7.47 | 3.52 / 3.51 |

In all of these the operators fit (0.68 GB, one part) and the ladder is the
same. Under a limit the iteration is 2.5% and 8% shorter. The step has
0.19 GB more to plan with, hence larger slabs and blocks: the construction
of the operators before, numpy gathers in threads, left about 0.2 GB of
anonymous memory that was no array any more when the budget made its
batches small (219 MB more than the operators with 4000 MB, none with
24000 MB, not given back by `malloc_trim`; measured on the operators alone,
on the login node), and `max_memory` counts what the process holds. The new
construction leaves nothing. With ZGEMM (4000 MB) the copy of the column
blocks of W(mbej) is gone as well (below); the two were not measured apart.

**The plan of the step, checked against the allocations.** The three
benzene/cc-pVTZ runs above peaked at 26.8 GB of anonymous memory under
`max_memory` 24000 MB, the code before as well: the step with Kramers pairs
had two arrays that were not in its plan, a copy of the two column blocks
of W(mbej) for the ZGEMM of pass 2 (a range of j and their partners) and
the rows of tau and of the result that the ladder gathers for one pair of
each two partners. The copy is gone (one GEMM per column block, into the
two halves of the columns of the result: `blas.gemm` writes into a block of
columns of a row-major array), the arrays of the ladder are counted, and
`test_amps_peak_within_plan` holds the peak of the allocations of a step
(tracemalloc) against the working set of the plan for a budget that forces
slabs and blocks; it fails for the code before (59.5 MB against 54.2 MB
planned). With the corrected plan benzene/cc-pVTZ peaks at 23.7 GB under
`max_memory` 24000 MB (job 31582707, c436: 196.2 s per iteration on that
node, the direct ladder 26.5 s of it in one pass over the integrals; the
ratio to c282 is 1.06, that of the benzene/cc-pVDZ iterations of the two
jobs 1.12).

### CCSD(T) against CFOUR: where it stands

One job (31554690, c081, 8 cores), CFOUR's CCSD(T) and then zccsdt's on
the same node; zccsdt with its defaults (Kramers pairs, three real GEMMs
where the plan says so) and `max_memory` 24000 MB:

| seconds | H2O/cc-pVTZ | H2O/cc-pVQZ | C6H6/cc-pVDZ |
|---|---|---|---|
| CCSD iteration: CFOUR | 0.33 | 2.76 | 26.89 |
| CCSD iteration: zccsdt | 0.067 (4.9x) | 0.428 (6.4x) | 9.95 (2.7x) |
| (T): CFOUR | 2.1 | 23.9 | 1499 |
| (T): zccsdt | 0.6 (3.5x) | 6.3 (3.8x) | 235 (6.4x) |
| integrals: CFOUR `xx2cvtran` | 0.49 | 3.38 | 11.9 |
| integrals: zccsdt | 0.7 | 5.7 | 12.0 |

- With the later changes (the intermediates completed by half-block
  copies, the barred a of (T) in ranges, batches of 4 GB) the same
  comparison gives 0.055 / 0.368 / 8.36 s per iteration against 0.26 / 2.06
  / 22.98 s (job 31554768, c228: 4.7x, 5.6x, 2.75x), and (T) for H2O in 0.5
  / 5.1 s against 2.0 / 23.6 s (job 31556042, c415: 4x, 4.6x) and in 0.5 /
  4.7 s against 1.7 / 17.3 s (job 31558357, c451: 3.4x, 3.7x). The nodes
  differ more for CFOUR than for zccsdt, so the factors are larger on the
  nodes where CFOUR is slow (c081, c415) than where it is fast (c228, c451).
- Without Kramers pairs (`kramers = False`, or a reference that has none)
  the factors are 3.3x, 3.9x and 1.6x per iteration and 2.3x, 2.4x and 3.2x
  for (T).
- The integral step of zccsdt includes the AO integrals (libcint) and the
  two AO operators of the ladder, 3.3 of the 5.7 s for H2O/cc-pVQZ;
  `xx2cvtran` reads AO integrals made before. The transformation itself
  (ket, bra, sort: 2.5 s for H2O/cc-pVQZ here) is that of the section on the
  integrals, a little faster than CFOUR's (2.6 s). Since these runs the AO
  operators come from an eighth of the integrals (the section above): on
  one node the integral step of H2O/cc-pVQZ and benzene/cc-pVDZ takes 4.6 s
  and 8.6 s where the code of this table took 5.1 s and 10.0 s (job
  31577428).

**A whole calculation, to convergence** (`bench/run_zccsd.py`; job 31572991,
c418, 8 cores, CFOUR first; MKL 2026.1). CFOUR with `CC_CONV=7` (the largest
change of an amplitude below 1e-7); zccsdt until the norm of the change of
[t1, packed t2] is below 1e-7 and the energy changes by less than 1e-9.
Seconds:

| | H2O/cc-pVQZ: CFOUR | zccsdt | C6H6/cc-pVDZ: CFOUR | zccsdt |
|---|---|---|---|---|
| integral transformation | 2.9 | 5.0 | 9.6 | 10.1 |
| CCSD: cycles, seconds | 10, 22.8 | 12, 4.5 | 12, 298.8 | 13, 108.8 |
| (T) | 17.4 | 4.7 | 1141.8 | 186.0 |
| transformation, CCSD and (T) (CFOUR: `xx2cvtran` + `xx2cccsd`) | 45.5 | 14.2 (3.2x) | 1453.8 | 304.9 (4.8x) |
| without Kramers pairs: CCSD, (T), all | | 7.7, 9.0, 21.5 (2.1x) | | 195.0, 374.4, 579.4 (2.5x) |
| peak memory, GB | 3.7 | 7.8 | 13.7 | 16.0 |
| E_corr(CCSD) | -0.3172894490 | -0.3172894448 | -0.8372715768 | -0.8372715247 |
| E(T) | -0.0092679387 | -0.0092679380 | -0.0363228986 | -0.0363228895 |

- The two programs solve the same problem: the SCF energies differ by
  1.5e-8 and 3.5e-8 Eh, E_corr(CCSD) by 4e-9 and 5e-8, E(T) by 7e-10 and
  9e-9. With and without Kramers pairs zccsdt gives the same E_corr and
  E(T) to the ten digits printed, for both molecules.
- zccsdt needs one or two cycles more, to a criterion that is the stricter
  one (the norm bounds the largest change).
- A 24-core job of another session ran on the node from the middle of
  CFOUR's benzene run on; the times per iteration are those of the quiet
  nodes (8.37 s, CFOUR 22.98 s).
- Properties: after the CCSD, the relaxed one-particle density of CCSD(T) -
  (T) with the sources of lambda and its densities, the Lambda equations,
  the orbital gradient, the Z-vector equations - takes 36.3 s and 967.5 s
  where CFOUR's chain takes 225.2 s and 6465.3 s (6.2x, 6.7x; the section
  on the Lambda equations below).

**Memory.** Benzene/cc-pVDZ with Kramers pairs, job 31554690. Everything
is planned from `max_memory`: slabs and blocks of the CCSD step, the DIIS
vectors, the operands and batches of (T).

| `max_memory` | in scratch files | CCSD: plan of the step | s / iteration | peak memory | (T) | peak memory |
|---|---|---|---|---|---|---|
| 24000 MB | nothing | one slab, one block, three real GEMMs | 9.95 | 11.9 GB | 235.0 s (three real GEMMs) | 14.0 GB |
| 8000 MB | ovvv_p; DIIS vectors; the operand A of (T) | slabs of 9, blocks of 14 of 42 occupied, three real GEMMs | 11.76 | 7.4 GB | 230.5 s (ZGEMM, batches of 1.9 GB) | 7.6 GB |
| 4000 MB | <mn\|\|ef>, <mb\|\|je>, ovvv_p; W(mbej); DIIS vectors; both operands of (T) | slabs of 2, blocks of 6, ZGEMM | 13.94 | 3.6 GB | 259.4 s (ZGEMM, batches of 0.45 GB) | 3.8 GB |

With 4000 MB (PySCF's default `max_memory`) the iteration is 1.4 times
slower than with 24000 MB and (T) 1.1 times; both stay well ahead of CFOUR
in 11 GB. (Peak memory is anonymous memory; the mapped scratch arrays are
page cache.)

The size of the batches of (T) (benzene with Kramers pairs, `max_memory`
24000 MB, job 31556395 on c236, a node as fast as c081):

| x of a batch, at most | 0.5 GB | 1 GB | 2 GB | 4 GB | 8 GB |
|---|---|---|---|---|---|
| ZGEMM, seconds | 252.6 | 233.5 | 228.2 | 228.7 | 228.8 |
| three real GEMMs, seconds | | | 234.1 | 223.5 | 223.9 |
| peak memory (GB), ZGEMM / three real GEMMs | 10.9 | 11.4 | 12.5 / 14.0 | 14.6 / 16.0 | 18.4 / 20.2 |

The real parts hold two thirds of the elements in the same bytes and reach
the plateau later. The tables above were measured with 2 GB; the default
(`triples.BATCH_BYTES`) is 4 GB since this scan, i.e. 223.5 s for benzene
where the 2 GB of the comparison with CFOUR gave 234-235 s. For H2O/cc-pVQZ
it makes no difference in time (4.7 s with 4 GB, 4.5 s with 2 GB; job
31558357) and costs 2 GB of memory.

**Without MKL.** The matrix products go through one library: the MKL build
of the kernel library (all numbers above), the build of the same library on
OpenBLAS (0.3.33 with its own threads), or SciPy's BLAS if there is no
kernel library; without `libzc_zccsd.so` numpy does the work of the OpenMP
kernels on one thread. The energies are the same in all four
(`tests/test_zccsd.py` passes in each; here SciPy's BLAS is MKL as well,
and PySCF's own libraries are linked with MKL: a process without any MKL
was not tested). MKL against OpenBLAS, the same code on the same node (job
31564346, c022, 8 cores):

| seconds | H2O/cc-pVQZ: MKL | OpenBLAS | C6H6/cc-pVDZ: MKL | OpenBLAS |
|---|---|---|---|---|
| CCSD iteration | 0.423 | 0.421 | 9.47 | 12.96 |
| of which the two o^3 v^3 products | 0.149 | 0.149 | 5.13 | 7.22 |
| (T) | 5.3 | 5.2 | 227.9 | 277.2 |
| of which GEMMs / kernel | 3.7 / 1.3 | 3.7 / 1.3 | 179.9 / 47.0 | 222.4 / 53.6 |

For H2O the BLAS makes no difference; for benzene OpenBLAS is 1.4 times
slower per iteration (its DGEMM: 1.4 times in the six real products; the
other terms 1.3-1.5 times) and 1.2 times in (T).

The MKL of the environment changed from 2026.0 to 2026.1 at 14:23 on
2026-10-03: the runs of jobs 31554690, 31554768 and 31556042 and the first
four of the batch scan (ZGEMM, 0.5 to 4 GB) used 2026.0, the last four of
the scan (ZGEMM 8 GB, the three real GEMMs), job 31558357 and this one
2026.1. No difference shows where the two can be compared: benzene's (T)
with three real GEMMs and batches of 2 GB took 235.0 s with 2026.0 (c081)
and 234.1 s with 2026.1 (c236), with ZGEMM 228.7 s (4 GB, 2026.0) and
228.8 s (8 GB, 2026.1) in the scan, and the CCSD iterations of the eight
runs of the scan 9.36-9.38 s before and 9.36-9.37 s after.

The change (`libblas=*=*mkl`) also put numpy and SciPy on MKL, so that a
`DirectZCCSD` run now holds no OpenBLAS at all: MKL 2026.1 for numpy,
SciPy, PySCF's libraries and the kernel library, and one OpenMP runtime,
conda's `libomp.so`, for MKL, PySCF and the OpenMP kernels (it serves the
GNU interface as well: `libgomp.so.1` of the environment is a link to it).
One runtime is one pool of threads; `blas.backend()` names the runtimes
that are loaded, and a second one is a warning in the log.

What is not done:

- the integral transformation does not use Kramers pairs (the CCSD step
  does not read the rows of <mn||ef>, <mb||je> and <ma||ef> with a barred
  first index; (T) reads all of <ma||ef>);
- the integral-direct ladder computes all N^4/8 integrals of a pass,
  without screening, and a unit is at least one shell against one shell
  (16 s^2 N^2 bytes for a shell of s functions: a generally contracted
  shell is not split);
- the page faults of the work arrays of a CCSD step (3-6%, above);
- for a few occupied spinors the kernel of (T) is a fifth to a quarter of
  (T): its pieces of the x arrays are short (o(o-1)/2 elements) and come
  from memory one cache miss at a time, and the loops over k < j are a few
  elements long.

### The Lambda equations (`zccsd/lam.py`), their (T) terms, the Z-vector

**CCSD Lambda** (`DirectZCCSD.solve_lambda`). The equations of
`pyscf.cc.gccsd_lambda` for lambda1[i,a] and lambda2 packed over i > j and
a > b, in the layouts of the amplitude step:

- what depends on t alone is made once, by the first pass of the amplitude
  step with the full product <mn||ef> t2: elements of the similarity-
  transformed Hamiltonian - its one-particle blocks, <ij|Hbar|kl> on packed
  pairs, <mb|Hbar|ej> in the ring layout (o^2 v^2 elements, in memory or
  mapped), an intermediate with three occupied indices, the T1 equation;
- no intermediate with three virtual indices is made (PySCF's and the
  earlier socutils code keep one of o v^3 elements). lambda2 with
  <ab|Hbar|ci> in the lambda1 equation is evaluated from its parts: the
  integral with the rows of lambda2; the ring part by a product of the ring
  layouts of lambda2 and t2, in blocks of columns that are contracted with
  the rows of <ma||ef>; the t1 <mb|Hbar|ej> part with the rows of W as they
  are read for the ring term; the ladder part as the AO ladder of lambda2
  contracted with t1;
- per iteration: two o^3 v^3 products (the ring term of the lambda2
  equation and the product above), the AO ladder with the two sets of
  coefficients exchanged, one pass over <ma||ef>, one over W;
- Kramers pairs: the sums over the first index of <ma||ef> run over the
  unbarred m and are completed by their time-reversed image (for the
  packed array in place), of the ring term the rows with unbarred i are
  computed, the ladder is done for one pair of each two partners; three
  real GEMMs per product where the plan of the amplitude step uses them;
- memory: the plan of the amplitude step for the budget less what that
  plan does not know (`lam.reserve_bytes`); with ZGEMM the blocks of the
  product with t2 are two thirds of the plan's, since it holds three arrays
  of a block of columns where the step holds two.

Tests (`tests/test_zccsd_lambda.py`, H2O X2C spinors): one step and the
intermediates against PySCF's on integrals made by brute force from the AO
integrals, for random complex t and lambda (1e-12; observed 1e-15), without
and with Kramers pairs, in blocks, W mapped, three real GEMMs, numpy
kernels; the converged lambda against PySCF's iterations; the one-particle
density of t and lambda against the derivative of the CCSD energy with
respect to a complex Hermitian one-particle perturbation at fixed spinors
(central difference: 1.5e-9); the allocations of a step against
`lam.working_set` (tracemalloc), for budgets from the smallest up.

Against CFOUR (`xx2chbar`, then `xx2clcc`), one job, CFOUR first on the same
node (31600070, c516, 8 cores, `max_memory` 24000 MB and MEMORY_SIZE 24 GB,
MKL 2026.1; zccsdt as of d4520a8; `bench/profile_lambda.py`). CFOUR with
CC_CONV=7, zccsdt until the norm of the change is below 1e-7. Seconds:

| | H2O/cc-pVQZ: CFOUR | zccsdt | C6H6/cc-pVDZ: CFOUR | zccsdt |
|---|---|---|---|---|
| CCSD iteration on this node | 2.75 | 0.44 | 26.92 | 9.46 |
| Hbar / the intermediates | 29.1 | 0.23 | 158.7 | 7.0 |
| Lambda: cycles | 10 | 11 | 12 | 11 |
| Lambda: seconds | 46.9 | 6.5 | 321.9 | 130.6 |
| per cycle | 4.7 | 0.590 | 26.8 | 11.88 |
| Hbar and Lambda | 76.0 | 6.7 (11x) | 480.5 | 137.5 (3.5x) |
| peak memory, GB (zccsdt: of the whole run, integrals and CCSD included) | 2.8 | 5.3 | 12.6 | 14.1 |

The time of `xx2clcc` per cycle is its wall time over its cycles. c516 is
one of the nodes on which CFOUR is slow (26.9 s per CCSD iteration for
benzene; 23.0 s on the fast ones), so the factors would be smaller there, as
for the amplitude equations.

Where an iteration of zccsdt goes (the same job, seconds):

| | H2O/cc-pVQZ | C6H6/cc-pVDZ |
|---|---|---|
| the two o^3 v^3 products | 0.163 | 5.05 |
| the terms on the rows of <ma\|\|ef> (GEMM with lambda2 t1; rows; contraction with the product) | 0.187 (0.026; 0.042; 0.119) | 1.71 (0.85; 0.30; 0.57) |
| ring layouts, packing | 0.025 | 0.91 |
| AO ladder | 0.088 | 1.06 |
| pair-pair products, mba, lambda2 t1 | 0.034 | 1.34 |
| virtual and occupied one-particle terms | 0.045 | 1.11 |
| lambda1 terms, DIIS | 0.031 | 0.56 |
| iteration (with what is under no timer) | 0.590 | 11.88 |
| the CCSD iteration | 0.44 | 9.46 |

An iteration is 1.26-1.34 times a CCSD iteration. For benzene the two
products take what the two of the amplitude step take (4.7-5.8 s in the
CCSD iterations of the other jobs; not timed in this one), and the rest is
more than there. Known reserves, not measured: the two
one-particle terms on the occupied side are dense GEMMs on the pairs
(about 0.5 s for benzene); the ring layout of lambda2 and the real parts of
W are made twice and every iteration (0.9 s); t2, lambda2 and <ij||ab> are
unpacked in three passes (about 0.6 s); for H2O the contraction of the
product with the unpacked rows of <ma||ef> is a fifth of the iteration.
Since this job the two one-particle terms on the occupied side are part of
the pair products (6c8a393; on the shared node 12.7 s where 13.0 s were).

**Under memory limits and for benzene/cc-pVTZ** (job 31600261, c516, zccsdt
as of ac46d03; the cc-pVTZ run with three cycles of CCSD and of Lambda, on
amplitudes that are not converged, for the time and the memory of an
iteration):

| | C6H6/cc-pVDZ, 24000 MB | C6H6/cc-pVDZ, 8000 MB | C6H6/cc-pVTZ, 24000 MB |
|---|---|---|---|
| plan of the Lambda iterations | one block, three real GEMMs, W in memory | blocks of 8 and 14 of 42 occupied, ZGEMM, W and DIIS in scratch files | blocks of 2 and 10, ZGEMM, W and DIIS in scratch files, direct ladder |
| CCSD iteration, s | 9.46 | 11.21 | 194.3 |
| Lambda intermediates, s | 7.0 | 8.1 | 114.1 |
| Lambda iteration, s | 11.88 | 13.40 | 217.8 |
| of which the two products / the ladder | 5.05 / 1.06 | 6.91 / 1.07 | 121.8 / 32.4 |
| Lambda cycles | 11 | 11 | (3) |
| peak anonymous memory | 14.1 GB | 7.94 GB | 26.89 GB |

With 8000 MB lambda converges in the same 11 iterations to the same
amplitudes. The job showed two faults, both in the drivers and both fixed
in 2be7e95:

- the 26.89 GB of benzene/cc-pVTZ are the CCSD iterations, not Lambda:
  `amps.kernel` called its callback with `locals()`, and the dictionary of
  the frame kept the old and the new packed amplitudes alive through the
  next step - two packed arrays (2 x 1.62 GB) outside the plan, whenever a
  callback is set (the benchmark scripts set one to time the cycles; the
  23.7 GB of job 31582707 were measured without one). The callback gets a
  dictionary of its own now; `test_callback_keeps_no_amplitudes`;
- with `max_memory` 4000 MB the Lambda iterations of benzene/cc-pVDZ could
  not be planned (MemoryError: 1.14 GB left of the 1.77 GB of the smallest
  plan): the unpacked t2 of the CCSD object, four times the packed array,
  was in memory while lambda was solved for. `lam.kernel` now packs it,
  releases it and unpacks it at the end. The same array took 6.5 GB from
  the plan of the cc-pVTZ run (11.9 GB for the blocks instead of about 18).

The runs after the two fixes (job 31600838, c588, zccsdt as of 2be7e95):

| | C6H6/cc-pVDZ, 4000 MB | C6H6/cc-pVTZ, 24000 MB (three cycles each) |
|---|---|---|
| plan of the Lambda iterations | blocks of 1 and 4 of 42 occupied, ZGEMM, W and DIIS in scratch files, direct ladder (the operators kept in memory gave way) | blocks of 1 and 10, ZGEMM, W in memory, DIIS in a scratch file, direct ladder |
| CCSD iteration, s | 12.66 | 194.1 |
| Lambda intermediates, s | 10.8 | 119.6 |
| Lambda iteration, s | 18.59 | 233.6 |
| Lambda cycles | 11 | (3) |
| peak anonymous memory: integrals and CCSD / Lambda | 3.40 / 2.88 GB | 23.72 / 18.56 GB |

Both runs stay within `max_memory`, the CCSD of benzene/cc-pVTZ with a
callback set as well (23.72 GB where 26.89 GB were). With 4000 MB lambda
converges in the same 11 iterations to the same amplitudes as with 8000 and
24000 MB.

The Lambda iteration of benzene/cc-pVTZ took longer than in the first job
(233.6 s; 217.8 s) although its plan had 17.0 GB for the blocks instead of
11.9 GB: the plan now keeps W in memory and has slabs of one occupied index,
and the product L T took its rows of lambda2 in blocks of the size of its
columns, so that it was made of 882 products of one occupied index against
one (73.0 s and 10.8 s of ring layouts where 63.1 s and 5.8 s were). Since
2555685 the rows of lambda2 are in blocks of their own: for benzene/cc-pVDZ
with 4000 MB that product takes 4.6 s in 189 products where it took 5.5 s
in 882 (job 31601345, c237, a node with the same CCSD iteration, 12.67 s).
The iteration as a whole took 20.4 s in that job, more than the 18.6 s
before, because the same code had a larger reserve for an array of the
intermediates and so smaller blocks (3 instead of 4 occupied indices, 78
instead of 136 pairs of the ladder); ae5a717 makes the intermediates without
that array and the reserve is what it was.

The code as it is (ae5a717; job 31601552, c025, four cycles each): for
benzene/cc-pVDZ an iteration of Lambda takes 11.16 s next to a CCSD
iteration of 9.44 s with 24000 MB (1.18 times; the two products 5.05 s, the
pair and one-particle parts 1.73 s instead of 2.45 s), and 18.61 s next to
12.64 s with 4000 MB (the product L T 4.60 s in 126 products, the direct
ladder 3.04 s in blocks of 109 pairs; peak 2.86 GB). For benzene/cc-pVTZ
the iteration has not been timed again.

The memory that is set aside for the arrays of Lambda (`lam.reserve_bytes`)
counted them on top of the plan of a step, which already holds the
intermediates of a step that are not there during a Lambda iteration
(`amps.step_arrays`): only what the arrays of Lambda have more is set aside
now (benzene: 139 MB instead of 501 MB; the tracemalloc test holds with it,
by 13 MB and more for H2O/cc-pVTZ). With 4000 MB the blocks are larger for
it: 13.24 s per iteration next to a CCSD iteration of 11.71 s, with a peak
of 3.04 GB; with 24000 MB 9.64 s next to 8.39 s (job 31606686, c398, four
cycles each).

**The (T) terms** (`zccsd/lam_t.py`, `zccsd/triples.py`,
`solve_lambda(with_t=True)`). The earlier code of the user is on the branch
`claude/ccsd-lambda-socutils` of socutils (8965b9f; `cc/lambda_zccsd.py`,
`cc/lambda_t_zccsd.py`, `cc/rdm_zccsd.py`): PySCF's equations on full
arrays with the pp ladder through the AO ladder, and for (T) PySCF's source
terms with the bare integrals conjugated, which complex spinors need. The
(T) source terms are copied from there, with the (T) terms of the
one-particle density of `pyscf.cc.gccsd_t_rdm`, as a reference on full
o^3 v^3 arrays (`lam_t.sources`, `lam_t.gamma1`). Test: E(T) of these
triples against `triples.kernel` (1e-10), and the density with lambda and
the (T) terms against the central difference of E(CCSD) + E(T) for a
perturbation of the Fock matrix in its occupied-virtual blocks and on its
diagonal (1e-8, of a (T) part of 4e-5 to 1e-3), without and with Kramers
pairs.

For molecules of any size the terms are evaluated in the batches of the (T)
energy (`triples.kernel(terms=True)`, which `solve_lambda(with_t=True)` and
`make_rdm1(with_t=True)` call). Every term is a sum over the virtual triples
a > b > c, each in three roles (one of its virtual indices, the other two
as a pair). The two that cost o^3 v^4 / 4 and o^4 v^3 / 4 - the triples
with <ke||pq> and with <mn||js>, as much as the intermediates x of the
energy took - have the shape of the products that made x, with result and
operand exchanged: the kernel writes M = conj(2w + v) / D of a triple, with
the sign of the role, over its three x (each belongs to that triple alone),
and for every product x = A B^T of a batch two products follow, A[:, :v]^T M
and M B[:, v:] (GEMM; three real ones with `three_m`), which are
antisymmetrized into the terms of lambda2 kept transposed, [(a>b), (i>j)],
so that rows are written as a whole. The terms on <bc||jk>, on t2 and on
the Fock matrix and the diagonals of the density are sums in the kernel.
With Kramers pairs the triples that are done (unbarred b) give half of every
sum and the time-reversed image the other half. Memory: the terms of lambda2
take the place of the third packed T2 of the plan, the products leave GEMM
in an array of at most 64 MB. Tests: against the assembly from full arrays
for one batch, ranges of a and the smallest batches, ZGEMM and three real
GEMMs, C and numpy kernels, with and without f_ov and Kramers pairs (1e-15
without, 1e-12 with them: the symmetry of the integrals); the finite
difference above through `solve_lambda(with_t=True)`; what the kernel
allocates against `max_memory` from the least to all in memory.

One job (31602768, c280, 8 cores, MKL; `bench/profile_t_lambda.py`: the (T)
energy, then the energy with the terms, on the amplitudes of the same
CCSD), seconds:

| | (T) energy | with the terms | ratio | peak memory |
|---|---|---|---|---|
| H2O/cc-pVQZ (10 + 220 spinors), three real GEMMs | 5.1 | 13.8 | 2.7 | 7.0 GB |
| C6H6/cc-pVDZ (42 + 186), three real GEMMs | 224.4 | 573.8 | 2.56 | 14.9 GB |
| C6H6/cc-pVDZ, ZGEMM | 228.1 | 560.4 | 2.46 | 13.4 GB |
| C6H6/cc-pVDZ, `max_memory` 4000 MB (A in a scratch file, ZGEMM, 765 batches) | 232.6 | 579.6 | 2.49 | 3.86 GB |

Where the 574 s of benzene go: the products of the energy 177 s as before,
the two products of the terms 251 s, the kernel 138 s (47 s for the energy
alone), their antisymmetrization 7 s. The terms cost 1.5 times the energy,
not the 1.0 of their multiply-adds:

- the two products of the terms have the multiply-adds of the product that
  made x, but the large array is now an operand, which GEMM packs: at the
  shapes of a benzene batch ZGEMM does 573 GFLOPS for A B^T and 277 and 282
  for the two others (DGEMM: 488, 371 and 125; one run on the tunnel node,
  8 threads, not a benchmark). Three real GEMMs gain nothing here (251 s
  against 247 s). One pass over M in blocks that stay in the cache, feeding
  both products, is not written;
- the kernel writes M three times (nine real arrays with three real GEMMs)
  where the energy only reads: its time follows the memory traffic. The
  role of the first index could be taken from the array of the second
  (products for one b at a time), which would save one of the three.

CFOUR's (T) terms of Lambda have not been timed.

**Z-vector** (`zccsd/zvector.py`, `zccsd/orbgrad.py`,
`make_rdm1(relaxed=True)`). For C -> C exp(-kappa) the energy with its
Lagrangian changes by sum kappa_xy G_yx; R(z) = -G_vo with the orbital
Hessian of the SCF (orbital energy differences and the Fock matrix of the
density change; Jacobi steps, DIIS); the relaxed density is the density with
z in its virtual-occupied block.

*The orbital gradient.* With densities that are not Hermitized,
G[a,i] = (Gnh[a,i] - conj(Gnh[i,a])) / 2, and Gnh[i,a], the derivative along
the excitation a+_a a_i, is minus the lambda1 equation: zero for converged
lambda (H2O/6-31G: 2e-8 with lambda converged to 1e-10). What is left,
Gnh[a,i], is the Lagrangian L = E + l1 R1 + l2 R2 / 4 with the integrals
one-index transformed by a+_i a_a: a sum over the blocks of integrals of the
transformed block with its density, the coefficient of that block in L.
`zvector.gradient_terms` evaluates it from full arrays, block by block (each
block against L itself on the transformed integrals: 1e-17; the sum against
the gradient from PySCF's Hermitized two-particle density: 1e-8).
`zccsd/orbgrad.py` evaluates the same terms in the layouts of the amplitude
step:

- m = L T, the product of the ring layouts of lambda2 and t2 (o^3 v^3), is
  kept like W(mbej) of a step (o^2 v^2 elements, in memory or mapped) and
  read by rows; the density of <mn||ef> is packed like t2, with the second
  o^3 v^3 product, m^T (t2 - 2 t1 t1), antisymmetrized block by block; the
  rest are arrays with three occupied indices and three passes over <ma||ef>;
- only the density of <ia||bc> meets <ab||cd>. Its lambda1 tau and
  (lambda2 tau) t1 parts take the pp ladder of tau (the AO ladder of the
  amplitude step, once), its (lambda2 t2) t1 part is a Fock matrix of a
  v x v density, and its m t1 part - the virtual pair of the contraction
  split between m and t1 - is contracted with the integrals <j~ e||ab> of
  the spinors j~ = sum_f t1[j,f] f in the place of an occupied one, which
  the integral transformation delivers slab by slab (`ao2mo.slabs` with j~
  as the first index: nothing with four virtual indices and nothing of
  o v^3 elements is stored);
- with Kramers pairs the two products are done for the rows with an unbarred
  occupied index, and the passes over <ma||ef> and <j~ e||ab> for an unbarred
  first index;
- memory: the plan of the amplitude step for what `max_memory` leaves next
  to the packed t2 and lambda2, with the arrays of the gradient in the place
  of those of a step that are not made here; a slab of the <j~ e||ab>
  (16 v^3 bytes) has to fit, m goes to a scratch file if that makes room.

*Frozen occupied spinors.* A frozen spinor is an occupied one whose
amplitudes are zero: with t and lambda padded, the Lagrangian of the
frozen-core CCSD is that of the CCSD of all spinors, and
`zvector.gradient_frozen` takes the gradient from the code for all spinors -
Gnh[a,x] from `orbgrad` for all occupied x; the lambda1 residual of the
frozen spinors for the excitation half, which no amplitude makes vanish; and
the rotations between frozen and active occupied spinors from the residuals,
with one frozen index, of one step of the amplitude equations and one of the
Lambda equations (a rotation among occupied spinors is a change of the
amplitudes). The integrals are made once more with all occupied spinors.
The rotations between frozen and active occupied spinors themselves follow
from the Fock matrix staying diagonal (y_ic = G_ic / (e_c - e_i) in the
density, its Fock matrix on the right-hand side of the equations).

*Fock matrices.* The two of the gradient and those of the orbital Hessian
are J - K of the bare Coulomb operator on the scalar AO integrals
(`orbgrad.Coulomb`: PySCF's GHF contraction of the spin blocks), with the
integrals kept in memory if they take no more than a quarter of what
`max_memory` leaves; `zvector.scf_veff` checks once that this reproduces the
Fock operator of the SCF object. The spinor integrals of PySCF's 2c SCF took
1.5 s per build for benzene/cc-pVDZ, these 0.17 s.

Tests (H2O/6-31G unless said otherwise, without and with Kramers pairs): the
energy from the densities (1e-9); G anti-Hermitian, without
occupied-occupied and virtual-virtual blocks (1e-11); every block of
`orbgrad` against `gradient_terms` (1e-11); the relaxed dipole against the
central difference of E(SCF) + E_corr with the SCF repeated in the field
(3e-8, where the unrelaxed density is off by 2e-3), all electrons and with
two frozen spinors; the three blocks of `gradient_frozen` against the
gradient from the full densities (1e-8: the convergence of lambda, 5e-10,
4e-14); for H2O/cc-pVTZ the same gradient under ten memory plans (m mapped
or in memory, one to ten blocks), its allocations within the budget of the
plan (tracemalloc), and the Kramers reductions against the plain sums
(1e-10).

Time, against CFOUR's `xx2cccdenocbp` (densities) and `xx2czvec`: one job
(31605159, c079, 8 cores), CFOUR's chain `xx2cccsd; xx2chbar; xx2clcc;
xx2cccdenocbp; xx2czvec; xsox2cprop` first, then zccsdt
(`bench/profile_zvector.py`); all electrons, CCSD and Lambda converged in
both; seconds:

| | H2O/cc-pVQZ (10 + 220) | C6H6/cc-pVDZ (42 + 186) |
|---|---|---|
| CFOUR `xx2cccdenocbp` | 34.7 | 159.1 |
| CFOUR `xx2czvec` | 4.1 | 16.9 |
| CFOUR, both | 38.8 | 176.0 |
| zccsdt: orbital gradient | 2.3 | 20.0 |
| zccsdt: the Fock operator made and checked | 2.0 | 2.0 |
| zccsdt: Z-vector equations (9 and 8 Fock builds) | 1.6 | 1.4 |
| zccsdt, the three | 5.9 (6.6x) | 23.4 (7.5x) |
| for scale: a CCSD iteration of zccsdt | 0.37 | 8.42 |

(CFOUR's two programs took 36.2 s and 183.6 s in a second job on the same
node, 31604805. The relaxed density as `make_rdm1(relaxed=True)` makes it -
the one-particle density of t and lambda and the three parts above - took
5.8 s and 23.8 s in a later job, 31606686 on c398, where the three parts
took the same as here.) Where the 20.0 s of
benzene go: the two o^3 v^3 products 7.8 s, the small terms 3.6 s, the
integrals <j~ e||ab> and their contraction 2.5 s, the passes over <ma||ef>
2.5 s, the products on the pairs 1.7 s, the ladder 1.2 s, the Fock matrices
0.8 s. The gradient is 2.4 CCSD iterations for benzene and 6 for H2O, where
the integrals <j~ e||ab> are 40% of it. With `max_memory` 4000 MB (m and
the integrals in scratch files) the gradient of benzene takes 22.4 s with a
peak of 2.86 GB, and the relaxed density as a whole 38.8 s with a peak of
3.34 GB: the scalar AO integrals are then not kept, and the eight Fock
builds of the equations take 12.6 s, AO direct. Before the Kramers
reductions and with the Fock builds of the SCF object (job 31603549) the
gradient took 31.3 s and the equations 12.3 s.

With 12 frozen spinors (benzene/cc-pVDZ, 30 active occupied spinors, a CCSD
iteration of 3.43 s; job 31606686) the relaxed density takes 85.1 s with a
peak of 15.5 GB: the integrals with all 42 occupied spinors, one step of
the amplitude equations, the intermediates and one step of the Lambda
equations, the gradient itself (31 s, without Kramers pairs: the frozen
spinors are not paired), the Z-vector equations. That is 25 iterations of
the frozen-core CCSD, and 3.6 times what the relaxed density of the same
molecule takes without frozen spinors. With `max_memory` 4000 MB it takes
142.8 s with a peak of 3.70 GB: the three phases are planned one by one for
all occupied spinors, the unpacked amplitudes are released, and the
integrals of the active space are given up when a phase does not fit next
to them (they were here: 1.2 GB; `mycc.ao2mo()` makes them again). What it
needs at least is what a step of the CCSD of all spinors needs.

Values, against CFOUR (`xx2cccdenocbp`, `xx2czvec`, `xsox2cprop` after the
Lambda equations; job 31600371, H2O/cc-pVDZ, all electrons, CC_CONV=9;
`bench/check_dipole.py`, then with the gradient from the full two-particle
density), the z component of the dipole moment in atomic units:

| | CFOUR | zccsdt | difference |
|---|---|---|---|
| E(SCF) | -76.075431217 | -76.075431227 | 1.0e-8 |
| E_corr(CCSD) | -0.213384851 | -0.213384837 | 1.4e-8 |
| SCF dipole | 0.808722476 | 0.808722401 | 7.5e-8 |
| CCSD dipole, relaxed | 0.766033160 | 0.766033118 | 4.2e-8 |
| its correlation part | -0.042689316 | -0.042689283 | 3.3e-8 |

The numbers of zccsdt are for an SCF that is converged in its orbital
gradient (`conv_tol_grad = 1e-9`). PySCF's default for it is sqrt(conv_tol):
with `conv_tol = 1e-12` the Fock matrix kept occupied-virtual elements of
3e-8, the SCF dipole came out as 0.808722456 and the relaxed CCSD dipole as
0.766033146 - 5e-8 and 3e-8 off, and by accident closer to CFOUR's. A
relaxed property has what is left of the orbital gradient in first order;
`zvector.kernel` warns above 1e-7. CFOUR's numbers do not move when its
thresholds are tightened (SCF_CONV 13, CC_CONV 10, LINEQ_CONV 10 for the
Z-vector equations: 0.808722476 again): the 7.5e-8 between the SCF dipoles
is the difference of the two Hamiltonians, which E(SCF) shows as 1e-8 (the
speed of light is the same, 137.0359991; with X2CKEY=SOATMX CFOUR's
`xsox2catm` appears to make the X matrix atom by atom, which was not
followed up; PySCF's own atomic approximation, `approx='ATOM1E'`, is 6e-7
away in E(SCF)). The dipole operator has the
picture-change transformation of the X2C Hamiltonian
(`rdm.dipole_integrals`, PySCF's `with_x2c.picture_change`); with the bare
operator of the 2c AO basis, which the socutils code used, both dipoles of
zccsdt are 1.5e-6 lower (0.808720863 and 0.766031612). CFOUR also prints
an "unrelaxed" coupled-cluster dipole (0.784942); the density of t and
lambda at fixed spinors gives 0.764129 here, and that same density without
its occupied-virtual blocks 0.7964: CFOUR's number is a third quantity,
which was not identified.

**Orbital relaxation of CCSD(T)** (`make_rdm1` / `dip_moment` with
`with_t=True, relaxed=True`; `zccsd/orbgrad_t.py`, `zvector.kernel`). With
E(T) in the energy three things change in the orbital gradient
(`zvector.gradient_t` and `gradient_nh` have them from full arrays):

- lambda is that of CCSD(T), and `orbgrad` gives the de-excitation half of
  the gradient of the CCSD Lagrangian with it, as before;
- the derivative of the Lagrangian with respect to t no longer vanishes - it
  is minus the sources of (T) -, so the excitation half is the source of
  lambda1, and the rotations among occupied and among virtual spinors, which
  are changes of t, give contractions of the sources with t1 and t2;
- E(T) itself depends on the spinors through <bc||ei>, <ma||jk>, <bc||jk>,
  f[a,i] and the orbital energies. The coefficients of these in the
  first-order change of Re E(T) are the densities of (T)
  (`lam_t.densities`), and a rotation takes a bra index of an integral to
  the rotated spinor and a ket index to minus it: every density is
  contracted with integrals in all four blocks.

The occupied-occupied and virtual-virtual blocks of the gradient do not
vanish, because E(T) has the orbital energies in its denominators. As for
frozen spinors, these rotations follow from the Fock matrix staying
diagonal: y_pq = G_pq / (e_q - e_p) goes into the density, its Fock matrix
to the right-hand side of the Z-vector equations. G_pq is (e_q - e_p) times
the off-diagonal one-particle density of (T) - the derivative of E(T) with
respect to an off-diagonal Fock matrix element, a product of the triples -
exactly, for any amplitudes: 1e-15 for all pairs of H2O/6-31G, provided the
Fock matrix in the gradient is the one E(T) has (the orbital energies on
its diagonal; with what the SCF left off the diagonal, 2e-8, the two differ
by 1e-12, which divided by the splitting of two levels that only spin-orbit
coupling separates would be an error of its own). Between degenerate
spinors G vanishes (2e-19 between Kramers partners) and there is no
rotation.

The densities come from the pass over the triples that makes the sources of
lambda (`triples.kernel` with `terms='gradient'`): the two products of the
triples of a batch with the operands transposed run over all of A and of B,
not only over their integrals - the columns of M B that belong to the t2 in
B are the density of <bc||ek>, the rows of A^T M that belong to the t2 in A
minus that of <ma||jk> - and the kernel adds up the density of <bc||jk>
like the f term of lambda2, with t1 in the place of f. Both passes over a
batch then have the arithmetic of the forward product (v + o multiply-adds
per element): the products of the terms cost twice those of the energy. The
density of <bc||ek> (8 o v^3 bytes, as much as the stored <ke||bc>) is added
up in the layout of the batches, in memory after the operands if it fits,
and sorted into the layout of the integrals at the end.

`orbgrad_t` contracts the densities with the integrals that the amplitude
step keeps: products on the pairs, one pass over <ke||bc> and the density
of <bc||ek> by k (rows unpacked a few at a time), one by e. One contraction
needs <ab||cd>: the density of <bc||ek> with <bc||ea>. The three virtual
indices of the density of one k are transformed to the AO basis (the first
on the packed pairs: 2 N v^3 and 2 N^2 v^2 multiply-adds) and all k are
contracted with the AO integrals in one pass over them, computed for ranges
of shells and not kept (2 o N^4). A first version made the integrals
(ea|..) with the virtual e as the first index of an integral transformation:
v N^4, 18.5 s for H2O/cc-pVQZ in a rough run on the shared node, against
1.3 s. With Kramers pairs the sums over k and over e run over the unbarred
half and are completed by the time-reversed image.

Frozen occupied spinors: E(T) is that of the active spinors and cannot be
had from the code for all spinors with zero amplitudes; `orbgrad_t` runs on
the integrals of all occupied spinors with the densities and the sources of
the active ones, zero where an index is frozen, inside `gradient_frozen`.

Checks (`tests/test_zccsd_zvector.py`, `tests/test_zccsd_lambda.py`;
H2O/6-31G with and without Kramers pairs unless noted): the densities
against central differences of Re E(T) in random directions of each block
of integrals (1e-11); those of `triples.kernel` against `lam_t.densities`
in all variants of the batches - one batch, ranges of a, one virtual per
batch, ZGEMM and three real GEMMs, the C and the numpy kernels, f_ov,
Kramers pairs (1e-10); every origin of `orbgrad_t` (the sources, the three
densities, the Fock matrix) in all four blocks against the full arrays
(1e-11), also with two frozen spinors; the occupied-occupied and
virtual-virtual blocks against (e_q - e_p) times the products of the triples
(1e-15); the relaxed density against the central difference of E(SCF) +
E(CCSD) + E(T) with the SCF repeated in the field (5e-8, where the density
at fixed spinors is off by 6e-4), from the full arrays and from
`zvector.kernel`, with and without two frozen spinors; allocations within
`max_memory` for the pass over the triples (C2H4/6-31G, from the least to
everything in memory, the density of <bc||ek> in memory and mapped) and for
`orbgrad_t` (H2O/cc-pVTZ, one occupied spinor and one shell of AO integrals
at a time to all at once), MemoryError below the least.

Values against CFOUR, whose chain for CCSD(T) properties is `xx2cccsd` with
CALC=CCSD(T) and PROPS=FIRST_ORDER (its (T) step then makes the sources of
lambda and all densities of (T): `t3ptgradoc`), `xx2chbar`, `xx2clcc`,
`xx2cccdenocbp`, `xx2czvec`, `xsox2cprop`; all electrons, both programs with
tight thresholds (CFOUR: SCF_CONV 12, CC_CONV 10, LINEQ_CONV 10; zccsdt:
the SCF to 5e-10 in its orbital gradient, the CCSD energy to 1e-11, lambda
and t to 1e-9, the Z-vector equations to 1e-10; `bench/profile_t_relaxed.py`),
dipole moments (z, a.u.) with the picture-change transformed operator:

| | H2O/cc-pVDZ: CFOUR | zccsdt | H2O/cc-pVQZ: CFOUR | zccsdt |
|---|---|---|---|---|
| E(SCF) | -76.075431217 | -76.075431227 | -76.113956790 | -76.113956805 |
| E_corr(CCSD) | -0.213384851 | -0.213384837 | -0.317289455 | -0.317289444 |
| E(T) | -0.003063602 | -0.003063601 | -0.009267940 | -0.009267938 |
| SCF dipole | 0.808722476 | 0.808722401 | 0.789013661 | 0.789013598 |
| CCSD(T) dipole, relaxed | 0.760590014 | 0.760589981 | 0.744218715 | 0.744218681 |
| its correlation part | -0.048132462 | -0.048132420 | -0.044794945 | -0.044794917 |
| (T) alone (H2O/cc-pVDZ: against the relaxed CCSD dipole) | -0.005443146 | -0.005443137 | | |

The relaxed CCSD(T) dipoles agree to 3.3e-8 and 3.4e-8, their correlation
parts to 4e-8 and 3e-8, the (T) part to 1e-8: as the SCF dipoles do (7.5e-8
and 6.3e-8), i.e. within the difference of the two Hamiltonians.
For H2O/cc-pVDZ the finite difference of zccsdt's own energies, the SCF
repeated in the field, gives 0.760589980. Before the SCF was converged in
its orbital gradient (3e-8 left) the analytic value was 0.760590031 (see
above). The H2O/cc-pVQZ numbers were the same on two nodes (jobs 31611246
and 31609182); with CFOUR's default thresholds for the Lambda and Z-vector
equations and CC_CONV 7 its dipole is 0.744218622, with zccsdt's defaults
(1e-9 in the CCSD energy) 0.744218678.

Time, in one job (31609182, c435, 8 cores; CFOUR first: once with
CALC=CCSD(T) alone for the time of its (T) energy, then the chain above;
then zccsdt, `bench/profile_t_lambda.py` and `bench/profile_t_relaxed.py`,
code of commit 2f9a8b1 - synchronized into the snapshot of the job while
CFOUR ran); all electrons, CC_CONV 7 in CFOUR and 1e-9 in the CCSD energy in
zccsdt; seconds:

| | H2O/cc-pVQZ (10 + 220): CFOUR | zccsdt | C6H6/cc-pVDZ (42 + 186): CFOUR | zccsdt |
|---|---|---|---|---|
| CCSD (cycles) | 30.2 (10) | 5.2 (12) | 348.8 (12) | 122.4 (13) |
| (T) energy alone | 23.7 | 5.1 | 1493.1 | 223.2 |
| (T) with the sources of lambda | | 13.9 | | 572.6 |
| (T) with the sources and the densities | 97.5 | 19.8 | 5745.2 | 777.1 |
| Hbar and the Lambda equations (cycles) | 28.9 + 46.7 (10) | 7.1 (11) | 158.3 + 322.5 (12) | 154.1 (13) |
| densities, orbital gradient, Z-vector equations | 46.5 + 5.6 | 9.5 | 216.9 + 22.4 | 36.3 |
| all after the CCSD | 225.2 | 36.3 (6.2x) | 6465.3 | 967.5 (6.7x) |
| of zccsdt's last row: `orbgrad`, `orbgrad_t`, Fock operator, Z-vector | | 2.7, 2.6, 2.1, 2.0 | | 23.4, 8.6, 2.0, 1.5 |

By phase CFOUR takes 4.9 and 7.4 times as long for (T) with the terms of
the gradient, 10.6 and 3.1 times for Hbar and the Lambda equations, 5.5 and
6.6 times for what follows. Its (T) with these terms takes 4.1 and 3.8
times its (T) energy (it loops over the occupied triples, with three sets
of GEMMs for each); zccsdt's 3.9 and 3.5 times: the products of the terms
are twice those of the energy by count (benzene: 408 s against 176 s), the
kernel writes the triples back and adds up the other sums (147 s against
47 s), and the density of <bc||ek> is added up from the products (44 s with
the antisymmetrization of the sources). Where the 36.3 s of benzene after
the Lambda equations go: the Lagrangian's own part 23.4 s (`orbgrad`), the
(T) terms 8.6 s - the passes over <ke||bc> and the density of <bc||ek>
3.5 s, the contraction with <bc||ea> 3.0 s (2.1 s for the densities to the
AO basis, 0.9 s for the AO integrals), the products on the pairs 1.2 s -,
the Fock operator of the orbital Hessian made and checked 2.0 s, the
Z-vector equations 1.5 s.

The rotations among occupied and among virtual spinors of benzene: the
largest elements of y are 2.0e-4 and 3.3e-4 (z: 4.5e-4), between levels 0.33
and 0.69 Hartree apart. 41 occupied and 189 virtual pairs are closer than
1e-4 - 21 and 93 of them the Kramers partners, which are not rotated, the
others levels that spin-orbit coupling alone separates, down to 4e-8 and
1.2e-6 Hartree - and y between those is 2e-9 and 1e-7 at most: the gradient
is the multiple of the energy difference that it should be, and the division
makes nothing up.

With `max_memory` 4000 MB (benzene, job 31611246 on c399, where the run
with 24000 MB took the same 967.7 s after the CCSD as on c435): the pass
over the triples with the sources and the densities takes 840.3 s with A
and the density of <bc||ek> in scratch files (242.9 s for the energy alone),
the Lambda equations 246.1 s, and the peak of the anonymous memory of the
two is 3.79 GB; the relaxed density then takes 52.6 s with a peak of
3.35 GB, 1139 s in all. The (T) terms of the gradient are made first and
their arrays released - the Lagrangian's own part takes the memory of an
amplitude step -, and the two small densities wait in the scratch file of
the large one while the Lambda equations are solved, whose plan was at its
floor here (1.96 GB of 1.96 GB).

## Correctness references

| Reference | Agreement | Test |
|---|---|---|
| PySCF GCCSD (real and complex-rotated GHF) | 1e-10 | `tests/test_ccsd.py` |
| FCI, 3 electrons (real UHF; X2C spinor) | 1e-9 | `tests/test_ccsdt.py` |
| MRCC and CFOUR ncc/ecc/vcc CCSD and CCSDT, 5 small molecules | 1e-8 (observed about 5e-11) | `tests/test_references.py` |
| Determinant-space brute force, T3 residual element-wise | 1e-12 | `tests/test_ccsdt.py` |
| Invariance under complex occ-occ / vir-vir rotations (RHF -> GHF -> U) | 1e-10 | `tests/test_ccsdt.py` |
