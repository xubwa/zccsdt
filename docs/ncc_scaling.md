# Thread scaling: ncc and zccsdt on one node

How the CCSDT iteration and the (Q) correction of CFOUR's `ncc` and of
zccsdt's closed-shell program (`RCCSDT`) speed up with threads, on the same
node and the same molecules.  Everything in the tables is measured; what was
not measured is listed at the end.  Scripts: `bench/ncc_scaling/`
(`slurm_scaling.sh`, the lists `o3_tz.runs` and `c6h6_dz.runs` written by
`mkruns.py`, `summary.py`); the full tables of every run, with CPU time and
memory, are in `bench/ncc_scaling/results/`.

## Set-up

| | O3 / cc-pVTZ | benzene / cc-pVDZ |
|---|---|---|
| geometry (`bench/ncc_scaling/geom/*.bohr`, the same file for both programs) | r = 1.272 A, angle 116.8 degrees (the O3 of the Matthews and Stanton cases) | r(CC) = 1.397 A, r(CH) = 1.084 A, D6h (`bench/profile_sa.py`) |
| basis functions, electrons correlated | 90, all 24 | 114, 30 (6 core orbitals frozen) |
| occupied, virtual orbitals | 12, 78 | 15, 93 |
| point group used by `ncc` with `SYMMETRY=ON` | C2v | D2h |
| iterations timed (mean of) | 2 - 5 of 5 | 2 - 3 of 3 |
| job, node | 31782852, c304 | 31782853, c255 |
| wall time of the job | 10 h 30 min | 9 h 45 min |

- Node: 2 x 24 cores Xeon Gold 6248R, 188 GB, the whole node per job
  (`-N 1 -n 1 -c 48 --mem=180G`), nothing else on it.  The tree ran from a
  copy on the local disk (`/tmp`), scratch of both programs on `/tmp`.
- Binding: n threads on cores 0 .. n-1.  Up to 24 threads
  `numactl --physcpubind=0-(n-1) --membind=0` (one socket and its memory);
  47 and 48 threads on both sockets, memory where it is first touched (and
  one run each with `--interleave=all`).  The converged benzene run of zccsdt
  (24 threads, 81 GB resident) had `--preferred=0` instead of `--membind=0`.
- `OMP_NUM_THREADS` = `MKL_NUM_THREADS` = n.  `OMP_WAIT_POLICY` not set (the
  runs marked "passive" / "active" set it).
- Order in a job: all `ncc` runs with symmetry, all without, then zccsdt; one
  run of each program was repeated at the end of the job and agreed with the
  first within 1 % (O3, 8 threads: 8.43 / 8.52, 75.9 / 76.0, 19.52 / 19.50 s;
  benzene, 24 threads: 15.44 / 15.44, 180.0 / 180.6, 34.37 / 34.35 s).
- zccsdt: revision 1cc2e34 (the code of a5045bf), `zccsdt.CCSDT` on RHF
  (`RCCSDT`), PySCF 2.13.1, numpy 2.4.6, `max_memory` 70 GB, DIIS with T3, 8
  vectors, **all DIIS vectors in memory** (`incore_complete`; with the T3
  vectors in a scratch file an iteration is slower, `docs/usage.md`).  Peak
  resident memory of the iteration runs: 16 GB (O3), 30 GB (benzene).
- `ncc`: CFOUR (banner "Version 1.2", 64-bit integers) of
  `/home/xwang405/apps/cfour-x2cso`, with `xncc` rebuilt by
  `bench/cfour/rebuild_ncc.sh gcc` (g++ 9.3 `-O3 -march=native -fopenmp`, the
  fastest of the builds compared in `docs/benchmarks.md`; the installation
  has `-O2`), threaded MKL 2022.1 (`libmkl_intel_thread`, `libiomp5`) next
  to `libgomp`.  `CC_PROG=NCC`, `MEMORY_SIZE` 60 GB (memory limit 55.9 GiB),
  `SCF_CONV=10`, basis as `BASIS=SPECIAL` from PySCF's data, `UNITS=BOHR`.
  Peak resident memory of the iteration runs: 1.2 GB / 4.7 GB (O3, with /
  without symmetry), 2.0 GB / 14 GB (benzene).
- Times: the per-iteration wall times both programs print; (Q): the wall time
  `ncc` prints for it, and the wall time of `ccsdt_q()`.  `ncc` does (Q) only
  after converged iterations, so its (Q) runs converge CCSDT (to `CC_CONV=3`
  for the timing runs); the iteration-only runs stop at `CC_MAXCYC`.  zccsdt
  does (Q) on the amplitudes after 2 iterations (the time does not depend on
  them).

### Energies

The 24-thread (Q) runs of both programs are converged (`CC_CONV=10`; zccsdt
|dE| < 1e-11):

| | `ncc`, symmetry | zccsdt | difference |
|---|---|---|---|
| O3: E(SCF) | -224.340411087556 | -224.340411087550 | 6e-12 |
| O3: E(CCSDT) | -225.169291054214 | -225.169291054225 | 1.1e-11 |
| O3: E(CCSDT(Q)) | -225.176643879028 | -225.176643879041 | 1.3e-11 |
| benzene: E(SCF) | -230.721903098552 | -230.721903098547 | 5e-12 |
| benzene: E(CCSDT) | -231.581421583844 | -231.581421583844 | 1e-13 |
| benzene: E(CCSDT(Q)) | -231.584163569651 | -231.584163569652 | 1e-12 |

`ncc` without symmetry was not converged here; the correlation energies of
its iterations are those of the runs with symmetry to 6.0e-12 (O3, 5
iterations) and 6.6e-12 (benzene, 3 iterations).

### How `ncc` is threaded in this build

- OpenMP in its own code (`#pragma omp` in `core/contractions.cxx`,
  `transformations.cxx`, `lists.cxx`: the blocks of a contraction are dealt
  to the threads) and the threaded MKL outside those regions.  It reads
  `OMP_NUM_THREADS` (without it: one thread; it prints "Running with n
  threads/proc") and calls `omp_set_dynamic(false)`, `omp_set_nested(false)`.
- It uses the cores: 1 -> 8 threads gives 5.0 (no symmetry) and 4.9 / 3.2
  (symmetry) per iteration; `ps` saw exactly n threads in every run.
- `MKL_NUM_THREADS=1` is slower, so MKL's threads do part of the work: O3 8
  threads 10.1 against 8.4 s, 24 threads 8.6 against 5.8 s (symmetry), 62.5
  against 57.3 s (no symmetry, 24); benzene 24 threads 24.5 against 15.4 s
  and 198 against 180 s.
- `OMP_WAIT_POLICY=PASSIVE` slows `ncc` with symmetry: O3 9.49 against 8.43
  s (8 threads) and 7.69 against 5.77 s (24); benzene 23.1 against 15.1 s (8)
  and 33.0 against 15.4 s (24).  Without symmetry and for (Q) it makes no
  difference (O3 (Q), 24 threads: 865 against 863 s); `ACTIVE` equals the
  default.  The tables below are with the default.  (For zccsdt `PASSIVE`
  and the default give the same times.)
- Number of threads: `ncc` deals blocks to threads by a rule on the block
  indices that is uneven when the number of threads shares factors with the
  number of occupied orbitals (`docs/benchmarks.md`), so 7, 23 and 47 threads
  were run as well; without symmetry they are faster than 8, 24 and 48.
- Memory: the iterations need little ("optimal" 0.9 - 13 GiB, limit 55.9).
  For (Q) `ncc` would like 259 GiB (O3) and 606 GiB (benzene), more than the
  node has; with `MEMORY_SIZE` 150 GB instead of 60 (48 threads) (Q) took 834
  against 812 s (O3) and 1980 against 1626 s (benzene): no gain.

## O3 / cc-pVTZ, all electrons (job 31782852, c304)

Seconds per CCSDT iteration:

| threads | 1 | 2 | 4 | 7 | 8 | 16 | 23 | 24 | 47 | 48 |
|---|---|---|---|---|---|---|---|---|---|---|
| `ncc`, C2v | 41.09 | 23.01 | 13.40 | 9.44 | 8.43 | 6.14 | 5.70 | 5.77 | 9.14 | 9.98 |
| `ncc`, no symmetry | 380.6 | 215.1 | 131.9 | 66.2 | 75.9 | 53.3 | 39.3 | 57.3 | 30.7 | 41.1 |
| zccsdt RCCSDT | 134.4 | 72.2 | 37.4 | | 19.52 | 11.74 | | 9.56 | | 8.26 |
| speedup from 1 thread: `ncc`, C2v | 1 | 1.79 | 3.07 | 4.35 | 4.87 | 6.69 | 7.22 | 7.12 | 4.50 | 4.12 |
| `ncc`, no symmetry | 1 | 1.77 | 2.89 | 5.75 | 5.01 | 7.14 | 9.68 | 6.64 | 12.39 | 9.25 |
| zccsdt | 1 | 1.86 | 3.59 | | 6.88 | 11.45 | | 14.05 | | 16.27 |
| speedup from 8 threads: `ncc`, C2v | | | | | 1 | 1.37 | 1.48 | 1.46 | 0.92 | 0.85 |
| `ncc`, no symmetry | | | | | 1 | 1.42 | 1.93 | 1.32 | 2.47 | 1.85 |
| zccsdt | | | | | 1 | 1.66 | | 2.04 | | 2.36 |
| `ncc` C2v / zccsdt | 0.31 | 0.32 | 0.36 | | 0.43 | 0.52 | | 0.60 | | 1.21 |
| `ncc` no symmetry / zccsdt | 2.83 | 2.98 | 3.52 | | 3.89 | 4.54 | | 6.00 | | 4.98 |

With the best number of threads of `ncc` up to the count of the column (7 for
8, 23 for 24, 47 for 48), `ncc` without symmetry / zccsdt is 3.39, 4.11 and
3.72, and its speedup from "8" to "24" is 1.68.

Seconds of (Q) (`ncc` without symmetry: not run, see below):

| threads | 1 | 2 | 4 | 8 | 16 | 24 | 48 |
|---|---|---|---|---|---|---|---|
| `ncc`, C2v | 4579 | 2691 | 1617 | 1057 | 825 | 863 | 812 |
| zccsdt RCCSDT | 4635 | 2575 | 1191 | 635 | 406 | 359 | 317 |
| speedup from 1 thread: `ncc` | 1 | 1.70 | 2.83 | 4.33 | 5.55 | 5.31 | 5.64 |
| zccsdt | 1 | 1.80 | 3.89 | 7.29 | 11.41 | 12.91 | 14.62 |
| speedup from 8 threads: `ncc` | | | | 1 | 1.28 | 1.23 | 1.30 |
| zccsdt | | | | 1 | 1.56 | 1.77 | 2.01 |
| `ncc` C2v / zccsdt | 0.99 | 1.05 | 1.36 | 1.66 | 2.03 | 2.40 | 2.56 |

## Benzene / cc-pVDZ, frozen core (job 31782853, c255)

Seconds per CCSDT iteration:

| threads | 1 | 2 | 4 | 7 | 8 | 16 | 23 | 24 | 47 | 48 |
|---|---|---|---|---|---|---|---|---|---|---|
| `ncc`, D2h | 48.80 | 27.75 | 18.62 | 15.32 | 15.08 | 15.39 | 15.61 | 15.44 | 23.13 | 22.87 |
| `ncc`, no symmetry | 1232 | 631 | 346 | 229.5 | 243.8 | 206.3 | 161.0 | 180.0 | 125.3 | 135.5 |
| zccsdt RCCSDT | 536.1 | 274.7 | 140.8 | | 72.7 | 41.0 | | 34.4 | | 28.3 |
| speedup from 1 thread: `ncc`, D2h | 1 | 1.76 | 2.62 | 3.19 | 3.24 | 3.17 | 3.13 | 3.16 | 2.11 | 2.13 |
| `ncc`, no symmetry | 1 | 1.95 | 3.56 | 5.37 | 5.05 | 5.97 | 7.65 | 6.85 | 9.83 | 9.09 |
| zccsdt | 1 | 1.95 | 3.81 | | 7.38 | 13.07 | | 15.60 | | 18.92 |
| speedup from 8 threads: `ncc`, D2h | | | | | 1 | 0.98 | 0.97 | 0.98 | 0.65 | 0.66 |
| `ncc`, no symmetry | | | | | 1 | 1.18 | 1.51 | 1.35 | 1.95 | 1.80 |
| zccsdt | | | | | 1 | 1.77 | | 2.11 | | 2.57 |
| `ncc` D2h / zccsdt | 0.09 | 0.10 | 0.13 | | 0.21 | 0.38 | | 0.45 | | 0.81 |
| `ncc` no symmetry / zccsdt | 2.30 | 2.30 | 2.46 | | 3.36 | 5.03 | | 5.24 | | 4.78 |

With the best number of threads of `ncc` (7, 23, 47): `ncc` without symmetry
/ zccsdt 3.16, 4.68 and 4.42; its speedup from "8" to "24" is 1.43.

Seconds of (Q), from 8 threads (one thread: about 6 hours for zccsdt, an
estimate from its O3 speedup, not run):

| threads | 8 | 16 | 24 | 48 |
|---|---|---|---|---|
| `ncc`, D2h | 1607 | 1467 | 1518 | 1626 |
| zccsdt RCCSDT | 2948 | 1835 | 1604 | 1284 |
| speedup from 8 threads: `ncc` | 1 | 1.10 | 1.06 | 0.99 |
| zccsdt | 1 | 1.61 | 1.84 | 2.30 |
| `ncc` D2h / zccsdt | 0.55 | 0.80 | 0.95 | 1.27 |

## What the numbers say

- **zccsdt scales better than `ncc` in every comparison made.**  From 8 to 24
  threads (ideal 3) the CCSDT iteration of zccsdt gains 2.04 (O3) and 2.11
  (benzene), (Q) 1.77 and 1.84; `ncc` without symmetry gains 1.32 and 1.35
  (1.68 and 1.43 with its best thread counts, 7 and 23), `ncc` with symmetry
  1.46 and 0.98 per iteration and 1.23 and 1.06 in (Q).  From one thread to
  24, zccsdt reaches 14.1 and 15.6 per iteration, `ncc` at best 9.7 and 7.7
  without symmetry and 7.2 and 3.2 with it.  The hypothesis holds: zccsdt's
  8 -> 24 gain is about 2.0 - 2.1 of 3, and `ncc`'s is smaller.
- **The larger molecule does not help `ncc`.**  zccsdt scales a little better
  on benzene than on O3; `ncc` scales worse on it, with symmetry (D2h: no
  gain at all beyond 7 threads; the blocks are small) and without (8 -> 24:
  1.35).
- **In absolute time `ncc` with point-group symmetry is still ahead per
  iteration on one socket**: 0.60 (O3) and 0.45 (benzene) of zccsdt's time
  at 24 threads, down from 0.31 and 0.09 on one thread.  On 48 threads zccsdt
  is ahead for O3 (1.21) and nearly level for benzene (0.81).  In (Q) zccsdt
  is faster for O3 from 4 threads on (2.4 times at 24) and for benzene only
  on 48 threads (0.95 at 24, 1.27 at 48).  Against `ncc` without symmetry,
  the like-for-like algorithm, the ratio grows from 2.3 - 2.8 on one thread
  to 5.2 - 6.0 on 24 (4.1 - 4.7 with `ncc`'s best thread count).
- **The second socket**: zccsdt gains 1.16 (O3) and 1.21 (benzene) per
  iteration from 24 to 48 threads (1.24 and 1.35 with `--interleave=all`:
  7.70 and 25.4 s); `ncc` with symmetry gets slower (5.8 -> 10.0 s, 15.4 ->
  22.9 s), without symmetry it gains 1.33 - 1.39.

## Limits of the comparison

- CPU time is not a measure of work for `ncc` here: with the default wait
  policy its idle threads spin, so "cores busy" (CPU / wall) is 23.9 of 24
  for the benzene iteration with symmetry while the speedup is 3.2; with
  `PASSIVE` it is 5.1 (and the run is slower).  For zccsdt CPU / wall is
  20.8 (O3) and 22.6 (benzene) of 24 with either policy.  Both columns are
  in `bench/ncc_scaling/results/`.
- One-thread runs have the highest clock rate of the processor (turbo), so
  speedups from one thread understate the parallel efficiency of both
  programs alike.
- `ncc` without symmetry: (Q) was not run (an earlier job, 31777062, took
  19148 s for O3 on 8 threads with 24 GB); its converged energy was not
  recomputed here.
- `ncc`'s (Q) ran with less memory than it asks for, because the node does
  not have it; more memory within the node did not help (above).
- (Q) of O3 on one and two threads took 76 - 77 minutes per program, above
  the hour set as the limit; benzene (Q) below 8 threads was not run.
- The iteration times are means of 4 (O3) and 2 (benzene) iterations of a
  single run per thread count; the repeats at the end of each job differ by
  at most 1 %.
- One unexplained outlier: zccsdt for O3 with `OMP_WAIT_POLICY=ACTIVE` on 24
  threads took 74.4 s per iteration (default and `PASSIVE`: 9.56 and 9.58 s);
  the same setting for benzene gave 34.3 s against 34.4 s.  It was not
  repeated.
- The spin-orbital program (`GCCSDT` on the GHF form of the same RHF) was to
  be run for O3 at 8 and 24 threads and was **not measured**: with
  `incore_complete` it stopped with a MemoryError (8 pairs of DIIS vectors
  would take 160.8 GB).
- c255 (benzene) started with 61 GB and 30 GB free on its two memory nodes
  (c304: 80 GB each).  Whether that affected the 48-thread runs on c255 was
  not examined.
