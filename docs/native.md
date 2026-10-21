# xzccsdt: the native program in CFOUR's two-component chain

`cfour/xzccsdt` is a program without Python for spinor CCSDT, CCSDT(Q) and
CCSDT(Q)_Lambda energies, for lambda of CCSDT, for the one-particle density
and orbital gradient of CCSDT that CFOUR's `xx2czvec` turns into the relaxed
density, and for the excitation energies of EOM-EE-CCSDT.  It runs as a member of CFOUR's X2CSO chain, after `xjoda`,
`xsox2catm`, `xsox2chf` (and `xx2camf` for X2CAMF) and `xx2cvtran` with
`ABCDTYPE=STANDARD`, in their directory:

    xjoda; xsox2catm; xsox2chf; xx2cvtran; xzccsdt

The entry point and the scheduler are Fortran in the manner of CFOUR's x2c
programs, the kernels are the C kernels of this repository (`csrc/`), and all
working memory is the work array that CFOUR's `CRAPSI` allocates
(`MEMORY_SIZE` of the ZMAT).  The algorithms are those of the Python program
(`src/zccsdt/cc/ccsdt_opt.py`, `quadruples.py`, `t3engine.py`, `lambda_opt.py`,
`lambda_blk.py`, `quadruples_lambda.py`, `eom_opt.py`, `t3shift.py`;
docs/factorization.md, docs/lambda.md, docs/eom.md), which is the reference
of every number below.

State of this document: the program at the commit that runs the CCSDT
iterations and the Lambda iterations over MPI ranks (section 15; (Q) over
ranks with whole groups and a mapped T3: section 14; the density stage:
section 13; the threading of the iterations: section 12; the Lambda
equations and (Q)_Lambda: section 11).  What is measured says so (job, node);
what is not done is listed at the end.  In short: CCSDT, (Q), the Lambda
equations of CCSDT and (Q)_Lambda - dense or in the blocks of symmetry
labels, complex or real, the iterations and (Q), (Q)_Lambda over MPI ranks,
(Q) with restart - are there and validated against the Python program; so is the density stage of
CCSDT with the files for `xx2czvec`, frozen and dropped spinors included
(section 13), validated against the Python hand-over of section 10 and
through `xx2czvec` and `xsox2cprop`.  Excited states: EOM-EE-CCSDT
(`EXCITE=EOMEE`) in all four modes, in blocks for the states of a label shift,
with CFOUR's solver design and input - section 16; over MPI ranks: 16.7.  CC3:
without stored triples (the batches of `csrc/cc3_fly.c`, over MPI ranks;
EOM-CC3 folded, the totally symmetric states) and, as a check, with T3
stored (its EOM in every label shift) - section 17.  CC4 and EOM-EE-CC4
without stored quadruples, over MPI ranks, with checkpoints after every
evaluation of the T4 terms - section 19.  EOM-DEA-CCSDT (two electrons
attached, 2p + 3p1h + 4p2h on the sector itself, the roots of the targets 2
Omega of `%xzdea`), over MPI ranks too - section 20; with complete 5p3h
(`level 5p3h`), every array of the size of r5 distributed - section 21.

**The program takes no argument** (since commit a38e493): the job comes from
the ZMAT as joda stored it, and the steps that were arguments before (`(q)`,
`lambda`, `density`, `eom` on the files of an earlier run) happen by
themselves when the files `XZAMPS`, `XZLAMB` fit the run (section 1).  The
measurements below name the argument that was given then:

| argument (before a38e493) | now: ZMAT |
|---|---|
| `ccsdt`, `ccsdt[q]`, `ccsdt(q)` | `CALC=CCSDT`, `CCSDT[Q]`, `CCSDT(Q)` |
| `ccsdt(q)_l` | `CALC=CCSDT(Q)_L` |
| `ccsdt[q]_l` | not available (joda does not know `CCSDT[Q]_L`); `E[Q]_L` is printed with `CCSDT(Q)_L` |
| `ccsdt_density` | `CALC=CCSDT,PROPS=FIRST_ORDER` (or `DERIV_LEV=1`) |
| `ccsdt_eom` | `CALC=CCSDT,EXCITE=EOMEE` |
| `(q)`, `[q]`, `lambda`, `(q)_l`, `[q]_l`, `density`, `eom` | the same ZMAT as the full job, in the directory of an earlier run: `XZAMPS` (`XZLAMB`) are taken |
| `mp2` | `CALC=MP2` |
| `ccsd_eom` | removed (CFOUR's `xx2ceomcc` is EOM-CCSD) |

## 1. Build and run

    source /data/lcheng24/intel/oneapi/setvars.sh --force
    cd cfour/xzccsdt
    make                # -> build/xzccsdt/xzccsdt           (serial: no MPI in it)
    make MPI=1          # -> build/xzccsdt_mpi/xzccsdt       (the iterations and (Q) over ranks)

Inside CFOUR (the program directory `zccsdt`, the binaries `bin/xzccsdt`
and `bin/xzccsdt_mpi`): `cfour/merge/README.md`.

Compilers and flags are those of the CFOUR build (`ifort`, `icc`, `-i8`,
`-assume byterecl`; `make.config` of `~/apps/cfour-x2cso-rebuild`).  Linked:
`liblibr.a` of that build (`CRAPSI`, `GETREC`, `PUTREC`, `GETMEM`, `ERREX`,
`CRAPSO`), MKL with 8-byte integers (`-lmkl_intel_ilp64 -lmkl_intel_thread
-lmkl_core -liomp5`, the link line of CFOUR), and for `MPI=1` Intel MPI through
`mpiifort` / `mpiicc`; and for the density with frozen or dropped spinors
`libx2cint.a` (`int_mintv2e`, the module `cvwint_module` of `libx2c`) and
three objects of the program directory `x2czvec` (`vtranpsog.o`,
`vtran1sparse.o`, `expnd2p.o`; `FRZOBJ` of the Makefile; section 13).  Nothing
is written to the CFOUR tree and no CFOUR source is copied.  The run-time libraries are found through an rpath: the
executable runs without the oneAPI environment (the MPI launcher needs it).

The C kernels (`KERNELS` of the Makefile: `ccsdt_t3.c`, `t3pack.c`,
`tnpack.c`, `lambda_t3.c`, `qlambda.c`) are compiled from `csrc/` with `-DZC_USE_MKL -DMKL_ILP64`: their
BLAS calls go through `mkl_cblas.h`, whose integer is then 64 bits, as the
kernels' own `int64_t` arguments.  The program checks at start that the
kernels were compiled for 8-byte BLAS integers (`sizeof(MKL_INT)`) and stops
otherwise.  A GEMM with more than 2^31 elements in one operand was not run
(32 GB for one complex operand); the results of all cases below agree with
the LP64 Python build.

    xzccsdt

The program takes no argument (one stops it: `@XZCCSDT-F, xzccsdt takes no
argument ...` with the keywords below).  As every CFOUR program it does what
the ZMAT asks for, from the flags that joda stored in JOBARC (`xzjob` in
`xzdrv.F90`):

| ZMAT | job |
|---|---|
| `CALC=CCSDT` (`CC(n)` with `EXCITATION=3`) | CCSDT |
| `CALC=CCSDT[Q]` | CCSDT, [Q] (the final energy; (Q) printed too) |
| `CALC=CCSDT(Q)` (`CC(n-1)(n)` with `EXCITATION=4`) | CCSDT, (Q) and [Q] |
| `CALC=CCSDT(Q)_L` (`CC(n-1)(n)_L` with `EXCITATION=4`) | CCSDT, the Lambda equations, (Q)_Lambda and [Q]_Lambda (section 11) |
| `CALC=CCSDT,EXCITE=EOMEE` | CCSDT, then EOM-EE-CCSDT (section 16: the roots, their label shifts and the thresholds from `%eomnroot`, `%xzeom`, `ESTATE_CONV`, `ESTATE_MAXCYC`) |
| `CALC=CCSDT,PROPS=FIRST_ORDER` or `DERIV_LEV=1` | CCSDT, Lambda, the unrelaxed density and the orbital gradient as the files of `xx2czvec` (section 13) - the keywords with which CFOUR's own spinor CC programs (`xx2cccsd`: `max(DERIV_LEV, PROPS)`) make the density |
| `CALC=MP2` | the MP2 energy of the integral files (a check; `TOTENERG` not written) |
| `CALC=CC3` (with or without `EXCITE=EOMEE`) | stops: not in this version (the place of the next method in `xzjob`) |
| `CALC=CC4` (with or without `EXCITE=EOMEE`) | CC4 without stored quadruples, from the CCSDT amplitudes; EOM-EE-CC4 with the quadruples folded in (section 19) |
| `CALC=CCSDT` and the section `%xzdea` | CCSDT, then EOM-DEA-CCSDT: the states with two electrons more (section 20: the targets 2 Omega and their roots from `%xzdea`, the thresholds as for EOM-EE; `level 5p3h`: section 21) |

What joda (x2cso branch, `cfour-x2cso/bin/xjoda`, with `RELATIVISTIC=X2C1E`)
stores for these, measured 2026-10-12 on the test case of the package
(`cfour/merge/test`): `CALC=CCSDT,EXCITE=EOMEE` becomes `PROPS=EXCITATION`,
`EXCITE=NONE`, `DERIV_LEV=SECOND`, `CC_PROG=ECC` (joda's rule for EOM-EE with
CCSDT unless `CC_PROG=NCC` or `MRCC`) - the program reads either form as
EOM; `PROPS=FIRST_ORDER` sets `DERIV_LEV=FIRST` and `RELAX_DENS=ON`;
`DERIV_LEV=1` sets `CC_PROG=ECC`; `CCSDT[Q]` and `CCSDT(Q)` set `CC_PROG=NCC`.
None of this changes what the chain writes: E(SCF) and E(CCSDT) of all these
runs agree to all digits printed; `DROPMO` is not touched.  `CALC=CCSDT[Q]_L`
is not a value of joda: it is stored as `SCF` without a message, and the
program stops (`CALC ... is not one of ...`).  EOM or the density with
another `CALC` than `CCSDT` stops the program.

**Files of an earlier run.**  The amplitudes go to `XZAMPS`, lambda to
`XZLAMB`, and the file `XZKEY` (text) records what they were solved for: the
active occupied and virtual spinors, all spinors, the frozen spinors,
E(SCF) of JOBARC, the sum of the spinor energies of `scf2ceig` and the sum
of (p + 1) times the energy of spinor p (the order), `CC_CONV` of the
amplitudes and of lambda, and (the last line, `XZKEY02`) the model of the
amplitudes (CCSDT or CC3), their cycles in all runs and the largest element
of their last step, the same two numbers of lambda.  At the start (`xzreuse`; rank 0 decides, the
other ranks are told) `XZAMPS` is taken in the place of the CCSDT iterations
if `XZKEY` and `XZAMPS` are there, the header of `XZAMPS` has the active
spaces, the sizes agree exactly and the three energies to 1e-9 (relative),
and it was converged at least as tightly (`CC_CONV` not smaller); `XZLAMB`
in the place of the Lambda iterations if moreover it is lambda of these
amplitudes (its header, the CCSDT energy, as before) and its `CC_CONV` is
not smaller.  The output says which: `amplitudes: the file XZAMPS of an
earlier run belongs to these integrals (XZKEY): taken, the CCSDT iterations
are skipped` or `amplitudes: solved here`, likewise for lambda.  The labels
and the arithmetic are decided again from the same files; a file written in
another mode (`labels`, `real` changed between the runs) stops the run
when it is read.  So the steps can be separate runs in one directory:
`CALC=CCSDT`, then xjoda again with `CALC=CCSDT(Q)` (or `EXCITE=EOMEE`, or
`PROPS=FIRST_ORDER`) and the program again: only the new part is computed
(rerunning xjoda with another `CALC` in a finished directory: run, the files
of the SCF and of xx2cvtran stay valid).  Remove the files to force a new
solution.

**Iterations that did not finish are continued.**  While the amplitudes
are iterated, rank 0 writes them to `XZAMPS` every K cycles - the first time
after cycle 2; K = 1 + 100 t(write) / t(cycle) from that write, so the
writes take about 1 % of the time; the output states K - and once more when
`CC_MAXCYC` cycles are over without convergence; the Lambda iterations do
the same with `XZLAMB`.  `XZKEY` then has, in the place of `CC_CONV`, the
digits d that were reached (the largest element of the last step below
10^-d), so the reuse rule above needs no change: a run that asks for no
more than d takes the file as it is, and a run that asks for more - the
same ZMAT again - finds a file of its integrals and its model that is not
converged as tightly and STARTS its iterations from it instead of from the
second-order amplitudes (lambda: instead of from conj(t)):
`amplitudes: the file XZAMPS of an earlier run has CCSDT amplitudes of these
integrals that are not converged as tightly as asked (XZKEY: 12 cycles, the
largest element of the last step 1.178E-07): the iterations CONTINUE from
them`.  The DIIS vectors are not saved: the sequence starts anew, which
costs one or two cycles in all (check below).  At `CC_MAXCYC` the program
stops with CFOUR's error as before, and the message says that the
amplitudes are saved and that a rerun continues.  A write goes to
`XZAMPS.new` (`XZLAMB.new`), which then takes the name, and `XZKEY` is
removed before and written after: a run that is killed while it writes
leaves the file of before, or no key.  The model is part of the key: CCSDT
continues CCSDT amplitudes, CC3 continues CC3 amplitudes (the only use of
CC3 amplitudes of a file - a finished CC3 run leaves no `XZKEY`, section
17), CC4 continues the CCSDT iterations of its start (its own iterations
have the checkpoint `XZCC4`, 19.5).  A file of another mode (`labels`,
`real`, `cc3 stored` changed) that is not converged is left aside with a
line, and the iterations start as without it.  `XZKEY01` files of earlier
versions (converged CCSDT amplitudes, by the tighter test of then) are read
as before.

Check (2026-10-13, node c260, 8 threads; `CC_MAXCYC` cut, the same ZMAT run
again until it finishes; cycles of all runs against one run without a stop;
1 rank and 2 ranks with slices of T3 gave the same counts and energies):

| case | cut to | uninterrupted | continued | energies, continued against uninterrupted |
|---|---|---|---|---|
| HBr `hbr_dglin_fc` (8 + 28), `CALC=CCSDT(Q)_L`, CC_CONV=10 | 6 | CCSDT 17, Lambda 16 | CCSDT 6 + 6 + 6 + 1 = 19, Lambda 6 + 6 + 5 = 17 (6 runs) | E(CCSDT) 6e-12, E(CCSDT(Q)_L) 6e-12 |
| HBr `hbr_fc` (complex, dense), `CALC=CCSDT(Q)`, CC_CONV=10 | 7 | 17 | 7 + 7 + 5 = 19 | E(CCSDT), E(CCSDT(Q)) 4e-12 |
| H2O `h2o_631g`, `PROPS=FIRST_ORDER`, CC_CONV=9 | 5 | CCSDT 15, Lambda 13 | CCSDT 5 + 5 + 5 + 1 = 16, Lambda 5 + 5 + 4 = 14 | E(CCSDT) 9.5e-11 |
| HBr `hbr_dglin_fc`, `CALC=CC3` | 6 | 15 | 6 + 6 + 5 = 17 | E(CC3) identical (12 decimals) |
| HBr `hbr_dglin_fc`, `CALC=CC4` (the cut hits its CCSDT iterations) | 6 | CCSDT 17, then 23 cycles of CC4 (`CC4 converged in 40 cycles`) | CCSDT 6 + 6 + 6 + 1 = 19, then 23 cycles of CC4 in the last run | E(CC4) identical |
| HBr / cc-pVTZ-DK `hbr_tz_dglin` (8 + 78), `CALC=CCSDT`, killed (SIGKILL) in cycle 10 of 18; K = 3: saved after cycle 8 | - | 18 | 8 kept + 11 = 19 | E(CCSDT) 1e-12 |
| the same on 2 ranks, killed in cycle 4; K = 2: saved after cycle 4 | - | 18 | 4 + 14 = 18 | 6e-12 |

**`CC_CONV=N`: the iterations of the amplitudes and of lambda stop when the
largest element, in absolute value, of the change of the amplitudes in the
cycle - new minus old before the DIIS extrapolation, which is the residual
divided by the denominator of spinor energies - over T1, T2 and T3 (l1, l2,
l3) is below 10^-N.**  This is the test of CFOUR's spinor CCSD: `xx2cccsd`
(`x2c_ccsdsoao.f90`) leaves its loop on `rmax.lt.conv`, with `conv =
10**(-abs(iflags(4)))` (`setsocc.f90`) and `rmax = max(r2max, r1max)` set at
the end of `hbarso.f90` from `t1normso` / `t2normso` of r1, r2 *after*
`r1updso` / `r2updso` have divided the residuals by the denominators - the
largest modulus of an element of the update that `t12updso.f90` then adds
to the amplitudes; its Lambda equations (`x2cccdenoc/lbdeq.f90`) use the
same line.  CFOUR has no test of the energy, and neither has this program:
the energy change is a column of the table only.  (One difference in the
order: `xx2cccsd` tests the update before it is added and DIIS is applied,
this program after - its amplitudes at convergence include the last step.)
Until 2026-10-13 the test was |dT1| + |dT2| + |dT3| of the full tensors
below 10^-N and the energy change below 10^-(N+2), the convention of the
Python front end (`zccsdt.cfour`, unchanged: there `CC_CONV=N` still means
the norms and the energy).  The norm of the full T3 counts every element 36
times and sums over all of them, so that test was tighter by a factor that
grows with the size of the tensors: 20 for HBr / cc-pVTZ-DK (8 + 78), and
a run of a U- atom (15 + 315 active spinors, CC_CONV=7) had stopped at
`CC_MAXCYC=100` with the three norms at 1.2e-8, 4.3e-8, 5.7e-8 and the
energy changing by 2e-10.  The table of the iterations keeps the three
norms and has the new measure as the column `max|dT|` (`max|dL|`):

      cycle     correlation energy      energy change      |dT1|      |dT2|      |dT3|   max|dT|   DIIS    time/s
         17      -0.19890660878678      -1.541792E-11  2.840E-10  1.215E-09  2.065E-09  1.49E-10      8      0.12
         18      -0.19890660878767      -8.885115E-13  1.324E-10  3.981E-10  6.680E-10  5.82E-11      8      0.12

CC3 uses the same loop and test; CC4 uses the same measure in its macro /
micro logic (19.2), without the energy.  The excited states have their own
threshold, `ESTATE_CONV` (16.1), which already was CFOUR's: `eomene2.f90`
calls `eomconv` - the largest modulus of an element of the residual `H c -
omega c` at most `eomthr = 10**(-abs(iflags(98)))` or the number of
`%eomthrs` - and the complex solver here calls the same routine.
`CC_MAXCYC` as before.  Old against new on the regression cases
(`regress.sh --tol --eom`, 2026-10-13; serial and 2 ranks alike): CCSDT 17
-> 15 cycles (H2O, CC_CONV=9), 19 -> 17 (the five neutral HBr cases,
CC_CONV=10), 22 -> 20 (HBr+); Lambda 15 -> 13, 18 -> 16 (17 -> 15 with
dropped virtuals), 20 -> 16; every energy within 3.4e-11 (H2O) and 5e-12
(HBr), the elements of the density files within 9.3e-11 and 1.7e-11, the
excitation energies identical to the 8 decimals of `XZEOM`, with the same
number of EOM iterations.

Frozen and dropped spinors are what CFOUR left
(`DROPMO`, `%vircut`): the program takes the active spinors of the integral
files.

Exit: status 0 and CFOUR's `@CHECKOUT-I` line at the end (the line before:
`XZCCSDT finished: <method> energy done.`); every error goes through CFOUR's
`ERREX` (status 1, the `ERROR` banner), with a line `@XZCCSDT-F, ...` that
says what and, for memory, how many words are needed and how many are there.
The final energy is written to JOBARC as `TOTENERG`, as `xx2cccsd` does
(`lambda`: the CCSDT energy again).

Threads: `OMP_NUM_THREADS` (kernels and MKL share Intel's OpenMP runtime).

### MPI: the iterations and (Q) over ranks

    export PATH=$PATH:/cm/shared/apps/slurm/current/bin   # in a job: mpirun needs srun (section 8)
    export OMP_NUM_THREADS=24 I_MPI_PIN_DOMAIN=omp        # threads per rank
    mpirun -np 8 -ppn 2 xzccsdt_mpi                       # Intel MPI, inside a Slurm allocation

e.g. `sbatch -N 4 --ntasks-per-node=1 -c 48 --mem=180G` - one Slurm task per
node with all its cores; Intel MPI starts `-ppn` ranks on every node and
pins each to `OMP_NUM_THREADS` cores (`-ppn 2`: one rank per socket) - or `-N
1 --ntasks-per-node=1 -c 48` and `mpirun -np 2` on one node
(`I_MPI_HYDRA_BOOTSTRAP=fork` then needs no `srun`).  What was run, with
which fabric, and what is specific to Intel MPI: section 15.

Rank 0 is the CFOUR program (CRAPSI, JOBARC, the output).  The other ranks
get joda's flags from it, write their output to `xzccsdt.rNNNN`, wait for
the plan of rank 0 and allocate a work array themselves (`GETMEM`) - not
`MEMORY_SIZE` but the words that their part takes with that plan (section
4) -, and read the integral files themselves: **the run directory must
exist, with the integral files (and `dgcosym`, `xzlabels`: the symmetry
labels, section 6), under the same path on
every node** - a shared file system, or a copy in the node-local scratch of
every node.  All ranks read the same files and make the same decisions
(blocks, real or complex, the batches and their ranks, the task list of
(Q)).  Rank 0 sends what it decided (classes of the labels, real or complex,
the stored elements of T3) with its plan, and a rank that decided otherwise
from its files stops the run (`@XZCCSDT-F, this rank and rank 0 have not
decided alike ...`).

- **The CCSDT iterations and the Lambda iterations** run on all ranks
  (section 15): every rank has the amplitudes in its own work array, does
  its batches of pass 1, pass 2 and the pp ladder, and the parts of the
  residual are summed.  Nothing is read from `XZAMPS` or `XZLAMB` for them:
  rank 0 sends the amplitudes, so a node-local directory on every node is
  enough for `ccsdt`, `lambda` and `ccsdt_density`.
- **(Q) and (Q)_Lambda** run on all ranks as before (sections 5, 14): the
  ranks read T1 and T2 of `XZAMPS` (and lambda of `XZLAMB`) and map T3 -
  which rank 0 has written - so for them the run directory must be one that
  all nodes see (a blocking broadcast tells the other ranks when the files
  are there; with Intel MPI a waiting rank spins on one core unless
  `I_MPI_WAIT_MODE=1`).
- The density stage (`density`, the last step of `ccsdt_density`) is rank
  0's.
- **The excited states** (EOM-EE-CCSDT) run on all ranks (section 16.7):
  rank 0 runs the solver and sends the converged amplitudes once; every
  rank does its batches of a sigma vector and holds its rows of R3 of every
  vector of the subspace.  A node-local directory on every node is enough.
  (EOM-CC3 folded: section 17.4; EOM-CC4: section 19.4.  The check route
  `cc3 stored` with `EXCITE=EOMEE` runs with one rank: with more the program
  stops with a message.)

The steps can be separate invocations in one directory: `CALC=CCSDT`
(serial or one rank), then `CALC=CCSDT(Q)` with `mpirun ...`: the second
takes `XZAMPS` of the first.

## 2. Files

Read (formats: `src/zccsdt/cfour.py`, docs/usage.md "Integrals from another
program"; all integral files are one record of an unformatted sequential
file, complex, the first index fastest, antisymmetrized integrals):

| file | content |
|---|---|
| `JOBARC`, `JAINDX` | through CFOUR's routines: `IFLAGS`, `IFLAGS2` (CRAPSI), `NUMDROPA`, `SCFENEG` |
| `nmototal`, `scf2ceig` | number of spinors, their energies (order: frozen, active occupied, active virtual, dropped) |
| `moinhhhh`, `moinhhhp`, `moinhhpp`, `moinphhp` | `<ij||kl>`, `<ij||ka>`, `<ij||ab>`, `<ia||bj>` |
| `PPPH0001` ... | `<ab||ci>`, one file per occupied spinor |
| `moinpppp` | `<ab||cd>` on pairs (written only with `ABCDTYPE=STANDARD`) |
| `xzvvvv` | optional, not a file of CFOUR: `<ab||cd>` for a<b, c<d in the blocks of the pairs of one label sum - what `bench/dea/native_rundir.py` writes from a block file of `zccsdt.hblocks`; if it is there, `moinpppp` is not read (format: 20.1) |
| `dgcosym` | optional: the block of the SCF of every spinor (`xsox2chf`; one record of `nmototal` 8-byte integers, the order of `scf2ceig`; all 1 without a section `%dglin` in the ZMAT). The symmetry labels of the run if it has more than one block and there is no `xzlabels` (section 6) |
| `xzlabels` | optional, not a file of CFOUR: a symmetry label per spinor; goes before `dgcosym` (section 6) |
| `XZAMPS` | the amplitudes of an earlier run, for `(q)` / `[q]`, `lambda`, `(q)_l` / `[q]_l` and for the ranks other than 0 |
| `XZLAMB` | lambda of an earlier run, for `(q)_l` / `[q]_l`, `density` and for the ranks other than 0 |
| `xzparity` | optional, not a file of CFOUR: a parity (0 g, 1 u) per spinor of the SCF, the format of `xzlabels` without the modulus; EOM-DEA with `parity g` / `u` in `%xzdea` (section 20.1) |
| `IIII`; `JOBARC`: `scf2ccmo` | only for the density with frozen or dropped spinors: the AO integrals of `xsox2catm` (read by CFOUR's `int_mintv2e`) and the spinors of the SCF |

`moinpphh` is not read (it is the conjugate of `moinhhpp`; the Python reader
compares the two).  The numbers of active spinors come from the lengths of
`moinhhhh` and `moinhhpp`, as in the Python reader, and are checked against
`nmototal` and `NUMDROPA`.

Written:

| file | content |
|---|---|
| `JOBARC` | `TOTENERG`: the energy of the method |
| `XZAMPS` | after converged CCSDT: unformatted stream; `'XZAMPS01'`, 5 integers of 8 bytes (occupied, virtual, words per element: 1 real / 2 complex, classes of the labels, elements of T3 as stored), the correlation energy (real, 8 bytes), then T1 `[i][a]`, T2 `[i][j][a][b]`, T3 as stored (packed i<j<k, a<b<c; its blocks one after the other), row-major, in the order of the spinors of the run that wrote it |
| `XZLAMB` | after converged Lambda equations: the format of `XZAMPS` with the tag `'XZLAMB01'`; the real number is the CCSDT correlation energy of the amplitudes that lambda belongs to (checked against `XZAMPS` when both are read); then l1 `[i][a]`, l2 `[i][j][a][b]`, l3 as T3 is stored |
| `XZDIIS.NNNN` | with several ranks: the slice of rank NNNN of the DIIS vectors, in its run directory, where rank 0 has `XZDIIS` (section 15) |
| `XZDIIS` | only if the DIIS vectors do not all fit in the work array (section 4: all in memory or all in the file): direct-access file with the T3 parts of the vectors (of the amplitudes, then of lambda); deleted at the end |
| `XZQCHK.NNNN` | one per rank during (Q): a first line that names the calculation and the task list (kind, sizes, tasks, a number computed from the list), then a line per finished task (task, its two energies, seconds); deleted when (Q) has finished |
| `xfull`, `dpoo`, `dpvv`, `dii0`, `daa0`, `dpopo`, `dpvpv` | `density`, `ccsdt_density`: the input of CFOUR's `xx2czvec` (section 10.2: one record each, complex, CFOUR's order of the spinors) |
| `xzccsdt.rNNNN` | output of the ranks other than 0 (MPI) |
| `XZEOM`; `JOBARC`: `TOTENER2` | `eom`, `ccsdt_eom`: a text line per excited state (shift, root, omega, total energy, residual, weights); the total energy of the lowest state (section 16.1) |
| `XZEOMR` | the vectors of the roots (section 16.1; not with `%xzeomvec 0`, not if the disk is short); EOM-CC4: written root by root (19.3) |
| `XZCC4` | CC4: the amplitudes and the held T4 terms of the last completed T4 evaluation, or the converged CC4 amplitudes - what a later run of the same ZMAT continues from or takes (19.5) |
| `XZE4R.nnn`, `XZE4Q.nnn` | EOM-CC4, per problem (label shift) nnn: the roots that are done and the vectors; the last T4 evaluation of the root in hand (19.5) |
| `EOMCnnnn`, `EOMHnnnn` | only if the subspace of the EOM solver does not fit in the work array: its vectors and their sigma vectors (CFOUR's names), deleted at the end (section 16.4); over ranks `EOMCnnnn.<rank>`, `EOMHnnnn.<rank>`: the rows of a rank other than 0 (16.7) |
| `XZDEA`; `JOBARC`: `TOTENER2` | EOM-DEA: a text line per state (2 Omega, root, omega, total energy, residual, weights of 2p, 3p1h, 4p2h, the leading pair); the total energy of the lowest state (section 20.1) |
| `XZDXV` | EOM-DEA, only if the work array does not hold it: the rows of T3 and the six-index Xv of the ground state (R13) pair by pair; `XZDXV.<rank>` of a rank other than 0; deleted at the end (section 20.4) |

A run that reads `XZAMPS` or `XZLAMB` must run in the mode of the run that wrote it
(labels used or not, real or complex: both follow from the same integral
files, so this only matters with `labels` / `real`); the header is
checked.

**`XZAMPS` and `XZLAMB` are also what (Q) computes from**: (Q) and (Q)_Lambda
do not read T3 and lambda3 into the work array but map the two files
read-only (`mmap`; `xz_map_file` of `xzutil.c`) and read the amplitudes where
the system has them - in the page cache of the node, once for all the ranks
there (section 5).  This needs no conversion in any mode: the T3 of the file
is the stored T3 of the program byte for byte - the packed dense array, or
its blocks one after the other, real (8 bytes) or complex (16) - behind a
header of 56 bytes and T1 and T2, i.e. at an offset that is a multiple of 8.
The files must not be changed while a (Q) runs on them.  `qmap 0` reads
T3 (lambda3) into the work array of every rank as before; a run directory on
a network file system (NFS, Lustre, GPFS, BeeGFS, ...) gets a warning that
says so - mapping works there, but pages are then read over the network when
first touched or after the system has dropped them.

**Files beyond 2 GB.**  ifort writes a record of more than 2^31 - 9 bytes
as a chain of sub-records, each between two markers of 4 bytes (the leading
marker negative if another sub-record follows, the trailing one if one came
before); `moinpppp` of 160 virtual spinors is 2 654 310 416 bytes in two
sub-records of 2 147 483 639 and 506 826 761 bytes.  The native reader takes
a file with one Fortran READ, which follows the chain; only its checks of the
lengths count the markers (`xzrecbytes`).  The Python reader and writer
(`zccsdt.cfour._record`, `_write_record`, `record_bytes`) read and write the
same chain.  Validated on files of CFOUR's own `xx2cvtran` (section 7).
`xzccsdt mp2` reads the integrals and prints their second-order energy - the
number that `xx2cvtran` prints as "mp2 energy" - without the memory of a CC
calculation: a check of the files for cases too large to iterate.

What stays CFOUR's business: `xx2cvtran` with `ABCDTYPE=STANDARD` transforms
the whole (pq|rs) in core, n^4 complex numbers for n spinors (196 spinors:
25.8 GB resident, measured); 300 virtual spinors would be about 150 GB.

## 3. Sources

| file | what |
|---|---|
| `xzccsdt.f` | main program in CFOUR's style: MPI start, CRAPSI on rank 0, the work array `ICORE(I0)`, `XZDRV`, CRAPSO |
| `xzdrv.F90` | driver: job (`xzjob`), labels, the mode loop (labels x real), `xzsetmode` (classes, groups, block sizes), `xzlayout` (tables and integrals in the work array), energies, TOTENERG |
| `xzread.F90` | readers of CFOUR's files and of the labels (`xzrdlab`: `xzlabels`; `xzrddgc`: `dgcosym`); checks of the labels and of the imaginary parts |
| `xzmaps.F90` | index tables of pairs and triples (grouped by classes), layout of T3 in blocks |
| `xzmem.F90` | the stack on the work array, dry runs |
| `xzplan.F90` | the memory plan |
| `xzdata.F90` | sizes, the job, offsets of the arrays that live for the run |
| `xzdiis.F90` | DIIS: slots, the file, the small linear problem (ZHEEV) |
| `xzpar.F90` | ranks and the task list of (Q) (groups, pieces, the distribution by the cost model), checkpoint files |
| `xzdist.F90` | the iterations and the sigma vectors of EOM over ranks: the batches of pairs and triples and their ranks (cost model), the slices of T3, the times of the ranks |
| `xzkern.F90`, `xzkern.inc` | explicit `bind(C)` interfaces of the kernels, one include for `_d` and `_z` |
| `xzmpi.c`, `xzutil.c` | C: all MPI calls of the program behind a few functions (stubs without MPI) - broadcasts and sums of a few numbers, and arrays of the size of T3 in chunks: broadcast, send / receive, sum onto rank 0, sum on all ranks, the two exchanges of slices; free disk space, resident set (anonymous and mapped), BLAS integer size, a file mapped read-only (`xz_map_file`: T3 of (Q)), whether a directory is on a network file system |
| `xztype.h` | the element type of a compilation of the type-generic sources |
| `xzimd.inc` | T1-dressed integrals, W and F intermediates, the CCSD part, the occupied operator |
| `xzstep.inc` | one CCSDT step: the order of the stages, the placement of every array |
| `xzblk.inc` | the T3 passes in blocks of the labels |
| `xzdiisg.inc`, `xzcc.inc` | DIIS update, the iterations, `XZAMPS` |
| `xzlam.inc`, `xzlblk.inc` | the Lambda equations: intermediates, an iteration, the residuals of rank 1 and 2, `XZLAMB`; the passes over l3 in blocks of the labels |
| `xzden.inc` | the density stage: the passes over l3 and T3, the blocks of the two-particle density, the orbital gradient, the files of `xx2czvec`; its pass C in blocks |
| `xzdfrz.inc`, `xzfrz.F90` | frozen and dropped spinors in the density stage: the contractions with integrals of one such index, the rotations, the rows and columns of `xfull`; the integrals through CFOUR's `int_mintv2e` and `vtranpsog` |
| `xzq.inc`, `xzqs.inc` | (Q) and (Q)_Lambda: the task loop and the dense quadruple; the quadruple in sectors of the labels |
| `xzeom.inc`, `xzeoms.inc`, `xzeomd.F90` | EOM-EE-CCSDT: the intermediates of the ground state and the sigma vector; the solver, the driver, the test hooks and the vector operations over ranks; the input (flags, ZMAT sections), the label shifts and their layout, the slices and commands of the ranks, the results (section 16) |
| `xzcc3.inc` | CC3 (section 17): the step without stored triples, the batches of occupied triples and their fold, the diagonal Fock operator of T3, the folded EOM sigma vector and its driver |
| `xzcc4.inc` | CC4 (section 19): the T4 terms of the doubles and triples equations per occupied quadruple (dense and in sectors, with label shifts), the extrapolation over the macro iterations, the T4 terms of a sigma vector and the folded EOM driver, the files of a later run (`XZCC4`, `XZE4R`, `XZE4Q`, `XZEOMR`) |
| `xzdead.F90`, `xzdea.inc`, `xzdeas.inc`, `xzdeam.inc` | EOM-DEA-CCSDT (section 20): the input, the blocks of the sector and the tables, the slices of the ranks, the results; the intermediates of the ground state (Xv, the forms of T3); the sigma vector, the preconditioner, the masks of the parity; the driver |
| `xzdea5d.F90`, `xzdea5.inc`, `xzdea5s.inc`, `xzdea5r.inc`, `xzdea5x.inc` | EOM-DEA-CCSDT with complete 5p3h (section 21): the layout of the 5p3h blocks and of the intermediates, the rows of r5 over the ranks; the intermediates of the ground state and the kernels for rank 5; the 5p3h part of a sigma vector (the small intermediates, Z4, the pp ladder); the ring and the hh ladder; the rows of sigma5, Xv5 and E1x |
| `xzcc_z.F90`, `xzcc_d.F90` | the two instances of the `.inc` files |

Kernels of `csrc/` that are called: `zc_t3_pass1r`, `zc_t3_pass2r` (the
passes for a range of occupied pairs - four kernels added to `ccsdt_t3.c`
for the ranks, with `zc_l3_pass1r`, `zc_l3_pass2r`: the loops of
`zc_t3_pass1w` etc. from `plo` to `phi`; the scheduler calls them batch by
batch), `zc_t3_pass1w`, `zc_t3_pass2w` (the density stage),
`zc_t3_occop`, `zc_t3_update`, `zc_gemm` (ccsdt_t3.c); `zc_gather_vir_half`,
`zc_reduce_pair_first_rows`, `zc_pair_antisym_batch_add`,
`zc_pair_t1_contract`, `zc_add_transposed`, `zc_vvvv_dress`,
`zc_pair_one_body_add`, `zc_vsub`, `zc_vdot`, `zc_vmdot`, `zc_vlincomb`,
`zc_vzero_d`, and in blocks `zc_bgather2`, `zc_breduce2`, `zc_bgather_vir`,
`zc_breduce_ppl_rows` (t3pack.c); `zc_tn_expand`, `zc_tn_reduce`, `zc_tn_rows`,
`zc_tn_energy2`, `zc_tn_tuple_sums`, `zc_tn_zero`, and in sectors
`zc_q_sector_energy`, `zc_rows_gemm`, `zc_xv_sector` (tnpack.c); for lambda
`zc_l3_pass1w`, `zc_l3_pass2w` (ccsdt_t3.c), `zc_lam_pair_expand`,
`zc_lam_pair_matvec` and their `_tab` forms (lambda_t3.c), `zc_ql_energy`,
`zc_ql_sector_energy` (qlambda.c); for the EOM solver `zc_vtransform`,
`zc_vprecond` (t3pack.c; interfaces added, the kernels are those of the
Python solver).  Linked from CFOUR besides `liblibr.a`: the objects of
section 1, and for the EOM solver in complex arithmetic `eomreorder.o`,
`eomconv.o` of its program directory `x2ceomcc` (section 16.3).

**One change to a kernel file:** `zc_t3_pass1` and `zc_t3_pass2` of
`csrc/ccsdt_t3.c` allocated their batch buffers themselves.  The bodies are
now `zc_t3_pass1w` / `zc_t3_pass2w`, which take the buffers from the caller;
the old entry points allocate and call them (interfaces and behaviour
unchanged; the Python program runs on them).  **Two kernels added** to the
same file for the Lambda equations, `zc_l3_pass1w` and `zc_l3_pass2w`: the
two passes over the packed l3 that `lambda_opt.DenseL3` does with gathers and
GEMMs from Python, with the buffers of the caller (the Python program does
not call them).  The other kernels work on
arrays of the caller; what they allocate themselves is small (tables of
binomials, a line per thread).

### Real and complex: one source

The kernels are one C source instantiated for `double` and `double complex`
(`_d`, `_z`).  The Fortran that touches array elements or calls a kernel is
one source as well: the `.inc` files are compiled twice, by `xzcc_z.F90` and
`xzcc_d.F90` (the Lambda equations and (Q)_Lambda as well), with `xztype.h` giving the element type (`ZT`), the words per
element (`NW`), the suffix (`N(name)`) and the conjugation (`CONJ`).  Nothing
is written twice.  Everything outside the `.inc` files knows only the number
of words per element (`nwe`) and dispatches on it in six places (`xzdrv`:
iterations, XZAMPS, the Lambda equations, XZLAMB, (Q); `xzplan`: the dry runs;
and the density stage in both).

Typed Fortran, i.e. the places that are instantiated per type rather than
type-agnostic: all of `xzimd.inc` (loops over elements for the small
contractions that the Python scheduler does with numpy: dressed Fock matrix
and small blocks, the CCSD part, the occupied operator, operands of the
passes), the copies and accumulations of `xzblk.inc` and `xzq.inc`, the small
DIIS parts of `xzdiisg.inc`, and - because they pass typed constants to GEMM
and call suffixed kernels - the scheduling routines `xzstep.inc`,
`xzblk.inc`, `xzcc.inc`, `xzq.inc` themselves.  **Not done:** a scheduler
compiled once that reaches all arithmetic through typed C helpers or one
dispatch routine; today the scheduler is one source but two object codes.

Which instance runs is decided at run time from the integrals: if no
integral read has an imaginary part above 1e-10 the real one (the integral
files are complex; the real parts are compacted in place while reading).
`real 0` forces the complex instance; the test of section 7 runs a real
case through both.

## 4. Memory: the work array, the stack, the plan

`CRAPSI` returns one array of `MAXCOR` double words.  Every array of the
program is a region of it (`xzmem.F90`): `xzget(n, name)` takes the next n
words from a stack, on a 64-byte boundary; a stage marks the stack when it
starts (`xzmark`) and releases to the mark when it ends (`xzrel`).  A complex
number is two words.  There is no `allocate` and no `malloc` of anything that
grows with the problem, except: the labels and classes of the spinors (5
integer arrays of the number of spinors, needed before the plan), the task
list of (Q) (a few numbers per task), and what the kernels allocate for
tables (above).  MKL and the OpenMP runtime have their own buffers outside
the work array, as for every CFOUR program (measured resident sets below).

**Layout** (bottom to top): index tables (pairs, triples, signs; the layout
of T3 in blocks), spinor energies and their sums over triples, the integrals
`<oo||oo>`, `<oo||ov>`, `<oo||vv>` (full, on a<b, on i<j and a<b), `<ov||vo>`,
`<ov||vv>` on a<b, `<vv||vv>` on a<b, c<d (its diagonal pair blocks); then
T1, T2 (full), T3 (packed; its blocks), new T1 and T2, the DIIS vectors; then
the arrays of a step, which are taken and released stage by stage
(`xzstep.inc`):

1. everything of the size of T2 or smaller (dressed small blocks, F, W_ovvo,
   W_vooo, the occupied operator C(o,3)^2, operands of the passes);
2. the dressed `<ov||vv>` on a<b (D), the pass-1 result X, the pass-1 operand
   and the batch of T3 (released after pass 1);
3. W_vvvo, with the batch arrays of its terms; then W_vvvo is moved down
   over D and X, which are no longer needed;
4. the sweep over slabs of rows of the dressed `<ab||cd>`: the slab, the
   pp-ladder batch of T3 and its product;
5. the right-hand side of pass 2, made from W_vvvo above it and moved down
   over it; the two batch arrays of pass 2.

`moinpppp` is read into the place where the packed block stays: the record
of the file (v(v+1)/2 squared complex numbers) is compacted to the strict
pairs in place, rows and columns are permuted to the order of the program
(cycles of the permutation, one row of work), the diagonal blocks of the
labels are moved together, and real parts are compacted.  The transient is
the size of the file, not twice the packed block.

**The plan** (`xzplan.F90`) is made before anything is placed, and again for
the mode that the integrals turn out to have.  It is not computed from
formulae: the scheduler itself is run *dry* - through every `xzget` of the
layout, of the iterations and of (Q), with nothing touched - and the peak of
the stack is what the batch sizes take.  The plan and the placement therefore
cannot disagree; the output ends with the words actually used, which equal
the plan (checked in the scans below: `work array used` = `CCSDT with this
plan` to the word).

- The floor: every batch at its smallest (one occupied pair per batch of the
  passes, one occupied triple per pp-ladder batch, a slab of v rows of the
  dressed `<ab||cd>`, one first index per batch of the W_vvvo terms; (Q): Xv
  for three occupied indices, products in 16 batches of columns), the T3
  parts of the DIIS vectors in the file `XZDIIS`.  If the floor does not fit
  the program stops before reading the large files:

      @XZCCSDT-F, the work array is too short for this calculation.
        words needed:          2022168   words there:          2000000
        Set MEMORY_SIZE to at least          2022232 words (        16.2 MB).

- Above the floor the batches are raised to their targets one after the
  other, each as far as it fits before the next starts (so that no batch
  shrinks when the memory grows): rows of a slab (up to all pairs), triples
  of a pp-ladder batch (up to 1.3 GB of buffers), pairs of a batch of the
  passes (up to 2 GB: the defaults of the Python program), first indices of
  a W_vvvo batch (up to v/8).
- The DIIS vectors (8, each with its error vector): **memory first, all or
  nothing** (the user's decision, 2026-10-09: no split between memory and
  file).  They are in the work array if all of them fit next to the batches
  at the sizes just chosen - the batches come first, because small batches
  slow every pass.  Otherwise the T3 parts of all of them are records of the
  direct-access file `XZDIIS` and are passed over in chunks (2^20 elements);
  the same place holds for the amplitudes and for lambda.  The plan says
  which (`DIIS:   8 vectors in the work array` or `DIIS:   8 vectors, T3
  parts in the file XZDIIS (... MB)`); what the file costs when all cores
  are busy is measured in section 14.  Further:
  if the disk of the run directory does not hold all of them (free space is
  asked from the system before the first iteration), fewer vectors are used -
  in the file or in the work array, whichever holds more - and with fewer
  than two the iterations run without DIIS, with a warning.  A write error
  later stops the run with a message that names the file and its size.
- (Q) has the work array to itself after the iterations: Xv for more
  occupied indices first, then the two products in fewer batches of columns,
  then more rows per step of the formation of Xv.  T3 (and lambda3) are not
  part of it: (Q) reads them where the files `XZAMPS` (`XZLAMB`) are mapped
  (section 2), and the plan prints both numbers - the words private to a
  rank, and the mapped files that a node holds once:

        (Q) with this plan:                   1446046720  =    11568.4 MB
          (private to a rank.  Not in the work array: T3, read where the file XZAMPS is mapped -
           one copy in the page cache of a node for all its ranks:     1591.0 MB)

  With `qmap 0` T3 is in the work array of every rank and the line says
  `(private to a rank, T3 included: qmap 0)`.  A run that iterates first
  (`ccsdt(q)`) keeps T3 in the work array for the iterations, writes
  `XZAMPS`, releases T3 and maps the file like the other ranks.
- The excited states (`eom`, `ccsdt_eom`): the dry run goes through the
  allocations of every label shift of the input with the smallest subspace
  that can run; the subspace of the solver then takes what is left, per
  shift - all of it in the work array, a smaller one in the work array, or
  files (section 16.4).  With more than one rank the same dry run fixes the
  subspace of every shift (vectors, work array or files) and the words that
  a rank other than 0 needs for its rows of it; the run follows that plan
  (section 16.7).
- **The work array of a rank other than 0 is not `MEMORY_SIZE`.**  Rank 0
  makes the plan - the batch sizes, the place of the DIIS vectors - in its
  `MEMORY_SIZE`, then runs the scheduler dry once more *as a rank other
  than 0* (`asworker` of `xzdist.F90`: the same allocations without the DIIS
  vectors, and with the largest number of threads that a rank has, because
  the passes in blocks have buffers per thread) and sends the peak of that
  run - or what (Q) takes, if that is more - with the batch sizes.  The
  other ranks allocate that many words (plus 4096) and use the batch sizes
  as they are; a rank whose work array is too short for the plan of rank 0
  stops with both numbers.  What such a rank holds in the iterations: the
  tables and the integrals (as rank 0: it reads them itself), T1, T2, T3,
  one region of the size of T3 for its part of the residual (for the Lambda
  equations T3, lambda3 and the residual: three), and the arrays of a step
  at the batch sizes of the plan.  The plan prints it, and - node by node,
  from the names of the nodes and `MemAvailable` of each at the start - the
  work arrays of the ranks of a node against what the node had, with a
  warning if they are more:

        a rank other than 0 takes:            1943019168  =    15544.2 MB
          (its own work array: the integrals, the amplitudes, its part of the residual, the batches; no DIIS vectors)
        the work arrays of the ranks, node by node (MB; rank 0: its plan):
          node of rank     0:    2 ranks, work arrays      72127.4   memory available at the start     171230.4

  (26 + 120 dense real, two ranks on one node, `slices 0`.)

  **With several ranks on one node `MEMORY_SIZE` is rank 0's alone**: it
  must leave room for the other ranks of the node (the line above says how
  much), otherwise rank 0 plans the DIIS vectors into memory that the node
  does not have.  With slices (the default with more than one rank, section
  15) every rank holds its slice of every DIIS vector - in its work array,
  or in its own file `XZDIIS.<rank>` - and takes what rank 0 takes; the
  description above is that of `slices 0`.
- With more than one rank the batches of pairs and of triples are what is
  dealt to the ranks, so their targets are capped: at least 8 batches of
  each kind per rank (`bpr`).  One rank: the targets of before.

**The smallest MEMORY_SIZE** for o occupied and v virtual spinors, complex,
dense (one class): with p = v(v-1)/2, f = o v p (the size of `<ov||vv>` on
pairs) and t = C(o,3) C(v,3) (T3), the floor is about

    2 [ p^2 + f + 2 t + 4 f + 9 (o v)^2 + C(o,3)^2 ]  words

(`<vv||vv>`, `<ov||vv>`, T3 and its residual, the largest stage of a step -
pass 1: D, X, the operand and a batch of one pair, each about f -, the arrays
of the size of T2, the occupied operator); (Q) needs about
`2 [ p^2 + f + t + C(o,3) o p + 3 v C(v,3) + 6 o C(v,3) + 3 C(v,4) ]`.  The
exact numbers are printed by every run ("the least that CCSDT takes"):

| case | o, v | least for CCSDT | least for (Q) | default plan, CCSDT | T3 |
|---|---|---|---|---|---|
| H2O / 6-31G | 10, 16 | 16.2 MB | 8.0 MB | 56.7 MB | 1.1 MB |
| HBr / cc-pVDZ-DK, 28 frozen | 8, 28 | 38.2 MB | 24.5 MB | 147.8 MB | 2.9 MB |
| H2O / cc-pVDZ | 10, 38 | 144.3 MB | 89.6 MB | 725.6 MB | 16.2 MB |
| HBr / cc-pVTZ-DK, 28 frozen | 8, 78 | 596 MB | 810 MB | 2811 MB | 68 MB |
| H2O / cc-pVTZ | 10, 106 | 2123 MB | 2879 MB | 9124 MB | 370 MB |

Real arithmetic halves these; blocks of labels reduce T3 and `<vv||vv>` to
their allowed parts (section 6).  The (Q) columns of the table are those of
the private copy (`qmap 0`); with the map they are smaller by T3.  **Not
done:** T3 of the *iterations* in a mapped scratch file - T3, its residual
and (in the file-DIIS mode) nothing else of the size of T3 must fit in the
work array; the map of section 2 is read-only and serves (Q) alone.

Options of the plan and test hooks (lines of the section `%xzccsdt` of the ZMAT, section 18 -
where this document names an option, e.g. `labels 0`, it is such a line): `maxcor n` uses only n words of the work array;
`plan nqs nbt nbp nba qxv qc22 qc13 qrows` gives the batch sizes
(0: as planned), `diis m8|f8|m0` the place (memory / file) and number of
the DIIS vectors (read together with `plan`: `plan 0 0 0 0 0 0 0 0` and
`diis f8` is the planned run with the vectors in the file).  `qmap 0`:
(Q) with a private T3 (above).  `ncyc`, `start`, `hh`: section 12;
`qtasks`, `qranks`, `qsample`, `qstop`, `qcost`: section 5.
The iterations over ranks (section 15): `mpichunk n` (words of a
message), `mpitest n` (a test of the chunked transfers on arrays of n
words), `dist digits` (the kinds of batches that are dealt out: 1, 2, 3
pass 1, pass 2, pp ladder; 4, 5, 6 those of the Lambda equations; `0`:
rank 0 does all), `bpr`, `bcost file` (the seconds of every batch),
`slices 0`, `rebal`, `r0load`, `wfail`.  `maxcor` is read by
rank 0 only.  The excited states: `eomtest`, `eomsig`, `eomsd`,
`eomvec` (section 16.8).

Measured (H2O / 6-31G, login node): the same energies to 1e-12 with the work
array at its floor (2022176 words: DIIS in the file, one pair per batch),
at 2.3, 3 and 5 million words, and with explicit small batches (slabs of 16
and 17 rows of 120, one triple per pp batch, Xv for 3 to 6 occupied indices
of 9, products in 16 batches of columns); without DIIS and with 2 or 3
vectors the iterations converge to the same energy in more cycles (30, 27,
22 instead of 17).

## 5. (Q): tasks, ranks, restart

(Q) is the algorithm of `quadruples.py`: for one occupied quadruple at a
time, r and z over all virtual quadruples from five stacked matrix products,
each reduced onto a<b<c<d by the kernels of `tnpack.c`; the intermediate Xv
of T2 with `<vv||vv>` is held for the pairs (x, l) of one largest index l.

The unit of work is a **task** (`xzpar.F90`): a *group* - the quadruples
i<j<k<l of one l, which share the Xv(x, l) of that l - or a *piece* of a
group: those of one l whose smallest index i lies in a range.  A task needs
Xv(x, l) only for x >= its smallest i; it reads T2, T3 and the integrals and
returns two numbers (its parts of E[Q] and of E(Q) - E[Q]).  The serial
program is the one-rank case of the same code: the list, a loop that does
the tasks of this rank, a sum.  Consecutive tasks of one l on a rank keep
the Xv they share.

**Ranks: whole groups, a static partition.**  The groups go whole to the
ranks - the largest task first, each to the rank with the least cost so far
- so that no Xv is formed twice.  The groups are of very different size (l
quadruples C(l, 3): for 26 occupied spinors the last group is 15 % of all),
so with many ranks whole groups cannot be balanced; then, and only as far as
needed, groups are cut: while the slowest rank of the cost model is more
than 3 % above the mean, the group of the largest task gets one piece more
(pieces of equal cost, ranges of i), and the list is distributed again.
Pieces of one group on different ranks each form the Xv they need: that
repeat is accepted - passing Xv on between ranks was not tried, because a
repeat is small: Xv(x, l) for all x of the largest group costs about as much
as a dozen of its 2300 quadruples (measured: section 14).  The output says what
the list means:

       tasks:     33   ranks:    12   tasks of this rank:      3   rank     0
       groups (one largest index l, one Xv):     23;  Xv(x, l):    322 pairs,    500 formed over all ranks
       cost model: the slowest rank   2.50 % above the mean; the repeated Xv   0.25 % of the work

For 26 occupied spinors (the model in blocks, 26 + 78; `qranks "2 3 4 6 8
12 16 24 48"` prints these lines for any numbers of ranks without running
them):

| ranks | tasks | Xv formed (of 322 pairs) | slowest rank above the mean | repeated Xv, of the work |
|---|---|---|---|---|
| 2, 3, 4 | 23 | 322 | 0.00, 0.12, 0.69 % | 0 |
| 6 | 23 | 322 | 2.19 % | 0 |
| 8 | 34 | 462 | 1.64 % | 0.19 % |
| 12 | 33 | 500 | 2.50 % | 0.25 % |
| 16 | 41 | 633 | 2.93 % | 0.45 % |
| 24 | 100 | 1352 | 2.72 % | 1.61 % |
| 48 | 166 | 2178 | 2.95 % | 2.88 % |

Every rank builds the same list (it depends on the number of ranks and on
the sizes of the classes, not on anything a rank computes) and does its own
tasks in the order of l; one `MPI_Reduce` adds the energies.  Chosen over
pulling tasks from rank 0 because no rank then serves another while it
computes with all its threads, the list is fixed (which is what restart
needs), and nothing is communicated before the end.  The price: the balance
is that of the cost model.  The output ends with a line per rank - tasks,
quadruples, seconds, the seconds of Xv and Xo, its own memory and the mapped
files - and "sum over ranks / slowest", the speedup that the partition
allowed.

**The cost model** (`xzqmodel` of `xzdrv.F90`, `xzq_costs`), in units of one
virtual quadruple: an occupied quadruple of total class M costs the number
of virtual quadruples a<b<c<d of total class M (their sector; without labels
all of them, so every quadruple costs the same); Xv(x, l) costs 0.0012 times
the elements of its products (`<bc||ef>` on (bc), e times T2 on f, d, in the
blocks of the classes of x and l).  It was measured, not derived: section 14
(per group within 5 % on 6 threads and 13 % on 24, where one cost per
quadruple is off by up to 21 %).
`qcost file` writes the times of every quadruple and of every Xv of rank
0 to a file, which is what the model was fitted to.

**The left side** of the correction is an argument of the routine of one
quadruple (`xzquad`, `xzqsquad`): for (Q), z from T2 with the conjugated
integrals, and E(Q) adds the [V,T3] vector; for (Q)_Lambda, the left vectors
from lambda2 and lambda3 with their own products (section 11).  The task
layer, the reader, the reduction, restart, progress and sampling do not
depend on what is on the left; the checkpoint files carry its kind, so a
checkpoint of (Q) is not taken for (Q)_Lambda.

**Memory per rank, and what a node holds once.**  Each rank has its own work
array (section 4: the words that (Q) takes with the plan of rank 0) for what
is private: the integrals (`<vv||vv>` on pairs, `<ov||vv>` on pairs, the
small blocks), T1, T2, Xo, Xv for the occupied indices of the plan, and the
work arrays of a quadruple.  **T3 (and lambda3 for (Q)_Lambda) are not in
it**: the ranks read them where `XZAMPS` (`XZLAMB`) are mapped, and the page
cache of the node holds the file once for all of them.  The (Q) routines
only read T3 (the map is read-only: a write would stop the program), so
nothing else changes - the address of the map enters as the index `jt3`
(`jl3`) of the work array's address space, as CFOUR's `ICORE(I0)` does for
the work array itself.  Measured sizes and the resident memory of the ranks:
section 14.  What a rank still holds of the size of T3 or larger is Xo
(C(o,3) o v(v-1)/2 numbers, dense in every mode: 6.9 GB for 26 + 160 real),
formed by every rank - the next thing to share (not done).

**Restart, progress, sampling.**  Every rank appends a line per finished
task to `XZQCHK.<rank>`.  A later run with the same task list (the same
number of ranks, or the same `qtasks`) takes the finished tasks of all
such files as done and continues - the first line of a file carries a number
computed from the list, so the file of another list is not taken (with
another number of ranks the calculation starts again) -; all ranks read the
files before any of them writes one; the files are deleted when (Q) has
finished.  After every task a rank prints the task, its time, the per cent
of its own work done and the time left by the cost model.  `qsample f`
(0 < f < 1) does the fraction f of the quadruples of every task, evenly
spread over the task (at least one), with Xv formed as in the whole
calculation, and prints the estimated wall time of the whole (Q) (per task:
the time of Xv plus the time of its quadruples in proportion; the slowest
rank); no energy is printed and TOTENERG is not written.  (Before 2026-10-09
a sample took the *first* quadruples of a task, whose sectors are not the
average ones, and formed Xv again for every task: its estimates are not
comparable - section 14.)  `qtasks n` asks for about n tasks of equal
numbers of quadruples instead of the rule above, `qstop n` ends the run
after n tasks of a rank (test hooks of the list and of restart).

## 6. Symmetry labels and blocks

The interface is one integer label per spinor of the SCF, in the order of
`scf2ceig` (frozen, active occupied, active virtual, dropped).  A label
adds: `<pq||rs>` vanishes unless l_p + l_q = l_r + l_s (modulo a modulus if
that is not 0) - 2 m_j of a linear molecule with modulus 0.

**Where the labels of a run come from** (the same rule in the native
program, `xzdrv.F90`, and in the Python reader, `zccsdt.cfour.read_labels`):

1. the file **`xzlabels`** of the run directory, if it is there;
2. else the file **`dgcosym`** of CFOUR's SCF, if it has more than one
   block: label +(2k - 1) for block 2k - 1, -(2k - 1) for block 2k, modulus
   0 (`xzrddgc` of `xzread.F90`, `zccsdt.cfour.read_dgcosym`);
3. else no labels.

Nothing is parsed from the ZMAT.  The output says which it was:

    symmetry labels: the file xzlabels, modulus    0
    symmetry labels: the file dgcosym of CFOUR's SCF (m_j blocks), block numbers up to 6: labels 2 m_j, |2 m_j| up to 5, modulus 0
    symmetry labels: none (no file xzlabels; dgcosym: one block, an SCF without %dglin)
    symmetry labels: none (no file xzlabels, no file dgcosym)

`labels 0` switches the labels of either file off.  The labels of either
file are checked against the integrals (below).  In an MPI run every rank
reads the two files itself in its run directory; the ranks must find the
same (section 1: a rank that decides otherwise than rank 0 stops the run).

**`dgcosym` and `%dglin`.**  CFOUR's two-component SCF `xsox2chf` does the
SCF in blocks of m_j when the ZMAT has a section `%dglin` after the basis
lines: the number of |m_j| values, then for |m_j| = 1/2, 3/2, ... a line
with the numbers of occupied spinors of +m_j and of -m_j.  HBr (36
electrons; sigma and pi shells give 12 + 12 spinors of |m_j| = 1/2, 5 + 5 of
3/2, 1 + 1 of 5/2):

    hbr X2C cc-pvdz-dk
    BR 0.000000000000 0.000000000000 0.000000000000
    H 0.000000000000 0.000000000000 2.672072740135

    *CFOUR(CALC=CC(n-1)(n),EXCITATION=4,BASIS=SPECIAL
    SYM=OFF,COORDINATES=CARTESIAN,UNITS=BOHR,CHARGE=0,MULT=1
    RELATIVISTIC=X2C1E,X2CKEY=SOATMX,ABCDTYPE=STANDARD
    SCF_CONV=10,SCF_MAXCYC=200
    CC_CONV=9,CC_MAXCYC=100
    MEMORY_SIZE=4,MEM_UNIT=GB)

    BR:ZCCCPVDZDK
    H:ZCCCPVDZDK

    %dglin
    3
    12 12
    5 5
    1 1

(the molecule on the z axis; the occupations are the user's statement of the
state - the program checks the labels, not the state).  The SCF then writes
the block of every spinor to `dgcosym`: one Fortran unformatted sequential
record of `nmototal` 8-byte integers in the order of `scf2ceig`, block
2k - 1 for m_j = +(2k - 1)/2 and block 2k for m_j = -(2k - 1)/2 (the
numbering of `linsym` of sox2chf); without `%dglin` the file is there with
all numbers 1.  With `%dglin` the MO integrals of `xx2cvtran` are also real
(largest imaginary part 2e-11 for the input above, 1.5e-12 with 18 frozen
spinors) and 17 % of them are nonzero, against 0.16 and 75 % for the same
molecule without the section: the program runs in real blocks instead of
complex dense, with no file of ours in the run directory.  A record of
another length than `nmototal` integers or a block number below 1 stops the
program:

    @XZCCSDT-F, the file dgcosym has          504 bytes of data; one 8-byte integer for each of the       64 spinors of the SCF
                 is expected (nmototal),          512 bytes.  (The file xzlabels, if it is there, is used instead.)
    @XZCCSDT-F, the file dgcosym: a block number            0; the blocks of the SCF are numbered from 1.

A block number is not a label that adds; the numbering above is known in one
routine on each side (`xzrddgc`, `read_dgcosym`, which also takes a table
for another numbering).  Frozen and dropped spinors need nothing: the file
has all spinors of the SCF in the order of `scf2ceig`, and the program takes
the labels of the active ones.

Measured on that input (HBr / cc-pVDZ-DK, 36 + 28 spinors all-electron;
login node, 4 threads, `xzccsdt ccsdt`; 6 classes, T3 15.25 % and `<ab||cd>`
17.98 % of the dense arrays; the times are of a shared login node and only
say the order of magnitude):

| run directory | labels from | arithmetic, classes | E(CCSDT) | cycles | s per cycle |
|---|---|---|---|---|---|
| `%dglin`, all-electron (36 + 28) | `dgcosym` | real, 6 | -2605.010449740676 | 19 | 0.61 |
| | hand-written `xzlabels` (2 m_j) | real, 6 | -2605.010449740676 | 19 | 0.63 |
| | `dgcosym`, `labels 0` | real, 1 | -2605.010449740676 | 19 | 5.70 |
| | `dgcosym`, `real 0` | complex, 6 | -2605.010449740676 | 19 | 1.37 |
| | `dgcosym`, 2 ranks x 2 threads | real, 6 | -2605.010449740676 | 19 | 0.75 |
| | `dgcosym` with two block numbers exchanged | refused (forbidden `<oo||oo>` 1.41), real, 1 | -2605.010449740676 | 19 | 5.78 |
| | `xzlabels` with modulus 4 | not used (warning), real, 1 | -2605.010449740676 | 19 | 5.74 |
| the same input, the chain run again | `dgcosym` | real, 6 / 1 | -2605.010449740678 | 19 | 0.63 / 5.74 |
| `%dglin`, `DROPMO=1>9` (18 + 28 active) | `dgcosym` | real, 6 | -2605.002712127468 | 18 | 0.06 |
| | `dgcosym`, `labels 0` | real, 1 | -2605.002712127468 | 18 | 0.38 |
| | `dgcosym`, 2 ranks x 2 threads | real, 6 | -2605.002712127468 | 18 | 0.07 |
| no `%dglin`, `DROPMO=1>9` | none (`dgcosym`: one block) | complex, 1 | -2605.002712127468 | 18 | 0.99 |
| `%dglin`, `DROPMO=1>9`, `%vircut` 1.5 (18 + 22 active, 6 dropped) | `dgcosym` | real, 6 / 1 | -2604.991484860860 | 18 | 0.04 / 0.18 |
| `%dglin`, `%vircut` 1.5 (36 + 22 active) | `dgcosym` | real, 6 / 1 | -2604.998290453414 | 18 | 0.39 / 2.62 |
| HBr+ doublet, `%dglin` 12 12 / 5 4 / 1 1, `DROPMO=1>9` (17 + 29 active) | `dgcosym` | real, 6 / 1 | -2604.591824417278 | 21 | 0.07 / 0.38 |

("6 / 1": the run with the labels and the run with `labels 0`, the same
energy to all digits printed.)  The all-electron case without `%dglin` is
complex and dense: 18.7 s per cycle against 0.60 s, the same E(CCSDT)
-2605.010449740676 (measured before this rule, with a hand-written
`xzlabels`; then also `ccsdt_density` and `xx2czvec` on the labelled real
spinors: relaxed electronic dipole 1.07498654471939 against 1.07498654470416
by the complex dense route).  The doublet is one case of unequal
occupations of +m_j and -m_j (the hole in m_j = -3/2): the SCF converges,
the integrals are real (3.3e-11) and the labels pass the check; nothing
more was looked at for open shells.

**`xzlabels`** is the way to give labels from outside (it goes before
`dgcosym`), a text file:

    # comment lines start with #
    <number of spinors> <modulus>
    <label> <label> ...          (free format, any number per line)

The Python side reads and writes the same file:
`zccsdt.cfour.write_labels`, `read_labels`; `cfour.write` adds it for a
container with labels, `from_cfour(labels=True)` gives the container the
labels (default: not, as before).  `bench/cfour/labelled_run.py` makes test
directories: m_j-adapted real spinors of a linear molecule from a PySCF SCF,
written in CFOUR's formats with their labels; optionally with a phase on
every spinor (complex integrals with the same labels).

What the program does with the labels:

- classes: (label - smallest) / step, step the greatest common divisor of
  the differences.  The occupied and the virtual spinors are put in the order
  of their classes (a permutation applied while the integral files are read;
  CFOUR's order inside a class), pairs and triples are numbered by the sum of
  the classes of their indices (their group) first.  A block is then a range.
- T3, its residual and the DIIS vectors are stored on the blocks of equal
  occupied and virtual group only; `<vv||vv>` on its diagonal pair blocks.
  The passes over T3 run per allowed combination of classes (`xzblk.inc`,
  the scheduler of `SpinBlockT3` of the Python program for any number of
  classes).  Everything else of a step runs on the dense arrays with the
  grouped tables.
- no labels (neither file, or one block in `dgcosym`) is one class: one
  block, the dense kernels, in the same step routine.  It is not a separate
  program.
- the labels are not trusted: while the integrals are read, the largest
  element that the labels forbid is taken over all blocks; above 1e-8 the
  program says so with the block and the element, drops the labels and reads
  the integrals again for one class.  The spinors of an SCF without the
  symmetry fail this check (the two partners of a Kramers pair are mixed):
  that is the SCF's business - CFOUR's does it with `%dglin` (above).  The
  warning names the file the labels came from:

      @XZCCSDT-W, the labels of xzlabels are not symmetry labels of these spinors:
                   an element of <oo||vv> that they forbid is   1.18E-01.  The labels are not used.

- blocks and real arithmetic are independent: complex integrals with labels
  run in complex blocks (tested with phases on the spinors).

- (Q) runs in the sectors of the labels (`xzqs.inc`, the scheduler of
  `quadruples_sym.py`): per occupied quadruple the five products in blocks
  by groups of pairs and classes of spinors, Xv in blocks per class, one
  kernel (`zc_q_sector_energy`) over the virtual quadruples of the sector.
  It reads T3 **from its blocks** - the rows that a product needs are rows
  of the block of one group (`zc_rows_gemm` with the block as its matrix;
  the operands on two pairs through the layout tables) - so no dense T3 is
  made, unlike the Python front end, which expands the blocks for (Q).  The
  tasks, ranks, restart and sampling of section 5 are the same code.

`labels 0`: the labels are not used.  **Not done:** labels with a modulus
- the file format of `xzlabels` has one, the readers pass it on (`modulus` of
`xzdata.F90`; `xzrddgc` returns one too, 0 for the m_j blocks), and the
driver decides in one place: with a modulus that is not 0 the program warns
(`@XZCCSDT-W, labels with a modulus are not used by this program: one
block.`) and runs dense, whichever file the labels came from.  What assumes
labels without a modulus is the block code itself: the classes are (label -
smallest) / step and the group of a pair or triple is the sum of the classes
(`xzclasses`, `xzsetmode`, `xzblk.inc`, `xzlblk.inc`, `xzqs.inc`).  Also not
done: the rotation of unadapted Kramers pairs into j_z eigenfunctions (for a
run of CFOUR's SCF without `%dglin`); other sections of the SCF (`%dg2lin`,
...) were not run - their block numbers would need their own table in
`xzrddgc`, and are refused by the check against the integrals if the m_j
numbering does not fit them.

## 7. Validation

The Python program on the same run directory and the same integral files
(`python -m zccsdt.cfour --method "ccsdt(q)" RUNDIR`) against the native one;
chain made by `python -m zccsdt.cfour --chain --no-cc`; `CC_CONV=9` (energy
1e-11, amplitudes 1e-9).  Total energies in Hartree; all runs on the login
node with 4 threads unless a job is named.

| case (o + v active spinors) | | Python | native | cycles |
|---|---|---|---|---|
| H2O / 6-31G, X2C-1e (10 + 16) | CCSDT | -76.169208460811 | -76.169208460811 | 17 / 17 |
| | CCSDT[Q] | -76.169657261819 | -76.169657261819 | |
| | CCSDT(Q) | -76.169665313369 | -76.169665313369 | |
| the same, DROPMO=1>1 (8 + 16) | CCSDT | -76.168292779221 | -76.168292779221 | 17 / 17 |
| | CCSDT(Q) | -76.168748671427 | -76.168748671427 | |
| the same, DROPMO=1>1, %vircut 1.2 (8 + 10; 6 dropped) | CCSDT | -76.099716291715 | -76.099716291715 | 17 / 17 |
| | CCSDT[Q] | -76.099714131733 | -76.099714131733 | |
| | CCSDT(Q) | -76.099725173896 | -76.099725173896 | |
| H2O / cc-pVDZ, X2C-1e (10 + 38) | CCSDT | -76.292039876888 | -76.292039876888 | 19 / 19 |
| | CCSDT(Q) | -76.292528074414 | -76.292528074414 | |
| HBr / cc-pVDZ-DK, X2C-1e, 28 frozen (8 + 28) | CCSDT | -2604.989297832629 | -2604.989297832629 | 18 / 18 |
| | CCSDT[Q] | -2604.989486010692 | -2604.989486010692 | |
| | CCSDT(Q) | -2604.989515284327 | -2604.989515284327 | |
| HBr / cc-pVDZ-DK, X2CAMF (`%amfiMethod`), 28 frozen | CCSDT | -2602.405452129587 | -2602.405452129587 | 17 / 17 |
| | CCSDT(Q) | -2602.405669692619 | -2602.405669692619 | |
| H2O / cc-pVTZ, X2C-1e (10 + 106), job 31778128 | CCSDT | -76.394890322516 | -76.394890322516 | 19 / 19 |
| | CCSDT[Q] | -76.395054901781 | -76.395054901781 | |
| | CCSDT(Q) | -76.395259065600 | -76.395259065600 | |
| HBr / cc-pVTZ-DK, X2C-1e, 28 frozen (8 + 78), job 31778128 | CCSDT | -2605.165534626440 | -2605.165534626440 | 19 / 19 |
| | CCSDT[Q] | -2605.165561661789 | -2605.165561661789 | |
| | CCSDT(Q) | -2605.165759197891 | -2605.165759197891 | |
| HBr / cc-pVQZ-DK, X2C-1e, 28 frozen (8 + 160), `moinpppp` 2.65 GB, job 31779060 | CCSDT | -2605.192252825424 | -2605.192252825424 | 19 / 19 |
| | CCSDT[Q] | -2605.192239841907 | -2605.192239841907 | |
| | CCSDT(Q) | -2605.192558400552 | -2605.192558400552 | |

**Files beyond 2 GB** (HBr / cc-pVQZ-DK, files of CFOUR's `xx2cvtran`, job
31779060, node c388): the row above, and with 10 frozen spinors (26 + 160:
too large to iterate here, T3 would be 27.9 GB) the second-order energy of
the files: `xx2cvtran` prints -0.394844126182089; the Python reader gives
-0.394844126182; `xzccsdt mp2` gives -0.394844126182 (3.2 s to read and
sort, 4.6 GB of the work array); the same files written again by the Python
writer (sub-records of 2147483639 and 506826761 bytes, as CFOUR's) and read
by the native program: -0.394844126182.  For 8 + 160 `xx2cvtran` prints
-0.191879981425920 and the native program -0.191879981426.

The first H2O case is that of `/home/xwang405/.claude/jobs/7ba4fcf7/tmp/cf_n4`
(the three reference energies given with the task).

During development: the first run of the CCSDT step agreed with the Python
energy; (Q) was off by 4.5e-8 at first, found at the term (the T2 T2 part on
one index and a triple, by its contribution to z.r per occupied quadruple
against `quadruples.Quadruples`) - a work region declared too short, which
overwrote the first Xv.  `qdebug 1` prints the four parts of the
correction (z.[V,T3], z.T2T2, [V,T3].[V,T3], [V,T3].T2T2).
`debug file` does one CCSDT step on amplitudes read from a file and writes
the result (for comparisons with `ccsdt_opt.update_amps` on dumped arrays).

**Labels, blocks, real** (HBr / cc-pVDZ-DK, 28 frozen; m_j-adapted spinors
written by `bench/cfour/labelled_run.py`; 6 classes, T3 17.3 % and
`<ab||cd>` 18.0 % of the dense arrays):

| integrals | program, mode | E(CCSDT) |
|---|---|---|
| real, labels | Python, blocks (`symmetry` True) | -2604.989297832537 |
| | Python, dense (labels not used) | -2604.989297832537 |
| | native, blocks, real | -2604.989297832537 |
| | native, dense, real (`labels 0`) | -2604.989297832537 |
| | native, blocks, complex (`real 0`) | -2604.989297832537 |
| | native, dense, complex (both) | -2604.989297832537 |
| complex (a phase on every spinor), labels | Python, blocks | -2604.989297832545 |
| | native, blocks (complex found) | -2604.989297832545 |
| | native, dense | -2604.989297832545 |
| CFOUR's own spinors with these labels | native | labels refused (forbidden `<oo||vv>` 0.118), dense: -2604.989297832629 |

The same files with the active spinors in a random order (`labelled_run.py
--shuffle`, seed 11: the classes are then not ranges in the files, and the
program's own ordering, with the signs of the pairs that change their order,
is what is tested): CCSDT, [Q] and (Q) as below in all four native modes.

The real case through the complex path (the override `real 0`) is the
third and fourth native row.  CCSDT(Q) on the same files: E(CCSDT[Q]) =
-2604.989486010600, E(CCSDT(Q)) = -2604.989515284235 from Python in blocks,
Python dense, and the native program in all four modes (blocks and dense,
real and complex), with Xv held for 3 of 7 occupied indices, and with 1 and 3
ranks; with the phases (complex blocks) -2604.989486010608 and
-2604.989515284243 from Python and native.  The same at cc-pVTZ-DK (8 + 78,
8 classes), CCSDT: -2605.165534626448 from all six runs of section 8.

**Tasks and MPI** (one node; the MPI build with `mpirun -np`):

| case | ranks | E(CCSDT[Q]) | E(CCSDT(Q)) |
|---|---|---|---|
| H2O / 6-31G | serial build | -76.169657261819 | -76.169665313369 |
| | 1, 2, 4 | -76.169657261819 | -76.169665313369 |
| HBr / cc-pVDZ-DK | 1, 2, 4 | -2604.989486010692 | -2604.989515284327 |

(the same to all printed digits for 7, 14, 15, 18, 20 and 25 tasks).  `(q)`
alone from `XZAMPS`, with 20 tasks asked for, interrupted by `qstop 5` and
started again (serial: 5 tasks restored; 4 ranks: 10 restored): the same
energies; the checkpoint files are gone afterwards.  `qsample 0.3`: 62 of
210 (H2O) and 22 of 70 (HBr) quadruples, the estimate printed, no TOTENERG.

## 8. Timings

All on 8 cores of one node (`sbatch -p parallel -c 8 --mem=30G`, Xeon Gold
6248R), the Python program first, both reading the same integral files of
the CFOUR chain in the node-local `/tmp`; `MEMORY_SIZE=22 GB` for both
(the Python front end takes it as `max_memory`); `CC_CONV=9`.  Wall seconds.

**Complex, dense** (job 31778128, node c388; the energies of both programs
agree to all digits, section 7):

| | H2O / cc-pVTZ (10 + 106) | | HBr / cc-pVTZ-DK, 28 frozen (8 + 78) | |
|---|---|---|---|---|
| | Python | native | Python | native |
| CCSDT, per iteration (19 each) | 15.26 | 13.80 | 2.28 | 1.83 |
| (Q) | 164.1 | 168.2 | 11.6 | 13.8 |
| whole run, wall | 458.3 | 430.9 | 57.2 | 48.7 |
| peak resident set | 10.3 GB | 9.09 GB | 3.48 GB | 2.96 GB |
| work array used (of 22 GB) | - | 9124 MB | - | 2811 MB |
| reading the integrals | | 0.45 | | 0.13 |

Where the native iteration differs (H2O, 19 iterations, seconds; Python's
own table in brackets): pass 2 101.2 [104.1], pp ladder 99.6 [100.7] (GEMM
95.6 [96.4]), pass 1 36.8 [41.0], W_vvvo 9.7 [13.2], T1 dressing and the
dressed `<ab||cd>` rows 6.7 [12.2], CCSD part 1.3 [3.5], DIIS 2.9 [6.1],
other 0.6 [2.7].  The GEMM-bound stages are equal within 1-3 %; the native
program is faster in what the Python scheduler does itself (dressing,
intermediates, DIIS, Python between the kernels): 1.5 s per iteration here,
0.45 s for HBr.  The native program keeps its 8 DIIS vectors in the work
array (5.9 GB of the 9.1 GB for H2O); the Python program has them in a
mapped scratch file.

(Q) is the same products and reductions in both.  Its time varied between
runs by more than the two programs differ: HBr on the same node type, three
jobs: Python 11.6, 15.3 s; native 13.8, 15.4, 12.7 s (jobs 31778128,
31778290, 31778313).  One real difference was found from the timers of both
(job 31778313: Python `q.gemm` 8.4, `q.reduce` 2.2, `q.xv` 2.1; native
products 6.9, reductions 1.9, Xv 3.4): the native Xv expanded the rows of
`<vv||vv>` once per occupied index instead of once per set of indices.  That
is fixed (commit b36d7c8; H2O / cc-pVDZ on the login node: Xv 0.34 -> 0.19
s); at H2O / cc-pVTZ, where the rows come in two batches and the GEMM
dominates, it changes little (Xv 28.6 s of a (Q) of 170.5 s in job 31778544,
30.5 s of 168.2 s before; another node).  HBr / cc-pVTZ-DK was **not timed
again** after the change.  Excluded as causes, on the same
node: the MKL version (the native program linked with MKL 2026 of the Python
environment: CCSDT 34.56 against 34.59 s, (Q) 15.33 against 15.35 s, job
31778290), the compiler of the kernels (gcc instead of icc, login node, H2O
/ cc-pVDZ: 0.40 against 0.41 s per iteration, (Q) 2.96 against 2.94 s),
`KMP_BLOCKTIME=0` (no gain).

**HBr / cc-pVQZ-DK, 28 frozen (8 + 160)**, complex, dense, 16 cores of one
node (`-c 16 --mem=60G`, job 31779060, node c388, the native program first;
`MEMORY_SIZE=45 GB`): CCSDT 18.97 s per iteration native, 22.10 s Python;
(Q) 165.5 s native, 171.1 s Python; peak resident set 21.2 GB native (the
work array used: 21.2 GB, of which the 8 DIIS vectors 9.6 GB; the least that
CCSDT takes is 5.7 GB, (Q) 12.1 GB), 21.9 GB Python.

**Labels and real arithmetic**: HBr / cc-pVTZ-DK, 28 frozen (8 + 78), the
m_j-adapted real spinors written by `bench/cfour/labelled_run.py` (8
classes; T3 14.6 %, `<ab||cd>` 14.3 % of the dense arrays), CCSDT only, 19
iterations each, E(CCSDT) = -2605.165534626448 in all six runs (job
31778290, node c640):

| program, mode | s per iteration | whole run | peak resident set | work array |
|---|---|---|---|---|
| Python, blocks (the files are complex: complex arithmetic) | 0.74 | 15.6 | 1.12 GB | - |
| Python, dense, complex | 2.34 | 46.1 | 3.46 GB | - |
| native, blocks, real | 0.16 | 3.3 | 0.33 GB | 236 MB |
| native, blocks, complex (`real 0`) | 0.35 | 6.9 | 0.60 GB | 463 MB |
| native, dense, real (`labels 0`) | 0.61 | 11.9 | 1.69 GB | 1589 MB |
| native, dense, complex (both) | 1.82 | 34.9 | 2.95 GB | 2811 MB |

In blocks the T3 passes are a third of an iteration (native real: pass 1
0.27, pp-ladder GEMM 0.09, pass 2 0.61 of 3.0 s for 19 iterations); the rest
is the dense part of the step.  The Python numbers of docs/benchmarks.md
("CCSDT and CCSDTQ in blocks of the labels") are for the Python program on
its own real integrals and were not rerun here; the Python rows above are on
the files, where the Python reader keeps complex arrays.  (Q) in sectors
and HBr / cc-pVQZ-DK were **not timed**.

**(Q) over MPI ranks**: H2O / cc-pVTZ, X2C-1e (10 + 106), complex, dense;
the MPI build, **all ranks on one node** (`sbatch -N 1 --ntasks-per-node=4
-c 8 --mem-per-cpu=3700M`, `mpirun -np n -ppn n`, 8 threads per rank,
`I_MPI_PIN_DOMAIN=omp`; job 31778544, node c518; `MEMORY_SIZE=12 GB` per
rank).  (Q) from `XZAMPS` for 2 and 4 ranks; the energies of all four runs
are those of section 7 to all digits.

| ranks x threads | tasks | (Q) wall | speedup | sum over ranks / slowest | Xv, all ranks | products, all ranks | resident set per rank |
|---|---|---|---|---|---|---|---|
| 1 x 8 | 7 | 170.5 | 1 | 1 | 28.6 | 117.4 | 9.1 GB (with CCSDT) |
| 2 x 8 | 18 | 101.0 | 1.69 | 1.98 | 48.2 | 124.5 | 6.6 GB |
| 4 x 8 | 25 | 64.1 | 2.66 | 3.64 | 71.8 | 130.6 | 6.6 GB |

Where it falls short of linear, at 4 ranks: (1) Xv is formed on every rank
that has a task of the pair (x, l) - 71.8 s over the ranks against 28.6 s
once, i.e. 43 s of extra work, 11 s per rank of 64; the tasks of one l that
are split by the smallest index share most of their x.  (2) The balance of
the static partition: the slowest rank takes 64.1 s, the mean is 58.3 s
(sum / slowest 3.64 of 4).  (3) The products take 11 % more in total than on
one rank - four ranks with 8 threads each on one node share its memory
bandwidth; this part is a property of the one-node run and would not be
there with one rank per node.  The same run with CCSDT on rank 0 and (Q) on
all four (`ccsdt(q)`, 4 ranks): CCSDT 13.91 s per iteration (13.90 s with
one rank: the three waiting ranks did not disturb it), (Q) 63.7 s, the whole
run 329 s against 436 s with one rank.

**Not measured:** one rank per node (`-N 4 --ntasks-per-node=1 -c 8
--mem-per-cpu=3700M`, the run directory on the shared scratch).  The first
job of that shape (31778288, nodes c105, c387, c388, c399) failed at the
launch - see below - and its repetitions were not started by the queue
within the session (expected start a day later); they were cancelled.

**Launching under Slurm** (what got in the way): Intel MPI's `mpirun` starts
its proxies through `srun`, and the shell of a batch job on a compute node of
this cluster has no Slurm commands in its `PATH` (`which srun` finds nothing
there; on the login node it is `/cm/shared/apps/slurm/current/bin/srun`).
`mpirun` then stops with

    [mpiexec@c105] check_exit_codes (...hydra_demux_poll.c:117): unable to run bstrap_proxy on c105 (pid 3561325, exit code 65280)
    [mpiexec@c105] HYD_print_bstrap_setup_error_message (...): error setting up the bootstrap proxies

for any number of ranks, one included (jobs 31778288, 31778493; an attempt
with `srun --mpi=pmi2` in between, job 31778480, ended with `srun: command
not found`, which was the hint).  With

    source /data/lcheng24/intel/oneapi/setvars.sh --force
    export PATH=$PATH:/cm/shared/apps/slurm/current/bin
    export OMP_NUM_THREADS=8 I_MPI_PIN_DOMAIN=omp
    mpirun -np 4 -ppn 4 /path/to/build/xzccsdt_mpi/xzccsdt "ccsdt(q)"

in the job script it works (job 31778544).  CFOUR's library needed nothing:
only rank 0 calls CRAPSI and CRAPSO (JOBARC is opened by one process), the
other ranks call `MCHPRM` and `GETMEM` of `liblibr.a` only.

## 9. Not done

- Lambda: the residuals of rank 1 and 2 and the o v^3-sized operands in
  blocks of the labels (they are dense, as in the Python program: with
  labels they are most of the memory of an iteration, and were 39 % of its
  time before their loops became GEMMs - section 12: now a sixth at 26 +
  120); l3, T3 or the residual in a mapped scratch file; the Lambda
  equations with constant terms (lambda of CCSDT(Q), the equations of the
  first derivatives of (Q)_Lambda: docs/lambda.md 11, 12); the densities of
  (Q) and (Q)_Lambda (the density of CCSDT is there: section 13).
- The amplitude iterations with the Krylov solver of the Lambda equations
  (Newton-Krylov / Anderson with the full history; section 11: Sc 49
  instead of 77 cycles in an experiment); the Krylov vectors of Lambda in
  the file in fewer passes (each cycle reads the subspace about six times).
- The iterations over ranks: section 15.10 (what is not distributed - in
  blocks of the labels the small intermediates, a quarter of an iteration,
  are made by every rank with its own threads -; T3 shared between the
  ranks of one node; the final commit timed on several nodes).
- (Q) over ranks on separate nodes was not timed (section 8).  Whole groups
  now go to the ranks and T3 is shared between the ranks of a node (sections
  5, 14).  Not done there: Xo shared in the same way (it is formed by every
  rank - 21 to 25 s per rank at 12 x 4 - and is the largest private array
  of (Q): 6.9 GB of 11.6 GB per rank for 26 + 160 in real blocks, which is
  what limits the number of ranks on a node now); Xv passed between ranks instead of formed again by the pieces of a
  cut group; the mapped T3 on a network file system (the warning and
  `qmap 0` are there, no such run was made); a cost model with a constant
  of Xv per mode (one constant, between the two measured ones).
- (Q) in sectors was validated at cc-pVDZ only and not timed; its memory
  plan has one knob (the occupied indices whose Xv is held), no batches of
  columns.
- Labels with a modulus; the rotation of unadapted Kramers pairs into j_z
  eigenfunctions (CFOUR's SCF gives labelled, real integrals itself with a
  section `%dglin`, and the program takes the labels from its `dgcosym`:
  section 6).
- A scheduler compiled once for both element types (section 3): today one
  source, two instances.
- T3 of the iterations in a mapped scratch file ((Q) reads a mapped T3:
  section 5); `<ab||cd>` in a file (it is in the work array, bare, once; the
  dressed operator is never stored whole).  A split of the DIIS vectors
  between the work array and the file: decided against (section 4: all in
  memory, or all in the file).
- CCSDTQ, (P), properties, the AO-driven CCSD(T); CFOUR's nonrelativistic
  programs.
- Excited states: EOM-EE-CCSDT energies and right-hand vectors are there
  (section 16); not done: left-hand vectors and properties, IP / EA, more
  than one MPI rank, restart and root following (16.9).
- CC3: section 17.8.
- The relaxed density for `xsox2cprop` (`denfinal`): the density stage and
  the files of `xx2czvec` are there for CCSDT, with frozen and dropped
  spinors (section 13).  Not done there: the contributions of (Q); the held
  stage in blocks and in real arithmetic (not validated) and its timing at
  production size; a relaxed density for labelled real spinors (`xx2czvec`
  needs the spinors of CFOUR's SCF, which have no labels); a Z-vector solver
  of our own where `xx2czvec` is wrong (section 10.4).
- The one-rank-per-node timing of (Q), labels with a modulus: as before.
- Of the ZMAT only the sections of the excited states are read
  (`%eomnroot`, `%eomthrs`, `%eomndav0`, `%xzeom`, `%xzeomguess`,
  `%xzeomvec`: section 16.1), not `%zccsdt` of the Python front end: the
  method by argument or CALC, thresholds by CC_CONV / CC_MAXCYC (ESTATE_CONV
  / ESTATE_MAXCYC), everything else by the options of the section `%xzccsdt` (section 18)
  (test hooks and overrides).

## 10. Relaxed density through xx2czvec

The specification of the hand-over of a coupled-cluster density to CFOUR's
own Z-vector program, so that `xsox2cprop` gets its `denfinal` without the
orbital response being solved here.  Pinned with the Python program
(`zccsdt.cfour`: `handover_files`, `write_handover`, `handover`,
`handover_density`, `xx2czvec_sound`; `python -m zccsdt.cfour --properties
--zvector cfour RUNDIR`) on the files that CFOUR's `xx2cccdenocbp` writes
for CCSD and CCSD(T).  Every statement below that is not marked "from
reading" is a measurement of 2026-10-08: H2O and HF / 6-31G, X2C-1e,
SYM=OFF, geometry in bohr, SCF_CONV=11, CC_CONV=12; the binaries of
`~/apps/cfour-x2cso/bin`; login node, 4 threads.  The native program writes
these files since section 13 (`xzccsdt density`), which also has the answer
to how `xx2cccdenocbp` makes the parts with a frozen index.

### 10.1 What xx2czvec does

`xx2czvec` (sources `x2czvec/`) runs after `xjoda`, `xsox2catm`, `xsox2chf` -
`xx2cvtran` is not needed - and reads, besides the seven files below (from
reading): `ZMAT` (for a section `%vircut` only), `JOBARC` (NUMDROPA,
NUMVIRTA, IFLAGS, the spinors `scf2ccmo`, the Fock matrix `scf2cfck`,
`scf2ceig`), `nmototal`, `dgcosym`, the AO integrals (`IIII`, or its own
`ao2enum`, `ao2efil` of an earlier run) and the property integrals of
`xsox2catm`.  It makes the response integrals `<ab||ij>` and `<aj||ib>` over
**all** occupied and **all** virtual spinors from the AO integrals itself
(files `intpphh2`, `intphhp2`), solves

    R(d) = -x,     R(d)_ai = (e_a - e_i) d_ai + sum_bj ( <aj||ib> d_bj + <ab||ij> conj(d_bj) )

for d[a,i] (a: all virtual, i: all occupied spinors; x: the file `xfull`;
R: the orbital Hessian of the SCF, `zccsd.zvector.response`; `<pq||rs> =
(pr|qs) - (ps|qr)` as everywhere here) in a Krylov space preconditioned
with 1 / (e_a - e_i) - at most IFLAGS(31) vectors, until the new one is
below 10^-IFLAGS(30); 64 and 16 in these runs -, and assembles `denfinal`
(N spinors in CFOUR's order [frozen occupied, active occupied, active
virtual, dropped virtual]; `D[p,q] = <q+ p>`, the convention of
`bench/cfour/write_density.py`):

| block of `denfinal` | from |
|---|---|
| D[a,i], all virtual by all occupied | d (also written as `dpvofull`) |
| D[i,a] | conj(d[a,i]) |
| active occupied, D[i,j] | `dpoo`[i,j] + conj(`dpoo`[j,i]) + delta_ij (`dii0`[i,i] + 1) |
| active virtual, D[a,b] | `dpvv`[a,b] + conj(`dpvv`[b,a]) + delta_ab `daa0`[a,a] |
| D[frozen f, active occupied i] | `dpopo`[f,i]; D[i,f] its conjugate |
| D[dropped d, active virtual a] | `dpvpv`[d,a]; D[a,d] its conjugate |
| frozen by frozen | the unit matrix |
| dropped by dropped | zero |

(`cfour.handover_density` is this table; xx2czvec's `denfinal` equals it to
rounding in every run below.)  Nothing else enters: `xfull` is the whole
right-hand side - xx2czvec adds no term from the other files to it - and of
`dii0`, `daa0` only the diagonals are read.  It also writes `denfina2` (a
copy), `dennoscf` (without the 1 on the occupied diagonal) and `denpcorr`.
No two-particle density crosses the interface.

### 10.2 The files

Each of the seven is one record of a Fortran sequential unformatted file
(4-byte length, data, 4-byte length; `cfour._write_complex`), complex
numbers as (re, im) of 8 bytes, the first index fastest, in the basis of
CFOUR's spinors (`scf2ccmo`) and in its order.  A block without elements is
a record of length zero (a file of 8 bytes), and the file must be there.
With no = active occupied, nv = active virtual, nf = frozen occupied, nd =
dropped virtual spinors:

| file | shape | content |
|---|---|---|
| `xfull` | (nv + nd, nf + no): rows active virtual then dropped, columns frozen then active | x[a,i] below |
| `dpoo` | (no, no) | (D - 1) / 2 on the active occupied spinors |
| `dpvv` | (nv, nv) | D / 2 on the active virtual spinors |
| `dii0` | (no, no) | zeros |
| `daa0` | (nv, nv) | zeros |
| `dpopo` | (nf, no) | D[frozen, active occupied] |
| `dpvpv` | (nd, nv) | D[dropped, active virtual] |

D is the Hermitian one-particle density **without** the response of the
virtual-occupied rotations, in the names of `zccsd/zvector.py` (`_relax`)
and `cc/density_opt.py` (`relaxed_rdm1`):

    D = D0 + y

- D0: the unrelaxed density, reference included, Hermitized - `dm1 = (d1^T
  + conj(d1)) / 2` of `density_opt.kernel` in the active block, 1 on the
  frozen diagonal; for (T) with the terms of E(T) on its diagonals
  (`zvector.kernel`), for CCSDT(Q) plus `diag(q.de)`, the derivative of
  E(Q) with respect to the orbital energies (docs/lambda.md 11.2).
- y: the rotations that follow from the Fock matrix staying diagonal,
  `y[p,q] = G[p,q] / (e_q - e_p)` with the orbital gradient G[y,x] of
  `zccsd/zvector.py` (anti-Hermitian), zero between spinors closer than
  `zvector.DEGENERATE`: for the pairs (active occupied, frozen), (active
  virtual, dropped) and their transposes - the frozen-index blocks of
  `orbgrad_frozen` / `density_opt` (`frozen_terms`) plus the Fock matrix of
  the correlation density - and, for (T) and (Q) only, among the active
  occupied and among the active virtual spinors.  y is Hermitian and has no
  virtual-occupied block.  For CCSD and CCSDT without frozen or dropped
  spinors y = 0.

So `dpopo` and `dpvpv` are y alone (D0 vanishes there), and the oo and vv
rotations of (T) and (Q) (the Fock-density route of docs/lambda.md 11) ride
in `dpoo` and `dpvv`.  The right-hand side is

    x[a,i] = G[a,i] + F[y][a,i] - R(D0_vo)[a,i]          (all virtual a, all occupied i)
           = -( gvo + R(D_vo) ),       F[M][p,q] = sum_rs <ps||qr> M[r,s],

with `gvo = -G_vo - F[y]_vo` the right-hand side of the Z-vector equations
`R(z) = gvo` of `zvector._relax` (for CCSDT without frozen spinors `gvo =
-gnh / 2`, `Result.gnh`), and D0_vo = D_vo the virtual-occupied block of the
unrelaxed density (non-zero between active spinors only).  xx2czvec's
solution is then the whole block of the relaxed density, d = D0_vo + z:
the unrelaxed virtual-occupied block is **not** added by xx2czvec, it has
to be inside x through R(D0_vo) - one contraction of D0_vo with `<aj||ib>`
and `<ab||ij>` (a, i over all spinors of their kind, b, j active), or one AO
Fock build, which is what the Python program does.  Sign and conjugation
as written: `xfull` is minus R of the final D_vo, not conjugated (factor
-1.0000000000 in a fit of CFOUR's file, deviation 5e-13).  The two formulas
with MO integrals, R and F, agree with the AO-direct ones of `zvector` to
rounding, and x made with them with `xx2cccdenocbp`'s `xfull` to 1e-13
(CCSD(T) with a frozen core, CCSD with dropped spinors).

The split between `dpoo` and `dii0` is free.  `xx2cccdenocbp` writes a
`dpoo` that is not Hermitian and puts the diagonal into `dii0` (likewise
`dpvv`, `daa0`); only `dpoo + dpoo^+ + diag(dii0)` and `dpvv + dpvv^+ +
diag(daa0)` are comparable with it, and they are what was compared.

The spinors.  Each of CFOUR's SCF runs leaves its own mixture within a
Kramers pair, and the amplitudes of the Python program may live on another
one (`DirectZCCSD`: adapted pairs).  With U = C(CFOUR)^+ S C(ours), which
mixes only degenerate spinors: D -> U D U^+, x -> U_v x U_o^+ before
writing (`cfour.handover`).  The native program runs on CFOUR's spinors and
needs no such step.

### 10.3 Validation

Our files against those of `xx2cccdenocbp` in the same run directory (its
CCSD / Lambda chain first), element by element, max |difference|; then
`xx2czvec` on our files against its solution (`dpvofull`) and `denfinal` on
CFOUR's own files:

| H2O / 6-31G, 26 spinors | nf, nd | `xfull` | `dpopo` | `dpvpv` | oo combination | vv combination | `dpvofull` | `denfinal` |
|---|---|---|---|---|---|---|---|---|
| CCSD | 0, 0 | 5.3e-13 | - | - | 1.2e-13 | 2.7e-13 | 2.6e-13 | 2.7e-13 |
| CCSD, DROPMO=1>1 | 2, 0 | 1.2e-13 | 8.2e-15 | - | 1.2e-13 | 2.9e-13 | 2.8e-13 | 2.9e-13 |
| CCSD(T) | 0, 0 | 5.3e-13 | - | - | 2.3e-13 | 2.8e-13 | 2.9e-13 | 2.9e-13 |
| CCSD(T), DROPMO=1>1 | 2, 0 | 1.1e-13 | 8.7e-15 | - | 2.4e-13 | 2.8e-13 | 3.0e-13 | 3.0e-13 |
| CCSD, DROPMO=1>1, %vircut 1.5 | 2, 2 | 1.2e-13 | 7.7e-15 | 8.6e-15 | 1.9e-13 | 1.7e-13 | 2.8e-13 | 2.8e-13 |
| CCSD(T), DROPMO=1>1, %vircut 1.5 | 2, 2 | 1.5e-13 | 8.7e-15 | 8.9e-15 | 1.9e-13 | 1.6e-13 | 3.3e-13 | 3.3e-13 |

(sizes of the entries: `xfull` 1e-2, `dpopo` 3e-5, `dpvpv` 1e-2, the oo and
vv combinations 1e-2.)  The frozen columns and dropped rows of `xfull`, and
`dpopo`, `dpvpv`, of CFOUR are therefore the frozen-index quantities of the
Python route as they are.

The three routes - CFOUR's own chain, the Z-vector solver of the Python
program (`--zvector python`), the hand-over (`--zvector cfour`) - in one run
directory (the same spinors): the relaxed electronic dipole moment (z) that
`xsox2cprop` prints, the largest difference between `denfinal` of the
hand-over and of the Python solver, and between the 48 numbers of
`xsox2cprop` with the two (dipole, field gradients, contact and effective
densities, ...):

| | CFOUR's chain | Python Z-vector | xx2czvec on our files | `denfinal` | properties |
|---|---|---|---|---|---|
| CCSD | 0.01580313752031 | 0.01580313752061 | 0.01580313752035 | 4.2e-12 | 9.0e-12 |
| CCSD, frozen core | 0.01567701616479 | 0.01567701616334 | 0.01567701616502 | 4.2e-12 | 5.0e-12 |
| CCSD(T) | 0.01227486493558 | 0.01227486493506 | 0.01227486493588 | 1.6e-11 | 1.3e-10 |
| CCSD(T), frozen core | 0.01215439608933 | 0.01215439608680 | 0.01215439608968 | 9.1e-12 | 4.3e-11 |
| CCSD, frozen core, %vircut 1.5 | 0.02620131976691 | 0.02620131976469 | 0.02620131976709 | 1.3e-11 | 5.1e-11 |
| CCSD(T), frozen core, %vircut 1.5 | 0.02358891376170 | 0.02358891376432 | 0.02358891376219 | 1.3e-11 | 9.9e-11 |
| CCSDT | - | 0.01207714055096 | 0.01207714054634 | 4.9e-12 | 2.8e-11 |
| CCSDT, frozen core | - | 0.01195694144941 | 0.01195694144947 | 2.9e-12 | 1.1e-11 |
| CCSDT, frozen core, %vircut 1.5 | - | 0.02344559047768 | 0.02344559047585 | 9.5e-12 | 1.0e-10 |
| CCSDT(Q) | - | 0.01193066698001 | 0.01193066697872 | 3.7e-12 | 1.4e-11 |
| CCSDT(Q), frozen core | - | 0.01181130358927 | 0.01181130358956 | 2.1e-12 | 1.4e-11 |
| CCSDT(Q), frozen core, %vircut 1.5 | - | 0.02324580802959 | 0.02324580802765 | 7.9e-12 | 1.0e-10 |

The differences are of the size of the threshold of the Python solver (a
residual of 1e-11); against CFOUR's own `denfinal` the hand-over is at
3e-13.  xx2czvec converged in 15 cycles and took 0.14 to 0.19 s in these
runs.  So the oo and vv rotations of (T) and of (Q), the diagonal `q.de`,
the frozen core and the dropped virtual spinors all pass through the seven
files, and CFOUR's Z-vector solve reproduces the Python relaxed density to
1e-11.  CFOUR has no CCSDT or CCSDT(Q) density of its own to compare with.

### 10.4 Where xx2czvec is wrong: few active virtual spinors

With 4 of the 16 virtual spinors dropped (`%vircut 1.3`) the same comparison
gives files that agree with CFOUR's as before (`xfull` 1.4e-13, `dpvpv`
3.8e-14, the hand-over against CFOUR's own `denfinal` 2.3e-13) and a
density that is wrong in both: xx2czvec's solution is not the solution of
the equations.  Its residual in R (with integrals made here, `resid.py`)
is 5.5e-3 for a right-hand side of 1.6e-2, and its files `intpphh2`,
`intphhp2` differ from the integrals of CFOUR's spinors made here by 0.1 to
0.9, the block of the active virtual spinors included (with all virtual
spinors active: 2e-15, 8e-15).  The defect is not one of dropped spinors:
HF / 6-31G, all electrons, nothing dropped, has it too.

| run | no, nv, N | 1 + 3 C(nv,3) + nv(nv+1) + no(no+1) | 2 N^2 | relaxed dipole z, CFOUR's chain | Python Z-vector | residual of xx2czvec's solution |
|---|---|---|---|---|---|---|
| H2O CCSD | 10, 16, 26 | 2063 | 1352 | 0.01580313752 | 0.01580313752 | 3e-14 |
| H2O CCSD, DROPMO=1>1 | 8, 16, 26 | 2025 | 1352 | 0.01567701616 | 0.01567701616 | 2e-14 |
| H2O CCSD, DROPMO=1>1, %vircut 1.5 | 8, 14, 26 | 1375 | 1352 | 0.02620131977 | 0.02620131976 | 4e-13 |
| H2O CCSD, DROPMO=1>1, %vircut 1.3 | 8, 12, 26 | 889 | 1352 | 0.00462113289 | -0.02252083884 | 5.5e-3 |
| H2O CCSD, %vircut 1.3 | 10, 12, 26 | 927 | 1352 | 0.00418314407 | -0.02236195666 | 6.0e-3 |
| H2O CCSD, DROPMO=1>1, %vircut 1.19 | 8, 10, 26 | 543 | 1352 | 0.42621843425 | 0.20560900178 | 6.5e-2 |
| HF CCSD, nothing frozen or dropped | 10, 12, 22 | 927 | 968 | -0.26473249072 | -0.00099017762 | 4.2e-2 |

For HF the total dipole moment of `xsox2cprop` is -1.12462471501 with
CFOUR's own density and -0.86088240191 with the one of the Python solver;
the difference of CFOUR's own CCSD energies in fields of +-1e-4
(ZFIELD=+-1000, `xx2cccsd`) is -0.86088237 (for the SCF -0.90178027, against
the -0.90178030 that `xsox2cprop` prints).  CFOUR's relaxed CCSD properties
are thus wrong, without a message, in a plain small calculation; its
energies are not affected.  The Python values of the three `%vircut 1.3`
and `1.19` rows were not checked against finite differences here (their
kind was, in docs/benchmarks.md, "Frozen virtual spinors").

The rule of the table - sound if and only if the third column is at least
the fourth (`cfour.xx2czvec_sound`) - separates these seven runs, two of
them by 23 and 41 words, and agrees with the eight of docs/benchmarks.md
(H2O, N = 80, with 42, 24, 8 active virtual spinors: right, wrong, wrong;
HI, N = 256, with 130, 94, 78, 68, 58: right down to 68, wrong at 58 - the
rule puts the border at 64).  Its
cause, from reading the source and not verified in a debugger:
`x2czvec/x2cccdendrv.f90` hands `work1(tmp2)` to `frmfck2c3` as a scratch
matrix of N x N complex numbers; `tmp2` is never set in
`x2czvec/zvecini.f90` (the assignment is commented out), so the matrix lies
at the start of the work array, where the offset tables of the triples
(3 C(nv,3) + nv(nv+1) + no(no+1) words) are followed by the list of AO
integrals, read before.  Where the tables are shorter than 2 N^2 words,
the first AO integrals are overwritten.  One assignment in `zvecini.f90`
would remove it; the tree is read-only for this work and was not changed.
Until then the hand-over is exact where the rule holds, and the Python
program checks it in any case: `cfour.handover` computes `R(d) + x` for
xx2czvec's solution with one more Fock build, and with a residual above
1e-6 of the right-hand side it moves the file to `denfinal.wrong` and stops
(exit status 1, the reason named).  A native program without a Fock build
of its own has to apply the rule before it calls xx2czvec.

### 10.5 What does not go through this interface

- A density with elements between two frozen spinors other than the unit
  matrix, or between two dropped ones: xx2czvec sets these blocks itself
  (`handover_files` raises).  None of the present methods has them.
- A reference whose orbital Hessian is not the R above: xx2czvec builds it
  from the Coulomb integrals of the AO basis and the orbital energies (from
  reading).  A two-electron operator beyond that in the SCF, an SCF that
  is not converged (the relaxed density then carries its gradient in first
  order, `zvector.SCF_GRADIENT`) and non-canonical spinors are not covered.
  X2CAMF, whose mean field is a one-electron operator, should be: not run.
- Large cases, as far as the memory of xx2czvec goes (from reading, not
  measured): it holds the AO integrals in core and at least two arrays of
  (occupied x virtual spinors)^2 complex numbers, frozen and dropped ones
  included, without symmetry.
- A failed solve gives no signal: when the Krylov iteration does not
  converge the program goes on with what it has (from reading); the line
  "Z-vector equation converged in n cycles" of its output is the only sign
  (`handover` reports the solve as not converged without it, and its
  residual check catches a wrong solution).
- CCSDT(Q)_Lambda, CCSDTQ, (P): the Python front end writes relaxed
  densities for CCSD, CCSD(T), CCSDT and CCSDT(Q) only.  The density of
  CCSDT(Q)_Lambda goes through the same `solver` hook of `relaxed_rdm1`;
  not run through xx2czvec.
- Everything two-particle: the orbital gradient G (its virtual-occupied
  block, its frozen-index blocks, for (T) and (Q) its oo and vv blocks) and
  the Fock matrices F[y], R(D0_vo) have to be made before the hand-over.
  xx2czvec replaces the iterations of the Z-vector equations and the
  assembly of `denfinal`, nothing else.

### 10.6 For the Fortran port

In the order of the work.  (1) D0 and `gnh` (the virtual-occupied orbital
gradient, G_vo = gnh / 2) from the Lambda and density passes; without frozen
or dropped spinors and without (Q) that is all: `dpoo = (D0_oo - 1) / 2`,
`dpvv = D0_vv / 2`, `xfull = gnh / 2 - R(D0_vo)` with R from `<aj||ib>` and
`<ab||ij>` of the active spinors (in the integral files of `xx2cvtran`:
`moinphhp`, `moinpphh`), empty `dpopo`, `dpvpv`, zero `dii0`, `daa0`.  The
CCSD limit of exactly this is the first row of the tables above, and
`xx2cccdenocbp`'s files in the same directory are the element-wise
reference (the combinations of 10.2, not `dpoo` itself).  (2) With frozen
or dropped spinors: the frozen-index blocks of G, then y, F[y] and the rows
and columns of R(D0_vo) with a frozen index need integrals with one frozen
index and, for the block [dropped, frozen], the Fock matrix of the
correlation density (`relaxed_rdm1`: `V`); these are not in the files of
`xx2cvtran`.  (3) (Q): `q.de` on the diagonal, the oo and vv blocks of G
into y.  `tests/test_cfour.py` (`test_handover_files`,
`test_handover_to_a_stand_in_for_xx2czvec`) fix the formats and the
identities without a CFOUR binary; the scripts and logs of the measurements
are in `/home/xwang405/.claude/jobs/7ba4fcf7/tmp/zvec/` (`route.sh`,
`cmp.py`, `pin.py`, `resid.py`, `ints.py`, `formulas.py`, `logs/`).

**Done in the native program** (section 13): steps (1) and (2).  For (2)
the integrals with a frozen index are not taken from new files: they are
transformed in the program by the routines CFOUR's own density program
uses (`int_mintv2e`, `vtranpsog`), one frozen or dropped spinor against all
spinors, and that one class also gives the Fock matrices with two and three
such indices, so no Fock build in the AO basis is needed.  Step (3) is not
done.

## 11. The Lambda equations and (Q)_Lambda

    xzccsdt ccsdt                   # amplitudes -> XZAMPS
    xzccsdt lambda                  # XZAMPS -> lambda -> XZLAMB
    mpirun ... xzccsdt "(q)_l"      # XZAMPS, XZLAMB -> E[Q]_L, E(Q)_L   (or serial)

or `xzccsdt "ccsdt(q)_l"` for all three in one run (with MPI: rank 0
iterates, the other ranks wait and then read both files).  `[q]_l`,
`ccsdt[q]_l` put the fifth-order energy into `TOTENERG`.

### The Lambda equations (`xzlam.inc`, `xzlblk.inc`)

The equations and their factorization are those of `zccsdt.cc.lambda_opt`
(docs/lambda.md, section 9; in blocks `lambda_blk`, 9.7): T1-dressed
integrals, nothing conjugated, l1 and l2 dense, l3 and its residual stored
as T3 is (the packed block, or the blocks of the labels).  Start from
conj(t), Jacobi steps accelerated by a Krylov subspace (since 2026-10-14,
below: `xzlkry.inc`, up to 40 pairs of vectors in the work array or their l3
parts in `XZLKRY`; with option `lsolver diis` DIIS as before, `xzdiis.F90`,
the code of the amplitudes: 8 vectors in the work array or their l3 parts in
`XZDIIS`), converged when the largest element of the Jacobi step over l1,
l2, l3 is below 10^-`CC_CONV` (section 1; until 2026-10-13: |dL1| + |dL2| +
|dL3| of the full tensors).
The output has, per cycle, `1/4 sum <ab||ij> l2[ij][ab]` (the pseudo-energy
that the Python tests compare), the three norms, the largest element and
the time.  Lambda that is not converged is saved and continued as the
amplitudes are (section 1).

**Once** (`xzlamim`): what an amplitude iteration makes, by its routines
(`xzimd.inc`, pass 1 over T3) - the dressed integrals, W_vvvo with its T3
part, W_vooo (plus `f t2`), W_ovvo, F_vv, F_oo, W_oooo, the operator on
occupied triples (transposed) - and the two contractions of T3, `Xq` and
`X2`.  **The pp operator of the T3 equation is made in the place of the bare
`<ab||cd>`** (`~<ab||de> + sum_{m<n} t2[mn][ab] <mn||de> + 1/2 A[F_vv]`, in
its diagonal pair blocks with labels): the Python program holds it next to
the bare integrals, a second v^4/4 array; here there is none, and after the
Lambda equations the work array has no bare `<ab||cd>` any more.  A run that
goes on to (Q)_Lambda (`ccsdt(q)_l`) reads the integral files and `XZAMPS`
again (0.5 s for 10 + 106 spinors).

**An iteration** (`xzlamstep`), in the order of `lambda_opt.residuals`:

| step | dense | in blocks of the labels |
|---|---|---|
| pass 1: l3 with t2 and the W's (two stacked GEMMs), the density `Dr` of l3 with T3, `D1` of l2 with T3 | `zc_l3_pass1w`: two half-expanded batches of `nbp` occupied pairs | `xzlbpass1`: per batch of pairs of a group and group of virtual pairs, `[P][D][Q]` with D the (d, m) of one difference of classes; the stacked products per class of d and m by the routine of the T3 equation (`xzbp1xy`) |
| Gram matrix of l3 and T3 | one GEMM of the packed arrays | per group of triples |
| pass 2: ring with the transposed operator, two outer products, four drive products (K = 2 (o + v)), one reduction | `zc_l3_pass2w` | `xzbpass2`, the pass of the T3 equation, which now takes the number of parts of the summed index and the outer products as arguments |
| ladder with the transposed pp operator; `Xq`, `X2` on the same gathered batches of `nbt` triples | `zc_gather_vir_half`, GEMM with the transposed flag, `zc_reduce_pair_first_rows` | `xzlbladder`: per group of triples and class of the free index |
| the transposed operator on occupied triples | `zc_t3_occop` | `xzbocc` |
| residuals of rank 1 and 2, the Jacobi step | `xzldens`, `xzlr1`, `xzlr2a`, `xzlr2b`, `xzlupd`: loops and GEMMs on dense arrays (the o^2 v^4 product of `Y2` with `~<ej||bg>` and the o^3 v^3 ones are GEMMs); `zc_lam_pair_expand`, `zc_lam_pair_matvec` | the same routines; the pp operator in its pair blocks (`_tab` kernels) |

Nothing of this is written per element type: the `.inc` files are compiled
for real and complex elements as the rest (section 3).

**Memory.**  The regions, bottom to top: tables and integrals (section 4),
T1, T2, T3 (from `XZAMPS`), l1, l2, l3, new l1 and l2, the vectors of the
solver (the Krylov pairs, or the DIIS vectors),
then what is kept for the iterations - `GmQ` and `GPd` (the operands of pass
1: t2 with W_vvvo, t2 with W_vooo), `Bdrv` and `Adrv` (2 (o + v) rows and
columns of the drive products), `~<ej||bg>` (o v^3, both orders of the pair),
`Xq`, `X2`, W_ovvo twice, the occupied operator - and the arrays of the
setup or of an iteration, taken and released stage by stage: the batch
buffers of pass 1 (two), of pass 2 (two), of the ladder (two and a batch of
`X2`), and after the passes `Y2` on both orders of its pair (o v^3) with
three (o v)^2 arrays for the residuals of rank 1 and 2.  T3, l3 and the
residual are three regions of the size of T3 (the residual is a DIIS slot,
or the image of the free Krylov pair, when the vectors are in the work
array).  The plan is that of section 4: a
dry run of `xzlam` after the layout (and after the dry run of the
amplitude iterations in a run that does both), the floor checked before the
large files are read, then `nbt`, `nbp`, `nba` raised one after the other,
then the place of the DIIS vectors, then the Krylov pairs (below).  A run
prints "the least that Lambda takes"; below it:

    @XZCCSDT-F, the work array is too short for this calculation.
      words needed:          2861968   words there:          2800000
      Set MEMORY_SIZE to at least          2862032 words (        22.9 MB).

| case | o, v | mode | least for Lambda | default plan, Lambda | least for (Q)_L | T3 = l3 |
|---|---|---|---|---|---|---|
| H2O / 6-31G | 10, 16 | complex, dense | 22.9 MB | 62.2 MB | 9.6 MB | 1.1 MB |
| the same, DROPMO=1>1, %vircut 1.2 | 8, 10 | complex, dense | 4.5 MB | 8.5 MB | | 0.1 MB |
| HBr / cc-pVDZ-DK, 28 frozen | 8, 28 | complex, dense | 50.7 MB | 158.8 MB | 28.8 MB | 2.9 MB |
| the same, labelled real spinors | 8, 28 | real, 6 classes | 18.6 MB | 22.1 MB | | 0.25 MB |
| | | real, dense (`labels 0`) | 25.5 MB | 79.6 MB | | 1.5 MB |
| | | complex, 6 classes (`real 0`) | 36.8 MB | 43.8 MB | | 0.5 MB |

(the cc-pVTZ cases: under Timings below.)  In blocks the floor is the dense operands,
not l3: for the labelled HBr case 18.6 MB where the three T3-sized regions
are 0.8 MB.

### The solver: a Krylov subspace (`xzlkry.F90`, `xzlkry.inc`; 2026-10-14)

**Why.**  The Lambda equations are linear in lambda, `R(L) = eta + L A`
with A the Jacobian of CCSDT, and the Jacobi step `g(x) = R(x) / D` is
affine: `g(x) = c + M x`, M = A / D.  A direction with an eigenvalue theta
of M is contracted by `1 + theta` per Jacobi step; the eigenvalues of A are
the excitation energies of EOM-CCSDT of the totally symmetric shift, so an
open shell whose determinant has low excited states of the same m_j (a
fine-structure partner, d-d and s-d excitations) has |theta| of 1e-3 to
1e-1 against denominators of 0.3 to 0.6 Eh.  DIIS with 8 vectors loses
these directions while it extrapolates over nearly parallel errors, and
stalls; each restart from `XZLAMB` gains a factor and stalls again (the U
atom of 2026-10-14: 234 cycles in chunks of 15 to 1e-8).

**The reproducer**: the Sc atom, 2D (`%dglin` 8 8 / 3 2 / 0 0: the 3d
electron in m_j = 3/2), cc-pVDZ-DK, X2C-1e, `DROPMO=1>5` (11 + 65 active
spinors), real, 8 classes, CC_CONV=10; 0.14 s per Lambda cycle on 8 threads.
DIIS: 14 cycles to 1.5e-5, then 1e-6 .. 1.3e-7 for cycles 30 to 200
(`did not converge`); in chunks of 15 cycles with a new DIIS sequence each
(as the U runs): 1.3e-8 after 900 cycles.  The orbital Hessian of the
determinant (active spinors, m_j-conserving rotations) has its lowest
eigenvalues at -0.0049, 0.0026 (real) and -0.0005, 0.0017 (imaginary)
Eh - a saddle point in the parity-breaking rotations, 0.0031 / 0.0017 for
the parity-conserving ones.  The softest direction of M that the solver
finds (option `lritz`): theta = -0.00315 (contraction 0.99685; A eigenvalue
about 0.0017 Eh, the 2D3/2 -> 2D5/2 splitting at this level), its largest
l1 elements the singly occupied 3d (spinor 11, eps -0.343) -> virtuals of
m_j 3/2 at 0.195 and 0.096 Eh; then a cluster at theta = -0.078, -0.084,
-0.088, -0.097, -0.116 (contraction 0.92 .. 0.88).

**The method** (Anderson's form of GMRES with the full history; one
Jacobi step - one residual of the Lambda equations - per cycle, as DIIS):

1. g = (x + g) - x, the step of the present lambda x; its largest element is
   the test (CC_CONV) and the three norms are printed, as before;
2. the difference to the previous cycle, (dx, dg) = (x - x', g - g'), is a
   new pair: dg orthogonalized against the images s_i of the subspace
   (classical Gram-Schmidt; a second time if the first took more than half
   of its norm) and normalized, in the same pass; dx kept raw, the
   directions b = dX T through a small triangular T;
3. y = -S^H g minimizes |g + S y| (S orthonormal), and the next lambda is
   `x + B y + (g + S y)`: the point of the subspace with the smallest step,
   plus that step;
4. when the subspace is full it is compacted (thick restart) to the
   harmonic Ritz directions of M with the smallest |theta| -
   (S^H B) y = y / theta - half of it: the soft directions stay, which a
   short history loses.  The kept directions are orthonormalized in the
   small space, so the images stay orthonormal; both sets are transformed
   in place (`zc_vtransform`).

**The singles of the step** are not r1 / D but delta with K delta = -r1, K
the singles block of the Jacobian as the l1 residual has it (`xzlr1`:
`delta_mi F_vv[a,e] - delta_ae F_oo[m,i] + W_ovvo[m,a,e,i]`, the dressed
intermediates of the Lambda equations) on the singles that the labels allow
(691 of 4424 for the U atom), LU-factorized once after the intermediates
(rank 0; at most 6000 singles, else the denominators; option `lprec 0`:
the denominators).  The fixed point is the same; the step is a better one
for the excitations within the open shells (`-<ia||ia>`, the particle-hole
attraction of a compact shell, is most of the difference between eps_a -
eps_i and the excitation energy): the largest l1 element of the step of the
U atom at a given cycle 5 to 8 times smaller, 101 instead of 126 cycles.

The overlaps of a cycle are one pass over the images (`zc_vgram`, added to
`csrc/t3pack.c`: blocks of all vectors in cache once).  The convergence
measure is that of DIIS: the largest element of the Jacobi step at the
present lambda (a residual, not an extrapolated estimate).

**Storage and plan.**  A vector is (l1, l2 on pairs) - rank 0's, always in
the work array - and l3 as stored.  Slots: the columns of the subspace, the
previous iterate with its step, and a free slot into whose image region the
next cycle writes its new l3 (no separate residual region in the work
array).  The memory plan (section 4) places, after the DIIS vectors of the
amplitudes, as many pairs as fit up to 40 (option `lkry`); if fewer than 10
fit, the l3 parts go to the file `XZLKRY` of the run directory (40 pairs,
as many as the disk holds, records that fit the work array), read and
written in chunks, deleted at the end.  Over MPI ranks the solver is called
where DIIS was: with slices (section 15) every rank holds the slice of its
rows of every vector (its own `XZLKRY.<rank>` with files) and the overlaps
are summed over the ranks, the small numbers broadcast from rank 0; without
slices rank 0 alone.  A run that continues from `XZLAMB` starts a new
subspace.  Option `lsolver diis`: DIIS as before.

**Measured** (job 31994523, node c428, 8 threads; Lambda only, from
`XZAMPS`; Lambda cycles):

| case | DIIS | Krylov, default (40 pairs, singles block) | Krylov, `lprec 0` | Krylov stage / all |
|---|---|---|---|---|
| Sc 2D (11 + 65), CC_CONV=10 | not converged in 200 (1e-6 .. 1.3e-7); 900 in chunks of 15: 1.3e-8 | **73**; 97 with `lkry f12` (12 pairs, files) | 79; 77 with 63 pairs, 103 with 12 (work array or files), 81 at the floor of the work array (files, 28 pairs) | 3.1 of 12.8 s |
| h2o_631g, CC_CONV=9 | 13 | 13 | 13 | |
| hbr_fc, CC_CONV=10 | 16 | 14 | 15 | |
| hbr_lab, hbr_ph, hbr_dglin_fc | 16 | 15 | 15 | |
| hbr_dglin_fc_vc | 15 | 14 | 15 | |
| hbrp_dglin_fc (HBr+) | 16 | 16 | 16 | |

Sc: E(Q)_L -0.000395048222 in every Krylov run (12, 40, 63 pairs, in the
file, 2 ranks with and without slices - their cycles those of the serial
run in every printed digit -, at the floor of the work array, with and
without the singles block); -0.000395048199 from the DIIS lambda of 900
cycles (1.3e-8).  The regression cases: `regress.sh --tol --eom` serial and
`--np 2` OK, every energy and density element within 1e-10, serial and 2
ranks identical in every printed number.

**The U atom** (CFOUR `%dg2lin` determinant, U:ADYVTZ-SO, 14 + 316 active
spinors, real, 12 classes, T3 1370 MB; the production directory of
2026-10-14 with its `XZAMPS`; lambda from conj(t), CC_CONV=8, one run each;
bigmem nodes, 24 threads, `MEMORY=330GB`):

| | DIIS (the production log: 16 threads, bigmem15) | Krylov, `lprec 0` (job 31996609, bigmem17, shared) | Krylov, default (job 31997464, bigmem19) |
|---|---|---|---|
| Lambda cycles to max\|dL\| < 1e-8 | 234, in 6 runs (104 + 71 + 4 x 15; each run a new DIIS) | **126**, one run | **101**, one run |
| seconds per cycle | 31 | 32.6 (Krylov stage 3.6) | 24.2 (Krylov stage 2.6) |
| Lambda wall | about 7300 s in the 6 runs (+ 6 x 70 s integrals, intermediates) | 4125 s | 2455 s |
| Lambda plan / peak resident | 47.8 GB (8 DIIS vectors) / 50.4 GB | 137.9 GB (40 pairs: 109.6 GB) / 138.8 GB | 137.9 GB / 138.8 GB |
| E[Q]_L | 0.000195642398 | 0.000195642416 | 0.000195642421 |
| E(Q)_L | -0.001280128680 | -0.001280128534 | -0.001280128688 |

The energies agree within the convergence of lambda (1e-8 in the largest
element; DIIS 9.9e-9, Krylov 9.6e-9 and 9.9e-9).  The softest directions
(`lritz 6` at the end): theta = -0.0251, -0.0301, -0.0460, -0.0607, -0.0735
+- 0.0021i (`lprec 0`; with the singles block -0.0276, -0.0310, -0.0540,
...): contractions 0.975, 0.970 - the 0.97 of the stall; the first is in
l1 the occupied spinor 9 (eps -0.355) -> virtual 180 (0.344) and the singly
occupied 6d (spinor 13, eps -0.232) -> virtual 219 (0.118), the element the
DIIS runs showed.  Late in the iterations the step of a cycle is often
larger than the minimal residual of the cycle before (|g(x_k+1)| about 1.08
|g + S y|): the Jacobi step expands some directions (theta below -2 or a
non-normal transient), which DIIS cannot follow and the subspace can.

**The anions U-** (15 + 315 spinors, T3 1698 MB, two determinants `Um1`,
`Um2` of the production directory; Krylov default, one run each, jobs
31999735 / 31999736, bigmem19 / bigmem17, 24 threads):

| | DIIS (production log, 16 threads) | Krylov, default |
|---|---|---|
| Um1: Lambda cycles | 150 in 5 runs, then stopped at 4.0e-8 | **117**, one run, to 1e-8 (32.3 s per cycle, Krylov stage 3.1 s; 3789 s) |
| Um1: E[Q]_L, E(Q)_L | 0.000196249706, -0.001367110959 (lambda at 4.0e-8, `CC_CONV=7`) | 0.000196249651, -0.001367111296 |
| Um2: Lambda cycles | 207 in 8 runs, then stopped at 8.1e-8 | **92**, one run, to 1e-8 (37.4 s per cycle on a shared node, Krylov stage 4.2 s; 3460 s) |
| Um2: E[Q]_L, E(Q)_L | 0.000146694203, -0.001712301114 (lambda at 8.1e-8, `CC_CONV=7`) | 0.000146693652, -0.001712310933 |
| Lambda plan / peak resident | | 167.2 GB (40 pairs: 135.8 GB) / 168.1 GB |

Softest directions theta = -0.0286 (Um1), -0.0268 (Um2), the singly occupied
open shells (spinors 9, 14 of Um1; 9, 10 of Um2) to virtuals of the same
m_j.  The difference of E(Q)_L of Um2 (1e-8 Eh) is that of the DIIS lambda
at 8.1e-8: in a direction with theta = -0.027 the error of lambda is the
residual divided by |theta|, 37 times the residual.  An independent Krylov run of Um2
(`lprec 0`, job 32003950: 100 cycles, 3774 s) gives E(Q)_L -0.001712310932,
E[Q]_L 0.000146693652: the two Krylov solutions agree to 1e-12 Eh.

**The amplitude equations.**  The CCSDT iterations of the same cases are
slow late for the same reason: near convergence their step is that of the
same Jacobian (its right action; the same eigenvalues), so the soft
directions of the Lambda equations are those of the T equations.  Measured
on Sc (an experimental build that drives the T iterations through the same
update; not in the program): CCSDT in 49 instead of 77 cycles, E(CCSDT)
the same to 1e-12, and the harmonic Ritz value of the softest direction
-0.00314 (Lambda: -0.00315).  What would fix it: the same subspace solver
as a Newton-Krylov / Anderson method with the full history for the
nonlinear equations (the T2 and T3 of a cycle change the Jacobian a little:
compaction keeps the soft directions, refreshed by new differences), or
DIIS with the soft directions of the Lambda (or EOM) subspace as a
deflating preconditioner.  Not done (section 9).

### (Q)_Lambda (`xzq.inc`, `xzqs.inc`)

`E[Q]_L = sum z2 r / D`, `E(Q)_L = sum (z2 + z3) r / D` with r of (Q),
`z2 = <0| Lambda_2 V |Q>` and `z3 = <0| Lambda_3 V |Q>`, nothing conjugated
(docs/lambda.md, section 5; `quadruples_lambda.py`).  In the routine of one
occupied quadruple, with `left` set: the operands of z are l2 and
`<kl||cd>` instead of t2 and its conjugate; after r is there, the stacked
operands of [V,T3] are filled again with the integrals the other way round
(not conjugated) and with l3 - the expanded rows of the splits (i|jkl) and
the rows of the splits (ij|kl) -, and their two products are reduced into
z3; `zc_ql_energy` (dense) or `zc_ql_sector_energy` (sectors: the blocks of
z3 on two pairs and on an index and a triple next to those of r) gives the
two energies.  l3 is read from its blocks in the sectors, as T3 is.  More
than (Q) takes: l2 and l3 in the work array, l2 on pairs, z3 (C(v,4)
elements; in sectors a set of blocks of each kind) - the operands and the
product arrays are those of (Q), used a second time.  The memory plan is the
plan of (Q) with these regions in its dry run.

### Timings

One job (31779386, node c697, Xeon Gold 6248R, `sbatch -p parallel -c 8
--mem=30G`, snapshot of commit ca6f489), 8 threads, the Python program first
and then the native one on the same integral files of the CFOUR chain in the
node-local `/tmp`; `MEMORY_SIZE=22 GB` for both, `CC_CONV=9`.  Python: one
process that does CCSDT, (Q), Lambda, (Q)_L in this order (its peak resident
set is that of the process up to the step); native: four invocations
(`ccsdt`, `(q)`, `lambda`, `(q)_l`), each with its own peak.  Wall seconds.
All energies of the two programs agree to the 12 decimals printed:

| | H2O / cc-pVTZ (10 + 106) | HBr / cc-pVTZ-DK, 28 frozen (8 + 78) | the same, labelled real spinors |
|---|---|---|---|
| E(CCSDT) | -76.394890322517 | -2605.165534626435 | -2605.165534626450 |
| E(CCSDT(Q)) | -76.395259065600 | -2605.165759197886 | -2605.165759197901 |
| E(CCSDT[Q]_L) | -76.395053160857 | -2605.165560675042 | -2605.165560675057 |
| E(CCSDT(Q)_L) | -76.395261775543 | -2605.165762234866 | -2605.165762234881 |
| cycles, CCSDT / Lambda (both programs) | 19 / 17 | 19 / 17 | 19 / 17 |

| | H2O / cc-pVTZ, complex, dense | | HBr / cc-pVTZ-DK, complex, dense | | HBr, labelled real spinors (8 classes) | |
|---|---|---|---|---|---|---|
| | Python | native | Python | native | Python | native (real, blocks) |
| amplitude iteration | 15.85 | 13.89 | 2.34 | 1.82 | 0.74 (blocks, complex) | 0.16 |
| **Lambda iteration** | **19.46** | **18.68** | **2.54** | **2.30** | 2.55 (packed block, complex) | **0.16** |
| Lambda intermediates, once | 5.62 | 4.24 | 1.08 | 0.69 | 1.07 | 0.13 |
| Lambda, whole step | 337.2 | 321.9 | 44.4 | 39.7 | 44.5 | 2.91 |
| (Q) | 160.5 | 153.4 | 12.6 | 11.8 | 2.74 | 0.63 |
| **(Q)_L** | **226.8** | **214.3** | **17.6** | **16.2** | 3.09 | **0.85** |
| (Q)_L / (Q) | 1.41 | 1.40 | 1.39 | 1.37 | 1.13 | 1.35 |
| peak resident set: CCSDT | 10.1 GB | 9.08 GB | 3.40 GB | 2.96 GB | 1.09 GB | 0.32 GB |
| - Lambda | 10.6 GB | 9.50 GB | 3.59 GB | 3.21 GB | 3.64 GB | 0.40 GB |
| - (Q)_L | (10.6 GB) | 6.99 GB | (3.59 GB) | 2.19 GB | (3.64 GB) | 0.32 GB |
| work array used: CCSDT | - | 9125 MB | - | 2812 MB | - | 236 MB |
| - Lambda (planned = used) | - | 9549 MB | - | 3019 MB | - | 319 MB |
| - (Q), (Q)_L | - | 6563, 7035 MB | - | 1963, 2061 MB | - | 185, 194 MB |
| the least that Lambda takes | - | 2947 MB | - | 798 MB | - | 248 MB |
| the least that CCSDT, (Q), (Q)_L take | - | 2125, 2880, 3352 MB | - | 597, 810, 909 MB | - | 185, 185, 185 MB |

The Python Lambda of the labelled directory ran on the packed block in
complex arithmetic (`Result.blocks` False, as on the cc-pVDZ-DK files), so
the last pair of columns compares the native blocks in real arithmetic with
that, not with the Python block code; the Python (Q) and (Q)_L there ran in
sectors.  In sectors the floors of the three native steps are the same
number: the transient of reading `moinpppp` (the record of the file) is above
what the steps take.

Where a native Lambda iteration goes (17 iterations, seconds; H2O dense, HBr
dense, HBr blocks): pass 2 95.1, 12.4, 0.79; the ladder GEMM 87.4, 8.8, 0.11;
the density of l3 with T3 in pass 1 70.3, 8.6, 0.15; the stacked GEMMs of
pass 1 25.1, 2.9, 0.17; the residuals of rank 1 and 2 20.8, 3.2, 1.14; the
gathers of pass 1 7.6, 1.3, 0.12; DIIS and update 4.1, 0.8, 0.11.  Against
the amplitude iteration of the same job (pass 1 36.4, pp ladder GEMM 97.8,
pass 2 97.3 s in 19 iterations for H2O) the Lambda iteration has the density
(one more product of the size of the ring) and the dense residuals of rank 1
and 2, and no dressing of `<ab||cd>`: 18.7 against 13.9 s.  In blocks the
dense residuals of rank 1 and 2 are 39 % of the iteration - what the labels
do not touch (section 9).  (Q)_L: the left side with Lambda_3 takes 61.3 of
214.3 s (H2O), 4.4 of 16.2 s (HBr), 0.14 of 0.85 s (sectors).

Memory: the work array used equals the plan in every run (the last line of
the plan and "work array used" are the same number of words).  The 8 DIIS
vectors are in the work array in all of them (5.9 GB of the 9.5 GB of the
Lambda step of H2O); the resident set of the native Lambda step is 0.4 to 1.1
GB below that of the Python process.  **Not measured:** the Lambda step under
a `MEMORY_SIZE` near its floor at this size (the floor and file-DIIS paths ran
on the small cases only), (Q)_L over MPI ranks at this size, HBr /
cc-pVQZ-DK.

### Validation

The Python program on the same run directory and integral files (a script
that does what `zccsdt.cfour.run` does for `ccsdt(q)_l`: `GCCSDT.from_integrals`,
`kernel()`, `solve_lambda()`, `ccsdt_q_lambda()`; `symmetry` set for the
labelled directories) against the native one; `CC_CONV=9`; login node.
Lambda: the largest element of the difference of l1, l2, l3 between the two
programs, both in CFOUR's order of the spinors on the packed block (the
native arrays of a run with labels put back from the order of the classes
and from the blocks); the amplitudes of the two programs agree to 1e-16 as
well.

| case (o + v) | native mode | cycles (Python / native) | max \|dl1\| | max \|dl2\| | max \|dl3\| | pseudo-energy, both |
|---|---|---|---|---|---|---|
| H2O / 6-31G, X2C-1e (10 + 16) | complex, dense | 15 / 15 | 4e-17 | 4e-17 | 4e-18 | -0.13609781981803 |
| the same, DROPMO=1>1, %vircut 1.2 (8 + 10; 6 dropped) | complex, dense | 15 / 15 | 3e-17 | 5e-17 | 4e-18 | -0.06636667305326 (Python ...325) |
| HBr / cc-pVDZ-DK, X2C-1e, 28 frozen (8 + 28) | complex, dense | 16 / 16 | 1e-17 | 8e-17 | 2e-18 | -0.13899853036388 |
| the same, m_j-adapted real spinors with labels | real, 6 classes | 16 / 16 | 1e-17 | 1e-16 | 7e-18 | -0.13899853035651 |
| | real, dense (`labels 0`) | 16 | 3e-17 | 7e-17 | 6e-18 | |
| | complex, 6 classes (`real 0`) | 16 | 8e-18 | 7e-17 | 7e-18 | |
| | complex, dense (both) | 16 | 1e-17 | 7e-17 | 7e-18 | |
| the same with a phase on every spinor | complex, 6 classes | 16 / 16 | 2e-17 | 7e-17 | 6e-18 | -0.13899853035992 |
| the labelled spinors in a random order (`--shuffle`, seed 11; Python dense) | real, 6 classes | 16 / 16 | 1e-17 | 7e-17 | 7e-18 | -0.13899853035651 |

(largest elements: l1 0.009 to 0.014, l2 0.054 to 0.066, l3 0.0016 to
0.0031.)  The Python Lambda of the labelled directories ran on the packed
block (below `blocks_min` elements), so the native blocks are compared with
the dense Python code.  Batches and memory: the same agreement with one
occupied pair and one triple per batch and with 2 and 3 (`plan`), with
the DIIS vectors in the file (`diis f8`), with the work array at the floor
of the plan (H2O / 6-31G: 2861976 words, one pair per batch, 12 triples, DIIS
in the file) and at 3.5 million words; without DIIS (`diis m0`) 27 cycles
instead of 15 and lambda to 1e-10.  The first run of the first case gave
these numbers; no term had to be found.

(Q)_Lambda, total energies in Hartree, the two programs:

| case | | Python | native |
|---|---|---|---|
| H2O / 6-31G (10 + 16) | CCSDT[Q]_L | -76.169658038682 | -76.169658038682 |
| | CCSDT(Q)_L | -76.169666381377 | -76.169666381377 |
| the same, DROPMO=1>1, %vircut 1.2 (8 + 10) | CCSDT[Q]_L | -76.099714414588 | -76.099714414588 |
| | CCSDT(Q)_L | -76.099724741931 | -76.099724741931 |
| HBr / cc-pVDZ-DK, 28 frozen (8 + 28) | CCSDT[Q]_L | -2604.989481085566 | -2604.989481085566 |
| | CCSDT(Q)_L | -2604.989509911702 | -2604.989509911702 |
| the same, labelled real spinors (Python and native in sectors) | CCSDT[Q]_L | -2604.989481085473 | -2604.989481085473 |
| | CCSDT(Q)_L | -2604.989509911610 | -2604.989509911610 |
| the same with phases (complex sectors) | CCSDT[Q]_L | -2604.989481085482 | -2604.989481085482 |
| | CCSDT(Q)_L | -2604.989509911618 | -2604.989509911618 |
| the labelled spinors in a random order | CCSDT(Q)_L | -2604.989509911610 | -2604.989509911610 |

The native numbers are those of `ccsdt(q)_l` in one run (serial build), and
the same to all printed digits from: the three invocations `ccsdt`,
`lambda`, `(q)_l`; the labelled case dense real, in complex sectors and dense
complex (`labels`, `real`); Xv for 3 of 7 occupied indices with the
products in batches of 20 and 300 columns, and the work array at the floor of
(Q)_L (HBr dense: 3594976 words); the MPI build with 1, 2 and 4 ranks from
the files and with 4 ranks for `ccsdt(q)_l` (H2O, HBr dense, the labelled case
real and with phases: 5 to 25 tasks); `(q)_l` with 20 tasks asked for,
stopped by `qstop` after 5 tasks (serial) or 3 (4 ranks) and started again
(5 and 3 tasks restored; the checkpoint files gone afterwards); `(q)_l` after
an interrupted `(q)`, whose checkpoint is not taken.  `[q]_l` writes the
fifth-order energy to `TOTENERG`.  `qsample 0.3`: 21 of 70 quadruples
(HBr), the estimate printed, no `TOTENERG`.  CCSDT, [Q] and (Q) of the six
cases after the changes: the numbers of section 7.

## 12. Threads: where an iteration goes, and what was changed

Measured on HBr with 10 frozen spinors (26 occupied spinors), m_j-adapted
real spinors with labels written from Python (`bench/cfour/labelled_run.py`;
120 virtual spinors: the 120 of lowest energy of cc-pVQZ-DK), whole nodes
(`sbatch -p parallel -N 1 -n 1 -c 48 --mem=180G`, two sockets of Xeon Gold
6248R, `MEMORY_SIZE=150 GB`), three cycles per run (`ncyc 3`: nothing
converged, the tables of the stages printed).  The nodes of this partition
differ by up to 40 % among themselves in these runs (the same binary, 26 +
120 in blocks on 8 threads: 10.8 s per cycle on c011 and c195, 16.0 s on
c306), so every comparison below is of binaries run one after the other in
one job on one node, the old one first.

**The timers.**  Every stage of the amplitude step, of the Lambda
intermediates and of a Lambda iteration has a named timer (`xztic`, `xztoc`
of `xzdata.F90`): wall time and the CPU time of all threads of the process;
the output has both and their ratio.  That ratio is **not** the number of
cores that work: a thread of the OpenMP runtime that waits spins for
`KMP_BLOCKTIME` (200 ms) and counts as busy, so the ratio is close to the
number of threads in every stage of a short iteration.  With
`KMP_BLOCKTIME=0` it is the cores that work (and the run is a few per cent
slower); the serial or unbalanced stages below were found that way and by
the wall time of a stage against the number of threads.

**Before** (commit 51c6773: the program of section 11 with the timers).
Nothing of size was serial - the loops of `xzimd.inc` had OpenMP directives -
but in blocks the passes over T3 did not scale: a pass was a sequence of
hundreds of small gathers and GEMMs per batch, each a parallel region of its
own, and from 24 to 48 threads an iteration got slower.  The operator on the
occupied triples, which is 30 % of an iteration of the Python program at
this size, was never the problem here: it is built by a threaded loop in
3 ms for 26 occupied spinors, and its products are 1 % of an iteration in
blocks and 4 % of a dense one (26 + 78).

**Changes** (commits 6d73752, 1b7270d, 3bc2a40):

- *Passes in blocks as tasks* (`xzblk.inc`, `xzlblk.inc`).  Pass 1: the
  combinations of classes (d, m) of a batch; pass 2: one (group of Q, pair
  P); pass 1 of the Lambda equations: the gathers per (group, class), the
  densities per (group, pair) with a buffer per thread, the stacked products
  per (group, class).  A task gathers its own block of T3 or l3 and does its
  products with the BLAS sequential inside; what two tasks add to the same
  place (`yall`, `ra`, the densities) is added after the loop or under a
  lock.  A task of more than three fair shares of the work of its loop runs
  before the loop, alone, with threaded kernels (`xzbig`).
- *Batches of the passes in blocks*: the targets in pairs and triples grow
  by the dense fraction of T3, so that the buffers keep their size in bytes
  (33 pairs per batch instead of 3 for 26 + 120).  Without this a batch of
  26 + 160 had one pair, pass 2 a few huge tasks, and the tasks alone were a
  regression there (24 threads, job 31780679: pass 2 5.9 s before, 14.6 s).
- *Element loops that are now GEMMs or vector loops*: the dressed
  `<ab||ij>` (tau `<oo||oo>` as a GEMM), W_vooo (`sum_ne ~<mn||ie> t2`, o^4
  v^2, one GEMM), F_vv of T3 and G_vv, G_oo of the Lambda equations (one
  GEMM each on t2 and l2 as stored), the Lambda_1 residual (its o^2 v^3 sums
  as vector operations over a, threaded over i), the Lambda_2 residual (Y1
  with `~<jo||mb>`, o^4 v^2, a GEMM); the dressed Fock matrix threaded.
- *Not kept as the default: the hh ladder through occupied pairs*
  (`xzwpair`, `xzhhapply`; `hh pair`): T3 gathered by occupied pair in
  chunks of columns, one GEMM with `W(mn,ij) - 1/2 (F_mi d_nj - ...)`, the
  three terms of a triple added to the residual; (o - 2)^2 / (9 o) times
  fewer operations than the operator on triples.  Measured in dense mode (v
  = 78, job 31780679, c195, seconds per cycle in the stage):

  | o | 8 threads: triples | pairs | 48 threads: triples | pairs |
  |---|---|---|---|---|
  | 10 | 0.005 | 0.050 | 0.005 | 0.060 |
  | 14 | 0.045 | 0.145 | 0.010 | 0.180 |
  | 18 | 0.220 | 0.335 | 0.045 | 0.290 |
  | 22 | 0.690 | 0.670 | 0.180 | 0.450 |
  | 26 | 1.710 | 1.305 | 0.645 | 0.820 |
  | 30 | 4.110 | 2.830 | 1.255 | 1.295 |

  The gather and the reduction of the pair form are bound by the memory and
  do not scale; the one GEMM of the operator on triples does.  It wins on 8
  threads from 22 occupied spinors and on 48 threads nowhere up to 30.
- *Not done: the CCSD kernels of `csrc/zccsd_kernels.c`.*  They are for
  complex elements only and for pairs in numpy's `tril` order; the native
  CCSD part runs in real arithmetic for these cases, on pairs in its own
  (grouped) order, and its large terms are already one GEMM each (ladder on
  the dressed `<ab||cd>` slabs, W_klij, the ring).  It is 2 to 4 % of an
  iteration in blocks.

**Before and after, one node** (job 31781558, c306: the binary of 51c6773,
then the final one, alternating, twice; 26 + 120 in blocks, seconds per
cycle; the second repetition agrees within 1 %):

| threads | 8 | 16 | 24 | 48 | 8 -> 24 |
|---|---|---|---|---|---|
| CCSDT, before | 15.97 | 9.63 | 7.87 | 8.75 | 2.03 |
| CCSDT, after | 11.38 | 6.36 | 5.04 | 4.19 | 2.26 |
| ratio | 1.40 | 1.51 | 1.56 | 2.09 | |
| Lambda, before | 25.01 | 15.70 | 12.98 | 14.66 | 1.93 |
| Lambda, after | 14.95 | 8.88 | 7.28 | 6.29 | 2.05 |
| ratio | 1.67 | 1.77 | 1.78 | 2.33 | |

The same job, 26 + 160 in blocks (8 / 24 / 48 threads): CCSDT 49.26 / 26.61
/ 31.10 s before, 32.12 / 14.28 / 11.87 s after; Lambda 77.84 / 44.41 /
59.59 s before, 41.53 / 19.53 / 16.63 s after.  26 + 78 (24 / 48 threads):
CCSDT 2.24 / 2.88 -> 1.44 / 1.31 s, Lambda 3.51 / 3.75 -> 2.24 / 2.15 s.
An earlier job with the same alternation on a faster node (31781091, c011,
the state of 85a86b0, before the batch targets): CCSDT 10.84 / 7.18 / 5.99 /
5.45 s before, 8.62 / 5.34 / 4.97 / 4.09 s after.

The stages of the amplitude step, 26 + 120 in blocks (job 31781558, seconds
per cycle; stages below 0.1 s left out):

| stage | before: 8 | 16 | 24 | 48 | after: 8 | 16 | 24 | 48 |
|---|---|---|---|---|---|---|---|---|
| pass 1 (gather, products) | 2.53 | 2.05 | 1.97 | 2.76 | 1.49 | 1.09 | 1.05 | 0.97 |
| pass 2: gather, products | 5.60 (with the reduction) | 3.36 | 2.75 | 3.15 | 2.67 | 1.55 | 1.22 | 0.90 |
| pass 2: reduction | | | | | 1.07 | 0.58 | 0.41 | 0.29 |
| pp ladder: gather | 0.34 | 0.19 | 0.13 | 0.14 | 0.37 | 0.19 | 0.15 | 0.13 |
| pp ladder: GEMM | 1.23 | 0.81 | 0.54 | 0.36 | 1.25 | 0.81 | 0.51 | 0.28 |
| pp ladder: reduction | 0.82 | 0.45 | 0.33 | 0.28 | 0.78 | 0.40 | 0.30 | 0.21 |
| W_vvvo: t2 with `<ov||vv>` | 1.60 | 0.47 | 0.31 | 0.31 | 1.56 | 0.45 | 0.31 | 0.33 |
| dressed `<ab||ij>` | 0.93 | 0.49 | 0.32 | 0.19 | < 0.1 | | | |
| W_ovvo, F_vv, F_oo | 0.59 | 0.35 | 0.32 | 0.39 | 0.19 | 0.10 | 0.08 | 0.20 |
| W_vooo | 0.26 | 0.15 | 0.11 | 0.08 | < 0.1 | | | |
| dressing: `<ov||vo>`, `<ov||oo>` | 0.34 | 0.19 | 0.14 | 0.10 | 0.34 | 0.19 | 0.14 | 0.10 |
| dressing: Fock | 0.23 | 0.21 | 0.22 | 0.22 | < 0.1 | | | |
| CCSD part | 0.32 | 0.19 | 0.14 | 0.21 | 0.32 | 0.19 | 0.16 | 0.18 |
| hh ladder, F_oo | 0.27 | 0.09 | 0.06 | 0.06 | 0.26 | 0.09 | 0.06 | 0.04 |
| DIIS | 0.27 | 0.19 | 0.17 | 0.16 | 0.28 | 0.20 | 0.17 | 0.16 |
| total | 15.97 | 9.63 | 7.87 | 8.75 | 11.38 | 6.36 | 5.04 | 4.19 |

and of a Lambda iteration:

| stage | before: 8 | 16 | 24 | 48 | after: 8 | 16 | 24 | 48 |
|---|---|---|---|---|---|---|---|---|
| pass 1: gather | 1.56 | 1.15 | 0.98 | 1.25 | 1.32 | 0.91 | 0.85 | 0.66 |
| pass 1: density | 3.25 | 1.98 | 1.76 | 1.67 | 1.86 | 1.13 | 1.13 | 1.31 |
| pass 1: stacked GEMMs | 2.54 | 2.21 | 2.00 | 3.46 | 1.05 | 0.78 | 0.73 | 0.80 |
| pass 2 (with its reduction) | 6.35 | 3.96 | 3.41 | 4.06 | 4.14 | 2.33 | 1.70 | 1.30 |
| ladder: gather, GEMM, reduction | 2.79 | 1.70 | 1.22 | 1.37 | 2.83 | 1.66 | 1.12 | 0.86 |
| R1, R2: densities | 0.68 | 0.37 | 0.26 | 0.22 | 0.19 | 0.10 | 0.07 | 0.06 |
| R1 | 4.80 | 2.76 | 2.08 | 1.48 | 1.10 | 0.65 | 0.66 | 0.39 |
| R2 (a, b, the GEMMs on pairs) | 1.72 | 0.90 | 0.65 | 0.58 | 1.12 | 0.61 | 0.41 | 0.34 |
| total | 25.01 | 15.70 | 12.98 | 14.66 | 14.95 | 8.88 | 7.28 | 6.29 |

**What limits the gain from 8 to 24 threads (2.3 for CCSDT, 2.1 for
Lambda).**  Pass 1 (1.49 -> 1.05 s): its tasks are the pairs of classes of a
batch, a few of them (both indices of the largest class) hold most of the
work, and their products have one small dimension (the spinors of a class),
which a threaded BLAS does not speed up either - with `KMP_BLOCKTIME=0` and
48 threads 17 cores work in this stage, 40 in the products of pass 2.  The
density of the Lambda pass 1 (1.86 -> 1.13 s, no gain beyond 16 threads):
tasks of one GEMM each that add their product under one lock.  DIIS (0.28 ->
0.17 s): sums over vectors of the size of T3, bound by the memory.  The
gathers and reductions of the pp ladder and of pass 2 gain 2.4 to 2.6, the
GEMMs of the pp ladder 2.4.  **Not done:** pass 1 with its large tasks split
over the pairs of the batch.  The densities without the lock were tried
later and are slower: the lock is not what limits that stage (section 13).

**Dense mode** is the kernels of `csrc/ccsdt_t3.c` and was not changed: 26
+ 78, 48 threads, 11.8 s per cycle before and 12.1 s after (job 31780679,
with the pair form of the hh ladder, which was then the default); 26 + 120
on 24 threads 68.8 and 66.4 s (job 31781091).  Its stages are GEMMs with
one small dimension and gathers, bound by the memory on two sockets: pass 1
of 26 + 78 takes 2.4 s on 24 threads and 2.3 s on 48.

**Memory on two sockets.**  All arrays are regions of one work array, whose
pages lie where the thread that wrote them first ran - the integrals where
the reader ran.  Measured with the unchanged program and `numactl
--interleave=all` (48 threads): 26 + 120 dense 63.5 -> 53.1 s per cycle
(job 31780679, c195; 66.0 -> 54.0 s with the binary of 1b7270d); 26 + 78
dense 10.68 -> 9.93 s, with `OMP_PROC_BIND=spread OMP_PLACES=cores` alone
10.08 s, with both 9.16 s, the Lambda iteration 16.18 -> 13.80 s with both
(job 31780541, c380); in blocks the binding gives 1.33 -> 1.18 s and the
interleaving nothing (26 + 78), and 26 + 160 in blocks is not faster
interleaved (24.9 -> 24.5 s, before the batch targets).  So for dense
calculations on a whole node:

    export OMP_PROC_BIND=spread OMP_PLACES=cores
    numactl --interleave=all xzccsdt ccsdt

**Not done:** the pages of the regions touched first in parallel by the
program itself (or its memory policy set at the start), which would make the
wrapper unnecessary.

**Validation after these changes.**  The cases of sections 7 and 11 (H2O /
6-31G, the frozen + vircut case, HBr dense, the labelled HBr case in its four
modes, with phases and shuffled): CCSDT, [Q], (Q), [Q]_L, (Q)_L and the
cycles as in the tables, to the 12 decimals printed, with the final commit
(login node, 4 threads; the labelled case also on 16 and 48 threads); the
same with `hh pair`; one pair and one triple per batch, 2 and 3, the DIIS
vectors in the file, and the Lambda equations with the work array at the
floor of the plan (H2O: 2893968 words, the labelled HBr case 2339480; the
floors grew by 1 %: a second work array of W_vooo and the buffers per thread
of the passes, which make the plan depend on the number of threads).  HBr
/ cc-pVTZ-DK, 10 frozen (26 + 78), labelled, converged (`CC_CONV=9`; job
31780679, 48 threads, the binary of 1b7270d - not rerun with the last
commit), the native program in blocks, dense, and the Python program on the
same files (CCSDT and (Q) in blocks, Lambda on the packed block):

| | native, blocks | native, dense | Python |
|---|---|---|---|
| E(CCSDT) | -2605.269299541794 | -2605.269299541794 | -2605.269299541794 |
| E(CCSDT[Q]) | -2605.268881367147 | -2605.268881367147 | -2605.268881367147 |
| E(CCSDT(Q)) | -2605.269257610951 | -2605.269257610951 | -2605.269257610951 |
| E(CCSDT[Q]_L) | -2605.268884407884 | -2605.268884407884 | -2605.268884407884 |
| E(CCSDT(Q)_L) | -2605.269268719850 | -2605.269268719850 | -2605.269268719850 |
| cycles, CCSDT / Lambda | 20 / 18 | 20 / 18 | 20 / 18 |
| CCSDT, s per cycle | 1.09 | 11.7 | 12.3 |
| Lambda, s per cycle | 1.76 | 18.4 | 100.9 |

(the Python CCSDT of this job read the files as complex arrays; its own
numbers for real integrals are in docs/benchmarks.md.)

**Other additions.**  `ncyc n`: n cycles of every iteration, then on as
if converged, with a warning (timing runs).  `start ccsd`: T1 and T2 of
CFOUR's `xx2cccsd` as the start, from its files `t1guess` (`t1(a, i)`) and
`t2guess` (`t2(ab, ij)` on a <= b, i <= j, the diagonals zero) of the run
directory - one unformatted sequential record each, complex, CFOUR's order
of the active spinors; without the files, with other sizes, or with complex
amplitudes in real arithmetic, the second-order start stays, with a warning.
Measured (login node): the correlation energy of the start is the CCSD
energy that `xx2cccsd` prints (H2O / 6-31G: -0.135423693022), and CCSDT then
takes 17, 15 and 17 cycles instead of 17, 17 and 18 (H2O, the frozen +
vircut case, HBr / cc-pVDZ-DK), to the same energy within the threshold.

## 13. The relaxed density: the density stage and the hand-over

    xzccsdt ccsdt; xzccsdt lambda        # XZAMPS, XZLAMB
    xzccsdt density                      # -> xfull dpoo dpvv dii0 daa0 dpopo dpvpv
    xx2czvec; xsox2cprop                 # CFOUR: Z-vector equations, denfinal, properties

or `xzccsdt ccsdt_density` for CCSDT, the Lambda equations and the density
in one run.  The stage makes the unrelaxed one-particle density D0 and the
orbital gradient of CCSDT and writes the seven files of section 10.2;
CFOUR's `xx2czvec` then solves the Z-vector equations and writes `denfinal`
for `xsox2cprop`.  Nothing of the Z-vector equations is here.  With frozen
occupied or dropped virtual spinors the stage also makes the gradient
between them and the active spinors (below).  Rank 0 only; no energy changes
(`TOTENERG` is the CCSDT energy).  Sources: `xzden.inc`, `xzdfrz.inc`
(type-generic), `xzfrz.F90`; the reference is `zccsdt.cc.density_opt`
(`kernel`, `_frozen_stage`, `frozen_terms`, `relaxed_rdm1`; docs/lambda.md,
section 10) and `zccsdt.cfour.handover_files`.

### What is computed (`xzden.inc`)

The equations are those of `density_opt.kernel`: T1-dressed integrals,
nothing conjugated, every block of the two-particle density g2 made and
contracted with its dressed integrals, except the block with four virtual
indices.  After the Lambda equations the integral files, `XZAMPS` and
`XZLAMB` (now with l1) are read again, as for (Q)_Lambda.

| step | what | by |
|---|---|---|
| dressing | Fock matrix, `~<mn||ie>`, `~<mn||ij>`, `~<mb||ej>`, `~<mb||ij>`, `~<mb||Q>`, the bra-dressed `<ab||ei>` and `<ab||ij>` | the routines of the amplitude step (`xzimd.inc`) |
| `~<ab||de>` | made **in the place of the bare `<ab||de>`**, block by block; its parts of `~<ab||ei>` and `~<ab||ij>` | `xzdrrows`, `zc_pair_t1_contract`, one GEMM with tau1 |
| pass A | l3 with t2 on both sides (`C3`, `Z` of density_opt), l3 with T3 (the ring density `R3`), l2 with T3 | pass 1 of a Lambda iteration (`zc_l3_pass1w`, `xzlbpass1`) with operands that have only their t2 rows and columns |
| pass B | T3 with (l2, Z) and with (l2, C3), stacked; T3 with l1 and with l2 | pass 1 of an amplitude iteration (`zc_t3_pass1w`, `xzbpass1`) with other operands |
| pass C | `(t2 . Lv) Tv^T` (the term l3 t2 t3 of the block ijab) and `(~<ef||gi> . Lv) Tv^T` (the l3 t3 part of the block abcd), `Lv[Q][I][d] = l3[I, d, Q]` | two gathered batches of `nbt` triples, four GEMMs (`xzden`, `xzdbpassc`) |
| Gram matrix | the l3 t3 part of the block ijkl | as in a Lambda iteration (`xzlgramd`) |
| blocks of g2 | ijkl, ijak, ijab, iabj, aibc on dense arrays with pairs in the order of their groups | `xzdg1`, `xzdblk`: GEMMs for the o^3 v^3 and o^2 v^4 products, loops with OpenMP for the rest |
| gradient | `gnh[a][i]`, the 81 terms of the table GRADIENT in the order of `density_opt.kernel` | `xzdgrad`; the two products with `~<ae||gj>` and the one with `~<ab||de>` as GEMMs onto an o v^3 / 2 array, folded |
| hand-over | `D0 = (d1^T + conj(d1)) / 2`, `xfull = gnh / 2 - R(D0_vo)` with R from `<ia||bj>` and `<ij||ab>` in the work array | `xzdxact`, `xzdwrite` |

**Pass C replaces Gamma.**  The Python program has two routes for the l3 t3
part of the block abcd: `Y` (lambda3 with the integrals first, o^4 v^4 / 4)
and `gamma` (`Gamma[(e<f)][(g<h)] = 3 sum l3 t3`, a v^4 / 4 array; always
in blocks).  Here neither: the two places where Gamma is needed between the
active spinors are a product with an operand on its left, so the operand is
multiplied with the gathered batch of l3 first and the result with the batch
of T3 - `2 (C(o,2) + o v) v^2 / 2 . C(o,3) v` multiply-adds, less than the
`v^4 / 4 . C(o,3) v` of Gamma whenever `C(o,2) + o v < v^2 / 4`, and no array
beyond `o v^3 / 2`.  It is the way the ladder of the Lambda equations treats
`Xq` and `X2` (section 11).  With frozen or dropped spinors Gamma is made as
well (below).

In blocks of the labels only the three passes and the Gram matrix differ;
everything after them is the same code on the same dense arrays.  Real and
complex: the two instances of the `.inc` files, as everywhere.

### Frozen occupied and dropped virtual spinors (`xzdfrz.inc`, `xzfrz.F90`)

**How CFOUR's own chain does it** (from reading `x2cccdenocbp/` and
`x2czvec/` of `~/apps/cfour-x2cso-rebuild`; the format of the files measured
in section 10).  `xx2czvec` does not make anything with a frozen index out
of active quantities.  It reads `xfull` with (nu + ndropvir) (no + ndrop)
elements (`x2cccdendrv.f90`) and `dpopo`, `dpvpv` (`rddp.f90`) as they are,
and what it builds from the AO integrals is the matrix of the Z-vector
equations: `genintzvec.f90` calls the transformation `vtranpsog` three times
for `<ab||ij>` and `<aj||ib>` over all occupied and all virtual spinors
(files `intpphh2`, `intphhp2`).  There is no Fock build of a trial vector
in the AO basis.  The frozen rows and columns come from the density program:
`xx2cccdenocbp` holds the AO integrals in core (the file `IIII` through
`int_mintv2e`), transforms the integrals with one index in a small range
and three over **all** spinors itself (`giintccsdp4.f90`: `vtranpsog(nao,
imo, ntot, ntot, ntot)`), contracts them with its densities to
intermediates with a primed (frozen or dropped) index (`iopo`, `ivpv`,
`ivop`, `ivpo`, ...; `giintccsdp3` ... `giintccsdp5`), turns those into
`dpopo`, `dpvpv` with the orbital energies (`ipp2dp1`) and assembles `xfull`
from four blocks (`xmatfull.f90`).  The files of `xx2cvtran` have no
integral with a frozen index.

**What the native program does.**  The same integrals by the same routines:
`xzfrz.F90` reads the AO integrals of `IIII` into the work array with
`int_mintv2e` (libx2cint), takes the spinors from `JOBARC` (`scf2ccmo`) and
their energies from `scf2ceig`, and calls `vtranpsog` for one frozen or
dropped ("held") spinor h at a time: `(h q|r s)`, q, r, s over all N spinors.
`vtranpsog`, `vtran1sparse` and `expnd2p` exist only as objects of CFOUR's
program directories; the Makefile links `x2czvec/vtranpsog.o`,
`vtran1sparse.o`, `expnd2p.o` and `-lx2cint` (nothing is copied; a rebuilt
CFOUR tree must have these objects).  CFOUR's routine returns `conj((h q|r
s))` as `v(s, r, q)` in its order of the spinors (measured against integrals
made with PySCF on CFOUR's spinors: 1.4e-14).  Its debug lines are kept out
of the output (`xz_stdout`).

From these, per held spinor (the integrals are transformed twice and never
stored):

- sweep 1: `bf`, `kf` of `density_opt._frozen_stage` - the blocks of g2
  with the active part of `(h q|r s)`, T1-dressed (`'yxy'`) and, conjugated,
  dressed the other way (`'xyx'`), antisymmetrized to `~<h a||b c>`: about
  thirty contractions as loops (`xzdfrzb`, `xzdfrzk`).  The l3 t3 part of
  the block abcd needs Gamma on both of its sides here, so with held spinors
  pass C also accumulates `sum Lv Tv^T` (the size of `<ab||de>`: its diagonal
  pair blocks);
  and the Fock matrices `F[M][h][q] = sum_rs <h s||q r> M[r][s]` of five
  densities (tau^T, conj(tau), A^T, conj(A) of `frozen_terms`, and D0 - n);
- then `G[h][x] = (B + conj(K)) / 2 + (n_x - n_h) V[h][x]` between a held
  and an active spinor, `G[dropped][frozen] = V`, and the rotations `y[h][x] =
  -G[h][x] / (e_h - e_x)` with the active spinors of the kind of h (none
  between spinors closer than 1e-9 Hartree; a warning if the gradient there
  is above 1e-8);
- sweep 2: the Fock matrix of `y - (D0_vo + D0_ov)` in the rows and columns
  of a held spinor, and of y between the active spinors;
- `xfull[a][i] += F[y][a][i]`; `xfull[dropped][i] = G + F`, `xfull[a][frozen]
  = -conj(G[frozen][a]) + conj(F[frozen][a])`, `xfull[dropped][frozen] = V +
  F`; `dpopo = y[frozen][active occupied]`, `dpvpv = y[dropped][active
  virtual]`.

Integrals with two or three held indices (the block [dropped, frozen], the
Fock matrix of y in a held row) come out of the same `(h q|r s)` because q,
r, s run over all spinors: no Fock build in the AO basis is needed, unlike
in the Python program.

**Where it applies.**  The held stage needs the spinors of CFOUR's SCF, and
those are complex and carry no symmetry labels today: every run through
`xx2czvec` is the complex dense instance.  The labelled real directories of
the timings (made from a PySCF SCF, `bench/cfour/labelled_run.py`) have no
`scf2ccmo` of their spinors, so their relaxed density cannot be completed
by `xx2czvec`; for them the files of the active spinors alone can be
written (`den_active 1`, a test hook: xfull, dpoo, dpvv of the active
space, the records with a frozen index empty), which is what the block and
real code paths were validated with.  **Not validated:** the held stage in
blocks of the labels and in real arithmetic (the code is the same typed
source; no run directory has both labels and CFOUR's spinors).

### Memory

All of it in the work array, in the plan (`xzpeak_cc` runs `xzden` dry after
the iterations; "the least that ... density takes", `work array used` =
plan to the word in every run below, also at the floor).  Regions: the
integrals, T1, T2, T3, l1, l2, l3 (two arrays of the size of T3, no
residual, no DIIS vectors), then

- to the end of the stage: the dressed small blocks, t2 and l2 on pairs,
  `~<mb||Q>`, `~<Q||gi>`, `C3`, the result X of pass B whose first rows
  become the block aibc (four arrays of about o v^3 / 2), the ring density
  and its relatives ((o v)^2 each), the pass operands;
- per pass: the half-expanded batches of `nbp` occupied pairs (two in pass
  A, one in pass B), the two gathered batches and a product of `nbt`
  triples in pass C, the Gram matrix C(o,3)^2;
- at the end: an o v^3 / 2 array for the folded products, and 2 o v^2
  elements per thread for the block aibc.

With held spinors, in addition: `sum Lv Tv^T` (the size of `<ab||de>`), the
list of AO integrals (2 words per integral; the region is planned for
`ntri (ntri + 1)` words, ntri = n (n + 1) / 2 for n AO functions = N / 2),
the spinors, per held spinor `(h q|r s)` twice (CFOUR's complex array and
the typed one in the order of the program: 2 N^3 + NW N^3 words), the
half-transformed integrals (8 n ntri words), two arrays of (o + v)^3 for
the dressed active part, and six N^2 matrices.  For HBr / cc-pVQZ-DK with 10
frozen spinors (N = 196, n = 98, ntri = 4851): the AO list at most 188 MB
(the file `IIII` of that run is 35 MB), `(h q|r s)` 120 MB + 120 MB, the
dressed parts 2 x 103 MB (26 + 160 active, complex) - computed from the
sizes, not run at this size.

| case | mode | least for the density alone | default plan | T3 = l3 |
|---|---|---|---|---|
| H2O / 6-31G (10 + 16) | complex, dense | 12.4 MB | 37.8 MB | 1.1 MB |
| HBr / cc-pVDZ-DK, 28 frozen (8 + 28), active part | complex, dense | 29.2 MB | 102.3 MB | 2.9 MB |
| the same, labelled real spinors, active part | real, 6 classes | 11.4 MB | 14.5 MB | 0.25 MB |
| HBr / cc-pVQZ-DK, 10 frozen, 26 + 120, labelled real, active part | real, 10 classes | 3.64 GB (8 threads) | 5.67 GB | 655 MB |

In a run that also iterates (`ccsdt_density`) the floor is that of the
Lambda equations (H2O 23.2 MB, the labelled HBr case 18.7 MB): the density
stage is below it.  Run at the floor: H2O `density` (1549104 words), the
labelled HBr case `ccsdt_density` (2339488), H2O with DROPMO and `%vircut`
`ccsdt_density` with the held stage (1184944): the same files to 1e-16,
and 100 words below the error of section 4 with its numbers.  Batches of
1, 2 and 3 pairs and triples (`plan`): the same files.

### Validation

**The files against the Python program**, element by element (max
|difference|; `density_opt.kernel` on the same integral files with the
amplitudes and lambda of the Python program on the packed block, R from the
MO integrals, `cfour.handover_files`; login node unless a job is named):

| case | native mode | `xfull` | `dpoo` | `dpvv` |
|---|---|---|---|---|
| H2O / 6-31G (10 + 16) | complex, dense | 5.8e-17 | 1.7e-16 | 5.2e-18 |
| HBr / cc-pVDZ-DK, 28 frozen (8 + 28), active part | complex, dense | 6.8e-17 | 2.2e-16 | 5.2e-18 |
| the same, labelled real spinors, active part | real, 6 classes | 2.4e-17 | 5.6e-17 | 9.5e-18 |
| | real, dense (`labels 0`) | 2.0e-17 | 5.6e-17 | 2.2e-18 |
| | complex, 6 classes (`real 0`) | 1.9e-17 | 5.6e-17 | 6.9e-18 |
| the same with a phase on every spinor | complex, 6 classes | 2.5e-17 | 5.6e-17 | 1.1e-17 |
| HBr / cc-pVTZ-DK, 28 frozen (8 + 78), labelled real, active part (job 31789129) | real, 8 classes | 6.8e-17 | 5.6e-17 | 8.7e-18 |
| | real, dense | 6.6e-17 | 1.6e-19 | 4.3e-18 |
| | complex, 8 classes | 7.1e-17 | 5.6e-17 | 6.9e-18 |
| | complex, dense | 6.6e-17 | 5.6e-17 | 9.5e-18 |

(largest elements: `xfull` 0.009 to 0.014, `dpoo` 0.008 to 0.014, `dpvv`
0.004 to 0.006.)  The response density d1 itself agreed with the first run
of the first case (3e-16); the gradient needed one correction (the block
ijak was first assembled through an array of the wrong shape).

**The chain**: CFOUR's SCF and `xx2cvtran`, then in the same directory (the
same spinors) the Python Z-vector solver (`--zvector python`), the Python
hand-over (`--zvector cfour`) and the native program followed by `xx2czvec`
and `xsox2cprop`.  The seven native files against those of the Python
hand-over, `denfinal` of the two, the relaxed electronic dipole moment (z)
that `xsox2cprop` prints, the largest difference between its numbers:

| case | nf, nd | `xfull` | `dpopo` | `dpvpv` | `denfinal` | dipole z: Python Z-vector | Python hand-over | native + xx2czvec | properties |
|---|---|---|---|---|---|---|---|---|---|
| H2O / 6-31G, CCSDT | 0, 0 | 1.4e-13 | - | - | 5.3e-14 | 0.01207714055097 | 0.01207714054634 | 0.01207714054635 | 7.0e-12 |
| the same, DROPMO=1>1 | 2, 0 | 2.4e-14 | 8.2e-15 | - | 1.9e-14 | 0.01195694144940 | 0.01195694144946 | 0.01195694144946 | 1.0e-12 |
| the same, DROPMO=1>1, %vircut 1.5 | 2, 2 | 2.5e-14 | 8.1e-15 | 1.6e-15 | 3.5e-14 | 0.02344559047767 | 0.02344559047584 | 0.02344559047586 | 2.1e-12 |
| HBr / cc-pVDZ-DK, DROPMO=1>14 (8 + 28) | 28, 0 | 7.4e-14 | 1.2e-14 | - | 9.3e-14 | 1.07735527333876 | 1.07735527334354 | 1.07735527334354 | 5.7e-12 |
| the same, %vircut 2.0 (8 + 26) | 28, 2 | 7.3e-14 | 1.1e-14 | 5.5e-16 | 9.2e-14 | 1.07451246361978 | 1.07451246362169 | 1.07451246362162 | 5.3e-12 |
| HBr / cc-pVDZ-DK, all electrons (36 + 28; job 31789129, 24 threads, `CC_CONV=10`) | 0, 0 | 2.5e-12 | - | - | 9.2e-14 | 1.07498654471431 | 1.07498654470426 | 1.07498654470416 | 1.0e-10 |

(`CC_CONV=12` for H2O, 11 for HBr with frozen spinors; sizes of the entries: `xfull`
1e-2, `dpopo` 3e-5 to 2e-4, `dpvpv` 1e-2.)  The native route and the Python
hand-over differ by rounding; both differ from the Python solver by its
threshold, as in section 10.3, whose numbers these reproduce.  The first
run with frozen spinors gave these numbers.  `xx2czvec` converged in 15
cycles in all of them.  All these cases are in the range where `xx2czvec`
is right (section 10.4).

**Outside that range the program warns.**  The rule of section 10.4 is
applied to every run of the stage, e.g. for 26 + 20 active spinors of 56:

     @XZCCSDT-W, xx2czvec will NOT give the relaxed density of these files: with so few
                 active virtual spinors its response integrals are wrong (an unset work
                 offset in its source; CFOUR's own relaxed CC densities have the defect too).
                 1 + 3 C(nv,3) + nv (nv+1) + no (no+1) =  ...   is below   2 N^2 =  ...

The files are right in that case too (they do not depend on `xx2czvec`); it
is `xx2czvec` that cannot be used.  CFOUR's source was not changed.

Scripts and logs: `/home/xwang405/.claude/jobs/7ba4fcf7/tmp/natden/`
(`pyden.py`: the Python files and the comparison; `nroute.sh`, `jroute.sh`,
`ncmp.py`: the chain; `modes.sh`, `floor.sh`, `floor2.sh`, `batches.sh`,
`regr2.sh`; `den.sbatch` and `job_31789129/`; `vt/`: CFOUR's transformation
called from a test program, against PySCF).

**The existing methods**: CCSDT, [Q], (Q), [Q]_L, (Q)_L and the pseudo-energy
of the twelve runs of sections 7 and 11 (H2O, the frozen + vircut case, HBr
dense, the labelled case in four modes, with phases, shuffled) with the
binary before this work and the final one: identical to the 12 decimals
printed.

### The lock of the Lambda pass 1 (section 12): not the limit

Section 12 named the lock under which the density products of pass 1 are
added as the reason why that stage does not gain beyond 16 threads.  It was
replaced by tasks that own a range of rows of a density block and add to
it alone (no lock, no buffer per thread; commit e38fdeb) and measured
against the binary before, alternating in one job (31789129, c142, HBr 26 +
120 in blocks, three cycles; "pass 1: density", seconds for the three
cycles; the second repetition agrees within 2 %):

| threads | 8 | 16 | 24 | 48 |
|---|---|---|---|---|
| with the lock (before) | 4.72 | 2.66 | 2.19 | 2.33 |
| tasks on ranges of rows, no lock | 8.02 | 5.72 | 5.04 | 4.30 |
| Lambda iteration, with the lock | 10.40 | 6.35 | 5.42 | 4.56 |
| Lambda iteration, without | 11.52 | 7.32 | 6.32 | 5.35 |

Slower by 1.7 to 2.3: a task of the new kind is a sequence of thin products
(a few rows of l3 with the whole batch of T3 of one pair), and every task
reads that batch again.  The lock was not the limit - the stage is bound by
the memory traffic of the products.  **The locked loop was restored**
(commit 9fe25c8); the energies were identical with both.  **Not measured:**
accumulators per thread merged at the end (the other form of the same
idea; it keeps the square products and would remove only the lock, which
these numbers say costs little).

### Timings

One job (31789129, node c142, two sockets of Xeon Gold 6248R, `sbatch -p
parallel -N 1 -n 1 -c 48 --mem=180G`, snapshot of commit e38fdeb, run
directories in the node-local `/tmp`).  HBr / cc-pVQZ-DK, 10 frozen spinors,
26 + 120 active, m_j-adapted real spinors with labels (10 classes; T3 655
MB, 11 % of the dense array), `MEMORY_SIZE=150 GB`; the iterations three
cycles each (`ncyc 3`), the density stage on the amplitudes and lambda
of those cycles, active part (`den_active 1`: the directory has no
spinors of CFOUR).  Wall seconds; the two repetitions agree within 3 %:

| | 8 threads | 16 | 24 | 48 |
|---|---|---|---|---|
| CCSDT iteration | 7.6 | | 3.7 | |
| Lambda iteration | 10.4 | 6.4 | 5.4 | 4.6 |
| Lambda intermediates, once | 4.6 | | 2.6 | |
| **density and gradient stage** | **10.5** | 6.9 | **5.8** | 4.8 |
| - of it: writing the seven files | 0.01 | | 0.01 | |
| density stage, whole invocation (reading integrals, XZAMPS, XZLAMB) | 12.3 | 8.8 | 7.6 | 6.8 |

(CCSDT and Lambda iterations: the binary before this work, which the final
one equals in these routines.  The density stage was run with the binary of
the snapshot, whose pass A used the slower density tasks of the table
above: 2.71 s of the 10.5 s on 8 threads and 1.66 s of the 5.8 s on 24.
With the restored loop that part is 1.57 s and 0.73 s in a Lambda
iteration of the same job, so the stage of the final binary should be about
1 s shorter; **not measured.**)  Where the stage goes on 8 / 24 threads:
pass A 4.25 / 2.84 s (gather 0.73 / 0.60, ring density 2.71 / 1.66, stacked
products 0.81 / 0.58), pass B 0.92 / 0.69, pass C 2.17 / 0.94, the blocks of
g2 1.01 / 0.40, the gradient 0.98 / 0.39, dressing 0.53 / 0.27, Gram matrix
0.24 / 0.08.  The stage costs one Lambda iteration.  Work array used 5.67
GB = plan; peak resident set 6.0 GB.

`xx2czvec` at this size: CFOUR's own CCSD chain of HBr / cc-pVQZ-DK with 10
frozen spinors (26 + 160 active, 196 spinors, its own complex spinors;
`xx2cccsd` 77 s, `xx2chbar` 28 s, `xx2clcc` 62 s, `xx2cccdenocbp` 68 s on
48 threads), then `xx2czvec` alone on the files of `xx2cccdenocbp`:

| threads | 8 | 24 | 48 |
|---|---|---|---|
| `xx2czvec`, wall s | 11.2 | 10.9 | 14.0 |

(15 cycles; 2.1 GB resident; its integral files `intpphh2`, `intphhp2` 137
and 531 MB.)  Its time does not depend on who wrote `xfull`.

The price of CFOUR's spinors, same job: the chain of HBr / cc-pVDZ-DK with
all electrons (36 + 28, 24 threads) took 216 s in the native program
(`ccsdt_density`: CCSDT 3.91 s per cycle, 21 cycles; Lambda 6.66 s, 19
cycles; the density stage 4.86 s; 14.6 GB of the work array) - dense complex
arithmetic, because CFOUR's spinors have no labels - against 50 s for the
whole Python route, which runs on its own j_z-adapted spinors in blocks and
rotates the result to CFOUR's (section 10.2).  `xx2czvec` 1.6 s.

**Not measured:** the held stage at production size.  The one timing job
was submitted before the held stage existed, and a run through it needs
CFOUR's complex spinors without labels, i.e. the dense complex CCSDT of 26
+ 120 or more (66 s per cycle on 24 threads, section 12).  Small cases
(login node, 4 threads): H2O with 2 frozen and 2 dropped spinors 0.02 s for
the whole stage; HBr / cc-pVDZ-DK with 28 frozen spinors (64 spinors, 8 + 28
active) 0.90 s for the stage, most of it the 2 x 28 transformations and the
contractions per held spinor, against 0.07 s for the stage of the active
part.  The loops of `xzdfrzb` / `xzdfrzk` are (o + v)^4-like per held
spinor with OpenMP, not GEMMs; their cost at 26 + 160 with 10 frozen
spinors is not known.

### Not done

- (Q): `q.de` on the diagonal of the density and the oo and vv blocks of
  the gradient (section 10.6, step 3); the density of CCSDT(Q)_Lambda.
- The held stage in blocks of the labels and in real arithmetic: not
  validated (no directory with labels and CFOUR's spinors); its timing at
  production size; its contractions as GEMMs; batches of several held
  spinors per transformation (one at a time today, twice).
- The relaxed density of a labelled real calculation: needs an SCF of CFOUR
  in j_z-adapted (real) spinors, or a Z-vector solver of our own.
- `xx2czvec` is called by the user (or the front end) after the stage, not
  by the program; where the rule of section 10.4 fails the program warns
  and there is no native Z-vector solver to fall back on.
- The link to three objects of CFOUR's program directory `x2czvec` is by
  path (`FRZOBJ` of the Makefile): a CFOUR tree without them (a clean
  install of binaries and libraries only) does not link; they would have
  to move into a library of CFOUR at the final merge.
- The AO integrals in core (2 words per integral, as `xx2czvec` holds them)
  bound the held stage in memory for large basis sets; the stage of the
  active spinors has no such array.
- T3 or l3 in a mapped file; the pages of the work array touched in
  parallel (section 12).

## 14. (Q) over the ranks of a node: whole groups, a shared T3; the DIIS file

Three things were decided on 2026-10-09 and are in the program at this
commit: (A) the DIIS vectors stay all in memory or all in the file - no
split (section 4; nothing changed in the code, the rule is stated and its
cost measured below); (B) (Q) and (Q)_Lambda give whole groups to the ranks
(section 5); (C) the ranks of a node read one T3 (lambda3) from the mapped
files (sections 2, 4, 5).  One timing job on a whole node: **job 31790244,
node c426** (2 x 24 cores Xeon Gold 6248R, 189 GB; `sbatch -p parallel -N 1
-n 1 -c 48 --mem=180G`), HBr / cc-pVQZ-DK, 10 spinors frozen, labelled real
spinors written by `bench/cfour/labelled_run.py` (26 + 160, 10 classes; T3
1591 MB, 11.4 % of the dense array), run directories and `XZAMPS` in the
node-local `/tmp`, `MEMORY_SIZE=150 GB`; the binary of master before these
changes ("old", commit 429d755) and the present one ("new", 676d901)
alternating; Intel MPI on the one node (`I_MPI_HYDRA_BOOTSTRAP=fork`,
`I_MPI_PIN_DOMAIN=omp`, `OMP_PROC_BIND=close OMP_PLACES=cores`: each rank on
its own cores).  The validation of the final state is job 31790248 (node
c105, 24 cores) and the login node.  Scripts, logs and all outputs:
`/home/xwang405/.claude/jobs/7ba4fcf7/tmp/natq/` (`job.sbatch`,
`job-31790244.log`, `job_31790244/`; `d2.sbatch`, `d2-31790248.log`,
`d2_31790248/`; `tab.py`, `stages.py`, `fitcost.py`; `regrD.sh`, `tq.sh`,
`tc.sh`).

### What the earlier numbers were, and what (Q) over ranks really takes

The reason for (B) was the table "1 x 48, 2 x 24, 4 x 12, 6 x 8 ranks x
threads: 15.0, 8.1, 6.1, 6.8 minutes", read as "the finer split loses to the
repeated Xv".  Those were *estimates of a 0.5 % sample* (`qsample`), and
the sample of that binary was biased twice: it took the first quadruples of
every task (sectors that are not the average ones), and it formed Xv again
for every task.  **The whole calculation, measured** (wall time of `mpirun`,
reading the integrals included; converged amplitudes, CCSDT 20 cycles of
9.17 s on 48 threads):

| ranks x threads | old: wall | new: wall | old: sum over ranks / slowest | new | new: slowest rank above the mean |
|---|---|---|---|---|---|
| 1 x 48 | not run | 729.9 s | - | - | - |
| 2 x 24 | 413.2 s | 421.6 s | 1.99 | 1.99 | 0.3 % |
| 4 x 12 | 391.3 s | 383.4 s | 3.82 | 3.96 | 1.1 % |
| 6 x 8 | 386.0 s | 376.3 s | 5.76 | 5.90 | 1.6 % |
| 12 x 4 | 391.5 s | 386.4 s | 11.71 | 11.85 | 1.3 % |

E(CCSDT[Q]) = -2605.380721510858 and E(CCSDT(Q)) = -2605.381268110720 in
all nine runs (old and new; 1, 2, 4, 6 and 12 ranks).

So, plainly: **the finer split never lost to the repeated Xv, and (B) gains
2 %, not a factor.**  One rank per socket is 1.7 to 1.8 times one rank on both
sockets; beyond that the wall time of (Q) on this node does not depend on
how the 48 cores are cut into ranks (376 to 392 s), with the old binary as
with the new one.  What (B) changed, measured: the balance (sum over ranks /
slowest 3.96 of 4, 5.90 of 6, 11.85 of 12, from 3.82, 5.76, 11.71) and the
seconds of Xv (rank 0: 7.1 -> 4.0 s at 4 x 12, 8.6 -> 3.9 s at 6 x 8; all
ranks of the new binary together: 10.0, 17.3, 25.1, 77.6 rank-seconds for 2,
4, 6, 12 ranks, i.e. at most 1.7 % of the 12 x 375 rank-seconds).  Why one
node does not get faster beyond two ranks was **not determined** here (the
three parts of a quadruple - operands, products, reductions: 73, 122, 167 s
of rank 0 at 4 x 12 - all run at their full number of threads; the same
flat curve shows at 26 + 78 on 24 cores, below).  Ranks on separate nodes
were not measured (section 8).

What is formed by every rank and *does* grow with the number of ranks is
**Xo**: 2.2 s on one rank with 48 threads, but 5.3 to 8.0 s per rank at 4 x
12, 9.4 to 12.6 s at 6 x 8 and 21 to 25 s per rank at 12 x 4 (6 % of the
375 s; 290 rank-seconds), three times the Xv there.  It is also the largest
private array (below).  Sharing Xo as T3 is shared is the next step; it was
not part of this work.

**The sample** (`qsample 0.005`, 74 of 14950 quadruples; each run twice):
estimated wall time of the whole (Q) against the slowest rank of the whole
calculation above.

| ranks x threads | old: estimates | old: measured | new: estimates | new: measured |
|---|---|---|---|---|
| 1 x 48 | 745, 640 s | - | 760, 752 s | 725 s |
| 2 x 24 | 332, 349 s | 408 s | 421, 429 s | 415 s |
| 4 x 12 | 371, 359 s | 386 s | 420, 418 s | 378 s |
| 6 x 8 | 405, 427 s | 380 s | 448, 507 s | 372 s |
| 12 x 4 | 490, 492 s | 384 s | 442, 440 s | 375 s |

The old sample was off by -19 to +28 % and put the rank counts in the wrong
order; the new one (quadruples evenly spread over a task, Xv as in the
whole calculation) is high by 1 to 36 %, more with more ranks: with one or
two quadruples between two formations of Xv the sampled quadruples run on
cold caches.  It is an upper estimate to plan a job with, not a
measurement; a ranking of rank counts needs whole calculations.

### (B) The distribution and its cost model

The rule is in section 5.  For 26 occupied spinors no Xv is formed twice up
to 6 ranks (23 tasks = the 23 groups, 322 pairs (x, l)); with 12 ranks 33
tasks and 499 formations (the old list: 101 tasks), and the slowest rank of
the model 2.5 % above the mean.  Energies do not depend on the number of
ranks or on the list: 12 decimals at 26 + 160 (above), at 26 + 78 with 1, 2,
4, 6, 12 ranks in blocks and 1, 4 ranks dense (below), and in the small
cases with 1 to 4 ranks, with `qtasks 20`, interrupted (`qstop`) and
restarted with the same and - the files then being of another list and not
taken - with another number of ranks (`tq.sh`).

**The model was measured** with `qcost` (the wall time of every
quadruple and of every Xv of a whole (Q); `fitcost.py`), HBr 26 + 78:

- *Blocks (8 classes).*  The time of a quadruple follows the number of
  virtual quadruples of its sector; operation counts of the five products
  add nothing to the fit.  Per group l, model over measured time, for the
  groups from l = 15 on (93 % of the time): 0.95 to 1.03 on 6 threads (login
  node), 0.87 to 1.04 on 24 threads (job 31790248: small sectors do not get
  cheaper in proportion with many threads).  One cost for every quadruple,
  the former model: 0.89 to 1.21 and 0.91 to 1.21 - the largest group (l =
  25, 15 % of all quadruples) is overrated by a fifth; at 26 + 160 on 48
  threads the quadruples of l = 25 take 41.9 ms against 48 to 52 ms for l =
  13 to 24.
- *Dense.*  One cost per quadruple is right within 5 % per group (24
  threads, real).
- *Xv*, in seconds per element of its products over seconds per virtual
  quadruple: 0.0017 (blocks, 6 threads), 0.0025 (blocks, 24 threads), 0.0005
  (dense real, 24 threads).  The program uses one constant, 0.0012; Xv is at
  most 2 % of the work, so a factor of two in it moves the balance by a per
  cent.  Xv(x, l) for all x of the largest group costs about as much as a
  dozen of its 2300 quadruples (26 + 160, 12 threads: 54 ms per pair, 100 ms
  per quadruple), which is why a cut group forms its Xv again on every rank
  that has a piece instead of passing it on.

What the model achieves is the "sum over ranks / slowest" column above: the
slowest rank within 1.6 % of the mean at 26 + 160, within 5 % at 26 + 78 (6
x 4: 37.8 to 41.4 s).

### (C) One T3 per node

`XZAMPS` (`XZLAMB`) is mapped read-only by every rank; the T3 of the file is
the stored T3 of every mode, so no mode needs a conversion or a second file
(validated: real and complex, blocks and dense, with phases - `tc.sh`: all
of section 7's small cases, serial and with 2 and 3 ranks, from the files
and in one run with the iterations, against `qmap 0`; identical to 12
decimals).  The plan of a rank (26 + 160, real blocks):

| | old / `qmap 0` | mapped |
|---|---|---|
| (Q): least, per rank | 10923.9 MB | 9332.9 MB |
| (Q): with the default plan, per rank | 13159.4 MB | 11568.4 MB, and 1591.0 MB once per node |
| (Q)_Lambda: with the default plan, per rank | 15177 MB | 11994.7 MB, and 3182.1 MB once per node |

Measured resident memory (sampled every 5 s from `/proc/<pid>/status`; MB;
the sample with the largest sum; `RssAnon` is a rank's own memory, `RssFile`
the pages of mapped files that it has touched, which it shares):

| run | binary | RssAnon, sum of the ranks | largest rank | RssFile per rank | `Cached` of the node |
|---|---|---|---|---|---|
| (Q), 2 x 24 | old | 25579 | 12850 | 21 | 15721 |
| | new | 22609 | 11306 | 1671 | 15721 |
| (Q), 4 x 12 | old | 50015 | 12700 | 21 | 15106 |
| | new | 44123 | 11081 | 1671 | 15106 |
| | new, `qmap 0` (sample) | 47523 | 12017 | 21 | 15113 |
| (Q), 6 x 8 | old | 73985 | 12474 | 21 | 15137 |
| | new | 64804 | 10905 | 1671 | 15137 |
| (Q), 12 x 4 | old | 147134 | 12397 | 21 | 15231 |
| | new | 127089 | 10886 | 1670 | 15218 |
| (Q)_Lambda, 4 x 12 | old | 57705 | 14623 | 21 | 16745 |
| | new | 45781 | 11510 | 3320 | 16745 |

Every rank's own memory is smaller by T3 (by T3 and lambda3: 3.1 GB per
rank for (Q)_Lambda), every rank shows the mapped file as 1650 MB of
`RssFile`, and the page cache of the node is the same with 1 and with 12
ranks and with the old and the new binary - the file is in it once (it was
there before as well: the old ranks had read it through the same cache into
their own copies).  At 12 x 4 the node had 8 GB free with the old binary
and 28 GB with the new one.  The time is unchanged by the map (4 x 12
sample: 23.7 s mapped, 24.1 s private; the whole (Q)_Lambda: 508.2 s new,
509.0 s old).  The ranks other than 0 allocate 11.6 GB instead of 150 GB.

**24 x 2 was not run**: the least that a rank takes is 9.33 GB, and 24 of
them are 224 GB on a node of 189 GB.  T3 is no longer the reason - Xo is
(C(26,3) x 26 x 12720 numbers = 6.9 GB of the 9.3 GB, in every rank, dense in
every mode).  For the sizes named with the task: complex dense 26 + 160 has
a T3 of 27.9 GB and an Xo of 13.8 GB per rank; 26 + 320 a T3 of 225 GB,
which must then fit the page cache of the node once (it had to fit the work
array of every rank before) - neither was run.

**Not validated:** a run directory on a network file system (the warning of
section 2 and `qmap 0` exist; `xz_fs_local` was not exercised on such a
directory); more than one node.

### (A) The DIIS vectors in the work array or in the file

`ncyc 6`, the new binary, `plan "0 0 0 0 0 0 0 0"` with `diis m8`
(all in the work array) or `f8` (the T3 parts in `XZDIIS`); seconds per
cycle, mean of six and the sixth (5 vectors in the subspace by then; the
runs in blocks and dense on 48 threads were made twice):

| case | threads | in memory: mean, 6th | in the file: mean, 6th | DIIS stage, 6 cycles: memory, file | work array: memory, file |
|---|---|---|---|---|---|
| 26 + 160, real blocks (T3 1.6 GB) | 24 | 11.50, 11.60 (11.49, 11.60) | 14.17, 16.14 (14.19, 16.41) | 1.9, 17.0 s | 32.8, 9.0 GB |
| | 48 | 9.40, 9.82 (9.28, 9.11) | 12.96, 15.67 (12.90, 15.82) | 1.5, 19.4 s | |
| 26 + 120, real dense (`labels 0`; T3 5.8 GB) | 24 | 70.55, 71.77 | 80.11, 88.37 | 5.4, 60.1 s | 103.5, 15.9 GB |
| | 48 | 48.09, 48.83 (47.95, 48.19) | 59.04, 67.29 (59.19, 68.72) | 3.3, 69.9 s | |

With five vectors in the subspace a cycle with the file is 23 to 65 % longer
than one with the vectors in memory (dense 24 threads: 23 %, 48 threads: 38
%; blocks: 39 and 60 to 65 %), and it grows with every vector (by about 4 s
at 26 + 120; cycles with all eight were not run).  **Where** it costs is not where
the Python program lost it (docs/benchmarks.md, "The DIIS file and the last
cores": the write-back slowed the products of the following passes): here
the passes are unchanged - dense, 48 threads, six cycles: pass 2 154.4 s in
memory and 150.3 s with the file, pass 1 46.1 and 46.3 s, the pp-ladder
GEMM 62.8 and 63.1 s - and the whole difference is the DIIS stage itself,
which reads the earlier vectors record by record for the overlaps and again
for the extrapolation (69.9 s against 3.3 s; the file of 93.5 GB, 70 GB of
it in the page cache of the node at the end).  So "memory first" is right
for the native program too, and a plan that has the memory should use it:
the default plan did (`DIIS:   8 vectors in the work array`, 32.8 GB for 26
+ 160, 103.5 GB for 26 + 120 dense with `MEMORY_SIZE=150 GB`).  **Not
done:** a DIIS stage in the file with fewer passes over it (the overlaps of
the new error vector and the extrapolation in one pass per old vector would
halve the reads).

### Validation of the final state

(1) **The regression set** of sections 7, 11, 13 with the new binary (login
node, 4 threads; `regrD.sh`): `ccsdt(q)_l` of H2O / 6-31G, the frozen +
`%vircut` case, HBr / cc-pVDZ-DK dense, the labelled case in its four modes,
with phases (blocks and dense), shuffled, and `ccsdt(q)` of two - cycles,
E(CCSDT), E(CCSDT[Q]_L), E(CCSDT(Q)_L), E(CCSDT[Q]), E(CCSDT(Q)) identical
to the 12 decimals of the lines kept from before this work
(`native3/base.regr`; one pseudo-energy differs in its 14th decimal).  The
density stage (`ccsdt_density`) with the old and the new binary, the seven
files of `xx2czvec` number by number, largest |difference|: H2O 8.7e-19;
H2O with frozen and dropped spinors 6.9e-18 (`dpopo`, `dpvpv` identical);
HBr / cc-pVDZ-DK with 28 frozen 1.7e-18 and with 28 frozen, 2 dropped
1.1e-19 (`dpopo` 2e-28, `dpvpv` 1e-25); the labelled case, active part, in
blocks 5.6e-17, dense complex 1.4e-20; with phases 5.6e-17.

(2) **HBr / cc-pVTZ-DK, 10 frozen, labelled (26 + 78)**, converged
(`CC_CONV=9`), the native program in blocks and dense and the Python program
(`pytime.py`: the front end's calls, labels on) on the same files; job
31790248, node c105, 24 threads, the binary of 676d901:

| | native, blocks | native, dense | Python |
|---|---|---|---|
| E(CCSDT) | -2605.269299541779 | -2605.269299541779 | -2605.269299541779 |
| E(CCSDT[Q]) | -2605.268881367132 | -2605.268881367132 | -2605.268881367132 |
| E(CCSDT(Q)) | -2605.269257610936 | -2605.269257610936 | -2605.269257610936 |
| E(CCSDT[Q]_L) | -2605.268884407869 | -2605.268884407869 | -2605.268884407869 |
| E(CCSDT(Q)_L) | -2605.269268719835 | -2605.269268719835 | -2605.269268719835 |
| cycles, CCSDT / Lambda | 20 / 18 | 20 / 18 | 20 / 18 |

(the files of this job are not those of the table in section 12, hence the
other last digits; the three columns here are one set of files.)  The same
numbers from: (Q) and (Q)_Lambda in blocks with 2 x 12, 4 x 6, 6 x 4 and 12
x 2 ranks x threads, mapped; (Q) with `qmap 0`, serial and 4 x 6;
`ccsdt(q)_l` in one run on 3 x 8; dense (Q) and (Q)_Lambda on 4 x 6.  Wall
times of (Q) there (24 cores): blocks 78.8 s on one rank, 41.2, 41.0, 41.8,
43.4 s on 2, 4, 6, 12 ranks; dense 458 s on one rank, 310 s on 4 x 6;
(Q)_Lambda in blocks 107.3 s, and 62.2, 57.4 s on 2, 4 ranks.  The Python
times of this job are not a comparison of the programs (its Lambda ran on
the packed block) and are not quoted.

(3) `tests/test_cfour.py` (the hand-over formats; the worktree's libraries):
12 passed.

### Not done

- Xo formed once per node and mapped like T3 (above: the largest private
  array and the one repeated cost that grows with the ranks).
- Why (Q) on one node does not get faster beyond one rank per socket.
- Ranks on several nodes; a run directory on a network file system.
- The DIIS stage in the file with fewer passes; cycles with all eight
  vectors in the file were not timed.
- The estimate of `qsample` is an upper estimate (1 to 36 % high here).

## 15. The CCSDT iterations and the Lambda iterations over MPI ranks

    mpirun -np 8 -ppn 2 xzccsdt ccsdt          # 4 nodes, one rank per socket
    mpirun -np 4 -ppn 1 xzccsdt lambda         # 4 nodes, one rank per node

Until this commit only (Q) ran over ranks.  Now an iteration of the T3
equation and of the Lambda equations does: the target is the calculation
without symmetry (complex, dense), which takes days on one node.  What is
measured below says so (job, nodes); in short, for HBr 26 + 120 dense real
on Xeon Gold 6248R nodes: a CCSDT cycle takes 48 s serial, 40 s with one rank per socket
of one node (1.2 times), and on other nodes 56.7 s serial against 30.8 s on
two nodes (1.84), 17.4 s on four (3.26) and 15.1 s on four with two ranks
each (3.75); the Lambda equations the same (1.81, 3.28, 3.75); complex T3
1.92 and 3.51 on two and four nodes; in blocks of the labels 1.5 and 2.0,
because a quarter of an iteration there is not distributed.

### 15.1 The design as built

**Every rank runs the same step on its own copy of the amplitudes.**  The
step is `xzstep` (`xzlamstep`) as before, with three differences: the three
loops that are 95 % of a dense iteration - pass 1 and pass 2 over batches of
occupied pairs, the pp ladder over batches of occupied triples (in the
Lambda equations: pass 1 with the densities, pass 2, the ladder) - skip the
batches of other ranks; the ranks meet twice to sum what the batches made;
and the rows of T3 have owners.  `xzdist.F90` holds the list of batches, the
slices and the clocks; all MPI calls are in `xzmpi.c`.

*What a rank holds* (its own work array): the tables and the integrals - it
reads the files itself, as the ranks of (Q) do -, T1, T2, **all of T3**,
one region of the size of T3 for the residual (Lambda: T3, lambda3 and the
residual), the arrays of a step at the batch sizes of rank 0's plan, and
its slice of the DIIS vectors.  The small intermediates (dressed blocks,
F, W_ovvo, W_vooo, W_vvvo, the operands of the passes, the occupied
operator; for the Lambda equations everything that `xzlamim` makes once,
the pp operator in the place of `<ab||cd>`, `Xq`, `X2`) are **made by every
rank** from T1, T2 and the summed results of pass 1, with its own threads:
they are small, the code is the serial code, and nothing has to be sent or
kept consistent.  (The design named both ways; recomputing is the simpler
and is exact.  Its price is in 15.6: what is not distributed is done by
every rank with its own threads, so on *one* node it gets slower with more
ranks.)

*One iteration of the T3 equation:*

| step | who | what goes between the ranks |
|---|---|---|
| T1, T2 of rank 0 | rank 0 -> all | 2 small broadcasts |
| dressing, operands of pass 1 | every rank | - |
| **pass 1** | every rank: its batches of pairs | - |
| meeting 1 | all | sums on every rank (`MPI_Allreduce`, in place, in chunks): the T3 parts of W_vvvo (`X`: (o + v) o v(v-1)/2 numbers) and of W_vooo (`Y`: o(o-1)/2 v (o + v)), of the T1 residual (o v) and of the T2 residual (on pairs) |
| W_vvvo, W_vooo, W_ovvo, F, operands of pass 2 | every rank | - |
| sweep over the slabs of the dressed `<ab||cd>`; **pp ladder** | every rank dresses every slab; its batches of triples | - |
| CCSD part (new T1, T2), occupied operator | every rank (only rank 0's T1, T2 are used) | - |
| **pass 2** | every rank: its batches of pairs | - |
| hh ladder + F_oo | every rank: the rows of its slice | - |
| meeting 2 | all | **the residual to the owners of its rows** (`xzmpi_slices_add`) |
| Jacobi step, DIIS | every rank: its slice | the overlaps (a few numbers: sum, then rank 0's with the T1, T2 parts to all) |
| new T3 | all | **the slices to every rank** (`xzmpi_slices_put`) |
| energy, convergence | rank 0 | one word: "one more" or "done" |

The Lambda equations: the same with lambda3 in the place of T3 (T3 is sent
once, by broadcast, before the intermediates); meeting 1 sums the result of
the stacked products (`X`, `Y`) and the densities `Dr` ((o v)^2) and `D1`;
the Gram matrix of lambda3 and T3 is made by rows on the owners and summed
onto rank 0; meeting 2 sums the two small results of the ladder (`r2q`,
`r1x`) onto rank 0 and exchanges the residual; the residuals of rank 1 and
2 and the update of l1, l2 are rank 0's.

**Slices.**  Rank r owns the elements `soff(r) <= e < soff(r + 1)` of the
stored T3 - equal numbers of elements, cut at rows (occupied triples; in
blocks of the labels at the rows of a block).  A pass of this rank adds to
rows of every owner (pass 2 scatters; the pp ladder writes the rows of its
own triples only, which are not the rows of its slice), so every rank has a
whole residual array, and at meeting 2 the parts go home: in `nrank - 1`
rounds, in round d every rank sends what it has for rank + d and receives
what rank - d has for it (`MPI_Sendrecv`, chunk by chunk, a threaded add).
All ranks transfer at once, each moves (n - 1) / n of a T3 out and in, and
no rank relays or collects for another.  The same rounds carry the slices
of the new T3 to all ranks.  Then:

- the operator on occupied triples, the Jacobi step and DIIS are done by
  every rank on its rows (`xzslocc`, `xzslupd`, `xzdiis_upd` on a range of
  the stored array);
- **no rank holds more of the DIIS history than its slice**: 8 vectors and
  8 error vectors take 16 T3 / n per rank, in the work array if rank 0's
  plan has them there, else in the rank's own file `XZDIIS.<rank>` in its
  run directory ("memory first, all or nothing" is decided once, by rank
  0's plan, for all); the overlaps are sums of the ranks' dot products, the
  extrapolation is done slice by slice;
- rank 0 sends no T3 and receives no residual of other rows.

This is stage two of the design, and the default with more than one rank.
**Stage one** - rank 0 broadcasts T3, the residual is summed onto rank 0
(`MPI_Reduce` in place), rank 0 alone does the hh ladder, the Jacobi step
and DIIS and holds all vectors - was built first and validated, is still
there (`slices 0`) and is slower everywhere (15.6, 15.7): the reduction
of a T3 onto rank 0 ran at 0.4 to 1.9 GB/s, the exchange between owners at
4 to 12 GB/s.

**Deviations from the design as it was given.**  (1) The small T1/T2
dependent arrays are recomputed, not sent (above).  (2) A split of the
residual by rows was not used to avoid sending zeros: with the owners of
rows nothing is collected on one rank at all.  (3) Stage two became the
default, and with it the hh ladder, the Jacobi step and DIIS are
distributed too.  (4) T3 shared between the ranks of one node through a
mapped file was not built - it does not fall out for free here: the
amplitudes of the iterations change every iteration, so it would be a
shared *writable* segment (one per node, every rank writing its slice into
it and reading all of it), with the T3 of every rank addressed outside its
work array as (Q) addresses the mapped `XZAMPS`.  It would save (ranks per
node - 1) copies of T3 and the transfer of the slices within a node
(1.0 s of a 40 s cycle for 26 + 120 dense on 2 x 24, 15.6).

### 15.2 The chunked transfers (`xzmpi.c`)

An array of n double words goes in messages of at most `xz_chunk` words
(2^23 = 64 MB; `mpichunk`, capped at 2^27 so that no count in bytes
passes 2^31 - 1 either): `xzmpi_bcast_w` (from rank 0), `xzmpi_send_w` /
`xzmpi_recv_w`, `xzmpi_sum0_w` (the sum onto rank 0, in place),
`xzmpi_allsum_w` (the sum on every rank, in place), `xzmpi_slices_add` and
`xzmpi_slices_put` (above; every rank does the same number of chunks per
round, that of the longest slice, with empty messages where its own slice
is shorter).  Calls: `MPI_Bcast`, `MPI_Send`, `MPI_Recv`, `MPI_Sendrecv`,
`MPI_Reduce`, `MPI_Allreduce` with `MPI_IN_PLACE`, `MPI_Gather`,
`MPI_Barrier`, `MPI_Get_processor_name`, all with `int` counts of
`MPI_DOUBLE` or `MPI_BYTE` on `MPI_COMM_WORLD`.  The Fortran side passes
regions of the work array and 8-byte integers; it never sees an MPI handle
or an MPI integer.

`mpitest n` runs every routine on arrays of n words with patterns that
depend on rank and index (slices of unequal lengths, one of them empty) and
counts wrong words.  Run (login node): n = 1 000 003 with chunks of 4097
words on 1, 2, 3, 4, 6 ranks and of 1 word on 3 ranks; n = 300 000 000 (2.4
GB per array, more bytes than 2^31) with the default chunk and with the
largest; on four nodes n = 100 003 in chunks of 5000 - no wrong word.  The
whole regression set was run with chunks of 1000 to 7000 words (15.5).

A rank other than 0 that stops with an error (a file that is not there, a
work array too short for the plan) must not call CFOUR's `ERREX`: that
closes JOBARC as if the rank had opened it and writes an empty index
(`JAINDX`) over that of the run directory, which then cannot be read until
the chain is run again (found here; it held for the ranks of (Q) as well).
All errors of the program go through `xzerrex` now: `ERREX` on rank 0,
`MPI_Abort` on the others (their message is in `xzccsdt.rNNNN`).
`wfail` tests it; the directory runs again afterwards.

### 15.3 The batches and their ranks

The batches are those of the loops: `nbp` occupied pairs (pass 1, pass 2),
`nbt` occupied triples (ladder), never across the groups of the labels.
With more than one rank the plan keeps at least 8 batches of each kind per
rank (`bpr`; 26 + 120 dense on 8 ranks: 109 batches of 3 pairs, 65 of 40
triples).  `xzdeal` deals them, the most expensive first, **each to the
rank that is done with it first**; every rank makes the same list.

*The cost model* (`xzbcost1`): the operations of the products of a batch
by classes - pass 1: sum over the classes (d, m) of `n_d n_m n_Q (8 + n_e +
n_j)`; pass 2: sum over the groups Q of `n_delta n_Q (n_delta + 64)` (96 in
the Lambda equations); ladder: sum over the classes of the free index of
`n_c n_Q (n_Q + 16)`; Lambda pass 1 adds the densities `n_delta^2 n_Q`.
With one class every pair (triple) costs the same.  Measured with
`bcost` (seconds of every batch; `fitb.py`), seconds over model cost,
normalized, for the batches that hold 90 % of the time: 26 + 78 in blocks,
8 threads: pass 1 0.94 to 1.07, pass 2 0.97 to 1.01, ladder 0.96 to 1.12;
26 + 160 in blocks, one rank of 12 threads: 0.92 to 1.18, 0.95 to 1.02, 0.97
to 1.03; four ranks of 12 threads: 0.85 to 1.55, 0.74 to 1.09, 0.94 to 1.08;
one rank of 48 threads: 0.71 to 1.92 in pass 1 (a few large tasks do not
use 48 threads: section 12).  The imbalance of the ranks that the model
left (the slowest rank above the mean, job 31814617, one node, no
correction): dense 0.4 to 1.8 %; blocks 0.2 to 7.6 % (pass 1 the worst).

*The speeds of the ranks are measured, not assumed.*  On four nodes the
slowest rank was 7 to 15 % above the mean with a model that was right (the
dense case: every pair costs the same) - the nodes of the partition are not
equally fast (section 12: up to 40 %) - and the other ranks waited 2.0 to
2.7 s of a 17 s cycle (real) and 11 s of 54 s (complex).  So after every
step the ranks send rank 0 their seconds in their batches of each loop and
up to the two meetings (`xzrebal`); if the last rank came more than 3 %
(`rebal`) after the mean, rank 0 sends back, per rank, the model cost per
second in each loop and the seconds of everything else before each
meeting, and the batches are dealt again with those numbers (pass 2 of a
rank starts when its ladder is done).  That also takes care of whatever a
rank does besides its batches.  Measured on the login node with two ranks
of 2 and 6 threads (26 + 78 in blocks): 4.02 s per cycle without, 2.37 s
after one deal.  On four nodes of nearly equal speed it changed a
cycle by 0 to 6 % (15.7); on nodes of unequal speed it was not timed.  Energies do not depend on the deal: the regression
set with a deal after every step (`rebal 0.0001`, 15 to 18 deals per
case) is identical to 12 decimals.

### 15.4 Memory per rank

Planned by the dry run (section 4), printed by rank 0, checked node by node
against `MemAvailable`.  Work arrays as planned = as used (the table of the
ranks prints both the work array and the peak resident set of every rank),
MB, `MEMORY_SIZE` 150 GB, 4 DIIS vectors (`diis m4`; the default of 8
adds 8 more T3 / n per rank with slices, 8 T3 on rank 0 without):

| case (T3) | step | serial | stage one: rank 0, others | slices: every rank of 2 | of 4 | of 8 |
|---|---|---|---|---|---|---|
| 26 + 120 dense real (5.8 GB) | CCSDT | 56 583 | 56 583, 15 544 | 39 059 | 27 198 | 21 013 |
| | Lambda | 67 540 | 67 540, 26 501 | 50 016 | 38 135 | 32 132 |
| 26 + 160 real blocks (1.6 GB) | CCSDT | 19 820 | 19 820, 8 418 | 15 047 | 11 705 | 9 844 |
| | Lambda | | | 26 947 | 23 283 | 21 172 |
| 26 + 120 dense complex (11.7 GB) | CCSDT | 111 852 | 111 852, 29 774 | 76 803 | 53 437 | |

(resident sets: 85 to 97 % of these.)  With slices a rank of the CCSDT
iterations holds 2 T3 + 2 x (DIIS vectors) T3 / n + the integrals and the
batches; a rank of the Lambda iterations 3 T3 + `X2` and `Xq` (dense: 3.9 +
0.8 GB real at 26 + 120) + the same.  Floors and errors: rank 0 as before
(`the work array is too short ... Set MEMORY_SIZE to at least`); a rank
other than 0 has no floor of its own - it takes rank 0's plan and stops
with both numbers if its array does not hold it (ranks with different
numbers of threads are planned for the largest).  Tested (login node, 3
ranks): rank 0 at the floor of its plan (the DIIS vectors then in the files
`XZDIIS`, `XZDIIS.0001`, ...), below it (the error, exit 255), `diis f8`,
`diis m0`, ranks of 1 and 3 threads.

### 15.5 Validation

All against the serial program of before this work (tag
`xzccsdt-openmp-20261009b`, `bin_old`), to the 12 decimals printed.

(1) **The regression set** (sections 7, 11: `ccsdt(q)_l` of H2O / 6-31G, the
frozen + `%vircut` case, HBr / cc-pVDZ-DK dense, the labelled case in its
four modes, with phases in blocks and dense, shuffled; `ccsdt(q)` of two;
12 lines of `native3/base.regr`: cycles, E(CCSDT), E(CCSDT[Q]_L),
E(CCSDT(Q)_L), the pseudo-energy, E(CCSDT[Q]), E(CCSDT(Q))) -
`natmpi/regr.sh`, login node: identical with the serial build, and with
the MPI build on 1, 2, 3, 4 and 6 ranks, in the order of the work: nothing
distributed (`dist 0`), the pp ladder alone (`3`), the T3 equation
(`123`), everything; stage one (`slices 0`) and slices; chunks of 1000
to 7000 words; batches of 2 pairs and 3 triples; the DIIS vectors in the
files; a deal after every step.  The final commit: serial, 1 (`m1`), 2, 3,
4 ranks with slices, 2 and 3 ranks with `slices 0`.
(2) `ccsdt_density` on 2 to 4 ranks against the serial files of `xx2czvec`
(`tden.sh`: H2O, with frozen and dropped spinors, HBr with 28 frozen, the
labelled case in blocks and dense complex): largest difference 7e-17.
(3) The steps as separate invocations on 3 ranks (`ccsdt`, `lambda`,
`(q)_l`: `tsep.sh`).
(4) **HBr / cc-pVTZ-DK, 10 frozen, labelled, 26 + 78, converged**
(`CC_CONV=9`; job 31814617, node c393, snapshot 3eb4282), `ccsdt(q)_l`:

| | serial before | 1 x 48 | 2 x 24 | 4 x 12 | 2 x 24, 4 x 12 slices |
|---|---|---|---|---|---|
| blocks: E(CCSDT) | -2605.269299541783 | the same | the same | the same | the same |
| E(CCSDT[Q]_L) | -2605.268884407873 | | | | |
| E(CCSDT(Q)_L) | -2605.269268719839 | | | | |
| pseudo-energy | -0.29411064336175 | | | | |
| cycles CCSDT / Lambda | 20 / 18 | | | | |
| wall time of the chain / s | 146.1 | 149.4 | 90.2 | 87.2 | 83.9, 84.0 |
| dense (`labels 0`): the five numbers | the same as in blocks | the same | the same | the same | the same (4 x 12) |
| wall time of the chain / s | 1155.6 | 1175.7 | 795.0 | 811.5 | 757.5 (4 x 12) |

("the same": every digit printed.  Section 14 has -2605.269299541779 for
this case: another run of the SCF that writes the files, as said there; the
columns here are one set of files.)
(5) **Four nodes** (job 31814618, c292, c314, c395, c495; the run directory
in the shared scratch, a network file system - (Q) then maps `XZAMPS`
there, with its warning): `ccsdt(q)_l` of the labelled case and of HBr /
cc-pVDZ-DK dense on 4 x 1 ranks (chunks of 5000 words, `mpitest`) and 4
x 2 ranks, `ccsdt_density` on 2 x 2: all numbers those of the serial
program on the same files.

### 15.6 One node

Job 31814617, node c393 (2 x 24 cores Xeon Gold 6248R, 189 GB), snapshot
3eb4282 (before slices became the default and before the measured speeds:
"slices" below is `slices 1` of that binary), run directories in the
node-local `/tmp`, `ncyc 4`, 4 DIIS vectors in memory, Intel MPI on the
node (`I_MPI_HYDRA_BOOTSTRAP=fork`, `I_MPI_PIN_DOMAIN=omp`,
`OMP_PROC_BIND=close`).  Seconds per cycle, mean of cycles 2 to 4; the
serial program of before first.

| HBr, 10 frozen | serial before, 48 threads | the same, `numactl --interleave=all`, spread | 1 x 48 | 2 x 24 stage one | 4 x 12 stage one | 2 x 24 slices | 4 x 12 slices |
|---|---|---|---|---|---|---|---|
| 26 + 120 dense real, CCSDT | 48.08 | 44.46 | 47.95 | 43.90 | 45.98 | **40.07** | 41.21 |
| Lambda | 72.69 | 68.13 | 72.40 | 64.77 | 71.32 | **59.66** | 63.20 |
| 26 + 160 real blocks, CCSDT | 9.34 | | 9.57 | 9.26 | 10.70 | **8.20** | 10.11 |
| Lambda | 12.57 | | 12.55 | 11.44 | 11.95 | **10.49** | 11.05 |

One rank per socket gains 1.20 (CCSDT) and 1.22 (Lambda) over the serial
program as it is run, 1.11 and 1.14 over the serial program with its pages
interleaved; four ranks gain less.  **Why so little** - per cycle, dense
CCSDT, the serial program against a rank of 2 x 24 (which does half):

| stage | serial, 48 threads | 2 x 24, per rank | gain |
|---|---|---|---|
| pass 1 | 7.6 | 5.1 | 1.5 |
| sweep and pp ladder | 11.5 | 10.5 | 1.1 |
| pass 2 | 26.0 | 18.4 | 1.4 |
| the three | 45.1 | 34.0 | 1.33 |
| residual to its rows; T3 slices to the ranks | - | 1.0; 1.0 | |
| on its rows (hh ladder, Jacobi step, DIIS), CCSD part | 2.0 | 2.0 | |
| waiting for the other rank | - | 1.0 | |

Two ranks of 24 threads do the passes exactly as fast as two serial
programs of 24 threads would (section 14: 70.5 s per cycle on 24 threads,
i.e. 35 s for half) - the ranks do not disturb each other -, but the serial
program on 48 threads is already at 1.5 times its speed on 24: the GEMM of
the pp ladder uses the second socket fully, pass 1 and pass 2 (gathers and
reductions, bound by the memory) partly.  What is left to win on one node
is that NUMA part, 1.33 in the passes, and 5 s of transfers, row work and
waiting take half of it back.  (Q) gained 1.7 to 1.8 from one rank per
socket (section 14) because a rank of (Q) shares nothing and waits for no
one.  In blocks every rank also makes the small intermediates with its own
threads (below): with four ranks of 12 threads an iteration is slower than
the serial one.

**What is not distributed** (shares of a serial iteration, 48 threads, job
31814617):

| | 26 + 120 dense real | 26 + 160 real blocks |
|---|---|---|
| pass 1, pp ladder, pass 2: over the ranks | 15.9 + 24.0 + 54.0 = 93.9 % | 21.7 + 18.2 + 33.5 = 73.4 % |
| hh ladder, Jacobi step, DIIS: on the rows of a rank (slices) | 3.0 + 0.1 + 1.0 % | 1.2 + 0.2 + 2.4 % |
| the rest: made by every rank with its own threads | 2.0 % | 22.8 % (W_vvvo: t2 with `<ov||vv>` 11.0, CCSD part 2.2, W_vvvo bare 1.7, dressing 2.6, W_ovvo 1.0, ...) |

Dense, the limit of the scheme is 1 / 0.02 before communication; in blocks
1 / 0.23, because the dense o^2 v^4 and o^3 v^3 terms of the intermediates
(section 12) weigh there - they are not split over the ranks (not done:
the largest, the t2 `<ov||vv>` term of W_vvvo, is a loop over batches of
first indices and could be dealt out like a pass).  Lambda, dense, serial:
pass 1 39.0 %, pass 2 37.2 %, ladder 17.0 % (93.2 % over the ranks), Gram
matrix 2.0 % and hh ladder 2.1 % (on the rows of a rank), the rest - the
residuals of rank 1 and 2 on rank 0 - 1.7 %.

### 15.7 Several nodes

Job 31814618: four nodes c292, c314, c395, c495 of the partition `parallel`
(`sbatch -N 4 --ntasks-per-node=1 -c 48 --mem=180G`: one Slurm task per
node with all its cores - not `--ntasks-per-node=2 -c 24` as proposed,
because Intel MPI's launcher then gets one task of 48 cores per node and
pins the `-ppn` ranks itself: `cores allowed: 0-47` for one rank per node,
`0-23` and `24-47` for two, printed by the job), Intel MPI 2021 started by
`mpirun -np N -ppn P -hosts ...` through Slurm (`srun` in the PATH),
**fabric: libfabric provider `mlx`** (`I_MPI_DEBUG=5`), snapshot 3eb4282,
`ncyc 4`, 4 DIIS vectors.  **Run directories: the node-local `/tmp` of
every node** - made on the first node (157 s), carried as one tar file
(5.7 GB) through `/scratch16/lcheng24/xwang405` and unpacked on all four
(10 s for both), the scratch emptied again.  The serial program of before
first, on the first node (c292 is 18 % slower than c393 of 15.6).  Seconds
per cycle, mean of cycles 2 to 4; in brackets the speedup over the serial
program on the first node.

| nodes x ranks x threads | 26 + 120 dense real: CCSDT | Lambda | 26 + 160 real blocks: CCSDT | Lambda | 26 + 120 dense complex: CCSDT |
|---|---|---|---|---|---|
| serial before, 48 threads | 56.7 | 85.7 | 12.38 | 16.52 | 188.1 |
| 1 x 1 x 48 | 56.7 | 85.8 | 12.13 | 16.90 | 186.2 |
| 1 x 2 x 24, stage one | 52.3 | 77.1 | 12.22 | | |
| 2 x 1 x 48, stage one | 34.9 (1.63) | 52.3 (1.64) | 9.18 (1.35) | | 113.9 (1.65) |
| **2 x 1 x 48, slices** | **30.8 (1.84)** | **47.4 (1.81)** | **8.06 (1.54)** | **10.29 (1.61)** | **97.9 (1.92)** |
| 2 x 2 x 24, stage one | 40.1 (1.42) | 55.5 (1.54) | 10.09 (1.23) | | |
| 4 x 1 x 48, stage one | 25.4 (2.24) | 35.5 (2.42) | 8.32 (1.49) | | 78.3 (2.40) |
| **4 x 1 x 48, slices** | **17.4 (3.26)** | **26.2 (3.28)** | **6.15 (2.01)** | **7.09 (2.33)** | **53.6 (3.51)** |
| 4 x 2 x 24, stage one | 23.4 (2.42) | 32.9 (2.61) | 7.59 (1.63) | | |
| **4 x 2 x 24, slices** | **15.1 (3.75)** | **22.8 (3.75)** | **5.84 (2.12)** | **6.89 (2.40)** | not run |

(the complex case: CCSDT only; its Lambda part and two ranks per node were
cut from the job to save machine time.)

Where a cycle goes on four nodes (slices, seconds per cycle, the slowest
rank; "waiting" is the largest time a rank stood at a meeting):

| | pass 1 | sweep, ladder | pass 2 | waiting | sums of pass 1 | residual to its rows | T3 slices | on its rows |
|---|---|---|---|---|---|---|---|---|
| dense real, 4 x 1 x 48 | 2.2 | 4.1 | 7.9 | 0.4 + 2.7 | 0.12 | 0.70 | 1.0 | 1.0 |
| dense real, 4 x 2 x 24 | 1.5 | 3.3 | 5.3 | 0.3 + 2.0 | 0.20 | 1.5 | 1.7 | 0.8 |
| dense complex, 4 x 1 x 48 | 6.7 | 12.2 | 26.6 | 0.9 + 11.1 | 0.23 | 1.35 | 1.3 | 3.3 |
| dense real Lambda, 4 x 1 x 48 | 8.4 | 4.2 | 8.4 | 1.0 + 4.2 | 0.14 | 0.76 | 1.2 | 3.7 |

**The transfers** (measured rates; a T3 of 5.84 GB real, 11.68 GB complex):

| | on one node (shared memory) | between nodes (`mlx`) |
|---|---|---|
| stage one: T3 from rank 0, into a receiving rank | 3.1 to 4.2 GB/s | 9.8 GB/s (2 ranks), 5.0 to 5.5 (4), 2.7 (8) |
| stage one: the residual summed onto rank 0 (`MPI_Reduce`, in place) | 1.1 to 1.9 GB/s | 1.5 GB/s (2 ranks), 0.7 to 0.9 (4, 8), 0.44 (2 nodes x 2) |
| slices: the residual to the owners of its rows, per rank | 4.1 to 5.8 GB/s | 11.4 to 11.8 GB/s (2 ranks), 8.3 to 8.5 (4), 3.9 (8) |
| slices: the slices of the new T3, into a rank | 3.5 to 3.8 GB/s | 7.6 GB/s (2), 5.6 to 7.1 (4), 3.5 (8) |

(GB of one whole T3 per rank over the seconds of the transfer; with slices
a rank sends and receives (n - 1) / n of a T3, in n - 1 rounds.)  In a
cycle of the dense real case on four nodes the two exchanges are 1.7 s of
17.4 (10 %), for complex T3 2.7 s of 53.6 (5 %); the sums of pass 1 0.1 to
0.2 s.  Stage one spent 7.0 s (real) and 17 s (complex) per cycle in the
reduction onto rank 0 alone.

**What this binary did not have, and the final one has** (the numbers
above are from snapshot 3eb4282; the last changes came out of them):
slices as the default (no number changes: the "slices" rows are that
mode); the batches dealt by the measured speeds of the ranks (15.3) - the
"waiting" column above is what it is for: 2.0 to 2.7 s of 15 to 17 s
(real), 11 s of 54 s (complex), 4.2 s of 26 s (Lambda) on four nodes; the
Gram matrix of the Lambda equations by rows on every rank (1.5 s per cycle
of rank 0 at this size, part of the Lambda waiting).  So the four-node
rows of the dense cases should improve by up to the waiting (an upper
bound: 3.26 -> 3.9 real, 3.5 -> 4.4 complex on 4 x 1), the two-node and
one-node rows hardly (waiting 0.5 to 1 s there).  

**The final commit** (a65ebe7: slices by default, batches by measured
speeds, the Gram matrix by rows; snapshot `snap8`), one job each, the two
main cases, the serial program of before first; seconds per cycle, mean of
cycles 2 to 4, in brackets the speedup over the serial program of the job.

*Four nodes* - job 31827703, nodes c097, c102, c106, c110 (the layout, the
launch and the fabric of job 31814618; 38 minutes), small cases first: the
energies of the serial program on 4 x 1 and 4 x 2 ranks.

| nodes x ranks x threads | 26 + 120 dense real: CCSDT | Lambda | 26 + 160 real blocks: CCSDT | Lambda |
|---|---|---|---|---|
| serial before, 48 threads | 52.32 | 79.41 | 11.09 | 15.41 |
| 2 x 1 x 48 | 28.59 (1.83) | 43.30 (1.83) | 7.63 (1.45) | 9.82 (1.57) |
| 4 x 1 x 48 | 16.83 (3.11) | 24.78 (3.20) | 6.18 (1.79) | 6.92 (2.23) |
| 4 x 1 x 48, `rebal 0` | 16.90 | 25.05 | 6.09 | 7.37 |
| 4 x 2 x 24 | 14.55 (3.60) | 19.95 (3.98) | 5.83 (1.90) | 6.87 (2.24) |

These four nodes were nearly equally fast (measured speeds of the ranks
0.93 to 1.08 of the mean; the batches were dealt again 2 to 4 times in 4
cycles), so the measured speeds changed little: the largest waiting per
cycle went from 0.85 to 0.69 s (CCSDT) and from 1.09 to 0.67 s (Lambda,
dense), and a cycle by 0 to 6 %.  The 11 s of waiting of the complex case
on the nodes of job 31814618 were **not** measured again with this commit:
whether the speeds take them back as the two-rank test of 15.3 suggests is
open.  The speedups are those of the earlier binary on other nodes within
5 %; the Lambda equations on 4 x 2 gained (3.75 -> 3.98: the Gram matrix).

*One node* - job 31827786, node c668; first HBr 26 + 78 in blocks,
converged, `ccsdt(q)_l`: the serial program of before, 1 x 48, 2 x 24, 4 x
12 and 4 x 12 with `slices 0` give E(CCSDT) -2605.269299541781,
E(CCSDT[Q]_L) -2605.268884407871, E(CCSDT(Q)_L) -2605.269268719836, the
pseudo-energy -0.29411064336159 and 20 / 18 cycles, all five alike (the
files of this job; wall times 146.1, 151.4, 84.7, 83.4, 87.3 s).

| | serial before | interleaved | 1 x 48 | 2 x 24 | 4 x 12 |
|---|---|---|---|---|---|
| 26 + 120 dense real, CCSDT | 46.72 | 44.33 | 46.32 | **39.49 (1.18)** | 41.10 |
| Lambda | 73.83 | 67.81 | 73.24 | **59.10 (1.25)** | 61.93 |
| 26 + 160 real blocks, CCSDT | 9.25 | | 9.23 | **8.19 (1.13)** | 10.13 |
| Lambda | 12.57 | | 12.77 | **10.64 (1.18)** | 11.10 |

(the complex dense case on one node was not run: the job was ended there
to save machine time.)


### 15.8 An estimate for benzene / cc-pVDZ (30 + 186, complex, dense)

**An estimate, not a measurement.**  From the complex dense case that was
measured (26 + 120, c292: 188 s per cycle serial on 48 threads; per cycle
pass 1 26 s, sweep and ladder 48 s, pass 2 104 s, the rest 8 s), scaled by
the operations of each stage (pass 1 o^3 v^3 (o + v): 8.5; ladder o^3 v^5:
13.7; pass 2 o^4 v^4: 10.2): a serial cycle of about 2000 s on a 48-core
node of this partition (the run that prompted this work reports about an
hour on 16 cores).  T3 is 68.5 GB.  With slices, one rank per node:

| nodes | passes / n (+ 3 % imbalance) | made by every rank (2 %) | exchanges (68.5 GB (n-1)/n out and in, at 7 to 8.5 GB/s) and sums | a cycle | speedup |
|---|---|---|---|---|---|
| 4 | 500 s | 40 s | 20 s | about 560 s | 3.6 |
| 8 | 250 s | 40 s | 25 s | about 315 s | 6.3 |

assumptions: the measured rates hold for transfers twelve times larger and
for 8 ranks at one rank per node (measured: 4 ranks at one per node, 8 at
two per node, where the rate per rank halved); the speeds of the nodes are
balanced as in 15.3 (measured on nearly equal nodes only).  **Memory**: a
rank holds 2 T3 = 137 GB and about 15 GB of integrals and batches - it fits
a 189 GB node, which the serial run does not (its DIIS vectors then go to a
file of 1.1 TB) -, and its slice of the DIIS history, 16 x 68.5 / n GB: 274
GB (4 nodes) or 137 GB (8), i.e. in the file `XZDIIS.<rank>` on the
node-local disk of every rank (the plan decides), read by n ranks at once
where the serial program reads all of it.  The Lambda equations need 3 T3
and `X2` (33 GB) per rank, 250 GB: they do not fit these nodes in any mode
(not done: `X2`, `Xq` and the T3 of the Lambda equations as rows of the
owner only).

### 15.9 Intel MPI and Open MPI

The source uses nothing of one vendor (15.2); `make MPI=1` needs only the
compiler wrappers.  Specific to Intel MPI in what was run:

- the build: the wrappers `mpiifort`, `mpiicc` (`LD`, `MPICC` of the
  Makefile; Open MPI: `mpifort`, `mpicc` wrapping ifort / icc, and its
  `mpi.h` - the program has no Fortran MPI module to match, so the 8-byte
  integers of CFOUR do not matter to either library);
- the launch: `I_MPI_HYDRA_BOOTSTRAP=fork` (one node without `srun`),
  `-ppn`, `-hosts`, `I_MPI_PIN_DOMAIN=omp` (a rank is pinned to
  `OMP_NUM_THREADS` cores), `I_MPI_DEBUG=5` (prints the fabric),
  `I_MPI_WAIT_MODE`.  Open MPI: `mpirun --map-by ppr:2:node:PE=24
  --bind-to core` (or `--map-by socket:PE=24`) for the placement, `--host`
  for the nodes; the fabric is chosen by its own components (UCX on this
  hardware);
- both libraries spin while a rank waits in a blocking call, so a rank
  that waits at a meeting uses one core, not its threads.

**Not run with Open MPI** (no Open MPI of the Intel compilers was at hand);
the transfer rates above are Intel MPI's, in particular the slow
`MPI_Reduce` of large arrays in place.

### 15.10 Not done

- The batches by measured speeds on nodes of unequal speed, and the
  complex case, with the final commit (15.7: timed on four nearly equal
  nodes, real cases only).
- T3 shared by the ranks of one node (15.1); the intermediates of the
  blocks split over the ranks (15.6); `X2`, `Xq` of the Lambda equations
  by rows of the owner (15.8).
- The sums of pass 1 use `MPI_Allreduce` (about 1 GB/s; 0.1 to 0.5 s per
  cycle here, about 2 s estimated for benzene); stage one keeps
  `MPI_Reduce`.
- (Q) after the iterations still needs a run directory that all nodes see
  (the ranks read `XZAMPS`); with node-local directories run `ccsdt` and
  `lambda` there and (Q) from a shared one.
- The first cycle waits for the other ranks to read the integrals (they
  start when rank 0 has planned: 3 to 7 s here, on its own clock).
- Open MPI; more than four nodes; more than 8 ranks.

### 15.11 The sigma vectors of EOM

The excited states use this machinery for a second kind of step: the kinds
of batches 7 to 12 of `xzdist.F90`, dealt per label shift (`xzdeal_eom`,
`xzrebal_eom`), the sums of pass 1, rows with owners.  Section 16.7.

## 16. EOM-EE-CCSDT: excited states

    CALC=CCSDT,EXCITE=EOMEE         # CCSDT, then the excited states (or the
                                    # states from XZAMPS of an earlier CCSDT run)

Excitation energies and right-hand vectors of EOM-EE-CCSDT on the converged
amplitudes, in every mode of the program: the packed block or the blocks of
the labels, real or complex.  In blocks a state has a **label shift** D -
its excitation operator changes the sum of the labels of the spinors by D
(`r1[i][a]` where l_a - l_i = D, `r2`, `R3` likewise with the sums) - and
every shift is an eigenproblem of its own in the blocks of that shift.  With
the labels 2 m_j of `dgcosym` (section 6) D = 2 (Omega' - Omega): for a
closed-shell reference 2 Omega of the state.  D = 0 is the totally symmetric
case and runs through the same code.  The Python program
(`src/zccsdt/cc/eom_opt.py`, `eom.py`, `t3shift.py`; docs/eom.md) is the
reference of the sigma vector and of every root below; the solver is not
its block Davidson but CFOUR's (16.3).

Not here: IP / EA, left-hand vectors, properties, CFOUR's CIS guess.  Over
MPI ranks: 16.7.

### 16.1 Input and output

The input is CFOUR's where CFOUR's two-component EOM program `xx2ceomcc`
has a keyword or a section for the thing (docs/eom.md, section 9), read the
same way: the flags of joda from JOBARC, the sections from the ZMAT (a line
with the tag, the value on the next line; `xzeomd.F90`: `xzeominp`).

| what | keyword / section | default |
|---|---|---|
| threshold | `ESTATE_CONV=N`: the largest element of the residual of a normalized vector at most 10^-N; the section `%eomthrs` (a real number) overrides | 10^-5 |
| iterations at most | `ESTATE_MAXCYC` | 40 |
| roots | `%eomnroot` - of every shift that `%xzeom` gives without a number | 1 |
| vectors of the subspace at most | `%eomndav0` (in CFOUR the size of its arrays only: `eomene2` compacts at a hard-wired 20 + roots + 2) | max(20 + roots + 2, 4 roots + 8); a given value at least 2 roots + 2 (40 roots in 50 vectors: 21 converged after 200 iterations) |
| the label shifts | `%xzeom` (ours): lines `shift [roots]` until a line that is not one or two integers; shifts in the labels of the run, as the output lists them, no sign convention beyond that | shift 0 |
| the start | `%xzeomguess` (ours): 0 unit vectors on the lowest single excitations of the shift (CFOUR's guess without CIS); 1 the roots of the singles-doubles block from these unit vectors first | 1 |
| the vectors of the roots | `%xzeomvec` (ours): 0 no file `XZEOMR` | 1 |

    ESTATE_CONV=6,ESTATE_MAXCYC=60          (in the *CFOUR(...) line)

    %eomnroot
    3

    %xzeom
    2
    4
    0 5

(three roots of the shifts 2 and 4, five of shift 0).  For a reference that
is symmetric under time reversal the states of -D are the time-reversed
states of +D with the same energies (measured: 2e-12): give one of the two
unless both are wanted; the program computes what the section lists.  A
shift that is not a difference of sums of the labels, or that no single
excitation has, is left out with a warning.  **Without blocks** (no labels,
`labels 0`) the states of all shifts are roots of one problem and the
program computes the sum of the roots asked for.

Output, per shift: the size of a vector, where the subspace is, CFOUR's
line per iteration and unconverged root (`EOM cycle`, the cycle, the vectors
of the subspace, the real part of the eigenvalue, the largest element of the
residual, the threshold: `eomconv` itself in complex arithmetic), then per
root the excitation energy in Eh, eV and cm-1, the largest element and the
norm of its residual (computed again from the final vectors), the cycle in
which it converged, the weights of R1, R2, R3 (the squared norms of the
parts of the normalized vector on i<j, a<b etc.), the total energy, and the
largest elements of R1 as `i -> a` with the number of the spinor in the
order of the SCF (`scf2ceig`), its label and its energy.  At the end one
table of all states.

Where a later program finds the results:

| | |
|---|---|
| `XZEOM` (text) | a line per state: label shift, root, omega / Eh, total energy / Eh, the largest element and the norm of the residual, the weights of R1, R2, R3, converged (T / F), the largest element of R1 as the spinors i -> a (numbers of the SCF) and its absolute value; two comment lines with E(SCF), E(CCSDT) |
| JOBARC `TOTENER2` | the total energy of the lowest converged state (the record that `xx2ceomcc` writes); `TOTENERG` stays E(CCSDT) |
| `XZEOMR` (unformatted stream) | `'XZEOMR01'`, 5 integers of 8 bytes (occupied, virtual, words per element, classes, the step of the labels), then per root: the shift in classes, the root, the elements of the vector (3 integers), the real and the imaginary part of omega, the vector `r1[i][a]`, `r2[P][Q]` on i<j, a<b, `R3` in the blocks of the shift, normalized, in the order of the spinors of the run |

### 16.2 The sigma vector (`xzeom.inc`)

`sigma = <mu| [Hbar, R] |0>` is the Jacobian of the amplitude equations on R
(docs/eom.md 1 to 3), and in the T1-dressed factorization of the CCSDT step

    sigma_3 = residual(R3; W_ring, W_pp, O_hh | W'_vvvo x t2, W'_vooo x t2)
            + residual(T3; W'_ring, W'_pp, O'_hh | W_vvvo x r2, W_vooo x r2)

with `W' = W[H'; t2] + W_linear[H~; r2, R3]` and `H' = [H~, R1]`: the
passes of the T3 equation twice with other operands, pass 1 over R3 (all
four outputs) and over T3 with H' (its T2 part).  No kernel was written or
changed; `zc_vtransform` and `zc_vprecond` of `csrc/t3pack.c` (the solver)
got interfaces.

**Once** (`xzeomim`): one amplitude step without its T3 residual, its
operands kept (`xzdata`: `efoo` ..): the dressed Fock matrix and small
blocks, the dressed `<mb||de>` on pairs, the bra-dressed `<ab||ei>`, W_ovvo,
F_vv, F_oo, W_oooo, `<mn||Q> t2[P][Q]`, the operands of pass 1, W_vvvo,
W_vooo, the operator on occupied triples, and - as for the Lambda equations
- **the pp operator of the T3 equation in the place of the bare `<ab||cd>`**
(a run that needs the bare integrals afterwards reads the files again).

**Per vector** (`xzesig`), in the order of `eom_opt.sigma`:

| step | how |
|---|---|
| H' (Fock matrix, `<mn||ie>'`, `<mn||ij>'`, `<ma||di>'`, `<ma||ij>'`) | loops and one `zc_pair_t1_contract` (`xzeh1`, `xzehs`) |
| W'_ovvo, F'_vv, F'_oo; W'_oooo | the routine of the ground state (`xzweff`) with (r2; H'): it is linear in both; `xzewkl` |
| sigma_1, sigma_2 without the pp terms | `xzes1`; `<ab||ij>'` (`xzes2g`); the terms that are bilinear in an amplitude and an intermediate by one routine called twice, (r2; W) and (t2; W') (`xzeb2`, `xzefvv`: GEMMs on the layouts of the CCSD part) |
| pass 1 over R3, over T3 with H' | `zc_t3_pass1w` / `xzbpass1` (in blocks the second without the rows of the operand that are zero) |
| W'_vvvo | `- P(ab) r1 W_ovvo` (the loop of `xzbv`), `W_pp r1` less its F_vv part, the T2 terms of W_vvvo with r2 (`xzwvoovo`, `xzwvx` as they are), the T3 part from pass 1 |
| W'_vooo | `xzewo` |
| sweep 1 over the blocks of W_pp | its ladders in sigma_2 (with r2 + tau1'; the T2 part taken out), `W_pp r1`, the pp ladder of R3 |
| R3 with the operators of the ground state | pass 2 (`zc_t3_pass2r` / `xzbpass2`), the operator on occupied triples |
| sweep 2 over the rows of W'_pp | slab by slab `- P(ab) r1[m][a] ~<mb||de>` (`zc_vvvv_dress` with r1 for t1 and a zero bare block), its ladder with t2 in sigma_2, then `+ r2 <mn||de> + 1/2 A[F'_vv]` and the pp ladder of T3 |
| T3 with the derivatives of the operators | pass 2 with (W'_ring, r2, W_vooo | W_vvvo, r2), the operator of (W'_oooo, F'_oo) on occupied triples |

Two things differ from the Python code: **W'_pp is never stored** - its
rows are made in slabs of `nqs` rows, as the dressed `<ab||cd>` of an
amplitude step - so a sigma vector takes no v^4 array of its own; and the
operator O'_hh of a shift is the full matrix on occupied triples, of which
the passes take the blocks between two groups.

**Label shifts in the blocks** (`xzblk.inc`, `xzdata.F90`: `xzshctx`).  The
block passes of the ground state take the shifts of their operands from
module variables - `sh_t` of the rank-3 operand, `sh_w` of the operators (of
the integrals in pass 1), `sh_a` of the amplitudes of the drive terms; the
result has `sh_o = sh_t + sh_w` - which are zero everywhere but in the two
residuals and the two passes 1 of a sigma vector:

- a rank-3 array of shift s has the blocks (occupied triples of group g) x
  (virtual triples of group g + s) (`xzeomlay`; `xzna`), and the gathers and
  reductions get the table `obase` of its layout - the kernels are driven
  by tables and do not know the layout;
- where the class of an index follows from the classes of the others the
  shifts enter (the group of Q in the gathered `T[P][(d, m)][Q]`, the two
  classes of the operands of pass 1 and its two conditions for the outputs
  of rank 1 and 2, the classes of e and m in the drive terms);
- the ring operator is a rectangular block per delta, [(a, i) of delta +
  sW][(d, m) of delta]; the rows of W'_pp are pairs of block h + s for the
  columns of block h; a block that is not there is not gathered (zero);
- `xzshctx(0, 0, 0)` gives the tables of `xzsetmode`: the ground state, the
  Lambda equations and the density run the same statements as before (their
  energies and files are unchanged, 16.5).

r1 and r2 of a vector are dense with zeros outside their shift, and sigma_1,
sigma_2 are masked to it at the end of every sigma vector (`xzemask`): the
integrals are symmetric to the rounding of the SCF only, and without the
mask the states of other shifts, which lie lower, grow in the iterations
(found in the Python program at production size, its commit 656148d).

### 16.3 The solver (`xzeoms.inc`)

The user's decision was CFOUR's design; it is that of `xx2ceomcc` with its
algorithm 2 (`eomene2`; docs/eom.md 9):

1. right-hand vectors only; **one orthonormal basis for all roots of a
   problem**, with the sigma vector of every basis vector;
2. the projected matrix `<C_i| H C_j>` (the new row and column per new
   vector), diagonalized by LAPACK's general solver (`zgeev`, `dgeev`), the
   eigenpairs sorted by the real parts of the eigenvalues;
3. the targets are the lowest roots asked for; **one correction vector per
   root that is not converged and iteration**: the residual `sum_j y_j (H
   C_j - omega C_j)` of the normalized Ritz vector, the test on its largest
   element, the correction `r / (omega - D)` with the differences of the
   spinor energies D, Gram-Schmidt against the basis twice, its sigma
   vector;
4. a converged root is kept by its coefficients in the subspace and
   recognized afterwards by its overlap with the eigenvectors
   (`eompkvec3`'s rule), so that no residual is formed for it again;
5. when the vectors of an iteration would exceed the subspace, it is
   compacted to the Ritz vectors of the converged roots and the lowest
   others, roots + 2 in all, C and H C transformed alike (`eomcompact`;
   here: the coefficients are orthonormalized in the subspace and both sets
   are transformed in one pass, in place).  CFOUR adds one vector per
   iteration, so 20 free places last 20 iterations; here every root that is
   not converged adds one, so an iteration adds at most subspace - (roots +
   2) of them (to the lowest roots; the compaction always keeps every root)
   and the default subspace has four places per root (40 roots of H2O /
   6-31G, CCSDT, threshold 1e-7: subspace 82 = 2 roots + 2, compacted every
   iteration, 2107 sigma vectors and 12 roots not converged after 120
   iterations; 164: 923 sigma vectors, 110 iterations; 202: 815, 65; 400,
   never compacted: 678, 40; the roots of 164 and 202 agree with those of
   400 to 5e-9 and 1.3e-8 Eh; the default 168: all 40 in 76 iterations,
   883 sigma vectors, 5e-9 Eh from 400).  Before this, the subspace was
   capped at 64 and a compaction kept fewer than roots + 2 vectors: the same
   40 roots came out wrong from the 25th on (0.790 Eh for 0.760).

What is CFOUR's code and what is written here:

| | |
|---|---|
| linked (complex arithmetic) | `eomreorder` (the sort of the eigenpairs) and `eomconv` (the test and its line of output): two objects of its program directory `x2ceomcc` (`EOMOBJ` of the Makefile); nothing is copied |
| written here, the same algorithm | these two for real arithmetic (`xzesmall`, `xzeconv`); and everything that touches a vector - store, project, Ritz vector and residual, precondition, orthogonalize, compact (`xzevput`, `xzevmdot`, `xzevcomb`, `xzeprec`, `xzeorth`, `xzevtrans`): CFOUR's routines for these are generic in the length but read whole vectors of complex singles and doubles through the offsets of its work array and the file names of its modules (`soccmem`, `soccmod`), and its vectors have another layout |

What was not taken over, and what is done instead:

- *Every root in a subspace of its own* (CFOUR's default algorithm 0, with
  which two of four roots of H2O came out twice): one subspace for all roots
  of a shift, the corrections orthogonal to all of it - a state cannot be
  found twice.
- *The collapse of algorithm 0*, suspected wrong in docs/eom.md 9: not
  used; the compaction here transforms the projected matrix with the same
  coefficients as the vectors (`Q^H G Q`), and a run with a subspace of 13
  instead of 25 vectors gives the same roots (16.4).
- *One vector per iteration, one root after the other*: here all
  unconverged roots get their vector in an iteration, so a degenerate or
  nearly degenerate level is built together.
- *A converged root is never looked at again*: here the residuals of all
  roots are computed once more from the final vectors (they are what the
  table prints); a root above the threshold after all - the vectors of a
  nearly degenerate level turn into each other while the subspace grows -
  is iterated on, and the record of a converged root that no eigenvector
  has any more (overlap below 0.95) is dropped.
- *The preconditioner*: the differences of the spinor energies for all
  three ranks (CFOUR: the diagonal of Hbar for the singles), a denominator
  below 1e-3 replaced by 1e-3 (CFOUR: the element set to zero) - as the
  Python solver.
- *Real arithmetic*: two roots that are degenerate but for the rounding
  come out of `dgeev` as a complex pair; the real and the imaginary part of
  its vector are taken as two vectors with the real part of the eigenvalue.
- `EOMFOLLOW`, `%eomroot`, `%irestart`, `%eomalg`, the core-valence and
  continuum masks of CFOUR are not read: the lowest roots of a shift.

**The start.**  Default (`%xzeomguess 1`): the roots of the singles-doubles
block of the same matrix (the sigma vector without R3 and without the terms
of T3 in sigma_2: no work of the size of T3) for roots + 2 unit vectors on
the lowest single excitations of the shift, converged to 1e-4; all of them
start the problem with triples, which takes its lowest roots.  With
`%xzeomguess 0` the unit vectors themselves, one per root, as CFOUR without
CIS.  Both converge to the same roots in every case below; the unit vectors
took 6, 3 and 6 sigma vectors of T3 size more for three roots of the three
shifts of HBr (56, 53, 68 against 50, 50, 62 at a threshold of 1e-9), and a
root without weight on the start vectors is found only if a correction
brings it in - hence the default.

**How tight.**  The largest element of the residual is the test (CFOUR's);
the norm of the residual is 5 to 15 times that.  HBr with `%dglin`, nine
roots, against the roots at 1e-9:

| threshold | cc-pVDZ-DK: largest error; sigma vectors per root | cc-pVTZ-DK |
|---|---|---|
| 1e-4 | 7.1e-6 Eh (1.6 cm-1); 7.0 | 1.2e-5 Eh (2.6 cm-1); 6.9 |
| 1e-5 (`ESTATE_CONV` default) | 1.9e-6 Eh (0.4 cm-1); 8.6 | 9.1e-7 Eh (0.2 cm-1); 9.3 |
| 1e-6 | 1.9e-7 Eh (0.04 cm-1); 11.4 | 5.6e-8 Eh (0.01 cm-1); 12.3 |
| 1e-9 | -; 18.0 | -; 20.3 |

### 16.4 Memory and disk

A vector is `[r1 | r2 on pairs | R3 in the blocks of its shift]`, one
region: o v + C(o,2) C(v,2) + (the blocks of the shift) elements.  In the
work array, bottom to top: tables and integrals (the pp operator in the
place of `<ab||cd>`), T1, T2, T3, the kept intermediates (five arrays of
the size of `<ov||vv>` on pairs, f = o v C(v,2): the dressed block, the
bra-dressed `<ab||ei>`, W_vvvo, the operand of pass 1 with its o / v more;
the operator on occupied triples; arrays of (o v)^2), the table of the
layout of the shift, the vectors to start from, **two vectors** (the new
one and its sigma vector), the subspace, and the arrays of a sigma vector,
taken and released stage by stage (three more of f at the peak: W'_vvvo,
the result and the operand of the second pass 1; the batch buffers of the
passes).

**The plan** is that of section 4: `xzeom` is run dry after the layout (and
after the iterations in `ccsdt_eom`) through the allocations of every shift
of the input, with the smallest subspace that can run - the singles-doubles
stage in the work array, or the buffer of the files - and the batch sizes
are raised as far as fit.  The floor is checked before the large files are
read:

    @XZCCSDT-F, the work array is too short for this calculation.
      words needed:          2636792   words there:          2631792
      Set MEMORY_SIZE to at least          2636856 words (        21.1 MB).

**The subspace** takes what is left after that (the batches come first, as
for the DIIS vectors), decided per shift when its problem starts - a dry run
of one sigma vector gives what must stay free:

1. all vectors of the subspace and their sigma vectors in the work array, if
   they fit;
2. else, if at least 3 roots + 4 of them fit: as many as fit, in the work
   array - the output says `the work array holds 18 vectors of the subspace
   .. (25 asked for)`.  A smaller subspace costs sigma vectors (HBr /
   cc-pVDZ-DK, three roots at 1e-9: 13 of 25 vectors 76 instead of 62 sigma
   vectors, 19 of 25: 53 instead of 50) but no reading;
3. else the files `EOMCnnnn` (vectors) and `EOMHnnnn` (sigma vectors) of the
   run directory - CFOUR's names -, one vector each, unformatted stream,
   read in chunks through a buffer of the work array (up to 2^20 elements
   per vector), kept open, deleted at the end.  Every new vector reads the
   subspace about seven times (the row and the column of the projected
   matrix, Ritz vector and residual, Gram-Schmidt twice).  The free space of
   the run directory is asked first: the subspace is lowered to what the
   disk holds, and below 2 roots + 2 vectors the program stops with the
   numbers.  A write error stops it with the name of the file and the size
   of a vector.  `eomvec f` forces the files.

The file `XZEOMR` of the vectors of the roots is written only if the disk
holds it (a warning otherwise; `%xzeomvec 0`: never).

The plan prints `the least that EOM takes` and `EOM with this plan` without
the subspace in the work array; `work array used` at the end has it.

| case | o, v | mode | least for EOM | plan (without the subspace) | a sigma vector | a vector | subspace of 25 |
|---|---|---|---|---|---|---|---|
| H2O / 6-31G | 10, 16 | complex, dense | 26.0 MB | 51.2 MB | | 1.3 MB | (28 for 6 roots) 59 MB |
| HBr / cc-pVDZ-DK, `%dglin`, 28 frozen | 8, 28 | real, 6 classes | 21.1 MB | 22.4 MB | 6.0 MB | 0.34 MB | 17.1 MB |
| HBr / cc-pVTZ-DK, `%dglin`, 28 frozen | 8, 78 | real, 8 classes | | 284 MB (with CCSDT) | 108 MB | 5.7 MB | 283 MB |
| HBr / cc-pVQZ-DK, `%dglin`, 10 frozen, `%vircut` (job 31847097) | 26, 120 | real, 10 classes | | 7.1 GB (EOM), 14.5 GB (with CCSDT and its DIIS vectors) | 2.29 GB | 674 MB (shift 0), 649 (2), 579 (4) | 33.7 GB; work array used 39.7 GB, peak resident 39.4 GB |

### 16.5 Validation

**The sigma vector against the derivative of the amplitude step**
(`eomtest seed`, with `ccsdt_eom`): for a random totally symmetric vector
R the five-point formula `[8 (Res(T + R) - Res(T - R)) - (Res(T + 2R) -
Res(T - 2R))] / 12` of the residual of the native CCSDT step - which is
exact for these polynomials (docs/eom.md 1) and is the reference of the
Python sigma vector too - against `xzesig`:

| case | mode | abs. difference in sigma_1, sigma_2, sigma_3 (norms 0.1 to 3, 2 to 23, 10 to 109) |
|---|---|---|
| H2O / 6-31G (10 + 16) | dense, complex | 9e-16, 9e-15, 7e-14 |
| | the same with batches of 2 pairs, 3 triples, slabs of 7 rows | 1e-15, 8e-15, 7e-14 |
| HBr / cc-pVDZ-DK, `%dglin` (8 + 28) | 6 classes, real | 4e-16, 5e-15, 4e-15 |
| | dense, real (`labels 0`) | 1e-16, 2e-15, 1e-14 |
| | 6 classes, complex (`real 0`) | 5e-16, 7e-15, 6e-15 |
| | 6 classes, real, small batches | 4e-16, 5e-15, 4e-15 |

This is the check of the dense sigma vector and of the block passes at
shift 0 after their generalization, in real arithmetic too.

**The sigma vector against the Python sigma vector, element by element**
(`eomsig file`: the vector of the file, in the layout of 16.4, and its
sigma vector written to `file.out`; `eomsd 1` with it: the
singles-doubles block).  `eom_opt.sigma` of the Python program for a random
vector of the shift (complex: the Python program takes integrals read from
files as complex, so the native program ran with `real 0` on the real
files), each program on its own converged amplitudes; where the spinors are
in the order of their labels in the files the two programs have the same
layouts of r1, r2 and of R3 in the blocks of a shift:

| case | mode, shift in labels | largest difference in sigma_1, sigma_2, sigma_3 (largest elements 2 to 42, 11 to 156, 25 to 188) |
|---|---|---|
| H2O / 6-31G, CFOUR's spinors (10 + 16) | dense, complex | 1e-10, 4e-10, 6e-10 (the amplitudes of the two programs: CC_CONV=9) |
| HBr / cc-pVDZ-DK, labelled real spinors (8 + 28) | dense (`labels 0`) | 1e-15, 3e-14, 5e-14 |
| | 6 classes, shift 0 | 1e-15, 2e-14, 4e-14 |
| | 6 classes, shift 2 | 1e-15, 2e-14, 4e-14 |
| | 6 classes, shift -4 | 1e-15, 2e-14, 4e-14 |
| the same with a phase on every spinor (complex integrals) | 6 classes, shifts 2 and 6 | 2e-15, 3e-14, 5e-14 |

The sigma vectors of real arithmetic with a shift are checked by their
roots: the real blocks give the roots of the complex blocks to all digits
printed, and those of the Python program and of the dense native run
(below).

**Roots against the Python EOM on the same integral files**
(`zccsdt.cfour.from_cfour(dir, labels=True)`, `GCCSDT.from_integrals`,
`eom_ee(shift=)`, residual norm 1e-9; native: largest element 1e-9, i.e.
norms of 3e-9 to 1e-8; CC_CONV=9 or 10).  Reference directories:
`/scratch16/lcheng24/xwang405/zccsdt_ref/<case>` (README in each).

| case | mode | roots | largest difference / Eh |
|---|---|---|---|
| H2O / 6-31G, X2C-1e, CFOUR's spinors (10 + 16) | dense, complex | 7 (a triplet 0.2843786409 / ..8923 / ..9590, 0.3111891002, a triplet at 0.36607) | 4e-11 |
| H2O / 6-31G, nonrelativistic RHF of PySCF as spin orbitals (the integrals of the MRCC reference, written by `zccsdt.cfour.write`) | dense, real | 16 | 1.8e-10 |
| HBr / cc-pVDZ-DK, `%dglin`, CFOUR's SCF, 28 frozen (8 + 28) | 6 classes, real; shifts 0, 2, 4 (three roots each) | 9: shift 0 0.243530585077, 0.243983671405, 0.336571921431; shift 2 0.237036344568, 0.260822952202, 0.336663368901; shift 4 0.233028661244, 0.544410080817, 0.634588206498 | 7e-11 |
| | 6 classes, complex (`real 0`) | the same 9 | equal to the real run to all digits printed |
| | dense, real and complex (`labels 0`): the 9 lowest of all shifts | 0.233028661246 (twice), 0.237036344589 (twice), 0.243530585084, 0.243983671519, 0.260822952225 (twice), 0.336571921481 | 1.1e-10 to the roots of the shifts |
| | shifts 0, -2, -4 from unit vectors (`%xzeomguess 0`) | 9 | 1e-10 to the roots of +2, +4 |
| | the subspace in files (`eomvec f`), the work array at its floor | 9 | equal to all digits printed |
| the same with `%vircut` 1.5 (8 + 22, 6 dropped) | 6 classes, real; shifts 0, 2, 4 | 9 (0.247567085051, ..) | 1.2e-10 |
| HBr+ doublet, `%dglin` 12 12 / 5 4 / 1 1, 28 frozen (7 + 29): an open-shell reference with unequal occupations of +-m_j | 6 classes, real; shifts 0, 2, -2, 4, -4 | 13 (shift -2: 0.012125852499, 0.157178984394, ..; shift -4: 0.011838429525, ..; shift 0: 0.273088295771, ..) | 1.1e-10 |
| HBr / cc-pVDZ-DK, m_j-adapted spinors of a PySCF SCF in the order of their labels (`bench/cfour/labelled_run.py`; the Python program in its blocks) | 6 classes, real; shifts 0, 2, -2, 4, 6 | 14 | 7e-11; +2 against -2: 2e-12 |
| the same with a phase on every spinor | 6 classes, complex | 9 | equal to the real spinors to all digits printed |

**MRCC** (EOM-CCSDT, `bench/refs/reference_energies.json`, `eom_ee` /
`h2o_631g`, MRCC's own RHF integrals; the native program on PySCF's
integrals of the same molecule, row 2 above; 16 roots asked for):

| state | MRCC / Eh | native | difference |
|---|---|---|---|
| E(CCSDT) | -76.120425238932 | -76.120425238916 | 2e-11 |
| triplet 1 | 0.284831364143 | 0.284831363508 (three roots) | 6e-10 |
| singlet 1 | 0.311687769200 | 0.311687768552 | 6e-10 |
| triplet 2 | 0.366400122961 | 0.366400122669, ..669, ..770 | 3e-10 |
| triplet 3 | 0.375682797356 | 0.375682796596 (three roots) | 8e-10 |
| singlet 2 | 0.394170561703 | 0.394170560987 | 7e-10 |
| singlet 3 | 0.404172632463 | 0.404172631686 | 8e-10 |

(The Python program on the same files is at 0.284831363565, 0.311687768585:
the 6e-10 to MRCC are the integrals of the two SCF programs, not the EOM.)

**CFOUR's `xx2ceomcc`** (EOM-CCSD): an EOM-CCSD mode of this program on
the amplitudes of CFOUR's `xx2cccsd` (`ccsd_eom`) was compared with
`xx2ceomcc` (H2O, HBr: the roots agree to 1e-10) and removed since:
CFOUR's own EOM-CCSD serves.

**Regression** (`regr.sh` of the work directory): the binary of the commit
before this work (e9be264) and the present one on the same files -
`ccsdt(q)_l` for seven cases (H2O; HBr complex, with `%dglin`, with
`%vircut`, the doublet, labelled spinors, with phases) and the labelled
case dense, complex and both, `ccsdt(q)` for two, small batches with the
DIIS vectors in the file - give the same E(CCSDT), E[Q]_L, E(Q)_L, E[Q],
E(Q) and Lambda pseudo-energies to all 12 (14) decimals printed, 14 lines;
the seven files of `ccsdt_density` are identical byte for byte for H2O and
HBr (complex, dense) and equal to 2e-19 in blocks, where two runs of the
old binary differ as much; the MPI build with 2 ranks gives the energies of
the serial build.

### 16.6 Timing

Job 31847097, node c210 (2 x Xeon Gold 6248R), 24 threads on the cores of
socket 0 with its memory, no other job on the node, the binary built there
from the snapshot of commit a948521, scratch on the node-local disk
(`qz/eom_time.sbatch` of the work directory).  HBr / cc-pVQZ-DK, X2C-1e,
CFOUR's SCF with `%dglin`, `DROPMO=1>5` and `%vircut 4.124587` (the 120
lowest virtual spinors): 26 + 120, real, 10 classes; the chain took 24 s.
`xzccsdt ccsdt_eom`, three roots of each of the shifts 0, 2, 4, the largest
element of a residual below 1e-7 (norms 5.9e-7 to 1.8e-6), the subspace of
25 vectors in the work array (`MEMORY_SIZE` 60 GB).  Measured once.  The
Python numbers are those of docs/eom.md, section 8 (job 31828491, node c603
of the same type, 24 cores, PySCF's spinors of the same molecule at a bond
length 0.0005 Angstrom longer, residual norm 1e-6).

| | native | Python |
|---|---|---|
| CCSDT iteration | 4.73 s (19 cycles) | 6.05 s |
| **shift 0**: triples of a vector; a sigma vector; with the solver; sigma vectors (per root) | 81 875 732; 7.78 s = 1.64 iterations; 8.99 s; 45 (15.0) | 81 875 732; -; 11.06 s; 15.3 per root |
| **shift 2** | 78 744 914; 7.45 s = 1.58; 8.81 s; 46 (15.3) | 78 744 914; -; 10.47 s; 15.7 |
| **shift 4** | 70 043 592; 6.63 s = 1.40; 7.71 s; 49 (16.3) | 70 043 592; -; 9.71 s; 17.7 |
| the vector operations of the solver per sigma vector | 1.21, 1.36, 1.08 s | 1.55 s (shift 0) |
| singles-doubles stage: sigma vectors, time | 54, 54, 42; 27.2, 27.2, 20.9 s (0.50 s each) | 52 to 73; 98, 98, 70 s (1.35 s each) |
| time of a shift | 432, 432, 399 s | 613, 596, 591 s |
| roots / Eh: shift 0 | 0.2420088840, 0.2424537236, 0.3324832462 | (another geometry) 0.2418705730, 0.2423160390, 0.3322661699 |
| shift 2; shift 4 | 0.2350868824, 0.2560543447, 0.3325560201; 0.2312098041, 0.3776223127, 0.4154809801 | |
| memory | work array used 39.7 GB, peak resident 39.4 GB (33.7 GB of it the subspace) | 31.3 GB growth, plan 36.2 GB |

A sigma vector is 1.4 to 1.64 CCSDT iterations, 1.19 to 1.26 times
faster than the Python program's with the solver included (the iteration
itself 1.28 times); the sigma vectors per root are those of the Python
block Davidson.  Of the 1089 s in the intermediates and the 140 sigma
vectors of T3 size and 150 of the singles-doubles block: pass 2 of the two
residuals 442 s (gathers and products 357, reduction 85), the two passes 1
220 s, the pp ladders 200 s, the small dense terms 135 s (the
singles-doubles stages in it), W'_vvvo 55 s, the rows of W'_pp 12 s, the
operators on occupied triples 15 s.

**The subspace in files** (the same job, afterwards: `xzccsdt eom`, shift 0,
`eomvec f`; 33.7 GB of files on the node-local NVMe): the same roots and
the same 45 sigma vectors of 7.78 s, but 22.8 s per sigma vector in the
vector operations of the solver instead of 1.2 - 2983 GB read and written
in 961 s (3.1 GB/s; 98 readings of a vector per new vector, about seven
times the subspace) - 1405 s for the shift instead of 432.  The work array
used was 6.3 GB.  Hence the rule of 16.4: a smaller subspace in the work
array before files.

An indicative run of the same script on 16 cores of a shared node (c510,
inside a holding job, not a measurement): CCSDT 5.69 s, sigma vectors 9.46,
9.17, 8.13 s, the same numbers of vectors.

Not measured: complex arithmetic and the dense layout at production size;
the Python program on this node and these spinors (its numbers are of
another job and another SCF).

### 16.7 Over MPI ranks

    mpirun -np 8 -ppn 2 xzccsdt_mpi          # CALC=CCSDT,EXCITE=EOMEE; 4 nodes, one rank per socket

EOM-EE-CCSDT runs over the ranks as the iterations do (section 15), in
every mode: real and complex, the blocks of every label shift and dense.
Rank 0 runs the solver.  No new option; `slices`, `bpr`, `rebal`,
`mpichunk` act as in the iterations.

**The sigma vector** (`xzesig` with `edist`; `xzepsig`).  Every rank has
the amplitudes (rank 0 broadcasts T1, T2, T3 once, before the
intermediates), the pp operator in the place of `<ab||cd>` and the kept
intermediates of 16.2, **made by every rank itself** (`xzeomim`, one
amplitude step without its residual; 1 to 4 s at the sizes below).  Per
vector:

| step | who | what goes between the ranks |
|---|---|---|
| r1, r2 of rank 0; the rows of R3 of every rank | all | a broadcast of the singles and doubles; the slices of R3 to every rank (`xzmpi_slices_put`): every rank then has the whole vector |
| H', the terms of sigma_1, sigma_2 without T3-sized arrays, W'_vvvo, W'_vooo, the rows of W'_pp | every rank, the same numbers | - |
| **pass 1 over R3**; **pass 1 over T3 with H'** | every rank: its batches of pairs | after each: the four (three) results summed on every rank (`MPI_Allreduce`, as meeting 1 of an iteration) |
| sweep over W_pp with the **pp ladder of R3**; **pass 2 with R3** | every rank: its batches of triples / pairs | - |
| operator on occupied triples with R3 | every rank: its rows (`xzeslocc`) | - |
| sweep over the rows of W'_pp with the **pp ladder of T3**; **pass 2 with T3** | every rank: its batches | - |
| operator of (W'_oooo, F'_oo) with T3 | every rank: its rows | - |
| sigma_3 | all | **the rows to their owners** (`xzmpi_slices_add`) |

sigma_1 and sigma_2 come out the same on every rank (after the sums of the
passes 1); rank 0's are used.  The six loops are six more kinds of batches
of `xzdist.F90` (7 to 12; the batches themselves are those of the plan,
`nbp` pairs and `nbt` triples), dealt **per label shift** by a cost model
that has the shifts of the operands (`xzbcost1`: the classes of the summed
indices with `sh_t`, `sh_w` as the passes of `xzblk.inc` have them), and
dealt again by the measured speeds of the ranks after a sigma vector whose
last rank came more than `rebal` after the mean at one of the three
meetings (`xzdeal_eom`, `xzrebal_eom`; the ladder and pass 2 of T3 start on
a rank when those of R3 are done).  The slices are those of the layout of
the shift (`xzeslices`: equal numbers of elements of R3, cut at rows of its
blocks), not those of T3.

**The solver and where the subspace is** (the scalable form of the two that
the design named).  Rank 0 runs `xzesolve` as before: the projected
matrix, its eigenpairs, the convergence test, compaction.  **No rank holds
more of a vector of the subspace than its rows**: rank r has, of every
basis vector and of every sigma vector, the elements of its slice of R3 -
rank 0 with the singles and doubles in front - in its own work array, or
in its own files (below).  Every operation of the solver on vectors is a
command of rank 0 (`xzecmd`: an operation and up to five numbers,
broadcast) that all ranks carry out on their part (`xzeserve` on the other
ranks; the routines are the serial ones on the range `evoff .. evoff +
evlen - 1` of a vector): store, the dot products with the subspace
(`MPI_Reduce` of n numbers onto rank 0), linear combinations and the
compaction (the coefficients broadcast), scale, the preconditioner by
rows, the largest element of a residual (`MPI_Reduce` with max; CFOUR's
`eomconv` gets that element and prints its line), the sigma vector.  The
two vectors of the work array (the new vector, its sigma vector) are whole
on every rank - the passes read all of R3 and add to rows of all owners.
For the file `XZEOMR` the rows of a root go to all ranks once more and
rank 0 writes the vector.  The singles-doubles stage (the start) is rank
0's alone: it has nothing of the size of T3.

**The subspace in files** works per rank: with `eomvec f`, or when the
plan of rank 0 says files, every rank keeps its rows in its own files
`EOMCnnnn.<rank>`, `EOMHnnnn.<rank>` of its run directory (rank 0:
`EOMCnnnn`, `EOMHnnnn`), read through its own buffer.  Work array or files
is decided once, by rank 0, for all - as for the DIIS vectors.

**Memory per rank, planned before anything is allocated.**  The dry run
of rank 0 (section 4) goes through the EOM stage twice, as itself and as a
rank other than 0.  As itself it fixes, per label shift, what the run will
do with this `MEMORY_SIZE` by the rule of 16.4 applied to *its own part* of
a vector (singles, doubles, its slice): all vectors of the subspace in the
work array, fewer, or files (`xzevplace`; `epm`, `epf`).  The run follows
that plan (it never takes more vectors than planned), and the words that
another rank needs for its rows of that subspace are part of what rank 0
sends it (`rplan(17)`), so a rank other than 0 allocates

    tables, integrals (the pp operator), T1, T2, T3, the kept intermediates
    + 2 whole vectors + the arrays of a sigma vector (as rank 0, with the
      chunk of the exchange)
    + 2 x (vectors of the subspace) x (its slice of R3)        [or the file buffer]

and stops with both numbers if its array does not hold the plan.  Rank 0
holds the same with its own slice and the singles and doubles.  The plan
prints both and the sum per node against `MemAvailable`:

      a rank other than 0 takes:            1715130792  =    13721.0 MB
        (EOM: the vectors of the subspace by rows of R3 - rank 0 holds up to     9127.3 MB of them besides the plan above, every other rank     8197.9 MB, which is in its number)
        node of rank     0:    1 ranks, work arrays      15784.4   memory available at the start     171932.8

Measured (jobs below; MB; "used" is the table of the ranks at the end, the
work array and the peak resident set):

| case, subspace of 25 | serial: work array, resident | 2 ranks: rank 0 / another, plan = used; resident | 4 ranks | 8 ranks |
|---|---|---|---|---|
| HBr 26 + 120, real blocks (a vector 674 MB) | 39 742, 39 109 | 23 144 / 22 121; 23 425 / 22 418 | 14 743 / 13 721 (used 13 700 to 13 714); 15 489 / 14 386 to 14 437 | 10 533 / 9 516 (used 9 498 to 9 516); 11 234 / 10 118 to 10 288 |
| HBr 18 + 78, complex dense (a vector 1001 MB) | 55 647, 51 014 | 30 656 / 30 250; 28 972 / 28 522 | 17 581 / 17 175; 16 913 / 16 417 to 16 466 | 11 238 / 10 833; 11 494 / 11 030 to 11 127 |

(The resident sets of the ranks of an MPI run are up to 8 % above their
work arrays: buffers of the MPI library.)  The subspace is what shrinks: 33.7 GB (50.0 GB) on one rank, 2 x 25 x
(slice) on each of n.  What does not shrink with n: T3, the two whole
vectors and the arrays of a sigma vector on every rank.

**Not distributed**, and what it costs: the small dense terms of a sigma
vector, W'_vvvo and the rows of W'_pp are made by every rank with its own
threads (HBr 26 + 120 in blocks: about 0.6 s of a serial sigma vector of 4.9 s
on 48 threads; they are what bounds the gain in blocks, as the
intermediates do in an iteration, 15.6); the singles-doubles stage runs on
rank 0 (22 to 27 s per shift there, at any number of ranks: 10 % of a shift
on one node, 18 % on four); the intermediates of the ground state are made
by every rank.

**Validation** (holding job 31913218, node c611; the binaries of the
commit before this work as the reference; scripts and outputs
`.claude/jobs/7ba4fcf7/tmp/eomm`).

*Roots, serial against ranks* (`XZEOM`, 14 decimals):

| case | runs | largest difference of a root / Eh |
|---|---|---|
| HBr / cc-pVDZ-DK, `%dglin`, real blocks, shifts 2, 4, 6, 0, three roots each, 1e-9 | serial on 8 and 1 thread; MPI build on 1, 2, 3, 4 ranks (8, 4 and 1 thread per rank); `slices 0`; small batches with `mpichunk 3001` and a deal after every vector (`plan 7 3 2 2 ..`, `rebal 0.0001`); `eomvec f` on 2 and 3 ranks | 1.0e-14 |
| the same, complex blocks (`real 0`) | serial, 2, 3, 4 ranks | 0 (all 14 decimals) |
| H2O / 6-31G, complex dense, 4 roots, 1e-5 | serial, 2, 3, 4 ranks, 4 ranks of 1 thread | 0 |
| HBr dense real (`labels 0`), 3 roots | serial, 2, 4 ranks | 0 |
| HBr, the start from unit vectors and a subspace of 13 (21 compactions) | serial, 3 ranks | 0 |
| HBr, EOM alone from `XZAMPS` of a run on 2 ranks | 3 ranks, serial | 1.0e-14 |
| HBr, rank 0 near the least memory (`maxcor`; 2 and 3 ranks: subspaces of 13 to 24 vectors by the plan, the work array used 99.7 to 100.0 % of it) | | 1.8e-10 (another subspace; threshold 1e-9) |

*The sigma vector of a fixed vector* (`eomsig`: a random vector on the
pattern of the roots of the shift; the largest difference of an element
of sigma to the binary of before this work, serial; largest elements 3.8
to 24.6):

| case | new serial, 8 threads | 1 thread | 1 rank | 2, 3, 4 ranks; 2 ranks of 1 thread; 3 ranks with small batches and `mpichunk 3001` |
|---|---|---|---|---|
| HBr real blocks, shift 0 | 0 | 1.3e-15 | 0 | 1.8e-15 to 2.2e-15 |
| shift 2 | 0 | 1.6e-15 | 0 | 1.8e-15 to 2.2e-15 |
| shift 4 | 0 | 1.3e-15 | 0 | 2.2e-15 to 3.6e-15 |
| HBr complex blocks, shift 2 | 0 | 1.8e-15 | 0 | 1.8e-15 to 3.6e-15 |
| shift 6 | 0 | 1.3e-15 | 0 | 2.2e-15 to 3.1e-15 |
| H2O complex dense | 0 | 7.1e-15 | 0 | 1.1e-14 |
| HBr real dense | 0 | 1.8e-15 | 0 | 1.8e-15 to 2.7e-15 |

(the serial program and one rank of the MPI build give the bits of the
binary of before; ranks differ from it as one thread does.)

*Regression*: `regress.sh --eom` serial, and `--np 2 --eom` - the EOM step
of the reference with one rank, of the new binary with two - on the seven
cases: REGRESSION OK both (all printed lines identical, 38 lines of roots
each; density files to 5.6e-17).  `eomtest` serial: the numbers of before
(3.6e-16, 5.1e-15, 4.2e-15); with more than one rank it stops with a
message, as `cc3 stored` with EOM does.  The folded EOM-CC3 on 2 ranks:
its roots as before.

**Not tested**: files chosen by the plan over ranks (in the cases tried
the smaller subspace in the work array always won; files over ranks ran
with `eomvec f`); a rank whose work array is too short for the plan; Open
MPI.

**Timings.**  Two jobs, snapshot of commit 9664495 built on the node, run
directories in the node-local `/tmp` of every node, Intel MPI 2021
(`I_MPI_PIN_DOMAIN=omp`, `OMP_PROC_BIND=close`; one node:
`I_MPI_HYDRA_BOOTSTRAP=fork`, rank 0 on the cores 0-23 and rank 1 on 24-47;
several nodes: through Slurm, provider `mlx`), Xeon Gold 6248R (2 x 24
cores, 189 GB).  The serial program first, on the first node; 24 threads:
the cores and the memory of socket 0.  EOM from `XZAMPS` of the first run
in all runs but the first.  **Seconds per sigma vector with R3** (the
solver not included; measured once):

*A. HBr / cc-pVQZ-DK, `%dglin`, 26 + 120, real, 10 classes* (the case of
16.6; T3 655 MB; three roots of each of the shifts 0 and 2 to 1e-6: 36 + 36
sigma vectors in every run; the mean of the two shifts):

| | job 31913519, node c694 | job 31913521, nodes c045, c075, c138, c143 | vector operations of the solver per sigma vector |
|---|---|---|---|
| serial, 24 threads | 7.55 | 6.67 | 1.05 to 1.36 |
| serial, 48 threads | 6.39 | 4.91 | 0.62 to 1.04 |
| 1 node x 2 ranks x 24 | 5.79 | 4.60 | 0.55 to 0.83 |
| 1 node x 4 x 12 | 6.29 | | 0.80 |
| 2 nodes x 1 x 48 | | 3.59 | 0.42 to 0.49 |
| 4 nodes x 1 x 48 | | 2.62 | 0.24 to 0.28 |
| 4 nodes x 2 x 24 | | 2.67 | 0.22 to 0.25 |

Speedups (same node for the serial run): on one node 2 x 24 against the
serial program on 24 threads 1.30 and 1.45, against 48 threads 1.10 and
1.07; two nodes 1.37 (1.86 against 24 threads); **four nodes 1.87 (2.55)**,
with two ranks each 1.84.  CCSDT itself: 4.68 and 4.07 s per cycle on 24
threads.

*B. HBr / cc-pVTZ-DK, `DROPMO=1>9`, `labels 0`, `real 0`: 18 + 78, complex,
dense* (62.1 million triples, T3 993 MB; three roots to 1e-5, 29 sigma
vectors in every run; CCSDT 13.45 and 12.86 s per cycle on 24 threads):

| | job 31913519, c694 | job 31913521, c045 .. | vector operations per sigma vector |
|---|---|---|---|
| serial, 24 threads | 25.60 | 24.71 | 1.52 to 1.99 |
| serial, 48 threads | 19.93 | 17.02 | 0.87 to 1.16 |
| 1 node x 2 ranks x 24 | 14.95 | 13.77 | 0.81 to 1.12 |
| 1 node x 4 x 12 | 13.93 | | 1.13 |
| 2 nodes x 1 x 48 | | 10.00 | 0.62 |
| 4 nodes x 1 x 48 | | 5.74 | 0.36 |
| 4 nodes x 2 x 24 | | 4.64 | 0.32 |

Speedups: on one node 2 x 24 against 24 threads 1.71 and 1.79, against 48
threads 1.33 and 1.24 (4 x 12: 1.43); two nodes 1.70 (2.47 against 24
threads); **four nodes 2.97 (4.30), with two ranks each 3.67 (5.33)**.

The roots of all runs of a case agree to the 14 decimals of `XZEOM` (one
run: 1.0e-14).

Where a sigma vector goes (seconds per sigma vector and rank; "rest": what
every rank makes itself; the table of the ranks of the output):

| | pass 1 (both) | sweeps, ladders | pass 2 (both) | sums of pass 1 | vector to the ranks, sigma_3 to its rows | on its rows | waiting | rest |
|---|---|---|---|---|---|---|---|---|
| blocks, 1 node x 2 x 24 (c694) | 0.87 | 0.95 | 1.76 | 0.72 | 0.14, 0.14 | 0.06 | 0.13 | 0.87 |
| blocks, 4 nodes x 1 x 48 | 0.41 | 0.37 | 0.59 | 0.22 | 0.05, 0.07 | 0.02 | 0.21 | 0.62 |
| dense complex, 4 nodes x 1 x 48 | 1.00 | 0.99 | 2.50 | 0.09 | 0.09, 0.12 | 0.18 | 0.38 | 0.30 |
| dense complex, 4 nodes x 2 x 24 | 0.51 | 0.90 | 1.83 | 0.15 | 0.20, 0.22 | 0.15 | 0.30 | 0.17 |

The sums of the passes 1 (`MPI_Allreduce` of 0.53 GB per sigma vector in
blocks: 0.7 GB/s on one node, 2.4 GB/s between four) are 8 to 13 % of a
sigma vector in blocks; the exchanges of the vector and of sigma_3 ran at
2.3 to 4.0 GB/s per rank on one node and 6 to 9.7 GB/s between nodes (8
ranks on four nodes: 3.7 to 4.2).

**An estimate for ThF+ 28 + 330** (real, 12 classes; **not a measurement**;
no job was run).  Serial, 24 cores of a bigmem node (job 31861231): 449 s
per sigma vector of shift 2 (a vector 15.4 GB) and 21.5 s of vector
operations; of a CCSDT iteration there 85 % are the three loops that the
ranks share (pass 1 15.8, ladder 22.9, pass 2 46.4 %) and 15 % what every
rank would make itself (the t2 `<ov||vv>` term of W_vvvo alone 7 %).
With the measured ratios of case A - the passes of a rank of 48 threads
1.3 times those of 24, three quarters of the ideal gain over four ranks,
the rest 1.4 times faster on 48 threads - and the sums of pass 1 (9.4 GB
per sigma vector) and the exchanges at the rates above:

| | passes | made by every rank | sums, exchanges, waiting | a sigma vector | against 449 s |
|---|---|---|---|---|---|
| 2 nodes x 1 x 48 | 170 s | 50 s | 15 s | about 235 s | 1.9 |
| 4 nodes x 1 x 48 | 95 s | 50 s | 15 s | about 160 s | 2.8 |

(of the 2.8, 1.36 is 48 threads against 24.)  **Memory** is what decides
where it can run: without the subspace 107 GB per rank (the serial plan:
integrals, T3 15.7 GB, the kept intermediates, two vectors, a sigma vector
21 GB), which every rank has; the subspace of 25 vectors, 770 GB on one
rank, is 2 x 25 x 15.4 / n GB per rank - 193 GB on four: 300 GB per rank,
i.e. four nodes of the bigmem kind, where the serial run took 680 GB and
held 18 vectors.  On the 189 GB nodes of `parallel` a rank has room for 9
vectors of four ranks' slices: enough for one root per run (3 roots + 4 =
7), not for three (13) - then the plan puts the subspace into the files of
every rank (193 GB on its node-local disk), which costs seven readings of
them per new vector (16.6).

**Not done**: the small dense terms, W'_vvvo and the rows of W'_pp split
over the ranks; the singles-doubles stage over ranks; the intermediates
of the ground state made once and sent, or their pass 1 dealt out; the
sums of pass 1 by an exchange in the place of `MPI_Allreduce`; whole
vectors (R3, sigma_3) held by rows only, which the passes do not allow;
EOM of CC3 with T3 stored over ranks; more than four nodes.

### 16.8 Test hooks

`eomtest seed` (16.5), `eomsig file` [`eomsd 1`],
`eomvec f` (the subspace in files), `eomcyc n` (iterations at most:
`ESTATE_MAXCYC` is in JOBARC and needs xjoda to change).

### 16.9 Not done

- Left-hand vectors, r0, transition moments, densities and properties of
  the excited states; IP and EA (the Python program has them by a ghost
  spinor with short vectors, docs/eom.md 5).
- T3 or the vectors in a mapped file.  Over ranks (16.7): the small dense
  terms of a sigma vector split over the ranks; `cc3 stored` with EOM.
- Root following (`EOMFOLLOW=OVERLAP`), given start vectors (`%eomroot`),
  restart from the files of the subspace (`%irestart`), the core-valence
  separation, CFOUR's CIS guess.  A state without a single excitation in its
  shift has no start vector (the Python program goes on with doubles).
- A degenerate level cut by the number of roots is not completed (H2O: six
  roots end inside the second triplet and converge slowly: 7 are 268 sigma
  vectors, 6 did not converge in 40 iterations).
- The residuals of all roots in one pass over the subspace; the dense small
  terms of a sigma vector and the o v^3 arrays in blocks of the labels.
## 17. CC3

    CALC=CC3                    CC3 without stored triples
    CALC=CC3, EXCITE=EOMEE      CC3, then the folded excited states (the label shifts of %xzeom)
    cc3 stored               (a line of %xzccsdt; a check route) CC3 with T3 stored; with
                                EXCITE=EOMEE its excited states in every label shift
    mpirun -np 4 xzccsdt        the iterations over ranks (both routes)

The job comes from the ZMAT as for every method (section 1); `CALC=CC3` is
joda's level 32, and EXCITE=EOMEE is read in either of joda's forms.  The
validation and timings below were run while the program still took the
method as an argument (`cc3`, `cc3_eom`, `cc3_stored`, `cc3_stored_eom`:
the four cases above).  CC3 amplitudes are never reused from `XZAMPS` as a
result: a CC3 run removes `XZKEY` after writing its converged amplitudes,
and the key of amplitudes that are not converged (section 1: saved every
few cycles and at `CC_MAXCYC`) names their model, so no CCSDT run takes or
continues them, no CC3 run those of CCSDT - a CC3 run continues only its
own unfinished iterations.

CC3 of docs/ccn.md: the singles and doubles equations of CCSDT, the triples
equation to the lowest order, `<T| [V~, T2] + [F, T3] |0> = 0`, F the Fock
operator of the reference.  The two routes of the Python program, with its
numbers as the reference: T3 stored (a switch of the CCSDT step,
`ccsdt_opt.update_amps(model='cc3')`) and no T3 (the batches of occupied
triples, `cc.cc3fly`, the kernel `csrc/cc3_fly.c` - docs/ccn.md, sections 9
and 10).  The model is the variable `cc3mode` of `xzdata` (0 CCSDT, 1 CC3
with T3 stored, 2 without); the output says CC3 where it said CCSDT, the
energy goes to JOBARC as `TOTENERG`, the amplitudes to `XZAMPS` (without T3
there: its count in the header is 0), the excited states to `XZEOM` and
`TOTENER2` (section 16.1).

### 17.1 The Fock operator: canonical spinors

The triples equation takes F as the diagonal of the spinor energies
(`scf2ceig`; `[F, T3]` is `(e_abc - e_ijk) T3`), which is the model only for
canonical spinors.  The program has no Fock matrix of its own; it reads that
of CFOUR's SCF in the AO basis (JOBARC `scf2cfck`) with its spinors
(`scf2ccmo`), transforms it to the active spinors and prints the largest
elements off the diagonal - nothing is cut and nothing rotated:

    Fock matrix of CC3: the spinor energies of scf2ceig; the Fock matrix of the SCF (scf2cfck, scf2ccmo)
      on the active spinors: largest |F_oo| off the diagonal   3.31E-13,  |F_vv|   5.27E-13,  |F_ov|   3.00E-12;  |F_pp - e_p|   3.44E-15

(HBr / cc-pVDZ-DK with `%dglin`; H2O / 6-31G: 1.3e-13, 1.1e-13, 4.3e-13,
3.8e-15.)  Above 1e-6 off the diagonal of F_oo or F_vv the run stops with a
message (`focktol` raises it); without the two records, or when the
diagonal of that matrix is not `scf2ceig` (integral files written by
another program, e.g. `bench/cfour/labelled_run.py`), the line says that the
reader has the orbital energies only and canonical spinors are assumed.

### 17.2 T3 stored (`cc3mode` 1)

The CCSDT step of section 4 (`xzstep.inc`) with the T3 couplings of the
triples equation off, as the Python stored route:

| stage of the CCSDT step | CC3 |
|---|---|
| pass 1 (T3 -> T1, T2, W_vvvo, W_vooo) | the T3 terms of R1 and R2 kept; the T3 parts of W_vvvo, W_vooo not added |
| W_vvvo | the T1-dressed `<ab\|\|ei>` alone (`xzbv` and the rows of the dressed `<ab\|\|cd>` in the sweep; no `xzwvoovo`, `xzwvx`) |
| W_vooo | the dressed `<ma\|\|ij>` alone (`xzwo3`) |
| sweep | the CCSD ladder and the part of W_vvvo; the pp operator of T3 and its ladder left out |
| pass 2 | the drive terms alone, T3 not read: dense `zc_t3_driver` (`csrc/cc3_native.c`, the drive of `zc_t3_drive` for a range of pairs with the buffer of the caller - so the batches go to the MPI ranks as those of pass 2), in blocks `xzbpass2` with `noring` (the gather and the ring product skipped) |
| hh ladder, F_oo | `[F, T3]` by its diagonal (`zc_t3_fdiag`, per block; with slices on the rows of the rank) |
| update, DIIS, MPI | as CCSDT |

The excited states (`cc3_stored_eom`): the EOM-CCSDT sigma vector of
section 16.2 with the same switches - `<ab||ei>' = - P(ab) r1[m][a]
~<mb||ei> + W_pp r1` less the F_vv and the T2 parts of W_pp (`xzewv0`,
`xzwvt2out`), `<ma||ij>'` the derivative of the dressed integral alone,
the pass 1 over R3 without its parts of W', no pp ladders of R3 and of T3,
`[F, R3]` by the diagonal, the T3 residual with the derivatives of the
drive operators only - in every label shift and mode, with the solver and
input of section 16.

### 17.3 No stored triples (`cc3mode` 2)

`xzstep3` (`xzcc3.inc`): the CCSD part of the CCSDT step - the dressing,
the vovv term of R1 from the pass-1 operand G, the dressed `<ab||ij>`,
W_ovvo, F_vv, F_oo, the sweep over the rows of the dressed `<ab||cd>` with
the CCSD ladder and their part of the dressed `<ab||ei>` - then the T3 terms
of R1 and R2 from all batches of occupied triples (`xzfly`), then the CCSD
part and the Jacobi step (`xzccsd`).  The operands of `zc_cc3_batch`
(docs/ccn.md, section 10) in the native layouts:

| operand | from |
|---|---|
| At | T2 `[i][j][a][b]` as it is |
| Ao `[hi][lo][c][m]` | the dressed ovoo block `ovood[m][c][hi][lo]` (`xzao3`) - the operand that pass 2 has as `wo[m][c][(lo<hi)] = - ovood[m][c][lo][hi]`; the table of docs/ccn.md writes it in the convention of the Python containers |
| Bv `[e][i][Q]` | W_vvvo of CC3 transposed (`xzbvq`) |
| Bo | `t2[m][i][Q]` as it is |
| Gv `[m][Q][y]` | the columns `no + y` of the pass-1 operand G (`xzgvq`) |
| Gsq | the dressed `<mn\|\|ie>` as it is |
| gsq | `joovvp` (`<mn\|\|Q>`): the negative of the kernel's `<lo hi\|\|Q>`, so the singles term is folded in with the other sign |
| fov | the dressed `f[m][e]` |

n3 is 0 in this mode: T3, its residual and the T3 parts of the DIIS
vectors are empty regions, DIIS runs on T1 and T2, the slices of section 15
are off.  The batches: per pair (j < k) and class s of i the i < j of class
s, as many per call as the work of a batch holds - a parameter of the plan
(section 4): at least one occupied triple of the largest group
(`zc_cc3_work`), raised up to 2 GB or all the triples of a pair, after the
slab of the dressed `<ab||cd>` (the batches of pass 1, pass 2 and the pp
ladder are not used); `cc3w n` gives it (elements, at least one triple).
The plan prints it:

    CC3: the work of a batch of occupied triples:          41958 elements (       0.3 MB; one triple:         6993)

and the floor stops the run with the words needed as any other (section 4).

**MPI.**  Every rank runs `xzstep3` on the amplitudes that rank 0 sends;
in `xzfly` it does the pairs (j, k) that the cost model gives it - the cost
of a pair the work of its batches (triples times the work per triple of
their group), the most expensive first, each to the rank with the least so
far; every rank makes the same list - and the four sums (o^2 v^2 + 2 o^2
v(v-1)/2 + o v numbers) are summed on all ranks (`xzmpi_allsum_w`, chunked,
`MPI_Allreduce`).  The fold, the CCSD part (rank 0) and DIIS (rank 0) follow.

### 17.4 EOM-CC3 folded (`cc3_eom`)

The folded eigenproblem of docs/ccn.md, section 4, in the space of singles
and doubles.  The sigma vector at a frequency omega (`xzesig` with
`efold`): the singles-doubles block of section 16 (the sigma vector without
R3 and without the terms of T3) plus `xzefold`:

- `<ab||ei>'` (`xzewv0` with the dressed `<mb||ei>`, the sweep over W_pp
  with r1, the F_vv and T2 parts taken out) and `<ma||ij>'` of H'
  (`xzehs`);
- the batches with nset = 2: (t2, Ao', Bv', t2) and (r2, Ao, Bv, r2) over
  `omega + e_ijk - e_abc`, contracted with the operands of the ground state
  -> sigma_1, sigma_2;
- the batches with nset = 1: the operands of the ground state, omega = 0,
  contracted with those of H' - `Gv' = sum_n r1[n][y] <mn||Q>` (`xzgv1`),
  `<mn||ie>'`, f' - -> sigma_2.

The operands of the ground-state batches (Ao, Bv, Gv) are kept with the
intermediates of section 16.2 (`e3ao`, `e3bv`, `e3gv`); a sigma vector takes
o^3 v + 2 o v^2 (v-1)/2 more for its own and the four sums.

The outer loop (`xzeom3`, the algorithm of `ccn.folded_roots`):

1. the roots of the singles-doubles block from unit vectors on the lowest
   single excitations (roots + 2 of them), to 1e-4 (no triples);
2. root by root, from the lowest: the solver of section 16.3 on the matrix
   at a fixed omega, started from the vectors of the step before and
   following the roots with the largest weight on them (`nfollow`), with a
   threshold that follows `|lambda(omega) - omega|` down to `ESTATE_CONV`;
   omega replaced by lambda(omega), or by the secant step of the last two;
   converged when `|lambda - omega|` is below `ESTATE_CONV / 100`
   (`omegatol`) and the residual below `ESTATE_CONV`.  Roots of stage 1
   closer than 5e-4 Eh to a neighbour (`foldwin`) form a cluster, solved
   together at every frequency and taken by their place in it: following
   one vector alone ran the second component of the nearly degenerate
   lowest triplet of H2O (split by 2.8e-7 Eh) into the first (the Python
   folded solver has the same weakness: its third root of that case is
   2.3e-7 off).

Per frequency step a line: root, step, omega, lambda(omega), their
difference, the largest element of the residual, sigma vectors, seconds;
per root its omega, the frequencies and sigma vectors it took.  In blocks of
the labels one problem per label shift of `%xzeom` (below); without labels
the states of all shifts are roots of one problem.  No file `XZEOMR`.

**Label shifts** (`csrc/cc3_shift.c`, added: `zc_cc3_batchs`, the batch
kernel with four shifts in classes; all zero it is `zc_cc3_batch`, and the
native program calls it everywhere).  For a vector of shift s the triples
of a batch of occupied triples of group g are the virtual triples of group
g + s:

| argument | meaning | triples of the vector | ground-state triples with H' |
|---|---|---|---|
| `st` | of the triples: virtual group g + st (layout of the work, tables, `ev3`) | s | 0 |
| `sa[z]` | of the amplitudes of drive set z: `At[P][c][e]` with class(c) + class(e) = class(P) + sa, `Bo[m][i][Q]` with group(Q) = class(m) + class(i) + sa | 0 (t2), s (r2) | 0 |
| `sw[z]` | of its operators: `Ao[P][c][m]` with class(c) + class(m) = class(P) + sw, `Bv[e][i][Q]` with group(Q) = class(e) + class(i) + sw; sa + sw = st | s (H'), 0 | 0 |
| `sc` | of the contraction operands: Gv with class(y) + class(m) = group(Q) + sc, Gsq with class(x) + class(d) = class(P) - sc, gsq with group(Q) = class(P) - sc, fov with class(d) = class(m) - sc; the four sums have the shift st + sc | 0 | s |

In the kernel that is: the summed index e of a drive product has the class
class(P) + sa - class(c), m the class class(P) + sw - class(c), per set (one
class for both in the unshifted kernel); H and the amplitudes are those of
group g + st; the four contractions take the classes above.  The Fortran
side passes s = the shift of the vector (`xzefold`), the work per triple is
that of group g + s, and sigma_1, sigma_2 are masked to the shift as in
section 16.2.  **What the Python batch route would need** (not changed
here): `cc3fly.Fly.run` with these four shifts - the kernel is in the
library's directory already, only not in its build list -, `per_i` and the
batches with g + st, `eom_opt.sigma` passing (s, {0, s}, {s, 0}, 0) and
(0, 0, 0, s) for a `shift` in the fly branch (the masks m1, m2 of the shift
are there), and `ccn.eom_folded` taking the singles of the shift as the
guess (`eom.symmetric_singles(eris, shift)`).

**More than one rank.**  The folded EOM runs over the ranks (EOM-CCSDT
does too, in another way: 16.7; the stored route of CC3 does not): rank 0
runs the solver; it sends the
converged T1, T2 once, every rank makes the intermediates of the ground
state, and for every folded sigma vector rank 0 broadcasts the vector, the
frequency and the shift (`xzefsend`), every rank computes the same sigma
vector with its share of the batches (`xzfly`: the four sums are summed on
all ranks) and the other ranks drop theirs.

**A limit of the folded route**: it follows the lowest roots of the
singles-doubles stage.  A state that is dominated by doubles there lies
high in that stage and is not found unless enough roots are asked for:
HBr, shift 4 - the third root with R3 stored, 0.652304807 (weights of R1,
R2, R3: 0.025, 0.946, 0.028), is missing from three folded roots, which
give the stored fourth, 0.671637923, as their third.

### 17.5 Validation

Native against the Python program (the CC3 branch, commit c21f2b5, on the
same files: `zccsdt.cfour.from_cfour`, `GCC3.from_integrals`, `store_t3`
True / False, `eom_ee`; conv_tol 1e-12, roots 1e-9).  Reference directories
of section 16.5; H2O with `%eomthrs 1e-9` and four roots of shift 0.

| case | quantity | Python | native |
|---|---|---|---|
| H2O / 6-31G, X2C-1e (10 + 16), complex dense, CC_CONV=9 | E_corr(CC3), T3 stored | -0.13646018280846 | -0.136460182812 |
| | no T3 | -0.13646018280879 | -0.136460182806 |
| | EOM-CC3, T3 stored, shift 0 | 0.284178235164, ..515488, ..9182094, 0.311028170186 | 0.284178235179, ..515466, ..9182095, 0.311028170220 |
| | EOM-CC3 folded | (0.284178235126, ..514353, ..8954671: the third off by 2.3e-7, 0.311028170190) | 0.284178235134, ..515478, ..9182079, 0.311028170177 |
| HBr / cc-pVDZ-DK, `%dglin`, 28 frozen (8 + 28), CC_CONV=10 | E_corr(CC3), T3 stored (blocks and dense) | -0.14264585619981 | -0.142645856200 |
| | no T3 | -0.14264585620031 | -0.142645856200 |
| | EOM-CC3, T3 stored, shift 0 | 0.243652194013, 0.244106956035, 0.336657669623 | 0.243652193990, ..6955, ..9613 |
| | shift 2 | 0.237157681885, 0.260912563601, 0.336749228834 | 0.237157681858, ..563598, ..228823 |
| | shift 4 | 0.233152111958, 0.544409202291, 0.652304807416 | 0.233152111958, ..202364, ..807412 |
| | shift 6 | 0.738624111774, 0.773900482145, 0.786520835590 | 0.738624111703, ..482154, ..835680 |
| | EOM-CC3 folded, shift 0 | | 0.243652194009, 0.244106956022, 0.336657669619 |

Largest differences: energies 3.5e-12 (H2O, CC_CONV=9: the convergence),
2e-13 (HBr); roots with T3 stored 9e-11, folded against Python's stored
roots 3e-11 (H2O), 1.3e-11 (HBr).  The eomtest hook (16.5) on
`cc3_stored_eom`: the sigma vector against the five-point derivative of
the CC3 step, 4e-16, 5e-15, 2e-15 (6 classes, real) and 4e-16, 2e-15, 7e-15
(dense, complex) in sigma_1, sigma_2, sigma_3.

Label shifts in the folded route (HBr, `%xzeom` 2, 4, 6, 0, three roots
each, 1e-9), against the roots with T3 stored of the table above: shift 2
0.237157681878, 0.260912563587, 0.336749228828 (2e-11); shift 4
0.233152111948, 0.544409202278 (9e-11; the third: 17.4); shift 6
0.738624111778, 0.773900482120, 0.786520835593 (9e-11); shift 0 as
before.  The same digits in real blocks, complex blocks (`real 0`), with
one triple per batch on one thread, and with 2 and 3 MPI ranks; complex
dense on 2 ranks: the 12 roots of the dense table; H2O, 4 roots, 1 and 2
ranks: 0.284178235134, ..515478, ..9182079, 0.311028170177.

Modes (all the same to the digits printed): HBr in real blocks (6
classes), complex blocks (`real 0`), real dense (`labels 0`), complex
dense; the labelled PySCF spinors and the same with phases (`hbr_lab`,
`hbr_ph`: -0.142645856190 / ..191 with and without T3); one triple per batch
(`cc3w 1`); 1 and 8 threads; the MPI build with 2, 3 and 4 ranks on one
node (T3 stored also with `slices 0`): E(CC3) -2604.989088528366 / ..367
and -76.169163882631 / ..625 in every run.  The folded roots in real and
complex blocks, with one triple per batch and on one thread: equal to all
12 decimals of `XZEOM`; without labels, real and complex (12 roots of all
shifts), against the stored dense roots: 5e-11.  The memory floor:
`maxcor` below the least that CC3 takes stops with the words needed;
just above it the batches get smaller (11 928 elements instead of 41 958)
and the energy is the same.  `regress.sh` (serial with `--eom`, and 2
ranks) against the binary before this work: REGRESSION OK after each of the
three steps.

### 17.6 Timings

Job 31905441, node c048 (2 x Xeon Gold 6248R), 24 threads on the cores of
socket 1 with its memory, no other job on the node, the binary built there
from commit 12da931 (`cc3_time.sbatch` of the work directory).  HBr /
cc-pVQZ-DK, X2C-1e, `%dglin`, `DROPMO=1>5`, `%vircut 4.124587`: 26 + 120,
real, 10 classes (the case of 16.6; the chain took 23 s).  CC_CONV=8,
`%eomthrs 1e-6`, three roots of shift 0, MEMORY_SIZE 60 GB.  Measured once.

| | CC3, no T3 (`cc3_eom`) | CC3, T3 stored (`cc3_stored_eom`) | CCSDT (`ccsdt`, `ncyc 6`) |
|---|---|---|---|
| s per iteration | 1.54 (15 cycles) | 2.13 (17 cycles) | 4.10 |
| relative to CCSDT | 0.38 | 0.52 | 1 |
| a sigma vector | 2.58 s (folded, at one omega) | 3.23 s (EOM-CC3 with R3) | (16.6: 7.78 s on another node) |
| sigma vectors for 3 roots | 51 (26 + 7 + 18; 4, 3, 5 frequencies) | 34 | |
| EOM in all | 157 s | 171 s | |
| work array used | 4.0 GB | 39.6 GB (34 GB of it the subspace) | 14.5 GB (8 DIIS vectors in it) |
| peak resident set | 4.3 GB | 36.5 GB | 10.8 GB |
| E(CC3) / roots / Eh | -2605.341236402851; 0.242409172773, 0.242857579014, 0.332737188934 | -2605.341236402876; 0.242409336079, 0.242857728239, 0.332737269979 | |

The roots of the two routes differ by up to 1.6e-7 Eh at this threshold
(the largest element of a residual 1e-6: the error of 16.3), the energies
by 2.5e-11 (CC_CONV=8).  Of an iteration without T3, the batches are 14.8 s
of 23.2 s (64 %; 11.3 of 24 cores busy): products 3.5, amplitudes 2.1,
gather 4.0, contractions 5.2 s (the kernel's clocks); the rest is the CCSD
part of the step.  A folded sigma vector is two passes over the batches
(105 s of the 131 s of 51 sigma vectors) and the singles-doubles terms (41
s).  Against the Python program (docs/ccn.md, section 9, job 31901611,
another node of the same type, PySCF's spinors): an iteration 2.54 s
without T3 and 3.63 s with T3 stored, a sigma vector 4.2 and 5.4 s.

### 17.7 ThF+

Job 31905442, bigmem12, 24 cores (`thf_cc3.sbatch` of the work directory;
commit a1ca399, the program still with the method as argument).  ThF+ at
3.743547 bohr, X2CAMF, Th DYVTZ-SO + F ADYVTZ-SO, 70 spinors frozen: 28 +
330 active spinors, closed shell, `%dglin` 29 29 / 14 14 / 5 5 / 1 1: real,
12 classes (the case of the EOM-CCSDT job 31861231; `labtol 1e-5`,
`imagtol 1e-6` as there).  CC_CONV=7, ESTATE_CONV=5, MEMORY_SIZE 400 GB.
The chain took 10.5 min (`xx2cvtran` 389 GB resident - the largest of the
job).  The Fock check: largest |F_oo| off the diagonal 1.3e-11, |F_vv|
4.4e-9, |F_ov| 1.8e-10, |F_pp - e_p| 3.6e-11.

| | CC3, no T3 | CC3, T3 stored (3 cycles, `ncyc`) | CCSDT (job 31861231) |
|---|---|---|---|
| s per iteration | 37.75 (38 cycles, 23.3 cores busy) | 128.2 | 258.5 (59 cycles) |
| work array: plan | 67.3 GB with the EOM (the least: 65.9 GB) | 302.8 GB | 302.8 GB |
| peak resident set | 60.0 GB | 110.9 GB in 3 cycles (the DIIS vectors not all touched) | 298.1 GB |
| total energy / Eh | -26538.335568361723 | | -26538.328976480821 |

E(CC3) - E(CCSDT) = -6.59 mEh.  The work of a batch: 175 805 422 elements
(1.4 GB; one occupied triple 6 761 747).

EOM-CC3 folded, three roots of shift 0 (the totally symmetric states; the
largest element of a residual 1e-5, |lambda - omega| below 1e-7): the
singles-doubles stage 54 sigma vectors in 414 s (roots 0.02295, 0.02467,
0.04217 Eh); a folded sigma vector 77.8 s (81 of them in 6300 s; it takes
19.6 GB of the work array) = 2.1 CC3 iterations; EOM in all 6742 s.

| root | omega / Eh | cm-1 | frequencies, sigma vectors | max \|r\| |
|---|---|---|---|---|
| 1 | 0.019859519401 | 4358.66 | 4, 20 | 8.6e-6 |
| 2 | 0.020829587614 | 4571.57 | 5, 27 | 7.6e-6 |
| 3 | 0.039378072332 | 8642.49 | 4, 34 | 4.8e-6 |

(No EOM-CCSDT roots of shift 0 are at hand to compare; the states of
Omega = 1, 2, 3 are those of shifts 2, 4, 6.)

### 17.8 Not done

- The folded EOM: states dominated by doubles (17.4: they are not among
  the lowest roots of the singles-doubles stage); the vectors of the roots
  (no `XZEOMR`).  The Python batch route has no label shifts (17.4 says
  what it needs).
- EOM of CC3 with T3 stored (`cc3 stored`) over MPI ranks - the program
  stops; CC3 amplitudes are not reused (no `XZKEY` for them).
- Spinors that are not canonical: the program stops (17.1); the
  semicanonical rotation of docs/ccn.md, section 1, belongs before the
  integral transformation (CFOUR's spinors are canonical).
- IP / EA, left-hand vectors, properties of CC3; CC2 and CC4.

## 18. Options: the section %xzccsdt of the ZMAT

The program is controlled by the ZMAT alone, as CFOUR's programs are: the
job by `CALC`, `EXCITE`, `PROPS` (section 1), the thresholds by `CC_CONV`
(the largest element of the step of the amplitudes below 10^-N, as in
`xx2cccsd`; section 1), `CC_MAXCYC` (the amplitudes are saved when it is
reached, and a rerun continues; section 1), `ESTATE_CONV`,
`ESTATE_MAXCYC`, the memory by `MEMORY_SIZE`,
the excited states by `%eomnroot`, `%eomthrs`, `%eomndav0`, `%xzeom`,
`%xzeomguess`, `%xzeomvec` (16.1) - and everything else by one section of
its own (`xzopt.F90`):

    %xzccsdt
    cc3 stored
    labtol 1e-6

one option per line, the name first (either case), its value the rest of
the line (a name alone: 1); the section ends at an empty line, at the next
line that starts with `%` or at the end of the file.  A name that is not
an option stops the program with the list of the options.  **No
environment variable is read** (`OMP_NUM_THREADS` and those of the MPI
library are not ours); the variables `XZ_*` of earlier versions, which the
sections above still describe by the name of their option, are gone.  In an
MPI run every rank reads the ZMAT of its run directory.  An ordinary run
needs no option.

**User options.**

| option | default | what |
|---|---|---|
| `labtol x` | 1e-5 | the largest integral that the symmetry labels forbid and that is taken as zero; above it the labels are not used (section 6) |
| `imagtol x` | 1e-7 | the largest imaginary part of an integral that is taken as zero (real arithmetic); above it complex arithmetic |
| `labels 0` | 1 | the labels of `dgcosym` / `xzlabels` are not used: one block |
| `real 0` | 1 | complex arithmetic whatever the integrals |
| `cc3 stored` | batch | CC3 with T3 stored (17.2; with EXCITE=EOMEE the sigma vector with R3): a check route with the memory of CCSDT |
| `focktol x` | 1e-6 | CC3: the largest element of F_oo, F_vv of the SCF off the diagonal that is accepted (17.1) |
| `omegatol x` | ESTATE_CONV / 100 | EOM-CC3 folded: `\|lambda(omega) - omega\|` at convergence (17.4) |
| `foldwin x` | 5e-4 | EOM-CC3 folded: roots of the singles-doubles stage closer than this (Eh) are solved together |
| `start ccsd` | second order | the iterations from T1, T2 of CFOUR's `xx2cccsd` (`t1guess`, `t2guess`) |
| `hh pair` | op | the hh ladder of dense T3 through occupied pairs (section 12) |
| `qmap 0` | 1 | (Q): T3 in the work array of every rank instead of the mapped `XZAMPS` (section 5) |
| `eomvec f` | work array | EOM: the vectors of the subspace in files (16.4) |
| `eomvdir dir` | the run directory | EOM: the files of the subspace in `dir` (made if it is not there), e.g. a node-local disk; over ranks every rank its part (21.1) |
| `slices 0` | 1 | iterations over ranks without slices of T3 (section 15) |
| `bpr n` | 8 | batches per rank at least |
| `rebal x` | 0.03 | the batches are dealt again when the last rank is this fraction behind the mean (0: never) |
| `dist digits` | 123456 | the kinds of batches that are dealt to the ranks |
| `mpichunk n` | 2^23 | words of an MPI message |
| `qtasks n`, `qranks n` | from the ranks | tasks of (Q), ranks it is planned for (section 5) |
| `diis m8` / `f8` / `m0` | planned | with `plan`: the DIIS vectors in memory / in the file, their number |
| `lsolver diis` | krylov | the Lambda equations with the DIIS of the amplitudes instead of the Krylov solver (section 11) |
| `plan 8 numbers` | planned | the batch sizes nqs nbt nbp nba qxv qc22 qc13 qrows (0: as planned) |
| `cc3w n` | planned | CC3: elements of the work of a batch of occupied triples (at least one triple) |

**The defaults of `labtol` and `imagtol`** (changed with this section from
1e-8 and 1e-10, so that ordinary runs need no option): measured in the
integrals of CFOUR's chain with `%dglin`, the forbidden elements are 4.4e-7
to 6.1e-7 and the imaginary parts 1e-9 to 2e-9 for ThF+ (a basis with very
large exponents; with the old defaults the run dropped its labels or went
complex) and 1e-12 or less for light molecules; labels that are not
symmetry labels of the spinors leave elements of 0.1 and more (section 6:
0.118, 1.41), spinors that are not real imaginary parts of the size of the
integrals.  1e-5 and 1e-7 are a factor of 16 and 50 above the largest noise
seen and three or more orders below a violation.  What is below a
tolerance is dropped (set to zero), so the tolerances also bound what is
neglected; the program prints the largest values it found next to the
tolerance in every run:

    the largest imaginary part of the integrals:   1.50E-12 (below the tolerance  1.00E-07: taken as zero)

The two numbers are in `xzlayout` of `xzdrv.F90`.

**CFOUR keywords in the place of earlier variables**: `eomcyc` is gone -
`ESTATE_MAXCYC` of the ZMAT (the program takes its job from the ZMAT and
xjoda is run again when it changes); the amplitudes of CFOUR's CCSD as
the whole problem (`eomccsd`, `ccsd_eom`) went with the argument.

**Developer and test options** (the same section; not in the README of the
package):

| option | what |
|---|---|
| `ncyc n` | a timing run: n cycles of every iteration, converged or not (section 12) |
| `maxcor n` | only n words of the work array are used (rank 0; the floor and the plan, section 4) |
| `debug file` | one amplitude step on the amplitudes of the file, the result written to `file.out` (section 7) |
| `ldebug file`, `ddebug file` | the same for a Lambda iteration; dumps of the density stage |
| `qdebug` | (Q): its four parts printed |
| `den_active` | the density files of the active spinors alone |
| `wfail` | an error on a rank other than 0 (the exit through `MPI_Abort`, 15.2) |
| `eomtest seed` | the sigma vector against the derivative of the amplitude step (16.5); one rank |
| `deasig file`, `deaonly pieces`, `deabatch n` | EOM-DEA: the sigma vector of the vector of a file (the first target) to `file.out`; only the pieces named (pp hh ring x22 x13 row3 row2, as `dea_sector.Engine.sigma(only=...)`); batches of n elements (20.5); over ranks too |
| `deaxv m` / `deaxv f` | EOM-DEA: R13 (Xv) in the work array / in the file `XZDXV`, whatever fits (20.4) |
| `deaplan n1 n2 ...` | EOM-DEA: the memory of every rank for n1, n2, ... ranks from dry runs, then the program stops (one rank; 20.4, 21.5) |
| `dea5pool n` | EOM-DEA 5p3h: the pool of the quintuple lists in words (21.2) |
| `dea5z4 n` | EOM-DEA 5p3h: at most n slots of Z4 on every rank, not made over the ranks (21.3) |
| `dea5xb n` | EOM-DEA 5p3h over ranks: at most n rounds of the rows of Xv5 and E1x in one exchange (21.4) |
| `dea5xa 0` / `1` | EOM-DEA 5p3h on one rank: Xv5 k by k / of all k (21.3) |
| `eomsig file`, `eomsd` | the sigma vector of the vector of a file; its singles-doubles block (16.5); over ranks too (16.7) |
| `mpitest n` | the chunked transfers on arrays of n words (15.2) |
| `qsample x`, `qstop n` | (Q): a sample of the quadruples; stop after n tasks (restart test, section 5) |
| `cc4stop n` | CC4: the run stops after its n-th evaluation of the T4 terms (restart test, 19.5) |
| `qcost file`, `bcost file` | the seconds of the tasks of (Q) / of the batches of the iterations |
| `r0load a b` | a load of rank 0 by hand in the deal of the batches |
| `lkry n`, `lkry m<n>`, `lkry f<n>` | the Krylov solver of the Lambda equations with up to n pairs of vectors (3 .. 63; default 40, as many as fit), in the work array / in the file `XZLKRY` (section 11) |
| `lprec 0` | the Krylov solver: the singles of the step from the denominators instead of the singles block of the Jacobian (section 11) |
| `lritz n` | the Krylov solver: per cycle the norm of the step and of the minimal residual, at the end the n harmonic Ritz values of the smallest \|theta\| and the largest l1 elements of the softest direction (section 11) |

Checked (HBr / cc-pVDZ-DK with `%dglin`, `CALC=CC3`): no section - real, 6
classes, E(CC3) -2604.989088528367; `cc3 stored` - ..366; `imagtol 1e-12`
and `Labels 0` - complex (1.5e-12 found), one block, the same energy; a
line `foo 3` - the program stops with the two lists; the variables
`XZ_CC3=stored XZ_REAL=0` in the environment - no effect; `cc3w 1`, `real
0` on 2 ranks with EXCITE=EOMEE - the roots of 17.5.

## 19. CC4

    CALC=CC4                    CC4, the quadruples never stored
    CALC=CC4, EXCITE=EOMEE      CC4, then EOM-EE-CC4 with the quadruples folded in (the label shifts of %xzeom)
    mpirun -np 4 xzccsdt        the ground state and the excited states over ranks (19.4)

`CALC=CC4` is joda's level 52.  CC4 of docs/ccn.md (sections 1, 2, 11): the
singles, doubles and triples equations of CCSDTQ, the quadruples equation to
the lowest order, `<Q| [V~, T3] + 1/2 [[V~, T2], T2] + [F, T4] |0> = 0`, F
the Fock operator of the reference - the diagonal of the spinor energies:
the check of 17.1 runs for CC4 too.  The reference of every number is the
Python program (`ccn.cc4_kernel(store_t4=False)`, `cc.cc4fly`), which is
validated against MRCC's CC4.  Sources: `xzcc4.inc`, `csrc/cc4_native.c`
(added; not in the Python build), the loop of `xzcc.inc`, two lines of
`xzstep.inc`, and for the excited states the solver of section 16
(`xzeoms.inc`, `xzeomd.F90`).

A run of production size takes days, so every completed evaluation of the T4
terms is written down: a run of the same ZMAT in the same directory goes on
after the last one (19.5).

### 19.1 The T4 terms (`xzt4`, `xz4run`)

T4 enters the doubles and the triples equations through four sums that are
linear in it (docs/ccn.md 11).  `xzt4` makes them for given T1, T2, T3 -
`R2[P][Q]` and a rank-3 array in the layout of T3 - and nothing of the size
of T4 exists at any time:

1. **operands** (once per evaluation): the T1-dressed blocks with the
   routines of the CCSDT step (`xzfock`, `xzdroo`, `xzdrovvq`, `xzdrov`,
   `xzbv` and a sweep over the rows of the dressed `<ab||cd>` for
   `<ab||ei>~`), T2 in three layouts, Xo (`xz4xo`: `xzqxo` of (Q) with the
   dressed `<mn||jk>`, `<md||el>`).  **The rows of the dressed `<ab||cd>` of
   that sweep are kept** - one array in the blocks of the bare integral -
   for Xv.  (Q) reads `<ab||ei>` and `<ma||ij>` as conjugates of the stored
   `<ie||ab>`, `<ij||ma>`, which holds for bare integrals only: here the two
   dressed blocks are laid out as (Q) wants its operands, `- <ab||ex>~` as
   `[x][e][Q]` and `<ma||ij>~` as `[i][j][m][a]` (`xz4q12`), and nothing is
   conjugated.
2. **per occupied quadruple** - the task list of (Q) (section 5: groups of
   one largest index l that share Xv(x, l), dealt to the ranks) - a row of
   T4 is made and contracted.  In the sectors of the labels (`xz4sfw`,
   `zc_cc4_sector_t4i`, `xz4sbk`) the row lives twice, as in (Q): blocks on
   two pairs `[(ab)][(cd)]` (one per group of (ab)) and blocks on an index
   and a triple `[a][(bcd)]` (one per class of a), about 8 n_v^4 / (4
   classes) and 8 n_v^4 / (6 classes) bytes.  A pass over them is what an
   evaluation costs, so every block is touched as few times as the algebra
   allows:
   - *right-hand side, two pairs*: one product per block.  The rows of both
     kinds of splits - `- <ab||ex>~` with `t(J; e, cd)` and `t(x m, ab)` with
     `Xo(m; J; cd)` - are stacked along the summed index (all blocks of the
     row in one parallel loop over block and split) and the block is set by
     one GEMM;
   - *right-hand side, an index and a triple*: one pass over the columns of
     a block (`zc_cc4_rowsmm`).  Its right operands are rows that lie
     elsewhere - rows of T3 for `<ma||ij>~ t(m kl; bcd)`, rows of three Xv
     for `2 t(ij, ae) Xv(e; kl; bcd)` -; a thread takes blocks of 512
     columns, gathers the rows of a block into its cache and multiplies them
     there;
   - *the row over `e_ijkl - e_abcd`* in place (`zc_cc4_sector_t4i`): every
     place of the blocks with four different indices is one split of one
     virtual quadruple, so the quadruple reads its ten places, and the
     amplitude, with the sign of the split, is written over them.  The ten
     places are found from the tables for the first d of a run of
     consecutive d and stepped from there (one element where d is in the
     columns, one row where it is in the rows).  The places with a common
     index belong to no quadruple and are set to zero; nothing else is;
   - *contractions, two pairs*: a block is read once - one GEMM whose left
     operand has the rows of `<mc||ef>~` of the four splits (m | J) and
     those of `<mn||ef>` of the doubles term stacked; its result is the
     doubles term and, reduced (c | ab) onto the triples (`zc_cc4_red12`),
     the term `R3[J] += <mc||ef>~ X22`;
   - *contractions, an index and a triple*: one pass over the columns of a
     block (`zc_cc4_addmm`): `R3[J] += f~[m] X13` and `R3[(k, P)] +=
     <P'||ke>~ X13` are added to the rows of R3 where they are.

   Without labels (`xz4fwd`, `zc_cc4_divide`, `xz4back`) the row is one
   packed array C(v,4) and the products run in batches of columns, as in
   (Q): the buffers of a quadruple are those of (Q) plus `[nv][columns]` and
   `[4 + 6 o][columns]`.
3. **ranks**: every rank its tasks and its own two sums; the doubles term is
   summed on all ranks, the triples term onto rank 0 (chunked, section 15).

**Xv** is that of (Q) (`xzqsxv`, `xzqxv`) on the kept dressed `<ab||cd>`:
per pair (x, l) the block `[(bc)][e][f]` of a group of pairs and two classes
is read and multiplied with `t2[x][l][f][d]`.  (Until 2026-10-13 the block
was dressed again for every pair, with `<my||ef>` and `<mn||ef>` gathered
for all m: 34 s of 358 s for 26 + 120, 0.45 s of 0.79 s for 8 + 78.)

**What bounds an evaluation now** (19.8): the passes over memory.  For 26 +
120 a row is 67 MB, and a quadruple reads or writes about 0.4 GB (counted
from the sizes of the arrays, not measured) - 46 MB of stacked operands and
blocks on two pairs, 143 MB for the blocks on an index and a triple (the
rows of three Xv are 81 MB of it), the row once in place, and 130 MB in the
contractions (66 MB of it rows of R3) - in 11.8 ms on 24 cores of one
socket.  The two GEMMs with a long summed index (two pairs,
forward and back) are a fifth of the time.

### 19.2 The iterations: from CCSDT, macro / micro, extrapolated

The CCSDT step takes the two sums as fixed parts of its residuals (`cc4` of
`xzdata`; `jd42`, `jd43`: one array of the size of T3 more on every rank).
A T4 evaluation costs as much as 0.61 (Q) - 42 CCSDT iterations for 26 +
120 (19.8; 1.24 (Q) before 2026-10-13) -, so the iterations are arranged to
need few of them:

- **Start: the CCSDT amplitudes.**  If `XZAMPS` belongs to these integrals
  (`XZKEY`, section 1: a CCSDT run of this directory, converged at least as
  tightly) it is the start; otherwise CCSDT is solved first in the same run
  and written to `XZAMPS` / `XZKEY` exactly as a CCSDT run writes it.
  `XZAMPS` is always that of CCSDT, the start of CC4 and never its result;
  the CC4 amplitudes go to a file of their own, `XZCC4` (19.5).
- **Macro / micro.**  The T4 terms are evaluated at the present amplitudes
  (a cycle marked `T4 n`), the DIIS sequence starts again, and the equations
  of CCSDT with these terms are iterated - cycles of the cost of CCSDT -
  until the change of the amplitudes is below `cc4micro` (0.01) times that
  of the first step after the evaluation; then the terms are evaluated
  again.  The change of the amplitudes is its largest element (section 1,
  `CC_CONV`; until 2026-10-13 the sum of the three norms, together with the
  energy change - the tables below are of then).  Converged when the first step after an evaluation is below the
  threshold of `CC_CONV` (the test of a plain iteration), or - saving that
  last evaluation - when the first steps of the evaluations say that the
  next would be: `2 kappa d` below the threshold, d the last first step
  and kappa the largest factor between two of them that was seen (the
  output says so: `new T4 terms would change the amplitudes by about ..`).
  `cc4 confirm` asks for the confirming evaluation.
- **Extrapolation over the macro iterations** (`xz4acc`; Anderson with one
  step of history).  A macro iteration is a map G: amplitudes x, at which
  the terms are evaluated, to the amplitudes g that solve the equations with
  them.  With f = g - x of this and of the macro iteration before, the next
  evaluation is made at

      x' = g - theta (g - g_before),   theta = <f, f - f_before> / |f - f_before|^2

  (measured theta: -0.01 to -0.04; a theta beyond 1/2 in size is not
  taken).  The first steps then fall by a factor of 0.01 to 0.06 per
  evaluation instead of 0.04 to 0.08, which **saves one evaluation in five
  of the seven cases below and two in one**.  It takes three sets of
  amplitudes on rank 0 (x, g_before, f_before: three times T3 and T2); the
  plan puts them into the work array if they fit there before any batch
  grows, and says so if they do not (`no room for its three sets of
  amplitudes: about one T4 evaluation more`).  `cc4acc 0` switches it off.
- `cc4 plain`: the T4 terms in every iteration (from the CCSDT amplitudes);
  `cc4 scratch`: the same from the second-order amplitudes without CCSDT -
  the iteration of the Python program.

Every macro iteration prints the energy that its T4 terms give and the
factor by which the first step fell.

**T4 evaluations to convergence** (all runs converge to the same energy
within the threshold; 8 threads, the holding jobs; QZ: 24 threads, 19.8):

| case | CC_CONV | macro / micro (default) | `cc4acc 0` | `cc4 confirm` | `cc4 confirm`, `cc4acc 0` | `cc4 plain` (from CCSDT) | `cc4 scratch` |
|---|---|---|---|---|---|---|---|
| H2O / 6-31G (10 + 16) | 9 | 5 | 6 | 6 | 7 | 13 | 17 |
| | 7 | 4 | 5 | 5 | 5 | 9 | 13 |
| HBr / cc-pVDZ-DK (8 + 28) | 10 | 5 | 6 | 6 | 7 | 15 | 19 |
| | 7 | 4 | 4 | 4 | 5 | 9 | 14 |
| HBr / cc-pVTZ-DK (8 + 78) | 10 | 6 | 8 | 7 | 8 | 17 | 21 |
| | 7 | 4 | 5 | 5 | | 11 | 14 |
| HBr / cc-pVQZ-DK (26 + 120; 19.8) | 7 | 5 | 6 | | | 13 | |

The first steps of 26 + 120, CC_CONV=7 (the change of the amplitudes; job
of the holding node, 24 threads): 1.6e-2, 1.2e-3, 5.0e-5, 1.5e-6, 9.2e-8
with the extrapolation, 1.6e-2, 1.2e-3, 9.0e-5, 7.2e-6, 6.0e-7 and a sixth
evaluation without; E(CC4) -2605.340463043361 and ..043035 (3.3e-10 apart,
CC_CONV=7).

**How good are few evaluations?**  The error of the correlation energy
after the equations with the T4 terms of evaluation n are solved (Eh; the
effect of T4 is E(CC4) - E(CCSDT); without the extrapolation, which acts
from n = 3 on):

| case | E(CC4) - E(CCSDT) | n = 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| H2O / 6-31G (10 + 16) | -4.65e-4 | 5.1e-6 | 2.5e-8 | 2.2e-9 | 2e-10 |
| HBr / cc-pVDZ-DK, 28 frozen (8 + 28) | -2.13e-4 | 2.0e-7 | 3.5e-8 | 1.9e-9 | 9e-11 |
| HBr / cc-pVDZ-DK, 18 frozen (18 + 28) | -2.03e-4 | 1.1e-6 | 9.5e-8 | 4.6e-9 | 2e-10 |
| HBr / cc-pVDZ-DK, 10 frozen (26 + 28) | -1.88e-4 | 2.4e-6 | 1.7e-7 | 8e-9 | 4e-10 |
| HBr / cc-pVTZ-DK, 28 frozen (8 + 78) | -2.21e-4 | 6.6e-6 | 4.3e-7 | 2.3e-8 | 1.4e-9 |
| HBr / cc-pVQZ-DK, 10 frozen, 120 virtual (26 + 120) | +1.01e-4 | 3.4e-5 | 2.1e-6 | 1.2e-7 | 7e-9 |

One evaluation on the converged CCSDT amplitudes gives 97 to 99.9 % of the
effect where the effect is ordinary and two thirds of it for 26 + 120
(where E(CC4) - E(CCSDT) is a small sum of larger parts: (Q) is +8.7e-5
there); every further evaluation gains a factor of 12 or more.  Three
evaluations are within 1.2e-7 Eh in all cases (26 + 120 with the
extrapolation: 5e-8 after three, 1e-9 after four).

### 19.3 EOM-EE-CC4 (`xze4`, `xzeom4f`)

The folded eigenproblem of docs/ccn.md, section 4, in the space of singles,
doubles and triples: `[A_PP + dA(omega)] R = omega R` with A_PP the matrix
of EOM-CCSDT at the CC4 amplitudes (section 16, unchanged) and dA the T4
terms.  `N(xzeom)` solves A_PP first (its roots are the start), then
`xzeom4f` refines every root.

**The T4 terms of a sigma vector** (`xze4`, called by `xzesig`; one pass
over the occupied quadruples, `xz4run` with `eom`):

    r4 = ([V', T3] + [V~, R3] + [[V', T2], T2] / 2 + [[V~, R2], T2]) / (omega + e_ijkl - e_abcd)   with H~   -> sigma_2, sigma_3
    t4 of the ground state (made again)                                                            with H'   -> sigma_2, sigma_3

H' = [H~, R1]: `<ab||ei>' = - P(ab) r1[m][a] <mb||ei>~ + <ab||ef>~ r1[i][f]`
(the second part from the dressed `<ab||cd>`, of which `xzeomim` keeps a
copy for CC4 - the pp operator is in the place of the bare one - that
`xz4eomim` dresses where it is), the derivatives of the small blocks that
`xzesig` has (`xzehs`, `xzeh1`), `<mb||ef>' = - r1[n][b] <mn||ef>`, and for
Xv `<bc||ef>' = - P(bc) r1[m][b] <mc||ef>~`, made once per sigma vector in
blocks of the shift (rows of group q, pairs (ef) of group q - s).  The
right-hand side of r4 is not made by polarization (the Python program:
three passes with T +- R) but directly, in two calls of the products: (H'
operands; t2, T3; Xo and Xv of (H', t2) plus those of (H~, r2), which enter
with the same left operand and are summed once) and (H~ operands; r2, R3;
Xo, Xv of the ground state).  So a sigma vector with the T4 terms is three
right-hand sides and two contractions - measured 2.3 to 2.4 T4 evaluations of
the ground state (19.8) - and holds two Xo and two sets of Xv.

**Label shifts.**  Every operand of a product has a shift in classes (0:
ground state; s: r1, r2, R3 and H').  A quadruple of total class M has its
r4 row on the virtual quadruples of total class M + s, and every range of a
summed index, every block of the rank-3 operands (the layout of the shift,
16.2) and of Xv follows from the shifts of the two operands of its product
(`xz4sfw`, `xz4sbk`, `xz4sxv`, `xz4slay`); the ground state is the case of
all shifts zero of the same routines.  Without labels all states are roots
of one problem (16.1).

**Macro / micro for a root** (the default).  A sigma vector with the T4
terms costs several T4 evaluations of the ground state, one without them
two CCSDT iterations.  So per macro iteration the T4 terms are evaluated
once on the present vector q of the root, at its present frequency (F =
[A_PP + dA(omega)] q), and held on q while the eigenproblem of

    B x = A_PP (x - q (q|x)) + F (q|x)

is solved with EOM-CCSDT sigma vectors (`xzesigm`; the solver of 16.3,
started from q); its root and vector are the next omega and q.  Roots of the
start closer than `foldwin` form a cluster as in 17.4: q is an orthonormal
basis of their vectors, each evaluated at the frequency of the root in
hand.  Converged when omega changes by less than `omegatol`
(ESTATE_CONV / 100), or when the ratio of the last two changes times the
last says the next change is below it.  Measured: the change falls by a
factor of 7 to 20 per evaluation (the T4 terms act strongly on the R3 part
of the vector, which follows them only in the next macro iteration).

`cc4eom davidson`: the solver on the folded matrix at fixed omega, every
sigma vector with the T4 terms, omega by the secant step - the algorithm of
17.4, as a check.

**The file `XZEOMR`** has the vectors of the roots in the format of the
other EOM methods (16.1; not with `%xzeomvec 0`): a root is written when it
is done with the T4 terms - singles, doubles and triples at its own
frequency -, so that a later run goes on behind it (19.5).

**Is one evaluation enough?**  Per root the output has (i) omega without
the T4 terms (A_PP at the CC4 amplitudes), (ii) the Rayleigh quotient of
that vector with the folded matrix - one evaluation, the vector not relaxed
-, (iii) omega after the first macro iteration - one evaluation, the vector
relaxed with the T4 terms held -, and the converged omega:

| case, state | (i) without T4 | (ii) Rayleigh | (iii) one evaluation | converged | error of (ii), (iii) |
|---|---|---|---|---|---|
| H2O / 6-31G, lowest triplet (three components) | 0.284570900 | 0.284247741 | 0.284432987 | 0.284455231 | -2.1e-4, -2.2e-5 |
| lowest singlet | 0.311385371 | 0.311075218 | 0.311258820 | 0.311283423 | -2.1e-4, -2.5e-5 |
| HBr / cc-pVDZ-DK (8 + 28), shift 0, root 1 | 0.243634874 | 0.243491684 | 0.243559530 | 0.243564745 | -7.3e-5, -5.2e-6 |
| shift 2, root 1 | 0.237140998 | 0.236997087 | 0.237064960 | 0.237070154 | -7.3e-5, -5.2e-6 |
| shift 2, root 2 | 0.260930000 | 0.260746767 | 0.260825030 | 0.260830785 | -8.4e-5, -5.8e-6 |
| shift 4, root 1 | 0.233133231 | 0.232992070 | 0.233059167 | 0.233064310 | -7.2e-5, -5.1e-6 |
| shift 4, root 2 (doubles) | 0.544495117 | 0.544284210 | 0.544512513 | 0.544532240 | -2.5e-4, -2.0e-5 |
| shift 6, root 1 | 0.737989434 | 0.737862060 | 0.737851629 | 0.737850626 | +1.1e-5, +1.0e-6 |
| HBr / cc-pVQZ-DK (26 + 120), shift 2, root 1 (ESTATE_CONV 1e-5) | 0.235277196 | 0.234961321 | 0.235142242 | 0.235170618 | -2.1e-4, -2.8e-5 |

(Hartree.)  The Rayleigh quotient with the right-hand vector alone is not a
usable estimate: for the states dominated by single excitations it
overshoots the effect of the T4 terms by a factor of two to three (the
matrix is not Hermitian, and the first-order formula wants the left-hand
vector, which this program does not have).  One evaluation followed by the
cheap relaxation - (iii) - has 73 to 99 % of the effect of the T4 terms for
these states (half of it for the doubles-dominated root of shift 4), an
error of 1e-6 to 2.8e-5 Eh (up to 0.8 meV, 6 cm-1); the second evaluation
brings it to 2e-6 Eh or better in the small cases and to 7e-8 for 26 + 120.
So one evaluation is enough for 1e-4 Eh, not for 1e-5.  Note that
most of what CC4 changes in an excitation energy is in (i) already, through
the ground-state amplitudes.

### 19.4 EOM-EE-CC4 over MPI ranks

    mpirun -np 4 xzccsdt        # CALC=CC4,EXCITE=EOMEE; no option

The excited states run over the ranks in every mode (real and complex, the
blocks of every label shift and dense, both solvers).  Two things are
distributed, each as its ground-state or EOM-CCSDT counterpart is:

- **The T4 terms of a sigma vector** as those of the ground state: the pass
  over the occupied quadruples goes over the tasks of (Q), every rank its
  tasks (`xz4run`).  A rank adds its part of the triples term to all rows
  of its sigma_3 and its part of the doubles term to an array of its own;
  the doubles term is summed on all ranks (`MPI_Allreduce`), and the rows of
  sigma_3 go to their owners in the one exchange that ends a distributed
  sigma vector (16.7) - the T4 terms are made before it.  Every rank makes
  the operands itself (the kept ones once, `xz4eomim`; those of the vector
  per sigma vector), as it makes the intermediates of 16.7.
- **The rest of a sigma vector and the vectors of the solver** by the route
  of 16.7: rank 0 runs the solver and commands; every rank does its batches
  of the two residuals and holds its rows of R3 of every vector of the
  subspace.  Macro / micro adds the two sets of vectors q and F: **every
  rank holds its rows of them** (kinds 3 and 4 of the vector store of the
  solver, `xzevput`, `xzevmdot`, `xzevcomb`, `xzevget`), and the matrix B of
  19.3 is applied by rows - the overlaps (q|x) summed onto rank 0, `x - q
  (q|x)` on the rows of every rank, the distributed sigma vector, `+ F
  (q|x)` on the rows; the vector of the basis comes back from the store.
  Three commands were added: `fold` (the sigma vectors that follow have the
  T4 terms at this frequency), `qopen` (the place of q and F), `get` (a
  vector of the store into the work vector).  Rank 0 has the whole vectors
  of the roots of the problem (the start of every root, the result).

**What a rank holds** (the plan of rank 0 runs the EOM stage as itself and
as another rank, 16.7; the excited states of CC4 are in it):

| | every rank | rank 0 besides |
|---|---|---|
| for all sigma vectors | integrals with the pp operator, T1, T2, **T3**, the kept intermediates of 16.2, the dressed `<ab||cd>`, **Xo of the ground state** | the vectors of the roots (roots x a vector) |
| a sigma vector with the T4 terms | 2 whole vectors, the arrays of 16.2, **Xo of the vector**, `<bc||ef>'`, two sets of Xv, the row of one quadruple with its stacks | |
| macro / micro | its rows of q and F (2 x roots of a cluster x its slice) | with the singles and doubles |
| the solver | its rows of the subspace (16.7) | |

Measured (MB; HBr / cc-pVDZ-DK 8 + 28, shifts 0, 2, 4, 6, two ranks): rank
0 plans 27.4 and holds up to 10.3 of the subspace besides, used 29.7; the
other rank plans 28.0, used 24.5.  For 26 + 120: 19.8.

**Xo is the large piece: 40 GB per rank at the size of ThF+, twice (ground
state and vector), and the same numbers on every rank.**  The ranks of a
node could share Xo of the ground state, the dressed `<ab||cd>` and T3 - all
read-only during the excited states - as (Q) shares T3, by mapping a file;
Xo of the vector is the same on all ranks as well but changes with every
sigma vector, so sharing it means one rank of a node writing it and the
others waiting.  **Not implemented**: the four arrays are regions of the
stack of the work array, which the plan runs dry rank by rank; taking them
out needs a second kind of region (mapped, per node, written by one rank),
its place in the plan, the wait of the other ranks per sigma vector, and a
file of 40 to 80 GB per node in the page cache or in `/dev/shm`.  It would
save 80 GB and T3 for every further rank of a node; on one rank per node
(the layout of 19.9) it saves nothing.

**Validation** (holding job 31915920, node c432; the roots of `XZEOM`,
12 digits printed; serial against ranks):

| case | runs | largest difference of a root / Eh |
|---|---|---|
| HBr / cc-pVDZ-DK, `%dglin`, real blocks, shifts 0, 2, 4, 6 (2, 2, 2, 1 roots; shift 0 a cluster of two), 1e-8 | serial; MPI build on 1, 2, 3, 4 ranks (8 and 4 threads per rank); `eomvec f` on 2 ranks; 3 ranks of one thread with `plan 5 2 1 1 3 0 0 0`, `mpichunk 3001`, `slices 0` | 0 (all 12 digits) |
| the same, `cc4eom davidson` | serial, 2, 3 ranks | 0 |
| the same, complex blocks (`real 0`) | serial, 2, 3 ranks | 0 |
| H2O / 6-31G, complex dense, 4 roots (a cluster of three), 1e-9 | serial, 2, 3, 4 ranks | 0 |
| HBr, complex dense (`labels 0`, `real 0`), 4 roots | serial, 2, 3, 4 ranks | 0 |
| HBr, real dense (`labels 0`), 14 roots, 1e-8 | serial, 2, 4 ranks | 6.3e-10 (roots 10, 11, 13, 14: clusters; the solver of the stage without T4 terms takes another path over ranks, 16.7; threshold 1e-8) |

E(CC4) is that of the serial run in all of them.  `regress.sh --eom`
serial and `--np 2 --eom`: REGRESSION OK.

### 19.5 Checkpoint and restart

Nothing has to be asked for: the files are written by rank 0 in its run
directory, each as `<name>.new` that then takes the name (an interrupted
write leaves the file of before), each with the numbers of `XZKEY` (the
active spaces, E(SCF), the sums of the spinor energies: section 1), so that
the files of another calculation in the directory are not taken.  **A run
of the same ZMAT in the directory goes on after the last completed
evaluation and says so**:

      amplitudes: the file XZCC4 of an earlier run of these integrals: its CC4 iterations after    2 T4 evaluations
                  (the amplitudes of the last completed one and its T4 terms) - this run CONTINUES from there

| file | written | content |
|---|---|---|
| `XZCC4` | after every T4 evaluation of the ground state (macro / micro), and at convergence | the amplitudes at which the evaluation was made, its two held terms (R2, R3), the count, g and f of the extrapolation; at convergence the converged amplitudes alone |
| `XZE4R.nnn` | per problem (label shift) nnn of the excited states: when the roots without the T4 terms are there, and after every root | these roots and the vectors (the start), the roots that are done (the line of the table), the place in `XZEOMR` |
| `XZE4Q.nnn` | after every T4 evaluation of the root in hand; removed when the root is done | q, F, the frequency of the evaluation, the counts |

What a rerun does:

- `XZCC4` with a completed evaluation: the amplitudes and the held terms are
  read (over ranks: rank 0 sends the doubles term), and the equations with
  them are iterated - the first step is the first one after that
  evaluation, DIIS from new, as in the run that wrote the file.  The
  micro iterations that the earlier run did after it are done again (cheap).
- `XZCC4` with converged amplitudes, `CC_CONV` at least that of this run:
  taken, no iteration (`taken, the CC4 iterations are skipped`).  Converged
  less tightly: they are the start (an evaluation first).  **This is what
  the excited states of a later run start from**; `XZAMPS` stays the CCSDT
  file.
- `XZE4R.nnn`: the stage without T4 terms of that problem is skipped, the
  roots that are done are printed from the file, and the root in hand
  continues - with `XZE4Q.nnn` from the eigenproblem of its last evaluation
  (q, F and the frequency from the file; q is also the start of the
  solver), otherwise from its first evaluation.
- The files of a finished calculation stay; a rerun prints the result
  without an evaluation.  A file of another mode (option `labels`, `real`
  changed) stops the run with the message of `XZAMPS`.  `cc4 plain` and
  `cc4 scratch` write the converged amplitudes only, and `cc4 scratch` does
  not read `XZCC4`; `cc4eom davidson` has no files of its roots but
  `XZEOMR`.

`cc4stop n` (a test hook) stops the run after its n-th T4 evaluation -
those of the ground state and the sigma vectors with the T4 terms counted
together.

**Check** (HBr / cc-pVDZ-DK 8 + 28, CC_CONV=10; `XZEOM` to 12 digits; the
counts are evaluations of the earlier runs + of the last run):

| chain | result | evaluations |
|---|---|---|
| CC4: stopped after evaluation 2 (serial), continued on 3 ranks | E_corr -0.143067769768; theta of the three extrapolations as in the uninterrupted run (-0.00938, -0.03367, -0.01038) | 2 + 3 = 5 |
| stopped after 3 (2 ranks), continued serial; a third run | the same; the third run takes the converged amplitudes | 3 + 2 = 5; 5 + 0 |
| HBr / cc-pVTZ-DK (8 + 78): stopped after 4 | E_corr -0.199127895284 | 4 + 2 = 6 |
| `cc4acc 0` (the state of commit 7562fba): stopped after 1, after 2, after 3, on 1 to 3 ranks | E(CC4) -2604.989510441935 in every chain | 2 + 4, 3 + 3, 1 + 2 + 3 = 6 |
| CC_CONV=11 on the file of CC_CONV=10 | the amplitudes are the start | 6 + 2 |
| EOM-CC4, 4 shifts (7 roots): stopped inside the cluster of shift 0 (2 sigma vectors with the T4 terms), again after its first root and the first evaluation of the second, again inside shift 2; 1, 2, 1, 3 ranks | all roots to the 12 digits; `XZEOMR` as that of the uninterrupted run (1 - \|<v\|v'>\| below 1e-15); a fifth run prints everything from the files | shift 0, root 1: 10 = 2 + 8; root 2: 4 = 2 + 2; shift 2, root 1: 5 = 1 + 4 |
| H2O / 6-31G, the cluster of three: stopped after 6 and after 12 sigma vectors with the T4 terms; 1, 2, 1 ranks | the four roots to the 12 digits | 21 = 6 + 6 + 9 |
| the same with `XZEOMR` removed after the first run | the roots; the program says that the file of the earlier run is not there and writes none | |

### 19.6 Memory

The plan (section 4) runs the scheduler dry through the T4 evaluation as a
stage of the iterations, with the batch parameters of (Q) (`qxv`, `qc22`,
`qc13`, `qrows`: raised after those of the CCSDT step, before the DIIS
vectors are placed), on every rank; the floor stops the run with the words
needed.  On top of what CCSDT holds: the two sums (one array of the size of
T3); on rank 0 the three sets of amplitudes of the extrapolation if they
fit (19.2; they are not part of the floor); during an evaluation the
dressed `<mb||ef>`, `<ab||ex>` (2 o v v(v-1)/2) and `<ab||cd>` (as the bare
one), Xo (C(o,3) o v(v-1)/2: the largest), Xv for the planned occupied
indices, and the buffers of one quadruple - in sectors the row once (its
blocks on two pairs and on an index and a triple), the stacked operands of
its blocks on two pairs (2 x 4 (v + o) x the largest group of pairs) and of
the contractions, dense one row C(v,4) and the column batches.  The arrays
of a CCSDT step and those of an evaluation are not held together.
Measured, 26 + 120 in real blocks (19.8): the plan 20.4 GB with the 8 DIIS vectors and the
three sets of the extrapolation (2.2 GB) in the work array (CCSDT alone:
14.5 GB; the least: 8.3 GB), used to the word, peak resident set 16.6 GB;
(Q) of the same case: 5.5 GB.  The Python program
peaked at 36 times the memory of (Q) for its CC4.

The excited states: 19.4.  The solver keeps the vectors of the roots and,
for the macro iterations, two vectors per member of a cluster; its subspace
goes to files if the work array is short (16.4).  HBr / cc-pVDZ-DK with
`maxcor` 1 % above the floor of 26.8 MB: the same roots (5e-12), the
work array used to 100.0 %.

### 19.7 Validation

Native against Python on the same CFOUR files (`zccsdt.cfour.from_cfour`,
`ccn.cc4_kernel(eris, 1e-12, 1e-10, store_t4=False)`; reference directories
of 16.5):

| case | Python | native |
|---|---|---|
| H2O / 6-31G, X2C-1e (10 + 16), complex dense, CC_CONV=9: E_corr(CCSDT) | -0.13650476098947 | -0.136504760992 |
| E_corr(CC4) | -0.13697006920966 | -0.136970069209 (default), ..211 (`cc4acc 0`), ..210 (confirm, plain), ..212 (scratch) |
| HBr / cc-pVDZ-DK, `%dglin`, 28 frozen (8 + 28), CC_CONV=10: E_corr(CC4) | -0.14306776976830 | -0.143067769768 (default, plain, scratch), ..769 (`cc4acc 0`, confirm) |
| H2O / 6-31G, nonrelativistic RHF as spin orbitals (`h2o_631g_nr`: the integrals of the MRCC reference): E(CC4) | -76.120890191123 (docs/ccn.md 12; MRCC: 8e-13) | -76.120890191106 (CC_CONV=9, `cc4acc 0`) |
| HF / 6-31G, X2C-1e (10 + 12; the package test, without labels): E_corr(CC4) | -0.13230791419589 | -0.132307914197 |
| EOM-CC4, the three lowest roots (`cc4fly.eom`, 1e-8; native `%eomthrs 1e-9`, macro / micro) | 0.377265907539 (twice), 0.378090274216 | 0.377265907522 (twice), 0.378090274212: 1.7e-11 (`cc4acc 0`); 0.377265907520 (twice), 0.378090274211 with the extrapolation in the ground state, serial and on 2 ranks |
| H2O / 6-31G (10 + 16), EOM-CC4, the lowest triplet | its folded solver does not settle on the three nearly degenerate components (the weakness of 17.4): root 1 moves within 0.284455224 .. 232 over 30 frequencies, root 2 reached 0.284455510848 with |lambda - omega| 1e-8 and then drifts | 0.284455230604, 0.284455512076 (inside that scatter, 5e-9) |

Modes, all to the digits printed (HBr 8 + 28): real blocks of 6 classes,
real dense (`labels 0`), complex blocks (`real 0`), complex dense; 1 and 8
threads; 2, 3 and 4 ranks with and without slices; the start from `XZAMPS`
of an earlier run on 1 and 2 ranks; `maxcor` 1 % above the floor (DIIS in
the file, no room for the extrapolation: the path of `cc4acc 0`); explicit
small batches (`plan 0 0 0 0 3 7 50 11` dense: Xv for 3 occupied indices of
7, 7 and 50 columns, 11 rows; `plan 7 3 2 1 4 1 1 1` for H2O: one column
per product) - and below the floor the message with the words needed.  H2O
likewise on 1, 2, 3 and 4 ranks.  Larger occupied spaces, blocks against
dense (before the extrapolation): HBr / cc-pVDZ-DK with 10 frozen spinors
(26 + 28, the 3d shell occupied): E(CC4) -2605.010443509274 in both; 18 +
28: -2605.002915235616 (E(CCSDT(Q)) -2605.002920087064).

**The restructured evaluation of 2026-10-13** (19.1) against the one before
it, same amplitudes: E(CC4) and all roots of the table of 19.4 to the 12
digits printed, in real and complex blocks, serial and over ranks; the
first cycle of 26 + 120 and of 8 + 78 (energy, the three norms) to all
digits printed.

EOM-CC4, internal checks (HBr 8 + 28, `%eomthrs 1e-8`, `cc4acc 0`; Eh):

| shift, root | real blocks, macro / micro | `cc4eom davidson` | complex blocks | dense (one problem, 14 roots) |
|---|---|---|---|---|
| 0, 1 | 0.243564745147 | 0.243564745956 | 0.243564745147 | 0.243564745169 |
| 0, 2 | 0.244017034892 | 0.244017035286 | 0.244017034892 | 0.244017034892 |
| 2, 1 | 0.237070153885 | 0.237070154654 | 0.237070153885 | 0.237070153885 (twice: shifts +-2) |
| 2, 2 | 0.260830784608 | 0.260830785105 | 0.260830784608 | 0.260830784610 (twice) |
| 4, 1 | 0.233064310317 | 0.233064310829 | 0.233064310317 | 0.233064310317 (twice) |
| 4, 2 | 0.544532239662 | 0.544532239701 | 0.544532239662 | 0.544532239652, ..672 |
| 6, 1 | 0.737850625992 | 0.737850625900 | 0.737850625992 | (above the 14 lowest) |

The same digits on one thread with small batches (`plan 5 2 1 1 3 0 0 0`).
The two solvers agree to 8e-10 (their threshold 1e-8), blocks and dense to
2e-11.  H2O (complex dense, four roots, `%eomthrs 1e-9`): macro / micro
0.284455230604, 0.284455512076, 0.284456180025, 0.311283423249; `cc4eom
davidson` 0.284455230693, ..512127, ..6180139, 0.311283423118.  With the
extrapolation in the ground state (the default) the roots of HBr are these
digits but the last of shift 6 (..991), those of H2O move by 2e-12.

`regress.sh` (serial with `--eom`, and 2 ranks with `--eom`) against the
binary of master (29ff500): REGRESSION OK with the final sources - the
numbers of the other methods are unchanged.  `cfour/merge/test/run.sh`:
E(CC4) of HF / 6-31G -100.196781330906 (5 evaluations; `cc4acc 0`:
..907, 6 evaluations; the reference of the test is ..907 to 1e-8); PASS.

### 19.8 Timings

HBr / cc-pVQZ-DK, X2C-1e, `%dglin`, `DROPMO=1>5`, `%vircut 4.124587`: 26 +
120, real, 10 classes (the case of 16.6 and 17.6), CC_CONV=7, MEMORY_SIZE
60 GB; 2 x Xeon Gold 6248R (48 cores, 189 GB).

**One T4 evaluation, stage by stage** (node c432, holding job 31915920; 24
threads on the cores and the memory of socket 0; the first cycle from the
CCSDT amplitudes, `ncyc 1`; seconds, measured once each):

| HBr / cc-pVQZ-DK, 26 + 120 (14 950 occupied quadruples) | master 29ff500 | Xv from the kept `<ab||cd>` (bc22bc9) | the row with half the passes (22cc8d5) |
|---|---|---|---|
| operands | 2.05 | 2.05 | 2.12 |
| Xv | 34.49 | 2.19 | 2.20 |
| right-hand side, blocks on two pairs | 38.62 | 38.83 | 25.18 |
| right-hand side, blocks on an index and a triple | 88.14 | 88.09 | 37.47 |
| the row over the denominators | 104.38 | 104.71 | 64.30 |
| contractions | 77.31 | 78.55 | 44.87 |
| **an evaluation** | **357.85** | 327.33 | **176.23** |

| HBr / cc-pVTZ-DK, 8 + 78 (70 occupied quadruples) | master | bc22bc9 | 22cc8d5 |
|---|---|---|---|
| operands | 0.04 | 0.04 | 0.04 |
| Xv | 0.45 | 0.06 | 0.06 |
| right-hand side, blocks on two pairs | 0.05 | 0.05 | 0.03 |
| right-hand side, blocks on an index and a triple | 0.07 | 0.07 | 0.04 |
| the row over the denominators | 0.09 | 0.09 | 0.08 |
| contractions | 0.08 | 0.08 | 0.05 |
| **an evaluation** | **0.79** | 0.40 | **0.28** |

(Xv was 10 % of an evaluation for 26 + 120 and 57 % for 8 + 78; it is 1 %
and 21 %.)  Inside the stages of 26 + 120 before the second change
(temporary clocks): of the blocks on an index and a triple 59 s in the
product with rows of T3 - a loop over columns, not vectorized -; of the
contractions 45 s in five readings of every block on two pairs; 26 s in
setting the blocks to zero twice per quadruple; the stacking of small
operands on one thread.  After it: the two long GEMMs 18 and 20 s, the pass
of `zc_cc4_rowsmm` 37 s, the row 64 s, `zc_cc4_addmm` 15 s, the reduction
(c | ab) 4 s.  Tried and not kept: the virtual quadruples of the row in
tiles of 8^4 (82 s against 78 s for the plain order: the row is not bound
by its cache lines but by what it reads and writes once); one range of the
first index per thread (117 s: the ranges cannot be balanced).

**CC4 to convergence and the excited states, before and after** (one job,
one node, built there from `git archive`; node c432, holding job 31915920, nothing else on the node;
`cc4f_time.sbatch` of the work directory; master 29ff500 against 398d0c1;
24 threads on the cores and the memory of socket 0, or 2 ranks x 24 threads
with `I_MPI_PIN_DOMAIN=omp`; measured once):

| | master 29ff500 | 398d0c1 | ratio |
|---|---|---|---|
| CCSDT iteration; (Q), once (24 threads) | 4.16 s (17 cycles); 289.0 s | | |
| **24 threads: one T4 evaluation** | 358.2 s = 1.24 (Q) = 86 CCSDT iterations | **176.2 s** = 0.61 (Q) = 42 iterations | 2.03 |
| CC4 from the CCSDT amplitudes, to CC_CONV=7 | 2262.8 s: 6 evaluations (2149.6 s), 27 cycles | **995.2 s: 5 evaluations (882.2 s), 25 cycles** | 2.27 |
| **2 ranks x 24: one T4 evaluation** | 183.8 s | **91.6 s** (1.92 of 24 threads) | 2.01 |
| CC4 from the CCSDT amplitudes | 1197.3 s: 6 evaluations (1103.0 s), 27 cycles | **554.1 s**: 5 evaluations (457.5 s), 25 cycles | 2.16 |
| E(CC4) / Eh | -2605.340463043037 | -2605.340463043362 (both) | |
| work array, 24 threads: plan = used; peak resident set | 18150.9 MB; 14.3 GB | 20360.8 MB (2198.7 MB: the extrapolation); 16.6 GB | |
| 2 ranks: plan of rank 0 / of the other; resident (rank 0) | 13565.8 / 13565.8 MB; 12.7 GB | 15775.7 / 13577.0 MB; 15.0 GB | |
| **EOM-CC4, one root** of shift 2, `%eomthrs 1e-5`, from `XZCC4` (3 macro iterations, 3 + 10 sigma vectors), 24 threads: a sigma vector with the T4 terms; the root | one rank only; job 31913400 on c332, waiting threads asleep: 1073 s; 3300 s | 410.9 s = 2.33 evaluations; **1304.7 s** | |
| 2 ranks x 24 | stops | 212.2 s; **692.9 s** (1.88 of 24 threads) | |
| EOM, 24 threads: the least; used; resident | 17.0 GB; 40.6 GB; 30.1 GB | 16995.3 MB; 40594.7 MB (the subspace takes what is free); 29.0 GB | |
| EOM, 2 ranks: plan of rank 0 (its rows of the subspace besides) / of the other (with its rows); used and resident, rank 0 | | 16763.4 / 29770.7 MB; 25886.3 MB, 21.8 GB (the other rank, in the run of the same plan on c054: 24320.4 MB, 21.3 GB) | |

omega of the root: 0.235170618224 Eh in all runs of this section; in
`XZEOM`, within a job (each has its own SCF): 0.23517061822375 on 24 threads
and ..371 on 2 ranks and on four nodes (job 31916937), ..382 on 4 and on 8
ranks (job 31916947).

**The waiting of the threads.**  The jobs of 2026-10-12 set
`OMP_WAIT_POLICY=PASSIVE`; the numbers above are with the default of the
OpenMP library (a thread spins for 200 ms before it sleeps).  An evaluation
is tens of short parallel regions per occupied quadruple, and threads
that sleep between them cost: master 419.7 s for an evaluation on c432 with
PASSIVE (16 of 24 cores busy; 380 s on c134 in the job of 2026-10-12)
against 358.2 s, 398d0c1 213 s against 185 s (c054).  The README of the
package asks for `OMP_NUM_THREADS` alone.

**Over nodes** (two jobs of four nodes each, 2 x Xeon Gold 6248R, run
directories in the node-local `/tmp` of every node, Intel MPI 2021 with
`I_MPI_PIN_DOMAIN=omp`; `cc4f_nodes.sbatch`, `cc4f_nodes2.sbatch`; commit
398d0c1; CC4 from the CCSDT amplitudes and the root as above in one run):

| | job 31916937 (c054, c058, c069, c577) | job 31916947 (c488, c507, c520, c526) |
|---|---|---|
| serial, 24 threads: a CCSDT iteration | 4.10 s | 4.68 s |
| serial, 24 threads: a T4 evaluation; CC4; a sigma vector with the T4 terms; the root | 185.0 s; 1033.9 s; 442.1 s; 1399.0 s | |
| 1 node x 2 ranks x 24 | 107.7 s; 626.8 s; 258.4 s; 829.4 s (1.72, 1.65, 1.71, 1.69) | |
| 4 nodes x 1 rank x 48 | 72.8 s (70.3 to 76.7); 432.9 s; 162 to 168 s; 532.8 s | 68.7 s (65.3 to 72.0); 416.6 s; 158 to 182 s; 535.2 s |
| **4 nodes x 2 ranks x 24** | | **31.3 s; 225.1 s; 71.5 s; 247.8 s** |

(CC4: 5 evaluations and 25 cycles in every run.)  Eight ranks of 24 threads
on four nodes make an evaluation in 31.3 s - 5.6 times the 24 threads of
c432, 5.9 times those of c054 - and the root in 248 s (5.3, 5.6).  One rank
of 48 threads per node is half as fast as two of 24 (12 to 17 ms per
occupied quadruple on 48 threads; 24 threads on one socket of c432 take
11.8) - presumably because the work array of a rank lies on the socket of
the thread that touched it first; not examined.  **One rank per socket.**  Memory per rank of the run on 8 ranks (EOM included): rank 0
plans 16291.2 MB and used 15547.6, the others plan 18436.9 and used 14805;
resident 15.2 to 16.0 GB each, 37 to 39 GB of work arrays per node.

A first job of four nodes (31916803) did not get through its first
evaluation: the rank of one node (c060) was still at it after 19 minutes
while the three others had done their quarters within a minute and waited.
The node was excluded from the two jobs above; what was wrong with it was
not found (no other job on it, cores and memory as the others).

### 19.9 An estimate for ThF+ (28 + 330, real, 12 classes)

**An estimate, not a measurement** - no CC4 job was run for ThF+.
Measured there (17.7; job 31861231, 24 cores): a CCSDT iteration 258 s,
(Q) 5.4 h.  Taken from 19.8: one T4 evaluation = 0.61 (Q) (it was
1.24 to 1.30 before the work of 2026-10-13) - if the ratio of 26 + 120
holds for classes of 27 virtual spinors; 4 to 5 evaluations from converged
CCSDT amplitudes at CC_CONV=7 with the extrapolation (19.2), about 25
CCSDT-like cycles in all; a sigma vector with the T4 terms = 2.3 to 2.4
evaluations; the ratios of 19.8 for the ranks.

| | one node, 24 cores | one node, 2 ranks x 24 | 4 nodes x 2 ranks x 24 |
|---|---|---|---|
| one T4 evaluation | 0.61 x 5.4 h = 3.3 h (estimate of 2026-10-12: 7.0 h) | 1.7 to 1.9 h | 0.6 h, if the ranks scale as for 26 + 120 (5.6 to 5.9 on eight ranks) |
| CC4 from the CCSDT amplitudes, converged (4 to 5 evaluations, 25 cycles of 258 s) | 15 to 18 h (was 37 h) | 9 to 11 h | 3 to 4 h (the cycles between at the rates of 15.7) |
| the energy to about 1e-7 Eh (3 evaluations) | 11 h | 6 h | 2.5 h |
| one evaluation (93 to 99.9 % of the effect of T4 in the small cases, two thirds in the worst) | 3.7 h | 2 h | 0.8 h |
| excited state: one sigma vector with the T4 terms | 7.6 to 7.9 h (was 17 to 20 h) | 4 h | 1.4 h |
| excited state, one evaluation and the relaxation (error 5e-6 to 2.5e-5 Eh in the small cases) | about 8.5 h per state, after CC4 and the roots without the T4 terms | 4.5 h | 1.6 h |
| excited state, converged to 1e-5 Eh (3 evaluations; 1e-7: 4) | 24 to 26 h per state (was 55 to 80 h) | 13 h | 4.5 h |

Every evaluation is followed by a checkpoint (19.5), so a job limit inside
this costs at most the evaluation in hand: 3.3 h, or one sigma vector.

Memory: Xo alone is C(28,3) x 28 x C(330,2) = 5.0e9 numbers = 40 GB per rank
(dense in every mode), T3 and its two companions 3 x 13 GB, the three sets
of the extrapolation 3 x 13 GB more on rank 0 (left out by the plan if they
do not fit: one evaluation more); the ground state fits a 192 GB node per
rank without the DIIS vectors in memory (they go to the file) and, on rank
0, without the extrapolation or with it and little else.  The excited
states: 107 GB of the EOM-CCSDT sigma vector (16.7), two Xo (80 GB), the
dressed `<ab||cd>` and `<bc||ef>'` (2 x 2 GB), Xv of two kinds for 3 to 27
occupied indices (2 x 1.3 GB each), the row and its stacks (about 5 GB):
200 to 270 GB per rank, on rank 0 the vector of every root (15.4 GB)
besides - nodes of the bigmem kind, as the EOM-CCSDT job.  Disk: `XZCC4`
is 2 x T3 and the history of the extrapolation, 52 GB at most; `XZE4Q` 31
GB per root of a cluster, `XZE4R` and `XZEOMR` 15.4 GB per root.

### 19.10 Options

Lines of `%xzccsdt` (section 18; an ordinary run needs none):

| option | default | what |
|---|---|---|
| `cc4 confirm` | macro | macro / micro, converged only when new T4 terms leave the first step below the thresholds (one evaluation more) |
| `cc4 plain` | | the T4 terms in every iteration, from the CCSDT amplitudes |
| `cc4 scratch` | | the same from the second-order amplitudes, no CCSDT first (`XZAMPS` and `XZCC4` not read) |
| `cc4micro x` | 0.01 | the equations with the T4 terms held are iterated until the change is x times that of the first step |
| `cc4acc 0` | 1 | no extrapolation over the macro iterations (19.2) |
| `cc4eom davidson` | macro | EOM-CC4: every sigma vector with the T4 terms (the solver of 17.4); no checkpoints |
| `omegatol x`, `foldwin x` | ESTATE_CONV / 100, 5e-4 | EOM-CC4 as EOM-CC3 (17.4): the change of omega at convergence; roots closer than this are solved together |
| `plan`, `diis`, `maxcor` | | the plan by hand: its numbers 5 to 8 are the batches of the T4 evaluation (section 4) |
| `cc4stop n` | | a test hook: the run stops after its n-th T4 evaluation (19.5) |

### 19.11 Not done

- **A left-hand vector** and with it the perturbative estimate of the T4
  effect on an excitation energy (the analogue of (Q)); the Rayleigh
  quotient with the right-hand vector overshoots (19.3).  An estimate of
  the work: the left eigenproblem of EOM-CCSDT is not in this program
  either (16.9) - the transposed sigma vector is the Lambda machinery of
  section 11 with the label shifts of 16.2 and the terms of `xzlam.inc`
  without the inhomogeneity, about the size of `xzeom.inc` (1500 lines)
  with its validation against the Python program, which has the left
  vectors of EOM-CCSDT only without labels; the correction itself is one
  pass of `xz4run` with a left contraction (the products of (Q)_Lambda,
  `xzq.inc`, in the place of the contractions of 19.1) at the cost of one
  sigma vector with the T4 terms.  Two to three weeks at the pace of this
  work, most of it the left sigma vector.
- An extrapolation over the macro iterations of a root (its changes fall by
  a constant factor as those of the ground state do).
- The ranks of a node sharing Xo, the dressed `<ab||cd>` and T3 in the
  excited states (19.4); Xo is formed by every rank (as in (Q), section 5).
- A restart inside a T4 evaluation (checkpoints of its tasks, as (Q) has
  them): an interrupted evaluation is done again from its start.
- The three sets of amplitudes of the extrapolation in a file when the work
  array has no room for them.
- The row of T4 without labels is not restructured (it is the dense code of
  (Q) with the divide and the contractions in batches of columns).
- Density, Lambda, IP / EA for CC4; the label-shifted Python reference
  (cc4fly has shift 0 only).

## 20. EOM-DEA-CCSDT: two electrons attached

    CALC=CCSDT  and the section %xzdea      CCSDT, then the states with two electrons more
    mpirun -np 4 -ppn 2 xzccsdt_mpi         the same over ranks (20.6)

The double electron attachment of docs/eom.md, section 11, on the sector
itself (11.9): R = 2p + 3p1h + 4p2h - `r2[(a<b)]`, `r3[i, (a<b<c)]`,
`r4[(i<j), (a<b<c<d)]` - on the ground state of CCSDT, the eigenvalues
E(N+2) - E_CCSDT(N), in the blocks of the symmetry labels: a problem is a
target 2 Omega, the sum of the labels of the two attached electrons.  The
application is the zero-field splitting of O2 X 3Sigma_g- from O2(2+)
(docs/eom.md 11.6).  The Python program (`src/zccsdt/cc/dea_sector.py`,
`Engine.sigma`) is the reference of the sigma vector and of the roots; its
factorization, its layouts and its passes are taken over one to one (the
term list: the module docstring of `dea_sector`, docs/eom.md 11.9).
Sources: `xzdead.F90` (input, tables, slices, the results),
`xzdea.inc` (the intermediates of the ground state), `xzdeas.inc` (the
sigma vector, the preconditioner, the masks of the parity), `xzdeam.inc`
(the driver); the solver is that of section 16 (`xzeoms.inc`, generalized
to the vectors of the sector).  No kernel was written: the expansions and
reductions of `csrc/t4sym.c` (`zc_t4s_x22`, `zc_t4s_x13`, `zc_t4s_x12`,
`zc_t4s_quads`, added to `KERNELS` of the Makefile), `zc_tn_rows`
(tnpack.c), `zc_pair_one_body_add`, `zc_vprecond` (t3pack.c), `zc_gemm`.

R to complete 5p3h (`level 5p3h`): section 21.  Not here: the folded 5p3h
row of docs/eom.md 11.3, the T of CCSDTQ or CC4 (the two T4 terms of
11.9), DIP, left-hand vectors, the file of the root vectors.

### 20.1 Input and output

    *CFOUR(CALC=CCSDT, ..., CC_CONV=10, ESTATE_CONV=9, ESTATE_MAXCYC=120, MEMORY_SIZE=100, MEM_UNIT=GB)
    ...
    %xzdea
    0 1
    2 1
    parity g

| what | ZMAT | default |
|---|---|---|
| the targets | `%xzdea` (ours: CFOUR has no keyword for DEA - `EXCITE` stops at `EOMDIP`): lines `target [roots]`, 2 Omega in the labels of the run, until an empty line or a line that starts with `%` | - (the section is what asks for EOM-DEA) |
| the parity of the states | `parity g` / `parity u` in `%xzdea`; needs the file `xzparity` | all |
| the start | `guess n` in `%xzdea`: unit vectors on the n lowest pairs (a<b) of the target (the masks) | max(4, roots + 2) |
| threshold, iterations, roots, subspace | `ESTATE_CONV` (or `%eomthrs`), `ESTATE_MAXCYC`, `%eomnroot`, `%eomndav0` - as for EOM-EE (16.1) | 10^-5, 40, 1, max(22 + roots, 4 roots + 8) |

`%xzdea` goes with `CALC=CCSDT` without `EXCITE` and `PROPS`; the
amplitudes are those of the run or of `XZAMPS` (section 1).  The file
`xzparity` (ours, written by `bench/dea/native_rundir.py`): the format of
`xzlabels` without the modulus - a line with the number of spinors of the
SCF, then one parity (0 g, 1 u) per spinor in the order of `scf2ceig`.  The
mask keeps the elements whose spinors have the total parity of the target
(the occupied ones counting negatively), as `dea_sector.ProblemSector`.

Output per target: the sizes of the parts, the batches and the place of the
subspace, CFOUR's line per iteration (`EOM cycle`), the time per sigma
vector and its pieces, per root omega (Eh, eV, cm-1), the largest element
and the norm of the residual, the weights of 2p, 3p1h, 4p2h, the total
energy and the largest elements of r2 (the two spinors in the numbering of
the SCF, their labels and energies); at the end one table of all states
with the energies above the lowest.  Files: `XZDEA` (text: a line per
state - 2 Omega, root, omega, total energy, residual, weights, converged,
the leading pair and its weight - and two comment lines with E(SCF),
E(CCSDT)); JOBARC `TOTENER2`: the total energy of the lowest converged
state; `TOTENERG` stays E(CCSDT).

**Hamiltonians from outside CFOUR's chain.**  The X2Ccorr Hamiltonians of
O2(2+) (the files `o2_hf2plus_*_x2ccorr.npz` of bench/dea/common.py) are
not made by CFOUR.  `bench/dea/native_rundir.py <file.npz> <dir> [--freeze n]
[--targets 0,2] [--roots 1] [--cc-conv 10] [--estate-conv 9] [--mem-gb 100]
[--xz 'option'] [--phases seed]` makes a run directory: CFOUR's `xjoda` on a
stand-in molecule (HF / 6-31G with the GENBAS of cfour/merge/test) for JOBARC
and joda's flags, then the integral files, `scf2ceig`, `nmototal`,
`xzlabels` (2 Omega) and `xzparity` of the spinors of the Hamiltonian file in
the order of their labels (`zccsdt.cfour.write`; the frozen spinors folded
into the one-electron part by `common.load`), E(SCF) as `SCFENEG` and
`NUMDROPA` 0 in JOBARC; the ZMAT has `%xzdea` with the targets and
`parity g`.  Nothing of the stand-in molecule is read by `xzccsdt`.  The
files have the spinor energies only (canonical spinors): f_ov of the SCF of
the Hamiltonian file (1e-10 for O2) is left out, as by every CFOUR chain.
`--phases seed` multiplies every spinor by a random phase: complex integrals
with the same labels (a test of the complex arithmetic).

**From a block file, without an n^4 array** (the X2Ccorr Hamiltonian in
label blocks of docs/eom.md 11.10, `o2_hf2plus_*_x2ccorr.h5`; 2026-10-20).
`native_rundir.py <file.h5> <dir> ...` (the same options) writes the blocks
with an occupied index from the label blocks of the file
(`common.load_blocks(names=...)` without `vvvv`: `<ov||vv>` is the largest,
o v^3) and `<ab||cd>` not as `moinpppp` - a record of (v(v+1)/2)^2 complex
numbers, 60 GB for 350 virtual spinors, which the reader holds whole while it
sorts it (section 4) - but as the file **`xzvvvv`**: one dense block per sum
of the labels of a pair (2 Omega_a + 2 Omega_b), made group by group from the
blocks (g_a g_c | g_b g_d) of the file (`write_vvvv_blocks`: (ac|bd) on the
ordered pairs of the group, then (ac|bd) - (ad|bc) on the pairs a<b; the
largest group of cc-pV5Z-X2C-TP, label sum 0, has 8173 pairs: 2.1 GB of
ordered pairs while it is made).  The format (stream, 8 bytes per number):

    'XZVVVV01', nv, nwf (1 real, 2 complex), ngrp
    cnt(ngrp)                 the pairs of each group (their sum: nv (nv - 1) / 2)
    pair(nvp)                 the pairs, group after group: a + b (b - 1) / 2 for a < b,
                              CFOUR's numbers of the virtual spinors from 0
    the blocks, group after group: [i][j] = <pair_i || pair_j>, row-major

The program reads it when it is there (`xzvvhdr`, `xzrdvvb` of `xzread.F90`;
the output says `<vv||vv>: the file xzvvvv`): block by block of the file into
the diagonal blocks of its own groups of pairs, every element to its place
with the signs of its two pairs (the program may order the spinors
differently: `vperm`); the transient is one block of the file, not the record
of `moinpppp`.  The groups of the file need not be those of the program
(`labels 0`: one group of all pairs, the blocks of the file are parts of it);
an element of the file between two groups of the program is checked against
`labtol` as the elements of `moinpppp` are, a complex file against `imagtol`.
`--moinpppp` writes a block file the old way (the full container and
`moinpppp`: the check of the two routes on the same integrals, 20.5).  For
cc-pV5Z-X2C-TP (360 spinors, 10 + 350) the run directory takes 67 s and 11 GB
(`xzvvvv` 2.8 GB, 23 groups; the ten files `PPPH` 3.4 GB).
`bench/dea/native_blocks.sbatch SNAP H5 RUNDIR TAG [MAXCYC] [MEMWORDS]`
makes the run directory and runs it with one rank of 48 threads per node
(`MEMORY_SIZE` at most the least `MemAvailable` of the nodes less 6 GB:
GETMEM of a rank of 155 GB failed on a node that had 141 GB, job 32272419);
`bench/dea/jobarc_flags.py <dir> 97=3` sets `ESTATE_MAXCYC` (or the memory,
36) in JOBARC without xjoda - a timing run of three iterations per target,
whose CCSDT amplitudes the next run takes.

### 20.2 The sigma vector

The vector, one region `[r2 | r3 | r4]` of the shift s (the class of a pair
of two attached electrons, (target - 2 l_min) / step): r2 the pairs a<b of
group s; r3 one block per group K of the virtual triples, its rows the
occupied i of class K - s; r4 one block per class M of the virtual
quadruples, its rows the occupied pairs of group M - s in their order, its
columns the quadruples of class M (lexicographic, `zc_t4s_quads`) - the
layout of `dea_sector.Sector`, so that a vector of the Python program is
read element for element (the test hook `deasig`, 20.5).  Since the pairs
are numbered by groups, the rows of r4 in the order of the blocks are the
occupied pairs in their order: a range of pairs is a range of r4 (20.6).

**The ground state, once** (`N(xzdim)`, after `N(xzeomim)` of EOM-EE): the
operators of the T3 equation of the native program are those of the T4
equation that the sector needs - W_vvvo is Wv (with its T2 and T3 parts),
W_ovvo is Wr, F_vv, F_oo, W_oooo is Whh on pairs, and the pp operator of
the T3 equation in the place of `<ab||cd>` is Wpp + F_vv / 2 on pairs (Wq
of the 3p1h row) -, except Wo: the W_vooo of the native step leaves its f_ov
term to the T3 equation, so `Wo = - W_vooo - sum_e f[m][e] t2[P][a][e]`.
Made here: Wv as [i][e][Q], [Wo | t2] by pairs (`la13` of the Python
program), Wr as [(i a)][(m e)], Wh = Whh - F_oo on pairs, Wq of the 4p2h row
(Wpp + F_vv / 3) and Wpp as [(bc) e][f] (from the pp operator and F_vv,
`zc_pair_one_body_add`), `<dm||ef>` in blocks, T3 on (e | cd) and by an
index and a pair, and R13: per occupied pair (k, l) and class mu the rows
`t(m kl; K)` of the m of class mu above the six-index `Xv(e; kl; K)` of the
T4 equation without T4 (`N(xzdxv)`: the term `- P(kl) <mn||el> t(mnk, bcd)`
directly, the other three as products per (class of e, class of d) on all
threads, reduced over (d | bc) by `zc_t4s_x12`).  R13 is the largest array
(1.0 GB for 116 spinors, 5.2 GB for 168, about 12 GB for 216).

**A sigma vector** (`N(xzdsig)`, the order of `Engine.sigma`): the small
intermediates (r2 on all pairs, r3 on (a | bc) by `zc_t4s_x12`, h, WhR, YR,
H, WoR, Q3, XoR, ZR, U and XvR without their 4p2h terms); the pp ladder per
block in batches of rows (`zc_t4s_x22` expand, one GEMM per group of (ef)
with Wq, `zc_t4s_x22` reduce) with its thin products per row (the 4p2h terms
of sigma2, of sigma3 and of XoR); the ring per spectator j and class
difference d (`zc_t4s_x13` expand of the rows (m, j), one GEMM with the block
of the stacked operator [Wr rows (i a) | f_ov | -1/2 <mj||ek> | 1/2 <mj||ef>]
gathered per block, `zc_t4s_x13` reduce into the rows (i, j), the thin rows
to sigma3 and XvR); the hh ladder per block (one GEMM with the pairs of the
block); the rows of sigma4 one occupied pair at a time (per group of (ab)
one GEMM of the operands stacked by `zc_tn_rows`, per class of a one GEMM
of [Wo | t2 | WoR | r2 | -YR] with [r3; XvR; T3 rows; Xv; T3P], each reduced
once); the 3p1h row; the 2p row; the masks of the parity.  Real arithmetic
where the integrals are real (as the rest of the program); the same source
for complex numbers.

### 20.3 The solver

`N(xzesolve)` of section 16, unchanged in its algorithm (CFOUR's design:
one orthonormal basis for all roots of a target, a correction r / (omega -
D) per unconverged root and iteration, the test on the largest element of
the residual, compaction to roots + 2 Ritz vectors), with hooks for the
vector of the sector (`edea` of `xzeomd`: the part that rank 0 holds -
r2, r3 -, the first part - r2, for the weights and the leading elements -,
the rest - r4): the sigma vector `N(xzdsig)`, the preconditioner
`N(xzdprec)` (the differences of the spinor energies per block,
`zc_vprecond`, floor 1e-3 as the Python solver), the start from unit
vectors on the lowest pairs of the target in the masks.  The sigma vector
applies the masks; the corrections stay in them.  The subspace: in the work
array, smaller in the work array, or in the files `EOMCnnnn`, `EOMHnnnn`
(16.4).

### 20.4 Memory and disk

In the work array, after the integrals and the amplitudes: the kept arrays
of `N(xzeomim)` and those of 20.2 (with R13), then per target the 2p
operator, two vectors, the subspace and the arrays of a sigma vector (the
small intermediates, then pass by pass the buffers of the pp ladder, of the
ring and of the rows, each released before the next).  **The batches** of
the passes over r4 (an operand of the pp ladder, an operand and a product of
a batch of the ring) are as large as the work array leaves them next to
the buffer of the subspace in files, at most 1 GB: the largest that fits is
found by bisection on the dry run of a sigma vector (monotone in the batch).
The plan of the program (section 4) runs EOM-DEA dry with the smallest
batches and the smallest subspace (the floor); the subspace then takes
what is left, as in 16.4.  **R13** is in the work array if EOM-DEA fits there
with it (a dry run of all of it at the floor); otherwise in the file
`XZDXV` (`XZDXV.<rank>`) of the run directory, written pair by pair when it
is made and read pair by pair in the rows of every sigma vector (the free
disk is checked; the file is deleted at the end); option `deaxv m` / `deaxv
f` forces the one or the other.  Over ranks every rank holds R13 of its
pairs only (20.6).

| case | the work array | used (MB) | peak resident (MB) | subspace |
|---|---|---|---|---|
| 116 spinors, MEMORY_SIZE 100 GB | 100 GB | 13 884 | 14 133 | 23 vectors in the work array (9.9 GB) |
| the same, `maxcor 500000000` (4 GB) | 4 GB | 3 677 | 3 868 | in files (10.4 GB); the same roots (4e-11) |
| the same, `maxcor 400000000` (3.2 GB; three iterations) | 3.2 GB | 3 190 | 3 197 | in files; the batches 323 and 340 MB (bisected), the iterations those of the run above |
| 168 spinors, MEMORY_SIZE 100 GB | 100 GB | 68 807 | 68 078 | 23 vectors in the work array (55 GB) |
| the same, `maxcor 2000000000` (16 GB; job 32265685) | 16 GB | 13 431 | 13 800 | in files (52.6 GB); the roots identical to 14 decimals (52 s per sigma vector with the solver: the files) |
| DYVQZ-SOG (216 spinors), MEMORY_SIZE 100 GB | 100 GB | 99 755 | 98 478 | 13 vectors in the work array (73 GB) |

The resident set follows the words of the work array that are used (MKL
and the OpenMP runtime add 0.2 - 0.4 GB).

**The plan of a rank for n ranks: option `deaplan n1 n2 ...`** (one rank, in
the run directory of the case; the program reads the integrals, runs the
memory plan of section 4, then the dry runs of EOM-DEA for every rank of
every n - with R13 of the pairs of that rank and its rows of r4 - prints a
line per rank and stops; nothing is computed, 24 s for cc-pV5Z-X2C-TP).
The columns: the arrays of the ground state that stay (with R13 of the
rank), the transient of making them, the parts of the two vectors that the
rank holds (r2, r3 and its rows of r4, 20.6) with the 2p operator - two
whole vectors until 2026-10-20 -, a sigma vector with batches of 1 GB and
of one element, the part of a vector that the rank holds, and the totals
with the subspace (the default 23 vectors) in the work array or in files,
also with R13 in `XZDXV`.  **cc-pV5Z-X2C-TP** (10 + 350, 12 classes, real;
a vector 2.55e9 elements, 20.4 GB; R13 84.5 GB in all; `pv5z_run`,
2026-10-20), GB of the rank that takes the most, with two whole vectors on
every rank (the program before the vectors were distributed):

| ranks | R13 of a rank | stays, with it | part of a vector | subspace in files (the floor) | vectors of the subspace that fit in 120 / 170 GB |
|---|---|---|---|---|---|
| 1 | 84.5 | 112.1 | 20.4 | 176.3 (R13 in the file: 93.8) | 0 / 1 (R13 in the file) |
| 2 | 43.2 | 70.8 | 10.6 | 135.0 | 0 / 1 |
| 4 | 22.0 | 49.6 | 5.6 | 113.8 | 0 / 5 |
| 6 | 15.5 | 43.1 | 3.7 | 107.3 | 1 / 8 |
| 8 | 11.6 | 39.3 | 2.8 | 103.5 | 2 / 11 |
| 12 | 7.8 | 35.4 | 1.9 | 99.6 | 5 / 18 |
| 16 | 5.8 | 33.4 | 1.4 | 97.6 | 8 / 23 |

What every rank holds whatever their number: the integrals and the
amplitudes 6.4 GB, the arrays of the ground state without R13 27.6 GB (the
table of the quadruples of the classes 4.9, W_pp as [(bc) e][f] 5.7, the pp
operator of the 4p2h row 2.8, T3 on (e | cd) and by an index and a pair
4.7, the operands of EOM-EE 8.6), their transient 11.9 GB (the products of
Xv of one pair, 11.3), **the two whole vectors 41.3 GB** and a sigma vector
16.3 GB (U 5.7, XvR 1.9, the buffers of a batch of the passes; one element
per batch saves nothing: the expansion of one row is 2.8 GB) - 91.6 GB
before R13 and the subspace.  The solver keeps a smaller subspace in the
work array only if at least 3 roots + 4 = 7 vectors fit (16.4), else it
goes to files (2 x 2.8 GB per vector on the disk of every node at 8 ranks,
read about seven times per new vector).  So a rank of cc-pV5Z-X2C-TP fits a
whole node (48 cores, `--mem=180G`, `MemAvailable` 167 - 172 GB at the
start) from 6 ranks on, and 120 GB only from 16 ranks on.
Measured, 8 ranks x 48 threads, `MEMORY_SIZE` 170 GB (job 32272311): 11
vectors in the work array (the plan above), the largest rank used 168 641
of 169 999 MB, the others 150 901 - 168 204 MB; resident 133 - 150 GB after
three iterations per target (the subspace not yet filled).  The production
run (12 ranks, `MEMORY_SIZE` 154.9 GB - the least `MemAvailable` of its
nodes less 6 GB -, job 32272434): 14 and 15 vectors in the work array; rank
0 used 153 297 MB (resident 151 150 MB), the largest other rank 153 691 of
154 871 MB (resident 151 518 MB), the smallest 135 353 MB.

**With the vectors distributed** (20.6; 2026-10-20) a rank holds its part
of the two vectors instead of the two whole vectors, and a sigma vector
takes the two buffers of the rows of a spectator of the ring more (2 x 5.4
GB at cc-pwCVQZ-X2C-TP).  **cc-pwCVQZ-X2C-TP, all electrons** (14 + 322 =
336 spinors, 12 classes, real; a vector 4.56e9 elements, 36.5 GB;
`deaplan`, the plan directory of `o2cv/pwcvqz/run_ae`), GB of the rank that
takes the most:

| ranks | R13 of a rank | stays, with it | the two parts | a sigma vector | part of a vector | subspace in files (the floor) | before: two whole vectors (73.5 GB), sigma 19.4 GB | vectors of the subspace that fit in 142 / 155 GB |
|---|---|---|---|---|---|---|---|---|
| 4 | 39.2 | 74.3 | 19.3 | 30.2 | 9.3 | 131.6 | - | 0 / 1 |
| 8 | 20.2 | 55.3 | 10.3 | 30.2 | 4.8 | 103.7 | - | 4 / 5 |
| 12 | 14.1 | 49.2 | 7.4 | 30.2 | 3.3 | 94.6 | - | 7 / 9 |
| 16 | 10.6 | 45.6 | 5.8 | 30.2 | 2.5 | 89.4 | - | 10 / 13 |
| 24 | 7.1 | 42.1 | 4.1 | 30.2 | 1.7 | 84.2 | 142.9 | 17 / 21 |
| 32 | 5.1 | 40.1 | 3.2 | 30.2 | 1.3 | 81.3 | 141.2 | 23 / 23 |

(The floor of the earlier program, at least 137.6 GB with R13 in `XZDXV`,
left no subspace in the work array of a parallel node, 146 - 149 GB
`MemAvailable`; with the subspace on GPFS a sigma vector took 8.2 min on
32 nodes.)  The production run (16 ranks x 48 threads, `MEMORY_SIZE` 155.7
GB, job 32291868; 20.7): 13 vectors in the work array (56.7 GB of rank 0's
part, 64.3 GB for the second target); rank 0 used 151 391 MB (resident 149
464 MB), the other ranks 139 846 - 154 278 MB of 155 725 MB (resident 138
- 152 GB).  Two ranks per node (32 ranks, 74 GB each) do not fit: the floor
of a rank is 80 - 81 GB, of it 42.7 GB that every rank holds whatever their
number (the integrals and amplitudes 7.7, the arrays of the ground state
without R13 35.1) and the sigma vector 30.2 GB.

### 20.5 Validation

**The sigma vector element by element against `dea_sector.Engine.sigma`**
(`bench/dea/native_check.py <rundir> <binary> [--target t] [--batch n]
[--complex] [--xz option]`: the integrals of the run directory read by
`zccsdt.cfour.from_cfour`, the amplitudes of the native `XZAMPS`, a random
vector written to a file; the native program with the option `deasig file`
writes sigma of it to `file.out`; the masks of the parity applied to both):

| case | largest difference (largest element) in 2p, 3p1h, 4p2h |
|---|---|
| O2 26 spinors (10 + 16), 4 classes, real; targets 0, 2, -2; batches of 1, 50 elements and 1 GB | 2.7e-15 (1.4), 4.0e-15 (6.7), 7.1e-15 (12) |
| the same, complex arithmetic (`real 0`) | 3.1e-15, 4.1e-15, 7.1e-15 |
| the same with phases on the spinors (complex integrals, largest imaginary part 0.21) | 2.3e-15, 4.8e-15, 7.2e-15 |
| O2 DYVDZ (10 + 42), 6 classes; targets 0 and 2 | 6.0e-14 (7.5), 1.2e-13 (29), 2.1e-13 (54) |
| the same, R13 in the file `XZDXV` | identical |
| the same over 3 and 4 ranks (20.6; 3 ranks with `XZDXV.<rank>`); 26 spinors with phases over 3 ranks | 1.8e-13 (52); 1.1e-14 (14) |
| 168 spinors (10 + 158, 8 classes; 1.5e8 elements of r4), targets 0 and 2 (job 32265685) | 3.2e-11 (570), 1.2e-10 (1350), 1.7e-10 (2200): 8e-14 of the largest elements |

**One class** (`labels 0`, 26 spinors: all pairs one problem): the three
lowest roots are the root of 2 Omega = 0 and the two of +-2 of the blocks,
1e-10 (the threshold).

**The roots** (ESTATE_CONV / `%eomthrs` as given; the Python roots: residual
norm 1e-9 unless said otherwise; E in Eh):

| case | native | Python | difference |
|---|---|---|---|
| 26 spinors, split | 1.5767221e-05 | 1.576720396e-05 (JSON of 11.6) | 1.8e-11 |
| DYVDZ, CC_CONV=12, `%eomthrs 1e-11`: E(2 Omega = 0), E(2), split | -150.08744260782260, -150.08742672225537, 1.588556722e-05 | the sector, the norm of dT below 1e-11, residual norm 1e-11: -150.0874426078221, -150.08742672225455, 1.588556756e-05 | 4.8e-13, 8.2e-13, 3.4e-13 |
| the same against the JSON of 11.6 (residual norm 1e-9) | | 1.588555955e-05 | 7.7e-12 (the threshold of the JSON) |
| TZ (10 + 106), CC_CONV=10, `%eomthrs 1e-10` | -150.21709561322, -150.21707919864, 1.6414585e-05 | -150.21709561323, -150.21707919864, 1.6414583399e-05 (11.6) | 1e-11, 1e-11, 1.6e-12 |
| 168 spinors (10 + 158, 1s frozen), CC_CONV=10, `%eomthrs 1e-10` | -150.22882293513, -150.22880648606, 1.6449066e-05 | -150.22882293512, -150.22880648609, 1.6449031676e-05 (11.6) | 8e-12, 2.6e-11, 3.4e-11 (the JSON: residual norm 1e-9) |
| DYVQZ-SOG (10 + 206), CC_CONV=10, `%eomthrs 1e-10`: E(CCSDT), E(2 Omega = 0), E(2), split | -148.895224614241, -150.25266284304, -150.25264609144, 1.6751600e-05 (3.6766 cm-1) | the production run of the sector (job 32236524): -148.89522461421, -150.25266284301, -150.25264609143, 1.6751580e-05 (3.6765 cm-1) | 3e-11, 2.7e-11, 7e-12, 2.0e-11 |
| DYVDZ over 1, 2, 4 ranks (20.6) | identical to 14 decimals, the same 33 + 32 sigma vectors | | |

**From the block file** (20.1; 2026-10-20; the files of docs/eom.md 11.10
with the default options, `x2cblk/{dzdef,tzd,pvqzd}`): the same block file
written both ways - `xzvvvv`, and `--moinpppp` - gives the same roots to all
digits printed; against the runs of the dense `.npz` files above the
differences are those of the two Hamiltonian files (11.10: integrals to
1.5e-11).

| case | xzvvvv: E(2 Omega = 0), E(2) | moinpppp, the same file | reference |
|---|---|---|---|
| DYVDZ, CC_CONV=12, ESTATE_CONV=11 | -150.08744260781970, -150.08742672225239 | the same digits; also with `real 0`, with `xzvvvv` written complex, and over 2 and 3 ranks | the .npz (above): -150.08744260782260, -150.08742672225537 |
| the same, `labels 0` (one problem of all pairs; its lowest root found is the pair of 2 Omega = +-2) | -150.08742672225264 | -150.08742672225264 | |
| TZ (10 + 106), CC_CONV=10, ESTATE_CONV=10 | -150.21709561323652, -150.21707919864454 | the same digits | the .npz: -150.21709561322, -150.21707919864 |
| the same over 3 ranks, `maxcor 500000000` (subspace 9 vectors) | -150.21709561322419, -150.21707919865020 | | |
| cc-pVQZ-X2C-TP (10 + 206), MEMORY_SIZE 100 GB (13 vectors), serial | -150.26441254321131, -150.26439571881491 | the same digits | the Python sector on the dense file (11.6): -150.26441254319857, -150.26439571882986, split 1.682436871419668e-05 Eh |
| the same, 170 GB (23 vectors), 1 rank x 48 threads, MPI binary | -150.26441254319741, -150.26439571882486 (split 1.68243726e-05) | | 1.2e-12, 5.0e-12 (split 3.8e-12) |
| the same, 2 ranks x 48 threads on two nodes | -150.26441254319735, -150.26439571882480 | | |

(The serial run with the smaller subspace converges to the same threshold,
1e-10 on the largest element of the residual, by another path: 1.3e-11 and
1.5e-11 from the reference.)

**The vectors distributed over the ranks** (20.6; 2026-10-20).  The sigma
vector against `dea_sector` (`native_check.py`, the MPI binary through a
wrapper `mpirun -np n`): DYVDZ (the `.npz` file), targets 0 and 2, 3 and 4
ranks, batches of 1 GB and of 50 elements (many pieces of the hh ladder,
many batches of the ring): 6.6e-14, 1.2e-13, 2.1e-13 in 2p, 3p1h, 4p2h (as
before the change); 26 spinors with phases (complex), the same: 4.8e-15,
1e-14.  The roots against the binary of master (c1d8df3) on the same
nodes, the same run directories (from the block files; CC_CONV=10,
ESTATE_CONV=10; 1 rank x 48 threads, 2 ranks x 48 on two nodes, 4 ranks x
24 on two nodes; jobs 32291726, 32291727):

| case | master: E(2 Omega = 0), E(2) over 1 / 2 / 4 ranks | the branch | difference |
|---|---|---|---|
| DYVDZ | -150.08744260782476, -150.08742672224628 (all three) | the same digits, all three | 0 |
| TZ | -150.21709561323649, -150.21707919864451 (all three) | the same digits, all three | 0 |
| cc-pVQZ-X2C-TP | 1 rank -150.26441254320349, -150.26439571882486; 2 and 4 ranks -150.26441254319735, -150.26439571882480 | the same digits at every number of ranks | 0 |
| cc-pwCVTZ-X2C-TP, 16 electrons (14 + 158) | 1 rank -150.33918005110073, -150.33916313538737 (the production run, `pwcvtz_ae.json`); 2 and 4 ranks -150.33918005110070, -150.33916313538734 | the same digits at every number of ranks | 0 |

(The reference of cc-pVQZ-X2C-TP, the Python sector: -150.26441254319857,
-150.26439571882986; the runs of 1 and of 2 or 4 ranks differ by 6e-12 in
E(2 Omega = 0) in either binary: the paths of the solver.)
`cfour/merge/test/run.sh`: PASS with the serial and with the MPI binary.

### 20.6 Over MPI ranks

The rows of r4 are dealt to the ranks by occupied pairs: one range of
pairs per rank, the same for all targets, of about equal sums of the
elements of their rows over the targets (`xzdranges`); in the layout of
r4 a range of pairs is a range of elements (20.2), so that every rank
holds one slice of every vector - rank 0 with r2 and r3 in front, as the
singles and doubles of EOM-EE (16.7) - and the vector operations of the
solver are the commands of 16.7 on these slices (`N(xzeserve)`; the
`op_shift` of a target sets up its sector and the batches of rank 0,
`N(xzdshift)`).

**Every rank holds only its part of a vector** (2026-10-20; until then
every rank held two whole vectors, the trial vector and its sigma vector:
2 x 36.5 GB at cc-pwCVQZ-X2C-TP whatever the number of ranks, 20.4).  The
two vectors of the solver and of a sigma vector are on every rank r2, r3
(rank 0's; a copy on the other ranks, which the sigma vector reads and to
which it adds partial sums) and the rows of r4 of the rank (`xzdpart`);
the addresses of the vectors are those that whole vectors would have
(`ekx = start - esoff(rank)`, as for R13 in `XZDXV`), of which the solver
and the sigma vector touch only the rows of the rank.  The subspace holds
the same parts (16.7).  The test hook `deasig` sends the rows of the vector
of its file from rank 0 to their owners and collects sigma4 from them
(`N(xzepslices)`).

A sigma vector (`N(xzdsig)` on every rank): r2 and r3 of rank 0 go to all
ranks (`xzmpi_bcast_w`); the small intermediates are sums over an occupied
index of r3, split by that index over the ranks (mod(index, ranks)): YR and
ZR are summed at once, XoR and XvR stay partial sums until the passes over
r4 have added to them; every rank makes the rows of sigma4 of its pairs in
the pp ladder and the six-index products from its own rows; the thin
products of the pp ladder are summed with XoR (on all ranks; sigma2 and the
3p1h parts on rank 0).  Two passes need rows of other ranks:

- **the hh ladder** makes the rows of a block from all rows of the block:
  if other ranks hold some of them, the block goes in pieces of columns (at
  most `dbatch` elements, the same pieces on all ranks), each piece of every
  row of the block sent to the other ranks that hold rows of it;
- **the ring** of a spectator j reads the rows (m, j) of all m and writes
  the rows (i, j) of all i.  Its work items are the batches of the serial
  code (spectator, batch of blocks) with a cost that counts the products,
  the elements moved by the expansion and the reduction, and the walks
  through the quadruples of the rows (m, j).  The spectators go to the ranks
  in rounds: the largest first to the round with the least cost that has
  fewer spectators than ranks (one round if there are at least as many
  ranks as spectators); in a round every spectator has one rank, the ranks
  left go one by one to the spectator with the largest cost per rank, and
  the batches of a spectator, the largest first, to the least loaded of its
  ranks (the same deal on every rank; the output prints the cost of the
  deal, the sum over the rounds of the largest cost of a rank against the
  mean: 1.15 and 1.20 at cc-pwCVQZ-X2C-TP over 16 ranks, 14 spectators).  In
  a round a rank takes the rows (m, j) of its spectator from their owners
  into a buffer, makes its batches into the rows (i, j) of a second buffer,
  and gives these back to their owners, who add them to their rows of
  sigma4 (two buffers of the rows of the spectator with the most: 10.8 GB
  at cc-pwCVQZ-X2C-TP); the thin rows of sigma3 and XvR are summed
  afterwards.

The exchanges are point to point (`xzmpi_segs` of `xzmpi.c`: a list of
segments of the work array to and from other ranks, all sends posted
first, then the receives - stored in place, or added through a few
buffers of one message each -, in messages of at most `mpichunk` words).
The 3p1h and 2p rows are rank 0's.  Every rank makes R13 (Xv) of its pairs
only.  The other ranks take as many words as rank 0 (`MEMORY_SIZE`).  Per
sigma vector a rank receives r2 and r3, the rows of the blocks that it
shares with other ranks (hh), the rows (m, j) of its spectator in every
round and the rows of sigma4 of its pairs that the rings of the other ranks
made - at cc-pwCVQZ-X2C-TP over 16 ranks about 5 GB for the ring and a few
GB for the hh ladder, where a rank received a whole vector (36.5 GB) and
sent its sum by slices (36.5 GB) before.  The output gives the seconds of
the exchanges of the ring and of the hh ladder (in them the waiting for
the other ranks: the ring of a rank gives its rows back when the rings of
the ranks that hold its rows are done).

Measured with the earlier program - two whole vectors on every rank, the
vector to all ranks and sigma4 summed by slices (`xzmpi_slices_put`,
`xzmpi_slices_add`) in every sigma vector - (one job each, two nodes of 2 x Xeon Gold 6248R, `-N 2
--ntasks-per-node=1 -c 32`, `mpirun -ppn`; CCSDT once; `%eomthrs 1e-10`; the
same numbers of sigma vectors and the same roots, to all digits printed, in
every run; `bench/dea/native_mpi.sbatch`):

| TZ (10 + 106), job 32265683 | a sigma vector with the solver (s) | of it: intermediates, pp, ring, rows; the transfers (s) | EOM-DEA, both targets (s) | work array used / resident, rank 0 (MB) |
|---|---|---|---|---|
| 1 rank x 32 threads, one node | 1.43, 1.37 | 0.10, 0.38, 0.29, 0.45; - | 97.8 | 13 884 / 14 218 |
| 2 x 16, one node | 1.09, 1.06 | 0.11, 0.25, 0.30, 0.18; 0.10 | 75.5 | 7 767 / 8 384 |
| 2 x 32, two nodes | 0.88, 0.86 | 0.11, 0.18, 0.22, 0.21; 0.05 | 62.1 | 7 767 / 8 228 |
| 4 x 16, two nodes | 0.66, 0.66 | 0.11, 0.13, 0.15, 0.09; 0.10 | 46.3 | 5 074 / 5 724 |
| **168 spinors** (10 + 158), job 32265684 | | | | |
| 1 rank x 32 threads, one node | 7.99, 7.68 | 0.35, 2.99, 1.82, 1.73; - | 557 | 68 807 / 67 974 |
| 2 x 16, one node | 6.67, 6.47 | 0.45, 2.16, 1.72, 0.96; 0.53 | 468 | 38 696 / 38 742 |
| 2 x 32, two nodes | 4.84, 4.60 | 0.41, 1.43, 1.22, 0.86; 0.39 | 339 | 38 696 / 38 603 |
| 4 x 16, two nodes | 3.81, 3.72 | 0.50, 1.01, 0.86, 0.46; 0.64 | 269 | 23 536 / 23 866 |
| 4 x 16, two nodes, the intermediates split and the ring by batches (job 32269591, c092/c094, `--exclusive`) | 3.40, 3.33 | 0.37, 1.01, 0.61, 0.46; 0.61 | 242 | 23 536 |
| the same nodes, the ring by blocks (the binary of 93164be) | 3.63, 3.59 | 0.37, 1.01, 0.86, 0.46; 0.59 | 258 | |
| **DYVQZ-SOG** (10 + 206), jobs 32268202 / 32269044 (c227/c232 both) | | | | |
| 1 rank x 32 threads (job 32265420, c301) | 29.9, 29.2 | 0.7, 9.9, 10.2, 5.0; - | 2483 | 99 800 / 98 500 |
| 4 x 16, two nodes, the ring by blocks (93164be) | 15.22, 15.12 | 1.19, 3.58, 5.90, 1.55; 1.55 | 1085 | 48 963 |
| 4 x 16, two nodes, the ring by batches (fceb0a9) | 13.09, 13.14 | 1.21, 3.58, 3.77, 1.55; 1.55 | 941 | 48 963 |

**Before and after the distribution of the vectors**, the same nodes (two
nodes of 48 cores each, `--exclusive`; the binary of master c1d8df3 and the
branch; job 32291726, c193/c194, and job 32291727, c196/c197):

| case, ranks x threads | a sigma vector with the solver, targets 0 / 2 (s): master, branch | the transfers and sums, master, branch (s) | work array used, rank 0 (MB): master, branch | the other ranks (MB): master, branch |
|---|---|---|---|---|
| cc-pVQZ-X2C-TP, 1 x 48 (the same code) | 30.7 / 28.3, 28.1 / 27.9 | - | 151 080, 151 080 | |
| 2 x 48, two nodes | 16.4 / 17.3, 15.5 / 15.3 | 0.67 / 0.53, 0.72 / 0.63 | 84 832, 83 217 | 86 712, 84 484 |
| 4 x 24, two nodes (two per node) | 10.7 / 10.8, 9.8 / 9.9 | 1.61 / 1.64, 1.27 / 1.08 | 53 274, 50 407 | 50 094 - 51 245, 43 509 - 47 605 |
| cc-pwCVTZ-X2C-TP 16 e, 1 x 48 | 19.3 / 18.9, 19.8 / 19.6 | - | 136 732, 136 732 | |
| 2 x 48, two nodes | 11.3 / 11.3, 11.6 / 11.2 | 1.38 / 1.42, 1.12 / 0.84 | 73 990, 71 792 | |
| 4 x 24, two nodes | 8.0 / 7.9, 7.6 / 7.3 | 1.40 / 1.43, 1.22 / 0.92 | 45 107, 41 767 | |

At these sizes the vectors are small next to the rest (2.8 GB for
cc-pVQZ-X2C-TP), and the two programs take about the same time: the ring
and the hh ladder now move the rows they need where the earlier program
moved whole vectors; two ranks of 24 threads per node, which fitted before
at cc-pVQZ-X2C-TP, are again 1.5 - 1.6 times one rank of 48 per node.  The
change matters where the vectors are large: cc-pwCVQZ-X2C-TP (36.5 GB a
vector, 20.4, 20.7).

The pp ladder, the ring, the rows of sigma4 and the hh ladder are divided;
the small intermediates were made by every rank in the first runs of the table (0.1 s
for 116 spinors, 0.5 s for 168 on four ranks; split by the occupied index
since, 0.37 s), and the transfers (the vector to all ranks and sigma4 by
slices: 0.2 GB each for 116 spinors, 1.2 GB for 168, 2.8 GB for DYVQZ)
take 0.05 - 0.1 s, 0.4 - 0.6 s and 1.6 s: two nodes are 2.2 times one rank
for 116 spinors, 2.35 times for 168 and 2.6 times for DYVQZ (one rank on
another node of the same type).  At DYVQZ the ring of ranks 0 and 1 (the first node, c227) takes 262
s of the run against 118 - 120 s on ranks 2 and 3 (c232), with the pp ladder and the rows equal on all ranks (the per-rank sums of the
last output lines of the ranks; not understood: at 168 spinors on c092/c094 the four rings are equal); the time ranks 2 and 3 wait is in their
transfers (354 - 361 s against 181 s on rank 1).  The 168-spinor run of job
32269043 shared a node with other jobs (every pass 30 % slower) and is not
in the table.  (The
ring of an earlier version, in which every rank expanded all rows of every
spectator for its own output rows, took 0.25 s on four ranks; job
32265195.)  The ranks hold their parts: rank 0 of two takes 7.8 GB where one
rank takes 13.9 GB (R13 and the subspace divided).

**Every rank has the words of rank 0, and rank 0 plans for the rank that
holds the most** (2026-10-20).  The ranges of pairs are cut by the elements
of r4 summed over the targets, so the parts of R13 and the rows of one
target differ from rank to rank; until then rank 0 chose the batches and the
subspace for its own part.  At cc-pV5Z-X2C-TP over 8 ranks rank 3 holds
11.6 GB of R13 and 2.80 GB of a vector where rank 0 holds 10.7 GB and 2.63
GB, and with 12 vectors of the subspace (rank 0's choice) rank 3 stopped at
its first sigma vector, 0.36 GB short (job 32272248).  Now every rank runs
the dry runs of all ranks alike (the pairs of a rank are known to all,
`xzdranges`; `asrank` in `xzdeam.inc` puts a rank in the place of rank r
for a dry run): R13 goes to `XZDXV` on all ranks if one rank does not hold
it; the batches are bisected on the largest sigma vector of any rank plus
what that rank holds of R13 and of the parts of the two vectors more than
rank 0 (`r13more`, `vecmore`); and the words of the subspace are those of
the rank that holds the fewest vectors of its part (`ranksub`).
TZ over 3 ranks with `maxcor 500000000`: 9 vectors, the largest rank uses
3941 of 4000 MB (resident 4190 MB); the ranks other than 0 now print their
work array used and peak resident set at the end.

### 20.7 Timings

One job per comparison, the Python sector first (`bench/dea/native_bench.sbatch`:
`sector_bench.py` - ground state, three sigma vectors, the roots of 2 Omega
= 0 and 2 -, then the native program in node-local /tmp on the same
Hamiltonian), 32 cores of a Xeon Gold 6248R node, 120 GB:

| | Python sector | native | |
|---|---|---|---|
| **116 spinors** (job 32260659, c686) | | | |
| a sigma vector | 3.4 - 3.6 s | 1.45 s (pieces: rows 0.50, ring 0.43, pp 0.40, intermediates 0.10) | 2.4x |
| the roots of 2 Omega = 0 and 2 | 62 + 38 sigma vectors, 399 s | 37 + 32, 121 s (1.81, 1.66 s per sigma vector with the solver) | 3.3x |
| CCSDT | 22 s | 9.2 s | |
| the run | 469 s (integrals, CCSDT, preparation, three sigma vectors, roots) | 132 s | |
| memory | peak 10.9 GB | work array used 13.9 GB, resident 14.1 GB (the subspace of 23 vectors in it) | |
| **168 spinors** (job 32260660, c572) | | | |
| a sigma vector | 13.3 - 15.0 s | 6.97 s (pieces: pp 2.95 - of it the products 2.05 -, ring 1.83, rows 1.76, intermediates 0.35) | 2.0x |
| the roots of 2 Omega = 0 and 2 | 64 + 40 sigma vectors, 1615 s (14.1, 13.3 s each) | 38 + 32, 552 s (7.99, 7.78 s per sigma vector with the solver) | 2.9x |
| CCSDT | 58 s | 25 s | |
| the run | 1838 s | 588 s | 3.1x |
| memory | peak 48.7 GB (plan 9.3 GB fixed + 3.9 GB per sigma vector; a subspace of 12 vectors) | work array used 68.8 GB, resident 68.1 GB (the subspace of 23 vectors in it: 55 GB) | |

Every pass of the native sigma vector is faster (Python, 116 spinors, the
same node: rows 1.15, ring 0.79, pp 0.68, small intermediates 0.56 s); the
products are the same.

**DYVQZ-SOG** (10 + 206, 9 classes; 3.5e8 elements of a vector, 2.8 GB): the
native program alone in job 32265420 (node c301), against the production
run of the Python sector on a node of the same type (job 32236524, c705, 32
cores; not the same node):

| | Python sector | native | |
|---|---|---|---|
| a sigma vector | 47.4 - 48.3 s | 25.9 s (pp 9.9 - of it the products 7.3 -, ring 10.2, rows 5.0, intermediates 0.7; the expansions and reductions 9.5 of it) | 1.9x |
| the roots of 2 Omega = 0 and 2 | 57 + 42 sigma vectors, 5736 s | 43 + 41, 2483 s (29.9, 29.2 s per sigma vector with the solver) | 2.3x |
| CCSDT | 127 s | 81 s | |
| intermediates of the ground state | (in the sigma vector of the first target) | 21 s (Xv 11.9 GB) | |
| memory | MaxRSS 110 GB (three levels) | work array used 99.8 GB of 100 GB (the subspace: 13 vectors in the work array, 73 GB), resident 98.5 GB | |

The native runs needed fewer sigma vectors: the solver of section 16 (one
correction per unconverged root, a subspace of 23; 13 for DYVQZ) against
the block Davidson of the Python program (a subspace of 12), at a threshold
on the largest element (1e-10) instead of the norm (1e-9).  At DYVQZ the
expansions and reductions of the passes (memory-bound) are a third of a
sigma vector.

**From the block files, whole nodes** (2026-10-20; nodes of 2 x Xeon Gold
6248R, 48 cores, `--exclusive --mem=180G`, one run directory on /scratch16;
`%eomthrs` 1e-10, CC_CONV=10).  cc-pVQZ-X2C-TP (10 + 206), job 32272007
(c507, c512; the CCSDT amplitudes of the first run taken by the others):

| ranks x threads | a sigma vector with the solver (s) | of it: intermediates, pp, ring, rows; the transfers (s) | EOM-DEA, both targets (s) | work array used / resident, rank 0 (MB) |
|---|---|---|---|---|
| 1 x 48, one node | 25.0, 26.3 | 0.82, 8.21, 7.97, 4.36; - | 1686 (35 + 30 sigma vectors) | 155 747 / 153 745 (23 vectors) |
| 2 x 48, two nodes | 15.9, 14.7 | 0.81, 3.85, 4.11, 2.01; 3.02 | 1014 | 84 832 / 83 720 |
| 4 x 24, two nodes | 10.5, 10.6 | 1.13, 2.61, 2.75, 1.18; 1.80 | 695 | 53 274 / 53 056 |

(CCSDT: 28 cycles, 2.27 s per cycle on one node.)  The same roots in the
three runs (20.5).  Two ranks of 24 threads on a node are 1.5 times one of
48 (the threads of the second socket add little to one rank) - but at
cc-pV5Z-X2C-TP a rank needs 92 GB before its part (20.4), so two of them
do not fit a node.

**cc-pV5Z-X2C-TP** (10 + 350; the run directory from the block file in 67
s, 20.1), 8 ranks x 48 threads on 8 nodes: CCSDT 28 cycles, 11.6 s per
cycle, 326 s (job 32272248), E(CCSDT) = -148.916391284243 (E(SCF)
-148.325694978171); the intermediates of EOM-DEA 31 - 42 s; a sigma vector
66 s (target 0) and 87 s (target 2) with the solver, of it intermediates
10.6 / 18.9, pp 16.3 / 16.7, ring 12.0 / 11.7, rows 4.7 / 5.3, and the
transfers - the vector to the ranks and sigma4 by slices, 20 GB each - 23.7
/ 43.0 s (job 32272311, three iterations per target).  From the QZ rates at
one rank the pieces grow with the work of the case (`sizes`: the products
of the pp ladder 18 times those of QZ, the ring 11 times, the expanded rows
7.3 times): pp 8.2 x 18 / (8 ranks, about 9 x the cores used well) = 16 s
as measured; the transfers were not in that estimate and are now a third
of a sigma vector.

The production run of cc-pV5Z-X2C-TP, 12 ranks x 48 threads on 12 nodes
(job 32272434, the amplitudes of the timing run): the intermediates 38 s;
target 0: 51 sigma vectors (48 cycles, 14 vectors of the subspace), 2296 s,
45.0 s per sigma vector with the solver (intermediates 6.8, pp 10.5, ring
7.4, rows 2.7, 3p1h 0.8; the transfers 15.6); target 2: 38 sigma vectors (35
cycles, 15 vectors), 1947 s, 51.2 s (pp 12.3, ring 7.5, rows 3.4; the
transfers 18.8); EOM-DEA 4281 s, the run 4305 s (12 x 1.2 node-hours).

| | E(2 Omega = 0) / Eh | E(2 Omega = 2) / Eh | split / Eh | cm-1 |
|---|---|---|---|---|
| cc-pVQZ-X2C-TP (10 + 206), 1 x 48 | -150.26441254319741 | -150.26439571882486 | 1.6824373e-05 | 3.6925 |
| cc-pV5Z-X2C-TP (10 + 350), 12 x 48 | -150.27870443446366 | -150.27868757728118 | 1.6857182e-05 | 3.6997 |

(DEA-EOM-CCSDT, 4p2h, T of CCSDT, X2Ccorr; residuals 5.7e-11, 5.3e-11;
E(CCSDT) of 5Z -148.916391284243; the weights of 2p, 3p1h, 4p2h 0.8405,
0.1462, 0.0133.)

**cc-pwCVQZ-X2C-TP, all electrons** (14 + 322; the vectors distributed,
20.6; 2026-10-20): 16 ranks x 48 threads on 16 nodes, `MEMORY_SIZE` 155.7
GB, the CCSDT amplitudes of the run directory (`XZAMPS`, E(CCSDT) =
-149.026251166161), job 32291868: the intermediates 57 s; target 0: 40 sigma
vectors (38 cycles, 13 vectors of the subspace), 2064 s, 51.6 s per sigma
vector with the solver (intermediates 2.1, pp 11.9, ring 11.7, rows 4.1,
3p1h 0.9; the transfers and the sums over the ranks 19.6); target 2: 37
sigma vectors, 1857 s, 50.2 s (pp 14.2, ring 11.5, rows 4.8; the transfers
15.7); EOM-DEA 3921 s, the run 67 min (16 x 1.1 node-hours).  The
earlier program did not fit the nodes with a subspace in the work array
(20.4).  Of the transfers most is waiting: the per-rank sums of a timing
run on other 16 nodes (job 32303635, three iterations per target) give the
pp ladder of the ranks 9.0 - 14.7 s per sigma vector and the rows of
sigma4 2.8 - 4.7 s, with ranks of the same rows of r4 (the same blocks,
the same elements) 1.4 times apart (the pp ladder 142 and 205 s on two
ranks of 299 294 490 + 262 495 068 elements, 126 and 177 s on two of
260 567 780 + 249 412 075), so that the fast ranks wait in the exchanges of the ring (2 -
9 s per sigma vector) and of the hh ladder (0.7 - 3.5 s) and in the sums
that follow the passes; ranges cut by an estimated cost of the rows instead
of their elements moved the cuts by one pair and changed nothing.

| | E(2 Omega = 0) / Eh | E(2 Omega = 2) / Eh | split / Eh | cm-1 |
|---|---|---|---|---|
| cc-pwCVQZ-X2C-TP, 16 electrons (14 + 322), 16 x 48 | -150.38831805949553 | -150.38830096789087 | 1.7091604661e-05 | 3.75117 |
| the same, 1s frozen (12 electrons, 10 + 322; 10 ranks, earlier program) | -150.26877691671027 | -150.26876014093125 | 1.6775779017e-05 | 3.68186 |

(Residuals 9.8e-11, 3.8e-11; the weights of 2p, 3p1h, 4p2h 0.8429,
0.1441, 0.0130.)  The O 1s contribution to the splitting with
cc-pwCVQZ-X2C-TP: 3.15825644e-07 Eh = 0.06932 cm-1 (cc-pwCVTZ-X2C-TP:
0.05339 cm-1); the files in `o2zfs_coord/zccsdt_results/o2_1s_pwcv`
(`pwcvqz_ae.json`, `summary.json`).

### 20.8 Not done

- The folded 5p3h row (complete 5p3h: section 21); the T of CCSDTQ / CC4
  in the sector; DIP; left-hand vectors; the vectors of the roots in a file.
- Only the elements of the parity of the target are needed (half of the
  vector): the masks are applied to a vector of both parities, as in the
  Python program.
- Over ranks: the exchanges of the ring and of the hh ladder and the sums
  over the ranks are synchronous; nothing overlaps them with the products.
  The deal of the ranges and of the ring is static: nodes of one job that
  differ by 1.4 times in the same work (cc-pwCVQZ-X2C-TP, 20.7; at DYVQZ the
  ring of the ranks of the first node of a pair took twice as long) make
  the others wait, a third of a sigma vector at 16 ranks.  The rows of the
  six-index products are reduced row by row (one entry per call of the
  kernel).
- The subspace in files reads it about seven times per new vector (16.4):
  168 spinors under 16 GB, 50 s per sigma vector with the solver instead of
  8 s.  A subspace smaller than 3 roots + 4 vectors in the work array is not
  tried.
- Every rank holds the whole small intermediates of a sigma vector (U,
  XvR), the two buffers of the rows of a spectator of the ring (10.8 GB at
  cc-pwCVQZ-X2C-TP) and the arrays of the ground state but R13 - 73 GB of a
  rank at cc-pwCVQZ-X2C-TP before its part (20.4), so that two ranks do not
  fit a node there; the table of the quadruples of the classes is on every
  rank as well.

## 21. EOM-DEA-CCSDT with complete 5p3h

    %xzdea
    0 1
    2 1
    parity g
    level 5p3h                            R = 2p + 3p1h + 4p2h + 5p3h on CCSDT

The level of docs/eom.md 11.11: the right-hand vector to 5p3h, complete,
on the ground state of CCSDT (T1-dressed integrals, T2, T3) - the target
states described as CCSDT describes a ground state.  The Python program
(`src/zccsdt/cc/dea5_sector.py`, `Engine5.sigma`) is the reference of the
sigma vector and of the roots; its factorization, its layouts and its
passes are taken over one to one (the term list: the module docstring of
`dea5_sector`).  The application is the zero-field splitting of O2 X
3Sigma_g- at cc-pVQZ-X2C-TP (10 + 206 spinors), whose r5 (3.4e10 elements,
269 GB) is not one node's (docs/eom.md 11.11: the Python plan 99 GB + 806
GB): every array of the size of r5 is distributed over the ranks.  Sources:
`xzdea5d.F90` (the layout of the 5p3h blocks, of the intermediates, the
distribution of the rows of r5), `xzdea5.inc` (the intermediates of the
ground state, the kernels for rank 5), `xzdea5s.inc` (the 5p3h part of a
sigma vector, the small intermediates, Z4, the pp ladder), `xzdea5r.inc`
(the ring, the hh ladder), `xzdea5x.inc` (the rows of sigma5, Xv5 and
E1x); the driver, the solver and the vectors of section 20 with r5 behind
r4.  Kernels (`csrc/t4sym.c`): `zc_t5s_quints`, `zc_t5s_quints_par` (the
quintuples of a class, of one parity), `zc_t5s_x23`, `zc_t5s_x14`,
`zc_t5s_x14p` (the expansions and reductions of a row of r5 on
[(ab)][(cde)], [a][(bcde)], and on [a][(bcde)] with the quadruples of one
parity).

### 21.1 Input and output

| what | ZMAT | default |
|---|---|---|
| the level | `level 5p3h` in `%xzdea` (`level 4p2h`: section 20) | 4p2h |
| the subspace in files elsewhere | `eomvdir dir` in `%xzccsdt`: the files `EOMCnnnn`, `EOMHnnnn` in `dir` (made if it is not there; e.g. node-local `/tmp/...`, every rank its part) | the run directory |

The rest as in 20.1.  A run directory: `bench/dea/native_rundir.py
<file.npz | file.h5> <dir> --level 5p3h [...]`.  Output per target, in
addition: the elements of r5 in the blocks and those stored (the parity),
the quintuple lists (their pool), Xv5 of one k (and of all k on one rank);
per sigma vector the seconds of the pieces of the 5p3h part:

    5p3h part per sigma vector (s): r5 and r4 to the passes, G and XoRp, Z4, XoR5, pp, ring, hh, rows, Xv5 and E1x
      (of them: the expansions and reductions, the products of pp, of the ring, of the rows [(ab),(cde)] + [a,(bcde)]; sigma4, sigma5 back)
      (over the ranks: the rows of the ring, of the hh ladder, of Xv5 and E1x, the sums of the intermediates)
      (Z4: the slots or the slabs made once over the ranks; Xv5 and E1x: the rounds of the exchanges at once)

`XZDEA` and JOBARC as in 20.1; the weights are those of 2p, 3p1h, 4p2h
(the rest is 5p3h).

### 21.2 The vector

r5 of the shift s: one block per class M of the virtual quintuples, its
rows the occupied triples (i<j<k) of group M - s, its columns the
quintuples a<b<c<d<e of class M (lexicographic, `zc_t5s_quints`) - of a row
only the quintuples whose parity goes with that of the triple and of the
target (`parity g`): half of r5, everywhere - in the vectors of the solver
and in every pass of a sigma vector; a kernel takes the quintuple list of
the parity of its row.  The lists (the pairs (block, parity), 16 GB at
cc-pVQZ-X2C-TP) are made when they are used, in a pool (the lists of the
rows of the rank and the ten largest others; the least recently used goes;
`dea5pool`).  The natural layout (`dea5_sector.ProblemSector5.pack5`, the
test hook `deasig`): per block the rows of the triples of parity 0, then
those of parity 1.  The part of a rank (21.4) is its rows of r4 followed
by its rows of r5, per block those of parity 0, then 1.  cc-pVQZ-X2C-TP, 2
Omega = 0: r5 3.36e10 elements in the blocks, 1.68e10 stored (134 GB), a
row 1.15 GB.

### 21.3 The sigma vector

`N(xzd5sig)` after the rows to 4p2h of section 20, the order of
`Engine5.sigma`: r4 of all rows on every rank; the small intermediates (G,
G3, G2, G5, r3 on (h | pair), XoRp; XvR5; Y4, XoR5 and XoRp from r4); **the
pp ladder** per block and parity group of rows in batches (`zc_t5s_x23`
expand to [(fg)][(cde)], one product per group of (fg) with Wq5 = Wpp + F_vv
/ 4, reduce), its thin products (XoR5, sigma3, sigma4 on (a | bcd)); **the
ring** per spectator pair (u, w): the rows (m, u, w) on [(m e)][(bcde)] by
`zc_t5s_x14p`, grouped by d = class(m) - class(e) and by the parity of the
quadruples - two products of a quarter of the work and half the size
instead of one -, times the stacked operator [Wr rows (i a) | f_ov |
<mu||ei'> | - <mw||ei'>], reduced into the rows (i, u, w) and sigma4; **the
hh ladder** per block and parity group; **the rows of sigma5**, one triple
at a time: X[(ab), (cde)] per group of (ab), one product of the stacked
operands D1 - D8 (Wv | r4, t2 | XoR5, r3 | Xv2, Y4 | t3, XoRp | t3, t3 | Z4,
t3 | XvR5, Xo3 | r3) in parts of at most a row on (a | bcde), then X[a,
(bcde)] (E1: Wo | r4, E2: W2 | r4), each reduced once; **Xv5 and E1x**
(Xv5(f; k; bcde) from r4, G5 | t3 and the rows (l, n, k) of r5; E1x: t2 |
Xv5 into the rows (x, y, k)); sigma4 from the terms on (a | bcd).  Real
arithmetic where the integrals are real, the same source for complex.

**Z4** (Z4(mf; k; K) = sum <ml||fh> r4(lk; h K), 25 GB at cc-pVQZ-X2C-TP)
is made in the pass of the rows, slab k by slab: on one rank in as many
slots as there are indices in the triples of the rank if they fit below
the peak of the ring pass (else at least three, the least recently used
replaced); over ranks once over all ranks (21.4).  Before (2026-10-21) a
slab was made again for almost every triple: 14.6 of 137 s per sigma
vector at cc-pVTZ-X2C-TP.

**Xv5 on one rank** (11.8 GB per k at cc-pVQZ-X2C-TP): all k at once when
they take at most a quarter of the free work array (decided with the
layout of the target, so that the plan of a sigma vector holds them):
made before the rows, the term of r5 from every row once into the slabs
of its three indices, E1x in the products of E1 (as `Engine5`).  Else k by
k after the rows: a row expanded three times for the term of r5 and
reduced three times for E1x (35 of 137 s at cc-pVTZ-X2C-TP).

### 21.4 Over MPI ranks

The rows of r5 go to the ranks by occupied triples: one range of triples
per rank (`dtb`), the same for all targets, of about equal sums of the
elements of their rows over the targets; a rank holds its rows of r4 (the
pairs of section 20.6) and its rows of r5 - its part of every vector of
the solver and of the subspace.  In a sigma vector: r4 of all rows on
every rank (gathered, 2.8 GB at cc-pVQZ-X2C-TP), the terms of r5 in sigma4
summed on their owners; the small intermediates summed over the ranks; the
pp ladder and the hh ladder on the rows of the rank (the hh ladder
exchanges the column pieces of a block); the ring: the spectator pairs go
to the ranks in rounds (the deal of the ring of r4, 20.6), a rank takes
the rows (m, u, w) of its pair from their owners and gives the rows (i, u,
w) back (added there); the rows of sigma5: the triples of the rank.

**Z4 over ranks**: the blocks (k, mu, fe) of the slabs that some rank
needs go to the ranks by their cost (the largest first to the least loaded
rank); a rank makes its blocks from the rows (l, k) of r4 (made once per
k) and sends each to the ranks that need its slab (`N(xzd5z4sh)`) - when
the slabs of every rank fit below its peak so far, else every rank makes
the slabs it needs (the slots of one rank).

**Xv5 over ranks**: the index k of Xv5 has its ranks (`xzd5zgroups`: with
at least as many ranks as occupied spinors, every k a group of ranks that
share the pairs (l, n) of the term of r5 and the pairs (x, y) of E1x; with
fewer, the k in slots); the ranks of k sum Xv5(.; k) over a
sub-communicator (`xzmpi_sub_split`, `xzmpi_sub_allsum_w`); a rank takes
the rows (l, n, k) from their owners and gives its rows (x, y, k) of
sigma5 back, as many rows per exchange as fit below the peak of the ring
pass (`dea5xb`; one per exchange before 2026-10-21: 14 of 88 s per sigma
vector at cc-pVTZ-X2C-TP on 2 ranks were the waits of the synchronous
rounds).

### 21.5 Memory and disk

The plan of section 20.4 with the arrays of 21.3: the quintuple pool, Z4,
Xv5, XoR5, Y4, XvR5 and the buffers of the passes are in the dry run of a
sigma vector; the batches of the passes bisected as in 20.4; the subspace
takes what is left (in the work array, smaller, or in files - with
`eomvdir` on the disk of every node).  **`deaplan`** prints the plan of
every rank for 5p3h (also the largest regions at the peak of a sigma
vector).  cc-pVQZ-X2C-TP (`pvqz_plan`), GB of the work array of the rank
that takes the most:

| ranks | stays | two parts (with the pool, 16 GB) | a sigma vector | part of a vector | total, subspace in files | vectors of 23 in 150 GB |
|---|---|---|---|---|---|---|
| 16 | 11.1 - 13.2 | 31.5 - 34.9 | 56.4 - 61.0 | 8.2 - 9.5 | 102 - 110 | 2 - 3 |
| 24 | 10.6 - 11.7 | 24.4 - 28.4 | 53.3 - 59.9 | 4.8 - 6.7 | 92 - 101 | 3 - 6 |
| 32 | 10.1 - 11.2 | 22.0 - 26.0 | 52.8 - 59.4 | 3.5 - 5.6 | 87 - 97 | 4 - 9 |
| 48 | 9.3 - 10.3 | 19.5 - 23.2 | 52.3 - 58.9 | 2.2 - 3.5 | 84 - 94 | 8 - 15 |

The peak of a sigma vector is the pp ladder at 16 ranks (a batch of 22 GB
and its product), the ring from 24 ranks on (a batch 11 GB, its product 11
GB, the rows of the spectator pair in and out 9 + 9 GB), on top of what
stays (about 10 GB: B4, Xv2, r4 and the terms in sigma4 of all rows).

### 21.6 Validation

**The sigma vector element by element against `Engine5.sigma`**
(`bench/dea/native_check5.py <rundir> <binary> [--target t] [--batch n]
[--complex] [--mpi n] [--piece p] [--xz option]`; `check5_runs.sh`,
`check5_pieces.sh`), the largest difference of every part:

| case | runs | 2p, 3p1h, 4p2h | 5p3h |
|---|---|---|---|
| O2 n26 (10 + 16) X2Ccorr, 2 Omega = 0, 2, -2 | serial; batches of 1 and 50 elements; MPI 1, 2, 3, 4, 10, 11, 13 ranks; small pools, `dea5z4 3`, `dea5xb 2`, `dea5xa 0` / 1 | <= 1.1e-14 | 1.4e-14 of 17 - 20 |
| n26, random spinor phases (complex) | serial; 1, 2, 3, 11 ranks | <= 1.3e-14 | 1.6e-14 of 21 |
| DYVDZ-SOG (10 + 42) X2Ccorr | serial; 2, 3, 4, 8, 13 ranks | 1.6e-13 of 55 | 1.8e-13 of 74 - 77 |
| the pieces (pp ring hh x23 x14 e1x) alone | serial, MPI | - | 1e-15 |

**Roots** (ESTATE_CONV 10):

| case | native | Python sector | |
|---|---|---|---|
| n26, complex phases, the subspace in files of `eomvdir` (3 ranks) | -149.869782541593, -149.869766407516 | -149.86978254160, -149.86976640752 | split 3.5410 cm-1 |
| DYVDZ-SOG X2Ccorr (serial) | -150.094871569832, -150.094854897417; split 1.6672415e-05 Eh | -150.09487156985, -150.09485489743; 1.6672424e-05 (dea_strings: 1.66724049e-05) | 3.6592 cm-1 |
| cc-pVDZ-X2C-TP X2Ccorr (10 + 42; serial, 8 threads: 34 + 31 sigma vectors, 168 s with CCSDT) | -150.096289292237, -150.096272539749; split 1.6752487e-05 Eh | -150.09628929224, -150.09627253975; 1.6752489e-05 (52 + 40 sigma vectors, 707 s) | 3.6767 cm-1 |
| cc-pVTZ-X2C-TP X2Ccorr (one node each, jobs 32331084, 32331086) | -150.228488820219, -150.228471307068; split 1.7513151e-05 Eh | -150.22848882023, -150.22847130708; 1.7513154e-05 | 3.8437 cm-1 |

The 4p2h level is unchanged (DYVDZ-SOG -150.087442607823, -150.087426722255,
20.5); `cfour/merge/test/run.sh`: PASS with the serial and the MPI binary.

### 21.7 Timings

**One node, against the Python sector** (`bench/dea/native5_bench.sbatch`:
the Python sector's sigma vectors of random vectors first, then the native
program in node-local /tmp, the serial binary on 48 threads; Xeon Gold
6248R, 180 GB), cc-pVTZ-X2C-TP (10 + 106, 2 Omega = 0; 6.8e8 elements of r5
stored):

| | Python sector | native | |
|---|---|---|---|
| a sigma vector, job 32333565 (c456) | 202.0, 194.6 s (ring 71, rows 43, pp 35, small 16, pack and unpack 34) | 93.7 / 85.0 s with the solver, 2 Omega = 0 / 2 (rows 34.3 - of it the products 25.1 -, pp 24.4 - 11.5 -, ring 14.3 - 3.5 -, Xv5 and E1x 13.4, Z4 2.5, XoR5 2.1; the expansions and reductions 20.8) | 2.2x |
| the same before 2026-10-21 (job 32331022, c074) | 140.1, 138.3 s | 136.7 / 125.7 s (Xv5 and E1x 34.9, Z4 14.6, ring 27.4) | 1.0x |
| the roots of 2 Omega = 0 and 2 (the code before 2026-10-21) | 77 + 48 sigma vectors, 17 901 + 10 581 s (jobs 32325450, 32325451) | 45 + 40, 6 233 + 6 379 s (jobs 32331084, 32331086; 138.5, 159.5 s per sigma vector; 12 vectors of the subspace in the work array) | 2.9x, 1.7x |
| memory | RSS 55.6 GB (5.7 GB fixed + 37 GB per sigma vector) | a sigma vector 9.4 GB of the work array (batches of 4.3 GB; Xv5 of all k 4.9 GB); 150 GB used with the subspace | |

The Python times vary by a factor of 1.45 between nodes of the same type
(the ring 40 s on c074, 71 s on c456); one job per comparison.  On 8
threads (the holding job, one sigma vector of a random vector, `deasig`;
24 GB of work array): 423 s before the masks of the kernels, 377 s with
them, 312 s with batches of 3.2 GB (the ring 189, 140, 80 s).

**Over nodes** (one rank of 48 threads per node, `native5_mpi.sbatch`, two
iterations of 2 Omega = 0 - six sigma vectors -, the subspace in the work
array; seconds per sigma vector with the solver):

| nodes | job | per sigma vector | pp | ring | rows | Xv5 and E1x (of it the exchanges) | Z4 | speed-up |
|---|---|---|---|---|---|---|---|---|
| 1 | 32334622 (c207) | 90.8 | 25.3 | 11.5 | 33.0 | 14.0 (Xv5 of all k) | 0.9 | 1 |
| 2 | 32334855 | 63.9 | 12.1 | 9.8 | 10.7 | 18.2 (4.1) | 0.3 | 1.42 |
| 4 | 32334862 | 36.3 | 7.3 | 5.4 | 6.5 | 11.3 (0.6) | 0.2 | 2.50 |
| 8 | 32334863 | 25.2 | 3.3 | 2.9 | 2.8 | 10.4 (3.9) | 0.1 | 3.61 |

The passes of the rows of r5 (pp, ring, rows) scale with the ranks (7.8x
from 1 to 8 nodes for the three together); Xv5 and E1x do not: with fewer
ranks than occupied spinors (10) a rank holds Xv5 of its k (k mod ranks)
and takes, walks and gives back every row with one of them - most rows on
every rank (on 8 ranks two ranks have two k, the others one: the waits of
the exchanges).  With slot by slot (before; job 32334257, 2 nodes) Xv5 and
E1x took 25.8 s.  On the node
c058 (jobs 32333658, 32334057; its cores at 2.7 GHz against 4.0 GHz of
the other node) the rows of its rank took more than 25 minutes instead of
14 s, with or without Z4 made over the ranks, while the other rank waited
in the exchanges of Xv5; the same binary on two other nodes ran normally
(job 32334257): the jobs exclude c058.

### 21.8 O2 / cc-pVQZ-X2C-TP

TODO

### 21.9 Not done

- The expansions and reductions of the rows of r5 dominate the passes
  (about 60% of a sigma vector at cc-pVTZ-X2C-TP on 48 threads): every row
  is expanded and reduced six times in the ring (three spectator pairs),
  twice in the pp ladder and in the rows, and the kernels walk the
  quintuple lists element by element.
- Over ranks Xv5 is held per k by groups of ranks; the term of r5 and E1x
  move every row three times each way (the Python route - the term of r5
  in the ring, E1x in E1 - needs Xv5 of all k on every rank: 117 GB at
  cc-pVQZ-X2C-TP).
- The rounds of the ring and of Xv5 are synchronous; nothing overlaps the
  exchanges with the products.
- The T of CCSDTQ in the sector; DIP; left-hand vectors; the vectors of the
  roots in a file.
