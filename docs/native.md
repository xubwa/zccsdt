# xzccsdt: the native program in CFOUR's two-component chain

`cfour/xzccsdt` is a program without Python for spinor CCSDT and CCSDT(Q)
energies.  It runs as a member of CFOUR's X2CSO chain, after `xjoda`,
`xsox2catm`, `xsox2chf` (and `xx2camf` for X2CAMF) and `xx2cvtran` with
`ABCDTYPE=STANDARD`, in their directory:

    xjoda; xsox2catm; xsox2chf; xx2cvtran; xzccsdt

The entry point and the scheduler are Fortran in the manner of CFOUR's x2c
programs, the kernels are the C kernels of this repository (`csrc/`), and all
working memory is the work array that CFOUR's `CRAPSI` allocates
(`MEMORY_SIZE` of the ZMAT).  The algorithms are those of the Python program
(`src/zccsdt/cc/ccsdt_opt.py`, `quadruples.py`, `t3engine.py`;
docs/factorization.md), which is the reference of every number below.

State of this document: the program at the commit that adds it.  What is
measured says so (job, node); what is not done is listed at the end.  In
short: CCSDT and (Q) - dense or in the blocks of symmetry labels, complex or
real, (Q) over MPI ranks with restart - are there and validated against the
Python program; the Lambda equations and (Q)_Lambda are not.

## 1. Build and run

    source /data/lcheng24/intel/oneapi/setvars.sh --force
    cd cfour/xzccsdt
    make                # -> build/xzccsdt/xzccsdt           (serial: no MPI in it)
    make MPI=1          # -> build/xzccsdt_mpi/xzccsdt       ((Q) parallel over ranks)

Compilers and flags are those of the CFOUR build (`ifort`, `icc`, `-i8`,
`-assume byterecl`; `make.config` of `~/apps/cfour-x2cso-rebuild`).  Linked:
`liblibr.a` of that build (`CRAPSI`, `GETREC`, `PUTREC`, `GETMEM`, `ERREX`,
`CRAPSO`), MKL with 8-byte integers (`-lmkl_intel_ilp64 -lmkl_intel_thread
-lmkl_core -liomp5`, the link line of CFOUR), and for `MPI=1` Intel MPI through
`mpiifort` / `mpiicc`.  Nothing is written to the CFOUR tree and no CFOUR
source is copied.  The run-time libraries are found through an rpath: the
executable runs without the oneAPI environment (the MPI launcher needs it).

The C kernels (`KERNELS` of the Makefile: `ccsdt_t3.c`, `t3pack.c`,
`tnpack.c`) are compiled from `csrc/` with `-DZC_USE_MKL -DMKL_ILP64`: their
BLAS calls go through `mkl_cblas.h`, whose integer is then 64 bits, as the
kernels' own `int64_t` arguments.  The program checks at start that the
kernels were compiled for 8-byte BLAS integers (`sizeof(MKL_INT)`) and stops
otherwise.  A GEMM with more than 2^31 elements in one operand was not run
(32 GB for one complex operand); the results of all cases below agree with
the LP64 Python build.

    xzccsdt [method]

`method`: `ccsdt`, `ccsdt[q]`, `ccsdt(q)`; `(q)` or `[q]` for the correction
alone from the amplitudes of an earlier run (file `XZAMPS`).  Without the
argument: `CALC` of the ZMAT as joda stored it (`CALC=CCSDT`, `CCSDT[Q]`,
`CCSDT(Q)`, `CC(n)` with `EXCITATION=3`, `CC(n-1)(n)` with `EXCITATION=4`).
Thresholds: `CC_CONV=N` - the amplitudes (|dT1| + |dT2| + |dT3| of the full
tensors) below 10^-N and the energy below 10^-(N+2), as the Python front end
reads it - and `CC_MAXCYC`.  Frozen and dropped spinors are what CFOUR left
(`DROPMO`, `%vircut`): the program takes the active spinors of the integral
files.

Exit: status 0 and CFOUR's `@CHECKOUT-I` line at the end (the line before:
`XZCCSDT finished: <method> energy done.`); every error goes through CFOUR's
`ERREX` (status 1, the `ERROR` banner), with a line `@XZCCSDT-F, ...` that
says what and, for memory, how many words are needed and how many are there.
The final energy is written to JOBARC as `TOTENERG`, as `xx2cccsd` does.

Threads: `OMP_NUM_THREADS` (kernels and MKL share Intel's OpenMP runtime).

### MPI: (Q) over ranks

    export PATH=$PATH:/cm/shared/apps/slurm/current/bin   # in a job: mpirun needs srun (section 8)
    export OMP_NUM_THREADS=8 I_MPI_PIN_DOMAIN=omp         # threads per rank
    mpirun -np 4 -ppn 1 xzccsdt "ccsdt(q)"                # Intel MPI, inside a Slurm allocation

e.g. `sbatch -N 4 --ntasks-per-node=1 -c 8 --mem-per-cpu=3700M` (one rank
per node; `-ppn 1`) or `-N 1 --ntasks-per-node=4 -c 8` (`-ppn 4`: the case
that was run).
Only (Q) is parallel over ranks; the CCSDT iterations are those of rank 0,
with its threads.  Rank 0 is the CFOUR program (CRAPSI, JOBARC, the output).
The other ranks get joda's flags from it, allocate a work array of the same
size themselves (`GETMEM`), write their output to `xzccsdt.rNNNN`, wait until
rank 0 has the amplitudes (a blocking broadcast: with Intel MPI a waiting rank
spins on one core unless `I_MPI_WAIT_MODE=1`), then read the integral files
and `XZAMPS` themselves.  The run directory must therefore be on a file system
that all nodes see.  All ranks read the same files and make the same decisions
(blocks, real or complex, the task list): nothing but the flags, the sizes of
JOBARC and the final sums is communicated.

The two steps can be two invocations, as the X2CSO programs are called one
after the other: `xzccsdt ccsdt` (serial or one rank), then
`mpirun ... xzccsdt "(q)"`.

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
| `xzlabels` | optional, not a file of CFOUR: a symmetry label per spinor (section 6) |
| `XZAMPS` | the amplitudes of an earlier run, for `(q)` / `[q]` and for the ranks other than 0 |

`moinpphh` is not read (it is the conjugate of `moinhhpp`; the Python reader
compares the two).  The numbers of active spinors come from the lengths of
`moinhhhh` and `moinhhpp`, as in the Python reader, and are checked against
`nmototal` and `NUMDROPA`.

Written:

| file | content |
|---|---|
| `JOBARC` | `TOTENERG`: the energy of the method |
| `XZAMPS` | after converged CCSDT: unformatted stream; `'XZAMPS01'`, 5 integers of 8 bytes (occupied, virtual, words per element: 1 real / 2 complex, classes of the labels, elements of T3 as stored), the correlation energy (real, 8 bytes), then T1 `[i][a]`, T2 `[i][j][a][b]`, T3 as stored (packed i<j<k, a<b<c; its blocks one after the other), row-major, in the order of the spinors of the run that wrote it |
| `XZDIIS` | only if the DIIS vectors do not fit in the work array: direct-access file with the T3 parts of the vectors; deleted at the end |
| `XZQCHK.NNNN` | one per rank during (Q): a line per finished task (task, its two energies, seconds); deleted when (Q) has finished |
| `xzccsdt.rNNNN` | output of the ranks other than 0 (MPI) |

A file that reads `XZAMPS` must run in the mode of the run that wrote it
(labels used or not, real or complex: both follow from the same integral
files, so this only matters with `XZ_LABELS` / `XZ_REAL`); the header is
checked.

**Limit of the reader:** `moinpppp` is read as one Fortran record into the
place where the packed block stays (section 4).  ifort writes a record of
more than 2 GB as sub-records, which the Fortran reader takes; the Python
reader and writer (`zccsdt.cfour`) assume one record with 4-byte markers and
do not go beyond 2 GB (v > about 150 spinors).  No case of that size was run.

## 3. Sources

| file | what |
|---|---|
| `xzccsdt.f` | main program in CFOUR's style: MPI start, CRAPSI on rank 0, the work array `ICORE(I0)`, `XZDRV`, CRAPSO |
| `xzdrv.F90` | driver: job (`xzjob`), labels, the mode loop (labels x real), `xzsetmode` (classes, groups, block sizes), `xzlayout` (tables and integrals in the work array), energies, TOTENERG |
| `xzread.F90` | readers of CFOUR's files and of `xzlabels`; checks of the labels and of the imaginary parts |
| `xzmaps.F90` | index tables of pairs and triples (grouped by classes), layout of T3 in blocks |
| `xzmem.F90` | the stack on the work array, dry runs |
| `xzplan.F90` | the memory plan |
| `xzdata.F90` | sizes, the job, offsets of the arrays that live for the run |
| `xzdiis.F90` | DIIS: slots, the file, the small linear problem (ZHEEV) |
| `xzpar.F90` | ranks and the task list of (Q), checkpoint files |
| `xzkern.F90`, `xzkern.inc` | explicit `bind(C)` interfaces of the kernels, one include for `_d` and `_z` |
| `xzmpi.c`, `xzutil.c` | C: the MPI calls behind a few functions (stubs without MPI); free disk space, resident set, BLAS integer size |
| `xztype.h` | the element type of a compilation of the type-generic sources |
| `xzimd.inc` | T1-dressed integrals, W and F intermediates, the CCSD part, the occupied operator |
| `xzstep.inc` | one CCSDT step: the order of the stages, the placement of every array |
| `xzblk.inc` | the T3 passes in blocks of the labels |
| `xzdiisg.inc`, `xzcc.inc` | DIIS update, the iterations, `XZAMPS` |
| `xzq.inc`, `xzqs.inc` | (Q): the task loop and the dense quadruple; the quadruple in sectors of the labels |
| `xzcc_z.F90`, `xzcc_d.F90` | the two instances of the `.inc` files |

Kernels of `csrc/` that are called: `zc_t3_pass1w`, `zc_t3_pass2w`,
`zc_t3_occop`, `zc_t3_update`, `zc_gemm` (ccsdt_t3.c); `zc_gather_vir_half`,
`zc_reduce_pair_first_rows`, `zc_pair_antisym_batch_add`,
`zc_pair_t1_contract`, `zc_add_transposed`, `zc_vvvv_dress`,
`zc_pair_one_body_add`, `zc_vsub`, `zc_vdot`, `zc_vmdot`, `zc_vlincomb`,
`zc_vzero_d`, and in blocks `zc_bgather2`, `zc_breduce2`, `zc_bgather_vir`,
`zc_breduce_ppl_rows` (t3pack.c); `zc_tn_expand`, `zc_tn_reduce`, `zc_tn_rows`,
`zc_tn_energy2`, `zc_tn_tuple_sums`, `zc_tn_zero`, and in sectors
`zc_q_sector_energy`, `zc_rows_gemm`, `zc_xv_sector` (tnpack.c).

**One change to a kernel file:** `zc_t3_pass1` and `zc_t3_pass2` of
`csrc/ccsdt_t3.c` allocated their batch buffers themselves.  The bodies are
now `zc_t3_pass1w` / `zc_t3_pass2w`, which take the buffers from the caller;
the old entry points allocate and call them (interfaces and behaviour
unchanged; the Python program runs on them).  The other kernels work on
arrays of the caller; what they allocate themselves is small (tables of
binomials, a line per thread).

### Real and complex: one source

The kernels are one C source instantiated for `double` and `double complex`
(`_d`, `_z`).  The Fortran that touches array elements or calls a kernel is
one source as well: the `.inc` files are compiled twice, by `xzcc_z.F90` and
`xzcc_d.F90`, with `xztype.h` giving the element type (`ZT`), the words per
element (`NW`), the suffix (`N(name)`) and the conjugation (`CONJ`).  Nothing
is written twice.  Everything outside the `.inc` files knows only the number
of words per element (`nwe`) and dispatches on it in four places (`xzdrv`:
iterations, XZAMPS, (Q); `xzplan`: the dry runs).

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
`XZ_REAL=0` forces the complex instance; the test of section 7 runs a real
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
- The DIIS vectors (8, each with its error vector) are in the work array if
  all of them fit next to that.  Otherwise their T3 parts are records of the
  direct-access file `XZDIIS` and are passed over in chunks (2^20 elements);
  if the disk of the run directory does not hold all of them (free space is
  asked from the system before the first iteration), fewer vectors are used -
  in the file or in the work array, whichever holds more - and with fewer
  than two the iterations run without DIIS, with a warning.  A write error
  later stops the run with a message that names the file and its size.
- (Q) has the work array to itself after the iterations: Xv for more
  occupied indices first, then the two products in fewer batches of columns,
  then more rows per step of the formation of Xv.

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
their allowed parts (section 6).  **Not done:** T3 in a mapped scratch file -
T3, its residual and (in the file-DIIS mode) nothing else of the size of T3
must fit in the work array.

Test hooks (environment): `XZ_MAXCOR=n` uses only n words of the work array;
`XZ_PLAN="nqs nbt nbp nba qxv qc22 qc13 qrows"` gives the batch sizes
(0: as planned), `XZ_DIIS=m8|f8|m0` the place (memory / file) and number of
the DIIS vectors.

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

The unit of work is a **task** (`xzpar.F90`): the quadruples i<j<k<l of one
l, or - where one l is more than a fair share - those of one l whose smallest
index i lies in a range.  A task needs Xv(x, l) only for x >= its smallest i;
it reads T2, T3 and the integrals and returns two numbers (its parts of E[Q]
and of E(Q) - E[Q]).  The serial program is the one-rank case of the same
code: the list, a loop that does the tasks of this rank, a sum.  One task
per l for one rank (no Xv is formed twice); about eight tasks per rank
otherwise (`XZ_QTASKS=n` sets the number asked for).  Consecutive tasks of
one l on a rank keep the Xv they share.

**Ranks: a static partition.**  The tasks are given to the ranks by their
estimated cost (operation counts of the products and of Xv), the largest
first, each to the rank with the least so far; every rank builds the same
list and does its own tasks in the order of l; one `MPI_Reduce` adds the
energies.  Chosen over pulling tasks from rank 0 because no rank then serves
another while it computes with all its threads, the list is fixed (which is
what restart needs), and nothing is communicated before the end.  The price:
the balance is that of the cost model.  The output gives the time of every
rank's part and "sum over ranks / slowest", the speedup that the partition
allows.

**The left side** of the correction is built in the routine of one
quadruple (`xzquad`: z from T2 with the conjugated integrals; E(Q) adds the
[V,T3] vector).  (Q)_Lambda - the same tasks with the left vectors from
lambda2, lambda3 and their own products - is **not implemented**; the task
layer, the reader, the reduction, restart, progress and sampling do not
depend on what is on the left.

**Memory per rank:** each rank has its own work array of MAXCOR words and
its own plan.  A rank holds the integrals (`<vv||vv>` on pairs, `<ov||vv>` on
pairs, the small blocks), T2 and one T3 (read-only: the dense packed T3
without labels, its blocks with labels), Xo, Xv for the occupied indices of
the plan, and the work arrays of a quadruple.  H2O / cc-pVTZ, complex, dense:
at least 2879 MB per rank, 6561 MB with the default plan (T3 370 MB, the
integrals 1.5 GB, Xv 2.9 GB, the vectors of a quadruple 0.24 GB, operands).

**Restart, progress, sampling.**  Every rank appends a line per finished
task to `XZQCHK.<rank>`.  A later run with the same task list (the same
number of ranks, or the same `XZ_QTASKS`) takes the finished tasks of all
such files as done and continues; the files are deleted when (Q) has
finished.  After every task a rank prints the task, its time, the per cent of
its own work done and the time left by the cost model.  `XZ_QSAMPLE=f`
(0 < f < 1) does the fraction f of the quadruples of every task and prints
the estimated wall time of the whole (Q) (from above: Xv is formed for every
task in full); no energy is printed and TOTENERG is not written.
`XZ_QSTOP=n` (a test hook) ends the run after n tasks of a rank.

## 6. Symmetry labels and blocks

The interface is one integer label per spinor, read from the file
**`xzlabels`** of the run directory (nothing is derived from CFOUR's files):

    # comment lines start with #
    <number of spinors> <modulus>
    <label> <label> ...          (free format, any number per line)

One label per spinor of the SCF in the order of `scf2ceig` (frozen, active
occupied, active virtual, dropped).  A label adds: `<pq||rs>` vanishes unless
l_p + l_q = l_r + l_s (modulo the modulus if it is not 0) - 2 m_j of a linear
molecule with modulus 0.  The Python side reads and writes the same file:
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
- no labels (no file) is one class: one block, the dense kernels, in the
  same step routine.  It is not a separate program.
- the labels are not trusted: while the integrals are read, the largest
  element that the labels forbid is taken over all blocks; above 1e-8 the
  program says so with the block and the element, drops the labels and reads
  the integrals again for one class.  The spinors of an SCF without the
  symmetry fail this check (the two partners of a Kramers pair are mixed):
  that is the SCF's business.

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

`XZ_LABELS=0`: the labels are not used.  **Not done:** labels with a modulus
(the block code is for labels that add without one; with a modulus the
program warns and runs dense); the rotation of unadapted Kramers pairs into
j_z eigenfunctions.

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

The first H2O case is that of `/home/xwang405/.claude/jobs/7ba4fcf7/tmp/cf_n4`
(the three reference energies given with the task).

During development: the first run of the CCSDT step agreed with the Python
energy; (Q) was off by 4.5e-8 at first, found at the term (the T2 T2 part on
one index and a triple, by its contribution to z.r per occupied quadruple
against `quadruples.Quadruples`) - a work region declared too short, which
overwrote the first Xv.  `XZ_QDEBUG=1` prints the four parts of the
correction (z.[V,T3], z.T2T2, [V,T3].[V,T3], [V,T3].T2T2).
`XZ_DEBUG=file` does one CCSDT step on amplitudes read from a file and writes
the result (for comparisons with `ccsdt_opt.update_amps` on dumped arrays).

**Labels, blocks, real** (HBr / cc-pVDZ-DK, 28 frozen; m_j-adapted spinors
written by `bench/cfour/labelled_run.py`; 6 classes, T3 17.3 % and
`<ab||cd>` 18.0 % of the dense arrays):

| integrals | program, mode | E(CCSDT) |
|---|---|---|
| real, labels | Python, blocks (`symmetry` True) | -2604.989297832537 |
| | Python, dense (labels not used) | -2604.989297832537 |
| | native, blocks, real | -2604.989297832537 |
| | native, dense, real (`XZ_LABELS=0`) | -2604.989297832537 |
| | native, blocks, complex (`XZ_REAL=0`) | -2604.989297832537 |
| | native, dense, complex (both) | -2604.989297832537 |
| complex (a phase on every spinor), labels | Python, blocks | -2604.989297832545 |
| | native, blocks (complex found) | -2604.989297832545 |
| | native, dense | -2604.989297832545 |
| CFOUR's own spinors with these labels | native | labels refused (forbidden `<oo||vv>` 0.118), dense: -2604.989297832629 |

The same files with the active spinors in a random order (`labelled_run.py
--shuffle`, seed 11: the classes are then not ranges in the files, and the
program's own ordering, with the signs of the pairs that change their order,
is what is tested): CCSDT, [Q] and (Q) as below in all four native modes.

The real case through the complex path (the override `XZ_REAL=0`) is the
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
alone from `XZAMPS`, with 20 tasks asked for, interrupted by `XZ_QSTOP=5` and
started again (serial: 5 tasks restored; 4 ranks: 10 restored): the same
energies; the checkpoint files are gone afterwards.  `XZ_QSAMPLE=0.3`: 62 of
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
| native, blocks, complex (`XZ_REAL=0`) | 0.35 | 6.9 | 0.60 GB | 463 MB |
| native, dense, real (`XZ_LABELS=0`) | 0.61 | 11.9 | 1.69 GB | 1589 MB |
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

- **The Lambda equations of CCSDT and (Q)_Lambda**: not started.  The (Q)
  task layer is ready for another left side (section 5), nothing else of it
  exists; there is no file for lambda (the hand-over would be a second file
  in the format of `XZAMPS`).
- (Q) over ranks on separate nodes was not timed (section 8); a smaller
  duplication of Xv between the ranks (tasks of one l kept on one rank where
  the balance allows it) was not tried.
- (Q) in sectors was validated at cc-pVDZ only and not timed; its memory
  plan has one knob (the occupied indices whose Xv is held), no batches of
  columns.
- Labels with a modulus; the rotation of unadapted Kramers pairs into j_z
  eigenfunctions (to get labelled, real integrals out of CFOUR's present
  SCF).
- A scheduler compiled once for both element types (section 3): today one
  source, two instances.
- T3 in a mapped scratch file; `<ab||cd>` in a file (it is in the work
  array, bare, once; the dressed operator is never stored whole).
- CCSDTQ, (P), properties, the AO-driven CCSD(T); CFOUR's nonrelativistic
  programs.
- `moinpppp` beyond 2 GB was not run (section 2); HBr / cc-pVQZ-DK (8 + 160)
  was therefore not timed.
- No input section of the ZMAT is read (`%zccsdt` of the Python front end):
  method by argument or CALC, thresholds by CC_CONV / CC_MAXCYC, everything
  else by the environment variables named above (test hooks and overrides).
