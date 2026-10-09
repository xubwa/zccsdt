# xzccsdt: the native program in CFOUR's two-component chain

`cfour/xzccsdt` is a program without Python for spinor CCSDT, CCSDT(Q) and
CCSDT(Q)_Lambda energies, for lambda of CCSDT and for the one-particle density
and orbital gradient of CCSDT that CFOUR's `xx2czvec` turns into the relaxed
density.  It runs as a member of CFOUR's X2CSO chain, after `xjoda`,
`xsox2catm`, `xsox2chf` (and `xx2camf` for X2CAMF) and `xx2cvtran` with
`ABCDTYPE=STANDARD`, in their directory:

    xjoda; xsox2catm; xsox2chf; xx2cvtran; xzccsdt

The entry point and the scheduler are Fortran in the manner of CFOUR's x2c
programs, the kernels are the C kernels of this repository (`csrc/`), and all
working memory is the work array that CFOUR's `CRAPSI` allocates
(`MEMORY_SIZE` of the ZMAT).  The algorithms are those of the Python program
(`src/zccsdt/cc/ccsdt_opt.py`, `quadruples.py`, `t3engine.py`, `lambda_opt.py`,
`lambda_blk.py`, `quadruples_lambda.py`; docs/factorization.md, docs/lambda.md),
which is the reference of every number below.

State of this document: the program at the commit that gives whole groups
of (Q) to the ranks and maps T3 for them (section 14; the density stage:
section 13; the threading of the iterations: section 12; the Lambda
equations and (Q)_Lambda: section 11).  What is measured says so (job, node);
what is not done is listed at the end.  In short: CCSDT, (Q), the Lambda
equations of CCSDT and (Q)_Lambda - dense or in the blocks of symmetry
labels, complex or real, (Q) and (Q)_Lambda over MPI ranks with restart - are
there and validated against the Python program; so is the density stage of
CCSDT with the files for `xx2czvec`, frozen and dropped spinors included
(section 13), validated against the Python hand-over of section 10 and
through `xx2czvec` and `xsox2cprop`.

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

    xzccsdt [method]

`method`: `ccsdt`, `ccsdt[q]`, `ccsdt(q)`; `(q)` or `[q]` for the correction
alone from the amplitudes of an earlier run (file `XZAMPS`); `lambda` for the
Lambda equations of CCSDT for the amplitudes of `XZAMPS` (written to
`XZLAMB`); `(q)_l` or `[q]_l` for CCSDT(Q)_Lambda from `XZAMPS` and `XZLAMB`;
`ccsdt(q)_l`, `ccsdt[q]_l` for CCSDT, Lambda and the correction in one run
(section 11); `density` for the unrelaxed density and the orbital gradient of
CCSDT from `XZAMPS` and `XZLAMB`, written as the files of `xx2czvec`;
`ccsdt_density` for CCSDT, Lambda and the density in one run (section 13).  Without the argument: `CALC` of the ZMAT as joda stored it (`CALC=CCSDT`, `CCSDT[Q]`,
`CCSDT(Q)`, `CC(n)` with `EXCITATION=3`, `CC(n-1)(n)` with `EXCITATION=4`).
Thresholds: `CC_CONV=N` - the amplitudes (|dT1| + |dT2| + |dT3| of the full
tensors) below 10^-N and the energy below 10^-(N+2), as the Python front end
reads it - and `CC_MAXCYC`; lambda: |dL1| + |dL2| + |dL3| below 10^-N.  Frozen and dropped spinors are what CFOUR left
(`DROPMO`, `%vircut`): the program takes the active spinors of the integral
files.

Exit: status 0 and CFOUR's `@CHECKOUT-I` line at the end (the line before:
`XZCCSDT finished: <method> energy done.`); every error goes through CFOUR's
`ERREX` (status 1, the `ERROR` banner), with a line `@XZCCSDT-F, ...` that
says what and, for memory, how many words are needed and how many are there.
The final energy is written to JOBARC as `TOTENERG`, as `xx2cccsd` does
(`lambda`: the CCSDT energy again).

Threads: `OMP_NUM_THREADS` (kernels and MKL share Intel's OpenMP runtime).

### MPI: (Q) over ranks

    export PATH=$PATH:/cm/shared/apps/slurm/current/bin   # in a job: mpirun needs srun (section 8)
    export OMP_NUM_THREADS=8 I_MPI_PIN_DOMAIN=omp         # threads per rank
    mpirun -np 4 -ppn 1 xzccsdt "ccsdt(q)"                # Intel MPI, inside a Slurm allocation

e.g. `sbatch -N 4 --ntasks-per-node=1 -c 8 --mem-per-cpu=3700M` (one rank
per node; `-ppn 1`) or `-N 1 --ntasks-per-node=4 -c 8` (`-ppn 4`: the case
that was run).
Only (Q) and (Q)_Lambda are parallel over ranks; the CCSDT iterations and the
Lambda equations are those of rank 0, with its threads.  Rank 0 is the CFOUR program (CRAPSI, JOBARC, the output).
The other ranks get joda's flags from it, write their output to
`xzccsdt.rNNNN`, wait for the plan of rank 0 and allocate a work array
themselves (`GETMEM`) - not `MEMORY_SIZE` but the words that (Q) takes with
that plan (section 4) -, wait until
rank 0 has the amplitudes (a blocking broadcast: with Intel MPI a waiting rank
spins on one core unless `I_MPI_WAIT_MODE=1`), then read the integral files,
T1 and T2 of `XZAMPS` themselves and map the T3 of that file (section 5: one
copy per node).  The run directory must therefore be on a file system
that all nodes see (with lambda on the left: `XZLAMB` too).  All ranks read
the same files and make the same decisions
(blocks, real or complex, the task list): nothing but the flags, the sizes of
JOBARC, the size of the plan and the final sums is communicated.

The steps can be separate invocations, as the X2CSO programs are called one
after the other: `xzccsdt ccsdt` (serial or one rank), then
`mpirun ... xzccsdt "(q)"`; or `xzccsdt ccsdt`, `xzccsdt lambda`,
`mpirun ... xzccsdt "(q)_l"`.

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
| `XZAMPS` | the amplitudes of an earlier run, for `(q)` / `[q]`, `lambda`, `(q)_l` / `[q]_l` and for the ranks other than 0 |
| `XZLAMB` | lambda of an earlier run, for `(q)_l` / `[q]_l`, `density` and for the ranks other than 0 |
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
| `XZDIIS` | only if the DIIS vectors do not all fit in the work array (section 4: all in memory or all in the file): direct-access file with the T3 parts of the vectors (of the amplitudes, then of lambda); deleted at the end |
| `XZQCHK.NNNN` | one per rank during (Q): a first line that names the calculation and the task list (kind, sizes, tasks, a number computed from the list), then a line per finished task (task, its two energies, seconds); deleted when (Q) has finished |
| `xfull`, `dpoo`, `dpvv`, `dii0`, `daa0`, `dpopo`, `dpvpv` | `density`, `ccsdt_density`: the input of CFOUR's `xx2czvec` (section 10.2: one record each, complex, CFOUR's order of the spinors) |
| `xzccsdt.rNNNN` | output of the ranks other than 0 (MPI) |

A run that reads `XZAMPS` or `XZLAMB` must run in the mode of the run that wrote it
(labels used or not, real or complex: both follow from the same integral
files, so this only matters with `XZ_LABELS` / `XZ_REAL`); the header is
checked.

**`XZAMPS` and `XZLAMB` are also what (Q) computes from**: (Q) and (Q)_Lambda
do not read T3 and lambda3 into the work array but map the two files
read-only (`mmap`; `xz_map_file` of `xzutil.c`) and read the amplitudes where
the system has them - in the page cache of the node, once for all the ranks
there (section 5).  This needs no conversion in any mode: the T3 of the file
is the stored T3 of the program byte for byte - the packed dense array, or
its blocks one after the other, real (8 bytes) or complex (16) - behind a
header of 56 bytes and T1 and T2, i.e. at an offset that is a multiple of 8.
The files must not be changed while a (Q) runs on them.  `XZ_QMAP=0` reads
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
| `xzread.F90` | readers of CFOUR's files and of `xzlabels`; checks of the labels and of the imaginary parts |
| `xzmaps.F90` | index tables of pairs and triples (grouped by classes), layout of T3 in blocks |
| `xzmem.F90` | the stack on the work array, dry runs |
| `xzplan.F90` | the memory plan |
| `xzdata.F90` | sizes, the job, offsets of the arrays that live for the run |
| `xzdiis.F90` | DIIS: slots, the file, the small linear problem (ZHEEV) |
| `xzpar.F90` | ranks and the task list of (Q) (groups, pieces, the distribution by the cost model), checkpoint files |
| `xzkern.F90`, `xzkern.inc` | explicit `bind(C)` interfaces of the kernels, one include for `_d` and `_z` |
| `xzmpi.c`, `xzutil.c` | C: the MPI calls behind a few functions (stubs without MPI); free disk space, resident set (anonymous and mapped), BLAS integer size, a file mapped read-only (`xz_map_file`: T3 of (Q)), whether a directory is on a network file system |
| `xztype.h` | the element type of a compilation of the type-generic sources |
| `xzimd.inc` | T1-dressed integrals, W and F intermediates, the CCSD part, the occupied operator |
| `xzstep.inc` | one CCSDT step: the order of the stages, the placement of every array |
| `xzblk.inc` | the T3 passes in blocks of the labels |
| `xzdiisg.inc`, `xzcc.inc` | DIIS update, the iterations, `XZAMPS` |
| `xzlam.inc`, `xzlblk.inc` | the Lambda equations: intermediates, an iteration, the residuals of rank 1 and 2, `XZLAMB`; the passes over l3 in blocks of the labels |
| `xzden.inc` | the density stage: the passes over l3 and T3, the blocks of the two-particle density, the orbital gradient, the files of `xx2czvec`; its pass C in blocks |
| `xzdfrz.inc`, `xzfrz.F90` | frozen and dropped spinors in the density stage: the contractions with integrals of one such index, the rotations, the rows and columns of `xfull`; the integrals through CFOUR's `int_mintv2e` and `vtranpsog` |
| `xzq.inc`, `xzqs.inc` | (Q) and (Q)_Lambda: the task loop and the dense quadruple; the quadruple in sectors of the labels |
| `xzcc_z.F90`, `xzcc_d.F90` | the two instances of the `.inc` files |

Kernels of `csrc/` that are called: `zc_t3_pass1w`, `zc_t3_pass2w`,
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
`zc_ql_sector_energy` (qlambda.c).

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

  With `XZ_QMAP=0` T3 is in the work array of every rank and the line says
  `(private to a rank, T3 included: XZ_QMAP=0)`.  A run that iterates first
  (`ccsdt(q)`) keeps T3 in the work array for the iterations, writes
  `XZAMPS`, releases T3 and maps the file like the other ranks.
- The work array of a rank other than 0 is not `MEMORY_SIZE`: rank 0 sends
  the words that (Q) takes with its plan, and the other ranks allocate that
  (plus 4096 words) and make the same plan in it.  `MEMORY_SIZE` (and
  `XZ_MAXCOR`) therefore bound the plan of every rank, and the memory of a
  node is ranks x the private words + the mapped files once.

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
the private copy (`XZ_QMAP=0`); with the map they are smaller by T3.  **Not
done:** T3 of the *iterations* in a mapped scratch file - T3, its residual
and (in the file-DIIS mode) nothing else of the size of T3 must fit in the
work array; the map of section 2 is read-only and serves (Q) alone.

Test hooks (environment): `XZ_MAXCOR=n` uses only n words of the work array;
`XZ_PLAN="nqs nbt nbp nba qxv qc22 qc13 qrows"` gives the batch sizes
(0: as planned), `XZ_DIIS=m8|f8|m0` the place (memory / file) and number of
the DIIS vectors (read together with `XZ_PLAN`: `XZ_PLAN="0 0 0 0 0 0 0 0"
XZ_DIIS=f8` is the planned run with the vectors in the file).  `XZ_QMAP=0`:
(Q) with a private T3 (above).  `XZ_NCYC`, `XZ_START`, `XZ_HH`: section 12;
`XZ_QTASKS`, `XZ_QRANKS`, `XZ_QSAMPLE`, `XZ_QSTOP`, `XZ_QCOST`: section 5.

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

For 26 occupied spinors (the model in blocks, 26 + 78; `XZ_QRANKS="2 3 4 6 8
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
`XZ_QCOST=file` writes the times of every quadruple and of every Xv of rank
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
number of ranks, or the same `XZ_QTASKS`) takes the finished tasks of all
such files as done and continues - the first line of a file carries a number
computed from the list, so the file of another list is not taken (with
another number of ranks the calculation starts again) -; all ranks read the
files before any of them writes one; the files are deleted when (Q) has
finished.  After every task a rank prints the task, its time, the per cent
of its own work done and the time left by the cost model.  `XZ_QSAMPLE=f`
(0 < f < 1) does the fraction f of the quadruples of every task, evenly
spread over the task (at least one), with Xv formed as in the whole
calculation, and prints the estimated wall time of the whole (Q) (per task:
the time of Xv plus the time of its quadruples in proportion; the slowest
rank); no energy is printed and TOTENERG is not written.  (Before 2026-10-09
a sample took the *first* quadruples of a task, whose sectors are not the
average ones, and formed Xv again for every task: its estimates are not
comparable - section 14.)  `XZ_QTASKS=n` asks for about n tasks of equal
numbers of quadruples instead of the rule above, `XZ_QSTOP=n` ends the run
after n tasks of a rank (test hooks of the list and of restart).

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

- Lambda: the residuals of rank 1 and 2 and the o v^3-sized operands in
  blocks of the labels (they are dense, as in the Python program: with
  labels they are most of the memory of an iteration, and were 39 % of its
  time before their loops became GEMMs - section 12: now a sixth at 26 +
  120); l3, T3 or the residual in a mapped scratch file; the Lambda
  equations with constant terms (lambda of CCSDT(Q), the equations of the
  first derivatives of (Q)_Lambda: docs/lambda.md 11, 12); the densities of
  (Q) and (Q)_Lambda (the density of CCSDT is there: section 13).
- (Q) over ranks on separate nodes was not timed (section 8).  Whole groups
  now go to the ranks and T3 is shared between the ranks of a node (sections
  5, 14).  Not done there: Xo shared in the same way (it is formed by every
  rank - 21 to 25 s per rank at 12 x 4 - and is the largest private array
  of (Q): 6.9 GB of 11.6 GB per rank for 26 + 160 in real blocks, which is
  what limits the number of ranks on a node now); Xv passed between ranks instead of formed again by the pieces of a
  cut group; the mapped T3 on a network file system (the warning and
  `XZ_QMAP=0` are there, no such run was made); a cost model with a constant
  of Xv per mode (one constant, between the two measured ones).
- (Q) in sectors was validated at cc-pVDZ only and not timed; its memory
  plan has one knob (the occupied indices whose Xv is held), no batches of
  columns.
- Labels with a modulus; the rotation of unadapted Kramers pairs into j_z
  eigenfunctions (to get labelled, real integrals out of CFOUR's present
  SCF).
- A scheduler compiled once for both element types (section 3): today one
  source, two instances.
- T3 of the iterations in a mapped scratch file ((Q) reads a mapped T3:
  section 5); `<ab||cd>` in a file (it is in the work array, bare, once; the
  dressed operator is never stored whole).  A split of the DIIS vectors
  between the work array and the file: decided against (section 4: all in
  memory, or all in the file).
- CCSDTQ, (P), properties, the AO-driven CCSD(T); CFOUR's nonrelativistic
  programs.
- The relaxed density for `xsox2cprop` (`denfinal`): the density stage and
  the files of `xx2czvec` are there for CCSDT, with frozen and dropped
  spinors (section 13).  Not done there: the contributions of (Q); the held
  stage in blocks and in real arithmetic (not validated) and its timing at
  production size; a relaxed density for labelled real spinors (`xx2czvec`
  needs the spinors of CFOUR's SCF, which have no labels); a Z-vector solver
  of our own where `xx2czvec` is wrong (section 10.4).
- The one-rank-per-node timing of (Q), labels with a modulus: as before.
- No input section of the ZMAT is read (`%zccsdt` of the Python front end):
  method by argument or CALC, thresholds by CC_CONV / CC_MAXCYC, everything
  else by the environment variables named above (test hooks and overrides).

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
conj(t), Jacobi steps with DIIS (`xzdiis.F90`, the code of the amplitudes: 8
vectors in the work array or their l3 parts in `XZDIIS`), converged when
|dL1| + |dL2| + |dL3| of the full tensors is below 10^-`CC_CONV`.  The output
has, per cycle, `1/4 sum <ab||ij> l2[ij][ab]` (the pseudo-energy that the
Python tests compare), the three norms and the time.

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
T1, T2, T3 (from `XZAMPS`), l1, l2, l3, new l1 and l2, the DIIS vectors,
then what is kept for the iterations - `GmQ` and `GPd` (the operands of pass
1: t2 with W_vvvo, t2 with W_vooo), `Bdrv` and `Adrv` (2 (o + v) rows and
columns of the drive products), `~<ej||bg>` (o v^3, both orders of the pair),
`Xq`, `X2`, W_ovvo twice, the occupied operator - and the arrays of the
setup or of an iteration, taken and released stage by stage: the batch
buffers of pass 1 (two), of pass 2 (two), of the ladder (two and a batch of
`X2`), and after the passes `Y2` on both orders of its pair (o v^3) with
three (o v)^2 arrays for the residuals of rank 1 and 2.  T3, l3 and the
residual are three regions of the size of T3 (the residual is a DIIS slot
when the vectors are in the work array).  The plan is that of section 4: a
dry run of `xzlam` after the layout (and after the dry run of the
amplitude iterations in a run that does both), the floor checked before the
large files are read, then `nbt`, `nbp`, `nba` raised one after the other,
then the place of the DIIS vectors.  A run prints "the least that Lambda
takes"; below it:

    @XZCCSDT-F, the work array is too short for this calculation.
      words needed:          2861968   words there:          2800000
      Set MEMORY_SIZE to at least          2862032 words (        22.9 MB).

| case | o, v | mode | least for Lambda | default plan, Lambda | least for (Q)_L | T3 = l3 |
|---|---|---|---|---|---|---|
| H2O / 6-31G | 10, 16 | complex, dense | 22.9 MB | 62.2 MB | 9.6 MB | 1.1 MB |
| the same, DROPMO=1>1, %vircut 1.2 | 8, 10 | complex, dense | 4.5 MB | 8.5 MB | | 0.1 MB |
| HBr / cc-pVDZ-DK, 28 frozen | 8, 28 | complex, dense | 50.7 MB | 158.8 MB | 28.8 MB | 2.9 MB |
| the same, labelled real spinors | 8, 28 | real, 6 classes | 18.6 MB | 22.1 MB | | 0.25 MB |
| | | real, dense (`XZ_LABELS=0`) | 25.5 MB | 79.6 MB | | 1.5 MB |
| | | complex, 6 classes (`XZ_REAL=0`) | 36.8 MB | 43.8 MB | | 0.5 MB |

(the cc-pVTZ cases: under Timings below.)  In blocks the floor is the dense operands,
not l3: for the labelled HBr case 18.6 MB where the three T3-sized regions
are 0.8 MB.

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
| | real, dense (`XZ_LABELS=0`) | 16 | 3e-17 | 7e-17 | 6e-18 | |
| | complex, 6 classes (`XZ_REAL=0`) | 16 | 8e-18 | 7e-17 | 7e-18 | |
| | complex, dense (both) | 16 | 1e-17 | 7e-17 | 7e-18 | |
| the same with a phase on every spinor | complex, 6 classes | 16 / 16 | 2e-17 | 7e-17 | 6e-18 | -0.13899853035992 |
| the labelled spinors in a random order (`--shuffle`, seed 11; Python dense) | real, 6 classes | 16 / 16 | 1e-17 | 7e-17 | 7e-18 | -0.13899853035651 |

(largest elements: l1 0.009 to 0.014, l2 0.054 to 0.066, l3 0.0016 to
0.0031.)  The Python Lambda of the labelled directories ran on the packed
block (below `blocks_min` elements), so the native blocks are compared with
the dense Python code.  Batches and memory: the same agreement with one
occupied pair and one triple per batch and with 2 and 3 (`XZ_PLAN`), with
the DIIS vectors in the file (`XZ_DIIS=f8`), with the work array at the floor
of the plan (H2O / 6-31G: 2861976 words, one pair per batch, 12 triples, DIIS
in the file) and at 3.5 million words; without DIIS (`XZ_DIIS=m0`) 27 cycles
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
complex (`XZ_LABELS`, `XZ_REAL`); Xv for 3 of 7 occupied indices with the
products in batches of 20 and 300 columns, and the work array at the floor of
(Q)_L (HBr dense: 3594976 words); the MPI build with 1, 2 and 4 ranks from
the files and with 4 ranks for `ccsdt(q)_l` (H2O, HBr dense, the labelled case
real and with phases: 5 to 25 tasks); `(q)_l` with 20 tasks asked for,
stopped by `XZ_QSTOP` after 5 tasks (serial) or 3 (4 ranks) and started again
(5 and 3 tasks restored; the checkpoint files gone afterwards); `(q)_l` after
an interrupted `(q)`, whose checkpoint is not taken.  `[q]_l` writes the
fifth-order energy to `TOTENERG`.  `XZ_QSAMPLE=0.3`: 21 of 70 quadruples
(HBr), the estimate printed, no `TOTENERG`.  CCSDT, [Q] and (Q) of the six
cases after the changes: the numbers of section 7.

## 12. Threads: where an iteration goes, and what was changed

Measured on HBr with 10 frozen spinors (26 occupied spinors), m_j-adapted
real spinors with labels written from Python (`bench/cfour/labelled_run.py`;
120 virtual spinors: the 120 of lowest energy of cc-pVQZ-DK), whole nodes
(`sbatch -p parallel -N 1 -n 1 -c 48 --mem=180G`, two sockets of Xeon Gold
6248R, `MEMORY_SIZE=150 GB`), three cycles per run (`XZ_NCYC=3`: nothing
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
  (`xzwpair`, `xzhhapply`; `XZ_HH=pair`): T3 gathered by occupied pair in
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
same with `XZ_HH=pair`; one pair and one triple per batch, 2 and 3, the DIIS
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

**Other additions.**  `XZ_NCYC=n`: n cycles of every iteration, then on as
if converged, with a warning (timing runs).  `XZ_START=ccsd`: T1 and T2 of
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
written (`XZ_DEN_ACTIVE=1`, a test hook: xfull, dpoo, dpvv of the active
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
1, 2 and 3 pairs and triples (`XZ_PLAN`): the same files.

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
| | real, dense (`XZ_LABELS=0`) | 2.0e-17 | 5.6e-17 | 2.2e-18 |
| | complex, 6 classes (`XZ_REAL=0`) | 1.9e-17 | 5.6e-17 | 6.9e-18 |
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
cycles each (`XZ_NCYC=3`), the density stage on the amplitudes and lambda
of those cycles, active part (`XZ_DEN_ACTIVE=1`: the directory has no
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
repeated Xv".  Those were *estimates of a 0.5 % sample* (`XZ_QSAMPLE`), and
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

**The sample** (`XZ_QSAMPLE=0.005`, 74 of 14950 quadruples; each run twice):
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
cases with 1 to 4 ranks, with `XZ_QTASKS=20`, interrupted (`XZ_QSTOP`) and
restarted with the same and - the files then being of another list and not
taken - with another number of ranks (`tq.sh`).

**The model was measured** with `XZ_QCOST` (the wall time of every
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
and in one run with the iterations, against `XZ_QMAP=0`; identical to 12
decimals).  The plan of a rank (26 + 160, real blocks):

| | old / `XZ_QMAP=0` | mapped |
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
| | new, `XZ_QMAP=0` (sample) | 47523 | 12017 | 21 | 15113 |
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
section 2 and `XZ_QMAP=0` exist; `xz_fs_local` was not exercised on such a
directory); more than one node.

### (A) The DIIS vectors in the work array or in the file

`XZ_NCYC=6`, the new binary, `XZ_PLAN="0 0 0 0 0 0 0 0"` with `XZ_DIIS=m8`
(all in the work array) or `f8` (the T3 parts in `XZDIIS`); seconds per
cycle, mean of six and the sixth (5 vectors in the subspace by then; the
runs in blocks and dense on 48 threads were made twice):

| case | threads | in memory: mean, 6th | in the file: mean, 6th | DIIS stage, 6 cycles: memory, file | work array: memory, file |
|---|---|---|---|---|---|
| 26 + 160, real blocks (T3 1.6 GB) | 24 | 11.50, 11.60 (11.49, 11.60) | 14.17, 16.14 (14.19, 16.41) | 1.9, 17.0 s | 32.8, 9.0 GB |
| | 48 | 9.40, 9.82 (9.28, 9.11) | 12.96, 15.67 (12.90, 15.82) | 1.5, 19.4 s | |
| 26 + 120, real dense (`XZ_LABELS=0`; T3 5.8 GB) | 24 | 70.55, 71.77 | 80.11, 88.37 | 5.4, 60.1 s | 103.5, 15.9 GB |
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
x 2 ranks x threads, mapped; (Q) with `XZ_QMAP=0`, serial and 4 x 6;
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
- The estimate of `XZ_QSAMPLE` is an upper estimate (1 to 36 % high here).
