# The native program as a program directory of CFOUR

`cfour/xzccsdt/` (docs/native.md) is built in this repository by its own
Makefile against a built CFOUR.  This kit turns it into one more program
directory of CFOUR's source tree (the two-component development version),
built by CFOUR's `make.config` and `make.rules` as `x2cccsd` or `x2czvec`
are.  The directory is generated; the sources stay here.

| file | what |
|---|---|
| `assemble.sh` | writes the directory from the repository as it is |
| `regress.sh` | two binaries on the regression cases, compared |

## The merge

1. Assemble into the CFOUR source tree (`NAME`: the directory, default
   `xzccsdt`; the program is `x<NAME>` unless `--prog` says otherwise):

       cfour/merge/assemble.sh --cfour <cfour> --out <cfour>/xzccsdt [--name xzccsdt] [--prog xxzccsdt]

2. Add the name to the list of program directories in `<cfour>/Makefile.in`
   (and in `<cfour>/Makefile` of a tree that is already configured) - the one
   line of CFOUR that changes:

       sox2chfcd x2ciimcd x2ceomccdencd x2cmp xzccsdt

3. Build as any directory of CFOUR: `./configure ...; make` at the top, or
   `make` in `<cfour>/xzccsdt` of a built tree (it needs `lib/libx2cint.a`,
   `lib/liblibr.a` and the modules of `libx2c`).  Result: `bin/xxzccsdt`.
   For the program over MPI ranks, in the directory:

       make XZMPI=1 MPICC=mpiicc          # -> bin/xxzccsdt_mpi, beside the serial one

4. Check (in a job; reference binaries: `make` and `make MPI=1` in
   `cfour/xzccsdt` of this repository):

       cfour/merge/regress.sh --ref build/xzccsdt/xzccsdt     --new <cfour>/bin/xxzccsdt --eom
       cfour/merge/regress.sh --ref build/xzccsdt_mpi/xzccsdt --new <cfour>/bin/xxzccsdt_mpi --np 2

5. When this repository changes: run `assemble.sh` again with the same
   arguments (it empties the directory, which it recognises by its file
   `.assembled_from_zccsdt`, and writes it anew - new source files are found
   by themselves), `make`, `regress.sh`.  Nothing is edited in CFOUR's copy;
   its `README` names the commit it was made from.

## What the directory is

- Every file of `cfour/xzccsdt/` except its Makefile (a glob).  `make.rules`
  compiles each `*.f`, `*.f90`, `*.f03`, `*.c` of a directory and has no rule
  for `%.F90`: the `.F90` files are copied as `.f90`.  CFOUR's `FCFLAGS`
  preprocess every Fortran file (`-cpp`), so nothing else is needed for the
  `#include` of the typed sources (`*.inc`, `xztype.h`; `-I.` is added).
- The C kernels of `csrc/` that the program links: `KERNELS` of
  `cfour/xzccsdt/Makefile`, read at assembly time (now `ccsdt_t3.c`,
  `t3pack.c`, `tnpack.c`, `lambda_t3.c`, `qlambda.c`).
- The routines of other program directories of CFOUR that the program
  calls: every variable of `cfour/xzccsdt/Makefile` of the form
  `X := $(addprefix $(CFOUR)/<directory>/, a.o b.o)`, read at assembly time -
  `FRZOBJ` (`x2czvec/vtranpsog.f90`, `vtran1sparse.f90`, `expnd2p.f90`: the
  density stage) and, with the EOM sources, `EOMOBJ`
  (`x2ceomcc/eomreorder.f90`, `eomconv.f90`).  The repository links their
  objects from a built CFOUR; here their sources are copied from `--cfour`
  into the directory, as CFOUR itself keeps copies of a routine in the
  directories that use it.  They are not stored in this repository.  If
  CFOUR changes them, assemble again.
- A `Makefile` of CFOUR's form (`topdir`, `builddir`, `make.config`,
  `INST_PROGS`, `make.rules`, `libs`, the link rule with `$(FC) $(FLDFLAGS)`
  and `$(BLAS_LIBS)`), with
  - `FCFLAGS+=-qopenmp $(XZ_ARCH) -I. -I../libx2c`, `FLDFLAGS+=-qopenmp`
    (`make.rules` removes `-fopenmp` for programs outside its
    `openmp_whitelist`; the Makefile adds the flag after the include, as
    `x2cccsd/Makefile` does, so the whitelist need not change);
  - `CFLAGS+=-O3 $(XZ_ARCH) -std=c99 -qopenmp -complex-limited-range
    -DZC_USE_MKL -DMKL_ILP64` and `-I$(MKLROOT)/include` if the environment
    has `MKLROOT` (`make.config` has no variable for MKL; CFOUR finds it
    through the compiler's environment).  CFOUR's own `CFLAGS` have no
    optimisation level;
  - `XZ_ARCH?=-xHost`, the instruction set, as the repository's Makefile
    (`make XZ_ARCH=` builds without);
  - `libs:= -lx2cint -llibr`;
  - the order of compilation at its end: per object the files it includes
    and the objects of the modules it uses.  CFOUR's `config/mkdep90.py`
    (`.f90deps`) reads `use` only in the compiled file itself, and here most
    of them are in the includes.
- A `README` (what the program is, the commit, how to regenerate, the origin
  of every file).

No source of the repository had to be changed for CFOUR's rules.

## MPI

The serial program is what `make` builds, in every configuration of CFOUR.
The program over ranks is an opt-in of the directory's Makefile and needs
nothing of `make.config` beyond what the serial one uses:

    make XZMPI=1 MPICC=mpiicc

- `xzmpi.c`, the one file with MPI calls, is compiled once more by `$(MPICC)`
  with `-DXZ_MPI` into `xzmpi_mpi.o`; all other objects are those of the
  serial program (the Fortran side has no MPI, no `mpif.h`, no module).
- The link is CFOUR's (`$(FC) $(FLDFLAGS) ... $(BLAS_LIBS)`) plus
  `$(XZ_MPILIBS)`: the `-L`, `-l` and `-Wl,` words of what the wrapper says
  it links with (`$(MPICC) --showme:link`, else `$(MPICC) -show`).
- The program is installed as `x<NAME>_mpi`, beside the serial one.

Intel MPI (tested): `MPICC=mpiicc`; `mpiicc -show` gives
`-L<impi>/lib/release -L<impi>/lib -lmpifort -lmpi -ldl -lrt -lpthread`.
(`mpicc` of the oneAPI environment is Intel MPI's wrapper of gcc, which does
not take CFOUR's icc flags.)

Open MPI (not run in this form): `MPICC=mpicc` with `OMPI_CC=icc` if the
wrapper was built for another compiler - the C flags are icc's -;
`mpicc --showme:link` gives the link words (`-L<ompi>/lib -lmpi` and
`-Wl,-rpath` words); or `XZ_MPILIBS="-L<ompi>/lib -lmpi"` by hand.  Run with
that Open MPI's `mpirun` (docs/native.md 15.9).

Why not CFOUR's MPI configuration: `configure --enable-mpi=<openmpi|lam|mpich|scali>`
sets `MPI_TYPE`, replaces `CC`, `FC`, `CXX` by the MPI wrappers for the whole
of CFOUR and compiles everything with `-DMPI` (then `initenv` of libr calls
`MPI_INIT` and changes into a directory per rank); `make.config` gets no MPI
compile or link variables.  It is not a configuration in use, and this
program needs MPI in one C file only.

## What the directory needs from CFOUR

- `lib/libx2cint.a` (`int_mintv2e` and what it calls), `lib/liblibr.a`
  (`CRAPSI`, `CRAPSO`, `GETREC`, `PUTREC`, `GETMEM`, `ERREX`, `initenv`,
  `walltime`, ...);
- the module files of `libx2c` (`-I../libx2c`: `cvwint_module`, `kinds`,
  `reals`, `complexs`);
- copied at assembly: `x2czvec/vtranpsog.f90`, `vtran1sparse.f90`,
  `expnd2p.f90`; with the EOM sources `x2ceomcc/eomreorder.f90`,
  `eomconv.f90`;
- `make.config`: `FC`, `CC`, `FCFLAGS` with `-cpp -i8 -assume byterecl`,
  `FLDFLAGS`, `CFLAGS`, `BLAS_LIBS` with 8-byte integers
  (`-lmkl_intel_ilp64 ...`: the kernels are compiled with `-DMKL_ILP64` and
  the program stops at start if the two do not agree); `make.rules`
  unchanged.

## Tested

In a private copy of a configured and built tree (`~/apps/cfour-x2cso-rebuild`
copied to `/scratch16/lcheng24/xwang405/cfour_merge_test/`; neither that tree
nor `~/apps/cfour-x2cso` was written to): `make` in the directory, `make
XZMPI=1 MPICC=mpiicc`, and `make` at the top level.  ifort / icc 2021.6.0
(oneAPI 2022.1, classic compilers), MKL 2022.1 ILP64, Intel MPI 2021.6, GNU
make 4.2.1; `MPI_TYPE=none`.  Results: section "Results" below.

Not tested: gfortran / gcc (the flags of the Makefile are Intel's, as those
of CFOUR's x2c directories: `-qopenmp`; the C flags `-xHost`,
`-complex-limited-range` would need their GNU forms), ifx / icx, Open MPI
through this Makefile, a CFOUR configured with `--enable-mpi`.

## Open points

- **The name.**  Directory `xzccsdt` gives the program `xxzccsdt` by CFOUR's
  convention (`--name`, `--prog`).  The repository's own build, its
  documentation, the output files of the ranks (`xzccsdt.rNNNN`) and the
  Python front end (`python -m zccsdt.cfour --chain`) say `xzccsdt`.
- **`-xHost`** (`XZ_ARCH`): CFOUR's flags name no instruction set; the
  timings of docs/native.md are with it.  A binary built with it runs only on
  processors that have the instruction set of the build host.
- **Messages of the build that are CFOUR's**: `-extend_source ... deprecated`
  (its `FCFLAGS`), and `Circular ../xzccsdt/xzplan.o <- ../xzccsdt/xzplan.o
  dependency dropped` (`mkdep90.py` writes a dependency of a file on itself
  when it uses a module that it defines); both harmless.
- **`make clean`** without `XZMPI=1` leaves `bin/x<NAME>_mpi` (CFOUR's rule
  removes `INST_PROGS`).
- **A CFOUR configured with `--enable-mpi`** (read, not run): the serial
  program would work as every CFOUR module does there; `XZMPI=1` would
  initialise MPI twice (`xzmpi_init`, then `MPI_INIT` in libr's
  `initenv('MODULE')` on rank 0) - `xzccsdt.f` would have to leave out the
  call of `initenv` in that case.
- GENBAS and the driver (`xcfour`, the scripts that call the chain) are not
  part of this: the program is called by hand or by the front end.
- Nothing in `make.rules`, `make.config.in` or `configure.ac` has to change.
  Optional: the program's name in `openmp_whitelist` of `make.rules.in`.

## Results

2026-10-10, node c510 (8 cores), `regress.sh`; reference = the binaries of
`cfour/xzccsdt/Makefile` (`make`, `make MPI=1`) of the same sources, built
in the same session.  Run twice: on this branch's base (d199087, before the
EOM sources) and on master c15f493 (with EOM: 32 files of `cfour/xzccsdt`,
5 kernels, 5 routines of CFOUR), the second from an export of master
(`git archive`), nothing merged.  Numbers below: master c15f493.

Build with CFOUR's make in the private copy: `make` in the directory and
`make` at the top level (which compiled this directory only) - exit 0,
`bin/xxzccsdt`, no MPI library in it; `make XZMPI=1 MPICC=mpiicc` - exit 0,
`bin/xxzccsdt_mpi` (libmpi, libmpifort); also with `make.rules` as
`make.rules.in` has it (`VPATH` with `$(srcdir)`; the built tree's lacks it).

Per case and binary: `ccsdt(q)_l`, `(q)`, `density`, serial also `eom`.
Compared with the reference binary: exit status, mode, cycles, every energy
as printed (12 decimals), the excitation energies (8 decimals), the seven
density files.

| binaries | runs | printed lines | differ | density files: byte-identical / all | largest difference of an element |
|---|---|---|---|---|---|
| `xxzccsdt` vs repository, serial (with `eom`) | 28 + 28, all exit 0 | 307 | 0 | 26 / 49 | 5.6e-17 |
| `xxzccsdt_mpi` vs repository `MPI=1`, 2 ranks | 21 + 21, all exit 0 | 203 | 0 | 25 / 49 | 5.6e-17 |
| `xxzccsdt_mpi` vs repository `MPI=1`, 4 ranks | 21 + 21, all exit 0 | 203 | 0 | 21 / 49 | 5.6e-17 |

(Differences of 1e-17 in elements of 1e-2: taken to be the order of sums
over threads; one binary run twice was not compared.)  The printed lines of the serial program and of 2
and 4 ranks are identical too.

| case | mode | E(CCSDT) | E(CCSDT(Q)) | E(CCSDT(Q)_L) | lowest omega (`eom`) |
|---|---|---|---|---|---|
| `h2o_631g` (10 + 16) | complex, dense | -76.169208460811 | -76.169665313369 | -76.169666381377 | 0.28437865 |
| `hbr_fc` (8 + 28, DROPMO) | complex, dense | -2604.989297832626 | -2604.989515284324 | -2604.989509911699 | 0.23302866 |
| `hbr_lab` (`xzlabels`) | real, 6 classes | -2604.989297832497 | -2604.989515284195 | -2604.989509911570 | 0.23302866 |
| `hbr_ph` (`xzlabels`, phases) | complex, 6 classes | -2604.989297832495 | -2604.989515284193 | -2604.989509911568 | 0.73790600 (shift 6) |
| `hbr_dglin_fc` (`%dglin`: `dgcosym`) | real, 6 classes | -2604.989297832622 | -2604.989515284320 | -2604.989509911695 | 0.23302866 (12 roots) |
| `hbr_dglin_fc_vc` (`%vircut`, 8 + 22) | real, 6 classes | -2604.979399715844 | -2604.979617028076 | -2604.979611696038 | 0.23708252 (9 roots) |
| `hbrp_dglin_fc` (HBr+, 7 + 29) | real, 6 classes | -2604.578962949237 | -2604.579086210195 | -2604.579082911749 | 0.01183843 (13 roots) |

Scripts and logs of these runs: `~/.claude/jobs/7ba4fcf7/tmp/mergekit/`.
