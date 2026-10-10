#!/bin/bash
#
# assemble.sh: the native program cfour/xzccsdt as one program directory of
# CFOUR (two-component development version), built by CFOUR's own make.config
# and make.rules.
#
#   cfour/merge/assemble.sh --cfour CFOUR_SOURCE_TREE [--name NAME] [--prog PROG] [--out DIR]
#
#   --cfour DIR   a CFOUR source tree: the routines that the density stage takes
#                 from other program directories (FRZOBJ, EOMOBJ of cfour/xzccsdt/Makefile:
#                 objects of x2czvec, x2ceomcc) are copied from there.  Read only, unless
#                 --out points into it.
#   --name NAME   the name of the directory (default xzccsdt)
#   --prog PROG   the name of the installed program (default x<NAME>, CFOUR's
#                 convention: x2cccsd -> xx2cccsd)
#   --out DIR     the directory to create (default ./<NAME>); for the merge:
#                 --out CFOUR_SOURCE_TREE/<NAME>
#
# What is assembled is the repository as it is now: every source of
# cfour/xzccsdt/ (a glob, no list), the C kernels that its Makefile names
# (KERNELS) from csrc/, the routines of CFOUR that it names (objects below $(CFOUR)), a
# Makefile in CFOUR's style and a README.  Nothing of CFOUR is stored in this
# repository.  See cfour/merge/README.md.
#
set -eu

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
src=$repo/cfour/xzccsdt
csrc=$repo/csrc

name=xzccsdt
prog=
out=
cfour=
while [ $# -gt 0 ]; do
  case $1 in
    --name)  name=$2;  shift 2 ;;
    --prog)  prog=$2;  shift 2 ;;
    --out)   out=$2;   shift 2 ;;
    --cfour) cfour=$2; shift 2 ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "assemble.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$prog" ] || prog=x$name
[ -n "$out" ] || out=$PWD/$name
die() { echo "assemble.sh: $*" >&2; exit 1; }
[ -n "$cfour" ] || die "--cfour CFOUR_SOURCE_TREE is needed (routines of CFOUR are taken from it)"
[ -d "$cfour" ] || die "no directory $cfour"
[ -f "$src/Makefile" ] || die "no $src/Makefile"
case $name in *[!A-Za-z0-9_]*|'') die "the name '$name' is not a plain directory name" ;; esac

marker=.assembled_from_zccsdt

# ---- the output directory: new, or one that this script made before -------
if [ -e "$out" ]; then
  [ -d "$out" ] || die "$out exists and is not a directory"
  if [ -n "$(ls -A "$out")" ]; then
    [ -f "$out/$marker" ] || die "$out exists, is not empty and was not made by this script"
    # all of it is generated: empty it (objects and modules of a build too)
    find "$out" -mindepth 1 -maxdepth 1 -type f -delete
    [ -z "$(ls -A "$out")" ] || die "$out has subdirectories; remove them by hand"
  fi
else
  mkdir -p "$out"
fi
out=$(cd "$out" && pwd)
: > "$out/$marker"

manifest=$(mktemp)
trap 'rm -f "$manifest"' EXIT

# put FROM TO ORIGIN: copy one file, never over another
put() {
  local to=$out/$2
  # CFOUR's rules make one object per base name, whatever the suffix
  local base=${2%.*} f
  for f in "$out/$base".*; do
    [ -e "$f" ] || continue
    case ${f##*.} in
      f|F|f90|F90|f03|F03|c|cxx)
        case ${2##*.} in
          f|F|f90|F90|f03|F03|c|cxx) die "two sources for the object $base.o: $(basename "$f") and $2 ($3)" ;;
        esac ;;
    esac
  done
  [ ! -e "$to" ] || die "the file $2 would be written twice ($3)"
  cp -p "$1" "$to"
  printf '%-22s %s\n' "$2" "$3" >> "$manifest"
}

# ---- 1. the program's own sources: everything in cfour/xzccsdt -------------
# make.rules compiles *.f *.f90 *.f03 *.c (found with find -iname) and has
# pattern rules for %.f %.F %.f90 %.f03 %.c, none for %.F90: the free-form
# sources that need the preprocessor are copied as .f90.  CFOUR's FCFLAGS
# preprocess every Fortran file (-cpp), as the repository's Makefile does.
nsrc=0
for f in "$src"/*; do
  [ -f "$f" ] || continue
  b=$(basename "$f")
  case $b in
    Makefile|makefile|GNUmakefile|README*|*.o|*.mod|*.d|.*) continue ;;
    *.F90) put "$f" "${b%.F90}.f90" "cfour/xzccsdt/$b" ;;
    *.F03) put "$f" "${b%.F03}.f03" "cfour/xzccsdt/$b" ;;
    *.f|*.F|*.f90|*.f03|*.c|*.h|*.inc) put "$f" "$b" "cfour/xzccsdt/$b" ;;
    *) echo "assemble.sh: note: $b has a suffix that CFOUR's rules do not compile; copied as it is" >&2
       put "$f" "$b" "cfour/xzccsdt/$b" ;;
  esac
  nsrc=$((nsrc + 1))
done

# the words of a variable of the repository's Makefile (continuation lines joined)
mkvar() {
  sed -e ':a' -e '/\\$/{N;s/\\\n/ /;ba' -e '}' "$src/Makefile" |
    sed -n "s/^$1[[:space:]]*[:?]\{0,1\}=[[:space:]]*//p" | head -1
}

# ---- 2. the C kernels of csrc/ that the program links -----------------------
kernels=$(mkvar KERNELS)
[ -n "$kernels" ] || die "no KERNELS in $src/Makefile"
# a kernel file and what it includes from csrc/
putc() {
  local f=$1 h
  [ -f "$csrc/$f" ] || die "csrc/$f (KERNELS, or included by a kernel) is not there"
  [ -e "$out/$f" ] && return 0
  put "$csrc/$f" "$f" "csrc/$f"
  for h in $(sed -n 's/^[[:space:]]*#[[:space:]]*include[[:space:]]*"\([^"]*\)".*/\1/p' "$csrc/$f"); do
    [ -f "$csrc/$h" ] && putc "$h"
  done
  return 0
}
for k in $kernels; do putc "$k.c"; done

# ---- 3. routines of other program directories of CFOUR ----------------------
# The repository links objects of a built CFOUR: every variable of its
# Makefile of the form  X := $(addprefix $(CFOUR)/<directory>/, a.o b.o ...)
# (FRZOBJ: x2czvec; EOMOBJ: x2ceomcc).  In CFOUR a program directory has its
# own copy of such routines (dgoccinp.f90 is in nine directories): their
# sources are copied from the tree given with --cfour.
frzlist=
cfvars=$(mktemp)
sed -e ':a' -e '/\\$/{N;s/\\\n/ /;ba' -e '}' "$src/Makefile" |
  sed -n 's/^\([A-Za-z0-9_]*\)[[:space:]]*[:?]\{0,1\}=.*\$(CFOUR)\/\([A-Za-z0-9_]*\)\/[,[:space:]]*\(.*\)$/\1 \2 \3/p' > "$cfvars"
while read -r var frzdir frz; do
  n=0
  for w in $(echo "$frz" | tr ',()' '   '); do
    case $w in
      *.o)
        r=$(basename "$w" .o); found=
        for s in f90 f F f03 F90; do
          if [ -f "$cfour/$frzdir/$r.$s" ]; then
            t=$r.$s; [ $s = F90 ] && t=$r.f90
            put "$cfour/$frzdir/$r.$s" "$t" "CFOUR $frzdir/$r.$s"
            found=1; n=$((n + 1)); frzlist="$frzlist $frzdir/$r.$s"; break
          fi
        done
        [ -n "$found" ] || die "no source of $r in $cfour/$frzdir ($var of $src/Makefile)"
        ;;
    esac
  done
  [ $n -gt 0 ] || die "$var of $src/Makefile names $cfour/$frzdir but no objects: $frz"
done < "$cfvars"
rm -f "$cfvars"

# ---- 4. the order of compilation -------------------------------------------
# CFOUR's config/mkdep90.py (.f90deps) reads "use" only in the compiled files
# themselves; here most of them are in the typed includes.  The Makefile gets
# the full list: per object, the files it includes and the objects of the
# modules it uses.
incs_of() {   # the files of $out that FILE includes, transitively (one per line)
  local f=$1 h
  for h in $(sed -n -e 's/^[[:space:]]*#[[:space:]]*include[[:space:]]*"\([^"]*\)".*/\1/p' \
                    -e "s/^[[:space:]]*[Ii][Nn][Cc][Ll][Uu][Dd][Ee][[:space:]]*['\"]\([^'\"]*\)['\"].*/\1/p" "$out/$f"); do
    [ -f "$out/$h" ] || continue
    case " $seen " in *" $h "*) continue ;; esac
    seen="$seen $h"
    incs_of "$h"
  done
}
modtab=$(mktemp)
for f in "$out"/*.f "$out"/*.F "$out"/*.f90 "$out"/*.f03; do
  [ -f "$f" ] || continue
  seen=; incs_of "$(basename "$f")"
  for g in $(basename "$f") $seen; do
    sed -n 's/^[[:space:]]*[Mm][Oo][Dd][Uu][Ll][Ee][[:space:]][[:space:]]*\([A-Za-z0-9_]*\)[[:space:]]*\(!.*\)\{0,1\}$/\1/p' "$out/$g" |
      tr 'A-Z' 'a-z' | grep -v '^procedure$' | sed "s/\$/ $(basename "${f%.*}")/" >> "$modtab" || true
  done
done
deps=
for f in "$out"/*.f "$out"/*.F "$out"/*.f90 "$out"/*.f03; do
  [ -f "$f" ] || continue
  b=$(basename "$f"); o=${b%.*}
  seen=; incs_of "$b"; incs=$seen
  objs=
  for m in $(for g in $b $incs; do
               sed -n 's/^[[:space:]]*[Uu][Ss][Ee][[:space:]][[:space:]]*\([A-Za-z0-9_]*\).*/\1/p' "$out/$g"
             done | tr 'A-Z' 'a-z' | sort -u); do
    p=$(awk -v m="$m" '$1 == m { print $2; exit }' "$modtab")
    [ -n "$p" ] && [ "$p" != "$o" ] && objs="$objs \$(o)/$p.o"
  done
  [ -n "$incs$objs" ] || continue
  deps="$deps\$(o)/$o.o:$incs$objs
"
done
rm -f "$modtab"
for f in "$out"/*.c; do
  [ -f "$f" ] || continue
  b=$(basename "$f")
  seen=; incs_of "$b"
  [ -n "$seen" ] && deps="$deps\$(o)/${b%.c}.o:$seen
"
done

# ---- 5. the Makefile --------------------------------------------------------
# (an export of the repository without .git: a file COMMIT at its top)
commit=$(git -C "$repo" rev-parse HEAD 2>/dev/null || cat "$repo/COMMIT" 2>/dev/null || echo unknown)
if [ -n "$(git -C "$repo" status --porcelain -- cfour/xzccsdt csrc 2>/dev/null)" ]; then
  commit="$commit (with uncommitted changes in cfour/xzccsdt or csrc)"
fi
today=$(date +%Y-%m-%d)

cat > "$out/Makefile" <<EOF
#
# $name: spinor CCSDT, CCSDT(Q), Lambda, (Q)_Lambda and the CCSDT density on
# the files of the two-component (X2CSO) programs.
#
# Generated by cfour/merge/assemble.sh of zccsdt ($today), commit
# $commit
# Do not edit here: change the repository and assemble again (README).
#
#DEBUG=1
#PROF=1
#RANGE=1

topdir:=..
builddir:=..

include \$(builddir)/make.config

INST_PROGS:=$prog
# the program over MPI ranks, a second program, only when asked for:
#     make XZMPI=1 MPICC=mpiicc          (Intel MPI; MPICC=mpicc for Open MPI)
# All MPI calls are in xzmpi.c (C; the Fortran side has none), which is stubs
# without -DXZ_MPI.  CFOUR's own MPI configuration (--enable-mpi: MPI_TYPE,
# the wrappers as CC and FC) is not used: $prog is the serial program in
# every configuration.
ifeq (\$(XZMPI),1)
INST_PROGS+=${prog}_mpi
endif
INST_LIBS:=
INST_INCLUDES:=
INST_DATA:=
# module names are listed in smallcaps and wihtout any .mod
INST_MODULES:=


include \$(builddir)/make.rules

# OpenMP (make.rules takes -fopenmp away from programs that are not in its
# openmp_whitelist), the typed includes of this directory, the modules of libx2c
# XZ_ARCH: the instruction set of the kernels and of the scheduler; CFOUR's
# flags name none.  make XZ_ARCH= builds without.
XZ_ARCH?=-xHost
FCFLAGS+=-qopenmp \$(XZ_ARCH) -I. -I../libx2c
FLDFLAGS+=-qopenmp

# the C kernels: optimised (CFOUR's CFLAGS have no -O), BLAS through MKL's
# cblas with 8-byte integers, as the Fortran side (-i8, mkl_intel_ilp64)
CFLAGS+=-O3 \$(XZ_ARCH) -std=c99 -qopenmp -complex-limited-range -DZC_USE_MKL -DMKL_ILP64
# mkl_cblas.h: make.config has no variable for MKL; the compiler finds it
# through its environment (CPATH), or below \$MKLROOT if that is set
ifneq (\$(MKLROOT),)
CFLAGS+=-I\$(MKLROOT)/include
endif

libs:= -lx2cint -llibr

all: \$(INST_PROGS) \$(INST_LIBS)


\$(bindir)/$prog: \$(objs) \$(libs)
	\$(FC) \$(FLDFLAGS) -o \$@ \$(objs) \$(libs) \$(BLAS_LIBS)

# XZMPI=1: the same objects, with xzmpi.c compiled once more by the MPI C
# wrapper (xzmpi_mpi.o), linked by CFOUR's Fortran compiler with the MPI C
# libraries.  XZ_MPILIBS: the -L, -l and -Wl words of what the wrapper says it
# links with (Open MPI: --showme:link; MPICH, Intel MPI: -show); set it by
# hand for a wrapper that answers neither.
MPICC?=mpicc
ifeq (\$(XZMPI),1)
xz_comma:=,
xz_mpishow:=\$(shell \$(MPICC) --showme:link 2>/dev/null || \$(MPICC) -show 2>/dev/null)
XZ_MPILIBS?=\$(filter -L% -l% -Wl\$(xz_comma)%,\$(xz_mpishow))
endif

xzmpi_mpi.o: xzmpi.c
	\$(MPICC) \$(CFLAGS) -DXZ_MPI -c \$< -o \$@

\$(bindir)/${prog}_mpi: \$(objs) xzmpi_mpi.o \$(libs)
	\$(FC) \$(FLDFLAGS) -o \$@ \$(filter-out %/xzmpi.o,\$(objs)) xzmpi_mpi.o \$(libs) \$(BLAS_LIBS) \$(XZ_MPILIBS)

# the order of compilation: what an object includes, and the objects of the
# modules it uses (config/mkdep90.py does not look into the includes)
o:=\$(subst \$(topdir),\$(builddir),\$(srcdir))
$deps
EOF

# ---- 6. the README of the directory -----------------------------------------
{
  cat <<EOF
$name - spinor CCSDT in CFOUR's two-component chain
=====================================================

The program $prog runs after xjoda, xsox2catm, xsox2chf and xx2cvtran
(ABCDTYPE=STANDARD) in their directory, on their integral files:

    $prog [method]     ccsdt, ccsdt(q), (q), lambda, (q)_l, ccsdt(q)_l, density,
                       ccsdt_density, ... (docs/native.md, section 1; the
                       header of xzccsdt.f); without the argument what CALC
                       of the ZMAT asks for

CCSDT, the (Q) and [Q] corrections, the Lambda equations, (Q)_Lambda, the
unrelaxed CCSDT density with the orbital gradient as the input files of
xx2czvec, and what the repository has added since.  A Fortran scheduler on CFOUR's work array (CRAPSI) with C kernels;
real or complex arithmetic and the blocks of the symmetry labels (dgcosym of
xsox2chf with %dglin) are chosen at run time.

make builds the serial program $prog.  The program over MPI ranks (the
iterations and (Q); mpirun -np N ${prog}_mpi ...) is a second program, built
only when asked for:

    make XZMPI=1 MPICC=mpiicc      (Intel MPI; MPICC=mpicc for Open MPI)

All MPI calls are in xzmpi.c, which is stubs without -DXZ_MPI; the Fortran
side has no MPI.  CFOUR's own MPI configuration (configure --enable-mpi) is
not used: it replaces CC and FC for all of CFOUR and makes libr's initenv
call MPI_INIT, and it is not a configuration in use.

This directory is GENERATED.  Its sources are those of the repository zccsdt
(cfour/xzccsdt and csrc), commit

    $commit

assembled on $today by cfour/merge/assemble.sh.  The documentation is
docs/native.md of that repository.  To change something, change the
repository and assemble again:

    <zccsdt>/cfour/merge/assemble.sh --cfour <cfour> --name $name --prog $prog --out <cfour>/$name

(the directory is emptied and written anew; <zccsdt>/cfour/merge/README.md has
the whole procedure and the regression check).

Needs of CFOUR: lib/libx2cint.a, lib/liblibr.a, the modules of libx2c
(cvwint_module, kinds, reals, complexs), MKL with 8-byte integers
(BLAS_LIBS of make.config), OpenMP.

Files (name here, where it comes from):

EOF
  cat "$manifest"
  cat <<EOF
Makefile               generated: CFOUR's form (make.config, make.rules), with the
                       order of compilation at its end
README                 generated
EOF
} > "$out/README"

echo "assemble.sh: $out"
echo "  program $prog; $nsrc files of cfour/xzccsdt; kernels: $kernels"
echo "  from $cfour:${frzlist:- nothing}"
echo "  commit $commit"
