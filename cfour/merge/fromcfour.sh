#!/bin/bash
#
# fromcfour.sh: the inverse of assemble.sh - CFOUR's directory zccsdt/ (or a
# copy of it, e.g. `git -C <cfour> archive <commit> zccsdt | tar -x -C DIR`)
# into the layout of this repository, so that changes made in CFOUR's tree
# can be committed here (and merged) as changes of these files.
#
#   cfour/merge/fromcfour.sh ZCCSDT_DIR
#
# *.f90, *.f, *.inc, xztype.h, xzmpi.c, xzutil.c -> cfour/xzccsdt/ (same names);
# the C kernels (KERNELS of cfour/xzccsdt/Makefile) and the headers of csrc/
# they include (zcred.h, ...) -> csrc/; test/* -> cfour/merge/test/.  Left out:
# the generated Makefile and README (their changes go by hand into
# cfour/xzccsdt/Makefile and the README template of assemble.sh), the marker
# file, and the routines of other CFOUR directories that assemble.sh copies
# (FRZOBJ, EOMOBJ of the Makefile).  Files of cfour/xzccsdt that ZCCSDT_DIR
# does not have are listed (new here, or removed there).  Nothing is
# committed.  The branch lan/x2cso-zccsdt was made this way, one commit per
# commit of CFOUR (docs/native.md, section 22).
#
set -eu
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
src=${1:?usage: fromcfour.sh ZCCSDT_DIR}
src=$(cd "$src" && pwd)
mk=$repo/cfour/xzccsdt/Makefile
[ -f "$mk" ] || { echo "fromcfour.sh: no $mk" >&2; exit 1; }
mkvar() {
  sed -e ':a' -e '/\\$/{N;s/\\\n/ /;ba' -e '}' "$mk" | sed -n "s/^$1[[:space:]]*[:?]\{0,1\}=[[:space:]]*//p" | head -1
}
kernels=" $(for k in $(mkvar KERNELS); do printf '%s.c ' "$k"; done)"
# the routines of other directories of CFOUR (objects named in the Makefile)
cfcopy=" $(sed -e ':a' -e '/\\$/{N;s/\\\n/ /;ba' -e '}' "$mk" | sed -n 's/.*\$(CFOUR)\/[A-Za-z0-9_]*\/[,[:space:]]*\(.*\))$/\1/p' |
           tr ' ,' '\n\n' | sed -n 's/\.o$//p' | sed 's/$/.f90/' | tr '\n' ' ')"
n=0
for f in "$src"/*; do
  [ -f "$f" ] || continue
  b=$(basename "$f")
  case $b in Makefile|README|.assembled_from_zccsdt|*.o|*.mod) continue ;; esac
  case "$cfcopy" in *" $b "*) continue ;; esac
  case "$kernels" in
    *" $b "*) cp -p "$f" "$repo/csrc/$b"; n=$((n + 1)); continue ;;
  esac
  # (the headers of the kernels live in csrc/; xztype.h is the program's)
  case $b in
    xztype.h) ;;
    *.h) cp -p "$f" "$repo/csrc/$b"; n=$((n + 1)); continue ;;
  esac
  cp -p "$f" "$repo/cfour/xzccsdt/$b"
  n=$((n + 1))
done
if [ -d "$src/test" ]; then
  for t in "$src"/test/*; do [ -f "$t" ] && cp -p "$t" "$repo/cfour/merge/test/"; done
fi
echo "fromcfour.sh: $n files from $src"
for f in "$repo"/cfour/xzccsdt/*; do
  b=$(basename "$f")
  [ "$b" = Makefile ] && continue
  [ -e "$src/$b" ] || echo "  not in $src: cfour/xzccsdt/$b"
done
