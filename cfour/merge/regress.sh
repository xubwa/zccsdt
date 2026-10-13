#!/bin/bash
#
# regress.sh: two builds of the native program on the regression cases - the
# one of CFOUR's make against the one of the repository's Makefile.
#
#   cfour/merge/regress.sh --ref BIN --new BIN [--ref-args] [--np N] [--cases "c1 c2 ..."]
#                          [--refdir DIR] [--scratch DIR] [--threads T] [--eom]
#                          [--cfourbin DIR]
#
#   --ref BIN      the reference binary (build/xzccsdt/xzccsdt of the repository;
#                  with --np: build/xzccsdt_mpi/xzccsdt)
#   --new BIN      the binary under test (<cfour>/bin/xzccsdt; with --np: xzccsdt_mpi)
#   --ref-args     the reference binary is one of before the interface without
#                  arguments (commit a38e493): it is given the method as its
#                  argument.  Otherwise (and always for --new) the method is set
#                  in the ZMAT and xjoda is run again before the program
#   --cfourbin DIR the xjoda for that (default ~/apps/cfour-x2cso/bin); the ZMAT of a
#                  case has CALC=... followed by a comma
#   --np N         both through "mpirun -np N" (the MPI environment must be set)
#   --refdir DIR   the run directories of the cases, one per case, each with the
#                  files of xjoda .. xx2cvtran (default
#                  /scratch16/lcheng24/xwang405/zccsdt_ref; each has a README
#                  that says how it was made)
#   --eom          also "eom" from XZAMPS (the excited states that the ZMAT of
#                  the case asks for): the file XZEOM - shift, root, omega to
#                  8 decimals, converged.  With --np the reference binary
#                  runs this step with one rank and the binary under test
#                  with N: the roots over N ranks against those of one rank
#                  (a reference of before EOM over ranks stops with more)
#   --scratch DIR  where the copies are run (default /tmp/$USER.xzregress);
#                  never a source directory
#
# Per case and binary, in a copy of the run directory: "ccsdt(q)_l" (CCSDT,
# the Lambda equations, (Q)_Lambda; ZMAT: CALC=CCSDT(Q)_L), "(q)" from XZAMPS
# (CALC=CCSDT(Q), the amplitudes taken), "density" from XZAMPS and XZLAMB
# (CALC=CCSDT,PROPS=FIRST_ORDER: the seven files of xx2czvec), with --eom
# "eom" (CALC=CCSDT,EXCITE=EOMEE).  Compared: the exit status, the
# mode (labels, real or complex), the cycles, every energy as printed (12
# decimals), and the density files (bytes; where they differ, the largest
# difference of an element).  Exit status 0 if nothing printed differs and
# no element of a density file differs by more than 1e-10.
#
# Run it in a job, not on a login node (8 threads; about 2 minutes per
# binary for the default cases).
#
set -u
ref=; new=; np=0; threads=8; eom=0; refargs=0; cfourbin=$HOME/apps/cfour-x2cso/bin
refdir=/scratch16/lcheng24/xwang405/zccsdt_ref
scratch=/tmp/$USER.xzregress
cases="h2o_631g hbr_fc hbr_lab hbr_ph hbr_dglin_fc hbr_dglin_fc_vc hbrp_dglin_fc"
while [ $# -gt 0 ]; do
  case $1 in
    --ref) ref=$2; shift 2 ;;
    --new) new=$2; shift 2 ;;
    --np) np=$2; shift 2 ;;
    --cases) cases=$2; shift 2 ;;
    --refdir) refdir=$2; shift 2 ;;
    --scratch) scratch=$2; shift 2 ;;
    --threads) threads=$2; shift 2 ;;
    --eom) eom=1; shift ;;
    --ref-args) refargs=1; shift ;;
    --cfourbin) cfourbin=$2; shift 2 ;;
    -h|--help) sed -n '2,45p' "$0"; exit 0 ;;
    *) echo "regress.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -x "$ref" ] && [ -x "$new" ] || { echo "regress.sh: --ref BIN and --new BIN (executables) are needed" >&2; exit 2; }
ref=$(cd "$(dirname "$ref")" && pwd)/$(basename "$ref")
new=$(cd "$(dirname "$new")" && pwd)/$(basename "$new")
mkdir -p "$scratch" || exit 2
scratch=$(cd "$scratch" && pwd)

if [ "$np" -gt 0 ]; then
  export OMP_NUM_THREADS=$((threads / np)) I_MPI_HYDRA_BOOTSTRAP=${I_MPI_HYDRA_BOOTSTRAP:-fork}
  [ "$OMP_NUM_THREADS" -ge 1 ] || export OMP_NUM_THREADS=1
else
  export OMP_NUM_THREADS=$threads
fi

# one binary on all cases: the lines to compare in $scratch/WHO.txt, the density
# files in $scratch/WHO.den/
# the ZMAT keywords of a method of the old interface
zkw() {
  case $1 in
    'ccsdt(q)_l') echo 'CALC=CCSDT(Q)_L' ;;
    '(q)') echo 'CALC=CCSDT(Q)' ;;
    density) echo 'CALC=CCSDT,PROPS=FIRST_ORDER' ;;
    eom) echo 'CALC=CCSDT,EXCITE=EOMEE' ;;
  esac
}
runset() {
  local who=$1 bin=$2 useargs=$3 c m f n d o st n1
  local txt=$scratch/$who.txt
  mkdir -p "$scratch/$who.den" "$scratch/$who.out"
  : > "$txt"
  for c in $cases; do
    [ -d "$refdir/$c" ] || { echo "$c | no directory $refdir/$c" >> "$txt"; continue; }
    d=$scratch/$who.run
    mkdir -p "$d"
    find "$d" -mindepth 1 -delete
    for f in "$refdir/$c"/*; do
      n=$(basename "$f")
      case $n in
        XZAMPS|XZLAMB|XZKEY|XZEOM*|EOMC[0-9]*|EOMH[0-9]*|XZDIIS*|XZQCHK.*|xzccsdt.r*|xfull|dpoo|dpvv|dii0|daa0|dpopo|dpvpv|denfinal|*.out|*.log|README) ;;
        *) [ -f "$f" ] && cp -p "$f" "$d/" ;;
      esac
    done
    methods='ccsdt(q)_l (q) density'
    [ $eom = 1 ] && methods="$methods eom"
    for m in $methods; do
      o=$scratch/$who.out/$c.$(echo "$m" | tr -d '()').out
      args=$m
      if [ "$useargs" = 0 ]; then
        args=
        sed -i -e 's/,EXCITATION=[0-9]*//' -e 's/,PROPS=[A-Z_]*//' -e 's/,EXCITE=[A-Z]*//' \
               -e "s/CALC=[^,]*,/$(zkw "$m"),/" "$d/ZMAT"
        (cd "$d" && "$cfourbin/xjoda" > "$o.xjoda" 2>&1)
      fi
      if [ "$np" -gt 0 ]; then
        n1=$np
        [ "$m" = eom ] && [ "$who" = ref ] && n1=1
        (cd "$d" && mpirun -np "$n1" "$bin" $args > "$o" 2>&1)
      else
        (cd "$d" && "$bin" $args > "$o" 2>&1)
      fi
      st=$?
      {
        echo "$c | $m | exit $st | $(grep -c 'XZCCSDT-W' "$o") warnings | $(grep -m1 'symmetry labels:' "$o" | sed 's/ *symmetry labels: *//' | cut -c1-40) | $(grep -m1 'arithmetic:' "$o" | sed 's/  */ /g')"
        grep -E 'converged in' "$o" | sed "s/^ */$c | $m |   /"
        grep -E '^ *E\(|^ *E[\[(]Q' "$o" | sed "s/^ */$c | $m |   /; s/  */ /g"
      } >> "$txt"
      if [ "$m" = eom ]; then
        if [ -f "$d/XZEOM" ]; then
          awk -v c="$c" '!/^#/ { printf "%s | eom | shift %s root %s omega %.8f converged %s\n", c, $1, $2, $3, $10 }' "$d/XZEOM" >> "$txt"
        else
          echo "$c | eom | no file XZEOM" >> "$txt"
        fi
      fi
    done
    for f in xfull dpoo dpvv dii0 daa0 dpopo dpvpv; do
      if [ -f "$d/$f" ]; then
        echo "$c | file $f | $(stat -c %s "$d/$f") bytes" >> "$txt"
        cp -p "$d/$f" "$scratch/$who.den/$c.$f"
      else
        echo "$c | file $f | missing" >> "$txt"
      fi
    done
    find "$d" -mindepth 1 -delete
  done
  echo "$who ($bin): $(grep -c '| exit 0 |' "$txt") runs with exit 0, $(grep -c '| exit [1-9]' "$txt") failed, $(grep -c ' E[(\[]' "$txt") energy lines"
}

runset ref "$ref" $refargs
runset new "$new" 0

bad=0
if diff "$scratch/ref.txt" "$scratch/new.txt" > "$scratch/diff.txt"; then
  echo "printed lines (status, mode, cycles, energies to 12 decimals, sizes of the density files): $(wc -l < "$scratch/ref.txt"), all identical"
else
  echo "printed lines DIFFER:"; cat "$scratch/diff.txt"; bad=1
fi
grep -q '| exit [1-9]' "$scratch/new.txt" && bad=1
nf=0; nsame=0; ndiff=
for f in "$scratch"/ref.den/*; do
  [ -f "$f" ] || continue
  g=$scratch/new.den/$(basename "$f")
  nf=$((nf + 1))
  if cmp -s "$f" "$g"; then nsame=$((nsame + 1)); else ndiff="$ndiff $(basename "$f")"; fi
done
echo "density files: $nf, byte-identical: $nsame"
if [ -n "$ndiff" ]; then
  # one record each: 4 bytes, the numbers (8-byte reals), 4 bytes
  python - "$scratch" $ndiff <<'EOF' || bad=1
import sys, os
import numpy as np
s = sys.argv[1]; worst = 0.0
for n in sys.argv[2:]:
    a = os.path.join(s, 'ref.den', n); b = os.path.join(s, 'new.den', n)
    if not os.path.exists(b) or os.path.getsize(a) != os.path.getsize(b):
        print('   %-28s missing or of another size' % n); worst = 1.0; continue
    x = np.fromfile(a, dtype=np.uint8)[4:-4].view(np.float64)
    y = np.fromfile(b, dtype=np.uint8)[4:-4].view(np.float64)
    d = abs(x - y).max() if x.size else 0.0
    worst = max(worst, d)
    print('   %-28s largest difference %.2e (largest element %.3e)' % (n, d, abs(x).max() if x.size else 0.0))
print('largest difference of an element of a density file: %.2e' % worst)
sys.exit(0 if worst <= 1e-10 else 1)
EOF
fi
if [ $bad = 0 ]; then echo "REGRESSION OK"; else echo "REGRESSION FAILED (outputs in $scratch)"; fi
exit $bad
