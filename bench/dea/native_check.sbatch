#!/bin/bash
#SBATCH -p parallel -A lcheng24 -N 1 -n 1 -c 32 --mem=120G -t 8:00:00
#SBATCH -J zc_dea_chk
#
# EOM-DEA of the native program at production size: the sigma vector element
# by element against dea_sector (bench/dea/native_check.py, both targets) and
# the roots under a small work array (option maxcor: smaller batches, Xv in
# the file XZDXV when it does not fit) against the roots with a large one.
#
#   sbatch native_check.sbatch SNAP NPZ FREEZE OUTDIR MAXCOR_WORDS [ESTATE_THR]
set -u
snap=$1
npz=$2
freeze=$3
outdir=$4
maxcor=$5
thr=${6:-1e-10}
tag=$(basename "$npz" .npz)
mkdir -p "$outdir"
export OMP_NUM_THREADS=32
export PYTHONPATH=$snap/src:$HOME/apps/socutils-git-env:${PYTHONPATH:-}
echo "node $(hostname), job $SLURM_JOB_ID, $(date)"
cd "$snap"
d=/tmp/zc_dea_chk_$SLURM_JOB_ID
mkdir -p "$d"
python -u bench/dea/native_rundir.py "$npz" "$d" --freeze "$freeze" --mem-gb 100 --cc-conv 10 --estate-conv 9 \
  > "$outdir/rundir_$tag.log" 2>&1
printf '\n%%eomthrs\n%s\n\n' "$thr" >> "$d/ZMAT"
echo "=== native: CCSDT and the roots, MEMORY_SIZE 100 GB"
(cd "$d" && /usr/bin/time -v "$snap/xzccsdt" > xzccsdt.full.out 2> xzccsdt.full.time)
cp "$d/xzccsdt.full.out" "$outdir/full_$tag.out"
cp "$d/XZDEA" "$outdir/full_$tag.XZDEA"
grep "sigma vectors:\|per sigma\|takes\|work array used\|peak resident\|Xv with" "$outdir/full_$tag.out"
grep "Maximum resident" "$d/xzccsdt.full.time"
for t in 0 2; do
  echo "=== the sigma vector of target $t against dea_sector"
  python -u bench/dea/native_check.py "$d" "$snap/xzccsdt" --target $t --threads 32 2>&1 | grep -v "pyscf\|PYSCF"
done
echo "=== native: the roots in $maxcor words (option maxcor)"
printf '\n%%xzccsdt\nmaxcor %s\n\n' "$maxcor" >> "$d/ZMAT"
(cd "$d" && /usr/bin/time -v "$snap/xzccsdt" > xzccsdt.small.out 2> xzccsdt.small.time)
cp "$d/xzccsdt.small.out" "$outdir/small_$tag.out"
cp "$d/XZDEA" "$outdir/small_$tag.XZDEA"
grep "sigma vectors:\|per sigma\|takes\|work array used\|peak resident\|Xv with\|subspace" "$outdir/small_$tag.out"
grep "Maximum resident" "$d/xzccsdt.small.time"
grep -v "^#" "$outdir/full_$tag.XZDEA"
grep -v "^#" "$outdir/small_$tag.XZDEA"
rm -f "$d"/moin* "$d"/PPPH* "$d"/XZ* "$d"/EOM* "$d"/* 2> /dev/null
rmdir "$d"
echo "done $(date)"
