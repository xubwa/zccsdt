#!/bin/bash
#SBATCH -p parallel -A lcheng24 -N 1 --ntasks-per-node=1 -c 16 --mem=60G
#SBATCH -J zc_dea_plan
#
# The run directory of the native EOM-DEA from a block file (native_rundir.py, if it has no ZMAT yet)
# and the option deaplan (docs/native.md 20.4: the memory of every rank for n ranks, nothing computed)
# in a copy of it whose files are links:
#
#   sbatch -t 1:00:00 native_plan.sbatch SNAP H5 RUNDIR FREEZE "RANKS" MEMGB
#
#   SNAP    a snapshot with the serial binary SNAP/xzccsdt and the Python package (src/, bench/)
#   FREEZE  native_rundir.py --freeze (the first FREEZE spinors of the block file frozen)
#   RANKS   the numbers of ranks, e.g. "1 2 4 8 12 16"
#   MEMGB   MEMORY_SIZE of the plan in GB
zargs=("$@")
source /data/lcheng24/intel/oneapi/setvars.sh --force > /dev/null 2>&1
set -- "${zargs[@]}"
snap=$1
h5=$2
rdir=$3
freeze=$4
ranks=$5
memgb=$6
export PYTHONPATH=$snap/src:$HOME/apps/socutils-git-env:${PYTHONPATH:-}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
echo "node $(hostname), job $SLURM_JOB_ID, $(date)"
if [ ! -f "$rdir/ZMAT" ]; then
  /usr/bin/time -v python -u "$snap/bench/dea/native_rundir.py" "$h5" "$rdir" --cc-conv 10 --estate-conv 10 \
    --freeze "$freeze" > "$rdir.rundir.log" 2>&1 || { tail -30 "$rdir.rundir.log"; exit 1; }
  tail -25 "$rdir.rundir.log"
fi
plan=${rdir}_plan
mkdir -p "$plan"
for f in "$rdir"/*; do
  case $(basename "$f") in
    ZMAT|JOBARC|JAINDX|xzccsdt*|time.err|XZDEA*|XZAMPS|XZKEY|EOM*|XZDXV*) ;;
    *) ln -sf "$f" "$plan/" ;;
  esac
done
cp "$rdir/JOBARC" "$rdir/JAINDX" "$plan/"
sed "s/MEMORY_SIZE=[0-9]*/MEMORY_SIZE=$memgb/" "$rdir/ZMAT" > "$plan/ZMAT"
printf '%%xzccsdt\ndeaplan %s\n\n' "$ranks" >> "$plan/ZMAT"
cd "$plan" || exit 1
/usr/bin/time -v "$snap/xzccsdt" > xzccsdt.out 2> time.err
echo "exit $?"
sed -n '/Memory plan/,$p' xzccsdt.out | head -100
grep -E 'Maximum resident|Elapsed' time.err
echo "done $(date)"
