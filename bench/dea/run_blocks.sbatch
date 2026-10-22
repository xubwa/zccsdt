#!/bin/bash
#SBATCH -p parallel -A lcheng24 -N 1 --ntasks-per-node=1 -c 48 --mem=180G --exclusive
#SBATCH -J zc_runblocks
#
# The X2Ccorr block file of O2(2+) with the code of a snapshot
# (o2_x2ccorr.py run-blocks: the SCF of the checkpoint, the spinors as
# o2_scf.spinors makes them - parity eigenfunctions -, default options).
#
#   sbatch -t 2:00:00 run_blocks.sbatch SNAP BASIS OUTDIR
#   (BASIS: e.g. pvqz_x2c_tp; the checkpoint ~/scr4/zccsdt_o2dea/o2_2plus_krhf_x2cmp_<BASIS>.chk)
snap=$1
basis=$2
out=$3
export PYTHONPATH=$snap/src:$HOME/apps/socutils-git-env:${PYTHONPATH:-}
export OMP_NUM_THREADS=48
export ZCCSDT_TMPDIR=/tmp
mkdir -p "$out"
echo "node $(hostname), job $SLURM_JOB_ID, $(date)"
/usr/bin/time -v python -u "$snap/bench/dea/o2_x2ccorr.py" run-blocks "$HOME/scr4/zccsdt_o2dea" --basis "$basis" --out "$out" \
  > "$out/run.log" 2> "$out/run.time"
echo "exit $?"
grep -v -i plugin "$out/run.log" | grep -v pyscf | tail -30
grep "Elapsed\|Maximum resident" "$out/run.time"
echo "done $(date)"
