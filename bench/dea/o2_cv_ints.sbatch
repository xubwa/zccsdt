#!/bin/bash
#SBATCH -p parallel -A lcheng24 -N 1 --ntasks-per-node=1 -c 48 --mem=180G --exclusive
#SBATCH -J zc_o2cv_int
#
# The X2Ccorr block files of O2(2+) for the 1s contribution to the DEA splitting (all electrons
# minus 1s frozen, the same basis and the same spinors):
#
#   sbatch -t 4:00:00 o2_cv_ints.sbatch SNAP BASIS DIR MODE
#
#   BASIS  pwcvtz_x2c_tp, pwcvqz_x2c_tp (bench/dea/o2_scf.py --basis)
#   MODE   ae      the SCF of O2(2+) (once; checkpoint DIR/o2_2plus_krhf_x2cmp_<BASIS>_ae.chk) and the
#                  block file of all spinors, nothing frozen (o2_x2ccorr.py run-blocks --ae)
#          fcdef   the same checkpoint copied to the name of the frozen-core tag (the same spinors, no
#                  new SCF), and the block file with the four-component field of the frozen 1s spinors
#                  (the default of run-blocks)
#          check   bench/dea/o2_cv_check.py: spinors kept, E(SCF), the orbitals of the two checkpoints
zargs=("$@")
source /data/lcheng24/intel/oneapi/setvars.sh --force > /dev/null 2>&1
source ~/.conda_activate 2>/dev/null
set -- "${zargs[@]}"
snap=$1
basis=$2
dir=$3
mode=$4
export PYTHONPATH=$snap/src:$HOME/apps/socutils-git-env:${PYTHONPATH:-}
export OMP_NUM_THREADS=48 MKL_NUM_THREADS=48
export TMPDIR=/tmp/zc_o2cv_$SLURM_JOB_ID
mkdir -p "$TMPDIR" "$dir"
echo "node $(hostname), job $SLURM_JOB_ID, $(date)"
ae=$dir/o2_2plus_krhf_x2cmp_${basis}_ae.chk
fc=$dir/o2_2plus_krhf_x2cmp_${basis}.chk
case $mode in
  ae)
    python -u "$snap/bench/dea/o2_cv_check.py" scf "$dir" "$basis" > "$dir/scf.log" 2>&1 || { tail -30 "$dir/scf.log"; exit 1; }
    grep "converged SCF\|E(SCF)" "$dir/scf.log"
    /usr/bin/time -v python -u "$snap/bench/dea/o2_x2ccorr.py" run-blocks "$dir" --basis "$basis" --ae --max-memory 150000
    ;;
  fcdef)
    [ -f "$ae.done" ] || { echo "no converged SCF $ae"; exit 1; }
    cp "$ae" "$fc" && touch "$fc.done"
    /usr/bin/time -v python -u "$snap/bench/dea/o2_x2ccorr.py" run-blocks "$dir" --basis "$basis" --max-memory 150000
    ;;
  check)
    python -u "$snap/bench/dea/o2_cv_check.py" check "$dir" "$basis"
    ;;
esac
echo "exit $?"
rm -rf "/tmp/zc_o2cv_$SLURM_JOB_ID"
echo "done $(date)"
