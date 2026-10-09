#!/bin/bash -l
#SBATCH --job-name=eom2
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH -A lcheng24
#SBATCH --partition=parallel
#SBATCH --mem=90G
#SBATCH --exclusive
# EOM-CCSDT of HBr in the blocks of the symmetry labels (bench/eom2_bench.py):
# states with a label shift, IP with short vectors; a node of its own, the
# threads on the cores of one socket and the memory of that socket.
#   usage: sbatch -o LOGDIR/%x-%j.out --export=ALL,ZC_ROOT=$SNAP bench/slurm_eom2.sh
#   ZC_ROOT   snapshot of the repository (with src/zccsdt/lib/*.so)
#   E2_RUNS   the runs, "case:parts" separated by blanks (default "tz:ee,ip qz:ee,ip")
#   E2_THREADS (default 24), and the variables of bench/eom2_bench.py
cd ${ZC_ROOT:?snapshot of the repository}
ROOT=$(pwd)
export PYTHONPATH=$ROOT/src:$HOME/apps/socutils-git-env:$PYTHONPATH
S=/tmp/zccsdt_eom2_${SLURM_JOB_ID:-$$}
mkdir -p $S
export ZCCSDT_TMPDIR=$S PYSCF_TMPDIR=$S
n=${E2_THREADS:-24}
sel=$(python3 - "$n" <<'PY'
import glob, os, sys
n = int(sys.argv[1]); mine = sorted(os.sched_getaffinity(0)); nodes = {}
for d in glob.glob('/sys/devices/system/node/node[0-9]*'):
    r = open(d + '/cpulist').read().strip(); cpus = set()
    for part in r.split(','):
        lo, _, hi = part.partition('-'); cpus.update(range(int(lo), int(hi or lo) + 1))
    nodes[int(d.rsplit('node', 1)[1])] = sorted(cpus & set(mine))
best = max(nodes, key=lambda k: len(nodes[k])) if nodes else None
if best is not None and len(nodes[best]) >= n:
    print(','.join(map(str, nodes[best][:n])) + '|%d' % best)
else:
    print(','.join(map(str, mine[:n])) + '|')
PY
)
cpus=${sel%|*}; node=${sel#*|}
bind=(taskset -c "$cpus")
if [ -n "$node" ] && command -v numactl >/dev/null; then bind=(numactl --physcpubind="$cpus" --membind="$node"); fi
echo "host $(hostname)  $(date)  threads $n  cores $cpus${node:+, memory of socket $node}  commit ${ZC_COMMIT:-?}"
lscpu | grep 'Model name'; lscpu | grep -E "NUMA node[0-9]"; free -g | sed -n 2p
echo "cpus of the job: $(taskset -cp $$ | sed 's/.*: //')  load $(cut -d' ' -f1-3 /proc/loadavg)"
echo "other jobs on this node:"; squeue -w "$(hostname)" -h -o "  %i %u %j %C cores" 2>/dev/null | grep -v " ${SLURM_JOB_ID:-none} " || true
filter() { grep --line-buffered -v -i -E "plugins found|site-packages/pyscf|shciscf|PYSCF_EXT_PATH"; }
runs=${E2_RUNS:-tz:ee,ip qz:ee,ip}
for run in $runs; do
  echo "=== $(date '+%F %T')  ${run}  load $(cut -d' ' -f1-3 /proc/loadavg)"
  OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n OPENBLAS_NUM_THREADS=$n OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=PASSIVE \
    "${bind[@]}" python -u bench/eom2_bench.py "${run%%:*}" "${run#*:}" 2>&1 | filter
done
find "$S" -delete
echo "done $(date)  load $(cut -d' ' -f1-3 /proc/loadavg)"
