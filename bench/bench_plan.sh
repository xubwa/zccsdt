#!/bin/bash
# A spinor CCSDT(Q) benchmark as an ordered list of measurements (bench/bench_step.py), cheapest
# and most telling first, each its own process, each result appended to one JSON-lines file:
#   1. CCSDT iterations: every mode x every number of threads (largest thread count first)
#   2. (Q) from a sample of the occupied quadruples: every mode x every thread count, for each fraction of
#      BP_FRACTION in turn (e.g. "0.005 0.02": a first idea quickly, a better number after it)
#   3. the whole (Q), only where asked for (BP_FULLQ="mode:threads ..."), last
# A measurement already in the file is skipped, so a job that ran out of time is resubmitted as it is.
# Threads are bound to cores (one per core, in order); a run that fits one socket also takes its
# memory from that socket.
#
# usage (in a Slurm job with a whole node): bench/bench_plan.sh OUT.jsonl
# environment: BP_GEOM, BP_BASIS, BP_FROZEN, BP_MODES="blocks dense", BP_THREADS="48 24 8",
#              BP_NITER=4, BP_FRACTION="0.05" ("" or "none": no (Q)), BP_ITER=1 (0: no iterations), BP_FULLQ="", BP_REAL=1, BP_TAG, BP_DRY=1 (print the list only),
#              BP_NUMA=interleave, BP_LOGDIR (keep the output of every measurement there)
set -u
out=$1
: "${BP_GEOM:?}" "${BP_BASIS:?}"
frozen=${BP_FROZEN:-0}; modes=${BP_MODES:-"blocks dense"}; threads=${BP_THREADS:-"48 24 8"}
niter=${BP_NITER:-4}; fraction=${BP_FRACTION:-0.05}; fullq=${BP_FULLQ:-}; real=${BP_REAL:-1}; tag=${BP_TAG:-}
root=$(cd "$(dirname "$0")/.." && pwd)
spinors=${out%.jsonl}.spinors.npz
pick() {    # N -> "cpu list|node": N cores of the job's own set, of one socket (and that socket) if they fit there
  python3 - "$1" <<'PY'
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
}
step() {    # threads mode what extra...
  local n=$1 mode=$2 what=$3; shift 3
  local sel; sel=$(pick "$n"); local cpus=${sel%|*} node=${sel#*|}
  local bind=(taskset -c "$cpus")
  if [ -n "$node" ] && command -v numactl >/dev/null; then bind=(numactl --physcpubind="$cpus" --membind="$node"); fi
  # BP_NUMA=interleave: the pages of every array spread over the sockets (a run on both sockets whose arrays
  # are first written by one thread has them all on that thread's socket otherwise)
  if [ "${BP_NUMA:-}" = interleave ] && command -v numactl >/dev/null; then bind=(numactl --physcpubind="$cpus" --interleave=all); node="all, interleaved"; fi
  echo "=== $(date '+%F %T')  $what $mode, $n threads  load $(cut -d' ' -f1-3 /proc/loadavg)  cores $cpus${node:+, memory of socket $node}"
  [ -n "${BP_DRY:-}" ] && return 0
  local log; log=$(mktemp)
  OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n OPENBLAS_NUM_THREADS=$n OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=PASSIVE \
    "${bind[@]}" python -u "$root/bench/bench_step.py" --geom "$BP_GEOM" --basis "$BP_BASIS" --frozen "$frozen" \
    --mode "$mode" --what "$what" --real "$real" --out "$out" --spinors "$spinors" --tag "$tag" "$@" > "$log" 2>&1
  local rc=$?
  grep -E "^RESULT|already in" "$log" || { echo "!!! no result (exit $rc); the end of the output:"; tail -n 15 "$log"; }
  if [ -n "${BP_LOGDIR:-}" ]; then mkdir -p "$BP_LOGDIR"; cp "$log" "$BP_LOGDIR/${tag}_${what}_${mode}_${n}_$(date +%H%M%S).log"; fi
  rm -f "$log"
}
echo "host $(hostname)  $(date)  $(lscpu | grep 'Model name' | sed 's/  */ /g')"; lscpu | grep -E "NUMA node[0-9]"; free -g | sed -n 2p
echo "other jobs on this node:"; squeue -w "$(hostname)" -h -o "  %i %u %j %C cores" 2>/dev/null | grep -v " ${SLURM_JOB_ID:-none} " || true
if [ "${BP_ITER:-1}" != 0 ]; then for n in $threads; do for mode in $modes; do step "$n" "$mode" iter --niter "$niter"; done; done; fi
[ "$fraction" = none ] && fraction=
for fr in $fraction; do for n in $threads; do for mode in $modes; do step "$n" "$mode" q --fraction "$fr"; done; done; done
for item in $fullq; do step "${item#*:}" "${item%:*}" q --fraction 0; done
echo "done $(date)  load $(cut -d' ' -f1-3 /proc/loadavg)"
