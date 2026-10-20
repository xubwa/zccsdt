#!/bin/bash
#SBATCH -p parallel -A lcheng24 --ntasks-per-node=1 -c 48 --mem=180G --exclusive
#SBATCH -J zc_dea_blk
#
# EOM-DEA-CCSDT of the native program from a block file of zccsdt.hblocks
# (docs/native.md, sections 20.1, 20.4, 20.7): the run directory on a file
# system that all nodes see (made by native_rundir.py if it has no ZMAT, no
# n^4 array), then one rank per node with 48 threads.
#
#   sbatch -N 12 -t 8:00:00 native_blocks.sbatch SNAP H5 RUNDIR TAG [MAXCYC] [MEMWORDS]
#
#   SNAP      a snapshot with the MPI binary SNAP/xzccsdt_mpi and the Python package (src/, bench/)
#   MAXCYC    ESTATE_MAXCYC (default 120; 3: a timing run - the CCSDT amplitudes stay in XZAMPS and
#             the next run takes them)
#   MEMWORDS  MEMORY_SIZE in words (default 165 GB), at most the least MemAvailable of the nodes
#             less 6 GB: GETMEM of every rank must succeed (nodes of the partition have had 140 -
#             175 GB available at the start of a job)
# (setvars.sh of oneAPI first: it reads unset variables and sets some of its own)
zargs=("$@")
source /data/lcheng24/intel/oneapi/setvars.sh --force > /dev/null 2>&1
set -- "${zargs[@]}"
snap=$1
h5=$2
rdir=$3
tag=$4
maxcyc=${5:-120}
memw=${6:-20625000000}
export PYTHONPATH=$snap/src:$HOME/apps/socutils-git-env:${PYTHONPATH:-}
export PATH=$PATH:/cm/shared/apps/slurm/current/bin
export I_MPI_PIN_DOMAIN=omp
export OMP_NUM_THREADS=48
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
nn=${#nodes[@]}
hosts=$(IFS=,; echo "${nodes[*]}")
echo "nodes ${nodes[*]}, job $SLURM_JOB_ID, $(date)"
if [ ! -f "$rdir/ZMAT" ]; then
  python -u "$snap/bench/dea/native_rundir.py" "$h5" "$rdir" --cc-conv 10 --estate-conv 10 > "$rdir.rundir.log" 2>&1 || exit 1
fi
cd "$rdir" || exit 1
avail=$(srun -N $nn --ntasks-per-node=1 awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo | sort -n | head -1)
cap=$(( (avail - 6000) * 1000000 / 8 ))
[ "$memw" -gt "$cap" ] && memw=$cap
echo "least MemAvailable of the nodes: $avail MB; MEMORY_SIZE $memw words"
python -u "$snap/bench/dea/jobarc_flags.py" . 97=$maxcyc 36=$memw
/usr/bin/time -v mpirun -np $nn -ppn 1 -hosts "$hosts" "$snap/xzccsdt_mpi" > "xzccsdt.$tag.out" 2> "xzccsdt.$tag.time"
echo "exit $?"
grep "vv||vv\|converged in\|sigma vectors:\|per sigma\|the vector to the ranks\|work array used\|peak resident\|intermediates of the ground\|subspace" "xzccsdt.$tag.out"
grep -h "work array used" xzccsdt.r0*
grep "^ *[-0-9]" XZDEA 2> /dev/null
grep "Elapsed\|Maximum resident" "xzccsdt.$tag.time"
cp XZDEA "XZDEA.$tag" 2> /dev/null
echo "done $(date)"
