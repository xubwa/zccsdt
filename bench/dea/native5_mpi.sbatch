#!/bin/bash
#SBATCH -p parallel -A lcheng24 --ntasks-per-node=1 -c 48 --mem=180G --exclusive
#SBATCH -J zc_dea5_mpi
#
# EOM-DEA with complete 5p3h of the native program over MPI ranks, one rank
# of 48 threads per node (docs/native.md, section 21): a run directory that
# all nodes see (native_rundir.py --level 5p3h), the subspace of the solver
# in the work array or - option eomvec f with eomvdir - in files on the disk
# of every node.
#
#   sbatch -N 32 -t 24:00:00 native5_mpi.sbatch SNAP RUNDIR TAG [MAXCYC] [MEMWORDS] ['option value' ...]
#
#   SNAP      a snapshot (bench/dea/snapshot.sh) with SNAP/xzccsdt_mpi
#   MAXCYC    ESTATE_MAXCYC (default 120; 2: a timing run of two iterations per target)
#   MEMWORDS  MEMORY_SIZE in words (default 165 GB), at most the least MemAvailable of the nodes
#             less 6 GB (GETMEM of every rank must succeed)
#   options   the lines of the section %xzccsdt of the ZMAT (replaced; none: the section as it is)
zargs=("$@")
source /data/lcheng24/intel/oneapi/setvars.sh --force > /dev/null 2>&1
set -- "${zargs[@]}"
snap=$1
rdir=$2
tag=$3
maxcyc=${4:-120}
memw=${5:-20625000000}
shift 5 2> /dev/null || shift $#
export PYTHONPATH=$snap/src:$HOME/apps/socutils-git-env:${PYTHONPATH:-}
export PATH=$PATH:/cm/shared/apps/slurm/current/bin
export I_MPI_PIN_DOMAIN=omp
export OMP_NUM_THREADS=48
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
nn=${#nodes[@]}
hosts=$(IFS=,; echo "${nodes[*]}")
echo "nodes ${nodes[*]}, job $SLURM_JOB_ID, $(date)"
cd "$rdir" || exit 1
if [ $# -gt 0 ]; then
  python -u "$snap/bench/dea/zmat_xz.py" . "$@"
fi
avail=$(srun -N $nn --ntasks-per-node=1 awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo | sort -n | head -1)
cap=$(( (avail - 6000) * 1000000 / 8 ))
[ "$memw" -gt "$cap" ] && memw=$cap
echo "least MemAvailable of the nodes: $avail MB; MEMORY_SIZE $memw words"
python -u "$snap/bench/dea/jobarc_flags.py" . 97=$maxcyc 36=$memw
/usr/bin/time -v mpirun -np $nn -ppn 1 -hosts "$hosts" "$snap/xzccsdt_mpi" > "xzccsdt.$tag.out" 2> "xzccsdt.$tag.time"
echo "exit $?"
grep "converged in\|sigma vectors:\|per sigma\|5p3h\|of them\|over the ranks\|work array used\|peak resident\|intermediates of the ground\|subspace" "xzccsdt.$tag.out"
grep -h "work array used" xzccsdt.r0* 2> /dev/null | sort -k5 -n | tail -3
grep "^ *[-0-9]" XZDEA 2> /dev/null
grep "Elapsed\|Maximum resident" "xzccsdt.$tag.time"
cp XZDEA "XZDEA.$tag" 2> /dev/null
# (the directory of option eomvdir on every node: empty after the run - the program deletes its files)
for o in "$@"; do
  case $o in
    eomvdir\ *) srun -N $nn --ntasks-per-node=1 rmdir "${o#eomvdir }" 2> /dev/null ;;
  esac
done
echo "done $(date)"
