#!/bin/bash -l
#SBATCH --job-name=nccscal
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -A lcheng24
#SBATCH --partition=parallel
#SBATCH --mem=180G
#SBATCH --exclude=c105
# Thread scaling of CFOUR's ncc and of zccsdt on one whole node: the runs of a
# list one after the other, each bound to its cores.
# usage: sbatch -J nccscal_<case> -o LOG/%x-%j.out --export=ALL,ZC_ROOT=SNAPSHOT,NS_KEEP=LOG \
#            bench/ncc_scaling/slurm_scaling.sh RUNS
#   lines of the file RUNS (# starts a comment), run in the order given:
#     ncc TAG GEOM BASIS FROZEN SYM CALC CONV MAXCYC MEM_GB THREADS BIND MKL [WAIT]
#        GEOM: file of bench/ncc_scaling/geom;  FROZEN, SYM: ON | OFF
#        CALC: T (CCSDT) | Q (CCSDT(Q); ncc does (Q) only after convergence)
#        CONV, MAXCYC: CC_CONV and CC_MAXCYC (a run that stops at MAXCYC ends
#        with an error status after printing its iterations: timing only)
#        MKL: MKL_NUM_THREADS (n: as many as THREADS)
#        WAIT: OMP_WAIT_POLICY of the run, PASSIVE | ACTIVE (default: not set,
#        what a user gets; idle threads then spin for a while, which counts
#        as CPU time - with PASSIVE they sleep and CPU time is work)
#     zc TAG KIND GEOM BASIS NCORE NITER Q MAXMEM_MB THREADS BIND [WAIT]
#        bench/ncc_scaling/zc_run.py
#   THREADS: OMP_NUM_THREADS;  BIND:
#   (cores: the first THREADS cores of the job, 0..THREADS-1 on a whole node)
#     socket  numactl --physcpubind=cores --membind=0     (one socket and its memory)
#     pref    numactl --physcpubind=cores --preferred=0   (memory of the other socket when the first is full)
#     both    numactl --physcpubind=cores                 (memory where it is first touched)
#     inter   numactl --physcpubind=cores --interleave=all
#   CFOUR_BIN: the executables (the best ncc build of docs/benchmarks.md: the
#              one of bench/cfour/rebuild_ncc.sh gcc)
#   NS_TEST=1: outside Slurm, for a set-up test with tiny runs
runs=$(readlink -f "$1")
if [ "$NS_TEST" = 1 ]; then SLURM_JOB_ID=${SLURM_JOB_ID:-test$$}; fi
if [ -z "$SLURM_JOB_ID" ] || [ -z "$ZC_ROOT" ] || [ -z "$NS_KEEP" ]; then
    echo "needs a Slurm job, ZC_ROOT and NS_KEEP"; exit 1
fi
source /data/lcheng24/intel/oneapi/setvars.sh --force > /dev/null 2>&1
cfour_root=/home/xwang405/apps/cfour-x2cso
cfour_bin=${CFOUR_BIN:-/home/xwang405/.claude/jobs/7ba4fcf7/tmp/cfour_bin_gcc}
keep=$NS_KEEP/$(basename "$runs" .runs)_$SLURM_JOB_ID
scratch=/tmp/nccscal_$SLURM_JOB_ID
export ZCCSDT_TMPDIR=$scratch/zc PYSCF_TMPDIR=$scratch/zc
mkdir -p "$keep" $ZCCSDT_TMPDIR $scratch/bin
cp "$runs" "$keep/runs"
cleanup() {
    [ -d "/tmp/nccscal_$SLURM_JOB_ID" ] && find "/tmp/nccscal_$SLURM_JOB_ID" -mindepth 1 -delete
    rmdir "/tmp/nccscal_$SLURM_JOB_ID" 2> /dev/null
}
trap cleanup EXIT
SQUEUE=$(command -v squeue || echo /cm/shared/apps/slurm/current/bin/squeue)
# the tree runs from the local disk of the node
cp -r "$ZC_ROOT" $scratch/root
cd $scratch/root || exit 1
export PYTHONPATH=src:$HOME/apps/socutils-git-env:$PYTHONPATH PYTHONDONTWRITEBYTECODE=1
QUIET="pyscf plugins|site-packages/pyscf|shciscf/pyscf|PYSCF_EXT_PATH|omp_set_nested"

# the executables of CFOUR, with xncc under /usr/bin/time: its own CPU time,
# wall time and peak memory, apart from the SCF and the integral programs
for f in $cfour_bin/*; do ln -sf $f $scratch/bin/$(basename $f); done
rm -f $scratch/bin/xncc
cat > $scratch/bin/xncc <<WEOF
#!/bin/bash
exec /usr/bin/time -v -o \${NCC_TIME_FILE:-ncc_time.txt} $cfour_bin/xncc "\$@"
WEOF
chmod +x $scratch/bin/xncc

echo "job $SLURM_JOB_ID  host $(hostname)  $(date)  cpus $(taskset -cp $$ | sed 's/.*: //')  mem ${SLURM_MEM_PER_NODE} MB"
echo "root $PWD (copy of $ZC_ROOT)  revision $(cat REVISION 2> /dev/null)  cfour $cfour_bin"
lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Socket|NUMA node[0-9]'
numactl -H | grep -E 'cpus|size|free'
df -h /tmp | tail -1
echo "xncc: $(ls -l $cfour_bin/xncc)"
ldd $cfour_bin/xncc | grep -E 'mkl|omp' | sed 's/ (0x.*//'
echo "other jobs on this node:"; $SQUEUE -h -w $(hostname -s) -t R -o "  %i %u %j %C cores %M"

# the cores of the job in rising order (a whole node: 0-47, socket 0 first)
allowed=($(taskset -cp $$ | sed 's/.*: //' | tr ',' '\n' | awk -F- '{if (NF == 2) for (i = $1; i <= $2; i++) print i; else print $1}'))
bind() {        # THREADS BIND -> the numactl command (the first THREADS cores of the job)
    local list=$(echo ${allowed[@]:0:$1} | tr ' ' ',')
    case $2 in
      socket) echo "numactl --physcpubind=$list --membind=0";;
      pref)   echo "numactl --physcpubind=$list --preferred=0";;
      both)   echo "numactl --physcpubind=$list";;
      inter)  echo "numactl --physcpubind=$list --interleave=all";;
      *)      echo "false";;
    esac
}

state() {       # load and free memory of the two NUMA nodes
    echo "load $(cut -d' ' -f1-3 /proc/loadavg)  free MB $(numactl -H | awk '/free/ {printf "%s ", $4}')"
}

# largest number of threads of a running xncc (sampled every 5 s), into a file
watch_threads() {
    local out=$1 max=0 n
    while [ ! -e $out.stop ]; do
        n=$(ps -o nlwp= -C xncc 2> /dev/null | sort -n | tail -1)
        [ -n "$n" ] && [ "$n" -gt "$max" ] && max=$n
        sleep 5
    done
    echo $max > $out
    rm -f $out.stop
}

ncc() {         # TAG GEOM BASIS FROZEN SYM CALC CONV MAXCYC MEM_GB THREADS BIND MKL
    local tag=$1 geom=$2 basis=$3 frozen=$4 sym=$5 calc=$6 conv=$7 maxcyc=$8 mem=$9 nt=${10} bd=${11} mkl=${12}
    local wait=${13:-default}
    unset OMP_WAIT_POLICY; [ $wait != default ] && export OMP_WAIT_POLICY=$wait
    local run=$scratch/$tag level=CCSDT
    [ "$calc" = Q ] && level='CCSDT(Q)'
    [ "$mkl" = n ] && mkl=$nt
    mkdir -p $run $keep/$tag
    cp $cfour_root/basis/GENBAS $run/GENBAS
    python bench/ncc_scaling/mkzmat.py bench/ncc_scaling/geom/$geom $basis $run/GENBAS $run/ZMAT \
        "calc=$level" sym=$sym frozen=$frozen conv=$conv maxcyc=$maxcyc mem=$mem 2>&1 | grep -v -E "$QUIET"
    local cmd=$(bind $nt $bd)
    echo "$cmd  OMP_NUM_THREADS=$nt MKL_NUM_THREADS=$mkl OMP_WAIT_POLICY=$wait  MEMORY_SIZE=$mem GB" | tee $keep/$tag/command.txt
    watch_threads $keep/$tag/nlwp.txt &
    (cd $run && OMP_NUM_THREADS=$nt MKL_NUM_THREADS=$mkl OPENBLAS_NUM_THREADS=$nt NCC_TIME_FILE=$run/ncc_time.txt \
        PATH=$scratch/bin:$PATH /usr/bin/time -v $cmd xcfour > out 2> time.txt)
    touch $keep/$tag/nlwp.txt.stop; wait
    cp $run/out $run/time.txt $run/ZMAT $run/ncc_time.txt $keep/$tag/ 2> /dev/null
    find $run -mindepth 1 -delete
    local out=$keep/$tag/out
    grep -E "Computational point group|threads/proc|Memory limit|memory requirement|Compromise memory" $out | sort -u
    grep -E "E\(SCF\)=|iterations converged|Total CCSDT|CCSDT\[Q\]|CCSDT\(Q\)|did not converge|ERROR|rror" $out | sort -u
    echo "xncc: $(grep -E 'Elapsed|Maximum resident|User time|System time' $keep/$tag/ncc_time.txt | sed 's/^\s*//' | tr '\n' ';')  threads seen $(cat $keep/$tag/nlwp.txt)"
}

zc() {          # TAG KIND GEOM BASIS NCORE NITER Q MAXMEM_MB THREADS BIND
    local tag=$1 kind=$2 geom=$3 basis=$4 ncore=$5 niter=$6 q=$7 mem=$8 nt=$9 bd=${10}
    local wait=${11:-default}
    unset OMP_WAIT_POLICY; [ $wait != default ] && export OMP_WAIT_POLICY=$wait
    mkdir -p $keep/$tag
    local cmd=$(bind $nt $bd)
    echo "$cmd  OMP_NUM_THREADS=$nt MKL_NUM_THREADS=$nt OMP_WAIT_POLICY=$wait  max_memory=$mem MB" | tee $keep/$tag/command.txt
    OMP_NUM_THREADS=$nt MKL_NUM_THREADS=$nt OPENBLAS_NUM_THREADS=$nt ZC_MAXMEM=$mem \
        /usr/bin/time -v -o $keep/$tag/time.txt $cmd \
        python -u bench/ncc_scaling/zc_run.py $kind bench/ncc_scaling/geom/$geom $basis $ncore $niter $q 2>&1 \
        | grep --line-buffered -v -E "$QUIET" > $keep/$tag/log
    grep -E "^ZC |Traceback|Error" $keep/$tag/log
    echo "python: $(grep -E 'Elapsed|Maximum resident|User time|System time' $keep/$tag/time.txt | sed 's/^\s*//' | tr '\n' ';')"
}

while read -r kind args <&3; do
    set -- $args
    echo "=== $kind $1  $(date +%F\ %T)  $(state)"
    t0=$(date +%s)
    case $kind in
      ncc|zc) $kind $args < /dev/null;;
      *) echo "unknown kind $kind";;
    esac
    echo "=== end $kind $1  $(( $(date +%s) - t0 )) s  $(state)  scratch $(du -sh $scratch 2> /dev/null | cut -f1)"
done 3< <(grep -v -E '^\s*(#|$)' "$runs")
echo "other jobs on this node:"; $SQUEUE -h -w $(hostname -s) -t R -o "  %i %u %j %C cores %M"
echo "=== summary"
python bench/ncc_scaling/summary.py $keep 2>&1 | grep -v -E "$QUIET"
echo "done $(date)"
