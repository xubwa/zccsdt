#!/bin/bash -l
#SBATCH --job-name=cfour
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH -A lcheng24
#SBATCH --partition=parallel
#SBATCH --mem=180G
# usage: sbatch bench/cfour/slurm_cfour.sh MOL BASIS CC_PROG REF MAXCYC
mol=$1; basis=$2; prog=$3; ref=$4; maxcyc=${5:-6}
export OMP_NUM_THREADS=48 MKL_NUM_THREADS=48
export PATH=/home/xwang405/apps/cfour-x2cso/bin:$PATH
run=/home/xwang405/.claude/jobs/7ba4fcf7/tmp/cfour_bench/${mol}_${basis}_${prog}_${ref}_${SLURM_JOB_ID}
mkdir -p $run && cd $run
cp /home/xwang405/apps/cfour-x2cso/basis/GENBAS GENBAS
case $mol in
  h2o) geom="O 0.0 0.0 0.0
H 0.0 0.757 0.587
H 0.0 -0.757 0.587";;
  n2) geom="N 0.0 0.0 0.0
N 0.0 0.0 1.0977";;
esac
cat > ZMAT <<ZEOF
$mol CCSDT $basis
$geom

*CFOUR(CALC=CCSDT,BASIS=$basis,REF=$ref,CC_PROG=$prog
COORDINATES=CARTESIAN,UNITS=ANGSTROM,SYMMETRY=OFF
FROZEN_CORE=OFF,SCF_CONV=10,CC_CONV=10,CC_MAXCYC=$maxcyc
SCF_MAXCYC=200,MEMORY_SIZE=150,MEM_UNIT=GB)

ZEOF
echo "host $(hostname) $(date) run dir $run"
/usr/bin/time -v xcfour > out 2> time.txt
grep -E "executable x(ncc|ecc|vcc|vtran|intprc) finished|Walltime|It\. " out | tail -40
grep -E "Maximum resident" time.txt
echo "done $(date)"
