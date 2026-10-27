#!/bin/bash
# O 1s core correlation of the O2 zero-field splitting with complete 5p3h
# (docs/native.md, section 21): the run directories of cc-pwCVTZ-X2C-TP,
# all electrons (16 e) and 1s frozen (12 e, native_rundir.py --freeze 4) from
# the same block file as the 4p2h runs of o2cv/pwcvtz (so the two share the
# spinors and the Hamiltonian), one directory per target, the CCSDT amplitudes
# (XZAMPS, XZKEY) of those 4p2h runs; then the option deaplan for
# cc-pwCVTZ-X2C-TP and cc-pwCVQZ-X2C-TP (plan directories whose files are
# links, the ZMAT with "level 5p3h").  One rank, nothing computed:
#
#   srun -A lcheng24 -p parallel -n1 -c8 --mem 30G -t 30 bash o2_cv5_prep.sh SNAP CVROOT [rundirs] [plans]
#
#   SNAP    a snapshot (snapshot.sh) with xzccsdt and src/, bench/
#   CVROOT  the directory of pwcvtz/ and pwcvqz/ (the 4p2h runs); the new
#           directories go to CVROOT/pwcvtz5 and CVROOT/pwcvqz5
zargs=("$@")
source /data/lcheng24/intel/oneapi/setvars.sh --force > /dev/null 2>&1
set -- "${zargs[@]}"
snap=$1
cv=$2
what=${3:-rundirs plans}
export PYTHONPATH=$snap/src:$HOME/apps/socutils-git-env:${PYTHONPATH:-}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "node $(hostname), job $SLURM_JOB_ID, $(date)"
tz=$cv/pwcvtz5
qz=$cv/pwcvqz5
mkdir -p "$tz" "$qz"

if [[ $what == *rundirs* ]]; then
  h5=$cv/pwcvtz/o2_hf2plus_pwcvtz_x2c_tp_ae_n172_x2ccorr.h5
  for spec in "ae 0 run_ae" "fc 4 run_frz"; do
    set -- $spec
    for t in 0 2; do
      d=$tz/${1}_t$t
      [ -f "$d/ZMAT" ] && continue
      python -u "$(dirname "$0")/native_rundir.py" "$h5" "$d" --cc-conv 10 --estate-conv 10 \
        --freeze $2 --targets $t --level 5p3h > "$d.rundir.log" 2>&1 || { tail -20 "$d.rundir.log"; exit 1; }
      tail -3 "$d.rundir.log"
      cp "$cv/pwcvtz/$3/XZAMPS" "$cv/pwcvtz/$3/XZKEY" "$d/"
    done
  done
fi

# plan: link the files of SRC into DST, ZMAT with level 5p3h and the deaplan line
plan() {
  local src=$1 dst=$2 ranks=$3 memgb=$4 f
  mkdir -p "$dst"
  for f in "$src"/*; do
    case $(basename "$f") in
      ZMAT|JOBARC|JAINDX|*xzccsdt*|time.err|XZDEA*|XZAMPS|XZKEY|EOM*|XZDXV*) ;;
      *) ln -sf "$f" "$dst/" ;;
    esac
  done
  cp "$src/JOBARC" "$src/JAINDX" "$dst/"
  python -u "$snap/bench/dea/jobarc_flags.py" "$dst" 36=$((memgb * 125000000))
  # (the targets 0 and 2 of the 4p2h ZMAT; "level 5p3h" after "parity g")
  sed -e "s/MEMORY_SIZE=[0-9]*/MEMORY_SIZE=$memgb/" -e '/^level /d' -e 's/^parity g$/parity g\nlevel 5p3h/' \
    -e '/^%xzccsdt/,$d' "$src/ZMAT" > "$dst/ZMAT"
  printf '%%xzccsdt\ndeaplan %s\n\n' "$ranks" >> "$dst/ZMAT"
  (cd "$dst" && /usr/bin/time -v "$snap/xzccsdt" > xzccsdt.out 2> time.err; echo "$dst: exit $?")
  grep -E 'Maximum resident|Elapsed' "$dst/time.err"
}

if [[ $what == *plans* ]]; then
  plan "$tz/ae_t0" "$tz/ae_plan" "7 10 12 14" 150
  plan "$tz/fc_t0" "$tz/fc_plan" "5 8 10 12" 150
  plan "$cv/pwcvqz/run_ae" "$qz/ae_plan" "14 28 42 56" 150
  plan "$cv/pwcvqz/run_frz" "$qz/fc_plan" "10 20 30 40" 150
fi
echo "done $(date)"
