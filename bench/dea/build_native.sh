#!/bin/bash
# Builds the native program of this checkout (cfour/xzccsdt): the serial and
# the MPI binary in build/xzccsdt and build/xzccsdt_mpi, and with "lib" the C
# kernels of the Python package (csrc -> src/zccsdt/lib, linked with MKL).
#   bash bench/dea/build_native.sh [serial] [mpi] [lib]     (default: serial mpi)
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source /data/lcheng24/intel/oneapi/setvars.sh --force > /dev/null 2>&1
what="$*"
[ -z "$what" ] && what="serial mpi"
for x in $what; do
  case $x in
    serial) make -C "$ROOT/cfour/xzccsdt" -j8 2>&1 | grep -E 'error|Error' -A3 || true
            ls -la "$ROOT/build/xzccsdt/xzccsdt" ;;
    mpi)    make -C "$ROOT/cfour/xzccsdt" -j8 MPI=1 2>&1 | grep -E 'error|Error' -A3 || true
            ls -la "$ROOT/build/xzccsdt_mpi/xzccsdt" ;;
    lib)    cmake -S "$ROOT/csrc" -B "$ROOT/build/clib" -DZC_MKL_DIR=/data/lcheng24/intel/oneapi/mkl/2022.1.0/lib/intel64 > /dev/null
            cmake --build "$ROOT/build/clib" -j8 2>&1 | grep -E 'error|Error' -A3 || true
            ls -la "$ROOT/src/zccsdt/lib" ;;
  esac
done
