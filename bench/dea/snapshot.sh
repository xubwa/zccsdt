#!/bin/bash
# A snapshot of this checkout for Slurm jobs (the jobs must not see files
# that change while they wait): src/ (with the C kernels built in
# src/zccsdt/lib), bench/, and the native binaries as xzccsdt, xzccsdt_mpi.
#   bash bench/dea/snapshot.sh <dir>
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
dst=$1
mkdir -p "$dst"
cp -r "$ROOT/src" "$ROOT/bench" "$dst/"
cp "$ROOT/build/xzccsdt/xzccsdt" "$dst/xzccsdt"
cp "$ROOT/build/xzccsdt_mpi/xzccsdt" "$dst/xzccsdt_mpi"
git -C "$ROOT" rev-parse --short HEAD > "$dst/HEAD"
git -C "$ROOT" diff --stat > "$dst/DIFF" || true
echo "snapshot $dst ($(cat "$dst/HEAD"))"
