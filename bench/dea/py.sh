#!/bin/bash
# python -u with the package of this checkout and socutils on the path
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
export PYTHONPATH=$ROOT/src:$HOME/apps/socutils-git-env:$PYTHONPATH
exec python -u "$@"
