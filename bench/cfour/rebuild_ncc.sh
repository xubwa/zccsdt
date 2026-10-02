#!/bin/bash
# Rebuild CFOUR's ncc with other compiler flags in a scratch copy of its source
# (the installation is not touched) and assemble a directory of executables to
# pass as CFOUR_BIN to slurm_cfour.sh: links to the installed programs, with
# xncc replaced.
# usage: rebuild_ncc.sh TAG CXX 'FLAGS' [CFOUR_ROOT]
#   rebuild_ncc.sh gcc  g++  '-O3 -march=native -Drestrict=__restrict__'
#   rebuild_ncc.sh icc  icpc '-O3 -xHost -qopt-zmm-usage=high -restrict'
#   rebuild_ncc.sh prof icpc '-O3 -xHost -qopt-zmm-usage=high -restrict -DPROFILE'
# -DPROFILE turns on ncc's own timers; bench/cfour/ncc_profile.py sums them up.
# NCC_SCRATCH: where to build (default /tmp);  NCC_JOBS: make -j (default 16)
set -e
tag=$1; cxx=$2; flags=$3; root=${4:-/home/xwang405/apps/cfour-x2cso}
b=${NCC_SCRATCH:-/tmp}/zccsdt_nccbuild_$tag
rm -rf $b; mkdir -p $b/bin $b/cfour_bin
cp -r $root/ncc $b/ncc
find $b/ncc -name '*.o' -delete
cp $root/make.config $root/make.rules $b/
for f in include lib config config.h; do ln -s $root/$f $b/$f; done
# only the definitions with literal flags, not the later "cxxldflags:=$(cxxldflags) -L..."
sed -i -E "s|^CXX:=.*|CXX:=$cxx|
           s|^(cxxflags:=)[^$].*|\1$flags -fopenmp -DINT64|
           s|^(cxxldflags:=)[^$].*|\1$flags -fopenmp -DINT64 -lrt|" $b/make.config
(cd $b/ncc && make -j${NCC_JOBS:-16} > ../build.log 2>&1) || { tail -20 $b/build.log; exit 1; }
for f in $root/bin/*; do ln -s $f $b/cfour_bin/$(basename $f); done
rm -f $b/cfour_bin/xncc
cp $b/bin/xncc $b/cfour_bin/xncc
echo "built $b/bin/xncc; use CFOUR_BIN=$b/cfour_bin"
