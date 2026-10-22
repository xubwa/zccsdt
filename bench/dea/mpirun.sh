#!/bin/bash
# mpirun of the oneAPI environment (the MPI binary of cfour/xzccsdt), for the
# test scripts; inside a job step the ranks start on its node (fork)
source /data/lcheng24/intel/oneapi/setvars.sh --force > /dev/null 2>&1
export I_MPI_HYDRA_BOOTSTRAP=${I_MPI_HYDRA_BOOTSTRAP:-fork}
exec mpirun "$@"
