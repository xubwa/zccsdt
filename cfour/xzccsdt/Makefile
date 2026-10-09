#
# xzccsdt: spinor CCSDT / CCSDT(Q) as a member of CFOUR's two-component chain.
#
#   make            builds $(BUILD)/xzccsdt        (needs ifort, icc, MKL:
#                   source the oneAPI environment first)
#   make clean
#
# The Fortran sources are here; the C kernels are those of csrc/ (KERNELS),
# compiled for the BLAS integers of the CFOUR build (ILP64).  CFOUR is only
# linked: its libraries are taken from $(CFOUR)/lib, nothing is written there.
#
CFOUR   ?= $(HOME)/apps/cfour-x2cso-rebuild
BUILD   ?= ../../build/xzccsdt
CSRC    := ../../csrc
MKLROOT ?= /data/lcheng24/intel/oneapi/mkl/2022.1.0
MKLLIB  ?= $(MKLROOT)/lib/intel64

FC      := ifort
CC      := icc
# MPI=1: (Q) parallel over ranks with Intel MPI (the wrappers of the oneAPI
# environment); the default is the serial program, which has no MPI in it
ifeq ($(MPI),1)
  BUILD  := ../../build/xzccsdt_mpi
  LD     := mpiifort
  MPICC  := mpiicc
  MPIDEF := -DXZ_MPI
else
  LD     := $(FC)
  MPICC  := $(CC)
  MPIDEF :=
endif
# the flags of CFOUR's x2c programs (make.config: -i8, byte record lengths)
FCFLAGS := -O3 -xHost -cpp -assume byterecl -assume buffered_io -i8 -qopenmp -mcmodel=medium -DINT64
CFLAGS  := -O3 -xHost -std=c99 -qopenmp -complex-limited-range -DZC_USE_MKL -DMKL_ILP64 \
           -I$(MKLROOT)/include
BLAS    := -L$(MKLLIB) -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 \
           -lpthread -lm -ldl
# (the run-time libraries are found without the oneAPI environment)
ICLIB   ?= $(dir $(shell which ifort 2>/dev/null))../../compiler/lib/intel64_lin
RPATH   := -Wl,-rpath,$(MKLLIB) -Wl,-rpath,$(abspath $(ICLIB))
LIBS    := -L$(CFOUR)/lib -llibr $(BLAS) -lrt $(RPATH) $(XLIBS)

# the compiler of the kernels (default: icc, as the rest; an experiment:
#   make KCC=gcc KCFLAGS='-O3 -march=native -std=gnu99 -fopenmp -fcx-limited-range \
#        -DZC_USE_MKL -DMKL_ILP64 -I$$MKLROOT/include' BUILD=../../build/xzccsdt_gcc
# - Intel's OpenMP runtime serves the GNU interface)
KCC     ?= $(CC)
KCFLAGS ?= $(CFLAGS)

# the C kernels that the program calls
KERNELS := ccsdt_t3 t3pack tnpack lambda_t3 qlambda
MODS    := xzkern xzmem xzdata xzdiis xzpar
SRCS    := xzmaps xzread xzplan xzdrv xzcc_z xzcc_d
INCS    := xztype.h xzkern.inc xzimd.inc xzstep.inc xzblk.inc xzdiisg.inc xzcc.inc xzlam.inc xzlblk.inc xzden.inc xzq.inc xzqs.inc

OBJS := $(addprefix $(BUILD)/, $(addsuffix .o, $(MODS) $(SRCS) xzccsdt xzutil xzmpi $(KERNELS)))

all: $(BUILD)/xzccsdt

$(BUILD)/xzccsdt: $(OBJS)
	$(LD) $(FCFLAGS) -o $@ $(OBJS) $(LIBS)

$(BUILD)/%.o: %.F90 $(INCS) | $(BUILD)
	$(FC) $(FCFLAGS) -module $(BUILD) -I. -c $< -o $@

$(BUILD)/xzccsdt.o: xzccsdt.f $(BUILD)/xzpar.o | $(BUILD)
	$(FC) $(FCFLAGS) -extend-source -module $(BUILD) -c $< -o $@

$(BUILD)/xzutil.o: xzutil.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/xzmpi.o: xzmpi.c | $(BUILD)
	$(MPICC) $(CFLAGS) $(MPIDEF) -c $< -o $@

$(BUILD)/%.o: $(CSRC)/%.c | $(BUILD)
	$(KCC) $(KCFLAGS) -c $< -o $@

# the modules before what uses them
$(addprefix $(BUILD)/, $(addsuffix .o, $(SRCS))): $(addprefix $(BUILD)/, $(addsuffix .o, $(MODS)))
$(BUILD)/xzmem.o $(BUILD)/xzdata.o $(BUILD)/xzdiis.o $(BUILD)/xzpar.o: $(BUILD)/xzkern.o

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -f $(BUILD)/*.o $(BUILD)/*.mod $(BUILD)/xzccsdt

.PHONY: all clean
