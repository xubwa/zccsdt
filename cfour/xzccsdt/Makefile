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

FC      := ifort
CC      := icc
# the flags of CFOUR's x2c programs (make.config: -i8, byte record lengths)
FCFLAGS := -O3 -xHost -cpp -assume byterecl -assume buffered_io -i8 -qopenmp -mcmodel=medium -DINT64
CFLAGS  := -O3 -xHost -std=c99 -qopenmp -complex-limited-range -DZC_USE_MKL -DMKL_ILP64 \
           -I$(MKLROOT)/include
BLAS    := -L$(MKLROOT)/lib/intel64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 \
           -lpthread -lm -ldl
# (the run-time libraries are found without the oneAPI environment)
ICLIB   ?= $(dir $(shell which ifort 2>/dev/null))../../compiler/lib/intel64_lin
RPATH   := -Wl,-rpath,$(MKLROOT)/lib/intel64 -Wl,-rpath,$(abspath $(ICLIB))
LIBS    := -L$(CFOUR)/lib -llibr $(BLAS) -lrt $(RPATH)

# the C kernels that the program calls
KERNELS := ccsdt_t3 t3pack tnpack
MODS    := xzkern xzmem xzdata xzdiis
SRCS    := xzmaps xzread xzplan xzdrv xzcc_z xzcc_d
INCS    := xztype.h xzkern.inc xzimd.inc xzstep.inc xzblk.inc xzdiisg.inc xzcc.inc xzq.inc

OBJS := $(addprefix $(BUILD)/, $(addsuffix .o, $(MODS) $(SRCS) xzccsdt xzutil $(KERNELS)))

all: $(BUILD)/xzccsdt

$(BUILD)/xzccsdt: $(OBJS)
	$(FC) $(FCFLAGS) -o $@ $(OBJS) $(LIBS)

$(BUILD)/%.o: %.F90 $(INCS) | $(BUILD)
	$(FC) $(FCFLAGS) -module $(BUILD) -I. -c $< -o $@

$(BUILD)/xzccsdt.o: xzccsdt.f | $(BUILD)
	$(FC) $(FCFLAGS) -extend-source -c $< -o $@

$(BUILD)/xzutil.o: xzutil.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/%.o: $(CSRC)/%.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

# the modules before what uses them
$(addprefix $(BUILD)/, $(addsuffix .o, $(SRCS))): $(addprefix $(BUILD)/, $(addsuffix .o, $(MODS)))
$(BUILD)/xzmem.o $(BUILD)/xzdata.o $(BUILD)/xzdiis.o: $(BUILD)/xzkern.o

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -f $(BUILD)/*.o $(BUILD)/*.mod $(BUILD)/xzccsdt

.PHONY: all clean
