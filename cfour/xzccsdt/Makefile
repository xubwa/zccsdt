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
# integrals with a frozen index for the density stage (xzfrz.F90): CFOUR's own
# transformation, three objects of its program directory x2czvec, and its
# reader of the AO integrals in libx2cint (module cvwint_module of libx2c)
FRZOBJ  := $(addprefix $(CFOUR)/x2czvec/, vtranpsog.o vtran1sparse.o expnd2p.o)
# the EOM solver in complex arithmetic (xzeoms.inc): the sort of the
# eigenpairs and the test of convergence of CFOUR's xx2ceomcc, two objects
# of its program directory x2ceomcc
EOMOBJ  := $(addprefix $(CFOUR)/x2ceomcc/, eomreorder.o eomconv.o)
LIBS    := $(FRZOBJ) $(EOMOBJ) -L$(CFOUR)/lib -lx2cint -llibr -lx2cint $(BLAS) -lrt $(RPATH) $(XLIBS)

# the compiler of the kernels (default: icc, as the rest; an experiment:
#   make KCC=gcc KCFLAGS='-O3 -march=native -std=gnu99 -fopenmp -fcx-limited-range \
#        -DZC_USE_MKL -DMKL_ILP64 -I$$MKLROOT/include' BUILD=../../build/xzccsdt_gcc
# - Intel's OpenMP runtime serves the GNU interface)
KCC     ?= $(CC)
KCFLAGS ?= $(CFLAGS)

# the C kernels that the program calls
KERNELS := ccsdt_t3 t3pack tnpack lambda_t3 qlambda cc3_fly cc3_native cc3_shift cc4_native t4sym
MODS    := xzkern xzopt xzmem xzdata xzdiis xzlkry xzpar xzdist xzfrzd xzfrz xzpland xzeomd xzdead xzdea5d xzampio
SRCS    := xzmaps xzread xzplan xzdrv xzcc_z xzcc_d
INCS    := xztype.h xzkern.inc xzimd.inc xzstep.inc xzsd.inc xzblk.inc xzdiisg.inc xzlkry.inc xzcc.inc xzlam.inc xzlblk.inc xzden.inc xzdfrz.inc xzq.inc xzqs.inc xzeom.inc xzeoms.inc xzcc3.inc xzcc4.inc xzdea.inc xzdeas.inc xzdeam.inc xzdea5.inc xzdea5s.inc xzdea5r.inc xzdea5x.inc

OBJS := $(addprefix $(BUILD)/, $(addsuffix .o, $(MODS) $(SRCS) xzccsdt xzutil xzmpi $(KERNELS)))

all: $(BUILD)/xzccsdt

$(BUILD)/xzccsdt: $(OBJS)
	$(LD) $(FCFLAGS) -o $@ $(OBJS) $(LIBS)

$(BUILD)/%.o: %.F90 $(INCS) | $(BUILD)
	$(FC) $(FCFLAGS) -module $(BUILD) -I. -I$(CFOUR)/libx2c -c $< -o $@

$(BUILD)/xzccsdt.o: xzccsdt.f $(BUILD)/xzpar.o $(BUILD)/xzdist.o | $(BUILD)
	$(FC) $(FCFLAGS) -extend-source -module $(BUILD) -c $< -o $@

$(BUILD)/xzutil.o: xzutil.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/xzmpi.o: xzmpi.c | $(BUILD)
	$(MPICC) $(CFLAGS) $(MPIDEF) -c $< -o $@

$(BUILD)/%.o: $(CSRC)/%.c $(CSRC)/zcred.h | $(BUILD)
	$(KCC) $(KCFLAGS) -c $< -o $@

# the modules before what uses them
$(addprefix $(BUILD)/, $(addsuffix .o, $(SRCS))): $(addprefix $(BUILD)/, $(addsuffix .o, $(MODS)))
$(BUILD)/xzmem.o $(BUILD)/xzdata.o $(BUILD)/xzdiis.o $(BUILD)/xzpar.o: $(BUILD)/xzkern.o
$(BUILD)/xzdata.o: $(BUILD)/xzopt.o
$(BUILD)/xzfrz.o: $(BUILD)/xzdata.o $(BUILD)/xzmem.o $(BUILD)/xzfrzd.o
$(BUILD)/xzpland.o $(BUILD)/xzampio.o: $(BUILD)/xzdata.o
$(BUILD)/xzeomd.o: $(BUILD)/xzdata.o $(BUILD)/xzmem.o $(BUILD)/xzpar.o
$(BUILD)/xzdead.o: $(BUILD)/xzdata.o $(BUILD)/xzkern.o $(BUILD)/xzopt.o $(BUILD)/xzeomd.o $(BUILD)/xzpar.o
$(BUILD)/xzdea5d.o: $(BUILD)/xzdead.o $(BUILD)/xzdata.o $(BUILD)/xzkern.o $(BUILD)/xzpar.o $(BUILD)/xzeomd.o
$(BUILD)/xzdrv.o: $(BUILD)/xzplan.o $(BUILD)/xzdead.o
$(BUILD)/xzplan.o: $(BUILD)/xzeomd.o $(BUILD)/xzdead.o
$(BUILD)/xzdist.o: $(BUILD)/xzdata.o $(BUILD)/xzpar.o $(BUILD)/xzopt.o

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -f $(BUILD)/*.o $(BUILD)/*.mod $(BUILD)/xzccsdt

.PHONY: all clean
