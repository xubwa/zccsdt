#
# x2cprobe: hbarso of CFOUR's xx2cccsd (libx2ccc) against the CCSD part of
# the native CCSDT step, in one executable.  docs/native_x2ccc.md.
#
#   source /data/lcheng24/intel/oneapi/setvars.sh --force
#   make -C ../xzccsdt        (the native objects and modules: build/xzccsdt)
#   make                      -> build/x2ccc_probe/x2cprobe
#
# Against a native program whose xzimd.inc has the argument lists after
# commit 6d73752 (xzvvoo, xzwo, xzwpair):
#   make NATIVE=<its build directory> BUILD=<another> PDEFS=-DXZ_NEWIMD
#
# CFOUR is linked, never written to and never copied: hbarso.o and the
# program-directory objects it needs from $(CFOUR)/x2cccsd, the libraries
# libx2ccc.a, libx2cint.a, liblibr.a from $(CFOUR)/lib, the module files of
# $(CFOUR)/x2cccsd (soccmem), libx2ccc (soccmod) and libx2c (kinds, ...).
#
CFOUR   ?= $(HOME)/apps/cfour-x2cso-rebuild
NATIVE  ?= ../../build/xzccsdt
BUILD   ?= ../../build/x2ccc_probe
MKLROOT ?= /data/lcheng24/intel/oneapi/mkl/2022.1.0
MKLLIB  ?= $(MKLROOT)/lib/intel64
PDEFS   ?=

FC      := ifort
# the flags of the native program for what is compiled here
FCFLAGS := -O3 -xHost -cpp -assume byterecl -assume buffered_io -i8 -qopenmp -mcmodel=medium -DINT64
BLAS    := -L$(MKLLIB) -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
ICLIB   ?= $(dir $(shell which ifort 2>/dev/null))../../compiler/lib/intel64_lin
RPATH   := -Wl,-rpath,$(MKLLIB) -Wl,-rpath,$(abspath $(ICLIB))

# the native program without its main program
NOBJ := $(addprefix $(NATIVE)/, xzkern.o xzmem.o xzdata.o xzdiis.o xzpar.o xzmaps.o xzread.o \
          xzplan.o xzdrv.o xzcc_z.o xzcc_d.o xzutil.o xzmpi.o ccsdt_t3.o t3pack.o tnpack.o \
          lambda_t3.o qlambda.o)

# xx2cccsd: the residual routine and what it calls that is not in libx2ccc
# (t2abcdaoso, the AO-driven ladder, is replaced by a stub in cfside.F90)
COBJ := $(addprefix $(CFOUR)/x2cccsd/, hbarso.o soccmem.o cvsmasksd1.o psmasksd.o conmasksd.o \
          convirmask.o oldgradmask.o pgmaskcount0.o pgsymmasksd0.o t2abcdmoso.o t2abcdso.o \
          t2ftaaso2.o t2ftiiso2.o fia2r1.o r1updso.o r2updso.o t1normso.o t2normso.o \
          ppphreadp.o wppphfname.o vreadp.o vreadf.o ztranspiv.o)

CLIBS := -L$(CFOUR)/lib -lx2ccc -lx2cint -llibr

OBJS := $(BUILD)/x2cprobe.o $(BUILD)/probe.o $(BUILD)/cfside.o

all: $(BUILD)/x2cprobe

$(BUILD)/x2cprobe: $(OBJS) $(NOBJ)
	$(FC) $(FCFLAGS) -o $@ $(OBJS) $(NOBJ) $(COBJ) $(CLIBS) $(BLAS) -lrt $(RPATH)

$(BUILD)/probe.o: probe.F90 | $(BUILD)
	$(FC) $(FCFLAGS) $(PDEFS) -module $(BUILD) -I$(NATIVE) -c $< -o $@

$(BUILD)/cfside.o: cfside.F90 | $(BUILD)
	$(FC) $(FCFLAGS) -module $(BUILD) -I$(CFOUR)/x2cccsd -I$(CFOUR)/libx2ccc -I$(CFOUR)/libx2c -c $< -o $@

$(BUILD)/x2cprobe.o: x2cprobe.f | $(BUILD)
	$(FC) $(FCFLAGS) -extend-source -c $< -o $@

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -f $(BUILD)/*.o $(BUILD)/*.mod $(BUILD)/x2cprobe

.PHONY: all clean
