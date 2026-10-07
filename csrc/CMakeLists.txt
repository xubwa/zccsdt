# OpenMP C kernels for the CCSDT residual (signed antisymmetric pack/unpack
# and permutation antisymmetrizers).  Ported from socutils lib/ccsdt.
cmake_minimum_required(VERSION 3.16)
project(zccsdt_clib C)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_C_STANDARD 99)
find_package(OpenMP REQUIRED)
add_library(ccsdt_clib SHARED ccsdt_pack.c ccsdt_asym.c ccsdt_asym_pack.c)
target_compile_options(ccsdt_clib PRIVATE -O3 -march=native)
target_link_libraries(ccsdt_clib PUBLIC OpenMP::OpenMP_C m)
set_target_properties(ccsdt_clib PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src/zccsdt/lib)

# packed-T3 gather / antisymmetrizing-reduction kernels (double and complex)
add_library(zc_t3pack SHARED t3pack.c)
target_compile_options(zc_t3pack PRIVATE -O3 -march=native)
target_link_libraries(zc_t3pack PUBLIC OpenMP::OpenMP_C m)
set_target_properties(zc_t3pack PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src/zccsdt/lib)

# packed antisymmetric tuples of any rank: expansion into two groups of
# indices and antisymmetrizing reduction (zccsdt/cc/t4pack.py)
add_library(zc_tnpack SHARED tnpack.c)
target_compile_options(zc_tnpack PRIVATE -O3 -march=native -fcx-limited-range)
target_link_libraries(zc_tnpack PUBLIC OpenMP::OpenMP_C m)
set_target_properties(zc_tnpack PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src/zccsdt/lib)

# (P) in symmetry sectors: rows of packed quadruples on the classes of the
# labels and the sum over the virtual pentuples of a sector
# (zccsdt/cc/pentuples_sym.py)
add_library(zc_psym SHARED psym.c)
target_compile_options(zc_psym PRIVATE -O3 -march=native -fcx-limited-range)
target_link_libraries(zc_psym PUBLIC OpenMP::OpenMP_C m)
set_target_properties(zc_psym PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src/zccsdt/lib)

# kernels of the spinor CCSD amplitude equations (zccsdt/zccsd/amps.py) and of
# its (T) correction (zccsdt/zccsd/triples.py)
add_library(zc_zccsd SHARED zccsd_kernels.c zccsd_t.c)
target_compile_options(zc_zccsd PRIVATE -O3 -march=native -fcx-limited-range)
target_link_libraries(zc_zccsd PUBLIC OpenMP::OpenMP_C m)
set_target_properties(zc_zccsd PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src/zccsdt/lib)

# T3 part of the CCSDT iteration (passes, ladders, update); needs CBLAS.
set(ZC_BLAS_LIB "$ENV{CONDA_PREFIX}/lib/libopenblas.so" CACHE FILEPATH "CBLAS library")
set(ZC_BLAS_INC "$ENV{CONDA_PREFIX}/include" CACHE PATH "directory with cblas.h")
add_library(zc_ccsdt_t3 SHARED ccsdt_t3.c)
target_include_directories(zc_ccsdt_t3 PRIVATE ${ZC_BLAS_INC})
target_compile_options(zc_ccsdt_t3 PRIVATE -O3 -march=native)
target_link_libraries(zc_ccsdt_t3 PUBLIC OpenMP::OpenMP_C ${ZC_BLAS_LIB} m)
get_filename_component(ZC_BLAS_DIR ${ZC_BLAS_LIB} DIRECTORY)
set_target_properties(zc_ccsdt_t3 PROPERTIES
  BUILD_RPATH ${ZC_BLAS_DIR} INSTALL_RPATH ${ZC_BLAS_DIR}
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src/zccsdt/lib)

# Optional MKL build of the same kernels (GNU OpenMP threading layer, so BLAS
# and the OpenMP loops share one thread pool).  Select it at run time with
# ZCCSDT_T3_LIB=.../libzc_ccsdt_t3_mkl.so
set(ZC_MKL_DIR "" CACHE PATH "MKL lib/intel64 directory (empty: skip)")
if(ZC_MKL_DIR)
  add_library(zc_ccsdt_t3_mkl SHARED ccsdt_t3.c)
  target_compile_definitions(zc_ccsdt_t3_mkl PRIVATE ZC_USE_MKL)
  target_include_directories(zc_ccsdt_t3_mkl PRIVATE ${ZC_BLAS_INC})
  target_compile_options(zc_ccsdt_t3_mkl PRIVATE -O3 -march=native)
  target_link_libraries(zc_ccsdt_t3_mkl PUBLIC
    ${ZC_MKL_DIR}/libmkl_intel_lp64.so ${ZC_MKL_DIR}/libmkl_gnu_thread.so
    ${ZC_MKL_DIR}/libmkl_core.so OpenMP::OpenMP_C m dl)
  set_target_properties(zc_ccsdt_t3_mkl PROPERTIES
    BUILD_RPATH ${ZC_MKL_DIR} INSTALL_RPATH ${ZC_MKL_DIR}
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src/zccsdt/lib)
endif()

# Spinor AO->MO kernels of socutils (lib/ao2mo/nrr_ao2mo_opt.c): the e1 pass
# on PySCF's s4 AO fill and the e2 pass, for zccsdt.zccsd.nrr_fast.  The
# library reuses AO2MOfill_nr_s4, AO2MOmmm_r_iltj, NPdunpack_tril ... of the
# PySCF that the configuring Python imports and links against its libraries
# (and, through them, its BLAS): rebuild after changing PySCF.  C99 keeps the
# K&R-style `(*fmmm)()` declarations valid (gcc >= 14 defaults to C23).
find_program(ZC_PYTHON NAMES python3 python)
execute_process(
  COMMAND ${ZC_PYTHON} -c
          "from pyscf import lib; import os; print(os.path.dirname(os.path.abspath(lib.__file__)))"
  OUTPUT_VARIABLE ZC_PYSCF_LIB OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE ZC_PYSCF_RC ERROR_QUIET)
if(ZC_PYSCF_RC EQUAL 0 AND ZC_PYSCF_LIB)
  find_library(ZC_AO2MO_LIB     NAMES ao2mo     PATHS ${ZC_PYSCF_LIB} NO_DEFAULT_PATH)
  find_library(ZC_NP_HELPER_LIB NAMES np_helper PATHS ${ZC_PYSCF_LIB} NO_DEFAULT_PATH)
  find_library(ZC_CVHF_LIB      NAMES cvhf      PATHS ${ZC_PYSCF_LIB} NO_DEFAULT_PATH)
  find_library(ZC_CINT_LIB      NAMES cint      PATHS ${ZC_PYSCF_LIB}/deps/lib NO_DEFAULT_PATH)
endif()
if(ZC_AO2MO_LIB AND ZC_NP_HELPER_LIB AND ZC_CVHF_LIB AND ZC_CINT_LIB)
  message(STATUS "nrr_opt: linking against PySCF at ${ZC_PYSCF_LIB}")
  add_library(nrr_opt SHARED nrr_ao2mo_opt.c)
  target_include_directories(nrr_opt PRIVATE ${ZC_PYSCF_LIB} ${ZC_PYSCF_LIB}/ao2mo
                             ${ZC_PYSCF_LIB}/deps/include)
  target_compile_options(nrr_opt PRIVATE -O3)
  target_link_libraries(nrr_opt PRIVATE ${ZC_AO2MO_LIB} ${ZC_NP_HELPER_LIB} ${ZC_CVHF_LIB}
                        ${ZC_CINT_LIB} OpenMP::OpenMP_C m)
  set_target_properties(nrr_opt PROPERTIES
    BUILD_RPATH "${ZC_PYSCF_LIB};${ZC_PYSCF_LIB}/deps/lib"
    INSTALL_RPATH "${ZC_PYSCF_LIB};${ZC_PYSCF_LIB}/deps/lib"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src/zccsdt/lib)
else()
  message(WARNING "nrr_opt: PySCF libraries not found through ${ZC_PYTHON}; "
                  "libnrr_opt.so (integrals of zccsdt.zccsd) is not built")
endif()
