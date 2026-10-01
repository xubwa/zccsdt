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
