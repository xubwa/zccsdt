'''The spinor CCSD and CCSD(T) of socutils, copied by the same author as the
starting point of zccsdt's CCSD(T) (~/apps/socutils-git at ffbe691; each file
says what was changed):

- ZCCSD (zccsd.py): pp ladder in the MO basis on <ab||ef> packed over
  (a>b, e>f).
- DirectZCCSD (zccsd_direct.py): pp ladder in the scalar AO basis (no vvvv);
  <ma||ef> stored, or its terms from the AO integrals (ovvv_direct);
  ladder='chol' needs socutils.
- gccsd_t.kernel: (T) over virtual (vir_loop) or occupied (occ_loop) triples.
- triples.kernel: (T) of DirectZCCSD in batches of virtual pairs, GEMMs on
  the packed <ie||bc> and a kernel that assembles the triples of one pair
  in the cache (csrc/zccsd_t.c).
- nrr_fast: the spinor AO->MO transformation of socutils (lib/ao2mo), both
  passes in C with the s4 symmetry of the AO integrals (csrc/nrr_ao2mo_opt.c,
  built into lib/libnrr_opt.so against the installed PySCF by
  `cmake -S csrc -B build && cmake --build build`).
- amps: the amplitude equations and the iterations of DirectZCCSD on packed
  amplitudes, [(i>j), (a>b)] throughout, sliced within max_memory (the
  equations of socutils' update_amps, which stays on PySCF's driver as
  DirectZCCSD.socutils_amps).
- kernels: the OpenMP kernels of amps (csrc/zccsd_kernels.c, built into
  lib/libzc_zccsd.so) with their numpy equivalents.
- aoladder: the two AO operators of the ladder of amps, kept as triangles
  or made from the AO integrals in every iteration (integral-direct).
- ao2mo: the integrals of both classes for a two-component reference on
  nrr_fast, within max_memory: slabs of the first MO index that are
  antisymmetrized into their blocks at once, the blocks that do not fit
  mapped from a scratch file (the route of socutils' cc through nrr_outcore
  stays as ZCCSD.ao2mo_outcore).
- lam: the Lambda equations of DirectZCCSD on packed amplitudes, in the
  layouts and within the memory plan of amps (solve_lambda).
- lam_t, rdm, zvector: the (T) terms of Lambda, the one-particle density and
  dipole moment (make_rdm1, dip_moment), the Z-vector equations of the
  orbital relaxation.  The (T) terms and the orbital gradient are references
  on full arrays for small molecules (lam_t and rdm from socutils' branch
  claude/ccsd-lambda-socutils, 8965b9f).

Both classes take a spinor SCF object such as PySCF's x2c.UHF and follow
PySCF's CCSD settings (conv_tol, diis_space, ...); max_memory is PySCF's (MB
for the whole process, 4000 by default).  For the speed of DirectZCCSD the
kernel libraries should be built with MKL on GNU OpenMP (csrc/CMakeLists.txt,
ZC_MKL_DIR): kernels and BLAS then share one pool of threads.
'''

from zccsdt.zccsd.zccsd import ZCCSD
from zccsdt.zccsd.zccsd_direct import DirectZCCSD
