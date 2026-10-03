'''The spinor CCSD and CCSD(T) of socutils, copied by the same author as the
starting point of zccsdt's CCSD(T) (~/apps/socutils-git at ffbe691; each file
says what was changed):

- ZCCSD (zccsd.py): pp ladder in the MO basis on <ab||ef> packed over
  (a>b, e>f).
- DirectZCCSD (zccsd_direct.py): pp ladder in the scalar AO basis (no vvvv);
  <ma||ef> stored, or its terms from the AO integrals (ovvv_direct);
  ladder='chol' needs socutils.
- gccsd_t.kernel: (T) over virtual (vir_loop) or occupied (occ_loop) triples.
- nrr_fast: the spinor AO->MO transformation of socutils (lib/ao2mo), both
  passes in C with the s4 symmetry of the AO integrals (csrc/nrr_ao2mo_opt.c,
  built into lib/libnrr_opt.so against the installed PySCF by
  `cmake -S csrc -B build && cmake --build build`).
- amps: the amplitude equations of DirectZCCSD on packed pairs and fixed
  layouts (the equations of socutils' update_amps, which stays as
  DirectZCCSD.socutils_amps; kernels in csrc/zccsd_kernels.c).
- ao2mo: the integrals of both classes for a two-component reference on
  nrr_fast, within max_memory: slabs of the first MO index that are
  antisymmetrized into their blocks at once, the blocks that do not fit
  mapped from a scratch file (the route of socutils' cc through nrr_outcore
  stays as ZCCSD.ao2mo_outcore).

Both classes run on PySCF's CCSD driver (DIIS, convergence) and take a
spinor SCF object such as PySCF's x2c.UHF; max_memory is PySCF's (MB for the
whole process, 4000 by default).
'''

from zccsdt.zccsd.zccsd import ZCCSD
from zccsdt.zccsd.zccsd_direct import DirectZCCSD
