'''The spinor CCSD and CCSD(T) of socutils, copied by the same author as the
starting point of zccsdt's CCSD(T) (~/apps/socutils-git at ffbe691; each file
says what was changed):

- ZCCSD (zccsd.py): pp ladder in the MO basis on <ab||ef> packed over
  (a>b, e>f).
- DirectZCCSD (zccsd_direct.py): pp ladder in the scalar AO basis (no vvvv);
  <ma||ef> stored, or its terms from the AO integrals (ovvv_direct);
  ladder='chol' needs socutils.
- gccsd_t.kernel: (T) over virtual (vir_loop) or occupied (occ_loop) triples.
- ao2mo: the integrals of both classes for a two-component reference, slabs
  of the first MO index from the packed scalar AO integrals (the route of
  socutils through nrr_outcore stays as ZCCSD.ao2mo_outcore).

Both classes run on PySCF's CCSD driver (DIIS, convergence) and take a
spinor SCF object such as PySCF's x2c.UHF.
'''

from zccsdt.zccsd.zccsd import ZCCSD
from zccsdt.zccsd.zccsd_direct import DirectZCCSD
