'''The spinor CCSD and CCSD(T) of socutils, copied by the same author as the
starting point of zccsdt's CCSD(T) (~/apps/socutils-git at ffbe691; each file
says what was changed):

- ZCCSD (zccsd.py): integrals transformed out of core, <ab||cd> stored in a
  scratch file and packed on (a>b, e>f) for the ladder.
- DirectZCCSD (zccsd_direct.py): the pp ladder and the <ma||ef> terms in the
  scalar AO basis (no vvvv, no ovvv); ladder='chol' needs socutils.
- gccsd_t.kernel: (T) over virtual (vir_loop) or occupied (occ_loop) triples.

Both classes run on PySCF's CCSD driver (DIIS, convergence) and take a
spinor SCF object such as PySCF's x2c.UHF.
'''

from zccsdt.zccsd.zccsd import ZCCSD
from zccsdt.zccsd.zccsd_direct import DirectZCCSD
