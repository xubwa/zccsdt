#!/usr/bin/env python
'''
Open-shell CCSDT on a UHF reference (spin blocks, U-form integrals), with a
memory budget: what does not fit goes to scratch files (ZCCSDT_TMPDIR).
'''

from pyscf import gto, scf
import zccsdt

mol = gto.M(atom='N 0 0 0; H 0 0.8036 0.6347; H 0 -0.8036 0.6347', basis='cc-pvtz',
            spin=1, verbose=4)
mf = scf.UHF(mol).run()

mycc = zccsdt.UCCSDT(mf, frozen=1)  # one core orbital per spin
mycc.max_memory = 4000              # MB for the whole process; default: 80% of what
                                    # the node and the job (Slurm cgroup) leave
mycc.run()
