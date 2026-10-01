'''Per-term timing of CCSDT iterations.

usage: python bench/profile_iter.py MOLECULE BASIS KIND [NITER] [FROZEN]
  MOLECULE: h2o | n2     KIND: ghf | x2c
environment: ZC_PACKED=1 builds the integrals block-wise with packed vvvv
             (needed for large bases); ZC_MEM_MB sets the batch buffer budget.
'''
import os
import sys
import time
import resource
import numpy as np
from pyscf import gto, scf, x2c

from zccsdt import eris as E, timing
from zccsdt.cc import solver

GEOM = {
    'h2o': 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
    'n2': 'N 0 0 0; N 0 0 1.0977',
}


def main():
    mol_name, basis, kind = sys.argv[1:4]
    niter = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    frozen = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    mol = gto.M(atom=GEOM[mol_name], basis=basis, verbose=0)
    if kind == 'x2c':
        mf = x2c.UHF(mol).run(conv_tol=1e-10)
    else:
        mf = scf.addons.convert_to_ghf(scf.RHF(mol).run(conv_tol=1e-10))
    t0 = time.perf_counter()
    packed = os.environ.get('ZC_PACKED', '0') == '1'
    eris = E.from_scf(mf, frozen=frozen, packed_vvvv=packed)
    t_ints = time.perf_counter() - t0
    nocc = eris.nocc
    nvir = eris.nmo - nocc
    print('%s/%s %s  nocc=%d nvir=%d dtype=%s  integrals %.1fs'
          % (mol_name, basis, kind, nocc, nvir, eris.dtype, t_ints))
    timing.reset()
    opts = solver.Options(max_cycle=niter, verbose=True,
                          mem_mb=float(os.environ.get('ZC_MEM_MB', 2000)))
    r = solver.kernel(eris, 'ccsdt', opts)
    per = len(r.timings)
    print('iterations: ' + ' '.join('%.2f' % t for t in r.timings))
    print('mean iteration %.3fs (first %.3fs)' % (np.mean(r.timings[1:] or r.timings),
                                                  r.timings[0]))
    for name, sec, calls in timing.report(per):
        print('  %-14s %9.3fs  %5.1f calls' % (name, sec, calls))
    print('peak RSS %.2f GB' % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))


if __name__ == '__main__':
    main()
