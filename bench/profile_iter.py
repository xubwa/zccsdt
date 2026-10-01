'''Per-term timing of CCSDT iterations.

usage: python bench/profile_iter.py MOLECULE BASIS KIND [NITER] [FROZEN]
  MOLECULE: h2o | n2 | h2s | nh2 | o2     KIND: ghf | x2c | uhf
  (uhf: U-form integrals straight from a UHF reference, UCCSDT)
environment: ZC_PACKED=1 builds the integrals block-wise with packed vvvv
             (needed for large bases); ZC_MEM_MB sets the batch buffer budget;
             ZC_SPIN=1 (ghf only) uses alpha-first orbitals and the spin-blocked
             T3 engine (UCCSDT); ZC_FLIP=1 adds closed-shell spin flip (RCCSDT);
             ZC_RU=1 (ghf only) builds R-form integrals straight from the RHF
             (eris_spin, no GHF step).
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
    'h2s': 'S 0 0 0; H 0 0.9618 0.9272; H 0 -0.9618 0.9272',
    'nh2': 'N 0 0 0; H 0 0.8036 0.6347; H 0 -0.8036 0.6347',
    'o2': 'O 0 0 0; O 0 0 1.2075',
}
SPIN = {'nh2': 1, 'o2': 2}       # 2S of the open-shell molecules


def main():
    mol_name, basis, kind = sys.argv[1:4]
    niter = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    frozen = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    mol = gto.M(atom=GEOM[mol_name], basis=basis, spin=SPIN.get(mol_name, 0), verbose=0)
    ru = os.environ.get('ZC_RU', '0') == '1' and kind == 'ghf'
    if kind == 'uhf':
        mf = scf.UHF(mol).run(conv_tol=1e-10)
    elif kind == 'x2c':
        mf = x2c.UHF(mol).run(conv_tol=1e-10)
    elif ru:
        mf = scf.RHF(mol).run(conv_tol=1e-10)
    else:
        mf = scf.addons.convert_to_ghf(scf.RHF(mol).run(conv_tol=1e-10))
    t0 = time.perf_counter()
    packed = os.environ.get('ZC_PACKED', '0') == '1'
    spin = os.environ.get('ZC_SPIN', '0') == '1' and kind == 'ghf'
    if kind == 'uhf':
        from zccsdt import eris_spin
        eris = eris_spin.from_scf(mf, frozen=frozen,
                                  vir_batch=int(os.environ.get('ZC_VBATCH', 16)))
    elif ru:
        from zccsdt import eris_spin
        eris = eris_spin.from_scf(mf, frozen=frozen,
                                  vir_batch=int(os.environ.get('ZC_VBATCH', 16)))
    else:
        eris = E.from_scf(mf, frozen=frozen, packed_vvvv=packed,
                          vir_batch=int(os.environ.get('ZC_VBATCH', 16)), spin_order=spin)
    t_ints = time.perf_counter() - t0
    nocc = eris.nocc
    nvir = eris.nmo - nocc
    print('%s/%s %s  nocc=%d nvir=%d dtype=%s  integrals %.1fs'
          % (mol_name, basis, kind, nocc, nvir, eris.dtype, t_ints))
    kw = dict(verbose=True, mem_mb=float(os.environ.get('ZC_MEM_MB', 2000)),
              spin_flip=os.environ.get('ZC_FLIP', '0') == '1')
    # one warm-up iteration (engine setup, caches, buffers), then the timed ones
    r0 = solver.kernel(eris, 'ccsdt', solver.Options(max_cycle=1, **kw))
    timing.reset()
    r = solver.kernel(eris, 'ccsdt', solver.Options(max_cycle=niter, **kw),
                      t1=r0.t1, t2=r0.t2, t3=r0.t3)
    per = len(r.timings)
    print('iterations: ' + ' '.join('%.2f' % t for t in r.timings))
    print('mean iteration %.3fs (warm-up iteration %.3fs; timers below exclude it)'
          % (np.mean(r.timings), r0.timings[0]))
    for name, sec, calls in timing.report(per):
        print('  %-14s %9.3fs  %5.1f calls' % (name, sec, calls))
    print('peak RSS %.2f GB' % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))


if __name__ == '__main__':
    main()
