'''Times of the integral step of DirectZCCSD and of the orbital gradient with
a frozen index (zccsd.orbgrad_frozen) at the size of a production run,
without the coupled cluster: random amplitudes, with the symmetry of the
Kramers pairs where the spinors have it.  For times and peaks only - the
amplitudes solve nothing -; the gradient is printed as a checksum, the same
for two trees with the same spinors (ZC_SPINORS of bench/spinors.py).

usage: python bench/profile_frozen.py MOL BASIS NFROZEN [WITH_T 0|1] [REPEAT]
  NFROZEN: frozen occupied spinors; WITH_T: with (random) densities of (T)
environment: ZC_VIRCUT: the virtual spinors above this orbital energy
             (Hartree) frozen as well; ZC_MAXMEM (MB, default 24000);
             ZC_SCF_ONLY=1: stop after the SCF (to make the file of
             ZC_SPINORS); and those of bench/spinors.py
'''
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spinors                                                          # noqa: E402
from profile_iter import MemWatch                                       # noqa: E402

import zccsdt                                                           # noqa: E402
from zccsdt import timing                                               # noqa: E402
from zccsdt.zccsd import DirectZCCSD, kramers, orbgrad_frozen, triples, zvector   # noqa: E402
from zccsdt.zccsd import kernels as K                                   # noqa: E402


def watched(f):
    '''(result, seconds, peak anonymous memory in GB) of f().'''
    watch = MemWatch()
    watch.start()
    t0 = time.perf_counter()
    out = f()
    wall = time.perf_counter() - t0
    watch.stop()
    return out, wall, watch.anon / 1e6


def main():
    mol_name, basis, nfz = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with_t = len(sys.argv) > 4 and sys.argv[4] == '1'
    repeat = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    maxmem = float(os.environ.get('ZC_MAXMEM', 24000))
    mol = spinors.molecule(mol_name, basis, max_memory=maxmem)
    t0 = time.perf_counter()
    mf = spinors.scf(mol, 1e-12, 1e-9)
    print('code %s  threads %s  max_memory %.0f MB' % (os.path.dirname(zccsdt.__file__),
                                                      os.environ.get('OMP_NUM_THREADS', '?'), maxmem))
    print('SCF of %s (or its spinors from a file): %.1f s, converged %s, E %.8f'
          % (os.environ.get('ZC_SCF', 'pyscf'), time.perf_counter() - t0, mf.converged, mf.e_tot),
          flush=True)
    if os.environ.get('ZC_SCF_ONLY') == '1':
        return
    vircut = float(os.environ.get('ZC_VIRCUT', np.inf))
    high = [p for p, e in enumerate(mf.mo_energy) if e > vircut and mf.mo_occ[p] == 0]
    cc = DirectZCCSD(mf, frozen=list(range(nfz)) + high if high else nfz)
    cc.verbose, cc.max_memory = 0, maxmem
    timing.reset()
    eris, wall, peak = watched(cc.ao2mo)
    o, v = eris.nocc, eris.fock.shape[0] - eris.nocc
    npo, npv = K.npair(o), K.npair(v)
    print('%s/%s  %d + %d active spinors, %d + %d frozen  Kramers pairs: %s'
          % (mol_name, basis, o, v, nfz, len(high), bool(eris.kramers)))
    # ('ao2mo.fill', the AO integrals, is a part of 'ao2mo.bra')
    print('integrals: %.1f s, peak anonymous memory %.2f GB;  %s'
          % (wall, peak, '  '.join('%s %.1f' % (k, w) for k, w, _ in timing.report() if w > 0.02 * wall)),
          flush=True)
    rng = np.random.default_rng(11)

    def random(shape, pairs=()):
        x = 0.02 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
        if eris.kramers:
            x = 0.5 * (x + kramers.image(x, pairs=pairs))
        return np.ascontiguousarray(x)

    t1, l1 = random((o, v)), random((o, v))
    t2p, l2p = random((npo, npv), (0, 1)), random((npo, npv), (0, 1))
    dens = None
    if with_t:
        dens = triples.Densities(random((o, v, npv), (2,)), random((o, v, npo), (2,)),
                                 random((npo, npv), (0, 1)))
    C, mo_e, nocc, fc = zvector.full_space(cc)
    held = np.concatenate((fc, zvector.frozen_virtuals(cc))).astype(int)
    Cf = np.vstack(mol.sph2spinor_coeff()) @ C[:, held]
    for _ in range(repeat):
        timing.reset()
        X, wall, peak = watched(lambda: orbgrad_frozen.kernel(cc, eris, t1, t2p, l1, l2p, Cf, dens,
                                                             mo_energy=mo_e[held]))
        print('gradient with a frozen index: %.2f s, peak anonymous memory %.2f GB  '
              '(largest element %.6e, sum of the moduli %.10e)' % (wall, peak, abs(X).max(), abs(X).sum()),
              flush=True)
        for name, w, calls in timing.report():
            if w > 0.005 * wall:
                print('    %-12s %7.2f s  %5.1f %%  (%d calls)' % (name, w, 100 * w / wall, calls))


if __name__ == '__main__':
    main()
