'''Profile of the dense X2Ccorr route of o2_x2ccorr.py (before zccsdt.eri4c):
time and anonymous memory of every ao2mo.general call inside
socutils.tools.fcidump_rel.x2c_active_eri_4c and of the steps after it
(complex -> real copy, the HF in the correlated space, the rotation by
tensordot, savez is not repeated).

usage: python x2ccorr_profile.py <dir> --basis <name> [--skip-hf]

RssAnon of /proc/self/status is sampled every 0.5 s by a thread; each step
prints its time, RssAnon at its end and the largest RssAnon during it.'''
import os
import sys
import threading
import time

import numpy as np
from pyscf import ao2mo, lib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SKIP_HF = '--skip-hf' in sys.argv
if SKIP_HF:
    sys.argv.remove('--skip-hf')
import o2_scf  # noqa: E402
import o2_x2ccorr  # noqa: E402


def rss_anon():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('RssAnon'):
                return int(line.split()[1]) * 1024
    return 0


class Sampler:
    def __init__(self):
        self.peak = 0
        self.stop = False
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while not self.stop:
            self.peak = max(self.peak, rss_anon())
            time.sleep(0.5)

    def step(self, what, t0):
        now = rss_anon()
        print('%-58s %8.1f s   RssAnon %7.2f GB   peak in step %7.2f GB' % (what, time.time() - t0, now / 1e9,
                                                                         max(self.peak, now) / 1e9), flush=True)
        self.peak = now


S = Sampler()
_general = ao2mo.general


def general(mol, mos, *args, **kw):
    t0 = time.time()
    out = _general(mol, mos, *args, **kw)
    S.step('  ao2mo.general %s' % kw.get('intor', args[0] if args else ''), t0)
    return out


ao2mo.general = general


def main():
    outdir = sys.argv[1]
    t0 = time.time()
    mf = o2_scf.run_scf(os.path.join(outdir, 'o2_2plus_krhf_x2cmp%s.chk' % o2_scf.TAG), verbose=0)
    C, om2, par, io, iv, e = o2_scf.spinors(mf)
    nf = o2_scf.NFROZEN
    cols = np.concatenate((io[nf:], iv))
    c, lab, pr = C[:, cols], om2[cols], par[cols]
    no, n = len(io) - nf, len(cols)
    print('%d correlated spinors, %d threads' % (n, lib.num_threads()), flush=True)
    S.step('SCF from the checkpoint', t0)
    t0 = time.time()
    hc, ecore = o2_x2ccorr.frozen_field(mf, C[:, io[:nf]])
    S.step('frozen-core four-component mean field', t0)
    t0 = time.time()
    h1, eri = o2_x2ccorr.assemble(mf, hc, c)
    S.step('x2c_active_eri_4c (all classes)', t0)
    t0 = time.time()
    h1, eri = h1.real.copy(), eri.real.copy()
    S.step('complex -> real copy', t0)
    if not SKIP_HF:
        t0 = time.time()
        U, eps, e_hf, e_start = o2_x2ccorr.hartree_fock(h1, eri, ecore, no, lab, pr)
        S.step('HF in the correlated space', t0)
    else:
        U = np.linalg.qr(np.random.default_rng(1).standard_normal((n, n)))[0]
    t0 = time.time()
    for axis in range(4):
        eri = np.moveaxis(np.tensordot(eri, U, axes=([axis], [0])), -1, axis)
        S.step('  rotation, index %d' % axis, t0)
    eri = np.ascontiguousarray(eri)
    S.step('rotation by tensordot (all four)', t0)


if __name__ == '__main__':
    lib.num_threads(int(os.environ.get('OMP_NUM_THREADS', 1)))
    main()
