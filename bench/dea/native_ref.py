'''The roots of EOM-DEA-CCSDT of the Python sector (zccsdt.cc.dea_sector) on
the integral files of a run directory of the native program - the reference
numbers of a regression test (cfour/merge/test/run.sh): the CCSDT ground
state of the Python program on the same files, then the lowest roots of the
targets 2 Omega given (all parities, the labels of the run directory:
xzlabels or dgcosym).

usage: python native_ref.py <rundir> <target> [<target> ...] [--roots n] [--tol 1e-10]
'''
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def main():
    args = sys.argv[1:]
    nroots, tol = 1, 1e-10
    if '--roots' in args:
        k = args.index('--roots')
        nroots = int(args[k + 1])
        del args[k:k + 2]
    if '--tol' in args:
        k = args.index('--tol')
        tol = float(args[k + 1])
        del args[k:k + 2]
    rundir, targets = args[0], [int(x) for x in args[1:]]
    from zccsdt import cfour
    from zccsdt.cc import dea, dea_sector
    eris = cfour.from_cfour(rundir, labels=True)
    imag = max(abs(np.asarray(getattr(eris, k)).imag).max() for k in ('oooo', 'ooov', 'oovv', 'ovvo', 'ovvv'))
    if imag < 1e-10:
        for k in ('oooo', 'ooov', 'oovv', 'ovov', 'ovvo', 'ovvv', 'fock', '_vvvv_pp'):
            setattr(eris, k, np.ascontiguousarray(np.asarray(getattr(eris, k)).real))
    eris.use_labels = True
    # the spinors in the order of their labels (the sector wants them so; the
    # native program sorts them itself, stably): occupied and virtual each
    no = eris.nocc
    lab = np.asarray(eris.labels)
    po = np.argsort(lab[:no], kind='stable')
    pv = np.argsort(lab[no:], kind='stable')
    if (po != np.arange(no)).any() or (pv != np.arange(len(pv))).any():
        sl = {'o': po, 'v': pv}
        for key in ('oooo', 'ooov', 'oovv', 'ovov', 'ovvo', 'ovvv'):
            x = np.asarray(getattr(eris, key))
            setattr(eris, key, np.ascontiguousarray(x[np.ix_(*[sl[c] for c in key])]))
        nv = len(pv)
        a, b = np.triu_indices(nv, 1)
        full = np.zeros((nv, nv, nv, nv), dtype=np.asarray(eris._vvvv_pp).dtype)
        full[a[:, None], b[:, None], a[None, :], b[None, :]] = eris._vvvv_pp
        full[b[:, None], a[:, None], a[None, :], b[None, :]] = -np.asarray(eris._vvvv_pp)
        full[a[:, None], b[:, None], b[None, :], a[None, :]] = -np.asarray(eris._vvvv_pp)
        full[b[:, None], a[:, None], b[None, :], a[None, :]] = eris._vvvv_pp
        full = full[np.ix_(pv, pv, pv, pv)]
        eris._vvvv_pp = np.ascontiguousarray(full[a[:, None], b[:, None], a[None, :], b[None, :]])
        p = np.concatenate((po, no + pv))
        eris.fock = np.asarray(eris.fock)[np.ix_(p, p)]
        eris.mo_energy = np.asarray(eris.mo_energy)[p]
        eris.labels = lab[p]
    st = common.ground(eris, 'ccsdt', 1e-11)
    ecc = eris.e_hf + st['e_corr']
    print('E(SCF) %.12f  E(CCSDT) %.12f' % (eris.e_hf, ecc))
    for t in targets:
        pb = dea_sector.ProblemSector(eris, st['t1'], st['t2'], st['t3'], rank=4, target=t)
        r = dea.kernel(None, None, None, problem=pb, nroots=nroots, conv_tol=tol, max_cycle=200)
        for k, w in enumerate(r.e):
            print('target %3d root %d  omega %.12f  E %.12f  |residual| %.1e' % (t, k + 1, w, ecc + w, r.residuals[k]))


if __name__ == '__main__':
    main()
