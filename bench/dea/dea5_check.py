'''The sigma vector with complete 5p3h on the sector (zccsdt.cc.dea5_sector)
against the determinant route (zccsdt.cc.dea_strings, hmax 3), element by
element, for a random vector of parity g, 2 Omega = 0 and 2, T of CCSDT, on
a Hamiltonian file.

usage: python dea5_check.py <file.npz> [--targets 0 2] [--pieces]'''
import os
import sys
import time
from math import comb

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def det_map(ds, dpos, no, nv):
    '''For each part h of the sector (positions in the dense packed arrays
    [C(no, h), C(nv, h + 2)]): (index into the determinant vector of part h or
    -1 where the determinant is not in its space, sign).'''
    from zccsdt.cc import ccsdtq_terms as Q
    from zccsdt.cc.dea_strings import _add
    out = []
    full = (1 << no) - 1
    for h, pos in enumerate(dpos):
        nvt = comb(nv, h + 2)
        rows, cols = pos // nvt, pos % nvt
        occ = Q.tuples(no, h) if h else np.zeros((1, 0), dtype=np.int64)
        vir = Q.tuples(nv, h + 2)
        holes = occ[rows]
        hmask = (1 << holes).sum(axis=1) if h else np.zeros(len(pos), dtype=np.int64)
        kmask = full ^ hmask
        k = np.searchsorted(ds.K[h], kmask)
        assert np.all(ds.K[h][k] == kmask)
        bmask = (1 << vir[cols]).sum(axis=1)
        idx = np.full(len(pos), -1, dtype=np.int64)
        groups = ds.vs.groups(h + 2)
        for kk in np.unique(k):
            sel = np.flatnonzero(k == kk)
            grp = groups.get(_add(ds.L, ds.Kkey[h][kk], -1))
            if grp is None:
                continue
            p = np.searchsorted(grp, bmask[sel])
            ok = (p < len(grp)) & (grp[np.minimum(p, len(grp) - 1)] == bmask[sel])
            idx[sel[ok]] = ds.rows[h][kk] + p[ok]
        sign = (-1.0) ** (h * (h - 1) // 2 + (holes.sum(axis=1) if h else 0))
        out.append((idx, np.broadcast_to(sign, idx.shape).astype(float)))
    return out


def main():
    from zccsdt.cc import ccsdt_opt, ccsdtq_ref as R, ccsdtq_terms as Q, dea_strings as DS, dea5_sector as D5
    args = sys.argv[1:]
    targets = (0, 2)
    if '--targets' in args:
        k = args.index('--targets')
        targets = tuple(int(x) for x in args[k + 1:k + 3])
        del args[k:k + 3]
    pieces = '--pieces' in args
    path = args[0]
    eris, lab, par, info = common.load(path, sort=True)
    no = eris.nocc
    nv = info['n'] - no
    st = common.ground(eris, 'ccsdt', 1e-11)
    e_cc = eris.e_hf + st['e_corr']
    print('%s: %d + %d spinors, E(CCSDT) = %.11f' % (path, no, nv, e_cc), flush=True)
    fock, g, _, _ = R.integrals(eris)
    fock, g = np.asarray(fock).real, np.asarray(g).real
    amps = [st['t1'], Q.unique(st['t2'], 2), np.asarray(ccsdt_opt.packed_t3(eris, st['t3']))]
    for tgt in targets:
        t0 = time.time()
        pb = D5.ProblemSector5(eris, st['t1'], st['t2'], st['t3'], target=tgt, parity=par, verbose=True)
        rng = np.random.default_rng(5)
        x = pb.project([rng.standard_normal(n) for n in pb.sizes])
        t1 = time.time()
        y = pb.sigma(x, pb.new())
        t2 = time.time()
        ds = DS.Problem(fock, g, no, amps, 3, labels=lab, parity=par, target=tgt, e_ref=eris.e_hf)
        e0 = e_cc - ds.const
        maps = det_map(ds, pb.dense_positions(), no, nv)
        xd = ds.new()
        for h, (idx, sg) in enumerate(maps):
            ok = idx >= 0
            assert abs(x[h][~ok]).max(initial=0) == 0, 'elements outside the determinant space'
            xd[h][idx[ok]] = sg[ok] * x[h][ok]
        assert sum(int((m[0] >= 0).sum()) for m in maps) == sum(ds.sizes), (pb.counts(), ds.sizes)
        t3 = time.time()
        yd = ds.sigma(xd, ds.new())
        t4 = time.time()
        print('2 Omega = %d: sector prepared %.1f s, sigma %.2f s; determinants %.1f s; sizes %s / %s'
              % (tgt, t1 - t0, t2 - t1, t4 - t3, pb.counts(), ds.sizes), flush=True)
        for h, (idx, sg) in enumerate(maps):
            ok = idx >= 0
            ref = sg[ok] * (yd[h][idx[ok]] - e0 * xd[h][idx[ok]])
            print('  row %d: max |sigma| %.3e, max |diff| %.3e' % (h + 2, abs(ref).max(), abs(ref - y[h][ok]).max()),
                  flush=True)
        if pieces:
            for piece in D5.PIECES5:
                yp = pb.sigma(x, pb.new(), only={piece})
                print('    piece %-4s max |sigma5| %.3e' % (piece, abs(yp[3]).max()), flush=True)


if __name__ == '__main__':
    main()
