'''The sigma vector of the sector (zccsdt.cc.dea_sector) against that of the
ghost route in label blocks (zccsdt.cc.dea.ProblemSym), element by element,
for a random vector (parity g), 2 Omega = 0 and 2, on a Hamiltonian file.

usage: python sector_check.py <file.npz> [ccsdt | ccsdtq]'''
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from zccsdt.cc import dea, dea_sector as D, ccsdtq_terms as Q

path = sys.argv[1]
level = sys.argv[2] if len(sys.argv) > 2 else 'ccsdt'
eris, lab, par, info = common.load(path, sort=True)
st = common.ground(eris, level, 1e-10)
no = eris.nocc
nv = np.asarray(eris.fock).shape[0] - no
for target in (0, 2):
    sp = D.ProblemSector(eris, st['t1'], st['t2'], st['t3'], rank=4, target=target, parity=par, t4=st['t4'])
    gp = dea.ProblemSym(eris, st['t1'], st['t2'], st['t3'], st['t4'], rank=4, target=target, parity=par)
    rng = np.random.default_rng(3)
    x = sp.project([rng.standard_normal(n) for n in sp.sizes])
    y = sp.sigma(x, sp.new())
    dpos = sp.dense_positions()
    full = []
    for k in range(3):
        row = k + 2
        n = (1 if row == 2 else (no if row == 3 else no * (no - 1) // 2)) * len(Q.tuples(nv, row))
        f = np.zeros(n)
        f[dpos[k]] = x[k]
        full.append(f)
    xg = gp.new()
    xg[0][:] = full[0]
    for k in (1, 2):
        xg[k][:] = full[k][gp.dpos[k]]
    yg = gp.sigma(xg, gp.new())
    for k in range(3):
        got = np.zeros_like(full[k])
        got[dpos[k]] = y[k]
        ref = yg[0] if k == 0 else np.zeros_like(full[k])
        if k:
            ref[gp.dpos[k]] = yg[k]
        print('%s %s target %d row %d: max |diff| %.2e, max |sigma| %.2e, elements %d / %d'
              % (path.split('/')[-1], level, target, k + 2, abs(got - ref).max(), abs(ref).max(), len(y[k]), len(yg[k])))
