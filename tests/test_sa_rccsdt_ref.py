'''Reference closed-shell spin-adapted CCSD / CCSDT (zccsdt.sa.rccsdt_ref).'''

import numpy as np
import pytest
from pyscf import cc, gto, scf

from zccsdt.sa import rccsdt_ref as R, somap


def _h2o(basis='6-31g'):
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis=basis, verbose=0)
    return scf.RHF(mol).run(conv_tol=1e-12)


def test_dressing_matches_spin_orbital():
    from zccsdt.cc import ccsdt
    no, nv = 2, 3
    d = somap.random_spatial(no, nv, seed=3)
    n = no + nv
    Vd, Fd = R.dress(d['V'], d['F'], d['T1'], no)
    sp = np.r_[np.arange(no), np.arange(no), no + np.arange(nv), no + np.arange(nv)]
    sg = np.r_[np.zeros(no), np.ones(no), np.zeros(nv), np.ones(nv)].astype(int)
    e = sg[:, None] == sg[None, :]

    def so(V):
        return (V[np.ix_(sp, sp, sp, sp)] * e[:, None, :, None] * e[None, :, None, :]
                - V.transpose(0, 1, 3, 2)[np.ix_(sp, sp, sp, sp)] * e[:, None, None, :] * e[None, :, :, None])

    t1so = somap.one_body(d['T1'])
    ge, fdr = ccsdt._t1_dress(so(d['V']).astype(complex), (d['F'][np.ix_(sp, sp)] * e).astype(complex),
                              t1so, 2 * no, 2 * n)
    assert abs(ge - so(Vd)).max() < 1e-12
    assert abs(fdr - Fd[np.ix_(sp, sp)] * e).max() < 1e-12


def test_null_projection():
    t = somap.pair_symmetrize(np.random.default_rng(0).standard_normal((3, 3, 3, 4, 4, 4)))
    p = R.project_null(t)
    assert abs(somap.triples(p) - somap.triples(t)).max() < 1e-13
    assert abs(R.project_null(p) - p).max() < 1e-14


def test_rccsd_vs_pyscf():
    mf = _h2o()
    r = R.kernel(R.from_scf(mf, ncore=1), 'ccsd', conv_tol=1e-11, conv_tol_normt=1e-8)
    mc = cc.RCCSD(mf, frozen=1).run(conv_tol=1e-12, conv_tol_normt=1e-9)
    assert r.converged and abs(r.e_tot - mc.e_tot) < 1e-9


def test_rccsdt_equals_spin_orbital():
    '''Converged energy equals the spin-orbital CCSDT code (GHF integrals).'''
    from zccsdt import eris as E
    from zccsdt.cc import solver
    mf = _h2o()
    r = R.kernel(R.from_scf(mf, ncore=1), 'ccsdt', conv_tol=1e-11, conv_tol_normt=1e-8)
    eris = E.from_scf(scf.addons.convert_to_ghf(mf), frozen=2)
    ref = solver.kernel(eris, 'ccsdt', solver.Options(conv_tol=1e-11, conv_tol_normt=1e-8))
    assert r.converged and ref.converged
    assert abs(r.e_tot - ref.e_tot) < 1e-9
