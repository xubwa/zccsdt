'''EOM-DEA through two ghost spinors (zccsdt.cc.dea; docs/eom.md, section
11): the sigma vector as one residual of the ground-state code at T + R

- with the complete operator (a reference of two electrons, R to 4p2h)
  the roots are the full CI of the four-electron system;
- against the dense term tables (ccsdtq_ref, ccn_ref) evaluated at T + R
  and against the dense EE Jacobian (eom_ref) among the elements with both
  ghost indices, real and complex, T up to T4;
- the masks of a label that adds.'''
import numpy as np
import pytest
from pyscf import fci, gto, scf

from test_ccsdtq_ref import _spin_mixed
from zccsdt import eris as E
from zccsdt.cc import ccn, ccn_ref as C, ccsdtq, ccsdtq_ref as R, ccsdtq_terms as Q, dea, eom_ref, solver

TIGHT = solver.Options(conv_tol=1e-13, conv_tol_normt=1e-11)
H4 = 'H 0 0 0; H 0 0 0.9; H 0 1.1 0; H 0 1.0 1.0'


def _ghf(atom, charge=0, basis='sto-3g'):
    mol = gto.M(atom=atom, basis=basis, charge=charge, verbose=0)
    return scf.RHF(mol).run(conv_tol=1e-13)


def _padded(amps, fock, g, no):
    '''The dense amplitudes and integrals with two ghost occupied spin
    orbitals in front.'''
    for _ in range(2):
        amps, fock, g, no, _ = eom_ref.ghost(amps, fock, g, no, 'ea')
    return amps, fock, g, no


def _dense(x, no, nv, dtype):
    '''The parts of a short vector as dense antisymmetric arrays on the
    padded spin orbitals: [0, r2, r3(, r4)].'''
    out = [np.zeros((no + 2, nv), dtype=dtype)]
    for k, p in enumerate(x):
        rank = k + 2
        occ = Q.tuples(no + 2, rank)
        u = np.zeros((len(occ), len(Q.tuples(nv, rank))), dtype=dtype)
        rows = np.flatnonzero((occ[:, 0] == 0) & (occ[:, 1] == 1))
        u[rows] = np.asarray(p).reshape(len(rows), -1)
        out.append(Q.expand(u, no + 2, nv, rank))
    return out


def _sector(dense, no, nv):
    out = []
    for k, d in enumerate(dense[1:]):
        rank = k + 2
        occ = Q.tuples(no + 2, rank)
        rows = np.flatnonzero((occ[:, 0] == 0) & (occ[:, 1] == 1))
        out.append(Q.unique(d, rank)[rows].ravel())
    return out


def test_complete_operator_is_full_ci():
    '''LiH(2+) / STO-3G, two electrons: CCSD is exact, R = 2p + 3p1h + 4p2h
    is complete, and the lowest roots are the full CI of LiH (four
    electrons in the orbitals of the dication).'''
    rhf = _ghf('Li 0 0 0; H 0 0 1.5957', charge=2)
    eris = E.from_scf(scf.addons.convert_to_ghf(rhf))
    cc = solver.kernel(eris, 'ccsd', TIGHT)
    res = dea.kernel(eris, cc.t1, cc.t2, rank=4, nroots=1, conv_tol=1e-9)
    mol = rhf.mol
    h1 = rhf.mo_coeff.T @ rhf.get_hcore() @ rhf.mo_coeff
    from pyscf import ao2mo
    eri = ao2mo.kernel(mol, rhf.mo_coeff)
    e_fci = fci.direct_spin1.kernel(h1, eri, rhf.mo_coeff.shape[1], (2, 2), ecore=mol.energy_nuc(), conv_tol=1e-13)[0]
    assert res.converged.all()
    assert abs(eris.e_hf + cc.e_corr + res.e[0] - e_fci) < 1e-9
    # rank 3 is not complete
    low = dea.kernel(eris, cc.t1, cc.t2, rank=3, nroots=1, conv_tol=1e-9)
    assert 1e-7 < low.e[0] - res.e[0] < 1e-2


@pytest.mark.parametrize('cplx', [False, True])
@pytest.mark.parametrize('level', ['ccsd', 'ccsdt', 'cc3', 'ccsdt4', 'ccsdtq', 'cc4', 'cc4full'])
def test_sigma_against_the_dense_references(cplx, level):
    '''H4 / STO-3G (4 + 4 spin orbitals, 6 + 4 with the ghosts): the sigma
    vector of every level for random vectors against the dense residuals
    of the model at T + R on the rows with both ghosts, and - where the
    model has one - against the dense EE Jacobian on R.'''
    mf = scf.addons.convert_to_ghf(_ghf(H4))
    eris = E.from_scf(mf, mo_coeff=_spin_mixed(mf)[0] if cplx else mf.mo_coeff)
    no, nv = 4, 4
    t3 = t4 = None
    rank, row3, row4, model = 3, 'full', 'full', 'ccsdtq'
    if level == 'ccsd':
        cc = solver.kernel(eris, 'ccsd', TIGHT)
    elif level in ('ccsdt', 'ccsdt4'):
        cc = solver.kernel(eris, 'ccsdt', TIGHT)
        t3, rank = cc.t3, 4 if level == 'ccsdt4' else 3
    elif level == 'cc3':
        cc = solver.kernel(eris, 'cc3', TIGHT)
        eris, t3, row3, model = cc.eris, cc.t3, 'cc3', 'cc3'
    elif level == 'ccsdtq':
        cc = ccsdtq.kernel(eris, ccsdtq.Options(conv_tol=1e-13, conv_tol_normt=1e-11))
        t3, t4, rank = cc.t3, cc.t4, 4
    else:
        cc = ccn.cc4_kernel(eris, conv_tol=1e-13, conv_tol_normt=1e-11)
        eris, t3, t4, rank = cc.eris, cc.t3, cc.t4, 4
        if level == 'cc4':
            row4, model = 'cc4', 'cc4'
    pb = dea.Problem(eris, cc.t1, cc.t2, t3, t4, rank=rank, row3=row3, row4=row4)
    fock, g, mo_e, _ = R.integrals(eris)
    dt = pb.dtype
    amps = [np.asarray(cc.t1, dtype=dt), np.asarray(cc.t2, dtype=dt),
            Q.expand(np.zeros((4, 4), dtype=dt) if t3 is None else np.asarray(t3, dtype=dt).reshape(4, 4), no, nv, 3)]
    if rank == 4:
        amps.append(Q.expand(np.zeros((1, 1), dtype=dt) if t4 is None else np.asarray(t4, dtype=dt).reshape(1, 1), no, nv, 4))
    ampsp, fp, gp, nop = _padded(amps, fock if cplx else fock.real, g if cplx else g.real, no)
    rng = np.random.default_rng(3)
    for _ in range(2):
        x = [rng.standard_normal(s) + (1j * rng.standard_normal(s) if cplx else 0) for s in pb.sizes]
        y = pb.sigma(x, pb.new())
        r = _dense(x, no, nv, dt)
        at = [a + b for a, b in zip(ampsp, r)]
        if model == 'ccsdtq':
            ref = R.residuals(at, fp, gp, nop)
        else:
            ref = C.residuals(model, at, fp, gp, nop)
        ref = _sector(ref, no, nv)
        scale = max(abs(p).max() for p in y)
        assert scale > 0.1
        for a, b in zip(y, ref):
            assert abs(a - b).max() < 1e-11 * max(scale, 1)
        if level in ('ccsd', 'ccsdt'):
            # the EE Jacobian of CCSDT on R (CCSD amplitudes: T3 = 0)
            jac = _sector(eom_ref.sigma(r, ampsp, fp, gp, nop), no, nv)
            for a, b in zip(y, jac):
                assert abs(a - b).max() < 1e-11 * max(scale, 1)
        # linear in R: the residual of the ground state has no part on these rows
        y2 = pb.sigma([2.0 * p for p in x], pb.new())
        assert max(abs(2 * a - b).max() for a, b in zip(y, y2)) < 1e-11 * max(scale, 1)


def test_roots_and_masks():
    '''LiH / STO-3G, CCSDT, R to 3p1h: the roots of kernel() are the lowest
    eigenvalues of the dense matrix; with the label 2 m_s of the spin
    orbitals the roots of the targets -2, 0, 2 together are those roots,
    and the targets +-2 have the same.'''
    mf = scf.addons.convert_to_ghf(_ghf('Li 0 0 0; H 0 0 1.5957'))
    eris = E.from_scf(mf)
    cc = solver.kernel(eris, 'ccsdt', TIGHT)
    pb = dea.Problem(eris, cc.t1, cc.t2, cc.t3)
    w = np.linalg.eigvals(pb.matrix())
    w = np.sort(w.real[abs(w.imag) < 1e-10])
    res = dea.kernel(eris, cc.t1, cc.t2, cc.t3, nroots=4, nguess=28, conv_tol=1e-9)
    assert res.converged.all() and abs(res.e - w[:4]).max() < 1e-9
    nao = mf.mol.nao_nr()
    C_ = np.hstack((mf.mo_coeff[:, mf.mo_occ > 0], mf.mo_coeff[:, mf.mo_occ == 0]))
    ms2 = np.where((abs(C_[:nao]) ** 2).sum(axis=0) > 0.5, 1, -1)
    roots = {}
    for tgt in (-2, 0, 2):
        r = dea.kernel(eris, cc.t1, cc.t2, cc.t3, labels=ms2, target=tgt, nroots=2, nguess=28, conv_tol=1e-9)
        assert r.converged.all()
        roots[tgt] = r.e
        assert all(abs(w - x).min() < 1e-9 for x in r.e)
    assert abs(roots[2] - roots[-2]).max() < 1e-9
    assert abs(min(roots[0][0], roots[2][0]) - w[0]) < 1e-9
