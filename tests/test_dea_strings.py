'''EOM-DEA on determinants (zccsdt.cc.dea_strings), the reference route of
any rank:

- all holes: the full CI of the system with two electrons more;
- R to 3p1h and to 4p2h: the matrix of the route through the ghost spinors
  (zccsdt.cc.dea), T to T4 - the same eigenvalues;
- R to 5p3h: the matrix P exp(-T) H exp(T) P of second-quantized operators
  in the Fock space (tests/fockspace.py), projected on the determinants
  with at most three holes;
- labels: the roots of the blocks are roots of the whole.'''
import numpy as np
import pytest
from pyscf import ao2mo, fci, scf

from fockspace import Fock
from test_dea import H4, TIGHT, _ghf
from zccsdt import eris as E
from zccsdt.cc import ccsdtq, ccsdtq_ref as R, ccsdtq_terms as Q, dea, dea_strings as DS, solver

LIH = 'Li 0 0 0; H 0 0 1.5957'


def _problem(eris, amps, hmax, **kw):
    fock, g, _, no = R.integrals(eris)
    return DS.Problem(np.asarray(fock).real, np.asarray(g).real, no, amps, hmax, e_ref=eris.e_hf, **kw)


def test_all_holes_is_full_ci():
    '''LiH / STO-3G from LiH(2+) with all holes (2), and - T of CCSDTQ, which
    is exact for four electrons - LiH(2-) from LiH with all four holes.'''
    rhf = _ghf(LIH, charge=2)
    eris = E.from_scf(scf.addons.convert_to_ghf(rhf))
    cc = solver.kernel(eris, 'ccsd', TIGHT)
    e, res = DS.kernel(_problem(eris, [cc.t1, Q.unique(cc.t2, 2)], 2), nroots=1, nguess=6, conv_tol=1e-9)
    mol = rhf.mol
    h1 = rhf.mo_coeff.T @ rhf.get_hcore() @ rhf.mo_coeff
    eri = ao2mo.kernel(mol, rhf.mo_coeff)
    e_fci = fci.direct_spin1.kernel(h1, eri, rhf.mo_coeff.shape[1], (2, 2), ecore=mol.energy_nuc(), conv_tol=1e-13)[0]
    assert res.converged.all() and abs(e[0] - e_fci) < 1e-9
    # the neutral molecule as the reference: six electrons, R complete with four holes
    rhf = _ghf(LIH)
    eris = E.from_scf(scf.addons.convert_to_ghf(rhf))
    cc = ccsdtq.kernel(eris, ccsdtq.Options(conv_tol=1e-13, conv_tol_normt=1e-11))
    amps = [cc.t1, Q.unique(cc.t2, 2), np.asarray(cc.t3).reshape(4, 56), np.asarray(cc.t4).reshape(1, 70)]
    e, res = DS.kernel(_problem(eris, amps, 4), nroots=1, nguess=28, conv_tol=1e-9)
    h1 = rhf.mo_coeff.T @ rhf.get_hcore() @ rhf.mo_coeff
    eri = ao2mo.kernel(rhf.mol, rhf.mo_coeff)
    e6 = min(fci.direct_spin1.kernel(h1, eri, 6, na, ecore=rhf.mol.energy_nuc(), conv_tol=1e-13)[0]
             for na in ((3, 3), (4, 2)))
    assert res.converged.all() and abs(e[0] - e6) < 1e-9


@pytest.mark.parametrize('level', ['ccsd', 'ccsdt', 'ccsdtq'])
def test_against_the_ghost_route(level):
    '''H4 / STO-3G (4 + 4): the eigenvalues of the matrix on determinants
    with one (two) holes are those of the route through the ghost spinors
    with R to 3p1h (4p2h), plus the energy of the ground state.'''
    eris = E.from_scf(scf.addons.convert_to_ghf(_ghf(H4)))
    if level == 'ccsd':
        cc = solver.kernel(eris, 'ccsd', TIGHT)
        t3 = t4 = None
        amps, cases = [cc.t1, Q.unique(cc.t2, 2)], [(3, 1)]
    elif level == 'ccsdt':
        cc = solver.kernel(eris, 'ccsdt', TIGHT)
        t3, t4 = cc.t3, None
        amps, cases = [cc.t1, Q.unique(cc.t2, 2), np.asarray(t3).reshape(4, 4)], [(3, 1), (4, 2)]
    else:
        cc = ccsdtq.kernel(eris, ccsdtq.Options(conv_tol=1e-13, conv_tol_normt=1e-11))
        t3, t4 = cc.t3, cc.t4
        amps, cases = [cc.t1, Q.unique(cc.t2, 2), np.asarray(t3).reshape(4, 4), np.asarray(t4).reshape(1, 1)], [(4, 2)]
    for rank, hmax in cases:
        ghost = dea.Problem(eris, cc.t1, cc.t2, t3, t4 if rank == 4 else None, rank=rank)
        if level == 'ccsdtq' and rank == 3:
            continue
        w = np.sort(np.linalg.eigvals(ghost.matrix()).real) + eris.e_hf + cc.e_corr
        pb = _problem(eris, amps, hmax)
        assert sum(pb.sizes) == sum(ghost.sizes)
        v = np.sort(np.linalg.eigvals(pb.matrix()).real) + pb.const
        assert abs(v - w).max() < 1e-9


def test_three_holes_against_the_fock_space():
    '''LiH / STO-3G (4 + 8), random amplitudes T1 .. T4 and real integrals:
    the matrix with up to three holes (2p .. 5p3h, 896 determinants) has
    the eigenvalues of P exp(-T) H exp(T) P built from second-quantized
    operators; in the blocks of a label that adds, their union.'''
    no, nv = 4, 8
    n = no + nv
    eris = E.from_scf(scf.addons.convert_to_ghf(_ghf(LIH)))
    fock, g, _, _ = R.integrals(eris)
    fock, g = np.asarray(fock).real, np.asarray(g).real
    rng = np.random.default_rng(7)
    from fockspace import random_antisym
    dense = [0.05 * rng.standard_normal((no, nv))] + [0.05 * random_antisym(rng, no, nv, k) for k in (2, 3, 4)]
    amps = [dense[0]] + [Q.unique(t, k) for t, k in zip(dense[1:], (2, 3, 4))]
    pb = DS.Problem(fock, g, no, amps, 3, e_ref=eris.e_hf)
    assert pb.sizes == [28, 224, 420, 224]
    v = np.sort(np.linalg.eigvals(pb.matrix()).real)
    # the same in the Fock space: H = sum h1 p+ q + 1/4 sum <pq||rs> p+ q+ s r, T, the determinants by their occupations
    fk = Fock(no, nv)
    import scipy.sparse as sp
    h1 = fock - np.einsum('piqi->pq', g[:, :no, :, :no])
    H = fk.two_body(g) + sum((h1[p, q] + np.einsum('piqi->pq', g[:, :no, :, :no])[p, q]) * (fk.ad[p] @ fk.a[q])
                             for p in range(n) for q in range(n))
    T = sum(fk.excitation(t) for t in dense)
    dim = 2 ** n

    def expm(sign, X):
        out, term = X.copy(), X.copy()
        for k in range(1, n + 1):
            term = (sign / k) * (T @ term)
            out = out + term
        return out
    # the determinants with six electrons and at most three holes (bit p of the index: spin orbital n - 1 - p)
    occ = np.array([[(d >> (n - 1 - p)) & 1 for p in range(n)] for d in range(dim)])
    sel = np.flatnonzero((occ.sum(axis=1) == no + 2) & (occ[:, :no].sum(axis=1) >= no - 3))
    assert len(sel) == 896
    Pm = sp.csr_matrix((np.ones(len(sel)), (sel, np.arange(len(sel)))), shape=(dim, len(sel)))
    M = (Pm.T @ expm(-1.0, sp.csr_matrix(H @ expm(1.0, Pm)))).toarray()
    w = np.sort(np.linalg.eigvals(M).real)
    assert abs(v - w).max() < 1e-9
    # labels: 2 m_s of the spin orbitals
    mf = scf.addons.convert_to_ghf(_ghf(LIH))
    nao = mf.mol.nao_nr()
    C_ = np.hstack((mf.mo_coeff[:, mf.mo_occ > 0], mf.mo_coeff[:, mf.mo_occ == 0]))
    ms2 = np.where((abs(C_[:nao]) ** 2).sum(axis=0) > 0.5, 1, -1)
    cc = solver.kernel(eris, 'ccsdt', TIGHT)
    amps = [cc.t1, Q.unique(cc.t2, 2), np.asarray(cc.t3).reshape(4, 56)]
    whole = np.sort(np.linalg.eigvals(DS.Problem(fock, g, no, amps, 2, e_ref=eris.e_hf).matrix()).real)
    parts = []
    for tgt in (-2, 0, 2):
        blk = DS.Problem(fock, g, no, amps, 2, labels=ms2, target=tgt, e_ref=eris.e_hf)
        parts.append(np.linalg.eigvals(blk.matrix()).real)
        assert all(abs(whole - x).min() < 1e-9 for x in parts[-1])
    assert abs(np.sort(parts[0]) - np.sort(parts[2])).max() < 1e-9
