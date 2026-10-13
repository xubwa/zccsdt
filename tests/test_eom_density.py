'''Densities and transition densities of EOM-EE-CCSDT states
(zccsdt.cc.eom_density; docs/eom.md, section 13):

- against second-quantized operators in a small Fock space (tests/fockspace.py)
  for amplitudes and vectors that solve nothing: <0| L Obar R |0>, the two
  transition densities, the contraction L' and xi'';
- the density with the response of the amplitudes is the derivative of the
  energy of the state with respect to a field at fixed orbitals (central
  differences of sixth order of E(CCSDT) + omega from the dense matrix);
- two electrons, where the method is exact: the densities of the excited
  states and the transition strengths are those of the eigenvectors of H;
- in the blocks of the symmetry labels against the packed block, a state
  with a label shift; the front end with a frozen core.'''
from types import SimpleNamespace

import numpy as np
import pytest
from pyscf import gto, scf

import fockspace as F
import zccsdt
from test_ccsdtq_ref import _scf, _spin_mixed
from test_eom_opt import pack2, pack3
from test_eom_shift import hbr, lih    # noqa: F401  (fixtures)
from test_lambda_opt import _amps, _model_eris
from zccsdt import eris as E
from zccsdt.cc import (ccsdtq_ref as R, eom, eom_density as ED, eom_left as EL, eom_opt,
                       eom_ref as EOM, lambda_opt, solver, t3pack)


def _fock_density(fk, bra, ket):
    '''d[p,q] = <bra| p+ q |ket> (bra as a vector: nothing conjugated).'''
    Lp = [fk.a[p] @ bra for p in range(fk.n)]
    Rq = [fk.a[q] @ ket for q in range(fk.n)]
    return np.array([[l @ r for r in Rq] for l in Lp])


@pytest.mark.parametrize('cplx', [True, False])
def test_densities_against_the_fock_space(cplx):
    '''4 + 5 spin orbitals, random Hermitian integrals, amplitudes, Lambda,
    L and R that solve nothing (L R = 1): the density of the state, the
    two transition densities and xi'' are the matrix elements of the
    operators, to 1e-11 of their size.'''
    no, nv = 4, 5
    eris = _model_eris(no, nv, seed=7, cplx=cplx, scale=0.1)
    fock, g, mo_e, _ = R.integrals(eris)
    rng = np.random.default_rng(11)
    amps = _amps(rng, no, nv, cplx, 0.1)
    r, l, lam = _amps(rng, no, nv, cplx, 1.0), _amps(rng, no, nv, cplx, 1.0), _amps(rng, no, nv, cplx, 0.2)
    om, vm = t3pack.maps_for(eris)
    fk = F.Fock(no, nv)
    ex = F.Left(fk, fock, g, amps)
    rv, lv, lamv = ex.vector(r), ex.vector(l), ex.vector(lam)
    s = lv @ rv
    l = [x / s for x in l]
    lv = lv / s
    T3p = pack3(amps[2], om, vm).ravel()
    gs = eom_opt.prepare(eris, amps[0], amps[1], T3p)
    omega = 0.7
    right = SimpleNamespace(ground=gs, shift=None, e=np.array([omega]), r1=[r[0]], r2=[pack2(r[1], om, vm)],
                            r3=[pack3(r[2], om, vm).ravel()])
    left = SimpleNamespace(ops=EL.LeftOps(eris, amps[0], amps[1], T3p), roots=[0], l1=[l[0]],
                           l2=[pack2(l[1], om, vm)], l3=[pack3(l[2], om, vm).ravel()])
    st = ED.kernel(eris, amps[0], amps[1], T3p, right, left, lam=(lam[0], lam[1], pack3(lam[2], om, vm).ravel()),
                   transition=True)[0]
    # the reference row
    eta = ex.lambda_residual(np.zeros(len(ex.index)))
    assert abs(st.r0 - eta @ rv / omega) < 1e-12 and abs(st.r0) > 1e-3
    ref = fk.ref.astype(complex)
    ket = ex.exp(st.r0 * ref + ex.dets @ rv, 1)             # exp(T) R |0>
    bra = ex.exp(ex.dets @ lv, -1, left=True)               # <0| L exp(-T)
    d = _fock_density(fk, bra, ket)
    assert abs(st.d1 - d).max() < 1e-11 * abs(d).max() and abs(np.trace(st.d1) - no) < 1e-11
    assert abs(d - d.T.conj()).max() > 1e-3                 # (nothing is Hermitian here)
    dl = _fock_density(fk, bra, ex.right)
    assert abs(st.left - dl).max() < 1e-11 * abs(dl).max()
    dr = _fock_density(fk, ex.exp(ex.bra(lamv), -1, left=True), ket)
    assert abs(st.right - dr).max() < 1e-11 * abs(dr).max() and abs(dr).max() > 0.1
    assert abs(st.ground_overlap - abs(st.r0 + lamv @ rv)) < 1e-12
    # the part of L R that de-excites
    a1, a2 = ED.contract(gs, l[1], left.l3[0], r[0], right.r2[0])
    w1 = np.einsum('ijab,jb->ia', l[1], r[0]) + 0.25 * np.einsum('ijkabc,jkbc->ia', l[2], r[1])
    w2 = np.einsum('ijkabc,kc->ijab', l[2], r[0])
    assert abs(a1 - w1).max() < 1e-12 and abs(a2 - pack2(w2, om, vm)).max() < 1e-12
    # xi''[kappa] = <0| L [[Hbar, tau_kappa], R] |0>
    view = ED.View(eris, gs.dtype)
    xi = ED._xi(view, amps[0], amps[1], T3p, (r[0], r[1], right.r3[0]), (l[0], l[1], left.l3[0]), solver.Options())
    Rop = sum(fk.excitation(x) for x in r)
    lb = ex.dets @ lv
    hb, rr = ex.hbar(ref), Rop @ ref
    hr = ex.hbar(rr)
    want = []
    for _, occ, vir in ex.index:
        tau = ex.tau(occ, vir)
        x = ex.hbar(tau @ rr) - tau @ hr - Rop @ ex.hbar(tau @ ref) + Rop @ (tau @ hb)
        want.append(lb @ x)
    want = np.array(want)
    got = np.concatenate([xi[0].ravel(), xi[1].ravel(), xi[2].ravel()])
    assert abs(got - want).max() < 1e-10 * abs(want).max() and abs(want).max() > 1e-2


@pytest.mark.parametrize('cplx', [False, True])
def test_response_density_is_the_energy_derivative(cplx):
    '''LiH / STO-3G, 4 + 8 spin orbitals (complex: every pair mixed by an
    SU(2) matrix): the dipole of the density with the response of the
    amplitudes is the derivative of E(CCSDT) + omega with respect to a
    field z in the Hamiltonian at fixed orbitals - the lowest singlet, and
    the triplet below it as a level - to 1e-8; the density of the EOM
    state without the response differs by 3e-6 (four electrons: CCSDT is
    close to exact).'''
    mf = scf.addons.convert_to_ghf(_scf('lih'))
    C = _spin_mixed(mf)[0] if cplx else mf.mo_coeff
    nao = mf.mol.nao_nr()
    with mf.mol.with_common_orig((0, 0, 0)):
        z = mf.mol.intor('int1e_r')[2]
    zmo = C[:nao].conj().T @ z @ C[:nao] + C[nao:].conj().T @ z @ C[nao:]
    eris = E.from_scf(mf, mo_coeff=C)
    fock, g, mo_e, no = R.integrals(eris)
    nv = len(mo_e) - no
    opts = solver.Options(conv_tol=1e-13, conv_tol_normt=1e-11)
    rt = solver.kernel(eris, 'ccsdt', opts)
    lr = lambda_opt.kernel(eris, rt.t1, rt.t2, rt.t3, opts)
    res = eom.kernel(eris, rt.t1, rt.t2, rt.t3, nroots=4, conv_tol=1e-10)
    left = EL.kernel(eris, rt.t1, rt.t2, rt.t3, res, conv_tol=1e-10)
    assert res.converged.all() and left.converged.all() and left.levels == [[0, 1, 2], [3]]
    states = ED.kernel(eris, rt.t1, rt.t2, rt.t3, res, left, lam=(lr.l1, lr.l2, lr.l3), response=True,
                       transition=True, opts=opts)
    d0 = ED.density_opt.kernel(eris, rt.t1, rt.t2, rt.t3, lr.l1, lr.l2, lr.l3, gradient=False).d1

    def energies(eps):
        f = fock + eps * zmo
        x = R.kernel(f, g, mo_e, no, 'ccsdt', conv_tol=1e-13, conv_tol_normt=1e-11)
        w = EOM.eigenvalues(x.amps if hasattr(x, 'amps') else [x.t1, x.t2, x.t3], f, g, no)
        e0 = np.trace(f[:no, :no]).real + x.e_corr
        return e0, w.real[:4]
    # (central differences of sixth order: the fifth derivative is large here - with the
    # formula of fourth order the ground state is off by 2e-7 at 1e-3 and 4e-6 at 2e-3)
    eps = 5e-4
    e = {k: energies(k * eps) for k in (-3, -2, -1, 1, 2, 3)}

    def fd(f):
        return (45 * (f(e[1]) - f(e[-1])) - 9 * (f(e[2]) - f(e[-2])) + (f(e[3]) - f(e[-3]))) / (60 * eps)
    dz = [np.einsum('pq,pq', zmo, s.d1_response) for s in states]
    dx = [np.einsum('pq,pq', zmo, s.d1) for s in states]
    g0 = np.einsum('pq,pq', zmo, d0)
    assert abs(fd(lambda x: x[0]) - g0.real) < 1e-9
    # the singlet
    want = fd(lambda x: x[0] + x[1][3])
    assert abs(dz[3].real - want) < 1e-8 and abs(dz[3].imag) < 1e-9
    assert 1e-7 < abs(dx[3].real - want) < 1e-4
    # the triplet: the sum over the level
    want3 = fd(lambda x: 3 * x[0] + x[1][:3].sum())
    assert abs(sum(dz[:3]).real - want3) < 3e-8
    print('LiH %s: d(E0 + omega)/dz finite field %.10f, response density %.10f, EOM-CC density %.10f; ground %.10f'
          % ('complex' if cplx else 'real', want, dz[3].real, dx[3].real, g0.real))
    for s in states:
        assert abs(np.trace(s.d1) - no) < 1e-10 and abs(np.trace(s.d1_response) - no) < 1e-10
        assert s.ground_overlap < 1e-8 and abs(np.trace(s.left)) < 1e-10 and abs(np.trace(s.right)) < 1e-10
    assert abs(states[3].r0) > 1e-3 and max(abs(s.r0) for s in states[:3]) < 1e-8


def _two_electrons():
    mol = gto.M(atom='He 0 0 0; H 0 0 0.9', charge=1, basis='6-31g', verbose=0)
    return scf.addons.convert_to_ghf(scf.RHF(mol).run(conv_tol=1e-13))


def test_two_electrons_are_exact():
    '''HeH+ / 6-31G (2 + 6 spin orbitals), where CCSD is exact and the
    code runs without triples: the densities of the excited states are
    those of the eigenvectors of H among the determinants (the singlets by
    themselves, a triplet as a level), the density with the response is the
    same, and the transition strength is |<0| z |k>|^2.'''
    mf = _two_electrons()
    cc = zccsdt.CCSDT(mf)
    cc.verbose, cc.conv_tol, cc.conv_tol_normt = 0, 1e-13, 1e-11
    cc.kernel()
    e, res = cc.eom_ee(nroots=8, conv_tol=1e-9, nguess=12)
    pr = cc.eom_properties(response=True)
    left, states = res.left, res.states
    assert left.converged.all() and abs(left.e - e).max() < 1e-9
    # the eigenvectors of H in the Fock space, two electrons
    fock, g, mo_e, no = R.integrals(E.from_scf(mf))         # (with the full <ab||cd>)
    nv = len(mo_e) - no
    fk = F.Fock(no, nv)
    H = (F.one_body(fk, fock) + fk.two_body(g)).toarray()
    N = sum((fk.ad[p] @ fk.a[p]).toarray() for p in range(fk.n)).diagonal()
    sel = np.flatnonzero(abs(N - 2) < 1e-9)
    w, V = np.linalg.eigh(H[np.ix_(sel, sel)])
    assert abs((w[1:9] - w[0]) - e).max() < 1e-8

    def vec(k):
        x = np.zeros(2 ** fk.n, dtype=V.dtype)
        x[sel] = V[:, k]
        return x
    C, nf, na = cc._all_spinors()
    r = np.einsum('pi,xpq,qj->xij', C.conj(), cc._dipole_ao(), C)
    assert nf == 0
    n = 0
    for lv in left.levels:
        ks = [left.roots[i] for i in lv]
        exact = sum(_fock_density(fk, vec(k + 1).conj(), vec(k + 1)) for k in ks)
        for kind in ('d1', 'd1_response'):
            got = sum(getattr(states[i], kind) for i in lv)
            assert abs(got - exact).max() < 2e-7, (lv, kind)
        s_exact = sum(abs(np.einsum('xpq,pq->x', r, _fock_density(fk, vec(0).conj(), vec(k + 1)))) ** 2 for k in ks).sum()
        assert abs(pr.strength[lv].sum() - s_exact) < 1e-7
        n += s_exact > 1e-3
    assert n >= 1 and abs(pr.dipole_ground - (pr.dipole_ground * 0 + cc_dipole(cc, fk, vec(0), r))).max() < 1e-7
    # the front end: the density in the convention of make_rdm1, the table
    D = cc.eom_rdm1(states[0])
    assert abs(D - D.conj().T).max() < 1e-12 and abs(np.trace(D) - 2) < 1e-10
    assert abs(D - cc.eom_rdm1(states[0].root)).max() == 0 and len(pr.text()) >= 2 * len(e) + 1
    Dao = cc.eom_rdm1(states[0], response=True, ao_repr=True)
    assert abs(np.einsum('pq,qp', Dao, mf.get_ovlp()) - 2) < 1e-9


def cc_dipole(cc, fk, x, r):
    mol = cc.mol
    nuc = np.einsum('i,ix->x', mol.atom_charges(), mol.atom_coords())
    return nuc - np.einsum('xpq,pq->x', r, _fock_density(fk, x.conj(), x)).real


def test_blocks_and_a_label_shift(lih):
    '''LiH with X2C spinors (4 + 18): the densities of the totally
    symmetric states in the blocks of the labels are those of the packed
    block (levels of the packed block that lie in shift 0 with all their
    roots; summed over the level), with the response and the transition
    strengths; a state of shift 2 through the packed block: its density
    conserves the labels, its transition densities change them by -+2.'''
    a, b = lih[True], lih[False]
    ea, ra = a.eom_ee(nroots=[4, 2], conv_tol=1e-8, shift=[0, 2], nguess=6)
    pa = a.eom_properties(ra[0], response=True)
    eb, rb = b.eom_ee(nroots=8, conv_tol=1e-8, nguess=12)
    pb = b.eom_properties(response=True)
    assert abs(pa.dipole_ground - pb.dipole_ground).max() < 1e-8
    n = 0
    for lv in pb.levels:
        ks = [k for k in range(len(ea[0])) if abs(ea[0][k] - eb[lv[0]]) < 1e-8]
        if len(ks) != len(lv):
            continue
        for x, y in ((pa.dipole, pb.dipole), (pa.dipole_response, pb.dipole_response)):
            assert abs(x[ks].sum(axis=0) - y[lv].sum(axis=0)).max() < 2e-6
        assert abs(pa.strength[ks].sum() - pb.strength[lv].sum()) < 2e-6
        n += 1
    assert n >= 2 and abs(pa.dipole - pa.dipole_response).max() > 1e-5
    # shift 2
    st = a.eom_density(ra[2], transition=True)
    lab = np.asarray(a._eris.labels)
    same = lab[:, None] == lab[None, :]
    for s in st:
        assert abs(np.trace(s.d1) - 4) < 1e-9 and abs(s.d1[~same]).max() < 1e-9 and s.r0 == 0
        up = lab[None, :] - lab[:, None]            # label of q minus label of p in <p+ q>
        assert abs(s.right[up != 2]).max() < 1e-9 and abs(s.right).max() > 1e-3
        assert abs(s.left[up != -2]).max() < 1e-9 and abs(s.left).max() > 1e-3
    # ... against the same level of the packed block (the triplet: shifts -2, 0, 2)
    k3 = [k for k in range(8) if abs(eb[k] - ea[2][0]) < 1e-8]
    if len(k3) == 3:
        k0 = [k for k in range(len(ea[0])) if abs(ea[0][k] - ea[2][0]) < 1e-8]
        assert len(k0) == 1
        C, nf, na = a._all_spinors()
        r = np.einsum('pi,xpq,qj->xij', C.conj(), a._dipole_ao(), C)
        mol = a.mol
        nuc = np.einsum('i,ix->x', mol.atom_charges(), mol.atom_coords())
        d2 = nuc - ED.moment(r, st[0].d1).real
        # (the partner of shift -2 has the dipole of shift 2: time reversal)
        assert abs(2 * d2 + pa.dipole[k0[0]] - pb.dipole[k3].sum(axis=0)).max() < 2e-6


def test_frozen_core_in_the_blocks(hbr, monkeypatch):
    """HBr / cc-pVDZ-DK, X2C spinors, 28 frozen and the higher virtual
    spinors dropped (8 + 16): the properties of the totally symmetric
    states in the blocks of the labels (the dipole operator with its
    picture change) against the packed block; the density of a state in
    the convention of make_rdm1 over all spinors - the frozen ones
    occupied, the trace the number of electrons, its dipole that of the
    table; MemoryError with its numbers when the arrays do not fit."""
    a, b = hbr[True], hbr[False]
    ea, ra = a.eom_ee(nroots=3, conv_tol=1e-8, shift=0)
    pa = a.eom_properties()
    assert ra.left.plan.blocks and ra.conv_tol == 1e-8 and ra.left.residuals.max() < 1e-8 and pa.dipole_response is None
    eb, rb = b.eom_ee(nroots=10, conv_tol=1e-8, nguess=14)
    pb = b.eom_properties()
    assert abs(pa.dipole_ground - pb.dipole_ground).max() < 1e-7
    n = 0
    for lv in pb.levels:
        ks = [k for k in range(len(ea)) if abs(ea[k] - eb[lv[0]]) < 1e-8]
        if len(ks) == len(lv):
            assert abs(pa.dipole[ks].sum(axis=0) - pb.dipole[lv].sum(axis=0)).max() < 2e-6
            assert abs(pa.strength[ks].sum() - pb.strength[lv].sum()) < 2e-6
            n += 1
    assert n >= 1
    st = ra.states[0]
    D = a.eom_rdm1(st)
    nel = a.mol.nelectron
    assert D.shape[0] == a.mol.nao_2c() and abs(D - D.conj().T).max() < 1e-13 and abs(np.trace(D) - nel) < 1e-9
    assert abs(D[:28, :28] - np.eye(28)).max() == 0 and not D[28 + 24:].any()
    Dao = a.eom_rdm1(st, ao_repr=True)
    mol = a.mol
    nuc = np.einsum('i,ix->x', mol.atom_charges(), mol.atom_coords())
    dip = nuc - np.einsum('xpq,qp->x', a._dipole_ao(), Dao).real
    assert abs(dip - pa.dipole[0]).max() < 1e-9 and abs(dip[2]) > 0.1
    from zccsdt import memory
    monkeypatch.setattr(memory, 'budget', lambda max_memory=None: 1000)
    with pytest.raises(MemoryError, match='ensities of'):
        a.eom_density(ra)
