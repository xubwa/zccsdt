'''EOM-IP- and EOM-EA-CCSDT with short vectors (zccsdt.cc.eom.ghost_short;
docs/eom.md, section 5): the ghost spinor carries a label of its own, and
the vectors r[i], r[(i<j), a], r[(i<j<k), (a<b)] (EA: r[a], r[i, (a<b)],
r[(i<j), (a<b<c)]) are the vectors of one label shift

- against the dense IP / EA matrices of the reference and against the EE
  problem with the ghost spinor at its full length (LiH; real, complex;
  CCSD, CCSDT), against PySCF (H2O);
- in the blocks of the symmetry labels, by the label of the electron: LiH
  and HBr with X2C spinors against the packed layout and the ghost route.'''
import numpy as np
import pytest
from pyscf import gto, scf, x2c

from test_ccsdtq_ref import _scf, _spin_mixed
from test_eom_shift import _cc
from zccsdt import eris as E
from zccsdt.cc import ccsdtq_ref as R, ccsdtq_terms as Q, eom, eom_ref, solver, t3pack

TIGHT = solver.Options(conv_tol=1e-12, conv_tol_normt=1e-10)


@pytest.mark.parametrize('cplx', [False, True])
@pytest.mark.parametrize('method', ['ccsd', 'ccsdt'])
def test_short_vectors_against_the_reference_and_the_ghost_route(cplx, method):
    '''LiH / STO-3G (4 + 8): four roots of IP and EA with short vectors -
    rank-3 parts of C(4,3) C(8,2) = 112 and C(4,2) C(8,3) = 336 elements
    where the EE problem with the ghost has 336 and 560 - are the lowest
    eigenvalues of the dense matrices of the reference (1e-9) and the
    roots of the ghost route (1e-10), with as many sigma vectors.'''
    mf = scf.addons.convert_to_ghf(_scf('lih'))
    C = _spin_mixed(mf)[0] if cplx else mf.mo_coeff
    eris = E.from_scf(mf, mo_coeff=C, packed_vvvv=not cplx)
    rt = solver.kernel(eris, method, TIGHT)
    fock, g, mo_e, no = R.integrals(E.from_scf(mf, mo_coeff=C))
    om, vm = t3pack.maps_for(eris)
    with3 = method == 'ccsdt'
    t3 = rt.t3 if with3 else None
    amps = [rt.t1, rt.t2] + ([Q.expand(np.asarray(rt.t3).reshape(om.ntri, vm.ntri), no, vm.n, 3)] if with3 else [])
    for kind, n3, n1, n2 in (('ip', 112, (4,), (6, 8)), ('ea', 336, (8,), (4, 28))):
        sh = eom.ghost_short(eris, rt.t1, rt.t2, t3, kind)
        assert not sh.labelled and sh.class_shift() == 1 and sh.labels() == [0] and sh.t3 is t3
        with pytest.raises(ValueError, match='needs the symmetry labels in use'):
            sh.class_shift(1)
        res = eom.kernel(sh.eris, sh.t1, sh.t2, sh.t3, nroots=4, conv_tol=1e-9, name=kind.upper(), shift=sh.class_shift())
        w = eom_ref.eigenvalues(amps, fock, g, no, kind)
        assert res.converged.all() and abs(res.e - w.real[:4]).max() < 1e-9
        e2, t1, t2, t3g, singles = eom.ghost(eris, rt.t1, rt.t2, t3, kind)
        old = eom.kernel(e2, t1, t2, t3g, nroots=4, conv_tol=1e-9, singles=singles)
        assert abs(res.e - old.e).max() < 1e-10 and res.nsigma == old.nsigma
        if with3:
            assert res.r3[0].size == n3 and old.r3[0].size == (336 if kind == 'ip' else 560)
            assert res.r3[0].dtype == (np.complex128 if cplx else np.float64)
        # the vector without the ghost index: all of its norm
        x1, x2, x3 = sh.strip(res, 1)
        assert x1.shape == n1 and x2.shape == n2 and (x3 is None) == (not with3)
        norm = np.vdot(x1, x1).real + np.vdot(x2, x2).real + (np.vdot(x3, x3).real if with3 else 0)
        assert abs(norm - 1) < 1e-12 and abs(x1).max() > 0.5


def test_short_vectors_against_pyscf():
    '''EOM-IP- and EA-CCSD of H2O / STO-3G against PySCF's roots.'''
    from conftest import water_ghf
    from test_eom_ref import pyscf_roots
    eris = E.from_scf(water_ghf('sto-3g'))
    rt = solver.kernel(eris, 'ccsd', TIGHT)
    ref = pyscf_roots()
    for kind in ('ip', 'ea'):
        sh = eom.ghost_short(eris, rt.t1, rt.t2, None, kind)
        res = eom.kernel(sh.eris, sh.t1, sh.t2, None, nroots=4, conv_tol=1e-9, shift=sh.class_shift())
        assert abs(res.e - np.sort(ref[kind])[:4]).max() < 1e-8


@pytest.fixture(scope='module')
def lih():
    mf = x2c.UHF(gto.M(atom='Li 0 0 0; H 0 0 1.5957', basis='6-31g', verbose=0)).run(conv_tol=1e-12)
    return {sym: _cc(mf, sym) for sym in (True, False)}


def test_front_end_and_label_blocks_lih(lih):
    '''LiH / 6-31G, X2C (4 + 18; labels 2 m_j = -3 .. 3).  eom_ip() and
    eom_ea(): short vectors on the packed layout (612 and 4896 triples
    where the ghost route has 3876 and 8160) with the roots of the ghost
    route; by the label of the electron in the blocks of the labels (176
    of the 612; 1220 and 806 of the 4896): the same roots, each label an
    eigenproblem of its own, +-label degenerate and computed once.'''
    a, b = lih[True], lih[False]
    for kind, n, sizes, labels in (('ip', 4, (612, 3876), {-1: 176, 1: 176}),
                                   ('ea', 6, (4896, 8160), {-3: 806, -1: 1220, 1: 1220, 3: 806})):
        f = getattr(a, 'eom_' + kind)
        ep, rp = f(nroots=n, conv_tol=1e-9, nguess=n + 4)
        assert rp.converged.all() and rp.r3[0].size == sizes[0] and getattr(a, 'eom_%s_result' % kind) is rp
        # (a calculation on the packed block gives the same)
        ex, rx = getattr(b, 'eom_' + kind)(nroots=n, conv_tol=1e-9, nguess=n + 4)
        assert rx.r3[0].size == sizes[0] and abs(ex - ep).max() < 1e-9
        eo, ro = getattr(b, 'eom_' + kind)(nroots=n, conv_tol=1e-9, nguess=n + 4, short=False)
        assert ro.r3[0].size == sizes[1] and abs(eo - ep).max() < 1e-10
        eb, rb = f(nroots=2, conv_tol=1e-9, shift='all', partners=True)
        assert sorted(eb) == sorted(labels) and all(r.converged.all() for r in rb.values())
        assert {d: rb[d].r3[0].size for d in eb} == labels and rp.short.labels() == [0] and rb[1].short.labels() == sorted(labels)
        cut = ep[-1] + 1e-7
        union = np.sort(np.concatenate([e[e < cut] for e in eb.values()]))
        assert len(union) == n and abs(union - ep).max() < 1e-10
        for d in labels:
            assert abs(eb[d] - eb[-d]).max() < 1e-10 and rb[d].label_shift == d
        e2, r2 = f(nroots=1, conv_tol=1e-8, shift='all')
        assert r2[-1].partner_of == 1 and r2[1].partner_of is None and e2[-1][0] == e2[1][0] and abs(e2[1][0] - ep[0]) < 1e-8
        e1, r1 = f(nroots=1, conv_tol=1e-8, shift=-1)
        assert abs(e1[0] - ep[0]) < 1e-8 and r1.partner_of is None
        x1, x2, x3 = r1.short.strip(r1, 0)
        assert x1.shape == ((4,) if kind == 'ip' else (18,)) and x3.size == labels[-1] and abs(x1).max() > 0.9
        with pytest.raises(ValueError, match='no %s state changes the labels by 2' % kind.upper()):
            f(nroots=1, shift=2)
        with pytest.raises(ValueError, match='labels in use'):
            getattr(b, 'eom_' + kind)(nroots=1, shift=1)
    assert 0.2 < a.eom_ip(nroots=1)[0][0] < 0.4


def test_label_blocks_hbr():
    '''HBr / cc-pVDZ-DK, X2C, 28 frozen, 8 + 16: the eight ionization
    energies of the valence spinors (four Kramers pairs) by the labels
    -+1 (three each) and -+3 (one each) are those of the packed layout,
    to 1e-10; the plan of a label counts its short vectors.'''
    mol = gto.M(atom='Br 0 0 0; H 0 0 1.4145', basis='cc-pvdz-dk', verbose=0)
    mf = x2c.UHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-8
    mf.kernel()
    cc = _cc(mf, True, frozen=list(range(28)) + list(range(36 + 16, mol.nao_2c())))
    eb, rb = cc.eom_ip(nroots=[3, 1, 3, 1], conv_tol=1e-9, shift=[1, 3, -1, -3], nguess=3)
    assert all(r.converged.all() for r in rb.values()) and rb[-1].partner_of == 1 and rb[-3].partner_of == 3
    ep, rp = cc.eom_ip(nroots=8, conv_tol=1e-9)
    assert rp.converged.all() and rp.r3[0].size == 56 * 120
    union = np.sort(np.concatenate(list(eb.values())))
    assert abs(union - ep).max() < 1e-10 and 0.3 < ep[0] < 0.5
    # the lowest: the pi spinors with |2 m_j| = 3, then those with 1
    assert abs(eb[3][0] - ep[0]) < 1e-10 and abs(eb[1][0] - ep[2]) < 1e-10 and ep[2] - ep[0] > 1e-3
    r = rb[1]
    n3 = r.r3[0].size
    assert n3 < 56 * 120 / 4 and r.plan.vectors == 2 * r.plan.max_space * 8 * (8 * 17 + 28 * 136 + n3)
    ea, ra = cc.eom_ea(nroots=1, conv_tol=1e-8, shift=1)
    ex, rx = cc.eom_ea(nroots=2, conv_tol=1e-8)
    assert ra.converged.all() and abs(ex - ea[0]).min() < 1e-8
