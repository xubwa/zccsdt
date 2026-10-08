'''The first derivatives of CCSDT(Q) from packed amplitudes
(zccsdt.cc.quadruples_grad, the sources in zccsdt.cc.lambda_opt, the (Q)
terms of zccsdt.cc.density_opt.relaxed_rdm1, GCCSDT.make_rdm1(with_q=True))
against the dense reference (zccsdt.cc.qgrad_ref):

- the pass over the occupied quadruples: sources, densities, the
  derivatives with respect to the orbital energies and Gamma, for random
  amplitudes, in every plan of the memory, with and without the C kernels;
- the memory plan against what is allocated (tracemalloc);
- Lambda with the sources, the relaxed density and the orbital gradient
  for molecules: complex spin orbitals, X2C spinors, frozen and dropped
  spinors (Kramers pairs);
- the front end: on the packed block and in the blocks of the symmetry
  labels, and against a finite field.'''
import gc

import numpy as np
import pytest
from pyscf import gto, scf, x2c

from test_ccsdtq_ref import _scf, _spin_mixed
from test_quadruples import _container, _packed_data, _random
from zccsdt import eris as E
from zccsdt.cc import (ccsdtq_terms as Q, density_opt as D, lambda_opt as LO, qgrad_ref as GR, quadruples as QD,
                       quadruples_grad as QG, solver, t4pack as P)

WATER = 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587'


def _rel(a, b):
    return abs(a - b).max() / abs(b).max()


def _asym(d):
    d = d - d.transpose(1, 0, 2, 3)
    return (d - d.transpose(0, 1, 3, 2)) / 4


@pytest.mark.parametrize('no,nv,cplx', [(5, 6, True), (6, 5, False), (4, 7, True)])
@pytest.mark.parametrize('use_c', [True, False])
def test_pass_equals_reference(no, nv, cplx, use_c, monkeypatch):
    '''E(Q), s2, s3, the densities of the six blocks of integrals (as
    antisymmetric arrays over all spin orbitals), the derivatives with
    respect to the orbital energies and Gamma of the virtual spin orbitals:
    those of the dense reference, for amplitudes that solve nothing, with
    the products in one batch, in the smallest batches and in between.'''
    monkeypatch.setattr(P, 'USE_C', use_c and P.USE_C)
    arr, mo_e = _random(no, nv, cplx)
    eris = _container(arr['v'], mo_e, no)
    T3 = P.pack(arr['t3'])
    ref = GR.quadruples_gradient(arr['t2'], arr['t3'], arr['v'], mo_e, no)
    gam = GR.fock_density(arr['t2'], arr['t3'], arr['v'], mo_e, no)
    isz = 16 if cplx else 8
    for pl in (QG.plan(no, nv, isz), QG.explicit_plan(no, nv, isz, 1, 1, 3, 1), QG.explicit_plan(no, nv, isz, 3, 7, 3, 2)):
        g = QG.kernel(eris, arr['t2'], T3, plan_=pl)
        assert abs(g.e_q - ref.e_q) < 1e-12 * abs(ref.e_q)
        assert abs(g.e_q - QD.kernel(eris, arr['t2'], T3)[1]) < 1e-12 * abs(ref.e_q)
        assert _rel(g.s2, ref.s2) < 1e-12 and _rel(g.s3, Q.unique(ref.s3, 3)) < 1e-12
        assert _rel(g.de, ref.de) < 1e-12 and _rel(g.gamma_vv, gam[no:, no:]) < 1e-12
        d2, r2 = _asym(QG.dense_d2(g, no)), _asym(ref.d2)
        o, v = slice(0, no), slice(no, None)
        for name in ('vvoo', 'vvvo', 'ovoo', 'oooo', 'ovvo', 'vvvv'):
            s = tuple(o if c == 'o' else v for c in name)
            assert _rel(d2[s], r2[s]) < 1e-12, name
        assert _rel(d2, r2) < 1e-12                         # (and nothing elsewhere)


@pytest.mark.parametrize('no,nv,cplx', [(5, 20, True), (6, 26, False)])
def test_memory_plan(no, nv, cplx):
    '''plan(): monotone in the budget, within it, the whole plan when all
    fits, MemoryError with the numbers below the floor; s3 and the density
    of <bc||ef> in a scratch file where the largest batches do not fit, the
    density alone where s3 fits with them, and the results are those of the
    arrays in memory; what a run allocates (tracemalloc follows numpy's
    allocations, not the mapped arrays) stays within Plan.need, from the
    smallest batches up, and the plan is not a loose bound.'''
    from zccsdt.scratch import is_mapped
    import tracemalloc
    eris, t2, T3 = _packed_data(no, nv, cplx)
    isz = 16 if cplx else 8
    full = QG.plan(no, nv, isz)
    budgets = [int(b) for b in np.linspace(full.floor, 1.05 * full.need, 40)]
    plans = [QG.plan(no, nv, isz, b) for b in budgets]
    key = [(p.q.cols22, p.q.cols13, p.q.xv, p.q.rows) for p in plans]
    for a, b in zip(key, key[1:]):
        assert all(x <= y for x, y in zip(a, b))
    assert all(p.need <= b for p, b in zip(plans, budgets))
    # what is mapped: both arrays below the largest batches, then the density, then nothing
    order = [QG.MAPPED.index(p.mapped) for p in plans]
    assert order == sorted(order) and order[0] == 0 and order[-1] == 2 and 1 in order
    top = (full.q.cols22, full.q.cols13, full.q.xv, full.q.rows)
    assert all(k == top for k, p in zip(key, plans) if p.mapped != QG.MAPPED[0])
    # as large as fits: at least nine tenths of every budget are used while the batches grow
    assert all(p.need > 0.9 * b for p, b, k in zip(plans, budgets, key) if k != top)
    assert key[-1] == (full.q.cols22, full.q.cols13, full.q.xv, full.q.rows) and plans[-1].need == full.need
    with pytest.raises(MemoryError, match=r'derivatives of \(Q\) need [0-9.]+ MB .* [0-9.]+ MB are left'):
        QG.plan(no, nv, isz, full.floor - 1)
    least = QG.explicit_plan(no, nv, isz, *QD._least(nv)[:2], 3, QD._least(nv)[2], mapped=QG.MAPPED[0])
    assert least.need == least.floor == plans[0].need < full.need
    g0 = QG.kernel(eris, t2, T3, plan_=full)
    assert g0.scratch is None and not is_mapped(g0.vvvv) and not is_mapped(g0.s3)
    seen = []
    for pl in [least, plans[len(plans) // 4], plans[order.index(1)], full]:
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        g = QG.kernel(eris, t2, T3, plan_=pl)
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        assert is_mapped(g.vvvv) == ('vvvv' in pl.mapped) and is_mapped(g.s3) == ('s3' in pl.mapped)
        for a, b in ((g.s2, g0.s2), (g.s3, g0.s3), (g.vvvv, g0.vvvv), (g.vvvo, g0.vvvo), (g.gamma_vv, g0.gamma_vv),
                     (g.src3[0], g0.src3[0]), (g.src3[1], g0.src3[1])):
            assert _rel(a, b) < 1e-11
        assert peak <= pl.need, (peak, pl)
        seen.append(peak / pl.need)
        del g
    assert min(seen) > 0.8


def test_fewer_than_four():
    eris, t2, T3 = _packed_data(3, 6, False)
    with pytest.raises(ValueError):
        QG.kernel(eris, t2, T3)


# -------------------------------------------------------------------- molecules

def _case(name, kind, nfrozen, ndrop):
    '''(SCF, all spinors, integrals of the active ones).'''
    if kind == 'x2c':
        mol = gto.M(atom=WATER if name == 'h2o' else 'B 0 0 0; H 0 0 1.2324', basis='sto-3g', verbose=0)
        mf = x2c.UHF(mol)
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-10, 200
        mf.kernel()
        C = mf.mo_coeff
    else:
        rhf = _scf(name)
        rhf.conv_tol, rhf.conv_tol_grad = 1e-13, 1e-10
        rhf.kernel()
        mf = scf.addons.convert_to_ghf(rhf)
        C = _spin_mixed(mf)[0] if kind == 'complex' else mf.mo_coeff
    n = C.shape[1]
    eris = E.from_scf(mf, mo_coeff=C[:, :n - ndrop], mo_occ=mf.mo_occ[:n - ndrop], frozen=nfrozen)
    return mf, C, eris


@pytest.mark.parametrize('name,kind,nfrozen,ndrop', [('bh', 'real', 0, 0), ('bh', 'complex', 0, 0), ('lih', 'complex', 0, 0),
                                                    ('h2o', 'x2c', 2, 0), ('bh', 'x2c', 2, 2)])
def test_relaxed_density_equals_reference(name, kind, nfrozen, ndrop):
    '''Converged CCSDT, the pass over the quadruples, Lambda with the
    sources (zccsdt.cc.lambda_opt) and density_opt.relaxed_rdm1(q=): lambda,
    the density with the orbital energies of E(Q), the orbital gradient
    between all pairs of spinors and the relaxed density are those of the
    dense reference - with the rotations among occupied and among virtual
    spinors, between frozen and active ones (H2O and BH with X2C: the
    Kramers pairs of the frozen spinors), and t1 that is not small in the
    gradient (BH, LiH).'''
    mf, C, eris = _case(name, kind, nfrozen, ndrop)
    n = C.shape[1]
    r = solver.kernel(eris, 'ccsdt', solver.Options(conv_tol=1e-13, conv_tol_normt=1e-11, incore_complete=True))
    o = eris.nocc
    v = n - ndrop - nfrozen - o
    q = QG.kernel(eris, r.t2, r.t3)
    opts = solver.Options(conv_tol_normt=1e-11, incore_complete=True)
    rl = LO.kernel(eris, r.t1, r.t2, r.t3, opts, sources=(None, q.s2, q.s3))
    assert r.converged and rl.converged
    rel = D.relaxed_rdm1(mf, eris, r.t1, r.t2, r.t3, rl.l1, rl.l2, rl.l3, C[:, nfrozen:n - ndrop],
                         frozen_occ=C[:, :nfrozen] if nfrozen else None,
                         dropped_vir=C[:, n - ndrop:] if ndrop else None, tol=1e-11, q=q)
    t3 = Q.expand(np.asarray(r.t3).reshape(len(Q.tuples(o, 3)), -1), o, v, 3)
    ref = GR.relaxed_rdm1(mf, C, nfrozen, ndrop, amps=[r.t1, r.t2, t3])
    assert abs(q.e_q - ref.e_q) < 1e-13
    for a, b in zip((rl.l1, rl.l2, np.asarray(rl.l3).reshape(-1, len(Q.tuples(v, 3)))),
                    (ref.lams[0], ref.lams[1], Q.unique(ref.lams[2], 3))):
        assert abs(a - b).max() < 1e-10
    assert abs(rel.unrelaxed - ref.unrelaxed).max() < 1e-10
    assert abs(rel.gradient - ref.gradient).max() < 1e-10 and abs(ref.gradient).max() > 1e-7
    assert abs(rel.relaxed - ref.relaxed).max() < 1e-10
    assert abs(rel.relaxed - rel.relaxed.conj().T).max() < 1e-13
    assert abs(np.trace(rel.relaxed) - o - nfrozen) < 1e-10
    # lambda of CCSDT is another one, and without the (Q) terms the density is another
    r0 = LO.kernel(eris, r.t1, r.t2, r.t3, opts)
    assert abs(r0.l2 - rl.l2).max() > 1e-6
    if name != 'lih':               # (E(Q) of LiH is 1e-7)
        plain = D.relaxed_rdm1(mf, eris, r.t1, r.t2, r.t3, r0.l1, r0.l2, r0.l3, C[:, nfrozen:n - ndrop],
                               frozen_occ=C[:, :nfrozen] if nfrozen else None,
                               dropped_vir=C[:, n - ndrop:] if ndrop else None, tol=1e-11)
        assert abs(plain.relaxed - rel.relaxed).max() > 1e-6


def test_sources_in_the_blocks_of_the_labels():
    '''lambda_opt.kernel(sources=) with the equations in the blocks of the
    symmetry labels (BH with X2C spinors adapted to m_j): the solution of
    the packed block.'''
    import zccsdt
    mol = gto.M(atom='B 0 0 0; H 0 0 1.2324', basis='sto-3g', verbose=0)
    mf = x2c.UHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-10
    mf.kernel()
    cc = zccsdt.CCSDT(mf)
    cc.verbose, cc.conv_tol, cc.conv_tol_normt = 0, 1e-13, 1e-11
    cc.kernel()
    eris = cc._eris
    assert getattr(eris, 'use_labels', False) and np.size(cc.t3) < 20 * 20
    q = cc._q_terms()
    assert abs(q.e_q - cc.ccsdt_q()) < 1e-13
    opts = solver.Options(conv_tol_normt=1e-11, incore_complete=True)
    src = (None, q.s2, q.s3)
    rb = LO.kernel(eris, cc.t1, cc.t2, cc.t3, opts, blocks=True, sources=src)
    dense, t3 = cc._dense_view()
    rd = LO.kernel(dense, cc.t1, cc.t2, t3, opts, sources=src)
    assert rb.converged and rd.converged and rb.blocks and not rd.blocks
    assert abs(rb.l1 - rd.l1).max() < 1e-10 and abs(rb.l2 - rd.l2).max() < 1e-10
    assert abs(LO.packed_l3(eris, rb.l3) - np.asarray(rd.l3).reshape(20, 20)).max() < 1e-10
    plain = LO.kernel(dense, cc.t1, cc.t2, t3, opts)
    assert abs(plain.l2 - rd.l2).max() > 1e-6


def _hf(field, blocks_min=None, frozen=2):
    import zccsdt
    mol = gto.M(atom='F 0 0 0; H 0 0 0.9168', basis='6-31g', verbose=0)
    mf = x2c.UHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-10
    if field:
        h0, z = mf.get_hcore(), mol.intor('int1e_r_spinor')[2]
        mf.get_hcore = lambda *args, **kwargs: h0 + field * z
    mf.kernel()
    cc = zccsdt.CCSDT(mf, frozen=frozen)
    cc.verbose, cc.conv_tol, cc.conv_tol_normt = 0, 1e-12, 1e-10
    if blocks_min is not None:
        cc.blocks_min = blocks_min
    cc.kernel()
    return mol, mf, cc


def test_front_end():
    '''GCCSDT.make_rdm1(with_q=True) for HF / 6-31G with X2C spinors, two
    frozen (real spinors adapted to m_j, T3 in the blocks of the labels):
    the derivative of E(CCSDT) + E(Q) with the SCF repeated in the field;
    Lambda and the density in the blocks of the labels give the same
    density as on the packed block; lambda is solved again when the other
    one is asked for, and (Q)_Lambda takes that of CCSDT.'''
    mol, mf, cc = _hf(0.0)
    assert np.size(cc.t3) < 56 * 220
    z = mol.intor('int1e_r_spinor')[2]
    Dq = cc.make_rdm1(ao_repr=True, with_q=True)
    assert cc._l_with_q and cc.converged_lambda and not cc.rdm1_blocks
    lq = cc.l2.copy()
    assert abs(cc.e_q - cc._q_terms().e_q) == 0
    e_q = cc.e_q
    assert abs(cc.ccsdt_q() - e_q) < 1e-12
    assert abs(Dq - Dq.conj().T).max() < 1e-12
    assert abs(np.einsum('pq,qp', mol.intor('int1e_ovlp_spinor'), Dq).real - 10) < 1e-9
    h = 1e-3

    def energy(f):
        c = _hf(f)[2]
        return c.e_tot + c.ccsdt_q()
    e = {f: energy(f) for f in (-2 * h, -h, h, 2 * h)}
    ff = (e[-2 * h] - 8 * e[-h] + 8 * e[h] - e[2 * h]) / (12 * h)
    rel = np.einsum('pq,qp', z, Dq).real
    print('HF / 6-31G X2C, 2 frozen: relaxed CCSDT(Q) %.12f finite field %.12f, difference %.1e' % (rel, ff, rel - ff))
    assert abs(rel - ff) < 2e-9
    D0 = cc.make_rdm1(ao_repr=True)
    assert not cc._l_with_q and abs(cc.l2 - lq).max() > 1e-6
    assert abs(np.einsum('pq,qp', z, D0).real - rel) > 1e-5
    # in the blocks of the labels
    cb = _hf(0.0, blocks_min=0)[2]
    Db = cb.make_rdm1(ao_repr=True, with_q=True)
    assert cb.lambda_blocks and cb.rdm1_blocks and cb._l_with_q
    assert abs(Db - Dq).max() < 5e-9
    # (Q)_Lambda wants lambda of CCSDT
    e_ql = cb.ccsdt_q_lambda()
    assert not cb._l_with_q
    cc.solve_lambda()
    assert abs(cc.ccsdt_q_lambda() - e_ql) < 1e-10
