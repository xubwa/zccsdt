'''The first derivatives of CCSDT(Q)_Lambda from packed amplitudes
(zccsdt.cc.quadruples_lambda_grad, zccsdt.cc.zeta_opt, the terms of zeta in
zccsdt.cc.density_opt.relaxed_rdm1, GCCSDT.make_rdm1(with_q='lambda'))
against the dense reference (zccsdt.cc.qlgrad_ref; docs/lambda.md 12):

- the pass over the occupied quadruples: the sources of the two linear
  equations, the densities, the derivatives with respect to the orbital
  energies and Gamma, for random amplitudes, in every plan of the memory,
  with and without the C kernels; the memory plan against what is
  allocated (tracemalloc);
- the equation of zeta: every stencil gives the solution of the exact one,
  with the vectors in memory and in a scratch file; its memory plan;
- zeta, lambda', the densities, the orbital gradient between all pairs of
  spinors and the relaxed density for molecules: real and complex spin
  orbitals, X2C spinors, frozen and dropped spinors;
- the front end: against a finite field, in the blocks of the symmetry
  labels and on the packed block, with the other lambdas solved in
  between.'''
import gc

import numpy as np
import pytest
from pyscf import gto, x2c

from test_quadruples import _container, _packed_data
from test_quadruples_grad import _case
from test_quadruples_lambda import _random_lambda
from zccsdt.cc import (ccsdt_opt, ccsdtq_terms as Q, density_opt as D, lambda_opt as LO, qlgrad_ref as GR,
                       quadruples as QD, quadruples_grad as QG, quadruples_lambda as QL, quadruples_lambda_grad as LG,
                       solver, t4pack as P, zeta_opt as Z)


def _rel(a, b):
    return abs(a - b).max() / abs(b).max()


def _hermitized(d):
    '''The part of d2[p,q,r,s] that the derivative of a real energy with
    respect to Hermitian integrals sees.'''
    d = d - d.transpose(1, 0, 2, 3)
    d = (d - d.transpose(0, 1, 3, 2)) / 4
    return 0.5 * (d + d.conj().transpose(2, 3, 0, 1))


@pytest.mark.parametrize('no,nv,cplx,use_c', [(5, 6, True, True), (6, 5, False, True), (4, 7, True, False)])
def test_pass_equals_reference(no, nv, cplx, use_c, monkeypatch):
    '''e, s = de/dt, q = de/dlambda, the densities of the blocks of
    integrals (as an antisymmetric array over all spin orbitals, in the
    form that Hermitian integrals see), the derivatives with respect to the
    orbital energies and Gamma of the virtual spin orbitals: those of the
    dense reference, for amplitudes and lambda that solve nothing, with the
    products in one batch, in the smallest batches and in between.'''
    monkeypatch.setattr(P, 'USE_C', use_c and P.USE_C)
    arr, mo_e = _random_lambda(no, nv, cplx)
    eris = _container(arr['v'], mo_e, no)
    T3, L3 = P.pack(arr['t3']), P.pack(arr['l3'])
    ref = GR.quadruples_gradient(arr['t2'], arr['t3'], arr['l2'], arr['l3'], arr['v'], mo_e, no)
    gam = GR.fock_density(arr['t2'], arr['t3'], arr['l2'], arr['l3'], arr['v'], mo_e, no)
    want = _hermitized(ref.d2)
    isz = 16 if cplx else 8
    for pl in (LG.plan(no, nv, isz), LG.explicit_plan(no, nv, isz, 1, 1, 3, 1), LG.explicit_plan(no, nv, isz, 3, 7, 3, 2)):
        g = LG.kernel(eris, arr['t2'], T3, arr['l2'], L3, plan_=pl)
        assert abs(g.e_q - ref.e.real) < 1e-12 * abs(ref.e) and abs(g.e_imag - ref.e.imag) < 1e-12 * abs(ref.e)
        assert abs(g.e_q - QL.kernel(eris, arr['t2'], T3, arr['l2'], L3)[1]) < 1e-12 * abs(ref.e)
        assert _rel(g.s2, ref.s2) < 1e-12 and _rel(g.s3, Q.unique(ref.s3, 3)) < 1e-12
        assert _rel(g.q2, ref.q2) < 1e-12 and _rel(g.q3, Q.unique(ref.q3, 3)) < 1e-12
        assert _rel(g.de, ref.de) < 1e-12 and _rel(g.gamma_vv, gam[no:, no:]) < 1e-12
        got = _hermitized(QG.dense_d2(g, no))
        o, v = slice(0, no), slice(no, None)
        for name in ('vvoo', 'vvvo', 'ovoo', 'oooo', 'ovvo', 'vvvv'):
            s = tuple(o if c == 'o' else v for c in name)
            assert _rel(got[s], want[s]) < 1e-12, name
        assert _rel(got, want) < 1e-12                      # (and nothing elsewhere)
    # the sources of the two equations are not those of (Q)
    plain = QG.kernel(eris, arr['t2'], T3)
    assert _rel(g.s3, plain.s3) > 1e-2 and abs(g.q3).max() > 1e-2


@pytest.mark.parametrize('no,nv,cplx', [(5, 20, True), (6, 26, False)])
def test_memory_plan(no, nv, cplx):
    '''plan(): monotone in the budget, within it, the whole plan when all
    fits, MemoryError with the numbers below the floor; s3, q3 and the
    density of <bc||ef> in a scratch file where the largest batches do not
    fit, the density alone where s3 and q3 fit with them, and the results
    are those of the arrays in memory; what a run allocates (tracemalloc
    follows numpy's allocations, not the mapped arrays) stays within
    Plan.need, from the smallest batches up, and the plan is not a loose
    bound.'''
    from zccsdt.scratch import is_mapped
    import tracemalloc
    eris, t2, T3 = _packed_data(no, nv, cplx)
    _, l2, L3 = _packed_data(no, nv, cplx, seed=2)
    isz = 16 if cplx else 8
    full = LG.plan(no, nv, isz)
    budgets = [int(b) for b in np.linspace(full.floor, 1.05 * full.need, 40)]
    plans = [LG.plan(no, nv, isz, b) for b in budgets]
    key = [(p.q.cols22, p.q.cols13, p.q.xv, p.q.rows) for p in plans]
    for a, b in zip(key, key[1:]):
        assert all(x <= y for x, y in zip(a, b))
    assert all(p.need <= b for p, b in zip(plans, budgets))
    order = [LG.MAPPED.index(p.mapped) for p in plans]
    assert order == sorted(order) and order[0] == 0 and order[-1] == 2 and 1 in order
    top = (full.q.cols22, full.q.cols13, full.q.xv, full.q.rows)
    assert all(k == top for k, p in zip(key, plans) if p.mapped != LG.MAPPED[0])
    assert all(p.need > 0.9 * b for p, b, k in zip(plans, budgets, key) if k != top)
    assert key[-1] == top and plans[-1].need == full.need
    assert full.need > QG.plan(no, nv, isz).need and full.kept > QG.plan(no, nv, isz).kept
    with pytest.raises(MemoryError, match=r'derivatives of \(Q\)_Lambda need [0-9.]+ MB .* [0-9.]+ MB are left'):
        LG.plan(no, nv, isz, full.floor - 1)
    least = LG.explicit_plan(no, nv, isz, *QD._least(nv)[:2], 3, QD._least(nv)[2], mapped=LG.MAPPED[0])
    assert least.need == least.floor == plans[0].need < full.need
    g0 = LG.kernel(eris, t2, T3, l2, L3, plan_=full)
    assert g0.scratch is None and not is_mapped(g0.vvvv) and not is_mapped(g0.s3) and not is_mapped(g0.q3)
    seen = []
    for pl in [least, plans[len(plans) // 4], plans[order.index(1)], full]:
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        g = LG.kernel(eris, t2, T3, l2, L3, plan_=pl)
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        assert is_mapped(g.vvvv) == ('vvvv' in pl.mapped)
        assert is_mapped(g.s3) == ('s3' in pl.mapped) and is_mapped(g.q3) == ('q3' in pl.mapped)
        for a, b in ((g.s2, g0.s2), (g.s3, g0.s3), (g.q2, g0.q2), (g.q3, g0.q3), (g.vvvv, g0.vvvv), (g.vvvo, g0.vvvo),
                     (g.gamma_vv, g0.gamma_vv), (g.src3[0], g0.src3[0]), (g.src3[1], g0.src3[1])):
            assert _rel(a, b) < 1e-11
        print('%s: allocated %.2f MB, planned %.2f MB (%.2f)' % (pl.mapped, peak / 1e6, pl.need / 1e6, peak / pl.need))
        assert peak <= pl.need, (peak, pl)
        seen.append(peak / pl.need)
        del g
    assert min(seen) > 0.8
    gc.collect()                # (the mappings of the scratch files go with their arrays)


def test_fewer_than_four():
    eris, t2, T3 = _packed_data(3, 6, False)
    with pytest.raises(ValueError):
        LG.kernel(eris, t2, T3, t2, T3)


# -------------------------------------------------------------------- molecules

TOL = 1e-11


def _solve(eris):
    '''Converged CCSDT, Lambda and the pass over the quadruples.'''
    r = solver.kernel(eris, 'ccsdt', solver.Options(conv_tol=1e-13, conv_tol_normt=TOL, incore_complete=True))
    opts = solver.Options(conv_tol_normt=TOL, incore_complete=True)
    rl = LO.kernel(eris, r.t1, r.t2, r.t3, opts)
    assert r.converged and rl.converged
    return r, rl, LG.kernel(eris, r.t2, r.t3, rl.l2, rl.l3), opts


@pytest.mark.parametrize('name,kind,nfrozen,ndrop', [('bh', 'real', 0, 0), ('lih', 'complex', 0, 0), ('h2o', 'x2c', 2, 0),
                                                    ('bh', 'x2c', 2, 2)])
def test_relaxed_density_equals_reference(name, kind, nfrozen, ndrop):
    '''Converged CCSDT and Lambda, the pass over the quadruples, zeta
    (zccsdt.cc.zeta_opt.kernel), the sources of lambda' and lambda'
    (lambda_opt.kernel(sources=)), density_opt.relaxed_rdm1(zeta=): zeta,
    the sources, lambda', the density with the orbital energies of E(Q)_L,
    the orbital gradient between all pairs of spinors and the relaxed
    density are those of the dense reference - with the rotations among
    occupied and among virtual spinors, between frozen and active ones,
    and t1 that is not small in the gradient (BH, LiH).  The derivatives
    along zeta that are taken once: the exact stencil and the central
    differences give the same.'''
    mf, C, eris = _case(name, kind, nfrozen, ndrop)
    n = C.shape[1]
    r, rl, q, opts = _solve(eris)
    o = eris.nocc
    v = n - ndrop - nfrozen - o
    unique = lambda x: Q.unique(x, 3)                                           # noqa: E731
    packed = lambda x: np.asarray(x).reshape(len(Q.tuples(o, 3)), -1)           # noqa: E731
    t3, l3 = (Q.expand(packed(x), o, v, 3) for x in (r.t3, rl.l3))
    ref = GR.relaxed_rdm1(mf, C, nfrozen, ndrop, amps=[r.t1, r.t2, t3], lams=[rl.l1, rl.l2, l3], tol=TOL)
    assert abs(q.e_q - ref.e_ql) < 1e-13 and abs(q.e_imag) < 1e-13
    # zeta
    rz = Z.kernel(eris, r.t1, r.t2, r.t3, q.q2, q.q3, opts)
    zeta = (rz.z1, rz.z2, rz.z3)
    assert rz.converged and rz.forward > 3 and rz.stencil_error < 1e-13
    for a, b in zip((rz.z1, rz.z2, packed(rz.z3)), (ref.terms.zeta[0], ref.terms.zeta[1], unique(ref.terms.zeta[2]))):
        assert abs(a - b).max() < 1e-10
    assert abs(rz.z1).max() > 1e-7 and abs(rz.z2).max() > 1e-7
    # the sources of lambda' (their rank 1 does not vanish), and lambda'
    src = Z.lambda_sources(eris, r.t1, r.t2, r.t3, rl.l1, rl.l2, rl.l3, zeta, q.s2, q.s3, opts)
    for stencil, tol in (('exact', 1e-13), ('central', 1e-12)):
        s = Z.lambda_sources(eris, r.t1, r.t2, r.t3, rl.l1, rl.l2, rl.l3, ref.terms.zeta[:2] + [unique(ref.terms.zeta[2])],
                             q.s2, q.s3, opts, stencil=stencil)
        for a, b in zip((s[0], LG.pairs_to_dense(s[1], o, v), packed(s[2])),
                        (ref.terms.sources[0], ref.terms.sources[1], unique(ref.terms.sources[2]))):
            assert abs(a - b).max() < tol
    assert abs(src[0]).max() > 1e-9
    rq = LO.kernel(eris, r.t1, r.t2, r.t3, opts, l1=rl.l1, l2=rl.l2, l3=rl.l3, sources=src)
    assert rq.converged and rq.ncycle <= rl.ncycle
    for a, b in zip((rq.l1, rq.l2, packed(rq.l3)), (ref.terms.lams[0], ref.terms.lams[1], unique(ref.terms.lams[2]))):
        assert abs(a - b).max() < 1e-10
    assert abs(rq.l2 - rl.l2).max() > 1e-7
    # the densities
    kw = dict(frozen_occ=C[:, :nfrozen] if nfrozen else None, dropped_vir=C[:, n - ndrop:] if ndrop else None, tol=TOL)
    Ca = C[:, nfrozen:n - ndrop]
    rel = D.relaxed_rdm1(mf, eris, r.t1, r.t2, r.t3, rq.l1, rq.l2, rq.l3, Ca, q=q, zeta=zeta,
                         lams=(rl.l1, rl.l2, rl.l3), **kw)
    assert abs(rel.unrelaxed - ref.unrelaxed).max() < 1e-10
    assert abs(rel.gradient - ref.gradient).max() < 1e-10 and abs(ref.gradient).max() > 1e-7
    assert abs(rel.relaxed - ref.relaxed).max() < 1e-10
    assert abs(rel.relaxed - rel.relaxed.conj().T).max() < 1e-13
    assert abs(np.trace(rel.relaxed) - o - nfrozen) < 1e-10
    two = D.relaxed_rdm1(mf, eris, r.t1, r.t2, r.t3, rq.l1, rq.l2, rq.l3, Ca, q=q, zeta=zeta,
                         lams=(rl.l1, rl.l2, rl.l3), stencil='central', **kw)
    assert abs(two.relaxed - rel.relaxed).max() < 1e-12 and abs(two.gradient - rel.gradient).max() < 1e-12
    # without zeta and lambda-bar the density is another, and so is that of CCSDT
    plain = D.relaxed_rdm1(mf, eris, r.t1, r.t2, r.t3, rl.l1, rl.l2, rl.l3, Ca, **kw)
    assert abs(plain.relaxed - rel.relaxed).max() > (1e-7 if name == 'lih' else 1e-6)


def test_zeta_stencils():
    '''The equation of zeta for BH with complex spin orbitals: forward
    differences first, central differences, and the exact formula in every
    iteration end at the same vector, with fewer amplitude steps the lower
    the ladder starts; the central step is the exact one to the rounding;
    with the vectors of the equation in a scratch file the same.'''
    from zccsdt.scratch import is_mapped
    mf, C, eris = _case('bh', 'complex', 0, 0)
    r, rl, q, opts = _solve(eris)
    res = {}
    for stencil in ('exact', 'central', 'forward'):
        log = []
        opts.callback = lambda info: log.append(info['stencil'])
        res[stencil] = rz = Z.kernel(eris, r.t1, r.t2, r.t3, q.q2, q.q3, opts, stencil=stencil)
        assert rz.converged and not rz.plan.mapped and not is_mapped(rz.z3)
        assert len(log) == rz.ncycle and log[-1] != 'forward' and set(log) <= {'forward', 'central', 'exact'}
        assert log.count('forward') == rz.forward and (rz.forward > 3) == (stencil == 'forward')
        if stencil != 'exact':
            assert 'exact' not in log and rz.stencil_error < 1e-13
    opts.callback = None
    ex, ce, fo = (res[k] for k in ('exact', 'central', 'forward'))
    assert ex.steps == 4 * ex.ncycle and ce.steps == 2 * ce.ncycle + 2 and fo.steps < ce.steps < ex.steps
    assert fo.steps == 1 + fo.forward + 2 * (fo.ncycle - fo.forward) + 2
    for other in (ce, fo):
        for a, b in zip((other.z1, other.z2, other.z3), (ex.z1, ex.z2, ex.z3)):
            assert abs(a - b).max() < 1e-11
    assert abs(ex.z3).max() > 1e-6 and np.iscomplexobj(ex.z3)
    # in a scratch file
    plan = Z.plan_for(eris, ex.z3.dtype, opts)
    assert not plan.mapped and plan.vectors >= Z.NVEC * ex.z3.nbytes
    opts.plan = Z.Plan(plan.amps, True, plan.amps.need)
    opts.incore_complete = False
    rm = Z.kernel(eris, r.t1, r.t2, r.t3, q.q2, q.q3, opts)
    assert rm.converged and is_mapped(rm.z3) and rm.steps == fo.steps
    for a, b in zip((rm.z1, rm.z2, rm.z3), (fo.z1, fo.z2, fo.z3)):      # (two converged iterations)
        assert abs(a - b).max() < 1e-11
    with pytest.raises(ValueError):
        Z.kernel(eris, r.t1, r.t2, r.t3, q.q2, q.q3, opts, stencil='five')
    del rm
    gc.collect()


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
    '''GCCSDT.make_rdm1(with_q='lambda') for HF / 6-31G with X2C spinors, two
    frozen (real spinors adapted to m_j, T3 in the blocks of the labels):
    the derivative of E(CCSDT) + E(Q)_L with the SCF repeated in the field.
    lambda of CCSDT stays what the object holds, and (Q)_Lambda, the
    density of CCSDT(Q) and that of CCSDT(Q)_Lambda come out the same in
    any order; Lambda, lambda' and the densities in the blocks of the
    labels give the density of the packed block; GCCSDTQ hands the request
    to its CCSDT step.'''
    import zccsdt
    mol, mf, cc = _hf(0.0)
    assert np.size(cc.t3) < 56 * 220
    z = mol.intor('int1e_r_spinor')[2]
    Dq = cc.make_rdm1(ao_repr=True, with_q='lambda')
    assert cc.converged_zeta and cc.converged_lambda_ql and cc.converged_lambda and not cc._l_with_q
    assert cc.ql_gradient_sectors and not cc.rdm1_blocks and not cc.ql_info['lambda']['blocks']
    info = cc.ql_info['zeta']
    assert info['forward'] > 3 and info['steps'] < 2 * info['cycles'] and info['stencil_error'] < 1e-12
    l2, lq, zeta = cc.l2.copy(), cc.l2_ql.copy(), cc.zeta2.copy()
    assert abs(lq - l2).max() > 1e-6 and abs(zeta).max() > 1e-6
    assert np.size(cc.zeta3) == np.size(cc.t3) == np.size(cc.l3_ql)
    e_ql = cc.e_ql
    assert abs(cc.ccsdt_q_lambda() - e_ql) < 1e-13 and abs(cc.e_tot_ql - cc.e_tot - e_ql) < 1e-13
    assert abs(Dq - Dq.conj().T).max() < 1e-12
    assert abs(np.einsum('pq,qp', mol.intor('int1e_ovlp_spinor'), Dq).real - 10) < 1e-9
    h = 1e-3

    def energy(f):
        c = _hf(f)[2]
        return c.e_tot + c.ccsdt_q_lambda()
    e = {f: energy(f) for f in (-2 * h, -h, h, 2 * h)}
    ff = (e[-2 * h] - 8 * e[-h] + 8 * e[h] - e[2 * h]) / (12 * h)
    rel = np.einsum('pq,qp', z, Dq).real
    print('HF / 6-31G X2C, 2 frozen: relaxed CCSDT(Q)_L %.12f finite field %.12f, difference %.1e' % (rel, ff, rel - ff))
    assert abs(rel - ff) < 2e-9
    # the other densities are other numbers, and they leave nothing behind that this one takes
    D0 = cc.make_rdm1(ao_repr=True)
    D1 = cc.make_rdm1(ao_repr=True, with_q=True)
    assert cc._l_with_q and abs(cc.l2 - l2).max() > 1e-6
    for other in (D0, D1):
        assert abs(np.einsum('pq,qp', z, other).real - rel) > 5e-6
    assert abs(cc.ccsdt_q_lambda() - e_ql) < 1e-10 and not cc._l_with_q
    cc.solve_lambda(with_q=True)
    l1q, l2q, l3q = cc.solve_lambda(with_q='lambda')
    assert not cc._l_with_q and abs(cc.l2 - l2).max() < 1e-9 and abs(l2q - lq).max() < 1e-9
    assert l2q is cc.l2_ql and abs(cc.zeta2 - zeta).max() < 1e-9
    assert abs(cc.make_rdm1(ao_repr=True, with_q='lambda') - Dq).max() < 5e-9
    assert cc.solve_lambda(with_q='lambda')[2] is l3q            # (kept)
    with pytest.raises(ValueError):
        cc.make_rdm1(with_q='q')
    # CCSDTQ: its CCSDT step
    cq = zccsdt.GCCSDTQ(mf, frozen=2)
    cq.ccsdt_run = cc
    assert cq.solve_lambda(with_q='lambda')[2] is l3q
    assert abs(cq.make_rdm1(ao_repr=True, with_q='lambda') - Dq).max() < 5e-9
    # in the blocks of the labels
    cb = _hf(0.0, blocks_min=0)[2]
    Db = cb.make_rdm1(ao_repr=True, with_q='lambda')
    assert cb.lambda_blocks and cb.rdm1_blocks and cb.ql_info['lambda']['blocks'] and not cb._l_with_q
    assert abs(Db - Dq).max() < 5e-9
    # every iteration of zeta with the exact formula: the same density
    cb.zeta_stencil, cb._ql_solution = 'exact', None
    assert abs(cb.make_rdm1(ao_repr=True, with_q='lambda') - Db).max() < 1e-10
    assert cb.ql_info['zeta']['steps'] == 4 * cb.ql_info['zeta']['cycles']
    del cc, cb, cq
    gc.collect()


def test_zeta_memory_plan(monkeypatch):
    '''plan_for(): with memory for an amplitude step and the vectors of the
    equation everything is in memory, and what a run allocates stays within
    the plan; with less, the vectors and the DIIS vectors go to a scratch
    file and the amplitude step has the plan of the amplitude equations -
    the run allocates that much less, and gives the same zeta; below what
    an amplitude step needs, its MemoryError with the numbers.'''
    import tracemalloc
    from zccsdt import eris as E, memory
    from zccsdt.scratch import is_mapped
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='6-31g', verbose=0)
    mf = x2c.UHF(mol)
    mf.conv_tol = 1e-11
    mf.kernel()
    eris = E.from_scf(mf)               # (complex spinors, no labels: T3 on the packed block, 1 MB)
    r = solver.kernel(eris, 'ccsdt', solver.Options(conv_tol=1e-10, conv_tol_normt=1e-8, incore_complete=True))
    rl = LO.kernel(eris, r.t1, r.t2, r.t3, solver.Options(conv_tol_normt=1e-8, incore_complete=True))
    q = LG.kernel(eris, r.t2, r.t3, rl.l2, rl.l3)
    dtype = np.result_type(r.t3)
    t3 = np.size(r.t3) * dtype.itemsize
    opts = solver.Options(conv_tol_normt=1e-9, diis_space=4)
    monkeypatch.setattr(memory, 'budget', lambda max_memory=None: None)
    full = Z.plan_for(eris, dtype, opts)
    extra = (Z.NVEC + 2 * opts.diis_space) * t3
    assert t3 > 1e6 and not full.mapped and full.vectors == extra and full.need == full.amps.need + extra
    amps = ccsdt_opt.plan_for(eris, dtype, opts)
    assert amps.need == full.amps.need and not amps.t3_file
    # a budget that an amplitude step fits in, the vectors next to it not
    left = amps.need - ccsdt_opt.cached_bytes(eris) + extra // 2
    monkeypatch.setattr(memory, 'budget', lambda max_memory=None: left)
    monkeypatch.setattr(memory, 'available', lambda: left)
    tight = Z.plan_for(eris, dtype, opts)
    assert tight.mapped and tight.vectors == 0 and tight.need == tight.amps.need and not tight.amps.t3_file
    assert tight.amps.pp_bytes is None and not tight.amps.pp_file
    monkeypatch.setattr(memory, 'budget', lambda max_memory=None: left + extra)
    assert not Z.plan_for(eris, dtype, opts).mapped
    monkeypatch.setattr(memory, 'budget', lambda max_memory=None: -10 ** 11)
    monkeypatch.setattr(memory, 'available', lambda: -10 ** 11)
    with pytest.raises(MemoryError, match=r'CCSDT with 10 occupied and 16 virtual spin orbitals needs about [0-9.]+ GB'):
        Z.plan_for(eris, dtype, opts)
    out = {}
    for name, plan in (('memory', full), ('file', tight)):
        ccsdt_opt.release_cached(eris)
        gc.collect()
        opts.plan = plan
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        rz = Z.kernel(eris, r.t1, r.t2, r.t3, q.q2, q.q3, opts)
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        print('zeta, vectors in %s: allocated %.2f MB, planned %.2f MB, T3 %.2f MB, %d cycles'
              % (name, peak / 1e6, plan.need / 1e6, t3 / 1e6, rz.ncycle))
        assert rz.converged and rz.ncycle > 6 and is_mapped(rz.z3) == plan.mapped and peak < plan.need
        out[name] = (peak, rz)
    # (two vectors are in memory either way: the shifted amplitudes and the new vector)
    assert 0.8 * extra < out['memory'][0] - out['file'][0] < 1.1 * extra
    for k in ('z1', 'z2', 'z3'):        # (two converged iterations)
        assert abs(np.asarray(getattr(out['file'][1], k)) - getattr(out['memory'][1], k)).max() < 1e-9
    del out, rz
    gc.collect()
