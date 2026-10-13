'''Left-hand eigenvectors of EOM-EE-CCSDT (zccsdt.cc.eom_left; docs/eom.md,
section 12):

- the left product l A against the transposed dense Jacobian (eom_ref.matrix)
  at amplitudes that solve nothing, real and complex;
- with a label shift: in the blocks of the labels against the packed block,
  every shift, real and with a phase on every spinor;
- converged left vectors: their eigenvalues are those of the right vectors,
  L_k R_m = delta_km, the reference row (Lambda R_k = -r0); the blocks of
  the labels against the packed block; a degenerate level;
- the memory plan against what is allocated.'''
import gc
import tracemalloc

import numpy as np
import pytest

import zccsdt
from test_eom_opt import pack2, pack3, random_case
from test_eom_shift import _lex, _p2, _phase3, _phased, _phases, _random_vector
from zccsdt.cc import ccsdt_opt, ccsdtq_ref as R, eom, eom_left as EL, eom_opt, eom_ref as EOM, solver, t3pack


def _vec(om, vm, r, dtype):
    return [np.ascontiguousarray(r[0], dtype=dtype).ravel(), np.ascontiguousarray(pack2(r[1], om, vm), dtype=dtype).ravel(),
            np.ascontiguousarray(pack3(r[2], om, vm), dtype=dtype).ravel()]


@pytest.mark.parametrize('name', ['lih', 'h2o'])
@pytest.mark.parametrize('cplx', [False, True])
def test_left_product_against_the_transposed_jacobian(name, cplx):
    '''l A of the Lambda machinery = (dense Jacobian)^T l to 1e-11 of its
    size, for random amplitudes that solve nothing (the Jacobian at any
    point) and a random l; LiH / STO-3G (4 + 8) and H2O with the 1s pair
    frozen (8 + 4), real and complex spin orbitals.'''
    eris, amps, l = random_case(name, cplx, True, seed=3)
    fock, g, mo_e, no = R.integrals(eris)
    om, vm = t3pack.maps_for(eris)
    dtype = np.complex128 if cplx else np.float64
    A = EOM.matrix(amps, fock, g, no)
    ops = EL.LeftOps(eris, amps[0], amps[1], pack3(amps[2], om, vm).ravel())
    assert ops.mode == 'same' and not ops.blocks
    x = _vec(om, vm, l, dtype)
    y = [np.zeros_like(p) for p in x]
    ops.apply(x, y)
    want = np.concatenate(x) @ A
    got = np.concatenate(y)
    assert abs(got - want).max() < 1e-11 * abs(want).max()
    off = np.cumsum([0] + [p.size for p in x])
    for k in range(3):          # (each rank by itself)
        assert abs(want[off[k]:off[k + 1]]).max() > 1e-2
        assert abs(got - want)[off[k]:off[k + 1]].max() < 1e-11 * abs(want[off[k]:off[k + 1]]).max()
    # the constant term of the Lambda equations is the row of the reference: eta = <0| [Hbar, tau] |0>
    e1, e2 = ops.eta()
    z = [np.zeros_like(p) for p in x]
    from zccsdt.cc import lambda_opt
    r1, r2, r3 = lambda_opt.residuals(ops.im, np.zeros_like(amps[0]), np.zeros_like(amps[1]), z[2], ops.T3)
    assert abs(r1 - e1).max() < 1e-13 and abs(r2 - e2).max() < 1e-13 and abs(r3).max() < 1e-13


from test_eom_shift import lih    # noqa: E402,F401  (the fixture: CCSDT of LiH with X2C spinors)


def _apply(ops, x):
    y = [np.zeros_like(p) for p in x]
    ops.apply(x, y)
    return y


def test_left_product_with_a_label_shift(lih):
    '''LiH, X2C spinors (4 + 18, labels -3 .. 3): l A of a random vector of
    every shift - in the blocks of the labels for the totally symmetric
    one, through the packed block for the others - is the one of the packed
    block without labels, and l (A r) = (l A) r with the sigma vector of
    the blocks of that shift (zccsdt.cc.t3shift: other code); with a phase
    on every spinor the product has the conjugate phases.'''
    cc = lih[True]
    eb = cc._eris
    ed, t3d = cc._dense_view()
    omb, vmb = t3pack.maps_for(eb)
    omd, vmd = t3pack.maps_for(ed)
    gb = eom_opt.prepare(eb, cc.t1, cc.t2, cc.t3)
    sh = gb.shifter()
    opd = EL.LeftOps(ed, cc.t1, cc.t2, t3d)
    assert opd.mode == 'same' and not opd.blocks
    u = np.exp(2j * np.pi * np.random.default_rng(5).random(omb.n + vmb.n))
    f1, f2, po, pv = _phases(omb, vmb, u)
    ec = _phased(eb, u)
    t3c = _phase3(sh.layout(0), np.asarray(cc.t3), po, pv)
    for s in sh.shifts():
        L = sh.layout(s)
        ops = EL.LeftOps(eb, cc.t1, cc.t2, cc.t3, shift=s if s else None)
        assert (ops.mode, ops.blocks) == (('same', True) if s == 0 else ('dense', False))
        l1, l2, rng = _random_vector(omb, vmb, s, 40 + s)
        l3 = rng.standard_normal(L.size)
        x = [l1.ravel().copy(), _p2(l2, omb, vmb).ravel(), l3]
        y = _apply(ops, x)
        yd = _apply(opd, [l1.ravel().copy(), _p2(l2, omd, vmd).ravel(), _lex(L, l3).ravel()])
        y2 = t3pack.unpack_pairs(yd[1], omd, vmd)
        y3 = yd[2].reshape(omd.ntri, vmd.ntri)
        assert abs(y[0] - yd[0]).max() < 1e-10 * abs(y2).max()
        assert abs(t3pack.unpack_pairs(y[1], omb, vmb) - y2).max() < 1e-10 * abs(y2).max()
        assert abs(_lex(L, y[2]) - y3).max() < 1e-10 * abs(y3).max() and abs(y3).max() > 1e-3
        m1, m2 = gb.masks(s)
        assert not y[0].reshape(m1.shape)[~m1].any() and not y[1].reshape(m2.shape)[~m2].any()
        # l (A r) = (l A) r
        r1, r2, rng = _random_vector(omb, vmb, s, 60 + s)
        r3 = rng.standard_normal(L.size)
        r = [r1.ravel(), _p2(r2, omb, vmb).ravel(), r3]
        sb = eom_opt.sigma(gb, r1, _p2(r2, omb, vmb), r3, shift=s if s else None)
        a, b = EL._dotu(x, [np.asarray(p).ravel() for p in sb]), EL._dotu(y, r)
        assert abs(a - b) < 1e-10 * abs(a) and abs(a) > 1e-2
        if s == 0:      # the totally symmetric vectors through the packed block as well
            yy = _apply(EL.LeftOps(eb, cc.t1, cc.t2, cc.t3, dense=True), x)
            assert max(abs(p - q).max() for p, q in zip(y, yy)) < 1e-11
        if s in (0, 1, -2):
            oc = EL.LeftOps(ec, cc.t1 * f1, cc.t2 * f2, t3c, shift=s if s else None)
            assert oc.dtype == np.complex128 and oc.blocks == (s == 0)
            yc = _apply(oc, [(l1 * f1.conj()).ravel(), _p2(l2 * f2.conj(), omb, vmb).ravel(),
                             _phase3(L, l3, po.conj(), pv.conj())])
            assert abs(yc[0] - y[0] * f1.conj().ravel()).max() < 1e-11 * abs(y2).max()
            assert abs(yc[1] - y[1] * _p2(f2.conj(), omb, vmb).ravel()).max() < 1e-11 * abs(y2).max()
            assert abs(yc[2] - _phase3(L, y[2], po.conj(), pv.conj())).max() < 1e-11 * abs(y3).max()


def _rank_weights(right, left, lv):
    '''sum over the roots ``lv`` of a level of sum_mu l_mu r_mu by rank.'''
    return np.array([sum(np.dot(np.ravel(l[k]), np.ravel(r[k])) for k in lv)
                     for l, r in ((left.l1, right.r1), (left.l2, right.r2), (left.l3, right.r3))])


def test_left_vectors_lih(lih):
    '''LiH: the left vectors of the shifts 0 and 2 in the blocks of the
    labels and of eight roots of the packed block (levels of 3, 1, 2 and 2
    roots): the eigenvalues of the left problem are
    those of the right one to 1e-9, L_k R_m = delta_km, a degenerate level
    is biorthogonalized, and the weights of the ranks in L_k R_k of the
    singlets agree between the blocks and the packed block.'''
    a, b = lih[True], lih[False]
    ea, ra = a.eom_ee(nroots=[4, 3], conv_tol=1e-8, shift=[0, 2], nguess=6)
    la = a.eom_left(conv_tol=1e-8)
    assert set(la) == {0, 2} and ra[0].left is la[0]
    assert la[0].plan.blocks and la[0].ops.mode == 'same' and la[2].ops.mode == 'dense' and la[2].shift == 1
    for d in (0, 2):
        l, r = la[d], ra[d]
        assert l.converged.all() and abs(l.e - r.e).max() < 1e-9 and abs(l.e_imag).max() < 1e-9
        assert abs(l.overlap.diagonal() - 1).max() < 1e-12 and abs(l.overlap - np.eye(len(r.e))).max() < 1e-8
        assert l.biorthogonality < 1e-8
        assert l.l3[0].size == r.r3[0].size and l.l1[0].shape == r.r1[0].shape
        m1, m2 = r.ground.masks(d // 2)
        assert not l.l1[0][~m1].any() and not l.l2[0][~m2].any()
    eb, rb = b.eom_ee(nroots=8, conv_tol=1e-8, nguess=12)
    lb = b.eom_left(conv_tol=1e-8)
    assert rb.left is lb and not lb.plan.blocks and lb.ops.mode == 'same'
    assert lb.converged.all() and abs(lb.e - rb.e).max() < 1e-9
    assert abs(lb.overlap.diagonal() - 1).max() < 1e-12 and abs(lb.overlap - np.eye(8)).max() < 1e-8
    assert lb.biorthogonality < 1e-8
    assert sorted(len(x) for x in lb.levels) == [1, 2, 2, 3]
    # the levels that lie in shift 0 with all their roots (the singlet, a pair): the same
    # weights of the ranks in L R, summed over the level, in the blocks
    n = 0
    for lv in lb.levels:
        ks = [k for k in range(len(ea[0])) if abs(ea[0][k] - eb[lv[0]]) < 1e-8]
        if len(ks) == len(lv):
            wa, wb = _rank_weights(ra[0], la[0], ks), _rank_weights(rb, lb, lv)
            assert abs(wa - wb).max() < 1e-6 and abs(wb.sum() - len(lv)) < 1e-11 and abs(wb[2]) > 1e-6
            n += 1
    assert n >= 2
    # only some of the roots, and the intermediates used again
    one = EL.kernel(b._eris, b.t1, b.t2, b.t3, rb, roots=[3], conv_tol=1e-8, ops=lb.ops)
    assert one.roots == [3] and abs(one.e[0] - rb.e[3]) < 1e-9 and abs(one.overlap[0, 0] - 1) < 1e-12


def test_left_vectors_of_a_sector():
    """LiH / STO-3G, spin orbitals of RHF (the label 2 m_s, by masks on the
    packed block): the left vectors of shift 2 - the M_S = 1 components of
    the triplets - stay in the shift, with the eigenvalues of the right
    ones; shift 0 through the front end as well."""
    from pyscf import scf
    from test_ccsdtq_ref import _scf
    cc = zccsdt.CCSDT(scf.addons.convert_to_ghf(_scf('lih')))
    cc.verbose, cc.conv_tol, cc.conv_tol_normt = 0, 1e-12, 1e-10
    cc.kernel()
    e, res = cc.eom_ee(nroots=2, conv_tol=1e-8, shift=[2, 0])
    left = cc.eom_left(conv_tol=1e-8)
    for d in (2, 0):
        l, r = left[d], res[d]
        assert l.ops.sector is not None and l.ops.sector.shift == d and l.ops.mode == 'same'
        assert l.converged.all() and abs(l.e - r.e).max() < 1e-9
        assert abs(l.overlap.diagonal() - 1).max() < 1e-12 and l.biorthogonality < 1e-8
        sec = l.ops.sector
        for k in range(2):
            assert not l.l1[k][~sec.m1].any() and not l.l2[k][~sec.m2].any() and abs(l.l1[k]).max() > 0.1
    assert abs(e[2][0] - e[0][0]) < 1e-8        # the triplet: its components M_S = 1 and 0


@pytest.mark.parametrize('cplx', [False, True])
def test_left_vectors_against_the_dense_matrix(cplx):
    '''LiH / STO-3G, 4 + 8 spin orbitals, real and complex: the left vector
    of the lowest singlet is the left eigenvector of the dense matrix
    (scipy) scaled to L R = 1, and the projector sum_k R_k L_k of the
    triplet below it is that of the dense eigenvectors of the level.'''
    import scipy.linalg
    from pyscf import scf
    from test_ccsdtq_ref import _scf, _spin_mixed
    from zccsdt import eris as E
    from zccsdt.cc import ccsdtq_terms as Q
    mf = scf.addons.convert_to_ghf(_scf('lih'))
    eris = E.from_scf(mf, mo_coeff=_spin_mixed(mf)[0] if cplx else mf.mo_coeff)
    rt = solver.kernel(eris, 'ccsdt', solver.Options(conv_tol=1e-12, conv_tol_normt=1e-10))
    res = eom.kernel(eris, rt.t1, rt.t2, rt.t3, nroots=4, conv_tol=1e-9)
    left = EL.kernel(eris, rt.t1, rt.t2, rt.t3, res, conv_tol=1e-9)
    assert left.converged.all() and abs(left.e - res.e).max() < 1e-10 and left.levels == [[0, 1, 2], [3]]
    assert abs(left.overlap.diagonal() - 1).max() < 1e-12 and abs(left.overlap - np.eye(4)).max() < 1e-8
    assert left.biorthogonality < 1e-8
    fock, g, mo_e, no = R.integrals(eris)
    om, vm = t3pack.maps_for(eris)
    amps = [rt.t1, rt.t2, Q.expand(np.asarray(rt.t3).reshape(om.ntri, vm.ntri), no, vm.n, 3)]
    A = EOM.matrix(amps, fock, g, no)
    w, vl, vr = scipy.linalg.eig(A, left=True)
    rv = [np.concatenate([np.ravel(x[k]) for x in (res.r1, res.r2, res.r3)]) for k in range(4)]
    lv = [np.concatenate([np.ravel(x[k]) for x in (left.l1, left.l2, left.l3)]) for k in range(4)]
    # the singlet
    j = np.flatnonzero(abs(w - res.e[3]) < 1e-7)
    assert len(j) == 1
    ld = vl[:, j[0]].conj()
    ld = ld / (ld @ rv[3])
    assert abs(ld - lv[3]).max() < 1e-6 * abs(ld).max()
    # the triplet: the projector of the level
    j = np.flatnonzero(abs(w - res.e[0]) < 1e-7)
    assert len(j) == 3
    Vr, Vl = vr[:, j], vl[:, j].conj()
    P = Vr @ np.linalg.inv(Vl.T @ Vr) @ Vl.T
    P2 = sum(np.outer(rv[k], lv[k]) for k in range(3))
    assert abs(P - P2).max() < 1e-6 and abs(P2).max() > 0.1


def test_memory_against_the_plan(monkeypatch, tmp_path):
    '''The plan of the left vectors and what a run allocates (tracemalloc
    follows numpy's arrays): H2O / cc-pVDZ, 8 + 38 spin orbitals, two
    roots - below the plan (less T3, which is there, and the allowance for
    the threads) and above half of it; a budget that does not hold the
    subspace asked for gives a smaller one, then the scratch file, then
    the error with its numbers.'''
    from test_eom import _water
    from zccsdt import disk, memory
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    eris, rt = _water('cc-pvdz')
    dtype = np.float64
    res = eom.kernel(eris, rt.t1, rt.t2, rt.t3, nroots=2, conv_tol=1e-3, max_cycle=3, max_space=6)
    sizes = res.ground.sizes()
    res.ground.c.clear()                        # (the operands of the right-hand side go)
    ccsdt_opt.release_cached(eris)
    opts = solver.Options(incore_complete=True)
    plan = EL.plan_for(eris, rt.t1, dtype, 2, sizes, max_space=6, opts=opts)
    assert plan.max_space == 6 and not plan.mapped and plan.mode == 'same' and not plan.blocks
    om, vm = t3pack.maps_for(eris)
    t3b = om.ntri * vm.ntri * 8
    one, big = sum(sizes) * 8, sizes[2] * 8
    arrays = plan.need - t3b - ccsdt_opt.overhead()
    gc.collect()
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    left = EL.kernel(eris, rt.t1, rt.t2, rt.t3, res, conv_tol=1e-3, max_cycle=3, plan=plan)
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    assert left.nsigma >= 4
    assert 0.5 * arrays < peak <= arrays, (peak, arrays, plan)
    # the rank-3 parts of the subspace in a scratch file: the same vectors
    import dataclasses
    from zccsdt.scratch import is_mapped
    pm = dataclasses.replace(plan, mapped=True, disk=2 * 6 * big)
    lm = EL.kernel(eris, rt.t1, rt.t2, rt.t3, res, conv_tol=1e-3, max_cycle=3, plan=pm)
    assert is_mapped(lm.l3[0]) and not is_mapped(left.l3[0]) and lm.nsigma == left.nsigma
    assert abs(lm.e - left.e).max() < 1e-10 and abs(lm.l1[0] - left.l1[0]).max() < 1e-8
    # budgets
    monkeypatch.setattr(ccsdt_opt, 'cached_bytes', lambda eris: 0)
    # (the buffers of the passes shrink before the subspace does: the smallest need of a subspace)
    small = solver.Options(incore_complete=True, mem_mb=1e-3)
    need7 = EL.plan_for(eris, rt.t1, dtype, 2, sizes, max_space=7, opts=small).need
    assert need7 < plan.need
    monkeypatch.setattr(memory, 'budget', lambda max_memory=None: need7 - t3b + 1000)
    p = EL.plan_for(eris, rt.t1, dtype, 2, sizes)
    assert not p.mapped and p.max_space == 7 and 'in memory' in p.text() and p.need <= need7 + 1000
    assert p.lam.mem_mb < 2000.0
    monkeypatch.setattr(disk, 'budget', lambda max_disk=None, where=None: 2 * 8 * big + 5)
    lam0 = need7 - (2 * 7 * one - 2 * big)                  # the Lambda plan without the subspace
    monkeypatch.setattr(memory, 'budget', lambda max_memory=None: lam0 - t3b + 2 * 10 * (one - big) + 1000)
    p = EL.plan_for(eris, rt.t1, dtype, 2, sizes)
    assert p.mapped and p.max_space == 8 and p.disk == 16 * big and 'scratch file' in p.text()
    monkeypatch.setattr(disk, 'budget', lambda max_disk=None, where=None: 5)
    with pytest.raises(MemoryError, match='do not fit in memory'):
        EL.plan_for(eris, rt.t1, dtype, 2, sizes)
