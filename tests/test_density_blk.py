'''The densities and the orbital gradient of CCSDT with t3 and lambda3 in the
blocks of a symmetry label (zccsdt.cc.density_blk, density_opt.kernel(...,
blocks=True)) against the code on the packed block (density_opt without
blocks, itself against the tables in test_density_opt.py):

- the passes over t3 and l3, Gamma and its products one by one, and the
  density, the gradient and the contractions with the integrals of a frozen
  index for amplitudes and Lambda that solve nothing: random integrals with
  one label (no symmetry) and with four, real and complex, in every plan;
- HBr / cc-pVDZ-DK with X2C spinors (8 + 28 active, 28 frozen, the labels
  2 m_j of the front end): relaxed and unrelaxed density and the orbital
  gradient; with dropped virtual spinors; with one label; with complex
  amplitudes (a phase on every spinor);
- the front end: in blocks with ``symmetry``, on the packed block without it
  or without memory for the blocks;
- the memory plan against what a run allocates, and over a scan of budgets.'''
import gc
import tracemalloc

import numpy as np
import pytest
from pyscf import gto, x2c

import zccsdt
from test_lambda_blk import LABELS, MODELS, _dense, _labelled_model, _packed
from zccsdt.cc import ccsdt_opt, density_blk as B, density_opt as D, lambda_opt as LO, solver, t3engine, \
    t3kern, t3pack

pytestmark = pytest.mark.skipif(not (t3kern.USE_C and t3pack.USE_C), reason='C libraries not built')

MODELS = dict(MODELS, wide=([-1, -1, 1, 1], [-3] * 6 + [-1] * 16 + [1] * 16 + [3] * 6))


def _rel(a, b):
    return abs(a - b).max() / abs(b).max()


def _frozen(nf, n, cplx, seed=3):
    '''A source of random integrals <fu||vw> of nf "frozen spinors".'''
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((nf, n, n, n)) + (1j * rng.standard_normal((nf, n, n, n)) if cplx else 0)
    g = g - g.transpose(0, 1, 3, 2)
    return D._Given(nf, lambda f0, f1: g[f0:f1], n)


@pytest.mark.parametrize('labels,cplx', [('one', True), ('one', False), ('four', True), ('four', False)])
def test_passes_in_blocks(labels, cplx):
    '''Every product of the passes over t3 and l3 in blocks is the sum over
    the half-expanded arrays, with whole batches and with one occupied pair
    and one triple per batch; Gamma is block diagonal and its product with
    three-index integrals that of the full matrix.'''
    no, nv = 5, 7
    eris, t, l = _labelled_model(no, nv, *LABELS[labels], seed=11, cplx=cplx)
    T3b = ccsdt_opt.blocked_t3(eris, _packed(t[2], 3))
    L3b = ccsdt_opt.blocked_t3(eris, _packed(l[2], 3))
    eng = B.engine(eris, T3b.dtype)
    om, vm = eng.om, eng.vm
    assert om.ncls == (1 if labels == 'one' else 4) and eng.size == T3b.size
    Td, Ld = eng.L.to_dense(T3b), eng.L.to_dense(L3b)       # (the triples in the order of their groups)
    nP, nQ = om.npair, vm.npair
    mm, nn, md, ne = om.pairs[:, 0], om.pairs[:, 1], vm.pairs[:, 0], vm.pairs[:, 1]
    t2mn, l2pp = t[1][mm, nn], np.ascontiguousarray(l[1][mm, nn][:, md, ne])
    t2pp, t2P = np.ascontiguousarray(t2mn[:, md, ne]), np.ascontiguousarray(t[1][:, :, md, ne])
    T = t3pack.gather_half(Td, om, vm, np.arange(nP))
    L = t3pack.gather_half(Ld, om, vm, np.arange(nP))
    rng = np.random.default_rng(0)
    # operands with the labels of those of the densities
    co, cv, gP, gQ = om.spins, vm.spins, om.pair_grp, vm.pair_grp
    GxT = rng.standard_normal((nP, nv, nv + no)) * (
        gP[:, None, None] - cv[None, :, None] == np.concatenate((cv, co))[None, None, :])
    GyT = rng.standard_normal((no, nQ, no + nv)) * (
        gQ[None, :, None] - co[:, None, None] == np.concatenate((co, cv))[None, None, :])
    GxT, GyT = GxT.astype(T3b.dtype), GyT.astype(T3b.dtype)
    l1v = np.ascontiguousarray(l[0].T).ravel()
    want = dict(
        Z=np.einsum('PgmQ,imQ->Pgi', L, t2P), C3x=np.einsum('Pce,PeiQ->ciQ', t2mn, L),
        XT=np.einsum('Pdx,PdiQ->xiQ', GxT, T), YT=np.einsum('PdmQ,mQy->Pdy', T, GyT),
        B1=np.einsum('PaiQ,PQ->ai', T, l2pp).ravel(), A1=np.einsum('em,PemQ->PQ', l[0].T, T),
        R3=np.einsum('PajQ,PbiQ->ajbi', L, T).reshape(nv * no, nv * no),
        Kg=np.einsum('PdmR,PdmQ->RQ', np.einsum('PdmQ,RQ->PdmR', L, t2pp), T))
    for nb in (100, 1):
        got = {}
        got['Z'], got['C3x'] = eng.pass_l(L3b, t2mn, t2P, nb)
        XT = np.zeros((nv + no, no * nQ), dtype=T3b.dtype)
        got['YT'] = np.zeros((nP, nv, no + nv), dtype=T3b.dtype)
        got['B1'], got['A1'], got['R3'], got['Kg'] = eng.pass_tl(
            T3b, L3b, GxT, GyT.reshape(no * nQ, no + nv), l2pp, l1v, t2pp, nb, XT, got['YT'])
        got['XT'] = XT.reshape(nv + no, no, nQ)
        for k, w in want.items():
            assert abs(w).max() > 1 and _rel(got[k], w) < 1e-13, (k, nb)
    Tv = t3pack.gather_vir_half(Td, vm, np.arange(om.ntri))
    Lv = t3pack.gather_vir_half(Ld, vm, np.arange(om.ntri))
    full = 3 * Lv.reshape(nQ, -1) @ Tv.reshape(nQ, -1).T
    lex = t3pack.triple_map(nv)
    order = lex.pidx[md, ne]                    # the pairs of the groups in lexicographic numbers
    V = rng.standard_normal((nQ, nv, 4)).astype(T3b.dtype)
    for ng in (100, 1):
        Gam = eng.gamma(T3b, L3b, ng)
        rest = full.copy()
        for (lo, hi), G in zip(vm.blocks, Gam):
            assert G.shape == (hi - lo, hi - lo)
            assert hi == lo or abs(G - full[lo:hi, lo:hi]).max() < 1e-13 * abs(full).max()
            rest[lo:hi, lo:hi] = 0
        assert abs(rest).max() < 1e-12
    Glex = np.zeros_like(full)
    Glex[np.ix_(order, order)] = full
    Vlex = np.zeros_like(V)
    Vlex[order] = V
    for cols in (True, False):
        a = np.zeros((nv, 4), dtype=T3b.dtype)
        b = np.zeros((nv, 4), dtype=T3b.dtype)
        D._gamma_pairs(a, Glex, Vlex, lex, cols, 0.7)
        D._gamma_pairs(b, Gam, V, vm, cols, 0.7)
        assert abs(a).max() > 1 and _rel(b, a) < 1e-13


@pytest.mark.parametrize('labels,cplx', [('one', True), ('one', False), ('four', True), ('four', False)])
def test_kernel_against_the_packed_block(labels, cplx):
    '''d1, g1, the table GRADIENT and the contractions with the integrals of
    a frozen index: those of the packed block to 1e-13 of their size, in
    every plan - whole batches and kept pair blocks of <ab||de>, one pair
    and one triple per batch with one row of a block at a time, and between
    them - and the density alone.'''
    no, nv = 5, 7
    eris, t, l = _labelled_model(no, nv, *LABELS[labels], seed=11, cplx=cplx)
    T3p, L3p = _packed(t[2], 3), _packed(l[2], 3)
    T3b, L3b = ccsdt_opt.blocked_t3(eris, T3p), ccsdt_opt.blocked_t3(eris, L3p)
    assert T3b.size == (350 if labels == 'one' else 78)
    fz = (3, _frozen(3, no + nv, cplx))
    ref = D.kernel(_dense(eris), t[0], t[1], T3p, l[0], l[1], L3p, frozen=fz)
    assert not ref.plan.blocks
    for plan in (None, D.Plan(nb=1, ng=1, slab_bytes=1, nk=1, blocks=True),
                 D.Plan(nb=2, ng=3, slab_bytes=(16 if cplx else 8) * 9, nk=2, blocks=True)):
        r = D.kernel(eris, t[0], t[1], T3b, l[0], l[1], L3b, frozen=fz, blocks=True, plan=plan)
        assert r.plan.blocks and r.plan.route == 'gamma' and r.d1.dtype == ref.d1.dtype
        for k in ('d1', 'g1', 'gnh', 'bf', 'kf'):
            assert abs(getattr(ref, k)).max() > 0.1 and _rel(getattr(r, k), getattr(ref, k)) < 1e-13, k
    r = D.kernel(eris, t[0], t[1], T3b, l[0], l[1], L3b, blocks=True, gradient=False)
    assert r.gnh is None and r.plan.route is None and _rel(r.d1, ref.d1) < 1e-13
    # a plan of the other layout is refused; no blocks without labels in use
    with pytest.raises(AssertionError, match='other layout'):
        D.kernel(eris, t[0], t[1], T3b, l[0], l[1], L3b, blocks=True, plan=D.Plan(nb=1, slab_bytes=1))
    with pytest.raises(ValueError, match='no blocks'):
        D.kernel(_dense(eris), t[0], t[1], T3p, l[0], l[1], L3p, blocks=True)


# ------------------------------------------------------------------ HBr

RUN = {}


def _front_end(frozen):
    mol = gto.M(atom='Br 0 0 0; H 0 0 1.4145', basis='cc-pvdz-dk', verbose=0)
    if 'mf' not in RUN:
        mf = x2c.UHF(mol)
        mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-9
        mf.kernel()
        RUN['mf'] = mf
    cc = zccsdt.GCCSDT(RUN['mf'], frozen=frozen)
    cc.verbose = 0
    cc.conv_tol, cc.conv_tol_normt = 1e-12, 1e-10
    return cc


def _args(cc):
    '''(active spinors, keywords) of relaxed_rdm1 as make_rdm1 passes them.'''
    fo, o, v, fv = cc._active()
    o, v = cc._sorted
    Cs = cc._axial()
    return Cs[:, np.concatenate((o, v))], dict(frozen_occ=Cs[:, fo] if len(fo) else None,
                                              dropped_vir=Cs[:, fv] if len(fv) else None, tol=1e-11)


@pytest.fixture(scope='module')
def hbr():
    '''HBr / cc-pVDZ-DK on the z axis, 28 frozen and 8 + 28 active spinors
    with the labels 2 m_j in use: the front end with converged amplitudes
    and Lambda in blocks, and the relaxed density of the packed block.'''
    cc = _front_end(28)
    cc.kernel()
    cc.conv_tol_normt = 1e-11
    cc.solve_lambda()
    assert cc.converged and cc.converged_lambda and cc.lambda_blocks
    eris = cc._eris
    Ca, kw = _args(cc)
    dense, t3 = cc._dense_view()
    ref = D.relaxed_rdm1(cc._scf, dense, cc.t1, cc.t2, t3, cc.l1, cc.l2, cc._packed_l3(), Ca, **kw)
    assert not ref.result.plan.blocks
    return cc, eris, Ca, kw, ref


def _same(got, ref, tol=1e-10):
    assert abs(got.relaxed - ref.relaxed).max() < tol and abs(got.unrelaxed - ref.unrelaxed).max() < tol
    assert abs(got.result.gnh - ref.result.gnh).max() < tol and abs(got.result.d1 - ref.result.d1).max() < tol
    assert abs(got.gradient - ref.gradient).max() < tol and abs(got.z - ref.z).max() < tol
    assert abs(ref.relaxed - ref.unrelaxed).max() > 1e-4 and abs(ref.gradient).max() > 1e-3


def test_hbr_in_blocks(hbr):
    '''Six labels of the virtual spinors, 31820 of the 183456 elements of
    t3 and l3: the relaxed and the unrelaxed density, the orbital gradient
    over all spinors (28 frozen ones) and the solution of the Z-vector
    equations of the packed block to 1e-10 (1e-13 found), also with the
    smallest plan and when the plan is made by a call of its own.'''
    cc, eris, Ca, kw, ref = hbr
    assert t3pack.maps_for(eris)[0].ncls == 6 and cc.t3.size == cc.l3.size == 31820
    args = (cc._scf, eris, cc.t1, cc.t2, cc.t3, cc.l1, cc.l2, cc.l3, Ca)
    r = D.relaxed_rdm1(*args, blocks=True, **kw)
    assert r.result.plan.blocks and r.result.plan.route == 'gamma' and r.result.plan.slab_bytes is None
    _same(r, ref)
    assert abs(np.trace(r.relaxed) - 36) < 1e-9 and r.nocc == 36
    plan = D.relaxed_rdm1(*args, blocks=True, plan_only=True, **kw)
    assert isinstance(plan, D.Plan) and plan.blocks and plan.held is not None and plan.nk == 28
    _same(D.relaxed_rdm1(*args, blocks=True, plan=plan, **kw), ref)
    small = D.Plan(nb=1, ng=1, slab_bytes=1, nk=3, blocks=True)
    _same(D.relaxed_rdm1(*args, blocks=True, plan=small, **kw), ref)


def test_one_label(hbr):
    '''No symmetry is one label: the blocks are the packed block.'''
    cc, eris, Ca, kw, ref = hbr
    e1 = _dense(eris)
    e1.labels, e1.use_labels = np.zeros_like(eris.labels), True
    eng = B.engine(e1, cc.t3.dtype)
    assert eng.om.ncls == 1 and eng.size == 56 * 3276
    t3 = ccsdt_opt.packed_t3(eris, cc.t3)
    r = D.relaxed_rdm1(cc._scf, e1, cc.t1, cc.t2, t3, cc.l1, cc.l2, cc._packed_l3(), Ca, blocks=True, **kw)
    assert r.result.plan.blocks
    _same(r, ref)


def test_complex_amplitudes(hbr):
    '''A phase on every active spinor: complex integrals, amplitudes and
    Lambda with the labels they had.  In blocks against the packed block
    with these integrals, and the densities in the AO basis against the
    real ones.'''
    cc, eris, Ca, kw, ref = hbr
    no = eris.nocc
    nv = len(eris.labels) - no
    u = np.exp(2j * np.pi * np.random.default_rng(5).random(no + nv))
    uo, uv = u[:no], u[no:]
    ec = _dense(eris)
    ec.use_labels = True
    for key in ('oooo', 'ooov', 'oovv', 'ovov', 'ovvo', 'ovvv'):
        x = np.asarray(getattr(eris, key)).astype(complex)
        for ax, c in enumerate(key):
            f = uo if c == 'o' else uv
            x = x * (f.conj() if ax < 2 else f).reshape([-1 if a == ax else 1 for a in range(4)])
        setattr(ec, key, x)
    ec.fock = np.asarray(eris.fock) * u.conj()[:, None] * u[None, :]
    a, b = np.triu_indices(nv, 1)
    for key in ('_vvvv_blocks', '_bare_Q', '_bare_vvvo'):     # (the caches of the real integrals)
        ec.__dict__.pop(key, None)
    ec._vvvv_pp = np.asarray(eris._vvvv_pp) * (uv[a] * uv[b]).conj()[:, None] * (uv[a] * uv[b])[None, :]
    f1 = uo[:, None] * uv.conj()[None, :]
    f2 = (uo[:, None, None, None] * uo[None, :, None, None]
          * uv.conj()[None, None, :, None] * uv.conj()[None, None, None, :])
    from zccsdt.cc import t4pack as P
    o3, v3 = P.tuples(no, 3), P.tuples(nv, 3)
    f3 = uo[o3].prod(axis=1)[:, None] * uv[v3].prod(axis=1).conj()[None, :]
    t3p = ccsdt_opt.packed_t3(eris, cc.t3) * f3
    l3p = cc._packed_l3().reshape(f3.shape) * f3.conj()
    t3c, l3c = ccsdt_opt.blocked_t3(ec, t3p), ccsdt_opt.blocked_t3(ec, l3p)
    assert abs(t3c.imag).max() > 1e-4 and abs(l3c.imag).max() > 1e-4
    amps = (cc.t1 * f1, cc.t2 * f2)
    lams = (cc.l1 * f1.conj(), cc.l2 * f2.conj())
    Cc = Ca * u[None, :]
    r = D.relaxed_rdm1(cc._scf, ec, *amps, t3c, *lams, l3c, Cc, blocks=True, **kw)
    d = D.relaxed_rdm1(cc._scf, _dense(ec), *amps, t3p, *lams, l3p, Cc, **kw)
    assert r.result.plan.blocks and not d.result.plan.blocks and np.iscomplexobj(r.result.gnh)
    assert abs(r.result.gnh.imag).max() > 1e-5
    # (the frozen spinors are rotated into Kramers pairs level by level in both runs)
    _same(r, d)

    def ao(x):
        return x.mo_coeff @ x.relaxed @ x.mo_coeff.conj().T, x.mo_coeff @ x.unrelaxed @ x.mo_coeff.conj().T
    for x, y in zip(ao(r), ao(ref)):
        assert abs(x - y).max() < 1e-9


def test_dropped_virtual_spinors():
    '''28 frozen and 4 dropped spinors, 8 + 24 active ones; amplitudes after
    three iterations and their conjugates as Lambda (nothing solved): the
    densities and the gradient over all spinors in blocks and on the packed
    block.'''
    cc = _front_end(list(range(28)) + list(range(60, 64)))
    cc.max_cycle = 3
    cc.kernel()
    eris = cc._eris
    assert getattr(eris, 'use_labels', False) and eris.nmo - eris.nocc == 24
    Ca, kw = _args(cc)
    assert kw['dropped_vir'].shape[1] == 4 and kw['frozen_occ'].shape[1] == 28
    l3 = np.array(cc.t3)
    args = (cc.t1, cc.t2, cc.t3, cc.t1, cc.t2, l3)
    r = D.relaxed_rdm1(cc._scf, eris, *args, Ca, blocks=True, **kw)
    dense, t3 = cc._dense_view()
    d = D.relaxed_rdm1(cc._scf, dense, cc.t1, cc.t2, t3, cc.t1, cc.t2, t3, Ca, **kw)
    assert r.result.plan.blocks and r.result.plan.nk == 32 and r.relaxed.shape == (64, 64)
    _same(r, d)


def test_front_end(hbr, monkeypatch):
    '''GCCSDT.make_rdm1: in blocks with ``symmetry``, on the packed block
    without it or when the plan of the blocks raises - the same density.'''
    cc, eris, Ca, kw, ref = hbr
    D1 = cc.make_rdm1()
    assert cc.rdm1_blocks and abs(D1 - ref.relaxed).max() < 1e-10
    Du = cc.make_rdm1(relaxed=False)
    assert abs(Du - ref.unrelaxed).max() < 1e-10
    cc.symmetry = False
    try:
        D2 = cc.make_rdm1()
    finally:
        cc.symmetry = True
    assert not cc.rdm1_blocks and abs(D2 - D1).max() < 1e-10

    def short(*args, **kwargs):
        raise MemoryError('nothing for the blocks')
    monkeypatch.setattr(D, 'plan_memory_blocks', short)
    D3 = cc.make_rdm1()
    assert not cc.rdm1_blocks and abs(D3 - D1).max() < 1e-10


# --------------------------------------------------------------------- memory

def _floor(eng, nf, src, top):
    '''The smallest budget that the plan takes.'''
    lo, hi = 0, top
    while hi - lo > 1:
        mid = (lo + hi) // 2
        try:
            D.plan_memory_blocks(eng, mid, nf, src)
            hi = mid
        except MemoryError:
            lo = mid
    return hi


@pytest.mark.parametrize('labels,cplx,nf', [('six', True, 0), ('six', False, 12), ('one', True, 0), ('one', True, 3),
                                            ('wide', True, 0), ('wide', False, 20)])
def test_allocations_stay_within_the_plan(labels, cplx, nf):
    '''What kernel() allocates in blocks (tracemalloc follows numpy's
    allocations, the work array of the block engine among them) is within
    the plan - for the smallest plan (one occupied pair per batch, slabs of
    a pair block of <ab||de>, the integrals of one frozen spinor at a time)
    and with no limit -, the plan is not far above it, and the numbers are
    the same.  6 + 30 spin orbitals with one and with six labels, 4 + 44
    with four; with and without frozen spinors.'''
    olab, vlab = MODELS[labels]
    no, nv = len(olab), len(vlab)
    eris, t, l = _labelled_model(no, nv, olab, vlab, seed=5, cplx=cplx)
    T3b = ccsdt_opt.blocked_t3(eris, _packed(t[2], 3))
    L3b = ccsdt_opt.blocked_t3(eris, _packed(l[2], 3))
    del t[2], l[2]
    om, vm = t3pack.maps_for(eris)
    ccsdt_opt.vvvv_packed(eris, vm)             # (the packed bare <ab||de>: the integral object's)
    del eris.vvvv
    src = _frozen(nf, no + nv, cplx) if nf else None
    eng = B.engine(eris, T3b.dtype)
    assert D.bare_bytes(eris, vm) == 0
    free = D.plan_memory_blocks(eng, None, nf, src)
    tight = D.plan_memory_blocks(eng, _floor(eng, nf, src, free.need), nf, src)
    assert free.slab_bytes is None and tight.slab_bytes is not None and tight.need < free.need
    assert tight.ng == 1 or eng.gamma_elems(tight.ng) <= eng.pass_elems(1)
    assert (tight.nk, free.nk) == ((1, nf) if nf else (None, None))
    out = []
    for plan in (tight, free):
        t3engine.release_workspace()
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        r = D.kernel(eris, t[0], t[1], T3b, l[0], l[1], L3b, plan=plan, blocks=True,
                     frozen=(nf, src) if nf else None)
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        need = plan.need - ccsdt_opt.overhead()
        print('%s, %d frozen, %s pairs per batch: allocated %.1f MB, planned %.1f MB (%.2f)'
              % (labels, nf, plan.nb, peak / 1e6, need / 1e6, peak / need))
        assert 0.45 * need < peak < need
        assert t3engine.workspace_bytes() == 0          # (given back after the passes)
        out.append(r)
    for k in ('d1', 'gnh') + (('bf', 'kf') if nf else ()):
        assert _rel(getattr(out[0], k), getattr(out[1], k)) < 1e-12


@pytest.mark.parametrize('labels,nf', [('six', 0), ('six', 12), ('one', 3), ('wide', 20)])
def test_plan_over_budgets(labels, nf):
    '''The plan in blocks over a scan of budgets from its floor to
    everything: never more than the budget, never smaller with more, the
    batches as large as fit (one more pair, triple or frozen spinor does
    not), the pair blocks of <ab||de> kept where they fit next to these
    (the batches come first, as on the packed block); MemoryError with the
    numbers below the floor.'''
    olab, vlab = MODELS[labels]
    no, nv = len(olab), len(vlab)
    eris = _labelled_model(no, nv, olab, vlab, seed=5, cplx=True)[0]
    eng = B.engine(eris, np.complex128)
    isz, n = 16, no + nv
    src = D._Given(nf, None, n) if nf else None
    free = D.plan_memory_blocks(eng, None, nf, src)
    assert free.blocks and free.slab_bytes is None and free.route == 'gamma'
    assert (free.nb, free.ng) == (eng.most_pairs(), eng.most_triples())
    floor = _floor(eng, nf, src, free.need)
    with pytest.raises(MemoryError, match=r'in the blocks of the symmetry labels.* need about .* GB|frozen'):
        D.plan_memory_blocks(eng, floor - 1, nf, src)
    fixed = D.fixed_bytes(no, nv, isz, nf) + eng.operand_bytes() + eng.gamma_bytes()
    pp = eng.gamma_bytes()

    def stage(nk):
        return nk * 2 * 16 * n ** 3 + D.frozen_own(no, nv, nk, isz, 0) if nf else 0
    last = None
    seen = set()
    for budget in np.linspace(floor, free.need + 1, 120).astype(np.int64):
        p = D.plan_memory_blocks(eng, int(budget), nf, src)
        left = budget - fixed
        work = max(eng.pass_elems(p.nb), eng.gamma_elems(p.ng)) * isz
        assert p.need <= budget and work <= left and stage(p.nk or 0) <= left
        assert p.nb == free.nb or eng.pass_elems(p.nb + 1) * isz > left
        assert p.ng == free.ng or eng.gamma_elems(p.ng + 1) * isz > left
        assert not nf or p.nk == free.nk or stage(p.nk + 1) > left
        if p.slab_bytes is None:
            assert left >= pp + max(work, stage(p.nk or 0))
        else:
            assert eng.row_bytes() <= p.slab_bytes <= eng.block_bytes() and p.slab_bytes <= left
            assert p.slab_bytes == eng.block_bytes() or p.slab_bytes == left
        if last is not None:
            assert p.need >= last.need and p.nb >= last.nb and p.ng >= last.ng and (p.nk or 0) >= (last.nk or 0)
        last = p
        seen.add(p.slab_bytes is None)
    assert seen == {True, False}
    assert (last.nb, last.ng, last.nk, last.slab_bytes) == (free.nb, free.ng, free.nk, None) and last.need == free.need
    # without the gradient there is no Gamma and no frozen stage
    d = D.plan_memory_blocks(eng, None, nf, src, gradient=False)
    assert d.route is None and d.ng is None and d.nk is None and d.need < free.need


def test_plan_counts_the_bare_blocks(hbr):
    '''kernel() takes the diagonal pair blocks of the bare <ab||de> from
    the budget where the integral object does not hold them for the maps of
    the blocks.'''
    cc, eris, Ca, kw, ref = hbr
    e = _dense(eris)
    e.use_labels = True
    vm = t3pack.maps_for(e)[1]
    ccsdt_opt.vvvv_packed(e, vm)
    assert D.bare_bytes(e, vm) == 0
    e.__dict__.pop('_vvvv_blocks')
    assert D.bare_bytes(e, vm) == sum((hi - lo) ** 2 for lo, hi in vm.blocks) * 8 > 0
