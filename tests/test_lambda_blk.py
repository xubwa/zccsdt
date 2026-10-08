'''The Lambda equations of CCSDT in the blocks of a symmetry label
(zccsdt.cc.lambda_blk, lambda_opt.kernel(..., blocks=True)) against the
equations on the packed block (lambda_opt without blocks, itself against
the tables in test_lambda_opt.py):

- every group of terms of the Lambda_3 residual and the residuals of rank 1
  and 2 for amplitudes and Lambda that solve nothing: random integrals with
  one label (no symmetry: the blocks are the whole arrays) and with four
  labels, real and complex, with whole batches and with one occupied pair
  and one triple per batch;
- converged Lambda of HBr / cc-pVDZ-DK with X2C spinors (8 + 28 active, the
  labels 2 m_j of the front end): l1, l2, l3 and the pseudo-energy; the same
  with one label and with complex amplitudes (a phase on every spinor);
- the front end: l3 in the layout of T3, the packed block for its readers,
  the equations on the packed block without ``symmetry`` or without memory
  for the blocks;
- the memory plan against what a run allocates.'''
import copy
import gc
import tracemalloc

import numpy as np
import pytest
from pyscf import gto, x2c

import fockspace as F
import zccsdt
from test_lambda_opt import _model_eris
from zccsdt.cc import ccsdt_opt, lambda_blk, lambda_opt as LO, solver, t3kern, t3pack

pytestmark = pytest.mark.skipif(not (t3kern.USE_C and t3pack.USE_C), reason='C libraries not built')


def _dense(eris):
    '''The same integrals without their labels in use.'''
    out = copy.copy(eris)
    out.use_labels = False
    out.__dict__.pop('_buffers', None)
    return out


def _labelled_model(no, nv, olab, vlab, seed, cplx):
    '''Random Hermitian integrals that conserve the labels (occupied and
    virtual ones in the order of their labels), and amplitudes and Lambda
    with the labels that solve nothing.'''
    eris = _model_eris(no, nv, seed=seed, cplx=cplx)
    lab = np.array(list(olab) + list(vlab))
    assert len(olab) == no and len(vlab) == nv
    eris.fock = eris.fock * (lab[:, None] == lab[None, :])
    for key in ('oooo', 'ooov', 'oovv', 'ovov', 'ovvo', 'ovvv', 'vvvv'):
        k = 0
        for ax, c in enumerate(key):
            l = lab[:no] if c == 'o' else lab[no:]
            k = k + (1 if ax < 2 else -1) * l.reshape([-1 if a == ax else 1 for a in range(4)])
        setattr(eris, key, np.ascontiguousarray(np.asarray(getattr(eris, key)) * (k == 0)))
    eris.labels, eris.modulus, eris.use_labels = lab, 0, True
    rng = np.random.default_rng(seed + 1)
    amps = []
    for _ in range(2):
        xs = []
        for k in (1, 2, 3):
            x = F.random_antisym(rng, no, nv, k)
            if cplx:
                x = x + 1j * F.random_antisym(rng, no, nv, k)
            d = 0
            for ax in range(2 * k):
                l = lab[:no] if ax < k else -lab[no:]
                d = d + l.reshape([-1 if a == ax else 1 for a in range(2 * k)])
            xs.append(0.3 * x * (d == 0))
        amps.append(xs)
    return eris, amps[0], amps[1]


def _packed(x, n):
    '''The unique block [i<j<k, a<b<c] in lexicographic order of a full
    antisymmetric tensor of rank 3.'''
    from zccsdt.cc import ccsdtq_terms as Q
    return Q.unique(x, n)


def _rel(a, b):
    return abs(a - b).max() / abs(b).max()


LABELS = {'one': ([0] * 5, [0] * 7),
          'four': ([-1, -1, 1, 1, 3], [-3, -1, -1, 1, 1, 3, 3])}


@pytest.mark.parametrize('labels,cplx,mem_mb', [
    ('one', True, 2000.0), ('one', False, 1e-3), ('four', True, 2000.0), ('four', False, 2000.0),
    ('four', True, 1e-3)])     # (1e-3: one occupied pair and one triple per batch)
def test_residuals_against_the_packed_block(labels, cplx, mem_mb):
    '''Each group of terms of the Lambda_3 residual and the three residuals
    in blocks, equal to those on the packed block to 1e-12 of their size.'''
    no, nv = 5, 7
    eris, t, l = _labelled_model(no, nv, *LABELS[labels], seed=11, cplx=cplx)
    om, vm = t3pack.maps_for(eris)
    assert om.spins is not None and om.ncls == (1 if labels == 'one' else 4)
    T3p, L3p = _packed(t[2], 3), _packed(l[2], 3)
    T3b, L3b = ccsdt_opt.blocked_t3(eris, T3p), ccsdt_opt.blocked_t3(eris, L3p)
    assert T3b.size == np.count_nonzero(T3p) == (350 if labels == 'one' else 78)
    assert np.array_equal(ccsdt_opt.packed_t3(eris, T3b), T3p)
    ref = LO.make_intermediates(_dense(eris), t[0], t[1], T3p)
    im = LO.make_intermediates(eris, t[0], t[1], T3b, mem_mb=mem_mb, blocks=True)
    assert isinstance(im.eng, lambda_blk.BlockL3) and isinstance(ref.eng, LO.DenseL3)
    assert (im.eng.nb, im.eng.t3.nbt) == (1, 1) if mem_mb < 1 else im.eng.nb >= no * (no - 1) // 2
    unpack = ccsdt_opt._unpack_pairs
    for name in LO.GROUPS:
        want = LO.residuals(ref, l[0], l[1], L3p, T3p, terms=[name])[2]
        got = LO.residuals(im, l[0], l[1], L3b, T3b, terms=[name])[2]
        assert abs(want).max() > 0.1 and _rel(ccsdt_opt.packed_t3(eris, got), want) < 1e-12, name
    w1, w2, w3 = LO.residuals(ref, l[0], l[1], L3p, T3p)
    r1, r2, r3 = LO.residuals(im, l[0], l[1], L3b, T3b)
    assert _rel(r1, w1) < 1e-12 and _rel(unpack(r2, om, vm), unpack(w2, ref.om, ref.vm)) < 1e-12
    assert _rel(ccsdt_opt.packed_t3(eris, r3), w3) < 1e-12
    assert all(np.iscomplexobj(x) == cplx for x in (r1, r2, r3))
    # the parts of the residuals of rank 1 and 2 by the rank of Lambda in them
    for sel in range(3):
        ll = [x if k == sel else 0 * x for k, x in enumerate((l[0], l[1], L3p))]
        w1, w2, _ = LO.residuals(ref, ll[0], ll[1], ll[2], T3p)
        r1, r2, _ = LO.residuals(im, ll[0], ll[1], ccsdt_opt.blocked_t3(eris, ll[2]), T3b)
        assert _rel(r1, w1) < 1e-12 and _rel(unpack(r2, om, vm), unpack(w2, ref.om, ref.vm)) < 1e-12, sel
    # a Jacobi step, into a given array
    out = np.empty(L3b.size, dtype=r3.dtype)
    new = LO.update_lams(im, l[0], l[1], L3b, T3b, out=out)
    want = LO.update_lams(ref, l[0], l[1], L3p, T3p)
    assert np.shares_memory(new[2], out)
    assert _rel(new[0], want[0]) < 1e-12 and _rel(new[1], want[1]) < 1e-12
    assert _rel(ccsdt_opt.packed_t3(eris, new[2]), want[2].reshape(T3p.shape)) < 1e-12


def pseudo_energy(eris, l1, l2):
    '''sum_ia f_ai l1_ia + 1/4 sum <ab||ij> l2_ijab.'''
    no = eris.nocc
    return (np.einsum('ia,ia', np.asarray(eris.fock)[:no, no:].conj(), l1)
            + 0.25 * np.einsum('ijab,ijab', np.asarray(eris.oovv).conj(), l2))


RUN = {}


@pytest.fixture(scope='module')
def hbr():
    '''HBr / cc-pVDZ-DK on the z axis, 8 + 28 active spinors with the labels
    2 m_j in use: (eris, t1, t2, T3 in blocks) of the front end and Lambda
    on the packed block.'''
    mol = gto.M(atom='Br 0 0 0; H 0 0 1.4145', basis='cc-pvdz-dk', verbose=0)
    mf = x2c.UHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-8
    mf.kernel()
    cc = zccsdt.GCCSDT(mf, frozen=28)
    cc.verbose = 0
    cc.conv_tol, cc.conv_tol_normt = 1e-12, 1e-10
    cc.kernel()
    assert cc.converged
    eris = cc.ao2mo()
    RUN['cc'] = cc
    ref = LO.kernel(_dense(eris), cc.t1, cc.t2, ccsdt_opt.packed_t3(eris, cc.t3),
                    solver.Options(conv_tol_normt=1e-11))
    assert ref.converged and not ref.blocks
    return eris, cc.t1, cc.t2, cc.t3, ref


def _same(eris, got, ref, tol=1e-10):
    assert abs(got.l1 - ref.l1).max() < tol and abs(got.l2 - ref.l2).max() < tol
    assert abs(LO.packed_l3(eris, got.l3) - ref.l3.reshape(56, -1)).max() < tol
    e = pseudo_energy(eris, ref.l1, ref.l2)
    assert abs(e) > 0.05 and abs(pseudo_energy(eris, got.l1, got.l2) - e) < tol


def test_hbr_in_blocks(hbr):
    '''Six labels of the virtual spinors: 31820 of the 183456 elements of
    l3, the solution of the packed block in them.'''
    eris, t1, t2, t3, ref = hbr
    assert t3pack.maps_for(eris)[0].ncls == 6 and t3.size == 31820
    opts = solver.Options(conv_tol_normt=1e-11)
    r = LO.kernel(eris, t1, t2, t3, opts, blocks=True)
    assert r.converged and r.blocks and r.plan.blocks and r.l3.size == t3.size
    assert r.ncycle == ref.ncycle
    _same(eris, r, ref)
    assert abs(r.l2 - t2).max() > 1e-4 and abs(r.l3 - t3).max() > 1e-5
    # the blocks hold what there is: nothing of the solution on the packed block is outside
    out = ref.l3.reshape(56, -1) - LO.packed_l3(eris, LO.blocked_l3(eris, ref.l3))
    assert abs(out).max() < 1e-10
    # started from the solution: converged in one step, DIIS vectors in memory or not
    for incore in (False, True):
        r1 = LO.kernel(eris, t1, t2, t3, solver.Options(conv_tol_normt=1e-9, incore_complete=incore),
                       r.l1, r.l2, r.l3, blocks=True)
        assert r1.converged and r1.ncycle == 1
    # small buffers: one occupied pair and one triple per batch
    r = LO.kernel(eris, t1, t2, t3, solver.Options(conv_tol_normt=1e-11, mem_mb=1e-3), blocks=True)
    assert r.converged
    _same(eris, r, ref)


def test_one_label(hbr):
    '''No symmetry is one label: the blocks are the packed block.'''
    eris, t1, t2, t3, ref = hbr
    e1 = _dense(eris)
    e1.labels, e1.use_labels = np.zeros_like(eris.labels), True
    assert t3pack.maps_for(e1)[0].ncls == 1
    T3p = ccsdt_opt.packed_t3(eris, t3)
    r = LO.kernel(e1, t1, t2, T3p, solver.Options(conv_tol_normt=1e-11), blocks=True)
    assert r.converged and r.blocks and r.l3.size == T3p.size
    assert np.array_equal(LO.packed_l3(e1, r.l3).ravel(), r.l3)
    _same(e1, r, ref)


def test_complex_amplitudes(hbr):
    '''A phase on every spinor: complex integrals and amplitudes with the
    labels they had; Lambda takes the conjugate phases of t.'''
    eris, t1, t2, t3, ref = hbr
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
    # (the Fock matrix too: 4e-10 between spinors of one label, 1e-9 in l1)
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
    t3c = ccsdt_opt.blocked_t3(ec, ccsdt_opt.packed_t3(eris, t3) * f3)
    r = LO.kernel(ec, t1 * f1, t2 * f2, t3c, solver.Options(conv_tol_normt=1e-11), blocks=True)
    assert r.converged and r.blocks and np.iscomplexobj(r.l3) and abs(r.l3.imag).max() > 1e-4
    assert abs(r.l1 - ref.l1 * f1.conj()).max() < 1e-10 and abs(r.l2 - ref.l2 * f2.conj()).max() < 1e-10
    assert abs(LO.packed_l3(ec, r.l3) - ref.l3.reshape(f3.shape) * f3.conj()).max() < 1e-10
    e = pseudo_energy(eris, ref.l1, ref.l2)
    assert abs(pseudo_energy(ec, r.l1, r.l2) - e) < 1e-10
    # and the packed block with these integrals
    rd = LO.kernel(_dense(ec), t1 * f1, t2 * f2, ccsdt_opt.packed_t3(ec, t3c),
                   solver.Options(conv_tol_normt=1e-11))
    _same(ec, r, rd)


def test_front_end(hbr, monkeypatch):
    '''GCCSDT.solve_lambda: in blocks with ``symmetry``, l3 in the layout of
    T3; on the packed block without it or without memory for the blocks -
    the same Lambda, in the same layout.'''
    eris, t1, t2, t3, ref = hbr
    cc = RUN['cc']
    cc.conv_tol_normt = 1e-11
    l1, l2, l3 = cc.solve_lambda()
    assert cc.converged_lambda and cc.lambda_blocks and l3.size == cc.t3.size == 31820
    packed = cc._packed_l3()
    assert packed.size == 56 * 3276 and cc._packed_l3() is packed          # (put there once)
    assert abs(l1 - ref.l1).max() < 1e-10 and abs(l2 - ref.l2).max() < 1e-10
    assert abs(packed - ref.l3).max() < 1e-10
    cc.symmetry = False
    try:
        d1, d2, d3 = cc.solve_lambda()
    finally:
        cc.symmetry = True
    assert not cc.lambda_blocks and d3.size == 31820 and abs(d3 - l3).max() < 1e-10
    assert abs(cc._packed_l3() - ref.l3).max() < 1e-10

    def short(*args, **kwargs):
        raise MemoryError('nothing for the blocks')
    monkeypatch.setattr(LO, 'plan_memory_blocks', short)
    d1, d2, d3 = cc.solve_lambda()
    assert not cc.lambda_blocks and d3.size == 31820 and abs(d3 - l3).max() < 1e-10
    assert abs(d1 - l1).max() < 1e-10 and abs(d2 - l2).max() < 1e-10


MODELS = {'one': ([0] * 6, [0] * 30),
          'six': ([-3, -1, -1, 1, 1, 3], [-5] * 2 + [-3] * 5 + [-1] * 8 + [1] * 8 + [3] * 5 + [5] * 2)}


@pytest.mark.parametrize('labels,cplx', [('one', True), ('six', True), ('six', False)])
def test_memory_plan(labels, cplx):
    '''What the intermediates and an iteration in blocks allocate
    (tracemalloc follows numpy's allocations) stays within the plan, stage
    by stage, with whole batches and with the smallest buffers, and the plan
    is not far above it; the work array is the planned one; the buffers
    grow with the budget and take what fits; MemoryError below the floor.'''
    from zccsdt.cc import t3engine
    olab, vlab = MODELS[labels]
    no, nv = len(olab), len(vlab)
    eris, t, l = _labelled_model(no, nv, olab, vlab, seed=5, cplx=cplx)
    isz = 16 if cplx else 8
    T3b = ccsdt_opt.blocked_t3(eris, _packed(t[2], 3))
    L3b = ccsdt_opt.blocked_t3(eris, _packed(l[2], 3))
    del t[2], l[2]
    om, vm = t3pack.maps_for(eris)
    ccsdt_opt.vvvv_packed(eris, vm)             # (the packed bare <ab||de>: the integral object's)
    ccsdt_opt._bare_Q(eris, om, vm)
    del eris.vvvv
    f = no * nv ** 3 * isz
    pp = sum((hi - lo) ** 2 for lo, hi in vm.blocks) * isz
    assert T3b.size == (81200 if labels == 'one' else 12416)
    for mem_mb in (2000.0, 1e-3):
        eng = LO.block_engine(eris, T3b.dtype, mem_mb)
        plan = LO.plan_memory_blocks(eng, None, mem_mb)
        assert plan.blocks and plan.buffers == eng.buffer_bytes(mem_mb)
        t3engine.release_workspace()
        del eng
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        im = LO.make_intermediates(eris, t[0], t[1], T3b, mem_mb=plan.mem_mb, blocks=True)
        setup = tracemalloc.get_traced_memory()[1] - base
        kept = tracemalloc.get_traced_memory()[0] - base
        tracemalloc.reset_peak()
        out = np.empty(L3b.size, dtype=T3b.dtype)       # (the residual: one T3 of the plan)
        LO.update_lams(im, l[0], l[1], L3b, T3b, out=out)
        LO.update_lams(im, l[0], l[1], L3b, T3b, out=out)
        peak = tracemalloc.get_traced_memory()[1] - base - out.nbytes
        tracemalloc.stop()
        fixed = pp + plan.buffers
        where = (mem_mb, setup / f, kept / f, peak / f, (plan.setup + fixed) / f, (plan.kept + fixed) / f,
                 (plan.iteration + fixed) / f)
        assert t3engine.workspace_bytes() == plan.buffers, where
        assert setup <= plan.setup + fixed, where
        assert kept <= plan.kept + fixed, where
        assert peak <= plan.iteration + fixed, where
        assert peak > 0.85 * (plan.iteration + fixed), where
        assert plan.need >= max(plan.setup, plan.iteration) + fixed + 3 * L3b.nbytes
        del im, out
    eng = LO.block_engine(eris, T3b.dtype)
    floor = LO.plan_memory_blocks(eng, None, 1e-3)
    full = LO.plan_memory_blocks(eng, None, 2000.0)
    assert floor.need == plan.need < full.need and full.diis == 0
    assert LO.plan_memory_blocks(eng, None, 2000.0, diis_space=6).diis == 12 * L3b.nbytes
    # budgets from the floor to everything: within the budget, growing with it, and no
    # buffers that fit are left out
    grid = [LO.plan_memory_blocks(eng, None, m).need for m in np.geomspace(1e-3, 2000.0, 400)]
    assert all(a <= b for a, b in zip(grid, grid[1:])) and len(set(grid)) >= 5
    last = None
    for budget in np.linspace(floor.need, full.need, 25).astype(np.int64):
        p = LO.plan_memory_blocks(eng, int(budget), 2000.0)
        assert p.need <= budget and p.need >= max(n for n in grid if n <= budget), budget
        assert last is None or (p.need >= last.need and p.mem_mb >= last.mem_mb)
        last = p
    assert last.need == full.need
    with pytest.raises(MemoryError, match=r'in the blocks of the symmetry labels.* need about .* GB'):
        LO.plan_memory_blocks(eng, floor.need - 1, 2000.0)
