'''CCSDTQ in blocks of a symmetry label (zccsdt.cc.t4engine_sym,
zccsdt.cc.ccsdtq_sym) against the dense code: the blocks and the packed
array, the T4 equation piece by piece on random amplitudes and integrals
that respect the labels, the iterations for a linear molecule with X2C
spinors (labels 2 m_j from the front end) - also with one label and with
complex amplitudes -, and the memory plan.'''
import copy
import gc
import tracemalloc
from math import comb

import numpy as np
import pytest
from pyscf import gto, x2c

import fockspace as F
import zccsdt
from zccsdt.cc import ccsdt_opt, ccsdtq, ccsdtq_sym as QS, t4engine as T, t4engine_sym as S, t4pack as P

pytestmark = pytest.mark.skipif(S._lib is None, reason='libzc_t4sym is not built')


def _label_sum(ko, kv, k):
    '''Labels of the occupied minus those of the virtual indices of an
    amplitude of rank k, on the full array.'''
    s = 0
    for ax in range(2 * k):
        lab = ko if ax < k else kv
        s = s + (1 if ax < k else -1) * lab.reshape([-1 if a == ax else 1 for a in range(2 * k)])
    return s


def _random(ko, kv, cplx, seed=3):
    '''Random amplitudes, f and <pq||rs> that vanish where the labels do not
    add up.'''
    no, nv = len(ko), len(kv)
    n = no + nv
    lab = np.concatenate((ko, kv))
    rng = np.random.default_rng(seed)
    vr, vi = F.random_integrals(rng, n), F.random_integrals(rng, n)
    v = vr + 1j * (vi - vi.transpose(2, 3, 0, 1))
    f = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    v = v * (lab[:, None, None, None] + lab[None, :, None, None] == lab[None, None, :, None] + lab[None, None, None, :])
    arr = {'v': v, 'f': (f + f.conj().T) * (lab[:, None] == lab[None, :])}
    for k in (2, 3, 4):
        t = F.random_antisym(rng, no, nv, k) + 1j * F.random_antisym(rng, no, nv, k)
        arr['t%d' % k] = t * (_label_sum(ko, kv, k) == 0)
    if not cplx:
        arr = {k: np.ascontiguousarray(x.real) for k, x in arr.items()}
    return arr


CASES = [([0, 0, 1, 1, 2], [0, 0, 1, 1, 1, 2, 3], True, None),
         ([0, 1, 1, 2, 2, 3], [0, 1, 1, 2, 2, 3, 3], False, 20000),
         ([1, 1, 2, 2, 2], [0, 0, 0, 1, 2, 2, 4, 4], True, 3000),
         ([0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], True, None)]


@pytest.mark.parametrize('ko,kv,cplx,batch', CASES)
def test_blocks_and_packed_array(ko, kv, cplx, batch):
    '''The blocks hold what the labels allow and nothing else; to_packed()
    and from_packed() are inverse to each other on it.'''
    ko, kv = np.array(ko), np.array(kv)
    no, nv = len(ko), len(kv)
    lay = S.Layout(ko, kv, int(max(ko.max(), kv.max())) + 1)
    allowed = _label_sum(ko, kv, 4) == 0
    assert lay.size == P.pack(allowed).sum()
    assert int(lay.t3off[-1]) == P.pack(_label_sum(ko, kv, 3) == 0).sum()
    arr = _random(ko, kv, cplx)
    T4 = P.pack(arr['t4'])
    blocks = lay.from_packed(T4)
    assert blocks.shape == (lay.size,) and abs(lay.to_packed(blocks) - T4).max() == 0
    rng = np.random.default_rng(1)
    x = rng.standard_normal(lay.size)
    assert abs(lay.from_packed(lay.to_packed(x)) - x).max() == 0
    full = rng.standard_normal(T4.shape)
    assert abs(lay.to_packed(lay.from_packed(full)) - full * P.pack(allowed)).max() == 0
    T3 = P.pack(arr['t3'])
    assert abs(lay.t3_to_packed(lay.t3_from_packed(T3)) - T3).max() == 0
    # the lists of the virtual quadruples: lexicographic, one total label each
    for M in lay.sectors:
        q = lay.quads[M].astype(int)
        assert (kv[q].sum(axis=1) == M).all() and (np.diff(lay.lex(M)) > 0).all()
        assert (P.tuples(nv, 4)[lay.lex(M)] == q).all()
    assert sum(lay.nI[M] * comb(nv, 4) for M in lay.sectors) <= comb(no, 4) * comb(nv, 4)


@pytest.mark.parametrize('ko,kv,cplx,batch', CASES)
def test_pieces_against_dense(ko, kv, cplx, batch):
    '''Every piece of the T4 equation, and all of it, equal to the dense
    engine on the packed arrays: R4, the T4 terms of R3 and of R2 - in one
    batch and in many.'''
    ko, kv = np.array(ko), np.array(kv)
    no = len(ko)
    arr = _random(ko, kv, cplx)
    ints = T.Dressed.from_dense(arr['f'], arr['v'], no)
    T3, T4 = P.pack(arr['t3']), P.pack(arr['t4'])
    lay = S.Layout(ko, kv, int(max(ko.max(), kv.max())) + 1)
    T3s, T4s = lay.t3_from_packed(T3), lay.from_packed(T4)
    seen = [0, 0, 0]
    for only in [None] + [(p,) for p in T.PIECES]:
        ref = T.residual(arr['t2'], T3, T4, ints, only=only)
        R4, R3, R2 = S.residual(lay, arr['t2'], T3s, T4s, ints, batch_bytes=batch, only=only)
        assert R4.dtype == (np.complex128 if cplx else np.float64)
        for k, (r, g) in enumerate(zip(ref, (lay.to_packed(R4), lay.t3_to_packed(R3), R2))):
            assert abs(r - g).max() <= 1e-12 * max(abs(r).max(), 1.0), (only, k)
            seen[k] += abs(r).max() > 1.0
    assert seen[0] == 8 and seen[1] >= 3 and seen[2] == 2       # (every piece is there)


def test_residual_into_a_given_array():
    ko, kv = np.array(CASES[0][0]), np.array(CASES[0][1])
    arr = _random(ko, kv, False)
    ints = T.Dressed.from_dense(arr['f'], arr['v'], len(ko))
    lay = S.Layout(ko, kv, 4)
    T3s, T4s = lay.t3_from_packed(P.pack(arr['t3'])), lay.from_packed(P.pack(arr['t4']))
    ref = S.residual(lay, arr['t2'], T3s, T4s, ints)[0]
    out = np.full(lay.size, 7.0)
    assert S.residual(lay, arr['t2'], T3s, T4s, ints, out=out)[0] is out and abs(out - ref).max() == 0


# ------------------------------------------------------------------ a linear molecule

@pytest.fixture(scope='module')
def hbr():
    '''HBr / cc-pVDZ-DK, X2C: 8 occupied spinors and the lowest 16 virtual
    ones, CCSDT amplitudes in the blocks of 2 m_j.'''
    mol = gto.M(atom='Br 0 0 0; H 0 0 1.4145', basis='cc-pvdz-dk', verbose=0)
    mf = x2c.UHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-8
    mf.kernel()
    nmo = len(mf.mo_occ)
    cc = zccsdt.GCCSDT(mf, frozen=list(range(28)) + list(range(nmo - 12, nmo)))
    cc.verbose = 0
    cc.conv_tol, cc.conv_tol_normt = 1e-11, 1e-9
    cc.kernel()
    eris = cc.ao2mo()
    assert eris.modulus == 0 and eris.use_labels and len(np.unique(eris.labels)) == 4
    opts = ccsdtq.Options(conv_tol=1e-11, conv_tol_normt=1e-9)
    res = QS.kernel(eris, opts, cc.t1, cc.t2, cc.t3)
    return eris, cc, res


def _dense(eris):
    '''The same integrals for the dense code: the labels unused.'''
    ed = copy.copy(eris)
    ed.use_labels = False
    ed.__dict__.pop('_buffers', None)
    return ed


def test_iterations_against_dense(hbr):
    '''The CCSDTQ energy and amplitudes of the blocks and of the dense
    solver, from the same CCSDT amplitudes; one label - no symmetry - is the
    case of one block.'''
    eris, cc, res = hbr
    no, nv = eris.nocc, eris.nmo - eris.nocc
    assert QS.available(eris) and res.converged
    opts = ccsdtq.Options(conv_tol=1e-11, conv_tol_normt=1e-9)
    t3p = ccsdt_opt.packed_t3(eris, cc.t3)
    ref = ccsdtq.kernel(_dense(eris), opts, cc.t1, cc.t2, t3p)
    assert ref.converged and abs(ref.e_corr - cc.e_corr) > 1e-4
    assert abs(res.e_corr - ref.e_corr) < 1e-10 and res.ncycle == ref.ncycle
    lay = res.plan.layout
    assert res.t4.shape == (lay.size,) and lay.size < 0.25 * ref.t4.size
    t4p = QS.packed_t4(eris, res.t4)
    assert t4p.shape == ref.t4.shape == (comb(no, 4), comb(nv, 4))
    assert abs(t4p - ref.t4).max() < 1e-9 * abs(ref.t4).max() + 1e-12
    assert abs(QS.t4_blocks(eris, t4p) - res.t4).max() == 0
    assert abs(QS.packed_t3(eris, res.t3) - ref.t3.reshape(comb(no, 3), -1)).max() < 1e-10
    assert abs(res.t2 - ref.t2).max() < 1e-10
    # the amplitudes of the dense code outside the blocks are zero to rounding
    assert abs(ref.t4 - lay.to_packed(lay.from_packed(ref.t4))).max() < 1e-12
    # from packed amplitudes, and from none of T3 and T4
    again = QS.kernel(eris, ccsdtq.Options(max_cycle=2), ref.t1, ref.t2, ref.t3, ref.t4)
    assert abs(again.e_corr - ref.e_corr) < 1e-10
    # one label
    e1 = copy.copy(eris)
    e1.labels = np.zeros_like(eris.labels)
    e1.__dict__.pop('_buffers', None)
    one = QS.kernel(e1, opts, cc.t1, cc.t2, t3p)
    assert one.plan.layout.sectors == [0] and one.t4.size == ref.t4.size
    assert abs(one.e_corr - ref.e_corr) < 1e-10
    assert abs(one.t4.reshape(ref.t4.shape) - ref.t4).max() < 1e-9 * abs(ref.t4).max() + 1e-12


def test_complex_amplitudes(hbr):
    '''A phase on every spinor: complex integrals and amplitudes with the
    labels they had; two iterations give the energies of the real ones and
    their amplitudes with the phases.'''
    eris, cc, res = hbr
    no = eris.nocc
    nv = len(eris.labels) - no
    u = np.exp(2j * np.pi * np.random.default_rng(5).random(no + nv))
    uo, uv = u[:no], u[no:]
    ec = copy.copy(eris)
    ec.__dict__.pop('_buffers', None)
    ec.fock = np.asarray(eris.fock) * u.conj()[:, None] * u[None, :]
    for key in ('oooo', 'ooov', 'oovv', 'ovov', 'ovvo', 'ovvv'):
        x = np.asarray(getattr(eris, key)).astype(complex)
        for ax, c in enumerate(key):
            f = uo if c == 'o' else uv
            x = x * (f.conj() if ax < 2 else f).reshape([-1 if a == ax else 1 for a in range(4)])
        setattr(ec, key, x)
    a, b = np.triu_indices(nv, 1)
    ec.__dict__.pop('_vvvv_blocks', None)       # (the cache of the real block)
    ec._vvvv_pp = np.asarray(eris._vvvv_pp) * (uv[a] * uv[b]).conj()[:, None] * (uv[a] * uv[b])[None, :]

    def phase(k):
        o, v = P.tuples(no, k), P.tuples(nv, k)
        return uo[o].prod(axis=1)[:, None] * uv[v].prod(axis=1).conj()[None, :]
    t1c = cc.t1 * uo[:, None] * uv.conj()[None, :]
    t2c = cc.t2 * (uo[:, None, None, None] * uo[None, :, None, None]
                   * uv.conj()[None, None, :, None] * uv.conj()[None, None, None, :])
    t3p = ccsdt_opt.packed_t3(eris, cc.t3)
    opts = ccsdtq.Options(max_cycle=3)
    real = QS.kernel(eris, opts, cc.t1, cc.t2, cc.t3)
    cplx = QS.kernel(ec, opts, t1c, t2c, t3p * phase(3))
    assert cplx.t4.dtype == np.complex128 and real.t4.dtype == np.float64
    assert abs(cplx.e_corr - real.e_corr) < 1e-11 and abs(real.e_corr - cc.e_corr) > 1e-5
    t4r, t4c = QS.packed_t4(eris, real.t4), QS.packed_t4(ec, cplx.t4)
    assert abs(t4c - t4r * phase(4)).max() < 1e-11 and abs(t4c.imag).max() > 1e-6
    dense = ccsdtq.kernel(_dense(ec), opts, t1c, t2c, t3p * phase(3))
    assert abs(dense.e_corr - cplx.e_corr) < 1e-11 and abs(dense.t4 - t4c).max() < 1e-11


def test_memory_plan(hbr):
    '''The plan: MemoryError with numbers below the floor, nothing shrinks
    when the budget grows, and the peak of the allocations of the engine and
    of an iteration stays within what is planned.'''
    eris, cc, res = hbr
    lay = S.layout_for(eris)
    isz = 8
    free = QS.plan(lay, isz)
    assert free.batch_bytes == S.BATCH_BYTES and not free.diis_file and free.t4 == isz * lay.size
    with pytest.raises(MemoryError, match='MB besides the integrals'):
        QS.plan(lay, isz, free.floor - 1)
    last = None
    for budget in np.linspace(free.floor, 1.2 * free.need, 40):
        pl = QS.plan(lay, isz, int(budget))
        assert pl.need <= budget and pl.floor == free.floor
        if last is not None:
            assert pl.batch_bytes >= last.batch_bytes and pl.mem_mb >= last.mem_mb
            assert last.diis_file or not pl.diis_file
        last = pl
    # (at the floor: the smallest batches, or larger ones that take no more)
    lo = S.min_batch_bytes(lay, isz)
    assert S.memory(lay, isz, QS.plan(lay, isz, free.floor).batch_bytes) == S.memory(lay, isz, lo)
    assert last.batch_bytes == S.BATCH_BYTES and not last.diis_file
    ints = T.Dressed.from_eris(eris, res.t1)
    ref = S.residual(lay, res.t2, res.t3, res.t4, ints)
    for batch in (None, 1 << 18, lo):
        gc.collect()
        tracemalloc.start()
        out = S.residual(lay, res.t2, res.t3, res.t4, ints, batch)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        assert peak < S.memory(lay, isz, batch), (batch, peak)
        assert all(abs(x - y).max() < 1e-13 for x, y in zip(out, ref))
    del ints, ref, out
    # an iteration under the smallest plan: batches of one row, the DIIS vectors of T4 on file
    least = QS.plan(lay, isz, free.floor)
    least.batch_bytes = lo
    assert least.diis_file
    small = QS.kernel(eris, ccsdtq.Options(max_cycle=2, plan=least), cc.t1, cc.t2, cc.t3)
    big = QS.kernel(eris, ccsdtq.Options(max_cycle=2, plan=free), cc.t1, cc.t2, cc.t3)
    assert abs(small.e_corr - big.e_corr) < 1e-13 and abs(small.t4 - big.t4).max() < 1e-13
    # the whole iteration (CCSDT step, dressing, T4 equation, DIIS in memory) within its plan
    pl = QS.plan_for(eris, np.float64, incore_complete=True)
    ccsdt_opt.release_cached(eris)
    del small, big
    gc.collect()
    tracemalloc.start()
    QS.kernel(eris, ccsdtq.Options(max_cycle=3, plan=pl), cc.t1, cc.t2, cc.t3)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < pl.need, (peak, pl.need)


def test_no_blocks_without_labels(hbr):
    '''Labels with a modulus, or none, or unused: the dense code is to be
    taken (available() False, kernel() refuses).'''
    eris, cc, res = hbr
    for change in (dict(modulus=4), dict(labels=None), dict(use_labels=False)):
        e = copy.copy(eris)
        e.__dict__.update(change)
        assert not QS.available(e)
        with pytest.raises(ValueError, match='no blocks'):
            QS.kernel(e, None, cc.t1, cc.t2, cc.t3)
