'''The first derivatives of (Q) in symmetry sectors
(zccsdt.cc.quadruples_grad_sym) against the pass on the packed block
(zccsdt.cc.quadruples_grad): a linear molecule with X2C spinors (labels
2 m_j), the same labels modulo 4, one label, complex amplitudes; what the
orbital gradient makes of the density of <bc||ef> in the blocks of the pair
labels; the memory plan against what is allocated; the front end.

The amplitudes and integrals are projected on their label-conserving parts
first: the dense pass keeps the elements that break the symmetry (1e-10 of
the others after the iterations), the sectors do not have them, and the
derivatives are linear in them where the energy is quadratic.'''
import copy
import gc
import tracemalloc

import numpy as np
import pytest
from pyscf import gto, x2c

import zccsdt
from zccsdt.cc import ccsdt_opt, quadruples as QD, quadruples_grad as QG, quadruples_grad_sym as GS, \
    quadruples_sym as QS, t4pack as P

FIELDS = ('s2', 's3', 'de_occ', 'gamma_vv', 'vvoo', 'vvvo', 'ovoo', 'oooo', 'ovvo')


def project(eris, t2, t3, modulus=0):
    '''The label-conserving parts of the integrals and of the amplitudes.'''
    no = eris.nocc
    lab = np.asarray(eris.labels)
    lo, lv = lab[:no], lab[no:]
    ec = copy.copy(eris)
    ec.__dict__.pop('_vvvv_blocks', None)

    def bad(k):
        return (k % modulus if modulus else k) != 0
    for key in ('oooo', 'ooov', 'oovv', 'ovov', 'ovvo', 'ovvv'):
        x = np.array(getattr(eris, key))
        k = 0
        for ax, c in enumerate(key):
            l = lo if c == 'o' else lv
            k = k + (1 if ax < 2 else -1) * l.reshape([-1 if a == ax else 1 for a in range(4)])
        x[bad(k)] = 0
        setattr(ec, key, x)
    a, b = np.triu_indices(len(lv), 1)
    kp = lv[a] + lv[b]
    vpp = np.array(eris._vvvv_pp)
    vpp[bad(kp[:, None] - kp[None, :])] = 0
    ec._vvvv_pp = vpp
    t2 = np.array(t2)
    t2[bad(lo[:, None, None, None] + lo[None, :, None, None] - lv[None, None, :, None] - lv[None, None, None, :])] = 0
    o3, v3 = P.tuples(no, 3), P.tuples(len(lv), 3)
    t3 = np.array(t3).reshape(len(o3), len(v3))
    t3[bad(lo[o3].sum(axis=1)[:, None] - lv[v3].sum(axis=1)[None, :])] = 0
    return ec, t2, t3


@pytest.fixture(scope='module')
def hbr():
    mol = gto.M(atom='Br 0 0 0; H 0 0 1.4145', basis='cc-pvdz-dk', verbose=0)
    mf = x2c.UHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-8
    mf.kernel()
    cc = zccsdt.GCCSDT(mf, frozen=28)
    cc.verbose = 0
    cc.conv_tol, cc.conv_tol_normt = 1e-11, 1e-9
    cc.kernel()
    eris = cc.ao2mo()
    raw = (eris, cc.t2, ccsdt_opt.packed_t3(eris, cc.t3))
    return project(*raw) + (raw,)


def _rel(a, b):
    return abs(a - b).max() / abs(b).max()


def _same(g, ref, tol=1e-12):
    assert abs(g.e_q - ref.e_q) < tol * abs(ref.e_q)
    for k in FIELDS:
        assert _rel(getattr(g, k), getattr(ref, k)) < tol, k
    assert _rel(QG.dense_pairs(g.vvvv), QG.dense_pairs(ref.vvvv)) < tol
    assert _rel(g.src3[0], ref.src3[0]) < tol and _rel(g.src3[1], ref.src3[1]) < tol


def test_sectors_against_dense(hbr):
    '''E(Q), s2, s3, the densities, the derivatives with respect to the
    orbital energies and Gamma: those of the dense pass; the density of
    <bc||ef> in the blocks of the pair labels, a fraction of the array.'''
    eris, t2, t3, raw = hbr
    ref = QG.kernel(eris, t2, t3)
    assert abs(ref.e_q) > 1e-4 and abs(ref.e_q - QS.kernel(eris, t2, t3)[1]) < 1e-14
    g = GS.kernel(eris, t2, t3)
    _same(g, ref)
    assert isinstance(g.vvvv, QG.PairBlocks) and g.vvvv.nbytes < 0.25 * ref.vvvv.nbytes
    assert abs(QG.dense_pairs(g.vvvv)).max() > 1e-6
    # the same symmetry as a subgroup: C2, the labels modulo 4
    e4 = copy.copy(eris)
    e4.modulus = 4
    g4 = GS.kernel(e4, t2, t3)
    _same(g4, ref)
    assert g.vvvv.nbytes < g4.vvvv.nbytes < ref.vvvv.nbytes
    # and no symmetry: one label
    e1 = copy.copy(eris)
    e1.labels = np.zeros_like(eris.labels)
    _same(GS.kernel(e1, t2, t3), ref)
    # the amplitudes as the iterations left them: the symmetry-breaking elements, of 1e-10
    _same(GS.kernel(*raw), QG.kernel(*raw), 2e-9)


def test_complex_amplitudes(hbr):
    '''A phase on every spinor: complex integrals and amplitudes with the
    labels they had.'''
    eris, t2, t3, _ = hbr
    no = eris.nocc
    nv = len(eris.labels) - no
    u = np.exp(2j * np.pi * np.random.default_rng(5).random(no + nv))
    uo, uv = u[:no], u[no:]
    ec = copy.copy(eris)
    for key in ('oooo', 'ooov', 'oovv', 'ovov', 'ovvo', 'ovvv'):
        x = np.asarray(getattr(eris, key)).astype(complex)
        for ax, c in enumerate(key):
            f = uo if c == 'o' else uv
            x = x * (f.conj() if ax < 2 else f).reshape([-1 if a == ax else 1 for a in range(4)])
        setattr(ec, key, x)
    a, b = np.triu_indices(nv, 1)
    ec.__dict__.pop('_vvvv_blocks', None)
    ec._vvvv_pp = np.asarray(eris._vvvv_pp) * (uv[a] * uv[b]).conj()[:, None] * (uv[a] * uv[b])[None, :]
    t2c = t2 * (uo[:, None, None, None] * uo[None, :, None, None]
                * uv.conj()[None, None, :, None] * uv.conj()[None, None, None, :])
    o3, v3 = P.tuples(no, 3), P.tuples(nv, 3)
    t3c = t3.reshape(len(o3), len(v3)) * uo[o3].prod(axis=1)[:, None] * uv[v3].prod(axis=1).conj()[None, :]
    ref = QG.kernel(ec, t2c, t3c)
    assert abs(ref.e_q - QG.kernel(eris, t2, t3).e_q) < 1e-13 and abs(ref.s2.imag).max() > 1e-6
    _same(GS.kernel(ec, t2c, t3c), ref)


def test_orbital_gradient_from_the_blocks(hbr):
    '''The terms of the orbital gradient among the active spinors and those
    of frozen spinors (random integrals, a batch): the same from the density
    of <bc||ef> in blocks as from the array, and from the array in slabs of
    rows as at once.'''
    eris, t2, t3, _ = hbr
    no = eris.nocc
    nv = len(eris.labels) - no
    n = no + nv
    ref, g = QG.kernel(eris, t2, t3), GS.kernel(eris, t2, t3)
    Ga = QG.active_terms(eris, ref)
    assert abs(Ga[:no, no:]).max() > 1e-6 and abs(Ga[no:, no:]).max() == 0
    assert _rel(QG.active_terms(eris, g), Ga) < 1e-12
    G = np.random.default_rng(3).standard_normal((3, n, n, n))
    b, k = QG.FrozenTerms(ref, no, True)(G)
    bs, ks = QG.FrozenTerms(g, no, True)(G)
    assert _rel(bs, b) < 1e-12 and _rel(ks, k) < 1e-12
    S = np.random.default_rng(4).standard_normal((2, nv, nv * (nv - 1) // 2))
    want = QG._vvvv_terms(ref.vvvv, S, nv)
    for D, slab in ((ref.vvvv, 8 * 1000), (g.vvvv, None), (g.vvvv, 8 * 100)):
        got = QG._vvvv_terms(D, S, nv, slab)
        assert _rel(got[0], want[0]) < 1e-12 and _rel(got[1], want[1]) < 1e-12


def test_memory_plan(hbr):
    '''The peak of the allocations within the plan, and the plan not far
    above it; MemoryError where the blocks do not fit.'''
    eris, t2, t3, _ = hbr
    pl = GS.plan(eris, 8)
    assert pl.need > QS.plan(eris, 8).need + pl.kept
    with pytest.raises(MemoryError, match=r'derivatives of \(Q\) in symmetry sectors need [0-9.]+ MB'):
        GS.plan(eris, 8, pl.need // 50)
    gc.collect()
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    g = GS.kernel(eris, t2, t3, plan_=pl)
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    print('sectors: allocated %.2f MB, planned %.2f MB (%.2f)' % (peak / 1e6, pl.need / 1e6, peak / pl.need))
    assert 0.5 * pl.need < peak < pl.need
    assert g.plan is pl


def test_front_end(hbr, monkeypatch):
    '''GCCSDT._q_terms: the sectors by default; the packed block with
    ``symmetry`` off, and with too little memory for the blocks - there
    with s3 and the density of <bc||ef> in a scratch file.'''
    from zccsdt import memory
    eris, t2, t3, _ = hbr
    ref = QG.kernel(eris, t2, t3)
    for sym, tight, sectors in ((True, False, True), (False, False, False), (True, True, False)):
        cc = zccsdt.GCCSDT.from_integrals(eris, verbose=0)
        cc.symmetry = sym
        cc.t2, cc.t3, cc.e_corr, cc.e_hf = t2, t3, 0.0, eris.e_hf
        if tight:
            # (between the least of the packed block, with its large arrays mapped, and of the sectors)
            left = int(1.05 * QG.plan(eris.nocc, len(eris.labels) - eris.nocc, 8).floor)
            with pytest.raises(MemoryError):
                GS.plan(eris, 8, left)
            monkeypatch.setattr(memory, 'budget', lambda max_memory=None: left)
        g = cc._q_terms()
        assert cc.q_gradient_sectors == sectors and cc._q_terms() is g
        assert g.plan.mapped == QG.MAPPED[0] if tight else not getattr(g.plan, 'mapped', ())
        _same(g, ref)
        assert abs(cc.e_q - ref.e_q) < 1e-13
        del g, cc
        gc.collect()            # (the mappings of the scratch file go with their arrays)
