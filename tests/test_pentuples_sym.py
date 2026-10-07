'''(P) in symmetry sectors (zccsdt.cc.pentuples_sym) against the dense code
(zccsdt.cc.pentuples): HBr with X2C spinors and the labels 2 m_j of the
front end, the same labels modulo 4, one label, complex amplitudes; the
intermediates in blocks against those of the dense code; the memory plan.

(P) is a function of the amplitudes it is given: T2 here is that of second
order, T3 and T4 are random numbers on the elements that the labels allow.'''
import copy
import gc
import tracemalloc

import numpy as np
import pytest
from pyscf import gto, x2c

import zccsdt
from zccsdt.cc import pentuples as PT, pentuples_sym as PS, t4pack as P

NVIR = 20


def _allowed(lab, no, k, modulus=0):
    '''[C(no,k), C(nv,k)]: whether the labels of the element add up.'''
    d = lab[:no][P.tuples(no, k)].sum(axis=1)[:, None] - lab[no:][P.tuples(len(lab) - no, k)].sum(axis=1)[None, :]
    return (d % modulus if modulus else d) == 0


def amplitudes(eris, seed=3):
    '''(T2 of second order, random packed T3 and T4 with the symmetry of the labels).'''
    no = eris.nocc
    lab = np.asarray(eris.labels)
    mo_e = np.asarray(eris.mo_energy)
    eo, ev = mo_e[:no], mo_e[no:]
    t2 = np.conj(np.asarray(eris.oovv)) / (eo[:, None, None, None] + eo[None, :, None, None]
                                         - ev[None, None, :, None] - ev[None, None, None, :])
    rng = np.random.default_rng(seed)
    out = [t2]
    for k, scale in ((3, 3e-3), (4, 1e-3)):
        ok = _allowed(lab, no, k, int(getattr(eris, 'modulus', 0) or 0))
        out.append(scale * rng.standard_normal(ok.shape) * ok)
    return out


@pytest.fixture(scope='module')
def hbr():
    mol = gto.M(atom='Br 0 0 0; H 0 0 1.4145', basis='cc-pvdz-dk', verbose=0)
    mf = x2c.UHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-8
    mf.kernel()
    order = np.argsort(mf.mo_energy)
    nocc = int((mf.mo_occ > 0).sum())
    # (the virtual space is cut between two levels, not inside a pair +m_j, -m_j)
    assert mf.mo_energy[order[nocc + NVIR]] - mf.mo_energy[order[nocc + NVIR - 1]] > 1e-3
    cc = zccsdt.GCCSDT(mf, frozen=list(order[:28]) + list(order[nocc + NVIR:]))
    cc.verbose = 0
    eris = cc.ao2mo()
    assert eris.nocc == 8 and len(eris.labels) == 8 + NVIR and eris.modulus == 0
    assert len(np.unique(eris.labels[8:])) >= 4
    return (eris,) + tuple(amplitudes(eris))


def test_sectors_against_dense(hbr):
    eris, t2, t3, t4 = hbr
    ref = PT.kernel(eris, t2, t3, t4)
    assert abs(ref) > 1e-8
    assert PS.available(eris)
    assert abs(PS.kernel(eris, t2, t3, t4) - ref) < 1e-13 * max(1.0, abs(ref) / 1e-3)
    # the same symmetry as a subgroup: C2, the labels modulo 4
    e4 = copy.copy(eris)
    e4.modulus = 4
    assert abs(PS.kernel(e4, t2, t3, t4) - ref) < 1e-13
    # and no symmetry: one label
    e1 = copy.copy(eris)
    e1.labels = np.zeros_like(eris.labels)
    assert abs(PS.kernel(e1, t2, t3, t4) - ref) < 1e-13


def _random(no, nv, lab, modulus, cplx, seed=1):
    """A container with random blocks of integrals and random amplitudes, of
    which the elements that the labels allow are there."""
    import types
    rng = np.random.default_rng(seed)
    lab = np.asarray(lab)

    def rnd(*shape):
        x = rng.standard_normal(shape)
        return x + 1j * rng.standard_normal(shape) if cplx else x

    def block(key):
        x, k = rnd(*[no if c == 'o' else nv for c in key]), 0
        for ax, c in enumerate(key):
            l = lab[:no] if c == 'o' else lab[no:]
            k = k + (1 if ax < 2 else -1) * l.reshape([-1 if n == ax else 1 for n in range(4)])
        return x * ((k % modulus if modulus else k) == 0)
    mo_e = np.concatenate((np.sort(rng.uniform(-3, -1, no)), np.sort(rng.uniform(1, 3, nv))))
    eris = types.SimpleNamespace(nocc=no, fock=np.diag(mo_e), mo_energy=mo_e, vvvv=None, labels=lab, modulus=modulus,
                                 **{key: block(key) for key in ('oooo', 'ooov', 'oovv', 'ovov', 'ovvo', 'ovvv')})
    a, b = np.triu_indices(nv, 1)
    k = (lab[no:][a] + lab[no:][b])[:, None] - (lab[no:][a] + lab[no:][b])[None, :]
    eris._vvvv_pp = rnd(len(a), len(a)) * ((k % modulus if modulus else k) == 0)
    t2 = block('oovv')
    t2 = t2 - t2.transpose(1, 0, 2, 3)
    t2 = t2 - t2.transpose(0, 1, 3, 2)
    t3, t4 = (rnd(*ok.shape) * ok for ok in (_allowed(lab, no, 3, modulus), _allowed(lab, no, 4, modulus)))
    return eris, t2, t3, t4


@pytest.mark.parametrize('no,nv,cplx,modulus,labels', [
    (6, 9, True, 0, [-1, 1, 1, -1, 3, -3, 1, -1, 3, 1, -1, 5, -3, 1, -1]),      # spinors in any order
    (7, 8, False, 4, [1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 3, 1]),             # the two irreps of C2
    (6, 8, True, 3, [0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 2, 1, 1, 0]),                 # three classes, modulo 3
    (9, 7, False, 0, [0, 2, -2, 0, 2, -2, 0, 4, -4, 0, 2, -2, 6, -6, 2, 0]),     # classes of one spinor
    (5, 6, True, 0, [0, 0, 0, 0, 1, 1, 1, 1, -1, -1, 7]),                        # one pentuple, a label apart
])
def test_random_labels(no, nv, cplx, modulus, labels):
    """Random integrals and amplitudes with labels of several kinds - more
    occupied than virtual spinors, classes of one spinor, a sector of one
    virtual pentuple and a label that nothing goes with: the energy of the
    dense code."""
    eris, t2, t3, t4 = _random(no, nv, labels, modulus, cplx)
    ref = PT.kernel(eris, t2, t3, t4)
    assert abs(ref) > 1e-3
    assert abs(PS.kernel(eris, t2, t3, t4) - ref) < 1e-12 * abs(ref)
    if cplx:
        # (real amplitudes of the higher ranks with complex integrals)
        ref = PT.kernel(eris, t2, t3.real, t4.real)
        assert abs(PS.kernel(eris, t2, t3.real, t4.real) - ref) < 1e-12 * abs(ref)


def test_intermediates_in_blocks(hbr):
    """Xv and Xq in blocks are the elements of those of the dense code that
    the labels allow, and the others vanish there."""
    eris, t2, t3, t4 = hbr
    no, nv = eris.nocc, len(eris.labels) - eris.nocc
    d = PT.Pentuples(eris, t2, t3, t4)
    s = PS.Pentuples(eris, t2, t3, t4, PS.plan(eris, 8))
    sec = s.sec
    for dense, blocks, rows, k in ((d.Xv, s.Xv, sec.V, 1), (d.Xq, s.Xq, sec.O, -1)):
        scale = abs(dense).max()
        assert scale > 1e-6 and len(blocks) == len(dense)
        seen = 0
        for p, blk in enumerate(blocks):
            rest = np.array(dense[p])
            kp = sec.kpair[p] if k == 1 else sec.k4[p]
            for kr, x in blk.items():
                at = np.ix_(rows[kr], sec.A3[int(sec.red(k * kr + kp))])
                assert abs(x - rest[at]).max() < 1e-13 * scale
                rest[at] = 0
                seen += x.size
            assert abs(rest).max() < 1e-13 * scale
        assert seen < 0.3 * dense.size


def test_complex_amplitudes(hbr):
    """A phase on every spinor: complex integrals and amplitudes with the
    labels they had, and the energy of the real ones."""
    eris, t2, t3, t4 = hbr
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
    ec.__dict__.pop('_vvvv_blocks', None)       # (the cache of the real block)
    ec._vvvv_pp = np.asarray(eris._vvvv_pp) * (uv[a] * uv[b]).conj()[:, None] * (uv[a] * uv[b])[None, :]
    t2c = t2 * (uo[:, None, None, None] * uo[None, :, None, None]
                * uv.conj()[None, None, :, None] * uv.conj()[None, None, None, :])
    amps = [t2c]
    for k, t in ((3, t3), (4, t4)):
        ok, vk = P.tuples(no, k), P.tuples(nv, k)
        amps.append(t * uo[ok].prod(axis=1)[:, None] * uv[vk].prod(axis=1).conj()[None, :])
    ref = PT.kernel(eris, t2, t3, t4)
    assert abs(PT.kernel(ec, *amps) - ref) < 1e-13
    assert abs(PS.kernel(ec, *amps) - ref) < 1e-13


@pytest.mark.parametrize('how', ['2 m_j', 'modulo 4', 'one label'])
def test_memory_plan(hbr, how):
    """The arrays that a run allocates (tracemalloc follows numpy's
    allocations) stay within Plan.need, and the blocks take less than the
    arrays of the dense code; MemoryError where the blocks do not fit."""
    eris, t2, t3, t4 = hbr
    eris = copy.copy(eris)
    if how == 'modulo 4':
        eris.modulus = 4
    elif how == 'one label':
        eris.labels = np.zeros_like(eris.labels)
    no, nv = eris.nocc, len(eris.labels) - eris.nocc
    PT.kernel(eris, t2, t3, t4)                 # (the tables of the occupied tuples: made once)
    pl = PS.plan(eris, 8)
    with pytest.raises(MemoryError, match='in blocks of the symmetry labels'):
        PS.plan(eris, 8, pl.need - 1)
    assert PS.plan(eris, 8, pl.need).need == pl.need
    gc.collect()
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    PS.kernel(eris, t2, t3, t4, plan_=pl)
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    assert peak <= pl.need, (peak, pl.need)
    print(how, 'peak / plan', peak / pl.need, 'plan / dense plan', pl.need / PT.plan(no, nv, 8).need)
    assert peak > 0.5 * pl.need, (peak, pl.need)
    dense = PT.plan(no, nv, 8)
    if how != 'one label':
        assert pl.need < dense.need
        # (two classes of the triples modulo 4: half of the dense arrays)
        part = 0.3 if how == '2 m_j' else 0.5
        assert pl.parts['Xv'] <= part * PT._big(no, nv, 8)[1] and pl.parts['Xq'] <= part * PT._kept(no, nv, 8)[1]
