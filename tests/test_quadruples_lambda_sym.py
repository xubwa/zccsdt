'''(Q)_Lambda in symmetry sectors (zccsdt.cc.quadruples_lambda_sym) against
the dense code: a linear molecule with X2C spinors (labels 2 m_j from the
front end) with converged CCSDT and Lambda, the same with the labels modulo
4, with one label and with complex amplitudes; (Q) for lambda = conj(t);
the peak of the allocations within the plan.'''
import copy
import gc
import tracemalloc

import numpy as np
import pytest
from pyscf import gto, x2c

import zccsdt
from zccsdt.cc import (ccsdt_opt, quadruples as QD, quadruples_lambda as QL, quadruples_lambda_sym as QLS,
                       quadruples_sym as QS, t4pack as P)


@pytest.fixture(scope='module')
def hbr():
    '''HBr / cc-pVDZ-DK, 8 + 28 active spinors: (eris with labels, t2, T3,
    l2, Lambda_3), the triples packed in lexicographic order.'''
    mol = gto.M(atom='Br 0 0 0; H 0 0 1.4145', basis='cc-pvdz-dk', verbose=0)
    mf = x2c.UHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-8
    mf.kernel()
    cc = zccsdt.GCCSDT(mf, frozen=28)
    cc.verbose = 0
    cc.conv_tol, cc.conv_tol_normt = 1e-11, 1e-9
    cc.kernel()
    cc.solve_lambda()
    assert cc.converged and cc.converged_lambda
    eris = cc.ao2mo()
    return eris, cc.t2, ccsdt_opt.packed_t3(eris, cc.t3), cc.l2, np.asarray(cc.l3)


def test_lambda_has_the_labels(hbr):
    eris, t2, t3, l2, l3 = hbr
    no = eris.nocc
    lab = np.asarray(eris.labels)
    assert QLS.available(eris) and eris.modulus == 0 and len(np.unique(lab[no:])) == 6
    k = (lab[:no, None, None, None] + lab[None, :no, None, None]
         - lab[None, None, no:, None] - lab[None, None, None, no:])
    assert abs(l2[k != 0]).max() < 1e-9 and abs(l2[k == 0]).max() > 1e-2
    o3, v3 = P.tuples(no, 3), P.tuples(len(lab) - no, 3)
    k = lab[:no][o3].sum(axis=1)[:, None] - lab[no:][v3].sum(axis=1)[None, :]
    L3 = l3.reshape(len(o3), len(v3))
    assert abs(L3[k != 0]).max() < 1e-9 and abs(L3[k == 0]).max() > 1e-4
    # (and Lambda is not conj(t) at the level of the test below)
    assert abs(l2 - t2.conj()).max() > 1e-4


def test_sectors_against_dense(hbr):
    eris, t2, t3, l2, l3 = hbr
    ref = QL.kernel(eris, t2, t3, l2, l3)
    plain = QD.kernel(eris, t2, t3)
    assert abs(ref[1]) > 1e-4 and abs(ref[1] - plain[1]) > 1e-7 and abs(ref[0] - plain[0]) > 1e-7
    assert np.allclose(QLS.kernel(eris, t2, t3, l2, l3), ref, rtol=0, atol=1e-13)
    # the same symmetry as a subgroup: C2, the labels modulo 4
    e4 = copy.copy(eris)
    e4.modulus = 4
    assert np.allclose(QLS.kernel(e4, t2, t3, l2, l3), ref, rtol=0, atol=1e-13)
    # and no symmetry: one label
    e1 = copy.copy(eris)
    e1.labels = np.zeros_like(eris.labels)
    assert np.allclose(QLS.kernel(e1, t2, t3, l2, l3), ref, rtol=0, atol=1e-13)
    # lambda = conj(t): (Q), dense and in sectors
    assert np.allclose(QLS.kernel(eris, t2, t3, t2.conj(), t3.conj()), plain, rtol=0, atol=1e-13)
    assert np.allclose(QLS.kernel(eris, t2, t3, t2.conj(), t3.conj()), QS.kernel(eris, t2, t3), rtol=0, atol=1e-13)


def test_complex_amplitudes(hbr):
    '''A phase on every spinor: complex integrals, amplitudes and Lambda
    (which takes the conjugate phases of t) with the labels they had, and
    the energies of the real ones.'''
    eris, t2, t3, l2, l3 = hbr
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
    f2 = (uo[:, None, None, None] * uo[None, :, None, None]
          * uv.conj()[None, None, :, None] * uv.conj()[None, None, None, :])
    o3, v3 = P.tuples(no, 3), P.tuples(nv, 3)
    f3 = uo[o3].prod(axis=1)[:, None] * uv[v3].prod(axis=1).conj()[None, :]
    t2c, t3c = t2 * f2, t3.reshape(f3.shape) * f3
    l2c, l3c = l2 * f2.conj(), l3.reshape(f3.shape) * f3.conj()
    ref = QL.kernel(eris, t2, t3, l2, l3)
    assert np.allclose(QL.kernel(ec, t2c, t3c, l2c, l3c), ref, rtol=0, atol=1e-13)
    assert np.allclose(QLS.kernel(ec, t2c, t3c, l2c, l3c), ref, rtol=0, atol=1e-13)
    # (the phases matter: with those of t on lambda the number is another)
    assert abs(QLS.kernel(ec, t2c, t3c, l2 * f2, l3.reshape(f3.shape) * f3)[1] - ref[1]) > 1e-6


@pytest.mark.parametrize('cplx', [False, True])
def test_memory(hbr, cplx):
    '''The plan: that of (Q) in sectors and the arrays of the left-hand
    side; MemoryError where the blocks do not fit; the peak of the
    allocations within it.'''
    eris, t2, t3, l2, l3 = hbr
    if cplx:                    # (the arrays of complex amplitudes on real integrals)
        t2, t3, l2, l3 = (x.astype(complex) for x in (t2, t3, l2, l3))
    isz = 16 if cplx else 8
    pl = QLS.plan(eris, isz)
    assert pl.need > QS.plan(eris, isz).need
    with pytest.raises(MemoryError, match=r'\(Q\)_Lambda in symmetry sectors'):
        QLS.plan(eris, isz, pl.need // 50)
    at = QLS.plan(eris, isz, pl.need)
    assert at.need <= pl.need and at.dense.xv == pl.dense.xv and (at.n22, at.n13) == (pl.n22, pl.n13)
    ref = QL.kernel(eris, t2, t3, l2, l3)

    def peak_of(fn):
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        e = fn()
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        return e, peak
    e, peak = peak_of(lambda: QLS.kernel(eris, t2, t3, l2, l3, plan_=pl))
    assert np.allclose(e, ref, rtol=0, atol=1e-13)
    assert peak <= pl.need, (peak, pl.need)
    # what the left-hand side adds to (Q) in sectors is what the plan adds (the plan of (Q)
    # in sectors itself counts the intermediate of T2 with <vv||vv> as the dense code has it)
    plq = QS.plan(eris, isz)
    _, peak_q = peak_of(lambda: QS.kernel(eris, t2, t3, plan_=plq))
    more, planned = peak - peak_q, pl.need - plq.need
    assert 0.8 * planned < more <= planned, (more, planned)
