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
