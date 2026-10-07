'''(Q) in symmetry sectors (zccsdt.cc.quadruples_sym) against the dense
code: a linear molecule with X2C spinors (labels 2 m_j from the front end),
the same with the labels modulo 4 and with complex amplitudes, and spin
orbitals of RHF with the two fermion irreps of C2 from the integrals in
Kramers pairs.'''
import copy

import numpy as np
import pytest
from pyscf import gto, scf, x2c

import zccsdt
from zccsdt import dirac
from zccsdt.cc import quadruples as QD, quadruples_sym as QS, t4pack as P


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
    return cc.ao2mo(), cc.t2, np.asarray(cc.t3)


def _conserved(x, lab, key, no, modulus=0):
    '''Largest element of the block ``key`` whose labels do not add up.'''
    k = 0
    for ax, c in enumerate(key):
        l = lab[:no] if c == 'o' else lab[no:]
        k = k + (1 if ax < 2 else -1) * l.reshape([-1 if a == ax else 1 for a in range(4)])
    bad = (k % modulus if modulus else k) != 0
    return abs(np.asarray(x)[bad]).max()


def test_labels_of_a_linear_molecule(hbr):
    eris, t2, t3 = hbr
    no = eris.nocc
    lab = np.asarray(eris.labels)
    assert eris.modulus == 0 and sorted(lab[:no]) == [-3, -1, -1, -1, 1, 1, 1, 3]
    assert (np.abs(lab) % 2 == 1).all() and lab.sum() == 0
    for key in ('oooo', 'ooov', 'oovv', 'ovov', 'ovvv'):
        assert _conserved(getattr(eris, key), lab, key, no) < 1e-12
    assert _conserved(t2, lab, 'oovv', no) < 1e-9
    assert QS.available(eris)


def test_sectors_against_dense(hbr):
    eris, t2, t3 = hbr
    ref = QD.kernel(eris, t2, t3)
    assert abs(ref[1]) > 1e-4
    assert np.allclose(QS.kernel(eris, t2, t3), ref, rtol=0, atol=1e-14)
    # the same symmetry as a subgroup: C2, the labels modulo 4
    e4 = copy.copy(eris)
    e4.modulus = 4
    assert np.allclose(QS.kernel(e4, t2, t3), ref, rtol=0, atol=1e-14)
    # and no symmetry: one label
    e1 = copy.copy(eris)
    e1.labels = np.zeros_like(eris.labels)
    assert np.allclose(QS.kernel(e1, t2, t3), ref, rtol=0, atol=1e-14)


def test_complex_amplitudes(hbr):
    '''A phase on every spinor: complex integrals and amplitudes with the
    labels they had, and the energies of the real ones.'''
    eris, t2, t3 = hbr
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
    o3, v3 = P.tuples(no, 3), P.tuples(nv, 3)
    t3c = t3.reshape(len(o3), len(v3)) * uo[o3].prod(axis=1)[:, None] * uv[v3].prod(axis=1).conj()[None, :]
    ref = QD.kernel(eris, t2, t3)
    assert np.allclose(QD.kernel(ec, t2c, t3c), ref, rtol=0, atol=1e-13)
    assert np.allclose(QS.kernel(ec, t2c, t3c), ref, rtol=0, atol=1e-13)


def test_two_irreps_of_c2():
    '''Spin orbitals of RHF in Kramers pairs: the labels +1 and -1 modulo 4
    of the integral files.'''
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='6-31g', verbose=0)
    mf = scf.addons.convert_to_ghf(scf.RHF(mol).run(conv_tol=1e-12, conv_tol_grad=1e-8))
    eris = dirac.container(dirac.integrals(mf))
    assert eris.modulus == 4 and set(np.unique(eris.labels)) == {-1, 1}
    cc = zccsdt.GCCSDT.from_integrals(eris, verbose=0)
    cc.conv_tol, cc.conv_tol_normt = 1e-11, 1e-9
    cc.kernel()
    ref = QD.kernel(eris, cc.t2, cc.t3)
    assert np.allclose(QS.kernel(eris, cc.t2, cc.t3), ref, rtol=0, atol=1e-14)
