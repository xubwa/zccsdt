'''The first derivatives of (Q)_Lambda in symmetry sectors
(zccsdt.cc.quadruples_lambda_grad_sym) against the pass on the packed block
(zccsdt.cc.quadruples_lambda_grad), and CCSDT(Q)_Lambda properties with
everything in the blocks of the symmetry labels against everything on the
packed block: HBr / cc-pVDZ-DK with X2C spinors, 28 frozen (8 + 28 active;
labels 2 m_j), the same labels modulo 4, one label, complex amplitudes; the
memory plan against what is allocated; the front end.

As in test_quadruples_grad_sym, the amplitudes, lambda and the integrals are
projected on their label-conserving parts for the comparisons of the pass.'''
import copy
import gc
import tracemalloc

import numpy as np
import pytest
from pyscf import gto, x2c

import zccsdt
from test_quadruples_grad_sym import project
from zccsdt.cc import ccsdt_opt, quadruples_grad as QG, quadruples_lambda_grad as LG, quadruples_lambda_grad_sym as LS, \
    quadruples_lambda_sym as QLS, quadruples_sym as QS, t4pack as P

FIELDS = ('s2', 's3', 'q2', 'q3', 'de_occ', 'gamma_vv', 'vvoo', 'vvvo', 'ovoo', 'oooo', 'ovvo')


def _hbr(**settings):
    mol = gto.M(atom='Br 0 0 0; H 0 0 1.4145', basis='cc-pvdz-dk', verbose=0)
    mf = x2c.UHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-9
    mf.kernel()
    cc = zccsdt.GCCSDT(mf, frozen=28)
    cc.verbose = 0
    cc.conv_tol, cc.conv_tol_normt = 1e-11, 1e-9
    for k, v in settings.items():
        setattr(cc, k, v)
    cc.kernel()
    return mol, cc


@pytest.fixture(scope='module')
def hbr():
    mol, cc = _hbr()
    cc.solve_lambda()
    eris = cc.ao2mo()
    raw = (eris, cc.t2, ccsdt_opt.packed_t3(eris, cc.t3), cc.l2, np.asarray(cc._packed_l3()).reshape(56, -1))
    ec, t2, t3 = project(*raw[:3])
    _, l2, l3 = project(eris, raw[3], raw[4])
    return ec, t2, t3, l2, l3, raw


def _rel(a, b):
    return abs(a - b).max() / abs(b).max()


def _same(g, ref, tol=1e-12):
    assert abs(g.e_q - ref.e_q) < tol * abs(ref.e_q) and abs(g.e_imag) < tol * abs(ref.e_q)
    for k in FIELDS:
        assert _rel(getattr(g, k), getattr(ref, k)) < tol, k
    assert _rel(QG.dense_pairs(g.vvvv), QG.dense_pairs(ref.vvvv)) < tol
    assert _rel(g.src3[0], ref.src3[0]) < tol and _rel(g.src3[1], ref.src3[1]) < tol


def test_sectors_against_dense(hbr):
    '''E(Q)_L, the sources of the two equations, the densities, the
    derivatives with respect to the orbital energies and Gamma: those of
    the dense pass; the density of <bc||ef> in the blocks of the pair
    labels, a fraction of the array.'''
    eris, t2, t3, l2, l3, raw = hbr
    ref = LG.kernel(eris, t2, t3, l2, l3)
    assert abs(ref.e_q) > 1e-4 and abs(ref.e_q - QLS.kernel(eris, t2, t3, l2, l3)[1]) < 1e-14
    g = LS.kernel(eris, t2, t3, l2, l3)
    _same(g, ref)
    assert isinstance(g.vvvv, QG.PairBlocks) and g.vvvv.nbytes < 0.25 * ref.vvvv.nbytes
    assert abs(g.q3).max() > 1e-5 and abs(g.s3).max() > 1e-5 and abs(QG.dense_pairs(g.vvvv)).max() > 1e-6
    # the same symmetry as a subgroup: C2, the labels modulo 4
    e4 = copy.copy(eris)
    e4.modulus = 4
    g4 = LS.kernel(e4, t2, t3, l2, l3)
    _same(g4, ref)
    assert g.vvvv.nbytes < g4.vvvv.nbytes < ref.vvvv.nbytes
    # and no symmetry: one label
    e1 = copy.copy(eris)
    e1.labels = np.zeros_like(eris.labels)
    _same(LS.kernel(e1, t2, t3, l2, l3), ref)
    # the amplitudes as the iterations left them: the symmetry-breaking elements, of 1e-10
    _same(LS.kernel(*raw), LG.kernel(*raw), 2e-9)


def test_complex_amplitudes(hbr):
    '''A phase on every spinor: complex integrals, amplitudes and lambda
    with the labels they had - lambda with the conjugate phases of t.'''
    eris, t2, t3, l2, l3, _ = hbr
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
    ph2 = (uo[:, None, None, None] * uo[None, :, None, None]
           * uv.conj()[None, None, :, None] * uv.conj()[None, None, None, :])
    o3, v3 = P.tuples(no, 3), P.tuples(nv, 3)
    ph3 = uo[o3].prod(axis=1)[:, None] * uv[v3].prod(axis=1).conj()[None, :]
    args = (ec, t2 * ph2, t3 * ph3, l2 * ph2.conj(), l3 * ph3.conj())
    ref = LG.kernel(*args)
    assert abs(ref.e_q - LG.kernel(eris, t2, t3, l2, l3).e_q) < 1e-13 and abs(ref.e_imag) < 1e-15
    assert abs(ref.s2.imag).max() > 1e-6 and abs(ref.q2.imag).max() > 1e-6
    _same(LS.kernel(*args), ref)


def test_memory_plan(hbr):
    '''The peak of the allocations within the plan, and the plan not far
    above it; MemoryError where the blocks do not fit.'''
    eris, t2, t3, l2, l3, _ = hbr
    pl = LS.plan(eris, 8)
    assert pl.need > QS.plan(eris, 8).need + pl.kept
    with pytest.raises(MemoryError, match=r'derivatives of \(Q\)_Lambda in symmetry sectors need [0-9.]+ MB'):
        LS.plan(eris, 8, pl.need // 50)
    gc.collect()
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    g = LS.kernel(eris, t2, t3, l2, l3, plan_=pl)
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    print('sectors: allocated %.2f MB, planned %.2f MB (%.2f)' % (peak / 1e6, pl.need / 1e6, peak / pl.need))
    assert 0.5 * pl.need < peak < pl.need
    assert g.plan is pl


def test_front_end_pass(hbr, monkeypatch):
    '''GCCSDT._ql_terms: the sectors by default; the packed block with
    ``symmetry`` off, and with too little memory for the blocks - there
    with s3, q3 and the density of <bc||ef> in a scratch file.'''
    from zccsdt import memory
    eris, t2, t3, l2, l3, _ = hbr
    ref = LG.kernel(eris, t2, t3, l2, l3)
    for sym, tight, sectors in ((True, False, True), (False, False, False), (True, True, False)):
        cc = zccsdt.GCCSDT.from_integrals(eris, verbose=0)
        cc.symmetry = sym
        cc.t2, cc.t3, cc.l2, cc.l3, cc.e_corr, cc.e_hf = t2, t3, l2, l3, 0.0, eris.e_hf
        if tight:
            # (between the least of the packed block, with its large arrays mapped, and of the sectors)
            left = int(1.05 * LG.plan(eris.nocc, len(eris.labels) - eris.nocc, 8).floor)
            with pytest.raises(MemoryError):
                LS.plan(eris, 8, left)
            monkeypatch.setattr(memory, 'budget', lambda max_memory=None: left)
        g = cc._ql_terms()
        assert cc.ql_gradient_sectors == sectors and cc._ql_terms() is g
        assert g.plan.mapped == LG.MAPPED[0] if tight else not getattr(g.plan, 'mapped', ())
        _same(g, ref)
        assert abs(cc.e_ql - ref.e_q) < 1e-13 and abs(cc.e_tot_ql - eris.e_hf - ref.e_q) < 1e-13
        del g, cc
        gc.collect()            # (the mappings of the scratch file go with their arrays)


def test_properties_in_blocks_against_the_packed_block():
    """CCSDT(Q)_Lambda properties for HBr, 28 frozen spinors (Kramers
    pairs): the pass in sectors, the amplitude steps of zeta, Lambda,
    lambda' and the densities in the blocks of the labels (blocks_min = 0)
    against the same without the labels - everything on the packed block.
    zeta, lambda' and the relaxed density agree to the thresholds of the
    iterations, and the dipole moment is not that of CCSDT or CCSDT(Q)."""
    mol, cb = _hbr(blocks_min=0)
    Db = cb.make_rdm1(with_q='lambda')
    info = cb.ql_info
    assert cb.ql_gradient_sectors and cb.lambda_blocks and cb.rdm1_blocks and info['lambda']['blocks']
    assert cb.converged_zeta and cb.converged_lambda_ql and info['zeta']['stencil_error'] < 1e-12
    assert np.size(cb.t3) < 56 * 3276 and np.size(cb.zeta3) == np.size(cb.t3) == np.size(cb.l3_ql)
    C, lbar = cb.rdm1_basis, cb.l2_ql - cb.l2
    mol, cd = _hbr(symmetry=False)
    Dd = cd.make_rdm1(with_q='lambda')
    assert not cd.ql_gradient_sectors and not cd.lambda_blocks and not cd.rdm1_blocks
    assert np.size(cd.t3) == 56 * 3276 == np.size(cd.zeta3)
    assert abs(cb.e_ql - cd.e_ql) < 1e-11 and abs(cb.e_ql) > 1e-4
    # (the two runs have the spinors in different orders: compare in the AO basis)
    Cd = cd.rdm1_basis
    Ab, Ad = C @ Db @ C.conj().T, Cd @ Dd @ Cd.conj().T
    assert abs(Ab - Ad).max() < 2e-8
    z = mol.intor('int1e_r_spinor')[2]
    mu = np.einsum('pq,qp', z, Ab).real
    assert abs(mu - np.einsum('pq,qp', z, Ad).real) < 2e-8
    for other in (False, True):
        Do = cb.make_rdm1(with_q=other, ao_repr=True)
        assert abs(np.einsum('pq,qp', z, Do).real - mu) > 1e-6
    # zeta and lambda' - lambda themselves (the spinors of the two runs are in different orders)
    for a, b in ((cb.zeta2, cd.zeta2), (lbar, cd.l2_ql - cd.l2), (cb.zeta3, cd.zeta3)):
        assert abs(np.linalg.norm(a) - np.linalg.norm(b)) < 1e-8 and np.linalg.norm(a) > 1e-4
    del cb, cd
    gc.collect()
