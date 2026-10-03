'''The spinor CCSD and (T) copied from socutils (zccsdt.zccsd): the same
energies as zccsdt's CCSD and PySCF's GCCSD(T), both (T) loops alike; the
integrals of zccsdt.zccsd.ao2mo against the route of socutils (nrr_outcore).'''

import numpy as np
import pytest
from pyscf import cc, gto, lib, scf, x2c
from pyscf.cc import gccsd_t as pyscf_gccsd_t

from zccsdt import eris as E
from zccsdt.cc import solver
from zccsdt.cc import gintermediates as imd
from zccsdt.zccsd import DirectZCCSD, ZCCSD, ao2mo, gccsd_t
from zccsdt.zccsd.zccsd_direct import get_ovvv
from zccsdt.scratch import is_mapped as scratch_is_mapped


@pytest.fixture(scope='module')
def x2c_h2o():
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='6-31g', verbose=0)
    return x2c.UHF(mol).run(conv_tol=1e-11)


def _run(cls, mf):
    mycc = cls(mf, frozen=2)
    mycc.verbose, mycc.conv_tol, mycc.conv_tol_normt = 0, 1e-11, 1e-8
    mycc.kernel()
    assert mycc.converged
    return mycc


def test_ccsd_equals_zccsdt(x2c_h2o, monkeypatch, tmp_path):
    monkeypatch.setattr(lib.param, 'TMPDIR', str(tmp_path))     # ZCCSD's scratch files
    ref = solver.kernel(E.from_scf(x2c_h2o, frozen=2), 'ccsd',
                        solver.Options(conv_tol=1e-11, conv_tol_normt=1e-8))
    for cls in (ZCCSD, DirectZCCSD):
        mycc = _run(cls, x2c_h2o)
        assert abs(mycc.e_corr - ref.e_corr.real) < 1e-9, cls.__name__


def test_integrals_equal_nrr_outcore(x2c_h2o, monkeypatch, tmp_path):
    monkeypatch.setattr(lib.param, 'TMPDIR', str(tmp_path))
    mol = x2c_h2o.mol
    old = ZCCSD(x2c_h2o, frozen=2)
    old.verbose, old.ao2mo_outcore = 0, True
    ref = old.ao2mo()                               # all MO integrals, HDF5
    imd._make_vvvv_tril(ref)
    nocc, nao = ref.nocc, mol.nao_nr()
    Ca, Cb = ao2mo.spin_components(mol, ref.mo_coeff)
    nmo, nvir = Ca.shape[1], Ca.shape[1] - nocc

    def check(blocks, vvvv):
        for key in ao2mo.BLOCKS:
            assert abs(blocks[key] - np.asarray(getattr(ref, key))).max() < 1e-12, key
        assert abs(vvvv['vvvv_tril'] - np.asarray(ref.vvvv_tril)).max() < 1e-12
        assert all(np.array_equal(x, y) for x, y in zip(vvvv.idx, ref.vvvv_tril_idx))

    # everything in memory, one pass
    blocks = ao2mo.occupied_blocks(mol, Ca, Cb, nocc)
    vvvv = ao2mo.packed_vvvv(mol, Ca, Cb, nocc)
    check(blocks, vvvv)
    assert not blocks.on_disk() and blocks.scratch is None and blocks.plan.nbra == nocc
    # one orbital per pass and one shell pair of AO integrals at a time
    one = ao2mo.Plan(1, 1, 1, 0, 0, None)
    check(ao2mo.occupied_blocks(mol, Ca, Cb, nocc, plan=one),
          ao2mo.packed_vvvv(mol, Ca, Cb, nocc, plan=one))
    assert 'ovvv' not in ao2mo.occupied_blocks(mol, Ca, Cb, nocc, with_ovvv=False)
    # the AO operators of the ladder, as socutils builds them from the 4-index array
    E = mol.intor('int2e')
    ops = ao2mo.ladder_operators(mol)
    pi, pj = ops.idx
    assert abs(ops['ao_eri'] - E.transpose(1, 3, 0, 2).reshape(nao * nao, nao * nao)).max() < 1e-12
    W_ref = (E[pi[:, None], pi[None, :], pj[:, None], pj[None, :]]
             - E[pi[:, None], pj[None, :], pj[:, None], pi[None, :]])
    assert abs(ops['W_same'] - W_ref).max() < 1e-12

    # short of memory: the least that works puts the blocks into a scratch
    # file and slices the passes; below it, MemoryError
    from zccsdt import memory
    monkeypatch.setattr(memory, 'resident', lambda: 0)
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    need = max(ao2mo.plan_slabs(mol, nocc, nmo).need,
               ao2mo.plan_slabs(mol, nvir, nvir, sort_arrays=0).need)
    low = 1.01 * need / 1e6                         # MB, the whole budget
    blocks = ao2mo.occupied_blocks(mol, Ca, Cb, nocc, max_memory=low)
    vvvv = ao2mo.packed_vvvv(mol, Ca, Cb, nocc, max_memory=low)
    check(blocks, vvvv)
    assert 'ovvv' in blocks.on_disk() and vvvv.on_disk() == ['vvvv_tril']
    assert blocks.plan.nbra < nocc and blocks.plan.work <= low * 1e6
    assert 'scratch file' in blocks.describe()
    with pytest.raises(MemoryError, match='one orbital at a time'):
        ao2mo.occupied_blocks(mol, Ca, Cb, nocc, max_memory=0.5 * need / 1e6)
    small = ao2mo.ladder_operators(mol, max_memory=low)
    assert small.on_disk() and abs(small['W_same'] - W_ref).max() < 1e-12
    assert abs(small['ao_eri'] - ops['ao_eri']).max() < 1e-12
    # the default route of ZCCSD.ao2mo gives the same object
    eris = ZCCSD(x2c_h2o, frozen=2).ao2mo()
    assert eris.vvvv is None and abs(eris.ovvv - np.asarray(ref.ovvv)).max() < 1e-12
    assert abs(np.asarray(eris.vvvv_tril) - np.asarray(ref.vvvv_tril)).max() < 1e-12


def test_socutils_routes_kept(x2c_h2o, monkeypatch, tmp_path):
    '''ovvv from the AO integrals in every term, and the integrals through
    nrr_outcore: the same energies as the defaults.'''
    monkeypatch.setattr(lib.param, 'TMPDIR', str(tmp_path))
    e_ref = _run(DirectZCCSD, x2c_h2o).e_corr
    for cls, switch in ((DirectZCCSD, 'ovvv_direct'), (DirectZCCSD, 'socutils_amps'),
                        (ZCCSD, 'ao2mo_outcore')):
        mycc = cls(x2c_h2o, frozen=2)
        setattr(mycc, switch, True)
        mycc.verbose, mycc.conv_tol, mycc.conv_tol_normt = 0, 1e-11, 1e-8
        mycc.kernel()
        assert mycc.converged and abs(mycc.e_corr - e_ref) < 1e-9, switch
        if switch == 'ovvv_direct':
            assert mycc.eris.ovvv is None and mycc.eris.E4 is not None
        if switch == 'socutils_amps':
            assert mycc.eris.residual is None and mycc.eris.ovvv is not None


def test_amps_equal_socutils(x2c_h2o, monkeypatch):
    '''The amplitude equations on packed amplitudes (zccsdt.zccsd.amps)
    against update_amps as copied from socutils, for amplitudes with a large
    T1: in one block, in blocks of columns and pairs, and with the numpy
    equivalents of the kernels.'''
    from zccsdt.zccsd import kernels as K, zccsd_direct as zd
    new = DirectZCCSD(x2c_h2o, frozen=2)
    old = DirectZCCSD(x2c_h2o, frozen=2)
    old.socutils_amps = True
    new.verbose = old.verbose = 0
    e_new, e_old = new.ao2mo(), old.ao2mo()
    assert e_new.ovvv is None and e_new.ovvo is None and e_new.ao_eri is None
    nocc, nvir = e_new.nocc, e_new.fock.shape[0] - e_new.nocc
    b, c = np.tril_indices(nvir, -1)
    assert abs(e_new.ovvv_p - e_old.ovvv[:, :, b, c]).max() < 1e-12
    rng = np.random.default_rng(7)
    t1 = 0.05 * (rng.standard_normal((nocc, nvir)) + 1j * rng.standard_normal((nocc, nvir)))
    t2 = rng.standard_normal((nocc,) * 2 + (nvir,) * 2) + 1j * rng.standard_normal((nocc,) * 2 + (nvir,) * 2)
    t2 = 0.02 * (t2 - t2.transpose(1, 0, 2, 3))
    t2 = t2 - t2.transpose(0, 1, 3, 2)
    for shift in (0.0, 0.3):
        new.level_shift = old.level_shift = shift
        t1n, t2n = new.update_amps(t1, t2, e_new)
        t1o, t2o = zd.update_amps(old, t1, t2, e_old)
        assert abs(t1n - t1o).max() < 1e-12 and abs(t2n - t2o).max() < 1e-12
    res = e_new.residual
    npo = nocc * (nocc - 1) // 2
    # blocks of columns of the ring terms and of pairs; the numpy kernels
    res.plan.jblocks = [(0, 1), (1, 4), (4, nocc)]
    res.plan.mblocks = [(0, 2), (2, 3), (3, nocc)]
    res.plan.pblocks = [(0, 3), (3, npo)]
    res.plan.lblocks = [(0, 5), (5, 6), (6, npo)]
    res.plan.krows = 7
    t1b, t2b = res.update(t1, t2, shift)
    assert abs(t1b - t1o).max() < 1e-12 and abs(t2b - t2o).max() < 1e-12
    monkeypatch.setattr(K, 'USE_C', False)
    t1b, t2b = res.update(t1, t2, shift)
    assert abs(t1b - t1o).max() < 1e-12 and abs(t2b - t2o).max() < 1e-12
    monkeypatch.setattr(K, 'USE_C', True)
    # energy and first amplitudes on packed pairs
    io, jo = np.tril_indices(nocc, -1)
    t2p = t2[io, jo][:, b, c]
    assert abs(res.energy(t1, t2p) - (np.einsum('ia,ia', e_old.fock[:nocc, nocc:], t1)
                                      + 0.25 * np.einsum('ijab,ijab', t2, e_old.oovv)
                                      + 0.5 * np.einsum('ia,jb,ijab', t1, t1, e_old.oovv))) < 1e-12
    emp2, t1i, t2i = old.init_amps(e_old)
    e0, t10, t2p0 = res.init_amps()
    assert abs(e0 - emp2) < 1e-12 and abs(t10 - t1i).max() < 1e-12
    assert abs(t2p0 - t2i[io, jo][:, b, c]).max() < 1e-12
    # the ladder alone, and (T) from the packed <ia||bc>
    tau = imd.make_tau(t2, t1, t1)
    L = res.ladder(np.ascontiguousarray(tau[io, jo][:, b, c]),
                   np.zeros((npo, b.size), dtype=complex))
    assert abs(L - zd._ladder_lc(tau, e_old)[io, jo][:, b, c]).max() < 1e-12
    assert abs(get_ovvv(e_new) - e_old.ovvv).max() < 1e-12


def test_amps_within_memory(x2c_h2o, monkeypatch, tmp_path):
    '''The amplitude equations with the least memory that works: the
    integrals in scratch files, W(mbej) mapped, one occupied orbital per slab
    and block, the AO operators by rows; the same step and the same CCSD
    energy as in memory; MemoryError below that.'''
    from zccsdt import memory
    from zccsdt.zccsd import amps, kernels as K
    monkeypatch.setattr(memory, 'resident', lambda: 0)
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    monkeypatch.setattr(lib.param, 'TMPDIR', str(tmp_path))
    mol = x2c_h2o.mol
    ref = DirectZCCSD(x2c_h2o, frozen=2)
    ref.verbose, ref.conv_tol, ref.conv_tol_normt = 0, 1e-11, 1e-8
    ref.max_memory = 4000
    e_ref = ref.ao2mo()
    o, v, nao = e_ref.nocc, e_ref.fock.shape[0] - e_ref.nocc, mol.nao_nr()
    res = amps.Residual(e_ref, ref.max_memory)
    assert not res.plan.w_mapped and res.plan.diis_incore
    assert res.plan.jblocks == res.plan.mblocks == [(0, o)] and res.plan.krows == 0
    rng = np.random.default_rng(3)
    t1 = 0.05 * (rng.standard_normal((o, v)) + 1j * rng.standard_normal((o, v)))
    t2p = 0.02 * (rng.standard_normal((K.npair(o), K.npair(v)))
                  + 1j * rng.standard_normal((K.npair(o), K.npair(v))))
    t1r, t2r = res.update_packed(t1, t2p)

    # the least memory for the integrals, then for the step
    low = 1.01 * max(ao2mo.plan_slabs(mol, o, o + v).need, 1) / 1e6
    while True:
        try:
            cc = DirectZCCSD(x2c_h2o, frozen=2)
            cc.verbose, cc.conv_tol, cc.conv_tol_normt, cc.max_memory = 0, 1e-11, 1e-8, low
            eris = cc.ao2mo()
            break
        except MemoryError:                         # the AO operators need more
            low *= 1.2
    mapped = [k for k, a in (('V', eris.oovv_r), ('E', eris.ovov_r), ('ovvv_p', eris.ovvv_p),
                             ('W_same', eris.W_same), ('W_plus', eris.W_plus))
              if scratch_is_mapped(a)]
    assert {'V', 'E', 'ovvv_p'} <= set(mapped)
    need = amps.plan_step(o, v, nao, None, mapped).need
    plan = amps.plan_step(o, v, nao, int(1.01 * need), mapped)
    assert plan.w_mapped and not plan.diis_incore and plan.work <= 1.01 * need
    assert len(plan.jblocks) == len(plan.mblocks) == o
    with pytest.raises(MemoryError, match='three packed T2'):
        amps.plan_step(o, v, nao, int(0.9 * need), mapped)
    low_res = amps.Residual(eris, plan=plan)
    for _ in range(2):                              # the mapped W is used again
        t1n, t2n = low_res.update_packed(t1, t2p)
        assert abs(t1n - t1r).max() < 1e-12 and abs(t2n - t2r).max() < 1e-12
    assert 'scratch file' in low_res.describe()
    low_res.close()
    # more memory: larger blocks, W in memory
    mid = amps.plan_step(o, v, nao, int(1.5 * need), mapped)
    assert not mid.w_mapped and 1 < len(mid.jblocks) < o
    t1n, t2n = amps.Residual(eris, plan=mid).update_packed(t1, t2p)
    assert abs(t1n - t1r).max() < 1e-12 and abs(t2n - t2r).max() < 1e-12
    # the whole calculation within max_memory
    cc.max_memory = max(low, 1.01 * need / 1e6)
    cc.kernel(eris=eris)
    ref.kernel(eris=e_ref)
    assert cc.converged and abs(cc.e_corr - ref.e_corr) < 1e-9
    assert eris.residual.plan.w_mapped
    assert abs(cc.t2 - ref.t2).max() < 1e-7


def test_kernels_equal_numpy():
    '''Every kernel of csrc/zccsd_kernels.c against its numpy equivalent.'''
    from zccsdt.zccsd import kernels as K
    if not K.USE_C:
        pytest.skip('libzc_zccsd.so is not built')
    rng = np.random.default_rng(1)
    o, v, nao = 5, 7, 4
    npo, npv, n1, n2 = K.npair(o), K.npair(v), o * v, 2 * nao

    def rnd(*shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    def both(f, *args, **kw):
        '''f on copies of the arguments with the C and the numpy kernels:
        the results and the arguments after the calls.'''
        out = []
        for use in (True, False):
            K.USE_C = use
            a = [x.copy() if isinstance(x, np.ndarray) else x for x in args]
            out.append((f(*a, **kw), a))
        K.USE_C = True
        return out

    def same(x, y):
        x, y = (x, y) if isinstance(x, tuple) else ((x,), (y,))
        assert max(abs(a - b).max() for a, b in zip(x, y)) < 1e-12

    t1, u, t2p, R = rnd(o, v), rnd(o, v), rnd(npo, npv), rnd(npo, npv)
    try:
        for f, args in ((K.outer_pp, (t1, u, 0.7, R)), (K.unpack_t2, (t2p, o, v)),
                        (K.pack_t2, (K.unpack_t2(t2p, o, v),)),
                        (K.ring_cols, (t2p, o, v, 0, o)), (K.ring_cols, (t2p, o, v, 1, 3)),
                        (K.asym_pack2, (rnd(npo, v, v), -1.1, R, 'pab')),
                        (K.asym_pack2, (rnd(v, npo, v), -1.1, R, 'apb')),
                        (K.unpack_tril, (t2p, v, 'pab')), (K.unpack_tril, (t2p, v, 'apb')),
                        (K.pack_tril_add, (rnd(npo, v, v), 0.9, R, 'pab')),
                        (K.pack_tril_add, (rnd(v, npo, v), 0.9, R, 'apb')),
                        (K.div_packed, (R, np.arange(o) - 9.0, np.arange(v) + 2.0)),
                        (K.t2p_matvec, (t2p, rnd(o, v), rnd(o, v))),
                        (K.add_rows, (R, np.array([3, -1, 0, 7, 9]), np.array([1., 0, -1, 1, -1]),
                                      rnd(5, npv), False)),
                        (K.add_rows, (R, np.array([3, -1, 0, 7, 9]), np.array([1., 0, -1, 1, -1]),
                                      rnd(5, npv), True)),
                        (K.ladder_pack, (rnd(npo, n2, n2), npo, nao, 'pxy')),
                        (K.ladder_pack, (rnd(n2, npo, n2), npo, nao, 'xpy')),
                        (K.ladder_unpack, (rng.standard_normal((6 * npo, K.npair(nao))),
                                           rng.standard_normal((2 * npo, nao * (nao + 1) // 2)),
                                           npo, nao, 'pxy')),
                        (K.ladder_unpack, (rng.standard_normal((6 * npo, K.npair(nao))),
                                           rng.standard_normal((2 * npo, nao * (nao + 1) // 2)),
                                           npo, nao, 'xpy'))):
            (a, _), (b, _) = both(f, *args)
            same(a, b)
        # pack and unpack are inverses; the ring layout of t2 is symmetric
        same(K.pack_t2(K.unpack_t2(t2p, o, v)), t2p)
        T = K.ring_cols(t2p, o, v)
        same(T, T.T)
        same(T.reshape(o, v, o, v), K.unpack_t2(t2p, o, v).transpose(0, 2, 1, 3))
        # the antisymmetrizer by blocks of columns adds up to the formula
        x = rnd(n1, n1)
        tot = np.zeros((npo, npv), dtype=complex)
        for j0, j1 in ((0, 2), (2, 3), (3, o)):
            xc = np.ascontiguousarray(x[:, j0 * v:j1 * v])
            (a, _), (b, _) = both(K.asym_pack_cols, xc, o, v, j0, j1, 0.3, R)
            same(a, b)
            K.asym_pack_cols(xc, o, v, j0, j1, 1.0, tot)
        (io, jo), (av, bv) = K.tril(o), K.tril(v)
        i, j, x4 = io[:, None], jo[:, None], x.reshape(o, v, o, v)
        same(tot, x4[i, av, j, bv] - x4[j, av, i, bv] - x4[i, bv, j, av] + x4[j, bv, i, av])
        # <mb||ef> t1: outputs in the arguments
        (_, a), (_, b) = both(K.ovvv_t1, rnd(v, npv), t1, 3, rnd(v, n1), rnd(v, v), rnd(o, v))
        same(tuple(a[3:]), tuple(b[3:]))
    finally:
        K.USE_C = True


def test_t_loops_complex(x2c_h2o, monkeypatch, tmp_path):
    monkeypatch.setattr(lib.param, 'TMPDIR', str(tmp_path))
    mycc = _run(DirectZCCSD, x2c_h2o)
    e_t = mycc.ccsd_t()                              # rebuilds ovvv, vir_loop
    eris = mycc.eris
    assert eris.ovvv is not None
    np.testing.assert_allclose(get_ovvv(eris), eris.ovvv)
    e_occ = gccsd_t.kernel(mycc, eris, mycc.t1, mycc.t2, 0, alg='occ_loop')
    e_ref = pyscf_gccsd_t.kernel(mycc, eris, mycc.t1, mycc.t2, verbose=0)
    assert abs(e_t - e_occ) < 1e-12                 # socutils' occ_loop is wrong here
    assert abs(e_t.real - np.real(e_ref)) < 1e-12 and abs(e_t.imag) < 1e-12
    # ZCCSD (integrals out of core) gives the same (T)
    z = _run(ZCCSD, x2c_h2o)
    assert abs(z.ccsd_t(alg='occ_loop') - e_t) < 1e-10


def test_t_loops_noncanonical():
    '''ROHF orbitals in spin-orbital form: f_ai != 0.'''
    mol = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='6-31g', spin=1, verbose=0)
    ghf = scf.addons.convert_to_ghf(scf.ROHF(mol).run(conv_tol=1e-11))
    gcc = cc.GCCSD(ghf, frozen=2).run(conv_tol=1e-11, conv_tol_normt=1e-8)
    eris = gcc.ao2mo()
    nocc = gcc.t1.shape[0]
    assert abs(eris.fock[nocc:, :nocc]).max() > 1e-3
    e_ref = pyscf_gccsd_t.kernel(gcc, eris, gcc.t1, gcc.t2, verbose=0)
    for alg in ('vir_loop', 'occ_loop'):
        e = gccsd_t.kernel(gcc, eris, gcc.t1, gcc.t2, 0, alg=alg)
        assert abs(e - e_ref) < 1e-11, alg
