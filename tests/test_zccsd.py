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
from zccsdt.zccsd import kernels as _kernels
from zccsdt.zccsd.zccsd_direct import get_ovvv
from zccsdt.scratch import is_mapped as scratch_is_mapped

# the C kernels (libzc_zccsd.so) are there; without them the numpy kernels do
# everything and "the C kernels" below are the numpy kernels again
HAVE_C = _kernels.USE_C


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


def test_slabs_equal_dense_within_plan(x2c_h2o, monkeypatch):
    '''zccsd.ao2mo.slabs against the integrals transformed from the 4-index
    AO array: batches of first orbitals, ranges of AO shell pairs and, within
    a range, blocks of AO pairs (the last ones shorter).  What a
    transformation allocates (tracemalloc follows numpy's allocations) stays
    within its plan, down to the least that works.'''
    import tracemalloc
    from zccsdt import blas

    def dense(mol, Ca, Cb, first, rest):
        E = mol.intor('int2e')
        f, q = slice(*first), slice(*rest)
        half = sum(np.einsum('mi,nq,mnls->iqls', C[:, f].conj(), C[:, q], E, optimize=True)
                   for C in (Ca, Cb))
        return sum(np.einsum('iqls,lr,st->iqrt', half, C[:, q].conj(), C[:, q], optimize=True)
                   for C in (Ca, Cb))

    def run(mol, Ca, Cb, first, rest, plan, shape):
        out = np.empty(shape, dtype=np.complex128)
        for i0, i1, g in ao2mo.slabs(mol, Ca, Cb, first, rest, plan):
            out[i0 - first[0]:i1 - first[0]] = g
        return out

    # the integrals, in passes of every kind
    mol = x2c_h2o.mol
    nao = mol.nao_nr()
    Ca, Cb = ao2mo.spin_components(mol, x2c_h2o.mo_coeff)
    nmo = Ca.shape[1]
    kinds = set()
    for first, rest in (((0, 10), (0, nmo)), ((3, 12), (10, nmo)), ((nmo - 6, nmo), (2, 20))):
        ref = dense(mol, Ca, Cb, first, rest)
        nfirst, nq = first[1] - first[0], rest[1] - rest[0]
        for factor, pairs in ((None, None), (None, 7), (1.0, None), (1.0, 3), (1.3, 5), (3.0, 16)):
            if pairs is not None:
                monkeypatch.setattr(ao2mo, 'BRA_BYTES', pairs * ao2mo._bra_pair(nao, nq, 1))
            need = ao2mo.plan_slabs(mol, nfirst, nq, sort_arrays=0).need
            plan = ao2mo.plan_slabs(mol, nfirst, nq, None if factor is None else int(factor * need), 0)
            assert plan.work <= plan.need * (factor or np.inf)
            assert abs(run(mol, Ca, Cb, first, rest, plan, ref.shape) - ref).max() < 1e-12
            block = ao2mo._bra_block(nao, nq, plan.nbra, plan.nkl)
            kinds.add((plan.nbra < nfirst, plan.nket < plan.nbra, plan.nkl < nao * (nao + 1) // 2,
                       block < plan.nkl, bool(plan.nkl % block)))
            monkeypatch.undo()
    assert len(kinds) >= 6 and (True, True, True, True, True) in kinds

    # the arrays of a transformation against its plan, where they are large
    # next to what the calls through ctypes leave for the garbage collector
    # (PySCF's fill, nrr_fast: pointers, a few kB for each range of AO shell
    # pairs whatever the size of the arrays; 0.22 MB at most here, where the
    # half transform of one orbital takes 2 MB)
    garbage = 1 << 19
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='cc-pvtz', verbose=0)
    mf = x2c.UHF(mol).run(conv_tol=1e-10)
    nao = mol.nao_nr()
    Ca, Cb = ao2mo.spin_components(mol, mf.mo_coeff)
    nmo = Ca.shape[1]
    first, rest = (3, 9), (40, nmo)
    nfirst, nq = first[1] - first[0], rest[1] - rest[0]
    ref = dense(mol, Ca, Cb, first, rest)
    blas.gemm(np.eye(2), np.eye(2))                 # (what its first call imports)
    blocked = 0
    for factor, pairs in ((1.0, None), (1.0, 5), (1.5, None), (2.5, 40), (None, 100), (None, None)):
        if pairs is not None:
            monkeypatch.setattr(ao2mo, 'BRA_BYTES', pairs * ao2mo._bra_pair(nao, nq, 1))
        need = ao2mo.plan_slabs(mol, nfirst, nq, sort_arrays=0).need
        plan = ao2mo.plan_slabs(mol, nfirst, nq, None if factor is None else int(factor * need), 0)
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        out = run(mol, Ca, Cb, first, rest, plan, ref.shape)
        peak = tracemalloc.get_traced_memory()[1] - base - out.nbytes
        tracemalloc.stop()
        assert abs(out - ref).max() < 1e-12
        assert peak <= plan.work + garbage, (factor, pairs, peak, plan.work)
        assert peak > 0.8 * plan.work, (factor, pairs, peak, plan.work)     # the plan is what is used
        blocked += ao2mo._bra_block(nao, nq, plan.nbra, plan.nkl) < plan.nkl
        monkeypatch.undo()
    assert blocked >= 3


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


def test_amps_equal_socutils(x2c_h2o, monkeypatch, tmp_path):
    '''The amplitude equations on packed amplitudes (zccsdt.zccsd.amps)
    against update_amps as copied from socutils, for amplitudes with a large
    T1: in one block, in blocks of columns and pairs, with the AO operators
    of the ladder kept, made from the AO integrals and read from a file, and
    with the numpy equivalents of the kernels.'''
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    from zccsdt.zccsd import kernels as K, zccsd_direct as zd
    new = DirectZCCSD(x2c_h2o, frozen=2)
    old = DirectZCCSD(x2c_h2o, frozen=2)
    old.socutils_amps = True
    new.kramers = False                             # the same spinors for both
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
    # three real GEMMs for each o^3 v^3 product instead of one ZGEMM (the
    # default from o v = amps.THREE_M_MIN on)
    assert not res.plan.three_m
    res.plan.three_m = True
    t1b, t2b = res.update(t1, t2, shift)
    assert abs(t1b - t1o).max() < 1e-12 and abs(t2b - t2o).max() < 1e-12
    from zccsdt.zccsd import amps
    assert amps.plan_step(40, 100, 70, 10 ** 12).three_m
    assert not amps.plan_step(40, 100, 70, 10 ** 12, three_m=False).three_m
    # blocks of columns of the ring terms and of pairs; the numpy kernels
    res.plan.jblocks = [(0, 1), (1, 4), (4, nocc)]
    res.plan.mblocks = [(0, 2), (2, 3), (3, nocc)]
    res.plan.pblocks = [(0, 3), (3, npo)]
    res.plan.lblocks = [(0, 5), (5, 6), (6, npo)]
    for res.plan.three_m in (False, True):
        t1b, t2b = res.update(t1, t2, shift)
        assert abs(t1b - t1o).max() < 1e-12 and abs(t2b - t2o).max() < 1e-12
    monkeypatch.setattr(K, 'USE_C', False)
    t1b, t2b = res.update(t1, t2, shift)
    assert abs(t1b - t1o).max() < 1e-12 and abs(t2b - t2o).max() < 1e-12
    monkeypatch.setattr(K, 'USE_C', HAVE_C)
    # the AO operators of the ladder: made from the AO integrals in every block
    # of pairs (units of several shells, of one shell against a few), or read
    # from a scratch file
    from zccsdt.zccsd import aoladder
    kept, floor = res.aolad, aoladder.unit_floor(new.mol)
    assert kept.mode == 'memory' and res.plan.ladder_work == 0
    assert kept.nblocks == 1 and kept.nbytes == aoladder.stored_bytes(new.mol)
    assert aoladder.stored_bytes(new.mol, 4) < 0.75 * kept.nbytes
    assert aoladder.blocks_that_fit(new.mol, kept.nbytes - 1) is None      # too few rows for parts
    for mode, work, nb in (('direct', None, 1), ('direct', 10 * floor, 1), ('direct', floor, 1),
                           ('memory', floor, 4), ('file', 2 * floor, 3)):
        res.aolad = aoladder.AOLadder(new.mol, mode, work, nb)
        assert mode == 'direct' or len(res.aolad.plus) == nb
        for use_c in (HAVE_C, False):
            monkeypatch.setattr(K, 'USE_C', use_c)
            passes = res.aolad.passes
            t1b, t2b = res.update(t1, t2, shift)
            assert abs(t1b - t1o).max() < 1e-12 and abs(t2b - t2o).max() < 1e-12
            assert res.aolad.passes == passes + len(res.plan.lblocks)
        res.aolad.close()
    monkeypatch.setattr(K, 'USE_C', HAVE_C)
    res.aolad = kept
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
    and block, the AO operators of the ladder made from the AO integrals of
    one shell against one shell at a time; the same step and the same CCSD
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
    assert res.plan.jblocks == res.plan.mblocks == [(0, o)]
    assert e_ref.aoladder.mode == 'memory' and res.plan.ladder_work == 0
    rng = np.random.default_rng(3)
    t1 = 0.05 * (rng.standard_normal((o, v)) + 1j * rng.standard_normal((o, v)))
    t2p = 0.02 * (rng.standard_normal((K.npair(o), K.npair(v)))
                  + 1j * rng.standard_normal((K.npair(o), K.npair(v))))
    # the spinors are Kramers pairs here: amplitudes with their symmetry, so that
    # the step uses it (half of the rows, completed by time reversal)
    from zccsdt.zccsd import kramers
    assert e_ref.kramers and res.kramers
    t1 = 0.5 * (t1 + kramers.image(t1))
    t2p = np.ascontiguousarray(0.5 * (t2p + kramers.image(t2p, pairs=(0, 1))))
    assert res.symmetric(t1, t2p)
    t1r, t2r = res.update_packed(t1, t2p)

    # the least memory for the integrals, then for the step
    low = 1.01 * max(ao2mo.plan_slabs(mol, o, o + v).need, 1) / 1e6
    while True:
        try:
            cc = DirectZCCSD(x2c_h2o, frozen=2)
            cc.verbose, cc.conv_tol, cc.conv_tol_normt, cc.max_memory = 0, 1e-11, 1e-8, low
            eris = cc.ao2mo()
            break
        except MemoryError:
            low *= 1.2
    # the AO operators of the ladder: kept where they fit into the share of the
    # memory for stored integrals (these are tiny), else made from the AO
    # integrals in every iteration
    assert eris.aoladder.mode == 'memory'
    monkeypatch.setattr(ao2mo, 'STORE_SHARE', 1e-6)
    eris = cc.ao2mo()
    monkeypatch.undo()
    monkeypatch.setattr(memory, 'resident', lambda: 0)
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    monkeypatch.setattr(lib.param, 'TMPDIR', str(tmp_path))
    mapped = [k for k, a in (('V', eris.oovv_r), ('E', eris.ovov_r), ('ovvv_p', eris.ovvv_p))
              if scratch_is_mapped(a)]
    assert set(mapped) == {'V', 'E', 'ovvv_p'}
    from zccsdt.zccsd import aoladder
    lad = eris.aoladder
    assert eris.kramers and lad.mode == 'direct' and lad.nbytes == 0
    units = lad.unit_range()
    assert units[0] == aoladder.unit_floor(mol) <= units[1]
    need = amps.plan_step(o, v, nao, None, mapped, kramers=True, ladder_units=units).need
    assert need > amps.plan_step(o, v, nao, None, mapped, ladder_units=units).need
    assert need >= amps.plan_step(o, v, nao, None, mapped, kramers=True).need
    plan = amps.plan_step(o, v, nao, int(1.01 * need), mapped, kramers=True, ladder_units=units)
    assert plan.w_mapped and not plan.diis_incore and plan.work <= 1.01 * need
    assert not plan.three_m and units[0] <= plan.ladder_work <= units[1]
    assert len(plan.jblocks) == o // 2 and len(plan.mblocks) == o
    with pytest.raises(MemoryError, match='three packed T2'):
        amps.plan_step(o, v, nao, int(0.9 * need), mapped, kramers=True, ladder_units=units)
    low_res = amps.Residual(eris, plan=plan)
    assert lad.work <= plan.ladder_work and len(lad.units) > 1
    assert sum(u.rows(*c)[0] for u in lad.units for c in u.chunks) == nao * (nao - 1) // 2
    for _ in range(2):                              # the mapped W is used again
        t1n, t2n = low_res.update_packed(t1, t2p)
        assert abs(t1n - t1r).max() < 1e-12 and abs(t2n - t2r).max() < 1e-12
    text = low_res.describe()
    assert 'scratch file' in text and 'AO integrals of the ladder' in text
    assert lad.passes > 2 and lad._buf is None      # once per block of pairs; nothing is kept
    low_res.close()
    # more memory: larger blocks, W in memory
    mid = amps.plan_step(o, v, nao, int(1.5 * need), mapped, kramers=True, ladder_units=units)
    assert not mid.w_mapped and 1 < len(mid.jblocks) < o
    step = amps.Residual(eris, plan=mid)
    for step.kramers in (True, False):              # with and without the symmetry
        t1n, t2n = step.update_packed(t1, t2p)
        assert abs(t1n - t1r).max() < 1e-12 and abs(t2n - t2r).max() < 1e-12
    # the whole calculation within max_memory
    cc.max_memory = max(low, 1.01 * need / 1e6)
    cc.kernel(eris=eris)
    ref.kernel(eris=e_ref)
    assert cc.converged and abs(cc.e_corr - ref.e_corr) < 1e-9
    assert eris.residual.plan.w_mapped
    assert abs(cc.t2 - ref.t2).max() < 1e-7
    # (T) drops the kept AO operators of the ladder; the next CCSD makes them again
    assert e_ref.aoladder.nbytes > 0
    e_t = ref.ccsd_t()
    assert e_ref.aoladder.nbytes == 0 and abs(cc.ccsd_t() - e_t) < 1e-10
    ref.kernel(eris=e_ref)
    assert e_ref.aoladder.nbytes > 0 and abs(cc.e_corr - ref.e_corr) < 1e-9
    cc.ao_ladder = 'disk'
    with pytest.raises(ValueError, match='ao_ladder'):
        cc.ao2mo()
    # kept operators give way when the step does not fit next to them: a mode
    # that was not asked for becomes 'direct', 'memory' that was asked for is
    # an error that says so
    floor = aoladder.unit_floor(mol)
    need = amps.plan_step(o, v, nao, None, kramers=True,
                          ladder_units=(floor, aoladder.unit_bytes(mol))).need
    for asked in (None, 'memory'):
        cc = DirectZCCSD(x2c_h2o, frozen=2)
        cc.verbose, cc.max_memory, cc.ao_ladder = 0, 4000, asked
        eris = cc.ao2mo()
        lad = eris.aoladder
        kept = lad.nbytes
        assert lad.mode == 'memory' and lad.auto == (asked is None) and kept > 0
        monkeypatch.setattr(memory, 'resident', lambda: lad.nbytes)
        tight = (need + kept // 4 + 1000) / 1e6     # MB: the step, not the step and the operators
        assert tight * 1e6 - kept < amps.plan_step(o, v, nao, None, kramers=True).need
        if asked:
            with pytest.raises(MemoryError, match="ao_ladder = 'direct'"):
                amps.Residual(eris, tight)
            assert lad.mode == 'memory' and lad.nbytes == kept
            continue
        step = amps.Residual(eris, tight)
        assert lad.mode == 'direct' and lad.nbytes == 0 and step.plan.ladder_work >= floor
        assert 'did not fit next to the operators' in step.describe()
        t1n, t2n = step.update_packed(t1, t2p)
        assert abs(t1n - t1r).max() < 1e-12 and abs(t2n - t2r).max() < 1e-12
        step.close()


def test_amps_peak_within_plan(monkeypatch):
    '''The arrays that one step allocates stay within the working set of
    its memory plan (tracemalloc follows numpy's allocations): slabs and
    blocks of several sizes, ZGEMM and three real GEMMs, with and without
    Kramers pairs, the AO operators of the ladder kept and integral-direct.
    A copy of a block that the plan does not know shows here.'''
    import tracemalloc
    from zccsdt import memory
    from zccsdt.zccsd import amps, kernels as K, kramers
    monkeypatch.setattr(memory, 'resident', lambda: 0)
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='cc-pvtz', verbose=0)
    mf = x2c.UHF(mol).run(conv_tol=1e-10)
    rng = np.random.default_rng(1)
    blocked = 0
    for mode, kr in (('memory', None), ('direct', None), ('memory', False)):
        cc = DirectZCCSD(mf)
        cc.verbose, cc.kramers, cc.ao_ladder, cc.max_memory = 0, kr, mode, 100000
        eris = cc.ao2mo()
        o, v, nao = eris.nocc, eris.fock.shape[0] - eris.nocc, mol.nao_nr()
        npo, npv = K.npair(o), K.npair(v)
        t1 = 0.05 * (rng.standard_normal((o, v)) + 1j * rng.standard_normal((o, v)))
        t2p = 0.02 * (rng.standard_normal((npo, npv)) + 1j * rng.standard_normal((npo, npv)))
        if eris.kramers:
            t1 = 0.5 * (t1 + kramers.image(t1))
            t2p = np.ascontiguousarray(0.5 * (t2p + kramers.image(t2p, pairs=(0, 1))))
        units = eris.aoladder.unit_range()
        need = amps.plan_step(o, v, nao, None, (), kramers=eris.kramers, ladder_units=units).need
        ref = None
        for factor in (1.0, 1.3, 2.0, 4.0):
            for m3 in (False, True):
                try:
                    plan = amps.plan_step(o, v, nao, int(factor * need), (), diis_incore=False,
                                          three_m=m3, kramers=eris.kramers, ladder_units=units)
                except MemoryError:                 # the real parts need more than the floor
                    assert m3 and factor == 1.0
                    continue
                res = amps.Residual(eris, plan=plan)
                res.update_packed(t1, t2p)          # what is made once is there
                tracemalloc.start()
                base = tracemalloc.get_traced_memory()[0]
                new = res.update_packed(t1, t2p)
                peak = tracemalloc.get_traced_memory()[1] - base
                tracemalloc.stop()
                res.close()
                assert peak <= plan.work, (mode, kr, factor, m3, peak, plan.work)
                blocked += len(plan.jblocks) > 1
                if ref is None:
                    ref = new
                assert abs(new[0] - ref[0]).max() < 1e-12 and abs(new[1] - ref[1]).max() < 1e-12
    assert blocked > 6


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
        K.USE_C = HAVE_C
        return out

    def same(x, y):
        x, y = (x, y) if isinstance(x, tuple) else ((x,), (y,))
        assert max(np.abs(a - b).max() for a, b in zip(x, y)) < 1e-12

    t1, u, t2p, R = rnd(o, v), rnd(o, v), rnd(npo, npv), rnd(npo, npv)
    try:
        for f, args in ((K.outer_pp, (t1, u, 0.7, R)), (K.unpack_t2, (t2p, o, v)),
                        (K.pack_t2, (K.unpack_t2(t2p, o, v),)),
                        (K.ring_cols, (t2p, o, v, 0, o)), (K.ring_cols, (t2p, o, v, 1, 3)),
                        (K.ring_cols3, (t2p, o, v, 1, 3)), (K.split3, (rnd(4, 9),)),
                        (K.split3, (rnd(4, 9)[:, 2:7],)),
                        (K.combine3, tuple(rng.standard_normal((3, 4, 5))) + (0.5, rnd(4, 5), True)),
                        (K.combine3, tuple(rng.standard_normal((3, 4, 5))) + (0.5,)),
                        (K.copy, (t2p, None, True)), (K.dotu, (t2p, R)),
                        (K.diff_norm2, (t2p, R)),
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
        # blocks of larger arrays, read in place: the planes of a slab of
        # integrals, packed, and vectors with a stride on the packed rows
        x = rnd(v + 2, npo + 1, v + 3)[1:1 + v, 1:, 2:2 + v]
        q = rnd(3, npo, 2, v)[:, :, 1, :]
        out = []
        for use in (True, False):
            K.USE_C = use
            out.append((K.asym_pack_view(x, np.empty((npo, npv), dtype=complex)),
                        K.tril_matvec(t2p, q, np.zeros((3, v), dtype=complex))))
        K.USE_C = HAVE_C
        same(*out)
        (av, bv), U = K.tril(v), K.unpack_tril(t2p, v, 'pab')
        same(out[0][0], (x[av, :, bv] - x[bv, :, av]).T)
        same(out[0][1], np.einsum('rac,frc->fa', U, q))
        # pack and unpack are inverses; the ring layout of t2 is symmetric
        same(K.pack_t2(K.unpack_t2(t2p, o, v)), t2p)
        T = K.ring_cols(t2p, o, v)
        same(T, T.T)
        A, B = rnd(4, 6), rnd(6, 5)                 # a complex product from three real ones
        same(K.combine3(*[a @ b for a, b in zip(K.split3(A), K.split3(B))]), A @ B)
        # ... from blocks of the columns of the three, into such a block
        P = rng.standard_normal((3, 4, 9))[:, :, 2:7]
        for acc in (True, False):
            C = rnd(4, 11)
            ref = C.copy()
            ref[:, 3:8] = 0.5 * ((P[0] - P[1]) + 1j * (P[2] - P[0] - P[1])) + (ref[:, 3:8] if acc else 0)
            K.combine3(P[0], P[1], P[2], 0.5, C[:, 3:8], acc)
            same(C, ref)
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
        # the fills by time reversal (pairs [unbarred, barred]); a block of columns
        from zccsdt.zccsd import kramers
        kpv, sgv = (np.ascontiguousarray(x) for x in kramers.pair_maps(v + 1))
        o6, v8 = 6, 8
        (a, _), (b, _) = both(K.kfill_ring, rnd(o6 * v8, o6 * v8), o6, v8, 0, 3, 3)
        same(a, b)
        (a, _), (b, _) = both(K.kfill_ring, rnd(o6 * v8, 4 * v8), o6, v8, 1, 3, 2)
        same(a, b)
        kp8, sg8 = (np.ascontiguousarray(x) for x in kramers.pair_maps(v8))
        L = rnd(4, K.npair(v8))
        (a, _), (b, _) = both(K.kfill_packed, L, np.array([2, -1, 0, 5]), np.array([1., 1, -1, 1]),
                              kp8, sg8, rnd(7, K.npair(v8)))
        same(a, b)
        (a, _), (b, _) = both(K.add_rows_k, rnd(7, K.npair(v8)), np.array([2, -1, 0, 5]),
                              np.array([1., 0, -1, 1]), L, np.array([3, 0, 1, 2]), kp8, sg8)
        same(a, b)
        # a packed array plus its image, in place; a symmetric array is doubled
        kp6, sg6 = kramers.pair_maps(o6)
        x = rnd(K.npair(o6), K.npair(v8))
        (a, _), (b, _) = both(K.ksym_packed, x, kp6, sg6, kp8, sg8)
        same(a, b)
        same(a, x + kramers.image(x, pairs=(0, 1)))
        same(K.ksym_packed(a.copy(), kp6, sg6, kp8, sg8), 2 * a)
        (a, _), (b, _) = both(K.asym_pack_cols, rnd(n1, 3 * v + 5), o, v, 1, 3, 0.3, R, 4)
        same(a, b)
        # <mb||ef> t1: outputs in the arguments
        (_, a), (_, b) = both(K.ovvv_t1, rnd(v, npv), t1, 3, rnd(v, n1), rnd(v, v), rnd(o, v))
        same(tuple(a[3:]), tuple(b[3:]))
    finally:
        K.USE_C = HAVE_C


def test_t_complex(x2c_h2o, monkeypatch, tmp_path):
    '''(T) in batches of virtual pairs (zccsdt.zccsd.triples) against the
    loops copied from socutils and PySCF's gccsd_t, for complex integrals:
    in one batch per virtual, in the smallest batches, with the numpy
    kernel, and with <bc||ei> in a scratch file.'''
    from zccsdt import memory
    from zccsdt.zccsd import kernels as K, triples
    monkeypatch.setattr(lib.param, 'TMPDIR', str(tmp_path))
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    mycc = _run(DirectZCCSD, x2c_h2o)
    e_t = mycc.ccsd_t()                              # triples.kernel
    eris = mycc.eris
    assert eris.ovvv is None
    eris.ovvv = get_ovvv(eris)                       # for the loops on the full arrays
    e_vir = gccsd_t.kernel(mycc, eris, mycc.t1, mycc.t2, 0)
    e_occ = gccsd_t.kernel(mycc, eris, mycc.t1, mycc.t2, 0, alg='occ_loop')
    e_ref = pyscf_gccsd_t.kernel(mycc, eris, mycc.t1, mycc.t2, verbose=0)
    assert abs(e_vir - e_occ) < 1e-12               # socutils' occ_loop is wrong here
    assert abs(e_vir.real - np.real(e_ref)) < 1e-12 and abs(e_vir.imag) < 1e-12
    assert isinstance(e_t, float) and abs(e_t - e_vir.real) < 1e-12
    eris.ovvv = None
    o, v = mycc.t1.shape
    floor = triples.memory_need(o, v, False)[1]
    io, jo = np.tril_indices(o, -1)
    bv, cv = np.tril_indices(v, -1)
    t2p = np.ascontiguousarray(mycc.t2[io, jo][:, bv, cv])
    # one ZGEMM per product instead of three real GEMMs
    assert abs(triples.kernel(mycc, eris, mycc.t1, mycc.t2, 0, three_m=False) - e_t) < 1e-12
    # ranges of a few a with all b < a
    monkeypatch.setattr(triples, 'BATCH_BYTES', 2 * floor * v)
    assert 3 < triples._a_range(2, v, o, 2 * floor * v) < v
    for m3 in (True, False):
        assert abs(triples.kernel(mycc, eris, mycc.t1, mycc.t2, 0, three_m=m3) - e_t) < 1e-12
    # the smallest batches, packed t2; the numpy kernel
    monkeypatch.setattr(triples, 'BATCH_BYTES', floor)
    assert triples._a_range(v - 1, v, o, floor) == v - 1
    assert len(triples._batches(v - 1, o, floor)) > 2
    for m3 in (True, False):
        assert abs(triples.kernel(mycc, eris, mycc.t1, t2p, 0, three_m=m3) - e_t) < 1e-12
    monkeypatch.setattr(K, 'USE_C', False)
    for m3 in (True, False):
        assert abs(triples.kernel(mycc, eris, mycc.t1, mycc.t2, 0, three_m=m3) - e_t) < 1e-12
    monkeypatch.setattr(K, 'USE_C', HAVE_C)
    # the least memory: <bc||ei> and the layouts of t2 in a scratch file, written
    # and read in blocks; below it, MemoryError
    monkeypatch.setattr(memory, 'resident', lambda: 0)
    monkeypatch.setattr(triples, 'BATCH_BYTES', 4 << 30)
    for m3 in (True, False):
        need = sum(triples.memory_need(o, v, m3))
        for use_c in (True, False):
            monkeypatch.setattr(K, 'USE_C', use_c and HAVE_C)
            assert abs(triples.kernel(mycc, eris, mycc.t1, mycc.t2, 0, three_m=m3,
                                      max_memory=1.01 * need / 1e6) - e_t) < 1e-12
        monkeypatch.setattr(K, 'USE_C', HAVE_C)
        with pytest.raises(MemoryError, match='one virtual orbital'):
            triples.kernel(mycc, eris, mycc.t1, mycc.t2, 0, three_m=m3,
                           max_memory=0.9 * need / 1e6)
    monkeypatch.undo()
    monkeypatch.setattr(lib.param, 'TMPDIR', str(tmp_path))
    # ZCCSD (integrals out of core, the copied loops) gives the same (T)
    z = _run(ZCCSD, x2c_h2o)
    assert abs(z.ccsd_t(alg='occ_loop') - e_t) < 1e-10
    assert abs(triples.kernel(z, z.eris, z.t1, z.t2, 0) - e_t) < 1e-10


def test_t_noncanonical():
    '''ROHF orbitals in spin-orbital form: f_ai != 0 (real integrals of
    PySCF's GCCSD).'''
    from zccsdt.zccsd import triples
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
    assert abs(triples.kernel(gcc, eris, gcc.t1, gcc.t2, 0) - e_ref) < 1e-11


def test_kramers_pairs():
    '''Kramers pairs from a time-reversal symmetric reference (also with
    levels that are degenerate by spatial symmetry); CCSD and (T) on them
    against the spinors of the SCF; (T) on half of the virtual triples.'''
    from zccsdt.zccsd import kramers, triples
    ch4 = gto.M(atom='C 0 0 0; H 0.629 0.629 0.629; H -0.629 -0.629 0.629; '
                     'H -0.629 0.629 -0.629; H 0.629 -0.629 -0.629', basis='6-31g', verbose=0)
    mf = x2c.UHF(ch4).run(conv_tol=1e-12)
    nocc, e = ch4.nelectron, mf.mo_energy
    # the t2 shell splits into a pair and a four-fold level of the double group
    assert np.sum(abs(np.diff(e[:nocc])) < 1e-7) == nocc // 2 + 1
    c = np.vstack(ch4.sph2spinor_coeff()) @ mf.mo_coeff
    s2 = np.kron(np.eye(2), ch4.intor('int1e_ovlp'))
    for lo, hi in ((0, nocc), (nocc, e.size)):
        cp, order = kramers.pair(c[:, lo:hi], s2, e[lo:hi])
        h = (hi - lo) // 2
        assert abs(cp.conj().T @ s2 @ cp - np.eye(hi - lo)).max() < 1e-12
        assert abs(cp[:, h:] - kramers.time_reverse(cp[:, :h])).max() == 0
        u = c[:, lo:hi].conj().T @ s2 @ cp                      # within the levels
        assert abs(u.conj().T @ u - np.eye(hi - lo)).max() < 1e-10
        f = u.conj().T @ np.diag(e[lo:hi]) @ u
        assert abs(f - np.diag(e[lo:hi][order])).max() < 1e-9
    # an odd number of spinors, a reference without the symmetry
    assert kramers.pair(c[:, :3], s2, e[:3]) is None
    oh = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='6-31g', spin=1, verbose=0)
    mfo = x2c.UHF(oh).run(conv_tol=1e-11)
    co = np.vstack(oh.sph2spinor_coeff()) @ mfo.mo_coeff
    assert kramers.pair(co[:, oh.nelectron:], np.kron(np.eye(2), oh.intor('int1e_ovlp')),
                        mfo.mo_energy[oh.nelectron:]) is None

    res = {}
    for kr in (False, None):
        mycc = DirectZCCSD(mf, frozen=2)
        mycc.kramers = kr
        mycc.verbose, mycc.conv_tol, mycc.conv_tol_normt = 0, 1e-11, 1e-9
        mycc.kernel()
        assert mycc.converged and mycc.eris.kramers is (kr is None)
        res[kr] = mycc, mycc.ccsd_t()
    (plain, et0), (paired, et1) = res[False], res[None]
    assert abs(plain.e_corr - paired.e_corr) < 1e-9 and abs(et0 - et1) < 1e-10
    # the amplitudes obey t[Kp Kq, Kr Ks] = +- conj(t[pq, rs]); the Fock matrix is diagonal
    assert kramers.symmetry_error(paired.t1) < 1e-8 and kramers.symmetry_error(paired.t2) < 1e-8
    assert kramers.symmetry_error(plain.t2) > 1e-4
    f = paired.eris.fock
    assert abs(f - np.diag(f.diagonal())).max() < 1e-7
    # the CCSD step: half of the expensive terms completed by time reversal
    # against the step that computes all of them, for symmetric amplitudes
    from zccsdt.zccsd import amps, kernels as K
    o, v = paired.t1.shape
    npo, npv = K.npair(o), K.npair(v)
    rng = np.random.default_rng(5)
    t1 = 0.05 * (rng.standard_normal((o, v)) + 1j * rng.standard_normal((o, v)))
    t1 = 0.5 * (t1 + kramers.image(t1))
    t2p = 0.02 * (rng.standard_normal((npo, npv)) + 1j * rng.standard_normal((npo, npv)))
    t2p = np.ascontiguousarray(0.5 * (t2p + kramers.image(t2p, pairs=(0, 1))))
    step = amps.Residual(paired.eris, paired.max_memory)
    assert step.kramers and not amps.Residual(plain.eris, plain.max_memory).kramers
    step.kramers = False
    t1r, t2r = step.update_packed(t1, t2p)
    assert kramers.symmetry_error(t1r) < 1e-8
    assert abs(kramers.image(t2r, pairs=(0, 1)) - t2r).max() < 1e-8
    step.kramers = True
    blocks = dict(mblocks=[(0, 1), (1, 3), (3, o)], jblocks=[(0, 3), (3, 4), (4, o)],
                  lblocks=[(0, 2), (2, 7), (7, npo)], pblocks=[(0, 5), (5, npo)])
    for m3 in (False, True):
        step.plan.three_m = m3
        t1k, t2k = step.update_packed(t1, t2p)
        assert abs(t1k - t1r).max() < 1e-12 and abs(t2k - t2r).max() < 1e-12
    whole = {k: getattr(step.plan, k) for k in blocks}
    for k, b in blocks.items():
        setattr(step.plan, k, b)
    for m3, use_c in ((False, True), (True, True), (True, False)):
        step.plan.three_m, K.USE_C = m3, use_c and HAVE_C
        try:
            t1k, t2k = step.update_packed(t1, t2p)
        finally:
            K.USE_C = HAVE_C
        assert abs(t1k - t1r).max() < 1e-12 and abs(t2k - t2r).max() < 1e-12
    for k, b in whole.items():
        setattr(step.plan, k, b)
    # (T): half of the virtual triples, doubled, against all of them
    for m3 in (True, False):
        full = triples.kernel(paired, paired.eris, verbose=0, three_m=m3, kramers=False)
        assert abs(full - et1) < 1e-12
        for nbytes in (triples.memory_need(o, v, m3)[1], 2 * triples.memory_need(o, v, m3)[1] * v):
            triples.BATCH_BYTES, old = nbytes, triples.BATCH_BYTES
            try:
                assert abs(triples.kernel(paired, paired.eris, verbose=0, three_m=m3) - et1) < 1e-12
            finally:
                triples.BATCH_BYTES = old
    # amplitudes without the symmetry: all triples, unless told otherwise
    assert kramers.symmetric(paired.t1, paired.t2) and kramers.symmetric(paired.t1, K.pack_t2(
        np.ascontiguousarray(paired.t2)))
    t2a = paired.t2 + 3e-2 * rng.standard_normal(paired.t2.shape)
    assert not kramers.symmetric(paired.t1, t2a)
    full = triples.kernel(paired, paired.eris, paired.t1, t2a, 0, kramers=False)
    assert abs(triples.kernel(paired, paired.eris, paired.t1, t2a, 0) - full) < 1e-12 * abs(full)
    # (the pairs are rotated within their levels as the SCF left them, so the
    # error of the half sum varies from run to run: only its presence is tested)
    assert abs(triples.kernel(paired, paired.eris, paired.t1, t2a, 0, kramers=True) - full) > 1e-8
    # kramers = True insists on the symmetry
    bad = DirectZCCSD(mfo)
    bad.kramers, bad.verbose = True, 0
    with pytest.raises(RuntimeError, match='time reversal'):
        bad.ao2mo()
    bad.kramers = None
    assert bad.ao2mo().kramers is False


def test_callback_keeps_no_amplitudes(x2c_h2o, monkeypatch, tmp_path):
    '''A callback of the CCSD iterations must not keep packed amplitudes
    alive through the next step (callback(locals()) did: the frame held the
    old and the new ones, two packed arrays that the memory plan of the step
    does not know).  With the DIIS vectors in a scratch file the amplitudes
    that a step returns are dead when the next step starts.'''
    import gc
    import weakref
    from zccsdt.zccsd import amps
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    monkeypatch.setattr(lib.param, 'TMPDIR', str(tmp_path))
    mycc = DirectZCCSD(x2c_h2o, frozen=2)
    mycc.verbose, mycc.conv_tol, mycc.conv_tol_normt = 0, 1e-9, 1e-7
    made, alive, seen = [], [], []
    update = amps.Residual.update_packed
    diis = amps.DIIS
    # all parts of the vectors in the file, the small ones as well
    monkeypatch.setattr(amps, 'DIIS', lambda space, filename, incore, incore_size=None: diis(
        space, filename, False, 0))

    def watched(self, t1, t2p, level_shift=0.0):
        gc.collect()
        alive.append(sum(r() is not None for r in made))
        out = update(self, t1, t2p, level_shift)
        made.append(weakref.ref(out[1]))
        return out

    monkeypatch.setattr(amps.Residual, 'update_packed', watched)
    mycc.callback = lambda envs: seen.append(sorted(envs))
    mycc.kernel()
    # the first DIIS call hands the first amplitudes back as they are: they are
    # the input of the second step
    assert mycc.converged and len(alive) > 5 and alive[1] <= 1 and max(alive[2:]) == 0
    assert {'t1new', 't2pnew', 'istep', 'eccsd', 'mycc', 'eris'} <= set(seen[0])


def test_ladder_operators_made_again_within_memory(monkeypatch):
    '''(T) drops the AO operators of the ladder; the next plan of the step
    (Lambda after (T), another CCSD) makes them again - if they still fit
    into what max_memory leaves then, with a unit of their integrals next to
    them, and in units sized for that.  Allocations (tracemalloc) within it;
    the same operators; with no room they give way to the mode 'direct'
    before anything is allocated, or MemoryError if 'memory' was asked for.'''
    import tracemalloc
    from zccsdt import memory
    from zccsdt.zccsd import amps, aoladder
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='cc-pvtz', verbose=0)
    mf = x2c.UHF(mol).run(conv_tol=1e-9)
    cc = DirectZCCSD(mf)
    cc.verbose, cc.max_memory = 0, 100000
    eris = cc.ao2mo()
    res = amps.Residual(eris, plan=False)
    lad = eris.aoladder
    assert lad.mode == 'memory' and lad.auto and lad.nblocks == 1
    ref = [T.copy() for _, T, _ in lad.same + lad.plus]
    kept, floor, first = lad.nbytes, aoladder.unit_floor(mol), lad._work
    assert kept == aoladder.stored_bytes(mol) and kept > 10 * floor and first > 100 * floor
    monkeypatch.setattr(memory, 'resident', lambda: 0)

    def peak_of(f):
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        f()
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        return peak
    budget = kept + 3 * floor
    lad.close()
    assert lad.nbytes == 0
    peak = peak_of(lambda: res._keep_ladder(budget / 1e6))
    assert lad.mode == 'memory' and lad.nbytes == kept and kept < peak <= budget
    assert lad._work <= 1.5 * floor + 1 and not res.gave_way
    for x, y in zip(ref, [T for _, T, _ in lad.same + lad.plus]):
        assert abs(x - y).max() < 1e-12
    # no room for a unit next to them: nothing is allocated
    lad.close()
    assert peak_of(lambda: res._keep_ladder((kept + 0.5 * floor) / 1e6)) < 0.1 * kept
    assert lad.mode == 'direct' and res.gave_way and lad.nbytes == 0
    asked = res.aolad = aoladder.AOLadder(mol, 'memory', first, 1, auto=False)
    asked.close()
    with pytest.raises(MemoryError, match='cannot be made again'):
        res._keep_ladder((kept + 0.5 * floor) / 1e6)
    res._keep_ladder(None)                  # no limit known: as they were made first
    assert asked.nbytes == kept


def test_diis_vectors_where_the_plan_puts_them(monkeypatch, tmp_path):
    '''When the memory plan of the step puts the DIIS vectors into a scratch
    file, the packed T2 of every vector is there: the DIIS of zccsdt.cc keeps
    parts below 1e7 elements in memory by default (a packed T2 of 7.5 million
    for benzene/cc-pVDZ with a frozen core: 1.4 GB in twelve vectors that the
    plan had counted as none), and the CCSD and Lambda iterations pass their
    own threshold, amps.DIIS_SMALL.  Only t1 stays in memory.'''
    from zccsdt import memory
    from zccsdt.cc.diis import DIIS
    from zccsdt.zccsd import amps, lam
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    monkeypatch.setattr(lib.param, 'TMPDIR', str(tmp_path))
    rng = np.random.default_rng(0)
    default, small = DIIS(space=2, incore=False), DIIS(2, None, False, amps.DIIS_SMALL)
    for k in range(3):                      # the first vector has no error vector yet
        x = [rng.standard_normal(n) + 1j * rng.standard_normal(n) for n in (100, amps.DIIS_SMALL)]
        for d in (default, small):
            d.update([a.copy() for a in x])
    assert not default.on_disk() and small.on_disk()
    assert [scratch_is_mapped(a) for a in small.es[-1]] == [False, True]
    assert [scratch_is_mapped(a) for a in default.es[-1]] == [False, False]
    default.close()
    small.close()
    # the iterations of the CCSD and of Lambda under a plan with the vectors in the file
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='cc-pvtz', verbose=0)
    cc = DirectZCCSD(x2c.UHF(mol).run(conv_tol=1e-9))
    cc.verbose, cc.max_cycle = 0, 3
    eris = cc.ao2mo()
    o, v, nao = eris.nocc, eris.fock.shape[0] - eris.nocc, mol.nao_nr()
    assert amps.DIIS_SMALL < o * (o - 1) // 2 * (v * (v - 1) // 2) < 1e7
    need = amps.plan_step(o, v, nao, None, kramers=True).need
    resident = memory.resident()
    monkeypatch.setattr(memory, 'resident', lambda: resident)
    cc.max_memory = (resident + 1.2 * need + lam.reserve_bytes(o, v)) / 1e6
    made = []
    diis = amps.DIIS

    def watched(*args, **kwargs):
        made.append(diis(*args, **kwargs))
        return made[-1]
    monkeypatch.setattr(amps, 'DIIS', watched)
    monkeypatch.setattr(lam, 'DIIS', watched)
    seen = []
    cc.callback = lambda envs: seen.append(envs['adiis'].on_disk())
    cc.kernel(eris=eris)
    assert not eris.residual.plan.diis_incore and seen[-1] and len(made) == 1
    cc.solve_lambda(eris=eris)
    assert len(made) == 2 and all(d.incore_size == amps.DIIS_SMALL and not d.incore for d in made)
