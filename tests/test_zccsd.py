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
    for cls, switch in ((DirectZCCSD, 'ovvv_direct'), (ZCCSD, 'ao2mo_outcore')):
        mycc = cls(x2c_h2o, frozen=2)
        setattr(mycc, switch, True)
        mycc.verbose, mycc.conv_tol, mycc.conv_tol_normt = 0, 1e-11, 1e-8
        mycc.kernel()
        assert mycc.converged and abs(mycc.e_corr - e_ref) < 1e-9, switch
        if switch == 'ovvv_direct':
            assert mycc.eris.ovvv is None and mycc.eris.E4 is not None


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
