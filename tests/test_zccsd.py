'''The spinor CCSD and (T) copied from socutils (zccsdt.zccsd): the same
energies as zccsdt's CCSD and PySCF's GCCSD(T), both (T) loops alike.'''

import numpy as np
import pytest
from pyscf import cc, gto, lib, scf, x2c
from pyscf.cc import gccsd_t as pyscf_gccsd_t

from zccsdt import eris as E
from zccsdt.cc import solver
from zccsdt.zccsd import DirectZCCSD, ZCCSD, gccsd_t
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
