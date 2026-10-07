'''The MO integrals of DIRAC as an integral container (zccsdt.dirac): the
files MRCONEE and MDCINT of an X2C calculation on BH in tests/data/dirac -
with C2v, real integrals, and without symmetry, complex ones - against
DIRAC's own SCF and CCSD and against the relativistic CCSDT, CCSDT(Q) and
CCSDTQ of MRCC on the same integrals (bench/refs: bh_sto2g_x2c_dirac).'''
import json
import lzma
import os

import numpy as np
import pytest

from zccsdt import dirac
from zccsdt.cc import ccsdtq, quadruples, solver

HERE = os.path.dirname(__file__)
REF = json.load(open(os.path.join(HERE, '..', 'bench', 'refs', 'reference_energies.json')))['bh_sto2g_x2c_dirac']


@pytest.fixture(scope='module')
def files(tmp_path_factory):
    '''{symmetry: (MRCONEE, MDCINT)} unpacked.'''
    out = {}
    for sym in ('c2v', 'c1'):
        d = tmp_path_factory.mktemp(sym)
        for name in ('MRCONEE', 'MDCINT'):
            with lzma.open(os.path.join(HERE, 'data', 'dirac', 'bh_sto2g_' + sym, name + '.xz')) as f:
                open(d / name, 'wb').write(f.read())
        out[sym] = (str(d / 'MRCONEE'), str(d / 'MDCINT'))
    return out


@pytest.mark.parametrize('sym', ['c2v', 'c1'])
def test_read(files, sym):
    '''What is in the files: 24 spinors in 12 Kramers pairs, 6 electrons;
    the integrals real for C2v and complex without symmetry; the list of
    the file is not complete under permutations, the completed one is; the
    SCF energy and the spinor energies of DIRAC from the integrals.'''
    x = dirac.read(*files[sym])
    n = len(x.energy)
    assert n == 24 and x.kramers.shape == (12, 2) and x.nelec.sum() == 6
    assert x.group_type == {'c2v': 1, 'c1': 4}[sym]
    assert np.iscomplexobj(x.g) == (sym == 'c1') and np.iscomplexobj(x.h) == (sym == 'c1')
    if sym == 'c1':
        assert abs(x.g.imag).max() > 1e-3
    assert sorted(x.kramers.ravel()) == list(range(n))
    g = x.g
    assert abs(g - g.transpose(2, 3, 0, 1)).max() < 1e-12
    assert abs(g - g.transpose(1, 0, 3, 2).conj()).max() < 1e-12
    raw = dirac.read(*files[sym], fill=False).g
    assert abs(raw - raw.transpose(2, 3, 0, 1)).max() > 0.1         # (one of the partners only)
    assert np.count_nonzero(raw) < 0.6 * np.count_nonzero(g)
    eris = dirac.from_dirac(*files[sym])
    assert eris.nocc == 6 and eris.nmo == 24
    assert abs(eris.e_hf - REF['dirac']['e_scf']) < 1e-10
    assert abs(eris.fock - np.diag(eris.fock.diagonal())).max() < 1e-9
    assert abs(eris.mo_energy - x.energy[eris.spinors]).max() < 1e-9
    assert eris.dtype == (np.complex128 if sym == 'c1' else np.float64)


def test_complete_list_written(files, tmp_path):
    '''write_mdcint: the complete list in DIRAC's records - read back
    without filling in anything it is the same array (real and complex).'''
    for sym in ('c2v', 'c1'):
        x = dirac.read(*files[sym])
        path = str(tmp_path / ('MDCINT_' + sym))
        n = dirac.write_mdcint(path, x)
        y = dirac.read(files[sym][0], path, fill=False)
        assert abs(y.g - x.g).max() < 1e-14
        assert n == np.count_nonzero(abs(x.g[x.kramers[:, 0]]) > 1e-15)


@pytest.mark.parametrize('sym', ['c2v', 'c1'])
def test_cc_on_dirac_integrals(files, sym):
    '''CCSD of DIRAC's RELCCSD; CCSDT, CCSDT[Q], CCSDT(Q) and CCSDTQ of MRCC
    (its interface to DIRAC, same integrals, the reference flagged as
    Hartree-Fock) - with the real integrals of C2v and with the complex
    ones of the calculation without symmetry.'''
    eris = dirac.from_dirac(*files[sym])
    ref = REF['recommended']
    opts = solver.Options(conv_tol=1e-12, conv_tol_normt=1e-9, max_cycle=200)
    rd = solver.kernel(eris, 'ccsd', opts)
    assert rd.converged and abs(rd.e_tot - REF['dirac']['e_ccsd']) < 1e-9 and abs(rd.e_tot - ref['e_ccsd']) < 1e-10
    rt = solver.kernel(eris, 'ccsdt', opts)
    assert rt.converged and abs(rt.e_tot - ref['e_ccsdt']) < 1e-10
    assert np.asarray(rt.t3).dtype == eris.dtype
    e5, e6 = quadruples.kernel(eris, rt.t2, rt.t3)
    assert abs(rt.e_tot + e5 - ref['e_ccsdt[q]']) < 1e-10
    assert abs(rt.e_tot + e6 - ref['e_ccsdt(q)']) < 1e-10
    # (not the variants that MRCC gives for a reference that is not Hartree-Fock)
    assert abs(rt.e_tot + e6 - REF['mrcc']['e_ccsdt(q)/A']) > 1e-7
    rq = ccsdtq.kernel(eris, ccsdtq.Options(conv_tol=1e-12, conv_tol_normt=1e-9), t1=rt.t1, t2=rt.t2, t3=rt.t3)
    assert rq.converged and abs(rq.e_tot - ref['e_ccsdtq']) < 1e-10
    assert rq.t4.dtype == eris.dtype
