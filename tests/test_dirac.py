'''The MO integrals of DIRAC as an integral container (zccsdt.dirac): the
files MRCONEE and MDCINT of an X2C calculation on BH in tests/data/dirac -
with C2v, real integrals, and without symmetry, complex ones - against
DIRAC's own SCF and CCSD and against the relativistic CCSDT, CCSDT(Q) and
CCSDTQ of MRCC on the same integrals (bench/refs: bh_sto2g_x2c_dirac).

The other direction: the spinors of PySCF in Kramers pairs, their integrals
written in DIRAC's format and read back, and the energies of MRCC on the
files written that way (HBr with the X2C Hamiltonian: hbr_ccpvdzdk_x2c_mdcint).'''
import io
import json
import lzma
import os

import numpy as np
import pytest
from pyscf import gto, scf, x2c

import zccsdt
from zccsdt import dirac, eris as E
from zccsdt.cc import ccsdtq, quadruples, solver

HERE = os.path.dirname(__file__)
REFS = json.load(open(os.path.join(HERE, '..', 'bench', 'refs', 'reference_energies.json')))
REF = REFS['bh_sto2g_x2c_dirac']


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


def test_mrconee_as_dirac_writes_it(files, tmp_path):
    '''write_mrconee: the file of DIRAC byte for byte (the real group); the
    other kinds of groups are not written.'''
    path = str(tmp_path / 'MRCONEE')
    dirac.write_mrconee(path, dirac.read(*files['c2v']))
    assert open(path, 'rb').read() == open(files['c2v'][0], 'rb').read()
    with pytest.raises(NotImplementedError):
        dirac.write_mrconee(path, dirac.read(*files['c1']))


@pytest.fixture(scope='module')
def hbr():
    mol = gto.M(atom='Br 0 0 0; H 0 0 1.4145', basis='cc-pvdz-dk', verbose=0)
    mf = x2c.UHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-8
    mf.kernel()
    return mf


def test_kramers_pairs(hbr):
    '''The X2C spinors of HBr on the z axis in pairs (psi, T psi): real,
    T^2 = -1, orthonormal, the Fock matrix diagonal with the spinor
    energies, the unbarred ones of m_j = 1/2, -3/2, 5/2 and never coupled
    to a barred one; the 4p shell split by spin-orbit coupling.'''
    Cu, Cb, e = dirac.kramers_pairs(hbr)
    n = len(hbr.mo_energy)
    assert Cu.shape == (n, n // 2) and not np.iscomplexobj(Cu) and not np.iscomplexobj(Cb)
    assert abs(dirac.time_reversed(hbr, Cb) + Cu).max() == 0
    S, F = hbr.get_ovlp(), hbr.get_fock()
    C = np.hstack((Cu, Cb))
    assert abs(C.T @ S @ C - np.eye(n)).max() < 1e-12
    f = C.T @ F @ C
    assert abs(f - np.diag(np.concatenate((e, e)))).max() < 1e-7
    assert abs(np.sort(np.concatenate((e, e))) - np.sort(hbr.mo_energy)).max() < 1e-8
    mj = np.einsum('mp,mn,n,np->p', Cu, S, dirac._jz(hbr), Cu).real
    assert set(np.round(mj, 6)) <= {0.5, -1.5, 2.5} and len(set(np.round(mj, 6))) == 3
    assert abs(Cu.T @ S @ Cb).max() == 0
    assert 0.010 < e[17] - e[16] < 0.016                    # (4p pi: 13 mEh)


@pytest.mark.parametrize('kind,frozen', [('ghf', 0), ('ghf', 2), ('x2c', 0), ('x2c', 2)])
def test_own_integrals_written_and_read(tmp_path, kind, frozen):
    '''integrals(): spin orbitals of RHF (H2O) and X2C spinors of a linear
    molecule (BH), all active and with a frozen pair.  Real; written, they
    are read back as they were (the barred half is made by the reader from
    the relation of the pairs); the container has the SCF energy and gives
    the CCSD energy of the integrals of the SCF object.'''
    if kind == 'ghf':
        mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='sto-3g', verbose=0)
        mf = scf.addons.convert_to_ghf(scf.RHF(mol).run(conv_tol=1e-12))
    else:
        mf = x2c.UHF(gto.M(atom='B 0 0 0; H 0 0 1.2324', basis='sto-3g', verbose=0)).run(conv_tol=1e-13)
    x = dirac.integrals(mf, frozen)
    n = len(mf.mo_energy) - frozen
    assert x.g.shape == (n,) * 4 and x.g.dtype == np.float64 and x.group_type == 1
    assert list(x.nelec) == [(mf.mol.nelectron - frozen) // 2] * 2
    count = dirac.write(str(tmp_path), x)
    assert count == np.count_nonzero(abs(x.g[:n // 2]) > 1e-15)
    y = dirac.read(str(tmp_path / 'MRCONEE'), str(tmp_path / 'MDCINT'), fill=False)
    assert abs(y.g - x.g).max() < 1e-14 and abs(y.h - x.h).max() == 0 and y.e_core == x.e_core
    assert abs(y.e_scf - mf.e_tot) < 1e-12 and list(y.nelec) == list(x.nelec)
    eris = dirac.from_dirac(str(tmp_path / 'MRCONEE'), str(tmp_path / 'MDCINT'))
    ref = E.from_scf(mf, frozen=frozen, packed_vvvv=True)
    assert eris.nocc == ref.nocc and abs(eris.e_hf - mf.e_tot) < 1e-10
    assert abs(eris.fock - np.diag(eris.fock.diagonal())).max() < 1e-7
    opts = solver.Options(conv_tol=1e-12, conv_tol_normt=1e-9)
    assert abs(solver.kernel(eris, 'ccsd', opts).e_tot - solver.kernel(ref, 'ccsd', opts).e_tot) < 1e-10
    if frozen:
        with pytest.raises(ValueError):
            dirac.integrals(mf, 1)


def test_what_cannot_be_paired():
    '''A bent molecule with X2C spinors has no real integrals: refused, and
    so is an open shell.'''
    mf = x2c.UHF(gto.M(atom='C 0 0 0; H 0 0.87 0.67; H 0 -0.87 0.67', basis='sto-3g', verbose=0)).run(conv_tol=1e-12)
    with pytest.raises(ValueError, match='linear molecule'):
        dirac.kramers_pairs(mf)
    mf = scf.addons.convert_to_ghf(scf.UHF(gto.M(atom='O 0 0 0; H 0 0 1', basis='sto-3g', spin=1, verbose=0)).run())
    with pytest.raises(ValueError):
        dirac.integrals(mf)


def test_hbr_spinors_against_mrcc_on_the_same_integrals(hbr):
    '''HBr / cc-pVDZ-DK, X2C spinors, 28 frozen, 8 + 16 active: CCSDT,
    CCSDT[Q], CCSDT(Q), CCSDTQ and CCSDTQ(P) through the front end on the
    integrals in Kramers pairs, against MRCC on the files written from
    them.'''
    ref = REFS['hbr_ccpvdzdk_x2c_mdcint']['recommended']
    x = dirac.integrals(hbr, frozen=28, nvir=16)
    assert len(x.energy) == 24 and x.g.dtype == np.float64
    eris = dirac.container(x)
    assert eris.nocc == 8 and abs(eris.e_hf - ref['e_scf']) < 1e-9
    buf = io.StringIO()
    cc = zccsdt.GCCSDTQ.from_integrals(eris, verbose=4, stdout=buf)
    cc.conv_tol, cc.conv_tol_normt = 1e-11, 1e-9
    cc.kernel()
    e_q = cc.ccsdt_q()
    out = buf.getvalue()
    assert 'integrals given (spinors in Kramers pairs)' in out and '8 occupied, 16 virtual spinors' in out
    corr = lambda key: ref[key] - ref['e_scf']
    assert cc.converged and cc.ccsdt_run.converged
    assert abs(cc.ccsdt_run.e_corr - corr('e_ccsdt')) < 2e-10
    assert abs(cc.ccsdt_run.e_corr + cc.ccsdt_run.e_q5 - corr('e_ccsdt[q]')) < 2e-10
    assert abs(cc.ccsdt_run.e_corr + e_q - corr('e_ccsdt(q)')) < 2e-10
    assert abs(cc.e_corr - corr('e_ccsdtq')) < 2e-10
    assert abs(cc.e_tot - ref['e_ccsdtq']) < 1e-9 and abs(cc.e_ccsdt - ref['e_ccsdt']) < 1e-9
    e_p = cc.ccsdtq_p()
    assert abs(e_p - (ref['e_ccsdtq(p)'] - ref['e_ccsdtq'])) < 3e-12 and abs(e_p) > 1e-6
    assert abs(cc.e_tot_p - ref['e_ccsdtq(p)']) < 1e-9
    assert '(P) correction: 56 occupied x 4,368 virtual pentuples, real' in buf.getvalue()
    with pytest.raises(RuntimeError):
        cc.set_frozen()


def test_hbr_turned_off_the_axis():
    '''The same molecule turned away from the z axis: complex spinors and
    integrals (no m_j to adapt to), complex T3 and T4 - and the correlation
    energies of the molecule on the axis, which are those of MRCC: CCSDT,
    CCSDT[Q], CCSDT(Q), CCSDTQ, CCSDTQ(P).'''
    ref = REFS['hbr_ccpvdzdk_x2c_mdcint']['recommended']
    a, b = np.radians(37.0), np.radians(61.0)
    h = 1.4145 * np.array([np.cos(a) * np.sin(b), np.sin(a) * np.sin(b), np.cos(b)])
    mf = x2c.UHF(gto.M(atom=[['Br', (0, 0, 0)], ['H', tuple(h)]], basis='cc-pvdz-dk', verbose=0))
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-8
    mf.kernel()
    assert abs(mf.e_tot - ref['e_scf']) < 1e-9
    with pytest.raises(ValueError):
        dirac.kramers_pairs(mf)
    full = E.from_scf(mf, frozen=28, packed_vvvv=True)
    eris = E.drop_virtuals(full, 16)
    assert eris.nmo == 24 and eris._vvvv_pp.shape == (120, 120) and E.drop_virtuals(full, 28) is full
    assert abs(np.asarray(eris.oovv).imag).max() > 0.01 and abs(eris._vvvv_pp.imag).max() > 0.005
    cc = zccsdt.GCCSDTQ.from_integrals(eris, verbose=0)
    cc.conv_tol, cc.conv_tol_normt = 1e-11, 1e-9
    cc.kernel()
    e_q, e_p = cc.ccsdt_q(), cc.ccsdtq_p()
    assert cc.t4.dtype == np.complex128 and abs(cc.t4.imag).max() > 1e-5
    corr = lambda key: ref[key] - ref['e_scf']
    assert abs(cc.ccsdt_run.e_corr - corr('e_ccsdt')) < 2e-10
    assert abs(cc.ccsdt_run.e_corr + cc.ccsdt_run.e_q5 - corr('e_ccsdt[q]')) < 2e-10
    assert abs(cc.ccsdt_run.e_corr + e_q - corr('e_ccsdt(q)')) < 2e-10
    assert abs(cc.e_corr - corr('e_ccsdtq')) < 2e-10
    assert abs(cc.e_corr + e_p - corr('e_ccsdtq(p)')) < 2e-10
    assert abs(e_p - (ref['e_ccsdtq(p)'] - ref['e_ccsdtq'])) < 3e-12


def test_frozen_and_dropped_spinors_by_a_list(hbr):
    '''zccsdt.CCSDT on the X2C object with frozen as a list of spinors, as
    in PySCF - the 28 of the core and the virtual ones above the sixteenth:
    the space of the reference of MRCC, by the front end alone.'''
    ref = REFS['hbr_ccpvdzdk_x2c_mdcint']['recommended']
    order = np.argsort(hbr.mo_energy)
    frozen = list(order[:28]) + list(order[36 + 16:])
    buf = io.StringIO()
    cc = zccsdt.CCSDT(hbr, frozen=frozen)
    assert isinstance(cc, zccsdt.GCCSDT) and cc.ncore == 28
    cc.stdout, cc.verbose = buf, 4
    cc.conv_tol, cc.conv_tol_normt = 1e-11, 1e-9
    cc.kernel()
    assert '8 occupied, 16 virtual spinors; 28 frozen, 12 virtual ones dropped' in buf.getvalue()
    assert cc._eris.nocc == 8 and cc._eris.nmo == 24
    e_q = cc.ccsdt_q()
    corr = lambda key: ref[key] - ref['e_scf']
    assert abs(cc.e_corr - corr('e_ccsdt')) < 2e-10
    assert abs(cc.e_corr + cc.e_q5 - corr('e_ccsdt[q]')) < 2e-10 and abs(cc.e_corr + e_q - corr('e_ccsdt(q)')) < 2e-10
    # a number is the lowest occupied ones
    assert zccsdt.GCCSDT(hbr, frozen=28)._active()[0].tolist() == sorted(order[:28].tolist())
    with pytest.raises(ValueError):
        zccsdt.GCCSDT(hbr, frozen=[64]).ncore
    with pytest.raises(NotImplementedError):
        zccsdt.CCSDTQ(scf.RHF(gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='sto-3g', verbose=0)).run(),
                      frozen=[0])
