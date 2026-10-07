'''The PySCF-style front end (zccsdt.ccsdt): the same energies as the
engines, dispatch by the kind of reference, frozen core, output.'''

import io

import numpy as np
import pytest
from pyscf import dft, gto, scf, x2c

import zccsdt
from zccsdt import timing


def _water(basis='6-31g', verbose=0):
    return gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis=basis,
                 verbose=verbose)


@pytest.fixture(scope='module')
def rhf():
    return scf.RHF(_water()).run(conv_tol=1e-11)


@pytest.fixture(scope='module')
def uhf():
    mol = gto.M(atom='N 0 0 0; H 0 0.8036 0.6347; H 0 -0.8036 0.6347', basis='6-31g', spin=1,
                verbose=0)
    return scf.UHF(mol).run(conv_tol=1e-11)


def _quiet_run(mycc, verbose=4):
    buf = io.StringIO()
    mycc.stdout, mycc.verbose = buf, verbose
    mycc.conv_tol, mycc.conv_tol_normt = 1e-10, 1e-7
    mycc.kernel()
    return buf.getvalue()


def test_rccsdt_equals_engine(rhf):
    from zccsdt.sa import rccsdt, rccsdt_ref
    mycc = zccsdt.CCSDT(rhf, frozen=1)
    assert isinstance(mycc, zccsdt.RCCSDT)
    out = _quiet_run(mycc)
    ref = rccsdt.kernel(rccsdt_ref.from_scf(rhf, ncore=1, full=False), conv_tol=1e-10,
                        conv_tol_normt=1e-7)
    assert mycc.converged and abs(mycc.e_corr - ref.e_corr) < 1e-10
    assert abs(mycc.e_tot - (rhf.e_tot + ref.e_corr)) < 1e-10
    assert mycc.cycles == ref.ncycle and len(mycc.timings) == mycc.cycles
    for text in ('zccsdt RCCSDT', 'reference', 'amplitudes', 'memory plan', 'E_corr / Eh',
                 'RCCSDT converged', 'E(CCSDT)', 'diagnostics', 'largest amplitudes',
                 'timings', 'pp ladder', 'pair pass', 'peak memory'):
        assert text in out, text
    # one line per iteration in the table
    assert sum(1 for line in out.splitlines()
               if line[:6].strip().isdigit() and 'e-' in line) == mycc.cycles
    # same-spin + opposite-spin parts add up to the correlation energy (RHF: f_ov = 0)
    ss = float(out.split('E_corr(same-spin)')[1].split()[0])
    os_ = float(out.split('E_corr(oppo-spin)')[1].split()[0])
    assert abs(ss + os_ - mycc.e_corr) < 1e-9
    e_q = mycc.ccsdt_q()
    r = rccsdt.add_q(rccsdt_ref.from_scf(rhf, ncore=1, full=False), ref)
    assert abs(e_q - r.e_q) < 1e-10 and abs(mycc.e_q5 - r.e_q5) < 1e-10
    assert abs(mycc.e_tot_q - (mycc.e_tot + e_q)) < 1e-12
    assert 'E(CCSDT(Q))' in mycc.stdout.getvalue()


def test_uccsdt_equals_engine(uhf):
    from zccsdt import eris_spin
    from zccsdt.cc import solver
    mycc = zccsdt.CCSDT(uhf, frozen=1)          # one core orbital per spin
    assert isinstance(mycc, zccsdt.UCCSDT)
    out = _quiet_run(mycc)
    ref = solver.kernel(eris_spin.from_scf(uhf, frozen=2), 'ccsdt',
                        solver.Options(conv_tol=1e-10, conv_tol_normt=1e-7))
    assert mycc.converged and abs(mycc.e_corr - ref.e_corr) < 1e-9
    for text in ('zccsdt UCCSDT', '1 frozen per spin', '|dT3|', 'UCCSDT converged',
                 'largest amplitudes', 'a -> ', 'timings'):
        assert text in out, text


def test_gccsdt_equals_engine(rhf):
    from zccsdt import eris
    from zccsdt.cc import solver
    ghf = scf.addons.convert_to_ghf(rhf)
    mycc = zccsdt.CCSDT(ghf, frozen=2)          # spin orbitals
    assert isinstance(mycc, zccsdt.GCCSDT)
    out = _quiet_run(mycc)
    ref = solver.kernel(eris.from_scf(ghf, frozen=2, packed_vvvv=True), 'ccsdt',
                        solver.Options(conv_tol=1e-10, conv_tol_normt=1e-7))
    assert mycc.converged and abs(mycc.e_corr - ref.e_corr) < 1e-9
    # the closed-shell result in a third representation
    r = zccsdt.RCCSDT(rhf, frozen=1)
    _quiet_run(r, verbose=0)
    assert abs(mycc.e_corr - r.e_corr) < 1e-8
    assert 'zccsdt GCCSDT' in out and 'spin orbitals' in out


def test_dispatch_and_frozen(rhf, uhf):
    mol = rhf.mol
    rohf = scf.ROHF(gto.M(atom='O 0 0 0; H 0 0 1', basis='sto-3g', spin=1, verbose=0))
    assert isinstance(zccsdt.CCSDT(rohf), zccsdt.UCCSDT)
    assert isinstance(zccsdt.CCSDT(uhf), zccsdt.UCCSDT)
    assert isinstance(zccsdt.CCSDT(scf.GHF(mol)), zccsdt.GCCSDT)
    assert isinstance(zccsdt.CCSDT(x2c.UHF(mol)), zccsdt.GCCSDT)
    with pytest.raises(RuntimeError):
        zccsdt.CCSDT(dft.RKS(mol))
    with pytest.raises(NotImplementedError):
        zccsdt.CCSDT(scf.DHF(mol))
    # set_frozen: the chemical core, in the unit of each class
    assert zccsdt.RCCSDT(rhf).set_frozen().frozen == 1
    assert zccsdt.UCCSDT(uhf).set_frozen().frozen == 1
    assert zccsdt.GCCSDT(scf.addons.convert_to_ghf(rhf)).set_frozen().frozen == 2
    with pytest.raises(NotImplementedError):
        zccsdt.RCCSDT(rhf, frozen=[0, 1]).ncore
    with pytest.raises(NotImplementedError):
        zccsdt.UCCSDT(uhf).ccsdt_q()


def test_ccsdtq_front_end(rhf):
    '''zccsdt.CCSDTQ: the class by the kind of reference; H2O / 6-31G, RHF
    taken as GHF spin orbitals - CCSDT first, then CCSDTQ, (Q) from the
    CCSDT step, (P) - against MRCC on the same integrals (bench/refs:
    h2o_631g_pyscf_mdcint); the report; a second run from the amplitudes;
    the frozen core in the unit of the reference.'''
    import json
    import os
    from zccsdt import eris as E
    from zccsdt.cc import ccsdtq
    ref = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'bench', 'refs',
                                      'reference_energies.json')))['h2o_631g_pyscf_mdcint']['recommended']
    mol = rhf.mol
    assert isinstance(zccsdt.CCSDTQ(scf.GHF(mol)), zccsdt.GCCSDTQ)
    assert isinstance(zccsdt.CCSDTQ(x2c.UHF(mol)), zccsdt.GCCSDTQ)
    with pytest.raises(RuntimeError):
        zccsdt.CCSDTQ(dft.RKS(mol))
    with pytest.raises(NotImplementedError):
        zccsdt.CCSDTQ(scf.DHF(mol))
    mycc = zccsdt.CCSDTQ(rhf)
    assert isinstance(mycc, zccsdt.GCCSDTQ) and mycc.ncore == 0
    buf = io.StringIO()
    mycc.stdout, mycc.verbose = buf, 4
    mycc.conv_tol, mycc.conv_tol_normt = 1e-10, 1e-8
    mycc.kernel()
    out = buf.getvalue()
    assert mycc.converged and mycc.ccsdt_run.converged
    assert abs(rhf.e_tot - ref['e_scf']) < 1e-9
    assert abs(mycc.e_ccsdt - ref['e_ccsdt']) < 2e-9 and abs(mycc.e_tot - ref['e_ccsdtq']) < 2e-9
    e_q = mycc.ccsdt_q()
    assert abs(e_q - (ref['e_ccsdt(q)'] - ref['e_ccsdt'])) < 1e-10
    assert abs(mycc.ccsdt_run.e_q5 - (ref['e_ccsdt[q]'] - ref['e_ccsdt'])) < 1e-10
    assert mycc.t4.shape == (210, 1820) and mycc.t4.dtype == np.float64 and len(mycc.timings) == mycc.cycles
    e_p = mycc.ccsdtq_p()
    assert abs(e_p - (ref['e_ccsdtq(p)'] - ref['e_ccsdtq'])) < 1e-10 and mycc.e_p == e_p
    assert abs(mycc.e_tot_p - ref['e_ccsdtq(p)']) < 2e-9
    out = buf.getvalue()
    for text in ('zccsdt GCCSDT:', 'zccsdt GCCSDTQ:', '|dT4|', 'T4 382,200', 'GCCSDTQ converged',
                 'E(CCSDTQ) - E(CCSDT)', 'T4 equation', 'products by occupied quadruple', 'peak memory',
                 '(P) correction: 252 occupied x 4,368 virtual pentuples, real', 'E(CCSDTQ(P))'):
        assert text in out, text
    # from the converged amplitudes: nothing left to do
    again = zccsdt.GCCSDTQ(scf.addons.convert_to_ghf(rhf))
    again.verbose = 0
    again.conv_tol, again.conv_tol_normt = 1e-9, 1e-7
    again.kernel(mycc.t1, mycc.t2, mycc.t3, mycc.t4)
    assert again.cycles <= 2 and abs(again.e_tot - mycc.e_tot) < 1e-9 and again.e_ccsdt is None
    with pytest.raises(RuntimeError):
        again.ccsdt_q()
    with pytest.raises(RuntimeError):
        zccsdt.GCCSDTQ(scf.addons.convert_to_ghf(rhf)).ccsdtq_p()
    # frozen core: orbitals per spin of the RHF reference, spin orbitals of the class
    small = scf.RHF(_water('sto-3g')).run(conv_tol=1e-12)
    fc = zccsdt.CCSDTQ(small, frozen=1)
    fc.verbose = 0
    fc.conv_tol, fc.conv_tol_normt = 1e-11, 1e-9
    fc.kernel()
    assert fc.frozen == 2 and fc.t4.shape == (70, 1)
    direct = ccsdtq.kernel(E.from_scf(scf.addons.convert_to_ghf(small), frozen=2, packed_vvvv=True),
                           ccsdtq.Options(conv_tol=1e-11, conv_tol_normt=1e-9))
    assert abs(fc.e_tot - direct.e_tot) < 1e-9


def test_quiet_and_short_of_memory(rhf, monkeypatch, tmp_path):
    from zccsdt import memory
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    mycc = zccsdt.RCCSDT(rhf)
    assert _quiet_run(mycc, verbose=0) == ''
    e_ref = mycc.e_corr
    # the least memory that works: T3 and the operator in scratch files
    monkeypatch.setattr(memory, 'resident', lambda: 0)
    from zccsdt.sa import rccsdt
    no, nv = mycc._eris.nocc, mycc._eris.nvir
    low = rccsdt.plan_memory(no, nv).need
    while True:
        try:
            rccsdt.plan_memory(no, nv, low - 20_000)
        except MemoryError:
            break
        low -= 20_000
    short = zccsdt.RCCSDT(rhf)
    short.max_memory = low / 1e6
    out = _quiet_run(short)
    assert 'memory is short' in out and 'scratch file' in out
    assert short.plan_used.t3_file and abs(short.e_corr - e_ref) < 1e-9


def test_timing_tree():
    timing.reset()
    with timing.timer('a'):
        with timing.timer('b'):
            pass
        with timing.timer('b'):
            with timing.timer('c'):
                pass
        timing.add('d', 0.5, 1.0)
    snap = timing.snapshot()
    assert snap[('a', 'b')][2] == 2 and ('a', 'b', 'c') in snap
    assert snap[('a', 'd')] == (0.5, 1.0, 1)
    assert [r[0] for r in timing.children(snap, ('a',))] == ['d', 'b']
    with timing.timer('a'):
        pass
    assert timing.since(snap)[('a',)][2] == 1 and ('a', 'b') not in timing.since(snap)
    flat = dict((k, (s, n)) for k, s, n in timing.report())
    assert flat['b'][1] == 2 and flat['a'][1] == 2
    timing.reset()
