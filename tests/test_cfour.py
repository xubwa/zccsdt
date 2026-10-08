'''The MO integrals of CFOUR's two-component programs as an integral
container (zccsdt.cfour).

tests/data/cfour/h2o_sto3g: the files that xjoda, xsox2catm, xsox2chf,
xx2cvtran and xx2cccsd of cfour-x2cso left for H2O / STO-3G (X2C-1e with
spin-orbit coupling, no symmetry, CCSD(T), all electrons; 10 occupied and 4
virtual spinors, complex integrals), with the converged amplitudes of
xx2cccsd.  The installed xx2cvtran does not write <ab||cd> (moinpppp), so
what is compared with CFOUR is what does not need it: the second-order
energy, and with CFOUR's amplitudes the CCSD energy, (T) and the T1
equation.  (bench/cfour/check_integrals.py compares every block with the
integrals PySCF makes for CFOUR's spinors.)

The whole way - a container written in CFOUR's files, read back, and the
calculation that the ZMAT asks for - is tested on integrals of PySCF.'''
import io
import lzma
import os
import stat
from types import SimpleNamespace

import numpy as np
import pytest
from pyscf import gto, x2c
from pyscf.cc import gccsd, gccsd_t

import zccsdt
from zccsdt import cfour, eris as E
from zccsdt.cc import ccsd, solver

HERE = os.path.dirname(__file__)
# printed by xsox2chf, xx2cvtran and xx2cccsd
REF = {'e_scf': -75.002378932014523, 'e_mp2': -3.558664173578302e-02, 'e_ccsd': -0.049502919581008,
       'e_t': -0.000067244390488, 'e_ccsd_t_tot': -75.051949095986018}


@pytest.fixture(scope='module')
def rundir(tmp_path_factory):
    d = tmp_path_factory.mktemp('h2o_sto3g')
    src = os.path.join(HERE, 'data', 'cfour', 'h2o_sto3g')
    for name in os.listdir(src):
        with lzma.open(os.path.join(src, name)) as f:
            open(d / name[:-3], 'wb').write(f.read())
    return str(d)


def test_read(rundir):
    '''The files as they are, and the container: 10 + 4 spinors, the SCF
    energy of JOBARC, complex antisymmetrized integrals, <ij||ab> the same
    in its two files, the second-order energy of xx2cvtran.  Without
    moinpppp there is no container for the CC classes, and the message
    says which keyword is missing.'''
    x = cfour.read(rundir)
    assert (x.nocc, x.nvir, x.nfrozen, x.ndropped) == (10, 4, 0, 0)
    assert abs(x.e_scf - REF['e_scf']) < 1e-12 and x.pppp is None
    assert x.hhhh.shape == (55, 55) and x.ppph.shape == (10, 4, 10) and x.hhpp.shape == (4, 10, 4, 10)
    assert np.all(np.diff(x.energy[:10]) > -1e-9) and x.energy[9] < 0 < x.energy[10]
    eris = cfour.container(x)
    assert eris.nocc == 10 and eris.nmo == 14 and eris.vvvv is None and not hasattr(eris, '_vvvv_pp')
    assert eris.dtype == np.complex128 and abs(eris.ovvv.imag).max() > 0.01
    for key in ('oooo', 'ooov', 'oovv'):
        blk = getattr(eris, key)
        assert abs(blk + blk.transpose(1, 0, 2, 3)).max() < 1e-13
    assert abs(eris.oooo - eris.oooo.conj().transpose(2, 3, 0, 1)).max() < 1e-12
    assert abs(eris.ovvo - eris.ovvo.conj().transpose(3, 2, 1, 0)).max() < 1e-12
    assert abs(eris.ovvv + eris.ovvv.transpose(0, 1, 3, 2)).max() < 1e-13
    assert abs(cfour.mp2(eris) - REF['e_mp2']) < 1e-13
    with pytest.raises(FileNotFoundError, match='ABCDTYPE'):
        cfour.from_cfour(rundir)
    # files of two calculations
    y = cfour.read(rundir)
    y.pphh = y.pphh * 1.001
    with pytest.raises(ValueError, match='moinpphh'):
        cfour.container(y)


def test_amplitudes_of_cfour(rundir):
    '''With the amplitudes xx2cccsd converged to (t1guess, t2guess): its
    CCSD correlation energy and its (T) correction from the container, and
    the T1 equation, which has every block but <ab||cd> and <ij||kl>.'''
    x = cfour.read(rundir)
    eris = cfour.container(x, packed_vvvv=False)
    no, nv = x.nocc, x.nvir
    t1 = np.ascontiguousarray(cfour._complex(os.path.join(rundir, 't1guess'), (nv, no)).T)
    t2 = cfour._complex(os.path.join(rundir, 't2guess'), (nv * (nv + 1) // 2, no * (no + 1) // 2))
    t2 = np.ascontiguousarray(cfour._unpack(cfour._unpack(t2, nv, 0), no, 2).transpose(2, 3, 0, 1))
    assert abs(ccsd.energy(t1, t2, eris) - REF['e_ccsd']) < 1e-13
    pe = gccsd._PhysicistsERIs()
    for key in ('fock', 'mo_energy', 'oovv', 'ooov', 'ovvv'):
        setattr(pe, key, getattr(eris, key))
    e_t = gccsd_t.kernel(SimpleNamespace(verbose=0, stdout=None), pe, t1, t2, verbose=0)
    assert abs(e_t - REF['e_t']) < 1e-13
    assert abs(x.e_scf + REF['e_ccsd'] + REF['e_t'] - REF['e_ccsd_t_tot']) < 1e-11
    eris.vvvv = np.zeros((nv,) * 4, dtype=np.complex128)
    n1, _ = ccsd.update_amps(SimpleNamespace(level_shift=0.0), t1, t2, eris, alg='dense')
    r1 = (n1 - t1) * (eris.mo_energy[:no, None] - eris.mo_energy[None, no:])
    assert abs(r1).max() < 1e-11 and abs(t1).max() > 1e-3


def test_zmat_and_job(rundir, tmp_path):
    '''The keywords and sections of a ZMAT, and what the CC step makes of
    them: CALC and the section %zccsdt for the method, CC_CONV, CC_MAXCYC
    and MEMORY_SIZE as joda understood them (JOBARC).'''
    kw, sections = cfour.parse_zmat(os.path.join(rundir, 'ZMAT'))
    assert kw['CALC'] == 'CCSD(T)' and kw['ABCDTYPE'] == 'STANDARD' and kw['CC_CONV'] == '12' and kw['MEM_UNIT'] == 'GB'
    assert sections == {}
    assert cfour.job(rundir).method == 'ccsd(t)'
    with pytest.raises(FileNotFoundError, match='scf2ccmo'):        # (the files of the SCF are not in the test data)
        cfour.run(rundir, verbose=0, stdout=io.StringIO())
    j = cfour.job(rundir, method='CCSDT(Q)')
    assert (j.method, j.conv_tol_normt, j.conv_tol, j.max_cycle) == ('ccsdt(q)', 1e-12, 1e-14, 50)
    assert abs(j.max_memory - 2000.0) < 1e-9 and j.source['max_memory'] == 'MEMORY_SIZE'
    assert cfour.programs(rundir) == ['xjoda', 'xsox2catm', 'xsox2chf', 'xx2cvtran']
    # a section of our own, and one of CFOUR's
    d = tmp_path / 'run'
    d.mkdir()
    for name in ('JOBARC', 'JAINDX'):
        open(d / name, 'wb').write(open(os.path.join(rundir, name), 'rb').read())
    text = open(os.path.join(rundir, 'ZMAT')).read().rstrip() + '\n\n%amfiMethod*\n0\n1\n1\n0\n\n'
    open(d / 'ZMAT', 'w').write(text + '%zccsdt\nmethod=CCSDTQ(P)\nmax_memory 3000\nconv_tol=1e-9\n\n')
    kw, sections = cfour.parse_zmat(str(d / 'ZMAT'))
    assert sections['amfimethod'] == ['0', '1', '1', '0'] and len(sections['zccsdt']) == 3
    j = cfour.job(str(d))
    assert (j.method, j.max_memory, j.conv_tol, j.conv_tol_normt) == ('ccsdtq(p)', 3000.0, 1e-9, 1e-12)
    assert j.source['method'] == '%zccsdt' and cfour.job(str(d), max_memory=500).max_memory == 500
    assert cfour.programs(str(d))[:2] == ['xjoda', 'xx2camf']
    open(d / 'ZMAT', 'w').write(text + '%zccsdt\nmemory=3\n\n')
    with pytest.raises(ValueError, match='memory'):
        cfour.job(str(d))
    # CFOUR's own way to (Q) and (P): CALC=CC(n-1)(n) with EXCITATION=n as joda stored it; (Q)_Lambda by name
    open(d / 'ZMAT', 'w').write(text.replace('CALC=CCSD(T)', 'CALC=CC(n-1)(n),EXCITATION=5'))
    arc = cfour.JobArc(str(d))
    flags2 = np.array(arc.get('IFLAGS2'))
    for n, method in ((5, 'ccsdtq(p)'), (4, 'ccsdt(q)')):
        flags2[37] = n
        arc.put('IFLAGS2', flags2)
        j = cfour.job(str(d))
        assert j.method == method and j.source['method'].endswith('EXCITATION=%d' % n)
    flags2[37] = 7
    arc.put('IFLAGS2', flags2)
    with pytest.raises(ValueError, match='EXCITATION=7'):
        cfour.job(str(d))
    assert cfour.job(str(d), method='CCSDT(Q)_L').method == 'ccsdt(q)_l'


def test_jobarc_records(rundir, tmp_path):
    '''JOBARC: a record written over (TOTENERG, which xx2cccsd left), a new
    one appended as PUTREC does, the others untouched.'''
    for name in ('JOBARC', 'JAINDX'):
        open(tmp_path / name, 'wb').write(open(os.path.join(rundir, name), 'rb').read())
    old = cfour.JobArc(rundir)
    assert abs(old.get('TOTENERG', '<f8')[0] - REF['e_ccsd_t_tot']) < 1e-12
    arc = cfour.JobArc(str(tmp_path))
    arc.put('TOTENERG', np.array([-75.5]))
    arc.put('ZCCSDTEN', np.array([1.5, 2.5, 3.5]))
    new = cfour.JobArc(str(tmp_path))
    assert new.get('TOTENERG', '<f8')[0] == -75.5 and list(new.get('ZCCSDTEN', '<f8')) == [1.5, 2.5, 3.5]
    k = new.names.index('ZCCSDTEN')
    assert new.names[k - 1] == 'TOTENERG' and new.loc[k] == new.loc[k - 1] + 1
    assert os.path.getsize(tmp_path / 'JOBARC') % 1024 == 0
    assert os.path.getsize(tmp_path / 'JAINDX') == os.path.getsize(os.path.join(rundir, 'JAINDX'))
    for key in ('IFLAGS', 'NUMDROPA', 'COORD'):
        assert np.array_equal(new.get(key), old.get(key))
    assert new.get('SCFENEG', '<f8')[0] == old.get('SCFENEG', '<f8')[0]
    with pytest.raises(ValueError):
        arc.put('TOTENERG', np.zeros(2))
    with pytest.raises(KeyError):
        arc.get('NOTTHERE')


@pytest.fixture(scope='module')
def bh():
    '''BH / STO-3G away from the z axis: X2C spinors, complex integrals.'''
    mol = gto.M(atom='B 0 0 0; H 0.5 0.3 1.0', basis='sto-3g', verbose=0)
    mf = x2c.UHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-8
    mf.kernel()
    return mf


@pytest.mark.parametrize('frozen,packed', [(0, True), (2, False)])
def test_written_and_read(bh, tmp_path, frozen, packed):
    '''A container of PySCF's integrals in CFOUR's files and back: every
    block, the packed <ab||cd> in the order of the CC code, the energies,
    the frozen pair.'''
    ref = E.from_scf(bh, frozen=frozen, packed_vvvv=packed)
    cfour.write(str(tmp_path), ref, nfrozen=frozen)
    x = cfour.read(str(tmp_path))
    assert (x.nocc, x.nvir, x.nfrozen, x.ndropped) == (6 - frozen, 6, frozen, 0)
    eris = cfour.from_cfour(str(tmp_path), packed_vvvv=packed)
    assert abs(eris.e_hf - bh.e_tot) < 1e-10 and list(eris.spinors) == list(range(frozen, 12))
    for key in ('oooo', 'ooov', 'oovv', 'ovov', 'ovvo', 'ovvv'):
        assert abs(np.asarray(getattr(eris, key)) - np.asarray(getattr(ref, key))).max() < 1e-13, key
    assert abs(eris.fock - np.diag(np.diag(ref.fock))).max() < 1e-12
    if packed:
        assert eris.vvvv is None and abs(eris._vvvv_pp - ref._vvvv_pp).max() < 1e-13
        assert abs(ref._vvvv_pp.imag).max() > 1e-3
    else:
        assert abs(eris.vvvv - ref.vvvv).max() < 1e-13


def test_calculation_of_the_zmat(bh, tmp_path, capsys):
    '''The front end on a directory in CFOUR's files: the method from CALC
    (CCSDT(Q)) and from the section %zccsdt (CCSDTQ(P), which joda does not
    know), thresholds and memory from joda's flags; the energies of the
    classes on the original container; the final energy in JOBARC.'''
    ref = E.from_scf(bh, packed_vvvv=True)
    d = str(tmp_path)
    cfour.write(d, ref, flags={2: 54, 4: 9, 7: 60, 36: 500000000})
    zmat = 'BH\nB\nH 1 1.2\n\n*CFOUR(CALC=CCSDT(Q),BASIS=STO-3G\nABCDTYPE=STANDARD,CC_CONV=9)\n\n'
    open(os.path.join(d, 'ZMAT'), 'w').write(zmat)
    buf = io.StringIO()
    e = cfour.run(d, stdout=buf, jobarc=True)
    out = buf.getvalue()
    assert 'CCSDT(Q) (CALC of the ZMAT)' in out and '6 occupied, 6 virtual spinors; 0 frozen' in out
    assert 'conv_tol 1e-11 (CC_CONV)' in out and 'max_cycle 60 (CC_MAXCYC)' in out and '4000 MB (MEMORY_SIZE)' in out
    assert 'integrals given (CFOUR, %s)' % d in out
    cc = zccsdt.GCCSDTQ.from_integrals(ref, verbose=0)
    cc.conv_tol, cc.conv_tol_normt = 1e-11, 1e-9
    cc.kernel()
    e_q, e_p = cc.ccsdt_q(), cc.ccsdtq_p()
    assert abs(e['scf'] - bh.e_tot) < 1e-10 and abs(e['ccsdt'] - cc.ccsdt_run.e_tot) < 1e-9
    assert abs(e['ccsdt(q)'] - (cc.ccsdt_run.e_tot + e_q)) < 1e-9 and abs(e_q) > 1e-6
    assert abs(e['ccsdt[q]'] - (cc.ccsdt_run.e_tot + cc.ccsdt_run.e_q5)) < 1e-9
    assert cfour.JobArc(d).get('TOTENERG', '<f8')[0] == e['ccsdt(q)']
    # the section of our own, through the command line
    open(os.path.join(d, 'ZMAT'), 'w').write(zmat + '%zccsdt\nmethod=ccsdtq(p)\nmax_memory=2000\n\n')
    assert cfour.main([d, '--verbose', '0']) == 0
    out = capsys.readouterr().out
    assert 'CCSDTQ(P) (%zccsdt)' in out and '2000 MB (%zccsdt)' in out
    last = {line.split()[0]: float(line.split()[2]) for line in out.splitlines() if line.startswith('E(')}
    assert abs(last['E(CCSDTQ)'] - cc.e_tot) < 1e-9 and abs(last['E(CCSDTQ(P))'] - cc.e_tot_p) < 1e-9
    assert abs(e_p) > 1e-9
    # CCSD, by the argument
    r = solver.kernel(ref, 'ccsd', solver.Options(conv_tol=1e-12, conv_tol_normt=1e-9))
    assert abs(cfour.run(d, method='ccsd', stdout=io.StringIO())['ccsd'] - r.e_tot) < 1e-9
    assert cfour.main([d, '--method', 'ccsd(t)']) == 1 and 'scf2ccmo' in capsys.readouterr().err


def _program(path, body):
    with open(path, 'w') as f:
        f.write('#!/bin/sh\n' + body + '\n')
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)


def test_chain_of_programs(tmp_path, capsys):
    '''The runner of CFOUR's programs, on stand-ins: the order, an output
    and a line of times per program, the command before a program; a
    program that ends with a status, one without the last line of CFOUR's
    programs and one that does not leave its files stop the chain, named.'''
    bindir, d = tmp_path / 'bin', tmp_path / 'run'
    bindir.mkdir()
    d.mkdir()
    open(d / 'ZMAT', 'w').write('x\nH\n\n*CFOUR(CALC=CCSDT)\n\n')
    open(d / 'GENBAS', 'w').write('\n')
    ok = 'echo " @CHECKOUT-I, Total execution time"'
    _program(bindir / 'xjoda', 'cat ZMAT > MOL; touch JOBARC JAINDX; echo joda >> order; ' + ok)
    _program(bindir / 'xsox2catm', 'echo atm >> order; ' + ok)
    _program(bindir / 'xsox2chf', 'test -f guess || exit 3; touch scf2ceig nmototal; echo hf >> order; ' + ok)
    _program(bindir / 'xx2cvtran', 'echo vtran >> order; ' + ok)
    with pytest.raises(cfour.ProgramFailed, match='xsox2chf FAILED: exit status 3'):
        cfour.run_programs(str(d), bindir=str(bindir), log=lambda s: None)
    with pytest.raises(cfour.ProgramFailed, match='xx2cvtran FAILED: it did not write moinhhhh'):
        cfour.run_programs(str(d), bindir=str(bindir), before={'xsox2chf': 'touch guess'}, log=lambda s: None)
    assert open(d / 'order').read().split() == ['joda', 'atm', 'joda', 'atm', 'hf', 'vtran']
    assert 'CHECKOUT' in open(d / 'xsox2chf.out').read() and 'xsox2chf wall' in open(d / 'xsox2chf.time').read()
    _program(bindir / 'xsox2catm', 'echo "Something very wrong"; ' + ok)
    with pytest.raises(cfour.ProgramFailed, match='xsox2catm FAILED: its output has'):
        cfour.run_programs(str(d), bindir=str(bindir), log=lambda s: None)
    _program(bindir / 'xsox2catm', 'echo stopped early')
    with pytest.raises(cfour.ProgramFailed, match='xsox2catm FAILED: its output does not end'):
        cfour.run_programs(str(d), bindir=str(bindir), log=lambda s: None)
    with pytest.raises(cfour.ProgramFailed, match='no executable'):
        cfour.run_programs(str(d), ['xjoda', 'xnothing'], bindir=str(bindir))
    # the command line: the chain in a scratch directory, the outputs back, the message of the failure
    start = tmp_path / 'start'
    start.mkdir()
    for name in ('ZMAT', 'GENBAS'):
        open(start / name, 'w').write(open(d / name).read())
    code = cfour.main([str(start), '--chain', '--bin', str(bindir), '--scratch', str(tmp_path / 'scr'), '--no-cc'])
    captured = capsys.readouterr()
    assert code == 1 and 'xsox2catm FAILED' in captured.err and 'xjoda wall' in captured.out
    assert sorted(os.listdir(start)) == ['GENBAS', 'ZMAT', 'xjoda.out', 'xjoda.time', 'xsox2catm.out', 'xsox2catm.time']
    assert os.listdir(tmp_path / 'scr') == []


def test_ao_driven_ccsd_t_on_cfours_spinors(tmp_path):
    '''CCSD and CCSD(T) of DirectZCCSD on the spinors of CFOUR's SCF
    (scf2ccmo), the basis read from GENBAS and the geometry from MOL, no
    integral file of xx2cvtran: the energies of xx2cccsd.'''
    pytest.importorskip('socutils.tools.label')
    src = os.path.join(HERE, 'data', 'cfour', 'h2o_sto3g_scf')
    for name in os.listdir(src):
        if name.endswith('.xz'):
            with lzma.open(os.path.join(src, name)) as f:
                open(tmp_path / name[:-3], 'wb').write(f.read())
    open(tmp_path / 'GENBAS', 'w').write(open(os.path.join(src, 'GENBAS.txt')).read())
    mol = cfour.molecule(str(tmp_path))
    assert mol.nao_2c() == 14 and mol.nelectron == 10
    buf = io.StringIO()
    e = cfour.run(str(tmp_path), method='ccsd(t)', verbose=0, stdout=buf)
    assert 'AO-driven' in buf.getvalue()
    assert abs(e['scf'] - REF['e_scf']) < 1e-9
    assert abs(e['ccsd'] - e['scf'] - REF['e_ccsd']) < 1e-9
    assert abs(e['ccsd(t)'] - e['ccsd'] - REF['e_t']) < 1e-9


def test_handover_files(tmp_path):
    '''The files xx2czvec reads for the relaxed density (docs/native.md):
    written and read back; the density xx2czvec assembles from them is the
    one handed over, also from the split of CFOUR's own files (dpoo not
    Hermitian, a diagonal in dii0); what the files cannot carry is refused.'''
    rng = np.random.default_rng(7)
    nf, no, nv, nd = 2, 4, 6, 2
    n, o = nf + no + nv + nd, nf + no
    c = lambda *shape: rng.standard_normal(shape) + 1j * rng.standard_normal(shape)     # noqa: E731
    D = c(n, n)
    D = D + D.conj().T
    D[:nf, :nf], D[o + nv:, o + nv:] = np.eye(nf), 0
    D[nf:o, nf:o] += np.eye(no)         # (its virtual-occupied block is not handed over: xx2czvec makes it)
    x, dvo = c(n - o, o), c(n - o, o)
    files = cfour.handover_files(D, x, o, nf, nd)
    assert [files[k].shape for k in cfour.HANDOVER] == [(8, 6), (4, 4), (6, 6), (4, 4), (6, 6), (2, 4), (2, 6)]
    cfour.write_handover(str(tmp_path), files)
    assert os.path.getsize(tmp_path / 'xfull') == 8 + 16 * 8 * 6    # one record, complex numbers
    back = cfour.read_handover(str(tmp_path), n, o, nf, nd)
    assert all(abs(back[k] - files[k]).max() == 0 for k in cfour.HANDOVER)
    assert abs(back['xfull'] - x).max() == 0
    assert abs(np.fromfile(tmp_path / 'xfull', dtype='<f8', offset=4, count=4)
               - [x[0, 0].real, x[0, 0].imag, x[1, 0].real, x[1, 0].imag]).max() == 0     # the first index fastest
    want = D.copy()
    want[o:, :o], want[:o, o:] = dvo, dvo.conj().T
    assert abs(cfour.handover_density(back, dvo) - want).max() < 1e-14
    # the split is free: only dpoo + dpoo^+ + diag(dii0) and dpvv + dpvv^+ + diag(daa0) are used
    other = dict(back)
    a, b = np.triu(c(no, no), 1), np.triu(c(nv, nv), 1)
    s, t = rng.standard_normal(no), rng.standard_normal(nv)
    other['dpoo'] = back['dpoo'] + a - a.conj().T - 0.5 * np.diag(s)
    other['dpvv'] = back['dpvv'] + b - b.conj().T - 0.5 * np.diag(t)
    other['dii0'], other['daa0'] = np.diag(s) + np.triu(c(no, no), 1), np.diag(t) + np.tril(c(nv, nv), -1)
    assert abs(cfour.handover_density(other, dvo) - want).max() < 1e-14
    # no frozen and no dropped spinors: two empty records
    files = cfour.handover_files(D[nf:o + nv, nf:o + nv], x[:nv, nf:], no)
    cfour.write_handover(str(tmp_path), files)
    assert os.path.getsize(tmp_path / 'dpopo') == 8 and os.path.getsize(tmp_path / 'dpvpv') == 8
    assert cfour.read_handover(str(tmp_path), no + nv, no)['dpopo'].shape == (0, no)
    for bad, what in ((D + 1e-3 * np.triu(np.ones((n, n)), 1), 'not Hermitian'),
                      (D + np.diag([1e-3] + [0.0] * (n - 1)), 'not 1 between the frozen'),
                      (D + np.diag([0.0] * (n - 1) + [1e-3]), 'not 0 between the dropped')):
        with pytest.raises(ValueError, match=what):
            cfour.handover_files(bad, x, o, nf, nd)
    with pytest.raises(ValueError, match='all virtual, all occupied'):
        cfour.handover_files(D, x[:nv], o, nf, nd)
    # where xx2czvec is wrong (measured: docs/native.md): active occupied, active virtual, all spinors
    for sizes, sound in (((10, 16, 26), True), ((8, 16, 26), True), ((8, 14, 26), True), ((8, 12, 26), False),
                         ((10, 12, 26), False), ((8, 10, 26), False), ((10, 12, 22), False),
                         # docs/benchmarks.md, "Frozen virtual spinors": H2O, HI
                         ((8, 42, 80), True), ((8, 24, 80), False), ((8, 8, 80), False),
                         ((18, 130, 256), True), ((18, 94, 256), True), ((18, 78, 256), True),
                         ((18, 68, 256), True), ((18, 58, 256), False)):
        assert cfour.xx2czvec_sound(*sizes) is sound


@pytest.mark.parametrize('method,nfrozen,ndropped', [('ccsd(t)', 0, 0), ('ccsd', 2, 2)])
def test_handover_to_a_stand_in_for_xx2czvec(tmp_path, monkeypatch, method, nfrozen, ndropped):
    """--zvector cfour without CFOUR: xx2czvec replaced by a function that
    does what it does with the files - solves R(d) = -xfull with the
    orbital Hessian of the SCF in CFOUR's spinors and writes dpvofull and
    the density it assembles.  Its denfinal is the one the Z-vector solver
    of this program writes: the right-hand side and the blocks of the
    density handed over are the whole relaxed density (the oo and vv
    rotations of (T), the frozen and dropped spinors)."""
    pytest.importorskip('socutils.tools.label')
    from zccsdt.zccsd import zvector
    src = os.path.join(HERE, 'data', 'cfour', 'h2o_sto3g_scf')
    for name in os.listdir(src):
        if name.endswith('.xz'):
            with lzma.open(os.path.join(src, name)) as f:
                open(tmp_path / name[:-3], 'wb').write(f.read())
    open(tmp_path / 'GENBAS', 'w').write(open(os.path.join(src, 'GENBAS.txt')).read())
    d = str(tmp_path)
    arc = cfour.JobArc(d)
    arc.put('NUMDROPA', np.array([nfrozen // 2], dtype=np.int64))
    arc.put('NUMVIRTA', np.array([ndropped], dtype=np.int64))
    j = cfour.job(d, method)
    j.conv_tol, j.conv_tol_normt = 1e-12, 1e-10
    cfour.run_direct(d, j, verbose=0, stdout=io.StringIO(), density=True)
    n, o = 14, 10
    own = cfour._complex(os.path.join(d, 'denfinal'), (n, n))
    os.remove(os.path.join(d, 'denfinal'))
    seen = {}

    def xx2czvec(directory, progs, bindir=None, log=print, **kw):
        assert progs == ['xx2czvec']
        mf = cfour.reference(directory)[0]
        files = cfour.read_handover(directory, n, o, nfrozen, ndropped)
        conv, dvo = zvector.solve(mf, np.asarray(mf.mo_coeff), np.asarray(mf.mo_energy), o, -files['xfull'],
                                  tol=1e-12, veff=zvector.scf_veff(mf))
        cfour._write_complex(os.path.join(directory, 'dpvofull'), dvo)
        cfour._write_complex(os.path.join(directory, 'denfinal'), cfour.handover_density(files, dvo))
        open(os.path.join(directory, 'xx2czvec.out'), 'w').write(' Z-vector equation converged in 9 cycles\n')
        seen.update(files)

    monkeypatch.setattr(cfour, 'run_programs', xx2czvec)
    out = io.StringIO()
    cfour.run_direct(d, j, verbose=0, stdout=out, density=True, zvector='cfour')
    assert 'max |R(D_vo) + xfull|' in out.getvalue()
    handed = cfour._complex(os.path.join(d, 'denfinal'), (n, n))
    assert abs(handed - handed.conj().T).max() < 1e-12 and abs(np.trace(handed) - o) < 1e-10
    assert abs(handed - own).max() < 2e-9
    assert seen['dpopo'].shape == (nfrozen, o - nfrozen) and seen['dpvpv'].shape == (ndropped, n - o - ndropped)
    assert abs(seen['dii0']).max() == 0 and abs(seen['daa0']).max() == 0
    if nfrozen:
        assert abs(seen['dpopo']).max() > 1e-7 and abs(seen['dpvpv']).max() > 1e-5
    # a solver that does not solve the equations is refused, and its density is not left as denfinal

    def wrong(directory, progs, bindir=None, log=print, **kw):
        xx2czvec(directory, progs)
        files = cfour.read_handover(directory, n, o, nfrozen, ndropped)
        dvo = 1.01 * cfour._complex(os.path.join(directory, 'dpvofull'), (n - o, o))
        cfour._write_complex(os.path.join(directory, 'dpvofull'), dvo)
        cfour._write_complex(os.path.join(directory, 'denfinal'), cfour.handover_density(files, dvo))

    monkeypatch.setattr(cfour, 'run_programs', wrong)
    with pytest.raises(cfour.ProgramFailed, match='did not solve the Z-vector equations'):
        cfour.run_direct(d, j, verbose=0, stdout=io.StringIO(), density=True, zvector='cfour')
    assert not os.path.isfile(os.path.join(d, 'denfinal')) and os.path.isfile(os.path.join(d, 'denfinal.wrong'))
