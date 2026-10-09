'''EOM-EE-CCSDT and EOM-EE-CCSD (zccsdt.cc.eom; docs/eom.md): the Davidson
solver on the sigma vectors of zccsdt.cc.eom_opt

- against all eigenvalues of the dense reference matrix (LiH, real and
  complex spin orbitals);
- against MRCC: H2O / 6-31G (MRCC's own integrals) and HBr with X2C spinors
  (the same integrals, written by zccsdt.dirac) - bench/refs, "eom_ee";
- in the blocks of the symmetry labels against the packed block;
- with the vectors of the subspace in a scratch file, the memory plan and
  what is allocated against it; the front end.'''
import gc
import io
import json
import os
import tracemalloc

import numpy as np
import pytest
from pyscf import gto, scf, x2c

import zccsdt
from test_ccsdtq_ref import _scf, _spin_mixed
from test_quadruples_lambda import _mf
from zccsdt import dirac, eris as E
from zccsdt.cc import ccsdt_opt, ccsdtq_ref as R, ccsdtq_terms as Q, eom, eom_opt, eom_ref, solver, t3pack

REFS = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'bench', 'refs', 'reference_energies.json')))['eom_ee']
TIGHT = solver.Options(conv_tol=1e-12, conv_tol_normt=1e-10)


def _nearest(e, values):
    return max(abs(np.asarray(e) - x).min() for x in np.atleast_1d(values))


@pytest.mark.parametrize('cplx', [False, True])
@pytest.mark.parametrize('method', ['ccsd', 'ccsdt'])
def test_roots_are_eigenvalues_of_the_reference_matrix(cplx, method):
    '''LiH / STO-3G, 4 + 8 spin orbitals (complex: every pair mixed by an
    SU(2) matrix): the six lowest roots - a triplet, a singlet, two of a
    sixfold level - are eigenvalues of the dense matrix (424 x 424 with
    triples) to 1e-10, and the lowest ones.'''
    mf = scf.addons.convert_to_ghf(_scf('lih'))
    eris = E.from_scf(mf, mo_coeff=_spin_mixed(mf)[0] if cplx else mf.mo_coeff)
    rt = solver.kernel(eris, method, TIGHT)
    with3 = method == 'ccsdt'
    res = eom.kernel(eris, rt.t1, rt.t2, rt.t3 if with3 else None, nroots=6, conv_tol=1e-9)
    assert res.converged.all() and res.residuals.max() < 1e-9 and len(res.e) == 6
    assert res.ground.dtype == (np.complex128 if cplx else np.float64) and res.ground.change < 1e-8
    fock, g, mo_e, no = R.integrals(eris)
    om, vm = t3pack.maps_for(eris)
    amps = [rt.t1, rt.t2]
    if with3:
        amps.append(Q.expand(np.asarray(rt.t3).reshape(om.ntri, vm.ntri), no, vm.n, 3))
    w = eom_ref.eigenvalues(amps, fock, g, no)
    assert abs(w.imag).max() < 1e-9
    assert abs(res.e - w.real[:6]).max() < 1e-10
    assert abs(res.e[0] - res.e[2]) < 1e-10 and res.e[3] - res.e[2] > 0.01          # the triplet, the singlet
    assert abs(res.weights.sum(axis=1) - 1).max() < 1e-12 and (res.weights[:, 0] > 0.8).all()
    assert (res.r3 is None) == (not with3) and res.nsigma >= 6
    if with3:       # (the guess: the roots of the singles-doubles block, followed)
        assert res.guess.converged.all() and abs(res.guess.e.real - res.e).max() < 1e-3
        assert res.weights[:, 2].max() > 1e-6


def test_h2o_against_mrcc():
    '''H2O / 6-31G: EOM-CCSD with the 1s pair frozen (8 + 16 spin orbitals)
    and EOM-CCSDT with all electrons (10 + 16), the lowest triplet and the
    lowest singlet (four roots of spin orbitals), and of the complex spin
    orbitals the same - against MRCC to 1e-9 (its energies have 12
    decimals).'''
    ghf = scf.addons.convert_to_ghf(_mf('h2o_631g'))
    ref = REFS['h2o_631g']['ccsd_frozen_core']
    eris = E.from_scf(ghf, frozen=2)
    rt = solver.kernel(eris, 'ccsd', TIGHT)
    assert abs(rt.e_tot - ref['e_ccsd']) < 1e-10
    res = eom.kernel(eris, rt.t1, rt.t2, None, nroots=12, nguess=16, conv_tol=1e-8)
    assert res.converged.all()
    assert _nearest(res.e, ref['singlets'] + ref['triplets']) < 1e-9
    assert abs(res.e[:4] - ([ref['triplets'][0]] * 3 + [ref['singlets'][0]])).max() < 1e-9
    ref = REFS['h2o_631g']['ccsdt_all_electron']
    for cplx in (False, True):
        eris = E.from_scf(ghf, mo_coeff=_spin_mixed(ghf)[0] if cplx else ghf.mo_coeff)
        rt = solver.kernel(eris, 'ccsdt', TIGHT)
        assert abs(rt.e_tot - ref['e_ccsdt']) < 1e-10
        res = eom.kernel(eris, rt.t1, rt.t2, rt.t3, nroots=4, conv_tol=1e-8)
        assert res.converged.all() and res.r3[0].dtype == (np.complex128 if cplx else np.float64)
        assert abs(res.e - ([ref['triplets'][0]] * 3 + [ref['singlets'][0]])).max() < 1e-9


@pytest.fixture(scope='module')
def hbr():
    mol = gto.M(atom='Br 0 0 0; H 0 0 1.4145', basis='cc-pvdz-dk', verbose=0)
    mf = x2c.UHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-8
    mf.kernel()
    return dirac.container(dirac.integrals(mf, frozen=28, nvir=16))


@pytest.mark.parametrize('method', ['ccsd', 'ccsdt'])
def test_hbr_spinors_against_mrcc_on_the_same_integrals(hbr, method):
    '''HBr / cc-pVDZ-DK, X2C spinors in Kramers pairs, 8 + 16 active: the
    states MRCC finds through its interface to DIRAC on the files written
    from these integrals (two degenerate ones, 6.73 eV, and one at 7.08 eV)
    are roots here to 1e-9; so is a pair in between that MRCC does not
    look for.'''
    ref = REFS['hbr_ccpvdzdk_x2c_mdcint'][method]
    rt = solver.kernel(hbr, method, TIGHT)
    assert abs(rt.e_tot - ref['e_' + method]) < 1e-9
    res = eom.kernel(hbr, rt.t1, rt.t2, rt.t3 if method == 'ccsdt' else None, nroots=6, nguess=8, conv_tol=1e-8)
    assert res.converged.all()
    assert _nearest(res.e, ref['states']) < 1e-9
    assert abs(res.e[0] - res.e[1]) < 1e-9 and abs(res.e[2] - res.e[3]) < 1e-9 and 0.004 < res.e[2] - res.e[1] < 0.006


# ---------------------------------------------------------------- label blocks

@pytest.fixture(scope='module')
def lih_x2c():
    '''CCSDT of LiH / 6-31G with X2C spinors (4 + 18), in the blocks of the
    symmetry labels and on the packed block.'''
    mf = x2c.UHF(gto.M(atom='Li 0 0 0; H 0 0 1.5957', basis='6-31g', verbose=0)).run(conv_tol=1e-12)
    out = {}
    for sym in (True, False):
        cc = zccsdt.CCSDT(mf)
        cc.symmetry, cc.verbose = sym, 0
        cc.conv_tol, cc.conv_tol_normt = 1e-11, 1e-9
        cc.kernel()
        out[sym] = cc
    return out


def test_sigma_in_label_blocks(lih_x2c):
    '''The sigma vector of a totally symmetric vector in the blocks of the
    labels (T3 and R3: 868 of the 3264 elements of the packed block) is the
    one on the packed block.'''
    cc = lih_x2c[True]
    eb = cc._eris
    ed, t3d = cc._dense_view()
    omb, vmb = t3pack.maps_for(eb)
    omd, vmd = t3pack.maps_for(ed)
    assert omb.spins is not None and omd.spins is None and np.size(cc.t3) == 868 and t3d.size == 3264
    rng = np.random.default_rng(2)
    lo, lv = np.asarray(omb.spins), np.asarray(vmb.spins)
    r1 = rng.standard_normal((omb.n, vmb.n)) * (lo[:, None] == lv[None, :])
    x = rng.standard_normal((omb.n,) * 2 + (vmb.n,) * 2)
    x = x - x.transpose(1, 0, 2, 3)
    x = x - x.transpose(0, 1, 3, 2)
    r2 = x * ((lo[:, None, None, None] + lo[None, :, None, None]) == (lv[None, None, :, None] + lv[None, None, None, :]))

    def p2(a, om, vm):
        return np.ascontiguousarray(a[om.pairs[:, 0][:, None], om.pairs[:, 1][:, None], vm.pairs[:, 0], vm.pairs[:, 1]])
    r3b = rng.standard_normal(np.size(cc.t3))
    r3d = ccsdt_opt.packed_t3(eb, r3b)
    gb = eom_opt.prepare(eb, cc.t1, cc.t2, cc.t3)
    sb = eom_opt.sigma(gb, r1, p2(r2, omb, vmb), r3b)
    gd = eom_opt.prepare(ed, cc.t1, cc.t2, t3d)
    sd = eom_opt.sigma(gd, r1, p2(r2, omd, vmd), r3d.ravel())
    assert abs(sb[0] - sd[0]).max() < 1e-11 * abs(sd[0]).max()
    s2 = t3pack.unpack_pairs(sd[1], omd, vmd)
    assert abs(t3pack.unpack_pairs(sb[1], omb, vmb) - s2).max() < 1e-11 * abs(s2).max()
    s3 = sd[2].reshape(omd.ntri, vmd.ntri)
    assert abs(ccsdt_opt.packed_t3(eb, sb[2]) - s3).max() < 1e-11 * abs(s3).max()
    # (what is outside the blocks vanishes on the packed block)
    assert abs(ccsdt_opt.packed_t3(eb, ccsdt_opt.blocked_t3(eb, s3)) - s3).max() < 1e-11 * abs(s3).max()


def test_roots_in_label_blocks(lih_x2c):
    '''The front end.  In the blocks of the labels: the totally symmetric
    states, each of them a root on the packed block, where the others are
    found too (the lowest level there is threefold, and one of the three
    is symmetric).'''
    buf = io.StringIO()
    cc = lih_x2c[True]
    cc.verbose, cc.stdout = 4, buf
    eb, res = cc.eom_ee(nroots=3, conv_tol=1e-8)
    cc.verbose = 0
    text = buf.getvalue()
    assert 'EOM-EE-CCSDT: 3 roots, vectors of 72 + 918 + 868 elements, real' in text
    assert 'root   omega / Eh' in text and 'sigma vectors' in text
    assert res.converged.all() and cc.e_ee is eb and cc.eom is res and res.r3[0].size == 868
    ed, resd = lih_x2c[False].eom_ee(nroots=6, conv_tol=1e-8)
    assert resd.r3[0].size == 3264
    assert _nearest(ed, eb) < 1e-8
    assert abs(ed[0] - ed[2]) < 1e-8 and abs(ed[0] - eb[0]) < 1e-8 and abs(ed[3] - eb[1]) < 1e-8


# ------------------------------------------------------------- memory and disk

def _water(basis='6-31g', frozen=2, max_cycle=3):
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis=basis, verbose=0)
    mf = scf.addons.convert_to_ghf(scf.RHF(mol).run(conv_tol=1e-12))
    eris = E.from_scf(mf, frozen=frozen, packed_vvvv=True)
    rt = solver.kernel(eris, 'ccsdt', solver.Options(max_cycle=max_cycle))
    ccsdt_opt.release_cached(eris)
    return eris, rt


def test_plan_and_mapped_vectors(monkeypatch, tmp_path):
    '''The plan: everything in memory if the budget holds it; the vectors
    of the subspace in a scratch file if only the fixed part fits (the same
    roots); a smaller subspace if the disk does not hold what was asked
    for; errors with numbers.'''
    from zccsdt import disk, memory
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    eris, rt = _water(max_cycle=30)
    dtype = np.float64
    fixed = eom.fixed_bytes(eris, dtype)
    one = (8 * 16 + 28 * 120 + 56 * 560) * 8
    p = eom.plan_for(eris, dtype, 3, opts=solver.Options(incore_complete=True))
    assert (p.max_space, p.mapped, p.disk) == (20, False, 0) and p.need == fixed + 40 * one and p.vectors == 40 * one
    assert 'in memory' in p.text()
    monkeypatch.setattr(memory, 'budget', lambda max_memory=None: fixed + 10 * one)
    monkeypatch.setattr(ccsdt_opt, 'cached_bytes', lambda eris: 0)
    p = eom.plan_for(eris, dtype, 3)
    assert p.mapped and p.max_space == 20 and p.disk == 40 * 56 * 560 * 8 and p.need == fixed + 40 * (one - 56 * 560 * 8)
    assert 'scratch file' in p.text()
    res = eom.kernel(eris, rt.t1, rt.t2, rt.t3, nroots=3, conv_tol=1e-7, plan=p)
    from zccsdt.scratch import is_mapped
    assert is_mapped(res.r3[0]) and res.converged.all()
    ref = eom.kernel(eris, rt.t1, rt.t2, rt.t3, nroots=3, conv_tol=1e-7, opts=solver.Options(incore_complete=True))
    assert not is_mapped(ref.r3[0]) and abs(res.e - ref.e).max() < 1e-9 and res.nsigma == ref.nsigma
    assert abs(abs(np.vdot(res.r3[1], ref.r3[1])) - np.vdot(ref.r3[1], ref.r3[1]).real) < 1e-6 or abs(res.e[0] - res.e[1]) < 1e-8
    # a disk that holds 11 vectors with their sigma vectors: the subspace shrinks
    monkeypatch.setattr(disk, 'budget', lambda max_disk=None, where=None: 22 * one + 5)
    p = eom.plan_for(eris, dtype, 3)
    assert p.mapped and p.max_space == 11 and p.asked == 20
    small = eom.kernel(eris, rt.t1, rt.t2, rt.t3, nroots=3, conv_tol=1e-7, plan=p)
    assert small.converged.all() and abs(small.e - ref.e).max() < 1e-8 and small.ncollapse >= ref.ncollapse
    # ... nor the smallest one (8 vectors for 3 roots): in memory if that holds them, else an error
    monkeypatch.setattr(disk, 'budget', lambda max_disk=None, where=None: 3 * one)
    monkeypatch.setattr(memory, 'budget', lambda max_memory=None: fixed + 17 * one)
    p = eom.plan_for(eris, dtype, 3)
    assert not p.mapped and p.max_space == 8
    monkeypatch.setattr(memory, 'budget', lambda max_memory=None: fixed + 10 * one)
    with pytest.raises(disk.DiskFullError, match='at least 8 vectors'):
        eom.plan_for(eris, dtype, 3)
    monkeypatch.setattr(memory, 'budget', lambda max_memory=None: fixed - 1)
    with pytest.raises(MemoryError, match='GB of memory for the operands'):
        eom.plan_for(eris, dtype, 3)


def test_memory_against_the_plan():
    '''What prepare() and the iterations allocate (tracemalloc follows
    numpy's arrays; the batch buffers of the C kernels and the buffers of
    the BLAS threads are outside of it, and are taken out of the plan for
    the comparison): H2O / cc-pVDZ, 8 + 38 spin orbitals, three roots,
    below the plan and above half of it.'''
    eris, rt = _water('cc-pvdz')
    dtype = np.float64
    eng = ccsdt_opt.engine_for(eris, dtype)
    plan = eom.plan_for(eris, dtype, 3, max_space=8, opts=solver.Options(incore_complete=True))
    arrays = plan.need - ccsdt_opt.overhead() - int(eng.buffer_bytes(2000.0))
    gc.collect()
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    res = eom.kernel(eris, rt.t1, rt.t2, rt.t3, nroots=3, conv_tol=1e-3, max_cycle=4, plan=plan)
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    assert res.nsigma >= 6
    assert 0.5 * arrays < peak <= arrays, (peak, arrays, plan)


# ------------------------------------------------- ionized and attached states

@pytest.mark.parametrize('cplx', [False, True])
@pytest.mark.parametrize('method', ['ccsd', 'ccsdt'])
def test_ip_and_ea_by_the_ghost_spinor(cplx, method):
    '''LiH / STO-3G: the four lowest ionization energies and the four
    lowest E(N+1) - E(N) of EOM-CCSD and EOM-CCSDT - the EE problem with a
    ghost spinor on packed amplitudes (the packed <ab||cd> with the real
    spin orbitals) - are the lowest eigenvalues of the dense IP and EA
    matrices of the reference to 1e-9; Kramers pairs.'''
    mf = scf.addons.convert_to_ghf(_scf('lih'))
    C = _spin_mixed(mf)[0] if cplx else mf.mo_coeff
    eris = E.from_scf(mf, mo_coeff=C, packed_vvvv=not cplx)
    rt = solver.kernel(eris, method, TIGHT)
    fock, g, mo_e, no = R.integrals(E.from_scf(mf, mo_coeff=C))
    om, vm = t3pack.maps_for(eris)
    with3 = method == 'ccsdt'
    amps = [rt.t1, rt.t2] + ([Q.expand(np.asarray(rt.t3).reshape(om.ntri, vm.ntri), no, vm.n, 3)] if with3 else [])
    for kind in ('ip', 'ea'):
        e2, t1, t2, t3, singles = eom.ghost(eris, rt.t1, rt.t2, rt.t3 if with3 else None, kind)
        assert e2.nocc == no + (kind == 'ea') and e2.fock.shape[0] == len(mo_e) + 1 and singles.sum() == (no if kind == 'ip' else vm.n)
        res = eom.kernel(e2, t1, t2, t3, nroots=4, conv_tol=1e-9, singles=singles, name=kind.upper())
        w = eom_ref.eigenvalues(amps, fock, g, no, kind)
        assert res.converged.all() and abs(res.e - w.real[:4]).max() < 1e-9
        assert abs(res.e[0] - res.e[1]) < 1e-9 and res.e[0] > 0.05
        # the vectors have the ghost index once: no singles outside of it
        assert abs(res.r1[0][~singles]).max() < 1e-12 and abs(res.r1[0][singles]).max() > 0.5


def test_ip_and_ea_against_pyscf_and_the_front_end(lih_x2c):
    '''EOM-IP- and EA-CCSD of H2O / STO-3G against PySCF's roots; the front
    end: eom_ip() and eom_ea() of a calculation in label blocks and of one
    on the packed block give the same energies (both run on the packed
    block with the ghost spinor).'''
    from conftest import water_ghf
    from test_eom_ref import pyscf_roots
    eris = E.from_scf(water_ghf('sto-3g'))
    rt = solver.kernel(eris, 'ccsd', TIGHT)
    ref = pyscf_roots()
    for kind in ('ip', 'ea'):
        e2, t1, t2, _, singles = eom.ghost(eris, rt.t1, rt.t2, None, kind)
        res = eom.kernel(e2, t1, t2, None, nroots=4, conv_tol=1e-9, singles=singles)
        assert abs(res.e - np.sort(ref[kind])[:4]).max() < 1e-8
    buf = io.StringIO()
    a, b = lih_x2c[True], lih_x2c[False]
    a.verbose, a.stdout = 4, buf
    ea, ra = a.eom_ip(nroots=2, conv_tol=1e-8)
    a.verbose = 0
    eb, rb = b.eom_ip(nroots=2, conv_tol=1e-8)
    assert 'EOM-IP-CCSDT: 2 roots' in buf.getvalue()
    assert ra.converged.all() and abs(ea - eb).max() < 1e-8 and abs(ea[0] - ea[1]) < 1e-8
    assert 0.2 < ea[0] < 0.4 and a.eom_ip_result is ra
    xa, _ = a.eom_ea(nroots=2, conv_tol=1e-8)
    xb, _ = b.eom_ea(nroots=2, conv_tol=1e-8)
    assert abs(xa - xb).max() < 1e-8 and abs(xa[0] - xa[1]) < 1e-8
