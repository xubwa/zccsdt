'''CC2 and CC3 on packed amplitudes (zccsdt.cc.ccn, ccsdt_opt.update_amps(model='cc3'),
eom_opt with the Jacobian of the model) and their front end (zccsdt.ccn):

- ground states and sigma vectors against the dense reference (ccn_ref),
  real and complex;
- spinors that are not canonical (the Fock operator of the triples equation
  off its diagonal): the energies do not change;
- H2O / 6-31G against CFOUR (xvcc, xvee: CC2 and CC3 with their
  linear-response excitation energies) and MRCC (CC3 ground state);
- CC3 in the blocks of the symmetry labels of a linear molecule against the
  packed block: ground state, excitation energies with and without a label
  shift, ionization energies with short vectors.'''
import os

import numpy as np
import pytest
from pyscf import gto, scf, x2c

import zccsdt
from conftest import water_ghf
from test_ccsdtq_ref import _scf, _spin_mixed
from test_eom_opt import pack2, pack3, random_case
from zccsdt import eris as E
from zccsdt.cc import ccn_ref as C, ccsdtq_ref as R, eom, eom_opt as EO, solver, t3pack
from zccsdt.genbas import read_genbas

TIGHT = dict(conv_tol=1e-12, conv_tol_normt=1e-10)


@pytest.fixture(autouse=True)
def _fock_as_it_is(monkeypatch):
    '''The comparisons with the dense reference are to rounding: the Fock
    operator of the triples equation with every off-diagonal element (by
    default those below 1e-6, the noise of an SCF, are left out).'''
    from zccsdt.cc import ccsdt_opt
    monkeypatch.setattr(ccsdt_opt, 'FOCK_OFFDIAG_TOL', 1e-13)


@pytest.mark.parametrize('cplx', [False, True])
@pytest.mark.parametrize('model', ['cc2', 'cc3'])
def test_ground_state_against_the_reference(model, cplx):
    mf = scf.addons.convert_to_ghf(_scf('lih'))
    eris = E.from_scf(mf, mo_coeff=_spin_mixed(mf)[0] if cplx else mf.mo_coeff)
    fock, g, mo_e, no = R.integrals(eris)
    ref = C.kernel(fock, g, mo_e, no, model)
    res = solver.kernel(eris, model, solver.Options(**TIGHT))
    assert res.converged and abs(res.e_corr - ref.e_corr) < 1e-11
    assert res.t1.dtype == (np.complex128 if cplx else np.float64)
    assert abs(res.t2 - ref.amps[1]).max() < 1e-8
    full = solver.kernel(eris, {'cc2': 'ccsd', 'cc3': 'ccsdt'}[model], solver.Options(**TIGHT))
    assert 1e-7 < abs(full.e_corr - res.e_corr) < 1e-2


@pytest.mark.parametrize('name', ['lih', 'h2o'])
@pytest.mark.parametrize('cplx', [False, True])
@pytest.mark.parametrize('model', ['cc2', 'cc3'])
def test_sigma_against_the_reference(name, cplx, model):
    '''The Jacobian of the model at amplitudes that solve nothing.'''
    with3 = model == 'cc3'
    eris, amps, r = random_case(name, cplx, with3)
    fock, g, mo_e, no = R.integrals(eris)
    om, vm = t3pack.maps_for(eris)
    gs = EO.prepare(eris, amps[0], amps[1], pack3(amps[2], om, vm).ravel() if with3 else None, model=model)
    s = EO.sigma(gs, r[0], pack2(r[1], om, vm), pack3(r[2], om, vm).ravel() if with3 else None)
    w = C.sigma(model, r, amps, fock, g, no)
    assert abs(s[0] - w[0]).max() < 1e-12 * abs(w[0]).max()
    assert abs(s[1] - pack2(w[1], om, vm)).max() < 1e-12 * abs(w[1]).max()
    if with3:
        assert abs(s[2].reshape(om.ntri, vm.ntri) - pack3(w[2], om, vm)).max() < 1e-12 * abs(w[2]).max()
    with pytest.raises(ValueError):
        EO.prepare(eris, amps[0], amps[1], None if with3 else np.zeros(om.ntri * vm.ntri), model=model)


def _rotated(mf, seed=7, frozen=0):
    '''Spin orbitals turned among the occupied and among the virtual ones
    (real orthogonal matrices that keep the spin): the same determinant,
    a Fock matrix that is not diagonal.'''
    Cm = np.array(mf.mo_coeff)
    nao = Cm.shape[0] // 2
    rng = np.random.default_rng(seed)
    occ = np.asarray(mf.mo_occ) > 0
    alpha = np.linalg.norm(Cm[:nao], axis=0) > 0.5
    for sel in (occ & alpha, occ & ~alpha, ~occ & alpha, ~occ & ~alpha):
        idx = np.flatnonzero(sel)
        idx = idx[idx >= frozen]
        q = np.linalg.qr(rng.standard_normal((len(idx), len(idx))))[0]
        Cm[:, idx] = Cm[:, idx] @ q
    return Cm


def test_spinors_that_are_not_canonical():
    '''CC2 and CC3 are invariant to rotations among the occupied and among
    the virtual spin orbitals: F is the Fock matrix as it is, and its
    off-diagonal part goes through the operators of the T3 passes.'''
    mf = scf.addons.convert_to_ghf(_scf('lih'))
    e0 = E.from_scf(mf)
    e1 = E.from_scf(mf, mo_coeff=_rotated(mf))
    f = np.asarray(e1.fock)
    no = e1.nocc
    assert abs(f[:no, :no] - np.diag(f.diagonal()[:no])).max() > 1e-2 and abs(f[:no, no:]).max() < 1e-8
    for model in ('cc2', 'cc3'):
        a = solver.kernel(e0, model, solver.Options(**TIGHT))
        b = solver.kernel(e1, model, solver.Options(max_cycle=400, **TIGHT))
        assert a.converged and b.converged and abs(a.e_corr - b.e_corr) < 1e-10
        # (with the guess of rotated spin orbitals not every component of a triplet is
        # found: the roots there are among those of the canonical ones)
        ra = eom.kernel(e0, a.t1, a.t2, a.t3, nroots=8, conv_tol=1e-8, model=model)
        rb = eom.kernel(e1, b.t1, b.t2, b.t3, nroots=4, conv_tol=1e-8, model=model, max_cycle=200)
        assert ra.converged.all() and rb.converged.all() and _in(rb.e, ra.e) < 1e-9 and abs(rb.e[0] - ra.e[0]) < 1e-9
        # the dense reference at the rotated spin orbitals
        fock, g, mo_e, _ = R.integrals(e1)
        ref = C.kernel(fock, g, mo_e, no, model, max_cycle=400)
        assert abs(ref.e_corr - b.e_corr) < 1e-10


@pytest.fixture(scope='module')
def water():
    '''H2O / 6-31G at the geometry (bohr) and with the basis (bench/refs/GENBAS,
    the entries of CFOUR's GENBAS) of the reference runs.'''
    genbas = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bench', 'refs', 'GENBAS')
    mol = gto.M(atom='O 0 0 0; H 0 1.430522676295752 1.109269235119691; H 0 -1.430522676295752 1.109269235119691',
                basis={x: read_genbas(genbas, x + ':6-31G') for x in ('O', 'H')}, unit='Bohr', verbose=0)
    return scf.RHF(mol).run(conv_tol=1e-13, conv_tol_grad=1e-9)


# CFOUR (xvcc, xvee; EOMEE-CC2 / CC3-LR, ESTATE_CONV=9): total energy, the two lowest singlets;
# MRCC: E(CC3) = -76.120380732033
CFOUR = {'CC2': (-76.113595476238, [0.307623345946, 0.392332664219]),
         'CCSD': (-76.119346380612, [0.308197284210, 0.391939281735]),
         'CC3': (-76.120380732032, [0.311525887921, 0.394096015139])}


@pytest.mark.parametrize('name', ['CC2', 'CCSD', 'CC3'])
def test_water_against_cfour(water, name):
    assert abs(water.e_tot + 75.983948495279) < 1e-10
    cc = getattr(zccsdt, name)(water)
    cc.verbose = 0
    cc.conv_tol, cc.conv_tol_normt = 1e-12, 1e-10
    cc.kernel()
    assert cc.converged and cc.model == name.lower() and (cc.t3 is None) == (name != 'CC3')
    assert abs(cc.e_tot - CFOUR[name][0]) < 2e-11
    # shift 0 of 2 m_s: the singlets and the M_S = 0 components of the triplets
    e, res = cc.eom_ee(nroots=6, shift=0, conv_tol=1e-8)
    assert res.converged.all()
    for w in CFOUR[name][1]:
        assert abs(e - w).min() < 2e-9
    with pytest.raises(NotImplementedError):
        cc.ccsdt_q()


@pytest.fixture(scope='module')
def lih_x2c():
    '''CC3 of LiH / 6-31G with X2C spinors (4 + 18), in the blocks of the
    symmetry labels and on the packed block.'''
    mf = x2c.UHF(gto.M(atom='Li 0 0 0; H 0 0 1.5957', basis='6-31g', verbose=0)).run(conv_tol=1e-12)
    out = {}
    for sym in (True, False):
        cc = zccsdt.CC3(mf)
        cc.symmetry, cc.verbose = sym, 0
        cc.conv_tol, cc.conv_tol_normt = 1e-12, 1e-10
        cc.kernel()
        out[sym] = cc
    return out


def _in(e, values):
    return max(abs(np.asarray(values) - x).min() for x in e)


def test_cc3_in_label_blocks(lih_x2c):
    cb, cd = lih_x2c[True], lih_x2c[False]
    assert np.size(cb.t3) == 868 and np.size(cd.t3) == 3264 and cb.t3.dtype == np.float64
    assert abs(cb.e_corr - cd.e_corr) < 1e-11
    ccsdt = zccsdt.CCSDT(cb._scf)
    ccsdt.verbose = 0
    ccsdt.kernel()
    assert 1e-8 < abs(ccsdt.e_corr - cb.e_corr) < 1e-4
    ed, _ = cd.eom_ee(nroots=12, conv_tol=1e-8)
    e0, r0 = cb.eom_ee(nroots=2, conv_tol=1e-8)
    assert r0.converged.all() and r0.r3[0].size == 868 and _in(e0, ed) < 1e-8
    es, rs = cb.eom_ee(nroots=2, shift=[2, -2, 4], conv_tol=1e-8)
    assert rs[-2].partner_of == 2 and abs(es[2] - es[-2]).max() == 0
    for d in (2, 4):
        assert rs[d].converged.all() and _in(es[d][:1 if d == 4 else 2], ed) < 1e-8      # (12 roots of the packed block)
    # the lowest level of the packed block: a threefold one, in the shifts 0 and +-2
    assert abs(ed[:3] - ed[0]).max() < 1e-8 and abs(e0[0] - ed[0]) < 1e-8 and abs(es[2][0] - ed[0]) < 1e-8
    # ionized and attached states: short vectors in the blocks, short vectors on the packed
    # layout, and the ghost spinor at full length
    for kind in ('ip', 'ea'):
        run = getattr(cb, 'eom_' + kind)
        full, _ = getattr(cd, 'eom_' + kind)(nroots=4, conv_tol=1e-8, short=False)
        short, _ = run(nroots=4, conv_tol=1e-8)
        assert abs(short - full).max() < 1e-8
        lab = -1 if kind == 'ip' else 1
        el, rl = run(nroots=2, shift=lab, conv_tol=1e-8)
        assert rl.converged.all() and _in(el, full) < 1e-8


def test_cc2_ionized_states_against_the_reference():
    '''CC2 and CCSD on the doubles level: IP and EA through the ghost
    spinor are the eigenvalues of the reference matrix among the vectors
    with the ghost index.'''
    mf = water_ghf('sto-3g')
    eris = E.from_scf(mf, packed_vvvv=False)
    fock, g, mo_e, no = R.integrals(eris)
    cc = zccsdt.CC2(mf)
    cc.verbose = 0
    cc.conv_tol, cc.conv_tol_normt = 1e-12, 1e-10
    cc.kernel()
    ref = C.kernel(fock, g, mo_e, no, 'cc2')
    assert abs(cc.e_corr - ref.e_corr) < 1e-11
    for kind in ('ip', 'ea'):
        w = C.eigenvalues('cc2', ref.amps, fock, g, no, kind).real
        e, res = getattr(cc, 'eom_' + kind)(nroots=2, conv_tol=1e-8)
        assert res.converged.all() and _in(e, w) < 1e-8 and abs(e[0] - w[np.argmin(abs(w - e[0]))]) < 1e-8
