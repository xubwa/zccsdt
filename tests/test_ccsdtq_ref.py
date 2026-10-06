'''The dense spin-orbital references of CCSDT(Q), CCSDTQ and CCSDTQ(P)
(zccsdt.cc.ccsdtq_ref) against MRCC, with real and with complex spin
orbitals.  Inputs and energies of the reference runs: bench/refs.'''
import json
import os

import numpy as np
import pytest
from pyscf import gto, scf

from zccsdt import eris as E
from zccsdt.cc import ccsdtq_ref as R
from zccsdt.genbas import read_genbas

ROOT = os.path.join(os.path.dirname(__file__), '..', 'bench', 'refs')
REFS = json.load(open(os.path.join(ROOT, 'reference_energies.json')))
GEOM = {'bh': 'B 0 0 0; H 0 0 1.2324', 'lih': 'Li 0 0 0; H 0 0 1.5957'}


def _scf(name):
    atoms = sorted({a.split()[0] for a in GEOM[name].split(';')})
    basis = {el: read_genbas(os.path.join(ROOT, 'GENBAS'), '%s:STO-3G' % el.upper()) for el in atoms}
    mol = gto.M(atom=GEOM[name], basis=basis, verbose=0)
    return scf.RHF(mol).run(conv_tol=1e-13)


def _spin_mixed(ghf, seed=4):
    '''(orbitals, u): the two spin orbitals of every spatial orbital mixed by
    a random SU(2) matrix of their own - complex orbitals that are still
    canonical; new = old @ u.'''
    rng = np.random.default_rng(seed)
    n = ghf.mo_coeff.shape[1]
    u = np.zeros((n, n), dtype=complex)
    order = np.argsort(ghf.mo_energy, kind='stable')
    for k in range(0, n, 2):
        p, q = order[k], order[k + 1]
        assert abs(ghf.mo_energy[p] - ghf.mo_energy[q]) < 1e-9
        a = rng.standard_normal(4)
        a /= np.linalg.norm(a)
        u[np.ix_([p, q], [p, q])] = [[a[0] + 1j * a[1], a[2] + 1j * a[3]],
                                     [-a[2] + 1j * a[3], a[0] - 1j * a[1]]]
    return ghf.mo_coeff @ u, u


def _rotate(t, uo, uv):
    '''The amplitudes of the same cluster operator in rotated spin orbitals
    (new = old @ u): occupied indices with u, virtual ones with conj(u).'''
    k = t.ndim // 2
    for axis in range(k):
        t = np.moveaxis(np.tensordot(t, uo, axes=([axis], [0])), -1, axis)
    for axis in range(k, 2 * k):
        t = np.moveaxis(np.tensordot(t, uv.conj(), axes=([axis], [0])), -1, axis)
    return t


@pytest.mark.parametrize('cplx', [False, True])
def test_lih_against_mrcc(cplx):
    '''CCSDT, CCSDT[Q], CCSDT(Q) and CCSDTQ (the full CI of four electrons)
    for LiH / STO-3G; with complex spin orbitals the same energies.'''
    mf = _scf('lih')
    ref = REFS['lih_sto3g']['recommended']
    assert abs(mf.e_tot - ref['e_scf']) < 1e-10
    ghf = scf.addons.convert_to_ghf(mf)
    eris = E.from_scf(ghf, mo_coeff=_spin_mixed(ghf)[0]) if cplx else E.from_scf(ghf)
    fock, g, mo_e, nocc = R.integrals(eris)
    assert (abs(g.imag).max() > 0.01) == cplx
    rt = R.kernel(fock, g, mo_e, nocc, 'ccsdt', conv_tol=1e-12, conv_tol_normt=1e-10)
    e5, e6 = R.quadruples(rt.amps[1], rt.amps[2], g, mo_e, nocc)
    rq = R.kernel(fock, g, mo_e, nocc, 'ccsdtq', conv_tol=1e-12, conv_tol_normt=1e-10)
    assert rt.converged and rq.converged
    assert abs(mf.e_tot + rt.e_corr - ref['e_ccsdt']) < 2e-11
    assert abs(mf.e_tot + rt.e_corr + e5 - ref['e_ccsdt[q]']) < 2e-11
    assert abs(mf.e_tot + rt.e_corr + e6 - ref['e_ccsdt(q)']) < 2e-11
    assert abs(mf.e_tot + rq.e_corr - ref['e_ccsdtq']) < 2e-11
    if cplx:
        assert abs(rq.amps[3].imag).max() > 1e-6


def test_bh_against_mrcc():
    '''BH / STO-3G, the smallest of the systems with pentuples: all five
    energies; then the amplitudes carried over to complex spin orbitals -
    they solve the equations there, and (Q) and (P) are the same numbers.'''
    mf = _scf('bh')
    ref = REFS['bh_sto3g']['recommended']
    ghf = scf.addons.convert_to_ghf(mf)
    fock, g, mo_e, nocc = R.integrals(E.from_scf(ghf))
    rt = R.kernel(fock, g, mo_e, nocc, 'ccsdt', conv_tol=1e-12, conv_tol_normt=1e-10)
    e5, e6 = R.quadruples(rt.amps[1], rt.amps[2], g, mo_e, nocc)
    rq = R.kernel(fock, g, mo_e, nocc, 'ccsdtq', conv_tol=1e-12, conv_tol_normt=1e-10)
    ep = R.pentuples(rq.amps[1], rq.amps[2], rq.amps[3], g, mo_e, nocc)
    assert rt.converged and rq.converged
    assert abs(mf.e_tot + rt.e_corr - ref['e_ccsdt']) < 2e-11
    assert abs(mf.e_tot + rt.e_corr + e5 - ref['e_ccsdt[q]']) < 2e-11
    assert abs(mf.e_tot + rt.e_corr + e6 - ref['e_ccsdt(q)']) < 2e-11
    assert abs(mf.e_tot + rq.e_corr - ref['e_ccsdtq']) < 2e-11
    # MRCC's energies have twelve decimals: -3.674e-9 +- 1e-12
    assert abs(ep - (ref['e_ccsdtq(p)'] - ref['e_ccsdtq'])) < 2e-12
    assert abs(ep + 3.674744e-9) < 1e-14

    C, u = _spin_mixed(ghf)
    fc, gc, mo_c, _ = R.integrals(E.from_scf(ghf, mo_coeff=C))
    assert abs(gc.imag).max() > 0.01 and abs(mo_c - mo_e).max() < 1e-9
    uo, uv = u[:nocc, :nocc], u[nocc:, nocc:]
    for res, e_pert in ((rt, (e5, e6)), (rq, ep)):
        amps = [_rotate(t, uo, uv) for t in res.amps]
        for k, r in enumerate(R.residuals(amps, fc, gc, nocc)):
            assert abs(r).max() < 1e-8, k
        if len(amps) == 3:
            got = R.quadruples(amps[1], amps[2], gc, mo_c, nocc)
            assert abs(got[0] - e_pert[0]) < 1e-13 and abs(got[1] - e_pert[1]) < 1e-13
        else:
            assert abs(R.pentuples(amps[1], amps[2], amps[3], gc, mo_c, nocc) - e_pert) < 1e-14
