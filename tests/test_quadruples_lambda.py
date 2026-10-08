'''CCSDT(Q)_Lambda on spin orbitals or spinors (zccsdt.cc.quadruples_lambda):
the left-hand products against the term table and the energies against the
dense reference (zccsdt.cc.lambda_ref), (Q) itself for lambda = conj(t),
every memory plan against the plain one, peak allocations against the plan,
and molecules with converged CCSDT and Lambda (zccsdt.cc.lambda_opt)
against the dense reference, MRCC and CFOUR.'''
import gc
import json
import os

import numpy as np
import pytest
from pyscf import gto, scf

import fockspace as F
from test_ccsdtq_ref import _scf, _spin_mixed
from test_quadruples import _container, _packed_data, _random
from zccsdt import eris as E
from zccsdt.cc import (ccsdtq_ref as R, ccsdtq_terms as Q, lambda_opt as LO, lambda_ref as LR,
                       quadruples as QD, quadruples_lambda as QL, solver, t4pack as P)
from zccsdt.genbas import read_genbas

ROOT = os.path.join(os.path.dirname(__file__), '..', 'bench', 'refs')
REFS = json.load(open(os.path.join(ROOT, 'reference_energies.json')))
H2O = 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587'


def _random_lambda(no, nv, cplx, seed=3):
    '''_random of test_quadruples with antisymmetric l2 and l3 that are not
    the conjugates of t2 and t3.'''
    arr, mo_e = _random(no, nv, cplx, seed)
    rng = np.random.default_rng(seed + 100)
    for k in (2, 3):
        x = F.random_antisym(rng, no, nv, k)
        arr['l%d' % k] = x + 1j * F.random_antisym(rng, no, nv, k) if cplx else x
    return arr, mo_e


def _close(a, b, tol=1e-12):
    return abs(a - b) < tol * abs(b)


@pytest.mark.parametrize('no,nv,cplx', [(5, 6, True), (6, 5, True), (7, 4, False), (5, 7, False)])
@pytest.mark.parametrize('use_c', [True, False])
def test_products_equal_terms(no, nv, cplx, use_c, monkeypatch):
    '''z2 and z3 for all quadruples are the terms Z4.T2 and Z4.T3 of the
    table with conj(l) for t, conjugated; r is that of (Q); the energies
    those of the dense reference - with the kernel of the energy and
    without.'''
    monkeypatch.setattr(QL, 'USE_C', use_c and QL.USE_C)
    arr, mo_e = _random_lambda(no, nv, cplx)
    eris = _container(arr['v'], mo_e, no)
    T3, L3 = P.pack(arr['t3']), P.pack(arr['l3'])
    z2, z3, rt3, rtt = QL.rows(eris, arr['t2'], T3, arr['l2'], L3)
    left = {'v': arr['v'], 't2': arr['l2'].conj(), 't3': arr['l3'].conj()}
    for x, name in ((z2, 'Z4.T2'), (z3, 'Z4.T3')):
        want = Q.asym_unique(Q.total(Q.LEFT[name][2], left, no, 4), 4).conj()
        assert abs(x - want).max() < 1e-12 * abs(want).max(), name
    plain = QD.rows(eris, arr['t2'], T3)
    assert abs(rt3 - plain[1]).max() == 0 and abs(rtt - plain[2]).max() == 0
    # each of the two products of Lambda_3 through its own integrals
    for k, (keep, drop) in enumerate((('ovvv', 'ooov'), ('ooov', 'ovvv'))):
        one = _container(arr['v'], mo_e, no)
        setattr(one, drop, np.zeros_like(getattr(one, drop)))
        want = Q.asym_unique(Q.total([Q.LEFT['Z4.T3'][2][k]], left, no, 4), 4).conj()
        assert abs(QL.rows(one, arr['t2'], T3, arr['l2'], L3)[1] - want).max() < 1e-12 * abs(want).max(), keep
    e = QL.kernel(eris, arr['t2'], T3, arr['l2'], L3)
    ref = LR.quadruples_lambda(arr['t2'], arr['t3'], arr['l2'], arr['l3'], arr['v'], mo_e, no)
    assert _close(e[0], ref[0]) and _close(e[1], ref[1])
    assert abs(ref[1] - R.quadruples(arr['t2'], arr['t3'], arr['v'], mo_e, no)[1]) > 1e-3 * abs(ref[1])


@pytest.mark.parametrize('no,nv,cplx', [(5, 6, True), (6, 7, False)])
def test_conjugate_amplitudes_give_q(no, nv, cplx):
    '''l2 = conj(t2), l3 = conj(T3): [Q] and (Q) of zccsdt.cc.quadruples.'''
    arr, mo_e = _random(no, nv, cplx)
    eris = _container(arr['v'], mo_e, no)
    T3 = P.pack(arr['t3'])
    e = QL.kernel(eris, arr['t2'], T3, arr['t2'].conj(), T3.conj())
    ref = QD.kernel(eris, arr['t2'], T3)
    assert _close(e[0], ref[0]) and _close(e[1], ref[1])


def test_real_amplitudes_complex_lambda():
    '''The types are those of the product: real t with complex lambda.'''
    no, nv = 5, 5
    arr, mo_e = _random_lambda(no, nv, False)
    rng = np.random.default_rng(9)
    l2 = arr['l2'] + 1j * F.random_antisym(rng, no, nv, 2)
    l3 = arr['l3'] + 1j * F.random_antisym(rng, no, nv, 3)
    eris = _container(arr['v'], mo_e, no)
    e = QL.kernel(eris, arr['t2'], P.pack(arr['t3']), l2, P.pack(l3))
    ref = LR.quadruples_lambda(arr['t2'], arr['t3'], l2, l3, arr['v'], mo_e, no)
    assert _close(e[0], ref[0]) and _close(e[1], ref[1])
    # (the imaginary part of lambda does not enter the real part of the energy)
    re = QL.kernel(eris, arr['t2'], P.pack(arr['t3']), arr['l2'], P.pack(arr['l3']))
    assert _close(e[0], re[0]) and _close(e[1], re[1])


def test_every_plan_gives_the_same():
    '''Products in batches of columns, Xv for a few occupied indices at a
    time, as in (Q).'''
    no, nv = 7, 6
    arr, mo_e = _random_lambda(no, nv, True)
    eris = _container(arr['v'], mo_e, no)
    T3, L3 = P.pack(arr['t3']), P.pack(arr['l3'])
    full = QL.plan(no, nv, 16)
    assert full.xv == no - 1
    ref = QL.rows(eris, arr['t2'], T3, arr['l2'], L3, full)
    e0 = QL.kernel(eris, arr['t2'], T3, arr['l2'], L3, plan_=full)
    for xv in (3, 6):
        for cols22, cols13, rows in ((1, 1, 1), (3, 7, 2), (full.cols22, full.cols13, full.rows)):
            pl = QL.explicit_plan(no, nv, 16, cols22, cols13, xv, rows)
            for a, b in zip(QL.rows(eris, arr['t2'], T3, arr['l2'], L3, pl), ref):
                assert abs(a - b).max() < 1e-12 * abs(b).max(), pl
            e = QL.kernel(eris, arr['t2'], T3, arr['l2'], L3, plan_=pl)
            assert _close(e[0], e0[0]) and _close(e[1], e0[1])


@pytest.mark.parametrize('no,nv,cplx', [(5, 20, True), (6, 26, False)])
def test_memory_plan(no, nv, cplx):
    '''plan(): monotone in the budget, the whole budget's worth at the top,
    MemoryError below the floor; the arrays that a run allocates
    (tracemalloc follows numpy's allocations) stay within Plan.need - for
    budgets from the floor up and for the smallest batches.'''
    import tracemalloc
    eris, t2, T3 = _packed_data(no, nv, cplx)
    _, l2, L3 = _packed_data(no, nv, cplx, seed=2)
    isz = 16 if cplx else 8
    full = QL.plan(no, nv, isz)
    plain = QD.plan(no, nv, isz)
    assert (full.cols22, full.cols13, full.xv, full.rows) == (plain.cols22, plain.cols13, plain.xv, plain.rows)
    assert full.need - plain.need == full.floor - plain.floor == QL._extra(no, nv, isz) > 0
    budgets = [int(b) for b in np.linspace(full.floor, 1.1 * full.need, 60)]
    plans = [QL.plan(no, nv, isz, b) for b in budgets]
    key = [(p.cols22, p.cols13, p.xv, p.rows) for p in plans]
    for a, b in zip(key, key[1:]):
        assert all(x <= y for x, y in zip(a, b))
    assert all(p.need <= b for p, b in zip(plans, budgets))
    assert key[-1] == (full.cols22, full.cols13, full.xv, full.rows) and plans[-1].need == full.need
    with pytest.raises(MemoryError, match=r'\(Q\)_Lambda needs'):
        QL.plan(no, nv, isz, full.floor - 1)
    e0 = QL.kernel(eris, t2, T3, l2, L3, plan_=full)
    least = QL.explicit_plan(no, nv, isz, *QD._least(nv)[:2], 3, QD._least(nv)[2])
    assert least.need == least.floor < full.need and least.need >= plans[0].need
    seen = []
    for pl in [least, plans[0], plans[len(plans) // 2], full]:
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        e = QL.kernel(eris, t2, T3, l2, L3, plan_=pl)
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        assert abs(e[0] - e0[0]) < 1e-11 * abs(e0[0]) and abs(e[1] - e0[1]) < 1e-11 * abs(e0[1])
        assert peak <= pl.need, (peak, pl)
        seen.append(peak / pl.need)
    # (not a loose bound either: the full plan is met to within the slack)
    assert seen[-1] > 0.8


def test_fewer_than_four():
    eris, t2, T3 = _packed_data(3, 6, False)
    assert QL.kernel(eris, t2, T3, t2, T3) == (0.0, 0.0)


# -------------------------------------------------------------------- molecules

def _mf(name):
    if name == 'bh_sto3g':
        return _scf('bh')
    basis = {el: read_genbas(os.path.join(ROOT, 'GENBAS'), '%s:%s' % (el, name.split('_')[1].upper().replace(
        '631G', '6-31G').replace('STO3G', 'STO-3G'))) for el in 'OH'}
    return scf.RHF(gto.M(atom=H2O, basis=basis, verbose=0)).run(conv_tol=1e-13)


def _solve(name, cplx):
    '''(scf, eris, CCSDT, Lambda) with the production codes.'''
    mf = _mf(name)
    ghf = scf.addons.convert_to_ghf(mf)
    eris = E.from_scf(ghf, mo_coeff=_spin_mixed(ghf)[0] if cplx else ghf.mo_coeff)
    assert (abs(np.asarray(eris.oovv).imag).max() > 0.01) == cplx
    rt = solver.kernel(eris, 'ccsdt', solver.Options(conv_tol=1e-12, conv_tol_normt=1e-10))
    rl = LO.kernel(eris, rt.t1, rt.t2, rt.t3, solver.Options(conv_tol_normt=1e-10))
    assert rt.converged and rl.converged
    return mf, eris, rt, rl


@pytest.mark.parametrize('name,cplx', [('h2o_sto3g', False), ('h2o_sto3g', True), ('bh_sto3g', False),
                                       ('bh_sto3g', True)])
def test_molecules_against_reference_mrcc_and_cfour(name, cplx):
    '''CCSDT and Lambda of the production codes, (Q)_Lambda on the packed
    amplitudes: the dense reference from the same amplitudes, CCSDT(Q)_L of
    MRCC and, for H2O, the two corrections of CFOUR's ncc; with conj(t)
    for lambda CCSDT(Q), another number.'''
    mf, eris, rt, rl = _solve(name, cplx)
    ref = REFS[name]
    e5, e6 = QL.kernel(eris, rt.t2, rt.t3, rl.l2, rl.l3)
    fock, g, mo_e, nocc = R.integrals(eris)
    nvir = len(mo_e) - nocc
    d5, d6 = LR.quadruples_lambda(rt.t2, LO.dense_l3(rt.t3, nocc, nvir), rl.l2, LO.dense_l3(rl.l3, nocc, nvir),
                                  g, mo_e, nocc)
    assert abs(e5 - d5) < 1e-13 and abs(e6 - d6) < 1e-13
    assert abs(rt.e_tot - ref['recommended']['e_ccsdt']) < 1e-10
    assert abs(rt.e_tot + e6 - ref['recommended']['e_ccsdt(q)_l']) < 1e-10
    assert abs(rt.e_tot + e6 - ref['mrcc']['bohr']['e_ccsdt(q)_l']) < 1e-10
    if name.startswith('h2o'):
        ncc = ref['cfour']['NCC']['bohr']
        assert abs(e5 - ncc['e_ccsdt[q]_l_corr']) < 2e-11 and abs(e6 - ncc['e_ccsdt(q)_l_corr']) < 2e-11
    q5, q6 = QL.kernel(eris, rt.t2, rt.t3, rt.t2.conj(), np.conj(rt.t3))
    p5, p6 = QD.kernel(eris, rt.t2, rt.t3)
    assert abs(q5 - p5) < 1e-14 and abs(q6 - p6) < 1e-14
    assert abs(rt.e_tot + q6 - ref['recommended']['e_ccsdt(q)']) < 1e-10
    assert abs(q6 - e6) > 1e-7


@pytest.mark.parametrize('cplx', [False, True])
def test_h2o_631g_against_mrcc_and_cfour(cplx):
    '''H2O / 6-31G (10 + 16 spin orbitals), real and with every pair of spin
    orbitals mixed by its own SU(2) matrix.'''
    mf, eris, rt, rl = _solve('h2o_631g', cplx)
    ref = REFS['h2o_631g']
    e5, e6 = QL.kernel(eris, rt.t2, rt.t3, rl.l2, rl.l3)
    assert abs(rt.e_tot - ref['recommended']['e_ccsdt']) < 1e-10
    assert abs(rt.e_tot + e6 - ref['recommended']['e_ccsdt(q)_l']) < 1e-9
    ncc = ref['cfour']['NCC']['bohr']
    assert abs(e5 - ncc['e_ccsdt[q]_l_corr']) < 5e-11 and abs(e6 - ncc['e_ccsdt(q)_l_corr']) < 5e-11
