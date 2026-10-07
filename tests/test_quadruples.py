'''CCSDT(Q) on spin orbitals or spinors (zccsdt.cc.quadruples): the products
against the term table, every memory plan against the plain one, peak
allocations against the plan, and energies against MRCC - through
zccsdt.CCSDT with GHF orbitals, real and complex, and for X2C spinors
against the dense reference.'''
import gc
import json
import os
import types

import numpy as np
import pytest
from pyscf import gto, scf, x2c

import fockspace as F
import zccsdt
from zccsdt import eris as E
from zccsdt.cc import ccsdtq_ref as R, ccsdtq_terms as Q, quadruples as QD, solver, t4pack as P
from zccsdt.genbas import read_genbas

ROOT = os.path.join(os.path.dirname(__file__), '..', 'bench', 'refs')
REFS = json.load(open(os.path.join(ROOT, 'reference_energies.json')))
GEOM = {'h2o': 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', 'n2': 'N 0 0 0; N 0 0 1.0977',
        'ch2': 'C 0 0 0; H 0 0.87 0.67; H 0 -0.87 0.67'}
SYSTEMS = {'h2o_631g': ('h2o', '6-31G', 0), 'h2o_ccpvdz_fc': ('h2o', 'cc-pvdz', 2),
           'n2_631g': ('n2', '6-31G', 0)}


def _random(no, nv, cplx, seed=3):
    '''(arrays of the term table, orbital energies): antisymmetric t2 and
    t3, Hermitian <pq||rs>.'''
    rng = np.random.default_rng(seed)
    vr = F.random_integrals(rng, no + nv)
    if cplx:
        vi = F.random_integrals(rng, no + nv)
        vr = vr + 1j * (vi - vi.transpose(2, 3, 0, 1))
    arr = {'v': vr}
    for k in (2, 3):
        t = F.random_antisym(rng, no, nv, k)
        arr['t%d' % k] = t + 1j * F.random_antisym(rng, no, nv, k) if cplx else t
    return arr, np.concatenate((np.sort(rng.uniform(-3, -1, no)), np.sort(rng.uniform(1, 3, nv))))


def _container(g, mo_e, nocc):
    o, v = slice(0, nocc), slice(nocc, None)
    blk = {k: np.ascontiguousarray(g[tuple(o if c == 'o' else v for c in k)])
           for k in ('oooo', 'ooov', 'oovv', 'ovov', 'ovvo', 'ovvv', 'vvvv')}
    return types.SimpleNamespace(nocc=nocc, fock=np.diag(mo_e).astype(g.dtype), mo_energy=mo_e, **blk)


@pytest.mark.parametrize('no,nv,cplx', [(5, 6, True), (6, 5, True), (7, 4, False), (5, 7, False)])
def test_products_equal_terms(no, nv, cplx):
    '''z, the [V,T3] part and the T2 T2 part of r for all quadruples are
    the terms of the table; the energies those of the dense reference.'''
    arr, mo_e = _random(no, nv, cplx)
    eris = _container(arr['v'], mo_e, no)
    T3 = P.pack(arr['t3'])
    got = QD.rows(eris, arr['t2'], T3)
    for x, terms in zip(got, (Q.LEFT['Z4.T2'][2], Q.GROUPS['R4: [V,T3]'][4],
                              Q.GROUPS['R4: 1/2 [[V,T2],T2]'][4])):
        want = Q.asym_unique(Q.total(terms, arr, no, 4), 4)
        assert abs(x - want).max() < 1e-12 * abs(want).max()
    # the three products of two T2, each through its own integrals
    for k, keep in enumerate(('vvvv', 'oooo', 'ovvo')):
        one = _container(arr['v'], mo_e, no)
        for b in ('vvvv', 'oooo', 'ovvo'):
            if b != keep:
                setattr(one, b, np.zeros_like(getattr(one, b)))
        want = Q.asym_unique(Q.total([Q.GROUPS['R4: 1/2 [[V,T2],T2]'][4][k]], arr, no, 4), 4)
        assert abs(QD.rows(one, arr['t2'], T3)[2] - want).max() < 1e-12 * abs(want).max(), keep
    e = QD.kernel(eris, arr['t2'], T3)
    ref = R.quadruples(arr['t2'], arr['t3'], arr['v'], mo_e, no)
    assert abs(e[0] - ref[0]) < 1e-12 * abs(ref[0]) and abs(e[1] - ref[1]) < 1e-12 * abs(ref[1])
    y4 = QD.second_order_quadruples(eris, arr['t2'])
    den = (mo_e[:no][P.tuples(no, 4)].sum(axis=1)[:, None] - mo_e[no:][P.tuples(nv, 4)].sum(axis=1)[None, :])
    assert abs(y4 * den - got[0]).max() < 1e-12 * abs(got[0]).max()


def test_every_plan_gives_the_same():
    '''Products in batches of columns, Xv for a few occupied indices at a
    time (the quadruples in groups), Xv formed from a few rows of
    <vv||vv> per step.'''
    no, nv = 7, 6
    arr, mo_e = _random(no, nv, True)
    eris = _container(arr['v'], mo_e, no)
    T3 = P.pack(arr['t3'])
    full = QD.plan(no, nv, 16)
    assert full.xv == no - 1
    ref = QD.rows(eris, arr['t2'], T3, full)
    e0 = QD.kernel(eris, arr['t2'], T3, plan_=full)
    for xv in (3, 4, 5, 6):
        for cols22, cols13, rows in ((1, 1, 1), (3, 7, 2), (full.cols22, full.cols13, full.rows)):
            pl = QD.explicit_plan(no, nv, 16, cols22, cols13, xv, rows)
            for a, b in zip(QD.rows(eris, arr['t2'], T3, pl), ref):
                assert abs(a - b).max() < 1e-12 * abs(b).max(), pl
            e = QD.kernel(eris, arr['t2'], T3, plan_=pl)
            assert abs(e[0] - e0[0]) < 1e-12 * abs(e0[0]) and abs(e[1] - e0[1]) < 1e-12 * abs(e0[1])
    q = QD.Quadruples(eris, arr['t2'], T3, QD.explicit_plan(no, nv, 16, 1, 1, 3, 1))
    sched = q.schedule()
    assert sorted(np.concatenate([Is for _, _, Is in sched])) == list(range(len(q.quads)))
    assert max(len(xs) for _, xs, _ in sched) == 3
    for l, xs, Is in sched:
        assert set(q.quads[Is, :3].ravel()) <= set(xs) and (q.quads[Is, 3] == l).all()


def _packed_data(no, nv, cplx, seed=1):
    '''A container with the packed <vv||vv> only (as GCCSDT has it), random.'''
    rng = np.random.default_rng(seed)

    def rnd(*shape):
        x = rng.standard_normal(shape)
        return x + 1j * rng.standard_normal(shape) if cplx else x
    o, v = no, nv
    mo_e = np.concatenate((np.sort(rng.uniform(-3, -1, o)), np.sort(rng.uniform(1, 3, v))))
    eris = types.SimpleNamespace(
        nocc=no, fock=np.diag(mo_e), mo_energy=mo_e, vvvv=None, oooo=rnd(o, o, o, o), ooov=rnd(o, o, o, v),
        oovv=rnd(o, o, v, v), ovov=rnd(o, v, o, v), ovvo=rnd(o, v, v, o), ovvv=rnd(o, v, v, v))
    eris._vvvv_pp = rnd(v * (v - 1) // 2, v * (v - 1) // 2)
    t2 = rnd(o, o, v, v)
    t2 = t2 - t2.transpose(1, 0, 2, 3)
    return eris, t2 - t2.transpose(0, 1, 3, 2), rnd(o * (o - 1) * (o - 2) // 6, v * (v - 1) * (v - 2) // 6)


@pytest.mark.parametrize('no,nv,cplx', [(5, 20, True), (6, 26, False)])
def test_memory_plan(no, nv, cplx):
    '''plan(): monotone in the budget, the whole budget's worth at the top,
    MemoryError below the floor; the arrays that a run allocates
    (tracemalloc follows numpy's allocations) stay within Plan.need - for
    budgets from the floor up and for the smallest batches.'''
    import tracemalloc
    eris, t2, T3 = _packed_data(no, nv, cplx)
    isz = 16 if cplx else 8
    full = QD.plan(no, nv, isz)
    assert (full.cols22, full.cols13, full.xv) == (nv * (nv - 1) // 2, nv * (nv - 1) * (nv - 2) // 6, no - 1)
    budgets = [int(b) for b in np.linspace(full.floor, 1.1 * full.need, 60)]
    plans = [QD.plan(no, nv, isz, b) for b in budgets]
    key = [(p.cols22, p.cols13, p.xv, p.rows) for p in plans]
    for a, b in zip(key, key[1:]):
        assert all(x <= y for x, y in zip(a, b))
    assert all(p.need <= b for p, b in zip(plans, budgets))
    assert key[-1] == (full.cols22, full.cols13, full.xv, full.rows) and plans[-1].need == full.need
    with pytest.raises(MemoryError, match=r'\(Q\) needs'):
        QD.plan(no, nv, isz, full.floor - 1)
    e0 = QD.kernel(eris, t2, T3, plan_=full)
    least = QD.explicit_plan(no, nv, isz, -(-full.cols22 // QD.PASSES), -(-full.cols13 // QD.PASSES), 3, 1)
    assert least.need < full.need
    seen = []
    for pl in [least, plans[0], plans[len(plans) // 2], full]:
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        e = QD.kernel(eris, t2, T3, plan_=pl)
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        assert abs(e[0] - e0[0]) < 1e-11 * abs(e0[0]) and abs(e[1] - e0[1]) < 1e-11 * abs(e0[1])
        assert peak <= pl.need, (peak, pl)
        seen.append(peak / pl.need)
    # (not a loose bound either: the full plan is met to within the slack)
    assert seen[-1] > 0.8


def _basis(atoms, name):
    if name.lower().startswith('cc-'):
        return name
    return {el: read_genbas(os.path.join(ROOT, 'GENBAS'), '%s:%s' % (el.upper(), name)) for el in atoms}


def _rhf(name):
    molname, basis, frozen = SYSTEMS[name]
    atoms = sorted({a.split()[0] for a in GEOM[molname].split(';')})
    mol = gto.M(atom=GEOM[molname], basis=_basis(atoms, basis), verbose=0)
    return scf.RHF(mol).run(conv_tol=1e-12), frozen


@pytest.mark.parametrize('name', ['h2o_631g', 'n2_631g', 'h2o_ccpvdz_fc'])
def test_gccsdt_q_against_mrcc(name):
    '''zccsdt.CCSDT on GHF orbitals and its ccsdt_q(): CCSDT[Q] and
    CCSDT(Q) of MRCC.'''
    rhf, frozen = _rhf(name)
    cc = zccsdt.CCSDT(scf.addons.convert_to_ghf(rhf), frozen=frozen)
    assert type(cc).__name__ == 'GCCSDT'
    cc.conv_tol, cc.conv_tol_normt = 1e-11, 1e-8
    cc.run()
    ref = REFS[name]['recommended']
    assert cc.converged and abs(cc.e_tot - ref['e_ccsdt']) < 1e-9
    assert cc.ccsdt_q() == cc.e_q
    assert abs(cc.e_tot + cc.e_q5 - ref['e_ccsdt[q]']) < 1e-9
    assert abs(cc.e_tot_q - ref['e_ccsdt(q)']) < 1e-9


def test_complex_spin_orbitals_against_mrcc():
    '''H2O / 6-31G with every pair of spin orbitals mixed by its own SU(2)
    matrix: complex integrals and amplitudes, the energies of MRCC.'''
    from test_ccsdtq_ref import _spin_mixed
    rhf, _ = _rhf('h2o_631g')
    ghf = scf.addons.convert_to_ghf(rhf)
    eris = E.from_scf(ghf, mo_coeff=_spin_mixed(ghf)[0])
    assert abs(np.asarray(eris.oovv).imag).max() > 0.01
    r = solver.kernel(eris, 'ccsdt', solver.Options(conv_tol=1e-11, conv_tol_normt=1e-8))
    assert abs(np.asarray(r.t3).imag).max() > 1e-5
    e5, e6 = QD.kernel(eris, r.t2, r.t3)
    ref = REFS['h2o_631g']['recommended']
    assert abs(r.e_tot + e5 - ref['e_ccsdt[q]']) < 1e-9
    assert abs(r.e_tot + e6 - ref['e_ccsdt(q)']) < 1e-9


def test_x2c_spinors_against_dense_reference():
    '''CH2 / STO-3G with the X2C Hamiltonian: spinors (complex integrals;
    those of a linear molecule on the z axis come out real), through
    zccsdt.CCSDT; the correction of the dense reference from the same
    amplitudes.'''
    mol = gto.M(atom=GEOM['ch2'], basis=_basis(['C', 'H'], 'STO-3G'), verbose=0)
    mf = x2c.UHF(mol).run(conv_tol=1e-12)
    cc = zccsdt.CCSDT(mf)
    cc.conv_tol = 1e-11
    cc.run()
    e_q = cc.ccsdt_q()
    fock, g, mo_e, nocc = R.integrals(E.from_scf(mf))       # (with the full <vv||vv>)
    assert abs(g.imag).max() > 1e-3 and cc.t3.dtype == np.complex128
    t3 = Q.expand(np.asarray(cc.t3).reshape(len(P.tuples(nocc, 3)), -1), nocc, len(mo_e) - nocc, 3)
    e5, e6 = R.quadruples(cc.t2, t3, g, mo_e, nocc)
    assert abs(cc.e_q5 - e5) < 1e-12 and abs(e_q - e6) < 1e-12
    # close to the nonrelativistic number of MRCC, and not it
    nr = REFS['ch2_sto3g']['recommended']
    assert 1e-10 < abs(e_q - (nr['e_ccsdt(q)'] - nr['e_ccsdt'])) < 1e-6


def test_small_spaces_and_other_references():
    '''Fewer than four occupied spin orbitals: no quadruples.  UCCSDT keeps
    T3 in spin blocks: no (Q) there.'''
    mol = gto.M(atom='Li 0 0 0', basis='6-31g', spin=1, verbose=0)
    uhf = scf.UHF(mol).run(conv_tol=1e-12)
    cc = zccsdt.CCSDT(scf.addons.convert_to_ghf(uhf)).run()
    assert cc.ccsdt_q() == 0.0 and cc.e_q5 == 0.0
    with pytest.raises(NotImplementedError, match='GCCSDT'):
        zccsdt.CCSDT(uhf).run().ccsdt_q()
