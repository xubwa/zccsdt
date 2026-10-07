'''CCSDTQ(P) on spin orbitals or spinors (zccsdt.cc.pentuples): every
product against the term table, every memory plan against the plain one,
peak allocations against the plan, and energies against MRCC.  (Spinors
with spin-orbit splitting against MRCC on the same integrals:
tests/test_dirac.py; through the front end: tests/test_frontend.py.)'''
import gc
import json
import os
import types

import numpy as np
import pytest
from pyscf import gto, scf

import fockspace as F
from zccsdt import eris as E
from zccsdt.cc import ccsdtq, ccsdtq_ref as R, ccsdtq_terms as Q, pentuples as PT, quadruples as QD, t4pack as P
from zccsdt.genbas import read_genbas

ROOT = os.path.join(os.path.dirname(__file__), '..', 'bench', 'refs')
REFS = json.load(open(os.path.join(ROOT, 'reference_energies.json')))
GEOM = {'bh': 'B 0 0 0; H 0 0 1.2324', 'ch2': 'C 0 0 0; H 0 0.87 0.67; H 0 -0.87 0.67', 'n2': 'N 0 0 0; N 0 0 1.0977'}


def _random(no, nv, cplx, seed=7):
    '''(arrays of the term table, orbital energies, container, packed T3
    and T4): antisymmetric amplitudes, Hermitian <pq||rs>.'''
    rng = np.random.default_rng(seed)
    vr = F.random_integrals(rng, no + nv)
    if cplx:
        vi = F.random_integrals(rng, no + nv)
        vr = vr + 1j * (vi - vi.transpose(2, 3, 0, 1))
    arr = {'v': vr}
    for k in (2, 3, 4):
        t = F.random_antisym(rng, no, nv, k)
        arr['t%d' % k] = t + 1j * F.random_antisym(rng, no, nv, k) if cplx else t
    mo_e = np.concatenate((np.sort(rng.uniform(-3, -1, no)), np.sort(rng.uniform(1, 3, nv))))
    o, v = slice(0, no), slice(no, None)
    blk = {k: np.ascontiguousarray(vr[tuple(o if c == 'o' else v for c in k)])
           for k in ('oooo', 'ooov', 'oovv', 'ovov', 'ovvo', 'ovvv', 'vvvv')}
    eris = types.SimpleNamespace(nocc=no, fock=np.diag(mo_e).astype(vr.dtype), mo_energy=mo_e, **blk)
    return arr, mo_e, eris, P.pack(arr['t3']), P.pack(arr['t4'])


@pytest.mark.parametrize('no,nv,cplx,npent', [(6, 6, True, 6), (7, 6, False, 3)])
def test_products_equal_terms(no, nv, cplx, npent):
    '''Each of the eight terms of r and the two parts of z, for occupied
    pentuples and all virtual ones, is its term of the table (the one with
    Y4: the term of T4 on the second-order quadruples); the energy is that
    of the dense reference.'''
    arr, mo_e, eris, T3, T4 = _random(no, nv, cplx)
    pents = P.tuples(no, 5)[:npent]
    y4 = QD.second_order_quadruples(eris, arr['t2'])
    with_y = dict(arr, t4=P.unpack(y4, no, nv, 4))
    G = Q.GROUPS
    cases = [('a', 0, G['R5: [V,T4]'][4][0], arr), ('b', 0, G['R5: [V,T4]'][4][1], arr)]
    cases += [(x, 0, G['R5: [[V,T2],T3]'][4][k], arr) for k, x in enumerate('cdef')]
    cases += [(x, 0, G['R5: 1/6 [[[V,T2],T2],T2]'][4][k], arr) for k, x in enumerate('gh')]
    cases += [('zt3', 1, Q.LEFT['Z5.T3'][2][0], arr), ('zy', 1, None, with_y)]
    for name, side, term, arrays in cases:
        got = PT.rows(eris, arr['t2'], T3, T4, only={name})
        terms = Q.LEFT['Z5.T4'][2] if term is None else [term]
        want = np.array([R._block(terms, arrays, no, 5, tuple(I)) for I in pents])
        assert abs(got[side][:npent] - want).max() < 1e-12 * abs(want).max(), name
        assert not got[1 - side].any(), name
    if npent == len(P.tuples(no, 5)):
        e = PT.kernel(eris, arr['t2'], T3, T4)
        ref = R.pentuples(arr['t2'], arr['t3'], arr['t4'], arr['v'], mo_e, no)
        assert abs(e - ref) < 1e-12 * abs(ref)


def test_every_plan_gives_the_same():
    '''Products in batches of columns, the intermediates in steps of a few
    rows, the T2 T2 T2 intermediate kept for all occupied quadruples or made
    for those of a pentuple when it comes.'''
    no, nv = 7, 6
    arr, mo_e, eris, T3, T4 = _random(no, nv, True, seed=11)
    full = PT.plan(no, nv, 16)
    assert full.store_xq and (full.cols23, full.cols14) == (20, 15)
    ref = PT.rows(eris, arr['t2'], T3, T4, full)
    e0 = PT.kernel(eris, arr['t2'], T3, T4, plan_=full)
    for store in (True, False):
        for cols23, cols14, rows in ((1, 1, 1), (3, 5, 2), (7, 2, 4), (full.cols23, full.cols14, full.rows)):
            pl = PT.explicit_plan(no, nv, 16, cols23, cols14, rows, store)
            for a, b in zip(PT.rows(eris, arr['t2'], T3, T4, pl), ref):
                assert abs(a - b).max() < 1e-12 * abs(b).max(), pl
            assert abs(PT.kernel(eris, arr['t2'], T3, T4, plan_=pl) - e0) < 1e-12 * abs(e0)
    # the second-order quadruples and Xv as arrays of a scratch file
    for y4_file in (False, True):
        pl = PT.explicit_plan(no, nv, 16, 3, 5, 2, False, y4_file, True)
        assert pl.xv_file and pl.y4_file == y4_file
        for a, b in zip(PT.rows(eris, arr['t2'], T3, T4, pl), ref):
            assert abs(a - b).max() < 1e-12 * abs(b).max(), pl
        assert abs(PT.kernel(eris, arr['t2'], T3, T4, plan_=pl) - e0) < 1e-12 * abs(e0)
    # fewer than five occupied or virtual spin orbitals: nothing
    arr, mo_e, eris, T3, T4 = _random(4, 6, False)
    assert PT.kernel(eris, arr['t2'], T3, T4) == 0.0


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
    return (eris, t2 - t2.transpose(0, 1, 3, 2), rnd(P.tuples(o, 3).shape[0], P.tuples(v, 3).shape[0]),
            rnd(P.tuples(o, 4).shape[0], P.tuples(v, 4).shape[0]))


@pytest.mark.parametrize('no,nv,cplx', [(6, 18, True), (7, 22, False)])
def test_memory_plan(no, nv, cplx):
    '''plan(): monotone in the budget, as large as fits, MemoryError below
    the floor; the arrays that a run allocates (tracemalloc follows numpy's
    allocations) stay within Plan.need - from the floor up, with the
    intermediate of the quadruples kept and made on the way.'''
    import tracemalloc
    eris, t2, T3, T4 = _packed_data(no, nv, cplx)
    isz = 16 if cplx else 8
    full = PT.plan(no, nv, isz, batch_bytes=1 << 40)
    nA3, nA4 = P.tuples(nv, 3).shape[0], P.tuples(nv, 4).shape[0]
    assert (full.cols23, full.cols14, full.store_xq) == (nA3, nA4, True)
    budgets = [int(b) for b in np.linspace(full.floor, 1.1 * full.need, 80)]
    plans = [PT.plan(no, nv, isz, b, batch_bytes=1 << 40) for b in budgets]
    key = [(not p.y4_file, not p.xv_file, p.store_xq, p.cols23, p.cols14, p.rows) for p in plans]
    for a, b in zip(key, key[1:]):
        assert all(x <= y for x, y in zip(a, b))
    assert all(p.need <= b for p, b in zip(plans, budgets))
    assert key[-1] == (True, True, True, full.cols23, full.cols14, full.rows) and plans[-1].need == full.need
    # at the floor the second-order quadruples and Xv are arrays of a scratch file: Xv alone
    # a little above it, none of the two where both fit
    assert plans[0].need == full.floor == min(PT._floors(no, nv, isz)) - sum(PT._big(no, nv, isz))
    assert plans[0].y4_file and plans[0].xv_file and not full.y4_file and not full.xv_file
    assert {(p.y4_file, p.xv_file) for p in plans} == {(True, True), (False, True), (False, False)}
    with pytest.raises(MemoryError, match=r'\(P\) needs'):
        PT.plan(no, nv, isz, full.floor - 1)
    e0 = PT.kernel(eris, t2, T3, T4, plan_=full)
    seen = []
    on_the_way = [PT.explicit_plan(no, nv, isz, *PT._least(no, nv), False),
                  PT.explicit_plan(no, nv, isz, full.cols23, full.cols14, full.rows, False)]
    for pl in [plans[0], plans[len(plans) // 2]] + on_the_way + [full]:
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        e = PT.kernel(eris, t2, T3, T4, plan_=pl)
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        assert abs(e - e0) < 1e-11 * abs(e0)
        assert peak <= pl.need, (peak, pl)
        seen.append(peak / pl.need)
    # (not a loose bound either)
    assert seen[-1] > 0.7, seen


def _scf(name, basis):
    atoms = sorted({a.split()[0] for a in GEOM[name].split(';')})
    bas = {el: read_genbas(os.path.join(ROOT, 'GENBAS'), '%s:%s' % (el.upper(), basis)) for el in atoms}
    return scf.RHF(gto.M(atom=GEOM[name], basis=bas, verbose=0)).run(conv_tol=1e-13)


@pytest.mark.parametrize('name,how', [('bh', 'real'), ('bh', 'mixed'), ('ch2', 'real'), ('n2', 'real')])
def test_against_mrcc(name, how):
    '''CCSDTQ(P) of MRCC (STO-3G: 6 + 6, 8 + 6 and 14 + 6 spin orbitals);
    BH also with complex spin orbitals (every spin pair mixed by its own
    SU(2) matrix), where T4 and the products are complex.'''
    ghf = scf.addons.convert_to_ghf(_scf(name, 'STO-3G'))
    if how == 'mixed':
        from test_ccsdtq_ref import _spin_mixed
        eris = E.from_scf(ghf, mo_coeff=_spin_mixed(ghf)[0], packed_vvvv=True)
    else:
        eris = E.from_scf(ghf, packed_vvvv=True)
    r = ccsdtq.kernel(eris, ccsdtq.Options(conv_tol=1e-12, conv_tol_normt=1e-9))
    ref = REFS['%s_sto3g' % name]['recommended']
    assert r.converged and abs(r.e_tot - ref['e_ccsdtq']) < 5e-11
    e_p = PT.kernel(eris, r.t2, r.t3, r.t4)
    want = ref['e_ccsdtq(p)'] - ref['e_ccsdtq']
    assert abs(e_p - want) < 3e-12 and abs(e_p) > 1e-9
    assert abs(r.e_tot + e_p - ref['e_ccsdtq(p)']) < 5e-11
    assert r.t4.dtype == (np.complex128 if how == 'mixed' else np.float64)


def test_open_shell_spinors_against_the_dense_reference():
    '''BeH / STO-3G, a doublet, turned away from the z axis, with the X2C
    Hamiltonian: five occupied and seven virtual spinors that are not
    Kramers pairs, complex integrals.  The amplitudes of CCSDTQ solve the
    equations of the dense reference, and (Q) and (P) are its numbers - the
    general case of the code, which no other program here takes.'''
    from pyscf import x2c
    from zccsdt.cc import solver
    mol = gto.M(atom='Be 0 0 0; H 0.5 0.7 1.03', basis='sto-3g', spin=1, verbose=0)
    mf = x2c.UHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-9
    mf.kernel()
    eris = E.from_scf(mf, packed_vvvv=True)
    no, nv = eris.nocc, eris.nmo - eris.nocc
    # (the SCF stops short of its flag at these thresholds; the Fock matrix is diagonal)
    assert abs(eris.fock - np.diag(eris.fock.diagonal())).max() < 1e-7
    assert (no, nv) == (5, 7) and abs(np.asarray(eris.oovv).imag).max() > 1e-5
    e = np.sort(mf.mo_energy)
    assert abs(e[1::2] - e[:-1:2]).max() > 1e-2                         # (no pairs: an open shell)
    r3 = solver.kernel(eris, 'ccsdt', solver.Options(conv_tol=1e-12, conv_tol_normt=1e-9))
    e_q5, e_q = QD.kernel(eris, r3.t2, r3.t3)
    r4 = ccsdtq.kernel(eris, ccsdtq.Options(conv_tol=1e-12, conv_tol_normt=1e-9), t1=r3.t1, t2=r3.t2, t3=r3.t3)
    assert r3.converged and r4.converged and r4.t4.dtype == np.complex128
    e_p = PT.kernel(eris, r4.t2, r4.t3, r4.t4)
    fock, g, mo_e, nocc = R.integrals(E.from_scf(mf))
    dense = lambda t, k: Q.expand(np.asarray(t).reshape(len(P.tuples(no, k)), -1), no, nv, k)
    ref = R.quadruples(r3.t2, dense(r3.t3, 3), g, mo_e, nocc)
    assert abs(e_q5 - ref[0]) < 1e-12 and abs(e_q - ref[1]) < 1e-12 and abs(e_q) > 1e-8
    amps = [r4.t1, r4.t2, dense(r4.t3, 3), dense(r4.t4, 4)]
    for k, res in enumerate(R.residuals(amps, fock, g, nocc)):
        assert abs(res).max() < 1e-8, k
    ref_p = R.pentuples(amps[1], amps[2], amps[3], g, mo_e, nocc)
    assert abs(e_p - ref_p) < 1e-13 and abs(e_p) > 1e-12
