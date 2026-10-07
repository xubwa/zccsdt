'''Spin-orbital CCSDTQ with packed T3 and T4 (zccsdt.cc.ccsdtq, t4engine,
t4pack): views and reductions of the packed block, the T4 equation piece by
piece and as a whole against the term table, the dressing of the integrals
by blocks, converged energies against MRCC and the amplitudes of an X2C
calculation in the equations of the dense reference.'''
import json
import os

import numpy as np
import pytest
from pyscf import gto, scf, x2c

import fockspace as F
from zccsdt import eris as E
from zccsdt.cc import ccsdtq, ccsdtq_ref as R, ccsdtq_terms as Q, solver, t4engine as T, t4pack as P
from zccsdt.genbas import read_genbas

ROOT = os.path.join(os.path.dirname(__file__), '..', 'bench', 'refs')
REFS = json.load(open(os.path.join(ROOT, 'reference_energies.json')))
GEOM = {'bh': 'B 0 0 0; H 0 0 1.2324', 'ch2': 'C 0 0 0; H 0 0.87 0.67; H 0 -0.87 0.67',
        'h2o': 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', 'n2': 'N 0 0 0; N 0 0 1.0977'}


def _random(no, nv, seed=3):
    '''Complex amplitudes, complex Hermitian f and <pq||rs>.'''
    n = no + nv
    rng = np.random.default_rng(seed)
    vr, vi = F.random_integrals(rng, n), F.random_integrals(rng, n)
    v = vr + 1j * (vi - vi.transpose(2, 3, 0, 1))
    f = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    arr = {'v': v, 'f': f + f.conj().T}
    for k in (2, 3, 4):
        arr['t%d' % k] = F.random_antisym(rng, no, nv, k) + 1j * F.random_antisym(rng, no, nv, k)
    return arr


def test_views_and_reductions():
    '''t4pack.view is the full array at sorted groups, t4pack.reduce the
    antisymmetrizer of an array given at sorted groups - for every kind of
    split, whole and in batches of the second groups.'''
    no, nv = 5, 7
    rng = np.random.default_rng(2)
    t4 = _random(no, nv)['t4']
    T = P.pack(t4)
    assert abs(P.unpack(T, no, nv, 4) - t4).max() < 1e-11
    for ko, kv in (((1, 3), (1, 3)), ((2, 2), (2, 2)), ((1, 3), (2, 2)), ((2, 2), (1, 3)),
                   ((1, 3), (3, 1)), (None, (2, 2)), ((2, 2), None)):
        so = P.split(no, *ko) if ko else None
        sv = P.split(nv, *kv) if kv else None
        if so is None:
            oi = [P.tuples(no, 4)[:, x].reshape(-1, 1, 1) for x in range(4)]
        else:
            o1, o2 = P.tuples(no, ko[0]), P.tuples(no, ko[1])
            tail = (None,) * (2 if sv else 1)
            oi = ([o1[:, x][(slice(None), None) + tail] for x in range(ko[0])]
                  + [o2[:, x][(None, slice(None)) + tail] for x in range(ko[1])])
        if sv is None:
            vi = [P.tuples(nv, 4)[:, x] for x in range(4)]
        else:
            v1, v2 = P.tuples(nv, kv[0]), P.tuples(nv, kv[1])
            vi = [v1[:, x][:, None] for x in range(kv[0])] + [v2[:, x][None, :] for x in range(kv[1])]
        idx = tuple(oi) + tuple(vi)
        want = t4[idx]
        assert abs(P.view(T, so, sv) - want).max() < 1e-11, (ko, kv)
        n2o, n2v = (so.n2 if so else T.shape[0]), (sv.n2 if sv else T.shape[1])
        jo, jv = (1, min(4, n2o)), (2, min(9, n2v))
        part = want[(slice(None), slice(*jo)) if so else (slice(*jo),)][(Ellipsis, slice(*jv))]
        assert abs(P.view(T, so, sv, jo, jv) - part).max() < 1e-11, (ko, kv)
        X = rng.standard_normal(want.shape) + 1j * rng.standard_normal(want.shape)
        full = np.zeros_like(t4)
        full[idx] = X
        ref = Q.asym_unique(full, 4)
        assert abs(P.reduce(np.zeros_like(T), X, so, sv) - ref).max() < 1e-11, (ko, kv)
        Rb = np.zeros_like(T)
        for b0 in range(0, n2o, 5):
            for c0 in range(0, n2v, 11):
                b, c = (b0, min(n2o, b0 + 5)), (c0, min(n2v, c0 + 11))
                Xb = X[(slice(None), slice(*b)) if so else (slice(*b),)][(Ellipsis, slice(*c))]
                P.reduce(Rb, Xb, so, sv, b, c)
        assert abs(Rb - ref).max() < 1e-11, (ko, kv)


@pytest.mark.parametrize('no,nv,batch,cplx', [(5, 6, None, True), (5, 6, 3e4, True), (6, 5, 1e5, True),
                                              (4, 8, 1e5, False), (6, 6, None, False)])
def test_residual_equals_table(no, nv, batch, cplx):
    '''The residual of T4 on the packed block, the T4 terms of R3 and R2:
    the terms of the table, in one batch and in many.'''
    arr = _random(no, nv)
    if not cplx:
        arr = {k: np.ascontiguousarray(x.real) for k, x in arr.items()}
    ints = T.Dressed.from_dense(arr['f'], arr['v'], no)
    R4, R3, R2 = T.residual(arr['t2'], P.pack(arr['t3']), P.pack(arr['t4']), ints, batch and int(batch))
    assert R4.dtype == (np.complex128 if cplx else np.float64)
    want = Q.unique(Q.residual(4, arr, no), 4)
    assert abs(R4 - want).max() < 1e-11 * abs(want).max()
    want = Q.asym_unique(Q.total(Q.GROUPS['R3: [F,T4]'][4] + Q.GROUPS['R3: [V,T4]'][4], arr, no, 3), 3)
    assert abs(R3 - want).max() < 1e-11 * abs(want).max()
    want = Q.asym_unique(Q.total(Q.GROUPS['R2: [V,T4]'][4], arr, no, 2), 2)
    assert abs(R2 - want).max() < 1e-11 * abs(want).max()


def test_pieces_of_the_residual():
    '''Each product of the engine is the group of terms it stands for; the
    three terms of T4 in R3 come with the ring and the two products of T2
    with a six-index intermediate.'''
    no, nv = 5, 6
    arr = _random(no, nv)
    T3, T4 = P.pack(arr['t3']), P.pack(arr['t4'])
    ints = T.Dressed.from_dense(arr['f'], arr['v'], no)
    G = Q.GROUPS

    def terms(*picks):
        return [G[name][4][x] for name, idx in picks for x in idx]
    pieces = {
        'pp': terms(('R4: [F,T4]', [0]), ('R4: [V,T4]', [0]), ('R4: [[V,T2],T4]', [2, 5])),
        'hh': terms(('R4: [F,T4]', [1]), ('R4: [V,T4]', [1]), ('R4: [[V,T2],T4]', [3, 6])),
        'ring': terms(('R4: [V,T4]', [2]), ('R4: [[V,T2],T4]', [4])),
        'b1': terms(('R4: [V,T3]', [0]), ('R4: [[V,T2],T3]', [0, 2, 4, 7]), ('R4: [[F,T2],T3]', [0]),
                    ('R4: 1/2 [[V,T3],T3]', [1]), ('R4: 1/2 [[V,T2],T2]', [1]),
                    ('R4: 1/6 [[[V,T2],T2],T2]', [1]), ('R4: [[V,T2],T4]', [0])),
        'b2': terms(('R4: [V,T3]', [1]), ('R4: [[V,T2],T3]', [1, 3, 5, 6]), ('R4: [[F,T2],T3]', [1]),
                    ('R4: 1/2 [[V,T3],T3]', [0]), ('R4: 1/2 [[V,T2],T2]', [0, 2]),
                    ('R4: 1/6 [[[V,T2],T2],T2]', [0, 2]), ('R4: [[V,T2],T4]', [1])),
        't3t3h': terms(('R4: 1/2 [[V,T3],T3]', [2])),
        't3t3r': terms(('R4: 1/2 [[V,T3],T3]', [3])),
    }
    assert sum(len(t) for t in pieces.values()) == 34 and set(pieces) | {'r2'} == set(T.PIECES)
    for piece, ts in pieces.items():
        want = Q.asym_unique(Q.total(ts, arr, no, 4), 4)
        got = T.residual(arr['t2'], T3, T4, ints, only={piece})[0]
        assert abs(got - want).max() < 1e-11 * abs(want).max(), piece
    for piece, pick in (('ring', ('R3: [F,T4]', [0])), ('b2', ('R3: [V,T4]', [0])), ('b1', ('R3: [V,T4]', [1]))):
        want = Q.asym_unique(Q.total(terms(pick), arr, no, 3), 3)
        got = T.residual(arr['t2'], T3, T4, ints, only={piece})[1]
        assert abs(got - want).max() < 1e-11 * abs(want).max(), piece


def _scf(name, basis):
    atoms = sorted({a.split()[0] for a in GEOM[name].split(';')})
    bas = {el: read_genbas(os.path.join(ROOT, 'GENBAS'), '%s:%s' % (el.upper(), basis)) for el in atoms}
    return scf.RHF(gto.M(atom=GEOM[name], basis=bas, verbose=0)).run(conv_tol=1e-13)


@pytest.mark.parametrize('kind', ['ghf', 'x2c'])
def test_dressing_by_blocks(kind):
    '''Dressed.from_eris - the T1-dressed blocks from the container, with
    the full and with the packed <vv||vv> - is the dressing of the full
    tensor.'''
    from zccsdt.cc import ccsdt
    mol = _scf('h2o', 'STO-3G').mol
    mf = scf.addons.convert_to_ghf(scf.RHF(mol).run(conv_tol=1e-12)) if kind == 'ghf' else x2c.UHF(mol).run()
    full, packed = E.from_scf(mf), E.from_scf(mf, packed_vvvv=True)
    fock, g, mo_e, nocc = R.integrals(full)
    rng = np.random.default_rng(0)
    t1 = 0.05 * rng.standard_normal((nocc, len(mo_e) - nocc))
    if kind == 'x2c':
        t1 = t1 + 0.05j * rng.standard_normal(t1.shape)
    ge, fdr = ccsdt._t1_dress(g, fock, t1, nocc, len(mo_e))
    a = T.Dressed.from_dense(fdr, ge, nocc)
    for eris in (full, packed):
        b = T.Dressed.from_eris(eris, t1)
        for k in ('f', 'oovv', 'oooo', 'oovo', 'ovvo', 'vovv', 'ovoo', 'vvvo_p', 'vvvv_pp'):
            assert abs(np.asarray(getattr(a, k)) - np.asarray(getattr(b, k))).max() < 1e-12, k


@pytest.mark.parametrize('name,basis,how', [('bh', 'STO-3G', 'mp2'), ('bh', 'STO-3G', 'ccsdt'),
                                            ('bh', 'STO-3G', 'mixed'), ('ch2', 'STO-3G', 'ccsdt'),
                                            ('n2', 'STO-3G', 'ccsdt'), ('h2o', '6-31G', 'ccsdt')])
def test_ccsdtq_against_mrcc(name, basis, how):
    '''Converged CCSDTQ energies against MRCC: from the amplitudes of MP2
    or of CCSDT; with the packed <vv||vv>; BH also with complex spin
    orbitals (every spin pair mixed by its own SU(2) matrix).'''
    ghf = scf.addons.convert_to_ghf(_scf(name, basis))
    if how == 'mixed':
        from test_ccsdtq_ref import _spin_mixed
        eris = E.from_scf(ghf, mo_coeff=_spin_mixed(ghf)[0])
    else:
        eris = E.from_scf(ghf, packed_vvvv=True)
    amps = {}
    if how != 'mp2':
        r3 = solver.kernel(eris, 'ccsdt', solver.Options(conv_tol=1e-11, conv_tol_normt=1e-8))
        amps = dict(t1=r3.t1, t2=r3.t2, t3=r3.t3)
    r = ccsdtq.kernel(eris, ccsdtq.Options(conv_tol=1e-11, conv_tol_normt=1e-9), **amps)
    ref = REFS['%s_%s' % (name, basis.lower().replace('-', ''))]['recommended']
    assert r.converged
    assert abs(r.e_tot - ref['e_ccsdtq']) < 5e-11
    assert r.t4.dtype == (np.complex128 if how == 'mixed' else np.float64)
    assert r.t4.shape == (len(P.tuples(eris.nocc, 4)), len(P.tuples(eris.nmo - eris.nocc, 4)))


def test_x2c_spinors_solve_the_reference_equations():
    '''CH2 / STO-3G with the X2C Hamiltonian (spinors, complex integrals):
    the converged amplitudes make the residuals of the dense reference
    (the term table on the full dressed tensor) vanish.'''
    mol = _scf('ch2', 'STO-3G').mol
    mf = x2c.UHF(mol).run(conv_tol=1e-12)
    eris = E.from_scf(mf, packed_vvvv=True)
    r = ccsdtq.kernel(eris, ccsdtq.Options(conv_tol=1e-11, conv_tol_normt=1e-9))
    assert r.converged and r.t4.dtype == np.complex128
    fock, g, mo_e, nocc = R.integrals(E.from_scf(mf))
    nvir = len(mo_e) - nocc
    assert abs(g.imag).max() > 1e-3
    amps = [r.t1, r.t2, Q.expand(np.asarray(r.t3).reshape(len(P.tuples(nocc, 3)), -1), nocc, nvir, 3),
            Q.expand(r.t4, nocc, nvir, 4)]
    for k, res in enumerate(R.residuals(amps, fock, g, nocc)):
        assert abs(res).max() < 1e-8, k
    assert abs(R.energy(r.t1, r.t2, fock, g, nocc) - r.e_corr) < 1e-11
