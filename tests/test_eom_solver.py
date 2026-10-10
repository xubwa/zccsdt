'''The vector operations of the EOM Davidson solver on threads
(zccsdt.cc.eom: transform, the preconditioner, mdot; csrc/t3pack.c
zc_vtransform, zc_vprecond) against numpy; a degenerate level is found
whole by the solver; on the packed block the states of one label shift are
selected by masks (eom.Sector; GCCSDT.eom_ee(shift=) with the spin label
of spin orbitals of pure spin): a triplet is computed once.'''
import json
import os

import numpy as np
import pytest
from pyscf import scf

import zccsdt
from test_quadruples_lambda import _mf
from zccsdt.cc import diis, eom, t3pack

REFS = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'bench', 'refs', 'reference_energies.json')))['eom_ee']


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_transform_in_place(dtype):
    '''The first m vectors become V Q, in place, for parts of lengths that
    are not multiples of the block of the kernel; with and without it.'''
    rng = np.random.default_rng(1)
    n, m, sizes = 7, 3, [5, 1000, 70001, 0]

    def rnd(*shape):
        x = rng.standard_normal(shape)
        return (x + 1j * rng.standard_normal(shape)).astype(dtype) if dtype == np.complex128 else x
    vecs = [[rnd(k) for k in sizes] for _ in range(n)]
    Q = rnd(n, m)
    ref = [[sum(vecs[j][p] * Q[j, k] for j in range(n)) for p in range(len(sizes))] for k in range(m)]
    keep = [[x.copy() for x in v] for v in vecs]
    assert diis._vec_c('vtransform', np.dtype(dtype)) is not None
    eom.transform(vecs, Q)
    for k in range(m):
        for p in range(len(sizes)):
            assert abs(vecs[k][p] - ref[k][p]).max(initial=0) < 1e-12
    for k in range(m, n):       # (the others are left as they were)
        assert all(np.array_equal(a, b) for a, b in zip(vecs[k], keep[k]))
    # the numpy path: strided parts are not given to the kernel
    wide = [[np.repeat(x, 2)[::2] for x in v] for v in keep]
    assert not wide[0][1].flags.c_contiguous
    eom.transform(wide, Q)
    for k in range(m):
        assert abs(wide[k][2] - ref[k][2]).max() < 1e-12
    # products with many vectors in one pass
    a, xs = keep[0], keep[1:]
    assert abs(eom.mdot(a, xs) - [eom.vdot(a, x) for x in xs]).max() < 1e-10
    assert abs(eom.mdot(a, xs) - [sum(np.vdot(a[p], x[p]) for p in range(3)) for x in xs]).max() < 1e-10


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_preconditioner_kernel(dtype):
    '''x[I, A] / (omega + eo[I] - ev[A]) with small denominators replaced.'''
    rng = np.random.default_rng(2)
    nI, nA, floor = 37, 501, 1e-3
    eo, ev = rng.standard_normal(nI), 2 + rng.standard_normal(nA)
    omega = 2.0 - eo[3] + ev[5] - 2.0 + 1e-5        # (one denominator below the floor)
    omega = omega + (0.3j if dtype == np.complex128 else 0)
    x = rng.standard_normal((nI, nA)).astype(dtype)
    den = omega + eo[:, None] - ev[None, :]
    if dtype == np.float64:
        assert abs(den[3, 5]) < floor
    den = np.where(abs(den) < floor, floor, den)
    ref = x / den
    f = diis._vec_c('vprecond', np.dtype(dtype))
    import ctypes
    f(t3pack._p(x), ctypes.c_int64(nI), ctypes.c_int64(nA), t3pack._p(eo), t3pack._p(ev),
      t3pack._p(np.array([omega], dtype=dtype)), ctypes.c_double(floor))
    assert abs(x - ref).max() < 1e-12 * abs(ref).max()


def _model(n, levels, seed=0, nonsym=1e-3):
    '''A nearly symmetric matrix with the eigenvalues ``levels`` (lists of
    degenerate ones) at the bottom of its spectrum, as an operator on
    vectors of three parts.'''
    rng = np.random.default_rng(seed)
    w = np.concatenate([np.full(k, e) for e, k in levels] + [np.linspace(2.0, 9.0, n - sum(k for _, k in levels))])
    U = np.linalg.qr(rng.standard_normal((n, n)))[0]
    S = np.eye(n) + nonsym * rng.standard_normal((n, n))
    A = S @ (U * w) @ U.T @ np.linalg.inv(S)
    cut = [3, n // 3]

    def split(x):
        return [x[:cut[0]].copy(), x[cut[0]:cut[1]].copy(), x[cut[1]:].copy()]

    def apply(x, y):
        out = A @ np.concatenate(x)
        for p, part in enumerate(split(out)):
            y[p][:] = part
    d = np.diag(A).copy()

    def precond(x, omega):
        den = omega - d
        den[abs(den) < 1e-3] = 1e-3
        for p, part in enumerate(split(np.concatenate(x) / den)):
            x[p][:] = part
        return x
    return A, w, apply, precond, [cut[0], cut[1] - cut[0], n - cut[1]], d


def test_degenerate_levels_are_found_whole():
    '''Davidson on a model with a threefold and a twofold level: asked for
    the six lowest roots it returns the triplet three times, the single
    root and the pair, each level with orthonormal vectors.'''
    A, w, apply, precond, sizes, d = _model(300, [(0.3, 3), (0.5, 1), (0.6, 2)])
    store = eom.Store(sizes, np.float64)
    gs = []
    for k in np.argsort(d)[:8]:
        v = store.new()
        full = np.zeros(300)
        full[k] = 1.0
        v[0][:], v[1][:], v[2][:] = full[:3], full[3:100], full[100:]
        gs.append(v)
    res = eom.davidson(apply, precond, gs, store, 6, conv_tol=1e-9, max_space=40, max_cycle=200)
    assert res.converged.all()
    e = np.sort(res.e.real)
    assert abs(e - w[:6]).max() < 1e-9
    X = np.array([np.concatenate(v) for v in res.vectors])
    assert abs(A @ X.T - X.T * res.e.real).max() < 1e-7
    assert np.linalg.matrix_rank(X, tol=1e-6) == 6


def test_states_of_one_spin_label_on_the_packed_block():
    '''H2O / 6-31G, RHF as GHF spin orbitals (label 2 m_s), all electrons,
    EOM-CCSDT against MRCC: the states of shift 2 are the M_S = 1
    components of the triplets, each once (the two lowest triplets, 1e-9);
    shift 0 has the M_S = 0 component of the lowest triplet and the lowest
    singlet; shift -2 is not computed; the vectors have their elements in
    the sector only.  Without a shift the four lowest roots are the triplet
    three times and the singlet.'''
    ref = REFS['h2o_631g']['ccsdt_all_electron']
    cc = zccsdt.CCSDT(scf.addons.convert_to_ghf(_mf('h2o_631g')))
    cc.verbose, cc.conv_tol, cc.conv_tol_normt = 0, 1e-12, 1e-10
    cc.kernel()
    assert abs(cc.e_tot - ref['e_ccsdt']) < 1e-9
    e, r = cc.eom_ee(nroots=[2, 2, 2], conv_tol=1e-8, shift=[2, 0, -2])
    assert r[2].converged.all() and r[0].converged.all() and r[-2].partner_of == 2 and r[2].shift is None
    assert abs(e[2] - ref['triplets'][:2]).max() < 1e-9
    assert abs(e[0] - [ref['triplets'][0], ref['singlets'][0]]).max() < 1e-9
    assert np.array_equal(e[-2], e[2])
    g = r[2].ground
    sec = eom.Sector(g.om, g.vm, cc._state_labels(cc._eris), 2)
    x = [r[2].r1[0].ravel().copy(), r[2].r2[0].ravel().copy(), r[2].r3[0].copy()]
    y = sec.project([v.copy() for v in x])
    assert all(np.array_equal(a, b) for a, b in zip(x, y)) and abs(x[0]).max() > 0.3
    assert 0 < sec.n3 < x[2].size / 3 and np.count_nonzero(x[2]) <= sec.n3
    assert eom.label_shifts(cc._eris, cc._state_labels(cc._eris)) == [-2, 0, 2]
    # complex spin orbitals without a spin label: no sectors
    with pytest.raises(ValueError, match='no symmetry labels for the states'):
        from test_ccsdtq_ref import _spin_mixed
        c2 = zccsdt.CCSDT(cc._scf, mo_coeff=_spin_mixed(cc._scf)[0])
        c2.verbose, c2.max_cycle = 0, 1
        c2.kernel()
        c2.eom_ee(nroots=1, shift=2)


def test_sixfold_level_and_a_cut_level():
    '''F- / 6-31G, RHF as GHF spin orbitals, 1s frozen, EOM-IP-CCSDT with
    short vectors: the lowest level, 2p^-1, is sixfold.  Six roots are six
    times the same energy (the first version matched the roots of two
    iterations one to one and lost the sixth to a vector far above); four
    roots cut the level - a warning and Result.cut_level -, and with
    complete=True all six come back.'''
    from pyscf import gto
    mf = scf.RHF(gto.M(atom='F 0 0 0', basis='6-31g', charge=-1, verbose=0)).run(conv_tol=1e-12)
    cc = zccsdt.CCSDT(mf.to_ghf(), frozen=2)
    cc.verbose, cc.symmetry, cc.conv_tol, cc.conv_tol_normt = 0, False, 1e-11, 1e-9
    cc.kernel()
    e, res = cc.eom_ip(nroots=6, conv_tol=1e-8)
    assert res.converged.all() and len(e) == 6 and e.max() - e.min() < 1e-9 and 0 < e[0] < 0.2 and not res.cut_level
    with pytest.warns(UserWarning, match='cut a degenerate level'):
        e4, r4 = cc.eom_ip(nroots=4, conv_tol=1e-8)
    assert len(e4) == 4 and r4.cut_level and abs(e4 - e[0]).max() < 1e-8
    e6, r6 = cc.eom_ip(nroots=4, conv_tol=1e-8, complete=True)
    assert len(e6) == 6 and r6.converged.all() and abs(e6 - e[0]).max() < 1e-8 and not r6.cut_level
    # the next level (2s^-1, twofold) after it
    e8, r8 = cc.eom_ip(nroots=8, conv_tol=1e-8)
    assert r8.converged.all() and abs(e8[:6] - e[0]).max() < 1e-8 and abs(e8[6] - e8[7]) < 1e-8 and e8[6] - e8[5] > 0.1


def test_more_roots_than_singles():
    '''Li / 6-31G, UHF as GHF, three electrons (CCSDT and the IP vectors
    are exact; the numbers are differences of FCI energies): four roots
    asked for with three single excitations - the guess goes on with
    doubles and four come back (the first version returned three without a
    word): the ground state of Li+ and the triplet.  The third component
    of the triplet has no single excitation in its sector (M_S = -1 of an
    alpha reference); it is among the four lowest once enough vectors are
    started (extra=10), else the singlet at 2.4521 takes its place.'''
    from pyscf import gto
    mf = scf.UHF(gto.M(atom='Li 0 0 0', basis='6-31g', spin=1, verbose=0)).run(conv_tol=1e-12)
    cc = zccsdt.CCSDT(mf.to_ghf())
    cc.verbose, cc.symmetry, cc.conv_tol, cc.conv_tol_normt = 0, False, 1e-11, 1e-9
    cc.kernel()
    e, res = cc.eom_ip(nroots=4, conv_tol=1e-8)
    assert len(e) == 4 and res.converged.all()
    assert abs(e - [0.19591086, 2.42362128, 2.42362128, 2.45206337]).max() < 1e-7
    e, res = cc.eom_ip(nroots=4, conv_tol=1e-8, extra=10)
    assert abs(e - [0.19591086, 2.42362128, 2.42362128, 2.42362128]).max() < 1e-7
    # with the ghost at full length only its singles are known: three roots, and a warning
    with pytest.warns(UserWarning, match='only 3 of the 4 roots'):
        e3, _ = cc.eom_ip(nroots=4, conv_tol=1e-8, short=False)
    assert len(e3) == 3
