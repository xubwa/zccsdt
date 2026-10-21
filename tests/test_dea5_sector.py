'''EOM-DEA with complete 5p3h operators (zccsdt.cc.dea_ref with R to 5p3h,
zccsdt.cc.dea5_sector; docs/eom.md, section 11.11):

- the terms that survive on the rows with both ghosts, derived from the
  groups of the T5 equation of CCSDTQP (Wick's theorem): the groups of the
  table of ccsdtq_terms are reproduced; the terms on dense arrays against
  the determinant route (dea_strings, three holes; real, T of CCSDT) and
  against <mu| [Hbar, R] |0> of second-quantized operators (complex, random
  T1 .. T3);
- the sigma vector of the sector in label blocks, every element, against
  those terms (LiH, X2C spinors in the blocks of 2 m_j);
- the roots against the determinant route; the memory of a sigma vector
  against its plan.'''
import tracemalloc
from fractions import Fraction
from math import comb

import numpy as np
import pytest
from pyscf import gto, scf, x2c

from fockspace import Fock, random_antisym
from test_ccsdtq_ref import _spin_mixed
from test_dea import TIGHT, _ghf
from test_dea5 import _from_det, _to_det, NO, NV
from zccsdt import eris as E
from zccsdt.cc import ccsdt_opt, ccsdtq_ref as R, ccsdtq_terms as Q, dea_ref, dea_strings as DS, solver, t4pack as P

LIH = 'Li 0 0 0; H 0 0 1.5957'


def dense_g(eris):
    """(fock, <pq||rs>) dense, <vv||vv> from its packed form if only that is kept."""
    from zccsdt.cc import ccsdtq_ref as R_
    fock, g, _, no = R_.integrals(eris)
    g = np.array(g)
    if getattr(eris, 'vvvv', None) is None:
        n = g.shape[0]
        nv = n - no
        a, b = np.triu_indices(nv, 1)
        vpp = np.asarray(eris._vvvv_pp)
        V = np.zeros((nv,) * 4, dtype=vpp.dtype)
        V[a[:, None], b[:, None], a[None, :], b[None, :]] = vpp
        V[b[:, None], a[:, None], a[None, :], b[None, :]] = -vpp
        V[a[:, None], b[:, None], b[None, :], a[None, :]] = -vpp
        V[b[:, None], a[:, None], b[None, :], a[None, :]] = vpp
        g[no:, no:, no:, no:] = V
    return np.asarray(fock), g


def test_derived_groups_reproduce_the_table():
    '''Wick's theorem with the canonical form of dea_ref: every group of
    ccsdtq_terms (rows 1 to 4 and the pentuples of (P)).'''
    for name, (k, kind, kinds, _, terms) in Q.GROUPS.items():
        ranks = tuple(int(x[1]) for x in kinds)
        if not ranks:
            continue
        der = {f: c for c, f in dea_ref.derived_group(k, kind, ranks)}
        tab = {}
        for c, f in terms:
            s, cf = dea_ref.canon(f)
            tab[cf] = tab.get(cf, Fraction(0)) + Fraction(c) * s
        assert der == tab, name
    assert [len(dea_ref.merged(dea_ref.sector_terms(r, 5), True)) for r in (2, 3, 4, 5)] == [7, 20, 43, 61]


def test_families_of_the_5p3h_row():
    """Random dense arrays: every term of the 5p3h row equals its family
    (dea_ref.FAMILIES5, the factorization of dea5_sector) with that term
    alone, and the families sum to the row."""
    rng = np.random.default_rng(2)
    no, nv = 4, 5
    n = no + nv
    v = rng.standard_normal((n, n, n, n))
    v = v - v.transpose(1, 0, 2, 3)
    v = v - v.transpose(0, 1, 3, 2)
    f = rng.standard_normal((n, n))
    arrays = {'v': v, 'f': f, 't2': random_antisym(rng, no, nv, 2), 't3': random_antisym(rng, no, nv, 3),
              'r2': dea_ref._asym(rng.standard_normal((nv, nv)), 0, 2),
              'r3': dea_ref._asym(rng.standard_normal((no, nv, nv, nv)), 1, 3),
              'r4': dea_ref._asym(rng.standard_normal((no, no, nv, nv, nv, nv)), 2, 4),
              'r5': dea_ref._asym(rng.standard_normal((no, no, no, nv, nv, nv, nv, nv)), 3, 5)}
    ts = dea_ref.merged(dea_ref.sector_terms(5, 5), True)
    for num, term in enumerate(ts, 1):
        fam, fi = dea_ref.ASSIGN5[num]
        sign, others = dea_ref.family_split(term, fi, fam)
        want = dea_ref.evaluate([term], arrays, no, 5)
        got = dea_ref.family_value(fam, dea_ref.family_B(fam, [(term[0], sign, others)], arrays, no), arrays, no)
        assert abs(want - got).max() < 1e-12 * max(1.0, abs(want).max()), num
    total = sum(dea_ref.family_value(fam, dea_ref.family_B(fam, items, arrays, no), arrays, no)
                for fam, items in dea_ref.families5().items())
    want = dea_ref.evaluate(ts, arrays, no, 5)
    assert abs(total - want).max() < 1e-12 * abs(want).max()


def _sizes():
    return [comb(NV, 2), NO * comb(NV, 3), comb(NO, 2) * comb(NV, 4), comb(NO, 3) * comb(NV, 5)]


def test_terms_against_the_determinant_route():
    '''LiH / STO-3G (4 + 8), T of CCSDT: the rows 2p .. 5p3h of dea_ref with
    R to 5p3h against P exp(-T) H exp(T) P on the determinants with up to
    three holes (converged amplitudes: Hbar; the energy on the diagonal).'''
    eris = E.from_scf(scf.addons.convert_to_ghf(_ghf(LIH)))
    cc = solver.kernel(eris, 'ccsdt', TIGHT)
    fock, g, _, _ = R.integrals(eris)
    fock, g = np.asarray(fock).real, np.asarray(g).real
    t3p = np.asarray(cc.t3).reshape(comb(NO, 3), comb(NV, 3))
    det = DS.Problem(fock, g, NO, [cc.t1, Q.unique(cc.t2, 2), t3p], 3, e_ref=eris.e_hf)
    e0 = eris.e_hf + cc.e_corr - det.const
    rng = np.random.default_rng(3)
    x = [rng.standard_normal(s) for s in _sizes()]
    yd = _from_det(det, det.sigma(_to_det(det, x), det.new()))
    rd = [dea_ref.expand(p, NO, NV, k + 2) for k, p in enumerate(x)]
    ref = dea_ref.sigma(rd, [np.asarray(cc.t1), np.asarray(cc.t2), Q.expand(t3p, NO, NV, 3)], fock, g, NO, rank=5)
    for k in range(4):
        a = yd[k].ravel() - e0 * x[k]
        assert abs(a).max() > 0.1
        assert abs(a - dea_ref.unique(ref[k], k + 2)).max() < 1e-11 * abs(a).max()


def test_terms_against_the_fock_space_complex():
    '''Complex spin orbitals and random complex T1 .. T3: the rows of
    dea_ref against <mu| exp(-T) [H, R] exp(T) |0> in the Fock space.'''
    import scipy.sparse as sp
    mf = scf.addons.convert_to_ghf(_ghf(LIH))
    eris = E.from_scf(mf, mo_coeff=_spin_mixed(mf)[0])
    fock, g, _, _ = R.integrals(eris)
    fock, g = np.asarray(fock), np.asarray(g)
    n = NO + NV
    rng = np.random.default_rng(5)
    t1 = 0.05 * (rng.standard_normal((NO, NV)) + 1j * rng.standard_normal((NO, NV)))
    t2 = 0.05 * (random_antisym(rng, NO, NV, 2) + 1j * random_antisym(rng, NO, NV, 2))
    t3 = 0.05 * (random_antisym(rng, NO, NV, 3) + 1j * random_antisym(rng, NO, NV, 3))
    fk = Fock(NO, NV)
    H = fk.two_body(g) + sum(fock[p, q] * (fk.ad[p] @ fk.a[q]) for p in range(n) for q in range(n) if fock[p, q] != 0)
    T = fk.excitation(t1) + fk.excitation(t2) + fk.excitation(t3)

    def ex(sign, v):
        out, term = v.copy(), v.copy()
        for j in range(1, n + 1):
            term = (sign / j) * (T @ term)
            out = out + term
        return out

    def op(holes, parts):
        m = sp.identity(2 ** n, format='csr')
        for a in parts:
            m = m @ fk.ad[NO + int(a)]
        for i in reversed(holes):
            m = m @ fk.a[int(i)]
        return m
    tup = {k: (P.tuples(NO, k - 2) if k > 2 else np.zeros((1, 0), dtype=int), P.tuples(NV, k)) for k in (2, 3, 4, 5)}
    ops = {k: [[op(h, p) for p in tup[k][1]] for h in tup[k][0]] for k in (2, 3, 4, 5)}
    x = [0.3 * (rng.standard_normal(s) + 1j * rng.standard_normal(s)) for s in _sizes()]
    Rop = sp.csr_matrix((2 ** n,) * 2, dtype=complex)
    for k in (2, 3, 4, 5):
        r = x[k - 2].reshape(len(tup[k][0]), len(tup[k][1]))
        for a in range(r.shape[0]):
            for b in range(r.shape[1]):
                Rop = Rop + r[a, b] * ops[k][a][b]
    hbar_r = ex(-1.0, H @ ex(1.0, Rop @ fk.ref))
    vec = hbar_r - Rop @ ex(-1.0, H @ ex(1.0, fk.ref))
    rd = [dea_ref.expand(p, NO, NV, k + 2) for k, p in enumerate(x)]
    ref = dea_ref.sigma(rd, [t1, t2, t3], fock, g, NO, rank=5)
    for k in (2, 3, 4, 5):
        want = np.array([[(ops[k][a][b] @ fk.ref) @ vec for b in range(len(tup[k][1]))]
                         for a in range(len(tup[k][0]))]).ravel()
        got = dea_ref.unique(ref[k - 2], k)
        assert abs(want).max() > 0.1 and abs(want - got).max() < 1e-11 * abs(want).max()


def _lih_x2c():
    import zccsdt
    mf = x2c.UHF(gto.M(atom=LIH, basis='sto-3g', verbose=0)).run(conv_tol=1e-12)
    c = zccsdt.CCSDT(mf)
    c.symmetry, c.verbose = True, 0
    c.conv_tol, c.conv_tol_normt = 1e-12, 1e-10
    c.kernel()
    return c


@pytest.fixture(scope='module')
def lih():
    return _lih_x2c()


@pytest.mark.parametrize('shift', [0, 2])
def test_sector_against_the_terms(lih, shift):
    '''LiH / STO-3G, X2C spinors in the blocks of 2 m_j: every element of the
    sigma vector of the sector (rows 2p .. 5p3h) against dea_ref on dense
    arrays; elements of the dense rows outside the blocks are zero.'''
    from zccsdt.cc import dea5_sector as D5
    eris = lih._eris
    no = eris.nocc
    nv = np.asarray(eris.fock).shape[0] - no
    pb = D5.ProblemSector5(eris, lih.t1, lih.t2, lih.t3, target=shift)
    dt = pb.dtype
    rng = np.random.default_rng(7)
    x = [rng.standard_normal(m) + (1j * rng.standard_normal(m) if dt.kind == 'c' else 0) for m in pb.sizes]
    y = pb.sigma(x, pb.new())
    dpos = pb.dense_positions()
    fock, g = dense_g(eris)
    if dt.kind == 'f':
        fock, g = fock.real, g.real
    full = []
    for k in range(4):
        row = k + 2
        f = np.zeros((1 if row == 2 else comb(no, row - 2)) * comb(nv, row), dtype=dt)
        f[dpos[k]] = x[k]
        full.append(f)
    t3p = ccsdt_opt.packed_t3(eris, lih.t3)
    amps = [np.asarray(lih.t1, dt), np.asarray(lih.t2, dt), Q.expand(np.asarray(t3p, dt), no, nv, 3)]
    ref = dea_ref.sigma([dea_ref.expand(p, no, nv, k + 2) for k, p in enumerate(full)], amps, fock, g, no, rank=5)
    scale = max(abs(p).max() for p in y)
    assert scale > 0.1
    for k in range(4):
        u = dea_ref.unique(ref[k], k + 2)
        got = np.zeros_like(u)
        got[dpos[k]] = y[k]
        assert abs(u - got).max() < 1e-11 * scale


def test_memory_of_a_sigma_vector(lih):
    '''tracemalloc: what one sigma vector allocates stays below the plan
    (memory5: per sigma vector), with batches that take several passes.'''
    from zccsdt.cc import dea5_sector as D5
    pb = D5.ProblemSector5(lih._eris, lih.t1, lih.t2, lih.t3, target=0, batch_bytes=1 << 14)
    x = [np.random.default_rng(1).standard_normal(n) for n in pb.sizes]
    y = pb.new()
    pb.sigma(x, y)
    tracemalloc.start()
    tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    pb.sigma(x, y)
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    assert peak <= pb.mem[1], (peak, pb.mem[1])
