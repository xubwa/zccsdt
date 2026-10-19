'''The DEA sigma vector on the sector itself, term by term: the terms of the
amplitude equations (zccsdt.cc.ccsdtq_terms, T1-dressed) that survive on the
rows with both ghost spinors (docs/eom.md, section 11.9), derived from the
table and evaluated on dense arrays - the reference of zccsdt.cc.dea_sector.

The ghost route (zccsdt.cc.dea) evaluates Res(T + R) on the rows (g1 g2 ..)
of the padded spinors.  No integral and no T has a ghost index, and R has
both, so in a surviving product R is one factor and g1, g2 are two of its
external occupied letters; the other factors are T and the dressed
integrals.  For every term coef * A(product) of the residual of rank k and
every amplitude factor with two external occupied letters (L1, L2):

    the factor becomes r of the sector, r_n[rest; virtuals] = t_n[g1, g2, rest; ..],
    with the sign that brings L1, L2 to the front of its occupied letters;
    the antisymmetrizer over the k external occupied letters keeps the
    permutations that put g1 on L1 and g2 on L2: the sign of that
    assignment times the antisymmetrizer over the k - 2 others.

The external occupied letters that are left are renamed i, j in their order.
Every ordered pair (L1, L2) counts: the pairs (L1, L2) and (L2, L1) give the
same product with the same sign.

Dense vectors: r2[a, b], r3[i, a, b, c], r4[i, j, a, b, c, d], antisymmetric.
'''

from fractions import Fraction
from itertools import permutations

import numpy as np

from zccsdt.cc import ccsdtq_terms as Q

OCC = Q.OCC
EXTO, EXTV = 'ijkl', 'abcd'


def _parity(seq, ref):
    '''Sign of the permutation that takes ``ref`` to ``seq``.'''
    return Q._sign([ref.index(c) for c in seq])


def sector_terms(row, rank=4, with_t3=True, with_t4=False):
    '''[(coef, factors, group)] of the sigma vector of the row ``row``
    (2: 2p, 3: 3p1h, 4: 4p2h) for R to ``rank`` (3: 2p + 3p1h, 4: .. +
    4p2h), T to T3 (``with_t3``) and T4 (``with_t4``).  Factors as in
    ccsdtq_terms, with the kinds 'r2', 'r3', 'r4' for the sector vector:
    their occupied letters are the real ones only.  The external letters
    are the first row - 2 of ijkl and the first row of abcd.'''
    out = []
    E = EXTO[:row]
    for name, (k, _, kinds, _, terms) in Q.GROUPS.items():
        if k != row or not name.startswith('R%d' % row):
            continue
        for coef, factors in terms:
            for fi, (kind, ix) in enumerate(factors):
                if kind not in ('t2', 't3', 't4') or int(kind[1]) > rank:
                    continue
                others = [kd for j, (kd, _) in enumerate(factors) if j != fi]
                if 't4' in others and not with_t4 or 't3' in others and not with_t3:
                    continue
                occ = [c for c in ix if c in OCC]
                ext = [c for c in occ if c in E]
                for L1 in ext:
                    for L2 in ext:
                        if L1 == L2:
                            continue
                        rest_f = [c for c in occ if c not in (L1, L2)]
                        s_f = _parity([L1, L2] + rest_f, occ)
                        rest = [c for c in E if c not in (L1, L2)]
                        s_a = _parity([L1, L2] + rest, list(E))
                        ren = {c: EXTO[n] for n, c in enumerate(rest)}
                        new = []
                        for j, (kd, x) in enumerate(factors):
                            if j == fi:
                                x = ''.join(c for c in x if c not in (L1, L2))
                                kd = 'r' + kind[1]
                            new.append((kd, ''.join(ren.get(c, c) for c in x)))
                        out.append((Fraction(coef) * s_f * s_a, tuple(new), name))
    return out


def merged(terms):
    '''The same terms with equal products (same factors) collected.'''
    acc = {}
    for coef, factors, group in terms:
        key = factors
        c, g = acc.get(key, (Fraction(0), group))
        acc[key] = (c + coef, g)
    return [(c, f, g) for f, (c, g) in acc.items() if c != 0]


def _asym(x, no_, nv_):
    '''Antisymmetrizer over the first no_ and the last nv_ axes (all signed
    permutations).'''
    for k, off in ((no_, 0), (nv_, no_)):
        if k < 2:
            continue
        out = np.zeros_like(x)
        nax = x.ndim
        for p in permutations(range(k)):
            ax = list(range(nax))
            ax[off:off + k] = [off + y for y in p]
            out += Q._sign(p) * x.transpose(ax)
        x = out
    return x


def evaluate(terms, arrays, nocc, row):
    '''sum coef * A(product) as a dense array [i.. (row - 2), a.. (row)].
    arrays: 'f', 'v' over all spin orbitals; 't2', 't3', 't4' full; 'r2',
    'r3', 'r4' as in the module docstring.'''
    n = arrays['v'].shape[0]
    outl = EXTO[:row - 2] + EXTV[:row]
    total = 0
    for coef, factors, _ in terms:
        ops = []
        for kind, ix in factors:
            a = arrays[kind]
            if kind in ('v', 'f'):
                a = a[tuple(slice(0, nocc) if c in OCC else slice(nocc, n) for c in ix)]
            ops.append(a)
        sub = ','.join(ix for _, ix in factors) + '->' + outl
        total = total + float(coef) * np.einsum(sub, *ops, optimize=True)
    return _asym(total, row - 2, row)


def sigma(r, amps, fock, g, nocc, rank=None):
    '''[sigma2, sigma3(, sigma4)] dense, for r = [r2, r3(, r4)] dense and the
    amplitudes [t1, t2, t3(, t4)] (dense, t3 may be None) of the
    Hamiltonian (fock, <pq||rs>) - T1 dressed here.'''
    from zccsdt.cc import ccsdt
    rank = rank or len(r) + 1
    ge, fdr = ccsdt._t1_dress(g, fock, amps[0], nocc, g.shape[0])
    arrays = {'v': ge, 'f': fdr, 't2': amps[1]}
    with_t3 = len(amps) > 2 and amps[2] is not None
    with_t4 = len(amps) > 3 and amps[3] is not None
    if with_t3:
        arrays['t3'] = amps[2]
    if with_t4:
        arrays['t4'] = amps[3]
    for k, x in enumerate(r):
        arrays['r%d' % (k + 2)] = x
    return [evaluate(sector_terms(row, rank, with_t3, with_t4), arrays, nocc, row) for row in range(2, rank + 1)]


def show(row, rank=4, with_t3=True, with_t4=False):
    '''Lines: the merged terms of a row.'''
    lines = []
    for c, f, gname in merged(sector_terms(row, rank, with_t3, with_t4)):
        lines.append('%-8s %-40s %s' % (c, ' '.join('%s_%s' % x for x in f), gname))
    return lines


if __name__ == '__main__':
    import sys
    wt4 = '--t4' in sys.argv
    for row in (2, 3, 4):
        print('row %d' % row)
        for line in show(row, with_t4=wt4):
            print('  ' + line)


# ------------------------------------------------------------------ vectors

def _tuples(n, k):
    return np.zeros((1, 0), dtype=np.int64) if k == 0 else Q.tuples(n, k)


def expand(x, no, nv, row):
    '''The dense antisymmetric array of a packed part [C(no, row - 2), C(nv, row)].'''
    ko, kv = row - 2, row
    occ, vir = _tuples(no, ko), _tuples(nv, kv)
    u = np.asarray(x).reshape(len(occ), len(vir))
    out = np.zeros((no,) * ko + (nv,) * kv, dtype=u.dtype)
    for p in permutations(range(ko)):
        for q in permutations(range(kv)):
            idx = (tuple(occ[:, p[s]][:, None] for s in range(ko))
                   + tuple(vir[:, q[s]][None, :] for s in range(kv)))
            out[idx] = Q._sign(p) * Q._sign(q) * u
    return out


def unique(x, row):
    '''The packed part of a dense antisymmetric array (inverse of expand).'''
    ko, kv = row - 2, row
    no = x.shape[0] if ko else 0
    nv = x.shape[-1]
    occ, vir = _tuples(no, ko), _tuples(nv, kv)
    return x[tuple(occ[:, s][:, None] for s in range(ko)) + tuple(vir[:, s][None, :] for s in range(kv))].ravel()
