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

Dense vectors: r2[a, b], r3[i, a, b, c], r4[i, j, a, b, c, d],
r5[i, j, k, a, b, c, d, e], antisymmetric.

R to 5p3h (``rank`` 5, docs/eom.md, section 11.11): the table of
ccsdtq_terms stops at rank 4 (and the pentuples of (P)); the groups that a
5p3h vector reaches - the T5 terms of the rows of rank 3 and 4 and every
group of the T5 equation of CCSDTQP that has at most one amplitude of rank
4 or 5 - are derived here with Wick's theorem (zccsdt.cc.wick), the
contractions collected by a canonical form of their products (canon()).
'''

from fractions import Fraction
from itertools import permutations

import numpy as np

from zccsdt.cc import ccsdtq_terms as Q

OCC = Q.OCC
EXTO, EXTV = 'ijklm', 'abcde'

# the groups of the rows of rank 3 to 5 that a 5p3h vector reaches (with the
# T of CCSDT or CCSDTQ in the other places): (rank, 'f' or 'v', ranks of the
# cluster operators), each the commutator 1/sym [..[H, T_a]..] with sym the
# factorials of equal ranks
GROUPS5 = [(3, 'v', (5,)), (4, 'f', (5,)), (4, 'v', (5,)),
           (5, 'f', (5,)), (5, 'v', (5,)), (5, 'v', (4,)), (5, 'v', (2, 3)), (5, 'f', (2, 4)), (5, 'f', (3, 3)),
           (5, 'v', (2, 4)), (5, 'v', (3, 3)), (5, 'v', (2, 2, 2)), (5, 'v', (2, 5)), (5, 'v', (3, 4)),
           (5, 'v', (2, 2, 3))]


def _groups(kind, ix):
    '''The antisymmetric groups of the indices of a factor.'''
    if kind == 'v':
        return [ix[0:2], ix[2:4]]
    if kind == 'f':
        return [ix[0], ix[1]]
    n = int(kind[1]) - 2 if kind[0] == 'r' else len(ix) // 2       # r_n: n - 2 occupied (the real ones)
    return [ix[:n], ix[n:]]


def canon(factors):
    '''(sign, canonical factors) of a product under A: the letters that occur
    once are external (the first ones of ijklm and of abcde), the others are
    summed.  A(product) = sign * A(canonical product).  Each letter gets a key
    that does not depend on its name - external or summed, the (factor,
    group) places where it stands, occupied or virtual -, the letters are
    renamed in the order of their keys and sorted inside every antisymmetric
    group; letters with the same key are interchangeable (the sign of the
    renaming and of the sorting cancel).  Amplitudes of the same kind
    commute: every order of them is tried, the smallest form is taken.'''
    best = None
    kinds = [k for k, _ in factors]
    for order in permutations(range(len(factors))):
        if any(kinds[o] != kinds[x] for x, o in enumerate(order)):
            continue
        fs = [factors[o] for o in order]
        where = {}
        for fi, (k, ix) in enumerate(fs):
            for gi, gr in enumerate(_groups(k, ix)):
                for c in gr:
                    where.setdefault(c, []).append((fi, gi))
        key = {c: (len(v) == 1, tuple(v), c in OCC) for c, v in where.items()}
        letters = sorted(key, key=lambda c: key[c])
        ext_o = [c for c in letters if key[c][0] and c in OCC]
        ext_v = [c for c in letters if key[c][0] and c not in OCC]
        sum_o = [c for c in letters if not key[c][0] and c in OCC]
        sum_v = [c for c in letters if not key[c][0] and c not in OCC]
        EO, EV = EXTO[:len(ext_o)], EXTV[:len(ext_v)]
        SO = [x for x in 'mnoijkl' if x not in EO][:len(sum_o)]
        SV = [x for x in 'efghabcd' if x not in EV][:len(sum_v)]
        ren = dict(zip(ext_o, EO))
        ren.update(zip(ext_v, EV))
        ren.update(zip(sum_o, SO))
        ren.update(zip(sum_v, SV))
        sign = 1
        for ext, tgt in ((ext_o, EO), (ext_v, EV)):
            if ext:
                sign *= _parity([ren[c] for c in sorted(ext)], sorted(tgt))
        out = []
        for k, ix in fs:
            new = ''
            for gr in _groups(k, ''.join(ren[c] for c in ix)):
                srt = ''.join(sorted(gr))
                sign *= _parity(list(srt), list(gr))
                new += srt
            out.append((k, new))
        cand = (tuple(out), sign)
        if best is None or cand[0] < best[0]:
            best = cand
    return best[1], best[0]


_TABLE5 = {}


def derived_group(rank, kind, ranks):
    '''[(coef, factors)] of 1/sym <rank| [..[H, T_a]..] |0>_C from Wick's
    theorem, the contractions collected by canon().'''
    key = (rank, kind, tuple(ranks))
    if key not in _TABLE5:
        from collections import Counter
        from math import factorial
        from zccsdt.cc import wick
        sym = 1
        for c in Counter(ranks).values():
            sym *= factorial(c)
        acc = {}
        for c, f, _ in wick.derive(kind, list(ranks), rank, sym):
            s, cf = canon(f)
            acc[cf] = acc.get(cf, Fraction(0)) + c * s
        _TABLE5[key] = [(c, f) for f, c in acc.items() if c != 0]
    return _TABLE5[key]


def table(rank=4):
    '''{name: (row, kind, kinds, terms)} of the amplitude equations that
    R to ``rank`` reaches: ccsdtq_terms for the rows to 4, and for rank 5
    the groups GROUPS5 in addition.'''
    out = {}
    for name, (k, kind, kinds, _, terms) in Q.GROUPS.items():
        if k <= 4 and name.startswith('R%d' % k):
            out[name] = (k, kind, kinds, terms)
    if rank >= 5:
        for k, kind, ranks in GROUPS5:
            name = 'R%d: %s.%s' % (k, kind.upper(), '.'.join('T%d' % x for x in ranks))
            out[name] = (k, kind, tuple('t%d' % x for x in ranks), derived_group(k, kind, ranks))
    return out


def _parity(seq, ref):
    '''Sign of the permutation that takes ``ref`` to ``seq``.'''
    return Q._sign([ref.index(c) for c in seq])


def sector_terms(row, rank=4, with_t3=True, with_t4=False):
    '''[(coef, factors, group)] of the sigma vector of the row ``row``
    (2: 2p, 3: 3p1h, 4: 4p2h, 5: 5p3h) for R to ``rank`` (3: 2p + 3p1h,
    4: .. + 4p2h, 5: .. + 5p3h), T to T3 (``with_t3``) and T4
    (``with_t4``).  Factors as in ccsdtq_terms, with the kinds 'r2' .. 'r5'
    for the sector vector: their occupied letters are the real ones only.
    The external letters are the first row - 2 of ijklm and the first row
    of abcde.'''
    out = []
    E = EXTO[:row]
    for name, (k, _, kinds, terms) in table(rank).items():
        if k != row:
            continue
        for coef, factors in terms:
            for fi, (kind, ix) in enumerate(factors):
                if kind not in ('t2', 't3', 't4', 't5') or int(kind[1]) > rank:
                    continue
                others = [kd for j, (kd, _) in enumerate(factors) if j != fi]
                if 't5' in others or 't4' in others and not with_t4 or 't3' in others and not with_t3:
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


def merged(terms, canonical=False):
    '''The same terms with equal products (same factors) collected; with
    ``canonical`` the products in the canonical form of canon() (equal up
    to a relabeling and the sign).'''
    acc = {}
    for coef, factors, group in terms:
        key = factors
        if canonical:
            sg, key = canon(factors)
            coef = coef * sg
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
    return [evaluate(merged(sector_terms(row, rank, with_t3, with_t4), True), arrays, nocc, row)
            for row in range(2, rank + 1)]


def show(row, rank=4, with_t3=True, with_t4=False, canonical=False):
    '''Lines: the merged terms of a row.'''
    lines = []
    for c, f, gname in merged(sector_terms(row, rank, with_t3, with_t4), canonical):
        lines.append('%-8s %-40s %s' % (c, ' '.join('%s_%s' % x for x in f), gname))
    return lines


if __name__ == '__main__':
    import sys
    wt4 = '--t4' in sys.argv
    rk = 5 if '--r5' in sys.argv else 4
    for row in range(2, rk + 1):
        print('row %d' % row)
        for line in show(row, rk, with_t4=wt4, canonical=rk == 5):
            print('  ' + line)


# ------------------------------------------------------------------ the families of the 5p3h row

# The factorization of zccsdt.cc.dea5_sector: every term of the 5p3h row is
# coef A(F B_t), F a fixed factor (the template below) and B_t the product of
# the others; the terms of a family add their B_t into one intermediate,
# B = sum_t coef_t sign_t P(B_t) (P: the antisymmetrizer over the external
# groups of B), and the family is A(F B) / (|O_B|! |V_B|!) =
# |O_F|! |V_F|! S[F B] (S: the distinct splits of the external indices).
# family: (kind of F, indices of F, indices of B); the external letters of F
# and B partition ijk / abcde, the other letters of F are shared with B.
FAMILIES5 = {
    'pp': ('r5', 'ijkfgcde', 'fgab'),        # Wpp(ab, fg) r5(ijk; fg cde)
    'Fv': ('r5', 'ijkfbcde', 'fa'),          # F_vv
    'hh': ('r5', 'mnkabcde', 'mnij'),        # Whh(mn, ij) r5(mnk; ..)
    'Fo': ('r5', 'mjkabcde', 'mi'),          # F_oo
    'ring': ('r5', 'mjkfbcde', 'mfia'),      # Wr(m, a, f, i) r5(mjk; f bcde)
    'D1': ('r4', 'jkfcde', 'fiab'),          # - Wv(ab, f, i) r4(jk; f cde)
    'D2': ('t2', 'imab', 'mjkcde'),          # t2(im; ab) XoR5(m; jk; cde)
    'D3': ('r3', 'kgde', 'gijabc'),          # r3(k; g de) Xv2(g; ij; abc)
    'D4': ('t3', 'ijmfab', 'mfkcde'),        # t3(ijm; f ab) Z4(m f; k; cde)
    'D5': ('t3', 'ijkfab', 'fcde'),          # t3(ijk; f ab) XvR5(f; cde)
    'D6': ('r3', 'mcde', 'mijkab'),          # r3(m; cde) Xo3(m; ijk; ab)
    'D7': ('t3', 'imnabc', 'mnjkde'),        # t3(imn; abc) Y4(mn; jk; de)
    'D8': ('t3', 'jkmabc', 'mide'),          # t3(jkm; abc) XoRp(m; i; de)
    'E1': ('r4', 'mkbcde', 'mija'),          # r4(mk; bcde) Wo(m, a, ij)
    'E1x': ('t2', 'ijaf', 'fkbcde'),         # t2(ij; a f) Xv5(f; k; bcde)
    'E2': ('r4', 'mnbcde', 'mnijka'),        # r4(mn; bcde) W2(mn; ijk; a)
}

# term (1-based, in the order of merged(sector_terms(5, 5), True)) -> (family, factor of F)
ASSIGN5 = {
    1: ('Fo', 1), 2: ('Fv', 1), 3: ('hh', 1), 4: ('ring', 1), 5: ('pp', 1),
    6: ('E1', 1), 7: ('D1', 1), 8: ('D6', 2), 9: ('D3', 2), 10: ('D8', 2), 11: ('D6', 2), 12: ('D5', 2),
    13: ('D3', 2), 14: ('D1', 2), 15: ('E1', 2), 16: ('D8', 2), 17: ('D6', 2), 18: ('D1', 2), 19: ('E1', 2),
    20: ('D2', 1), 21: ('E2', 2), 22: ('D1', 2), 23: ('D2', 1), 24: ('E1', 2), 25: ('E1x', 1), 26: ('D3', 1),
    27: ('D6', 1), 28: ('D8', 1), 29: ('D3', 1), 30: ('D5', 1), 31: ('D8', 2), 32: ('D6', 2),
    33: ('D2', 2), 34: ('D2', 3), 35: ('D2', 3), 36: ('Fv', 2), 37: ('pp', 2), 38: ('Fo', 2), 39: ('ring', 2),
    40: ('D2', 1), 41: ('hh', 2), 42: ('E1x', 1), 43: ('D1', 2), 44: ('D7', 1), 45: ('E1', 2), 46: ('D4', 1),
    47: ('D8', 1), 48: ('E2', 2), 49: ('D5', 1), 50: ('D5', 3), 51: ('D3', 3), 52: ('D3', 3), 53: ('D2', 1),
    54: ('D8', 3), 55: ('D6', 3), 56: ('D8', 3), 57: ('D6', 3), 58: ('D2', 2), 59: ('D2', 2), 60: ('E1x', 2),
    61: ('E1x', 1),
}


def family_split(term, fi, fam):
    '''(sign, the other factors renamed) with coef A(term) = coef sign A(F B_t)
    for the family ``fam`` and the factor ``fi`` of the term as F.'''
    _, factors, _ = term
    kind, tidx, bstr = FAMILIES5[fam]
    out_o, out_v = EXTO[:3], EXTV[:5]
    fk, fx = factors[fi]
    assert fk == kind, (term, fam)
    count = {}
    for _, ix in factors:
        for c in ix:
            count[c] = count.get(c, 0) + 1
    ext = {c for c, n in count.items() if n == 1}
    sign = 1
    ren = {}
    for T, L in zip(_groups(kind, tidx), _groups(kind, fx)):
        Te = [p for p, c in enumerate(T) if c in out_o + out_v]
        Ts = [p for p, c in enumerate(T) if c not in out_o + out_v]
        Le = [c for c in L if c in ext]
        Ls = [c for c in L if c not in ext]
        assert len(Te) == len(Le) and len(Ts) == len(Ls), (term, fam)
        new = [None] * len(T)
        for p, c in zip(Te + Ts, Le + Ls):
            new[p] = c
        sign *= _parity(new, list(L))
        for p, c in enumerate(new):
            ren[c] = T[p]
    used = set(tidx) | set(bstr)
    for out in (out_o, out_v):
        rest_t = sorted(c for c in out if c not in ren.values())
        rest_l = sorted(c for c in ext if c in out and c not in ren)
        ren.update(zip(rest_l, rest_t))
        src = sorted(c for c in ext if c in out)
        sign *= _parity([ren[c] for c in src], sorted(ren[c] for c in src))
    pools = {'o': [c for c in 'lnomijk' if c not in used], 'v': [c for c in 'hgfedcba' if c not in used]}
    for c in count:
        if c not in ren:
            ren[c] = pools['o' if c in OCC else 'v'].pop(0)
    return sign, [(k, ''.join(ren[c] for c in ix)) for j, (k, ix) in enumerate(factors) if j != fi]


def _einsum_factors(factors, arrays, nocc, out):
    n = arrays['v'].shape[0]
    ops = []
    for kind, ix in factors:
        a = arrays[kind]
        if kind in ('v', 'f'):
            a = a[tuple(slice(0, nocc) if c in OCC else slice(nocc, n) for c in ix)]
        ops.append(a)
    return np.einsum(','.join(ix for _, ix in factors) + '->' + out, *ops, optimize=True)


def family_B(fam, items, arrays, nocc):
    '''B of a family from [(coef, sign, other factors)], dense.'''
    bstr = FAMILIES5[fam][2]
    groups = [[p for p, c in enumerate(bstr) if c in EXTO[:3]], [p for p, c in enumerate(bstr) if c in EXTV[:5]]]
    B = 0
    for coef, sign, others in items:
        x = _einsum_factors(others, arrays, nocc, bstr)
        for g in groups:
            if len(g) < 2:
                continue
            y = np.zeros_like(x)
            for p in permutations(range(len(g))):
                ax = list(range(x.ndim))
                for a, b in zip(g, p):
                    ax[a] = g[b]
                y += Q._sign(p) * x.transpose(ax)
            x = y
        B = B + float(coef) * sign * x
    return B


def family_value(fam, B, arrays, nocc):
    '''A(F B) / (|O_B|! |V_B|!) as a dense [i, j, k, a, b, c, d, e].'''
    from math import factorial
    kind, tidx, bstr = FAMILIES5[fam]
    n = arrays['v'].shape[0]
    F = arrays[kind]
    if kind in ('v', 'f'):
        F = F[tuple(slice(0, nocc) if c in OCC else slice(nocc, n) for c in tidx)]
    x = np.einsum(tidx + ',' + bstr + '->' + EXTO[:3] + EXTV[:5], F, B, optimize=True)
    return _asym(x, 3, 5) / (factorial(sum(c in EXTO[:3] for c in bstr)) * factorial(sum(c in EXTV[:5] for c in bstr)))


def families5():
    '''{family: [(coef, sign, other factors)]} of the 5p3h row (T of CCSDT).'''
    ts = merged(sector_terms(5, 5), True)
    assert sorted(ASSIGN5) == list(range(1, len(ts) + 1))
    out = {}
    for num, term in enumerate(ts, 1):
        fam, fi = ASSIGN5[num]
        sign, others = family_split(term, fi, fam)
        out.setdefault(fam, []).append((term[0], sign, others))
    return out


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
