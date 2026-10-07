'''Derivation of the tables of zccsdt.cc.lambda_terms: the Lambda equations
of CCSDT, the densities and the orbital gradient, in the T1-dressed
formalism.  Everything here is symbolic (rational coefficients, index
strings); ``python -m zccsdt.cc.lambda_derive`` prints the tables as they
stand in lambda_terms.

The Lagrangian.  With the T1-dressed, normal-ordered F and V (f~ and <pq||rs>~
of zccsdt.cc.ccsdt._t1_dress) and T = T2 + T3,

    L = <0| (1 + Lambda) exp(-T) (F + V) exp(T) |0>
      = E0 + 1/4 sum v[m,n,e,f] t2[m,n,e,f] + sum_k (1/k!)^2 sum l_k R_k,

R_k the residuals of the amplitude equations (zccsdt.cc.ccsdtq_terms) and
Lambda_k = (1/k!)^2 sum l_k[i.., a..] i+ j+ .. b a,  k = 1, 2, 3.  A term
c A(P) of R_k gives the scalar c sum l_k P (the antisymmetrizer is taken
back by the antisymmetry of l_k): lagrangian().

Three derivatives of it, each by a rule on the index strings:

- with respect to t2 and t3 (lambda_by_differentiation): the factor is
  taken out and its letters become the external ones, c A(rest).  These are
  the residuals of the Lambda equations of rank 2 and 3;
- with respect to an integral (densities): the factor f or v is taken out;
  what is left is the coefficient of that block of dressed integrals in L;
- with respect to t1, which is in the dressed integrals only:
  d/dt1[i,a] h~ = [h~, a+_a a_i], a one-index transformation of them, a
  virtual bra index taking minus the values of i and an occupied ket index
  the values of a (one_index).  The dressed Fock matrix contains
  sum_m <pm||qm>~ and gets <pi||qa>~ besides; E0 gives f~[i,a].  The same
  with the de-excitation a+_i a_a - an occupied bra index takes minus the
  values of a, a virtual ket index those of i, the Fock matrix gets
  -<pa||qi>~, E0 gives -f~[a,i] - is the orbital gradient in the dressed
  basis.

The Lambda equations are derived a second time by Wick's theorem alone
(lambda_by_wick, zccsdt.cc.wick.derive_left), without the table of the
amplitude equations and without the rule for t1: all contractions of
<0| (1 + Lambda) (H tau T ..)_C |0> for tau of rank 1, 2, 3.  The two
derivations give the same tables, coefficient by coefficient
(tests/test_lambda_terms.py).

Terms are compared in a canonical form (canonical): the summed letters
renamed, the indices sorted within the antisymmetric groups of every
factor, the factors sorted, and for an antisymmetrized term the external
letters permuted - the smallest of all these, with the sign it takes.
'''

from fractions import Fraction
from itertools import combinations_with_replacement, permutations, product as iproduct
from math import factorial

from zccsdt.cc import ccsdtq_terms as Q

OCC = 'ijklmno'
VIR = 'abcdefgh'
_SUMMED = {'o': 'mnoijkl', 'v': 'efghabcd'}
_EXTERNAL = {'o': 'ijkl', 'v': 'abcd'}
_ORDER = {'l1': 0, 'l2': 1, 'l3': 2, 'f': 3, 'v': 4, 't2': 5, 't3': 6}


def _cls(letter):
    return 'o' if letter in OCC else 'v'


def _groups(kind, n):
    '''The antisymmetric groups of the indices of a factor.'''
    if kind == 'f':
        return ((0,), (1,))
    if kind == 'v':
        return ((0, 1), (2, 3))
    return (tuple(range(n // 2)), tuple(range(n // 2, n)))


def _sign(perm):
    return Q._sign(perm)


def _sorted(letters):
    '''(sorted letters, sign of the permutation).'''
    order = sorted(range(len(letters)), key=letters.__getitem__)
    return ''.join(letters[x] for x in order), _sign(order)


def canonical(coef, factors, ext, antisym):
    '''(coefficient, factors, external letters) of the term
    coef * product[ext] (``antisym``: coef * A(product), A over the external
    occupied and over the external virtual letters) in its canonical form,
    or None if the term vanishes by the antisymmetry of its factors.

    The k-th occupied (virtual) external letter is named ijkl[k] (abcd[k]),
    the summed ones by the first free letters of mnoijkl (efghabcd).  The
    letters are told apart by where they stand (factor and group), so only
    letters in the same places - and factors of the same kind - are tried
    in every order.'''
    kinds = [k for k, _ in factors]
    twins = [[x for x, k in enumerate(kinds) if k == kind] for kind in sorted(set(kinds))]
    letters = sorted({c for _, ix in factors for c in ix})
    assert all(c in letters for c in ext) and len(set(ext)) == len(ext)
    names = {}
    for cl in 'ov':
        e = [c for c in ext if _cls(c) == cl]
        names[cl] = (_EXTERNAL[cl][:len(e)], [c for c in _SUMMED[cl] if c not in _EXTERNAL[cl][:len(e)]])
    best, signs = None, set()
    for orders in iproduct(*[permutations(t) for t in twins]):
        # the factors of one kind numbered in this order
        number = {}
        for t, order in zip(twins, orders):
            for n, x in enumerate(order):
                number[x] = n
        place = {}
        for x, (kind, ix) in enumerate(factors):
            for g, group in enumerate(_groups(kind, len(ix))):
                for s in group:
                    place.setdefault(ix[s], []).append((_ORDER[kind], number[x], g))
        # classes of letters that may be exchanged: same space, external or
        # not (external ones only if antisymmetrized), same places
        classes = {}
        for c in letters:
            external = c in ext
            key = (_cls(c), external, tuple(sorted(place[c])), 0 if antisym or not external else ext.index(c))
            classes.setdefault(key, []).append(c)
        keys = sorted(classes)
        for perms in iproduct(*[permutations(classes[k]) for k in keys]):
            rename, count, sign = {}, {('o', True): 0, ('o', False): 0, ('v', True): 0, ('v', False): 0}, 1
            newext = {}
            for k, perm in zip(keys, perms):
                cl, external = k[0], k[1]
                for c in perm:
                    n = count[cl, external]
                    count[cl, external] += 1
                    rename[c] = names[cl][0][n] if external else names[cl][1][n]
            if not antisym:
                # the external letters keep their order
                for cl in 'ov':
                    e = [c for c in ext if _cls(c) == cl]
                    for n, c in enumerate(e):
                        rename[c] = names[cl][0][n]
            else:
                for cl in 'ov':
                    e = [rename[c] for c in ext if _cls(c) == cl]
                    sign *= _sign([names[cl][0].index(c) for c in e])
            new = []
            for kind, ix in factors:
                ix = ''.join(rename[c] for c in ix)
                out = list(ix)
                for group in _groups(kind, len(ix)):
                    srt, sg = _sorted([ix[s] for s in group])
                    sign *= sg
                    for s, c in zip(group, srt):
                        out[s] = c
                new.append((_ORDER[kind], kind, ''.join(out)))
            new = tuple(sorted(new))
            if best is None or new < best:
                best, signs = new, {sign}
            elif new == best:
                signs.add(sign)
    if len(signs) == 2:
        return None
    newext = ''
    count = {'o': 0, 'v': 0}
    for c in ext:
        newext += _EXTERNAL[_cls(c)][count[_cls(c)]]
        count[_cls(c)] += 1
    return coef * signs.pop(), tuple((kind, ix) for _, kind, ix in best), newext


def collect(terms, antisym):
    '''{external letters: [(coefficient, factors)]} of [(coef, factors,
    ext)] with equal terms added and the terms that vanish left out,
    sorted.'''
    out = {}
    for coef, factors, ext in terms:
        c = canonical(coef, factors, ext, antisym)
        if c is None:
            continue
        coef, factors, ext = c
        d = out.setdefault(ext, {})
        d[factors] = d.get(factors, 0) + coef
    return {ext: sorted(((c, f) for f, c in d.items() if c != 0), key=lambda t: (_key(t[1]), t[0]))
            for ext, d in out.items()}


def _key(factors):
    return tuple((_ORDER[k], ix) for k, ix in factors)


def _fresh(factors):
    '''The factors with every letter renamed to one that no external name
    can be: upper case.'''
    return tuple((kind, ix.upper()) for kind, ix in factors)


# ------------------------------------------------------------ the Lagrangian

def lagrangian(wick=False):
    '''[(coefficient, factors)]: L - E0 as a sum of scalars, every letter
    summed.  From the table of the amplitude equations, or (``wick``) from
    the contractions of zccsdt.cc.wick.'''
    out = [(Fraction(1, 4), (('v', 'mnef'), ('t2', 'mnef')))]
    if wick:
        from zccsdt.cc import wick as W
        out = []
        for k, kind, ranks, sym in _combinations(range(4)):
            for coef, factors, _ in W.derive(kind, ranks, k, sym):
                left = (('l%d' % k, Q.EXT[k]),) if k else ()
                out.append((coef, left + factors))
        return out
    for name, (k, _, kinds, _, terms) in Q.GROUPS.items():
        if k > 3 or 't4' in kinds:
            continue
        out += [(coef, (('l%d' % k, Q.EXT[k]),) + factors) for coef, factors in terms]
    return out


def _combinations(lranks):
    '''(rank of Lambda, kind of H, ranks of the cluster operators, symmetry
    factor) of everything that can be connected: at most two lines of F and
    four of V.'''
    for k in lranks:
        for kind, most in (('f', 2), ('v', 4)):
            for n in range(most + 1):
                for ranks in combinations_with_replacement((2, 3), n):
                    sym = 1
                    for r in set(ranks):
                        sym *= factorial(ranks.count(r))
                    yield k, kind, list(ranks), sym


# ------------------------------------------------------- Lambda equations

def lambda_by_wick(rank):
    '''The residual of the Lambda equation of ``rank``,
    <0| (1 + Lambda) [exp(-T) H exp(T), tau] |0> with the dressed H and
    T = T2 + T3, from Wick's theorem: [(coefficient, factors)], canonical.'''
    from zccsdt.cc import wick as W
    terms = []
    for k, kind, ranks, sym in _combinations(range(4)):
        if len(ranks) + 1 > (2 if kind == 'f' else 4):
            continue
        terms += W.derive_left(kind, ranks, k, rank, sym)
    out = collect(terms, antisym=rank > 1)
    assert list(out) == [Q.EXT[rank]]
    return out[Q.EXT[rank]]


def lambda_by_differentiation(rank):
    '''The same from the Lagrangian: its derivative with respect to t2 or
    t3 (rank 2, 3), or its one-index transformation by a+_a a_i (rank 1).'''
    if rank == 1:
        return one_index(excitation=True)
    terms = []
    for coef, factors in lagrangian():
        for x, (kind, ix) in enumerate(factors):
            if kind == 't%d' % rank:
                terms.append((coef, factors[:x] + factors[x + 1:], ix))
    out = collect(terms, antisym=True)
    return out[Q.EXT[rank]]


def one_index(excitation):
    '''L with the dressed integrals one-index transformed by a+_a a_i
    (``excitation``: dL/dt1[i,a], external letters 'ia') or by a+_i a_a
    (the orbital gradient of the dressed basis, external letters 'ai').'''
    I, A = 'i', 'a'                         # (the terms are in upper case meanwhile)
    terms = [(Fraction(1), (('f', 'ia'),), 'ia')] if excitation else [(Fraction(-1), (('f', 'ai'),), 'ai')]
    ext = 'ia' if excitation else 'ai'
    for coef, factors in lagrangian():
        factors = _fresh(factors)
        x = [k for k, _ in factors].index('f') if 'f' in [k for k, _ in factors] else [k for k, _ in factors].index('v')
        kind, ix = factors[x]
        for s, c in enumerate(ix):
            bra, occ = s < len(ix) // 2, c.lower() in OCC
            if excitation and bra and not occ:
                sign, here, there = -1, I, A
            elif excitation and not bra and occ:
                sign, here, there = 1, A, I
            elif not excitation and bra and occ:
                sign, here, there = -1, A, I
            elif not excitation and not bra and not occ:
                sign, here, there = 1, I, A
            else:
                continue
            new = []
            for y, (k2, ix2) in enumerate(factors):
                if y == x:
                    new.append((k2, ix2[:s] + here + ix2[s + 1:]))
                else:
                    new.append((k2, ix2.replace(c, there)))
            terms.append((sign * coef, tuple(_lower(new)), ext))
        if kind == 'f':
            p, q = ix
            v = ('v', p + I + q + A) if excitation else ('v', p + A + q + I)
            new = [v if y == x else f for y, f in enumerate(factors)]
            terms.append(((1 if excitation else -1) * coef, tuple(_lower(new)), ext))
    return collect(terms, antisym=False)[ext]


def _lower(factors):
    '''Upper-case letters (summed) to lower-case ones that are not i or a.'''
    pool = {'o': [c for c in OCC if c != 'i'], 'v': [c for c in VIR if c != 'a']}
    rename = {}
    for _, ix in factors:
        for c in ix:
            if c.isupper() and c not in rename:
                rename[c] = pool[_cls(c.lower())].pop(0)
    return [(kind, ''.join(rename.get(c, c) for c in ix)) for kind, ix in factors]


# ----------------------------------------------------------------- densities

def densities():
    '''{block: [(coefficient, factors)]}: the coefficients of the dressed
    integrals in L - E0.  The block is named by its external letters, those
    of the integral: 'ai' is the coefficient of f[a,i], 'aibc' that of
    <ai||bc>.'''
    terms = []
    for coef, factors in lagrangian():
        factors = _fresh(factors)
        kinds = [k for k, _ in factors]
        x = kinds.index('f') if 'f' in kinds else kinds.index('v')
        rest = _lower_all(factors[:x] + factors[x + 1:], factors[x][1])
        terms.append((coef, rest[0], rest[1]))
    return collect(terms, antisym=False)


def _lower_all(factors, ext):
    pool = {'o': list(OCC), 'v': list(VIR)}
    rename = {}
    for c in ext:
        rename[c] = pool[_cls(c.lower())].pop(0)
    for _, ix in factors:
        for c in ix:
            if c not in rename:
                rename[c] = pool[_cls(c.lower())].pop(0)
    return (tuple((kind, ''.join(rename[c] for c in ix)) for kind, ix in factors),
            ''.join(rename[c] for c in ext))


# -------------------------------------------------------------------- output

def _fmt(coef, factors):
    c = repr(str(coef)) if coef.denominator != 1 else str(coef.numerator)
    return '_t(%s, %s)' % (c, ', '.join("'%s_%s'" % f for f in factors))


def _print_list(name, terms, indent='    '):
    print('%s = [' % name)
    for coef, factors in terms:
        print('%s%s,' % (indent, _fmt(coef, factors)))
    print(']')


def tables():
    '''(LAMBDA, DENSITY, GRADIENT) as lambda_terms holds them.'''
    lam = {rank: lambda_by_wick(rank) for rank in (1, 2, 3)}
    return lam, densities(), one_index(excitation=False)


if __name__ == '__main__':
    lam, dens, grad = tables()
    print('LAMBDA = {')
    for rank, terms in lam.items():
        print('    %d: [' % rank)
        for coef, factors in terms:
            print('        %s,' % _fmt(coef, factors))
        print('    ],')
    print('}\n\nDENSITY = {')
    for block in sorted(dens, key=lambda b: (len(b), b)):
        print("    '%s': [" % block)
        for coef, factors in dens[block]:
            print('        %s,' % _fmt(coef, factors))
        print('    ],')
    print('}\n')
    _print_list('GRADIENT', grad)
