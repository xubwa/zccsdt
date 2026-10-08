'''The terms of the Lambda equations, densities and orbital gradient of
CCSDTQ that have t4 or l4 in them (zccsdt.cc.lambda4_terms), by the cost of
their best order of pairwise contractions: for docs/lambda.md, section 13.

usage: python bench/l4_scaling.py [o v]        (default 10 40)

For every term all orders of pairwise contraction are tried; the cost of a
step is o^x v^y with x, y the numbers of distinct occupied and virtual
letters of its two operands, the size of an intermediate that of its open
letters (no symmetry counted: the exponents are the point).  The best order
is the one whose most expensive step is cheapest at the given o and v, then
the one with the smallest largest intermediate.  A term is then named by
(the most expensive step, the largest intermediate that is not an operand).
'''
import sys
from collections import defaultdict
from itertools import combinations

from zccsdt.cc import lambda4_terms as L
from zccsdt.cc.ccsdtq_terms import EXT

OCC = set('ijklmno')


def _size(letters, o, v):
    x = sum(1 for c in letters if c in OCC)
    return o ** x * v ** (len(letters) - x), (x, len(letters) - x)


def best(factors, ext, o, v):
    '''((cost, exponents) of the most expensive step, (size, exponents) of
    the largest intermediate, the order as text).'''
    start = tuple((kind, frozenset(ix)) for kind, ix in factors)

    def search(ops):
        if len(ops) == 1:
            return (0, (0, 0)), (0, (0, 0)), ''
        out = None
        for a, b in combinations(range(len(ops)), 2):
            rest = [x for n, x in enumerate(ops) if n not in (a, b)]
            other = set(ext).union(*[x[1] for x in rest]) if rest else set(ext)
            both = ops[a][1] | ops[b][1]
            new = frozenset(c for c in both if c in other)
            step = _size(both, o, v)
            inter = _size(new, o, v) if rest else (0, (0, 0))
            name = '(%s %s)' % (ops[a][0], ops[b][0])
            sub = search(tuple(rest) + ((name, new),))
            cand = (max(step, sub[0]), max(inter, sub[1]), name if not sub[2] else sub[2])
            if out is None or cand[:2] < out[:2]:
                out = cand
        return out
    return search(start)


def fmt(e):
    return ''.join('%s^%d' % (c, n) if n > 1 else c for c, n in zip('ov', e) if n) or '1'


def main():
    o, v = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (10, 40)
    tables = [('LAMBDA[%d]' % k, EXT[k], L.quadruple(L.LAMBDA[k])) for k in (4, 3, 2, 1)]
    tables += [('DENSITY %s' % b, b, L.quadruple(t)) for b, t in L.DENSITY.items() if L.quadruple(t)]
    tables.append(('GRADIENT', 'ai', L.quadruple(L.GRADIENT)))
    print('o = %d, v = %d; T4 has o^4 v^4 elements, its ladders cost o^4 v^6 (pp), o^6 v^4 (hh), o^5 v^5 (ring)' % (o, v))
    for name, ext, terms in tables:
        groups = defaultdict(list)
        for coef, factors in terms:
            step, inter, order = best(factors, ext, o, v)
            groups[step[1], inter[1]].append((coef, factors, order))
        print('\n%s: %d terms with t4 or l4' % (name, len(terms)))
        for (step, inter), ts in sorted(groups.items(), key=lambda x: -_cost(x[0][0], o, v)):
            print('  step %-9s intermediate %-9s %2d terms' % (fmt(step), fmt(inter), len(ts)))
            if name.startswith('LAMBDA') or len(sys.argv) > 3:
                for coef, factors, order in ts:
                    print('      %-7s %-46s %s' % (coef, ' '.join('%s_%s' % f for f in factors), order))


def _cost(e, o, v):
    return o ** e[0] * v ** e[1]


if __name__ == '__main__':
    main()
