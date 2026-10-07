'''Wick's theorem for the connected coupled-cluster expressions of one
determinant: the terms of

    1/sym <Phi_k| (H T_a T_b ..)_connected |0>,
    H = F_N or V_N,   T_n = (1/n!)^2 sum t[i.., a..] a+ b+ .. j i,

which is the k-fold excited projection of the nested commutator
1/sym [..[[H, T_a], T_b]..] for commuting cluster operators.

Relative to the Fermi vacuum the cluster operators hold quasi-particle
creators only, and H is normal ordered, so the only contractions are those
of the quasi-particle annihilators of H (a+_m with m occupied, a_e with e
virtual) with operators of the T's to their right.  A term is one way of
contracting every annihilator of H so that every T has a contraction; its
sign is that of bringing each contracted pair together and then the open
operators into the order a+ b+ .. j i of the projection.  With the open
lines named by the external letters the term is

    coefficient * A(product),   A: all signed permutations of the external
                                occupied and of the external virtual letters,

the form of zccsdt.cc.ccsdtq_terms.  Contractions with the same topology -
the same numbers of hole and particle lines between H and each T - are the
same term up to the sign of a relabeling; ``derive`` leaves them apart and
``by_topology`` collects them.

Run as a script, the module prints the equations of CCSDTQ in the T1-dressed
formalism and the pentuples of the (P) correction.
'''

from fractions import Fraction
from itertools import product as iproduct
from math import factorial

OCC_LETTERS = 'ijklmno'
VIR_LETTERS = 'abcdefgh'


class _Op:
    __slots__ = ('label', 'dagger', 'space', 'owner')

    def __init__(self, label, dagger, space, owner):
        self.label, self.dagger, self.space, self.owner = label, dagger, space, owner

    @property
    def annihilator(self):
        '''Does it annihilate the Fermi vacuum?'''
        return self.dagger == (self.space == 'o')


def _hamiltonians(kind):
    '''[(coefficient, operators in normal order, labels in the order of the
    indices of the tensor)] of F_N = sum f[p,q] {p+ q} or of
    V_N = 1/4 sum <pq||rs> {p+ q+ s r}, by the classes of the indices.'''
    out = []
    n = 2 if kind == 'f' else 4
    for classes in iproduct('ov', repeat=n):
        labels = ['H%d' % x for x in range(n)]
        if kind == 'f':
            ops = [_Op(labels[0], True, classes[0], 0), _Op(labels[1], False, classes[1], 0)]
            coef = Fraction(1)
        else:
            ops = [_Op(labels[0], True, classes[0], 0), _Op(labels[1], True, classes[1], 0),
                   _Op(labels[3], False, classes[3], 0), _Op(labels[2], False, classes[2], 0)]
            coef = Fraction(1, 4)
        # normal order: the creators to the left, each kind in its order;
        # all operators of a normal-ordered product anticommute
        swaps = sum(1 for x in range(len(ops)) for y in range(x + 1, len(ops))
                    if ops[x].annihilator and not ops[y].annihilator)
        ops = [o for o in ops if not o.annihilator] + [o for o in ops if o.annihilator]
        out.append((coef * (-1) ** swaps, ops, list(zip(labels, classes))))
    return out


def _cluster(n, owner):
    '''T_n as (coefficient, operators a+ b+ .. j i, labels i.. a..).'''
    occ = ['T%do%d' % (owner, x) for x in range(n)]
    vir = ['T%dv%d' % (owner, x) for x in range(n)]
    ops = [_Op(v, True, 'v', owner) for v in vir] + [_Op(o, False, 'o', owner) for o in reversed(occ)]
    return Fraction(1, factorial(n) ** 2), ops, occ + vir


def derive(kind, ranks, rank, sym=1):
    '''All terms of 1/sym <Phi_rank| (H T_a T_b ..)_C |0> for H of ``kind``
    ('f' or 'v') and cluster operators of ``ranks``:
    [(coefficient, factors, topology)], one for every set of contractions.
    factors: ((kind, index string), ..) as in ccsdtq_terms; topology: the
    sorted (rank, hole lines, particle lines) of the cluster operators.'''
    terms = []
    ts = [_cluster(n, x + 1) for x, n in enumerate(ranks)]
    tcoef = Fraction(1, sym)
    for c, _, _ in ts:
        tcoef *= c
    tops = [op for _, ops, _ in ts for op in ops]
    for hcoef, hops, hlabels in _hamiltonians(kind):
        anns = [op for op in hops if op.annihilator]
        nopen = len(hops) - 2 * len(anns) + sum(2 * n for n in ranks)
        if nopen != 2 * rank:
            continue
        # every annihilator of H with an operator of a T: same space, the
        # other one of a+ / a
        choices = [[t for t in tops if t.space == a.space and t.dagger != a.dagger] for a in anns]
        for targets in iproduct(*choices):
            if len(set(map(id, targets))) != len(targets):
                continue
            if {t.owner for t in targets} != set(range(1, len(ranks) + 1)):
                continue                                    # a T without a contraction
            string = hops + tops
            sign = 1
            for a, t in zip(anns, targets):
                p, q = string.index(a), string.index(t)
                sign *= (-1) ** (q - p - 1)
                string = [o for o in string if o is not a and o is not t]
            # the open operators into a+ b+ .. (as they stand), then .. j i
            sign *= (-1) ** sum(1 for x in range(len(string)) for y in range(x + 1, len(string))
                                if string[x].space == 'o' and string[y].space == 'v')
            vir = [o for o in string if o.space == 'v']
            occ = [o for o in string if o.space == 'o']
            name = {o.label: VIR_LETTERS[x] for x, o in enumerate(vir)}
            name.update({o.label: OCC_LETTERS[rank - 1 - x] for x, o in enumerate(occ)})
            count = {'o': 0, 'v': 0}
            for a, t in zip(anns, targets):
                letters = OCC_LETTERS if a.space == 'o' else VIR_LETTERS
                name[a.label] = name[t.label] = letters[rank + count[a.space]]
                count[a.space] += 1
            factors = [(kind, ''.join(name[lab] for lab, _ in hlabels))]
            factors += [('t%d' % n, ''.join(name[lab] for lab in labels))
                        for n, (_, _, labels) in zip(ranks, ts)]
            topo = tuple(sorted((n, sum(1 for t in targets if t.owner == x + 1 and t.space == 'o'),
                                 sum(1 for t in targets if t.owner == x + 1 and t.space == 'v'))
                                for x, n in enumerate(ranks)))
            terms.append((hcoef * tcoef * sign, tuple(factors), topo))
    return terms


def derive_left(kind, ranks, lrank, rank, sym=1):
    '''All terms of the left-hand projection

        1/sym <0| Lambda_lrank (H tau T_a T_b ..)_C |0>,   tau = a+ b+ .. j i,

    the part of <0| (1 + Lambda) [exp(-T) H exp(T), tau] |0> with H of
    ``kind``, the de-excitation operator Lambda_n = (1/n!)^2 sum
    l[i.., a..] i+ j+ .. b a (lrank = 0: the 1) and cluster operators of
    ``ranks``: [(coefficient, factors, external letters of tau)], a term
    being coefficient * A(product) with A over the letters of tau (rank 1:
    the product).

    tau is of quasi-particle creators like a cluster operator and commutes
    with them, so the commutator is the connected product with tau as one
    more cluster operator, of unit amplitude - what derive() enumerates
    with the ranks ``ranks + [rank]``.  Lambda is of quasi-particle
    annihilators only: it closes the open lines of that product, and the
    n!^2 ways of doing so are relabelings of its summed indices, each with
    the sign that the antisymmetry of l takes back: 1/n!^2 sum l[I, A]
    <Phi_I^A| .. |0> is sum l[I, A] (representative product)[I, A].  The
    unit amplitude: a term of derive() is a sum c G[J, B] t[J, B] over all
    index values of the last operator, where the amplitude of one
    excitation stands rank!^2 times with its signs: its coefficient is
    c A(G).

    With the T1-dressed H, tau of rank 1 gives the derivative with respect
    to t1 (d/dt1[i,a] exp(-T1) H exp(T1) = [exp(-T1) H exp(T1), a+_a a_i]).'''
    out = []
    for coef, factors, _ in derive(kind, list(ranks) + [rank], lrank, sym):
        tau = factors[-1][1]
        left = (('l%d' % lrank, OCC_LETTERS[:lrank] + VIR_LETTERS[:lrank]),) if lrank else ()
        out.append((coef, left + factors[:-1], tau))
    return out


def topology(factors):
    '''The topology of a term given as factors: the sorted (rank, hole
    lines, particle lines) of its cluster operators.'''
    h = set(factors[0][1])
    out = []
    for kind, ix in factors[1:]:
        n = len(ix) // 2
        out.append((n, len(h & set(ix[:n])), len(h & set(ix[n:]))))
    return tuple(sorted(out))


def by_topology(terms):
    '''{topology: [(coefficient, factors)]} of derive().'''
    out = {}
    for coef, factors, topo in terms:
        out.setdefault(topo, []).append((coef, factors))
    return out


if __name__ == '__main__':
    from zccsdt.cc import ccsdtq_terms as Q
    for group, (rank, kind, kinds, sym, table) in Q.GROUPS.items():
        tops = by_topology(derive(kind, [int(k[1]) for k in kinds], rank, sym))
        print('%s: %d contractions, %d topologies (the table has %d terms)'
              % (group, sum(len(v) for v in tops.values()), len(tops), len(table)))
        for topo, ts in sorted(tops.items()):
            print('    %s: %4d contractions, sum of |coefficients| %s, e.g. %s %s'
                  % (topo, len(ts), sum(abs(c) for c, _ in ts), ts[0][0],
                     ' '.join('%s(%s)' % f for f in ts[0][1])))
