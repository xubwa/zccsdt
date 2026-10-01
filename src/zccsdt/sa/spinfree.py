"""Closed-shell spin adaptation of spin-orbital tensor expressions.

For a closed-shell (singlet) reference with spin-free cluster operators, every
spin-orbital tensor is a sum of spatial tensors times products of spin
Kronecker deltas.  In the non-orthogonal spin adaptation (NOSA) used here the
spatial T3 amplitudes t_ijk^abc are the coefficients of E_ai E_bj E_ck, and the
spin-orbital amplitudes are

    t(i j k, a b c) = sum_pi sgn(pi) t_ijk^{pi(abc)} prod_n delta(s_occ(n), s_vir(pi(n))).

A spin-orbital residual of the same form has a spatial residual Omega that is
the coefficient of the *identity pairing* delta(i,a) delta(j,b) delta(k,c), up
to the redundant component that is constant over the permutations of abc.
This module computes that coefficient mechanically:

1. every factor is expanded into spatial components, each with its spin
   deltas (pairs of index slots that share a spin);
2. for each product of components the deltas are traced as lines through the
   index graph: external indices end lines, internal (summed) indices continue
   them, and every closed loop of internal indices gives a factor 2;
3. a product contributes if its lines connect exactly the required external
   pairs (i-a, j-b, k-c).

Index letters: i j k l m n o (and I J K L M N O) are occupied, a b c d e f g h
(and A..H) are virtual.

Factor kinds (spin-orbital tensor -> spatial components, [lines]):

- ``'v'``: ``<pq||rs>`` from the spatial ``<pq|rs>`` (name ``V``):
  ``V_pqrs [p-r, q-s] - V_pqsr [p-s, q-r]``.
- ``'t1'``, ``'f'``, ``('one', N)``: ``X_pq [p-q]``.
- ``'t2'``: ``T2_ijab [i-a, j-b] - T2_ijba [i-b, j-a]``.
- ``'t3'``: ``sum_pi sgn(pi) T3_ijk^{pi(abc)} [n - pi(n)]``.
- ``('w', N)``: a general spin-free 4-index tensor
  ``N(pqrs) = N.J_pqrs [p-r, q-s] - N.K_pqrs [p-s, q-r]``.
- ``('wa', N)``: one antisymmetric in its first pair, ``N(pqrs) = -N(qprs)``:
  ``N_pqrs [p-r, q-s] - N_qprs [p-s, q-r]``.
"""

from fractions import Fraction
from itertools import permutations, product

import numpy as np

OCC = set('ijklmnoIJKLMNO')
VIR = set('abcdefghABCDEFGH')


def _sign(perm):
    s, p = 1, list(perm)
    for x in range(len(p)):
        while p[x] != x:
            y = p[x]
            p[x], p[y] = p[y], p[x]
            s = -s
    return s


class Comp:
    '''One spatial component: coef * name[slots in ``order``] with spin lines
    between the slot pairs ``pairs``.'''
    __slots__ = ('coef', 'name', 'order', 'pairs')

    def __init__(self, coef, name, order, pairs):
        self.coef, self.name, self.order, self.pairs = coef, name, order, pairs


def components(kind):
    '''Spatial components of a spin-orbital factor kind.'''
    if kind == 'v':
        return [Comp(1, 'V', (0, 1, 2, 3), ((0, 2), (1, 3))),
                Comp(-1, 'V', (0, 1, 3, 2), ((0, 3), (1, 2)))]
    if kind == 't2':
        return [Comp(1, 'T2', (0, 1, 2, 3), ((0, 2), (1, 3))),
                Comp(-1, 'T2', (0, 1, 3, 2), ((0, 3), (1, 2)))]
    if kind == 't3':
        return [Comp(_sign(pi), 'T3', (0, 1, 2) + tuple(3 + x for x in pi),
                     tuple((n, 3 + pi[n]) for n in range(3)))
                for pi in permutations(range(3))]
    if kind in ('t1', 'f'):
        return [Comp(1, kind.upper(), (0, 1), ((0, 1),))]
    if isinstance(kind, tuple):
        tag, name = kind
        if tag == 'w':
            return [Comp(1, name + '.J', (0, 1, 2, 3), ((0, 2), (1, 3))),
                    Comp(-1, name + '.K', (0, 1, 2, 3), ((0, 3), (1, 2)))]
        if tag == 'wa':
            return [Comp(1, name, (0, 1, 2, 3), ((0, 2), (1, 3))),
                    Comp(-1, name, (1, 0, 2, 3), ((0, 3), (1, 2)))]
        if tag == 'one':
            return [Comp(1, name, (0, 1), ((0, 1),))]
    raise ValueError(kind)


class Term:
    '''coef * prod_f factor_f, factors given as (kind, index string); letters
    that occur twice are summed.'''

    def __init__(self, coef, factors):
        self.coef = Fraction(coef)
        self.factors = [(k, ''.join(ix)) for k, ix in factors]

    def relabel(self, mapping, sign=1):
        return Term(self.coef * sign,
                    [(k, ''.join(mapping.get(c, c) for c in ix)) for k, ix in self.factors])

    def __repr__(self):
        return '%s %s' % (self.coef, self.factors)


def antisymmetrize(term, groups):
    '''Apply a product of antisymmetrizers; each group is a list of
    (relabeling dict, sign).'''
    terms = [term]
    for g in groups:
        terms = [t.relabel(m, s) for t in terms for m, s in g]
    return terms


def P_one_two(x, y, z):
    '''P(x/yz) = 1 - (xy) - (xz).'''
    return [({}, 1), ({x: y, y: x}, -1), ({x: z, z: x}, -1)]


def P_pair(x, y):
    '''P(xy) = 1 - (xy).'''
    return [({}, 1), ({x: y, y: x}, -1)]


def full_asym(*label_sets):
    '''All signed permutations within each label set (e.g. 'ijk', 'abc').'''
    groups = []
    for labels in label_sets:
        g = []
        for pi in permutations(range(len(labels))):
            g.append(({labels[n]: labels[pi[n]] for n in range(len(labels))}, _sign(pi)))
        groups.append(g)
    return groups


def _lines(edges, ext):
    '''Follow the spin lines.  Returns (endpoint pairs of the open lines,
    number of closed loops), or None if a node has the wrong degree.'''
    adj = {}
    for e, (u, v) in enumerate(edges):
        adj.setdefault(u, []).append(e)
        adj.setdefault(v, []).append(e)
    for node, es in adj.items():
        if len(es) != (1 if node in ext else 2):
            return None
    used = set()
    ends = []
    for start in sorted(ext):
        if start not in adj or adj[start][0] in used:
            continue
        node, e = start, adj[start][0]
        while True:
            used.add(e)
            u, v = edges[e]
            node = v if u == node else u
            if node in ext:
                break
            e = [x for x in adj[node] if x not in used][0]
        ends.append(frozenset((start, node)))
    loops = 0
    for e0 in range(len(edges)):
        if e0 in used:
            continue
        loops += 1
        node = edges[e0][0]
        e = e0
        while e not in used:
            used.add(e)
            u, v = edges[e]
            node = v if u == node else u
            nxt = [x for x in adj[node] if x not in used]
            if not nxt:
                break
            e = nxt[0]
    return ends, loops


def trace(term, ext):
    '''Spatial terms [(coef, [(name, index string), ...])] of the
    identity-pairing coefficient of one spin-orbital term.  ext: required
    external pairs, e.g. [('i','a'), ('j','b'), ('k','c')].'''
    ext_nodes = {c for p in ext for c in p}
    want = {frozenset(p) for p in ext}
    out = []
    for combo in product(*[components(k) for k, _ in term.factors]):
        coef = term.coef
        edges = []
        for (k, ix), c in zip(term.factors, combo):
            coef *= c.coef
            edges.extend((ix[x], ix[y]) for x, y in c.pairs)
        res = _lines(edges, ext_nodes)
        if res is None:
            continue
        ends, loops = res
        if len(ends) != len(want) or set(ends) != want:
            continue
        facs = [(c.name, ''.join(ix[s] for s in c.order)) for (k, ix), c in zip(term.factors, combo)]
        out.append((coef * 2 ** loops, facs))
    return out


def trace_all(terms, ext):
    out = []
    for t in terms:
        out.extend(trace(t, ext))
    return out


# ------------------------------------------------------------ simplify

# index permutations that leave a spatial tensor invariant
SYMMETRIES = {
    'V': [(0, 1, 2, 3), (1, 0, 3, 2)],          # <pq|rs> = <qp|sr>
    'T2': [(0, 1, 2, 3), (1, 0, 3, 2)],
    'T3': [tuple(list(p) + [3 + x for x in p]) for p in permutations(range(3))],
}


def _canonical(facs, out_idx, symmetries):
    '''Canonical key of a product: minimum over factor orders and tensor
    symmetries, internal indices renamed in order of appearance.'''
    best = None
    for order in permutations(range(len(facs))):
        choices = [symmetries.get(facs[n][0], [tuple(range(len(facs[n][1])))]) for n in order]
        for syms in product(*choices):
            ren = {}
            cnt = {'o': 0, 'v': 0}
            parts = []
            seq = []
            for n, s in zip(order, syms):
                name, ix = facs[n]
                ix2 = ''.join(ix[x] for x in s)
                seq.append((name, ix2))
                r = ''
                for c in ix2:
                    if c in out_idx:
                        r += c
                    else:
                        if c not in ren:
                            kind = 'o' if c in OCC else 'v'
                            ren[c] = '%s%d' % (kind, cnt[kind])
                            cnt[kind] += 1
                        r += '<' + ren[c] + '>'
                parts.append('%s[%s]' % (name, r))
            key = ' '.join(parts)
            if best is None or key < best[0]:
                best = (key, seq)
    return best


def simplify(terms, out_idx, symmetries=None):
    '''Merge identical products (up to internal renaming, factor order and
    tensor symmetries).  Returns [(coef, facs)] with nonzero coefficients.'''
    symmetries = SYMMETRIES if symmetries is None else symmetries
    acc, rep = {}, {}
    for coef, facs in terms:
        key, seq = _canonical(facs, set(out_idx), symmetries)
        acc[key] = acc.get(key, Fraction(0)) + coef
        rep.setdefault(key, seq)
    return [(acc[k], rep[k]) for k in sorted(acc) if acc[k] != 0]


def show(terms):
    return '\n'.join('%8s  %s' % (coef, '  '.join('%s[%s]' % f for f in facs))
                     for coef, facs in terms)


# ------------------------------------------------------------ numerics

def evaluate(terms, arrays, out_idx, dims, full_space=('V', 'F')):
    '''Sum of the spatial terms with numpy.einsum.  arrays: name -> ndarray;
    dims: {'o': nocc, 'v': nvir}.  Arrays named in ``full_space`` span the
    whole orbital space (occupied first) and are sliced per factor by the
    spaces of its index letters; the others are given in the factor's spaces.'''
    no, nv = dims['o'], dims['v']
    sl = {'o': slice(0, no), 'v': slice(no, no + nv)}
    shape = tuple(no if c in OCC else nv for c in out_idx)
    res = np.zeros(shape, dtype=np.result_type(*arrays.values()))
    for coef, facs in terms:
        letters = sorted({c for _, ix in facs for c in ix} | set(out_idx))
        m = {c: chr(ord('a') + n) for n, c in enumerate(letters)}
        spec = ','.join(''.join(m[c] for c in ix) for _, ix in facs) + '->' + \
            ''.join(m[c] for c in out_idx)
        ops = []
        for name, ix in facs:
            a = arrays[name]
            if name in full_space:
                a = a[tuple(sl['o' if c in OCC else 'v'] for c in ix)]
            ops.append(a)
        res += float(coef) * np.einsum(spec, *ops, optimize=True)
    return res
