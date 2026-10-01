'''Spatial (closed-shell, spin-adapted) -> spin-orbital tensors, for testing
the spin-adapted equations against the spin-orbital code.

Spin orbitals of a space of n spatial orbitals are ordered alpha first:
index p (alpha) and n + p (beta).  Each spatial dimension of size n becomes a
spin-orbital dimension of size 2n.
'''

from itertools import permutations

import numpy as np


def _sign(perm):
    s, p = 1, list(perm)
    for x in range(len(p)):
        while p[x] != x:
            y = p[x]
            p[x], p[y] = p[y], p[x]
            s = -s
    return s


def _spin_blocks(shape):
    '''For each spatial dimension n, the two spin slices of the 2n axis.'''
    return [(slice(0, n), slice(n, 2 * n)) for n in shape]


def one_body(x):
    '''X(p s, q s') = x_pq delta(s, s').'''
    n0, n1 = x.shape
    out = np.zeros((2 * n0, 2 * n1), dtype=x.dtype)
    out[:n0, :n1] = x
    out[n0:, n1:] = x
    return out


def two_body(J, K=None):
    '''X(p s1, q s2, r s3, s s4) = J_pqrs d(s1,s3) d(s2,s4) - K_pqrs d(s1,s4) d(s2,s3).
    K defaults to J with the last two indices swapped (antisymmetrized
    integral <pq||rs> from <pq|rs>, or t2 from t_ij^ab).'''
    if K is None:
        K = J.transpose(0, 1, 3, 2)
    sl = _spin_blocks(J.shape)
    out = np.zeros(tuple(2 * n for n in J.shape), dtype=np.result_type(J, K))
    for s1 in (0, 1):
        for s2 in (0, 1):
            blk = (sl[0][s1], sl[1][s2])
            out[blk + (sl[2][s1], sl[3][s2])] += J
            out[blk + (sl[2][s2], sl[3][s1])] -= K
    return out


def triples(t):
    '''t(i j k, a b c) = sum_pi sgn(pi) t_ijk^{pi(abc)} prod_n d(s_n, s'_{pi(n)})
    for a spatial t[i,j,k,a,b,c].'''
    no, nv = t.shape[0], t.shape[3]
    out = np.zeros((2 * no,) * 3 + (2 * nv,) * 3, dtype=t.dtype)
    so = [slice(0, no), slice(no, 2 * no)]
    sv = [slice(0, nv), slice(nv, 2 * nv)]
    for pi in permutations(range(3)):
        sg = _sign(pi)
        # spatial factor with the virtual of occupied position n at slot pi(n)
        inv = [pi.index(m) for m in range(3)]
        tp = t.transpose(0, 1, 2, 3 + inv[0], 3 + inv[1], 3 + inv[2])
        for s in np.ndindex(2, 2, 2):
            vs = [None] * 3
            for n in range(3):
                vs[pi[n]] = s[n]
            out[so[s[0]], so[s[1]], so[s[2]], sv[vs[0]], sv[vs[1]], sv[vs[2]]] += sg * tp
    return out


def aab_block(r, no, nv):
    '''The alpha-alpha-beta block R(i a, j a, k b; a a, b a, c b) of a
    spin-orbital triples tensor.'''
    return r[:no, :no, no:, :nv, :nv, nv:]


def nosa_aab(omega):
    '''Omega_ijk^abc - Omega_jik^abc: the alpha-alpha-beta spin-orbital block of
    a NOSA tensor with spatial component omega.'''
    return omega - omega.transpose(1, 0, 2, 3, 4, 5)


def pair_symmetrize(t):
    '''Average over the simultaneous permutations of the pairs (ia)(jb)(kc).'''
    out = np.zeros_like(t)
    for p in permutations(range(3)):
        out += t.transpose(p[0], p[1], p[2], 3 + p[0], 3 + p[1], 3 + p[2])
    return out / 6


def random_spatial(no, nv, seed=0, dtype=np.float64):
    '''Random spatial tensors with the symmetries of T1-dressed quantities:
    <pq|rs> = <qp|sr> (no bra-ket symmetry), t2_ijab = t2_jiba, pair-symmetric
    t3.  Returns dict with full-space V (n^4), F (n^2), T1, T2, T3.'''
    rng = np.random.default_rng(seed)
    n = no + nv
    V = rng.standard_normal((n, n, n, n))
    V = V + V.transpose(1, 0, 3, 2)
    F = rng.standard_normal((n, n))
    T1 = rng.standard_normal((no, nv))
    T2 = rng.standard_normal((no, no, nv, nv))
    T2 = T2 + T2.transpose(1, 0, 3, 2)
    T3 = pair_symmetrize(rng.standard_normal((no,) * 3 + (nv,) * 3))
    return {'V': V.astype(dtype), 'F': F.astype(dtype), 'T1': T1.astype(dtype),
            'T2': T2.astype(dtype), 'T3': T3.astype(dtype)}


def _spaces(ix):
    from zccsdt.sa.spinfree import OCC
    return ''.join('o' if c in OCC else 'v' for c in ix)


def factor_so(kind, ix, spatial, no):
    '''Spin-orbital array of one factor (see zccsdt.sa.spinfree.components).
    spatial: dict with full-space 'V' (n^4) and 'F' (n^2), 'T1', 'T2', 'T3',
    and named intermediates (name or name.J / name.K) in the factor's spaces.'''
    sp = _spaces(ix)
    n = spatial['V'].shape[0] if 'V' in spatial else None
    sl = {'o': slice(0, no), 'v': slice(no, n)}
    if kind == 'v':
        V = spatial['V']
        J = V[tuple(sl[c] for c in sp)]
        K = V[tuple(sl[c] for c in sp[:2] + sp[3] + sp[2])].transpose(0, 1, 3, 2)
        return two_body(J, K)
    if kind == 'f':
        return one_body(spatial['F'][tuple(sl[c] for c in sp)])
    if kind == 't1':
        return one_body(spatial['T1'])
    if kind == 't2':
        return two_body(spatial['T2'])
    if kind == 't3':
        return triples(spatial['T3'])
    tag, name = kind
    if tag == 'w':
        return two_body(spatial[name + '.J'], spatial[name + '.K'])
    if tag == 'wa':
        x = spatial[name]
        return two_body(x, x.transpose(1, 0, 2, 3))
    if tag == 'one':
        return one_body(spatial[name])
    raise ValueError(kind)


def evaluate_so(terms, spatial, out_idx, no):
    '''Numerical value of spin-orbital Terms (zccsdt.sa.spinfree.Term) built
    from spatial tensors; output indices out_idx (spin-orbital axes).'''
    res = None
    for t in terms:
        arrs = [factor_so(k, ix, spatial, no) for k, ix in t.factors]
        letters = sorted({c for _, ix in t.factors for c in ix} | set(out_idx))
        m = {c: chr(ord('a') + n) for n, c in enumerate(letters)}
        spec = ','.join(''.join(m[c] for c in ix) for _, ix in t.factors) + '->' + \
            ''.join(m[c] for c in out_idx)
        val = float(t.coef) * np.einsum(spec, *arrs, optimize=True)
        res = val if res is None else res + val
    return res
