'''Block storage of the spin-adapted T3 amplitudes.

t_ijk^abc is invariant under simultaneous permutations of the pairs (ia),
(jb), (kc), so it is stored only for the canonical (sorted) occupied triples
s = (s0 <= s1 <= s2), not all equal (t_iii does not enter): one full v^3 cube
t3b[B, a, b, c] per triple B.  Any ordered triple (p, q, r) maps to a block
and an axis permutation:

    t_pqr^{xyz} = t3b[blk[p,q,r]].transpose(axes[p,q,r])[x, y, z].

Residual contributions come in two shapes:

- **attached** terms X_{r;pq}^{x;yz} (p <= q, symmetric under exchanging the
  spectator pairs (p,y) <-> (q,z)), summed over the three positions:
  omega_ijk^abc += X_{i;jk}^{a;bc} + X_{j;ik}^{b;ac} + X_{k;ij}^{c;ab};
- **ordered** terms G_t^{xyz} of the six-term pair symmetrizer:
  omega_ijk^abc += sum_pi G_{pi(ijk)}^{pi(abc)}.

``uses_attached`` / ``uses_ordered`` list the (block, axes) placements of one
computed term, so that blocks with repeated indices receive every
contribution exactly once.
'''

from itertools import permutations, product

import numpy as np


class Blocks:
    def __init__(self, nocc):
        self.nocc = no = nocc
        tri = [(i, j, k) for i in range(no) for j in range(i, no) for k in range(j, no)
               if not (i == j == k)]
        self.tri = np.array(tri, dtype=np.int64).reshape(-1, 3)
        self.nblk = len(tri)
        self.index = {t: n for n, t in enumerate(tri)}
        self.blk = -np.ones((no, no, no), dtype=np.int64)
        self.axes = np.zeros((no, no, no, 3), dtype=np.int64)
        for p, q, r in product(range(no), repeat=3):
            if p == q == r:
                continue
            order = np.argsort((p, q, r), kind='stable')
            self.blk[p, q, r] = self.index[tuple(int(x) for x in np.array((p, q, r))[order])]
            self.axes[p, q, r] = np.argsort(order)
        t = self.tri
        # 0: all distinct, 1: s0 = s1 < s2, 2: s0 < s1 = s2
        self.kind = np.where(t[:, 0] == t[:, 1], 1, np.where(t[:, 1] == t[:, 2], 2, 0))
        self._att, self._ord = {}, {}

    def cube(self, t3b, p, q, r):
        '''t_pqr^{xyz} as a (v, v, v) view (zeros for p = q = r).'''
        b = self.blk[p, q, r]
        if b < 0:
            return np.zeros(t3b.shape[1:], dtype=t3b.dtype)
        return t3b[b].transpose(self.axes[p, q, r])

    def uses_attached(self, r, p, q):
        '''[(block, axes)]: omega_block += X.transpose(axes) for an attached
        term X_{r;pq} (p <= q).'''
        key = (r, p, q)
        if key not in self._att:
            out = []
            s = tuple(sorted((r, p, q)))
            if s in self.index:
                for n in range(3):
                    rest = [m for m in range(3) if m != n]
                    if s[n] == r and (s[rest[0]], s[rest[1]]) == (p, q):
                        out.append((self.index[s], tuple(int(x) for x in np.argsort([n] + rest))))
            self._att[key] = out
        return self._att[key]

    def uses_ordered(self, t):
        '''[(block, axes)]: omega_block += G_t.transpose(axes) for an ordered
        term G_t of the six-term symmetrizer.'''
        t = tuple(t)
        if t not in self._ord:
            out = []
            s = tuple(sorted(t))
            if s in self.index:
                for pi in permutations(range(3)):
                    if tuple(s[pi[m]] for m in range(3)) == t:
                        out.append((self.index[s], tuple(int(x) for x in np.argsort(pi))))
            self._ord[t] = out
        return self._ord[t]

    def to_dense(self, t3b):
        no, nv = self.nocc, t3b.shape[1]
        out = np.zeros((no,) * 3 + (nv,) * 3, dtype=t3b.dtype)
        for p, q, r in product(range(no), repeat=3):
            if self.blk[p, q, r] >= 0:
                out[p, q, r] = self.cube(t3b, p, q, r)
        return out

    def from_dense(self, t3):
        return np.ascontiguousarray(t3[self.tri[:, 0], self.tri[:, 1], self.tri[:, 2]])


def project_null(r3b):
    '''Remove the redundant component of every cube: r - (1/6) sum_sigma r^{sigma(abc)}.'''
    s = sum(r3b.transpose(0, 1 + p[0], 1 + p[1], 1 + p[2]) for p in permutations(range(3)))
    return r3b - s / 6


def denominators(blocks, eo, ev):
    '''e_i + e_j + e_k - e_a - e_b - e_c for every block.'''
    t = blocks.tri
    o = eo[t[:, 0]] + eo[t[:, 1]] + eo[t[:, 2]]
    v = ev[:, None, None] + ev[None, :, None] + ev[None, None, :]
    return o[:, None, None, None] - v[None]
