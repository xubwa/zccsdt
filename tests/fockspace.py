'''Brute-force second quantization in a small spin-orbital Fock space
(Jordan-Wigner matrices), to check connected coupled-cluster expressions
term by term without diagram rules.

Spin orbitals 0 .. n-1, the first ``nocc`` occupied in the reference
|0> = a+_0 ... a+_{nocc-1} |vac>.  Amplitudes are antisymmetric tensors
t[i.., a..] (virtual indices counted from 0), integrals v[p,q,r,s] = <pq||rs>.
'''

from itertools import combinations

import numpy as np
import scipy.sparse as sp


class Fock:
    def __init__(self, nocc, nvir):
        self.no, self.nv = nocc, nvir
        self.n = n = nocc + nvir
        I = sp.identity(2, format='csr')
        Z = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, -1.0]]))
        A = sp.csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]]))      # a |1> = |0>
        self.a = []
        for p in range(n):
            m = sp.identity(1, format='csr')
            for q in range(n):
                m = sp.kron(m, Z if q < p else (A if q == p else I), format='csr')
            self.a.append(m)
        self.ad = [x.T.tocsr() for x in self.a]
        ref = np.zeros(2 ** n)
        ref[0] = 1.0                                                # vacuum
        for p in reversed(range(nocc)):
            ref = self.ad[p] @ ref
        self.ref = ref

    def excitation(self, t):
        '''T_k = (1/k!^2) sum t[i.., a..] a+ b+ .. j i  as a sparse matrix.'''
        k = t.ndim // 2
        T = sp.csr_matrix((2 ** self.n,) * 2)
        for occ in combinations(range(self.no), k):
            for vir in combinations(range(self.nv), k):
                val = t[occ + vir]
                if val == 0:
                    continue
                op = sp.identity(2 ** self.n, format='csr')
                for a in vir:
                    op = op @ self.ad[self.no + a]
                for i in reversed(occ):
                    op = op @ self.a[i]
                T = T + val * op
        return T

    def two_body(self, v):
        '''W_N = 1/4 sum <pq||rs> {p+ q+ s r}, normal ordered with respect to
        the reference, up to a constant.'''
        n = self.n
        pairs = list(combinations(range(n), 2))
        ann = {(r, s): self.a[s] @ self.a[r] for r, s in pairs}     # s r
        W = sp.csr_matrix((2 ** n,) * 2)
        for p, q in pairs:
            cre = ann[p, q].T.tocsr()                               # (q p)^+ = p+ q+
            for r, s in pairs:
                if v[p, q, r, s] != 0:
                    W = W + v[p, q, r, s] * (cre @ ann[r, s])
        u = np.einsum('piqi->pq', v[:, :self.no, :, :self.no])
        for p in range(n):
            for q in range(n):
                if u[p, q] != 0:
                    W = W - u[p, q] * (self.ad[p] @ self.a[q])
        return W

    def project(self, vec, k):
        '''x[i.., a..] = <Phi_{i..}^{a..}| vec> as a full antisymmetric tensor
        (|Phi> = a+ b+ .. j i |0>).'''
        no, nv = self.no, self.nv
        out = np.zeros((no,) * k + (nv,) * k)
        from itertools import permutations
        for occ in combinations(range(no), k):
            for vir in combinations(range(nv), k):
                det = self.ref
                for i in occ:                                       # .. j i |0>
                    det = self.a[i] @ det
                for a in reversed(vir):
                    det = self.ad[no + a] @ det
                val = float(det @ vec)
                for po in permutations(range(k)):
                    for pv in permutations(range(k)):
                        idx = tuple(occ[x] for x in po) + tuple(vir[x] for x in pv)
                        out[idx] = _sign(po) * _sign(pv) * val
        return out


def _sign(perm):
    s, p = 1, list(perm)
    for x in range(len(p)):
        while p[x] != x:
            y = p[x]
            p[x], p[y] = p[y], p[x]
            s = -s
    return s


def random_antisym(rng, no, nv, k):
    '''Random amplitude tensor t[i.., a..], antisymmetric in both groups.'''
    from itertools import permutations
    x = rng.standard_normal((no,) * k + (nv,) * k)
    out = np.zeros_like(x)
    for po in permutations(range(k)):
        for pv in permutations(range(k)):
            out += _sign(po) * _sign(pv) * x.transpose(po + tuple(k + y for y in pv))
    return out


def random_integrals(rng, n):
    '''Random real <pq||rs>: antisymmetric in (p,q) and (r,s), symmetric
    under (pq) <-> (rs).'''
    x = rng.standard_normal((n,) * 4)
    x = x - x.transpose(1, 0, 2, 3)
    x = x - x.transpose(0, 1, 3, 2)
    return x + x.transpose(2, 3, 0, 1)
