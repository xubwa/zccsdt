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


# ---------------------------------------------------------------------------
# The left-hand state of coupled-cluster theory, exactly: complex amplitudes,
# the similarity-transformed Hamiltonian as a matrix, the Lambda equations as
# a linear system, densities as expectation values.

def unique_tuples(n, k):
    return list(combinations(range(n), k))


class Left:
    '''exp(-T) H exp(T) and the left-hand state <0| (1 + Lambda) for
    H = sum h[p,q] p+ q + 1/4 sum <pq||rs> p+ q+ s r, given by its Fock
    matrix f[p,q] = h[p,q] + sum_i <pi||qi> and <pq||rs>, and
    T = sum of the excitation operators of ``amps`` = [t1, t2, ..] (dense
    antisymmetric arrays, real or complex), in the Fock space ``fk``.

    The excitation manifold is that of the amplitudes: all determinants
    a+ b+ .. j i |0> of rank 1 .. len(amps), unique ones (i < j < .., a < b
    < ..), rank after rank.  Vectors are over the whole Fock space; the bra
    <x| of a vector x is its transpose, never its conjugate.'''

    def __init__(self, fk, f, v, amps):
        self.fk, self.f, self.v = fk, f, v
        self.nrank = len(amps)
        no, nv = fk.no, fk.nv
        self.H = (one_body(fk, f) + fk.two_body(v)).tocsr()
        self.T = None
        for t in amps:
            x = fk.excitation(t)
            self.T = x if self.T is None else self.T + x
        self.T = self.T.tocsr()
        self.index = [(k, occ, vir) for k in range(1, self.nrank + 1)
                      for occ in unique_tuples(no, k) for vir in unique_tuples(nv, k)]
        self.dets = np.array([self.det(occ, vir) for _, occ, vir in self.index]).T
        self.right = self.exp(fk.ref.astype(complex), 1)

    def det(self, occ, vir):
        '''a+ b+ .. j i |0>.'''
        fk = self.fk
        x = fk.ref
        for i in occ:
            x = fk.a[i] @ x
        for a in reversed(vir):
            x = fk.ad[fk.no + a] @ x
        return x

    def tau(self, occ, vir):
        '''a+ b+ .. j i as a matrix.'''
        fk = self.fk
        op = sp.identity(2 ** fk.n, format='csr')
        for a in vir:
            op = op @ fk.ad[fk.no + a]
        for i in reversed(occ):
            op = op @ fk.a[i]
        return op

    def exp(self, x, sign, left=False):
        '''exp(sign T) x (``left``: x exp(sign T), x a bra).'''
        T = self.T.T if left else self.T
        out = term = np.asarray(x, dtype=complex)
        for k in range(1, self.fk.n + 1):
            term = sign * (T @ term) / k
            out = out + term
        return out

    def hbar(self, x):
        '''exp(-T) H exp(T) x.'''
        return self.exp(self.H @ self.exp(x, 1), -1)

    def hbar_left(self, x):
        '''x exp(-T) H exp(T) for the bra x.'''
        return self.exp(self.H.T @ self.exp(x, -1, left=True), 1, left=True)

    def energy(self):
        '''<0| Hbar |0>, with the energy of the reference.'''
        return self.fk.ref @ self.hbar(self.fk.ref)

    def e_ref(self):
        return self.fk.ref @ (self.H @ self.fk.ref)

    def residuals(self):
        '''<Phi_mu| Hbar |0> over the excitation manifold.'''
        return self.dets.T @ self.hbar(self.fk.ref)

    def bra(self, lam):
        '''<0| (1 + Lambda) for the coefficients ``lam`` over the manifold.'''
        return self.fk.ref + self.dets @ np.asarray(lam)

    def solve(self):
        '''The coefficients of Lambda over the manifold from
        <0| (1 + Lambda) (Hbar - E) |Phi_mu> = 0 for all mu of it: a linear
        system with the matrix of Hbar among the determinants.'''
        E = self.energy()
        cols = np.array([self.hbar(d) for d in self.dets.T]).T          # Hbar |mu>
        A = self.dets.T @ cols - E * np.eye(len(self.index))            # <nu| Hbar - E |mu>
        b = self.fk.ref @ cols                                          # <0| Hbar |mu>
        return np.linalg.solve(A.T, -b)

    def lambda_residual(self, lam):
        '''<0| (1 + Lambda) [Hbar, tau_mu] |0> over the manifold, for any
        amplitudes (zero for the solution if T solves its equations).'''
        bra = self.bra(lam)
        first = self.hbar_left(bra) @ self.dets                         # <0|(1+L) Hbar tau |0>
        hb = self.hbar(self.fk.ref)
        second = np.array([bra @ (self.tau(occ, vir) @ hb) for _, occ, vir in self.index])
        return first - second

    def tensors(self, x):
        '''A vector over the manifold as dense antisymmetric arrays
        [x1[i,a], x2[i,j,a,b], ..].'''
        no, nv = self.fk.no, self.fk.nv
        out = [np.zeros((no,) * k + (nv,) * k, dtype=np.result_type(x)) for k in range(1, self.nrank + 1)]
        from itertools import permutations
        for val, (k, occ, vir) in zip(x, self.index):
            for po in permutations(range(k)):
                for pv in permutations(range(k)):
                    idx = tuple(occ[y] for y in po) + tuple(vir[y] for y in pv)
                    out[k - 1][idx] = _sign(po) * _sign(pv) * val
        return out

    def vector(self, tensors):
        '''The inverse of tensors().'''
        return np.array([tensors[k - 1][occ + vir] for k, occ, vir in self.index])

    def expectation(self, lam, op):
        '''<0| (1 + Lambda) exp(-T) op exp(T) |0>.'''
        return self.exp(self.bra(lam), -1, left=True) @ (op @ self.right)

    def rdm1(self, lam):
        '''d1[p,q] = <0| (1 + Lambda) exp(-T) p+ q exp(T) |0>.'''
        fk = self.fk
        left = self.exp(self.bra(lam), -1, left=True)
        L = [fk.a[p] @ left for p in range(fk.n)]                       # <left| p+
        R = [fk.a[q] @ self.right for q in range(fk.n)]
        return np.array([[l @ r for r in R] for l in L])

    def rdm2(self, lam):
        '''d2[p,q,r,s] = <0| (1 + Lambda) exp(-T) p+ q+ s r exp(T) |0>.'''
        fk = self.fk
        n = fk.n
        left = self.exp(self.bra(lam), -1, left=True)
        pairs = list(combinations(range(n), 2))
        L = np.array([fk.a[q] @ (fk.a[p] @ left) for p, q in pairs])    # <left| p+ q+
        R = np.array([fk.a[s] @ (fk.a[r] @ self.right) for r, s in pairs])
        m = L @ R.T
        out = np.zeros((n,) * 4, dtype=complex)
        for x, (p, q) in enumerate(pairs):
            for y, (r, s) in enumerate(pairs):
                out[p, q, r, s] = out[q, p, s, r] = m[x, y]
                out[q, p, r, s] = out[p, q, s, r] = -m[x, y]
        return out


def one_body(fk, f):
    '''sum f[p,q] p+ q.'''
    op = sp.csr_matrix((2 ** fk.n,) * 2, dtype=np.result_type(f))
    for p in range(fk.n):
        for q in range(fk.n):
            if f[p, q] != 0:
                op = op + f[p, q] * (fk.ad[p] @ fk.a[q])
    return op


def hermitian_integrals(rng, n):
    '''Random complex <pq||rs>, antisymmetric in (p,q) and in (r,s),
    <pq||rs> = conj(<rs||pq>).'''
    x = rng.standard_normal((n,) * 4) + 1j * rng.standard_normal((n,) * 4)
    x = x - x.transpose(1, 0, 2, 3)
    x = x - x.transpose(0, 1, 3, 2)
    return x + x.transpose(2, 3, 0, 1).conj()


def complex_antisym(rng, no, nv, k, scale=1.0):
    return scale * (random_antisym(rng, no, nv, k) + 1j * random_antisym(rng, no, nv, k))
