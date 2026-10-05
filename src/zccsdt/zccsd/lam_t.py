'''The (T) terms of the Lambda equations and of the one-particle density: a
reference on full arrays, for small molecules.

With E(T) in the energy, the Lambda equations get the derivative of E(T) with
respect to t1 and t2 as source terms: constants l1_t, l2_t that are added to
lambda1 and lambda2 after every step of the CCSD Lambda equations (zccsd.lam).
The expressions are those of the user's socutils (cc/lambda_t_zccsd.py on the
branch claude/ccsd-lambda-socutils, 8965b9f, copied here): pyscf.cc
.gccsd_t_lambda with the bare integrals of the source terms conjugated.
Complex spinors need that - lambda is conj(t) in first order, and PySCF's
expressions put the terms on the phase of t; for real integrals the two are
the same.  The one-particle density has (T) terms as well (pyscf.cc
.gccsd_t_rdm): the diagonals of its occupied and virtual blocks, from the
orbital energies in the denominators of E(T), and a virtual-occupied block
from the Fock matrix in the disconnected triples.

For the orbital relaxation (zccsd.zvector) E(T) is also differentiated with
respect to the integrals it is made of: densities() has the coefficients of
<bc||ei>, <ma||jk>, <bc||jk> and f[a,i].

The connected and the disconnected triples are formed here as full o^3 v^3
arrays: the reference.  The code for molecules of any size is in
zccsd.triples (kernel with terms=True: the same sums in the batches of the
(T) energy); by_triples and Terms here are the two steps between them - the
sums over the virtual triples a > b > c in their three roles, and the same
from the triples as the batches deliver them.
'''

import numpy as np

from zccsdt.zccsd import kernels as K


def _blocks(eris):
    '''(ovvv, ooov, oovv, fock, o, v): <ia||bc>, <ij||ka>, <ij||ab> as full
    arrays from the integrals of DirectZCCSD.'''
    from zccsdt.zccsd.zccsd_direct import get_ovvv
    o = eris.nocc
    v = eris.fock.shape[0] - o
    oovv = getattr(eris, 'oovv', None)
    if oovv is None:
        oovv = K.unpack_t2(np.ascontiguousarray(eris.oovv_p), o, v)
    return get_ovvv(eris), np.asarray(eris.ooov), np.asarray(oovv), eris.fock, o, v


def _antisymmetrize(x):
    '''P(a/bc) P(i/jk) of x[i,j,k,a,b,c].'''
    x = x - x.transpose(0, 1, 2, 4, 3, 5) - x.transpose(0, 1, 2, 5, 4, 3)
    return x - x.transpose(1, 0, 2, 3, 4, 5) - x.transpose(2, 1, 0, 3, 4, 5)


def triples(t1, t2, eris):
    '''(t3c, t3d, eia): the connected and the disconnected triples
    [i,j,k,a,b,c] of (T), divided by their denominators.'''
    ovvv, ooov, oovv, fock, o, v = _blocks(eris)
    bcei = ovvv.conj().transpose(3, 2, 1, 0)            # <bc||ei>
    majk = ooov.conj().transpose(2, 3, 0, 1)            # <ma||jk>
    bcjk = oovv.conj().transpose(2, 3, 0, 1)            # <bc||jk>
    mo_e = eris.mo_energy
    eia = mo_e[:o, None] - mo_e[o:]
    d3 = (eia[:, None, None, :, None, None] + eia[None, :, None, None, :, None]
          + eia[None, None, :, None, None, :])
    t3c = _antisymmetrize(np.einsum('jkae,bcei->ijkabc', t2, bcei)
                          - np.einsum('imbc,majk->ijkabc', t2, majk)) / d3
    t3d = _antisymmetrize(np.einsum('ia,bcjk->ijkabc', t1, bcjk)
                          + np.einsum('ai,jkbc->ijkabc', fock[o:, :o], t2)) / d3
    return t3c, t3d, eia


def energy(t1, t2, eris):
    '''E(T) from the full triples (the check of this module against
    zccsd.triples).'''
    t3c, t3d, eia = triples(t1, t2, eris)
    d3 = (eia[:, None, None, :, None, None] + eia[None, :, None, None, :, None]
          + eia[None, None, :, None, None, :])
    return (np.einsum('ijkabc,ijkabc,ijkabc', (t3c + t3d).conj(), d3, t3c) / 36).real


def sources(t1, t2, eris):
    '''(l1_t[i,a], l2_t[i,j,a,b]): the (T) terms of the Lambda equations,
    divided by the denominators of lambda1 and lambda2.'''
    ovvv, ooov, oovv, fock, o, v = _blocks(eris)
    t3c, t3d, eia = triples(t1, t2, eris)
    l1_t = 0.25 * np.einsum('ijkabc,jkbc->ia', t3c.conj(), oovv.conj()) / eia
    m3 = (2 * t3c + t3d).conj()
    tmp = 0.5 * np.einsum('ijkaef,kbfe->ijab', m3, ovvv.conj())
    l2_t = tmp - tmp.transpose(0, 1, 3, 2)
    tmp = 0.5 * np.einsum('imnabc,mnjc->ijab', m3, ooov.conj())
    l2_t -= tmp - tmp.transpose(1, 0, 2, 3)
    l2_t += np.einsum('kc,ijkabc->ijab', fock[:o, o:].conj(), t3c.conj())
    l2_t /= eia[:, None, :, None] + eia[None, :, None, :]
    return l1_t, l2_t


def gamma1(t1, t2, eris):
    '''(goo, gvv, dvo): the (T) terms of the one-particle density in the
    convention of pyscf.cc.gccsd_rdm._gamma1_intermediates - the diagonals
    of doo (subtracted) and dvv (added), and what is added to dvo[a,i].'''
    t3c, t3d, _ = triples(t1, t2, eris)
    goo = np.einsum('iklabc,iklabc->i', (t3c + t3d).conj(), t3c) / 12
    gvv = np.einsum('ijkacd,ijkacd->a', t3c + t3d, t3c.conj()) / 12
    dvo = 0.25 * np.einsum('ijab,ijkabc->ck', t2.conj(), t3c)
    return goo, gvv, dvo


def densities(t1, t2, eris):
    '''(d1[b,c,e,i], d2[m,a,j,k], d3[b,c,j,k], df[a,i]): the densities of (T)
    for the orbital gradient - the coefficients of <bc||ei>, <ma||jk>,
    <bc||jk> and of the Fock matrix elements f[a,i] in the first-order change
    of the real part of E(T),

        d Re E(T) = Re sum (d1 d<bc||ei> + d2 d<ma||jk> + d3 d<bc||jk> + df df[a,i])
                    - sum_i goo[i] de_i + sum_a gvv[a] de_a

    (goo, gvv of gamma1; df = conj(dvo)).  E(T) is not linear in the
    integrals: with t3c, t3d the connected and the disconnected triples it is
    conj(t3c + t3d) D t3c / 36, the integrals of t3c enter twice - through
    conj(2 t3c + t3d), as t2 does in sources() - and those of t3d once.'''
    t3c, t3d, _ = triples(t1, t2, eris)
    m3, wc = (2 * t3c + t3d).conj(), t3c.conj()
    return (0.25 * np.einsum('ijkabc,jkae->bcei', m3, t2),
            -0.25 * np.einsum('ijkabc,imbc->majk', m3, t2),
            0.25 * np.einsum('ijkabc,ia->bcjk', wc, t1),
            0.25 * np.einsum('ijkabc,jkbc->ai', wc, t2))


def by_triples(t1, t2, eris):
    '''(l1, l2, goo, gvv, dvo): the source terms of sources() before the
    division by the denominators (l1[i,a], l2[i,j,a,b]) and the density
    terms of gamma1(), assembled from the virtual triples a > b > c alone.

    This is the form in which the batches of zccsd.triples have the
    triples: w = t3c[:, :, :, a, b, c] and m = conj(2 t3c + t3d) of one
    triple, antisymmetric in i, j, k.  Each triple contributes in three
    roles, one for each of its virtual indices s, with the other two as a
    pair p > q and the sign of the permutation (s, p, q) of (a, b, c): + for
    s = a (b, c), - for s = b (a, c), + for s = c (a, b):

      l1[i, s]       += sign sum_{j>k} conj(w[i,j,k]) conj(<jk||pq>)
      tmp[i, j, s, e] -= sign sum_k m[i,j,k] conj(<ke||pq>)      l2 += tmp - tmp(s <-> e)
      l2[i, j, (pq)]  -= sign P(ij) sum_{m>n} m[i,m,n] conj(<mn||js>)
      l2[i, j, (pq)]  += sign sum_k conj(f[k,s]) conj(w[i,j,k])
      dvo[s, k]       += sign sum_{i>j} conj(t2[i,j,p,q]) w[i,j,k]
      goo[i]          += sum_{k>l} conj(w + v)[i,k,l] w[i,k,l]
      gvv[s]          += sum_{i>j>k} (w + v)[i,j,k] conj(w[i,j,k])

    (l2[i, j, (pq)]: the element p, q and minus it at q, p.)  The triples
    themselves come from the full arrays of triples() here: this function
    checks the assembly, for the evaluation in batches to follow it.'''
    ovvv, ooov, oovv, fock, o, v = _blocks(eris)
    t3c, t3d, _ = triples(t1, t2, eris)
    fov = fock[:o, o:]
    l1 = np.zeros((o, v), dtype=np.complex128)
    tmp = np.zeros((o, o, v, v), dtype=np.complex128)
    l2 = np.zeros((o, o, v, v), dtype=np.complex128)
    goo, gvv = np.zeros(o, dtype=np.complex128), np.zeros(v, dtype=np.complex128)
    dvo = np.zeros((v, o), dtype=np.complex128)
    for a in range(v):
        for b in range(a):
            for c in range(b):
                w = t3c[:, :, :, a, b, c]
                u = w + t3d[:, :, :, a, b, c]
                m = (w + u).conj()
                goo += 0.5 * np.einsum('ikl,ikl->i', u.conj(), w)
                x = np.einsum('ijk,ijk', u, w.conj()) / 6
                for s, p, q, sign in ((a, b, c, 1.0), (b, a, c, -1.0), (c, a, b, 1.0)):
                    gvv[s] += x
                    l1[:, s] += 0.5 * sign * np.einsum('ijk,jk->i', w.conj(), oovv[:, :, p, q].conj())
                    tmp[:, :, s] -= sign * np.einsum('ijk,ke->ije', m, ovvv[:, :, p, q].conj())
                    y = 0.5 * np.einsum('imn,mnj->ij', m, ooov[:, :, :, s].conj())
                    y = sign * (np.einsum('k,ijk->ij', fov[:, s].conj(), w.conj()) - y + y.T)
                    l2[:, :, p, q] += y
                    l2[:, :, q, p] -= y
                    dvo[s] += 0.5 * sign * np.einsum('ij,ijk->k', t2[:, :, p, q].conj(), w)
    return l1, l2 + tmp - tmp.transpose(0, 1, 3, 2), goo, gvv, dvo


class Terms:
    '''The (T) terms of by_triples from the triples as zccsd.triples.kernel
    visits them in its batches: add() takes the connected triples w and
    w + v of one virtual triple a > b > c, divided by the denominator and
    packed over i > j > k; result() returns (l1, l2, goo, gvv, dvo) as
    by_triples does.  With Kramers pairs the kernel visits the triples with
    an unbarred b; time reversal maps them onto the others, and every sum is
    completed by its image.

    With numpy, one triple at a time: the check of the batches and of the
    time reversal.  triples.kernel(terms=True) does the same sums with two
    GEMMs per product of a batch and in its kernel.'''

    def __init__(self, t2, eris):
        self.ovvv, self.ooov, self.oovv, fock, o, v = _blocks(eris)
        self.fov = fock[:o, o:]
        self.t2 = t2
        self.kramers = False
        self.ijk = np.array([(i, j, k) for i in range(o) for j in range(i) for k in range(j)]).T
        self.l1 = np.zeros((o, v), dtype=np.complex128)
        self.tmp = np.zeros((o, o, v, v), dtype=np.complex128)
        self.l2 = np.zeros((o, o, v, v), dtype=np.complex128)
        self.goo = np.zeros(o, dtype=np.complex128)
        self.gvv = np.zeros(v, dtype=np.complex128)
        self.dvo = np.zeros((v, o), dtype=np.complex128)

    def _full(self, x):
        '''x[i,j,k], antisymmetric, from the elements with i > j > k.'''
        o = self.goo.size
        i, j, k = self.ijk
        out = np.zeros((o, o, o), dtype=np.complex128)
        out[i, j, k] = out[j, k, i] = out[k, i, j] = x
        out[j, i, k] = out[i, k, j] = out[k, j, i] = -x
        return out

    def add(self, a, b, c, w, u):
        w, u = self._full(w), self._full(u)
        m = (w + u).conj()
        self.goo += 0.5 * np.einsum('ikl,ikl->i', u.conj(), w)
        x = np.einsum('ijk,ijk', u, w.conj()) / 6
        for s, p, q, sign in ((a, b, c, 1.0), (b, a, c, -1.0), (c, a, b, 1.0)):
            self.gvv[s] += x
            self.l1[:, s] += 0.5 * sign * np.einsum('ijk,jk->i', w.conj(),
                                                    self.oovv[:, :, p, q].conj())
            self.tmp[:, :, s] -= sign * np.einsum('ijk,ke->ije', m, self.ovvv[:, :, p, q].conj())
            y = 0.5 * np.einsum('imn,mnj->ij', m, self.ooov[:, :, :, s].conj())
            y = sign * (np.einsum('k,ijk->ij', self.fov[:, s].conj(), w.conj()) - y + y.T)
            self.l2[:, :, p, q] += y
            self.l2[:, :, q, p] -= y
            self.dvo[s] += 0.5 * sign * np.einsum('ij,ijk->k', self.t2[:, :, p, q].conj(), w)

    def result(self):
        l2 = self.l2 + self.tmp - self.tmp.transpose(0, 1, 3, 2)
        out = [self.l1, l2, self.goo, self.gvv, self.dvo]
        if self.kramers:
            from zccsdt.zccsd.kramers import image, partner_signs
            for n, x in enumerate(out):
                if x.ndim == 1:         # the diagonal of a matrix
                    out[n] = x + x[partner_signs(x.size)[0]].conj()
                else:
                    out[n] = x + image(x)
        return tuple(out)


def in_batches(mycc, eris, t1, t2):
    '''(E(T), (l1, l2, goo, gvv, dvo)): the (T) energy and the terms of
    by_triples from the batches of zccsd.triples.kernel, one triple at a
    time (Terms; l2[i,j,a,b] unpacked).'''
    from zccsdt.zccsd import triples as _triples
    terms = Terms(t2, eris)
    e = _triples.kernel(mycc, eris, t1, t2, mycc.verbose, terms=terms)
    return e, terms.result()
