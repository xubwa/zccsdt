'''The one-particle density and the orbital gradient of CCSDT from packed
amplitudes, and the relaxed density through the Z-vector equations.

The equations are those of zccsdt.cc.lambda_terms (tables DENSITY and
GRADIENT, T1-dressed formalism; docs/lambda.md) and the dense reference is
zccsdt.cc.lambda_ref.  Here t3 and lambda3 are the packed arrays [C(o,3),
C(v,3)] of the amplitude code, and no array with more than three virtual
indices is formed; those with three are o v^3 / 2 elements.

How the 81 terms of the gradient are evaluated.  The gradient is the
de-excitation one-index transformation of the Lagrangian in the dressed
basis,

    Gnh[a,i] = f[a,i] + sum_q f[a,q] g1[i,q] - sum_p f[p,i] g1[p,a]
             + sum_pq <pa||qi> g1[p,q]
             + 1/2 sum <aq||rs> g2[i,q,r,s] - 1/2 sum <pq||is> g2[p,q,a,s],

g1 and g2 the coefficients of the dressed f and <pq||rs> in the Lagrangian
(the blocks of DENSITY, g2 antisymmetrized).  Every block of g2 but one is
made and then contracted with its block of dressed integrals:

    block     elements     made of                              meets
    ijkl      o^4 / 4      l2 t2, l3 t3 (O4)                    <am||jk>
    ijak      o^3 v        l1 t2, l2 t3, l3 t2 t2               <am||ej>, <mn||ij>
    iajk      o^3 v / 2    l3 t2 (Z)                            <ae||jk>
    ijab      o^2 v^2 / 4  t2, l1 t3, l2 t2 t2, l3 t2 t3        <am||ef>, <mn||ie>
    iabj      o^2 v^2      l2 t2, l3 t3 (R3: the ring density)  <ae||fm>, <em||ij>
    abci      o v^3 / 2    l3 t2 (C3)                           <ef||im>
    aibc      o v^3 / 2    l1 t2, l2 t3, l3 t2 t2               <ae||fg>, <em||if>
    abcd      -            l2 t2, l3 t3: not made               <ef||gi>

The block abcd meets <ef||gi> through its factors: l2 with the integrals
first (o^2 v^2), and lambda3 with the integrals first,
Y[P,g,m,h,i] = sum_(e<f) l3[m,P,g,e,f] <ef||hi> for a batch of occupied
pairs P, which is contracted with t3 of the same batch and dropped.

Passes over the packed arrays.  Both are read through the half-expanded
batches of the amplitude code (t3pack.gather_half: X[P, d, m, Q] for
occupied pairs P and virtual pairs Q, a matrix (P d) x (m Q)):

  1. lambda3 alone: Z[P,g,i] = sum l3 t2 over (m, Q) and C3[Q,c,i] = sum
     l3 t2 over (P, e) - one GEMM each per batch (o^4 v^3 / 4, o^3 v^4 / 4);
  2. t3 and lambda3 together, batch by batch:
       - t3 with l2 and Z stacked, over (P, d): the l2 t3 part of aibc and
         the term of ijab with Z (one GEMM, o^3 v^4 / 4);
       - t3 with l2 and C3 stacked, over (m, Q): the l2 t3 part of ijak and
         the term of ijab with C3 (one GEMM, o^3 v^4 / 4);
       - t3 with l1 and with l2 (matrix-vector): ijab and the density ia;
       - the ring density R3[(a j), (b i)] = sum_(P,Q) l3 t3, one GEMM per
         occupied pair (o^4 v^4 / 4); its traces are the l3 t3 parts of the
         densities ab and ij;
       - lambda3 with <ef||hi> (Y, o^4 v^4 / 4) and then with t3;
       - lambda3 with t2 over Q and then with t3 (o^5 v^3 / 4): the term
         l3 t2[ij,ef] t3[mno,abg] of ijab;
  3. the Gram matrix of the rows, M[I,J] = sum_A l3[I,A] t3[J,A], one GEMM
     on the packed arrays as they are (C(o,3)^2 C(v,3)): O4, the l3 t3 part
     of ijkl.

So t3 is gathered once and lambda3 twice; the two o^4 v^4 / 4 products are
the cost (a CCSDT iteration has one, and the o^3 v^5 / 24 ladder, which is
not here).  The packed dressed <ab||de> is used twice in slabs of rows
(ccsdt_opt.dressed_vvvv_rows): for the dressed <ef||gi> and <ef||ij> before
the passes, and with the block aibc after them; it is dressed once if all
of it fits.

Memory (plan_memory): the o v^3 / 2 arrays (the dressed <ef||gi>, C3, the
block aibc, the dressed <ma||ef>, the bare packed <ma||ef>), the batch
buffers of pass 2 (two half-expanded batches and Y) and the slab of dressed
<ab||de>; MemoryError if one occupied pair per batch and the rows of one
first index do not fit.

Timers (zccsdt.timing): 'dens.dress', 'dens.vvvv', 'dens.pass_l',
'dens.pass_tl' (with .gather, .t, .ring, .y, .h, .frozen), 'dens.gram',
'dens.blocks', 'dens.gradient', 'dens.frozen', 'dens.zvector'.
'''

from dataclasses import dataclass

import numpy as np
from pyscf import lib

from zccsdt import memory
from zccsdt.blas import gemm
from zccsdt.cc import ccsdt_opt, t3pack
from zccsdt.timing import timer

einsum = lib.einsum
BATCH_BYTES = 4 << 30       # the batch buffers of a pass when memory is not short
FROZEN_BYTES = 2 << 30      # the integrals with one frozen index held at a time


@dataclass
class Plan:
    '''The memory plan of kernel() (plan_memory).'''
    nb: int = None              # occupied pairs per batch of the passes
    slab_bytes: int = None      # dressed rows of <ab||de> held at a time (None: all, kept)
    need: int = 0               # bytes of ordinary memory the plan takes


@dataclass
class Result:
    d1: np.ndarray              # d1[p,q] = <p+ q>: response density, reference included, not Hermitian
    gnh: np.ndarray             # Gnh[a,i] of zccsd.zvector (None without ``gradient``)
    g1: np.ndarray              # the coefficient of the dressed f[p,q] in the Lagrangian
    plan: Plan = None
    bf: np.ndarray = None       # with frozen spinors f: 1/2 sum <fq||rs> g2[x,q,r,s] and
    kf: np.ndarray = None       # 1/2 sum <pq||fs> g2[p,q,x,s] as [f, x], see frozen_terms


def frozen_batch(nocc, nvir, nfrozen, itemsize=16):
    '''Frozen spinors whose integrals <fq||rs> (n^3 each: the bare ones and
    two dressed forms, and what makes them) are held at a time:
    FROZEN_BYTES of them, at least one.'''
    per = 5 * (nocc + nvir) ** 3 * itemsize
    return int(max(1, min(nfrozen, FROZEN_BYTES // max(per, 1))))


def fixed_bytes(nocc, nvir, itemsize=16, nfrozen=0):
    '''Bytes of the arrays of kernel() that do not depend on the plan: five
    o v^3 / 2 arrays at a time, the transient of the term of aibc with two
    t2, about twenty o^2 v^2 arrays and the Gram matrix of the occupied
    triples, and what the threaded BLAS keeps.  With ``nfrozen`` frozen
    spinors: two arrays of v^3 / 2 elements for each of them, and the
    integrals of a batch of them (frozen_batch).'''
    o, v = nocc, nvir
    nvp, npo, nto = v * (v - 1) // 2, o * (o - 1) // 2, o * (o - 1) * (o - 2) // 6
    half = o * v * nvp
    frozen = (2 * v * nvp + 3 * npo * v) * nfrozen \
        + 5 * (o + v) ** 3 * (frozen_batch(o, v, nfrozen, itemsize) if nfrozen else 0)
    return (5 * (o + v) * o * nvp + 4 * o * v * v + 20 * o * o * v * v + 3 * o * o * nvp
            + 2 * nto * nto + 6 * o ** 3 * v + 4 * npo * npo + half // max(v, 1) + frozen) * itemsize \
        + ccsdt_opt.overhead()


def pair_bytes(nocc, nvir, itemsize=16, nfrozen=0):
    '''Bytes of the batch buffers of pass 2 per occupied pair: t3 and
    lambda3 half-expanded, Y and the product with t2 (and the two products
    with the integrals of the frozen spinors).'''
    o, v = nocc, nvir
    nvp, npo = v * (v - 1) // 2, o * (o - 1) // 2
    return (2 * v * o * nvp + v * o * (v * o + npo) + 2 * v * o * v * nfrozen) * itemsize


def plan_memory(nocc, nvir, itemsize=16, budget=None, pp_mapped=False, nfrozen=0):
    '''Plan for ``budget`` bytes of ordinary memory beyond the amplitudes,
    lambda and what the integral object holds (None: no limit).  The batches
    of the passes take what is left of the budget, up to BATCH_BYTES; the
    dressed <ab||de> is kept whole if it fits next to them, and is else
    dressed twice in slabs of rows, as thick as fit (the slab and the
    batches are not alive together).  MemoryError below one occupied pair
    per batch and the rows of one first virtual index.  ``pp_mapped``: the
    bare packed <ab||de> is a mapped array (else it is in memory and
    counted with the integral object).  ``nfrozen``: frozen occupied and
    dropped virtual spinors whose integrals are given (kernel(frozen=)).'''
    o, v = nocc, nvir
    nvp, npo = v * (v - 1) // 2, max(o * (o - 1) // 2, 1)
    fixed, unit = fixed_bytes(o, v, itemsize, nfrozen), max(pair_bytes(o, v, itemsize, nfrozen), 1)
    pp, row = nvp * nvp * itemsize, max(nvp * itemsize, 1)
    nb_most = min(npo, max(1, BATCH_BYTES // unit))
    if budget is None:
        return Plan(nb=nb_most, slab_bytes=None, need=fixed + nb_most * unit + pp)
    left = budget - fixed
    row_min = min(v * row, pp)
    if left < max(unit, row_min):
        raise MemoryError('The densities of CCSDT with %d occupied and %d virtual spin orbitals need '
                          'about %.1f GB of memory for their o v^3 arrays and work buffers; %.1f GB '
                          'are available (max_memory, or what the node and the memory limit of the '
                          'job leave)' % (o, v, (fixed + max(unit, row_min)) / 1e9, budget / 1e9))
    nb = int(min(nb_most, left // unit))
    if left >= pp + nb * unit:
        return Plan(nb=nb, slab_bytes=None, need=fixed + pp + nb * unit)
    slab = int(min(pp, left // row * row))
    return Plan(nb=nb, slab_bytes=slab, need=fixed + max(nb * unit, slab))


def _packed3(eris, x, dtype):
    '''t3 or lambda3 as a C-contiguous [C(o,3), C(v,3)] array in the
    lexicographic order of the triples (a view where it can be).'''
    om, vm = t3pack.maps_for(eris)
    if om.spins is not None:            # the blocks of an engine with labels
        x = ccsdt_opt.packed_t3(eris, x)
    o = eris.nocc
    v = np.asarray(eris.fock).shape[0] - o
    n3 = (o * (o - 1) * (o - 2) // 6, v * (v - 1) * (v - 2) // 6)
    x = np.asarray(x)
    if x.size == 0:
        return np.zeros((0, n3[1]), dtype=dtype)
    if x.ndim == 6:
        from zccsdt.cc import ccsdt
        x = ccsdt.pack_t3(x)
    return np.ascontiguousarray(x.reshape(n3), dtype=dtype)


def _unpack_occ(x, om):
    '''[i, j, ...] antisymmetric from x[(i<j), ...].'''
    mm, nn = om.pairs[:, 0], om.pairs[:, 1]
    out = np.zeros((om.n, om.n) + x.shape[1:], dtype=x.dtype)
    out[mm, nn] = x
    out[nn, mm] = -x
    return out


def _unpack_vir(x, vm):
    '''[..., a, b] antisymmetric from x[..., (a<b)].'''
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
    out = np.zeros(x.shape[:-1] + (vm.n, vm.n), dtype=x.dtype)
    out[..., md, ne] = x
    out[..., ne, md] = -x
    return out


def _bra_pairs(g, W, D, pairs, sign):
    '''g[x,i] += sign sum_X W[r,X] D[y,i,X] and g[y,i] -= sign sum_X W[r,X]
    D[x,i,X] for the rows r = (x<y) = pairs[r] of W: the contraction of an
    array antisymmetric in its first two (virtual) indices, of which W has
    rows of pairs, with D[e, i, X] over the second of them and X.'''
    xs, ys = pairs[:, 0], pairs[:, 1]
    nr = len(xs)
    if nr == 0:
        return
    W = W.reshape(nr, -1)
    nv, no = D.shape[0], D.shape[1]
    D = D.reshape(nv, no, -1)
    # by the first index: rows that follow each other
    cut = np.flatnonzero(np.diff(xs)) + 1
    for r0, r1 in zip(np.concatenate(([0], cut)), np.concatenate((cut, [nr]))):
        g[ys[r0:r1]] -= sign * gemm(W[r0:r1], D[xs[r0]].T)
    # by the second
    order = np.argsort(ys, kind='stable')
    cut = np.flatnonzero(np.diff(ys[order])) + 1
    for r0, r1 in zip(np.concatenate(([0], cut)), np.concatenate((cut, [nr]))):
        rows = order[r0:r1]
        g[xs[rows]] += sign * gemm(np.ascontiguousarray(W[rows]), D[ys[rows[0]]].T)


class _Slabs:
    '''The packed T1-dressed <ab||de> in slabs of its rows (a<b): all of it,
    dressed once and kept, or ``slab_bytes`` at a time, dressed again for
    every sweep.'''

    def __init__(self, eris, t1, om, vm, dtype, slab_bytes):
        self.args = (eris, t1, om, vm)
        nvp = vm.npair
        self.slabs = ccsdt_opt.pair_slabs(nvp, nvp * np.dtype(dtype).itemsize, slab_bytes)
        self.keep = slab_bytes is None
        self.dtype, self.buf, self.done = dtype, None, False

    def __iter__(self):
        nvp = self.args[3].npair
        if nvp == 0:
            return
        if self.buf is None:
            rows = max(q1 - q0 for q0, q1 in self.slabs)
            self.buf = np.empty((rows, nvp), dtype=self.dtype)
        for q0, q1 in self.slabs:
            W = self.buf[:q1 - q0]
            if not (self.keep and self.done):
                with timer('dens.vvvv'):
                    ccsdt_opt.dressed_vvvv_rows(*self.args, q0, q1, W)
            yield q0, q1, W
        self.done = True
        if not self.keep:
            self.buf = None

    def close(self):
        self.buf = None


def _dressed_frozen(S0, x, y, dtype):
    '''(S[f,q,r,s], W[f,s,p,q]) from the bare S0[f,u,v,w] = <fu||vw> of a
    batch of frozen spinors f with the active ones: S = <fq||rs> with q
    dressed as a bra index and r, s as ket indices, and W[f,s,p,q] =
    <pq||fs> = conj(<fs||pq>) with p, q dressed as bra indices and s as a
    ket index (h~ = x h y^T).'''
    S0 = np.asarray(S0, dtype=dtype)
    S = einsum('fuvw,qu->fqvw', S0, x)
    S = einsum('fqvw,rv->fqrw', S, y)
    S = einsum('fqrw,sw->fqrs', S, y)
    W = einsum('fwtu,sw->fstu', S0.conj(), y)
    W = einsum('fstu,pt->fspu', W, x)
    W = einsum('fspu,qu->fspq', W, x)
    return np.ascontiguousarray(S), np.ascontiguousarray(W)


def kernel(eris, t1, t2, t3, l1, l2, l3, gradient=True, max_memory=None, plan=None, frozen=None):
    '''The response density and the orbital gradient of CCSDT for the
    integrals ``eris`` (zccsdt.eris.ERIs, spin orbitals or spinors), the
    amplitudes t1[i,a], t2[i,j,a,b], t3 packed [C(o,3), C(v,3)] (as the
    solver returns it) and lambda in the same layouts.  Returns a Result:

      d1[p,q] = <0| (1 + Lambda) exp(-T) p+ q exp(T) |0>  (lambda_ref.densities)
      gnh[a,i]  the table GRADIENT: Gnh[a,i] of zccsd.zvector for converged
                t and lambda, twice the orbital gradient G[a,i]

    over the spin orbitals of ``eris`` [occupied, virtual].  ``max_memory``:
    MB for the whole process (default: 80% of what the node and the job
    leave); ``plan``: a Plan given explicitly.

    ``frozen`` = (number of frozen spinors, function): with frozen occupied
    or dropped virtual spinors f, function(f0, f1) gives the bare
    <fu||vw>[f0:f1] with the spinors u, v, w of ``eris`` (frozen_integrals;
    it is called twice for every batch of frozen_batch() of them).  The
    Result then has bf[f,x] and kf[f,x], the contractions of the
    two-particle density with these integrals that the orbital gradient
    between frozen and active spinors is made of (frozen_terms).'''
    nocc, nvir = t1.shape
    o, v, n = nocc, nvir, nocc + nvir
    om, vm = t3pack.triple_map(o), t3pack.triple_map(v)
    mm, nn = om.pairs[:, 0], om.pairs[:, 1]
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
    nP, nQ = om.npair, vm.npair
    if t3 is None or l3 is None:        # CCSD
        t3 = l3 = np.zeros(0)
    dtype = np.result_type(t1, t2, t3, l1, l2, l3, eris.fock, eris.oovv)
    isz = np.dtype(dtype).itemsize
    if plan is None:
        from zccsdt.scratch import is_mapped
        plan = plan_memory(o, v, isz, memory.budget(max_memory),
                           is_mapped(getattr(eris, '_vvvv_pp', None)), frozen[0] if frozen else 0)
    nfz, fints = frozen if frozen and gradient else (0, None)
    nfb = frozen_batch(o, v, nfz, isz) if nfz else 1
    t1, l1 = (np.ascontiguousarray(x, dtype=dtype) for x in (t1, l1))
    t2, l2 = (np.asarray(x, dtype=dtype) for x in (t2, l2))
    T3p, L3p = _packed3(eris, t3, dtype), _packed3(eris, l3, dtype)
    triples = T3p.size > 0
    t2mn, l2mn = np.ascontiguousarray(t2[mm, nn]), np.ascontiguousarray(l2[mm, nn])     # [P, a, b]
    t2pp, l2pp = np.ascontiguousarray(t2mn[:, md, ne]), np.ascontiguousarray(l2mn[:, md, ne])
    t2P, l2P = np.ascontiguousarray(t2[:, :, md, ne]), np.ascontiguousarray(l2[:, :, md, ne])    # [i, j, Q]

    # ---- the dressed integrals of the passes: <ef||gi> and <ef||ij>
    with timer('dens.dress'):
        fd = ccsdt_opt.dress_fock(eris, t1)
        Vvo = ccsdt_opt.bra_dressed_vvvo_pp(eris, t1, om, vm).astype(dtype, copy=False)    # [Q, g, i]
        vvoo = ccsdt_opt.dressed_vvoo_pp(eris, t1, Vvo, om, vm).astype(dtype, copy=False)  # [P, Q]
        tau = np.ascontiguousarray(ccsdt_opt._tau1_pairs(t1, om, vm), dtype=dtype)
    slabs = _Slabs(eris, t1, om, vm, dtype, plan.slab_bytes)
    for q0, q1, W in slabs:
        with timer('dens.dress'):
            t3pack.pair_t1_contract(Vvo[q0:q1], W, t1, vm)
            vvoo[:, q0:q1] += gemm(W, tau.T).T
    del tau
    xm = np.eye(n, dtype=dtype)
    xm[o:, :o] -= t1.T
    ym = np.eye(n, dtype=dtype)
    ym[:o, o:] += t1
    Vf = Sq = None
    if nfz:
        # the integrals of the passes with a frozen index: <ef||hf'> and <f'e||gh>, dressed
        with timer('dens.frozen'):
            Vf = np.empty((nQ, v, nfz), dtype=dtype)        # <Q||h f>
            Sq = np.empty((nQ, v, nfz), dtype=dtype)        # <f e||Q>
            for f0 in range(0, nfz, nfb):
                f1 = min(f0 + nfb, nfz)
                S0 = fints(f0, f1)
                with timer('dens.frozen.dress'):
                    S, Wf = _dressed_frozen(S0, xm, ym, dtype)
                del S0
                Sq[:, :, f0:f1] = S[:, o:, o:, o:][:, :, md, ne].transpose(2, 1, 0)
                Vf[:, :, f0:f1] = -Wf[:, o:, o:, o:][:, :, md, ne].transpose(2, 1, 0)
                del S, Wf

    # ---- pass 1: lambda3 with t2
    Z = np.zeros((nP, v, o), dtype=dtype)           # Z3[i,(n<o),g] = 2 Z[P,g,i]
    GyT = np.zeros((o, nQ, o + v), dtype=dtype)     # [m, Q, k | a]: l2[k,m,Q] | C3[Q,a,m]
    nb = max(1, plan.nb or 1)
    if triples:
        with timer('dens.pass_l'):
            C3x = np.zeros((v, o * nQ), dtype=dtype)                                # [c, (i Q)]
            GyL = np.ascontiguousarray(t2P.transpose(1, 2, 0)).reshape(o * nQ, o)   # [(m Q), i]
            for p0 in range(0, nP, nb):
                p1 = min(p0 + nb, nP)
                k = p1 - p0
                L = t3pack.gather_half(L3p, om, vm, np.arange(p0, p1))
                Lf = L.reshape(k * v, o * nQ)
                # sum_(P,e) t2[P,c,e] L[P,e,i,Q]
                gemm(t2mn[p0:p1].reshape(k * v, v).T, Lf, C3x, alpha=-1.0, beta=1.0)
                gemm(Lf, GyL, Z[p0:p1].reshape(k * v, o))
                del L, Lf
            t3pack.axpby(GyT[:, :, o:], C3x.reshape(v, o, nQ).transpose(1, 2, 0), 2.0, 0.0)
            del C3x, GyL
    t3pack.axpby(GyT[:, :, :o], l2P.transpose(1, 2, 0), 1.0, 0.0)
    C3 = GyT[:, :, o:]                              # C3[m, Q, a] = sum_(no,e) l3[m,n,o,Q,e] t2[n,o,a,e]

    # ---- pass 2: t3, and t3 with lambda3
    XT = np.zeros((v + o, o * nQ), dtype=dtype)     # E2[a,(i Q)] | X10[i,(j Q)]
    YT = np.zeros((nP, v, o + v), dtype=dtype)      # Y2k[P,a,k] | Y12[P,b,a]
    B1 = np.zeros(v * o, dtype=dtype)               # sum_(P,Q) T[P,a,i,Q] l2[P,Q]
    A1 = np.zeros((nP, nQ), dtype=dtype)            # sum_(m,e) l1[m,e] T[P,e,m,Q]
    R3 = np.zeros((v * o, v * o), dtype=dtype)      # [(a j), (b i)]
    Kg = np.zeros((nP, nQ), dtype=dtype)
    S13 = np.zeros((v, o), dtype=dtype)
    SK = np.zeros((v, nfz), dtype=dtype)            # as S13, with <Q||h f>
    SB = np.zeros((v, nfz), dtype=dtype)            # t3 with <f e||Q> first, then lambda3
    if triples:
        with timer('dens.pass_tl'):
            GxT = np.concatenate((-l2mn, 2.0 * Z), axis=2)                          # [P, d, a | i]
            GyT2 = GyT.reshape(o * nQ, o + v)
            l1v = np.ascontiguousarray(l1.T).ravel()                                # [(e m)]
            Vvo2 = Vvo.reshape(nQ, v * o)
            t2ppT = np.ascontiguousarray(t2pp.T)
            Y1 = np.empty((nb * v * o, v * o), dtype=dtype)
            H = np.empty((nb * v * o, nP), dtype=dtype)
            Oq = np.empty((nQ, o, o), dtype=dtype)
            Om = np.empty((o, nQ, o), dtype=dtype)

            def yterm(Ar, B2, ncol, C, out, Ybuf):
                '''out[a, y] += sum_(m, g<h) C[P, a, m, (gh)] (Y[g,h] - Y[h,g]) with
                Y[(P g m), (h y)] = Ar B2, for the pairs P of the batch.'''
                k = C.shape[0]
                gemm(Ar, B2, Ybuf[:k * v * o])
                oq, om_ = (Oq, Om) if ncol == o else (np.empty((nQ, o, ncol), dtype=dtype),
                                                      np.empty((o, nQ, ncol), dtype=dtype))
                for j in range(k):
                    oq[...] = 0
                    t3pack.pair_antisym_add(oq, Ybuf[j * v * o:(j + 1) * v * o], vm,
                                            (o * v * ncol, ncol, v * ncol, 1))
                    t3pack.axpby(om_, oq.transpose(1, 0, 2), 1.0, 0.0)
                    gemm(C[j].reshape(v, o * nQ), om_.reshape(o * nQ, ncol), out, beta=1.0)

            if nfz:
                Yf = np.empty((nb * v * o, v * nfz), dtype=dtype)
                Vf2, Sq2 = Vf.reshape(nQ, v * nfz), Sq.reshape(nQ, v * nfz)
            for p0 in range(0, nP, nb):
                p1 = min(p0 + nb, nP)
                k = p1 - p0
                pb = np.arange(p0, p1)
                with timer('dens.pass_tl.gather'):
                    T = t3pack.gather_half(T3p, om, vm, pb)
                    L = t3pack.gather_half(L3p, om, vm, pb)
                Tf, Tr = T.reshape(k * v, o * nQ), T.reshape(k * v * o, nQ)
                Lr = L.reshape(k * v * o, nQ)
                with timer('dens.pass_tl.t'):
                    gemm(GxT[p0:p1].reshape(k * v, v + o).T, Tf, XT, beta=1.0)
                    gemm(Tf, GyT2, YT[p0:p1].reshape(k * v, o + v))
                    for j in range(k):
                        Tm = Tr[j * v * o:(j + 1) * v * o]
                        B1 += Tm @ l2pp[p0 + j]
                        A1[p0 + j] = l1v @ Tm
                    del Tm                  # (a view: it would keep the batch alive)
                with timer('dens.pass_tl.ring'):
                    for j in range(k):
                        gemm(Lr[j * v * o:(j + 1) * v * o], Tr[j * v * o:(j + 1) * v * o].T, R3,
                             beta=1.0)
                with timer('dens.pass_tl.h'):
                    gemm(Lr, t2ppT, H[:k * v * o])
                    gemm(H[:k * v * o].T, Tr, Kg, beta=1.0)
                if gradient:
                    with timer('dens.pass_tl.y'):
                        # Y[(P g m), (h i)], antisymmetrized in (g, h), with T[P, a, m, (g<h)]
                        yterm(Lr, Vvo2, o, T, S13, Y1)
                if nfz:
                    with timer('dens.pass_tl.frozen'):
                        yterm(Lr, Vf2, nfz, T, SK, Yf)
                        yterm(Tr, Sq2, nfz, L, SB, Yf)
                del T, L, Tf, Tr, Lr
            del GxT, Y1, H, Oq, Om, Vvo2
            if nfz:
                del Yf, Vf2, Sq2
    R3 = R3.reshape(v, o, v, o)
    B1 = B1.reshape(v, o)

    # ---- pass 3: the Gram matrix of the occupied triples
    O4 = np.zeros((nP, nP), dtype=dtype)            # [(k<l), (i<j)]: sum_(m,A) l3[m,k,l,A] t3[m,i,j,A]
    if triples:
        with timer('dens.gram'):
            M = gemm(L3p, T3p.T)
            for m in range(o):
                idx, s = om.tri[m], om.sgn[m].astype(np.float64)
                O4 += (s[:, None] * s[None, :]) * M[np.ix_(idx, idx)]
            del M

    # ---- the blocks of the densities
    with timer('dens.blocks'):
        Rl2 = einsum('jmae,imbe->ajbi', l2, t2)
        g_ab = einsum('Pae,Pbe->ab', l2mn, t2mn) + einsum('ambm->ab', R3) / 3.0
        g_ij = -einsum('jmQ,imQ->ij', l2P, t2P) - einsum('ejei->ij', R3) / 3.0
        g_ia = einsum('me,imae->ia', l1, t2) + B1.T + einsum('Pgi,Pag->ia', Z, t2mn)
        g1 = np.zeros((n, n), dtype=dtype)
        g1[:o, :o], g1[:o, o:], g1[o:, :o], g1[o:, o:] = g_ij, g_ia, l1.T, g_ab
        occ = np.zeros(n)
        occ[:o] = 1.0
        d1 = xm.T @ (g1 + np.diag(occ)) @ ym
    if not gradient:
        slabs.close()
        return Result(d1, None, g1, plan)

    with timer('dens.blocks'):
        Rg = Rl2 + R3                                   # the block iabj as [a, j, b, i]
        Y4 = l2pp @ t2pp.T + O4                         # the block ijkl: g2[ij,kl] = Y4[(kl),(ij)]
        # ---- ijak, antisymmetrized: [i, j, a, k]
        Dk = -einsum('ke,Pae->Pak', l1, t2mn) - YT[:, :, :o]
        for m in range(o):
            Dk[:, :, m] -= 0.5 * gemm(t2pp, C3[m])
        Dk = _unpack_occ(Dk, om)
        Zu = _unpack_occ(Z, om)                         # [k, m, e, j]
        T4 = einsum('imae,kmej->ijak', t2, Zu)
        Dk += T4 - T4.transpose(1, 0, 2, 3)
        del T4, Zu
        # ---- ijab, antisymmetrized: [(i<j), (a<b)]
        Dab = A1 + t2pp + Y4.T @ t2pp + Kg / 3.0
        W = einsum('Pae,eb->Pab', t2mn, g_ab)
        W += 0.5 * YT[:, :, o:].transpose(0, 2, 1)      # 1/2 Y12[P, b, a] as [P, a, b]
        Dab -= W[:, md, ne] - W[:, ne, md]
        W = einsum('in,jnQ->ijQ', g_ij, t2P) + 0.5 * XT[v:].reshape(o, o, nQ)
        Dab -= W[mm, nn] - W[nn, mm]
        W = einsum('fnai,jnbf->ijab', 0.5 * Rl2 + R3, t2)
        W = W - W.transpose(1, 0, 2, 3)
        W = W - W.transpose(0, 1, 3, 2)
        Dab += W[mm, nn][:, md, ne]
        del W
        # ---- aibc, antisymmetrized: [a, i, (b<c)], in the place of its l2 t3 part
        Dai = XT[:v].reshape(v, o, nQ)
        Dai *= -1.0
        gemm(l1.T, t2P.reshape(o, o * nQ), XT[:v], beta=1.0)       # -sum_m l1[m,a] t2[i,m,Q]
        Zai = np.ascontiguousarray(Z.transpose(1, 2, 0)).reshape(v * o, nP)        # [(a i), P]
        gemm(Zai, t2pp, XT[:v].reshape(v * o, nQ), alpha=-1.0, beta=1.0)
        del Zai
        # 1/2 (S - S^T), S[a,i,b,c] = sum_(m,e) t2[i,m,b,e] C3[(a e),c,m]
        pfull = np.where(vm.pidx >= 0, vm.pidx, vm.pidx.T)
        psign = np.sign(np.arange(v)[None, :] - np.arange(v)[:, None]).astype(np.float64)
        pfull = np.where(pfull >= 0, pfull, 0)
        t2r = np.ascontiguousarray(t2.transpose(0, 2, 1, 3)).reshape(o * v, o * v)  # [(i b), (m e)]
        for a in range(v if nQ else 0):
            Ca = C3[:, pfull[a], :] * psign[a][None, :, None]                       # [m, e, c]
            Sa = gemm(t2r, Ca.reshape(o * v, v)).reshape(o, v, v)
            Dai[a] += 0.5 * (Sa[:, md, ne] - Sa[:, ne, md])
        del t2r

    # ---- the gradient
    with timer('dens.gradient'):
        db = ccsdt_opt.dress_block
        foo, fov, fvo, fvv = fd[:o, :o], fd[:o, o:], fd[o:, :o], fd[o:, o:]
        ovoo, ovvo = db(eris, t1, 'ovoo'), db(eris, t1, 'ovvo')
        oooo, oovo = db(eris, t1, 'oooo'), db(eris, t1, 'oovo')
        G = np.array(fvo, dtype=dtype)
        G += fvo @ g_ij.T + fvv @ g_ia.T - g_ia.T @ foo - g_ab.T @ fvo
        G += einsum('jaki,jk->ai', ovoo, g_ij) + einsum('jabi,jb->ai', ovvo, g_ia)
        Voo = ccsdt_opt._unpack_pairs(vvoo, om, vm)             # [i, j, a, b] = <ab||ij>
        G += einsum('jiba,jb->ai', Voo, l1)
        # <ba||ci> with the density ab
        Wb = Vvo.reshape(nQ, v * o)
        Dv = np.ascontiguousarray(np.broadcast_to(np.eye(o, dtype=dtype)[None, :, None, :],
                                                  (v, o, v, o)) * g_ab[:, None, :, None])
        _bra_pairs(G, Wb, Dv, vm.pairs, -1.0)
        del Dv
        # ijkl with <am||jk>
        G -= 0.5 * einsum('majk,imjk->ai', ovoo, _unpack_occ(_unpack_occ(Y4.T, om).transpose(
            2, 0, 1), om).transpose(2, 3, 0, 1))
        # ijak with <am||ej> and <mn||ij>
        G -= einsum('maej,imej->ai', ovvo, Dk)
        G -= 0.5 * einsum('mnij,mnaj->ai', oooo, Dk)
        # iajk with <ae||jk>, abci with <ef||im>
        G += einsum('Pae,Pei->ai', Voo[mm, nn], Z)
        VooQ = np.ascontiguousarray(Voo[:, :, md, ne])          # [i, m, Q]
        for m in range(o):
            G += 0.5 * gemm(VooQ[:, m], C3[m]).T
        del VooQ, Voo
        # iabj with <ae||fm> and <em||ij>
        _bra_pairs(G, Wb, np.ascontiguousarray(Rg.transpose(0, 3, 2, 1)), vm.pairs, 1.0)
        G -= einsum('meij,ejam->ai', ovoo, Rg)
        # ijab with <mn||ie> and <am||ef>
        G += einsum('Pei,Pae->ai', oovo[mm, nn], _unpack_vir(Dab, vm))
        Dij = _unpack_occ(Dab, om)                              # [i, m, Q]
        G1Q, oovvQ = ccsdt_opt._bare_Q(eris, om, vm)            # <my||Q>, <mn||Q>
        oovvQ = _unpack_occ(np.asarray(oovvQ), om)              # [m, n, Q]
        for m in range(o):
            # the dressed <ma||Q> = <ma||Q> - sum_n t1[n,a] <mn||Q>
            Wm = np.array(G1Q[m], dtype=dtype)
            gemm(t1.T, oovvQ[m], Wm, alpha=-1.0, beta=1.0)
            G -= gemm(Wm, Dij[:, m].T)
        del oovvQ
        # abcd with <ef||gi>: its l2 t2 part, and its l3 t3 part from pass 2
        U = gemm(l2pp, Wb).reshape(nP, v, o)
        G += einsum('Pag,Pgi->ai', t2mn, U)
        G -= S13 / 3.0
        del U, Wb
        # aibc with <em||if> and with <ae||fg>
        for e in range(v):
            G -= einsum('mfi,maf->ai', ovvo[:, e], _unpack_vir(Dai[e], vm))
        for q0, q1, W in slabs:
            _bra_pairs(G, W, Dai, vm.pairs[q0:q1], -1.0)
        slabs.close()
    if not nfz:
        return Result(d1, G, g1, plan)

    # ---- frozen spinors: the blocks of g2 with the integrals of one frozen index
    with timer('dens.frozen'):
        bf = np.zeros((nfz, n), dtype=dtype)
        kf = np.zeros((nfz, n), dtype=dtype)
        D4 = _unpack_occ(_unpack_occ(Y4.T, om).transpose(2, 0, 1), om).transpose(2, 3, 0, 1)   # [i, m, j, k]
        Dij = _unpack_occ(Dab, om)                              # [i, m, Q]
        Dabv = _unpack_vir(Dab, vm)                             # [P, a, e]
        Zu = _unpack_occ(Z, om)                                 # [i, j, e, m] = Z[(ij), e, m]
        UK = gemm(l2pp, Vf.reshape(nQ, v * nfz)).reshape(nP, v, nfz)
        Dai2 = XT[:v]                                           # [a, (m Q)]
        del Vf, Sq
        for f0 in range(0, nfz, nfb):
            f1 = min(f0 + nfb, nfz)
            k = f1 - f0
            S0 = fints(f0, f1)
            with timer('dens.frozen.dress'):
                S, Wf = _dressed_frozen(S0, xm, ym, dtype)
            del S0
            SeQ = np.ascontiguousarray(S[:, o:, o:, o:][:, :, md, ne])      # [f, e, Q]
            SmQ = np.ascontiguousarray(S[:, :o, o:, o:][:, :, md, ne])      # [f, m, Q]
            SeP = S[:, o:, :o, :o][:, :, mm, nn]                            # [f, e, P]
            SmP = S[:, :o, :o, :o][:, :, mm, nn]
            Sego, Smeo = S[:, o:, o:, :o], S[:, :o, o:, :o]                 # [f, e, g, m], [f, m, e, j]
            # B[f, i] and B[f, a]: 1/2 sum S[f,q,r,s] g2[p,q,r,s]
            bi = 0.5 * einsum('fmjk,imjk->fi', S[:, :o, :o, :o], D4)
            bi += einsum('fmQ,imQ->fi', SmQ, Dij) + einsum('fmej,imej->fi', Smeo, Dk)
            bi += einsum('feP,Pei->fi', SeP, Z) + einsum('fegm,emgi->fi', Sego, Rg)
            for e in range(v):
                bi -= gemm(SeQ[:, e], Dai[e].T)
            ba = -einsum('fmP,Pam->fa', SmP, Z) - einsum('fmej,ajem->fa', Smeo, Rg)
            if nQ:
                ba += gemm(Dai2, SmQ.reshape(k, o * nQ).T).T
            US = SeP + gemm(SeQ.reshape(k * v, nQ), t2pp.T).reshape(k, v, nP)
            ba += einsum('Pae,feP->fa', l2mn, US) - SB.T[f0:f1] / 3.0
            for a in range(v if nQ else 0):
                Ca = C3[:, pfull[a], :] * psign[a][None, :, None]           # [m, e, g]
                ba[:, a] -= 0.5 * einsum('fegm,meg->f', Sego, Ca)
            # K[f, a] and K[f, i]: 1/2 sum W[p,q,f,s] g2[p,q,r,s], W[p,q,f,s] = Wf[f,s,p,q]
            WmQ = np.ascontiguousarray(Wf[:, :o, o:, o:][:, :, md, ne])     # [f, m, Q]
            WhQ = np.ascontiguousarray(Wf[:, o:, o:, o:][:, :, md, ne])     # [f, h, Q]
            Wjem, Wgem = Wf[:, :o, o:, :o], Wf[:, o:, o:, :o]               # [f, j, e, m], [f, g, e, m]
            ka = 0.5 * einsum('fjmn,mnaj->fa', Wf[:, :o, :o, :o], Dk)
            ka += einsum('feP,Pae->fa', Wf[:, o:, :o, :o][:, :, mm, nn], Dabv)
            ka -= einsum('Pah,Phf->fa', t2mn, UK[:, :, f0:f1]) - SK.T[f0:f1] / 3.0
            ka -= einsum('fjem,ejam->fa', Wjem, Rg)
            for m in range(o):
                ka -= 0.5 * gemm(WmQ[:, m], C3[m])
            for e in range(v):
                ka += einsum('fgm,mag->fa', Wgem[:, :, e, :], _unpack_vir(Dai[e], vm))
            ki = 0.5 * einsum('fjmn,mnij->fi', Wf[:, :o, :o, :o], D4)
            ki -= 0.5 * einsum('femn,mnei->fi', Wf[:, o:, :o, :o], Dk)
            ki += einsum('fjQ,ijQ->fi', WmQ, l2P) - einsum('fjem,ijem->fi', Wjem, Zu)
            ki += einsum('fgem,eigm->fi', Wgem, Rg)
            for i in range(o):
                ki[:, i] += 0.5 * einsum('fhQ,Qh->f', WhQ, C3[i])
            # the densities of the bare integrals: x on the bra index, y on the ket index
            bf[f0:f1, :o], bf[f0:f1, o:] = bi - ba @ t1.T, ba
            kf[f0:f1, :o], kf[f0:f1, o:] = ki, ka + ki @ t1
            del S, Wf, SeQ, SmQ, WmQ, WhQ
    return Result(d1, G, g1, plan, bf, kf)


# ------------------------------------------------------------ relaxed density

@dataclass
class Relaxed:
    relaxed: np.ndarray         # D[p,q] = <q+ p>, Hermitian, with the response of the spinors
    unrelaxed: np.ndarray       # the same without it: (d1^T + conj(d1)) / 2
    z: np.ndarray               # z[a,i] of the Z-vector equations
    mo_coeff: np.ndarray        # the spinors of the two densities: [frozen occupied, active
                                # occupied, active virtual, dropped virtual]
    mo_energy: np.ndarray
    nocc: int                   # occupied ones among them
    converged: bool = True
    result: Result = None
    gradient: np.ndarray = None  # with frozen spinors: the orbital gradient G[y,x] of zccsd.zvector, all spinors


def relaxed_rdm1(mf, eris, t1, t2, t3, l1, l2, l3, mo_coeff, frozen_occ=None, dropped_vir=None,
                 tol=1e-9, max_memory=None, verbose=None, plan=None):
    '''The relaxed one-particle density of CCSDT: kernel() for the density
    of t and lambda and the orbital gradient, the Z-vector equations of the
    SCF ``mf`` (zccsd.zvector.solve) and zvector.relaxed_rdm1.

    ``mo_coeff``: the spin orbitals or spinors of ``eris`` in the AO basis of
    ``mf``, [active occupied, active virtual], canonical; ``frozen_occ`` and
    ``dropped_vir``: those of the frozen occupied and of the dropped virtual
    ones (None: none).  Returns a Relaxed: the densities in the convention
    of zccsd.zvector and pyscf.cc.gccsd_rdm, D[p,q] = <q+ p>, Hermitian, in
    the basis of all spinors, Relaxed.mo_coeff - so that an expectation
    value is sum V[p,q] D[q,p] with V = C^+ V(AO) C, and the density in the
    AO basis C D C^+.'''
    from pyscf.lib import logger
    from zccsdt.zccsd import zvector
    log = logger.new_logger(mf, verbose)
    o = eris.nocc
    mo_coeff = np.asarray(mo_coeff)
    nf = 0 if frozen_occ is None else np.asarray(frozen_occ).shape[1]
    nd = 0 if dropped_vir is None else np.asarray(dropped_vir).shape[1]
    fov = abs(np.asarray(eris.fock)[:o, o:]).max() if o and mo_coeff.shape[1] > o else 0.0
    if fov > zvector.SCF_GRADIENT:
        log.warn('The Fock matrix of the SCF has occupied-virtual elements of %.1e: the relaxed '
                 'density carries an error of that size. Converge the SCF further (PySCF: '
                 'conv_tol_grad; its default is sqrt(conv_tol)).', fov)
    held_out = None
    if nf or nd:
        held_out = np.hstack([np.asarray(c) for c in (frozen_occ, dropped_vir) if c is not None and c.shape[1]])
    real = not np.iscomplexobj(np.zeros(0, dtype=np.result_type(eris.fock, eris.oovv)))
    res = kernel(eris, t1, t2, t3, l1, l2, l3, gradient=True, max_memory=max_memory, plan=plan,
                 frozen=None if held_out is None else (
                     nf + nd, frozen_integrals(mf, mo_coeff, held_out, real, max_memory)))
    dm1 = res.d1.T
    dm1 = 0.5 * (dm1 + dm1.conj().T)
    mo_e = np.asarray(eris.mo_energy, dtype=np.float64)
    if held_out is None:
        with timer('dens.zvector'):
            veff = zvector.scf_veff(mf, max_memory)
            conv, z = zvector.solve(mf, mo_coeff, mo_e, o, -0.5 * res.gnh, tol=tol, verbose=verbose,
                                    veff=veff)
        if not conv:
            log.warn('Z-vector equations not converged')
        return Relaxed(zvector.relaxed_rdm1(dm1, z, o), dm1, z, mo_coeff, mo_e, o, bool(conv), res)

    # ---- frozen occupied and dropped virtual spinors: [frozen, active occupied, active virtual, dropped]
    n = mo_coeff.shape[1]
    na = nf + n + nd
    C = np.hstack((held_out[:, :nf], mo_coeff, held_out[:, nf:]))
    frozen, act = np.arange(nf), nf + np.arange(o)
    vir, fvir = nf + o + np.arange(n - o), nf + n + np.arange(nd)
    both, held = np.concatenate((act, vir)), np.concatenate((frozen, fvir))
    fock = held_out.conj().T @ mf.get_fock() @ np.hstack((held_out, mo_coeff))
    eh = fock.diagonal().real.copy()
    fock[np.arange(nf + nd), np.arange(nf + nd)] = 0
    off = abs(fock).max()
    if off > 1e-6:
        log.warn('The frozen spinors are coupled to the others by the Fock matrix (%.1e): not canonical '
                 'spinors of this SCF; the relaxed density assumes that they are.', off)
    mo_e = np.concatenate((eh[:nf], mo_e, eh[nf:]))
    occ = (np.arange(na) < nf + o).astype(np.float64)
    full = np.diag(occ).astype(np.complex128)
    full[np.ix_(both, both)] = dm1
    with timer('dens.zvector'):
        veff = zvector.scf_veff(mf, max_memory)

        def fock_mo(D):
            '''C^+ (J - K)(D) C over all spinors, for a density D of the AO basis
            that need not be Hermitian (two builds with Hermitian densities).'''
            anti = abs(D - D.conj().T).max() > 1e-14
            vh = veff(0.5 * (D + D.conj().T))
            if anti:
                vh = vh + 1j * veff(-0.5j * (D - D.conj().T))
            return C.conj().T @ vh @ C

        def fock_of(M):
            F = fock_mo(mo_coeff @ M.T @ mo_coeff.conj().T)
            return F[np.ix_(held, both)], F[np.ix_(both, held)]

        B, K = frozen_terms(res, np.asarray(t1), fock_of)
        V = fock_mo(C @ (full - np.diag(occ)) @ C.conj().T)
        G = np.zeros((na, na), dtype=np.complex128)
        G[np.ix_(vir, act)] = 0.5 * res.gnh
        G[np.ix_(act, vir)] = -0.5 * res.gnh.conj().T
        Gf = 0.5 * (B + K.conj()) + (occ[both][None, :] - occ[held][:, None]) * V[np.ix_(held, both)]
        G[np.ix_(held, both)], G[np.ix_(both, held)] = Gf, -Gf.conj().T
        Gf = V[np.ix_(fvir, frozen)]
        G[np.ix_(fvir, frozen)], G[np.ix_(frozen, fvir)] = Gf, -Gf.conj().T
        relaxed, z = zvector._relax(mf, mf, C, mo_e, nf + o, G, full, (act, frozen, vir, fvir), False,
                                    zvector.DEGENERATE, tol, verbose, veff)
    return Relaxed(relaxed, full, z, C, mo_e, nf + o, True, res, G)


# ------------------------------------------------------------ frozen spinors

def frozen_integrals(mf, mo_coeff, frozen_coeff, real=False, max_memory=None):
    '''The function (f0, f1) -> <fu||vw>[f0:f1] that kernel(frozen=) takes:
    the antisymmetrized integrals of the spinors f = frozen_coeff[:, f0:f1]
    with the spinors u, v, w = mo_coeff of the SCF ``mf`` (its AO basis), by
    an integral transformation with f as the first index - (o + v)^3
    numbers for each f.  ``real``: the integrals are real (the spinors of
    zccsdt.eris.axial_spinors).'''
    from zccsdt import eris as E
    mol, kind = mf.mol, E._orbital_kind(mf)
    C, Cf = np.asarray(mo_coeff), np.asarray(frozen_coeff)

    def make(f0, f1):
        with timer('dens.frozen.ao2mo'):
            if kind == 'ghf' and (np.iscomplexobj(C) or np.iscomplexobj(Cf)):
                # (complex spin orbitals: from all AO integrals, small cases)
                nao = mol.nao_nr()
                eri = mol.intor('int2e', aosym='s1')
                X = 0
                for s1 in (slice(0, nao), slice(nao, None)):
                    half = np.einsum('mf,mnls,nq->fqls', Cf[s1, f0:f1].conj(), eri, C[s1], optimize=True)
                    for s2 in (slice(0, nao), slice(nao, None)):
                        X = X + np.einsum('fqls,lr,st->fqrt', half, C[s2].conj(), C[s2], optimize=True)
            else:
                X = E._chem_block(mol, kind, Cf[:, f0:f1], C, C, C, memory.ao2mo_memory(max_memory))
            # <fu||vw> = (fv|uw) - (fw|uv)
            S0 = X.transpose(0, 2, 1, 3) - X.transpose(0, 2, 3, 1)
            return E._real_part(S0) if real else S0

    return make


def frozen_terms(res, t1, fock_of):
    '''(B[f,x], K[f,x]) for the frozen spinors f and the active ones x of a
    Result with bf and kf:

        B[f,x] = 1/2 sum_qrs <fq||rs> gamma2[x,q,r,s],
        K[f,x] = 1/2 sum_pqs <pq||fs> gamma2[p,q,x,s],

    gamma2 the two-particle density without its one-particle terms, of the
    bare integrals and not Hermitian (d2 = gamma2 + the antisymmetrized
    products of gamma1 = d1 - 1(occupied) with 1(occupied) + the reference).
    The orbital gradient of zccsd.zvector between f and x is

        G[f,x] = (B[f,x] + conj(K[f,x])) / 2 + (n_x - n_f) V[f,x],

    V the Fock matrix (Coulomb minus exchange) of the Hermitian part of
    gamma1.  In the dressed basis gamma2 is the antisymmetrized g2 of the
    tables transformed back, which is bf and kf, plus the antisymmetrized
    product of gamma1 - tau / 2 with tau (tau: t1 in the occupied-virtual
    block), which meets the integrals as Fock matrices: fock_of(M) returns
    (F[f, x], F[x, f]) of F[a,b] = sum_uw <au||bw> M[u,w] for a matrix M
    over the active spinors, not Hermitian.'''
    o = t1.shape[0]
    n = res.d1.shape[0]
    tau = np.zeros((n, n), dtype=res.d1.dtype)
    tau[:o, o:] = t1
    occ = np.zeros(n)
    occ[:o] = 1.0
    A = res.d1 - np.diag(occ) - 0.5 * tau
    ft, fa = fock_of(tau), fock_of(A)
    B = res.bf + ft[0] @ A.T + fa[0] @ tau.T
    K = res.kf + ft[1].T @ A + fa[1].T @ tau
    return B, K


def all_blocks(eris, t1, t2, t3, l1, l2, l3, plan=None):
    '''Gnh[y,x] of zccsd.zvector for all pairs of spinors of ``eris`` - the
    Lagrangian with the bare integrals one-index transformed, a bra index y
    going to x and a ket index x to minus y - from all integrals as one
    full array (n^4: a check for small cases).  Every spinor is taken as a
    "frozen" one of kernel(frozen=), its integrals <yq||rs> with the same
    blocks of the densities; then

        Gnh[y,x] = B[y,x] - K[x,y] + (f gamma1^T - gamma1^T f)[y,x]
                 + (n_x - n_y) (f + V)[y,x].

    For converged t and lambda the blocks among occupied, among virtual
    spinors and [occupied, virtual] (the Lambda_1 equation) vanish, and
    [virtual, occupied] is Result.gnh.  Returns (Gnh, Result).'''
    from zccsdt.cc import ccsdtq_ref
    fock, g, _, o = ccsdtq_ref.integrals(eris)
    n = fock.shape[0]
    res = kernel(eris, t1, t2, t3, l1, l2, l3, plan=plan, frozen=(n, lambda f0, f1: g[f0:f1]))

    def fock_of(M):
        F = np.einsum('aubw,uw->ab', g, M)
        return F, F
    B, K = frozen_terms(res, np.asarray(t1), fock_of)
    occ = (np.arange(n) < o).astype(np.float64)
    g1 = res.d1 - np.diag(occ)
    V = np.einsum('pyqx,pq->yx', g, g1)
    return B - K.T + fock @ g1.T - g1.T @ fock + (V + fock) * (occ[None, :] - occ[:, None]), res
