'''The orbital gradient of the CCSD Lagrangian for the Z-vector equations, in
the layouts of zccsdt.zccsd.amps.

For converged t and lambda the gradient of zccsd.zvector is half of its
de-excitation part,

    G[a, i] = 1/2 Gnh[a, i],

Gnh[a, i] the Lagrangian L = E + sum l1 R1 + 1/4 sum l2 R2 with the
integrals one-index transformed by a+_i a_a (zvector has the argument and,
in gradient_terms, every term from full arrays: the reference of this
module).  With D_B the coefficient of the block B of integrals in L,

    Gnh[a, i] = sum over the blocks with an occupied bra index of
                    D_B[i ...] <a ...||...>
              - sum over the blocks with a virtual ket index of
                    D_B[... a ...] <...||... i ...>,

and the Fock matrix with its own density.  The densities are those of
zvector.gradient_terms; here they are evaluated as far as the contraction
needs them:

- four matrices on the occupied pairs and the one-particle densities (the
  products of the packed lambda2 with the packed t2, tau and <ij||ab>);
- m[(i a), (j b)] = sum_kc lambda2[i,k,a,c] t2[k,j,c,b], the product of the
  two ring layouts (o^3 v^3), kept like W(mbej) of the amplitude step (o^2 v^2
  elements, in memory or a mapped scratch array) and read by rows;
- D[(m>n), (e>f)], the density of <mn||ef> packed like t2: tau, one-particle
  terms on both sides, tau times a matrix on the pairs, and the product
  m^T (t2 - 2 t1 t1) (the second o^3 v^3 product), antisymmetrized block by
  block;
- arrays with three occupied indices and one virtual index.

Only the density of <ia||bc> meets <ab||cd>.  It is a sum of products, and
so is its contraction:

- where its virtual pair is that of tau - lambda1 tau and
  (lambda2 tau) t1 -, with the pp ladder of tau from the AO ladder of the
  amplitude step (one application, as in a CCSD iteration), contracted
  with lambda1 and with lambda2 t1;
- (lambda2 t2 over three indices) t1: a v x v density with <ab||cd> is a
  Fock matrix, built AO-direct (together with the Fock matrix of the
  one-particle densities: one call for two Coulomb and exchange matrices);
- m t1, the pair split between m and t1: sum_f <ab||ef> t1[j,f] is the
  integral <j~ e||ab> of the spinor j~ = sum_f t1[j,f] f in the place of an
  occupied one.  These integrals come from the integral transformation
  (zccsd.ao2mo.slabs with j~ as the first index) slab by slab and are
  contracted with the rows of m at once, like <je||ab> itself.

Cost: two o^3 v^3 products, one AO ladder, three passes over <ma||ef> (one
of them over the integrals of the j~), two Fock matrices: about a Lambda
iteration and the (ov|vv) part of the integral transformation.

Kramers pairs.  With t and lambda symmetric under time reversal, the two
o^3 v^3 products are done for the rows with an unbarred occupied index and
completed by the symmetry (kernels.kfill_ring), and the sums over the first
index of <ma||ef> - of the stored integrals and of the <j~ e||ab> - run over
the unbarred half and are completed by the time-reversed image.  The ladder
does half of the pairs (amps).

Frozen spinors are not known here: zvector.gradient_frozen runs this module
on all spinors, with zero amplitudes for the frozen ones.

Timers (zccsdt.timing): 'og.pairs', 'og.ring' (the two products), 'og.ladder',
'og.ovvv', 'og.dressed', 'og.fock', 'og.small'.
'''

import numpy as np
from pyscf.lib import logger

from zccsdt import blas, memory, scratch
from zccsdt.timing import timer
from zccsdt.zccsd import amps
from zccsdt.zccsd import kernels as K

GROUPS = ('fock', 'oooo', 'ooov', 'oovv', 'ovov', 'ovvv', 'vvvv')
UNPACK_BYTES = 64 << 20     # rows of <ma||ef> unpacked at a time, at most


def _ovvv_x(Om, X, out):
    '''out[i,a] += sum_cb <mc||ab> X[i,c,b] from the packed rows Om[c,(a>b)],
    a few of them unpacked at a time.'''
    o, v = X.shape[:2]
    step = max(1, min(o, UNPACK_BYTES // (16 * v * v)))
    Xm = X.reshape(o, v * v)
    for c0 in range(0, v, step):
        c1 = min(v, c0 + step)
        U = K.unpack_tril(Om[c0:c1], v, 'pab')                          # [c, b, a] = <mc||ba>
        blas.gemm(Xm[:, c0 * v:c1 * v], U.reshape((c1 - c0) * v, v), C=out, alpha=-1.0, beta=1.0)
        del U
    return out


class Coulomb:
    '''J - K of the bare Coulomb operator for one-particle densities in the
    (spin, AO) basis of the scalar AOs, complex and not necessarily
    Hermitian: PySCF's GHF contraction of the spin blocks with the real
    scalar integrals.  The integrals are kept in memory (8-fold symmetry:
    nao^4 bytes) if they take no more than a quarter of what ``max_memory``
    leaves; else every call computes them (AO direct).'''

    def __init__(self, mol, max_memory=None):
        self.mol = mol
        budget = memory.budget(max_memory)
        self.eri = None
        if budget is None or 4 * mol.nao_nr() ** 4 <= budget:
            with timer('og.eri'):
                self.eri = mol.intor('int2e', aosym='s8')

    def veff(self, dms):
        '''J - K for the density D[(s mu), (s' nu)], or for a stack of them.'''
        from pyscf.scf import ghf, hf

        def jkbuild(mol, dm, hermi, with_j, with_k, omega=None):
            if self.eri is not None:
                return hf.dot_eri_dm(self.eri, dm, hermi, with_j, with_k)
            return hf.get_jk(mol, dm, hermi, None, with_j, with_k, omega)

        vj, vk = ghf.get_jk(self.mol, np.asarray(dms), hermi=0, jkbuild=jkbuild)
        return vj - vk


def fock_matrices(coulomb, C, densities):
    '''[C^+ (J - K)(D) C for D in densities]: the Coulomb and exchange
    matrices of the bare Coulomb operator (``coulomb``: a Coulomb) for
    one-particle densities given in the basis of the spinors C[(spin, AO), p]
    as d[p, q], the coefficient of f[p, q] (D[nu, mu] = sum C[nu, q] d[p, q]
    conj(C[mu, p])).  The densities need not be Hermitian.'''
    veff = coulomb.veff(np.array([C @ d.T @ C.conj().T for d in densities]))
    return [C.conj().T @ veff[k] @ C for k in range(len(densities))]


def own_bytes(nocc, nvir):
    '''Bytes of the arrays of this module besides two packed ones and m: two
    matrices on the pairs, the arrays with three occupied indices that are
    alive together (lambda2 t1 and two more of its size, one of o^3 v
    elements), and what the terms on the rows of <ma||ef> take for one
    occupied index.'''
    o, v = int(nocc), int(nvir)
    npo, npv = K.npair(o), K.npair(v)
    kept = 16 * (2 * npo * npo + 3 * npo * o * v + o ** 3 * v + 2 * o ** 4)
    rows = 64 * o * v * v + 32 * o * npv
    return kept + rows + UNPACK_BYTES


def reserve_bytes(nocc, nvir):
    '''Bytes to set aside when the amplitude step is planned for this
    module: what own_bytes has more than the plan counts for arrays that are
    not made here - the intermediates of a step (amps.step_arrays) and one of
    its three packed T2 (here: tau and the density of <mn||ef> or the ladder;
    t2 itself is given, like lambda2, and is there when the plan is made).'''
    spare = amps.step_arrays(nocc, nvir) + 16 * K.npair(nocc) * K.npair(nvir)
    return max(0, own_bytes(nocc, nvir) - spare)


def working_set(plan, nocc, nvir):
    '''Bytes that gradient() takes at most under the Plan ``plan`` of the
    amplitude step, next to the packed t2 and lambda2 it is given: the
    working set of a step (m in the place of W) and reserve_bytes.'''
    return plan.work + reserve_bytes(nocc, nvir)


def slab_plan(mol, plan, nocc, nvir):
    '''The ao2mo.Plan of the integrals <j~ e||ab>: slabs (j~ q|rs) of the
    integral transformation within what the Plan ``plan`` of the step leaves
    next to m (if it is in memory) and the arrays of one j.  MemoryError if
    one spinor j~ does not fit: 16 v^3 bytes for its slab, twice what a row
    of the packed <ma||ef> takes.'''
    from zccsdt.zccsd import ao2mo
    o, v = int(nocc), int(nvir)
    left = None
    if plan.budget is not None:
        left = (plan.budget + reserve_bytes(o, v) - (0 if plan.w_mapped else 16 * (o * v) ** 2)
                - 64 * o * v * v)
    try:
        return ao2mo.plan_slabs(mol, o, v, left, sort_arrays=0)
    except MemoryError as err:
        raise MemoryError('%s (the integrals <j~ e||ab> of the orbital gradient%s)'
                          % (err, '' if plan.w_mapped else
                             ', next to the product of lambda2 and t2 kept in memory')) from None


def kernel(mycc, eris=None, t1=None, t2=None, l1=None, l2=None, verbose=None, parts=None,
           coulomb=None):
    '''Gnh[a, i] for the amplitudes and lambda of ``mycc`` (default: its t1,
    t2, l1, l2; t2 and lambda2 as [i,j,a,b] or packed [(i>j), (a>b)]), within
    its max_memory: the plan of the amplitude step for what is left next to
    the packed t2 and lambda2, less reserve_bytes.'''
    log = logger.new_logger(mycc, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    if eris is None:
        eris = mycc.eris if mycc.eris is not None else mycc.ao2mo(mycc.mo_coeff)
    if getattr(eris, 'residual', None) is None:
        eris.residual = amps.Residual(eris, plan=False)
    res = eris.residual
    o, v = res.nocc, res.nvir
    t1 = np.ascontiguousarray(mycc.t1 if t1 is None else t1, dtype=np.complex128)
    l1 = np.ascontiguousarray(mycc.l1 if l1 is None else l1, dtype=np.complex128)

    def packed(x):
        x = np.ascontiguousarray(x, dtype=np.complex128)
        return x if x.ndim == 2 else K.pack_t2(x)
    # as lam.kernel: the arrays of the CCSD object are packed and released
    # while this runs, and unpacked again at the end
    own_t, own_l = t2 is None, l2 is None
    t2p, l2p = packed(mycc.t2 if own_t else t2), packed(mycc.l2 if own_l else l2)
    t2 = l2 = None
    if own_t:
        mycc.t2 = None
    if own_l:
        mycc.l2 = None
    try:
        try:
            res.replan(mycc.max_memory, 0, False, False, reserve=reserve_bytes(o, v))
        except MemoryError as err:
            raise MemoryError('%s (the orbital gradient: the plan of the amplitude step, with '
                              '%s set aside for its own arrays)'
                              % (err, amps._size(reserve_bytes(o, v)))) from None
        try:
            slab_plan(eris.mol, res.plan, o, v)
        except MemoryError:
            if res.plan.w_mapped:
                raise
            res.plan.w_mapped = True        # m in a scratch file leaves room for the slabs
            slab_plan(eris.mol, res.plan, o, v)
        log.info('Orbital gradient: %s', res.describe())
        return gradient(mycc, eris, t1, t2p, l1, l2p, parts, coulomb)
    finally:
        res.close()
        if own_t:
            mycc.t2 = K.unpack_t2(t2p, o, v)
        if own_l:
            mycc.l2 = K.unpack_t2(l2p, o, v)
        log.timer('Orbital gradient', *cput0)


def gradient(mycc, eris, t1, t2p, l1, l2p, parts=None, coulomb=None):
    '''Gnh[a, i] (see the module) for the amplitudes t1[i,a], t2 packed
    [(i>j), (a>b)] and lambda in the same layouts; the integrals and the
    memory plan are those of eris.residual.  ``parts``: a dict that gets the
    terms by block of integrals, as zvector.gradient_terms returns them.
    ``coulomb``: the Coulomb for the two Fock matrices (default: a new one).'''
    res = eris.residual
    o, v = res.nocc, res.nvir
    n1, npo, npv = o * v, K.npair(o), K.npair(v)
    io, jo = K.tril(o)
    plan = res.plan
    prow = np.array([r for r, _ in res.pair_rows])
    psgn = np.array([g for _, g in res.pair_rows])
    t1 = np.ascontiguousarray(t1, dtype=np.complex128)
    l1 = np.ascontiguousarray(l1, dtype=np.complex128)
    t1T = np.ascontiguousarray(t1.T)
    X = {k: np.zeros((v, o), dtype=np.complex128) for k in GROUPS}
    pblocks = plan.pblocks
    slabs = slab_plan(eris.mol, plan, o, v)             # before anything is done
    # Kramers pairs: the rows of the ring products with an unbarred occupied
    # index, the sums over the first index of <ma||ef> over its unbarred half;
    # Xh collects the halves that their time-reversed image completes
    kr = bool(res.kramers and res.symmetric(t1, t2p) and res.symmetric(l1, l2p))
    ho = o // 2 if kr else o
    Xh = {k: np.zeros((v, o), dtype=np.complex128) for k in GROUPS}
    size = max(j1 - j0 for j0, j1 in plan.jblocks)
    if kr:      # blocks of unbarred columns (each with its partners) and of unbarred rows
        cblocks, rblocks = amps._blocks(ho, max(1, size // 2)), amps._blocks(ho, size)
    else:
        cblocks, rblocks = plan.jblocks, plan.jblocks

    def ring(j0, j1, t1t1=False):
        '''The columns (j b) of t2 in the ring layout for the block of j (with
        Kramers pairs: followed by those of the partners of its j); with
        ``t1t1``: of t2[n,j,f,b] - 2 t1[n,b] t1[j,f].'''
        parts = [(j0, j1)] + ([(j0 + ho, j1 + ho)] if kr else [])
        nc = (j1 - j0) * v
        Tc = np.empty((n1, len(parts) * nc), dtype=np.complex128)
        for k, (a0, a1) in enumerate(parts):
            Tk = K.ring_cols(t2p, o, v, a0, a1).reshape(o, v, a1 - a0, v)
            if t1t1:
                Tk -= 2.0 * t1[:, None, None, :] * t1[a0:a1].T[None, :, :, None]
            Tc[:, k * nc:(k + 1) * nc] = Tk.reshape(n1, nc)
            del Tk
        return Tc

    with timer('og.pairs'):
        # M[(ij),(kl)] = sum_{c>d} lambda2[(ij),(cd)] t2[(kl),(cd)], with tau: Xl
        Xl = blas.gemm(l2p, t2p.T)
        mij = np.einsum('ik,jk,ijk->ij', psgn, psgn, Xl[prow[:, None, :], prow[None, :, :]])
        mba = np.zeros((v, v), dtype=np.complex128)
        lt = np.empty((npo, o, v), dtype=np.complex128)     # sum_d lambda2[p,c,d] t1[k,d]
        Pm = np.zeros((v, v), dtype=np.complex128)          # sum conj(<p||Ab>) lambda2[p,a,b]
        for p0, p1 in pblocks:
            n = p1 - p0
            l2u = K.unpack_tril(l2p[p0:p1], v, 'apb').reshape(v * n, v)         # [(c p), a]
            t2u = K.unpack_tril(t2p[p0:p1], v, 'apb').reshape(v * n, v)
            blas.gemm(t2u.T, l2u, C=mba, beta=1.0)
            del t2u
            lt[p0:p1] = blas.gemm(l2u, t1T).reshape(v, n, o).transpose(1, 2, 0)
            vu = K.unpack_tril(res.oovv_p[p0:p1], v, 'apb').reshape(v, n * v)   # [A, (p b)]
            blas.gemm(vu.conj(), l2u.reshape(v, n * v).T, C=Pm, beta=1.0)
            del l2u, vu
        Xl += blas.gemm(lt.reshape(npo * o, v), t1.T).reshape(npo, o, o)[:, jo, io]
        lvv = blas.gemm(t1T, l1)                        # [f, b] = sum_j t1[j,f] lambda1[j,b]
        loo = blas.gemm(l1, t1.T)                       # [j, n] = sum_b lambda1[j,b] t1[n,b]
        tmp1vv, tmp1oo = mba + lvv, mij + loo
        # the one-particle densities: the coefficients of f[p, q]
        Dvv = np.ascontiguousarray(tmp1vv.T)
        Doo = -np.ascontiguousarray(tmp1oo.T)
        u = K.t2p_matvec(t2p, l1, np.zeros((o, v), dtype=np.complex128))
        Dov = t1 + u - blas.gemm(t1, Dvv) - blas.gemm(mij.T, t1)
        w = u - 0.5 * blas.gemm(t1, mba.T) - 0.5 * blas.gemm(mij.T, t1)
        # M2[(ij),(kl)] = sum_{b>c} lambda2[(ij),(bc)] conj(<kl||bc>)
        M2 = np.empty((npo, npo), dtype=np.complex128)
        for p0, p1 in pblocks:
            blas.gemm(l2p, res.oovv_p[p0:p1].conj().T, M2[:, p0:p1])
        Zoo = np.einsum('ij,jk,ijk->ki', psgn, psgn, M2[prow[:, :, None], prow[None, :, :]])
        del M2
        taup = K.outer_pp(t1, t1, 0.5, t2p.copy())

        # <mn||ij>
        for n in range(o):
            blas.gemm(res.ooov_en[n].conj(), np.ascontiguousarray(Xl[:, prow[n]] * psgn[n]),
                      C=X['oooo'], beta=1.0)

    # ---- m = L T in the ring layout, [(i b), (m d)], by blocks of its columns
    m = res._w_array()
    with timer('og.ring'):
        for m0, m1 in cblocks:
            Tc = ring(m0, m1)
            nc = (m1 - m0) * v
            Q = np.empty((n1, Tc.shape[1]), dtype=np.complex128)
            for i0, i1 in rblocks:
                Li = K.ring_cols(l2p, o, v, i0, i1)
                blas.gemm(Li.T, Tc, C=Q[i0 * v:i1 * v])
                del Li
            if kr:
                K.kfill_ring(Q, o, v, 0, ho, m1 - m0)
                m[:, (m0 + ho) * v:(m1 + ho) * v] = Q[:, nc:]
            m[:, m0 * v:m1 * v] = Q[:, :nc]
            del Q, Tc

    with timer('og.small'):
        # ---- <mn||ie> and <mb||ij>: Dh[(m>n), i, e], antisymmetric in m, n
        Dh = (Doo[io][:, :, None] * t1[jo][:, None, :] - Doo[jo][:, :, None] * t1[io][:, None, :])
        for p0, p1 in pblocks:
            n = p1 - p0
            t2u = K.unpack_tril(t2p[p0:p1], v, 'apb').reshape(v, n * v)         # [a, (p e)]
            Dh[p0:p1] -= blas.gemm(l1, t2u).reshape(o, n, v).transpose(1, 0, 2)
            del t2u
        for i in range(o):
            # 1/2 sum_j Y[mn,ij] t1[j,e], Y[mn,ij] = 2 Xl[(ij),(mn)]
            A = np.ascontiguousarray((Xl[prow[i]] * psgn[i][:, None]).T)        # [(mn), j]
            Dh[:, i, :] += blas.gemm(A, t1)
            # sum_b (m[ib,me] t1[n,b] - m[ib,ne] t1[m,b])
            mt = blas.gemm(t1, np.asarray(m[i * v:(i + 1) * v])).reshape(o, o, v)   # [n, m, e]
            Dh[:, i, :] += mt[jo, io] - mt[io, jo]
        for i in range(o):
            Er = np.asarray(res.E[i * v:(i + 1) * v])                   # [A, (n e)]
            DI = psgn[:, :, None] * Dh[prow, i]                         # [I, n, e]
            blas.gemm(Er.conj(), np.ascontiguousarray(DI.reshape(o, n1).T), C=X['ooov'], beta=1.0)
            Oi = np.ascontiguousarray(res.oooo_p[:, prow[i]] * psgn[i])     # [(mn), I]
            blas.gemm(np.ascontiguousarray(Dh[:, i, :].T), Oi, C=X['ooov'], alpha=-1.0, beta=1.0)
        X['ooov'] -= blas.gemm(Pm, t1.T)
        del Dh

    # ---- <mn||ef>: Dp[(m>n), (e>f)]
    with timer('og.pairs'):
        # tau; the one-particle densities on both sides of tau~ = t2 + the half of
        # the t1 t1 part of tau (that half joins the outer product with t1); tau
        # with the matrix Xl on the pairs
        Dp = taup.copy()
        Gm = np.zeros((npo, npo), dtype=np.complex128)
        for rows, cols, f0, f1, sign in res._g:
            Gm[rows, cols] += sign * tmp1oo[f0, f1]
        blas.gemm(Gm.T, t2p, C=Dp, beta=1.0)
        del Gm
        for p0, p1 in pblocks:
            n = p1 - p0
            tu = K.unpack_tril(t2p[p0:p1], v, 'apb').reshape(v * n, v)          # [(a p), c]
            K.asym_pack2(blas.gemm(tu, Dvv).reshape(v, n, v), -1.0, Dp[p0:p1], 'apb')
            del tu
        K.outer_pp(w + 0.5 * blas.gemm(Doo, t1) - 0.5 * blas.gemm(t1, Dvv), t1, 1.0, Dp)
        blas.gemm(Xl.T, taup, C=Dp, beta=1.0)
    with timer('og.ring'):
        # 1/2 P(mn) P(ef) sum_jb m[jb,me] (t2[jn,bf] - 2 t1[j,f] t1[n,b])
        for n0, n1_ in cblocks:
            Tc = ring(n0, n1_, t1t1=True)
            R = np.empty((n1, Tc.shape[1]), dtype=np.complex128)
            blas.gemm(m[:, :ho * v].T, Tc, C=R[:ho * v])     # the rows (m e) with unbarred m
            if kr:
                K.kfill_ring(R, o, v, 0, ho, n1_ - n0)
                K.asym_pack_cols(R, o, v, n0 + ho, n1_ + ho, 0.5, Dp, offset=(n1_ - n0) * v)
            K.asym_pack_cols(R, o, v, n0, n1_, 0.5, Dp)
            del R, Tc

    with timer('og.small'):
        for p0, p1 in pblocks:
            n = p1 - p0
            Du = K.unpack_tril(Dp[p0:p1], v, 'apb').reshape(v * n, v)           # [(f p), A]
            X['oovv'] += blas.gemm(res.ooov_en[:, :, p0:p1].reshape(o, v * n), Du).T
            del Du

    # ---- the pass over <ma||ef>
    Jv = np.empty((ho, v, npo), dtype=np.complex128)    # sum conj(<If||bc>) lambda2[(ij),(bc)]
    c44 = np.zeros((o, v), dtype=np.complex128)         # conj of the <mb||je> term, [I, A]
    fstep = max(1, min(v, UNPACK_BYTES // (16 * npv)))
    with timer('og.ovvv'):
        for j in range(ho):
            Om = res.ovvv_p[j]                                          # [e, (a>b)]
            # <mn||ef> density with <nA||ef>
            Dn = np.ascontiguousarray((psgn[:, j][:, None] * Dp[prow[:, j]]).T)     # [(ef), I]
            blas.gemm(Om, Dn, C=Xh['oovv'], alpha=-1.0, beta=1.0)
            del Dn
            # <ab||ef> density: its lambda2 with conj(<jf||bc>), a few rows f at a time
            for f0 in range(0, v, fstep):
                f1 = min(v, f0 + fstep)
                blas.gemm(np.conj(Om[f0:f1]), l2p.T, Jv[j, f0:f1])
            # <mb||je> density D[m, b, e] of this j with conj(<je||Ab>) and conj(<jI||mb>)
            Aj = psgn[j][:, None, None] * lt[prow[j]]                   # [i, m, b]
            Dj = blas.gemm(np.ascontiguousarray(Aj.reshape(o, n1).T), t1).reshape(o, v, v)
            Dj -= t1[:, None, :] * l1[j][None, :, None]
            Dj -= np.asarray(m[j * v:(j + 1) * v]).reshape(v, o, v).transpose(1, 0, 2)
            _ovvv_x(Om, np.ascontiguousarray(np.conj(Dj.transpose(0, 2, 1))), c44)
            Xh['ovov'] -= blas.gemm(res.ooov[j].conj().reshape(o, n1), Dj.reshape(n1, v)).T
            scratch.unmap(Om)
            del Dj, Aj
        Xh['ovov'] += c44.conj().T
        del Dp
        for p0, p1 in pblocks:
            n = p1 - p0
            tu = K.unpack_tril(taup[p0:p1], v, 'apb').reshape(v * n, v)         # [(f p), A]
            Xh['vvvv'][:, :ho] += blas.gemm(Jv[:, :, p0:p1].reshape(ho, v * n), tu).T
            del tu
        del Jv

    # ---- <ma||ef> density with <mb||If> and <mb||eI> (E[(m f), (I b)] = -<mb||If>)
    N = 2.0 * mba.T                                     # sum_ijc lambda2[ij,ca] t2[ij,ce]
    with timer('og.small'):
        Q3 = np.empty((npo, v, o), dtype=np.complex128)                 # [p, f, I]
        Et2 = np.empty((o, o, v, o), dtype=np.complex128)               # [j, I, b, m]
        EN = np.empty((o, o), dtype=np.complex128)
        lt2 = lt.reshape(npo, n1)
        for i in range(o):
            Ec = np.conj(np.asarray(res.E[i * v:(i + 1) * v])).reshape(v, o, v)   # [b, m, f]
            # - (El[m,f,I,i'] - El[i',f,I,m]) with El = sum_b E[(m f),(I b)] lambda1[i',b]
            El = blas.gemm(l1, Ec.reshape(v, n1)).reshape(o, o, v)      # [i', m, f]
            Q3[:, :, i] = El[jo, io] - El[io, jo]
            # sum_mb lt[p,m,b] E[(m f),(I b)]
            Q3[:, :, i] += blas.gemm(lt2, np.ascontiguousarray(Ec.transpose(1, 0, 2))
                                     .reshape(n1, v))
            Et2[:, i] = blas.gemm(Ec.reshape(n1, v), t1.T).reshape(v, o, o).transpose(2, 0, 1)
            EN[:, i] = np.einsum('bmf,bf->m', Ec, N)
            del Ec, El
        Et = np.conj(blas.gemm(np.asarray(res.E), np.conj(t1).reshape(n1, 1))).reshape(o, v)
        X['ovvv'] += -0.5 * blas.gemm(Et, N).T + 0.5 * blas.gemm(t1.T, EN)
        for p0, p1 in pblocks:
            n = p1 - p0
            tu = K.unpack_tril(taup[p0:p1], v, 'apb').reshape(v * n, v)         # [(f p), A]
            Qb = np.ascontiguousarray(Q3[p0:p1].transpose(1, 0, 2)).reshape(v * n, o)
            X['ovvv'] -= blas.gemm(tu.T, Qb)
            del tu, Qb
        del Q3
        # the m t1 part: with Et2 and the rows of m, and its trace with E
        EM = np.empty((o, o), dtype=np.complex128)
        Eo = np.asarray(res.E).reshape(o, v * n1)
        for j in range(o):
            mj = np.asarray(m[j * v:(j + 1) * v])                       # [b, (m A)]
            X['ovvv'] += blas.gemm(Et2[j].reshape(o, n1), mj.reshape(n1, v)).T
            EM[:, j] = np.conj(blas.gemm(Eo, np.conj(mj).reshape(v * n1, 1))).ravel()
        X['ovvv'] -= blas.gemm(t1.T, EM.T)
        X['ovvv'] += blas.gemm(t1.T, Zoo.T)
        del Et2

    # ---- <ia||bc> density with <ab||cd>
    with timer('og.ladder'):
        Rl = np.zeros((npo, npv), dtype=np.complex128)
        saved, res.kramers = res.kramers, kr        # half of the pairs only for symmetric t
        try:
            res.ladder(taup, Rl)
        finally:
            res.kramers = saved
        X['ovvv'] += K.t2p_matvec(Rl, l1, np.zeros((o, v), dtype=np.complex128)).T
        Pl = np.zeros((v, v), dtype=np.complex128)
        for p0, p1 in pblocks:
            n = p1 - p0
            Ru = K.unpack_tril(Rl[p0:p1], v, 'apb').reshape(v, n * v)           # [A, (p b)]
            lu = K.unpack_tril(l2p[p0:p1], v, 'pab').reshape(n * v, v)          # [(p b), c]
            blas.gemm(Ru, lu, C=Pl, beta=1.0)
            del Ru, lu
        X['ovvv'] += blas.gemm(Pl, t1.T)
        del Rl, taup, lt, lt2, Xl
    with timer('og.dressed'):
        # - sum_jbe conj(<j~ e||Ab>) m[(j b), (I e)], j~ = sum_f t1[j,f] f, with
        # <j~ e||Ab> = (j~ A|eb) - (j~ b|eA) from the slabs g[q, r, s] = (j~ q|rs) of
        # the integral transformation as they come
        from zccsdt.zccsd import ao2mo
        nao = eris.nao_sph
        Cv = eris.mo_ao_spin[:, o:]
        Cd = np.hstack((Cv @ t1[:ho].T, Cv))
        c5 = np.zeros((o, v), dtype=np.complex128)                      # the conjugate, [I, A]
        for j0, j1, g in ao2mo.slabs(eris.mol, Cd[:nao], Cd[nao:], (0, ho), (ho, ho + v), slabs):
            for j in range(j0, j1):
                gj = g[j - j0]
                mj = np.conj(np.asarray(m[j * v:(j + 1) * v])).reshape(v, o, v)  # [b, I, e]
                # sum_eb (g[A,e,b] - g[b,e,A]) conj(m[(j b), (I e)])
                blas.gemm(np.ascontiguousarray(mj.transpose(1, 2, 0)).reshape(o, v * v),
                          gj.reshape(v, v * v).T, C=c5, alpha=-1.0, beta=1.0)
                blas.gemm(np.ascontiguousarray(mj.transpose(1, 0, 2)).reshape(o, v * v),
                          gj.reshape(v * v, v), C=c5, beta=1.0)
                del mj, gj
        del g
        Xh['ovvv'] += c5.conj().T
    res.close()
    del m

    with timer('og.fock'):
        C = eris.mo_ao_spin
        n = o + v
        Df = np.zeros((n, n), dtype=np.complex128)
        Df[:o, :o], Df[:o, o:], Df[o:, :o], Df[o:, o:] = Doo, Dov, l1.T, Dvv
        DN = np.zeros((n, n), dtype=np.complex128)
        DN[o:, o:] = N
        if coulomb is None:
            coulomb = Coulomb(eris.mol, mycc.max_memory)
        Vf, VN = fock_matrices(coulomb, C, (Df, DN))
        fvo = res.fov.conj().T
        X['fock'] += (blas.gemm(fvo, Doo.T) + blas.gemm(res.fvv, Dov.T) - blas.gemm(Dov.T, res.foo)
                      - blas.gemm(Dvv.T, fvo) + Vf[o:, :o])
        X['ovvv'] += 0.5 * blas.gemm(VN[o:, o:], t1.T)
    if kr:
        from zccsdt.zccsd.kramers import image
        for k in GROUPS:
            Xh[k] += image(Xh[k])
    for k in GROUPS:
        X[k] += Xh[k]
    if parts is not None:
        parts.update(X)
    return sum(X.values())
