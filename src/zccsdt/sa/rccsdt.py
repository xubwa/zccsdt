'''Closed-shell spin-adapted CCSDT with block-stored T3 (spatial orbitals).

Equations: zccsdt.sa.derive (checked against the spin-orbital code); the
reference implementation is zccsdt.sa.rccsdt_ref.  T3 is stored as one v^3
cube per sorted occupied triple (zccsdt.sa.t3blocks).  The T3 residual is

  omega = P[ D + E ] + sum_positions ring + sum_pair_positions (pp + hh ladder),

  D_ijk^abc = sum_e Wv_bcek t_ij^ae,   E_ijk^abc = -sum_m Wo_maji t_mk^bc,
  ring X_{i;jk}^{a;bc} = sum_md J_madi t~_mjk^dbc - K_madi t_mjk^dbc
                         - K_mbdi t_mjk^adc - K_mcdi t_mjk^abd,
  t~_mjk^dbc = 2 t_mjk^dbc - t_mjk^bdc - t_mjk^cbd,
  pp ladder (pair 12) sum_de W_ab^de t_ijk^dec,  hh ladder sum_mn W_mn^ij t_mnk^abc,

with P the six-term pair symmetrizer.  Wv, Wo, J, K are the J (resp. K)
spatial components of the intermediates of zccsdt.sa.derive.  In the block
engine (residuals) F_vv rides in the attached ring term and F_oo in the hh
operator; the first version (residuals_terms) folds both into the ladder
operators.  Design notes: docs/spin_adapted.md.
'''

import numpy as np

from zccsdt.blas import gemm


def pp_operator(Wj, Fv):
    '''Symmetric / antisymmetric (d,e) parts of the pp ladder operator
    W_ab^de = Wj[a,b,d,e] + 1/2 (Fv_ad d_be + Fv_be d_ad), packed over a<=b,
    d<=e (symmetric) and a<b, d<e (antisymmetric).'''
    v = Wj.shape[0]
    W = np.array(Wj, copy=True)
    for b in range(v):
        W[:, b, :, b] += 0.5 * Fv
        W[b, :, b, :] += 0.5 * Fv
    s0, s1 = np.triu_indices(v)
    a0, a1 = np.triu_indices(v, 1)
    Ws = 0.5 * (W + W.transpose(0, 1, 3, 2))
    Wa = 0.5 * (W - W.transpose(0, 1, 3, 2))
    Ws = np.ascontiguousarray(Ws[s0, s1][:, s0, s1])
    Wa = np.ascontiguousarray(Wa[a0, a1][:, a0, a1])
    return Ws, Wa


# ladder plan per block kind: (packed view, both parts?, placements of the result)
# kind 0 (distinct): three full views; kind 1 (s0 = s1): the (0,1) view is
# symmetric in (d,e) and the (1,2) result is the (0,2) result with a <-> b;
# kind 2 (s1 = s2): likewise with (1,2) symmetric and (0,2) from (0,1).
_PP_PLAN = {0: (('01', True, ('01',)), ('02', True, ('02',)), ('12', True, ('12',))),
            1: (('01', False, ('01',)), ('02', True, ('02', '12'))),
            2: (('12', False, ('12',)), ('01', True, ('01', '02')))}


def ladder_pp(blocks, t3b, Ws, Wa, r3b, batch=8):
    '''r3b += sum over the three pair positions of every block of
    sum_de W_xy^de t^{..de..} (block-local): the (d,e) symmetric and
    antisymmetric parts as half-size GEMMs, using the extra symmetry of the
    blocks with a repeated occupied index.'''
    from zccsdt.cc import t3pack
    from zccsdt.timing import timer
    lib = t3pack._lib
    nb, v = t3b.shape[0], t3b.shape[1]
    v2, v3 = v * v, v * v * v
    view = {'01': (v2, v, 1), '02': (v2, 1, v), '12': (v, 1, v2)}
    ps, pa = v * (v + 1) // 2, v * (v - 1) // 2
    I64, P = t3pack._I64, t3pack._p
    arr = lambda x: np.array(x, dtype=np.int64).reshape(-1, 4)   # noqa: E731
    # work arrays for the largest batch (3 full views per block)
    mcol = 3 * min(batch, nb) * v
    buf = (lambda name, n: blocks.buffer(name, (n,))) if hasattr(blocks, 'buffer') \
        else (lambda name, n: np.empty(n))
    Tsb, Tab = buf('pp.Ts', ps * mcol), buf('pp.Ta', pa * mcol)
    Sb, Ab = buf('pp.S', ps * mcol), buf('pp.A', pa * mcol)
    for b0 in range(0, nb, batch):
        full, sym, fplace, splace = [], [], [], []
        for b in range(b0, min(b0 + batch, nb)):
            for vw, both, places in _PP_PLAN[int(blocks.kind[b])]:
                lst, plc = (full, fplace) if both else (sym, splace)
                c = len(lst)
                lst.append((b * v3,) + view[vw])
                for pl in places:
                    plc.append((b * v3,) + view[pl] + (c,))
        nf, ns = len(full), len(sym)
        ncol = (nf + ns) * v
        Ts = Tsb[:ps * ncol].reshape(ps, ncol)
        Ta = Tab[:pa * nf * v].reshape(pa, nf * v)
        F, S_ = arr(full), arr(sym)
        _t = timer('pp.pack'); _t.__enter__()
        if nf:
            lib.zc_sa_pair_pack_d(P(t3b), I64(nf), P(np.ascontiguousarray(F[:, 0])),
                                  P(np.ascontiguousarray(F[:, 1])), P(np.ascontiguousarray(F[:, 2])),
                                  P(np.ascontiguousarray(F[:, 3])), I64(v), I64(v),
                                  P(Ts), I64(ncol), P(Ta), I64(nf * v))
        if ns:
            lib.zc_sa_pair_pack_d(P(t3b), I64(ns), P(np.ascontiguousarray(S_[:, 0])),
                                  P(np.ascontiguousarray(S_[:, 1])), P(np.ascontiguousarray(S_[:, 2])),
                                  P(np.ascontiguousarray(S_[:, 3])), I64(v), I64(v),
                                  t3pack.ctypes.c_void_p(Ts.ctypes.data + 8 * nf * v), I64(ncol),
                                  None, I64(0))
        _t.__exit__(None, None, None)
        with timer('pp.gemm'):
            S = gemm(Ws, Ts, Sb[:ps * ncol].reshape(ps, ncol))
            A = gemm(Wa, Ta, Ab[:pa * nf * v].reshape(pa, nf * v)) if nf else None
        _t = timer('pp.unpack'); _t.__enter__()
        for plc, Sp, Ap, lda, coff in ((fplace, S, A, nf * v, 0), (splace, S, None, 0, nf)):
            if not plc:
                continue
            Q = np.array(plc, dtype=np.int64)
            cols = np.ascontiguousarray(Q[:, 4] + coff)
            lib.zc_sa_pair_unpack_add_d(P(r3b), I64(len(Q)), P(np.ascontiguousarray(Q[:, 0])),
                                        P(np.ascontiguousarray(Q[:, 1])), P(np.ascontiguousarray(Q[:, 2])),
                                        P(np.ascontiguousarray(Q[:, 3])), I64(v), I64(v),
                                        P(Sp), I64(ncol), None if Ap is None else P(Ap), I64(lda),
                                        P(cols))
        _t.__exit__(None, None, None)


def ladder_hh(blocks, t3b, Wh, r3b):
    '''r3b += sum over pair positions of sum_mn W_mn^ij t_mnk^abc, with
    Wh[m,n,i,j] the hh ladder operator (F_oo folded in).'''
    from zccsdt.cc.t3pack import axpby
    no, v = blocks.nocc, t3b.shape[1]
    W2 = Wh.reshape(no * no, no * no)
    T = np.empty((no, no, v, v, v), dtype=t3b.dtype)
    for k in range(no):
        for m in range(no):
            for n in range(no):
                axpby(T[m, n], blocks.cube(t3b, m, n, k), 1.0, 0.0)
        H = gemm(np.ascontiguousarray(W2.T), T.reshape(no * no, -1)).reshape(no, no, v, v, v)
        for i in range(no):
            for j in range(i, no):
                X = H[i, j].transpose(2, 0, 1)          # X_{k;ij}^{c;ab}
                for B, axes in blocks.uses_attached(k, i, j):
                    axpby(r3b[B], X.transpose(axes), 1.0)


def ring(blocks, t3b, WrJ, WrK, r3b):
    '''r3b += attached ring terms X_{i;pq}^{a;bc} (see module docstring);
    WrJ / WrK[m, a, d, i].'''
    no, v = blocks.nocc, t3b.shape[1]
    J2 = np.ascontiguousarray(WrJ.transpose(1, 3, 0, 2)).reshape(v * no, no * v)   # [(a,i),(m,d)]
    K2 = np.ascontiguousarray(WrK.transpose(1, 3, 0, 2)).reshape(v * no, no * v)
    JK = np.hstack((J2, -K2))
    for p in range(no):
        for q in range(p, no):
            T = np.empty((no, v, v, v), dtype=t3b.dtype)
            for m in range(no):
                T[m] = blocks.cube(t3b, m, p, q)                 # t_mpq^{d b c}
            T1 = T.reshape(no * v, v * v)
            T2g = np.ascontiguousarray(T.transpose(0, 2, 1, 3)).reshape(no * v, v * v)  # t^{x d y}
            T3g = np.ascontiguousarray(T.transpose(0, 3, 1, 2)).reshape(no * v, v * v)  # t^{x y d}
            Tt = 2 * T1 - T2g - T3g.reshape(no * v, v, v).transpose(0, 2, 1).reshape(no * v, v * v)
            X1 = gemm(JK, np.vstack((Tt, T1)))                    # [(a,i),(b,c)]
            X23 = gemm(-K2, np.hstack((T2g, T3g)))                # [(b,i),(a,c)] | [(c,i),(a,b)]
            X = X1.reshape(v, no, v, v).transpose(1, 0, 2, 3) \
                + X23[:, :v * v].reshape(v, no, v, v).transpose(1, 2, 0, 3) \
                + X23[:, v * v:].reshape(v, no, v, v).transpose(1, 2, 3, 0)
            for i in range(no):
                for B, axes in blocks.uses_attached(i, p, q):
                    r3b[B] += X[i].transpose(axes)


def drive(blocks, t2, WvJ, WoJ, r3b):
    '''r3b += P[D + E] (module docstring); WvJ[a,b,e,i], WoJ[m,a,j,i].  For
    each ordered (i, j) one buffer G[a, (k, b, c)] = D - E from two GEMMs.'''
    from zccsdt.cc.t3pack import axpby
    no, v = blocks.nocc, t2.shape[2]
    Wv2 = np.ascontiguousarray(WvJ.transpose(2, 3, 0, 1)).reshape(v, no * v * v)   # [e,(k,b,c)]
    t2f = t2.reshape(no, -1)                                                    # [m,(k,b,c)]
    for i in range(no):
        for j in range(no):
            G = gemm(np.ascontiguousarray(t2[i, j]), Wv2)
            gemm(np.ascontiguousarray(WoJ[:, :, j, i].T), t2f, G, alpha=-1.0, beta=1.0)
            G = G.reshape(v, no, v, v)
            for k in range(no):
                for B, axes in blocks.uses_ordered((i, j, k)):
                    axpby(r3b[B], G[:, k].transpose(axes), 1.0)


# ------------------------------------------------------------------ driver

def hh_operator(Wj, Fo):
    '''hh ladder operator W_mn^ij = Wj[m,n,i,j] - 1/2 (Fo_mi d_nj + Fo_nj d_mi).'''
    no = Wj.shape[0]
    W = np.array(Wj, copy=True)
    for j in range(no):
        W[:, j, :, j] -= 0.5 * Fo
        W[j, :, j, :] -= 0.5 * Fo
    return W


def residuals_terms(ints, blocks, t1, t2, t3b):
    '''(R1, R2, R3 blocks) for the spin-adapted CCSDT equations.  First
    version (kept for testing): T1 dressing on full V, intermediates and the CCSD part from the
    derived term lists, T3 -> T1 / T2 / W on a dense T3; the T3 residual
    on blocks.'''
    from zccsdt.sa import rccsdt_ref
    from zccsdt.sa.spinfree import evaluate
    from zccsdt.sa.t3blocks import project_null
    no, nv = ints.nocc, ints.nvir
    Vd, Fd = rccsdt_ref.dress(ints.V, ints.fock, t1, no)
    imds, res, slots = rccsdt_ref._terms(True)
    dims = {'o': no, 'v': nv}
    base = {'V': Vd, 'F': Fd, 'T2': t2, 'T3': blocks.to_dense(t3b)}
    arr = dict(base)
    for key in ('Fv', 'Fo', 'Wr.J', 'Wr.K', 'Wpp.J', 'Whh.J', 'Wv.J', 'Wo.J'):
        arr[key] = evaluate(imds[key], base, slots[key.split('.')[0]], dims)
    r1 = evaluate(res['R1'], arr, 'ia', dims)
    r2 = evaluate(res['R2'], arr, 'ijab', dims)
    r3b = np.zeros_like(t3b)
    Ws, Wa = pp_operator(arr['Wpp.J'], arr['Fv'])
    ladder_pp(blocks, t3b, Ws, Wa, r3b)
    ladder_hh(blocks, t3b, hh_operator(arr['Whh.J'], arr['Fo']), r3b)
    ring(blocks, t3b, arr['Wr.J'], arr['Wr.K'], r3b)
    drive(blocks, t2, arr['Wv.J'], arr['Wo.J'], r3b)
    return r1, r2, project_null(r3b)


def kernel(ints, conv_tol=1e-10, conv_tol_normt=1e-7, max_cycle=200, diis_space=8,
           verbose=False):
    '''Closed-shell spin-adapted CCSDT (Jacobi steps with DIIS).  numpy's own
    BLAS (OpenBLAS) is kept single-threaded while it runs: its spinning
    threads would otherwise compete with the MKL threads of zccsdt.blas.'''
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return _kernel(ints, conv_tol, conv_tol_normt, max_cycle, diis_space, verbose)
    with threadpool_limits(limits={'openblas': 1}):
        return _kernel(ints, conv_tol, conv_tol_normt, max_cycle, diis_space, verbose)


def _kernel(ints, conv_tol, conv_tol_normt, max_cycle, diis_space, verbose):
    from zccsdt.cc.diis import DIIS
    from zccsdt.sa import rccsdt_ref
    from zccsdt.sa.t3blocks import Blocks, jacobi
    no, nv = ints.nocc, ints.nvir
    blocks = Blocks(no)
    e = ints.mo_energy
    eo, ev = e[:no], e[no:]
    d1 = eo[:, None] - ev[None, :]
    d2 = d1[:, None, :, None] + d1[None, :, None, :]
    eo3 = eo[blocks.tri].sum(axis=1)
    t1 = np.zeros((no, nv))
    t2 = ints.block('oovv') / d2
    t3b = np.zeros((blocks.nblk, nv, nv, nv))
    r3b = np.empty_like(t3b)
    adiis = DIIS(space=diis_space)
    # DIIS metric of the full t_ijk^abc: a block stands for 6 (distinct
    # occupied indices) or 3 (one repeated index) ordered triples; the T3
    # error vector carries the weight, the stored T3 vector does not
    wsq = np.sqrt(np.where(blocks.kind == 0, 6.0, 3.0))
    e_old = rccsdt_ref.energy(t1, t2, ints)
    conv = False
    import time
    from zccsdt.timing import timer
    timings = []
    for cycle in range(max_cycle):
        t0 = time.perf_counter()
        r1, r2, r3b = residuals(ints, blocks, t1, t2, t3b, r3b)
        with timer('sa.diis'):
            xs, es = adiis.recycle()
            x3 = xs[2].reshape(t3b.shape) if xs else np.empty_like(t3b)
            e3 = es[2].reshape(t3b.shape) if es else np.empty_like(t3b)
            n3 = jacobi(eo3, ev, wsq, t3b, r3b, x3, e3)
            normt = float(np.linalg.norm(r1) + np.linalg.norm(r2) + np.sqrt(n3))
            q1, q2 = r1 / d1, r2 / d2
            # the extrapolated T3 goes into the old t3b array (not a DIIS vector)
            out = adiis.update([t1 + q1, t2 + q2, x3], errors=[q1, q2, e3],
                               out=[None, None, t3b])
        t1 = out[0].reshape(no, nv)
        t2 = out[1].reshape(no, no, nv, nv)
        t3b = out[2].reshape(t3b.shape)
        e_new = rccsdt_ref.energy(t1, t2, ints)
        timings.append(time.perf_counter() - t0)
        if verbose:
            print('cycle %3d  E_corr = %.15f  dE = %.3e  |r| = %.3e  %.2fs'
                  % (cycle + 1, e_new, e_new - e_old, normt, timings[-1]))
        done = abs(e_new - e_old) < conv_tol and normt < conv_tol_normt
        e_old = e_new
        if done:
            conv = True
            break
    res = rccsdt_ref.Result(e_corr=e_old, e_tot=ints.e_hf + e_old, converged=conv,
                            ncycle=cycle + 1, t1=t1, t2=t2, t3=t3b)
    res.timings = timings
    return res


# ------------------------------------------------------- explicit (v2) pieces

def permute(A, perm, out=None):
    '''A.transpose(perm) as a C-contiguous array (``out`` if given); large
    arrays are copied with the OpenMP strided kernel (numpy copies on one
    thread).'''
    from zccsdt.cc.t3pack import axpby
    V = A.transpose(perm)
    if out is None and V.flags.c_contiguous:
        return V
    if V.ndim <= 4 and V.size >= 1 << 15 and V.dtype in (np.float64, np.complex128):
        return axpby(np.empty(V.shape, dtype=V.dtype) if out is None else out, V, 1.0, 0.0)
    if out is None:
        return np.ascontiguousarray(V)
    out[...] = V
    return out


def _matrix(X, sx, rows, cols, dims):
    '''X as a 2-D (rows, cols) operand for gemm: a view when the axes are
    already in (rows, cols) or (cols, rows) order, else a parallel copy.'''
    m = int(np.prod([dims[c] for c in rows], dtype=np.int64))
    n = int(np.prod([dims[c] for c in cols], dtype=np.int64))
    V = X.transpose([sx.index(c) for c in rows + cols])
    if V.flags.c_contiguous:
        return V.reshape(m, n)
    W = X.transpose([sx.index(c) for c in cols + rows])
    if W.flags.c_contiguous:
        return W.reshape(n, m).T
    return permute(X, [sx.index(c) for c in rows + cols]).reshape(m, n)


def contract(spec, A, B, out=None, alpha=1.0, beta=0.0):
    '''Two-operand einsum without traces or batch indices as one MKL GEMM.
    Operands are passed as (transposed) views where possible, else permuted
    in parallel; the GEMM orientation follows the output order when it can.
    With ``out``: out = alpha * result + beta * out (in place).'''
    from zccsdt.cc.t3pack import axpby
    lhs, res = spec.replace(' ', '').split('->')
    sa, sb = lhs.split(',')
    k = [c for c in sa if c in sb]
    fa = [c for c in sa if c not in k]
    fb = [c for c in sb if c not in k]
    dims = dict(zip(sa, A.shape))
    dims.update(zip(sb, B.shape))
    res = list(res)
    if res == fb + fa and res != fa + fb:            # C^T = B^T A^T lands in place
        A, B, sa, sb, fa, fb = B, A, sb, sa, fb, fa
    Am = _matrix(A, sa, fa, k, dims)
    Bm = _matrix(B, sb, k, fb, dims)
    shape = [dims[c] for c in fa + fb]
    order = [(fa + fb).index(c) for c in res]
    if order == list(range(len(order))):
        if out is None:
            return gemm(Am, Bm, alpha=alpha).reshape(shape)
        if out.flags.c_contiguous:
            gemm(Am, Bm, out.reshape(Am.shape[0], Bm.shape[1]), alpha=alpha, beta=beta)
            return out
    C = gemm(Am, Bm, alpha=alpha if out is None else 1.0).reshape(shape)
    if out is None:
        return permute(C, order)
    if out.ndim <= 4:
        return axpby(out, C.transpose(order), alpha, beta)
    out *= beta
    out += alpha * C.transpose(order)
    return out


def intermediates(D, Fd, t2, nocc, P=None, t1=None):
    '''T2 parts of the intermediates (spatial J / K components of
    zccsdt.sa.derive) from the T1-dressed blocks D (see DRESSED, plus the
    bare 'oovv') and Fock.  The pp ladder operator is built separately
    (pp_operator_packed).

    Wv (J) is returned as 'WvT'[i, e, b, a] = Wv[a, b, e, i] (the layout of
    the drive GEMM), from P[a,b,e,i]: the dressed <ab|ei> without its
    t2 (G t1) part (default D['vvvo'] with t1 = None), which is overwritten.
    Its t2 terms are one GEMM against
      Q[m,n,e,i] = <mn|ei> - sum_f <mn|ef> t1[i,f] - d_ni Fov[m,e],
    and its three ovvv terms two GEMMs sharing the operand <ma|ef>.'''
    o, v = slice(0, nocc), slice(nocc, None)
    G = D['oovv']
    L = 2 * G - G.transpose(0, 1, 3, 2)                  # L_mnef = 2<mn|ef> - <mn|fe>
    out = {}
    out['Fv'] = Fd[v, v] - contract('mndf,mnaf->ad', L, t2)
    out['Fo'] = Fd[o, o] + contract('mnef,inef->mi', L, t2)
    out['Fov'] = Fd[o, v]
    out['Whh'] = D['oooo'] + contract('mnef,ijef->mnij', G, t2)
    out['WrJ'] = D['ovvo'] + contract('mnde,inae->madi', L, t2) - contract('mnde,inea->madi', G, t2)
    out['WrK'] = D['ovov'].transpose(0, 1, 3, 2) - contract('mned,inea->madi', G, t2)
    no, nv = t2.shape[0], t2.shape[2]
    ov, vv = no * nv, nv * nv
    if P is None:
        P = np.array(D['vvvo'])
    Q = np.array(D['oovo'])                              # [m, n, e, i]
    if t1 is not None:
        Q -= contract('mnef,if->mnei', G, t1)
    for i in range(no):
        Q[:, i, :, i] -= Fd[o, v]
    gemm(t2.reshape(no * no, vv).T, Q.reshape(no * no, ov), P.reshape(vv, ov), beta=1.0)
    ovvv = D['ovvv']                                     # <ma|fe>
    A12 = np.empty((nv, nv, 2, no, nv))                  # [(a,e), (<ma|fe> | <ma|ef>), (m,f)]
    permute(ovvv, (1, 3, 0, 2), out=A12[:, :, 0])
    permute(ovvv, (1, 2, 0, 3), out=A12[:, :, 1])
    Bst = np.empty((2, no, nv, no, nv))                  # [(t2b | -t2)[m,i,f,b] as (m,f), (i,b)]
    permute(2 * t2 - t2.transpose(0, 1, 3, 2), (0, 2, 1, 3), out=Bst[0])
    permute(-t2, (0, 2, 1, 3), out=Bst[1])
    C45 = gemm(A12.reshape(vv, 2 * ov), Bst.reshape(2 * ov, ov)).reshape(nv, nv, no, nv)
    C6 = gemm(A12[:, :, 1].reshape(vv, ov), permute(-t2, (0, 3, 1, 2)).reshape(ov, ov))
    C6 = C6.reshape(nv, nv, no, nv)                      # [b, e, i, a]
    WvT = permute(P, (3, 2, 1, 0), out=np.empty((no, nv, nv, nv)))
    from zccsdt.cc.t3pack import axpby
    axpby(WvT, C45.transpose(2, 1, 3, 0), 1.0, 1.0)
    axpby(WvT, C6.transpose(2, 1, 0, 3), 1.0, 1.0)
    out['WvT'] = WvT
    oovo, ooov = D['oovo'], D['ooov']
    out['WoJ'] = D['vooo'].transpose(1, 0, 3, 2) \
        + contract('nmej,niea->maji', oovo, 2 * t2) - contract('nmje,niea->maji', ooov, t2) \
        - contract('nmej,niae->maji', oovo, t2) - contract('nmie,njae->maji', ooov, t2) \
        + contract('amfe,ijfe->maji', D['vovv'], t2)
    out['L'] = L
    out['G'] = G
    return out


def ladder_pairs(Ws, Wa, x):
    '''sum_cd W_ab^cd x[cd, n] for x[c, d, n] with the packed (c,d)
    symmetric / antisymmetric parts of W; returns [a, b, n].'''
    v = x.shape[0]
    s0, s1 = np.triu_indices(v)
    a0, a1 = np.triu_indices(v, 1)
    xs = x[s0, s1] + x[s1, s0]
    xs[s0 == s1] *= 0.5
    S = gemm(Ws, np.ascontiguousarray(xs))
    A = gemm(Wa, np.ascontiguousarray(x[a0, a1] - x[a1, a0]))
    out = np.empty((v, v, x.shape[2]), dtype=S.dtype)
    out[s0, s1] = S
    out[s1, s0] = S
    out[a0, a1] += A
    out[a1, a0] -= A
    return out


def ccsd_residual(D, Fd, t2, imds, nocc, Ws, Wa, t1=None):
    '''T1-dressed closed-shell CCSD residuals (no T3); the ladder through the
    packed pp operator W (which includes the T2 term, removed here).  With
    ``t1``, D['vvoo'] lacks the sum_fg t_i^f t_j^g <ab|fg> term, which is
    added here through the ladder with tau = t2 + t1 t1.'''
    o, v = slice(0, nocc), slice(nocc, None)
    Fv, Fo, Whh, G = imds['Fv'], imds['Fo'], imds['Whh'], imds['G']
    t2b = 2 * t2 - t2.transpose(0, 1, 3, 2)
    r1 = Fd[v, o].T + contract('kc,ikac->ia', Fd[o, v], t2b) \
        - contract('klci,klca->ia', 2 * D['oovo'] - D['ooov'].transpose(0, 1, 3, 2), t2) \
        + contract('kadc,kidc->ia', D['ovvv'], t2b)
    W2J = 0.5 * (D['ovvo'] + imds['WrJ'])
    W2K = 0.5 * (D['ovov'].transpose(0, 1, 3, 2) + imds['WrK'])
    no, nv = t2.shape[0], t2.shape[2]
    tau = t2 if t1 is None else t2 + t1[:, None, :, None] * t1[None, :, None, :]
    lad = ladder_pairs(Ws, Wa, np.ascontiguousarray(tau.transpose(2, 3, 0, 1)).reshape(nv, nv, -1))
    lad = lad.reshape(nv, nv, no, no).transpose(2, 3, 0, 1)
    lad = lad - contract('mnab,mnij->ijab', t2, contract('mncd,ijcd->mnij', G, tau))
    q = 0.5 * D['vvoo'].transpose(2, 3, 0, 1) + 0.5 * lad \
        + 0.5 * contract('klij,klab->ijab', Whh, t2) \
        + contract('bc,ijac->ijab', Fv, t2) - contract('kj,ikab->ijab', Fo, t2) \
        + contract('kbcj,ikac->ijab', W2J, t2b) - contract('kbcj,ikac->ijab', W2K, t2) \
        - contract('kacj,ikcb->ijab', W2K, t2)
    r2 = q + q.transpose(1, 0, 3, 2)
    return r1, r2


def pair_gather(blocks, t3b, p, q, W):
    '''W[k][x, m, y, z] for the cubes s(x,y,z) = t_mpq^{xyz} of the pair (p, q):
    k = 0: Y = 2 s(x,y,z) - s(y,x,z) - s(z,y,x),  1: T = s(x,y,z),
    2: T2 = s(y,x,z),  3: T3 = s(y,z,x),  4: Z = 2 s(x,z,y) - s(y,z,x) - s(z,x,y)
    (Z is Y with y <-> z).'''
    from zccsdt.cc import t3pack
    from zccsdt.cc.t3pack import axpby
    no, v = blocks.nocc, t3b.shape[1]
    lib = t3pack._lib
    if t3pack.USE_C and hasattr(lib, 'zc_sa_pair_gather_s_d') and t3b.dtype in (np.float64, np.complex128):
        bl = blocks.blk[:, p, q]
        off = np.where(bl >= 0, bl * v ** 3, -1).astype(np.int64)
        cst = np.array((v * v, v, 1), dtype=np.int64)
        st = np.ascontiguousarray(cst[blocks.axes[:, p, q]].ravel())
        P = t3pack._p
        ptr = [t3pack.ctypes.c_void_p(W[k].ctypes.data) for k in range(5)]
        getattr(lib, 'zc_sa_pair_gather_s_' + t3pack._sfx(t3b.dtype))(
            P(t3b), t3pack._I64(no), t3pack._I64(v), P(off), P(st), *ptr,
            t3pack._I64(no * v * v), t3pack._I64(v * v))
        return W
    for m in range(no):
        c = blocks.cube(t3b, m, p, q)
        axpby(W[1][:, m], c, 1.0, 0.0)
        axpby(W[2][:, m], c.transpose(1, 0, 2), 1.0, 0.0)
        axpby(W[3][:, m], c.transpose(2, 0, 1), 1.0, 0.0)
        axpby(W[0][:, m], 2 * c - c.transpose(1, 0, 2) - c.transpose(2, 1, 0), 1.0, 0.0)
        axpby(W[4][:, m], 2 * c.transpose(0, 2, 1) - c.transpose(2, 0, 1) - c.transpose(1, 2, 0),
              1.0, 0.0)
    return W


def t3_pass(blocks, t3b, imds, D, nocc, r3b):
    '''One loop over sorted occupied pairs (p <= q) with the gathered cubes
    W[k][x, m, y, z] of t_mpq (pair_gather; the contraction index (x, m) is
    leading so that each contraction over the gathered cubes is one GEMM over
    all m): adds the ring X_{i;pq}^{a;bc} (with F_vv) to r3b and returns the
    T3 parts (r1, r2, Eab, Eba, dWo) of R1, R2, Wv (J) and Wo (J), with
    dWv[a,b,e,i] = Eab[e,i,a,b] + Eba[e,i,b,a].  For the
    reversed pair (q, p) the roles are T2 <-> T3 and Y <-> Z.'''
    from zccsdt.sa.t3blocks import scatter, zero
    from zccsdt.timing import timer
    no, v = blocks.nocc, t3b.shape[1]
    ov, vv = no * v, v * v
    G, L, Fov = imds['G'], imds['L'], imds['Fov']
    oovo, ooov = D['oovo'], D['ooov']                              # <mn|ej>, <mn|je>
    # ring operators [(a,i), (d,m)]: J2 = WrJ[m,a,d,i], K2 = WrK[m,a,d,i]
    J2 = imds['WrJ'].transpose(1, 3, 2, 0).reshape(ov, ov)
    K2 = imds['WrK'].transpose(1, 3, 2, 0).reshape(ov, ov)
    JK = np.ascontiguousarray(np.hstack((J2, -K2)))
    mK2 = np.ascontiguousarray(-K2)
    # F_vv of the attached position: X_{i;pq}^{a;bc} += sum_d Fv_ad t_ipq^{dbc}
    JK5 = JK.reshape(v, no, 2, v, no)
    for i in range(no):
        JK5[:, i, 1, :, i] += imds['Fv']
    # (B) operator [(a | m), (x, m', y)]: <m'a|xy> + 1/2 d_ay Fov[m',x] (the
    # F_ov term rides in, since Z = Y with y <-> z), and G[m',m,x,y] for Wo
    VG = np.empty((v + no, v, no, v))
    VG[:v] = D['ovvv'].transpose(1, 2, 0, 3)
    for a in range(v):
        VG[a, :, :, a] += 0.5 * Fov.T
    VG[v:] = G.transpose(1, 2, 0, 3)
    VG = VG.reshape(v + no, -1)
    r1x = np.zeros((v, no))                         # r1 as [a, i]
    r2 = np.zeros((no, no, v, v))
    # T3 -> W / R2 accumulators [row, i, y, z], rows: r2b (j; [j,i,b,a]),
    # Eab (e; dWv[a,b,e,i]), r2a (j; [j,i,a,b]), Eba (e; dWv[b,a,e,i])
    ACC = zero(blocks.buffer('t3p.acc', (2 * no + 2 * v, no, v, v)))
    dWo = np.zeros((no, v, no, no))
    W = blocks.buffer('t3p.W', (5, v, no, v, v))
    Wm = W.reshape(5, v, no * vv)
    Xb = blocks.buffer('t3p.X', (3, ov, vv))
    for p in range(no):
        for q in range(p, no):
            with timer('t3p.gather'):
                pair_gather(blocks, t3b, p, q, W)
            # ---- ring: X_{i;pq}^{a;bc} = X1[(a,i),(b,c)] + X2[(b,i),(a,c)] + X3[(c,i),(a,b)]
            with timer('t3p.ring_gemm'):
                X1 = gemm(JK, W[0:2].reshape(2 * ov, vv), Xb[0]).reshape(v, no, v, v)
                X2 = gemm(mK2, W[2].reshape(ov, vv), Xb[1]).reshape(v, no, v, v)
                X3 = gemm(mK2, W[3].reshape(ov, vv), Xb[2]).reshape(v, no, v, v)
            with timer('t3p.scatter'):
                items = []
                for i in range(no):
                    parts = (X1[:, i], X2[:, i].transpose(1, 0, 2), X3[:, i].transpose(1, 2, 0))
                    for B, axes in blocks.uses_attached(i, p, q):
                        items.append((B, (0, 0, 0), [x.transpose(axes) for x in parts]))
                scatter(r3b, items)
            with timer('t3p.pass1'):
                Gpq, Gqp = G[p, q], G[q, p]
                Lp = lambda g: 2 * g - g.T                         # noqa: E731  [f, e]
                A = lambda a, b: (-2 * oovo[a, b] + ooov[a, b].T).T   # noqa: E731  [j, e]
                if p == q:
                    # (A) contractions over x for i = m: rows Eab, r2a and r2a, Eba
                    op1 = np.vstack((Gpq, oovo[p, p].T))
                    gemm(op1, Wm[1], ACC[no:2 * no + v].reshape(no + v, -1), beta=1.0)
                    op2 = np.vstack((A(p, p), -Lp(Gpq).T))
                    gemm(op2, Wm[2], ACC[no + v:].reshape(no + v, -1), beta=1.0)
                    lv = (L[p, p].ravel(), None, L[p, p].T.ravel())
                    cf = 0.5
                else:
                    op1 = np.vstack((oovo[q, p].T, Gpq, oovo[p, q].T, Gqp))
                    gemm(op1, Wm[1], ACC.reshape(2 * (no + v), -1), beta=1.0)
                    op2 = np.empty((no + v, 2, v))
                    op2[:no, 0], op2[:no, 1] = A(p, q), A(q, p)
                    op2[no:, 0], op2[no:, 1] = -Lp(Gpq).T, -Lp(Gqp).T
                    gemm(op2.reshape(no + v, 2 * v), Wm[2:4].reshape(2 * v, -1),
                         ACC[no + v:].reshape(no + v, -1), beta=1.0)
                    lv = ((L[p, q] + L[q, p].T).ravel(), L[q, p].T.ravel(), L[p, q].T.ravel())
                    cf = 1.0
                # R1: sum_{yz} over T (+), T2 and T3 (-)
                for k, sgn in ((1, 1.0), (2, -1.0), (3, -1.0)):
                    if lv[k - 1] is not None:
                        gemm(W[k].reshape(ov, vv), np.ascontiguousarray(lv[k - 1]).reshape(vv, 1),
                             r1x.reshape(ov, 1), alpha=sgn, beta=1.0)
                # (B): R2 (ovvv, F_ov) and Wo: Y and Z against [Vo; G]
                PY = gemm(VG, W[0].reshape(-1, v))
                PZ = gemm(VG, W[4].reshape(-1, v))
                r2[p, q] += cf * (PY[:v] + PZ[:v].T)
                dWo[:, :, q, p] += PZ[v:]
                if p != q:
                    dWo[:, :, p, q] += PY[v:]
    r2b, Eab = ACC[:no], ACC[no:no + v]
    r2a, Eba = ACC[no + v:2 * no + v], ACC[2 * no + v:]
    r2 = r2 + r2a.transpose(1, 0, 2, 3) + r2b.transpose(1, 0, 3, 2)
    r2 = r2 + r2.transpose(1, 0, 3, 2)
    return np.ascontiguousarray(r1x.T), r2, Eab, Eba, dWo


def drive_hh(blocks, t3b, t2, Wv, Wo, Wh, r3b, max_bytes=1 << 29):
    '''r3b += the attached terms of single index k and pair (i <= j):

      hh ladder  sum_mn Wh[m,n,i,j] t_mnk^{abc}  (= X_{k;ij}^{c;ab}, F_oo in Wh),
      drive      P[D + E] = sum_positions X, X_{k;ij}^{x;yz} = K_{k;ij}^{x;yz} + K_{k;ji}^{x;zy},
                 K_{k;pq}^{x;yz} = sum_e Wv[y,x,e,k] t_qp^{ze} - sum_m Wo[m,z,p,q] t_mk^{yx}

    (K_{k;pq}^{x;yz} = G_{qpk}^{zyx} for the ordered term G = D + E of the
    module docstring).  One loop over k: K from one GEMM over all (p, q),
    the hh terms from one GEMM over the gathered t_mnk, then a single tiled
    scatter of their sum.  The virtual of k (x) is processed in slabs so that
    K and the gathered cubes stay within ``max_bytes`` each.  ``Wv``: the
    array Wv[a,b,e,i], or a function Wv(k, out) writing out[e,x,y] = Wv[y,x,e,k].'''
    from zccsdt.sa.t3blocks import gather_boxes, scatter
    from zccsdt.timing import timer
    no, v = blocks.nocc, t3b.shape[1]
    vv = v * v
    pi, pj = np.triu_indices(no)
    W2T = np.ascontiguousarray(Wh.reshape(no * no, no * no)[:, pi * no + pj].T)   # [(i<=j), (m,n)]
    B = np.empty((v + no, no, no, v))
    B[:v] = t2.transpose(3, 1, 0, 2)                    # [e, p, q, z] = t_qp^{ze}
    B[v:] = -Wo.transpose(0, 2, 3, 1)                   # [m, p, q, z] = -Wo[m,z,p,q]
    B = B.reshape(v + no, -1)
    nx = int(max(1, min(v, max_bytes // (8 * no * no * vv))))
    Tbuf = blocks.buffer('dh.T', (no * no * vv * nx,))
    Kbuf = blocks.buffer('dh.K', (nx * vv * no * no,))
    Hbuf = blocks.buffer('dh.H', (len(pi) * vv * nx,))
    from zccsdt.cc.t3pack import axpby
    A = np.empty((v + no, v, v))
    for k in range(no):
        if callable(Wv):
            Wv(k, A[:v])                                  # [e, x, y] = Wv[y,x,e,k]
        else:
            axpby(A[:v], Wv[:, :, :, k].transpose(2, 1, 0), 1.0, 0.0)
        A[v:] = t2[:, k].transpose(0, 2, 1)               # [m, x, y] = t_mk^{yx}
        Am = A.reshape(v + no, vv).T
        views = [blocks.view(m, n, k, v) for m in range(no) for n in range(no)]
        off0 = np.array([o for o, _ in views], dtype=np.int64)
        st = np.array([s_ for _, s_ in views], dtype=np.int64)
        for x0 in range(0, v, nx):
            n = min(nx, v - x0)
            with timer('dh.K_gemm'):
                K = gemm(Am[x0 * v:(x0 + n) * v], B,
                         Kbuf[:n * vv * no * no].reshape(n * v, -1)).reshape(n, v, no, no, v)
            T = Tbuf[:no * no * vv * n].reshape(no * no, v, v, n)        # t_mnk^{ab, x0+c}
            with timer('dh.gather'):
                gather_boxes(t3b, np.where(off0 >= 0, off0 + x0 * st[:, 2], -1), st, T)
            with timer('dh.H_gemm'):
                H = gemm(W2T, T.reshape(no * no, -1),
                         Hbuf[:len(pi) * vv * n].reshape(len(pi), -1)).reshape(len(pi), v, v, n)
            with timer('dh.scatter'):
                items = []
                for c, (i, j) in enumerate(zip(pi, pj)):
                    srcs = (H[c].transpose(2, 0, 1), K[:, :, i, j], K[:, :, j, i].transpose(0, 2, 1))
                    for Bk, axes in blocks.uses_attached(k, i, j):
                        # block axis a carries source axis axes[a]; the slab is on source axis 0
                        lo = tuple(x0 if axes[a] == 0 else 0 for a in range(3))
                        items.append((Bk, lo, [x.transpose(axes) for x in srcs]))
                scatter(r3b, items)


def dressed_blocks(ints, t1):
    '''(D, Fd): the T1-dressed blocks of DRESSED plus vovv, vooo and the bare
    oovv, and the dressed Fock matrix.'''
    bb = BareBlocks.of(ints)
    memo = {}
    D = {k: dress_block(bb, t1, k, memo=memo) for k in DRESSED}
    # <am|fe> = <ma|ef> and <am|ji> = <ma|ij> after dressing as well
    D['vovv'] = D['ovvv'].transpose(1, 0, 3, 2)
    D['vooo'] = dress_block(bb, t1, 'ovoo', memo=memo).transpose(1, 0, 3, 2)
    D['oovv'] = bb.get('oovv')
    return D, dress_fock(bb, ints.fock, t1)


def residuals(ints, blocks, t1, t2, t3b, r3b=None):
    '''(R1, R2, R3 blocks): dressed small blocks, the packed pp operator in
    one pass, explicit intermediates and CCSD part, one loop over occupied
    pairs for the ring (with F_vv) and T3 -> T1/T2/W, then drive and the
    ladders (F_oo in the hh operator).  ``r3b``: optional output array.'''
    from zccsdt.sa.t3blocks import project_null, zero
    from zccsdt.timing import timer
    no = ints.nocc
    bb = BareBlocks.of(ints)
    with timer('sa.dress'):
        D, Fd = dressed_blocks(ints, t1)
        G = D['oovv']
    with timer('sa.Wpp'):
        Ws, Wa = pp_operator_packed(ints, t1, t2)
    with timer('sa.imds'):
        # vvvo: bra-dressed part + sum_f t1[i,f] (bra-dressed vvvv = W - t2.G)
        B = dress_block(bb, t1, 'vvvo', ket=False)
        P = vvvv_t1_packed(Ws, Wa, t1, out=permute(B, (0, 1, 2, 3), out=np.empty(B.shape)))
        im = intermediates(D, Fd, t2, no, P=P, t1=t1)
    with timer('sa.ccsd'):
        # vvoo: bra-dressed part and the single-t1 ket terms; the t1 t1 vvvv
        # term rides with the CCSD ladder (tau = t2 + t1 t1)
        Bvo = dress_block(bb, t1, 'vvoo', ket=False)
        D['vvoo'] = Bvo + contract('if,abfj->abij', t1, B) + contract('jf,bafi->abij', t1, B)
        r1, r2 = ccsd_residual(D, Fd, t2, im, no, Ws, Wa, t1=t1)
    r3b = np.zeros_like(t3b) if r3b is None else zero(r3b)
    with timer('sa.t3_pass'):
        r1t, r2t, Eab, Eba, dWo = t3_pass(blocks, t3b, im, D, no, r3b)
    with timer('sa.drive_hh'):
        from zccsdt.cc.t3pack import axpby
        WvT = im['WvT']

        def Wv(k, out):                     # out[e,x,y] = Wv[y,x,e,k] incl. the T3 part
            axpby(out, WvT[k], 1.0, 0.0)
            axpby(out, Eab[:, k].transpose(0, 2, 1), 1.0, 1.0)
            axpby(out, Eba[:, k], 1.0, 1.0)
        drive_hh(blocks, t3b, t2, Wv, im['WoJ'] + dWo, hh_operator(im['Whh'], im['Fo']), r3b)
    with timer('sa.pp_ladder'):
        ladder_pp(blocks, t3b, Ws, Wa, r3b)
    with timer('sa.project'):
        r3b = project_null(r3b)
    return r1 + r1t, r2 + r2t, r3b


# ------------------------------------------------- block dressing and W_pp

class BareBlocks:
    '''Contiguous bare blocks <pq|rs> of the spatial integrals by block key
    ('ovvv': occupied / virtual per slot), from ints.block (cut from the full
    V or transformed from the AO integrals, cached), and the matrix of the
    T1 term of the dressed Fock operator.'''

    def __init__(self, ints):
        self.ints = ints
        self.no = ints.nocc
        self._c = {}

    @classmethod
    def of(cls, ints):
        bb = getattr(ints, '_sa_bare', None)
        if bb is None:
            bb = ints._sa_bare = cls(ints)
        return bb

    def get(self, key):
        return self.ints.block(key)

    def fock_t1(self):
        '''M[(r,s), (i,a)] = 2<ri|sa> - <ri|as>: F_rs + (M t1)_rs is the Fock
        operator with the T1-dressed density.  By (r,s) block, with pair
        symmetry: <bi|la> = <ib|al>, <bi|ca> = <ib|ac>, ...'''
        M = self._c.get('fock_t1')
        if M is None:
            no = self.no
            n = self.ints.fock.shape[0]
            v = n - no
            g = self.get
            M = np.empty((n, n, no, v))
            M[:no, :no] = 2 * g('ooov').transpose(0, 2, 1, 3) - g('oovo').transpose(0, 3, 1, 2)
            M[:no, no:] = 2 * g('oovv').transpose(0, 2, 1, 3) - g('oovv').transpose(0, 3, 1, 2)
            M[no:, :no] = 2 * g('ovvo').transpose(1, 3, 0, 2) - g('ovov').transpose(1, 2, 0, 3)
            M[no:, no:] = 2 * g('ovvv').transpose(1, 3, 0, 2) - g('ovvv').transpose(1, 2, 0, 3)
            M = self._c['fock_t1'] = M.reshape(n * n, -1)
        return M


def dress_block(bb, t1, key, ket=True, memo=None):
    '''One T1-dressed block (key like 'ovvo') from the cached bare blocks:
    bra virtual slots transform with X = 1 - t1^T, ket occupied slots with
    Y = 1 + t1 (only if ``ket``), slot by slot as bare + t1 corrections,
    <..a..> - sum_m t1[m,a] <..m..> and <..i..> + sum_e t1[i,e] <..e..>
    (the corrections use blocks with that slot switched).  ``memo`` shares
    partially dressed blocks between calls.  The result must not be
    modified (it may be a cached bare block).'''
    slots = [s for s in range(4)
             if (s < 2 and key[s] == 'v') or (s >= 2 and key[s] == 'o' and ket)]
    return _dress(bb, t1, key, tuple(slots), {} if memo is None else memo)


def _dress(bb, t1, key, slots, memo):
    if not slots:
        return bb.get(key)
    if (key, slots) in memo:
        return memo[key, slots]
    s, rest = slots[-1], slots[:-1]
    sw = key[:s] + ('o' if key[s] == 'v' else 'v') + key[s + 1:]
    base = _dress(bb, t1, key, rest, memo)
    C = _dress(bb, t1, sw, rest, memo)
    out = permute(base, (0, 1, 2, 3), out=np.empty(base.shape))
    lab = 'pqrs'
    cl = lab[:s] + 'x' + lab[s + 1:]
    if key[s] == 'v':                       # bra virtual: - sum_m t1[m,a] C[..m..]
        contract('x%s,%s->%s' % (lab[s], cl, lab), t1, C, out=out, alpha=-1.0, beta=1.0)
    else:                                   # ket occupied: + sum_e t1[i,e] C[..e..]
        contract('%sx,%s->%s' % (lab[s], cl, lab), t1, C, out=out, alpha=1.0, beta=1.0)
    memo[key, slots] = out
    return out


def dress_fock(bb, F, t1):
    '''T1-dressed Fock matrix X (F + M t1) Y^T (see BareBlocks.fock_t1).'''
    no = t1.shape[0]
    n = F.shape[0]
    X = np.eye(n)
    X[no:, :no] -= t1.T
    Y = np.eye(n)
    Y[:no, no:] += t1
    Fd = F + gemm(bb.fock_t1(), t1.reshape(-1, 1)).reshape(n, n)
    return X @ Fd @ Y.T


DRESSED = ('oooo', 'ooov', 'oovo', 'ovov', 'ovvo', 'ovvv')


def vvvv_t1_packed(Ws, Wa, t1, out=None):
    '''X[a,b,e,i] = sum_f W_ab^ef t1[i,f] from the packed parts of W (added
    to ``out`` if given).'''
    from zccsdt.cc import t3pack
    no, v = t1.shape
    if out is None:
        out = np.zeros((v, v, v, no))
    lib = t3pack._lib
    if t3pack.USE_C and hasattr(lib, 'zc_sa_vvvv_t1_d'):
        lib.zc_sa_vvvv_t1_d(t3pack._p(out), t3pack._p(Ws), t3pack._p(Wa),
                            t3pack._p(np.ascontiguousarray(t1)), t3pack._I64(v), t3pack._I64(no))
        return out
    s0, s1 = np.triu_indices(v)
    a0, a1 = np.triu_indices(v, 1)
    W = np.zeros((v, v, v, v))
    S = np.zeros((len(s0), v, v))
    S[:, s0, s1] = Ws
    S[:, s1, s0] = Ws
    A = np.zeros((len(a0), v, v))
    A[:, a0, a1] = Wa
    A[:, a1, a0] = -Wa
    W[s0, s1] += S
    W[s1, s0] += S
    W[s0[s0 == s1], s1[s0 == s1]] -= S[s0 == s1]
    W[a0, a1] += A
    W[a1, a0] -= A
    out += np.einsum('abef,if->abei', W, t1)
    return out


def vvvv_packed(ints, max_bytes=1 << 30):
    '''Bare <ab|de> symmetric / antisymmetric in (d,e), packed over rows
    a<=b (a<b) and columns d<=e (d<e) in numpy.triu_indices order.  Without
    the full V: from the 4-fold packed (ad|be) when it fits in ``max_bytes``,
    else row batches of a from the AO integrals (<ab|de> = (ad|be)).'''
    no, v = ints.nocc, ints.nvir
    s0, s1 = np.triu_indices(v)
    a0, a1 = np.triu_indices(v, 1)
    if ints.V is not None:
        Vv = ints.V[no:, no:, no:, no:]
        Vs = 0.5 * (Vv[s0, s1][:, s0, s1] + Vv[s0, s1][:, s1, s0])
        Va = 0.5 * (Vv[a0, a1][:, a0, a1] - Vv[a0, a1][:, a1, a0])
        return Vs, Va
    from zccsdt.cc import t3pack
    from zccsdt.sa.rccsdt_ref import ao_block
    Cv = np.ascontiguousarray(ints.mo_coeff[:, no:])
    Vs = np.empty((len(s0), len(s0)))
    Va = np.empty((len(a0), len(a0)))
    lib = t3pack._lib
    if (8 * len(s0) ** 2 <= max_bytes and t3pack.USE_C and hasattr(lib, 'zc_sa_vvvv_pack_d')):
        # (ad|be) with 4-fold symmetry in one transformation, then repacked
        from pyscf import ao2mo
        E4 = np.ascontiguousarray(ao2mo.kernel(ints.mol, Cv, compact=True))
        lib.zc_sa_vvvv_pack_d(t3pack._p(E4), t3pack._I64(v), t3pack._p(Vs), t3pack._p(Va))
        return Vs, Va
    nb = int(max(1, min(v, max_bytes // (8 * v ** 3))))
    rs = ra = 0
    for b0 in range(0, v, nb):
        b1 = min(v, b0 + nb)
        X = ao_block(ints.mol, (Cv[:, b0:b1], Cv, Cv, Cv))          # [a, b, d, e]
        for a in range(b0, b1):
            Y = X[a - b0, a:]                                       # rows (a, b >= a)
            Vs[rs:rs + v - a] = 0.5 * (Y[:, s0, s1] + Y[:, s1, s0])
            Va[ra:ra + v - a - 1] = 0.5 * (Y[1:, a0, a1] - Y[1:, a1, a0])
            rs += v - a
            ra += v - a - 1
    return Vs, Va


def bare_packed(ints):
    '''Bare <ab|de> symmetric / antisymmetric in (d,e), packed (a<=b, d<=e) /
    (a<b, d<e), with the matching packed ovvv (U) and oovv (G) parts; cached
    on the integral object.'''
    c = getattr(ints, '_sa_packed', None)
    if c is None:
        no, v = ints.nocc, ints.nvir
        s0, s1 = np.triu_indices(v)
        a0, a1 = np.triu_indices(v, 1)
        Vs, Va = vvvv_packed(ints)
        U = ints.block('ovvv')                                  # <mb|de>
        Us = 0.5 * (U[:, :, s0, s1] + U[:, :, s1, s0])
        Ua = 0.5 * (U[:, :, a0, a1] - U[:, :, a1, a0])
        Gf = ints.block('oovv').reshape(no * no, v, v)          # <mn|de>
        Gs = 0.5 * (Gf[:, s0, s1] + Gf[:, s1, s0])
        Ga = 0.5 * (Gf[:, a0, a1] - Gf[:, a1, a0])
        c = {k: np.ascontiguousarray(x) for k, x in
             (('Vs', Vs), ('Va', Va), ('Us', Us), ('Ua', Ua), ('Gs', Gs), ('Ga', Ga))}
        ints._sa_packed = c
    return c


def pp_operator_packed(ints, t1, t2):
    '''W_ab^de = ~<ab|de> + sum_mn t_mn^ab <mn|de> (no F_vv), as its (d,e)
    symmetric / antisymmetric packed parts, in one pass over the bare
    packed blocks (zc_vvvv_dress).'''
    from zccsdt.cc import t3pack
    lib = t3pack._lib
    c = bare_packed(ints)
    no, v = t1.shape
    s0, s1 = np.triu_indices(v)
    a0, a1 = np.triu_indices(v, 1)
    tau = (t2 + t1[:, None, :, None] * t1[None, :, None, :]).reshape(no * no, v, v)
    out = []
    I64, P = t3pack._I64, t3pack._p
    mt = np.ascontiguousarray(-t1)
    for (r0, r1), bare, U, G, sgnB in (((s0, s1), c['Vs'], c['Us'], c['Gs'], -1.0),
                                        ((a0, a1), c['Va'], c['Ua'], c['Ga'], 1.0)):
        CT = np.ascontiguousarray(tau[:, r0, r1].T)              # [(ab), (mn)]
        ws = ints.__dict__.setdefault('_sa_wpp', {})
        W = ws.get(len(r0))
        if W is None or W.shape != bare.shape:
            W = ws[len(r0)] = np.empty_like(bare)
        tB = np.ascontiguousarray(sgnB * t1)
        lib.zc_vvvv_dress_d(P(W), P(bare), I64(len(r0)), I64(bare.shape[1]),
                            P(np.ascontiguousarray(r0, dtype=np.int64)),
                            P(np.ascontiguousarray(r1, dtype=np.int64)),
                            P(mt), I64(v), P(U), I64(no), I64(v),
                            P(tB), I64(v), P(U), I64(no), I64(v),
                            P(CT), P(G), I64(no * no))
        out.append(W)
    return out
