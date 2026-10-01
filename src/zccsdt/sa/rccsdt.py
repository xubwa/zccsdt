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

with P the six-term pair symmetrizer; F_vv and F_oo are folded into the
ladder operators.  Wv, Wo, J, K are the J (resp. K) spatial components of
the intermediates of zccsdt.sa.derive.
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
    lib = t3pack._lib
    nb, v = t3b.shape[0], t3b.shape[1]
    v2, v3 = v * v, v * v * v
    view = {'01': (v2, v, 1), '02': (v2, 1, v), '12': (v, 1, v2)}
    ps, pa = v * (v + 1) // 2, v * (v - 1) // 2
    I64, P = t3pack._I64, t3pack._p
    arr = lambda x: np.array(x, dtype=np.int64).reshape(-1, 4)   # noqa: E731
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
        Ts = np.empty((ps, ncol))
        Ta = np.empty((pa, nf * v))
        F, S_ = arr(full), arr(sym)
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
        S = gemm(Ws, Ts)
        A = gemm(Wa, Ta) if nf else None
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
    t2 = ints.V[:no, :no, no:, no:] / d2
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

def contract(spec, A, B):
    '''Two-operand einsum without traces or batch indices, as one MKL GEMM
    (transpose, transpose, GEMM, transpose).'''
    lhs, out = spec.replace(' ', '').split('->')
    sa, sb = lhs.split(',')
    k = [c for c in sa if c in sb]
    fa = [c for c in sa if c not in k]
    fb = [c for c in sb if c not in k]
    dims = dict(zip(sa, A.shape))
    dims.update(zip(sb, B.shape))
    At = np.ascontiguousarray(A.transpose([sa.index(c) for c in fa + k])) if fa + k != list(sa) else A
    Bt = np.ascontiguousarray(B.transpose([sb.index(c) for c in k + fb])) if k + fb != list(sb) else B
    m = int(np.prod([dims[c] for c in fa], dtype=np.int64))
    kk = int(np.prod([dims[c] for c in k], dtype=np.int64))
    n = int(np.prod([dims[c] for c in fb], dtype=np.int64))
    C = gemm(np.ascontiguousarray(At).reshape(m, kk), np.ascontiguousarray(Bt).reshape(kk, n))
    C = C.reshape([dims[c] for c in fa + fb])
    order = [(fa + fb).index(c) for c in out]
    return C.transpose(order) if order != list(range(len(order))) else C


_einsum = contract


def dress(V, F, t1, nocc):
    '''T1-dressed <pq|rs> and Fock (see rccsdt_ref.dress), index by index
    with MKL GEMMs.'''
    n = F.shape[0]
    X = np.eye(n)
    X[nocc:, :nocc] -= t1.T
    Y = np.eye(n)
    Y[:nocc, nocc:] += t1
    # slot 0 (bra): Vd[p,...] = sum_P X[p,P] V[P,...]; then rotate the slots
    Vd = V
    for M in (X, X, Y, Y):
        Vd = gemm(M, np.ascontiguousarray(Vd).reshape(n, -1)).reshape(n, n, n, n)
        Vd = Vd.transpose(1, 2, 3, 0)
    Vd = np.ascontiguousarray(Vd)
    o, v = slice(0, nocc), slice(nocc, n)
    Fd = F + np.einsum('risa,ia->rs', 2 * V[:, o, :, v] - V[:, o, v, :].transpose(0, 1, 3, 2), t1)
    return Vd, X @ Fd @ Y.T


def intermediates(D, Fd, t2, nocc):
    '''T2 parts of the intermediates (spatial J / K components of
    zccsdt.sa.derive) from the T1-dressed blocks D (see DRESSED, plus the
    bare 'oovv') and Fock.  The pp ladder operator is built separately
    (pp_operator_packed).'''
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
    ovvv = D['ovvv']                                     # <ma|fe>
    out['WvJ'] = D['vvvo'] - contract('me,miab->abei', Fd[o, v], t2) \
        + contract('mnab,mnei->abei', t2, D['oovo']) \
        + contract('mafe,mifb->abei', ovvv, 2 * t2 - t2.transpose(0, 1, 3, 2)) \
        - contract('maef,mifb->abei', ovvv, t2) - contract('mbef,miaf->abei', ovvv, t2)
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


def add3(dst, srcs, alphas=(1.0, 1.0, 1.0)):
    '''dst += sum_k alphas[k] srcs[k] for (v,v,v) arrays / strided views (C
    kernel zc_add3_strided; dst C-contiguous).'''
    from zccsdt.cc import t3pack
    lib = t3pack._lib
    if not (t3pack.USE_C and hasattr(lib, 'zc_add3_strided_d')):
        for a, x in zip(alphas, srcs):
            dst += a * x
        return dst
    isz = dst.itemsize
    ptrs = [None] * 3
    st = np.zeros(9, dtype=np.int64)
    for k, x in enumerate(srcs):
        ptrs[k] = t3pack.ctypes.c_void_p(x.ctypes.data)
        st[3 * k:3 * k + 3] = [y // isz for y in x.strides]
    dstr = np.array([y // isz for y in dst.strides], dtype=np.int64)
    al = np.array(list(alphas) + [0.0] * (3 - len(alphas)), dtype=np.float64)
    lib.zc_add3_strided_d(t3pack.ctypes.c_void_p(dst.ctypes.data), t3pack._p(dstr),
                          t3pack._I64(dst.shape[0]), ptrs[0], ptrs[1], ptrs[2],
                          t3pack._p(st), t3pack._p(al))
    return dst


def pair_gather(blocks, t3b, p, q, W):
    '''W[k][x, m, y, z] for the cubes s(x,y,z) = t_mpq^{xyz} of the pair (p, q):
    k = 0: Y = 2 s(x,y,z) - s(y,x,z) - s(z,y,x),  1: T = s(x,y,z),
    2: T2 = s(y,x,z),  3: T3 = s(y,z,x),  4: Z = 2 s(x,z,y) - s(y,z,x) - s(z,x,y)
    (Z is Y with y <-> z).'''
    from zccsdt.cc import t3pack
    from zccsdt.cc.t3pack import axpby
    no, v = blocks.nocc, t3b.shape[1]
    lib = t3pack._lib
    if t3pack.USE_C and hasattr(lib, 'zc_sa_pair_gather_t_d') and t3b.dtype in (np.float64, np.complex128):
        bl = blocks.blk[:, p, q]
        off = np.where(bl >= 0, bl * v ** 3, -1).astype(np.int64)
        cst = np.array((v * v, v, 1), dtype=np.int64)
        st = np.ascontiguousarray(cst[blocks.axes[:, p, q]].ravel())
        P = t3pack._p
        ptr = [t3pack.ctypes.c_void_p(W[k].ctypes.data) for k in range(5)]
        getattr(lib, 'zc_sa_pair_gather_t_' + t3pack._sfx(t3b.dtype))(
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
    T3 parts (r1, r2, dWv, dWo) of R1, R2, Wv (J) and Wo (J).  For the
    reversed pair (q, p) the roles are T2 <-> T3 and Y <-> Z.'''
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
    ACC = np.zeros((2 * no + 2 * v, no, v, v))
    dWo = np.zeros((no, v, no, no))
    W = np.empty((5, v, no, v, v))
    Wm = W.reshape(5, v, no * vv)
    for p in range(no):
        for q in range(p, no):
            with timer('t3p.gather'):
                pair_gather(blocks, t3b, p, q, W)
            # ---- ring: X_{i;pq}^{a;bc} = X1[(a,i),(b,c)] + X2[(b,i),(a,c)] + X3[(c,i),(a,b)]
            with timer('t3p.ring_gemm'):
                X1 = gemm(JK, W[0:2].reshape(2 * ov, vv)).reshape(v, no, v, v)
                X2 = gemm(mK2, W[2].reshape(ov, vv)).reshape(v, no, v, v)
                X3 = gemm(mK2, W[3].reshape(ov, vv)).reshape(v, no, v, v)
            with timer('t3p.scatter'):
                for i in range(no):
                    parts = (X1[:, i], X2[:, i].transpose(1, 0, 2), X3[:, i].transpose(1, 2, 0))
                    for B, axes in blocks.uses_attached(i, p, q):
                        add3(r3b[B], [x.transpose(axes) for x in parts])
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
    dWv = Eab.transpose(2, 3, 0, 1) + Eba.transpose(3, 2, 0, 1)
    return np.ascontiguousarray(r1x.T), r2, dWv, dWo


def residuals(ints, blocks, t1, t2, t3b, r3b=None):
    '''(R1, R2, R3 blocks): dressed small blocks, the packed pp operator in
    one pass, explicit intermediates and CCSD part, one loop over occupied
    pairs for the ring (with F_vv) and T3 -> T1/T2/W, then drive and the
    ladders (F_oo in the hh operator).  ``r3b``: optional output array.'''
    from zccsdt.sa.t3blocks import project_null, zero
    from zccsdt.timing import timer
    no = ints.nocc
    with timer('sa.dress'):
        D = {k: dress_block(ints.V, t1, no, k) for k in DRESSED}
        G = D['oovv'] = np.ascontiguousarray(ints.V[:no, :no, no:, no:])
        Fd = dress_fock(ints.V, ints.fock, t1, no)
    with timer('sa.Wpp'):
        Ws, Wa = pp_operator_packed(ints, t1, t2)
    with timer('sa.imds'):
        # vvvo: bra-dressed part + sum_f t1[i,f] (bra-dressed vvvv = W - t2.G)
        B = dress_block(ints.V, t1, no, 'vvvo', ket=False)
        D['vvvo'] = B + vvvv_t1_packed(Ws, Wa, t1) \
            - contract('mnab,mnei->abei', t2, contract('mnef,if->mnei', G, t1))
        im = intermediates(D, Fd, t2, no)
    with timer('sa.ccsd'):
        # vvoo: bra-dressed part and the single-t1 ket terms; the t1 t1 vvvv
        # term rides with the CCSD ladder (tau = t2 + t1 t1)
        Bvo = dress_block(ints.V, t1, no, 'vvoo', ket=False)
        D['vvoo'] = Bvo + contract('if,abfj->abij', t1, B) + contract('jf,bafi->abij', t1, B)
        r1, r2 = ccsd_residual(D, Fd, t2, im, no, Ws, Wa, t1=t1)
    r3b = np.zeros_like(t3b) if r3b is None else zero(r3b)
    with timer('sa.t3_pass'):
        r1t, r2t, dWv, dWo = t3_pass(blocks, t3b, im, D, no, r3b)
    with timer('sa.drive'):
        drive(blocks, t2, im['WvJ'] + dWv, im['WoJ'] + dWo, r3b)
    with timer('sa.pp_ladder'):
        ladder_pp(blocks, t3b, Ws, Wa, r3b)
    with timer('sa.hh_ladder'):
        ladder_hh(blocks, t3b, hh_operator(im['Whh'], im['Fo']), r3b)
    with timer('sa.project'):
        r3b = project_null(r3b)
    return r1 + r1t, r2 + r2t, r3b


# ------------------------------------------------- block dressing and W_pp

def dress_block(V, t1, nocc, key, ket=True):
    '''One T1-dressed block of <pq|rs> (key like 'vvvo'), from the bare full V:
    bra virtual slots transform with X = 1 - t1^T, ket occupied slots with
    Y = 1 + t1 (only if ``ket``); other slots are plain slices.  Ket
    (occupied) slots first.'''
    n = V.shape[0]
    sl = {'o': slice(0, nocc), 'v': slice(nocc, n)}
    X = np.eye(n)
    X[nocc:, :nocc] -= t1.T
    Y = np.eye(n)
    Y[:nocc, nocc:] += t1
    mats, idx = [], []
    for pos, c in enumerate(key):
        if pos < 2 and c == 'v':
            mats.append(np.ascontiguousarray(X[nocc:]))
            idx.append(slice(None))
        elif pos >= 2 and c == 'o' and ket:
            mats.append(np.ascontiguousarray(Y[:nocc]))
            idx.append(slice(None))
        else:
            mats.append(None)
            idx.append(sl[c])
    out = V[tuple(idx)]
    order = sorted((p for p in range(4) if mats[p] is not None),
                   key=lambda p: mats[p].shape[0])
    for pos in order:
        x = np.ascontiguousarray(np.moveaxis(out, pos, 0))
        shape = x.shape
        y = gemm(mats[pos], x.reshape(shape[0], -1)).reshape((mats[pos].shape[0],) + shape[1:])
        out = np.moveaxis(y, 0, pos)
    return np.ascontiguousarray(out)


def dress_fock(V, F, t1, nocc):
    n = F.shape[0]
    X = np.eye(n)
    X[nocc:, :nocc] -= t1.T
    Y = np.eye(n)
    Y[:nocc, nocc:] += t1
    o, v = slice(0, nocc), slice(nocc, n)
    Fd = F + np.einsum('risa,ia->rs', 2 * V[:, o, :, v] - V[:, o, v, :].transpose(0, 1, 3, 2), t1)
    return X @ Fd @ Y.T


DRESSED = ('oooo', 'ooov', 'oovo', 'ovov', 'ovvo', 'ovvv', 'vovv', 'vooo')


def vvvv_t1_packed(Ws, Wa, t1):
    '''X[a,b,e,i] = sum_f W_ab^ef t1[i,f] from the packed parts of W.'''
    from zccsdt.cc import t3pack
    no, v = t1.shape
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
    return np.einsum('abef,if->abei', W, t1)


def bare_packed(ints):
    '''Bare <ab|de> symmetric / antisymmetric in (d,e), packed (a<=b, d<=e) /
    (a<b, d<e), with the matching packed ovvv (U) and oovv (G) parts; cached
    on the integral object.'''
    c = getattr(ints, '_sa_packed', None)
    if c is None:
        no, n = ints.nocc, ints.V.shape[0]
        v = n - no
        s0, s1 = np.triu_indices(v)
        a0, a1 = np.triu_indices(v, 1)
        Vv = ints.V[no:, no:, no:, no:]
        Vs = 0.5 * (Vv[s0, s1][:, s0, s1] + Vv[s0, s1][:, s1, s0])
        Va = 0.5 * (Vv[a0, a1][:, a0, a1] - Vv[a0, a1][:, a1, a0])
        U = ints.V[:no, no:, no:, no:]                          # <mb|de>
        Us = 0.5 * (U[:, :, s0, s1] + U[:, :, s1, s0])
        Ua = 0.5 * (U[:, :, a0, a1] - U[:, :, a1, a0])
        Gf = ints.V[:no, :no, no:, no:].reshape(no * no, v, v)   # <mn|de>
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
        W = np.empty_like(bare)
        tB = np.ascontiguousarray(sgnB * t1)
        lib.zc_vvvv_dress_d(P(W), P(bare), I64(len(r0)), I64(bare.shape[1]),
                            P(np.ascontiguousarray(r0, dtype=np.int64)),
                            P(np.ascontiguousarray(r1, dtype=np.int64)),
                            P(mt), I64(v), P(U), I64(no), I64(v),
                            P(tB), I64(v), P(U), I64(no), I64(v),
                            P(CT), P(G), I64(no * no))
        out.append(W)
    return out
