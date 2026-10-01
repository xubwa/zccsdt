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
    '''Closed-shell spin-adapted CCSDT (Jacobi steps with DIIS).'''
    from zccsdt.cc.diis import DIIS
    from zccsdt.sa import rccsdt_ref
    from zccsdt.sa.t3blocks import Blocks, denominators
    no, nv = ints.nocc, ints.nvir
    blocks = Blocks(no)
    e = ints.mo_energy
    eo, ev = e[:no], e[no:]
    d1 = eo[:, None] - ev[None, :]
    d2 = d1[:, None, :, None] + d1[None, :, None, :]
    d3 = denominators(blocks, eo, ev)
    t1 = np.zeros((no, nv))
    t2 = ints.V[:no, :no, no:, no:] / d2
    t3b = np.zeros((blocks.nblk, nv, nv, nv))
    adiis = DIIS(space=diis_space)
    e_old = rccsdt_ref.energy(t1, t2, ints)
    conv = False
    for cycle in range(max_cycle):
        r1, r2, r3b = residuals(ints, blocks, t1, t2, t3b)
        normt = float(np.linalg.norm(r1) + np.linalg.norm(r2) + np.linalg.norm(r3b))
        out = adiis.update([t1 + r1 / d1, t2 + r2 / d2, t3b + r3b / d3])
        t1 = out[0].reshape(no, nv)
        t2 = out[1].reshape(no, no, nv, nv)
        t3b = out[2].reshape(t3b.shape)
        e_new = rccsdt_ref.energy(t1, t2, ints)
        if verbose:
            print('cycle %3d  E_corr = %.15f  dE = %.3e  |r| = %.3e'
                  % (cycle + 1, e_new, e_new - e_old, normt))
        done = abs(e_new - e_old) < conv_tol and normt < conv_tol_normt
        e_old = e_new
        if done:
            conv = True
            break
    return rccsdt_ref.Result(e_corr=e_old, e_tot=ints.e_hf + e_old, converged=conv,
                             ncycle=cycle + 1, t1=t1, t2=t2, t3=t3b)


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


def fvv_term(t3b, Fv, r3b):
    '''r3b += sum_d Fv_ad t^{dbc} on each of the three virtual axes of every
    cube (the F_vv term of all three positions).'''
    nb, v = t3b.shape[0], t3b.shape[1]
    for b in range(nb):
        c = t3b[b]
        r3b[b] += gemm(Fv, c.reshape(v, -1)).reshape(v, v, v)
        r3b[b] += gemm(c.reshape(-1, v), np.ascontiguousarray(Fv.T)).reshape(v, v, v)
        r3b[b] += gemm(Fv, np.ascontiguousarray(c.transpose(1, 0, 2)).reshape(v, -1)) \
            .reshape(v, v, v).transpose(1, 0, 2)


def t3_pass(blocks, t3b, imds, D, nocc, r3b):
    '''One loop over sorted occupied pairs (p <= q).  From the gathered cubes
    T[m, x, y, z] = t_mpq^{xyz} it forms T2g = T(0,2,1,3), T3g = T(0,3,1,2),
    Y = 2T - T(0,2,1,3) - T(0,3,2,1) (= t~ of the ring) and
    Z = 2T(0,1,3,2) - T(0,3,1,2) - T(0,2,3,1); for the reversed pair (q, p)
    the roles are T2g <-> T3g, Y <-> Z.  Adds the ring to r3b and returns the
    T3 parts (r1, r2, dWv, dWo) of R1, R2, Wv (J) and Wo (J).'''
    from zccsdt.cc.t3pack import axpby
    no, v = blocks.nocc, t3b.shape[1]
    o, vs = slice(0, no), slice(no, None)
    G, L, Fov = imds['G'], imds['L'], imds['Fov']
    oovo, ooov = D['oovo'], D['ooov']                              # <mn|ej>, <mn|je>
    Vo = np.ascontiguousarray(D['ovvv'].transpose(1, 0, 2, 3)).reshape(v, -1)        # [a,(m,f,e)]
    Gm = np.ascontiguousarray(G.transpose(1, 0, 2, 3)).reshape(no, -1)                 # [m,(n,f,e)]
    J2 = np.ascontiguousarray(imds['WrJ'].transpose(1, 3, 0, 2)).reshape(v * no, no * v)
    K2 = np.ascontiguousarray(imds['WrK'].transpose(1, 3, 0, 2)).reshape(v * no, no * v)
    r1 = np.zeros((no, v))
    r2 = np.zeros((no, no, v, v))
    r2a = np.zeros((no, no, v, v))                  # [i, j, a, b], symmetrized at the end
    Eab = np.zeros((no, v, v, v))                   # dWv as [i, e, a, b]
    Eba = np.zeros((no, v, v, v))                   # dWv as [i, e, b, a]
    dWo = np.zeros((no, v, no, no))
    T = np.empty((no, v, v, v))
    T2g, T3g, Y, Z = (np.empty_like(T) for _ in range(4))
    ov, vv = no * v, v * v
    for p in range(no):
        for q in range(p, no):
            for m in range(no):
                axpby(T[m], blocks.cube(t3b, m, p, q), 1.0, 0.0)
            axpby(T2g, T.transpose(0, 2, 1, 3), 1.0, 0.0)
            axpby(T3g, T.transpose(0, 3, 1, 2), 1.0, 0.0)
            axpby(Y, T, 2.0, 0.0)
            axpby(Y, T.transpose(0, 2, 1, 3), -1.0)
            axpby(Y, T.transpose(0, 3, 2, 1), -1.0)
            axpby(Z, T.transpose(0, 1, 3, 2), 2.0, 0.0)
            axpby(Z, T.transpose(0, 3, 1, 2), -1.0)
            axpby(Z, T.transpose(0, 2, 3, 1), -1.0)
            T2, Y2 = T.reshape(ov, vv), Y.reshape(ov, vv)
            T2g2, T3g2 = T2g.reshape(ov, vv), T3g.reshape(ov, vv)
            # ---- ring: X_{i;pq}^{a;bc} = X1[(a,i),(b,c)] + X2[(b,i),(a,c)] + X3[(c,i),(a,b)]
            X1 = gemm(J2, Y2)
            gemm(K2, T2, X1, alpha=-1.0, beta=1.0)
            X2 = gemm(K2, T2g2, alpha=-1.0).reshape(v, no, v, v)
            X3 = gemm(K2, T3g2, alpha=-1.0).reshape(v, no, v, v)
            X1 = X1.reshape(v, no, v, v)
            for i in range(no):
                parts = (X1[:, i], X2[:, i].transpose(1, 0, 2), X3[:, i].transpose(1, 2, 0))
                for B, axes in blocks.uses_attached(i, p, q):
                    for x in parts:
                        axpby(r3b[B], x.transpose(axes), 1.0)
            # ---- pass 1: (A) (m,n) = (pp,qq) summed, external x;  (B) m = x summed
            orders = [(p, q, T2g, T3g, Y, Z, False)]
            if q != p:
                orders.append((q, p, T3g, T2g, Z, Y, True))
            for pp, qq, Tp, Tq, Yc, Zc, rev in orders:
                Lpq = L[pp, qq]                                             # [e, f]
                Gpq = G[pp, qq]
                Lp = 2 * Gpq - Gpq.T                                        # [f, e]
                A = -2 * oovo[pp, qq] + ooov[pp, qq].T                      # [e, j]
                if not rev:
                    r1 += (gemm(T2, Lpq.reshape(vv, 1)) - gemm(T3g2, np.ascontiguousarray(Lpq.T).reshape(vv, 1))).reshape(no, v)
                else:
                    r1 += (gemm(T2, np.ascontiguousarray(Lpq.T).reshape(vv, 1))
                           - gemm(T2g2, np.ascontiguousarray(Lpq.T).reshape(vv, 1))).reshape(no, v)
                LpT, AT, ooT = (np.ascontiguousarray(x.T) for x in (Lp, A, oovo[pp, qq]))
                Gc = np.ascontiguousarray(Gpq)
                for i in range(no):
                    # dWv: -sum_f Lp[f,e] C[i,b,f,a] + sum_f G[e,f] C[i,f,a,b]
                    gemm(LpT, Tp[i].reshape(v, vv), Eba[i].reshape(v, vv), alpha=-1.0, beta=1.0)
                    gemm(Gc, T[i].reshape(v, vv), (Eba if rev else Eab)[i].reshape(v, vv),
                         beta=1.0)
                    # R2 (oovo), i-part: sum_e A[e,j] C[i,a,e,b] + oovo[e,j] C[i,e,a,b]
                    gemm(AT, Tp[i].reshape(v, vv), r2a[i].reshape(no, vv), beta=1.0)
                    if rev:
                        jb = gemm(ooT, T[i].reshape(v, vv)).reshape(no, v, v)   # [j, b, a]
                        axpby(r2a[i], jb.transpose(0, 2, 1), 1.0)
                    else:
                        gemm(ooT, T[i].reshape(v, vv), r2a[i].reshape(no, vv), beta=1.0)
                # (B): F, ovvv and Wo terms
                r2[pp, qq] += 0.5 * gemm(Fov.reshape(1, -1), Yc.reshape(ov, vv)).reshape(v, v)
                r2[pp, qq] += 0.5 * gemm(Vo, Yc.reshape(-1, v))
                r2[pp, qq] += 0.5 * gemm(Vo, Zc.reshape(-1, v)).T
                dWo[:, :, qq, pp] += gemm(Gm, Zc.reshape(-1, v))
    r2 = r2 + r2a
    r2 = r2 + r2.transpose(1, 0, 3, 2)
    dWv = Eab.transpose(2, 3, 1, 0) + Eba.transpose(3, 2, 1, 0)
    return r1, r2, dWv, dWo


def residuals(ints, blocks, t1, t2, t3b):
    '''(R1, R2, R3 blocks): dressed small blocks, the packed pp operator in
    one pass, explicit intermediates and CCSD part, one loop over occupied
    pairs for the ring and T3 -> T1/T2/W, then drive, ladders and F terms.'''
    from zccsdt.sa.t3blocks import project_null
    no = ints.nocc
    D = {k: dress_block(ints.V, t1, no, k) for k in DRESSED}
    G = D['oovv'] = np.ascontiguousarray(ints.V[:no, :no, no:, no:])
    Fd = dress_fock(ints.V, ints.fock, t1, no)
    Ws, Wa = pp_operator_packed(ints, t1, t2)
    # vvvo: bra-dressed part + sum_f t1[i,f] (bra-dressed vvvv = W - t2.G)
    B = dress_block(ints.V, t1, no, 'vvvo', ket=False)
    D['vvvo'] = B + vvvv_t1_packed(Ws, Wa, t1) \
        - contract('mnab,mnei->abei', t2, contract('mnef,if->mnei', G, t1))
    im = intermediates(D, Fd, t2, no)
    # vvoo: bra-dressed part and the single-t1 ket terms; the t1 t1 vvvv term
    # rides with the CCSD ladder (tau = t2 + t1 t1)
    Bvo = dress_block(ints.V, t1, no, 'vvoo', ket=False)
    D['vvoo'] = Bvo + contract('if,abfj->abij', t1, B) + contract('jf,bafi->abij', t1, B)
    r1, r2 = ccsd_residual(D, Fd, t2, im, no, Ws, Wa, t1=t1)
    r3b = np.zeros_like(t3b)
    r1t, r2t, dWv, dWo = t3_pass(blocks, t3b, im, D, no, r3b)
    drive(blocks, t2, im['WvJ'] + dWv, im['WoJ'] + dWo, r3b)
    ladder_pp(blocks, t3b, Ws, Wa, r3b)
    fvv_term(t3b, im['Fv'], r3b)
    ladder_hh(blocks, t3b, hh_operator(im['Whh'], im['Fo']), r3b)
    return r1 + r1t, r2 + r2t, project_null(r3b)


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
