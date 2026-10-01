'''Spin-orbital CCSDT working on packed T3 only.

Same equations as ``zccsdt.cc.ccsdt`` (the ported reference, T1-dressed
formalism), reorganized so that no full o^3 v^3 array and no full n^4
integral tensor is ever formed:

* T1 dressing is done block by block (at most o v^4 work, for vvvv).
* T3 is accessed through half-expanded batches T[m, P, d, Q]
  (t3pack.gather_half) over occupied pairs P:
    pass 1: T3 -> W_vvvo, W_vooo dressing, T3 -> T1 and T3 -> T2;
    pass 2: ring + drive terms accumulated in one half-expanded buffer and
            reduced (antisymmetrized + packed) once.
* pp ladder: batches of occupied triples, F_vv folded into W(ab,de).
* hh ladder and F_oo: one C(o,3) x C(o,3) operator applied to packed T3.
* The dtype of the integrals is kept (real stays real).

See docs/factorization.md for the derivation of each step.
'''

import numpy as np
from pyscf import lib

from zccsdt.blas import gemm
from zccsdt.cc import ccsd, t3pack, t3kern
from zccsdt.timing import timer

einsum = lib.einsum
STORED = ('oooo', 'ooov', 'oovv', 'ovov', 'ovvo', 'ovvv', 'vvvv')


# ---------------------------------------------------------------- integrals

def get_block(eris, key):
    '''Any <pq||rs> block (key like 'vovv') from the stored PySCF-convention
    blocks, via pair antisymmetry and hermiticity <pq||rs> = <rs||pq>^*.'''
    for conj in (False, True):
        k = key[2:] + key[:2] if conj else key
        for swap_bra in (False, True):
            for swap_ket in (False, True):
                s = k
                if swap_bra:
                    s = s[1] + s[0] + s[2:]
                if swap_ket:
                    s = s[:2] + s[3] + s[2]
                if s not in STORED:
                    continue
                b = np.asarray(getattr(eris, s))
                if swap_bra:
                    b = -b.transpose(1, 0, 2, 3)
                if swap_ket:
                    b = -b.transpose(0, 1, 3, 2)
                if conj:
                    b = b.transpose(2, 3, 0, 1).conj()
                return b
    raise KeyError(key)


def _contract_slot(b, M, p):
    '''b with its slot p contracted against M[source, target], using
    contiguous GEMMs (no moveaxis copies).'''
    sh = b.shape
    if p == 3:
        return (b.reshape(-1, sh[3]) @ M).reshape(sh[:3] + (M.shape[1],))
    if p == 0:
        return (M.T @ b.reshape(sh[0], -1)).reshape((M.shape[1],) + sh[1:])
    lead = int(np.prod(sh[:p]))
    out = np.matmul(M.T, b.reshape(lead, sh[p], -1))
    return out.reshape(sh[:p] + (M.shape[1],) + sh[p + 1:])


def dress_block(eris, t1, key):
    '''T1-dressed block of <pq||rs>: a virtual bra index a picks up
    -sum_m t1[m,a] <m..||..>, an occupied ket index i picks up
    +sum_e t1[i,e] <..||..e>.  Other indices are unchanged.

    The dressing is a product of one-index transformations, so it is applied
    slot by slot (ket slots first): each step combines the blocks that differ
    only in the source space of that slot.'''
    alt = {}
    for p, sp in enumerate(key):
        if p < 2 and sp == 'v':
            alt[p] = ('o', -t1)                      # M[m, a]
        elif p >= 2 and sp == 'o':
            alt[p] = ('v', t1.T)                     # M[e, i]
    choices = [(key[p], alt[p][0]) if p in alt else (key[p],) for p in range(4)]
    state = {''.join(c): get_block(eris, ''.join(c))
             for c in np.ndindex(*[len(x) for x in choices])
             for c in [tuple(choices[p][i] for p, i in enumerate(c))]}
    for p in sorted(alt, reverse=True):
        src, M = alt[p]
        new = {}
        for k, blk in state.items():
            if k[p] != key[p]:
                continue
            k_alt = k[:p] + src + k[p + 1:]
            new[k] = blk + _contract_slot(state[k_alt], M, p)
        state = new
    return state[key]


def vvvv_packed(eris, vm):
    '''Undressed <ab||de> packed over a<b, d<e, cached on the integral object
    (built once from the stored block).'''
    cache = getattr(eris, '_vvvv_pp', None)
    if cache is None:
        md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
        cache = np.ascontiguousarray(np.asarray(eris.vvvv)[md[:, None], ne[:, None], md, ne])
        try:
            eris._vvvv_pp = cache
        except AttributeError:
            pass
    return cache


def _pair_t1_matrix(t1, vm):
    '''M[R, (e, i)] with sum_R V[Q, R] M[R, (e, i)] = sum_f V[Q, e, f] t1[i, f]
    for V antisymmetric in (e, f) and packed over R = (e' < f'):
    M = delta(e, e') t1[i, f'] - delta(e, f') t1[i, e'].'''
    nocc, nvir = t1.shape
    M = np.zeros((vm.npair, nvir, nocc), dtype=t1.dtype)
    R = np.arange(vm.npair)
    e1, f1 = vm.pairs[:, 0], vm.pairs[:, 1]
    M[R, e1] = t1[:, f1].T
    M[R, f1] = -t1[:, e1].T
    return M.reshape(vm.npair, nvir * nocc)


def wvvvo_packed(eris, t1, t2, fov, ge, om, vm):
    '''W_vvvo[Q(a<b), e, i] of the T3 drive term (without the T3 dressing),
    computed packed and without the full dressed vvvv:
      ~<ab||ei> = K_vv[ab] - L[a,b] + L[b,a] + sum_{m<n} tau[mn,ab] K_oo[mn]
      K_xy = <xy||ei> + sum_f <xy||ef> t1[i,f],  L[a,b] = sum_m t1[m,a] K_ov[m,b]
    plus the T2 terms of W_vvvo with the dressed oovo, ovvv and Fock.'''
    nocc, nvir = t1.shape
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
    mm, nn = om.pairs[:, 0], om.pairs[:, 1]
    dtype = np.result_type(t1, t2, eris.oovv)
    V = vvvv_packed(eris, vm)
    # K_vv packed
    W = np.array(get_block(eris, 'vvvo')[md, ne], dtype=dtype, order='C', copy=True)
    gemm(V, _pair_t1_matrix(t1, vm), W.reshape(vm.npair, -1), beta=1.0)
    # K_ov[m,b,e,i] and L
    ovvv = np.asarray(eris.ovvv)
    # explicit copies: get_block may return the stored array itself
    Kov = np.array(get_block(eris, 'ovvo'), dtype=dtype, order='C', copy=True)
    gemm(ovvv.reshape(-1, nvir), np.asfortranarray(t1.T), Kov.reshape(-1, nocc), beta=1.0)
    L = gemm(np.asfortranarray(t1.T), Kov.reshape(nocc, -1)).reshape(nvir * nvir, -1)
    W.reshape(vm.npair, -1)[:] -= np.take(L, md * nvir + ne, axis=0)
    W.reshape(vm.npair, -1)[:] += np.take(L, ne * nvir + md, axis=0)
    del L, Kov
    # K_oo packed over m<n and the tau term
    # contiguous copies: gemm writes into reshaped views of these
    Koo = np.array(get_block(eris, 'oovo')[mm, nn], dtype=dtype, order='C', copy=True)
    gemm(np.asarray(eris.oovv)[mm, nn].reshape(-1, nvir), np.asfortranarray(t1.T),
         Koo.reshape(-1, nocc), beta=1.0)
    tau = t1[mm][:, md] * t1[nn][:, ne] - t1[nn][:, md] * t1[mm][:, ne]      # (P, Q)
    gemm(np.asfortranarray(tau.T), Koo.reshape(om.npair, -1), W.reshape(vm.npair, -1),
         beta=1.0)
    # + 0.5 sum_mn ~<mn||ei> t2[mn,ab] = sum_{m<n} t2[mn,ab] ~oovo[mn,e,i]
    t2mn_pp = t2[mm, nn][:, md, ne]
    gemm(np.asfortranarray(t2mn_pp.T), np.ascontiguousarray(ge['oovo'][mm, nn]).reshape(
        om.npair, -1), W.reshape(vm.npair, -1), beta=1.0)
    # - P(ab) sum_{mf} ~<mb||ef> t2[mi,af]: X[(a,i),(b,e)] = t2[a,i,m,f] ovvv[m,f,b,e]
    A = np.ascontiguousarray(t2.transpose(2, 1, 0, 3)).reshape(nvir * nocc, nocc * nvir)
    B = np.ascontiguousarray(ge['ovvv'].transpose(0, 3, 1, 2)).reshape(nocc * nvir, -1)
    X = gemm(A, B).reshape(nvir, nocc, nvir, nvir)                           # [a,i,b,e]
    W -= (X[md, :, ne, :] - X[ne, :, md, :]).transpose(0, 2, 1)
    del X, A, B
    # - sum_m f[m,e] t2[m,i,a,b]
    W -= gemm(np.asfortranarray(fov.T), t2[:, :, md, ne].reshape(nocc, -1)).reshape(
        nvir, nocc, vm.npair).transpose(2, 0, 1)
    return W


def dressed_vvvv_pp(eris, t1, vm):
    '''Packed T1-dressed <ab||de> over a<b, d<e without the full v^4 array:
    ~<ab||de> = <ab||de> - Y[a,b,de] + Y[b,a,de] + Z[a,b,de],
    Y[x,y,Q] = sum_m t1[m,x] <my||Q>,  Z[a,b,Q] = sum_mn t1[m,a] t1[n,b] <mn||Q>.'''
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
    nocc, nvir = t1.shape
    cache = vvvv_packed(eris, vm)
    ovvv_Q = np.asarray(eris.ovvv)[:, :, md, ne]                   # (m, y, Q)
    Y = gemm(np.ascontiguousarray(t1.T), ovvv_Q.reshape(nocc, -1))
    Y = Y.reshape(nvir * nvir, -1)                                 # [(x,y), Q]
    out = cache - np.take(Y, md * nvir + ne, axis=0)
    out += np.take(Y, ne * nvir + md, axis=0)
    del Y
    # Z[(a<b), Q] = sum_{m<n} (t[m,a] t[n,b] - t[n,a] t[m,b]) <mn||Q>
    om = t3pack.triple_map(nocc)
    mm, nn = om.pairs[:, 0], om.pairs[:, 1]
    tau = (t1[mm][:, md] * t1[nn][:, ne] - t1[nn][:, md] * t1[mm][:, ne])   # (mn, ab)
    oovv_pp = np.asarray(eris.oovv)[mm, nn][:, md, ne]
    gemm(np.ascontiguousarray(tau.T), oovv_pp, out, beta=1.0)
    return out


def add_one_body_pairs(W, F, vm, scale):
    '''W[(a<b),(d<e)] += scale * (F_ad d_be - F_bd d_ae - F_ae d_bd + F_be d_ad),
    the antisymmetrized one-body operator F acting on either pair index.

    Grouped by the index F acts on: acting on a, the column pair is {y, b}
    with value +F[a,y] if y < b else -F[a,y]; acting on b, the column pair
    is {y, a} with value -F[b,y] if y < a else +F[b,y].'''
    a, b = vm.pairs[:, 0], vm.pairs[:, 1]
    y = np.arange(vm.n)[None, :]
    rows = np.broadcast_to(np.arange(vm.npair)[:, None], (vm.npair, vm.n))
    for keep, act, s_lt, s_gt in ((b, a, 1, -1), (a, b, -1, 1)):
        k = keep[:, None]
        valid = y != k
        col = vm.pidx[np.minimum(y, k), np.maximum(y, k)]
        val = scale * np.where(y < k, s_lt, s_gt) * F[act[:, None], y]
        W[rows[valid], col[valid]] += val[valid]
    return W


def dress_fock(eris, t1):
    '''T1-dressed Fock matrix (same definition as ccsdt._t1_dress).'''
    nocc = t1.shape[0]
    fock = np.asarray(eris.fock)
    dtype = np.result_type(fock, t1)
    n = fock.shape[0]
    sp = {'o': slice(0, nocc), 'v': slice(nocc, n)}
    fp = fock.astype(dtype)
    for r in 'ov':
        for s in 'ov':
            g = get_block(eris, r + 'o' + s + 'v')
            fp[sp[r], sp[s]] += einsum('risa,ia->rs', g, t1)
    x = np.eye(n, dtype=dtype)
    x[nocc:, :nocc] -= t1.T
    y = np.eye(n, dtype=dtype)
    y[:nocc, nocc:] += t1
    return x @ fp @ y.T


# ---------------------------------------------------------------- helpers

def _batches(n, size):
    return [np.arange(i, min(i + size, n)) for i in range(0, n, size)]


def _pair_batch_size(nocc, nvir, itemsize, mem_mb):
    '''Occupied pairs per batch for half-expanded buffers of
    nocc * nvir * nvpair elements each (a few are alive at once).'''
    per_pair = 3 * nocc * nvir * nvir * max(nvir - 1, 1) // 2 * itemsize
    return max(1, int(mem_mb * 1e6 // max(per_pair, 1)))


# ---------------------------------------------------------------- main step

def update_amps(opts, t1, t2, T3p, eris, mem_mb=2000):
    '''One Jacobi step.  T3p is packed (notri, nvtri) or flat; returns
    (t1new, t2new, T3p_new) with T3p_new flat.'''
    nocc, nvir = t1.shape
    om, vm = t3pack.triple_map(nocc), t3pack.triple_map(nvir)
    dtype = np.result_type(t1, t2, T3p, eris.oovv, eris.fock)
    T3p = np.ascontiguousarray(np.asarray(T3p, dtype=dtype).reshape(om.ntri, vm.ntri))
    mo_e = eris.mo_energy
    eia = mo_e[:nocc, None] - (mo_e[None, nocc:] + opts.level_shift)
    eijab = eia[:, None, :, None] + eia[None, :, None, :]
    mm, nn = om.pairs[:, 0], om.pairs[:, 1]
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]

    with timer('t1_dress'):
        fdr = dress_fock(eris, t1)
        ge = {}
        for k in ('ovoo', 'oooo', 'ovvo', 'oovv', 'ovvv', 'ooov', 'oovo', 'vovv'):
            with timer('dress.' + k):
                ge[k] = dress_block(eris, t1, k)
        with timer('dress.vvvv_pp'):
            vvvv_pp = dressed_vvvv_pp(eris, t1, vm)
    foo, fov, fvv = fdr[:nocc, :nocc], fdr[:nocc, nocc:], fdr[nocc:, nocc:]
    goovv = ge['oovv']

    with timer('ccsd'):
        t1new, t2new = ccsd.update_amps(opts, t1, t2, eris)

    # ---- pass 1: quantities linear in T3 that feed T1, T2 and the W's
    t2mn = t2[mm, nn]                                   # (P, a, b)
    t2P = t2[:, :, md, ne]                              # (i, j, Q)
    goovv_pp = goovv[mm, nn][:, md, ne]                 # (P, Q)
    goovv_P = goovv[:, :, md, ne]                       # (m, n, Q)
    goovv_mn = goovv[mm, nn]                            # (P, e, f)
    gvovv_Q = ge['vovv'][:, :, md, ne]                  # (a, m, Q)
    goovo_mn = ge['oovo'][mm, nn]                       # (P, e, j)
    dWvT = np.zeros((nocc, nvir, vm.npair), dtype=dtype)  # [i, e, Q(ab)]
    dWo = np.zeros((nocc, nvir, om.npair), dtype=dtype)   # [m, a, P(ij)]
    r1 = np.zeros(nvir * nocc, dtype=dtype)
    rA = np.zeros((om.npair, vm.npair), dtype=dtype)
    B0 = np.zeros((om.npair, nvir, nvir), dtype=dtype)
    C0 = np.zeros((nocc, nocc, vm.npair), dtype=dtype)
    nb = _pair_batch_size(nocc, nvir, np.dtype(dtype).itemsize, mem_mb)
    batches = _batches(om.npair, nb)
    # stacked integral operands so that pass 1 is two GEMMs per batch
    G_Pd = np.concatenate((goovv_mn.transpose(1, 0, 2),        # [e, P, f]
                           goovo_mn.transpose(2, 0, 1)), axis=0)   # [j, P, e]
    G_mQ = np.concatenate((goovv_P.transpose(1, 2, 0),        # [n, R, m']
                           gvovv_Q.transpose(1, 2, 0)), axis=2)    # [m, R, a]
    G_mQ = G_mQ.reshape(nocc * vm.npair, nocc + nvir)
    fovT = np.ascontiguousarray(fov.T).ravel()                 # [(e, m)]
    if t3kern.USE_C:
        with timer('pass1'):
            Xacc, Yall, r1, rA = t3kern.pass1(T3p, om, vm, nb, G_Pd, G_mQ,
                                              goovv_pp, fovT)
        dWvT = Xacc[:nvir].reshape(nvir, nocc, vm.npair).transpose(1, 0, 2)
        C0 = -2 * Xacc[nvir:].reshape(nocc, nocc, vm.npair).transpose(1, 0, 2)
        dWo = Yall[:, :, :nocc].transpose(2, 1, 0)
        B0 = -2 * Yall[:, :, nocc:].transpose(0, 2, 1)
    else:
        with timer('pass1'):
            for pb in batches:
                nbp = len(pb)
                with timer('gather_half'):
                    T = t3pack.gather_half(T3p, om, vm, pb)     # (Pb, d, m, Q)
                Tf = T.reshape(nbp * nvir, nocc * vm.npair)
                with timer('pass1_gemm_Pd'):
                    # sum over (P, d): dW_vvvo[i,e,Q] and C0[i,j,Q]
                    X = np.matmul(np.ascontiguousarray(G_Pd[:, pb]).reshape(
                        nvir + nocc, nbp * nvir), Tf)
                    X = X.reshape(nvir + nocc, nocc, vm.npair)
                    dWvT += X[:nvir].transpose(1, 0, 2)
                    C0 -= 2 * X[nvir:].transpose(1, 0, 2)
                with timer('pass1_gemm_mQ'):
                    # sum over (m, Q): dW_vooo[m',a,P] and B0[P,a,b]
                    Y = np.matmul(Tf, G_mQ)
                    Y = Y.reshape(nbp, nvir, nocc + nvir)
                    dWo[:, :, pb] = Y[:, :, :nocc].transpose(2, 1, 0)
                    B0[pb] = -2 * Y[:, :, nocc:].transpose(0, 2, 1)
                with timer('pass1_gemv'):
                    # T3 -> T1: r1[a,i] += sum_{P,Q} g[P,Q] T[P,a,i,Q]
                    Tm = T.reshape(nbp, nvir * nocc, vm.npair)
                    for k, P in enumerate(pb):
                        r1 += Tm[k] @ goovv_pp[P]
                    # T3 -> T2: rA[P,Q] = sum_{m,e} f[m,e] T[P,e,m,Q]
                    rA[pb] = np.matmul(fovT, Tm)
                del T, Tm, X, Y
    r1 = r1.reshape(nvir, nocc).T
    dWv = dWvT.transpose(2, 1, 0)                       # [Q, e, i]
    t1new = t1new + r1 / eia
    rB = 0.5 * (B0[:, md, ne] - B0[:, ne, md])
    rC = -0.5 * (C0[mm, nn] - C0[nn, mm])
    r2 = _unpack_pairs(rA + rB + rC, om, vm)
    t2new = t2new + r2 / eijab

    # ---- effective intermediates for the T3 residual
    with timer('intermediates'):
        govoo = ge['ovoo']
        ovvv, ooov = ge['ovvv'], ge['ooov']
        t_ = timer('imd.Wvvvo'); t_.__enter__()
        Wv = wvvvo_packed(eris, t1, t2, fov, ge, om, vm) + dWv   # [Q(ab), e, i]
        t_.__exit__(None, None, None)
        t_ = timer('imd.Wvooo'); t_.__enter__()
        Wvooo = govoo.copy()
        tmp2 = einsum('mnie,jnbe->mbij', ooov, t2)
        Wvooo -= (tmp2 - tmp2.transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2)
        Wvooo -= (0.5 * einsum('mbef,ijef->mbij', ovvv, t2)).transpose(0, 1, 3, 2)
        Wo = Wvooo[:, :, nn, mm] + dWo                   # [m, a, P(ij)]
        t_.__exit__(None, None, None)
        t_ = timer('imd.ring_F'); t_.__enter__()
        Weff_ovvo = ge['ovvo'] + einsum('mnde,inae->madi', goovv, t2)
        Feff_vv = fvv - einsum('Paf,Pdf->ad', t2mn, goovv_mn)
        Feff_oo = foo + einsum('mnP,inP->mi', goovv_P, t2P)
        t_.__exit__(None, None, None)
        t_ = timer('imd.Wpp'); t_.__enter__()
        Wpp = vvvv_pp + t2mn[:, md, ne].T @ goovv_pp
        # F_vv folded into the ladder: W(ab,de) += 1/2 A[F_ad delta_be]
        add_one_body_pairs(Wpp, Feff_vv, vm, 0.5)
        t_.__exit__(None, None, None)
        t_ = timer('imd.O6'); t_.__enter__()
        Weff_oooo = ge['oooo'] + einsum('mnP,ijP->mnij', goovv_P, t2P)
        O6 = t3pack.occ_operator(om, lambda t: _occ_apply(t, Weff_oooo, Feff_oo))
        t_.__exit__(None, None, None)

    R = np.zeros((om.ntri, vm.ntri), dtype=dtype)

    # ---- pass 2: ring + drive in one buffer Z[P, a, i, Q], reduced once
    # ring operator rows (a, i), columns (d, m) to match T[P, d, m, Q]
    WringT = Weff_ovvo.transpose(1, 3, 2, 0).reshape(nvir * nocc, nvir * nocc)
    WvT2 = np.ascontiguousarray(Wv.transpose(1, 2, 0)).reshape(nvir, -1)   # [e, (i,Q)]
    t2P_flat = t2P.reshape(nocc, -1)                                     # [m, (k,Q)]
    if t3kern.USE_C:
        with timer('pass2'):
            t3kern.pass2(T3p, om, vm, nb, WringT, t2mn.transpose(0, 2, 1), WvT2,
                         Wo.transpose(2, 1, 0), t2P_flat, R)
    else:
        with timer('pass2'):
            for pb in batches:
                nbp = len(pb)
                with timer('gather_half'):
                    T = t3pack.gather_half(T3p, om, vm, pb)     # (Pb, d, m, Q)
                with timer('ring_gemm'):
                    Z = np.empty((nbp, nvir * nocc, vm.npair), dtype=dtype)
                    Tm = T.reshape(nbp, nvir * nocc, vm.npair)
                    for k in range(nbp):
                        # numpy matmul is ~2x faster than SciPy's gemm for this
                        # shape (same OpenBLAS); Z is overwritten, no beta needed
                        np.matmul(WringT, Tm[k], out=Z[k])     # Z[P, a, i, Q]
                del T, Tm
                with timer('drive'):
                    Zf = Z.reshape(nbp * nvir, nocc * vm.npair)
                    # Xr1: Z[P,c,i,Q] -= sum_e t2mn[P,e,c] Wv[Q,e,i]
                    gemm(t2mn[pb].transpose(0, 2, 1).reshape(nbp * nvir, nvir), WvT2,
                         Zf, alpha=-1.0, beta=1.0)
                    # Xr2: Z[P,a,k,Q] -= sum_m Wo[m,a,P] t2P[m,k,Q]
                    gemm(Wo[:, :, pb].transpose(2, 1, 0).reshape(nbp * nvir, nocc),
                         t2P_flat, Zf, alpha=-1.0, beta=1.0)
                with timer('reduce'):
                    t3pack.add_reduce_split(R, Z, om, vm, pb)
                    del Z

    # ---- pp ladder (+F_vv) over batches of occupied triples
    with timer('pp_ladder'):
        nbt = max(1, int(mem_mb * 1e6 // (3 * vm.npair * nvir * np.dtype(dtype).itemsize)))
        if t3kern.USE_C:
            t3kern.pp_ladder(T3p, om, vm, nbt, Wpp, R)
        else:
            for rows in _batches(om.ntri, nbt):
                T = t3pack.gather_vir_half(T3p, vm, rows)            # (Q, nb, c)
                X = gemm(Wpp, T.reshape(vm.npair, -1))                # (Q', nb*c)
                t3pack.add_reduce_pair_first(R, X, vm, rows)
                del T, X

    # ---- hh ladder (+F_oo): occupied-space operator on packed T3
    with timer('hh_ladder'):
        if t3kern.USE_C:
            t3kern.occ_operator(O6, T3p, R)
        else:
            R += O6.astype(dtype) @ T3p

    with timer('t3_update'):
        eo = mo_e[:nocc][om.triples].sum(axis=1)
        ev = (mo_e[nocc:] + opts.level_shift)[vm.triples].sum(axis=1)
        if t3kern.USE_C:
            T3new = T3p.copy()
            t3kern.update(T3new, R, eo, ev)
        else:
            T3new = T3p + R / (eo[:, None] - ev[None, :])
    return t1new, t2new, T3new.ravel()


def _unpack_pairs(packed, om, vm):
    '''Full antisymmetric r2[i,j,a,b] from its (i<j, a<b) block.'''
    mm, nn = om.pairs[:, 0][:, None], om.pairs[:, 1][:, None]
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
    r2 = np.zeros((om.n, om.n, vm.n, vm.n), dtype=packed.dtype)
    r2[mm, nn, md, ne] = packed
    r2[nn, mm, md, ne] = -packed
    r2[mm, nn, ne, md] = -packed
    r2[nn, mm, ne, md] = packed
    return r2


def _occ_apply(t, Woooo, Foo):
    '''(L t)[J,i,j,k] = P(k/ij) 1/2 sum_mn W[m,n,i,j] t[J,m,n,k]
                      - P(i/jk) sum_m F[m,i] t[J,m,j,k].'''
    x = 0.5 * einsum('mnij,Jmnk->Jijk', Woooo, t)
    out = x - x.transpose(0, 3, 2, 1) - x.transpose(0, 1, 3, 2)
    y = einsum('mi,Jmjk->Jijk', Foo, t)
    out -= y - y.transpose(0, 2, 1, 3) - y.transpose(0, 3, 2, 1)
    return out
