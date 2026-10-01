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
    '''Undressed <ab||de> packed over the pairs of ``vm`` (both indices),
    cached on the integral object.  The stored packed form (from the generator
    or the HDF5 file, attribute _vvvv_pp) is in triu (lexicographic) pair
    order; a spin-grouped ``vm`` gets a permuted copy.'''
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
    lex = getattr(eris, '_vvvv_pp', None)
    if vm.spins is None:
        if lex is None:
            lex = np.ascontiguousarray(np.asarray(eris.vvvv)[md[:, None], ne[:, None], md, ne])
            try:
                eris._vvvv_pp = lex
            except AttributeError:
                pass
        return lex
    cache = getattr(eris, '_vvvv_pp_spin', None)
    if cache is None:
        if lex is not None:
            n = vm.n
            perm = md * n - md * (md + 1) // 2 + (ne - md - 1)      # triu index
            cache = np.ascontiguousarray(lex[perm[:, None], perm])
        else:
            cache = np.ascontiguousarray(np.asarray(eris.vvvv)[md[:, None], ne[:, None], md, ne])
        try:
            eris._vvvv_pp_spin = cache
        except AttributeError:
            pass
    return cache


def bra_dressed_vvvo_pp(eris, t1, om, vm):
    '''<ab||ei> with only the bra (a, b) T1-dressed, packed over a<b:
    B = <ab||ei> - (L[a,b] - L[b,a]) + sum_{m<n} tau1[mn,ab] <mn||ei>,
    L[a,b,e,i] = sum_m t1[m,a] <mb||ei>.  Shape (npair_v, nvir, nocc).'''
    nocc, nvir = t1.shape
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
    mm, nn = om.pairs[:, 0], om.pairs[:, 1]
    dtype = np.result_type(t1, eris.oovv)
    # explicit copies: get_block may return the stored array itself
    B = np.array(get_block(eris, 'vvvo')[md, ne], dtype=dtype, order='C', copy=True)
    ovvo = np.ascontiguousarray(get_block(eris, 'ovvo'), dtype=dtype)
    L = gemm(np.asfortranarray(t1.T), ovvo.reshape(nocc, -1))      # [a, (b, e, i)]
    t3pack.pair_antisym_add(B, L, vm, (nvir * nvir * nocc, nvir * nocc, nocc, 1),
                            alpha=-1.0)
    del L
    tau = _tau1_pairs(t1, om, vm)                                    # (mn, ab)
    oovo = np.ascontiguousarray(get_block(eris, 'oovo')[mm, nn], dtype=dtype)
    gemm(np.asfortranarray(tau.T), oovo.reshape(om.npair, -1), B.reshape(vm.npair, -1),
         beta=1.0)
    return B


def _tau1_pairs(t1, om, vm):
    '''tau1[(m<n), (a<b)] = t1[m,a] t1[n,b] - t1[n,a] t1[m,b].'''
    mm, nn = om.pairs[:, 0], om.pairs[:, 1]
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
    return t1[mm][:, md] * t1[nn][:, ne] - t1[nn][:, md] * t1[mm][:, ne]


def wvvvo_packed(Bv, Vd, t1, t2, fov, ge, om, vm):
    '''W_vvvo[Q(a<b), e, i] of the T3 drive term (without the T3 dressing):
    the fully T1-dressed <ab||ei> = Bv + sum_f ~<ab||ef> t1[i,f] (Bv: bra-dressed
    vvvo, Vd: dressed packed vvvv), plus the T2 terms with the dressed oovo,
    ovvv and Fock.'''
    nocc, nvir = t1.shape
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
    mm, nn = om.pairs[:, 0], om.pairs[:, 1]
    W = np.array(Bv, order='C', copy=True)
    # sum_f ~<ab||ef> t1[i,f]: sparse in the packed (e<f) index, done directly
    t3pack.pair_t1_contract(W, Vd, t1, vm)
    # + 0.5 sum_mn ~<mn||ei> t2[mn,ab] = sum_{m<n} t2[mn,ab] ~oovo[mn,e,i]
    t2mn_pp = t2[mm, nn][:, md, ne]
    gemm(np.asfortranarray(t2mn_pp.T), np.ascontiguousarray(ge['oovo'][mm, nn]).reshape(
        om.npair, -1), W.reshape(vm.npair, -1), beta=1.0)
    # - P(ab) sum_{mf} ~<mb||ef> t2[mi,af]: X[(a,i),(b,e)] = t2[a,i,m,f] ovvv[m,f,b,e]
    A = np.ascontiguousarray(t2.transpose(2, 1, 0, 3)).reshape(nvir * nocc, nocc * nvir)
    B = np.ascontiguousarray(ge['ovvv'].transpose(0, 3, 1, 2)).reshape(nocc * nvir, -1)
    X = gemm(A, B)                                                           # [a,i,b,e]
    t3pack.pair_antisym_add(W, X, vm, (nocc * nvir * nvir, nvir, 1, nvir * nvir),
                            alpha=-1.0)
    del X, A, B
    # - sum_m f[m,e] t2[m,i,a,b]
    W -= gemm(np.asfortranarray(fov.T), t2[:, :, md, ne].reshape(nocc, -1)).reshape(
        nvir, nocc, vm.npair).transpose(2, 0, 1)
    return W


def dressed_vvoo_pp(eris, t1, Bv, om, vm):
    '''Fully T1-dressed <ab||ij> packed as [(i<j), (a<b)], except for the
    sum_ef ~<ab||ef> t1[i,e] t1[j,f] term, which is added in the ladder GEMM
    together with t2 (see ccsd_residual_packed):
      bra-dressed <ab||ij> + sum_e Bv[ab,e,j] t1[i,e] - (i <-> j).'''
    nocc, nvir = t1.shape
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
    mm, nn = om.pairs[:, 0], om.pairs[:, 1]
    dtype = np.result_type(t1, eris.oovv)
    # bra-dressed <ab||ij> = <ab||ij> - P(ab) t1[m,a] <mb||ij> + tau1 <mn||ij>
    vvoo = np.ascontiguousarray(get_block(eris, 'vvoo'), dtype=dtype)   # [a,b,i,j]
    G = np.array(vvoo[md, ne][:, mm, nn], order='C', copy=True)         # (Q, P)
    ovoo = np.ascontiguousarray(get_block(eris, 'ovoo')[:, :, mm, nn], dtype=dtype)
    L = gemm(np.asfortranarray(t1.T), ovoo.reshape(nocc, -1))           # [a, (b, P)]
    t3pack.pair_antisym_add(G.reshape(vm.npair, om.npair, 1), L, vm,
                            (nvir * om.npair, om.npair, 1, 0), alpha=-1.0)
    oooo = np.ascontiguousarray(np.asarray(eris.oooo)[mm, nn][:, mm, nn], dtype=dtype)
    gemm(np.asfortranarray(_tau1_pairs(t1, om, vm).T), oooo, G, beta=1.0)
    # ket dressing with Bv: Y[Q, i, j] = sum_e Bv[Q,e,j] t1[i,e]
    Y = np.matmul(t1, Bv)                                                # (Q, i, j)
    G += Y[:, mm, nn] - Y[:, nn, mm]
    return np.ascontiguousarray(G.T)                                     # (P, Q)


def ccsd_residual_packed(t1, t2, fdr, ge, Vd, vvoo_pp, Feff_vv, Feff_oo,
                         Weff_ovvo, om, vm):
    '''CCSD residuals in the T1-dressed formalism (CCSD with T1 = 0 and the
    dressed Hamiltonian; validated against the determinant-space oracle):
      R1 = f_ai + f_kc t_ikac + 1/2 g_akcd t_ikcd - 1/2 g_klic t_klac
      R2 = g_abij + 1/2 g_abcd (t + tau1)_ijcd + 1/2 W_klij t_klab
           + P(ab) F_bc t_ijac - P(ij) F_kj t_ikab + P(ij)P(ab) W_kbcj t_ikac
    with W_klij = g_klij + 1/2 g_klcd t_ijcd, W_kbcj = g_kbcj + 1/2 g_klcd t_jlbd,
    F = Feff (the same effective Fock blocks as in the T3 equation).
    R2 is returned packed as [(i<j), (a<b)].'''
    nocc, nvir = t1.shape
    mm, nn = om.pairs[:, 0], om.pairs[:, 1]
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
    o, v = slice(0, nocc), slice(nocc, None)
    t2pp = np.ascontiguousarray(t2[mm, nn][:, md, ne])                  # (P, Q)
    goovv_pp = np.ascontiguousarray(ge['oovv'][mm, nn][:, md, ne])
    # R1
    R1 = fdr[v, o].T + einsum('kc,ikac->ia', fdr[o, v], t2)
    R1 += einsum('akR,ikR->ia', ge['vovv'][:, :, md, ne], t2[:, :, md, ne])
    R1 -= einsum('Pic,Pac->ia', ge['ooov'][mm, nn], t2[mm, nn])
    # R2: driving + ladder (with the t1 t1 part of the vvoo dressing) + hh
    R2 = np.array(vvoo_pp, order='C', copy=True)
    gemm(t2pp + _tau1_pairs(t1, om, vm), Vd.T, R2, beta=1.0)
    Wkl = np.ascontiguousarray(ge['oooo'][mm, nn][:, mm, nn])           # (kl, ij)
    gemm(goovv_pp, t2pp.T, Wkl, beta=1.0)
    gemm(np.asfortranarray(Wkl.T), t2pp, R2, beta=1.0)
    # F terms and ring term in full (o,o,v,v) form, then packed
    X = einsum('bc,ijac->ijab', Feff_vv, t2)
    full = X - X.transpose(0, 1, 3, 2)
    X = einsum('kj,ikab->ijab', Feff_oo, t2)
    full -= X - X.transpose(1, 0, 2, 3)
    W2 = 0.5 * (ge['ovvo'] + Weff_ovvo)                                  # [k,b,c,j]
    A = np.ascontiguousarray(t2.transpose(0, 2, 1, 3)).reshape(nocc * nvir, -1)  # (ia,kc)
    B = np.ascontiguousarray(W2.transpose(0, 2, 3, 1)).reshape(nocc * nvir, -1)  # (kc,jb)
    X = gemm(A, B).reshape(nocc, nvir, nocc, nvir).transpose(0, 2, 1, 3)        # [i,j,a,b]
    X = X - X.transpose(1, 0, 2, 3)
    full += X - X.transpose(0, 1, 3, 2)
    R2 += full[mm, nn][:, md, ne]
    return R1, R2


def dressed_vvvv_pp(eris, t1, om, vm):
    '''Packed T1-dressed <ab||de> over a<b, d<e without the full v^4 array:
    ~<ab||de> = <ab||de> - Y[a,b,de] + Y[b,a,de] + Z[a,b,de],
    Y[x,y,Q] = sum_m t1[m,x] <my||Q>,  Z[a,b,Q] = sum_mn t1[m,a] t1[n,b] <mn||Q>.'''
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
    nocc, nvir = t1.shape
    cache = vvvv_packed(eris, vm)
    ovvv_Q = np.ascontiguousarray(np.asarray(eris.ovvv)[:, :, md, ne])  # (m, y, Q)
    out = np.array(cache, dtype=np.result_type(cache, ovvv_Q, t1), copy=True)
    # out[(x<y), Q] -= Y[x,y,Q];  out[(y<x), Q] += Y[x,y,Q], with
    # Y[x,y,Q] = sum_m t1[m,x] <my||Q>, built in batches of x (Y is v^2 n_Q)
    nb = max(1, int(2e9 // (nvir * vm.npair * out.itemsize)))
    for x0 in range(0, nvir, nb):
        x1 = min(x0 + nb, nvir)
        Y = gemm(np.ascontiguousarray(t1[:, x0:x1].T), ovvv_Q.reshape(nocc, -1))
        Y = Y.reshape(x1 - x0, nvir, vm.npair)
        t3pack.pair_antisym_batch_add(out, Y, vm, x0, alpha=-1.0)
        del Y
    # Z[(a<b), Q] = sum_{m<n} (t[m,a] t[n,b] - t[n,a] t[m,b]) <mn||Q>
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
    om, vm = t3pack.maps_for(eris)
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
            vvvv_pp = dressed_vvvv_pp(eris, t1, om, vm)
    foo, fov, fvv = fdr[:nocc, :nocc], fdr[:nocc, nocc:], fdr[nocc:, nocc:]
    goovv = ge['oovv']

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
    # T3 contributions to the T1 / T2 residuals (r2 packed over i<j, a<b)
    rB = 0.5 * (B0[:, md, ne] - B0[:, ne, md])
    rC = -0.5 * (C0[mm, nn] - C0[nn, mm])
    r2_t3 = rA + rB + rC

    # ---- effective intermediates for the T3 residual
    with timer('intermediates'):
        govoo = ge['ovoo']
        ovvv, ooov = ge['ovvv'], ge['ooov']
        t_ = timer('imd.Wvvvo'); t_.__enter__()
        Bv = bra_dressed_vvvo_pp(eris, t1, om, vm)
        Wv = wvvvo_packed(Bv, vvvv_pp, t1, t2, fov, ge, om, vm) + dWv   # [Q(ab), e, i]
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
        # CCSD residuals (T1-dressed) + T3 contributions, before the dressed
        # packed vvvv is turned into the T3 ladder operator in place
        t_ = timer('ccsd'); t_.__enter__()
        vvoo_pp = dressed_vvoo_pp(eris, t1, Bv, om, vm)
        R1, R2 = ccsd_residual_packed(t1, t2, fdr, ge, vvvv_pp, vvoo_pp, Feff_vv,
                                      Feff_oo, Weff_ovvo, om, vm)
        t1new = t1 + (R1 + r1) / eia
        t2new = t2 + _unpack_pairs(R2 + r2_t3, om, vm) / eijab
        del vvoo_pp, R1, R2
        t_.__exit__(None, None, None)
        t_ = timer('imd.Wpp'); t_.__enter__()
        Wpp = vvvv_pp                     # in place: W(ab,de) = ~<ab||de> + ...
        del vvvv_pp
        gemm(np.asfortranarray(t2mn[:, md, ne].T), goovv_pp, Wpp, beta=1.0)
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
            Adrv = np.concatenate((t2mn.transpose(0, 2, 1), Wo.transpose(2, 1, 0)),
                                  axis=2)                 # (P, c, e | m)
            Bdrv = np.concatenate((WvT2, t2P_flat), axis=0)  # (e | m, (i, Q))
            t3kern.pass2(T3p, om, vm, nb, WringT, Adrv, Bdrv, R)
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
            t3kern.update_into_r(T3p, R, eo, ev)      # R <- T3p + R / D
            T3new = R
        else:
            T3new = T3p + R / (eo[:, None] - ev[None, :])
    return t1new, t2new, T3new.ravel()


def ccsd_update_amps(opts, t1, t2, eris):
    '''One Jacobi step of CCSD in the T1-dressed formalism (the CCSD part of
    update_amps without T3).'''
    nocc, nvir = t1.shape
    om, vm = t3pack.maps_for(eris)
    mm, nn = om.pairs[:, 0], om.pairs[:, 1]
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
    mo_e = eris.mo_energy
    eia = mo_e[:nocc, None] - (mo_e[None, nocc:] + opts.level_shift)
    eijab = eia[:, None, :, None] + eia[None, :, None, :]
    with timer('t1_dress'):
        fdr = dress_fock(eris, t1)
        ge = {k: dress_block(eris, t1, k)
              for k in ('oooo', 'ovvo', 'oovv', 'ooov', 'vovv')}
        vvvv_pp = dressed_vvvv_pp(eris, t1, om, vm)
        Bv = bra_dressed_vvvo_pp(eris, t1, om, vm)
    with timer('ccsd'):
        goovv = ge['oovv']
        t2mn = t2[mm, nn]
        Feff_vv = fdr[nocc:, nocc:] - einsum('Paf,Pdf->ad', t2mn, goovv[mm, nn])
        Feff_oo = fdr[:nocc, :nocc] + einsum('mnP,inP->mi', goovv[:, :, md, ne],
                                              t2[:, :, md, ne])
        Weff_ovvo = ge['ovvo'] + einsum('mnde,inae->madi', goovv, t2)
        vvoo_pp = dressed_vvoo_pp(eris, t1, Bv, om, vm)
        R1, R2 = ccsd_residual_packed(t1, t2, fdr, ge, vvvv_pp, vvoo_pp, Feff_vv,
                                      Feff_oo, Weff_ovvo, om, vm)
    return t1 + R1 / eia, t2 + _unpack_pairs(R2, om, vm) / eijab


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
