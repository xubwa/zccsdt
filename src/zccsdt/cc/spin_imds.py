'''Spin-blocked (U / R form) building blocks of the CCSDT intermediates.

These mirror the functions of ccsdt_opt for integrals given as
eris_spin.SpinERIs: tensors with v^3 or v^4 elements stay spin-blocked
(SpinTensor or packed pair blocks), smaller ones are returned dense so that
the rest of ccsdt_opt can use them unchanged.
'''

import numpy as np

from zccsdt.blas import gemm
from zccsdt.cc import t3pack
from zccsdt.cc.spintensor import SpinTensor, einsum, pack_pairs, pair_space

STORED = ('oooo', 'ooov', 'oovv', 'ovov', 'ovvo', 'ovvv')


def get_block(se, key):
    '''Any <pq||rs> block of SpinERIs as a SpinTensor (antisymmetry of the
    pairs and hermiticity <pq||rs> = <rs||pq>^*).'''
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
                b = se.g[s]
                if swap_bra:
                    b = -b.transpose(1, 0, 2, 3)
                if swap_ket:
                    b = -b.transpose(0, 1, 3, 2)
                if conj:
                    b = b.transpose(2, 3, 0, 1).conj()
                return b
    raise KeyError(key)


def _t1_tensors(se, t1):
    o, v = se.spaces['o'], se.spaces['v']
    bra = SpinTensor.from_dense(-t1, (o, v), sig=(1, -1))      # M[m, a] = -t1
    ket = SpinTensor.from_dense(t1.T, (v, o), sig=(1, -1))     # M[e, i] = t1[i, e]
    return bra, ket


def dress_block(se, t1, key, mats=None):
    '''T1-dressed block (same transformation as ccsdt_opt.dress_block) as a
    SpinTensor.'''
    bra, ket = mats if mats is not None else _t1_tensors(se, t1)
    alt = {}
    for p, sp in enumerate(key):
        if p < 2 and sp == 'v':
            alt[p] = ('o', bra)
        elif p >= 2 and sp == 'o':
            alt[p] = ('v', ket)
    choices = [(key[p], alt[p][0]) if p in alt else (key[p],) for p in range(4)]
    state = {}
    for c in np.ndindex(*[len(x) for x in choices]):
        k = ''.join(choices[p][i] for p, i in enumerate(c))
        state[k] = get_block(se, k)
    letters = 'pqrs'
    for p in sorted(alt, reverse=True):
        src, M = alt[p]
        new = {}
        sub_in = letters
        sub_out = letters[:p] + 'x' + letters[p + 1:]
        spec = '%s,%sx->%s' % (sub_in, letters[p], sub_out)
        for k, blk in state.items():
            if k[p] != key[p]:
                continue
            k_alt = k[:p] + src + k[p + 1:]
            new[k] = blk + einsum(spec, state[k_alt], M)
        state = new
    return state[key]


def dress_fock(se, t1):
    '''T1-dressed Fock matrix (dense), as ccsdt_opt.dress_fock.'''
    nocc = t1.shape[0]
    fock = np.asarray(se.fock)
    n = fock.shape[0]
    o, v = se.spaces['o'], se.spaces['v']
    T1 = SpinTensor.from_dense(t1, (o, v), sig=(-1, 1))
    sl = {'o': slice(0, nocc), 'v': slice(nocc, n)}
    fp = fock.astype(np.result_type(fock, t1))
    for r in 'ov':
        for s in 'ov':
            g = get_block(se, r + 'o' + s + 'v')
            fp[sl[r], sl[s]] += einsum('risa,ia->rs', g, T1).to_dense()
    x = np.eye(n, dtype=fp.dtype)
    x[nocc:, :nocc] -= t1.T
    y = np.eye(n, dtype=fp.dtype)
    y[:nocc, nocc:] += t1
    return x @ fp @ y.T


def _pair_scatter_same_spin(out, Y, nloc, pa, pb, alpha):
    '''out[q] += alpha (Y[pa_q, pb_q] - Y[pb_q, pa_q]) for local pair lists.'''
    if t3pack.USE_C and hasattr(t3pack._lib, 'zc_pair_antisym_batch_add_d') \
            and out.flags.c_contiguous:
        Y = np.ascontiguousarray(Y, dtype=out.dtype)
        al = np.array(alpha, dtype=out.dtype)
        getattr(t3pack._lib, 'zc_pair_antisym_batch_add_' + t3pack._sfx(out.dtype))(
            t3pack._p(out), t3pack._p(Y), t3pack._I64(nloc), t3pack._I64(len(pa)),
            t3pack._p(np.ascontiguousarray(pa)), t3pack._p(np.ascontiguousarray(pb)),
            t3pack._I64(out.shape[1]), t3pack._I64(0), t3pack._I64(nloc), t3pack._p(al))
        return out
    out += alpha * (Y[pa, pb] - Y[pb, pa])
    return out


def dressed_vvvv_blocks(se, t1, om, vm):
    '''Packed T1-dressed <ab||de> as the aa / ab / bb pair blocks, with the
    alpha-beta Y terms written by GEMMs straight into the block.  For RHF the
    beta-beta block is the alpha-alpha block (same object).'''
    nocc, nvir = t1.shape
    o, v = se.spaces['o'], se.spaces['v']
    Qs = pair_space('Q', vm)
    Ps = pair_space('P', om)
    G = pack_pairs(se.g['ovvv'], 2, Qs)                    # (m, y, Q) blocks
    oovv = pack_pairs(pack_pairs(se.g['oovv'], 2, Qs), 0, Ps)  # (P, Q) blocks
    oa, ob = o.groups[0], o.groups[1]
    va, vb = v.groups[0], v.groups[1]
    nva = va[1] - va[0]
    ta = np.ascontiguousarray(t1[oa[0]:oa[1], va[0]:va[1]])   # alpha block
    tb = np.ascontiguousarray(t1[ob[0]:ob[1], vb[0]:vb[1]])   # beta block
    from zccsdt.cc.ccsdt_opt import _tau1_pairs
    tau = _tau1_pairs(t1, om, vm)
    out = [None, None, None]
    for h in (0, 1, 2):
        lo, hi = vm.blocks[h]
        if h == 2 and se.restricted:
            out[2] = out[0]
            continue
        blk = np.array(se.vvvv_blocks[h], dtype=np.result_type(se.vvvv_blocks[h], t1),
                       order='C', copy=True)
        if hi > lo:
            if h in (0, 2):
                s = h // 2
                tt = ta if s == 0 else tb
                Gs = G.blocks.get((s, s, h))
                if Gs is not None and tt.size:
                    nq = hi - lo
                    off = va[0] if s == 0 else vb[0]
                    nloc = Gs.shape[1]
                    Y = gemm(np.ascontiguousarray(tt.T), Gs.reshape(Gs.shape[0], -1))
                    Y = Y.reshape(nloc, nloc, nq)
                    pa = vm.pa_c[lo:hi] - off
                    pb = vm.pb_c[lo:hi] - off
                    _pair_scatter_same_spin(blk, Y, nloc, pa, pb, -1.0)
            else:
                nq = hi - lo
                G01 = G.blocks.get((0, 1, 1))                  # [m_a, b_b, Q]
                G10 = G.blocks.get((1, 0, 1))                  # [m_b, a_a, Q]
                blk3 = blk.reshape(nva, -1)                    # rows a, cols (b, Q)
                if G01 is not None and ta.size:
                    # blk[(a,b),Q] -= sum_m t1[m,a] <m b||Q>
                    gemm(np.ascontiguousarray(ta.T), G01.reshape(G01.shape[0], -1), blk3,
                         alpha=-1.0, beta=1.0)
                if G10 is not None and tb.size:
                    # blk[(a,b),Q] += sum_m t1[m,b] <m a||Q>
                    tbT = np.ascontiguousarray(tb.T)
                    for a in range(nva):
                        Ga = np.ascontiguousarray(G10[:, a, :])
                        gemm(tbT, Ga, blk[a * (tb.shape[1]):(a + 1) * tb.shape[1]],
                             beta=1.0)
            # + sum_{m<n} tau1[mn, ab] <mn||de> (only occupied pairs of group h)
            ob_h = oovv.blocks.get((h, h))
            if ob_h is not None:
                plo, phi = om.blocks[h]
                gemm(np.ascontiguousarray(tau[plo:phi, lo:hi].T), ob_h, blk, beta=1.0)
        out[h] = blk
    return out


def bra_dressed_vvvo(se, t1, om, vm):
    '''Bra-dressed <ab||ei> packed over a<b, dense (npair_v, nvir, nocc)
    (see ccsdt_opt.bra_dressed_vvvo_pp), written block by block:
      B[(ab), e, i] = <ab||ei> - L[a,b,e,i] + L[b,a,e,i] + sum_{m<n} tau1[mn,ab] <mn||ei>,
      <ab||ei> = -<ie||ab>^*,  L[a,b,e,i] = sum_m t1[m,a] <mb||ei>.'''
    nocc, nvir = t1.shape
    o, v = se.spaces['o'], se.spaces['v']
    Qs, Ps = pair_space('Q', vm), pair_space('P', om)
    dtype = np.result_type(t1, se.fock)
    B = np.zeros((vm.npair, nvir, nocc), dtype=dtype)
    G = pack_pairs(se.g['ovvv'], 2, Qs)                     # [i, e, Q] blocks
    for (si, se_, h), blk in G.blocks.items():
        lo, hi = vm.blocks[h]
        (e0, e1), (i0, i1) = v.groups[se_], o.groups[si]
        B[lo:hi, e0:e1, i0:i1] -= blk.transpose(2, 1, 0).conj()
    ovvo = se.ovvo                                          # dense [m, b, e, i]
    oa, ob = o.groups
    va, vb = v.groups
    ta = np.ascontiguousarray(t1[oa[0]:oa[1], va[0]:va[1]])
    tb = np.ascontiguousarray(t1[ob[0]:ob[1], vb[0]:vb[1]])
    nva, nvb = va[1] - va[0], vb[1] - vb[0]
    # same-spin pair blocks: Y[x,y,e,i] = sum_m t[m,x] <my||ei>, then
    # B[(x<y)] -= Y[x,y] - Y[y,x]
    for h, tt, (m0, m1), (x0, x1) in ((0, ta, oa, va), (2, tb, ob, vb)):
        lo, hi = vm.blocks[h]
        nx = x1 - x0
        if hi == lo or tt.size == 0:
            continue
        Y = gemm(np.ascontiguousarray(tt.T),
                 np.ascontiguousarray(ovvo[m0:m1, x0:x1]).reshape(m1 - m0, -1))
        Y = Y.reshape(nx, nx * nvir * nocc)
        Bh = B[lo:hi].reshape(hi - lo, -1)
        _pair_scatter_same_spin(Bh, Y.reshape(nx, nx, -1), nx, vm.pa_c[lo:hi] - x0,
                                vm.pb_c[lo:hi] - x0, -1.0)
    # alpha-beta block (a alpha, b beta), rows a-major
    lo, hi = vm.blocks[1]
    if hi > lo:
        B1 = B[lo:hi].reshape(nva, nvb * nvir * nocc)
        if ta.size:      # -L[a,b] = -sum_{m alpha} t[m,a] <m b||e i>
            gemm(np.ascontiguousarray(ta.T),
                 np.ascontiguousarray(ovvo[oa[0]:oa[1], vb[0]:vb[1]]).reshape(oa[1] - oa[0], -1),
                 B1, alpha=-1.0, beta=1.0)
        if tb.size:      # +L[b,a] = sum_{m beta} t[m,b] <m a||e i>
            tbT = np.ascontiguousarray(tb.T)
            for a in range(nva):
                Ga = np.ascontiguousarray(ovvo[ob[0]:ob[1], va[0] + a]).reshape(ob[1] - ob[0], -1)
                gemm(tbT, Ga, B[lo + a * nvb:lo + (a + 1) * nvb].reshape(nvb, -1), beta=1.0)
    # + sum_{m<n} tau1[mn, ab] <mn||ei>, per pair group
    from zccsdt.cc.ccsdt_opt import _tau1_pairs
    tau = _tau1_pairs(t1, om, vm)
    oovo = pack_pairs(get_block(se, 'oovo'), 0, Ps)        # [P, e, i] blocks
    for (h, se_, si), blk in oovo.blocks.items():
        lo, hi = vm.blocks[h]
        plo, phi = om.blocks[h]
        (e0, e1), (i0, i1) = v.groups[se_], o.groups[si]
        if hi > lo and phi > plo:
            B[lo:hi, e0:e1, i0:i1] += np.einsum('PQ,Pei->Qei', tau[plo:phi, lo:hi], blk)
    return B


def t2_ovvv_term(se, t2, ovvv_d, vm):
    '''sum_{m,f} t2[m,i,a,f] ~<mb||ef>, antisymmetrized in (a,b) and packed:
    returns X[(a<b), e, i] (dense) with X = Y[a,b] - Y[b,a],
    Y[a,b,e,i] = sum_{mf} t2[m,i,a,f] ovvv[m,b,e,f].'''
    o, v = se.spaces['o'], se.spaces['v']
    Qs = pair_space('Q', vm)
    T2 = SpinTensor.from_dense(t2, (o, o, v, v), sig=(-1, -1, 1, 1))
    Y = einsum('miaf,mbef->abei', T2, ovvv_d)
    return np.ascontiguousarray(pack_pairs(Y - Y.transpose(1, 0, 2, 3), 0, Qs).to_dense())


def vvvv_t1_blocks(W, Vd, t1, om, vm, batch=512):
    '''W[Q, e, i] += sum_f ~<ab||ef> t1[i, f] for spin-grouped pairs, as GEMMs
    per spin block (t1 conserves spin; Vd are the aa / ab / bb pair blocks).
    W: dense (npair_v, nvir, nocc).'''
    noa = int(np.sum(om.spins == 0))
    nva = int(np.sum(vm.spins == 0))
    nocc, nv = om.n, vm.n
    va, vb = (0, nva), (nva, nv)
    oa, ob = (0, noa), (noa, nocc)
    ta = np.ascontiguousarray(t1[oa[0]:oa[1], va[0]:va[1]])
    tb = np.ascontiguousarray(t1[ob[0]:ob[1], vb[0]:vb[1]])
    nvb = nv - nva
    # alpha-beta pair block: columns (e alpha, f beta), e-major
    lo, hi = vm.blocks[1]
    if hi > lo and nva and nvb:
        V1 = Vd[1].reshape(hi - lo, nva, nvb)
        if tb.size:      # e alpha, i beta: sum_{f beta} V[Q, e, f] t[i, f]
            X = gemm(np.ascontiguousarray(V1).reshape(-1, nvb), np.asfortranarray(tb.T))
            W[lo:hi, va[0]:va[1], ob[0]:ob[1]] += X.reshape(hi - lo, nva, -1)
        if ta.size:      # e beta, i alpha: -sum_{f alpha} V[Q, f, e] t[i, f]
            X = np.matmul(V1.transpose(0, 2, 1), ta.T)
            W[lo:hi, vb[0]:vb[1], oa[0]:oa[1]] -= X
    # same-spin blocks: unpack V rows to antisymmetric (e, f), then GEMM
    for h, (v0, v1), (o0, o1), tt in ((0, va, oa, ta), (2, vb, ob, tb)):
        lo, hi = vm.blocks[h]
        n = v1 - v0
        if hi == lo or n < 2 or tt.size == 0:
            continue
        iu, ju = np.triu_indices(n, 1)
        ttT = np.asfortranarray(tt.T)
        for q0 in range(lo, hi, batch):
            q1 = min(q0 + batch, hi)
            full = np.zeros((q1 - q0, n, n), dtype=np.result_type(Vd[h], tt))
            full[:, iu, ju] = Vd[h][q0 - lo:q1 - lo]
            full[:, ju, iu] = -Vd[h][q0 - lo:q1 - lo]
            X = gemm(full.reshape(-1, n), ttT).reshape(q1 - q0, n, -1)
            W[q0:q1, v0:v1, o0:o1] += X
    return W
