'''Spin-blocked (U / R form) building blocks of the CCSDT intermediates.

These mirror the functions of ccsdt_opt for integrals given as
eris_spin.SpinERIs: tensors with v^3 or v^4 elements stay spin-blocked
(SpinTensor or packed pair blocks), smaller ones are returned dense so that
the rest of ccsdt_opt can use them unchanged.
'''

import ctypes

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


def bare_packed(se):
    '''Bare ovvv packed over its virtual pair, [m, y, Q], and oovv packed over
    both pairs, [P, Q] (SpinTensors), cached on the integral object.'''
    c = se.__dict__.get('_bare_packed')
    if c is None:
        om, vm = t3pack.maps_for(se)
        Qs, Ps = pair_space('Q', vm), pair_space('P', om)
        c = (pack_pairs(se.g['ovvv'], 2, Qs),
             pack_pairs(pack_pairs(se.g['oovv'], 2, Qs), 0, Ps))
        se.__dict__['_bare_packed'] = c
    return c


def dressed_vvvv_blocks(se, t1, om, vm, t2pp=None, out=None):
    '''Packed T1-dressed <ab||de> as the aa / ab / bb pair blocks,
      ~<ab||de> = <ab||de> - sum_m t1[m,a] <mb||de> + sum_m t1[m,b] <ma||de>
                  + sum_{m<n} tau1[mn,ab] <mn||de>,
    plus sum_{m<n} t2[mn,ab] <mn||de> if t2pp (packed (P, Q)) is given, which
    is the T2 part of the T3 ladder operator.  One pass over each bare block
    (C kernel).  For RHF the beta-beta block is the alpha-alpha block (same
    object).  ``out``: optional list of reusable buffers for the blocks.'''
    nocc, nvir = t1.shape
    o, v = se.spaces['o'], se.spaces['v']
    G, oovv = bare_packed(se)
    oa, ob = o.groups[0], o.groups[1]
    va, vb = v.groups[0], v.groups[1]
    ta = np.ascontiguousarray(t1[oa[0]:oa[1], va[0]:va[1]])   # alpha block
    tb = np.ascontiguousarray(t1[ob[0]:ob[1], vb[0]:vb[1]])   # beta block
    from zccsdt.cc.ccsdt_opt import _tau1_pairs
    C = _tau1_pairs(t1, om, vm)
    if t2pp is not None:
        C = C + t2pp
    lib = t3pack._lib
    use_c = t3pack.USE_C and hasattr(lib, 'zc_vvvv_dress_d')
    res = [None, None, None]
    for h in (0, 1, 2):
        if h == 2 and se.restricted:
            res[2] = res[0]
            continue
        lo, hi = vm.blocks[h]
        plo, phi = om.blocks[h]
        bare = se.vvvv_blocks[h]
        dt = np.result_type(bare, t1, C)
        blk = out[h] if out is not None else None
        if blk is None or blk.shape != bare.shape or blk.dtype != dt:
            blk = np.empty(bare.shape, dtype=dt)
            if out is not None:
                out[h] = blk
        res[h] = blk
        if h in (0, 2):
            s_ = h // 2
            tt = ta if s_ == 0 else tb
            off = va[0] if s_ == 0 else vb[0]
            GA = GB = G.blocks.get((s_, s_, h))           # [m, y, Q]
            tA, tB = -tt, tt
            pa, pb = vm.pa_c[lo:hi] - off, vm.pb_c[lo:hi] - off
        else:
            GA = G.blocks.get((0, 1, 1))                  # [m_a, b_b, Q]
            GB = G.blocks.get((1, 0, 1))                  # [m_b, a_a, Q]
            tA, tB = -ta, tb
            pa, pb = vm.pa_c[lo:hi] - va[0], vm.pb_c[lo:hi] - vb[0]
        H = oovv.blocks.get((h, h))
        if hi == lo:
            continue
        if use_c:
            def arr(x):
                return None if x is None else np.ascontiguousarray(x, dtype=dt)
            tA, tB, GA, GB, H = arr(tA), arr(tB), arr(GA), arr(GB), arr(H)
            nmA = 0 if GA is None or tA.size == 0 else GA.shape[0]
            nmB = 0 if GB is None or tB.size == 0 else GB.shape[0]
            nP = 0 if H is None or phi == plo else H.shape[0]
            CT = np.ascontiguousarray(C[plo:phi, lo:hi].T, dtype=dt)
            ptr = lambda x: None if x is None else t3pack._p(x)   # noqa: E731
            I64 = t3pack._I64
            getattr(lib, 'zc_vvvv_dress_' + t3pack._sfx(dt))(
                t3pack._p(blk), t3pack._p(np.ascontiguousarray(bare, dtype=dt)),
                I64(hi - lo), I64(hi - lo), t3pack._p(pa), t3pack._p(pb),
                ptr(tA), I64(tA.shape[1]), ptr(GA), I64(nmA),
                I64(0 if GA is None else GA.shape[1]),
                ptr(tB), I64(tB.shape[1]), ptr(GB), I64(nmB),
                I64(0 if GB is None else GB.shape[1]),
                t3pack._p(CT), ptr(H), I64(nP))
            continue
        blk[...] = bare
        for tc, Gc, x, y in ((tA, GA, pa, pb), (tB, GB, pb, pa)):
            if Gc is None or tc.size == 0:
                continue
            # blk[q] += sum_m tc[m, x_q] Gc[m, y_q]
            blk += np.einsum('mq,mqc->qc', tc[:, x], Gc[:, y])
        if H is not None and phi > plo:
            blk += C[plo:phi, lo:hi].T @ H
    return res


def bra_dressed_vvvo(se, t1, om, vm):
    '''Bra-dressed <ab||ei> packed over a<b, dense (npair_v, nvir, nocc)
    (see ccsdt_opt.bra_dressed_vvvo_pp), written block by block:
      B[(ab), e, i] = <ab||ei> - L[a,b,e,i] + L[b,a,e,i] + sum_{m<n} tau1[mn,ab] <mn||ei>,
      <ab||ei> = -<ie||ab>^*,  L[a,b,e,i] = sum_m t1[m,a] <mb||ei>.
    Only the Ms-allowed (e, i) blocks of each pair group are touched.'''
    nocc, nvir = t1.shape
    o, v = se.spaces['o'], se.spaces['v']
    Ps = pair_space('P', om)
    dtype = np.result_type(t1, se.fock)
    B = np.zeros((vm.npair, nvir, nocc), dtype=dtype)
    G = bare_packed(se)[0]                                  # [i, e, Q] blocks
    for (si, se_, h), blk in G.blocks.items():
        lo, hi = vm.blocks[h]
        (e0, e1), (i0, i1) = v.groups[se_], o.groups[si]
        blk = np.asarray(blk, dtype=dtype)
        if np.iscomplexobj(blk):
            blk = blk.conj()
        # loop order (e, i, Q): contiguous reads of blk
        t3pack.axpby(B[lo:hi, e0:e1, i0:i1].transpose(1, 2, 0), blk.transpose(1, 0, 2), -1.0)
    ovvo = se.g['ovvo']                                     # [m, b, e, i] blocks
    oa, ob = o.groups
    va, vb = v.groups
    ta = np.ascontiguousarray(t1[oa[0]:oa[1], va[0]:va[1]], dtype=dtype)
    tb = np.ascontiguousarray(t1[ob[0]:ob[1], vb[0]:vb[1]], dtype=dtype)
    nva, nvb = va[1] - va[0], vb[1] - vb[0]
    # same-spin pair blocks (all four indices of spin s):
    # Y[x,y,e,i] = sum_m t[m,x] <my||ei>,  B[(x<y)] -= Y[x,y] - Y[y,x]
    for h, s, tt in ((0, 0, ta), (2, 1, tb)):
        lo, hi = vm.blocks[h]
        blk = ovvo.blocks.get((s, s, s, s))
        if hi == lo or tt.size == 0 or blk is None:
            continue
        (x0, x1), (i0, i1) = v.groups[s], o.groups[s]
        nx, ni = x1 - x0, i1 - i0
        Y = gemm(np.ascontiguousarray(tt.T), np.asarray(blk, dtype=dtype).reshape(blk.shape[0], -1))
        tmp = np.zeros((hi - lo, nx * ni), dtype=dtype)
        _pair_scatter_same_spin(tmp, Y.reshape(nx, nx, -1), nx, vm.pa_c[lo:hi] - x0,
                                vm.pb_c[lo:hi] - x0, -1.0)
        t3pack.axpby(B[lo:hi, x0:x1, i0:i1].transpose(0, 2, 1),
                     tmp.reshape(hi - lo, nx, ni).transpose(0, 2, 1))
    # alpha-beta pair block (a alpha, b beta; rows a-major), (e, i) of opposite spins
    lo, hi = vm.blocks[1]
    if hi > lo:
        B1 = B[lo:hi].reshape(nva, nvb, nvir, nocc)
        for se_, si in ((0, 1), (1, 0)):
            (e0, e1), (i0, i1) = v.groups[se_], o.groups[si]
            ne, ni = e1 - e0, i1 - i0
            if ne == 0 or ni == 0:
                continue
            out = B1[:, :, e0:e1, i0:i1].transpose(0, 1, 3, 2)     # (a, b, i, e)
            blk = ovvo.blocks.get((0, 1, se_, si))
            if ta.size and blk is not None:   # -L[a,b] = -sum_{m alpha} t[m,a] <m b||e i>
                X = gemm(np.ascontiguousarray(ta.T),
                         np.asarray(blk, dtype=dtype).reshape(blk.shape[0], -1))
                t3pack.axpby(out, X.reshape(nva, nvb, ne, ni).transpose(0, 1, 3, 2), -1.0)
            blk = ovvo.blocks.get((1, 0, se_, si))
            if tb.size and blk is not None:   # +L[b,a] = sum_{m beta} t[m,b] <m a||e i>
                X = gemm(np.ascontiguousarray(tb.T),
                         np.asarray(blk, dtype=dtype).reshape(blk.shape[0], -1))
                t3pack.axpby(out, X.reshape(nvb, nva, ne, ni).transpose(1, 0, 3, 2))
    # + sum_{m<n} tau1[mn, ab] <mn||ei>, per pair group
    from zccsdt.cc.ccsdt_opt import _tau1_pairs
    tau = _tau1_pairs(t1, om, vm)
    oovo = pack_pairs(get_block(se, 'oovo'), 0, Ps)        # [P, e, i] blocks
    for (h, se_, si), blk in oovo.blocks.items():
        lo, hi = vm.blocks[h]
        plo, phi = om.blocks[h]
        (e0, e1), (i0, i1) = v.groups[se_], o.groups[si]
        if hi > lo and phi > plo:
            X = gemm(np.ascontiguousarray(tau[plo:phi, lo:hi].T),
                     np.asarray(blk, dtype=dtype).reshape(phi - plo, -1))
            t3pack.axpby(B[lo:hi, e0:e1, i0:i1].transpose(0, 2, 1),
                         X.reshape(hi - lo, e1 - e0, i1 - i0).transpose(0, 2, 1))
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


def t2_ovvv_add(W, t2, ovvv_d, om, vm, alpha=-1.0):
    '''W[(a<b), e, i] += alpha (Y[a,b,e,i] - Y[b,a,e,i]),
    Y[a,b,e,i] = sum_{mf} t2[m,i,a,f] ovvv[m,b,e,f], without the dense Y:
    spin conservation leaves the blocks with s_i - s_a = s_b - s_e = s_f - s_m,
    computed as four GEMMs Y[(x,i), (y,e)] = A[(x,i), (m,f)] B[(m,f), (y,e)]
    (rows (x,i) = alpha-alpha and beta-beta stacked where they share B).
    W: dense (npair_v, nvir, nocc).'''
    o = (0, int(np.sum(om.spins == 0)), om.n)
    v = (0, int(np.sum(vm.spins == 0)), vm.n)
    rng_o = ((o[0], o[1]), (o[1], o[2]))
    rng_v = ((v[0], v[1]), (v[1], v[2]))
    dt = W.dtype
    nv, no = W.shape[1], W.shape[2]

    def bop(sm, sy, se_, sf):
        return ovvv_d.blocks.get((sm, sy, se_, sf))

    def aop(sm, si, sx, sf):
        (m0, m1), (i0, i1) = rng_o[sm], rng_o[si]
        (x0, x1), (f0, f1) = rng_v[sx], rng_v[sf]
        a = t2[m0:m1, i0:i1, x0:x1, f0:f1].transpose(2, 1, 0, 3)
        return np.ascontiguousarray(a, dtype=dt).reshape((x1 - x0) * (i1 - i0), -1)

    def ygemm(rows, sy, se_):
        '''Y for the row spin pairs ``rows`` [(sx, si), ...] stacked, column
        spins (sy, se_); returns {(sx, si): Y block (nx, ni, ny, ne)}.'''
        d = sy - se_
        terms = [(sm, sm + d) for sm in (0, 1) if 0 <= sm + d <= 1]
        terms = [(sm, sf) for sm, sf in terms if bop(sm, sy, se_, sf) is not None]
        rows = [(sx, si) for sx, si in rows if si - sx == d]
        if not terms or not rows:
            return {}
        ny, ne = rng_v[sy][1] - rng_v[sy][0], rng_v[se_][1] - rng_v[se_][0]
        # B[(m,f), (y,e)] = ovvv[m,y,e,f], the blocks of all terms stacked;
        # copy loop order (m, y, f, e) keeps each [m, y] slab in cache
        kdim = [bop(sm, sy, se_, sf).shape[0] * bop(sm, sy, se_, sf).shape[3]
                for sm, sf in terms]
        B = np.empty((sum(kdim), ny * ne), dtype=dt)
        k0 = 0
        for (sm, sf), k in zip(terms, kdim):
            blk = np.asarray(bop(sm, sy, se_, sf), dtype=dt)
            nm, nf = blk.shape[0], blk.shape[3]
            t3pack.axpby(B[k0:k0 + k].reshape(nm, nf, ny, ne).transpose(0, 2, 1, 3),
                         blk.transpose(0, 1, 3, 2), 1.0, 0.0)
            k0 += k
        A = np.block([[aop(sm, si, sx, sf) for sm, sf in terms] for sx, si in rows])
        Y = gemm(np.ascontiguousarray(A), B)
        out, r0 = {}, 0
        for sx, si in rows:
            nx, ni = rng_v[sx][1] - rng_v[sx][0], rng_o[si][1] - rng_o[si][0]
            out[(sx, si)] = Y[r0:r0 + nx * ni].reshape(nx, ni, ny, ne)
            r0 += nx * ni
        return out

    both = [(0, 0), (1, 1)]
    Ya = ygemm(both, 0, 0)          # (y, e) alpha alpha: Y(0000), Y(1001)
    Yb = ygemm(both, 1, 1)          # (y, e) beta beta:   Y(0110), Y(1111)
    Yc = ygemm([(0, 1)], 1, 0)      # Y(0101): x a, i b, y b, e a
    Yd = ygemm([(1, 0)], 0, 1)      # Y(1010): x b, i a, y a, e b
    # same-spin pair blocks: e, i of the pair spin
    for h, s, Yd_ in ((0, 0, Ya), (2, 1, Yb)):
        lo, hi = vm.blocks[h]
        Y = Yd_.get((s, s))
        if hi == lo or Y is None:
            continue
        (x0, x1), (i0, i1) = rng_v[s], rng_o[s]
        nx, ni = x1 - x0, i1 - i0
        Y = np.ascontiguousarray(Y)                       # [x, i, y, e]
        tmp = np.zeros((hi - lo, nx, ni), dtype=dt)       # [Q, e, i]
        t3pack.pair_antisym_add(tmp, Y, vm, (ni * nx * nx, nx, 1, nx * nx), alpha,
                                pairs=(vm.pa_c[lo:hi] - x0, vm.pb_c[lo:hi] - x0))
        t3pack.axpby(W[lo:hi, x0:x1, i0:i1].transpose(0, 2, 1), tmp.transpose(0, 2, 1))
    # alpha-beta pair block (a alpha, b beta), rows a-major
    lo, hi = vm.blocks[1]
    if hi > lo:
        (a0, a1), (b0, b1) = rng_v
        W1 = W[lo:hi].reshape(a1 - a0, b1 - b0, nv, no)
        for (se_, si), Yab, Yba in (((1, 0), Yb.get((0, 0)), Yd.get((1, 0))),
                                    ((0, 1), Yc.get((0, 1)), Ya.get((1, 1)))):
            (e0, e1), (i0, i1) = rng_v[se_], rng_o[si]
            if e1 == e0 or i1 == i0:
                continue
            Wv = W1[:, :, e0:e1, i0:i1].transpose(0, 1, 3, 2)      # (a, b, i, e)
            if Yab is not None:      # Y[a, b, e, i] from [a, i, b, e]
                t3pack.axpby(Wv, Yab.transpose(0, 2, 1, 3), alpha)
            if Yba is not None:      # - Y[b, a, e, i] from [b, i, a, e]
                t3pack.axpby(Wv, Yba.transpose(2, 0, 1, 3), -alpha)
    return W


def _vvvv_t1_c(lib, W, Vd, ta, tb, rows, nvs, nos):
    '''C kernels of vvvv_t1_blocks: one pass over each pair block.'''
    dt = W.dtype
    sfx = t3pack._sfx(dt)
    I64 = t3pack._I64
    nv, nocc = W.shape[1], W.shape[2]
    ldq, lde = nv * nocc, nocc
    nva, nvb = nvs
    noa, nob = nos
    ta = np.ascontiguousarray(ta, dtype=dt)
    tb = np.ascontiguousarray(tb, dtype=dt)

    def ptr(q, e, i):
        return ctypes.c_void_p(W.ctypes.data + dt.itemsize * (q * ldq + e * lde + i))

    lo, hi = rows[1]
    if hi > lo and noa + nob and Vd[1] is not None:
        V = np.ascontiguousarray(Vd[1], dtype=dt)
        getattr(lib, 'zc_vt1_ab_' + sfx)(
            ptr(lo, 0, noa), ptr(lo, nva, 0), t3pack._p(V), t3pack._p(ta),
            t3pack._p(tb), I64(hi - lo), I64(nva), I64(nvb), I64(noa), I64(nob),
            I64(ldq), I64(lde))
    for h, v0, n, o0, no, tt in ((0, 0, nva, 0, noa, ta), (2, nva, nvb, noa, nob, tb)):
        lo, hi = rows[h]
        if hi == lo or n < 2 or no == 0 or Vd[h] is None:
            continue
        V = np.ascontiguousarray(Vd[h], dtype=dt)
        getattr(lib, 'zc_vt1_same_' + sfx)(
            ptr(lo, v0, o0), t3pack._p(V), t3pack._p(tt), I64(hi - lo), I64(n),
            I64(no), I64(ldq), I64(lde))


def vvvv_t1_blocks(W, Vd, t1, om, vm, batch=512, rows=None):
    '''W[Q, e, i] += sum_f V[Q, (e,f)] t1[i, f] for V antisymmetric in (e, f)
    with columns in the spin-grouped pairs of ``vm``: Vd are the aa / ab / bb
    column blocks, whose rows are W[rows[h][0]:rows[h][1]] (default: the pair
    blocks of vm, i.e. V = packed dressed vvvv).  t1 conserves spin.
    W: dense (nrow, nvir, nocc).'''
    rows = vm.blocks if rows is None else rows
    noa = int(np.sum(om.spins == 0))
    nva = int(np.sum(vm.spins == 0))
    nocc, nv = om.n, vm.n
    va, vb = (0, nva), (nva, nv)
    oa, ob = (0, noa), (noa, nocc)
    ta = np.ascontiguousarray(t1[oa[0]:oa[1], va[0]:va[1]])
    tb = np.ascontiguousarray(t1[ob[0]:ob[1], vb[0]:vb[1]])
    nvb = nv - nva
    lib = t3pack._lib
    if (t3pack.USE_C and hasattr(lib, 'zc_vt1_same_d') and W.flags.c_contiguous
            and W.shape[1:] == (nv, nocc)):
        _vvvv_t1_c(lib, W, Vd, ta, tb, rows, (nva, nvb), (noa, nocc - noa))
        return W
    # alpha-beta pair block: columns (e alpha, f beta), e-major
    lo, hi = rows[1]
    if hi > lo and nva and nvb and Vd[1] is not None:
        V1 = Vd[1].reshape(hi - lo, nva, nvb)
        if tb.size:      # e alpha, i beta: sum_{f beta} V[Q, e, f] t[i, f]
            X = gemm(np.ascontiguousarray(V1).reshape(-1, nvb), np.asfortranarray(tb.T))
            W[lo:hi, va[0]:va[1], ob[0]:ob[1]] += X.reshape(hi - lo, nva, -1)
        if ta.size:      # e beta, i alpha: -sum_{f alpha} t[i, f] V[Q, f, e]
            X = np.matmul(ta, V1)                       # (Q, i, e), V1 not copied
            W[lo:hi, vb[0]:vb[1], oa[0]:oa[1]] -= X.transpose(0, 2, 1)
    # same-spin blocks: unpack V rows to antisymmetric (e, f), then GEMM
    for h, (v0, v1), (o0, o1), tt in ((0, va, oa, ta), (2, vb, ob, tb)):
        lo, hi = rows[h]
        n = v1 - v0
        if hi == lo or n < 2 or tt.size == 0 or Vd[h] is None:
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
