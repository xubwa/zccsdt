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

from zccsdt.cc import ccsd, t3pack
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


def dress_block(eris, t1, key):
    '''T1-dressed block of <pq||rs>: a virtual bra index a picks up
    -sum_m t1[m,a] <m..||..>, an occupied ket index i picks up
    +sum_e t1[i,e] <..||..e>.  Other indices are unchanged.'''
    opts = []
    for pos, sp in enumerate(key):
        if pos < 2 and sp == 'v':
            opts.append((('v', None), ('o', -t1)))           # M[m, a]
        elif pos >= 2 and sp == 'o':
            opts.append((('o', None), ('v', t1.T)))          # M[e, i]
        else:
            opts.append(((sp, None),))
    out = None
    for combo in np.ndindex(*[len(x) for x in opts]):
        src = ''.join(opts[p][c][0] for p, c in enumerate(combo))
        b = get_block(eris, src)
        for p, c in enumerate(combo):
            M = opts[p][c][1]
            if M is not None:
                b = np.moveaxis(np.tensordot(b, M, axes=([p], [0])), -1, p)
        out = b if out is None else out + b
    return out


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
    T3p = np.asarray(T3p, dtype=dtype).reshape(om.ntri, vm.ntri)
    mo_e = eris.mo_energy
    eia = mo_e[:nocc, None] - (mo_e[None, nocc:] + opts.level_shift)
    eijab = eia[:, None, :, None] + eia[None, :, None, :]
    mm, nn = om.pairs[:, 0], om.pairs[:, 1]
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]

    with timer('t1_dress'):
        fdr = dress_fock(eris, t1)
        ge = {k: dress_block(eris, t1, k) for k in
              ('vvvo', 'ovoo', 'vvvv', 'oooo', 'ovvo', 'oovv', 'ovvv',
               'ooov', 'oovo', 'vovv')}
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
    dWv = np.zeros((vm.npair, nvir, nocc), dtype=dtype)  # [Q(ab), e, i]
    dWo = np.zeros((nocc, nvir, om.npair), dtype=dtype)  # [m, a, P(ij)]
    r1 = np.zeros((nocc, nvir), dtype=dtype)
    rA = np.zeros((om.npair, vm.npair), dtype=dtype)
    B0 = np.zeros((om.npair, nvir, nvir), dtype=dtype)
    C0 = np.zeros((nocc, nocc, vm.npair), dtype=dtype)
    nb = _pair_batch_size(nocc, nvir, np.dtype(dtype).itemsize, mem_mb)
    batches = _batches(om.npair, nb)
    with timer('pass1'):
        for pb in batches:
            with timer('gather_half'):
                T = t3pack.gather_half(T3p, om, vm, pb)     # (m, Pb, d, Q)
            # dW_vvvo[Q, e, i] += sum_{P,f} g[P,e,f] T[i,P,f,Q]
            dWv += einsum('Pef,iPfQ->Qei', goovv_mn[pb], T)
            # dW_vooo[m, a, P] = sum_{n,R} g[m,n,R] T[n,P,a,R]
            dWo[:, :, pb] = einsum('mnR,nPaR->maP', goovv_P, T)
            # T3 -> T1: r1[i,a] += sum_{P,Q} g[P,Q] T[i,P,a,Q]
            r1 += einsum('PQ,iPaQ->ia', goovv_pp[pb], T)
            # T3 -> T2
            rA[pb] = einsum('me,mPeQ->PQ', fov, T)
            B0[pb] = -2 * einsum('amR,mPbR->Pab', gvovv_Q, T)
            C0 -= 2 * einsum('Pej,iPeQ->ijQ', goovo_mn[pb], T)
            del T
    t1new = t1new + r1 / eia
    rB = 0.5 * (B0[:, md, ne] - B0[:, ne, md])
    rC = -0.5 * (C0[mm, nn] - C0[nn, mm])
    r2 = _unpack_pairs(rA + rB + rC, om, vm)
    t2new = t2new + r2 / eijab

    # ---- effective intermediates for the T3 residual
    with timer('intermediates'):
        gvvvo, govoo = ge['vvvo'], ge['ovoo']
        ovvv, ooov, oovo = ge['ovvv'], ge['ooov'], ge['oovo']
        Wvvvo = gvvvo.copy()
        Wvvvo += 0.5 * einsum('mnei,mnab->abei', oovo, t2)
        tmp = einsum('mbef,miaf->abei', ovvv, t2)
        Wvvvo -= (tmp - tmp.transpose(1, 0, 2, 3))
        Wvvvo -= einsum('me,miab->abei', fov, t2)
        Wv = Wvvvo[md, ne] + dWv                         # [Q(ab), e, i]
        Wvooo = govoo.copy()
        tmp2 = einsum('mnie,jnbe->mbij', ooov, t2)
        Wvooo -= (tmp2 - tmp2.transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2)
        Wvooo -= (0.5 * einsum('mbef,ijef->mbij', ovvv, t2)).transpose(0, 1, 3, 2)
        Wo = Wvooo[:, :, nn, mm] + dWo                   # [m, a, P(ij)]
        Weff_ovvo = ge['ovvo'] + einsum('mnde,inae->madi', goovv, t2)
        Wring = Weff_ovvo.transpose(3, 1, 0, 2).reshape(nocc * nvir, nocc * nvir)
        Feff_vv = fvv - einsum('Paf,Pdf->ad', t2mn, goovv_mn)
        Feff_oo = foo + einsum('mnP,inP->mi', goovv_P, t2P)
        Wpp = ge['vvvv'][md[:, None], ne[:, None], md, ne] + t2mn[:, md, ne].T @ goovv_pp
        # F_vv folded into the ladder: W(ab,de) += 1/2 A[F_ad delta_be]
        I = np.eye(nvir)
        FI = (einsum('ad,be->abde', Feff_vv, I) - einsum('bd,ae->abde', Feff_vv, I)
              - einsum('ae,bd->abde', Feff_vv, I) + einsum('be,ad->abde', Feff_vv, I))
        Wpp += 0.5 * FI[md[:, None], ne[:, None], md, ne]
        del FI
        Weff_oooo = ge['oooo'] + einsum('mnP,ijP->mnij', goovv_P, t2P)
        O6 = t3pack.occ_operator(om, lambda t: _occ_apply(t, Weff_oooo, Feff_oo))

    R = np.zeros((om.ntri, vm.ntri), dtype=dtype)

    # ---- pass 2: ring + drive, one antisymmetrizing reduction per batch
    with timer('pass2'):
        for pb in batches:
            with timer('gather_half'):
                T = t3pack.gather_half(T3p, om, vm, pb)     # (m, Pb, d, Q)
            with timer('ring_gemm'):
                Tm = T.transpose(0, 2, 1, 3).reshape(nocc * nvir, -1)
                Z = (Wring @ Tm).reshape(nocc, nvir, len(pb), vm.npair)  # [i,a,P,Q]
            del T, Tm
            with timer('drive'):
                # Xr1[i,P,Q,c] = sum_e Wv[Q,e,i] t2mn[P,e,c]
                Z -= einsum('Qei,Pec->icPQ', Wv, t2mn[pb])
                # Xr2[k,P,Q,a] = sum_m Wo[m,a,P] t2P[m,k,Q]
                Z -= einsum('maP,mkQ->kaPQ', Wo[:, :, pb], t2P)
            with timer('reduce'):
                Xv = t3pack.reduce_vir(Z.transpose(0, 2, 1, 3), vm)   # (i, P, A)
                del Z
                t3pack.scatter_occ(R, Xv, om, pb)
                del Xv

    # ---- pp ladder (+F_vv) over batches of occupied triples
    with timer('pp_ladder'):
        nbt = max(1, int(mem_mb * 1e6 // (3 * vm.npair * nvir * np.dtype(dtype).itemsize)))
        for rows in _batches(om.ntri, nbt):
            T = t3pack.gather_vir_half(T3p, vm, rows)            # (nb, Q, c)
            X = (Wpp @ T.transpose(1, 0, 2).reshape(vm.npair, -1))
            X = X.reshape(vm.npair, len(rows), nvir).transpose(1, 0, 2)
            R[rows] += t3pack.reduce_vir_pair_first(X, vm)
            del T, X

    # ---- hh ladder (+F_oo): occupied-space operator on packed T3
    with timer('hh_ladder'):
        R += O6.astype(dtype) @ T3p

    with timer('t3_update'):
        eo = mo_e[:nocc][om.triples].sum(axis=1)
        ev = (mo_e[nocc:] + opts.level_shift)[vm.triples].sum(axis=1)
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
