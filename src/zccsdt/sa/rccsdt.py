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


def ladder_pp(blocks, t3b, Ws, Wa, r3b, batch=64):
    '''r3b += sum over the three pair positions of every block of
    sum_de W_xy^de t^{..de..} (block-local), with the (d,e) symmetric and
    antisymmetric parts as two half-size GEMMs.'''
    nb, v = t3b.shape[0], t3b.shape[1]
    s0, s1 = np.triu_indices(v)
    a0, a1 = np.triu_indices(v, 1)
    for b0 in range(0, nb, batch):
        b1 = min(b0 + batch, nb)
        n = b1 - b0
        c = t3b[b0:b1]
        # views (pair positions 01, 02, 12) as [d, e, x]
        views = np.stack((c, c.transpose(0, 1, 3, 2), c.transpose(0, 2, 3, 1)), axis=1)  # (n,3,d,e,x)
        Tv = views.transpose(2, 3, 0, 1, 4).reshape(v, v, n * 3 * v)
        Ts = Tv[s0, s1] + Tv[s1, s0]
        Ts[s0 == s1] *= 0.5
        Ta = Tv[a0, a1] - Tv[a1, a0]
        S = gemm(Ws, Ts)
        A = gemm(Wa, Ta)
        L = np.zeros((v, v, n * 3 * v), dtype=r3b.dtype)
        L[s0, s1] = S
        L[s1, s0] = S
        L[a0, a1] += A
        L[a1, a0] -= A
        L = L.reshape(v, v, n, 3, v).transpose(2, 3, 0, 1, 4)        # (n, view, x, y, spect)
        r3b[b0:b1] += L[:, 0] + L[:, 1].transpose(0, 1, 3, 2) + L[:, 2].transpose(0, 3, 1, 2)


def ladder_hh(blocks, t3b, Wh, r3b):
    '''r3b += sum over pair positions of sum_mn W_mn^ij t_mnk^abc, with
    Wh[m,n,i,j] the hh ladder operator (F_oo folded in).'''
    no, v = blocks.nocc, t3b.shape[1]
    W2 = Wh.reshape(no * no, no * no)
    for k in range(no):
        T = np.empty((no, no, v, v, v), dtype=t3b.dtype)
        for m in range(no):
            for n in range(no):
                T[m, n] = blocks.cube(t3b, m, n, k)
        H = gemm(np.ascontiguousarray(W2.T), T.reshape(no * no, -1)).reshape(no, no, v, v, v)
        for i in range(no):
            for j in range(i, no):
                X = H[i, j].transpose(2, 0, 1)          # X_{k;ij}^{c;ab}
                for B, axes in blocks.uses_attached(k, i, j):
                    r3b[B] += X.transpose(axes)


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
    '''r3b += P[D + E] (module docstring); WvJ[a,b,e,i], WoJ[m,a,j,i].'''
    no, v = blocks.nocc, t2.shape[2]
    Wv2 = np.ascontiguousarray(WvJ.transpose(2, 0, 1, 3)).reshape(v, v * v * no)    # [e,(b,c,k)]
    t2f = t2.reshape(no, -1)                                                      # [m,(k,b,c)]
    for i in range(no):
        for j in range(no):
            D = gemm(np.ascontiguousarray(t2[i, j]), Wv2).reshape(v, v, v, no)      # [a,b,c,k]
            E = gemm(np.ascontiguousarray(WoJ[:, :, j, i].T), t2f).reshape(v, no, v, v)
            for k in range(no):
                uses = blocks.uses_ordered((i, j, k))
                if not uses:
                    continue
                G = D[:, :, :, k] - E[:, k]
                for B, axes in uses:
                    r3b[B] += G.transpose(axes)


# ------------------------------------------------------------------ driver

def hh_operator(Wj, Fo):
    '''hh ladder operator W_mn^ij = Wj[m,n,i,j] - 1/2 (Fo_mi d_nj + Fo_nj d_mi).'''
    no = Wj.shape[0]
    W = np.array(Wj, copy=True)
    for j in range(no):
        W[:, j, :, j] -= 0.5 * Fo
        W[j, :, j, :] -= 0.5 * Fo
    return W


def residuals(ints, blocks, t1, t2, t3b):
    '''(R1, R2, R3 blocks) for the spin-adapted CCSDT equations.  First
    version: T1 dressing on full V, intermediates and the CCSD part from the
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
