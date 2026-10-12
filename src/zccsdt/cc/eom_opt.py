'''EOM-EE-CCSDT (and EOM-EE-CCSD) on packed amplitudes: the sigma vector

    sigma = <mu| [Hbar, R] |0> = (Jacobian of the amplitude equations) R

for spin orbitals or spinors, real or complex (docs/eom.md; the dense
reference is zccsdt.cc.eom_ref).

The amplitude step of zccsdt.cc.ccsdt_opt is linear in the dressed integrals
and in T3 and at most quadratic in T2, so its derivative along R = (r1, r2,
R3) is the same passes over packed arrays with other operands:

    sigma_3 = residual(R3; W)  +  residual(T3; W')
              the T3 passes with R3 in the place of T3 and the operators W
              of the ground state (ring, pp ladder with F_vv, hh ladder with
              F_oo), their drive terms being W'_vvvo x t2 and W'_vooo x t2;
              and once more with T3 and the derivatives W' of the operators,
              their drive terms being W_vvvo x r2 and W_vooo x r2.
    W'      = W[H'; t2] + (the part of W that is linear in t2, T3)[H~; r2, R3],
              H' = [H~, R1]: the dressed integrals with one index transformed
              by r1 (eom_ref.dressed_derivative).

ccsdt_opt.update_amps(capture=) hands over the dressed integrals and the
operators of the converged ground state (prepare()); nothing of the T3
machinery is written again here.  The parts of rank one and two are GEMMs
on the layouts of ccsdt_opt.ccsd_residual_packed, with pass 1 over R3 (and
over T3 for the three terms of sigma_2 that have H' and T3).

A vector is (r1[i,a], r2[(i<j),(a<b)], R3) with the pairs of the maps of the
integrals and R3 flat in the layout of T3 of the engine - the packed block
(i<j<k, a<b<c), or the blocks of the symmetry labels, for a totally
symmetric R.

Cost of a sigma vector: two T3 residuals and two passes 1, about two
amplitude iterations less the T1 dressing.
'''

from dataclasses import dataclass

import numpy as np
from pyscf import lib

from zccsdt.blas import gemm
from zccsdt.cc import ccsdt_opt, t3kern, t3pack
from zccsdt.cc.ccsdt_opt import transposed
from zccsdt.timing import timer

einsum = lib.einsum


@dataclass
class Ground:
    '''What sigma() needs of the ground state (prepare).'''
    eris: object
    om: object
    vm: object
    eng: object
    dtype: object
    t1: np.ndarray
    t2: np.ndarray
    T3: np.ndarray              # flat, layout of the engine; None: EOM-CCSD
    c: dict                     # the capture of ccsdt_opt.update_amps and what is made of it
    change: float = 0.0         # |Jacobi step - amplitudes| at these amplitudes
    model: str = 'cc'           # 'cc': CCSD / CCSDT; 'cc2', 'cc3': the Jacobian of that model (docs/ccn.md)

    @property
    def with_t3(self):
        return self.T3 is not None

    @property
    def fly(self):
        '''CC3 without stored triples: the vectors have two ranks, and the
        triples part of the Jacobian is folded in at a frequency omega.'''
        return self.model == 'cc3' and self.T3 is None

    def sizes(self, shift=None):
        '''Lengths of the three parts of a vector (``shift``: of a vector
        with that label shift, whose rank-3 part is in the blocks of
        t3shift.ShiftLayout).'''
        om, vm = self.om, self.vm
        n3 = self.eng.size if shift is None else self.shifter().size(shift)
        return [om.n * vm.n, om.npair * vm.npair, n3 if self.with_t3 else 0]

    def shifter(self):
        '''The passes of the engine for arrays with a label shift.'''
        from zccsdt.cc import eom
        return eom._shifter(self.eng)

    def masks(self, shift):
        '''([i,a], [(i<j),(a<b)]) True where an element has the label shift
        (classes of the maps; spin blocks: the number of beta indices).'''
        key = ('_mask', int(shift))
        if key not in self.c:
            om, vm = self.om, self.vm
            self.c[key] = (np.asarray(vm.spins)[None, :] - np.asarray(om.spins)[:, None] == shift,
                           np.asarray(vm.pair_grp)[None, :] - np.asarray(om.pair_grp)[:, None] == shift)
        return self.c[key]


def _dtau(t1, r1, om, vm):
    '''The derivative of tau1[(m<n), a, b] = t1[m,a] t1[n,b] - t1[n,a] t1[m,b] along r1.'''
    mm, nn = om.pairs[:, 0], om.pairs[:, 1]
    x = r1[mm][:, :, None] * t1[nn][:, None, :] - r1[nn][:, :, None] * t1[mm][:, None, :]
    return x + t1[mm][:, :, None] * r1[nn][:, None, :] - t1[nn][:, :, None] * r1[mm][:, None, :]


def prepare(eris, t1, t2, T3=None, opts=None, mem_mb=2000.0, model='cc', fly=False):
    '''The Ground of converged amplitudes: t1, t2 dense, T3 flat in the
    layout of the engine of ``eris`` (None: EOM-CCSD on CCSD amplitudes).
    One amplitude step with its operands kept.  ``model``: 'cc2' (T3 None)
    or 'cc3' for the Jacobian of that model at its amplitudes.  ``fly``
    (model 'cc3', T3 None): no T3 is stored; sigma(omega=) then folds the
    triples in at the frequency omega (zccsdt.cc.cc3fly).'''
    from zccsdt.cc import solver
    opts = opts or solver.Options()
    model = {'ccsd': 'cc', 'ccsdt': 'cc'}.get(model, model)
    if model not in ('cc', 'cc2', 'cc3') or (model == 'cc2' and T3 is not None) \
            or (model == 'cc3' and (T3 is None) != bool(fly)) or (fly and model != 'cc3'):
        raise ValueError('model %r with%s T3' % (model, 'out' if T3 is None else ''))
    om, vm = t3pack.maps_for(eris)
    dtype = np.dtype(np.result_type(t1, t2, eris.oovv, eris.fock, *([] if T3 is None else [T3])))
    eng = ccsdt_opt.engine_for(eris, dtype, mem_mb)
    t1 = np.ascontiguousarray(t1, dtype=dtype)
    t2 = np.ascontiguousarray(t2, dtype=dtype)
    if fly:
        T = None
    else:
        T = np.zeros(eng.size, dtype=dtype) if T3 is None else np.ascontiguousarray(np.asarray(T3, dtype=dtype).ravel())
    c = {}
    with timer('eom.prepare'):
        n1, n2, n3 = ccsdt_opt.update_amps(opts, t1, t2, T, eris, mem_mb=mem_mb, plan=ccsdt_opt.MemoryPlan(), capture=c,
                                           model='cc3' if model == 'cc3' else 'ccsdt')
        if model == 'cc2':
            from zccsdt.cc import ccn
            n1, n2 = ccn.cc2_update_amps(opts, t1, t2, eris)
        change = float(np.linalg.norm(n1 - t1) + np.linalg.norm(n2 - t2))
        if T3 is not None:
            change += 6.0 * float(solver._diff_norm(n3, T))
        del n1, n2, n3
        nocc, nvir = t1.shape
        mm, nn = om.pairs[:, 0], om.pairs[:, 1]
        ge = c['ge']
        ge['ovvv'] = c.pop('ovvv')
        c['t2mn'] = np.ascontiguousarray(t2[mm, nn])
        c['t2P'] = t3pack.pack_last2(t2, vm)
        c['t2pp'] = t3pack.pack_last2(c['t2mn'], vm)
        c['A'] = transposed(t2, (0, 2, 1, 3)).reshape(nocc * nvir, nocc * nvir)
        c['Gr'] = transposed(np.asarray(ge['oovv'], dtype=dtype), (0, 2, 1, 3)).reshape(nocc * nvir, nocc * nvir)
        # W_ovvo in the layouts [(m d), (i a)] (from the operator of the T3
        # ring term) and [m, (b, e, i)]; the operator of the CCSD ring term
        W4 = c['WringT'].reshape(nvir, nocc, nvir, nocc)                # [a, i, d, m]
        c['Wmbei'] = transposed(W4, (3, 0, 2, 1)).reshape(nocc, -1)
        g_l = transposed(ge['ovvo'], (0, 2, 3, 1), dtype).reshape(nocc * nvir, nocc * nvir)
        c['ovvo_l'] = g_l
        c['Br'] = 0.5 * (g_l + transposed(W4, (3, 2, 1, 0)).reshape(nocc * nvir, nocc * nvir))
        c['K'] = einsum('mnQ,ijQ->mnij', c['goovv_P'], c['t2P'])
        Wkl = np.array(np.asarray(c['Weff_oooo'])[mm, nn][:, mm, nn], dtype=dtype, order='C')
        Wkl += ccsdt_opt.occ_pair_one_body(np.asarray(c['Feff_oo']), om)
        c['Wkl'] = Wkl
        c['ovoo_P'] = np.ascontiguousarray(np.asarray(ge['ovoo'])[:, :, mm, nn], dtype=dtype)
        c['ovvv_Q'] = t3pack.pack_last2(np.ascontiguousarray(ge['ovvv'], dtype=dtype), vm)   # [m, y, Q]
        if T3 is None:
            for k in ('G_Pd', 'Wv', 'Wo'):
                c.pop(k, None)
        if fly:
            for k in ('WringT', 'O6'):
                c.pop(k, None)
    return Ground(eris, om, vm, eng, dtype, t1, t2, None if T3 is None else T, c, change, model)


def _pass1(g, T, G_Pd, G_mQ, gpp, fov, shifts=None, **kw):
    '''Pass 1 over a packed rank-3 array: (its rank-1 part [i,a], its rank-2
    part [P,Q], the part of W_vvvo [(e,i),Q], the part of W_vooo [m,a,P]).
    ``shifts`` = (of T, of the integrals): in the blocks of these shifts.'''
    om, vm = g.om, g.vm
    nocc, nvir = om.n, vm.n
    mm, nn = om.pairs[:, 0], om.pairs[:, 1]
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
    if shifts is None:
        Xacc, Yall, r1, rA = g.eng.pass1(T, G_Pd, G_mQ, gpp, fov)
    else:
        Xacc, Yall, r1, rA = g.shifter().pass1(T, shifts[0], G_Pd, G_mQ, gpp, fov, shifts[1], **kw)
    C0 = -2 * Xacc[nvir:].transpose(1, 0, 2)
    B0 = -2 * Yall[:, :, nocc:].transpose(0, 2, 1)
    r2 = rA + 0.5 * (B0[:, md, ne] - B0[:, ne, md]) - 0.5 * (C0[mm, nn] - C0[nn, mm])
    return r1.T, r2, Xacc[:nvir].reshape(nvir * nocc, vm.npair), Yall[:, :, :nocc].transpose(2, 1, 0)


def sigma(g, r1, r2p, R3=None, out=None, sd=False, shift=None, omega=None):
    '''(sigma_1[i,a], sigma_2[(i<j),(a<b)], sigma_3 flat) for the vector
    (r1, r2p, R3) - see the module; ``out``: a flat array for sigma_3.
    ``sd``: the singles-doubles block only - no R3 and none of the terms of
    T3 in sigma_2 (the guess of zccsdt.cc.eom).

    ``shift``: the label shift of the vector (classes of the maps; the
    integrals with their labels in use): r1, r2p have their elements where
    l_a - l_i (l_a + l_b - l_i - l_j) = shift, R3 and sigma_3 are flat in
    the layout g.shifter().layout(shift).  None: a totally symmetric
    vector in the layout of T3 (the same as shift = 0, by the engine
    itself).

    ``omega``: for a Ground of CC3 without stored triples (g.fly) - the
    folded sigma vector A_PP r + A_PQ (omega - A_QQ)^-1 A_QP r of the vector
    (r1, r2p); the third return value is None.'''
    s = 0 if shift is None else int(shift)
    sh = None if shift is None else g.shifter()
    om, vm, eng, dt, c = g.om, g.vm, g.eng, g.dtype, g.c
    nocc, nvir = om.n, vm.n
    mm, nn = om.pairs[:, 0], om.pairs[:, 1]
    t1, t2, ge, fdr = g.t1, g.t2, c['ge'], c['fdr']
    foo, fov, fvv = fdr[:nocc, :nocc], fdr[:nocc, nocc:], fdr[nocc:, nocc:]
    oooo, ooov, oovo, oovv, ovvo, ovvv = (np.asarray(ge[k]) for k in ('oooo', 'ooov', 'oovo', 'oovv', 'ovvo', 'ovvv'))
    goovv_P, gpp, t2pp, t2P, t2mn, A = c['goovv_P'], c['goovv_pp'], c['t2pp'], c['t2P'], c['t2mn'], c['A']
    with3 = g.with_t3 and not sd
    eng.io = None
    r1 = np.ascontiguousarray(r1, dtype=dt)
    r2p = np.ascontiguousarray(r2p, dtype=dt).reshape(om.npair, vm.npair)
    t_ = timer('eom.small'); t_.__enter__()
    r2 = t3pack.unpack_pairs(r2p, om, vm)
    r2mn = np.ascontiguousarray(r2[mm, nn])
    r2P = t3pack.pack_last2(r2, vm)
    Ar = transposed(r2, (0, 2, 1, 3)).reshape(nocc * nvir, nocc * nvir)

    # ---- H' = [H~, R1]: Fock matrix and the small blocks
    dfov = einsum('mnef,nf->me', oovv, r1)
    dfoo = fov @ r1.T + einsum('mnie,ne->mi', ooov, r1)
    dfvv = -r1.T @ fov - einsum('naef,nf->ae', ovvv, r1)
    dfvo = -r1.T @ foo + fvv @ r1.T + einsum('nafi,nf->ai', ovvo, r1)
    d_oovo = einsum('mnef,if->mnei', oovv, r1)
    d_ooov = -d_oovo.transpose(0, 1, 3, 2)
    d_oooo = einsum('mnej,ie->mnij', oovo, r1) + einsum('mnie,je->mnij', ooov, r1)
    d_ovvo = einsum('made,ie->madi', ovvv, r1) - einsum('na,mndi->madi', r1, oovo)
    x = einsum('maey,xe->maxy', ovvo, r1)
    d_ovoo = x - x.transpose(0, 1, 3, 2) - einsum('na,mnxy->maxy', r1, oooo)

    # ---- the intermediates that are linear in r2
    Xr = gemm(c['Gr'], Ar.T)                                    # [(m d), (i a)]
    X4 = Xr.reshape(nocc, nvir, nocc, nvir)
    dFoo = dfoo + 0.5 * np.einsum('mdid->mi', X4)
    dFvv = dfvv - 0.5 * np.einsum('mdma->ad', X4)
    d_l = transposed(d_ovvo, (0, 2, 3, 1), dt).reshape(nocc * nvir, nocc * nvir)
    dBr = d_l + 0.5 * Xr
    dWoooo = d_oooo + einsum('mnP,ijP->mnij', goovv_P, r2P)

    # ---- sigma_1
    s1 = dfvo.T + (A @ np.ascontiguousarray(dfov).ravel() + Ar @ np.ascontiguousarray(fov, dtype=dt).ravel()).reshape(nocc, nvir)
    M = gemm(goovv_P.reshape(nocc, -1), t2P.reshape(nocc, -1).T)        # [n, i]
    s1 -= M.T @ r1
    s1 += gemm(r2P.reshape(nocc, -1), c['G_mQ'][:, nocc:])
    s1 -= 0.5 * (einsum('klic,klac->ia', d_ooov, t2) + einsum('klic,klac->ia', ooov, r2))

    # ---- sigma_2: [H', T-independent] and the terms of the CCSD equations
    G = np.zeros((vm.npair, om.npair), dtype=dt)
    L = gemm(np.asfortranarray(r1.T), c['ovoo_P'].reshape(nocc, -1))            # [a, (b, P)]
    t3pack.pair_antisym_add(G.reshape(vm.npair, om.npair, 1), L, vm, (nvir * om.npair, om.npair, 1, 0), alpha=-1.0)
    t3pack.vvoo_ket_add(G, c['Bv'], r1, om)
    s2 = np.ascontiguousarray(G.T)
    del G, L
    dtau = _dtau(t1, r1, om, vm)                                # [P, a, b]
    xp = r2p + t3pack.pack_last2(dtau, vm)                      # (r2 + dtau1)[P, Q]
    cc2 = g.model == 'cc2'
    if cc2:
        # CC2: sigma_2 = <ab||ij>' + [F, R2] with the Fock matrix as it is.  The ladder of the
        # dressed <ab||de>~ acts on dtau1 alone (the ket part of <ab||ij>'): the pp operator
        # below is applied to dtau1, and its T2 and F_vv parts are taken out again
        xp = t3pack.pack_last2(dtau, vm)
        f0 = np.asarray(g.eris.fock)
        Y = gemm((-0.5 * dtau).reshape(-1, nvir), np.asarray(c['Feff_vv'], dtype=dt).T)
        gemm(r2mn.reshape(-1, nvir), np.ascontiguousarray(f0[nocc:, nocc:].T, dtype=dt), Y, beta=1.0)
        t3pack.add_asym_last2(s2, Y.reshape(-1, nvir, nvir), vm)
        del Y, dtau
        gemm(np.asfortranarray(ccsdt_opt.occ_pair_one_body(np.asarray(f0[:nocc, :nocc], dtype=dt), om).T), r2p, s2, beta=1.0)
    else:
        # P(ab) (F'_bc t_ijac + F_bc r_ijac); half of the second is in the ladder operator
        z = 0.5 * (r2mn - dtau)
        Y = gemm(z.reshape(-1, nvir), np.asarray(c['Feff_vv'], dtype=dt).T)
        gemm(t2mn.reshape(-1, nvir), np.ascontiguousarray(dFvv.T), Y, beta=1.0)
        t3pack.add_asym_last2(s2, Y.reshape(-1, nvir, nvir), vm)
        del Y, z, dtau
        # 1/2 W_klij r_klab + 1/2 W'_klij t_klab, with F_oo and F'_oo
        dWkl = np.array(dWoooo[mm, nn][:, mm, nn], dtype=dt, order='C')
        dWkl += ccsdt_opt.occ_pair_one_body(dFoo, om)
        gemm(np.asfortranarray(c['Wkl'].T), r2p, s2, beta=1.0)
        gemm(np.asfortranarray(dWkl.T), t2pp, s2, beta=1.0)
        # ring: t x W' + r x W
        X2 = gemm(A, dBr)
        gemm(Ar, c['Br'], X2, beta=1.0)
        t3pack.add_asym_pack4(s2, X2.reshape(nocc, nvir, nocc, nvir), om, vm, axes=(0, 2, 1, 3))
        del X2
    del dBr
    t_.__exit__(None, None, None)

    # ---- pp: <ab||de>' = -P(ab) r1[m,a] <mb||de>~ packed, its ladder with t2 (the one
    # with tau1 is in <ab||ij>');
    # then the derivative of the T3 ladder operator in the same array
    t_ = timer('eom.pp'); t_.__enter__()
    Wpp = c['Wpp']
    dWpp = []
    G_Q = c['ovvv_Q']
    # the derivative of the operator by its column block h; with a shift its rows are the
    # pairs of block h + s (lo2:hi2)
    for h, (lo, hi) in enumerate(vm.blocks):
        nq = hi - lo
        lo2, hi2 = vm.blocks[h + s] if 0 <= h + s < len(vm.blocks) else (0, 0)
        nq2 = hi2 - lo2
        blk = np.zeros((nq2, nq), dtype=dt)
        if nq:
            if nq2 and not cc2:
                Gh = np.ascontiguousarray(G_Q if nq == vm.npair else G_Q[:, :, lo:hi]).reshape(nocc, -1)
                nb = max(1, int(1e9 // (nvir * nq * blk.itemsize)))
                for xb in range(0, nvir, nb):
                    xe = min(xb + nb, nvir)
                    Y = gemm(np.ascontiguousarray(r1[:, xb:xe].T), Gh).reshape(xe - xb, nvir, nq)
                    t3pack.pair_antisym_batch_add(blk, Y, vm, xb, alpha=-1.0, block=h + s)
                    del Y
                t3pack.axpby(s2[:, lo2:hi2], gemm(blk, np.ascontiguousarray(t2pp[:, lo:hi].T)).T)
            # 1/2 <ab||cd>~ (r2 + dtau1): the ladder operator of T3 less its T2 part
            t3pack.axpby(s2[:, lo:hi], gemm(Wpp[h], np.ascontiguousarray(xp[:, lo:hi].T)).T)
            s2[:, lo:hi] -= (xp[:, lo:hi] @ gpp[:, lo:hi].T) @ t2pp[:, lo:hi]
            if with3 and nq2 and g.model != 'cc3':
                gemm(np.asfortranarray(r2p[:, lo2:hi2].T), np.ascontiguousarray(gpp[:, lo:hi]), blk, beta=1.0)
                if s == 0:
                    ccsdt_opt.add_one_body_pairs(blk, dFvv, vm, 0.5, block=h)
                else:
                    from zccsdt.cc import t3shift
                    blk += t3shift.one_body_pairs(dFvv, vm, h + s, h, 0.5, dt)
        dWpp.append(blk)
    t_.__exit__(None, None, None)
    # With labels in use the rank-1 and rank-2 parts are masked to the shift (0 for the
    # totally symmetric vectors): they are dense arrays, the integrals are symmetric to the
    # rounding of the SCF only, and what the rounding puts into other shifts would grow in a
    # Davidson iteration - the states of other shifts lie lower (HBr, 26 + 120: after 28
    # iterations of the singles-doubles stage the roots of the shifts +-2 and +-4 came up).
    labelled = om.spins is not None
    if labelled:
        m1, m2 = g.masks(s)
    fly = g.fly and not sd
    if fly and (omega is None or sh is not None):
        raise ValueError('CC3 without stored triples: sigma(omega=) for totally symmetric vectors')
    if not with3 and not fly:
        if labelled:
            s1 *= m1
            s2 *= m2
        return s1, s2, None

    # ---- pass 1 over R3 with the integrals of the ground state, over T3 with H'
    cc3 = g.model == 'cc3'
    Xv = None
    if not fly:
        with timer('eom.pass1'):
            a1, a2, Xv, dWo = _pass1(g, R3, c['G_Pd'], c['G_mQ'], gpp, fov, None if sh is None else (s, 0))
            s1 += a1
            s2 += a2
            G1 = np.zeros_like(c['G_Pd'])
            G1[nvir:] = d_oovo[mm, nn].transpose(2, 0, 1)
            G2 = np.zeros((nocc * vm.npair, nocc + nvir), dtype=dt)
            gemm(np.asfortranarray(goovv_P.reshape(nocc, -1).T), r1, G2[:, nocc:], alpha=-1.0)   # <am||Q>' = -r1[n,a] <nm||Q>
            if sh is None:
                s2 += _pass1(g, g.T3, G1, G2, np.zeros_like(gpp), dfov)[1]
            else:       # (of H' only <mn||ei>', <am||ef>' and f'[m,e] are not zero)
                s2 += _pass1(g, g.T3, G1, G2, None, dfov, (0, s), xe=False, yj=False)[1]
            del G1, G2

    # ---- the derivatives of W_vvvo and W_vooo
    with timer('eom.Wv'):
        dWv = np.zeros((vm.npair, nvir, nocc), dtype=dt)
        if not cc3:
            t3pack.add_transposed(dWv.reshape(vm.npair, -1), np.ascontiguousarray(Xv))
        del Xv
        # -P(ab) r1[m,a] W_ovvo[m,b,e,i]: <ab||ei>' and the t2 x <mb||ef>' term
        # (CC3: the dressed <mb||ei>~ alone - the drive operator is <ab||ei>')
        L = gemm(np.asfortranarray(r1.T), np.ascontiguousarray(ovvo, dtype=dt).reshape(nocc, -1) if cc3 else c['Wmbei'])
        t3pack.pair_antisym_add(dWv, L, vm, (nvir * nvir * nocc, nvir * nocc, nocc, 1), alpha=-1.0)
        del L
        # sum_f <ab||ef>~ r1[i,f], from the ladder operator: its T2 part gives
        # the term 1/2 <mn||ei>' t2[mn,ab]; its F_vv part is taken out
        F = np.asarray(c['Feff_vv'], dtype=dt)
        Fr = F @ r1.T                                           # [a, i]
        md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
        q = np.arange(vm.npair)
        for h, (lo, hi) in enumerate(vm.blocks):
            if hi > lo:
                t3pack.pair_t1_contract(dWv[lo:hi], Wpp[h], r1, vm, block=h)
        dWv -= 0.5 * (F[md][:, :, None] * r1.T[ne][:, None, :] - F[ne][:, :, None] * r1.T[md][:, None, :])
        dWv[q, ne] += 0.5 * Fr[md]
        dWv[q, md] -= 0.5 * Fr[ne]
        if cc3:     # the T2 part of the ladder operator on r1 is taken out too
            gemm(np.asfortranarray(t2pp.T), np.ascontiguousarray(d_oovo[mm, nn], dtype=dt).reshape(om.npair, -1),
                 dWv.reshape(vm.npair, -1), alpha=-1.0, beta=1.0)
        else:
            zero = np.zeros((nocc, nocc, nvir, nocc), dtype=dt)
            ccsdt_opt.wvvvo_packed(dWv, None, t1, t2, dfov, {'oovo': zero}, om, vm, xterm=lambda W: None, inplace=True)
            ccsdt_opt.wvvvo_packed(dWv, None, t1, r2, fov, ge, om, vm, inplace=True)
    if fly:
        # The triples folded in at the frequency omega (docs/ccn.md, section 4):
        #   r3 = (drive[t2; <ab||ei>', <ma||ij>'] + drive[r2; <ab||ei>~, <ma||ij>~]) / (omega - (e_abc - e_ijk))
        # contracted with the integrals of H~ into sigma_1 and sigma_2, and the triples of
        # the ground state contracted with those of H' = [H~, R1] into sigma_2; both made
        # per batch of occupied triples and discarded
        fl, d0 = c['fly'], c['fly_drive']
        with timer('eom.fly'):
            with timer('cc3fly.operands'):
                d1 = fl.drive_operands(t2, t2P, dWv, d_ovoo)
                d2 = (transposed(r2, (1, 0, 3, 2), dt), d0[1], d0[2], np.ascontiguousarray(r2P, dtype=dt))
            del dWv
            a1, a2 = fl.run([d1, d2], c['fly_contract'], omega=omega)
            s1 += a1
            s2 += a2
            del d1, d2
            with timer('cc3fly.operands'):
                G2 = gemm(np.asfortranarray(goovv_P.reshape(nocc, -1).T), r1, alpha=-1.0)      # <am||Q>' [(m, Q), a]
                contr = fl.contraction_operands(G2, d_oovo, np.zeros_like(goovv_P), dfov)
                del G2
            s2 += fl.run([d0], contr)[1]
        if labelled:
            s1 *= m1
            s2 *= m2
        return s1, s2, None
    if cc3:
        # sigma_3 = [F, R3] + drive(t2; <ab||ei>', <ma||ij>') + drive(r2; <ab||ei>~, <ma||ij>~):
        # T3 is not read
        dWo = np.ascontiguousarray(d_ovoo[:, :, nn, mm])
        Wpp3 = O63 = None           # (the Fock operator of the semicanonical spinors: its diagonal)
        if sh is not None:
            with timer('eom.r3'):
                S3 = sh.residual(R3, s, None, t2mn, dWv, dWo, t2P, Wpp3, O63, sW=0, sa=0, out=out)
                ccsdt_opt.fock_diag_add(g.eris, dt, R3, S3.reshape(-1), shift_layout=sh.layout(s))
            with timer('eom.t3'):
                sh.residual(g.T3, 0, None, r2mn, c['Wv'], c['Wo'], r2P, None, None, sW=s, sa=s, out=S3, clear=False)
        else:
            with timer('eom.r3'):
                S3 = eng.residual(R3, None, t2mn, dWv, dWo, t2P, Wpp3, O63, out=out)
                ccsdt_opt.fock_diag_add(g.eris, dt, np.asarray(R3).reshape(-1), S3.reshape(-1))
            with timer('eom.t3'):
                eng.residual(g.T3, None, r2mn, c['Wv'], c['Wo'], r2P, None, None, out=S3, clear=False)
        if labelled:
            s1 *= m1
            s2 *= m2
        return s1, s2, S3.reshape(-1)
    with timer('eom.small'):
        def t2_ooov(o_, a_):        # tmp2[m,b,i,j] = sum_ne o[m,n,i,e] t[j,n,b,e]
            Oo = np.ascontiguousarray(np.asarray(o_).transpose(0, 2, 1, 3), dtype=dt)
            return gemm(Oo.reshape(nocc * nocc, nocc * nvir), a_.T).reshape(nocc, nocc, nocc, nvir).transpose(0, 3, 1, 2)
        tmp2 = t2_ooov(d_ooov, A) + t2_ooov(ooov, Ar)
        dW = d_ovoo + tmp2 - tmp2.transpose(0, 1, 3, 2)
        x = gemm(np.ascontiguousarray(ovvv, dtype=dt).reshape(nocc * nvir, -1), r2.reshape(nocc * nocc, -1).T)
        dW -= 0.5 * x.reshape(nocc, nvir, nocc, nocc).transpose(0, 1, 3, 2)
        dW += einsum('nb,mnij->mbij', r1, c['K']).transpose(0, 1, 3, 2)
        dWo = dW[:, :, nn, mm] + dWo
        dW4 = (d_l + Xr).reshape(nocc, nvir, nocc, nvir)        # W'_ovvo [m, d, i, a]
        dWringT = transposed(dW4, (3, 2, 1, 0)).reshape(nvir * nocc, nvir * nocc)
        dO6 = t3kern.OccOperator(om, dWoooo, dFoo, dt)

    # ---- sigma_3
    if sh is not None:
        # R3 (shift s) with the operators of the ground state, the drive terms t2 x W' (shift s);
        # T3 with the derivatives of the operators (shift s), the drive terms r2 (shift s) x W
        with timer('eom.r3'):
            S3 = sh.residual(R3, s, c['WringT'], t2mn, dWv, dWo, t2P, Wpp, c['O6'], sW=0, sa=0, out=out)
        del dWv
        with timer('eom.t3'):
            sh.residual(g.T3, 0, dWringT, r2mn, c['Wv'], c['Wo'], r2P, dWpp, (dWoooo, dFoo), sW=s, sa=s,
                        out=S3, clear=False)
        s1 *= m1
        s2 *= m2
        return s1, s2, S3.reshape(-1)
    with timer('eom.r3'):
        S3 = eng.residual(R3, c['WringT'], t2mn, dWv, dWo, t2P, Wpp, c['O6'], out=out)
    del dWv
    with timer('eom.t3'):
        eng.residual(g.T3, dWringT, r2mn, c['Wv'], c['Wo'], r2P, dWpp, dO6, out=S3, clear=False)
    if labelled:
        s1 *= m1
        s2 *= m2
    return s1, s2, S3.reshape(-1)
