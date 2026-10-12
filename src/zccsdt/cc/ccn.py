'''The approximate iterative coupled-cluster models CC2, CC3 and CC4 on
packed amplitudes (docs/ccn.md; the dense reference: zccsdt.cc.ccn_ref).

CC2   cc2_update_amps(): the CCSD step of ccsdt_opt with the doubles
      equation <D| V~ + [F, T2] |0> (the singles equation is that of CCSD).
CC3   with T3 stored: ccsdt_opt.update_amps(model='cc3') - the CCSDT step
      without the ring term and the two ladders of the triples equation and
      with the drive operators of the dressed integrals alone - in the
      blocks of the symmetry labels, real or complex, with the memory plans
      of CCSDT.
Excitation energies: zccsdt.cc.eom with the Ground of
eom_opt.prepare(model=), whose sigma vector is the Jacobian of the model.

F is the Fock matrix of the integral container as it is (its occupied and
virtual blocks; not dressed with T1, not assumed diagonal).
'''

import numpy as np

from zccsdt.blas import gemm
from zccsdt.cc import ccsdt_opt, t3pack
from zccsdt.timing import timer


def folded_roots(apply_at, precond, guesses, store, nroots, conv_tol=1e-6, omega_tol=1e-10, omega0=0.0,
                 guess_tol=1e-3, max_outer=30, max_cycle=60, max_space=None, log=None, start=None):
    '''The ``nroots`` lowest roots of a folded, frequency-dependent
    eigenproblem A(omega) x = omega x (docs/ccn.md, section 4), each of them
    to self-consistency in omega.

    ``apply_at(omega)`` returns the operator apply(x, out) of
    zccsdt.cc.eom.davidson for the matrix at that frequency; ``precond``,
    ``guesses`` (vectors of ``store``) as there.  First all roots at
    ``omega0`` to ``guess_tol``; then root by root, from the lowest:
    omega <- the eigenvalue of A(omega) whose vector follows the root, by
    Davidson iterations started from the vector of the step before (with a
    threshold that follows |lambda(omega) - omega| down to ``conv_tol``),
    until the eigenvalue lambda(omega) is within ``omega_tol`` of omega with
    the residual below ``conv_tol``.  The frequencies are those of the
    secant method for lambda(omega) = omega (the plain fixed-point iteration
    converges at the rate of the weight of the folded rank in the state).

    ``start`` = [(omega, vector)] in the place of the first stage (and of
    ``guesses``): the roots of a cheaper matrix, e.g. of A_PP.

    Returns a zccsdt.cc.eom.Solved (history: per root the frequencies of
    the outer steps) with nsigma the number of applications in all.'''
    from zccsdt.cc import eom
    if start is None:
        nroots = min(nroots, len(guesses))
        space = max_space or 8 + 4 * nroots
        first = eom.davidson(apply_at(omega0), precond, guesses, store, nroots, guess_tol,
                             max(space, len(guesses) + nroots), max_cycle)
        order = np.argsort(first.e.real, kind='stable')
        start = [(float(first.e[k].real), first.vectors[k]) for k in order]
        nsigma, niter = first.nsigma, first.niter
    else:
        start = list(start)[:nroots]
        space = max_space or 8 + 4 * len(start)
        nsigma = niter = 0
    es, vecs, conv, resid, history = [], [], [], [], []
    for k, (omega, x) in enumerate(start):
        steps, gap, ok, prev = [omega], 1.0, False, None
        for outer in range(max_outer):
            tol = float(min(guess_tol, max(conv_tol, 0.1 * gap)))
            sol = eom.davidson(apply_at(omega), precond, [x], store, 1, tol, max(12, space // 2), max_cycle, follow=True)
            nsigma += sol.nsigma
            niter += sol.niter
            x, lam = sol.vectors[0], float(sol.e[0].real)
            gap = abs(lam - omega)              # how far omega is from the eigenvalue of A(omega)
            if log is not None:
                log(k, outer + 1, lam, gap, float(sol.residuals[0]), sol.nsigma)
            if gap < omega_tol and tol <= conv_tol and sol.converged[0]:
                omega, ok = lam, True
                steps.append(omega)
                break
            # the next frequency: lambda(omega), or the secant step for lambda(omega) = omega
            # from the last two evaluations (the slope is small: the weight of the folded rank)
            nxt = lam
            if prev is not None and abs(omega - prev[0]) > 1e-14:
                slope = (lam - prev[1]) / (omega - prev[0])
                if abs(slope) < 0.5:
                    nxt = omega + (lam - omega) / (1.0 - slope)
            prev, omega = (omega, lam), nxt
            steps.append(omega)
        es.append(omega)
        vecs.append(x)
        conv.append(ok)
        resid.append(float(sol.residuals[0]))
        history.append(steps)
    return eom.Solved(np.array(es, dtype=complex), vecs, np.array(conv), np.array(resid), niter, nsigma, 0, history)


def eom_folded(eris, t1, t2, nroots=3, conv_tol=1e-6, omega_tol=None, guess_tol=1e-4, opts=None, mem_mb=2000.0,
               max_cycle=60, max_outer=30, verbose=False, stdout=None, ground=None, extra=2):
    '''Excitation energies of CC3 without stored triples: the ``nroots``
    lowest totally symmetric roots of the folded eigenproblem
    [A_PP + A_PQ (omega - A_QQ)^-1 A_QP] R_P = omega R_P in the space of
    singles and doubles (docs/ccn.md, section 4), each to self-consistency
    in omega (folded_roots), started from the roots of the singles-doubles
    block A_PP (no triples work).  t1, t2: the CC3 amplitudes.  Returns a
    zccsdt.cc.eom.Result (r3 None; the vectors are normalized over their
    singles and doubles, the weights are those of the two ranks).'''
    import sys
    import time
    from zccsdt.cc import eom, eom_opt
    out = stdout or sys.stdout
    t_all = time.perf_counter()
    g = ground if ground is not None else eom_opt.prepare(eris, t1, t2, None, opts, mem_mb, model='cc3', fly=True)
    nocc, nvir = g.om.n, g.vm.n
    sizes = g.sizes()[:2]
    dtype = g.dtype

    def say(msg):
        if verbose:
            print(msg, file=out, flush=True)

    def apply_at(omega):
        def apply(x, y):
            s1, s2, _ = eom_opt.sigma(g, x[0].reshape(nocc, nvir), x[1], sd=omega is None, omega=omega)
            y[0][:] = s1.ravel()
            y[1][:] = s2.ravel()
        return apply
    say('EOM-EE-CC3 (triples folded, not stored): %d roots, vectors of %d + %d elements, %s' % (
        nroots, sizes[0], sizes[1], 'real' if dtype.kind == 'f' else 'complex'))
    store = eom.Store(sizes, dtype)
    precond = eom.make_precond(g, False)
    want = nroots + extra
    gs = []
    for k in eom.singles_guess(g, want, eom.symmetric_singles(eris, 0)):
        v = store.new()
        v[0][k] = 1.0
        gs.append(v)
    if not gs:
        raise ValueError('no single excitation to start from')
    t0 = time.perf_counter()
    nsolve = min(len(gs), want)
    first = eom.davidson(apply_at(None), precond, gs, store, nsolve, guess_tol, 4 * nsolve + 16, max_cycle)
    order = np.argsort(first.e.real, kind='stable')[:min(nroots, nsolve)]
    say('  singles-doubles block: %s  (%d sigma vectors, %.1f s)' % (
        ' '.join('%.6f' % first.e[k].real for k in order), first.nsigma, time.perf_counter() - t0))

    def log(k, outer, lam, gap, res, nsig):
        say('  root %d  step %d  omega = %.12f  |lambda(omega) - omega| = %.1e  |residual| %.1e  %d sigma vectors'
            % (k + 1, outer, lam, gap, res, nsig))
    tol_w = max(1e-10, 1e-2 * conv_tol) if omega_tol is None else omega_tol
    t0 = time.perf_counter()
    sol = folded_roots(apply_at, precond, None, store, nroots, conv_tol, tol_w, guess_tol=1e-3, max_outer=max_outer,
                       max_cycle=max_cycle, log=log, start=[(float(first.e[k].real), first.vectors[k]) for k in order])
    srt = np.argsort(sol.e.real, kind='stable')
    vec = [sol.vectors[k] for k in srt]
    wts = np.array([[float(np.vdot(p, p).real) for p in v] + [0.0] for v in vec])
    res = eom.Result(e=sol.e.real[srt], e_imag=sol.e.imag[srt], converged=sol.converged[srt], residuals=sol.residuals[srt],
                     r1=[v[0].reshape(nocc, nvir) for v in vec], r2=[v[1].reshape(g.om.npair, g.vm.npair) for v in vec],
                     r3=None, weights=wts, niter=sol.niter, nsigma=sol.nsigma, guess=first,
                     history=[sol.history[k] for k in srt], ground=g,
                     timings={'guess': t0 - t_all, 'iterations': time.perf_counter() - t0, 'total': time.perf_counter() - t_all})
    say('  root   omega / Eh       omega / eV   |residual|   %R1   %R2')
    for k in range(len(res.e)):
        say('  %3d  %14.10f  %11.6f   %.1e%s  %5.1f %5.1f' % (
            k + 1, res.e[k], res.e[k] * eom.HARTREE_EV, res.residuals[k], ' ' if res.converged[k] else '!',
            100 * wts[k][0] / wts[k].sum(), 100 * wts[k][1] / wts[k].sum()))
    say('  %d sigma vectors with triples; %.1f s' % (sol.nsigma, res.timings['total']))
    return res


def cc2_doubles(t1, t2, eris, vvvv_pp, vvoo_pp, om, vm, t2pp=None, tau=None):
    '''The doubles residual of CC2, packed [(i<j), (a<b)]:
        <ab||ij>~ + P(ab) f_bc t_ijac - P(ij) f_kj t_ikab,
    <ab||ij>~ fully T1-dressed (vvoo_pp of ccsdt_opt.dressed_vvoo_pp plus
    the ladder of the dressed <ab||de>~ with tau1), f the Fock matrix as it
    is.  ``tau``: in the place of tau1 (the Jacobian: its derivative).'''
    nocc, nvir = t1.shape
    mm, nn = om.pairs[:, 0], om.pairs[:, 1]
    dt = np.result_type(t2, vvoo_pp)
    if t2pp is None:
        t2pp = t3pack.pack_last2(np.ascontiguousarray(t2[mm, nn], dtype=dt), vm)
    if tau is None:
        tau = ccsdt_opt._tau1_pairs(t1, om, vm)
    tau = np.ascontiguousarray(tau, dtype=dt)
    R2 = np.array(vvoo_pp, dtype=dt, order='C', copy=True)
    for h, (lo, hi) in enumerate(vm.blocks):
        if hi > lo:
            X = gemm(np.ascontiguousarray(vvvv_pp[h], dtype=dt), np.ascontiguousarray(tau[:, lo:hi].T))
            t3pack.axpby(R2[:, lo:hi], X.T)
    f = np.asarray(eris.fock)
    foo, fvv = np.asarray(f[:nocc, :nocc], dtype=dt), np.asarray(f[nocc:, nocc:], dtype=dt)
    gemm(np.asfortranarray(ccsdt_opt.occ_pair_one_body(foo, om).T), np.ascontiguousarray(t2pp, dtype=dt), R2, beta=1.0)
    Y = gemm(np.ascontiguousarray(t2[mm, nn], dtype=dt).reshape(-1, nvir), np.ascontiguousarray(fvv.T))
    t3pack.add_asym_last2(R2, Y.reshape(-1, nvir, nvir), vm)
    return R2


def cc2_update_amps(opts, t1, t2, eris):
    '''One Jacobi step of CC2 in the T1-dressed formalism (the layouts and
    products of ccsdt_opt.ccsd_update_amps; the integrals of CCSD, <ab||cd>
    included - this is CC2 as a member of the hierarchy, not the o^2 v^3
    algorithm that never forms <ab||cd>).'''
    einsum = np.einsum
    nocc, nvir = t1.shape
    om, vm = t3pack.maps_for(eris)
    if ccsdt_opt._is_spin_eris(eris):
        raise NotImplementedError('CC2: for integrals without spin blocks (spin orbitals or spinors)')
    mm, nn = om.pairs[:, 0], om.pairs[:, 1]
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
    mo_e = eris.mo_energy
    eia = mo_e[:nocc, None] - (mo_e[None, nocc:] + opts.level_shift)
    with timer('t1_dress'):
        fdr = ccsdt_opt.dress_fock(eris, t1)
        ge = {k: ccsdt_opt.dress_block(eris, t1, k) for k in ('oooo', 'ovvo', 'oovv', 'ooov', 'vovv')}
        vvvv_pp = ccsdt_opt.dressed_vvvv_pp(eris, t1, om, vm)
        Bv = ccsdt_opt.bra_dressed_vvvo_pp(eris, t1, om, vm)
    with timer('ccsd'):
        goovv = ge['oovv']
        t2mn = t2[mm, nn]
        Feff_vv = fdr[nocc:, nocc:] - einsum('Paf,Pdf->ad', t2mn, goovv[mm, nn])
        Feff_oo = fdr[:nocc, :nocc] + einsum('mnP,inP->mi', goovv[:, :, md, ne], t2[:, :, md, ne])
        Weff_ovvo = ge['ovvo'] + einsum('mnde,inae->madi', goovv, t2, optimize=True)
        vvoo_pp = ccsdt_opt.dressed_vvoo_pp(eris, t1, Bv, om, vm)
        # (the singles equation is that of CCSD; its doubles part is not used)
        R1, _ = ccsdt_opt.ccsd_residual_packed(t1, t2, fdr, ge, vvvv_pp, vvoo_pp, Feff_vv, Feff_oo, Weff_ovvo, om, vm)
        R2 = cc2_doubles(t1, t2, eris, vvvv_pp, vvoo_pp, om, vm)
    return t1 + R1 / eia, t3pack.t2_update(t2, R2, om, vm, mo_e[:nocc], mo_e[nocc:] + opts.level_shift)


# ------------------------------------------------------------ CC4 with T4 stored

def cc4_rhs(t2, T3, ints, batch_bytes=None):
    '''The right-hand side of the quadruples equation of CC4,
    <Q| [V~, T3] + 1/2 [[V~, T2], T2] |0> (packed [C(o,4), C(v,4)]), from
    the T4 equation of CCSDTQ (zccsdt.cc.t4engine.residual) with T4 = 0:
    that is a polynomial in (T2, T3) - [V~, T3] is the part linear in T3,
    1/2 [[V~, T2], T2] the part quadratic in T2 at T3 = 0, and the other
    terms are of the orders T2 T3, T3 T3 and T2^3 - so the two parts are
    exact combinations of four evaluations:
        [V~, T3]             = (R(0, T3) - R(0, -T3)) / 2
        1/2 [[V~, T2], T2]   = (R(T2, 0) + R(-T2, 0)) / 2.'''
    from math import comb
    from zccsdt.cc import t4engine
    no, nv = ints.nocc, ints.nvir
    T4 = np.zeros((comb(no, 4), comb(nv, 4)), dtype=np.result_type(t2, T3, ints.oovv))
    z2, z3 = np.zeros_like(t2), np.zeros_like(T3)
    r = [t4engine.residual(a2, a3, T4, ints, batch_bytes)[0] for a2, a3 in ((z2, T3), (z2, -T3), (t2, z3), (-t2, z3))]
    return 0.5 * (r[0] - r[1] + r[2] + r[3])


def cc4_kernel(eris, conv_tol=1e-10, conv_tol_normt=1e-8, max_cycle=200, diis_space=8, verbose=False,
               t1=None, t2=None, t3=None, model='cc4', store_t4=True, opts=None):
    '''CC4 with T4 stored (docs/ccn.md, section 11): the singles, doubles
    and triples equations of CCSDTQ (the CCSDT step and the T4 terms of the
    T4 engine), the quadruples equation [F, T4] = -([V~, T3] + 1/2 [[V~, T2],
    T2]) solved in closed form for the semicanonical spinors (cc4_rhs).
    Everything in memory, dense layout: the reference of the route without
    stored T4.  ``model`` 'ccsdtq': the full quadruples equation (the limit;
    zccsdt.cc.ccsdtq does that in production).  ``store_t4`` False: no T4
    - the quadruples of every occupied quadruple are made from the current
    amplitudes, contracted into the doubles and triples equations and
    discarded (zccsdt.cc.cc4fly) - the production route: solver.kernel with
    method 'cc4', the CCSDT step in the layout of the integrals (the blocks
    of their labels), memory by plan; ``opts``: its solver.Options (default:
    the tolerances given here).  Returns a solver.Result with t4 (packed;
    None without T4) and the semicanonical integrals.'''
    if not store_t4:
        from zccsdt.cc import solver
        if model != 'cc4':
            raise ValueError('without stored T4: CC4')
        opts = opts or solver.Options(conv_tol=conv_tol, conv_tol_normt=conv_tol_normt, max_cycle=max_cycle,
                                      diis_space=diis_space, verbose=verbose)
        res = solver.kernel(eris, 'cc4', opts, t1=t1, t2=t2, t3=t3)
        res.t4 = None
        return res
    import time
    from math import comb
    from zccsdt.cc import ccsd, ccsdt_opt, solver, t4engine, t4pack as P
    from zccsdt.cc.diis import DIIS
    eris, rotation = semicanonicalize(eris)
    no = eris.nocc
    nv = np.asarray(eris.fock).shape[0] - no
    if t1 is None or t2 is None:
        t1, t2 = solver.init_amps(eris)
    dtype = np.result_type(t1, t2, eris.oovv, eris.fock)
    t1, t2 = t1.astype(dtype), t2.astype(dtype)
    nI3, nA3, nI, nA = comb(no, 3), comb(nv, 3), comb(no, 4), comb(nv, 4)
    T3 = np.zeros(nI3 * nA3, dtype=dtype) if t3 is None else np.array(t3, dtype=dtype).ravel()
    T4 = np.zeros((nI, nA), dtype=dtype)
    e = np.asarray(eris.fock).diagonal().real
    eo, ev = e[:no], e[no:]
    d2 = eo[P.tuples(no, 2)].sum(axis=1)[:, None] - ev[P.tuples(nv, 2)].sum(axis=1)[None, :]
    d3 = (eo[P.tuples(no, 3)].sum(axis=1)[:, None] - P.tuple_sums(ev, 3)[None, :]).ravel()
    d4 = eo[P.tuples(no, 4)].sum(axis=1)[:, None] - P.tuple_sums(ev, 4)[None, :]
    step = solver.Options()
    plan = ccsdt_opt.plan_for(eris, dtype, solver.Options(incore_complete=True))
    adiis = DIIS(space=diis_space, incore=True)
    e_old = ccsd.energy(t1, t2, eris)
    conv, timings = False, []
    for cycle in range(max_cycle):
        t0 = time.perf_counter()
        t1n, t2n, t3n = ccsdt_opt.update_amps(step, t1, t2, T3, eris, plan=plan)
        ints = t4engine.Dressed.from_eris(eris, t1)
        # the T4 terms of the doubles and triples equations (they do not depend on T2, T3)
        R4, r3, r2 = t4engine.residual(np.zeros_like(t2), np.zeros_like(T3), T4, ints)
        if model == 'cc4':
            T4n = cc4_rhs(t2, T3, ints) / d4
        else:
            R4 = t4engine.residual(t2, T3, T4, ints)[0]
            T4n = T4 + R4 / d4
        del ints, R4
        t2n = t2n + P.unpack(r2 / d2, no, nv, 2)
        t3n = t3n + r3.ravel() / d3
        normt = float(np.linalg.norm(t1n - t1) + np.linalg.norm(t2n - t2) + 6 * np.linalg.norm(t3n - T3)
                      + 24 * np.linalg.norm(T4n - T4))
        parts = adiis.update(solver._parts(t1n, t2n, t3n) + [T4n.ravel()])
        t1, t2, T3 = solver._from_parts(parts[:3], no, nv, True)
        T3 = np.array(T3)
        T4 = parts[3].reshape(nI, nA).copy()
        e_new = ccsd.energy(t1, t2, eris)
        timings.append(time.perf_counter() - t0)
        if verbose:
            print('cycle %3d  E_corr = %.15f  dE = %.3e  |dt| = %.3e' % (cycle + 1, e_new.real, (e_new - e_old).real, normt))
        done = abs(e_new - e_old) < conv_tol and normt < conv_tol_normt
        e_old = e_new
        if done:
            conv = True
            break
    adiis.close()
    res = solver.Result(e_corr=e_old, e_tot=eris.e_hf + float(np.real(e_old)), converged=conv, ncycle=cycle + 1,
                        t1=t1, t2=t2, t3=T3, timings=timings, eris=eris, rotation=rotation)
    res.t4 = T4
    return res


# ------------------------------------------------------------ semicanonical spinors

def fock_groups(f, labels=None):
    '''Groups of the indices of a block of the Fock matrix that it couples:
    the connected components of its nonzero elements (exact zeros, as the
    spin blocks of spin orbitals of pure spin have them), within the
    symmetry labels if given.  A list of index arrays, each in increasing
    order.'''
    from scipy.sparse.csgraph import connected_components
    n = f.shape[0]
    link = np.asarray(f) != 0
    if labels is not None:
        lab = np.asarray(labels)
        link &= lab[:, None] == lab[None, :]
    ncomp, comp = connected_components(link, directed=False)
    return [np.flatnonzero(comp == c) for c in range(ncomp)]


def semicanonical_rotation(f, nocc, labels=None):
    '''The unitary matrix U (block diagonal: occupied, virtual) that makes
    the occupied and the virtual block of the Fock matrix ``f`` diagonal,
    group by group of fock_groups (labels, spin blocks and real arithmetic
    kept: U is real for a real f), with the eigenvalues in increasing order
    inside a group and every column turned so that its largest element is
    real and positive; and the eigenvalues.  None if both blocks are
    exactly diagonal: nothing to do.  The models with the full blocks of F
    are invariant under such rotations (docs/ccn.md, section 1).'''
    f = np.asarray(f)
    n = f.shape[0]
    off = f - np.diag(f.diagonal())
    off[:nocc, nocc:] = off[nocc:, :nocc] = 0
    if not off.any():
        return None
    U = np.zeros((n, n), dtype=f.dtype)
    e = f.diagonal().real.copy()
    for lo, hi in ((0, nocc), (nocc, n)):
        blk = f[lo:hi, lo:hi]
        lab = None if labels is None else np.asarray(labels)[lo:hi]
        for g in fock_groups(blk, lab):
            sub = blk[np.ix_(g, g)]
            sub = 0.5 * (sub + sub.conj().T)
            w, u = np.linalg.eigh(sub)
            big = u[np.argmax(abs(u), axis=0), np.arange(len(g))]
            u = u * (abs(big) / big)[None, :]
            U[np.ix_(lo + g, lo + g)] = u
            e[lo + g] = w
    return U, e


def semicanonical_columns(mf, C, mo_occ, nfrozen, nocc, labels=None, real=False):
    '''The columns ``C`` (frozen occupied, active occupied, active virtual;
    occupations ``mo_occ``) with the active occupied and the active virtual
    ones rotated among themselves so that the Fock matrix of the SCF object
    ``mf`` (one Fock build of its density) is diagonal in both blocks
    (semicanonical_rotation); C itself if it already is, exactly.  The
    frozen columns are not touched.  For the front end: before the
    integrals are made.'''
    dm = mf.make_rdm1(C, mo_occ)
    fock_ao = mf.get_fock(dm=dm)
    Ca = C[:, nfrozen:]
    f = Ca.conj().T @ fock_ao @ Ca
    if real:
        f = f.real
    rot = semicanonical_rotation(f, nocc, labels)
    if rot is None:
        return C, None
    out = np.array(C, copy=True)
    out[:, nfrozen:] = Ca @ rot[0]
    return out, rot


def _rotate(x, mats):
    '''x with every axis k transformed by mats[k] (x'[.. p ..] = sum_a
    M[a, p] x[.. a ..]; None: unchanged).'''
    for k, M in enumerate(mats):
        if M is not None:
            x = np.moveaxis(np.tensordot(x, M, axes=([k], [0])), -1, k)
    return x


def semicanonicalize(eris, labels=None):
    '''(eris', U): an integral container of the same spinors rotated among
    the occupied and among the virtual ones so that the occupied and the
    virtual block of its Fock matrix are diagonal, with every block of
    integrals transformed (<pq||rs>' = sum U*_ap U*_bq U_cr U_ds <ab||cd>),
    or (eris, None) if the blocks are diagonal already.  For integrals given
    as they are (from_integrals, tests); <ab||cd> on pairs is unpacked to
    v^4 numbers for the transformation - the front end rotates the spinors
    before the integrals are made instead (semicanonical_columns).
    Amplitudes of eris' refer to the rotated spinors; t1 back is
    U_o t1 U_v^H (back_t1, back_t2).'''
    import copy
    from zccsdt.cc import ccsdt as ccsdt_ref
    no = eris.nocc
    lab = labels if labels is not None else (getattr(eris, 'labels', None) if not getattr(eris, 'modulus', 0) else None)
    rot = semicanonical_rotation(np.asarray(eris.fock), no, lab)
    if rot is None:
        return eris, None
    U, e = rot
    Uo, Uv = U[:no, :no], U[no:, no:]
    dt = np.result_type(eris.fock, eris.oovv, U)
    sp = {'o': Uo, 'v': Uv}
    out = copy.copy(eris)
    out.__dict__ = {k: v for k, v in eris.__dict__.items() if not k.startswith('_')}
    for key in ('oooo', 'ooov', 'oovv', 'ovov', 'ovvo', 'ovvv', 'vvvv'):
        x = getattr(eris, key, None)
        if x is None:
            continue
        mats = [sp[c].conj() if k < 2 else sp[c] for k, c in enumerate(key)]
        setattr(out, key, np.ascontiguousarray(_rotate(np.asarray(x, dtype=dt), mats)))
    pp = getattr(eris, '_vvvv_pp', None)
    if pp is not None:
        nv = Uv.shape[0]
        a, b = np.triu_indices(nv, 1)
        full = np.zeros((nv, nv, nv, nv), dtype=dt)
        P = np.asarray(pp)
        full[a[:, None], b[:, None], a[None, :], b[None, :]] = P
        full[b[:, None], a[:, None], a[None, :], b[None, :]] = -P
        full[a[:, None], b[:, None], b[None, :], a[None, :]] = -P
        full[b[:, None], a[:, None], b[None, :], a[None, :]] = P
        full = _rotate(full, [Uv.conj(), Uv.conj(), Uv, Uv])
        out._vvvv_pp = np.ascontiguousarray(full[a[:, None], b[:, None], a[None, :], b[None, :]])
        del full
    f = U.conj().T @ np.asarray(eris.fock, dtype=dt) @ U
    f[:no, :no] = np.diag(e[:no])
    f[no:, no:] = np.diag(e[no:])
    out.fock = f if np.iscomplexobj(eris.fock) or np.iscomplexobj(U) else f.real
    out.mo_energy = e.copy()
    out.source = (getattr(eris, 'source', '') or '') + ' (semicanonical)'
    out.semicanonical = (Uo, Uv)
    return out, (Uo, Uv)


def back_t1(t1, rot):
    '''t1[i, a] of the rotated spinors in the spinors before the rotation.'''
    Uo, Uv = rot
    return Uo.conj() @ t1 @ Uv.T


def back_t2(t2, rot):
    Uo, Uv = rot
    return _rotate(t2, [Uo.conj().T, Uo.conj().T, Uv.T, Uv.T])
