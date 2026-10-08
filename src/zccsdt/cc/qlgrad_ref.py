'''Dense reference of the analytic first derivatives of CCSDT(Q)_Lambda on
spin orbitals or spinors, real or complex: the two linear equations that
(Q)_Lambda adds to those of CCSDT, the densities, the orbital gradient and
the relaxed one-particle density.  docs/lambda.md, section 12.

The functional.  With canonical Hartree-Fock orbitals and bare integrals
(lambda_ref.quadruples_lambda, section 5),

    E(Q)_L = Re e,   e = sum z4 r4 / D4          (unique quadruples),
    r4 = A(g, t3) + B(g, t2, t2),                A = <Q| [V,T3] |0>, B = <Q| 1/2 [[V,T2],T2] |0>,
    z4 = Z(g^T, l2) + A(g^T, l3) = <0| (Lambda_2 + Lambda_3) V |Q>,   g^T[p,q,r,s] = g[r,s,p,q],

Z = <Q| V T2 |0> the disconnected product.  Nothing is conjugated: e is
holomorphic in t2, t3, l2, l3 and g, like the Lagrangian of CCSDT
L(t, lambda) = <0| (1 + Lambda) exp(-T) H exp(T) |0>; t1 and l1 do not enter.
(lambda_ref.quadruples_lambda writes z4 as conj(Z(g, conj(l)): the same
number for Hermitian integrals.  energy() below is e for any g.)

lambda is not a free variable: it solves the Lambda equations of CCSDT,
W_k(t, lambda) = dL/dt_k = 0, as t solves the amplitude equations R_k(t) =
0.  Both are constraints of the Lagrangian of CCSDT(Q)_Lambda, with the
multipliers zeta (for W; a vector like t) and lambda-bar (for R):

    F = L(t, lambda) + e(t, lambda) + sum zeta_k W_k(t, lambda) + sum lambda-bar_k R_k(t)
      = L(t, lambda') + e(t, lambda) + sum zeta_k W_k(t, lambda),    lambda' = lambda + lambda-bar

(L is linear in lambda; sums over the unique elements).  Stationary in
lambda (at fixed lambda'), since d(zeta . W)/dl_nu = sum_mu zeta_mu
dR_nu/dt_mu:

    sum_mu (dR_nu/dt_mu) zeta_mu + de/dl_nu = 0                         (1)

- the Jacobian of the amplitude equations on zeta, a right-hand linear
response equation with the source q_nu = de/dl_nu = <nu| V Y4 |0>, Y4 = r4 /
D4 (the terms of T4 in the doubles and triples equations of CCSDTQ; q1 = 0,
zeta_1 is not).  Stationary in t:

    W_k(t, lambda') + de/dt_k + sum_mu zeta_mu dW_mu/dt_k (t, lambda) = 0     (2)

- the Lambda equations of CCSDT for lambda' with the constant sources
s_k = de/dt_k + (the directional derivative of W(t, lambda) along zeta in
t), lambda_ref.kernel(sources=).  Then dF/dx is the derivative of the
explicit dependence on the integrals,

    d1, d2 = densities(t, lambda') + (the directional derivative of densities(t, lambda) along zeta),

in L = sum h d1 + 1/4 sum <pq||rs> d2, and those of e: its d2, and de on
the orbital energies.  The energy is Re F: the densities are made Hermitian
(lambda_ref.pyscf_rdms), and the orbitals follow as canonical Hartree-Fock
orbitals as for CCSDT(Q) (qgrad_ref.relaxed_rdm1).

How the directional derivatives are made here: exp(-T) H exp(T) ends at
the fourth power of T, so R, W and the densities are polynomials of degree
<= 4 in the amplitudes (t1 included), and the central five-point formula
[8 (f(t + z) - f(t - z)) - (f(t + 2z) - f(t - 2z))] / 12 is their exact
derivative along z (directional).  jacobian_dressed is the form an
implementation on packed amplitudes can take: three evaluations of the
residuals of the T1-dressed formalism.
'''

from dataclasses import dataclass

import numpy as np

from zccsdt.cc import ccsdt, ccsdtq_ref as R, ccsdtq_terms as Q, lambda_ref as LR, qgrad_ref as G
from zccsdt.cc.diis import DIIS

RIGHT = ('R4: [V,T3]', 'R4: 1/2 [[V,T2],T2]')
LEFT = ('Z4.T2', 'Z4.T3')


def _swap(g):
    '''g^T[p,q,r,s] = g[r,s,p,q]: bra and ket exchanged, nothing conjugated.'''
    return g.transpose(2, 3, 0, 1)


def energy(t2, t3, l2, l3, g, mo_energy, nocc):
    '''e = sum z4 r4 / D4, complex, for any arguments: E(Q)_L is its real
    part.  The function that quadruples_gradient differentiates.'''
    x, d = R._pieces(RIGHT, 4, {'v': g, 't2': t2, 't3': t3}, mo_energy, nocc)
    z = R._pieces(LEFT, 4, {'v': _swap(g), 't2': l2, 't3': l3}, mo_energy, nocc)[0]
    return complex(np.sum((z[LEFT[0]] + z[LEFT[1]]) * (x[RIGHT[0]] + x[RIGHT[1]]) / d))


@dataclass
class Gradient:
    e: complex                  # sum z4 r4 / D4; E(Q)_L = e.real
    s2: np.ndarray              # de/dt2 [i,j,a,b], antisymmetric: de = sum_(i<j,a<b) s2 dt2
    s3: np.ndarray              # de/dt3
    q2: np.ndarray              # de/dl2: the source of the equation of zeta
    q3: np.ndarray              # de/dl3
    d2: np.ndarray              # [p,q,r,s] over all spin orbitals: de = sum d2 d<pq||rs>
    de: np.ndarray              # [p], real: dE(Q)_L = sum de d e_p


def quadruples_gradient(t2, t3, l2, l3, g, mo_energy, nocc):
    '''e and its derivatives (see the module) for dense antisymmetric
    amplitudes and lambda, g[p,q,r,s] = <pq||rs> over all spin orbitals and
    the orbital energies: a Gradient.  With y = r4 / D4 and w = z4 / D4,

        de = sum [ y dz4 + w dr4 ] - sum w y dD4:

    the derivative of every term of r4 with w as the left-hand quadruple
    amplitudes, that of every term of z4 (in g^T, l2, l3) with y.'''
    n = g.shape[0]
    nvir = n - nocc
    right = {'v': g, 't2': t2, 't3': t3}
    left = {'v': _swap(g), 't2': l2, 't3': l3}

    def zeros():
        return {'v': np.zeros((n,) * 4, dtype=complex), 't2': np.zeros(t2.shape, dtype=complex),
                't3': np.zeros(t3.shape, dtype=complex)}
    dr, dz = zeros(), zeros()
    de = np.zeros(n, dtype=complex)
    eo, ev = mo_energy[:nocc], mo_energy[nocc:]
    vir = Q.tuples(nvir, 4)
    e = 0.0
    for occ in Q.tuples(nocc, 4):
        occ = tuple(int(x) for x in occ)
        r = sum(R._block(G._terms(name), right, nocc, 4, occ) for name in RIGHT)
        z = sum(R._block(G._terms(name), left, nocc, 4, occ) for name in LEFT)
        d = eo[list(occ)].sum() - ev[vir].sum(axis=1)
        y, w = r / d, z / d
        e += np.sum(z * y)
        c = w * y
        de[list(occ)] -= c.sum()
        np.add.at(de, nocc + vir, c[:, None])
        for names, arrays, lq, out in ((RIGHT, right, w, dr), (LEFT, left, y, dz)):
            lq = Q.expand(np.asarray(lq, dtype=complex)[None, :], 4, nvir, 4)     # [4,4,4,4,v,v,v,v]
            for name in names:
                for coef, factors in G._terms(name):
                    ops = [G._operand(kind, ix, arrays, nocc, occ) for kind, ix in factors]
                    for p, (kind, ix) in enumerate(factors):
                        rest = [(f[1], a) for k, (f, a) in enumerate(zip(factors, ops)) if k != p]
                        sub = ','.join(['ijklabcd'] + [s for s, _ in rest]) + '->' + ix
                        val = np.einsum(sub, lq, *[a for _, a in rest], optimize=True)
                        out[kind][G._where(kind, ix, nocc, nvir, occ)] += float(coef) * val
    # sums over all elements -> sums over the unique ones
    res = [Q.asym(dr['t2'], 2), Q.asym(dr['t3'], 3), Q.asym(dz['t2'], 2), Q.asym(dz['t3'], 3),
           dr['v'] + _swap(dz['v'])]
    if not any(np.iscomplexobj(x) for x in (t2, t3, l2, l3, g)):
        res = [x.real.copy() for x in res]
    return Gradient(complex(e), *res, de.real.copy())


def fock_density(t2, t3, l2, l3, g, mo_energy, nocc):
    '''Gamma[p,q], the derivative of e with respect to the Fock matrix in the
    form that needs no canonical orbitals, e = sum z4 y4 with y4 the
    solution of D y = r4 for the whole occupied-occupied and virtual-virtual
    blocks of f (the same number for canonical orbitals): de = sum Gamma[p,q]
    df[p,q],

        Gamma[a,e] = sum w[ijkl,abcd] y[ijkl,ebcd]      (i<j<k<l, b<c<d),
        Gamma[m,i] = -sum w[ijkl,abcd] y[mjkl,abcd]     (j<k<l, a<b<c<d),

    qgrad_ref.fock_density without the conjugation; the real part of its
    diagonal is Gradient.de.  That form does not change when the occupied
    or the virtual spin orbitals are rotated among themselves with t and
    lambda, so the orbital gradient of CCSDT(Q)_Lambda among them is

        G[p,q] = (e_q - e_p) (Gamma[q,p] + conj(Gamma[p,q])) / 2.

    From all quadruples as full arrays: for a few spin orbitals.'''
    n = g.shape[0]
    nvir = n - nocc
    x, d = R._pieces(RIGHT, 4, {'v': g, 't2': t2, 't3': t3}, mo_energy, nocc)
    z = R._pieces(LEFT, 4, {'v': _swap(g), 't2': l2, 't3': l3}, mo_energy, nocc)[0]
    y = Q.expand((x[RIGHT[0]] + x[RIGHT[1]]) / d, nocc, nvir, 4)
    w = Q.expand((z[LEFT[0]] + z[LEFT[1]]) / d, nocc, nvir, 4)
    out = np.zeros((n, n), dtype=complex)
    out[nocc:, nocc:] = np.einsum('ijklabcd,ijklebcd->ae', w, y) / 144
    out[:nocc, :nocc] = -np.einsum('ijklabcd,mjklabcd->mi', w, y) / 144
    return out


# ------------------------------------------------------ directional derivatives

def directional(f, amps, zeta):
    '''d/dx f([t_k + x zeta_k]) at x = 0 for f that returns a sequence of
    arrays, each a polynomial of degree <= 4 in the amplitudes: exact (the
    central five-point formula with step 1).'''
    def at(x):
        return f([t + x * z for t, z in zip(amps, zeta)])
    p1, m1, p2, m2 = at(1.0), at(-1.0), at(2.0), at(-2.0)
    return [(8.0 * (a - b) - (c - d)) / 12.0 for a, b, c, d in zip(p1, m1, p2, m2)]


def jacobian(zeta, amps, fock, g, nocc):
    '''sum_mu (dR_k/dt_mu) zeta_mu, k = 1, 2, 3: the Jacobian of the CCSDT
    amplitude equations at ``amps`` on a vector zeta = [z1, z2, z3] (dense,
    antisymmetric).'''
    return directional(lambda a: R.residuals(a, fock, g, nocc), amps, zeta)


def jacobian_dressed(zeta, amps, fock, g, nocc):
    '''The same from three evaluations of the residuals of the T1-dressed
    formalism, P(H~; t2, t3) - linear in the dressed integrals, quadratic in
    (t2, t3):

        [P(H~; t2 + z2, t3 + z3) - P(H~; t2 - z2, t3 - z3)] / 2 + P(dH~; t2, t3),

    dH~ = [H~, z1] the derivative of the dressed integrals along z1 (a
    one-index transformation; here by the five-point formula on the
    dressing).'''
    n = g.shape[0]
    t1, t2, t3 = amps

    def P(v, f, a2, a3):
        arrays = {'v': v, 'f': f, 't2': a2, 't3': a3}
        return [Q.residual(k, arrays, nocc, with_t4=False) for k in (1, 2, 3)]
    ge, fdr = ccsdt._t1_dress(g, fock, t1, nocc, n)
    dge, dfdr = directional(lambda a: ccsdt._t1_dress(g, fock, a[0], nocc, n), [t1], [zeta[0]])
    p = P(ge, fdr, t2 + zeta[1], t3 + zeta[2])
    m = P(ge, fdr, t2 - zeta[1], t3 - zeta[2])
    return [0.5 * (a - b) + c for a, b, c in zip(p, m, P(dge, dfdr, t2, t3))]


def lambda_response(zeta, amps, lams, fock, g, nocc):
    '''sum_mu zeta_mu dW_mu/dt_k (t, lambda), k = 1, 2, 3: the second
    derivative of the Lagrangian of CCSDT with respect to t, contracted with
    zeta - the directional derivative of the residuals of the Lambda
    equations along zeta in t, at fixed lambda.'''
    return directional(lambda a: LR.residuals(lams, a, fock, g, nocc), amps, zeta)


def density_response(zeta, amps, lams, nocc):
    '''(d1, d2): the directional derivative of lambda_ref.densities(t,
    lambda) along zeta in t - the coefficients of the integrals in
    sum zeta_mu W_mu(t, lambda).'''
    return directional(lambda a: LR.densities(a, lams, nocc), amps, zeta)


def _real(arrays, *others):
    if any(np.iscomplexobj(x) and abs(np.imag(x)).max() > 0 for x in others):
        return arrays
    return [np.real(x).copy() for x in arrays]


def solve_zeta(fock, g, mo_energy, nocc, amps, q2, q3, conv_tol_normt=1e-11, max_cycle=200,
               diis_space=8, verbose=False):
    '''Equation (1): jacobian(zeta) + [0, q2, q3] = 0 by Jacobi steps with
    DIIS, zeta += residual / D as for the amplitudes.  (converged, cycles,
    [z1, z2, z3]).'''
    nvir = g.shape[0] - nocc
    den = [R.denominator(mo_energy, nocc, k) for k in (1, 2, 3)]
    src = [np.zeros(amps[0].shape, dtype=complex), np.asarray(q2, dtype=complex), np.asarray(q3, dtype=complex)]
    zeta = [s / d for s, d in zip(src, den)]
    adiis = DIIS(space=diis_space, incore=True)
    conv = False

    def parts(a):
        return [a[0].ravel()] + [Q.unique(x, k + 1).ravel() for k, x in enumerate(a) if k]
    for cycle in range(max_cycle):
        res = [j + s for j, s in zip(jacobian(zeta, amps, fock, g, nocc), src)]
        new = [z + r / d for z, r, d in zip(zeta, res, den)]
        norm = sum(np.linalg.norm(a - b) for a, b in zip(new, zeta))
        if verbose:
            print('zeta: cycle %3d  |dz| = %.2e' % (cycle + 1, norm))
        if norm < conv_tol_normt:
            zeta, conv = new, True
            break
        ps = adiis.update(parts(new))
        zeta = [ps[0].reshape(nocc, nvir)] + [
            Q.expand(p.reshape(len(Q.tuples(nocc, k + 1)), -1), nocc, nvir, k + 1)
            for k, p in enumerate(ps) if k]
    adiis.close()
    return conv, cycle + 1, _real(zeta, fock, g, q2, q3, *amps)


# ------------------------------------------------------------- the Lagrangian

@dataclass
class Terms:
    e: complex                  # sum z4 r4 / D4
    q: Gradient
    zeta: list                  # [z1, z2, z3]
    sources: list               # [s1, s2, s3] of equation (2)
    lams: list                  # lambda' = lambda + lambda-bar
    d1: np.ndarray              # densities of F without those of e (q.d2, q.de): not Hermitian
    d2: np.ndarray
    ncycle: tuple = None        # iterations of (1) and of (2)


def lagrangian(fock, g, mo_energy, nocc, amps, lams, zeta, lams2):
    '''F = L(t, lambda') + e(t, lambda) + sum zeta_k W_k(t, lambda), complex,
    for any arguments (``lams2`` = lambda'; the reference energy is in L).'''
    L = LR.energy(fock, g, nocc, *LR.densities(amps, lams2, nocc))
    W = LR.residuals(lams, amps, fock, g, nocc)
    zw = sum(np.sum(z * w) / (f * f) for z, w, f in zip(zeta, W, (1, 2, 6)))
    return L + energy(amps[1], amps[2], lams[1], lams[2], g, mo_energy, nocc) + zw


def kernel(fock, g, mo_energy, nocc, amps, lams, conv_tol_normt=1e-11, verbose=False):
    '''Everything of the Lagrangian of CCSDT(Q)_Lambda for converged
    amplitudes and Lambda of CCSDT, at fixed orbitals: the derivatives of
    e, zeta from (1), lambda' from (2), and the densities.  A Terms.'''
    q = quadruples_gradient(amps[1], amps[2], lams[1], lams[2], g, mo_energy, nocc)
    conv, nz, zeta = solve_zeta(fock, g, mo_energy, nocc, amps, q.q2, q.q3, conv_tol_normt, verbose=verbose)
    if not conv:
        raise RuntimeError('the equations of zeta did not converge')
    dw = lambda_response(zeta, amps, lams, fock, g, nocc)
    sources = _real([dw[0], dw[1] + q.s2, dw[2] + q.s3], fock, g, *amps, *lams)
    rl = LR.kernel(fock, g, mo_energy, nocc, amps, conv_tol_normt=conv_tol_normt, sources=sources,
                   verbose=verbose)
    if not rl.converged:
        raise RuntimeError('the Lambda equations with the sources of (Q)_Lambda did not converge')
    d1, d2 = LR.densities(amps, rl.lams, nocc)
    z1, z2 = density_response(zeta, amps, lams, nocc)
    return Terms(q.e, q, zeta, sources, rl.lams, d1 + z1, d2 + z2, (nz, rl.ncycle))


# ------------------------------------------------------------ relaxed density

@dataclass
class Relaxed:
    relaxed: np.ndarray         # D[p,q] = <q+ p>, Hermitian, all spin orbitals
    unrelaxed: np.ndarray
    z: np.ndarray
    gradient: np.ndarray        # G[y,x] of zccsd.zvector, all spin orbitals
    e_corr: float               # E_corr(CCSDT)
    e_ql: float                 # E(Q)_L
    amps: list
    lams: list                  # lambda of CCSDT
    terms: Terms = None
    gradient_e: np.ndarray = None   # the part of ``gradient`` from the densities of e (d2, de)


def relaxed_rdm1(mf, mo_coeff, nfrozen=0, ndrop=0, amps=None, lams=None, tol=1e-11,
                 conv_tol_normt=1e-11, verbose=None):
    '''The relaxed one-particle density of CCSDT(Q)_Lambda for the SCF
    ``mf`` and its canonical spin orbitals or spinors ``mo_coeff`` (all of
    them, occupied first), the first ``nfrozen`` frozen and the last
    ``ndrop`` virtual ones left out: from full arrays over all spin
    orbitals, for small molecules.  ``amps``, ``lams``: converged dense
    [t1, t2, t3] and [l1, l2, l3] of CCSDT in the active spin orbitals
    (default: ccsdtq_ref.kernel, lambda_ref.kernel).  Returns a Relaxed; an
    expectation value is sum V[p,q] D[q,p].'''
    from zccsdt import eris as E
    from zccsdt.zccsd import zvector
    fock, g, mo_e, o = R.integrals(E.from_scf(mf, mo_coeff=mo_coeff))
    n = fock.shape[0]
    oa = o - nfrozen
    a = slice(nfrozen, n - ndrop)
    fa, ga, ea = fock[a, a], g[a, a, a, a], mo_e[a]
    if amps is None:
        r = R.kernel(fa, ga, ea, oa, 'ccsdt', conv_tol=1e-13, conv_tol_normt=conv_tol_normt)
        if not r.converged:
            raise RuntimeError('CCSDT did not converge')
        amps = r.amps
    e_corr = float(np.real(R.energy(amps[0], amps[1], fa, ga, oa)))
    if lams is None:
        rl = LR.kernel(fa, ga, ea, oa, amps, conv_tol_normt=conv_tol_normt)
        if not rl.converged:
            raise RuntimeError('the Lambda equations did not converge')
        lams = rl.lams
    t = kernel(fa, ga, ea, oa, amps, lams, conv_tol_normt)
    D1, D2 = G.full_densities(t.d1, t.d2, nfrozen, n)
    dm1, dm2 = LR.pyscf_rdms(D1, D2)
    h = fock - np.einsum('piqi->pq', g[:, :o, :, :o])
    eri = 0.5 * g.transpose(0, 2, 1, 3)                 # (pq|rs) - (ps|rq) = <pr||qs>
    Gr = zvector.orbital_gradient(h, eri, dm1, dm2)
    # the explicit dependence of e on the integrals and on the orbital energies
    d2 = np.zeros((n,) * 4, dtype=complex)
    d2[a, a, a, a] = t.q.d2
    d1 = np.zeros((n, n), dtype=complex)
    d1[np.arange(n)[a], np.arange(n)[a]] = t.q.de
    Gq = zvector.gradient_nh(zvector.t_fock(fock, o), g, d2, d1, o)
    Ge = 0.5 * (Gq - Gq.conj().T)
    Gr = Gr + Ge
    dm1 = dm1 + d1
    spaces = (np.arange(nfrozen, o), np.arange(nfrozen), np.arange(o, n - ndrop), np.arange(n - ndrop, n))
    relaxed, z = zvector._relax(mf, mf, np.asarray(mo_coeff), mo_e, o, Gr, dm1, spaces, True,
                                zvector.DEGENERATE_DENSE, tol, verbose)
    return Relaxed(relaxed, dm1, z, Gr, e_corr, float(t.e.real), amps, lams, t, Ge)
