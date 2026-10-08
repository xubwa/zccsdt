'''Dense reference of the analytic first derivatives of CCSDT(Q) on spin
orbitals or spinors, real or complex: what (Q) adds to the Lambda equations
of CCSDT (source terms), to the densities and to the orbital gradient, and
the relaxed one-particle density.  docs/lambda.md, section 11.

The functional.  With canonical Hartree-Fock orbitals and bare integrals

    E(Q) = Re sum conj(z4) r4 / D4          (unique quadruples i<j<k<l, a<b<c<d),
    r4 = A(g, t3) + B(g, t2, t2),           A = <Q| [V,T3] |0>, B = <Q| 1/2 [[V,T2],T2] |0>,
    z4 = Z(g, t2) + A(g, t3),               Z = <Q| V T2 |0>, the disconnected product,
    D4 = e_i + e_j + e_k + e_l - e_a - e_b - e_c - e_d

(zccsdt.cc.ccsdtq_ref.quadruples) is a real function of t2, t3, of the
antisymmetrized integrals g = <pq||rs> and of the orbital energies; t1 and
the off-diagonal Fock matrix do not enter.  It is not holomorphic: the
left-hand side is conjugated.  Its first-order change is

    dE(Q) = Re sum [ conj(y) dz + conj(w) dr ] - sum Re(conj(w) y) dD,
    y = r4 / D4,   w = z4 / D4,

and dz, dr are linear in dt and dg without conjugation.  So with
"left-hand quadruples" conj(y) and conj(w):

    dE(Q) = Re [ sum_(i<j,a<b) s2 dt2 + sum_(i<j<k,a<b<c) s3 dt3
                 + sum_pqrs d2[p,q,r,s] dg[p,q,r,s] ] + sum_p de[p] d e_p,

    s3 = <0| L(y+w) [V, tau3] |0>
    s2 = <0| L(y) (V tau2)_disconnected |0> + <0| L(w) [[V,T2], tau2] |0>
    d2:  conj(y+w) t3 on <cd||el> and <md||kl>, conj(y) t2 on <cd||kl>,
         conj(w) t2 t2 on <bc||ef>, <mn||jk>, <mb||ek>
    de[i] = -sum_(Q with i) Re(conj(w) y),   de[a] = +sum_(Q with a) Re(conj(w) y).

The Lagrangian of CCSDT(Q) is the real function

    F = Re L_CCSDT(t, lambda; f, g) + E(Q)(t2, t3; g, e),

L_CCSDT = <0| (1 + Lambda) exp(-T) H exp(T) |0> holomorphic (lambda_ref).
It is stationary in t when

    dL_CCSDT/dt_k + s_k = 0,   s1 = 0:

the Lambda equations of CCSDT with the constant sources s2, s3 in the
residuals of rank 2 and 3 (lambda_ref.kernel(sources=)).  For real
quantities s is the derivative of E(Q); for complex ones twice its Wirtinger
derivative (t and conj(t) independent), as in the CCSD(T) code (zccsd.lam_t).

The first derivative of F with respect to a perturbation of the Hamiltonian
is then that of its explicit dependence on the integrals: the densities of
the CCSDT Lagrangian with this lambda (lambda_ref.densities, made Hermitian:
the real part), d2 and de of (Q).  The orbitals follow the perturbation as
canonical Hartree-Fock orbitals: the rotations between occupied and virtual
spin orbitals through the Z-vector equations, those among occupied and among
virtual ones - on which F depends through D4 - through the Fock matrix
staying diagonal, y_pq = G_pq / (e_q - e_p) (zccsd.zvector._relax, as for
CCSD(T) and for frozen spin orbitals).  relaxed_rdm1 does that from full
arrays over all spin orbitals.

Everything is evaluated from the term table of zccsdt.cc.ccsdtq_terms, one
occupied quadruple at a time with dense arrays of the virtual indices; the
derivative of a term coefficient * A(product) contracted with an
antisymmetric left-hand quadruple L is coefficient * sum L product with the
factor left out.
'''

from dataclasses import dataclass

import numpy as np

from zccsdt.cc import ccsdtq_ref as R, ccsdtq_terms as Q, lambda_ref as LR

EXT = 'ijkl'
# (terms, which left-hand quadruples multiply them: conj(y), conj(w))
PARTS = (('Z4.T2', (1, 0)), ('R4: [V,T3]', (1, 1)), ('R4: 1/2 [[V,T2],T2]', (0, 1)))


@dataclass
class Gradient:
    e_q: float                  # E(Q)
    s2: np.ndarray              # [i,j,a,b], antisymmetric: the source of the Lambda_2 equation
    s3: np.ndarray              # [i,j,k,a,b,c]: that of the Lambda_3 equation
    d2: np.ndarray              # [p,q,r,s] over all spin orbitals: dE = Re sum d2 d<pq||rs>
    de: np.ndarray              # [p]: dE = sum de d e_p


def _terms(name):
    return Q.LEFT[name][2] if name in Q.LEFT else Q.GROUPS[name][4]


def _operand(kind, ix, arrays, nocc, occ):
    '''The array of a factor: its block of f or v, the external occupied
    letters on the spin orbitals ``occ``.'''
    a = arrays[kind]
    if kind in ('v', 'f'):
        n = a.shape[0]
        a = a[tuple(slice(0, nocc) if c in Q.OCC else slice(nocc, n) for c in ix)]
    for axis, c in enumerate(ix):
        if c in EXT:
            a = np.take(a, occ, axis=axis)
    return a


def _where(kind, ix, nocc, nvir, occ):
    '''The elements of the full array of a factor that _operand takes.'''
    off = nocc if kind in ('v', 'f') else 0
    return np.ix_(*[np.asarray(occ) if c in EXT else np.arange(nocc) if c in Q.OCC
                    else off + np.arange(nvir) for c in ix])


def quadruples_gradient(t2, t3, g, mo_energy, nocc):
    '''E(Q) and its derivatives (see the module) for dense antisymmetric
    t2[i,j,a,b], t3[i,j,k,a,b,c], g[p,q,r,s] = <pq||rs> over all spin
    orbitals and the orbital energies: a Gradient.'''
    n = g.shape[0]
    nvir = n - nocc
    arrays = {'v': g, 't2': t2, 't3': t3}
    out = {'v': np.zeros((n,) * 4, dtype=complex), 't2': np.zeros(t2.shape, dtype=complex),
           't3': np.zeros(t3.shape, dtype=complex)}
    de = np.zeros(n)
    eo, ev = mo_energy[:nocc], mo_energy[nocc:]
    vir = Q.tuples(nvir, 4)
    e_q = 0.0
    for occ in Q.tuples(nocc, 4):
        occ = tuple(int(x) for x in occ)
        x = {name: R._block(_terms(name), arrays, nocc, 4, occ) for name, _ in PARTS}
        d = eo[list(occ)].sum() - ev[vir].sum(axis=1)
        r = x['R4: [V,T3]'] + x['R4: 1/2 [[V,T2],T2]']
        z = x['Z4.T2'] + x['R4: [V,T3]']
        y, w = r / d, z / d
        e_q += np.vdot(z, y).real
        # the orbital energies: -Re(conj(w) y) dD
        c = (np.conj(w) * y).real
        de[list(occ)] -= c.sum()
        np.add.at(de, nocc + vir, c[:, None])
        # the left-hand quadruples on [4,4,4,4,v,v,v,v]
        left = [Q.expand(np.conj(u)[None, :], 4, nvir, 4) for u in (y, w)]
        for name, (cy, cw) in PARTS:
            lq = cy * left[0] + cw * left[1]
            for coef, factors in _terms(name):
                ops = [_operand(kind, ix, arrays, nocc, occ) for kind, ix in factors]
                for p, (kind, ix) in enumerate(factors):
                    rest = [(f[1], a) for q, (f, a) in enumerate(zip(factors, ops)) if q != p]
                    sub = ','.join(['ijklabcd'] + [s for s, _ in rest]) + '->' + ix
                    val = np.einsum(sub, lq, *[a for _, a in rest], optimize=True)
                    out[kind][_where(kind, ix, nocc, nvir, occ)] += float(coef) * val
    # dE = Re sum (over all elements) out dt: as sums over the unique elements
    s2, s3, d2 = Q.asym(out['t2'], 2), Q.asym(out['t3'], 3), out['v']
    if not any(np.iscomplexobj(x) for x in (t2, t3, g)):
        s2, s3, d2 = s2.real.copy(), s3.real.copy(), d2.real.copy()
    return Gradient(float(e_q), s2, s3, d2, de)


def energy(t2, t3, g, mo_energy, nocc):
    '''E(Q) of ccsdtq_ref.quadruples for any arguments (complex amplitudes,
    integrals that are not Hermitian): the function that quadruples_gradient
    differentiates.'''
    return R.quadruples(t2, t3, g, mo_energy, nocc)[1]


def fock_density(t2, t3, g, mo_energy, nocc):
    '''Gamma[p,q], the derivative of (Q) with respect to the Fock matrix in
    the form that does not need canonical orbitals (y the solution of
    D y = r with the whole occupied-occupied and virtual-virtual blocks of f
    in D; the same energy for canonical orbitals): dE = Re sum Gamma[p,q]
    df[p,q],

        Gamma[a,e] = sum conj(w[ijkl,abcd]) y[ijkl,ebcd]     (i<j<k<l, b<c<d),
        Gamma[m,i] = -sum conj(w[ijkl,abcd]) y[mjkl,abcd]    (j<k<l, a<b<c<d),

    its diagonal is Gradient.de.  That form of (Q) does not change when the
    occupied or the virtual spin orbitals are rotated among themselves
    with t, so the orbital gradient of CCSDT(Q) among them is

        G[p,q] = (e_q - e_p) (Gamma[q,p] + conj(Gamma[p,q])) / 2:

    the rotation y_pq = G_pq / (e_q - e_p) of zccsd.zvector._relax is the
    Hermitian part of Gamma, transposed.  From all quadruples as full
    arrays: for a few spin orbitals.'''
    n = g.shape[0]
    nvir = n - nocc
    arrays = {'v': g, 't2': t2, 't3': t3}
    x, d = R._pieces([name for name, _ in PARTS], 4, arrays, mo_energy, nocc)
    y = Q.expand((x['R4: [V,T3]'] + x['R4: 1/2 [[V,T2],T2]']) / d, nocc, nvir, 4)
    w = Q.expand((x['Z4.T2'] + x['R4: [V,T3]']) / d, nocc, nvir, 4)
    out = np.zeros((n, n), dtype=complex)
    out[nocc:, nocc:] = np.einsum('ijklabcd,ijklebcd->ae', w.conj(), y) / 144
    out[:nocc, :nocc] = -np.einsum('ijklabcd,mjklabcd->mi', w.conj(), y) / 144
    return out


# ------------------------------------------------------------ relaxed density

@dataclass
class Relaxed:
    relaxed: np.ndarray         # D[p,q] = <q+ p>, Hermitian, all spin orbitals
    unrelaxed: np.ndarray
    z: np.ndarray
    gradient: np.ndarray        # G[y,x] of zccsd.zvector, all spin orbitals
    e_corr: float               # E_corr(CCSDT)
    e_q: float
    amps: list
    lams: list
    q: Gradient = None


def full_densities(d1, d2, nfrozen, n):
    '''(D1[p,q], D2[p,q,r,s]) over ``n`` spin orbitals of which the first
    ``nfrozen`` are frozen occupied ones, from those of the active ones
    that follow them (lambda_ref.densities, reference included): the frozen
    ones occupied, E = sum h D1 + 1/4 sum <pq||rs> D2
    with the energy of the frozen ones and their potential in it.  Spin
    orbitals after the active ones (dropped virtual) are empty.'''
    na = d1.shape[0]
    act = slice(nfrozen, nfrozen + na)
    D1 = np.zeros((n, n), dtype=complex)
    D1[act, act] = d1
    D2 = np.zeros((n,) * 4, dtype=complex)
    D2[act, act, act, act] = d2
    if nfrozen:
        P = np.zeros((n, n))
        P[np.arange(nfrozen), np.arange(nfrozen)] = 1.0
        x = np.einsum('pr,qs->pqrs', D1 + 0.5 * P, P)
        x = x - x.transpose(1, 0, 2, 3)
        D2 += x - x.transpose(0, 1, 3, 2)
        D1 += P
    return D1, D2


def relaxed_rdm1(mf, mo_coeff, nfrozen=0, ndrop=0, with_q=True, amps=None, tol=1e-11,
                 conv_tol_normt=1e-11, verbose=None):
    '''The relaxed one-particle density of CCSDT(Q) (``with_q`` False: of
    CCSDT) for the SCF ``mf`` and its canonical spin orbitals or spinors
    ``mo_coeff`` (all of them, occupied first), the first ``nfrozen``
    frozen and the last ``ndrop`` virtual ones left out: from full arrays
    over all spin orbitals, for small molecules.  ``amps``: converged dense
    [t1, t2, t3] of the active spin orbitals (default: ccsdtq_ref.kernel).
    Returns a Relaxed; an expectation value is sum V[p,q] D[q,p].'''
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
    q = quadruples_gradient(amps[1], amps[2], ga, ea, oa) if with_q else None
    rl = LR.kernel(fa, ga, ea, oa, amps, conv_tol_normt=conv_tol_normt,
                   sources=[0, q.s2, q.s3] if with_q else None)
    if not rl.converged:
        raise RuntimeError('the Lambda equations did not converge')
    D1, D2 = full_densities(*LR.densities(amps, rl.lams, oa), nfrozen, n)
    dm1, dm2 = LR.pyscf_rdms(D1, D2)
    h = fock - np.einsum('piqi->pq', g[:, :o, :, :o])
    eri = 0.5 * g.transpose(0, 2, 1, 3)                 # (pq|rs) - (ps|rq) = <pr||qs>
    G = zvector.orbital_gradient(h, eri, dm1, dm2)
    if with_q:
        d2 = np.zeros((n,) * 4, dtype=complex)
        d2[a, a, a, a] = q.d2
        d1 = np.zeros((n, n), dtype=complex)
        d1[np.arange(n)[a], np.arange(n)[a]] = q.de
        Gq = zvector.gradient_nh(zvector.t_fock(fock, o), g, d2, d1, o)
        G = G + 0.5 * (Gq - Gq.conj().T)
        dm1 = dm1 + d1
    spaces = (np.arange(nfrozen, o), np.arange(nfrozen), np.arange(o, n - ndrop), np.arange(n - ndrop, n))
    relaxed, z = zvector._relax(mf, mf, np.asarray(mo_coeff), mo_e, o, G, dm1, spaces, with_q,
                                zvector.DEGENERATE_DENSE, tol, verbose)
    return Relaxed(relaxed, dm1, z, G, e_corr, q.e_q if with_q else 0.0, amps, rl.lams, q)
