'''Dense reference of the left-hand side of CCSDT on spin orbitals or
spinors, real or complex: the Lambda equations, the one- and two-particle
densities, the orbital gradient, and the CCSDT(Q)_Lambda energy.

Everything is evaluated term by term from the tables of
zccsdt.cc.lambda_terms on full arrays, like zccsdt.cc.ccsdtq_ref: for a
few dozen spin orbitals, as the reference of an implementation on packed
amplitudes.  fock[p,q] and g[p,q,r,s] = <pq||rs> are the bare ones over all
spin orbitals (ccsdtq_ref.integrals), amps = [t1, t2, t3] and lams =
[l1, l2, l3] dense antisymmetric arrays [i.., a..].

Conventions.  Lambda_k = (1/k!)^2 sum l_k[i.., a..] i+ j+ .. b a, the left
state is <0| (1 + Lambda) exp(-T): l1[i,a] and l2[i,j,a,b] are those of
pyscf.cc.gccsd_lambda and of zccsdt.zccsd.lam, conj(t) in first order.  The
densities are the response densities at fixed orbitals,

    d1[p,q]     = <0| (1 + Lambda) exp(-T) p+ q exp(T) |0>,
    d2[p,q,r,s] = <0| (1 + Lambda) exp(-T) p+ q+ s r exp(T) |0>,

not Hermitian, with the reference in them:

    E = sum h[p,q] d1[p,q] + 1/4 sum <pq||rs> d2[p,q,r,s],
    dE/dx = sum (dh/dx)[p,q] d1[p,q] + 1/4 sum (d<pq||rs>/dx) d2[p,q,r,s]

for a perturbation of the integrals at fixed orbitals.  pyscf.cc.gccsd_rdm
and zccsdt.zccsd (rdm, zvector) have D[p,q] = <q+ p> = d1[q,p] and
d[p,q,r,s] = <p+ r+ s q> = d2[p,r,q,s], both made Hermitian (pyscf_rdms).

They are made in the dressed basis: the tables give the coefficients of
the dressed f and <pq||rs> in the Lagrangian (no t1 in them), the
reference and the Fock matrix are put back, and the transformation that
dresses the integrals, h~ = x h y^T, takes them to the bare ones,
d1 = x^T d1~ y (four times for d2).

CCSDT(Q)_Lambda (canonical Hartree-Fock orbitals, bare integrals; t1 and
l1 do not enter), sums over the unique quadruples:

    E(Q)_L = sum z4 r4 / D4,   r4 = <Q| [V,T3] + 1/2 [[V,T2],T2] |0>,
                               z4 = <0| (Lambda_2 + Lambda_3) V |Q>,

the (Q) correction of ccsdtq_ref.quadruples with Lambda_2 and Lambda_3 in
the place of T2^+ and T3^+: Lambda_2 V the disconnected product and
Lambda_3 V the connected one.  E[Q]_L is the part of Lambda_2.
docs/lambda.md has the comparison with MRCC and CFOUR.
'''

from dataclasses import dataclass

import numpy as np

from zccsdt.cc import ccsdt, ccsdtq_ref as R, ccsdtq_terms as Q, lambda_terms as L
from zccsdt.cc.diis import DIIS


def _arrays(fock, g, nocc, amps, lams=None):
    '''The arrays of the tables: dressed f and v, the amplitudes.'''
    ge, fdr = ccsdt._t1_dress(g, fock, amps[0], nocc, g.shape[0])
    arrays = {'v': ge, 'f': fdr, 't2': amps[1], 't3': amps[2]}
    if lams is not None:
        arrays.update({'l1': lams[0], 'l2': lams[1], 'l3': lams[2]})
    return arrays


def residuals(lams, amps, fock, g, nocc, arrays=None, sources=None):
    '''The residuals of the three Lambda equations, dL/dt_k: dense.
    ``sources`` = [s1, s2, s3]: constant terms added to them (the
    derivatives of a correction to the energy with respect to t_k: those of
    (Q) in zccsdt.cc.qgrad_ref).'''
    if arrays is None:
        arrays = _arrays(fock, g, nocc, amps)
    arrays = dict(arrays, l1=lams[0], l2=lams[1], l3=lams[2])
    res = [L.residual(k, arrays, nocc) for k in (1, 2, 3)]
    if sources is not None:
        res = [r + s for r, s in zip(res, sources)]
    return res


@dataclass
class Result:
    converged: bool
    ncycle: int
    lams: list                  # [l1, l2, l3], dense


def kernel(fock, g, mo_energy, nocc, amps, conv_tol_normt=1e-9, max_cycle=200, diis_space=8,
           verbose=False, sources=None):
    '''The Lambda equations of CCSDT for the converged amplitudes
    ``amps`` = [t1, t2, t3], by Jacobi steps with DIIS from conj(t).
    ``sources``: see residuals.'''
    nvir = g.shape[0] - nocc
    arrays = _arrays(fock, g, nocc, amps)
    den = [R.denominator(mo_energy, nocc, k) for k in (1, 2, 3)]
    lams = [np.array(t, dtype=complex).conj() for t in amps]
    adiis = DIIS(space=diis_space, incore=True)
    conv = False

    def parts(a):
        return [a[0].ravel()] + [Q.unique(x, k + 1).ravel() for k, x in enumerate(a) if k]
    for cycle in range(max_cycle):
        res = residuals(lams, amps, fock, g, nocc, arrays, sources)
        new = [x + r / d for x, r, d in zip(lams, res, den)]
        norm = sum(np.linalg.norm(a - b) for a, b in zip(new, lams))
        if verbose:
            print('cycle %3d  |dl| = %.2e' % (cycle + 1, norm))
        if norm < conv_tol_normt:
            lams, conv = new, True
            break
        ps = adiis.update(parts(new))
        lams = [ps[0].reshape(nocc, nvir)] + [
            Q.expand(p.reshape(len(Q.tuples(nocc, k + 1)), -1), nocc, nvir, k + 1)
            for k, p in enumerate(ps) if k]
    adiis.close()
    if not (np.iscomplexobj(fock) or np.iscomplexobj(g) or any(np.iscomplexobj(t) for t in amps)
            or (sources is not None and any(np.iscomplexobj(s) and abs(np.imag(s)).max() > 0 for s in sources))):
        lams = [x.real.copy() for x in lams]
    return Result(conv, cycle + 1, lams)


def lambda_ccsdt(eris, t1, t2, t3, **kwargs):
    '''(l1, l2, l3) of CCSDT for the integrals of an ERIs container and
    converged dense amplitudes.'''
    fock, g, mo_e, nocc = R.integrals(eris)
    r = kernel(fock, g, mo_e, nocc, [t1, t2, t3], **kwargs)
    if not r.converged:
        raise RuntimeError('the Lambda equations of CCSDT did not converge')
    return r.lams


# ----------------------------------------------------------------- densities

def _dressing(t1, nocc, n):
    '''(x, y) of ccsdt._t1_dress: h~ = x h y^T.'''
    x = np.eye(n, dtype=complex)
    x[nocc:, :nocc] -= t1.T
    y = np.eye(n, dtype=complex)
    y[:nocc, nocc:] += t1
    return x, y


def dressed_densities(amps, lams, nocc):
    '''(d1~, d2~): the coefficients of the dressed one-electron integrals
    h~[p,q] and of the dressed <pq||rs> in L = sum h~ d1~ + 1/4 sum
    <pq||rs>~ d2~ - the densities <0| (1 + Lambda) exp(-T2 - T3) p+ q
    exp(T2 + T3) |0> and the same of p+ q+ s r.'''
    n = sum(amps[0].shape)
    arrays = {'v': np.empty((n, 0)), 't2': amps[1], 't3': amps[2],
              'l1': lams[0], 'l2': lams[1], 'l3': lams[2]}
    g1, g2 = L.density(arrays, nocc)
    occ = np.zeros(n)
    occ[:nocc] = 1.0
    d1 = g1 + np.diag(occ)
    # the reference, and the Fock matrix f~ = h~ + sum_i <pi||qi>~
    x = np.einsum('pr,qs->pqrs', g1 + 0.5 * np.diag(occ), np.diag(occ))
    x = x - x.transpose(1, 0, 2, 3)
    d2 = g2 + x - x.transpose(0, 1, 3, 2)
    return d1, d2


def densities(amps, lams, nocc):
    '''(d1[p,q], d2[p,q,r,s]) = <p+ q>, <p+ q+ s r> of the left and right
    CCSDT states (see the module), over all spin orbitals.'''
    n = sum(amps[0].shape)
    x, y = _dressing(amps[0], nocc, n)
    d1, d2 = dressed_densities(amps, lams, nocc)
    d1 = x.T @ d1 @ y
    d2 = np.einsum('pqrs,pt,qu->turs', d2, x, x)
    d2 = np.einsum('turs,rv,sw->tuvw', d2, y, y)
    return d1, d2


def pyscf_rdms(d1, d2, hermitian=True):
    '''(D[p,q] = <q+ p>, d[p,q,r,s] = <p+ r+ s q>) of pyscf.cc.gccsd_rdm
    and zccsdt.zccsd, made Hermitian (the real part of every expectation
    value of a Hermitian operator stays).'''
    dm1 = d1.T
    dm2 = d2.transpose(0, 2, 1, 3)
    if hermitian:
        dm1 = 0.5 * (dm1 + dm1.conj().T)
        dm2 = 0.5 * (dm2 + dm2.conj().transpose(1, 0, 3, 2))
    return dm1, dm2


def energy(fock, g, nocc, d1, d2):
    '''sum h d1 + 1/4 sum <pq||rs> d2: the electronic energy (of the
    Lagrangian) from the densities.'''
    h = fock - np.einsum('piqi->pq', g[:, :nocc, :, :nocc])
    return np.einsum('pq,pq', h, d1) + 0.25 * np.einsum('pqrs,pqrs', g, d2)


def one_index(h, v, d1, d2):
    '''G[x,y] = sum h'[p,q] d1[p,q] + 1/4 sum v'[p,q,r,s] d2[p,q,r,s] with
    h' = [h, x+ y], the integrals one-index transformed by x+ y: a ket
    index y takes the values of x, a bra index x minus those of y.
    G[a,i] is dL/dt1[i,a] with the dressed integrals and densities;
    for Hermitian integrals the orbital gradient of the Hermitian
    densities, zccsdt.zccsd.zvector.orbital_gradient, is
    Gz[y,x] = -1/2 (G[x,y] - conj(G[y,x])).'''
    G = h.T @ d1 - d1 @ h.T
    G += 0.5 * (np.einsum('pqxs,pqys->xy', v, d2) - np.einsum('yqrs,xqrs->xy', v, d2))
    return G


def orbital_gradient(fock, g, nocc, amps, lams, dressed=False):
    '''Gnh[a,i] = <0| (1 + Lambda) exp(-T) [a+_i a_a, H] exp(T) |0>: the
    Lagrangian with the integrals one-index transformed by the
    de-excitation a+_i a_a, at fixed t and lambda - Gnh[a,i] of
    zccsdt.zccsd.zvector, a bra index i taking the values of a and a ket
    index a minus those of i.  For converged t and lambda the other half,
    Gnh[i,a], and the blocks among occupied and among virtual spin
    orbitals vanish, and the orbital gradient of the Hermitian densities
    (zvector.orbital_gradient with those of pyscf_rdms) is
    G[a,i] = 1/2 Gnh[a,i], G[i,a] = -conj(G[a,i]).

    ``dressed``: from the table of the dressed basis alone
    (lambda_terms.GRADIENT), which is the same for converged t and lambda -
    the difference is the one-index transformations among occupied and
    among virtual spin orbitals and by a+_a a_i, contracted with t1, and
    those vanish with the residuals.  Else from the full densities.'''
    if dressed:
        return L.gradient(_arrays(fock, g, nocc, amps, lams), nocc)
    d1, d2 = densities(amps, lams, nocc)
    h = fock - np.einsum('piqi->pq', g[:, :nocc, :, :nocc])
    return -one_index(h, g, d1, d2)[:nocc, nocc:].T


# ------------------------------------------------------------- (Q)_Lambda

def quadruples_lambda(t2, t3, l2, l3, g, mo_energy, nocc):
    '''(E[Q]_L, E(Q)_L) from converged CCSDT amplitudes and Lambda.'''
    right = {'v': g, 't2': t2, 't3': t3}
    x, d = R._pieces(('R4: [V,T3]', 'R4: 1/2 [[V,T2],T2]'), 4, right, mo_energy, nocc)
    r = (x['R4: [V,T3]'] + x['R4: 1/2 [[V,T2],T2]']) / d
    # <0| Lambda_n V |Q> = conj(<Q| V Lambda_n^+ |0>)
    left = {'v': g, 't2': np.conj(l2), 't3': np.conj(l3)}
    z = R._pieces(('Z4.T2', 'Z4.T3'), 4, left, mo_energy, nocc)[0]
    e5 = np.vdot(z['Z4.T2'], r).real
    return float(e5), float(e5 + np.vdot(z['Z4.T3'], r).real)


def ccsdt_q_lambda(eris, t2, t3, l2, l3):
    '''The CCSDT(Q)_Lambda correction to the CCSDT energy.'''
    fock, g, mo_e, nocc = R.integrals(eris)
    return quadruples_lambda(t2, t3, l2, l3, g, mo_e, nocc)[1]
