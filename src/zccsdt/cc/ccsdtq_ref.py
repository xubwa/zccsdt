'''Dense reference implementations on spin orbitals or spinors, real or
complex: CCSDT and CCSDTQ, and the perturbative corrections (Q) and (P).

Everything is evaluated term by term from the table of
zccsdt.cc.ccsdtq_terms on full arrays t2[i,j,a,b], t3[i,j,k,a,b,c],
t4[i,j,k,l,a,b,c,d]: for a few dozen spin orbitals, as the reference of the
packed implementations and of their equations.  The integrals are the full
antisymmetrized tensor <pq||rs>.

The corrections (canonical orbitals, bare integrals; T1 does not enter),
sums over the unique quadruples and pentuples i<j<.., a<b<.., D the
differences of the orbital energies:

    E[Q] = sum conj(z4) r4 / D4,   r4 = <Q| [V,T3] + 1/2 [[V,T2],T2] |0>,
                                   z4 = <Q| V T2 |0>  (the disconnected product),
    E(Q): z4 = <Q| V (T2 + T3) |0>,

    E(P) = sum conj(z5) r5 / D5,   r5 = <P| [V,T4] + [[V,T2],T3]
                                           + 1/6 [[[V,T2],T2],T2] |0>,
                                   z5 = <P| V T3 |0> + <P| V Y4 |0>_connected,
                                   Y4 = <Q| V T2 |0> / D4

with the amplitudes of CCSDT for (Q) and of CCSDTQ for (P).  (Q) is the
correction of Bomble, Stanton, Kallay and Gauss.  The left-hand side of (P)
was not taken from a paper but fixed against MRCC: its CCSDTQ[P] and
CCSDTQ(P) are printed equal, and of the candidates only this one - the
quadruples of the bra to second order, not T4^+ - gives its numbers, for
BH (-3.6747e-9 for MRCC's -3.674e-9, its energies having twelve decimals)
and for CH2 (5.218704e-8 for 5.218700e-8), both with STO-3G; T4^+ in the
place of Y4 gives -9.3e-10 and -5.15e-8, and T2 T2 / 2 gives -4.02e-9 for BH.
'''

from dataclasses import dataclass
from itertools import permutations

import numpy as np

from zccsdt.cc import ccsdt, ccsdtq_terms as Q
from zccsdt.cc.bruteforce import build_g
from zccsdt.cc.diis import DIIS


def integrals(eris):
    '''(fock, <pq||rs>, orbital energies, nocc) of an ERIs container.'''
    nocc, nmo = eris.nocc, eris.fock.shape[0]
    return np.asarray(eris.fock), build_g(eris, nocc, nmo), np.asarray(eris.mo_energy), nocc


def residuals(amps, fock, g, nocc):
    '''[R1, .., Rn] for amps = [t1, t2, t3] (CCSDT) or [t1, .., t4] (CCSDTQ),
    all dense.'''
    t1 = amps[0]
    ge, fdr = ccsdt._t1_dress(g, fock, t1, nocc, g.shape[0])
    arrays = {'v': ge, 'f': fdr}
    arrays.update({'t%d' % (k + 1): a for k, a in enumerate(amps) if k})
    with_t4 = len(amps) == 4
    return [Q.residual(k + 1, arrays, nocc, with_t4=with_t4) for k in range(len(amps))]


def denominator(mo_energy, nocc, k):
    '''D[i.., a..] = e_i + .. - e_a - .. as a dense array of rank k.'''
    eo, ev = mo_energy[:nocc], mo_energy[nocc:]
    d = 0
    for x in range(k):
        d = d + eo.reshape([-1 if y == x else 1 for y in range(2 * k)])
        d = d - ev.reshape([-1 if y == k + x else 1 for y in range(2 * k)])
    return d


def energy(t1, t2, fock, g, nocc):
    o, v = slice(0, nocc), slice(nocc, None)
    e = np.einsum('ia,ia', fock[o, v], t1)
    e += 0.25 * np.einsum('ijab,ijab', g[o, o, v, v], t2)
    e += 0.5 * np.einsum('ijab,ia,jb', g[o, o, v, v], t1, t1)
    return e


@dataclass
class Result:
    e_corr: float
    converged: bool
    ncycle: int
    amps: list                  # [t1, t2, t3(, t4)], dense


def kernel(fock, g, mo_energy, nocc, method='ccsdtq', conv_tol=1e-11, conv_tol_normt=1e-9,
           max_cycle=200, diis_space=8, verbose=False):
    '''CCSDT or CCSDTQ by Jacobi steps with DIIS, from the MP2 amplitudes.'''
    n = {'ccsdt': 3, 'ccsdtq': 4}[method]
    nmo = g.shape[0]
    nvir = nmo - nocc
    o, v = slice(0, nocc), slice(nocc, None)
    dtype = np.result_type(fock, g)
    den = [denominator(mo_energy, nocc, k + 1) for k in range(n)]
    amps = [np.zeros((nocc,) * (k + 1) + (nvir,) * (k + 1), dtype=dtype) for k in range(n)]
    amps[0] = fock[v, o].T / den[0]
    amps[1] = g[v, v, o, o].transpose(2, 3, 0, 1) / den[1]
    e_old = energy(amps[0], amps[1], fock, g, nocc)
    adiis = DIIS(space=diis_space, incore=True)
    conv = False

    def parts(a):
        return [a[0].ravel()] + [Q.unique(x, k + 1).ravel() for k, x in enumerate(a) if k]
    for cycle in range(max_cycle):
        res = residuals(amps, fock, g, nocc)
        new = [t + r / d for t, r, d in zip(amps, res, den)]
        normt = sum(np.linalg.norm(a - b) for a, b in zip(new, amps))
        ps = adiis.update(parts(new))
        amps = [ps[0].reshape(nocc, nvir)] + [
            Q.expand(p.reshape(len(Q.tuples(nocc, k + 1)), -1), nocc, nvir, k + 1)
            for k, p in enumerate(ps) if k]
        e = energy(amps[0], amps[1], fock, g, nocc)
        if verbose:
            print('cycle %3d  E_corr = %.13f  dE = %.2e  |dt| = %.2e' % (cycle + 1, e.real, (e - e_old).real, normt))
        if abs(e - e_old) < conv_tol and normt < conv_tol_normt:
            conv = True
            e_old = e
            break
        e_old = e
    adiis.close()
    return Result(float(np.real(e_old)), conv, cycle + 1, amps)


# -------------------------------------------------- perturbative corrections

def _block(terms, arrays, nocc, k, occ):
    '''The antisymmetrized sum of terms for one sorted occupied tuple and
    all sorted virtual tuples: [C(nv,k)].'''
    x = Q.total(terms, arrays, nocc, k, occ=occ)        # [k]*k + [nv]*k
    y = 0
    for p in permutations(range(k)):
        y = y + Q._sign(p) * x[p]
    vir = Q.tuples(y.shape[-1], k)
    out = 0
    for p in permutations(range(k)):
        out = out + Q._sign(p) * y[tuple(vir[:, p[s]] for s in range(k))]
    return out


def _pieces(names, k, arrays, mo_energy, nocc):
    '''{name: [C(no,k), C(nv,k)]} for groups / left-hand quantities, and the
    denominator there.'''
    occ = Q.tuples(nocc, k)
    out = {}
    for name in names:
        terms = Q.LEFT[name][2] if name in Q.LEFT else Q.GROUPS[name][4]
        out[name] = np.array([_block(terms, arrays, nocc, k, tuple(o)) for o in occ])
    eo, ev = mo_energy[:nocc], mo_energy[nocc:]
    vir = Q.tuples(len(ev), k)
    d = eo[occ].sum(axis=1)[:, None] - ev[vir].sum(axis=1)[None, :]
    return out, d


def quadruples(t2, t3, g, mo_energy, nocc):
    '''(E[Q], E(Q)) from converged CCSDT amplitudes.'''
    arrays = {'v': g, 't2': t2, 't3': t3}
    x, d = _pieces(('R4: [V,T3]', 'R4: 1/2 [[V,T2],T2]', 'Z4.T2', 'Z4.T3'), 4, arrays, mo_energy, nocc)
    r = (x['R4: [V,T3]'] + x['R4: 1/2 [[V,T2],T2]']) / d
    e5 = np.vdot(x['Z4.T2'], r).real
    return float(e5), float(e5 + np.vdot(x['Z4.T3'], r).real)


def pentuples(t2, t3, t4, g, mo_energy, nocc, parts=False):
    '''E(P) from converged CCSDTQ amplitudes.  ``parts``: the energies
    {(left, right)} of the two parts of z5 with the three groups of r5.'''
    nvir = g.shape[0] - nocc
    arrays = {'v': g, 't2': t2, 't3': t3, 't4': t4}
    right = ('R5: [V,T4]', 'R5: [[V,T2],T3]', 'R5: 1/6 [[[V,T2],T2],T2]')
    x, d = _pieces(right + ('Z5.T3',), 5, arrays, mo_energy, nocc)
    z4, d4 = _pieces(('Z4.T2',), 4, arrays, mo_energy, nocc)
    y4 = Q.expand(z4['Z4.T2'] / d4, nocc, nvir, 4)
    left = {'V T3': x['Z5.T3'],
            'V Y4': _pieces(('Z5.T4',), 5, dict(arrays, t4=y4), mo_energy, nocc)[0]['Z5.T4']}
    out = {(a, b): float(np.vdot(z, x[b] / d).real) for a, z in left.items() for b in right}
    return out if parts else sum(out.values())
