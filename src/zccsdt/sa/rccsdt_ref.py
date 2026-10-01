'''Reference closed-shell spin-adapted CCSD and CCSDT on spatial orbitals.

Dense numpy implementation for validation and small systems: the residuals
are evaluated term by term from the equations of zccsdt.sa.derive (which are
checked against the spin-orbital code), on full spatial arrays.

Conventions (closed shell, RHF reference, T1-dressed formalism):

- V[p,q,r,s] = <pq|rs> (physicist, real spatial orbitals), occupied first.
- t1[i,a]; t2[i,j,a,b] = t_ij^ab = t2[j,i,b,a]; t3[i,j,k,a,b,c] = t_ijk^abc,
  invariant under simultaneous permutations of the pairs (ia), (jb), (kc).
  The spin-orbital amplitudes follow from zccsdt.sa.somap.
- T3 has a redundant component (constant over the permutations of abc for
  fixed ijk) that does not enter the wavefunction; it is projected out of
  every update, so t3 never contains it.
'''

from dataclasses import dataclass
from itertools import permutations

import numpy as np

from zccsdt.sa import derive
from zccsdt.sa.spinfree import evaluate


@dataclass
class RHFIntegrals:
    nocc: int                  # active doubly occupied spatial orbitals
    fock: np.ndarray           # (n, n) MO Fock matrix, active space
    V: np.ndarray              # (n, n, n, n) <pq|rs>, active space
    mo_energy: np.ndarray
    e_hf: float
    e_nuc: float

    @property
    def nvir(self):
        return self.fock.shape[0] - self.nocc


def from_scf(mf, ncore=0):
    '''Spatial MO integrals from a PySCF RHF object; ``ncore`` frozen core
    (spatial) orbitals.'''
    from pyscf import ao2mo
    occ = np.asarray(mf.mo_occ) > 0
    C = np.asarray(mf.mo_coeff)[:, ncore:]
    nocc = int(occ.sum()) - ncore
    n = C.shape[1]
    eri = ao2mo.general(mf.mol, (C, C, C, C), compact=False).reshape(n, n, n, n)
    V = np.ascontiguousarray(eri.transpose(0, 2, 1, 3))          # (pr|qs) -> <pq|rs>
    fock = C.T @ mf.get_fock() @ C
    return RHFIntegrals(nocc=nocc, fock=fock, V=V, mo_energy=fock.diagonal().copy(),
                        e_hf=float(mf.e_tot), e_nuc=float(mf.energy_nuc()))


def dress(V, F, t1, nocc):
    '''T1-dressed <pq|rs> and Fock: bra indices transform with X = 1 - t1^T
    (virtual picks up occupied), ket indices with Y = 1 + t1.'''
    n = F.shape[0]
    X = np.eye(n)
    X[nocc:, :nocc] -= t1.T
    Y = np.eye(n)
    Y[:nocc, nocc:] += t1
    Vd = np.einsum('PQRS,pP->pQRS', V, X, optimize=True)
    Vd = np.einsum('pQRS,qQ->pqRS', Vd, X, optimize=True)
    Vd = np.einsum('pqRS,rR->pqrS', Vd, Y, optimize=True)
    Vd = np.einsum('pqrS,sS->pqrs', Vd, Y, optimize=True)
    o, v = slice(0, nocc), slice(nocc, n)
    Fd = F + np.einsum('risa,ia->rs', 2 * V[:, o, :, v] - V[:, o, v, :].transpose(0, 1, 3, 2), t1)
    return Vd, X @ Fd @ Y.T


_TERMS = {}


def _terms(with_t3):
    if with_t3 not in _TERMS:
        imds = derive.spatial_intermediates()
        res = derive.spatial_residuals()
        if not with_t3:
            def drop(ts):
                return [(c, f) for c, f in ts if not any(n == 'T3' for n, _ in f)]
            imds = {k: drop(v) for k, v in imds.items()}
            res = {k: drop(v) for k, v in res.items() if k != 'R3'}
        slots = {name: s for name, (s, _) in derive.intermediate_terms().items()}
        _TERMS[with_t3] = (imds, res, slots)
    return _TERMS[with_t3]


def project_null(r3):
    '''Remove the redundant component: r - (1/6) sum_sigma r^{sigma(abc)}.'''
    s = sum(r3.transpose(0, 1, 2, 3 + p[0], 3 + p[1], 3 + p[2]) for p in permutations(range(3)))
    return r3 - s / 6


def residuals(t1, t2, t3, ints):
    '''Spatial residuals (R1, R2, R3); R3 is None for CCSD (t3 None).'''
    no, nv = ints.nocc, ints.nvir
    Vd, Fd = dress(ints.V, ints.fock, t1, no)
    imds, res, slots = _terms(t3 is not None)
    dims = {'o': no, 'v': nv}
    base = {'V': Vd, 'F': Fd, 'T2': t2}
    if t3 is not None:
        base['T3'] = t3
    arr = dict(base)
    for key, terms in imds.items():
        if t3 is None and key.split('.')[0] not in ('Fv', 'Fo', 'Wpp', 'Whh', 'Wr'):
            continue
        arr[key] = evaluate(terms, base, slots[key.split('.')[0]], dims)
    r1 = evaluate(res['R1'], arr, 'ia', dims)
    r2 = evaluate(res['R2'], arr, 'ijab', dims)
    r3 = evaluate(res['R3'], arr, 'ijkabc', dims) if t3 is not None else None
    return r1, r2, r3


def energy(t1, t2, ints):
    no = ints.nocc
    fov = ints.fock[:no, no:]
    W = ints.V[:no, :no, no:, no:]
    tau = t2 + np.einsum('ia,jb->ijab', t1, t1)
    return 2 * np.einsum('ia,ia', fov, t1) + \
        np.einsum('ijab,ijab', 2 * W - W.transpose(0, 1, 3, 2), tau)


@dataclass
class Result:
    e_corr: float
    e_tot: float
    converged: bool
    ncycle: int
    t1: np.ndarray
    t2: np.ndarray
    t3: np.ndarray


def kernel(ints, method='ccsdt', conv_tol=1e-10, conv_tol_normt=1e-7, max_cycle=200,
           diis_space=8, verbose=False):
    '''Solve closed-shell CCSD or CCSDT (Jacobi steps with DIIS).'''
    from zccsdt.cc.diis import DIIS
    no, nv = ints.nocc, ints.nvir
    e = ints.mo_energy
    eo, ev = e[:no], e[no:]
    d1 = eo[:, None] - ev[None, :]
    d2 = d1[:, None, :, None] + d1[None, :, None, :]
    with_t3 = method == 'ccsdt'
    t1 = np.zeros((no, nv))
    W = ints.V[:no, :no, no:, no:]
    t2 = W / d2
    t3 = None
    if with_t3:
        d3 = (eo[:, None, None, None, None, None] + eo[None, :, None, None, None, None]
              + eo[None, None, :, None, None, None] - ev[None, None, None, :, None, None]
              - ev[None, None, None, None, :, None] - ev[None, None, None, None, None, :])
        t3 = np.zeros((no,) * 3 + (nv,) * 3)
    adiis = DIIS(space=diis_space)
    e_old = energy(t1, t2, ints)
    conv = False
    for cycle in range(max_cycle):
        r1, r2, r3 = residuals(t1, t2, t3, ints)
        parts = [t1 + r1 / d1, t2 + r2 / d2]
        if with_t3:
            r3 = project_null(r3)
            parts.append(t3 + r3 / d3)
        normt = sum(float(np.linalg.norm(r)) for r in (r1, r2) + ((r3,) if with_t3 else ()))
        out = adiis.update(parts)
        t1 = out[0].reshape(no, nv)
        t2 = out[1].reshape(no, no, nv, nv)
        if with_t3:
            t3 = out[2].reshape((no,) * 3 + (nv,) * 3)
        e_new = energy(t1, t2, ints)
        if verbose:
            print('cycle %3d  E_corr = %.15f  dE = %.3e  |r| = %.3e' % (cycle + 1, e_new, e_new - e_old, normt))
        if abs(e_new - e_old) < conv_tol and normt < conv_tol_normt:
            conv = True
            e_old = e_new
            break
        e_old = e_new
    return Result(e_corr=e_old, e_tot=ints.e_hf + e_old, converged=conv, ncycle=cycle + 1,
                  t1=t1, t2=t2, t3=t3)
