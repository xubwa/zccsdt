'''A one-shot quadruples correction to the excitation energies of
EOM-EE-CCSDT (docs/eom.md, section 14): the first-order change of omega_k
under everything that CC4 adds to CCSDT, with the left and right vectors of
EOM-CCSDT,

    d omega_k = L_k dA(omega_k) R_k,        L_k R_k = 1.

The folded Jacobian of CC4 in the space of singles, doubles and triples
(docs/ccn.md, sections 4 and 11; cc4fly.sigma) differs from that of CCSDT by

  fold     A_PQ (omega - D_Q)^-1 A_QP: the quadruples of the state,
           r4 = ([V', T3] + [V~, R3] + [[V', T2], T2] / 2 + [[V~, R2], T2]) / (omega + e_ijkl - e_abcd),
           V' = [V~, R1], contracted into the doubles and triples rows with
           the integrals of H~;
  t4       the T4 of the ground state, ([V~, T3] + [[V~, T2], T2] / 2) / (e_ijkl - e_abcd),
           contracted with the integrals of H' = [H~, R1] (the derivative of
           the T4 terms of the doubles and triples equations along R1);
  ground   the change of the Jacobian of CCSDT with the amplitudes, which
           the T4 terms of the amplitude equations move: Res(T) + rho(T) = 0
           with rho the sums of T4 in the doubles and triples equations, so
           dT = - A^-1 rho to first order and

               L (dA/dT . dT) R = xi'' . dT = W . rho,    W A = - xi'',
               xi''_kappa = <0| L [[Hbar, tau_kappa], R] |0>:

           W = Zt - Lambda is the response vector of the density of the
           state (eom_density: State.zeta), so this term costs a dot
           product once that density is there.  In the same way
           Lambda . rho is the first-order change of the ground-state
           energy, a (Q)-like correction, and Zt . rho that of the energy
           of the state.

All three are of first order in the coupling to the quadruples and all
three are included: omega_k + fold + t4 + ground estimates the excitation
energy of CC4.  Without the left vector the first two are the Rayleigh
quotient of R_k (the estimate that docs/native.md, 19.3, found to
overshoot).

One evaluation costs what a sigma vector of the folded problem costs: four
passes over the quadruples (three drives and the ground-state T4 with H'),
and one more, once, for rho.  Spinors: semicanonical (the canonical ones of
the SCF are).  The quadruples are made on the packed block - in the sectors
of the (Q) code for the ground state and for the totally symmetric states
of a calculation in the blocks of the labels, in one process otherwise (a
state with a label shift; the states of the packed block, whose levels
hold components of several shifts).
'''

import time
from dataclasses import dataclass

import numpy as np

from zccsdt.cc import cc4fly, eom_density, t4engine
from zccsdt.timing import timer


@dataclass
class Correction:
    '''The one-shot quadruples correction of one state (Hartree).'''
    root: int
    omega: float                # of EOM-CCSDT
    fold: float                 # L (A_PQ (omega - D)^-1 A_QP) R
    t4: float                   # L (the T4 terms with H') R
    ground: float = None        # W . rho: the shift of the ground-state amplitudes (None: no response vector)
    rayleigh: float = None      # the first two with conj(R) / |R|^2 in the place of L
    seconds: float = 0.0

    @property
    def total(self):
        return self.fold + self.t4 + (self.ground or 0.0)

    @property
    def estimate(self):
        return self.omega + self.total


@dataclass
class Result:
    states: list                # the Correction of each root
    e_ground: float = None      # Lambda . rho: the first-order change of the ground-state energy
    seconds_rho: float = 0.0

    def text(self):
        out = []
        if self.e_ground is not None:
            out.append('ground state: Lambda . rho = %.10f Eh' % self.e_ground)
        for c in self.states:
            out.append('state %d  omega %.10f  fold %+.3e  t4 %+.3e  ground %s  total %+.3e  estimate %.10f Eh'
                       % (c.root + 1, c.omega, c.fold, c.t4, 'none' if c.ground is None else '%+.3e' % c.ground,
                          c.total, c.estimate))
        return out


def kernel(eris, t1, t2, t3, right, left, roots=None, lam=None, zetas=None, max_memory=None, workers=None,
           verbose=False, stdout=None):
    '''The Result for the states of ``left`` (an eom_left.Result of the
    eom.Result ``right``; ``roots``: positions in left.roots).  ``lam`` =
    (l1, l2, l3) of the ground state and ``zetas`` = [Zt per root]
    (eom_density.State.zeta, l3 in the layout of t3): the term of the
    ground-state amplitudes, left out (Correction.ground None) without
    them.'''
    import sys
    out = stdout or sys.stdout
    g = right.ground
    shift = right.shift
    dtype = g.dtype
    fock = np.asarray(eris.fock)
    no, nv = g.om.n, g.vm.n
    off = max(abs(fock[:no, :no] - np.diag(fock.diagonal()[:no])).max(), abs(fock[no:, no:] - np.diag(fock.diagonal()[no:])).max())
    if off > 1e-7:
        raise ValueError('the quadruples correction takes semicanonical spinors (a diagonal Fock matrix among the '
                         'occupied and among the virtual ones); largest off-diagonal element %.1e' % off)
    e = fock.diagonal().real
    view = eom_density.View(eris, dtype, shift, dense=True)        # the packed block
    # The sectors of the (Q) code are for totally symmetric operands: the ground state, and a
    # state that is known to be totally symmetric - one of shift 0 in the blocks of the labels.
    # A state with a shift, and the states of the packed block (all shifts together: a level
    # holds components of several), get the quadruples without sectors.
    import copy
    sym = g.om.spins is not None and not shift
    e_sym = view.eris
    e_any = e_sym
    if not sym and getattr(e_sym, 'labels', None) is not None:
        e_any = copy.copy(e_sym)
        e_any.no_sectors = True
    roots = list(range(len(left.roots))) if roots is None else list(roots)
    kw = dict(max_memory=max_memory, workers=workers)

    def say(msg):
        if verbose:
            print(msg, file=out, flush=True)
    with timer('eomq.dress'):
        ints = t4engine.Dressed.from_eris(view.eris, np.asarray(t1, dtype=dtype))
    T3m = view.rank3(t3).reshape(g.om.ntri, g.vm.ntri)
    t2 = np.asarray(t2, dtype=dtype)
    res = Result([])
    rho = None
    if lam is not None and zetas is not None:
        t0 = time.perf_counter()
        with timer('eomq.rho'):
            rho = cc4fly.run_for(e_sym, ints, e, t2, T3m, **kw)
        res.seconds_rho = time.perf_counter() - t0
        lam2, lam3 = view.pack2(np.asarray(lam[1], dtype=dtype)), view.rank3(lam[2])
        res.e_ground = _real(np.dot(lam2.ravel(), rho[0].ravel()) + np.dot(lam3, rho[1].ravel()), dtype)
        say('  the T4 terms of the ground state: %.1f s; Lambda . rho = %.10f Eh' % (res.seconds_rho, res.e_ground))
    for kk in roots:
        t0 = time.perf_counter()
        k = left.roots[kk]
        omega = float(right.e[k])
        r1 = np.asarray(right.r1[k], dtype=dtype)
        r2 = view.dense2(right.r2[k])
        R3m = view.rank3(right.r3[k], shift).reshape(T3m.shape)
        l2 = view.pack2(view.dense2(left.l2[kk])).ravel()
        l3 = view.rank3(left.l3[kk], shift)
        with timer('eomq.dint'):
            dints = cc4fly.dressed_derivative(view.eris, np.asarray(t1, dtype=dtype), r1)
        with timer('eomq.fold'):
            a2, a3 = cc4fly.run_for(e_any, ints, e, t2, T3m, omega=omega, back_ints=ints,
                                    drives=[(ints, t2 + r2, T3m + R3m, 0.5), (ints, t2 - r2, T3m - R3m, -0.5),
                                            (dints, t2, T3m, 1.0)], **kw)
        with timer('eomq.t4'):
            b2, b3 = cc4fly.run_for(e_any, ints, e, t2, T3m, back_ints=dints, **kw)
        fold = _real(np.dot(l2, a2.ravel()) + np.dot(l3, a3.ravel()), dtype)
        t4 = _real(np.dot(l2, b2.ravel()) + np.dot(l3, b3.ravel()), dtype)
        # the same with conj(R) on the left: the Rayleigh quotient of the right-hand vector
        rr2 = view.pack2(r2).ravel()
        nrm = np.vdot(r1, r1) + np.vdot(rr2, rr2) + np.vdot(R3m, R3m)
        ray = _real((np.vdot(rr2, (a2 + b2).ravel()) + np.vdot(R3m.ravel(), (a3 + b3).ravel())) / nrm, dtype)
        c = Correction(root=k, omega=omega, fold=fold, t4=t4, rayleigh=ray)
        del a2, a3, b2, b3, dints
        if rho is not None:
            z = zetas[kk]
            w2 = view.pack2(np.asarray(z[1], dtype=dtype)) - lam2
            w3 = view.rank3(z[2]) - lam3
            c.ground = _real(np.dot(w2.ravel(), rho[0].ravel()) + np.dot(w3, rho[1].ravel()), dtype)
        c.seconds = time.perf_counter() - t0
        say('  ' + Result([c]).text()[-1] + '  (%.1f s)' % c.seconds)
        res.states.append(c)
    return res


def _real(x, dtype):
    return float(np.real(x))
