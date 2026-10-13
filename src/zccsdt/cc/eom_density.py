'''One-particle densities of EOM-EE-CCSDT states and the transition densities
between them and the ground state (docs/eom.md, section 13), from the right
and left eigenvectors of zccsdt.cc.eom and zccsdt.cc.eom_left.

With Obar[p,q] = exp(-T) p+ q exp(T), R_k = r0 + R1 + R2 + R3 the right and
L_k = L1 + L2 + L3 the left eigenvector of the state k, L_k R_k = 1 (r0 =
eta . r / omega = - Lambda . r: the product with the left ground state
<0| (1 + Lambda) vanishes; r0 = 0 unless the state is totally symmetric):

    expectation   g(k)[p,q]  = <0| L_k Obar[p,q] R_k |0>
    response      gr(k)[p,q] = g(k)[p,q] + <0| Z_k Obar[p,q] |0>,
                  Z_k A = - xi,  xi_kappa = <0| L_k [Hbar, tau_kappa] R_k |0>
    transition    <0~| p+ q |k> = <0| (1 + Lambda) Obar[p,q] R_k |0>
                  <k~| p+ q |0> = <0| L_k Obar[p,q] |0>

g(k) is the density of the EOM-CC states as they are (the "unrelaxed
EOM-CC density": amplitudes and spinors fixed); gr(k) adds the response of
the ground-state amplitudes and is the derivative of the total energy of
the state with respect to a one-electron perturbation at fixed spinors
(what a finite field in the Hamiltonian gives).  Neither has the response
of the spinors.

**Nothing is contracted here that the ground-state codes do not contract.**
d1(T; l) = <0| (1 + l) exp(-T) p+ q exp(T) |0> is density_opt.kernel() for
any amplitudes T and any left vector l, linear in l and of second degree
in T, so with d1_ref = d1(T; 0) (the unit matrix of the occupied spinors
and t1) and R the excitation part of R_k:

    <0| L [Obar, R] |0>    = [d1(T + R; L) - d1(T - R; L)] / 2 - (r1 in the block [i,a])  =: F
    <0| L R Obar |0>       = d1(T; L') ,  L' = the part of L R that still de-excites:
                             l'1[i,a] = l2[ij,ab] r1[j,b] + 1/4 l3[ijk,abc] r2[jk,bc],
                             l'2[ij,ab] = l3[ijk,abc] r1[k,c]
    g(k)                   = d1(T; L') + r0 (d1(T; L) - d1_ref) + F
    <k~| p+ q |0>          = d1(T; L) - d1_ref
    <0~| p+ q |k>          = d1(T; r0 Lambda + Lambda') - d1_ref + [d1(T + R; Lambda) - d1(T - R; Lambda)] / 2
                             + (r0 + Lambda . R) d1_ref,
                             Lambda' = the same contraction of Lambda with R; the last term is zero
                             for converged vectors

and for the response

    xi  = r0 omega l + eta + L' A + xi'',   xi''_kappa = <0| L [[Hbar, tau_kappa], R] |0>
        = d/dx [L A(T + x R)] at x = 0:   the left product at four displaced
          amplitudes (the five-point formula, exact: A is a polynomial in T)
    Z   = Zt - L' - r0 L,   Zt A = - eta - xi'':  the Lambda equations with xi'' as a source
    gr(k) = d1(T; Zt) + F

(Zt = Lambda + the response to xi''; gr(k) minus the ground-state density
d1(T; Lambda) is the derivative of omega).  L' is one pass of the T3 engine
over l3 (its pass 1 with r in the place of the integrals).

Cost per state: three evaluations of the density for g(k), four more for
the two transition densities; the response four left products with their
intermediates, one solution of the Lambda equations and one density.  An
implementation with intermediates of its own would need one pass each.

Layouts: where the Lambda and density engines of the blocks of the symmetry
labels can take the arrays (totally symmetric states) everything runs in
the blocks; a state with a label shift is put on the packed block.
'''

import time
from dataclasses import dataclass

import numpy as np

from zccsdt.cc import ccsdt_opt, density_blk, density_opt, eom_left, eom_opt, lambda_opt, t3pack
from zccsdt.timing import timer


class View:
    '''The layout in which the Lambda and the density engine take the
    arrays of a state of the label shift ``shift`` (classes of the maps):
    as they are (no labels in use; or labels, a totally symmetric state and
    the engines of the blocks), or on the packed block in lexicographic
    order with the labels out of use.'''

    def __init__(self, eris, dtype, shift=None, dense=False):
        om, vm = t3pack.maps_for(eris)
        self.om, self.vm, self.dtype, self.shift = om, vm, np.dtype(dtype), shift
        labelled = om.spins is not None
        self.blocks = bool(labelled and not shift and not dense and lambda_opt.block_engine(eris, dtype) is not None
                           and density_blk.engine(eris, dtype) is not None)
        self.dense = labelled and not self.blocks
        self.eris = eris
        if self.dense:
            import copy
            self.eris = copy.copy(eris)
            self.eris.use_labels = False
            self.eris.__dict__.pop('_buffers', None)
            self.source = eris
            self.lex = (ccsdt_opt._lex(om), ccsdt_opt._lex(vm))
        vom, vvm = t3pack.maps_for(self.eris)
        self.pairs = (vom.pairs[:, 0][:, None], vom.pairs[:, 1][:, None], vvm.pairs[:, 0], vvm.pairs[:, 1])
        self.n3 = om.ntri * vm.ntri if self.dense else ccsdt_opt.engine_for(eris, dtype).size

    def rank3(self, x, shift=None):
        '''A rank-3 array of the label shift ``shift`` (None: the layout of
        T3) in the layout of the view, flat.'''
        if not self.dense:
            return np.ascontiguousarray(np.asarray(x, dtype=self.dtype).ravel())
        eng = ccsdt_opt.engine_for(self.source, self.dtype)
        from zccsdt.cc import eom
        layout = eng.L if shift is None else eom._shifter(eng).layout(shift)
        out = np.zeros((self.om.ntri, self.vm.ntri), dtype=self.dtype)
        out[np.ix_(*self.lex)] = layout.to_dense(np.asarray(x, dtype=self.dtype))
        return out.ravel()

    def back3(self, x):
        '''A totally symmetric rank-3 array of the view in the layout of T3.'''
        if not self.dense:
            return x
        return ccsdt_opt.blocked_t3(self.source, np.asarray(x).reshape(self.om.ntri, self.vm.ntri))

    def dense2(self, x2p):
        '''[i,j,a,b] of [(i<j),(a<b)] in the pairs of the maps of the vectors.'''
        return t3pack.unpack_pairs(np.asarray(x2p, dtype=self.dtype), self.om, self.vm)

    def pack2(self, x2):
        '''[(i<j),(a<b)] in the pairs of the view of a dense [i,j,a,b].'''
        return np.ascontiguousarray(x2[self.pairs])


def reference_density(t1):
    '''d1(T; 0)[p,q] = <0| exp(-T) p+ q exp(T) |0>: 1 on the occupied diagonal, t1 in [i,a].'''
    no, nv = t1.shape
    d = np.zeros((no + nv, no + nv), dtype=t1.dtype)
    d[:no, :no] = np.eye(no)
    d[:no, no:] = t1
    return d


def contract(g, l2, l3, r1, r2p, s3=None, sr=None):
    '''The part of (l R) that still de-excites, for a left vector (l2
    dense, l3 flat of the label shift ``s3``; None: the layout of T3) and a
    right one (r1, r2 on pairs, of the shift ``sr``):

        l'1[i,a] = sum l2[ij,ab] r1[j,b] + 1/4 sum l3[ijk,abc] r2[jk,bc],
        l'2[(i<j),(a<b)] = sum l3[ijk,abc] r1[k,c]

    by pass 1 of the T3 engine of the Ground ``g`` over l3, with r2 in the
    place of <mn||ef> and r1 in that of f[m,e].'''
    om, vm, dt = g.om, g.vm, g.dtype
    nocc, nvir = om.n, vm.n
    r1 = np.ascontiguousarray(r1, dtype=dt)
    r2p = np.ascontiguousarray(r2p, dtype=dt).reshape(om.npair, vm.npair)
    out1 = np.einsum('ijab,jb->ia', l2, r1)
    out2 = np.zeros((om.npair, vm.npair), dtype=dt)
    if np.size(l3):
        G1 = np.zeros((nvir + nocc, om.npair, nvir), dtype=dt)
        G2 = np.zeros((nocc * vm.npair, nocc + nvir), dtype=dt)
        g.eng.io = None
        l3 = np.ascontiguousarray(np.asarray(l3, dtype=dt).ravel())
        if om.spins is None or (s3 is None and not sr):
            a1, a2 = eom_opt._pass1(g, l3, G1, G2, r2p, r1)[:2]
        else:
            a1, a2 = eom_opt._pass1(g, l3, G1, G2, r2p, r1, (s3 or 0, -(sr or 0)), xe=False, yj=False)[:2]
        out1 = out1 + a1
        out2 = out2 + a2
    return out1, out2


@dataclass
class State:
    '''The densities of one state (kernel); all [p,q] over the active
    spinors [occupied, virtual], p+ q, not Hermitian.'''
    root: int
    omega: float
    r0: float                   # the component of R_k on the reference
    d1: np.ndarray              # <0| L_k Obar R_k |0>: the density of the state
    d1_response: np.ndarray = None      # ... with the response of the ground-state amplitudes
    left: np.ndarray = None     # <k~| p+ q |0> = <0| L_k Obar |0>
    right: np.ndarray = None    # <0~| p+ q |k> = <0| (1 + Lambda) Obar R_k |0>
    ground_overlap: float = 0.0     # |r0 + Lambda . R_k|: zero for converged vectors
    zeta: tuple = None          # Zt = Lambda + the response to xi'' (l1, l2 dense, l3 in the layout of T3)
    lambda_cycles: int = 0
    timings: dict = None


def kernel(eris, t1, t2, t3, right, left, roots=None, lam=None, response=False, transition=False, opts=None,
           verbose=False, stdout=None, dense=False):
    '''The densities of the states of ``left`` (an eom_left.Result of the
    eom.Result ``right``; ``roots``: positions in left.roots, default all):
    a list of State.  ``lam`` = (l1, l2, l3): Lambda of the ground state,
    l3 in the layout of t3 - needed for ``transition`` (the two transition
    densities with the ground state) and the start of the equations of
    ``response`` (the density that is the derivative of the energy of the
    state at fixed spinors).  ``opts``: solver.Options (max_memory; the
    thresholds of the Lambda equations of the response).'''
    import sys
    from zccsdt import memory
    from zccsdt.cc import solver
    opts = opts or solver.Options()
    out = stdout or sys.stdout
    g = right.ground
    shift = right.shift
    dtype = np.dtype(np.result_type(g.dtype, *([] if lam is None else [lam[0]])))
    if transition and lam is None:
        raise ValueError('transition densities need Lambda of the ground state (lam=)')
    view = View(eris, dtype, shift, dense)
    no, nv = g.om.n, g.vm.n
    isz = dtype.itemsize
    roots = list(range(len(left.roots))) if roots is None else list(roots)

    def say(msg):
        if verbose:
            print(msg, file=out, flush=True)

    # ---- memory: what is made here besides the arrays of the density code - T3, a vector on
    # each side and Lambda in the layout of the view (copies on the packed block only), the
    # displaced T3, a rank-3 part of zeros / of r0 Lambda
    budget = None if opts.incore_complete else memory.budget(opts.max_memory)
    own = view.n3 * isz * (2 + (4 if view.dense else 0) + (1 if lam is not None and view.dense else 0))
    dplan = density_opt.kernel(view.eris, t1, t2, np.zeros(0), t1, t2, np.zeros(0), gradient=False,
                               max_memory=opts.max_memory, blocks=view.blocks, plan_only=True) \
        if budget is not None else None
    if budget is not None and own + dplan.need > budget:
        raise MemoryError('the densities of EOM-CCSDT states need %.2f GB for the arrays of the size of T3 (%s) '
                          'next to %.2f GB of the density code; %.2f GB are available (max_memory)'
                          % (own / 1e9, 'on the packed block' if view.dense else 'in the layout of T3',
                             dplan.need / 1e9, budget / 1e9))
    T3v = view.rank3(t3)
    lamv = None if lam is None else (np.asarray(lam[0], dtype=dtype), np.asarray(lam[1], dtype=dtype), view.rank3(lam[2]))
    ref = reference_density(np.asarray(t1, dtype=dtype))
    zero3 = np.zeros(view.n3, dtype=dtype)
    eta = left.ops.eta()
    sym = shift in (None, 0) and (left.ops.sector is None or left.ops.sector.shift == 0)

    def dens(tt, ll):
        with timer('eom.density'):
            return density_opt.kernel(view.eris, tt[0], tt[1], tt[2], ll[0], ll[1], ll[2], gradient=False,
                                      max_memory=opts.max_memory, blocks=view.blocks).d1
    states = []
    for kk in roots:
        t_0 = time.perf_counter()
        k = left.roots[kk]
        times = {}
        omega = float(right.e[k])
        r1 = np.asarray(right.r1[k], dtype=dtype)
        r2p = np.asarray(right.r2[k], dtype=dtype)
        l1 = np.asarray(left.l1[kk], dtype=dtype)
        l2p = np.asarray(left.l2[kk], dtype=dtype)
        r2, l2 = view.dense2(r2p), view.dense2(l2p)
        R3v, L3v = view.rank3(right.r3[k], shift), view.rank3(left.l3[kk], shift)
        r0 = (np.dot(eta[0].ravel(), r1.ravel()) + np.dot(eta[1].ravel(), r2p.ravel())) / omega if sym else 0.0
        r0 = r0.real if dtype.kind == 'f' else r0
        T, L = (t1, t2, T3v), (l1, l2, L3v)
        # ---- <0| L R Obar |0> and r0 <0| L Obar |0>
        lp1, lp2 = contract(g, l2, left.l3[kk], r1, r2p, shift, shift)
        lp2 = view.dense2(lp2)
        A = dens(T, (lp1, lp2, zero3))
        B = dens(T, L)
        # ---- <0| L [Obar, R] |0>, and the same with Lambda
        Td = np.empty_like(T3v)
        F = G = None
        for sign in (1.0, -1.0):
            np.multiply(R3v, sign, out=Td)
            Td += T3v
            Tx = (t1 + sign * r1, t2 + sign * r2, Td)
            x = dens(Tx, L)
            F = 0.5 * sign * x if F is None else F + 0.5 * sign * x
            if transition:
                x = dens(Tx, lamv)
                G = 0.5 * sign * x if G is None else G + 0.5 * sign * x
        del Td
        F[:no, no:] -= r1
        d1 = A + r0 * (B - ref) + F
        st = State(root=k, omega=omega, r0=r0, d1=d1)
        times['density'] = time.perf_counter() - t_0
        if transition:
            t_1 = time.perf_counter()
            st.left = B - ref
            a1, a2 = contract(g, lamv[1], lam[2], r1, r2p, None, shift)
            x3 = lamv[2] * r0 if r0 else zero3
            D = dens(T, (r0 * lamv[0] + a1, r0 * lamv[1] + view.dense2(a2), x3))
            # (r0 + Lambda . R = 0 for converged vectors: the left ground state and R_k are orthogonal)
            lr = r0 + ((np.dot(lamv[0].ravel(), r1.ravel()) + 0.25 * np.dot(lamv[1].ravel(), r2.ravel())
                        + np.dot(lamv[2], R3v)) if sym else 0.0)
            st.right = D - ref + G + lr * ref
            st.ground_overlap = abs(lr)
            del x3
            times['transition'] = time.perf_counter() - t_1
        if response:
            t_1 = time.perf_counter()
            xi = _xi(view, t1, t2, T3v, (r1, r2, R3v), L, opts)
            times['xi'] = time.perf_counter() - t_1
            t_1 = time.perf_counter()
            lopts = solver.Options(conv_tol_normt=opts.conv_tol_normt, max_cycle=opts.max_cycle,
                                   diis_space=opts.diis_space, incore_complete=opts.incore_complete,
                                   max_memory=opts.max_memory, max_disk=opts.max_disk, mem_mb=opts.mem_mb)
            start = {} if lamv is None else dict(l1=lamv[0], l2=lamv[1], l3=lamv[2])
            zr = lambda_opt.kernel(view.eris, t1, t2, T3v, lopts, blocks=view.blocks,
                                   sources=lambda_opt.ResidualSources(xi), **start)
            if not zr.converged:
                import warnings
                warnings.warn('EOM density: the Lambda equations of the response of state %d did not converge in %d cycles'
                              % (k + 1, zr.ncycle))
            times['lambda'] = time.perf_counter() - t_1
            st.lambda_cycles = zr.ncycle
            del xi
            t_1 = time.perf_counter()
            st.d1_response = dens(T, (zr.l1, zr.l2, zr.l3)) + F
            st.zeta = (zr.l1, zr.l2, view.back3(zr.l3))
            times['density'] += time.perf_counter() - t_1
        st.timings = times
        say('  state %d: r0 = %.6f; %s' % (k + 1, abs(r0), ', '.join('%s %.1f s' % kv for kv in times.items())))
        states.append(st)
    return states


def _xi(view, t1, t2, T3v, R, L, opts):
    '''xi''[kappa] = <0| L [[Hbar, tau_kappa], R] |0> = d/dx (L A(T + x R)) at
    x = 0, in the layout of the residuals of the Lambda equations of the
    view: the left product at T +- R and T +- 2 R (the five-point formula,
    exact for a polynomial of degree four; A is one of degree three).'''
    r1, r2, R3v = R
    l1, l2, L3v = L
    x = [np.ascontiguousarray(l1).ravel(), view.pack2(l2).ravel(), L3v]
    acc = None
    Td = np.empty_like(T3v)
    for c, w in ((1.0, 8.0 / 12), (-1.0, -8.0 / 12), (2.0, -1.0 / 12), (-2.0, 1.0 / 12)):
        np.multiply(R3v, c, out=Td)
        Td += T3v
        ops = eom_left.LeftOps(view.eris, t1 + c * r1, t2 + c * r2, Td, mem_mb=opts.mem_mb)
        assert ops.mode == 'same'
        y = [np.zeros_like(p) for p in x]
        with timer('eom.density.xi'):
            ops.apply(x, y)
        del ops
        if acc is None:
            acc = [w * p for p in y]
        else:
            for a, p in zip(acc, y):
                a += w * p
        del y
    om, vm = t3pack.maps_for(view.eris)
    return acc[0].reshape(om.n, vm.n), acc[1].reshape(om.npair, vm.npair), acc[2]


# ---------------------------------------------------------------- properties

def hermitian(d1):
    '''D[p,q] = <q+ p> made Hermitian, the convention of make_rdm1: (d1^T + conj(d1)) / 2.'''
    return 0.5 * (d1.T + d1.conj())


def moment(op, d1):
    '''sum_pq op[p,q] d1[p,q] for an operator in the basis of the active
    spinors (several: op[x,p,q]) and a density or transition density
    [p,q] = <p+ q> as it is (not made Hermitian).'''
    return np.einsum('...pq,pq->...', np.asarray(op), d1)


def strength(op, st):
    '''The transition strength of the operator(s) op[x,p,q] between the
    ground state and the State ``st``: sum_x <0~| op_x |k> <k~| op_x |0>,
    the product of the two transition moments (its real part: for the
    exact states |<0| op |k>|^2).'''
    return float(np.sum(moment(op, st.right) * moment(op, st.left)).real)


@dataclass
class Properties:
    '''Dipole moments and transition moments of EOM states (GCCSDT.eom_properties), atomic units.'''
    e: np.ndarray               # excitation energies / Hartree
    states: list                # the State of each root
    dipole_ground: np.ndarray   # of the ground state (t and Lambda at fixed spinors), with the nuclei
    dipole: np.ndarray          # [root, x]: of the state, from g(k)
    dipole_response: np.ndarray     # ... from gr(k) (None without ``response``)
    moment_right: np.ndarray    # [root, x]: <0~| mu_x |k>
    moment_left: np.ndarray     # [root, x]: <k~| mu_x |0>
    strength: np.ndarray        # [root]: sum_x of their products (real part)
    oscillator: np.ndarray      # [root]: 2/3 omega strength
    levels: list = None         # the roots by degenerate levels: only the sums over a level are defined

    def text(self):
        out = ['ground state        dipole %s' % ' '.join('%12.8f' % x for x in self.dipole_ground)]
        for k, st in enumerate(self.states):
            line = 'state %d  %.8f Eh  dipole %s' % (st.root + 1, self.e[k], ' '.join('%12.8f' % x for x in self.dipole[k]))
            if self.dipole_response is not None:
                line += '   with the response of T %s' % ' '.join('%12.8f' % x for x in self.dipole_response[k])
            out.append(line)
            out.append('         transition moments <0~|mu|k> %s   <k~|mu|0> %s   strength %.8f   f = %.8f'
                       % (' '.join('%11.7f' % abs(x) for x in self.moment_right[k]),
                          ' '.join('%11.7f' % abs(x) for x in self.moment_left[k]), self.strength[k], self.oscillator[k]))
        for lv in self.levels or []:
            if len(lv) > 1:
                out.append('level of the roots %s: f = %.8f, mean dipole %s'
                           % (', '.join(str(self.states[k].root + 1) for k in lv), self.oscillator[lv].sum(),
                              ' '.join('%12.8f' % x for x in self.dipole[lv].mean(axis=0))))
        return out
