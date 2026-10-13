'''EOM-EE-CCSDT and EOM-EE-CCSD: the lowest right-hand eigenpairs of the
Jacobian of the amplitude equations (docs/eom.md), by a block Davidson
iteration on the sigma vectors of zccsdt.cc.eom_opt.

    kernel(eris, t1, t2, t3, nroots=3)      # Result: energies, vectors

The solver (davidson): right-hand vectors only; an orthonormal basis V, the
sigma vectors S = A V, the projected matrix G = V^H S, a general eigenvalue
problem of its size; the residual A x - omega x of every root that is not
converged, divided by omega - D with the differences of the orbital
energies D, orthogonalized and added to V.  The roots are the lowest ones
or, with ``follow``, those with the largest overlap with the roots of the
iteration before (with the guess at the start).  When V would exceed
``max_space`` vectors, it is replaced by an orthonormal basis of the
current roots.

The guess of EOM-CCSDT: the roots of the singles-doubles block of the same
matrix (the sigma vector without R3 and without the terms of T3 in
sigma_2), from unit vectors on the lowest singles, converged loosely; they
are followed.

Memory (plan_for): the operands of the ground state and of a sigma vector
are in memory (the pp operator twice, six arrays of the size of <ia||bc>,
T3); the 2 max_space vectors of the subspace are in memory if they fit in
``max_memory`` next to them, else they are mapped arrays of a scratch file
(counted against ``max_disk``); max_space is lowered to what fits.
'''

import ctypes
import os
import time
from dataclasses import dataclass

import numpy as np
import scipy.linalg

from zccsdt.cc import ccsdt_opt, diis, eom_opt, t3pack, zeta_opt
from zccsdt.timing import timer

HARTREE_EV = 27.211386245988
CHUNK = 1 << 21
# False: the collapse, the preconditioner and the projected matrix with numpy on one
# thread, as the first version had them (ZCCSDT_EOM_NUMPY=1; to measure the difference)
THREADED = not os.environ.get('ZCCSDT_EOM_NUMPY')


# ------------------------------------------------------------------- vectors

def vdot(x, y):
    '''<x|y> of two vectors (lists of flat parts).'''
    return sum(diis._dot(a, b) for a, b in zip(x, y) if a.size)


def mdot(a, xs):
    '''[<a|x> for x in xs] in one pass over a (threaded).'''
    if not THREADED:
        return np.array([vdot(a, x) for x in xs])
    out = 0
    for p in range(len(a)):
        if a[p].size:
            out = out + diis._mdot(a[p], [x[p] for x in xs])
    return np.atleast_1d(out)


def lincomb(out, terms):
    '''out = sum c * x over terms = [(c, x)]; out may be one of the x.'''
    cs = [c for c, _ in terms]
    for p in range(len(out)):
        if out[p].size:
            diis._lincomb([x[p] for _, x in terms], cs, out[p])
    return out


def transform(vecs, Q):
    '''The first m of ``vecs`` become sum_j vecs[j] Q[j, k], k < m (m
    columns of Q), in place and in chunks: no array of the size of a vector
    is made.'''
    n, m = Q.shape
    assert n == len(vecs) and m <= n
    Qt = np.ascontiguousarray(Q.T)
    for p in range(len(vecs[0])):
        size = vecs[0][p].size
        dt = vecs[0][p].dtype
        f = diis._vec_c('vtransform', dt) if size and THREADED else None
        if f is not None and all(v[p].flags.c_contiguous and v[p].dtype == dt for v in vecs):
            # (the threaded kernel: blocks of 256 elements of all vectors at a time)
            q = np.ascontiguousarray(Q, dtype=dt)
            ptrs = (ctypes.c_void_p * n)(*[v[p].ctypes.data for v in vecs])
            f(ctypes.c_int64(size), ctypes.c_int64(n), ctypes.c_int64(m), ptrs, t3pack._p(q))
            continue
        step = max(1, CHUNK // n)
        for c0 in range(0, size, step):
            blk = np.stack([v[p][c0:c0 + step] for v in vecs])
            new = Qt @ blk
            for k in range(m):
                vecs[k][p][c0:c0 + step] = new[k]


class Store:
    '''The vectors of the subspace: parts of the sizes ``sizes``; the last
    part in a scratch file if ``mapped``.'''

    def __init__(self, sizes, dtype, mapped=False):
        self.sizes, self.dtype, self.mapped = list(sizes), np.dtype(dtype), mapped
        self.scr, self.count, self.free = None, 0, []

    def new(self):
        if self.free:
            return self.free.pop()
        out = [np.zeros(n, dtype=self.dtype) for n in self.sizes[:-1]]
        n = self.sizes[-1]
        if self.mapped and n:
            from zccsdt.scratch import Scratch
            if self.scr is None:
                self.scr = Scratch(prefix='zccsdt_eom_', what='the vectors of the EOM subspace',
                                   hint=' or lower max_space')
            out.append(self.scr.create('v%d' % self.count, n, self.dtype))
        else:
            out.append(np.zeros(n, dtype=self.dtype))
        self.count += 1
        return out

    def release(self, vec):
        self.free.append(vec)

    def close(self):
        if self.scr is not None:
            self.scr.close()
            self.scr = None


# -------------------------------------------------------------------- solver

@dataclass
class Solved:
    e: np.ndarray               # eigenvalues (complex)
    vectors: list
    converged: np.ndarray       # per root
    residuals: np.ndarray       # |A x - omega x| per root, |x| = 1
    niter: int
    nsigma: int
    ncollapse: int = 0
    history: list = None        # per iteration: (size of the subspace, eigenvalues, residual norms, seconds)


def _orthonormal_columns(Y, real):
    '''An orthonormal basis of the span of the columns of Y (of their real
    and imaginary parts if ``real``).'''
    if real:
        Y = np.concatenate([Y.real, Y.imag], axis=1)
    u, s, _ = np.linalg.svd(Y, full_matrices=False)
    return u[:, s > 1e-8 * s[0]]


def davidson(apply, precond, guesses, store, nroots, conv_tol=1e-6, max_space=20, max_cycle=60,
             follow=False, log=None, lindep=1e-10):
    '''The ``nroots`` lowest (or followed) right eigenpairs of the operator
    ``apply(x, out)`` (out: a vector of the store, to be filled).
    ``precond(x, omega)`` divides the vector x by omega - D in place.
    ``guesses``: vectors of the store (taken over).  Converged: the norm
    of the residual of the normalized vector below ``conv_tol``.'''
    real = store.dtype.kind == 'f'
    V, S = [], []
    G = np.zeros((0, 0), dtype=store.dtype)
    new, nsigma, ncollapse, history = [], 0, 0, []
    # orthonormalize the guesses
    for g in guesses:
        g = _orthogonalize(g, new, lindep)
        if g is not None:
            new.append(g)
    nroots = min(nroots, len(new))
    prev = np.eye(len(new), nroots, dtype=complex) if follow else None
    w = Y = res = None
    conv = np.zeros(nroots, bool)
    for it in range(max_cycle):
        t0 = time.perf_counter()
        n0 = len(V)
        for v in new:
            s = store.new()
            with timer('eom.sigma'):
                apply(v, s)
            nsigma += 1
            V.append(v)
            S.append(s)
        n = len(V)
        Gn = np.zeros((n, n), dtype=store.dtype)
        Gn[:n0, :n0] = G
        with timer('eom.vectors'):
            for j in range(n0, n):          # (one pass over the new vector for a column, one for a row)
                Gn[:, j] = np.conj(mdot(S[j], V))
                if n0:
                    Gn[j, :n0] = mdot(V[j], S[:n0])
        G = Gn
        ew, ev = scipy.linalg.eig(G)
        if prev is not None:        # the roots that the previous ones have become
            pv = np.zeros((n, prev.shape[1]), dtype=complex)
            pv[:prev.shape[0]] = prev
            # by their weight in the space of the previous roots, not one to one: inside a
            # degenerate level the eigenvectors of two iterations are rotated against each
            # other at will, and a one-to-one match gave the last root of a sixfold level
            # (F-, 2p^-1) to a vector far above it
            Qp = _orthonormal_columns(pv, False)
            score = np.linalg.norm(Qp.conj().T @ (ev / np.linalg.norm(ev, axis=0)), axis=0)
            pick = sorted((int(x) for x in np.argsort(-score, kind='stable')[:prev.shape[1]]),
                          key=lambda j: (ew[j].real, j))
        else:
            pick = list(np.argsort(ew.real, kind='stable')[:nroots])
        w, Y = ew[pick], ev[:, pick]
        if real:
            # a complex pair of a real matrix - two roots that are degenerate
            # but for the rounding: the real and the imaginary part of its
            # vector span their space (the real part alone if only one of
            # the two is among the roots)
            Yr = Y.real.copy()
            for k in range(len(pick)):
                if w[k].imag > 0:
                    partner = [m for m in range(len(pick)) if m != k and abs(w[m] - w[k].conjugate()) < 1e-14 + 1e-10 * abs(w[k])]
                    if partner:
                        Yr[:, partner[0]] = Y[:, k].imag
            Y = Yr
        w, Y = _degenerate(G, w, Y, real, cluster_tol=max(1e-9, 1e-2 * conv_tol))
        Y = Y / np.linalg.norm(Y, axis=0)
        if follow:
            prev = Y.astype(complex)
        # collapse first if the corrections would not fit
        if n + nroots > max_space and n > nroots:
            Q = _orthonormal_columns(Y, False)
            Q = Q.real.copy() if real else Q
            with timer('eom.vectors'):
                transform(V, Q)
                transform(S, Q)
            for x in V[Q.shape[1]:] + S[Q.shape[1]:]:
                store.release(x)
            V, S = V[:Q.shape[1]], S[:Q.shape[1]]
            G = Q.conj().T @ G @ Q
            Y = Q.conj().T @ Y
            Y = Y / np.linalg.norm(Y, axis=0)
            if follow:
                prev = Y.astype(complex)
            n = len(V)
            ncollapse += 1
        # residuals and corrections
        new, res = [], np.zeros(nroots)
        with timer('eom.vectors'):
            for k in range(nroots):
                wk = w[k].real if real else w[k]
                y = Y[:, k].real if real else Y[:, k]
                t = store.new()
                lincomb(t, [(y[j], S[j]) for j in range(n)] + [(-wk * y[j], V[j]) for j in range(n)])
                res[k] = float(np.sqrt(abs(vdot(t, t))))
                if res[k] < conv_tol:
                    store.release(t)
                    continue
                precond(t, wk)
                t = _orthogonalize(t, V + new, lindep, store)
                if t is not None:
                    new.append(t)
        conv = res < conv_tol
        history.append((n, w.copy(), res.copy(), time.perf_counter() - t0))
        if log is not None:
            log(it + 1, n, w, res, conv, history[-1][3])
        if conv.all() or not new:
            break
    for x in new:
        store.release(x)
    # the eigenvectors, in the first slots of V
    Yf = (Y.real if real else Y)
    Qf = np.zeros((len(V), len(V)), dtype=store.dtype)
    Qf[:, :nroots] = Yf
    with timer('eom.vectors'):
        transform(V, np.ascontiguousarray(Qf[:, :nroots]))
    for x in V[nroots:] + S:
        store.release(x)
    return Solved(np.array(w), V[:nroots], conv, res, it + 1, nsigma, ncollapse, history)


def _degenerate(G, w, Y, real, cluster_tol):
    '''Roots that are degenerate (to ``cluster_tol``): the eigenvectors that
    a general eigensolver returns for them can be nearly parallel.  They
    are replaced by an orthonormal basis of their space - the right
    singular vectors of G - omega to its smallest singular values - turned
    as close to the given vectors as it can be, and get one eigenvalue.'''
    w, Y = w.copy(), np.array(Y, dtype=float if real else complex)
    n, done = len(w), set()
    for k in range(n):
        if k in done:
            continue
        cl = [m for m in range(n) if m not in done and abs(w[m] - w[k]) < cluster_tol]
        done.update(cl)
        if len(cl) < 2:
            continue
        om = np.mean(w[cl])
        om = om.real if real else om
        _, _, vh = np.linalg.svd(G - om * np.eye(G.shape[0]))
        B = vh[-len(cl):].conj().T
        u, _, v = np.linalg.svd(B.conj().T @ Y[:, cl])
        Y[:, cl] = B @ (u @ v)
        w[cl] = om
    return w, Y


def _orthogonalize(t, basis, lindep, store=None):
    '''t orthogonalized against ``basis`` (twice) and normalized, in place;
    None (and t released) if nothing is left of it.'''
    n0 = float(np.sqrt(abs(vdot(t, t))))
    if n0 == 0.0:
        if store is not None:
            store.release(t)
        return None
    lincomb(t, [(1.0 / n0, t)])
    for _ in range(2):
        if basis:
            c = np.conj(mdot(t, basis))
            lincomb(t, [(1.0, t)] + [(-ck, b) for ck, b in zip(c, basis)])
    nrm = float(np.sqrt(abs(vdot(t, t))))
    if nrm < lindep:
        if store is not None:
            store.release(t)
        return None
    lincomb(t, [(1.0 / nrm, t)])
    return t


# -------------------------------------------------------------------- memory

@dataclass
class Plan:
    '''Memory and disk of kernel() (plan_for).'''
    nroots: int
    max_space: int              # vectors of the subspace at most (as asked for, or lowered)
    mapped: bool                # the vectors of the subspace are in a scratch file
    need: int                   # bytes of ordinary memory
    fixed: int                  # of which the ground state and a sigma vector
    vectors: int                # bytes of the 2 max_space vectors (memory or disk)
    disk: int                   # bytes in scratch files
    asked: int = 0              # max_space as asked for

    def text(self):
        return ('subspace of at most %d vectors (%.2f GB with their sigma vectors) in %s; '
                'ground state and one sigma vector %.2f GB of memory'
                % (self.max_space, self.vectors / 1e9, 'a scratch file' if self.mapped else 'memory',
                   self.fixed / 1e9))


def _shifter(eng):
    '''The t3shift.ShiftT3 of an engine of label blocks (kept with it).'''
    from zccsdt.cc import t3shift
    if not hasattr(eng, 'L'):
        raise ValueError('a label shift needs symmetry labels in use (blocks of the labels)')
    if getattr(eng, '_shifter', None) is None:
        eng._shifter = t3shift.ShiftT3(eng)
    return eng._shifter


def fixed_bytes(eris, dtype, with_t3=True, mem_mb=2000.0, shift=None):
    '''Bytes of ordinary memory that prepare() and a sigma vector take
    besides the vectors of the subspace: the larger of

      prepare   an amplitude step with everything in memory and its
                operands kept: two more arrays of the size of <ia||bc>;
      sigma     the kept operands (the pp operator; <ia||bc>~, its packed
                form, the bra-dressed <ab||ei>, W_vvvo and the operand of
                pass 1: 3.5 f; T3; 16 arrays of o^2 v^2), the derivative of
                the pp operator, 3.5 f of transients (W'_vvvo, its parts and
                the drive operand), the batch buffers and the occupied
                operators.

    In the blocks of the symmetry labels the rank-3 arrays, the pp
    operator and the work array have the sizes of the blocks (the o v^3
    arrays are dense there too), and the amplitude step is counted with
    what the blocks of its occupied operator take - at most two blocks of
    T3 - instead of the quarter of ``mem_mb`` that the plan of the ground
    state sets aside for them.  ``shift``: a vector with that label shift
    (classes of the maps): the work array of the shifted passes, the
    derivative of the pp operator between the pair blocks h and h + shift,
    the blocks of the occupied operator between the groups g and g + shift.'''
    om, vm = t3pack.maps_for(eris)
    o, v = om.n, vm.n
    isz = np.dtype(dtype).itemsize
    eng = ccsdt_opt.engine_for(eris, dtype)
    f = o * v ** 3 * isz
    s = o * o * v * v * isz
    pp = sum((hi - lo) ** 2 for lo, hi in vm.blocks) * isz
    t3 = eng.size * isz if with_t3 else 0
    bufs = int(eng.buffer_bytes(mem_mb))
    if hasattr(eng, 'L'):
        L = eng.L
        blk = max([int(L.nI[g]) * int(L.nA[g]) for g in L.stored] + [0]) * isz
        small = 8 * s + 2 * om.ntri ** 2 * isz + ccsdt_opt.overhead()
        dress = min(2_000_000_000, 2 * pp) + f // 2
        prep = small + bufs + max(int(3.5 * f), int(1.75 * f) + max(dress, min(int(mem_mb * 250_000), 2 * blk))) \
            + 2 * eng.size * isz + pp + 2 * f
    else:
        step = ccsdt_opt.plan_memory(o, v, isz, mem_mb=mem_mb, t3_bytes=eng.size * isz, pp_bytes=pp, engine=eng)
        prep = step.need + 2 * f
    occ = 2 * om.ntri ** 2 * isz if o < t3kern_min_occ() else 4 * om.npair ** 2 * isz
    dpp = pp
    if shift is not None:
        sh = _shifter(eng)
        nq = [hi - lo for lo, hi in vm.blocks]
        dpp = sum(nq[h] * nq[h + shift] for h in range(len(nq)) if 0 <= h + shift < len(nq)) * isz
        if with_t3:
            bufs = max(bufs, int(sh.buffer_bytes(mem_mb, shift)))
            occ += sh.occ_bytes(shift)
    sig = pp + dpp + 7 * f + 22 * s + t3 + bufs + 2 * occ + ccsdt_opt.overhead()
    if not with_t3:
        sig = pp + dpp + 3 * f + 22 * s + ccsdt_opt.overhead()
    # the vectors of the singles-doubles stage (rank one and two; 2 (4 (nroots + extra) + 16)
    # of them: 80 for four roots)
    return max(prep, sig) + 80 * (o * v + om.npair * vm.npair) * isz


def t3kern_min_occ():
    from zccsdt.cc import t3kern
    return t3kern.HH_PAIR_MIN_OCC


def plan_for(eris, dtype, nroots, max_space=None, with_t3=True, opts=None, mem_mb=2000.0, shift=None):
    '''The Plan for ``nroots`` roots: ``max_space`` (default 8 + 4 nroots)
    vectors and their sigma vectors in memory if ``opts.max_memory`` (MB
    for the whole process; default 80% of what is available) holds them
    next to the fixed part, else in a scratch file (``opts.max_disk``);
    max_space is lowered, to 2 nroots + 2 at least, to what memory or disk
    hold.  MemoryError if the fixed part does not fit, DiskFullError if the
    smallest subspace fits nowhere.'''
    from zccsdt import disk, memory
    from zccsdt.cc import solver
    opts = opts or solver.Options()
    om, vm = t3pack.maps_for(eris)
    isz = np.dtype(dtype).itemsize
    eng = ccsdt_opt.engine_for(eris, dtype)
    asked = max_space or 8 + 4 * nroots
    floor = min(asked, 2 * nroots + 2)
    n3 = (eng.size if shift is None else _shifter(eng).size(shift)) if with_t3 else 0
    one = (om.n * vm.n + om.npair * vm.npair + n3) * isz
    fixed = fixed_bytes(eris, dtype, with_t3, mem_mb, shift=shift)
    if opts.incore_complete:
        return Plan(nroots, asked, False, fixed + 2 * asked * one, fixed, 2 * asked * one, 0, asked)
    budget = memory.budget(opts.max_memory)
    if budget is not None:
        budget += ccsdt_opt.cached_bytes(eris)
    if budget is None or fixed + 2 * asked * one <= budget:
        return Plan(nroots, asked, False, fixed + 2 * asked * one, fixed, 2 * asked * one, 0, asked)
    if fixed > budget:
        raise MemoryError('EOM-CC with %d occupied and %d virtual spin orbitals needs %.2f GB of memory for the '
                          'operands of the ground state and of a sigma vector (the pp operator and its '
                          'derivative, the <ia||bc>-sized arrays, T3); %.2f GB are available (max_memory, '
                          'or what the node and the job leave)' % (om.n, vm.n, fixed / 1e9, budget / 1e9))
    left = disk.budget(opts.max_disk)
    space = asked if left is None else int(min(asked, left // max(2 * one, 1)))
    if space < floor:
        # (in memory after all, if a smaller subspace fits there)
        space_mem = int((budget - fixed) // max(2 * one, 1))
        if space_mem >= floor:
            return Plan(nroots, space_mem, False, fixed + 2 * space_mem * one, fixed, 2 * space_mem * one, 0, asked)
        raise disk.DiskFullError(
            'the EOM subspace of %d roots needs at least %d vectors of %.2f GB with their sigma vectors: '
            '%.2f GB; memory has %.2f GB beside the fixed %.2f GB and the scratch disk %.2f GB '
            '(max_memory, max_disk, ZCCSDT_TMPDIR)'
            % (nroots, floor, one / 1e9, 2 * floor * one / 1e9, (budget - fixed) / 1e9, fixed / 1e9,
               (left or 0) / 1e9))
    big = n3 * isz                                  # (the parts of rank one and two stay in memory)
    return Plan(nroots, space, True, fixed + 2 * space * (one - big), fixed, 2 * space * one, 2 * space * big, asked)


# -------------------------------------------------------------------- driver

@dataclass
class Result:
    e: np.ndarray               # excitation energies / Hartree (real parts)
    e_imag: np.ndarray          # their imaginary parts
    converged: np.ndarray
    residuals: np.ndarray
    r1: list                    # [i,a] per root
    r2: list                    # [(i<j),(a<b)] per root, pairs of the maps of the integrals
    r3: list                    # flat, layout of t3 (None: EOM-CCSD)
    weights: np.ndarray         # [root, rank]: the norms^2 of r1, r2, r3 (unique elements; sum 1)
    niter: int = 0
    nsigma: int = 0
    ncollapse: int = 0
    guess: object = None        # the Solved of the singles-doubles stage
    history: list = None
    plan: object = None
    ground: object = None
    timings: dict = None
    shift: object = None        # the label shift in classes of the maps (None: totally symmetric, layout of t3)
    cut_level: bool = False     # the roots end inside a degenerate level (by the singles-doubles stage)
    short: object = None        # IP / EA with short vectors: the Short of the problem (strip() takes the ghost out)
    label_shift: object = None  # the same in units of the labels (set by the front end)
    partner_of: object = None   # these are the time-reversed states of that label shift (no vectors)
    left: object = None         # the eom_left.Result of these states, once computed (GCCSDT.eom_left)


def denominators(g):
    '''(D1 flat, D2 flat, (eo3, ev3)): e_a + .. - e_i - .. for the parts of
    a vector; the triples by the sums over occupied and virtual triples in
    the order of the maps.'''
    om, vm = g.om, g.vm
    e = np.asarray(g.eris.mo_energy).real
    eo, ev = e[:om.n], e[om.n:]
    d1 = (ev[None, :] - eo[:, None]).ravel()
    d2 = (ev[vm.pairs].sum(axis=1)[None, :] - eo[om.pairs].sum(axis=1)[:, None]).ravel()
    return d1, d2, (eo[om.triples].sum(axis=1), ev[vm.triples].sum(axis=1))


def make_precond(g, with_t3, floor=1e-3, shift=None):
    d1, d2, (eo3, ev3) = denominators(g)
    if not with_t3:
        blocks = []
    elif shift is None:
        blocks = zeta_opt._blocks(g.eris, g.dtype)
    else:
        blocks = [(None, None) + b for b in g.shifter().layout(shift).blocks()]

    def div(x, d, omega):
        den = omega - d
        den[abs(den) < floor] = floor
        x /= den

    def precond(x, omega):
        div(x[0], d1, omega)
        div(x[1], d2, omega)
        if with_t3:
            for rows, cols, where, shape, I, A in blocks:
                if not shape[0] or not shape[1]:
                    continue
                blk = x[2][where].reshape(shape)
                f = diis._vec_c('vprecond', blk.dtype)
                if f is not None and blk.flags.c_contiguous and THREADED:
                    f(t3pack._p(blk), ctypes.c_int64(shape[0]), ctypes.c_int64(shape[1]),
                      t3pack._p(np.ascontiguousarray(eo3[I])), t3pack._p(np.ascontiguousarray(ev3[A])),
                      t3pack._p(np.array([omega], dtype=blk.dtype)), ctypes.c_double(floor))
                    continue
                step = max(1, CHUNK // shape[1])
                for r0 in range(0, shape[0], step):
                    den = omega - (ev3[A][None, :] - eo3[I][r0:r0 + step, None])
                    den[abs(den) < floor] = floor
                    blk[r0:r0 + step] /= den
        return x
    return precond


def singles_guess(g, nguess, allowed=None):
    '''Unit vectors [i,a] on the ``nguess`` lowest differences of orbital
    energies (among the elements ``allowed`` [i,a] if given).'''
    d1 = denominators(g)[0].copy()
    if allowed is not None:
        d1[~np.asarray(allowed).ravel()] = np.inf
    order = np.argsort(d1, kind='stable')[:nguess]
    return [int(x) for x in order if np.isfinite(d1[x])]


def symmetric_singles(eris, shift=0):
    '''[i,a] True where a single excitation is totally symmetric for the
    labels in use (all of them without labels), or has the label shift
    ``shift`` (classes of the maps).'''
    om, vm = t3pack.maps_for(eris)
    if om.spins is None:
        return np.ones((om.n, vm.n), bool)
    return np.asarray(vm.spins)[None, :] - np.asarray(om.spins)[:, None] == shift


def class_shift(eris, label_shift):
    '''The shift in classes of the maps for a change ``label_shift`` of
    the sum of the symmetry labels (for 2 m_j, whose step is 2: half of
    it); ValueError if no excitation changes the labels by that.'''
    lab = getattr(eris, 'labels', None)
    om, _ = t3pack.maps_for(eris)
    if lab is None or om.spins is None:
        raise ValueError('a label shift needs the symmetry labels in use: a calculation in the blocks of the '
                         'labels (symmetry = True; a linear molecule on the z axis)')
    u = np.unique(np.asarray(lab))
    step = int(np.gcd.reduce(np.diff(u))) if len(u) > 1 else 1
    if int(label_shift) % step:
        raise ValueError('the labels of these spinors differ by multiples of %d: no state with a shift of %d'
                         % (step, label_shift))
    return int(label_shift) // step


class Sector:
    '''The vectors of one label shift on the packed block, by masks: the
    elements of r1[i,a], r2[(i<j),(a<b)] and of the packed rank-3 part
    whose labels (``labels``: one integer per spinor that adds, occupied
    then virtual - 2 m_j, 2 m_s) change by ``shift``.  The sigma vector
    maps such a vector to one of the same shift; project() removes what
    the rounding puts elsewhere.'''

    def __init__(self, om, vm, labels, shift):
        lab = np.asarray(labels, dtype=np.int64)
        assert om.spins is None and len(lab) == om.n + vm.n
        lo, lv = lab[:om.n], lab[om.n:]
        self.shift = int(shift)
        self.m1 = lv[None, :] - lo[:, None] == shift
        self.m2 = lv[vm.pairs].sum(axis=1)[None, :] - lo[om.pairs].sum(axis=1)[:, None] == shift
        self.lo3, self.lv3 = lo[om.triples].sum(axis=1), lv[vm.triples].sum(axis=1)
        self.n3 = int(sum((self.lv3 == shift + x).sum() * n for x, n in zip(*np.unique(self.lo3, return_counts=True))))

    def project(self, x):
        x[0] *= self.m1.ravel()
        x[1] *= self.m2.ravel()
        if len(x) > 2 and x[2].size:
            blk = x[2].reshape(len(self.lo3), len(self.lv3))
            step = max(1, CHUNK // max(len(self.lv3), 1))
            for r0 in range(0, len(self.lo3), step):
                blk[r0:r0 + step] *= self.lv3[None, :] == self.shift + self.lo3[r0:r0 + step, None]
        return x


def time_reversal_symmetric(eris, tol=1e-7, labels=None):
    '''Do the spinors come in pairs of labels l, -l with equal energies,
    occupied and virtual ones apart - the mark of a closed shell that is
    symmetric under time reversal?  Then the states of the label shifts D
    and -D are each other's time-reversed images, with equal energies.'''
    lab = getattr(eris, 'labels', None) if labels is None else labels
    if lab is None:
        return False
    lab = np.asarray(lab)
    e = np.asarray(eris.mo_energy).real
    no = eris.nocc
    for sl in (slice(0, no), slice(no, None)):
        l, x = lab[sl], e[sl]
        for v in np.unique(l):
            a, b = np.sort(x[l == v]), np.sort(x[l == -v])
            if len(a) != len(b) or (len(a) and abs(a - b).max() > tol):
                return False
    return True


def partner(res, label_shift, nroots=None):
    '''The Result of the time-reversed states of ``res`` (the label shift
    ``label_shift`` = minus that of res): the same energies, no vectors.'''
    import copy
    out = copy.copy(res)
    n = len(res.e) if nroots is None else min(nroots, len(res.e))
    out.e, out.e_imag, out.converged, out.residuals = (np.array(x[:n]) for x in (res.e, res.e_imag, res.converged, res.residuals))
    out.weights = np.array(res.weights[:n])
    out.r1 = out.r2 = out.r3 = None
    out.partner_of, out.label_shift = res.label_shift, label_shift
    out.shift = None if res.shift is None else -res.shift
    return out


def label_shifts(eris, labels=None):
    '''The changes of the sum of the labels that a single excitation can
    make (the shifts with singles), sorted.'''
    lab = np.asarray(eris.labels if labels is None else labels)
    no = eris.nocc
    return sorted({int(x) for x in (lab[None, no:] - lab[:no, None]).ravel()})


def ghost(eris, t1, t2, t3=None, kind='ip'):
    """(eris, t1, t2, t3, singles) of the problem with one more spinor that
    no integral touches and whose Fock element is zero - the last virtual
    one (kind 'ip') or the first occupied one ('ea') -, the amplitudes
    padded with zeros (t3: on the packed block in lexicographic order, in
    and out), and the single excitations [i,a] into (out of) it.  The
    EE problem there, among the vectors that have the ghost index once, is
    the IP (EA) problem: r[i,x] = r[i], r[ij,ax] = r[ij,a], r[ijk,abx] =
    r[ijk,ab] (EA: r[x,a], r[xi,ab], r[xij,abc]); its eigenvalues are the
    ionization energies (EA: E(N+1) - E(N)).  Dense integrals without
    symmetry labels; the vectors have the length of the EE problem with
    one spinor more - the cost of that problem, not of one with vectors an
    order shorter (docs/eom.md)."""
    from zccsdt.eris import ERIs
    no, n = eris.nocc, np.asarray(eris.fock).shape[0]
    nv = n - no
    at = n if kind == 'ip' else 0
    keep = np.array([p for p in range(n + 1) if p != at])
    sp = 'v' if kind == 'ip' else 'o'
    no2, nv2 = (no, nv + 1) if kind == 'ip' else (no + 1, nv)

    def pad(x, key):
        x = np.asarray(x)
        out = np.zeros(tuple(d + (c == sp) for d, c in zip(x.shape, key)), dtype=x.dtype)
        out[tuple(slice(0, d) if kind == 'ip' else slice(int(c == sp), None) for d, c in zip(x.shape, key))] = x
        return out
    fock = np.zeros((n + 1, n + 1), dtype=np.asarray(eris.fock).dtype)
    fock[np.ix_(keep, keep)] = eris.fock
    e = np.zeros(n + 1)
    e[keep] = np.asarray(eris.mo_energy).real
    blocks = {k: (None if getattr(eris, k) is None else pad(getattr(eris, k), k))
              for k in ('oooo', 'ooov', 'oovv', 'ovov', 'ovvo', 'ovvv', 'vvvv')}
    out = ERIs(nocc=no2, fock=fock, mo_energy=e, e_hf=eris.e_hf, e_nuc=eris.e_nuc,
               source=(eris.source or '') + ' + ghost', **blocks)
    pp = getattr(eris, '_vvvv_pp', None)
    if pp is not None:
        if kind == 'ip':            # the pairs of nv + 1 virtual indices that do not have the last one
            a, b = np.triu_indices(nv2, 1)
            sel = np.flatnonzero(b < nv)
            new = np.zeros((len(a), len(a)), dtype=np.asarray(pp).dtype)
            new[np.ix_(sel, sel)] = pp
            out._vvvv_pp = new
        else:
            out._vvvv_pp = np.ascontiguousarray(pp)
    t1p, t2p = pad(t1, 'ov'), pad(t2, 'oovv')
    t3p = None
    if t3 is not None:
        from math import comb
        io = np.array([k for k, (i, j, l) in enumerate(_triples(no2)) if kind == 'ip' or i > 0])
        iv = np.array([k for k, (a, b, c) in enumerate(_triples(nv2)) if kind == 'ea' or c < nv])
        t3p = np.zeros((comb(no2, 3), comb(nv2, 3)), dtype=np.asarray(t3).dtype)
        if len(io) and len(iv):
            t3p[np.ix_(io, iv)] = np.asarray(t3).reshape(comb(no, 3), comb(nv, 3))
        t3p = t3p.ravel()
    singles = np.zeros((no2, nv2), bool)
    if kind == 'ip':
        singles[:, nv] = True
    else:
        singles[0, :] = True
    return out, t1p, t2p, t3p, singles


def _triples(n):
    from itertools import combinations
    return combinations(range(n), 3)


@dataclass
class Short:
    '''The IP or EA problem with short vectors (ghost_short).'''
    kind: str
    eris: object                # the integrals with the ghost spinor and its label
    t1: np.ndarray              # the amplitudes there: t1, t2 padded, t3 as it was
    t2: np.ndarray
    t3: np.ndarray
    ghost_label: int
    step: int
    labelled: bool              # the spinors had symmetry labels in use

    def class_shift(self, label=None):
        '''The shift (classes of the maps of self.eris) of the vectors
        with the ghost index once and the label ``label`` of the electron
        that is removed (IP: minus the label of its spinor) or attached
        (EA) - the change of the sum of the labels; None without labels.'''
        if not self.labelled:
            if label not in (None, 0):
                raise ValueError('a label for the %s states needs the symmetry labels in use: a calculation in '
                                 'the blocks of the labels' % self.kind.upper())
            return 1
        if label is None:
            raise ValueError('with symmetry labels in use the %s states are asked for by the label of the '
                             'electron (shift=)' % self.kind.upper())
        try:
            return class_shift(self.eris, self.ghost_label + label if self.kind == 'ip' else label - self.ghost_label)
        except ValueError:
            raise ValueError('no %s state changes the labels by %d (the labels of the electrons: %s)'
                             % (self.kind.upper(), label, self.labels())) from None

    def labels(self):
        '''The labels an electron removed (IP: minus the labels of the
        occupied spinors) or attached (EA: the labels of the virtual ones)
        can carry, sorted.'''
        if not self.labelled:
            return [0]
        lab = np.asarray(self.eris.labels)
        no = self.eris.nocc
        return sorted({-int(x) for x in lab[:no]} if self.kind == 'ip' else {int(x) for x in lab[no:]})

    def strip(self, res, k=0):
        '''Root k of a Result without the ghost index: (r[i], r[(i<j), a],
        r3) for IP, (r[a], r[i, (a<b)], r3) for EA; the pairs in the order
        of the maps of self.eris, r3 flat in the blocks of the shift -
        [(i<j<k), (a<b)] (EA: [(i<j), (a<b<c)]) in the order of the maps
        without labels.'''
        om, vm = t3pack.maps_for(self.eris)
        r1, r2 = res.r1[k], res.r2[k]
        r3 = res.r3[k] if res.r3 else None
        if self.kind == 'ip':
            x = vm.n - 1
            return r1[:, x].copy(), np.ascontiguousarray(r2[:, vm.pidx[np.arange(x), x]]), r3
        return r1[0].copy(), np.ascontiguousarray(r2[om.pidx[0, np.arange(1, om.n)]]), r3


def ghost_short(eris, t1, t2, t3=None, kind='ip'):
    """The IP or EA problem as an EE problem with a ghost spinor (ghost())
    whose vectors are short: the ghost carries a symmetry label of its own,
    far from the labels of the others, so the vectors with the ghost index
    once are the vectors of one label shift, and their rank-3 part is
    stored and passed over in the blocks of that shift
    (t3shift.ShiftLayout) - r[(i<j<k), (a<b)] (IP), r[(i<j), (a<b<c)]
    (EA) and nothing else.  The passes of a sigma vector then lose a
    virtual (occupied) index wherever the ghost takes its place.

    Without symmetry labels in use the spinors get one label and the ghost
    another: the short vectors on the packed layout, all states.  With
    labels (n classes, the blocks of a linear molecule) the ghost is
    3 (n - 1) + 1 classes beyond the last (IP) or before the first (EA)
    of them: no element without the ghost has the shifts of the elements
    with it, and the label of the electron that is removed or attached
    selects the states (Short.class_shift).  t3: flat, in the layout of
    the engine of ``eris``; it is the T3 of the new integrals as it is.
    Returns a Short."""
    if getattr(eris, 'spin_occ', None) is not None:
        raise ValueError('IP and EA with short vectors: spin orbitals without spin blocks, or spinors')
    labelled = t3pack.classes_of(eris) is not None
    n = np.asarray(eris.fock).shape[0]
    if labelled:
        lab = np.asarray(eris.labels, dtype=np.int64)
        u = np.unique(lab)
        step = int(np.gcd.reduce(np.diff(u))) if len(u) > 1 else 1
        gap = step * (3 * int((u[-1] - u[0]) // step) + 1)
    else:
        lab, step, gap = np.zeros(n, dtype=np.int64), 1, 1
    glab = int(lab.max() + gap) if kind == 'ip' else int(lab.min() - gap)
    e2, t1p, t2p, _, _ = ghost(eris, t1, t2, None, kind)
    e2.labels = np.concatenate((lab, [glab])) if kind == 'ip' else np.concatenate(([glab], lab))
    e2.modulus, e2.use_labels = 0, True
    return Short(kind, e2, t1p, t2p, t3, glab, step, labelled)


def kernel(eris, t1, t2, t3=None, nroots=3, conv_tol=1e-6, max_space=None, max_cycle=60, guess=None,
           nguess=None, guess_tol=1e-4, opts=None, mem_mb=2000.0, verbose=False, plan=None, ground=None,
           stdout=None, singles=None, name='EE', shift=None, sector=None, complete=False, extra=2, model='cc'):
    '''EOM-EE-CCSDT for converged amplitudes (t3 flat in the layout of the
    engine of ``eris``; None: EOM-EE-CCSD for CCSD amplitudes): the
    ``nroots`` lowest excitation energies and right-hand vectors.

    ``guess``: [(r1[i,a], r2[(i<j),(a<b)])] to start from (followed);
    default: the roots of the singles-doubles block from ``nguess`` (default
    nroots) unit vectors on the lowest singles, converged to ``guess_tol``.
    With symmetry labels in use the states are the totally symmetric ones.
    ``singles`` [i,a]: the single excitations the unit vectors of the guess
    are taken from (default: the totally symmetric ones); with the
    integrals and amplitudes of ghost() they select the IP or EA problem.
    ``shift``: with symmetry labels in use, the states whose excitation
    operator changes the sum of the labels by ``shift`` (in classes of the
    maps: class_shift()) - r1[i,a] where l_a - l_i = shift, the rank-3
    part in the blocks of t3shift.ShiftLayout; default: the totally
    symmetric states, in the layout of t3.
    ``sector`` = (labels, D): on the packed block (no labels in use),
    the states whose excitation operator changes the sum of the labels
    ``labels`` (one integer per spinor) by D - the guess is taken from the
    singles with l_a - l_i = D and every sigma vector is projected on the
    elements of that shift (Sector): the other components of a degenerate
    level, which have other shifts, are not computed.
    ``extra``: the singles-doubles stage solves for this many roots more
    than asked for, to see whether ``nroots`` cuts a degenerate level (the
    next roots within 2e-6 Eh of the last one): then a warning is printed
    and Result.cut_level is set, or with ``complete`` the rest of the
    level is computed too (more roots than asked for come back).  Where
    the single excitations are fewer than the vectors wanted, the guess
    goes on with unit vectors on the lowest doubles; if fewer roots than
    asked for can be started, that is said (a warning), not passed over.
    ``model``: 'cc2' (t3 None) or 'cc3': the amplitudes are those of that
    model and the matrix is its Jacobian (docs/ccn.md) - the excitation
    energies of CC2 / CC3 linear response.
    ``conv_tol``: the norm of the residual of a normalized vector.
    ``opts``: solver.Options for max_memory, max_disk, incore_complete.
    Returns a Result.'''
    import sys
    from zccsdt import disk
    from zccsdt.cc import solver
    opts = opts or solver.Options()
    out = stdout or sys.stdout
    with3 = t3 is not None
    dtype = np.dtype(np.result_type(t1, t2, eris.oovv, eris.fock, *([t3] if with3 else [])))
    if plan is None:
        plan = plan_for(eris, dtype, nroots, max_space, with3, opts, mem_mb, shift=shift)
    if plan.mapped:
        disk.require(plan.disk, 'the vectors of the EOM subspace', opts.max_disk)
    t_all = time.perf_counter()
    g = ground if ground is not None else eom_opt.prepare(eris, t1, t2, t3, opts, mem_mb, model=model)
    sizes = g.sizes(shift)
    nocc, nvir = g.om.n, g.vm.n
    times = {'prepare': time.perf_counter() - t_all}

    def say(msg):
        if verbose:
            print(msg, file=out, flush=True)

    def note(msg):
        import warnings
        say('  warning: ' + msg)
        warnings.warn('EOM-%s: %s' % (name, msg), stacklevel=3)
    cut_level = False

    def logger(stage):
        def log(it, n, w, res, conv, sec):
            say('%s %3d  %3d vectors  %s  %.2f s' % (
                stage, it, n, '  '.join('%.8f %.1e%s' % (x.real, r, '*' if c else ' ')
                                        for x, r, c in zip(w, res, conv)), sec))
        return log

    sec = None
    if sector is not None:
        if shift is not None or g.om.spins is not None:
            raise ValueError('sector: for the packed block; in the blocks of the labels the states are chosen by shift')
        sec = Sector(g.om, g.vm, sector[0], sector[1])

    def apply_for(with_r3):
        def apply(x, y):
            s1, s2, _ = eom_opt.sigma(g, x[0].reshape(nocc, nvir), x[1], x[2] if with_r3 else None,
                                      out=y[2] if with_r3 else None, sd=not with_r3, shift=shift)
            y[0][:] = s1.ravel()
            y[1][:] = s2.ravel()
            if sec is not None:
                sec.project(y)
        return apply

    say('EOM-%s-%s: %d roots, vectors of %d + %d + %d elements, %s' % (
        name, {'cc2': 'CC2', 'cc3': 'CC3'}.get(g.model, 'CCSDT' if with3 else 'CCSD'), nroots, sizes[0], sizes[1], sizes[2],
        'real' if dtype.kind == 'f' else 'complex'))
    if shift is not None:
        m1, m2 = g.masks(shift)
        say('  label shift: %d classes; %d singles, %d doubles and %d triples have it'
            % (shift, m1.sum(), m2.sum(), sizes[2]))
    if sec is not None:
        say('  label shift %+d on the packed block: %d singles, %d doubles and %d triples have it'
            % (sec.shift, sec.m1.sum(), sec.m2.sum(), sec.n3 if with3 else 0))
    say('  ' + plan.text())
    if g.change > 1e-5:
        say('  warning: the amplitudes are not converged (a Jacobi step changes them by %.1e)' % g.change)
    # ---- the singles-doubles stage
    sd = Store(sizes[:2], dtype)
    first = None
    lowest = False
    t0 = time.perf_counter()
    if guess is None:
        allowed = symmetric_singles(eris, shift or 0) if sec is None else sec.m1
        if singles is not None:
            allowed = allowed & np.asarray(singles, bool)
        want = max(nguess or 0, nroots + extra)
        idx = singles_guess(g, want, allowed)
        gs = []
        for k in idx:
            v = sd.new()
            v[0][k] = 1.0
            gs.append(v)
        if not gs:
            raise ValueError('no single excitation to start from%s' % (
                '' if shift is None else ' with a label shift of %d classes' % shift))
        if len(gs) < want:
            # fewer singles than vectors wanted (IP of three electrons: three): the lowest doubles
            d2 = denominators(g)[1].copy()
            if shift is not None:
                d2[~g.masks(shift)[1].ravel()] = np.inf
            elif sec is not None:
                d2[~sec.m2.ravel()] = np.inf
            elif singles is not None or g.om.spins is not None:
                d2[:] = np.inf          # (which doubles belong to these singles is not known here)
            for k in np.argsort(d2, kind='stable')[:want - len(gs)]:
                if np.isfinite(d2[k]):
                    v = sd.new()
                    v[1][k] = 1.0
                    gs.append(v)
        asked = nroots
        nroots = min(nroots, len(gs))
        if nroots < asked:
            note('only %d of the %d roots asked for can be started: no more single%s excitations to start from'
                 % (nroots, asked, '' if singles is not None else ' or double'))
        final = not with3
        nsolve = min(len(gs), nroots + extra)
        first = davidson(apply_for(False), make_precond(g, False), gs, sd, nsolve,
                         conv_tol if final else guess_tol, max(4 * nsolve + 16, plan.max_space if final else 0),
                         max_cycle, log=logger('SD   ' if with3 else 'iter '))
        # does nroots cut a degenerate level?  (the roots beyond it, by the singles-doubles stage)
        order = np.argsort(first.e.real, kind='stable')
        e_sd = first.e.real[order]
        k = nroots
        while k < len(e_sd) and e_sd[k] - e_sd[k - 1] < 2e-6:
            k += 1
        if k > nroots:
            more = ' (and the level may go on: raise nroots)' if k == len(e_sd) else ''
            if complete:
                say('  %d roots asked for cut a degenerate level at %.8f Eh: %d roots are computed%s'
                    % (nroots, e_sd[nroots - 1], k, more))
                nroots = k
            else:
                cut_level = True
                note('%d roots cut a degenerate level at %.8f Eh (%s%d roots would complete it%s; complete=True '
                     'does that)' % (nroots, e_sd[nroots - 1], 'at least ' if more else '', k, ''))
        keep = order[:nroots]
        # the triples stage starts from all of them and takes the lowest roots: a state that
        # the singles-doubles block puts above its neighbours (the M_S = -1 component of the
        # triplet of Li+, a 2h1p state for a UHF reference) comes down with the triples
        start = [first.vectors[j] for j in order]
        lowest = True
        first = Solved(first.e[keep], [first.vectors[j] for j in keep], first.converged[keep], first.residuals[keep],
                       first.niter, first.nsigma, first.ncollapse, first.history)
    else:
        start = [[np.array(a, dtype=dtype).ravel(), np.array(b, dtype=dtype).ravel()] for a, b in guess]
        if sec is not None:
            start = [sec.project(x) for x in start]
        nroots = min(nroots, len(start))
    times['guess'] = time.perf_counter() - t0
    if not with3:
        if first is None:
            first = davidson(apply_for(False), make_precond(g, False), start, sd, nroots, conv_tol,
                             max(4 * nroots + 16, plan.max_space), max_cycle, follow=True, log=logger('iter '))
        res, store = first, sd
    else:
        store = Store(sizes, dtype, plan.mapped)
        gs = []
        for x in start:
            v = store.new()
            v[0][:], v[1][:] = x[0], x[1]
            gs.append(v)
        t0 = time.perf_counter()
        res = davidson(apply_for(True), make_precond(g, True, shift=shift), gs, store, nroots, conv_tol,
                       max(plan.max_space, len(gs) + nroots), max_cycle, follow=not lowest, log=logger('iter '))
        times['iterations'] = time.perf_counter() - t0
    wts = np.array([[float(np.vdot(p, p).real) if p.size else 0.0 for p in v] for v in res.vectors])
    wts = np.concatenate([wts, np.zeros((len(wts), 3 - wts.shape[1]))], axis=1)
    order = np.argsort(res.e.real, kind='stable')
    vec = [res.vectors[k] for k in order]
    out_res = Result(e=res.e.real[order], e_imag=res.e.imag[order], converged=res.converged[order],
                     residuals=res.residuals[order],
                     r1=[v[0].reshape(nocc, nvir) for v in vec], r2=[v[1].reshape(g.om.npair, g.vm.npair) for v in vec],
                     r3=[v[2] for v in vec] if with3 else None, weights=wts[order], niter=res.niter,
                     nsigma=res.nsigma, ncollapse=res.ncollapse, guess=first if with3 else None,
                     history=res.history, plan=plan, ground=g, timings=times, shift=shift, cut_level=cut_level)
    times['total'] = time.perf_counter() - t_all
    say('  root   omega / Eh       omega / eV   |residual|   %R1   %R2   %R3')
    for k in range(len(out_res.e)):
        say('  %3d  %14.10f  %11.6f   %.1e%s  %5.1f %5.1f %5.1f' % (
            k + 1, out_res.e[k], out_res.e[k] * HARTREE_EV, out_res.residuals[k],
            ' ' if out_res.converged[k] else '!', *(100 * out_res.weights[k] / out_res.weights[k].sum())))
    say('  %d iterations, %d sigma vectors, %d collapses; %.1f s' % (res.niter, res.nsigma, res.ncollapse, times['total']))
    return out_res
