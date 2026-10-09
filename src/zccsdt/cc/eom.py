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

import time
from dataclasses import dataclass

import numpy as np
import scipy.linalg

from zccsdt.cc import ccsdt_opt, diis, eom_opt, t3pack, zeta_opt
from zccsdt.timing import timer

HARTREE_EV = 27.211386245988
CHUNK = 1 << 21


# ------------------------------------------------------------------- vectors

def vdot(x, y):
    '''<x|y> of two vectors (lists of flat parts).'''
    return sum(diis._dot(a, b) for a, b in zip(x, y) if a.size)


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
        for j in range(n0, n):
            for i in range(n):
                Gn[i, j] = vdot(V[i], S[j])
            for i in range(n0):
                Gn[j, i] = vdot(V[j], S[i])
        G = Gn
        ew, ev = scipy.linalg.eig(G)
        if prev is not None:        # the roots that the previous ones have become
            pv = np.zeros((n, prev.shape[1]), dtype=complex)
            pv[:prev.shape[0]] = prev
            ov = abs(pv.conj().T @ (ev / np.linalg.norm(ev, axis=0)))
            pick = []
            for k in range(prev.shape[1]):
                order = np.argsort(-ov[k])
                pick.append(next(int(x) for x in order if x not in pick))
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
            c = [vdot(b, t) for b in basis]
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


def fixed_bytes(eris, dtype, with_t3=True, mem_mb=2000.0):
    '''Bytes of ordinary memory that prepare() and a sigma vector take
    besides the vectors of the subspace: the larger of

      prepare   an amplitude step with everything in memory and its
                operands kept: two more arrays of the size of <ia||bc>;
      sigma     the kept operands (the pp operator; <ia||bc>~, its packed
                form, the bra-dressed <ab||ei>, W_vvvo and the operand of
                pass 1: 3.5 f; T3; 16 arrays of o^2 v^2), the derivative of
                the pp operator, 3.5 f of transients (W'_vvvo, its parts and
                the drive operand), the batch buffers and the occupied
                operators.'''
    om, vm = t3pack.maps_for(eris)
    o, v = om.n, vm.n
    isz = np.dtype(dtype).itemsize
    eng = ccsdt_opt.engine_for(eris, dtype)
    f = o * v ** 3 * isz
    s = o * o * v * v * isz
    pp = sum((hi - lo) ** 2 for lo, hi in vm.blocks) * isz
    t3 = eng.size * isz if with_t3 else 0
    step = ccsdt_opt.plan_memory(o, v, isz, mem_mb=mem_mb, t3_bytes=eng.size * isz, pp_bytes=pp, engine=eng)
    prep = step.need + 2 * f
    occ = 2 * om.ntri ** 2 * isz if o < t3kern_min_occ() else 4 * om.npair ** 2 * isz
    sig = 2 * pp + 7 * f + 22 * s + t3 + int(eng.buffer_bytes(mem_mb)) + 2 * occ + ccsdt_opt.overhead()
    if not with_t3:
        sig = 2 * pp + 3 * f + 22 * s + ccsdt_opt.overhead()
    return max(prep, sig)


def t3kern_min_occ():
    from zccsdt.cc import t3kern
    return t3kern.HH_PAIR_MIN_OCC


def plan_for(eris, dtype, nroots, max_space=None, with_t3=True, opts=None, mem_mb=2000.0):
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
    one = (om.n * vm.n + om.npair * vm.npair + (eng.size if with_t3 else 0)) * isz
    fixed = fixed_bytes(eris, dtype, with_t3, mem_mb)
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
    big = (eng.size if with_t3 else 0) * isz        # (the parts of rank one and two stay in memory)
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


def make_precond(g, with_t3, floor=1e-3):
    d1, d2, (eo3, ev3) = denominators(g)
    blocks = zeta_opt._blocks(g.eris, g.dtype) if with_t3 else []

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


def symmetric_singles(eris):
    '''[i,a] True where a single excitation is totally symmetric for the
    labels in use (all of them without labels).'''
    om, vm = t3pack.maps_for(eris)
    if om.spins is None:
        return np.ones((om.n, vm.n), bool)
    return np.asarray(om.spins)[:, None] == np.asarray(vm.spins)[None, :]


def kernel(eris, t1, t2, t3=None, nroots=3, conv_tol=1e-6, max_space=None, max_cycle=60, guess=None,
           nguess=None, guess_tol=1e-4, opts=None, mem_mb=2000.0, verbose=False, plan=None, ground=None,
           stdout=None):
    '''EOM-EE-CCSDT for converged amplitudes (t3 flat in the layout of the
    engine of ``eris``; None: EOM-EE-CCSD for CCSD amplitudes): the
    ``nroots`` lowest excitation energies and right-hand vectors.

    ``guess``: [(r1[i,a], r2[(i<j),(a<b)])] to start from (followed);
    default: the roots of the singles-doubles block from ``nguess`` (default
    nroots) unit vectors on the lowest singles, converged to ``guess_tol``.
    With symmetry labels in use the states are the totally symmetric ones.
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
        plan = plan_for(eris, dtype, nroots, max_space, with3, opts, mem_mb)
    if plan.mapped:
        disk.require(plan.disk, 'the vectors of the EOM subspace', opts.max_disk)
    t_all = time.perf_counter()
    g = ground if ground is not None else eom_opt.prepare(eris, t1, t2, t3, opts, mem_mb)
    sizes = g.sizes()
    nocc, nvir = g.om.n, g.vm.n
    times = {'prepare': time.perf_counter() - t_all}

    def say(msg):
        if verbose:
            print(msg, file=out, flush=True)

    def logger(stage):
        def log(it, n, w, res, conv, sec):
            say('%s %3d  %3d vectors  %s  %.2f s' % (
                stage, it, n, '  '.join('%.8f %.1e%s' % (x.real, r, '*' if c else ' ')
                                        for x, r, c in zip(w, res, conv)), sec))
        return log

    def apply_for(with_r3):
        def apply(x, y):
            s1, s2, _ = eom_opt.sigma(g, x[0].reshape(nocc, nvir), x[1], x[2] if with_r3 else None,
                                      out=y[2] if with_r3 else None, sd=not with_r3)
            y[0][:] = s1.ravel()
            y[1][:] = s2.ravel()
        return apply

    say('EOM-EE-%s: %d roots, vectors of %d + %d + %d elements, %s' % (
        'CCSDT' if with3 else 'CCSD', nroots, sizes[0], sizes[1], sizes[2], 'real' if dtype.kind == 'f' else 'complex'))
    say('  ' + plan.text())
    if g.change > 1e-5:
        say('  warning: the amplitudes are not converged (a Jacobi step changes them by %.1e)' % g.change)
    # ---- the singles-doubles stage
    sd = Store(sizes[:2], dtype)
    first = None
    t0 = time.perf_counter()
    if guess is None:
        idx = singles_guess(g, nguess or nroots, symmetric_singles(eris))
        gs = []
        for k in idx:
            v = sd.new()
            v[0][k] = 1.0
            gs.append(v)
        nroots = min(nroots, len(gs))
        final = not with3
        first = davidson(apply_for(False), make_precond(g, False), gs, sd, nroots,
                         conv_tol if final else guess_tol, max(4 * nroots + 16, plan.max_space if final else 0),
                         max_cycle, log=logger('SD   ' if with3 else 'iter '))
        start = first.vectors
    else:
        start = [[np.array(a, dtype=dtype).ravel(), np.array(b, dtype=dtype).ravel()] for a, b in guess]
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
        res = davidson(apply_for(True), make_precond(g, True), gs, store, nroots, conv_tol, plan.max_space,
                       max_cycle, follow=True, log=logger('iter '))
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
                     history=res.history, plan=plan, ground=g, timings=times)
    times['total'] = time.perf_counter() - t_all
    say('  root   omega / Eh       omega / eV   |residual|   %R1   %R2   %R3')
    for k in range(len(out_res.e)):
        say('  %3d  %14.10f  %11.6f   %.1e%s  %5.1f %5.1f %5.1f' % (
            k + 1, out_res.e[k], out_res.e[k] * HARTREE_EV, out_res.residuals[k],
            ' ' if out_res.converged[k] else '!', *(100 * out_res.weights[k] / out_res.weights[k].sum())))
    say('  %d iterations, %d sigma vectors, %d collapses; %.1f s' % (res.niter, res.nsigma, res.ncollapse, times['total']))
    return out_res
