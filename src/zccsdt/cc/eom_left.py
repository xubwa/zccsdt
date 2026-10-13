'''Left-hand eigenvectors of EOM-EE-CCSDT (docs/eom.md, section 12):

    sum_nu l_nu A[nu, mu] = omega l_mu,      A[nu, mu] = <nu| [Hbar, tau_mu] |0>,

A the Jacobian of the amplitude equations, whose right-hand eigenvectors
zccsdt.cc.eom computes.  Nothing is conjugated: L = sum_mu l_mu <mu| over
the unique elements, as Lambda is held (zccsdt.cc.lambda_opt).

**The left product is the Lambda residual without its constant term.**  The
Lambda equations are the derivatives of the Lagrangian with respect to the
amplitudes,

    dL/dt_mu = <0| (1 + Lambda) [Hbar, tau_mu] |0> = eta_mu + sum_nu lambda_nu A[nu, mu],
    eta_mu   = <0| [Hbar, tau_mu] |0>:       eta_1 = f~[i,a], eta_2 = <ij||ab>, eta_3 = 0

for any amplitudes, converged or not.  So lambda_opt.residuals() with l in
the place of Lambda, less (f~[i,a], <ij||ab>), is l A: no term is written
here, and the engines of the Lambda equations - lambda_opt.DenseL3 on the
packed block, lambda_blk.BlockL3 in the blocks of the symmetry labels -
serve as they are.  An excited state has no component on the reference:
the column of the reference of L (Hbar - E0) = omega L reads l0 omega = 0.

A left vector of a state with the label shift D has its elements where
those of the right vector are - l1[i,a] where l_a - l_i = D, the rank-3
part in the blocks of t3shift.ShiftLayout(D) - (as an operator it lowers
the sum of the labels by D: the shift -D).  Where the Lambda engine has the
layout of the vector the product runs in it: the packed block without
labels (states of one shift by masks: eom.Sector), and the blocks of the
labels for the totally symmetric states.  A state with a shift in the
blocks of the labels: the Lambda engine of the blocks is written for
arrays without a shift, so the vector is put on the packed block, the
product is taken there and projected on the shift - right, at the cost and
with the memory of the packed block (LeftOps.mode == 'dense').

The solver is that of the right-hand vectors (eom.davidson) on the
transposed matrix, started from the conjugates of the converged right-hand
vectors, which it follows: an eigenvalue problem, not a linear equation at
the known omega - (A^T - omega) l = 0 is singular at the solution, and the
Rayleigh quotient of the start is omega to second order anyway.  Then

    sum_mu l_mu(k) r_mu(m) = delta_km:

L_k scaled by its product with R_k; inside a degenerate level (the
components of a multiplet on the packed block) the left vectors are
replaced by the combinations that are biorthogonal to the right ones.
'''

import time
from dataclasses import dataclass

import numpy as np

from zccsdt.cc import ccsdt_opt, eom, lambda_opt, t3pack
from zccsdt.timing import timer


def _dotu(x, y):
    '''sum_mu x_mu y_mu of two vectors (lists of flat parts): nothing conjugated.'''
    return sum(np.dot(np.asarray(a).ravel(), np.asarray(b).ravel()) for a, b in zip(x, y) if a.size)


class LeftOps:
    '''The left product l -> l A for the converged amplitudes (t1, t2 dense,
    t3 flat in the layout of the engine of ``eris``), for vectors in the
    layout of the right-hand vectors of eom.kernel(shift=, sector=): the
    intermediates of the Lambda equations, made once.

    mode 'same'   the Lambda engine has the layout of the vectors
         'dense'  labels in use and a shift (or no block engine): the
                  rank-3 part goes to the packed block in lexicographic
                  order and back, the product is projected on the shift.'''

    def __init__(self, eris, t1, t2, t3, shift=None, sector=None, mem_mb=2000.0, dense=False, plan=None):
        if ccsdt_opt._is_spin_eris(eris):
            raise NotImplementedError('left-hand vectors take spin-orbital or spinor integrals (zccsdt.eris)')
        if t3 is None:
            raise NotImplementedError('left-hand vectors: EOM-CCSDT (the Lambda machinery of CCSDT)')
        om, vm = t3pack.maps_for(eris)
        self.eris, self.om, self.vm = eris, om, vm
        self.shift, self.sector = shift, None
        self.dtype = dtype = np.dtype(np.result_type(t1, t2, t3, eris.oovv, eris.fock))
        labelled = om.spins is not None
        if sector is not None:
            if labelled or shift is not None:
                raise ValueError('sector: for the packed block; in the blocks of the labels the states are chosen by shift')
            self.sector = eom.Sector(om, vm, sector[0], sector[1])
        blocks = labelled and not shift and not dense and lambda_opt.block_engine(eris, dtype) is not None
        self.mode = 'same' if (blocks or not labelled) else 'dense'
        self.blocks = bool(blocks)
        self.masks = None
        mem = mem_mb if plan is None else plan.lam.mem_mb
        t0 = time.perf_counter()
        if self.mode == 'same':
            self.view, self.T3 = eris, np.ascontiguousarray(np.asarray(t3, dtype=dtype).ravel())
            if labelled:
                self.masks = _masks(om, vm, 0)
        else:
            import copy
            eng = ccsdt_opt.engine_for(eris, dtype)
            self.layout = eng.L if shift is None else eom._shifter(eng).layout(shift)
            self.T3 = np.ascontiguousarray(ccsdt_opt.packed_t3(eris, np.asarray(t3, dtype=dtype)))
            self.view = copy.copy(eris)
            self.view.use_labels = False
            self.view.__dict__.pop('_buffers', None)
            omd, vmd = t3pack.maps_for(self.view)
            assert omd.spins is None
            self.lex = (ccsdt_opt._lex(om), ccsdt_opt._lex(vm))
            # the pairs of the maps of the vectors among the lexicographic ones
            self.po = omd.pidx[om.pairs.min(axis=1), om.pairs.max(axis=1)]
            self.pv = vmd.pidx[vm.pairs.min(axis=1), vm.pairs.max(axis=1)]
            assert (om.pairs[:, 0] < om.pairs[:, 1]).all() and (vm.pairs[:, 0] < vm.pairs[:, 1]).all()
            lab = np.concatenate((np.asarray(om.spins), np.asarray(vm.spins))).astype(np.int64)
            self.project = eom.Sector(omd, vmd, lab, shift or 0)
            self.buf = None
        with timer('eom.left.prepare'):
            self.im = lambda_opt.make_intermediates(self.view, t1, t2, self.T3, mem_mb=mem, blocks=self.blocks)
        self.setup = time.perf_counter() - t0
        self.nocc, self.nvir = om.n, vm.n

    # ---- the rank-3 part between the layout of the vectors and the packed block
    def to_packed(self, x3):
        out = np.zeros((self.om.ntri, self.vm.ntri), dtype=self.dtype)
        out[np.ix_(*self.lex)] = self.layout.to_dense(np.asarray(x3))
        return out

    def from_packed(self, p):
        return self.layout.from_dense(np.asarray(p).reshape(self.om.ntri, self.vm.ntri)[np.ix_(*self.lex)])

    def apply(self, x, y):
        '''y = x A for the vector x = [l1[i,a] flat, l2[(i<j),(a<b)] flat,
        l3 flat] (pairs of the maps of the integrals); y: a vector to fill.'''
        im, om, vm = self.im, self.om, self.vm
        l1 = np.asarray(x[0]).reshape(self.nocc, self.nvir)
        l2 = t3pack.unpack_pairs(np.asarray(x[1]), om, vm)
        if self.mode == 'same':
            r1, r2, R = lambda_opt.residuals(im, l1, l2, x[2], self.T3, out=y[2] if y[2].size else None)
            y[0][:] = (r1 - im.fov).ravel()
            y[1][:] = (r2 - im.goovv_pp).ravel()
            if y[2].size and not np.shares_memory(R, y[2]):
                y[2][:] = np.asarray(R).ravel()
            if self.masks is not None:
                y[0] *= self.masks[0].ravel()
                y[1] *= self.masks[1].ravel()
            if self.sector is not None:
                self.sector.project(y)
            return y
        l3 = self.to_packed(x[2])
        if self.buf is None:
            self.buf = np.empty(l3.size, dtype=self.dtype)
        r1, r2, R = lambda_opt.residuals(im, l1, l2, l3, self.T3, out=self.buf if l3.size else None)
        del l3
        z = [np.ascontiguousarray((r1 - im.fov).ravel()), np.ascontiguousarray((r2 - im.goovv_pp).ravel()),
             np.asarray(R).reshape(-1)]
        self.project.project(z)
        y[0][:] = z[0]
        y[1][:] = self._pairs_back(z[1])
        y[2][:] = self.from_packed(z[2])
        return y

    def _pairs_back(self, r2d):
        '''[(i<j),(a<b)] in lexicographic order of the pairs -> in the order of the maps.'''
        npo, nvp = len(self.po), len(self.pv)
        return np.ascontiguousarray(np.asarray(r2d).reshape(npo, nvp)[np.ix_(self.po, self.pv)]).ravel()

    def eta(self):
        '''(eta_1[i,a], eta_2[(i<j),(a<b)]) = <0| [Hbar, tau_mu] |0>, the
        constant term of the Lambda equations, in the layout of the vectors:
        the row of the reference of Hbar - E0.'''
        e1, e2 = np.array(self.im.fov), np.array(self.im.goovv_pp)
        if self.mode == 'dense':
            e2 = self._pairs_back(e2).reshape(e2.shape)
        return e1, e2


def _masks(om, vm, shift):
    return (np.asarray(vm.spins)[None, :] - np.asarray(om.spins)[:, None] == shift,
            np.asarray(vm.pair_grp)[None, :] - np.asarray(om.pair_grp)[:, None] == shift)


# -------------------------------------------------------------------- memory

@dataclass
class Plan:
    '''Memory and disk of kernel() (plan_for).'''
    nroots: int
    max_space: int              # vectors of the subspace at most
    mapped: bool                # their rank-3 parts are in a scratch file
    need: int                   # bytes of ordinary memory: the Lambda machinery with T3 (which is there) and the subspace
    vectors: int                # bytes of the 2 max_space vectors (memory or disk)
    disk: int
    lam: object                 # the lambda_opt.LambdaPlan of the product
    mode: str = 'same'
    blocks: bool = False

    def text(self):
        return ('left product %s; subspace of at most %d vectors (%.2f GB with their products) in %s; %.2f GB of memory'
                % ('in the blocks of the labels' if self.blocks else
                   'on the packed block' + (' (vectors of a shift put there)' if self.mode == 'dense' else ''),
                   self.max_space, self.vectors / 1e9, 'a scratch file' if self.mapped else 'memory', self.need / 1e9))


def plan_for(eris, t1, dtype, nroots, sizes, max_space=None, opts=None, shift=None, dense=False):
    '''The Plan of the left-hand vectors of ``nroots`` states: the plan of
    the Lambda equations for the product (lambda_opt.plan_for: T3 is taken
    to be there, Lambda_3 and its residual are the vector and the product)
    and a subspace of ``max_space`` (default 4 + 3 nroots; lowered, to
    2 nroots + 2 at least, to what fits) vectors of the lengths ``sizes``
    with their products - in memory if ``opts.max_memory`` holds them, else
    their rank-3 parts in a scratch file.  A state with a shift in the
    blocks of the labels ('dense'): the packed T3 and the two copies of a
    conversion besides.  MemoryError / DiskFullError with the numbers.'''
    from zccsdt import disk
    from zccsdt.cc import solver
    opts = opts or solver.Options()
    om, vm = t3pack.maps_for(eris)
    isz = np.dtype(dtype).itemsize
    labelled = om.spins is not None
    blocks = labelled and not shift and not dense and lambda_opt.block_engine(eris, dtype) is not None
    mode = 'same' if (blocks or not labelled) else 'dense'
    view = eris
    if mode == 'dense':
        import copy
        view = copy.copy(eris)
        view.use_labels = False
        view.__dict__.pop('_buffers', None)
    one = sum(sizes) * isz
    big = sizes[2] * isz
    pk = om.ntri * vm.ntri * isz
    asked = max_space or 4 + 3 * nroots
    floor = min(asked, 2 * nroots + 2)
    # no DIIS vectors here; what the plan of the Lambda equations counts for them is not asked for
    lopts = solver.Options(mem_mb=opts.mem_mb, max_memory=opts.max_memory, max_disk=opts.max_disk,
                           incore_complete=opts.incore_complete, diis_space=0)

    def lam(reserve):
        return lambda_opt._plan_for(view, t1, dtype, lopts, blocks, int(reserve))
    left = None if opts.incore_complete else disk.budget(opts.max_disk)
    err = None
    for mapped in (False, True):
        if mapped and (opts.incore_complete or not big):
            break
        for space in range(asked, floor - 1, -1):
            if mode == 'same':
                # in memory, a vector and its product are Lambda_3 and the residual of the plan
                own = 2 * space * (one - big) if mapped else 2 * space * one - 2 * big
            else:
                # T3 on the packed block (the plan takes it to be there), and the copies of a
                # conversion (the array in the order of the maps and the gathered one)
                own = (2 * space * (one - big) if mapped else 2 * space * one) + 3 * pk
            if mapped and left is not None and 2 * space * big > left:
                continue
            try:
                p = lam(max(own, 0))
            except MemoryError as e:
                err = err or e
                continue
            return Plan(nroots, space, mapped, p.need, 2 * space * one, 2 * space * big if mapped else 0, p, mode,
                        bool(blocks))
    if err is None:
        raise disk.DiskFullError('the subspace of the left-hand vectors of %d roots needs at least %d vectors of '
                                 '%.2f GB with their products; the scratch disk has %.2f GB (max_disk, ZCCSDT_TMPDIR)'
                                 % (nroots, floor, one / 1e9, (left or 0) / 1e9))
    raise MemoryError('left-hand vectors of %d roots: the Lambda machinery and a subspace of %d vectors of %.2f GB '
                      'with their products do not fit in memory - %s' % (nroots, floor, one / 1e9, err))


# -------------------------------------------------------------------- driver

@dataclass
class Result:
    e: np.ndarray               # the eigenvalues of the left problem / Hartree (real parts)
    e_imag: np.ndarray
    e_right: np.ndarray         # those of the right-hand vectors they belong to
    converged: np.ndarray
    residuals: np.ndarray       # |l A - omega l| of the vector of norm one
    l1: list                    # [i,a] per root, sum_mu l_mu(k) r_mu(k) = 1
    l2: list                    # [(i<j),(a<b)]
    l3: list                    # flat, the layout of r3
    overlap: np.ndarray         # sum_mu l_mu(k) r_mu(m), after the normalization
    biorthogonality: float      # the largest |overlap[k,m]| between different levels, before the
                                # left vectors of a degenerate level were combined
    levels: list                # the roots by degenerate levels
    niter: int = 0
    nsigma: int = 0
    plan: object = None
    ops: object = None          # the LeftOps (the intermediates of the Lambda equations)
    timings: dict = None
    shift: object = None
    roots: list = None          # the roots of the right-hand Result these are


def kernel(eris, t1, t2, t3, right, roots=None, conv_tol=None, max_space=None, max_cycle=60, opts=None,
           mem_mb=2000.0, verbose=False, stdout=None, sector=None, ops=None, plan=None, dense=False,
           level_tol=1e-6):
    '''The left-hand eigenvectors of the states of ``right`` (an eom.Result
    of eom.kernel for the same integrals and amplitudes; ``roots``: of
    these roots, default all), in the layout of its vectors: with its label
    shift, or ``sector`` = (labels, D) as given there.  ``conv_tol``: the
    norm of l A - omega l for |l| = 1 (default: the threshold of the
    right-hand vectors).  ``ops``: a LeftOps
    of an earlier call (same shift) to use again.  ``dense``: with labels in
    use, the product on the packed block in any case (for the tests).
    Returns a Result; its vectors are normalized to L_k R_k = 1 and
    biorthogonal to the right-hand vectors of the other roots - which the
    eigenvalue equation gives between different levels (the value found
    before anything is done about it is Result.biorthogonality) and a
    linear combination inside a degenerate level (roots within
    ``level_tol`` Eh).'''
    import sys
    from zccsdt import disk
    from zccsdt.cc import solver
    opts = opts or solver.Options()
    out = stdout or sys.stdout
    if right.r3 is None or right.r1 is None:
        raise ValueError('left-hand vectors: the right-hand vectors of EOM-CCSDT are needed (a Result without '
                         'vectors - the time-reversed partner of another shift - has none)')
    g = right.ground
    if g is None or g.model != 'cc':
        raise NotImplementedError('left-hand vectors: EOM-CCSDT (not CC2 / CC3)')
    shift = right.shift
    if conv_tol is None:        # (that of the right-hand vectors: a density has the errors of both to first order)
        conv_tol = getattr(right, 'conv_tol', None) or 1e-6
    roots = list(range(len(right.e))) if roots is None else [int(k) for k in roots]
    n = len(roots)
    dtype = g.dtype
    sizes = g.sizes(shift)
    nocc, nvir = g.om.n, g.vm.n
    t_all = time.perf_counter()

    def say(msg):
        if verbose:
            print(msg, file=out, flush=True)
    if plan is None:
        plan = plan_for(eris, t1, dtype, n, sizes, max_space, opts, shift, dense)
    if plan.mapped:
        disk.require(plan.disk, 'the left-hand vectors of the EOM subspace', opts.max_disk)
    if ops is None:
        ops = LeftOps(eris, t1, t2, t3, shift, sector, mem_mb, dense, plan)
    times = {'prepare': ops.setup}
    say('EOM-EE-CCSDT, left-hand vectors of %d roots: vectors of %d + %d + %d elements, %s'
        % (n, sizes[0], sizes[1], sizes[2], 'real' if dtype.kind == 'f' else 'complex'))
    say('  ' + plan.text())
    store = eom.Store(sizes, dtype, plan.mapped)
    gs = []
    for k in roots:
        v = store.new()
        for p, x in zip(v, (right.r1[k], right.r2[k], right.r3[k])):
            p[:] = np.conj(np.asarray(x)).ravel()
        gs.append(v)

    def log(it, m, w, res, conv, sec):
        say('left %3d  %3d vectors  %s  %.2f s' % (
            it, m, '  '.join('%.8f %.1e%s' % (x.real, r, '*' if c else ' ') for x, r, c in zip(w, res, conv)), sec))
    t0 = time.perf_counter()
    res = eom.davidson(ops.apply, eom.make_precond(g, True, shift=shift), gs, store, n, conv_tol,
                       max(plan.max_space, 2 * n), max_cycle, follow=True, log=log)
    times['iterations'] = time.perf_counter() - t0
    e_right = np.asarray(right.e)[roots]
    # each left vector to the right one it has the largest product with (they were followed,
    # but inside a level and between close levels the order of the roots is not that of the start)
    R = [[np.asarray(right.r1[k]).ravel(), np.asarray(right.r2[k]).ravel(), np.asarray(right.r3[k]).ravel()]
         for k in roots]
    M = np.array([[_dotu(l, r) for r in R] for l in res.vectors])
    order = _assign(abs(M))
    vec = [res.vectors[j] for j in order]
    M = M[order]
    e = np.asarray(res.e)[order]
    conv, rn = np.asarray(res.converged)[order], np.asarray(res.residuals)[order]
    # levels: roots within level_tol of each other
    levels, seen = [], set()
    for k in np.argsort(e_right, kind='stable'):
        if k in seen:
            continue
        lv = [int(m) for m in range(n) if m not in seen and abs(e_right[m] - e_right[k]) < level_tol]
        seen.update(lv)
        levels.append(lv)
    scale = np.sqrt(abs(np.outer(M.diagonal(), M.diagonal())))
    off = abs(M) / np.where(scale > 0, scale, 1.0)
    for lv in levels:
        off[np.ix_(lv, lv)] = 0.0
    biorth = float(off.max()) if n else 0.0
    with timer('eom.vectors'):
        for lv in levels:
            C = np.linalg.inv(M[np.ix_(lv, lv)])            # L' = C L:  L' R = 1 in the level
            C = C.real.copy() if dtype.kind == 'f' else C
            new = [[np.zeros_like(np.asarray(p)) for p in vec[lv[0]]] for _ in lv]
            for a in range(len(lv)):
                for b in range(len(lv)):
                    for p in range(3):
                        if new[a][p].size:
                            new[a][p] += C[a, b] * np.asarray(vec[lv[b]][p])
            for a, k in enumerate(lv):
                for p in range(3):
                    if new[a][p].size:
                        vec[k][p][:] = new[a][p]
            del new
    M2 = np.array([[_dotu(l, r) for r in R] for l in vec])
    times['total'] = time.perf_counter() - t_all
    result = Result(e=e.real.copy(), e_imag=e.imag.copy(), e_right=e_right, converged=conv, residuals=rn,
                    l1=[v[0].reshape(nocc, nvir) for v in vec],
                    l2=[v[1].reshape(g.om.npair, g.vm.npair) for v in vec], l3=[v[2] for v in vec],
                    overlap=M2, biorthogonality=biorth, levels=levels, niter=res.niter, nsigma=res.nsigma,
                    plan=plan, ops=ops, timings=times, shift=shift, roots=roots)
    say('  root   omega(left) / Eh   - omega(right)   |residual|')
    for k in range(n):
        say('  %3d  %16.10f   %10.2e      %.1e%s' % (roots[k] + 1, result.e[k], result.e[k] - e_right[k], rn[k],
                                                      ' ' if conv[k] else '!'))
    say('  L_k R_m between different levels, before the normalization: %.1e at most; after it |L R - 1| = %.1e'
        % (biorth, abs(M2 - np.eye(n)).max() if n else 0.0))
    say('  %d iterations, %d left products; %.1f s' % (res.niter, res.nsigma, times['total']))
    return result


def _assign(W):
    '''For each column m the row with the largest W[row, m], every row once
    (greedy by the size of the elements).'''
    n = W.shape[0]
    order, rows, cols = [None] * n, set(), set()
    for idx in np.argsort(-W, axis=None):
        j, m = divmod(int(idx), n)
        if j in rows or m in cols:
            continue
        order[m] = j
        rows.add(j)
        cols.add(m)
        if len(rows) == n:
            break
    return order
