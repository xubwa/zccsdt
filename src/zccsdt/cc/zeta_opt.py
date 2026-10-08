'''The two linear equations of CCSDT(Q)_Lambda on packed amplitudes
(docs/lambda.md, section 12; the dense reference is zccsdt.cc.qlgrad_ref):

    sum_mu (dR_nu/dt_mu) zeta_mu + q_nu = 0                               (1)
    W_k(t, lambda') + de/dt_k + sum_mu zeta_mu dW_mu/dt_k (t, lambda) = 0   (2)

(1): the Jacobian of the CCSDT amplitude equations on zeta, with the source
q = de/dlambda of zccsdt.cc.quadruples_lambda_grad - kernel().  (2): the
Lambda equations of CCSDT with constant sources, zccsdt.cc.lambda_opt.kernel
with what lambda_sources() makes.

No linearized equations are coded.  The amplitude residuals R and the Lambda
residuals W are polynomials of degree <= 4 in the amplitudes, so a
derivative along zeta is a combination of the production codes at shifted
amplitudes (points()):

    'exact'    [8 (f(t + zeta) - f(t - zeta)) - (f(t + 2 zeta) - f(t - 2 zeta))] / 12
               four evaluations, the derivative to the rounding
    'central'  [f(t + h zeta) - f(t - h zeta)] / 2h,   h = DELTA / max |zeta|
               two evaluations; what is left out is h^2 times the cubic term
               of f along zeta, DELTA^2 = 1e-10 of J zeta times a number of
               order one, and the rounding of f is divided by h

The Jacobian.  ccsdt_opt.update_amps gives the Jacobi step N(t) = t + R(t) / D
of the amplitude equations, as it is - T3 on the packed block or in the
blocks of the symmetry labels, in memory or in a scratch file.  With weights
that sum to zero and reproduce the first derivative,

    zeta + (J zeta + q) / D = sum_k w_k N(t + x_k zeta) + q / D:

a Jacobi step of (1) costs as many amplitude iterations as the stencil has
points, and DIIS runs on it as on the amplitudes.  kernel() iterates with
the central differences (two amplitude iterations per iteration).  When
their change is below the threshold, the same iterate is done once more
with the points +-2h, which with the two at +-h are the exact formula at
step h: if the exact change is below the threshold too - the difference
between the two is Result.stencil_error - the iterations end, else they go
on with the exact stencil (four amplitude iterations each).

Memory (plan_for): the amplitude steps have the MemoryPlan of the amplitude
equations - their "T3" is the shifted amplitudes, their residual the new
vector.  The equation has four more arrays of the size of T3 (zeta_3,
q_3 / D, the accumulator of a stencil and a copy for the last check) and its
DIIS vectors.  If the amplitude steps do not fit in memory next to them,
they are mapped arrays of a scratch file, which every use passes over once,
front to back.

Timers (zccsdt.timing): 'zeta.step' (an amplitude step, with the timers of
ccsdt_opt below it), 'zeta.vectors', 'zeta.diis', 'zeta.response'.
'''

import time
from dataclasses import dataclass, replace

import numpy as np

from zccsdt.cc import ccsdt_opt, diis, lambda_opt, solver, t3pack
from zccsdt.cc.diis import DIIS
from zccsdt.timing import timer

DELTA = 1e-5        # central differences: the largest change of an amplitude
EXACT = ((1.0, 8.0 / 12.0), (-1.0, -8.0 / 12.0), (2.0, -1.0 / 12.0), (-2.0, 1.0 / 12.0))
NVEC = 4            # arrays of the size of T3 besides those of an amplitude step
CHUNK = 1 << 22


def points(stencil, scale=None, delta=DELTA):
    '''((x, w), ...) with sum_k w_k f(t + x_k zeta) the derivative of f along
    zeta: 'exact' for a polynomial of degree <= 4, or 'central' with the
    step ``delta`` / ``scale`` (scale: the largest element of zeta).'''
    if stencil == 'exact':
        return EXACT
    if stencil != 'central':
        raise ValueError("stencil: 'exact' or 'central', not %r" % (stencil,))
    h = float(delta / scale) if scale else 1.0
    return ((h, 0.5 / h), (-h, -0.5 / h))


def largest(zeta):
    '''The largest absolute value among the elements of the arrays.'''
    out = 0.0
    for a in zeta:
        a = np.asarray(a).reshape(-1) if not isinstance(a, np.memmap) else a.reshape(-1)
        for i in range(0, a.size, CHUNK):
            out = max(out, float(abs(a[i:i + CHUNK]).max()))
    return out


def lin(out, terms):
    '''out = sum c * x over ``terms`` = [(c, x)] of flat arrays of one type
    (out may be one of them), on threads.'''
    return diis._lincomb([x for _, x in terms], [c for c, _ in terms], out)


def _blocks(eris, dtype):
    '''[(rows of the lexicographic block or None, its columns or None, the
    slice of a flat T3 of the engine of ``eris``, the block's shape, the
    slices of the triples in the order of the maps)]: the packed block
    itself, or the blocks of the symmetry labels.'''
    om, vm = t3pack.maps_for(eris)
    if om.spins is None:
        return [(None, None, slice(0, om.ntri * vm.ntri), (om.ntri, vm.ntri), slice(0, om.ntri), slice(0, vm.ntri))]
    L = ccsdt_opt.engine_for(eris, dtype).L
    lo, lv = ccsdt_opt._lex(om), ccsdt_opt._lex(vm)
    out = []
    for g in L.stored:
        I, A = slice(om.tri_off[g], om.tri_off[g + 1]), slice(vm.tri_off[g], vm.tri_off[g + 1])
        out.append((lo[I], lv[A], slice(L.boff[g], L.boff[g] + int(L.nI[g] * L.nA[g])), (int(L.nI[g]), int(L.nA[g])), I, A))
    return out


def add_packed(eris, out, x3, alpha=1.0, denominators=None):
    '''out += alpha * x3, or with ``denominators`` = (eo, ev) - the sums of
    the orbital energies of the occupied and of the virtual triples in the
    order of the maps - out += alpha * x3 / (eo - ev).  ``out``: flat, in
    the layout of T3 of the engine of ``eris`` (the packed block, or the
    blocks of the symmetry labels); ``x3``: on the packed block [C(no,3),
    C(nv,3)] in lexicographic order, in memory or mapped.  Rows in chunks:
    no copy of the size of x3.'''
    om, vm = t3pack.maps_for(eris)
    x3 = x3.reshape(om.ntri, vm.ntri)
    for rows, cols, where, shape, I, A in _blocks(eris, out.dtype):
        if not shape[0] or not shape[1]:
            continue
        blk = out[where].reshape(shape)
        step = max(1, CHUNK // shape[1])
        for r0 in range(0, shape[0], step):
            r1 = min(shape[0], r0 + step)
            x = x3[r0:r1] if rows is None else x3[rows[r0:r1]][:, cols]
            if denominators is not None:
                x = x / (denominators[0][I][r0:r1, None] - denominators[1][A][None, :])
            blk[r0:r1] += alpha * x


# ------------------------------------------------------------------- memory

@dataclass
class Plan:
    '''The memory of kernel() (plan_for).'''
    amps: object                # the ccsdt_opt.MemoryPlan of the amplitude steps
    mapped: bool = False        # the vectors of the equation are in a scratch file
    need: int = 0               # bytes of ordinary memory
    vectors: int = 0            # of which are the vectors of the equation and its DIIS vectors


def plan_for(eris, dtype, opts):
    '''The Plan of kernel() with the solver options ``opts`` (max_memory in
    MB for the whole process; default: 80% of what is available): the
    MemoryPlan of the amplitude equations for what the NVEC vectors of the
    equation and the DIIS vectors that are not in the scratch file leave
    (the two arrays of the size of T3 of that plan are the shifted
    amplitudes and the new vector).  If an amplitude step does not fit in
    memory next to them, the vectors go to a scratch file and the step has
    the whole budget; its MemoryError where that is not enough.'''
    eng = ccsdt_opt.engine_for(eris, dtype)
    t3 = eng.size * np.dtype(dtype).itemsize
    incore = opts.incore_complete or eng.size < diis.INCORE_SIZE
    extra = NVEC * t3 + (2 * opts.diis_space * t3 if incore else 0)
    if opts.incore_complete:
        p = ccsdt_opt.plan_for(eris, dtype, opts)
        return Plan(p, False, p.need + extra, extra)
    try:
        p = ccsdt_opt.plan_for(eris, dtype, opts, reserve=extra)
        tight = p.t3_file or p.pp_file or p.pp_bytes is not None
    except MemoryError:
        tight = True
    if not tight:
        return Plan(p, False, p.need + extra, extra)
    p = ccsdt_opt.plan_for(eris, dtype, opts)
    return Plan(p, True, p.need, 0)


# ------------------------------------------------------------------- solver

@dataclass
class Result:
    converged: bool
    ncycle: int
    z1: np.ndarray              # [i,a]
    z2: np.ndarray              # [i,j,a,b]
    z3: np.ndarray              # flat, in the layout of t3 (in memory, or mapped: Plan.mapped)
    normt: float = 0.0          # the last change of the full tensors
    steps: int = 0              # amplitude steps taken
    stencil_error: float = None  # the central step against the exact one at the last iterate (full tensors)
    timings: list = None        # wall time per iteration
    plan: object = None


def kernel(eris, t1, t2, t3, q2, q3, opts=None, stencil='central', delta=DELTA):
    '''Solve J zeta + q = 0 for converged CCSDT amplitudes: t1, t2 dense, t3
    as solver.kernel returns it (the packed block, or the blocks of the
    symmetry labels where the integrals have them in use); q2[i,j,a,b] and
    q3 on the packed block [C(no,3), C(nv,3)] in lexicographic order (in
    memory or mapped): de/dl2 and de/dl3 of
    zccsdt.cc.quadruples_lambda_grad.  Jacobi steps with DIIS from zeta =
    q / D, converged when the norm of the change of the full tensors is
    below ``opts.conv_tol_normt``.  ``opts``: solver.Options (level_shift,
    max_cycle, diis_space, diis_file, incore_complete, mem_mb, max_memory,
    verbose, callback; ``plan`` takes a Plan of this module).

    ``stencil``: 'central' - two amplitude steps per iteration, and at the
    end the check with the exact formula described in the module (two more
    steps; four per iteration from then on if it fails) -, or 'exact': four
    amplitude steps in every iteration.  ``delta``: the largest change of an
    amplitude in a central step.  Returns a Result; zeta_3 has the layout of
    t3.'''
    from zccsdt import scratch
    from zccsdt.scratch import Scratch
    opts = opts or solver.Options()
    points(stencil)
    nocc, nvir = np.shape(t1)
    dtype = np.dtype(np.result_type(t1, t2, t3, q2, q3, eris.oovv, eris.fock))
    plan = opts.plan if isinstance(opts.plan, Plan) else plan_for(eris, dtype, opts)
    aplan = plan.amps
    aopts = replace(opts, plan=aplan)
    eng = ccsdt_opt.engine_for(eris, dtype)
    size = eng.size
    om, vm = t3pack.maps_for(eris)
    mo_e = np.asarray(eris.mo_energy)
    eia = mo_e[:nocc, None] - (mo_e[None, nocc:] + opts.level_shift)
    eijab = eia[:, None, :, None] + eia[None, :, None, :]
    eo = mo_e[:nocc][om.triples].sum(axis=1)
    ev = (mo_e[nocc:] + opts.level_shift)[vm.triples].sum(axis=1)
    t1, t2 = np.asarray(t1, dtype=dtype), np.asarray(t2, dtype=dtype)
    T3 = np.ascontiguousarray(np.asarray(t3, dtype=dtype).ravel()) if not isinstance(t3, np.memmap) or t3.dtype != dtype \
        else t3.reshape(-1)
    tight = bool(plan.mapped or aplan.t3_file or aplan.pp_file or aplan.pp_bytes is not None)
    r3_file = aplan.t3_file if aplan.r3_file is None else aplan.r3_file
    hints = scratch.hints(tight)
    scr = [None]
    adiis = None
    nsteps = [0]
    timings, conv, normt, err, icycle = [], False, 0.0, None, -1

    def new(name, mapped):
        '''An array of the size of T3, zero.'''
        if mapped and size:
            if scr[0] is None:
                scr[0] = Scratch(prefix='zccsdt_zeta_', what='the vectors of the equation of zeta',
                                 hint=' or give the calculation more memory (max_memory)')
            return scr[0].create(name, size, dtype)
        return np.zeros(size, dtype=dtype)

    try:
        ccsdt_opt.place_vvvv(eris, aplan)
        with timer('zeta.vectors'):
            z3, c3, acc = (new(k, plan.mapped) for k in ('z3', 'c3', 'acc'))
            if size:
                add_packed(eris, c3, q3, denominators=(eo, ev))
                lin(z3, [(1.0, c3)])
            A = new('in', plan.mapped and aplan.t3_file)
        c2 = np.asarray(q2, dtype=dtype) / eijab
        z1, z2 = np.zeros((nocc, nvir), dtype=dtype), c2.copy()
        B = keep = None

        def evaluate(x, out):
            '''The Jacobi step of the amplitude equations at t + x zeta: (t1, t2, T3 in ``out``).'''
            with timer('zeta.vectors'):
                lin(A, [(1.0, T3), (x, z3)])
            nsteps[0] += 1
            with timer('zeta.step'):
                n1, n2, n3 = ccsdt_opt.update_amps(aopts, t1 + x * z1, t2 + x * z2, A, eris, mem_mb=opts.mem_mb,
                                                   out=out, plan=aplan)
            assert n3.size == 0 or np.shares_memory(n3, out)
            return n1, n2

        def combine(pts, first, out):
            '''sum_k w_k N(t + x_k zeta) + first: (rank 1, rank 2, rank 3 in
            ``out``), ``first`` = (a1, a2, [(c, x)] of rank 3).'''
            a1, a2, terms = first
            for k, (x, wgt) in enumerate(pts):
                n1, n2 = evaluate(x, out)
                a1, a2 = a1 + wgt * n1, a2 + wgt * n2
                with timer('zeta.vectors'):
                    if k == len(pts) - 1:
                        lin(out, ([(1.0, acc)] if k else terms) + [(wgt, out)])
                    else:
                        lin(acc, ([(1.0, acc)] if k else terms) + [(wgt, out)])
            return a1, a2

        def change(a1, a2, a3):
            return [float(np.linalg.norm(a1 - z1)), float(np.linalg.norm(a2 - z2)), 6.0 * float(solver._diff_norm(a3, z3))]

        adiis = DIIS(space=opts.diis_space, filename=opts.diis_file, incore=opts.incore_complete,
                     incore_size=max(size, 1) if plan.mapped else None)
        mode = stencil
        for icycle in range(opts.max_cycle):
            t0, c0 = time.perf_counter(), time.process_time()
            scale = max(largest((z1, z2)), largest((z3,)))
            if scale == 0.0:            # (no source: zeta = 0)
                conv = True
                break
            if B is None:
                B = new('out', plan.mapped and r3_file)
            pts = points(mode, scale, delta)
            z1n, z2n = combine(pts, (0.0, c2, [(1.0, c3)]), B)
            norms = change(z1n, z2n, B)
            normt = sum(norms)
            if normt < opts.conv_tol_normt and mode == 'central':
                # the same iterate with the exact formula at step h: the points +-2h more
                h = pts[0][0]
                with timer('zeta.vectors'):
                    if keep is None:
                        keep = new('keep', plan.mapped)
                    lin(keep, [(1.0, B)])
                    first = ((4.0 * z1n) / 3.0, (4.0 * z2n - c2) / 3.0, [(4.0 / 3.0, keep), (-1.0 / 3.0, c3)])
                k1, k2 = z1n, z2n
                z1n, z2n = combine(((2.0 * h, -1.0 / (12.0 * h)), (-2.0 * h, 1.0 / (12.0 * h))), first, B)
                err = float(np.linalg.norm(z1n - k1) + np.linalg.norm(z2n - k2) + 6.0 * solver._diff_norm(B, keep))
                norms = change(z1n, z2n, B)
                normt = sum(norms)
                if normt >= opts.conv_tol_normt:
                    mode = 'exact'
            if normt < opts.conv_tol_normt:
                conv = True
                z1, z2 = z1n, z2n
                with timer('zeta.vectors'):
                    lin(z3, [(1.0, B)])
            else:
                with timer('zeta.diis'):
                    parts = adiis.update(solver._parts(z1n, z2n, B), out=[None, None, z3])
                    z1, z2, _ = solver._from_parts(parts, nocc, nvir, True)
                    # the array of the new vector is free again unless DIIS keeps it
                    if any(np.shares_memory(B, x[2]) for x in adiis.xs):
                        B = None
            timings.append(time.perf_counter() - t0)
            if opts.verbose:
                print('zeta cycle %3d  |dz| = %.3e  %s  %.2fs' % (icycle + 1, normt, mode, timings[-1]))
            if opts.callback is not None:
                opts.callback(dict(cycle=icycle + 1, norms=tuple(norms), normt=normt, diis=len(adiis.xs),
                                   wall=timings[-1], cpu=time.process_time() - c0, steps=nsteps[0], stencil=mode))
            if conv:
                break
    finally:
        scratch.hints(hints)
        if adiis is not None:
            adiis.close()
        if scr[0] is not None:
            scr[0].close()              # (the mapped zeta_3 stays valid: the file is unlinked)
    return Result(converged=conv, ncycle=icycle + 1, z1=z1, z2=z2, z3=z3, normt=normt, steps=nsteps[0],
                  stencil_error=err, timings=timings, plan=plan)


# ------------------------------------------------------- the second equation

def lambda_sources(eris, t1, t2, t3, l1, l2, l3, zeta, s2, s3, opts=None, blocks=False, stencil='exact',
                   delta=DELTA):
    '''The constant terms of the Lambda equations for lambda' of
    CCSDT(Q)_Lambda,

        s_k = de/dt_k + d/dx W_k(t + x zeta, lambda) at x = 0,

    W the residuals of the Lambda equations of CCSDT (lambda_opt.residuals)
    for lambda = (l1, l2, l3) of CCSDT: their intermediates are made and
    the residuals evaluated at the amplitudes of the stencil (four times for
    'exact', twice for 'central').  ``zeta`` = (z1, z2, z3) of kernel();
    t3, l3 and z3 in one layout, that of lambda_opt.kernel(blocks=) for
    these integrals; ``s2`` [i,j,a,b] and ``s3`` on the packed block in
    lexicographic order: de/dt of zccsdt.cc.quadruples_lambda_grad (s1 = 0
    there; the result has a rank 1).  Returns a lambda_opt.ResidualSources
    for lambda_opt.kernel(sources=).

    Memory: that of the Lambda equations (lambda_opt.plan_memory: T3, l3,
    the residual and the intermediates), with the shifted T3 in the place
    of T3 and one more array of that size, the source of rank 3.'''
    opts = opts or solver.Options()
    z1, z2, z3 = zeta
    if blocks:
        lay = lambda x: np.asarray(x).ravel()                                   # noqa: E731
    else:
        lay = lambda x: ccsdt_opt.packed_t3(eris, x).ravel()                    # noqa: E731
    T3p, L3p, Z3p = lay(t3), lay(l3), lay(z3)
    dtype = np.dtype(np.result_type(t1, t2, T3p, l1, l2, L3p, eris.oovv, eris.fock))
    T3p, L3p, Z3p = (np.ascontiguousarray(x, dtype=dtype) for x in (T3p, L3p, Z3p))
    size = T3p.size
    plan = lambda_opt._plan_for(eris, t1, dtype, opts, blocks, size * dtype.itemsize)
    t1, t2, z1, z2 = (np.asarray(x, dtype=dtype) for x in (t1, t2, z1, z2))
    S3, Tk, buf = np.zeros(size, dtype=dtype), np.empty(size, dtype=dtype), np.empty(size, dtype=dtype)
    a1 = a2 = 0.0
    pts = points(stencil, max(largest((z1, z2)), largest((Z3p,))), delta)
    with timer('zeta.response'):
        for x, wgt in pts:
            lin(Tk, [(1.0, T3p), (x, Z3p)])
            im = lambda_opt.make_intermediates(eris, t1 + x * z1, t2 + x * z2, Tk, mem_mb=plan.mem_mb, blocks=blocks)
            r1, r2, R = lambda_opt.residuals(im, l1, l2, L3p, Tk, out=buf if size else None)
            a1, a2 = a1 + wgt * r1, a2 + wgt * r2
            if size:
                lin(S3, [(1.0, S3), (wgt, R.reshape(-1))])
            om, vm = im.om, im.vm
            del im, r1, r2, R
        del Tk, buf
        mm, nn = om.pairs[:, 0], om.pairs[:, 1]
        md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
        a2 = a2 + np.asarray(s2)[mm[:, None], nn[:, None], md[None, :], ne[None, :]]
        if size:
            if blocks:
                add_packed(eris, S3, s3)
            else:
                S3 = S3.reshape(om.ntri, vm.ntri)
                s3 = s3.reshape(om.ntri, vm.ntri)
                step = max(1, CHUNK // max(vm.ntri, 1))
                for r0 in range(0, om.ntri, step):
                    S3[r0:r0 + step] += s3[r0:r0 + step]
                S3 = S3.reshape(-1)
    return lambda_opt.ResidualSources((np.ascontiguousarray(a1, dtype=dtype), np.ascontiguousarray(a2, dtype=dtype), S3))
