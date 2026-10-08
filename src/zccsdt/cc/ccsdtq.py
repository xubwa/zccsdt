'''CCSDTQ on spin orbitals or spinors, real or complex: the solver.

T1 and T2 are full arrays, T3 and T4 are kept on their unique blocks
(i<j<k(<l), a<b<c(<d)).  One iteration is

- the Jacobi step of CCSDT (zccsdt.cc.ccsdt_opt.update_amps: the T1, T2 and
  T3 equations without T4, on the packed T3),
- the T4 equation and the terms of T4 in the equations of T2 and T3
  (zccsdt.cc.t4engine, on T1-dressed integrals in blocks),
- DIIS on all four amplitudes (zccsdt.cc.diis: the vectors of T4 in a
  scratch file unless they are small).

The integrals come from a container (zccsdt.eris.ERIs), with <vv||vv> packed
or full.  The steps divide by the differences of the orbital energies
(the diagonal of the Fock matrix); canonical orbitals are not assumed.
'''

import time
from dataclasses import dataclass, field
from math import comb

import numpy as np

from zccsdt import memory
from zccsdt.cc import ccsd, ccsdt_opt, solver, t4engine, t4pack as P
from zccsdt.cc.diis import DIIS, INCORE_SIZE
from zccsdt.scratch import Scratch
from zccsdt.timing import timer


@dataclass
class Options:
    conv_tol: float = 1e-10
    conv_tol_normt: float = 1e-8
    max_cycle: int = 200
    diis_space: int = 8
    diis_file: str = None       # HDF5 file of the DIIS vectors (default: temporary)
    incore_complete: bool = False   # keep all DIIS vectors in memory
    max_memory: float = None    # MB for the whole process, as in PySCF (default:
                                # 80% of what the node and the job limit leave)
    max_disk: float = None      # MB for the scratch files of the process (default:
                                # ZCCSDT_MAX_DISK, else the free space of the scratch
                                # directory less a reserve)
    diis_t4: bool = None        # T4 in the DIIS vectors: None - yes, unless neither disk
                                # nor memory holds three of them; True - yes, or an
                                # error; False - T1, T2 and T3 only, T4 by its Jacobi step
    plan: object = None         # a Plan given explicitly
    diis_plan: object = None    # a zccsdt.disk.DiisPlan given explicitly
    mem_mb: float = 2000.0      # batch buffers of the CCSDT step
    verbose: bool = False
    callback: object = None     # callback(info) after every iteration


STEP_MB = 100.0                 # the batch buffers of the CCSDT step when memory is short


@dataclass
class Plan:
    '''Where the memory of a CCSDTQ calculation goes (plan()).'''
    batch_bytes: int            # an operand or a product of one batch of the T4 engine
    diis_file: bool             # the T4 vectors of DIIS in the scratch file
    need: int                   # bytes of ordinary memory besides the integrals
    floor: int                  # the least that a calculation takes
    t4: int                     # bytes of T4
    mem_mb: float = None        # batch buffers of the CCSDT step (None: as the options say)
    r4_file: bool = False       # the residual of T4 in a scratch file
    xv_file: bool = False       # the six-index intermediate of the engine in the scratch file
    t4_file: bool = False       # T4 in the scratch file
    disk: int = 0               # bytes of the arrays that the plan maps to the scratch file
    # bytes of the memory budget beyond ``need`` (None: no budget; not compared)
    spare: int = field(default=None, compare=False)


def _dressed_bytes(no, nv, isz):
    '''The dressed integrals of the T4 engine and what their dressing takes
    on the way: <vv||vv> and <vv||vo> on pairs, <vo||vv>.'''
    nQ = comb(nv, 2)
    return isz * (nQ * nQ + nv * nv * nQ + 2 * nQ * nv * no + 4 * no * nv ** 3 + 4 * (no * nv) ** 2)


def plan(nocc, nvir, isz, budget=None, diis_space=8, incore_complete=False, step_bytes=0, step_need=None,
         mem_mb=2000.0):
    '''The Plan for ``budget`` bytes of ordinary memory (None: no limit).

    With enough memory T4, its residual and the six-index intermediate Xv
    of the engine are held in memory.  With less they are arrays of a
    scratch file, which the operating system pages in and out - T4 first
    (the engine only reads it), then Xv, then the residual, which every
    product adds to: slow, and what there is for when nothing else fits.
    The floor (MemoryError below it) is the batches of the engine at their
    smallest, what it makes of T3, and the CCSDT step with its smallest
    buffers.  Above the three arrays the batches of the engine grow, up to
    their default; then the buffers of the CCSDT step, up to ``mem_mb``;
    with what is left then the DIIS vectors of T4 stay in memory
    (``incore_complete``: everything in memory in any case) - one after the
    other, so that no part of a plan shrinks when the budget grows.  What
    the CCSDT step takes: ``step_need(mem_mb)`` bytes (its own plan for
    these buffers), or ``step_bytes`` whatever the buffers.'''
    nI, nA = comb(nocc, 4), comb(nvir, 4)
    t4 = isz * nI * nA
    t3 = isz * comb(nocc, 3) * comb(nvir, 3)
    xv = isz * comb(nocc, 2) * nvir * comb(nvir, 3)
    if step_need is None:
        def step_need(mb):
            return step_bytes
    least_mb = min(mem_mb, STEP_MB)
    step_lo = step_need(least_mb)
    rest_fixed = t3 + _dressed_bytes(nocc, nvir, isz)
    lo, top = t4engine.min_batch_bytes(nocc, nvir, isz), t4engine.BATCH_BYTES
    top = max(lo, top)
    diis = 2 * diis_space * (t4 + t3)

    def engine(batch, r4=True, xv_in=True):
        return t4engine.memory(nocc, nvir, isz, batch, r4, xv_in)
    # the least with the three arrays in the file, then with the residual, Xv and T4 in memory
    f0 = step_lo + rest_fixed + engine(lo, False, False)
    f_r4, f_xv, f_t4 = f0 + t4, f0 + t4 + xv, f0 + 2 * t4 + xv
    floor = f_t4 + diis if incore_complete else f0
    r4_file = xv_file = t4_file = False
    if budget is None:
        batch, mb = top, mem_mb
        on_file = not incore_complete and nI * nA >= INCORE_SIZE
    else:
        if budget < floor:
            raise MemoryError(
                'CCSDTQ needs %.0f MB besides the integrals for %d occupied and %d virtual spin orbitals '
                '%s(T4 and its residual: %.0f MB each); %.0f MB are left'
                % (floor / 1e6, nocc, nvir, '' if incore_complete else
                   'with T4, its residual and the six-index intermediate in scratch files ', t4 / 1e6, budget / 1e6))
        r4_file, xv_file, t4_file = budget < f_r4, budget < f_xv, budget < f_t4
        batch, mb, on_file = lo, least_mb, True
        if not t4_file:
            rest = budget - step_lo - rest_fixed - t4 - (diis if incore_complete else 0)
            # the largest batch that fits: memory() grows with it
            if engine(top) <= rest:
                batch = top
            else:
                hi = top
                while hi - batch > max(1 << 16, batch // 64):
                    mid = (batch + hi) // 2
                    if engine(mid) <= rest:
                        batch = mid
                    else:
                        hi = mid
            if batch == top and mem_mb > least_mb:
                room = rest - engine(top) + step_lo
                if step_need(mem_mb) <= room:
                    mb = mem_mb
                else:
                    a, b = least_mb, mem_mb
                    for _ in range(24):
                        mid = 0.5 * (a + b)
                        a, b = (mid, b) if step_need(mid) <= room else (a, mid)
                    mb = a
            on_file = not incore_complete and not (
                batch == top and mb == mem_mb and step_need(mb) + rest_fixed + t4 + engine(batch) + diis <= budget)
    need = (step_need(mb) + rest_fixed + (0 if t4_file else t4) + engine(batch, not r4_file, not xv_file)
            + (0 if on_file else diis))
    mapped = (t4 if t4_file else 0) + (t4 if r4_file else 0) + (xv if xv_file else 0)
    return Plan(int(batch), bool(on_file), int(need), int(floor), int(t4), float(mb), bool(r4_file), bool(xv_file),
                bool(t4_file), int(mapped), None if budget is None else max(0, int(budget - need)))


def diis_plan_for(pl, sizes, isz, opts, mapped=0, tight=False):
    '''Where the DIIS vectors of a CCSDTQ run with the Plan ``pl`` go
    (zccsdt.disk.DiisPlan; the rule: zccsdt.disk, with T4 as the large part
    that can be left out).  ``sizes``: the elements of the parts of a vector
    (t1, packed t2, T3, T4), of which those from the size of T4 on go to the
    file.  A plan that has the vectors in memory (they are in its ``need``)
    stays as it is; else they get ``opts.max_disk`` less ``mapped`` - the
    arrays that the plan maps - and the memory that the plan leaves (none if
    it had to map: ``tight``).'''
    from zccsdt import disk
    big = disk.file_bytes(sizes, isz, min(INCORE_SIZE, max(1, sizes[-1])))
    with_t4 = getattr(opts, 'diis_t4', None)
    if not pl.diis_file and with_t4 is not False:
        if mapped:
            disk.require(mapped, 'T4, its residual and the six-index intermediate', opts.max_disk)
        return disk.DiisPlan(opts.diis_space, opts.diis_space, n_memory=opts.diis_space,
                             pair=2 * isz * (sizes[-1] + sizes[-2]), what='T4', case='m')
    return disk.diis_plan(big, opts.diis_space, opts.max_disk, opts.diis_file, mapped, getattr(pl, 'spare', None),
                          with_t4, False, 'T4', memory_tight=tight)


def plan_for(eris, dtype, max_memory=None, diis_space=8, incore_complete=False, mem_mb=2000.0):
    '''The Plan of a calculation on the integral container ``eris``:
    ``max_memory`` MB for the whole process (None: what the node and the
    job leave).'''
    nocc = eris.nocc
    nvir = np.asarray(eris.fock).shape[0] - nocc

    def step_need(mb):
        return ccsdt_opt.plan_for(eris, dtype, solver.Options(mem_mb=mb, incore_complete=True)).need
    return plan(nocc, nvir, np.dtype(dtype).itemsize, memory.budget(max_memory), diis_space, incore_complete,
                step_need=step_need, mem_mb=mem_mb)


@dataclass
class Result:
    e_corr: float
    e_tot: float
    converged: bool
    ncycle: int
    t1: np.ndarray
    t2: np.ndarray              # full
    t3: np.ndarray              # packed, flat
    t4: np.ndarray              # packed [C(no,4), C(nv,4)]
    timings: list = None        # wall time per iteration
    plan: object = None         # the Plan of the run
    diis_plan: object = None    # where its DIIS vectors were (zccsdt.disk.DiisPlan)


def _jacobi4(T4, R4, eo4, ev4):
    '''R4 <- T4 + R4 / D4 row by row (no array of the denominators).'''
    for I in range(R4.shape[0]):
        r = R4[I]
        r /= eo4[I] - ev4
        r += T4[I]
    return R4


def kernel(eris, opts=None, t1=None, t2=None, t3=None, t4=None):
    '''CCSDTQ by Jacobi steps with DIIS, from the given amplitudes (those of
    CCSDT, say: t1[i,a], t2[i,j,a,b], t3 packed) or from those of MP2.

    ``opts.max_disk`` (MB) bounds the scratch files of the run: the arrays
    that the plan maps (DiskFullError before anything is allocated if they do
    not fit) and the DIIS vectors (diis_plan_for; zccsdt.disk):
    ``opts.diis_space`` is a maximum, and where not even three vectors with
    T4 fit DIIS is done on T1, T2 and T3 and T4 takes its Jacobi step
    (``opts.diis_t4``).  The plan used is Result.diis_plan.'''
    from zccsdt import disk
    opts = opts or Options()
    with disk.limit(opts.max_disk):     # (every scratch file of the run counts against max_disk)
        return _kernel(eris, opts, t1, t2, t3, t4)


def _kernel(eris, opts, t1, t2, t3, t4):
    from zccsdt import disk
    nocc = eris.nocc
    nvir = np.asarray(eris.fock).shape[0] - nocc
    if t1 is None or t2 is None:
        t1, t2 = solver.init_amps(eris)
    dtype = np.result_type(t1, t2, eris.oovv, eris.fock)
    nI3, nA3, nI, nA = comb(nocc, 3), comb(nvir, 3), comb(nocc, 4), comb(nvir, 4)
    t1, t2 = t1.astype(dtype), t2.astype(dtype)
    T3 = np.zeros(nI3 * nA3, dtype=dtype) if t3 is None else np.array(t3, dtype=dtype).ravel()
    mo_e = np.asarray(eris.mo_energy)
    eo, ev = mo_e[:nocc], mo_e[nocc:]
    d2 = eo[P.tuples(nocc, 2)].sum(axis=1)[:, None] - ev[P.tuples(nvir, 2)].sum(axis=1)[None, :]
    d3 = (eo[P.tuples(nocc, 3)].sum(axis=1)[:, None] - P.tuple_sums(ev, 3)[None, :]).ravel() if nI3 else 0
    eo4, ev4 = eo[P.tuples(nocc, 4)].sum(axis=1), P.tuple_sums(ev, 4)
    pl = opts.plan or plan_for(eris, dtype, opts.max_memory, opts.diis_space, opts.incore_complete, opts.mem_mb)
    mem_mb = opts.mem_mb if pl.mem_mb is None else pl.mem_mb
    # (before anything is allocated: the disk holds the plan or the run stops here)
    dplan = opts.diis_plan or diis_plan_for(
        pl, [nocc * nvir, comb(nocc, 2) * comb(nvir, 2), nI3 * nA3, nI * nA], np.dtype(dtype).itemsize, opts,
        pl.disk, pl.t4_file or pl.r4_file or pl.xv_file)
    # T4, and with too little memory T4, the intermediate Xv and the residual as arrays of a
    # scratch file (new file space reads as zeros)
    scr = r4_buf = xv_buf = None
    if pl.t4_file or pl.r4_file or pl.xv_file:
        scr = Scratch(prefix='zccsdt_t4_', what='T4, its residual and the six-index intermediate',
                      hint=' or give the calculation more memory (max_memory)')
        if pl.r4_file:
            r4_buf = scr.create('r4', (nI, nA), dtype)
        if pl.xv_file:
            xv_buf = scr.create('xv', (comb(nocc, 2), nvir, comb(nvir, 3)), dtype)
    if pl.t4_file:
        T4 = scr.create('t4', (nI, nA), dtype)
        if t4 is not None:
            src = np.asarray(t4).reshape(nI, nA)
            for I in range(nI):
                T4[I] = src[I]
    else:
        T4 = np.zeros((nI, nA), dtype=dtype) if t4 is None else np.array(t4, dtype=dtype).reshape(nI, nA)
    step = solver.Options(mem_mb=mem_mb)                        # (of the CCSDT step: no level shift)
    step_plan = ccsdt_opt.plan_for(eris, dtype, solver.Options(mem_mb=mem_mb, incore_complete=True))
    adiis = DIIS(space=dplan.space, filename=opts.diis_file, incore=dplan.n_file == 0,
                 incore_size=min(INCORE_SIZE, max(1, nI * nA)), in_memory=dplan.n_memory)
    e_old = ccsd.energy(t1, t2, eris)
    conv, timings = False, []
    try:
        for cycle in range(opts.max_cycle):
            t0, c0 = time.perf_counter(), time.process_time()
            with timer('ccsdtq.t3'):
                t1n, t2n, t3n = ccsdt_opt.update_amps(step, t1, t2, T3, eris, mem_mb=mem_mb, plan=step_plan)
            with timer('ccsdtq.dress'):
                ints = t4engine.Dressed.from_eris(eris, t1)
            with timer('ccsdtq.t4'):
                R4, r3, r2 = t4engine.residual(t2, T3, T4, ints, pl.batch_bytes, out=r4_buf, xv=xv_buf)
            del ints
            with timer('ccsdtq.update'):
                t2n = t2n + P.unpack(r2 / d2, nocc, nvir, 2)
                if nI3:
                    t3n += r3.ravel() / d3
                _jacobi4(T4, R4, eo4, ev4)
            del r3, r2
            with timer('ccsdtq.diis'):
                first = adiis.xprev is None
                if first:
                    norms = [np.linalg.norm(t1n - t1), np.linalg.norm(t2n - t2),
                             6.0 * solver._diff_norm(t3n, T3), 24.0 * solver._diff_norm(R4.ravel(), T4.ravel())]
                if dplan.with_big:
                    parts = adiis.update(solver._parts(t1n, t2n, t3n) + [R4.ravel()],
                                         out=[None, None, T3, T4])
                else:
                    # DIIS without T4: the Jacobi step is the new T4, its old array the next residual
                    n4 = 24.0 * solver._diff_norm(R4.ravel(), T4.ravel())
                    parts = adiis.update(solver._parts(t1n, t2n, t3n), out=[None, None, T3])
                    if r4_buf is not None:
                        r4_buf = T4
                    T4 = R4.reshape(nI, nA)
                if not first:
                    en = adiis.err_norms
                    norms = [en[0], 2.0 * en[1], 6.0 * en[2], 24.0 * en[3] if dplan.with_big else n4]
                norms = [float(x) for x in norms]
                normt = sum(norms)
                t1, t2, _ = solver._from_parts(parts[:3], nocc, nvir, True)
            del R4, t1n, t2n, t3n, parts
            e_new = ccsd.energy(t1, t2, eris)
            timings.append(time.perf_counter() - t0)
            de = e_new - e_old
            e_old = e_new
            if opts.verbose:
                print('cycle %3d  E_corr = %.15f  dE = %.3e  |dt| = %.3e  %.2fs'
                      % (cycle + 1, e_new.real, de.real, normt, timings[-1]), flush=True)
            if opts.callback is not None:
                opts.callback(dict(cycle=cycle + 1, e_corr=float(np.real(e_new)), de=float(np.real(de)),
                                   norms=tuple(norms), normt=normt, diis=len(adiis.xs),
                                   wall=timings[-1], cpu=time.process_time() - c0))
            if abs(de) < opts.conv_tol and normt < opts.conv_tol_normt:
                conv = True
                break
    finally:
        adiis.close()
        if scr is not None:
            scr.close()                 # (a mapped T4 stays valid: the file is unlinked, not gone)
    return Result(e_corr=float(np.real(e_old)), e_tot=eris.e_hf + float(np.real(e_old)), converged=conv,
                  ncycle=cycle + 1, t1=t1, t2=t2, t3=T3, t4=T4, timings=timings, plan=pl, diis_plan=dplan)
