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
from dataclasses import dataclass
from math import comb

import numpy as np

from zccsdt import memory
from zccsdt.cc import ccsd, ccsdt_opt, solver, t4engine, t4pack as P
from zccsdt.cc.diis import DIIS, INCORE_SIZE
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
    plan: object = None         # a Plan given explicitly
    mem_mb: float = 2000.0      # batch buffers of the CCSDT step
    verbose: bool = False
    callback: object = None     # callback(info) after every iteration


@dataclass
class Plan:
    '''Where the memory of a CCSDTQ calculation goes (plan()).'''
    batch_bytes: int            # an operand or a product of one batch of the T4 engine
    diis_file: bool             # the T4 vectors of DIIS in the scratch file
    need: int                   # bytes of ordinary memory besides the integrals
    floor: int                  # the least that a calculation takes
    t4: int                     # bytes of T4


def _dressed_bytes(no, nv, isz):
    '''The dressed integrals of the T4 engine and what their dressing takes
    on the way: <vv||vv> and <vv||vo> on pairs, <vo||vv>.'''
    nQ = comb(nv, 2)
    return isz * (nQ * nQ + nv * nv * nQ + 2 * nQ * nv * no + 4 * no * nv ** 3 + 4 * (no * nv) ** 2)


def plan(nocc, nvir, isz, budget=None, diis_space=8, incore_complete=False, step_bytes=0):
    '''The Plan for ``budget`` bytes of ordinary memory (None: no limit).
    T4, its residual and the six-index intermediate Xv of the engine are
    held in memory: that, with the smallest batches and the DIIS vectors in
    the scratch file, is the floor (MemoryError below it).  From there the
    batches of the engine grow, up to its default; with what is left then
    the DIIS vectors of T4 stay in memory (``incore_complete``: in any
    case).  ``step_bytes``: what the CCSDT step takes (its own plan).'''
    t4 = isz * comb(nocc, 4) * comb(nvir, 4)
    t3 = isz * comb(nocc, 3) * comb(nvir, 3)
    fixed = step_bytes + t4 + t3 + _dressed_bytes(nocc, nvir, isz)
    lo, top = t4engine.min_batch_bytes(nocc, nvir, isz), t4engine.BATCH_BYTES
    top = max(lo, top)
    diis = 2 * diis_space * (t4 + t3)
    floor = fixed + t4engine.memory(nocc, nvir, isz, lo) + (diis if incore_complete else 0)
    if budget is None:
        batch = top
        on_file = not incore_complete and comb(nocc, 4) * comb(nvir, 4) >= INCORE_SIZE
    else:
        if budget < floor:
            raise MemoryError(
                'CCSDTQ needs %.0f MB besides the integrals for %d occupied and %d virtual spin orbitals '
                '(T4 and its residual: %.0f MB each); %.0f MB are left'
                % (floor / 1e6, nocc, nvir, t4 / 1e6, budget / 1e6))
        rest = budget - fixed - (diis if incore_complete else 0)
        # the largest batch that fits: memory() grows with it
        if t4engine.memory(nocc, nvir, isz, top) <= rest:
            batch = top
        else:
            batch, hi = lo, top
            while hi - batch > max(1 << 16, batch // 64):
                mid = (batch + hi) // 2
                if t4engine.memory(nocc, nvir, isz, mid) <= rest:
                    batch = mid
                else:
                    hi = mid
        on_file = not incore_complete and not (
            batch == top and fixed + t4engine.memory(nocc, nvir, isz, batch) + diis <= budget)
    need = fixed + t4engine.memory(nocc, nvir, isz, batch) + (0 if on_file else diis)
    return Plan(int(batch), bool(on_file), int(need), int(floor), int(t4))


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


def _jacobi4(T4, R4, eo4, ev4):
    '''R4 <- T4 + R4 / D4 row by row (no array of the denominators).'''
    for I in range(R4.shape[0]):
        r = R4[I]
        r /= eo4[I] - ev4
        r += T4[I]
    return R4


def kernel(eris, opts=None, t1=None, t2=None, t3=None, t4=None):
    '''CCSDTQ by Jacobi steps with DIIS, from the given amplitudes (those of
    CCSDT, say: t1[i,a], t2[i,j,a,b], t3 packed) or from those of MP2.'''
    opts = opts or Options()
    nocc = eris.nocc
    nvir = np.asarray(eris.fock).shape[0] - nocc
    if t1 is None or t2 is None:
        t1, t2 = solver.init_amps(eris)
    dtype = np.result_type(t1, t2, eris.oovv, eris.fock)
    nI3, nA3, nI, nA = comb(nocc, 3), comb(nvir, 3), comb(nocc, 4), comb(nvir, 4)
    t1, t2 = t1.astype(dtype), t2.astype(dtype)
    T3 = np.zeros(nI3 * nA3, dtype=dtype) if t3 is None else np.array(t3, dtype=dtype).ravel()
    T4 = np.zeros((nI, nA), dtype=dtype) if t4 is None else np.array(t4, dtype=dtype).reshape(nI, nA)
    mo_e = np.asarray(eris.mo_energy)
    eo, ev = mo_e[:nocc], mo_e[nocc:]
    d2 = eo[P.tuples(nocc, 2)].sum(axis=1)[:, None] - ev[P.tuples(nvir, 2)].sum(axis=1)[None, :]
    d3 = (eo[P.tuples(nocc, 3)].sum(axis=1)[:, None] - P.tuple_sums(ev, 3)[None, :]).ravel() if nI3 else 0
    eo4, ev4 = eo[P.tuples(nocc, 4)].sum(axis=1), P.tuple_sums(ev, 4)
    step = solver.Options(mem_mb=opts.mem_mb)                   # (of the CCSDT step: no level shift)
    step_plan = ccsdt_opt.plan_for(eris, dtype, solver.Options(mem_mb=opts.mem_mb, incore_complete=True))
    pl = opts.plan or plan(nocc, nvir, np.dtype(dtype).itemsize, memory.budget(opts.max_memory),
                           opts.diis_space, opts.incore_complete, step_plan.need)
    adiis = DIIS(space=opts.diis_space, filename=opts.diis_file, incore=not pl.diis_file,
                 incore_size=min(INCORE_SIZE, max(1, nI * nA)))
    e_old = ccsd.energy(t1, t2, eris)
    conv, timings = False, []
    try:
        for cycle in range(opts.max_cycle):
            t0, c0 = time.perf_counter(), time.process_time()
            with timer('ccsdtq.t3'):
                t1n, t2n, t3n = ccsdt_opt.update_amps(step, t1, t2, T3, eris, mem_mb=opts.mem_mb,
                                                      plan=step_plan)
            with timer('ccsdtq.dress'):
                ints = t4engine.Dressed.from_eris(eris, t1)
            with timer('ccsdtq.t4'):
                R4, r3, r2 = t4engine.residual(t2, T3, T4, ints, pl.batch_bytes)
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
                parts = adiis.update(solver._parts(t1n, t2n, t3n) + [R4.ravel()],
                                     out=[None, None, T3, T4])
                if not first:
                    en = adiis.err_norms
                    norms = [en[0], 2.0 * en[1], 6.0 * en[2], 24.0 * en[3]]
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
    return Result(e_corr=float(np.real(e_old)), e_tot=eris.e_hf + float(np.real(e_old)), converged=conv,
                  ncycle=cycle + 1, t1=t1, t2=t2, t3=T3, t4=T4, timings=timings, plan=pl)
