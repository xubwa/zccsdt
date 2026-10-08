'''CCSDTQ in blocks of a symmetry label: the solver of zccsdt.cc.ccsdtq for
spinors with labels that add without a modulus (2 m_j of a linear molecule).

One iteration is the same three steps, each in the blocks of the labels:

- the Jacobi step of CCSDT with T3 in blocks (zccsdt.cc.ccsdt_opt.update_amps
  with the block engine of zccsdt.cc.t3engine, which the labels of the
  integral container select),
- the T4 equation and the terms of T4 in the equations of T2 and T3
  (zccsdt.cc.t4engine_sym): T4 and its residual one dense block per total
  label,
- DIIS on all four amplitudes, T3 and T4 as the flat arrays of their blocks.

Result.t3 and Result.t4 are these flat arrays; packed_t3() and packed_t4()
put them on the packed arrays [C(no,k), C(nv,k)] of the dense code (zeros
outside the blocks), t4_blocks() is the inverse for T4.

Memory: plan() counts everything that an iteration holds - T4, its residual,
the six-index intermediate and the other blocks of the engine, its batches,
the CCSDT step and the DIIS vectors - and keeps the DIIS vectors of T4 in a
scratch file unless they fit as well.  Below the floor (the batches at
their smallest: one row of a block, one occupied triple of the ring) it
raises MemoryError with the numbers.  T4, the residual and the intermediate
are always in memory here: the scratch files that the dense solver falls
back on for them are not there, and a caller without the memory for the
blocks has the dense solver with its files left.

Labels with a modulus (the fermion irreps of C2 and its like) are not done:
available() is False for them and the dense solver is to be used.
'''

import time
from dataclasses import dataclass, field
from math import comb

import numpy as np

from zccsdt import memory
from zccsdt.cc import ccsd, ccsdt_opt, ccsdtq, solver, t4engine, t4engine_sym as S, t4pack as P
from zccsdt.cc.ccsdtq import Options, Result, STEP_MB
from zccsdt.cc.diis import DIIS, INCORE_SIZE
from zccsdt.timing import timer

__all__ = ['Options', 'Plan', 'available', 'kernel', 'packed_t3', 'packed_t4', 'plan', 'plan_for', 't4_blocks']


@dataclass
class Plan:
    '''Where the memory of a CCSDTQ calculation in blocks goes (plan()).'''
    batch_bytes: int            # an operand or a product of one batch of the T4 engine
    diis_file: bool             # the T4 vectors of DIIS in the scratch file
    need: int                   # bytes of ordinary memory besides the integrals
    floor: int                  # the least that a calculation takes
    t4: int                     # bytes of T4 in its blocks
    mem_mb: float = None        # batch buffers of the CCSDT step
    layout: object = None       # the blocks (zccsdt.cc.t4engine_sym.Layout)
    disk: int = 0               # bytes that the plan maps to scratch files (nothing)
    # bytes of the memory budget beyond ``need`` (None: no budget; not compared)
    spare: int = field(default=None, compare=False)


def available(eris):
    '''Whether the T4 equation can be done in the blocks of the labels of
    ``eris`` (zccsdt.cc.t4engine_sym.classes).'''
    return S.available(eris)


def packed_t4(eris, t4):
    '''T4 on the packed array [C(no,4), C(nv,4)] (lexicographic order of the
    quadruples), from the flat array of its blocks: zeros outside them.  An
    array that has the packed size already is returned as it is.'''
    no = eris.nocc
    nv = np.asarray(eris.fock).shape[0] - no
    shape = (comb(no, 4), comb(nv, 4))
    lay = S.layout_for(eris)
    t4 = np.asarray(t4)
    if lay is None or (t4.size == shape[0] * shape[1] and t4.size != lay.size):
        return t4.reshape(shape)
    return lay.to_packed(t4.ravel())


def t4_blocks(eris, t4):
    '''The flat array of the blocks of a packed T4 [C(no,4), C(nv,4)].'''
    return S.layout_for(eris).from_packed(t4)


def packed_t3(eris, t3):
    '''T3 on the packed array [C(no,3), C(nv,3)], from the flat array of its
    blocks (zccsdt.cc.ccsdt_opt.packed_t3).'''
    return ccsdt_opt.packed_t3(eris, t3)


def plan(lay, isz, budget=None, diis_space=8, incore_complete=False, step_need=None, mem_mb=2000.0):
    '''The Plan for ``budget`` bytes of ordinary memory (None: no limit) and
    the blocks ``lay``.  From the floor up: the batches of the engine grow
    to their default, then the buffers of the CCSDT step to ``mem_mb``, and
    with what is left the DIIS vectors of T4 stay in memory
    (``incore_complete``: they do in any case) - one after the other, so
    that no part of a plan shrinks when the budget grows.
    ``step_need(mem_mb)``: bytes of the CCSDT step for these buffers.'''
    no, nv = lay.no, lay.nv
    t4, t3 = isz * lay.size, isz * int(lay.t3off[-1])
    if step_need is None:
        def step_need(mb):
            return 0
    least_mb = min(mem_mb, STEP_MB)
    step_lo = step_need(least_mb)
    fixed = t3 + ccsdtq._dressed_bytes(no, nv, isz) + S.table_bytes(lay) + t4
    lo = S.min_batch_bytes(lay, isz)
    top = max(lo, S.BATCH_BYTES)
    diis = 2 * diis_space * (t4 + t3)

    def engine(batch):
        return S.memory(lay, isz, batch)
    floor = step_lo + fixed + engine(lo) + (diis if incore_complete else 0)
    if budget is None:
        batch, mb = top, mem_mb
        on_file = not incore_complete and lay.size >= INCORE_SIZE
    else:
        if budget < floor:
            raise MemoryError(
                'CCSDTQ in the blocks of the symmetry labels needs %.0f MB besides the integrals for %d '
                'occupied and %d virtual spinors (T4 and its residual in blocks: %.0f MB each%s); %.0f MB '
                'are left' % (floor / 1e6, no, nv, t4 / 1e6,
                              ', the DIIS vectors in memory' if incore_complete else '', budget / 1e6))
        rest = budget - step_lo - fixed - (diis if incore_complete else 0)
        batch, mb = lo, least_mb
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
            batch == top and mb == mem_mb and step_need(mb) + fixed + engine(batch) + diis <= budget)
    need = step_need(mb) + fixed + engine(batch) + (0 if on_file else diis)
    return Plan(int(batch), bool(on_file), int(need), int(floor), int(t4), float(mb), lay, 0,
                None if budget is None else max(0, int(budget - need)))


def plan_for(eris, dtype, max_memory=None, diis_space=8, incore_complete=False, mem_mb=2000.0):
    '''The Plan of a calculation on the integral container ``eris``:
    ``max_memory`` MB for the whole process (None: what the node and the
    job leave).'''
    lay = S.layout_for(eris)
    if lay is None:
        raise ValueError('no blocks for these integrals: no labels without a modulus in their order')

    def step_need(mb):
        return ccsdt_opt.plan_for(eris, dtype, solver.Options(mem_mb=mb, incore_complete=True)).need
    return plan(lay, np.dtype(dtype).itemsize, memory.budget(max_memory), diis_space, incore_complete,
                step_need=step_need, mem_mb=mem_mb)


def kernel(eris, opts=None, t1=None, t2=None, t3=None, t4=None):
    '''CCSDTQ by Jacobi steps with DIIS as zccsdt.cc.ccsdtq.kernel, in the
    blocks of the labels of ``eris``: from the given amplitudes (those of
    CCSDT, say: t1[i,a], t2[i,j,a,b], t3 in the blocks of the CCSDT step or
    packed, t4 in blocks or packed) or from those of MP2.  ``opts.max_disk``,
    ``opts.diis_t4``: as there (the DIIS vectors by zccsdt.disk; nothing
    else of this solver is in scratch files).'''
    from zccsdt import disk
    opts = opts or Options()
    with disk.limit(opts.max_disk):     # (every scratch file of the run counts against max_disk)
        return _kernel(eris, opts, t1, t2, t3, t4)


def _kernel(eris, opts, t1, t2, t3, t4):
    from zccsdt import disk
    lay = S.layout_for(eris)
    if lay is None:
        raise ValueError('no blocks for these integrals: no labels without a modulus in their order')
    nocc, nvir = lay.no, lay.nv
    if t1 is None or t2 is None:
        t1, t2 = solver.init_amps(eris)
    dtype = np.result_type(t1, t2, eris.oovv, eris.fock)
    t1, t2 = t1.astype(dtype), t2.astype(dtype)
    n3, n3p, n4p = int(lay.t3off[-1]), comb(nocc, 3) * comb(nvir, 3), comb(nocc, 4) * comb(nvir, 4)
    assert ccsdt_opt.engine_for(eris, dtype).size == n3     # (the blocks of the CCSDT step are these)
    if t3 is None:
        T3 = np.zeros(n3, dtype=dtype)
    elif np.size(t3) == n3:
        T3 = np.array(t3, dtype=dtype).ravel()
    else:
        assert np.size(t3) == n3p
        T3 = np.ascontiguousarray(lay.t3_from_packed(t3), dtype=dtype)
    if t4 is None:
        T4 = lay.zeros(dtype)
    elif np.size(t4) == lay.size:
        T4 = np.array(t4, dtype=dtype).ravel()
    else:
        assert np.size(t4) == n4p
        T4 = np.ascontiguousarray(lay.from_packed(t4), dtype=dtype)
    mo_e = np.asarray(eris.mo_energy)
    eo, ev = mo_e[:nocc], mo_e[nocc:]
    d2 = eo[P.tuples(nocc, 2)].sum(axis=1)[:, None] - ev[P.tuples(nvir, 2)].sum(axis=1)[None, :]
    eo3, ev3 = eo[lay.o3].sum(axis=1), P.tuple_sums(ev, 3)
    d3 = np.concatenate([(eo3[lay.O3cls[k]][:, None] - ev3[lay.A3cls[k]][None, :]).ravel()
                         for k in range(lay.n3)]) if n3 else 0
    pl = opts.plan or plan_for(eris, dtype, opts.max_memory, opts.diis_space, opts.incore_complete, opts.mem_mb)
    mem_mb = opts.mem_mb if pl.mem_mb is None else pl.mem_mb
    step = solver.Options(mem_mb=mem_mb)                        # (of the CCSDT step: no level shift)
    step_plan = ccsdt_opt.plan_for(eris, dtype, solver.Options(mem_mb=mem_mb, incore_complete=True))
    dplan = opts.diis_plan or ccsdtq.diis_plan_for(
        pl, [nocc * nvir, comb(nocc, 2) * comb(nvir, 2), n3, lay.size], np.dtype(dtype).itemsize, opts)
    adiis = DIIS(space=dplan.space, filename=opts.diis_file, incore=dplan.n_file == 0,
                 incore_size=min(INCORE_SIZE, max(1, lay.size)), in_memory=dplan.n_memory)
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
                R4, r3, r2 = S.residual(lay, t2, T3, T4, ints, pl.batch_bytes)
            del ints
            with timer('ccsdtq.update'):
                t2n = t2n + P.unpack(r2 / d2, nocc, nvir, 2)
                if n3:
                    r3 /= d3
                    t3n += r3
                S.jacobi(lay, T4, R4, eo, ev)
            del r3, r2
            with timer('ccsdtq.diis'):
                first = adiis.xprev is None
                if first:
                    norms = [np.linalg.norm(t1n - t1), np.linalg.norm(t2n - t2),
                             6.0 * solver._diff_norm(t3n, T3), 24.0 * solver._diff_norm(R4, T4)]
                if dplan.with_big:
                    parts = adiis.update(solver._parts(t1n, t2n, t3n) + [R4], out=[None, None, T3, T4])
                else:                   # DIIS without T4: its Jacobi step is the new T4
                    n4 = 24.0 * solver._diff_norm(R4, T4)
                    parts = adiis.update(solver._parts(t1n, t2n, t3n), out=[None, None, T3])
                    T4 = R4
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
    return Result(e_corr=float(np.real(e_old)), e_tot=eris.e_hf + float(np.real(e_old)), converged=conv,
                  ncycle=cycle + 1, t1=t1, t2=t2, t3=T3, t4=T4, timings=timings, plan=pl, diis_plan=dplan)
