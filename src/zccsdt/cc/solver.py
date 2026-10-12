'''Iterative CCSD / CCSDT driver (Jacobi update + DIIS).

Replaces the PySCF CCSD class machinery the socutils code relied on; works
directly on an ERIs container (zccsdt.eris.ERIs or PySCF's gccsd ERIs).

Memory: the DIIS vectors are in a scratch file unless ``incore_complete``
(zccsdt.cc.diis).  ``max_memory`` decides where the other large arrays of
CCSDT live (ccsdt_opt.plan_memory): when they do not fit, T3 and its residual
are mapped arrays of a scratch file and the packed <ab||de> is used in slabs.

Disk: ``max_disk`` (MB; default: the free space of the scratch directory
less a reserve) is what the scratch files of the run may take.  The mapped
arrays of the memory plan come first; the DIIS vectors get the rest
(zccsdt.disk, diis_plan_for): all ``diis_space`` of them in the file, else
some of them in memory, else fewer of them, else DIIS without T3 - or an
error before the first iteration, with the numbers.
'''

import time
from dataclasses import dataclass
import numpy as np
from pyscf import lib

from zccsdt.cc import ccsd, ccsdt, ccsdt_opt, t3pack
from zccsdt import scratch
from zccsdt.cc.diis import DIIS
from zccsdt.scratch import Scratch
from zccsdt.timing import timer


@dataclass
class Options:
    level_shift: float = 0.0
    conv_tol: float = 1e-10
    conv_tol_normt: float = 1e-8
    max_cycle: int = 200
    diis_space: int = 8         # stored vectors asked for: a maximum (zccsdt.disk)
    diis_file: str = None       # HDF5 file of the DIIS vectors (default: temporary)
    diis_t3: bool = None        # T3 in the DIIS vectors (CCSDT, Lambda): None - yes,
                                # unless neither disk nor memory holds three of them;
                                # True - yes, or an error; False - T1 and T2 only
    incore_complete: bool = False   # keep all DIIS vectors in memory (as in PySCF)
    mem_mb: float = 2000.0      # batch buffer budget of the packed-T3 code
    spin_flip: bool = False     # closed-shell alpha/beta symmetry (RCCSDT)
    store_t3: bool = True       # CC3 only.  False: no T3 is stored - the triples are made per
                                # batch of occupied triples and contracted (zccsdt.cc.cc3fly)
    verbose: bool = False
    max_memory: float = None    # MB for the whole process, as in PySCF (default:
                                # 80% of what the node and the job limit leave)
    max_disk: float = None      # MB for the scratch files of the process (default:
                                # ZCCSDT_MAX_DISK, else the free space of the scratch
                                # directory less a reserve)
    plan: object = None         # ccsdt_opt.MemoryPlan given explicitly
    diis_plan: object = None    # zccsdt.disk.DiisPlan given explicitly
    callback: object = None     # callback(info) after every iteration (kernel)


@dataclass
class Result:
    e_corr: float
    e_tot: float
    converged: bool
    ncycle: int
    t1: np.ndarray
    t2: np.ndarray
    t3: np.ndarray = None       # packed (i<j<k, a<b<c) for CCSDT
    timings: list = None        # wall time per iteration
    plan: object = None         # the MemoryPlan of a CCSDT run
    diis_plan: object = None    # where its DIIS vectors were (zccsdt.disk.DiisPlan)
    eris: object = None         # CC2 / CC3: the integrals of the spinors the amplitudes refer to
    rotation: object = None     # (U_o, U_v) if those are the given ones made semicanonical


def init_amps(eris):
    '''MP2 guess: t1 = f_ai / D_ia, t2 = <ab||ij> / D_ijab.'''
    nocc = eris.nocc
    mo_e = eris.mo_energy
    eia = mo_e[:nocc, None] - mo_e[None, nocc:]
    eijab = eia[:, None, :, None] + eia[None, :, None, :]
    fov = np.asarray(eris.fock)[:nocc, nocc:]
    t1 = fov.conj() / eia
    t2 = np.asarray(eris.oovv).conj() / eijab
    return t1, t2


def _parts(t1, t2, t3):
    '''Amplitudes as a list of flat parts for DIIS (t2 packed over i<j, a<b
    in triu order).'''
    om, vm = t3pack.lex_pairs(t1.shape[0]), t3pack.lex_pairs(t1.shape[1])
    t2p = t3pack.pack_last2(np.ascontiguousarray(t2[om.pa_c, om.pb_c]), vm)
    parts = [t1.ravel(), t2p.ravel()]
    if t3 is not None:
        parts.append(t3)
    return parts


def _from_parts(parts, nocc, nvir, with_t3):
    '''Inverse of _parts; keeps the dtype (real stays real).'''
    t1 = parts[0].reshape(nocc, nvir)
    t2 = t3pack.unpack_pairs(parts[1], t3pack.lex_pairs(nocc), t3pack.lex_pairs(nvir))
    t3 = parts[2] if with_t3 else None
    return t1, t2, t3


def _diff_norm(a, b, chunk=1 << 22):
    '''||a - b|| of two flat arrays without a temporary of their size.'''
    s = t3pack.diff_norm2(a, b) if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) else None
    if s is not None:
        return np.sqrt(s)
    s = 0.0
    for i in range(0, a.size, chunk):
        d = a[i:i + chunk] - b[i:i + chunk]
        s += float(np.vdot(d, d).real)
    return np.sqrt(s)


def diis_plan_for(eris, dtype, opts, plan):
    '''Where the DIIS vectors of a CCSDT run with the MemoryPlan ``plan`` go
    (zccsdt.disk.DiisPlan; the rule: zccsdt.disk): from ``opts.max_disk``
    less what the plan maps to scratch files, and the memory that the plan
    leaves of its budget.  Raises DiskFullError if the mapped arrays of the
    plan do not fit on the disk, or the vectors with an explicit
    ``diis_t3 = True``; MemoryError if ``incore_complete`` cannot be met.'''
    from zccsdt import disk, memory
    size = ccsdt_opt.engine_for(eris, dtype, flip=getattr(opts, 'spin_flip', False)).size
    t3 = size * np.dtype(dtype).itemsize
    o = eris.nocc
    v = eris.fock.shape[0] - o
    big = disk.file_bytes([o * v, o * (o - 1) // 2 * (v * (v - 1) // 2), size], np.dtype(dtype).itemsize)
    r3_file = bool(plan.t3_file if plan.r3_file is None else plan.r3_file)
    spare = plan.spare
    if opts.incore_complete:    # (whatever max_memory: what the node and the job leave)
        avail = memory.available()
        spare = None if avail is None else max(0, avail + ccsdt_opt.cached_bytes(eris) - plan.need)
    dp = disk.diis_plan(big, opts.diis_space, opts.max_disk, opts.diis_file, plan.disk, spare,
                        opts.diis_t3, opts.incore_complete, 'T3',
                        memory_tight=bool(plan.t3_file or r3_file))
    if r3_file and not dp.with_big and size:
        # the residual is not built in the DIIS storage then: one more array of the scratch file
        disk.require(plan.disk + t3, 'T3 and its residual', opts.max_disk)
    return dp


def kernel(eris, method='ccsdt', opts=None, t1=None, t2=None, t3=None):
    '''Solve CCSD or CCSDT.  Returns a Result; t3 is kept packed.
    method: 'ccsd', 'ccsdt' (packed-T3 implementation) or 'ccsdt_ref'
    (the ported reference implementation, for validation); 'cc2', 'cc3':
    the approximate models (docs/ccn.md), CC3 with T3 stored as in CCSDT.

    For 'ccsdt', ``opts.max_memory`` (MB; default: 80% of what the node and
    the memory cgroups of the process leave) decides where the large arrays
    live (ccsdt_opt.plan_memory); ``opts.incore_complete`` keeps everything
    in memory whatever the budget and ``opts.plan`` gives a MemoryPlan
    explicitly.  The plan used is returned as Result.plan.  ``opts.max_disk``
    (MB) bounds the scratch files of the run; where the DIIS vectors go
    follows from both (diis_plan_for; returned as Result.diis_plan), and a
    disk that does not hold what the run needs is an error before the first
    iteration.

    ``opts.callback(info)``, if given, is called after every iteration with
    a dict: cycle, e_corr, de, norms (|dt1|, |dt2|, |dt3| of the full
    tensors), normt, diis (vectors in the subspace), wall, cpu (seconds).'''
    opts = opts or Options()
    method = method.lower()
    # the approximate models (docs/ccn.md): CC3 is the CCSDT code with the triples equation
    # of model 'cc3' (T3 stored), CC2 the CCSD-level step of zccsdt.cc.ccn
    model = method
    if method == 'cc3':
        method = 'ccsdt' if opts.store_t3 else 'cc3_fly'
    assert method in ('ccsd', 'cc2', 'cc3_fly', 'ccsdt', 'ccsdt_ref')
    with_t3 = method not in ('ccsd', 'cc2', 'cc3_fly')
    rotation = None
    if model in ('cc2', 'cc3'):
        # the models with the full Fock blocks are invariant under rotations among the occupied
        # and among the virtual spinors: they are solved for semicanonical ones (exactly diagonal
        # blocks), and the amplitudes refer to those (Result.eris, Result.rotation)
        from zccsdt.cc import ccn
        eris, rotation = ccn.semicanonicalize(eris)
        if rotation is not None and (t1 is not None or t2 is not None or t3 is not None):
            raise ValueError('starting amplitudes for spinors that are not semicanonical: give those of '
                             'zccsdt.cc.ccn.semicanonicalize(eris)')
    nocc = eris.nocc
    nvir = eris.fock.shape[0] - nocc
    if t1 is None or t2 is None:
        t1, t2 = init_amps(eris)
    dtype = np.result_type(t1, t2, eris.oovv, eris.fock)
    t1 = t1.astype(dtype)
    t2 = t2.astype(dtype)
    plan = scr = dplan = None
    tight = r3_file = False
    from zccsdt import disk
    if method == 'ccsdt':
        plan = opts.plan or ccsdt_opt.plan_for(eris, dtype, opts)
        # (before anything is allocated: the disk holds the plan or the run stops here)
        with disk.limit(opts.max_disk):
            dplan = opts.diis_plan or diis_plan_for(eris, dtype, opts, plan)
        # memory is short if the plan has put anything into a scratch file:
        # the kernels then tell the operating system what they do with mapped arrays
        tight = plan.t3_file or plan.pp_file or plan.pp_bytes is not None
        # the residual of T3 in the scratch file (with T3, unless the plan
        # keeps it in memory): it is then built in the storage of the DIIS vectors
        r3_file = plan.t3_file if plan.r3_file is None else plan.r3_file
    hints = scratch.hints(tight)
    # DIIS without T3 (opts.diis_t3 = False, or nothing holds the vectors):
    # T3 takes its Jacobi step, into a second array that alternates with it
    diis_t3 = dplan is None or dplan.with_big
    other = None
    limit = disk.limit(opts.max_disk)
    limit.__enter__()
    adiis = None
    timings = []
    conv = False
    # packed-T3 code: the DIIS extrapolation is written over the old T3 (two
    # T3-sized arrays at a time, not three); out of core, the new T3 is
    # moreover built in the storage of the DIIS vector
    in_place = method == 'ccsdt'
    try:
        if plan is not None:
            ccsdt_opt.place_vvvv(eris, plan)
        if with_t3:
            size = ccsdt_opt.engine_for(eris, dtype, flip=opts.spin_flip).size
            if plan is not None and plan.t3_file and size:
                # T3 as a mapped array of a scratch file (new file space reads as zeros)
                scr = Scratch(prefix='zccsdt_t3_', what='T3',
                              hint=' or give the calculation more memory (max_memory)')
                mapped = scr.create('t3', size, dtype)
                if r3_file and not diis_t3:
                    other = scr.create('r3', size, dtype)
                if t3 is not None:
                    mapped[:] = np.asarray(t3).ravel()
                t3 = mapped
            elif t3 is None:
                t3 = np.zeros(size, dtype=dtype)
            elif method == 'ccsdt':
                t3 = np.array(t3, dtype=dtype).ravel()  # (overwritten below: not the caller's)

        e_old = ccsd.energy(t1, t2, eris)
        if dplan is None:
            adiis = DIIS(space=opts.diis_space, filename=opts.diis_file,
                         incore=opts.incore_complete)
        else:
            adiis = DIIS(space=dplan.space, filename=opts.diis_file,
                         incore=opts.incore_complete or dplan.n_file == 0,
                         in_memory=dplan.n_memory)
        for icycle in range(opts.max_cycle):
            t0, c0 = time.perf_counter(), time.process_time()
            if with_t3:
                if method == 'ccsdt':
                    if not diis_t3:
                        rbuf = other
                    else:
                        rbuf = adiis.buffers(_parts(t1, t2, t3))[0][2] if r3_file else None
                    t1n, t2n, t3n = ccsdt_opt.update_amps(opts, t1, t2, t3, eris,
                                                          mem_mb=opts.mem_mb, out=rbuf,
                                                          plan=plan, model=model)
                else:
                    t1n, t2n, t3n = ccsdt.update_amps(opts, t1, t2, t3, eris)
            elif method == 'cc2':
                from zccsdt.cc import ccn
                t1n, t2n = ccn.cc2_update_amps(opts, t1, t2, eris)
                t3n = None
            elif method == 'cc3_fly':
                t1n, t2n, t3n = ccsdt_opt.update_amps(opts, t1, t2, None, eris, mem_mb=opts.mem_mb, model='cc3')
            else:
                t1n, t2n = ccsdt_opt.ccsd_update_amps(opts, t1, t2, eris)
                t3n = None
            with timer('diis'):
                first = adiis.xprev is None
                if first:
                    norms = [np.linalg.norm(t1n - t1), np.linalg.norm(t2n - t2), 0.0]
                if with_t3 and (first or not diis_t3):
                    with timer('diis.norm0'):
                        n3 = 6.0 * _diff_norm(t3n, t3)
                if diis_t3:
                    parts = adiis.update(_parts(t1n, t2n, t3n),
                                         out=[None, None, t3] if in_place else None)
                else:
                    parts = adiis.update(_parts(t1n, t2n, None))
                if not first:
                    # ||dt2_full|| = 2 ||dt2_packed||, ||dt3_full|| = 6 ||dt3_packed||
                    en = adiis.err_norms
                    norms = [en[0], 2.0 * en[1], 6.0 * en[2] if with_t3 and diis_t3 else 0.0]
                if with_t3 and (first or not diis_t3):
                    norms[2] = n3
                norms = [float(x) for x in norms]
                normt = sum(norms)
                with timer('diis.unpack'):
                    if diis_t3:
                        t1, t2, t3 = _from_parts(parts, nocc, nvir, with_t3)
                    else:
                        t1, t2, _ = _from_parts(parts, nocc, nvir, False)
                        # the new T3 as it is; its array of the last iteration is the next target
                        scratch.written(t3n)
                        other, t3 = t3, t3n
            e_new = ccsd.energy(t1, t2, eris)
            timings.append(time.perf_counter() - t0)
            de = e_new - e_old
            e_old = e_new
            if opts.verbose:
                print('cycle %3d  E_corr = %.15f  dE = %.3e  |dt| = %.3e  %.2fs'
                      % (icycle + 1, e_new.real, de.real, normt, timings[-1]))
            if opts.callback is not None:
                opts.callback(dict(cycle=icycle + 1, e_corr=float(np.real(e_new)),
                                   de=float(np.real(de)), norms=tuple(norms), normt=normt,
                                   diis=len(adiis.xs), wall=timings[-1],
                                   cpu=time.process_time() - c0))
            if abs(de) < opts.conv_tol and normt < opts.conv_tol_normt:
                conv = True
                break
    finally:
        scratch.hints(hints)
        limit.__exit__(None, None, None)
        if adiis is not None:
            adiis.close()
        if scr is not None:
            scr.close()                 # the mapped t3 stays valid (the file is unlinked)
    e_corr = e_old
    return Result(e_corr=e_corr, e_tot=eris.e_hf + e_corr.real, converged=conv,
                  ncycle=icycle + 1, t1=t1, t2=t2, t3=t3, timings=timings, plan=plan,
                  diis_plan=dplan, eris=eris if model in ('cc2', 'cc3') else None, rotation=rotation)
