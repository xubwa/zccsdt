'''Iterative CCSD / CCSDT driver (Jacobi update + DIIS).

Replaces the PySCF CCSD class machinery the socutils code relied on; works
directly on an ERIs container (zccsdt.eris.ERIs or PySCF's gccsd ERIs).

Memory: the DIIS vectors are in a scratch file unless ``incore_complete``
(zccsdt.cc.diis).  ``max_memory`` decides where the other large arrays of
CCSDT live (ccsdt_opt.plan_memory): when they do not fit, T3 and its residual
are mapped arrays of a scratch file and the packed <ab||de> is used in slabs.
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
    diis_space: int = 8
    diis_file: str = None       # HDF5 file of the DIIS vectors (default: temporary)
    incore_complete: bool = False   # keep all DIIS vectors in memory (as in PySCF)
    mem_mb: float = 2000.0      # batch buffer budget of the packed-T3 code
    spin_flip: bool = False     # closed-shell alpha/beta symmetry (RCCSDT)
    verbose: bool = False
    max_memory: float = None    # MB for the whole process, as in PySCF (default:
                                # 80% of what the node and the job limit leave)
    plan: object = None         # ccsdt_opt.MemoryPlan given explicitly


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
    (m, n), (d, e) = np.triu_indices(t1.shape[0], 1), np.triu_indices(t1.shape[1], 1)
    t2p = t2[m[:, None], n[:, None], d, e]
    parts = [t1.ravel(), t2p.ravel()]
    if t3 is not None:
        parts.append(t3)
    return parts


def _from_parts(parts, nocc, nvir, with_t3):
    '''Inverse of _parts; keeps the dtype (real stays real).'''
    (m, n), (d, e) = np.triu_indices(nocc, 1), np.triu_indices(nvir, 1)
    t1 = parts[0].reshape(nocc, nvir)
    p = parts[1].reshape(len(m), len(d))
    m, n = m[:, None], n[:, None]
    t2 = np.zeros((nocc, nocc, nvir, nvir), dtype=p.dtype)
    t2[m, n, d, e] = p
    t2[n, m, d, e] = -p
    t2[m, n, e, d] = -p
    t2[n, m, e, d] = p
    t3 = parts[2] if with_t3 else None
    return t1, t2, t3


def _diff_norm(a, b, chunk=1 << 22):
    '''||a - b|| of two flat arrays without a temporary of their size.'''
    s = 0.0
    for i in range(0, a.size, chunk):
        d = a[i:i + chunk] - b[i:i + chunk]
        s += float(np.vdot(d, d).real)
    return np.sqrt(s)


def kernel(eris, method='ccsdt', opts=None, t1=None, t2=None, t3=None):
    '''Solve CCSD or CCSDT.  Returns a Result; t3 is kept packed.
    method: 'ccsd', 'ccsdt' (packed-T3 implementation) or 'ccsdt_ref'
    (the ported reference implementation, for validation).

    For 'ccsdt', ``opts.max_memory`` (MB; default: 80% of what the node and
    the memory cgroups of the process leave) decides where the large arrays
    live (ccsdt_opt.plan_memory); ``opts.incore_complete`` keeps everything
    in memory whatever the budget and ``opts.plan`` gives a MemoryPlan
    explicitly.  The plan used is returned as Result.plan.'''
    opts = opts or Options()
    method = method.lower()
    assert method in ('ccsd', 'ccsdt', 'ccsdt_ref')
    with_t3 = method != 'ccsd'
    nocc = eris.nocc
    nvir = eris.fock.shape[0] - nocc
    if t1 is None or t2 is None:
        t1, t2 = init_amps(eris)
    dtype = np.result_type(t1, t2, eris.oovv, eris.fock)
    t1 = t1.astype(dtype)
    t2 = t2.astype(dtype)
    plan = scr = None
    tight = False
    if method == 'ccsdt':
        plan = opts.plan or ccsdt_opt.plan_for(eris, dtype, opts)
        # memory is short if the plan has put anything into a scratch file:
        # the kernels then tell the operating system what they do with mapped arrays
        tight = plan.t3_file or plan.pp_file or plan.pp_bytes is not None
    hints = scratch.hints(tight)
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
                # T3 as a mapped array of a scratch file (new file space reads
                # as zeros); its residual is built in the storage of the DIIS vectors
                scr = Scratch(prefix='zccsdt_t3_', what='T3',
                              hint=' or give the calculation more memory (max_memory)')
                mapped = scr.create('t3', size, dtype)
                if t3 is not None:
                    mapped[:] = np.asarray(t3).ravel()
                t3 = mapped
            elif t3 is None:
                t3 = np.zeros(size, dtype=dtype)
            elif method == 'ccsdt':
                t3 = np.array(t3, dtype=dtype).ravel()  # (overwritten below: not the caller's)

        e_old = ccsd.energy(t1, t2, eris)
        adiis = DIIS(space=opts.diis_space, filename=opts.diis_file,
                     incore=opts.incore_complete)
        for icycle in range(opts.max_cycle):
            t0 = time.perf_counter()
            if with_t3:
                if method == 'ccsdt':
                    rbuf = adiis.buffers(_parts(t1, t2, t3))[0][2] if scr is not None else None
                    t1n, t2n, t3n = ccsdt_opt.update_amps(opts, t1, t2, t3, eris,
                                                          mem_mb=opts.mem_mb, out=rbuf,
                                                          plan=plan)
                else:
                    t1n, t2n, t3n = ccsdt.update_amps(opts, t1, t2, t3, eris)
            else:
                t1n, t2n = ccsdt_opt.ccsd_update_amps(opts, t1, t2, eris)
                t3n = None
            with timer('diis'):
                first = adiis.xprev is None
                if first:
                    normt = np.linalg.norm(t1n - t1) + np.linalg.norm(t2n - t2)
                    if with_t3:
                        with timer('diis.norm0'):
                            normt += 6.0 * _diff_norm(t3n, t3)
                parts = adiis.update(_parts(t1n, t2n, t3n),
                                     out=[None, None, t3] if in_place else None)
                if not first:
                    # ||dt2_full|| = 2 ||dt2_packed||, ||dt3_full|| = 6 ||dt3_packed||
                    en = adiis.err_norms
                    normt = en[0] + 2.0 * en[1] + (6.0 * en[2] if with_t3 else 0.0)
                with timer('diis.unpack'):
                    t1, t2, t3 = _from_parts(parts, nocc, nvir, with_t3)
            e_new = ccsd.energy(t1, t2, eris)
            timings.append(time.perf_counter() - t0)
            de = e_new - e_old
            e_old = e_new
            if opts.verbose:
                print('cycle %3d  E_corr = %.15f  dE = %.3e  |dt| = %.3e  %.2fs'
                      % (icycle + 1, e_new.real, de.real, normt, timings[-1]))
            if abs(de) < opts.conv_tol and normt < opts.conv_tol_normt:
                conv = True
                break
    finally:
        scratch.hints(hints)
        if adiis is not None:
            adiis.close()
        if scr is not None:
            scr.close()                 # the mapped t3 stays valid (the file is unlinked)
    e_corr = e_old
    return Result(e_corr=e_corr, e_tot=eris.e_hf + e_corr.real, converged=conv,
                  ncycle=icycle + 1, t1=t1, t2=t2, t3=t3, timings=timings, plan=plan)
