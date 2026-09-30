'''Iterative CCSD / CCSDT driver (Jacobi update + DIIS).

Replaces the PySCF CCSD class machinery the socutils code relied on; works
directly on an ERIs container (zccsdt.eris.ERIs or PySCF's gccsd ERIs).
'''

import time
from dataclasses import dataclass
import numpy as np
from pyscf import lib

from zccsdt.cc import ccsd, ccsdt, ccsdt_opt, t3pack
from zccsdt.timing import timer


@dataclass
class Options:
    level_shift: float = 0.0
    conv_tol: float = 1e-10
    conv_tol_normt: float = 1e-8
    max_cycle: int = 200
    diis_space: int = 8
    diis_incore: bool = True    # PySCF's default keeps DIIS vectors on disk
    mem_mb: float = 2000.0      # batch buffer budget of the packed-T3 code
    verbose: bool = False


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


def _pack(t1, t2, t3):
    om, vm = t3pack.triple_map(t1.shape[0]), t3pack.triple_map(t1.shape[1])
    t2p = t2[om.pairs[:, 0][:, None], om.pairs[:, 1][:, None],
             vm.pairs[:, 0], vm.pairs[:, 1]]
    parts = [t1.ravel(), t2p.ravel()]
    if t3 is not None:
        parts.append(t3)
    return np.hstack(parts)


def _unpack(vec, nocc, nvir, with_t3):
    '''Inverse of _pack; keeps the dtype of vec (real stays real).'''
    om, vm = t3pack.triple_map(nocc), t3pack.triple_map(nvir)
    n1 = nocc * nvir
    n2 = om.npair * vm.npair
    t1 = vec[:n1].reshape(nocc, nvir)
    t2 = ccsdt_opt._unpack_pairs(vec[n1:n1 + n2].reshape(om.npair, vm.npair), om, vm)
    t3 = vec[n1 + n2:].copy() if with_t3 else None
    return t1, t2, t3


def kernel(eris, method='ccsdt', opts=None, t1=None, t2=None, t3=None):
    '''Solve CCSD or CCSDT.  Returns a Result; t3 is kept packed.
    method: 'ccsd', 'ccsdt' (packed-T3 implementation) or 'ccsdt_ref'
    (the ported reference implementation, for validation).'''
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
    if with_t3 and t3 is None:
        t3 = np.zeros(t3pack.triple_map(nocc).ntri * t3pack.triple_map(nvir).ntri,
                      dtype=dtype)

    e_old = ccsd.energy(t1, t2, eris)
    adiis = lib.diis.DIIS(incore=opts.diis_incore)
    adiis.space = opts.diis_space
    timings = []
    conv = False
    for icycle in range(opts.max_cycle):
        t0 = time.perf_counter()
        if with_t3:
            if method == 'ccsdt':
                t1n, t2n, t3n = ccsdt_opt.update_amps(opts, t1, t2, t3, eris,
                                                      mem_mb=opts.mem_mb)
            else:
                t1n, t2n, t3n = ccsdt.update_amps(opts, t1, t2, t3, eris)
            normt = (np.linalg.norm(t1n - t1) + np.linalg.norm(t2n - t2)
                     + 6.0 * np.linalg.norm(t3n - t3))
        else:
            t1n, t2n = ccsd.update_amps(opts, t1, t2, eris)
            t3n = None
            normt = np.linalg.norm(t1n - t1) + np.linalg.norm(t2n - t2)
        with timer('diis'):
            vec = adiis.update(_pack(t1n, t2n, t3n))
            t1, t2, t3 = _unpack(vec, nocc, nvir, with_t3)
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
    e_corr = e_old
    return Result(e_corr=e_corr, e_tot=eris.e_hf + e_corr.real, converged=conv,
                  ncycle=icycle + 1, t1=t1, t2=t2, t3=t3, timings=timings)
