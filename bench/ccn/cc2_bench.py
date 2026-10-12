'''Timing of the Cholesky CC2 of spinors (zccsdt.zccsd.cc2) against a CCSD
iteration of DirectZCCSD and against the GEMM rate of the node
(docs/ccn.md, section 8):

    python bench/ccn/cc2_bench.py MOLECULE BASIS FROZEN [WHAT]

  MOLECULE  a key of GEOM of bench/profile_iter.py (c6h6, ..)
  BASIS     a PySCF name, or 'heavy,light' (e.g. cc-pvtz,cc-pvdz)
  FROZEN    frozen occupied spinors
  WHAT      comma separated (default cc2,sigma,gemm):
            cc2     NITER iterations of CC2 with the timers of every phase
            sigma   one sigma vector of EOM-CC2 (after two more iterations)
            gemm    the ZGEMM rate at the shape of the pair loop
            ccsd    NITER CCSD iterations of DirectZCCSD(ladder='ao'), the
                    production CCSD of this package (zccsdt.zccsd.amps)
            ccsd-chol  two CCSD iterations of DirectZCCSD(ladder='chol') on
                    the same vectors (socutils' einsum route; o v^3 and
                    n_chol n^2 complex arrays in memory)
environment: ZC_MAXMEM (MB, default 48000), ZC_NITER (default 4),
             ZC_MAX_ERROR (default 1e-5: the default of the class),
             ZC_CDERI: a .npy file to keep the Cholesky vectors in between runs
'''
import os
import sys
import time

import numpy as np
from pyscf import gto, x2c

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from profile_iter import GEOM                                            # noqa: E402

from zccsdt import memory, timing                                        # noqa: E402
from zccsdt.blas import gemm                                             # noqa: E402
from zccsdt.zccsd import DirectZCCSD                                     # noqa: E402

name, basis, frozen = sys.argv[1], sys.argv[2], int(sys.argv[3])
what = (sys.argv[4] if len(sys.argv) > 4 else 'cc2,sigma,gemm').split(',')
maxmem = float(os.environ.get('ZC_MAXMEM', 48000))
niter = int(os.environ.get('ZC_NITER', 4))
max_error = float(os.environ.get('ZC_MAX_ERROR', 1e-5))
if ',' in basis:
    heavy, light = basis.split(',')
    atoms = {a.split()[0] for a in GEOM[name].split(';')}
    basis = {a: (light if a.upper() == 'H' else heavy) for a in atoms}
mol = gto.M(atom=GEOM[name], basis=basis, verbose=0, max_memory=maxmem)
t0 = time.perf_counter()
mf = x2c.UHF(mol)
mf.conv_tol = 1e-10
mf.kernel()
nthr = int(os.environ.get('OMP_NUM_THREADS', 1))
print('%s: %d AO functions, %d spinors, E(SCF) = %.10f (%.0f s), %d threads' % (
    name, mol.nao_nr(), 2 * mol.nao_nr(), mf.e_tot, time.perf_counter() - t0, nthr), flush=True)


def tree(times, parent=(), indent=1, total=None):
    for nm, w, c, k in timing.children(times, parent):
        if total is None or w > 0.002 * total:
            print('%s%-28s %9.3f s  %5.1f cores  %6d calls' % ('  ' * indent, nm, w, c / w if w > 0 else 0, k))
            tree(times, parent + (nm,), indent + 1, total or w)


cderi = None
path = os.environ.get('ZC_CDERI')
if path and os.path.exists(path):
    cderi = np.load(path)
cc = DirectZCCSD(mf, frozen=frozen, ladder='chol', cderi=cderi, max_error=max_error)
cc.verbose, cc.max_memory = 0, maxmem
if 'cc2' in what or 'sigma' in what or 'gemm' in what:
    t0 = time.perf_counter()
    eng = cc.cc2_engine()
    if path and cderi is None:
        np.save(path, eng.L)
    print('Cholesky vectors and static three-index integrals: %.1f s' % (time.perf_counter() - t0))
    print(eng.describe(), flush=True)
    o, v, nchol = eng.o, eng.v, eng.nchol
    flops = 1.5 * o * o * v * v * nchol
    print('pair loop: 1.5 o^2 v^2 n_chol = %.3e complex multiply-adds per iteration' % flops)
if 'gemm' in what:
    # the three products of one occupied index i: (v, n_chol) x (n_chol, nj v) and its two mirrors
    nj = max(1, o // 2)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((v, nchol)) + 1j * rng.standard_normal((v, nchol))
    B = rng.standard_normal((nchol, nj * v)) + 1j * rng.standard_normal((nchol, nj * v))
    C = np.empty((v, nj * v), dtype=complex)
    gemm(A, B, C)
    t0 = time.perf_counter()
    for _ in range(3):
        gemm(A, B, C)
    rate = 3 * v * nchol * nj * v / (time.perf_counter() - t0)
    print('ZGEMM (%d x %d x %d): %.3e complex multiply-adds per second (%.0f GFLOP/s counting 8 flops each)'
          % (v, nchol, nj * v, rate, 8 * rate / 1e9))
    print('ideal time of the pair loop at that rate: %.2f s' % (flops / rate), flush=True)
    del A, B, C
if 'cc2' in what or 'sigma' in what:
    t1 = np.zeros((o, v), dtype=complex)
    eia = eng.eo[:, None] - eng.ev[None, :]
    rss0 = memory.resident()
    for it in range(niter):
        snap = timing.snapshot()
        t0, c0 = time.perf_counter(), time.process_time()
        R1, e = eng.residual(t1, keep=it == niter - 1)
        t1 = t1 + R1 / eia
        w, c = time.perf_counter() - t0, time.process_time() - c0
        print('CC2 iteration %d: %.2f s, %.1f cores busy; E_corr = %.10f; resident %.2f GB' % (
            it + 1, w, c / w, e.real, memory.resident() / 1e9), flush=True)
    print('the last iteration by phase (wall, cores busy):')
    tree(timing.since(snap))
    print('memory: three-index arrays planned %.2f GB, work %.2f GB; resident now %.2f GB (%.2f GB before the iterations)'
          % (eng.plan['stored'] / 1e9, eng.plan['work'] / 1e9, memory.resident() / 1e9, rss0 / 1e9), flush=True)
if 'sigma' in what:
    eng.t1 = t1
    r1 = np.zeros((o, v), dtype=complex)
    r1[o - 1, 0] = 1.0
    for rep in range(2):
        snap = timing.snapshot()
        t0, c0 = time.perf_counter(), time.process_time()
        s = eng.sigma(r1, 0.2)
        w, c = time.perf_counter() - t0, time.process_time() - c0
        print('EOM-CC2 sigma vector: %.2f s, %.1f cores busy; resident %.2f GB' % (w, c / w, memory.resident() / 1e9), flush=True)
    tree(timing.since(snap))
for variant in ('ccsd', 'ccsd-chol'):
    if variant not in what:
        continue
    stamps = []
    ref = DirectZCCSD(mf, frozen=frozen, ladder='ao' if variant == 'ccsd' else 'chol',
                      cderi=cc._cc2.L if (variant == 'ccsd-chol' and getattr(cc, '_cc2', None) is not None) else cderi,
                      max_error=max_error)
    ref.verbose, ref.max_memory = 0, maxmem
    ref.max_cycle = niter if variant == 'ccsd' else 2
    ref.callback = lambda envs: stamps.append(time.perf_counter())
    t0 = time.perf_counter()
    eris = ref.ao2mo()
    t1 = time.perf_counter()
    stamps.append(t1)
    try:
        ref.kernel(eris=eris)
    except Exception as err:                # (not converged in the cycles allowed: fine)
        print('  (%s)' % err)
    it = np.diff(stamps)
    print('%s of DirectZCCSD(ladder=%r): integrals %.1f s; iterations %s s; E_corr = %.10f; resident %.2f GB' % (
        'CCSD', ref.ladder, t1 - t0, ' '.join('%.2f' % x for x in it), ref.e_corr, memory.resident() / 1e9), flush=True)
