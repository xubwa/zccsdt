'''zccsdt: spin-orbital / spinor CCSDT, CCSDT(Q), CCSDTQ and CCSDTQ(P).'''

import os

# OpenMP threads of the C kernels must not spin while BLAS threads run (and
# vice versa); this only takes effect if libgomp is not loaded yet.
os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')

# PySCF's libraries before the kernel libraries of this package: the OpenMP
# runtime that they bring (in a conda environment libgomp.so.1 is LLVM's
# libomp, which also serves numpy's MKL) then serves the kernels too.  Loaded
# first, the kernel libraries take the libgomp of the compiler from the
# library path instead, and the process has two runtimes: with OMP_PROC_BIND
# each binds its threads, the second one within the single core that the
# first has left the main thread, and everything runs on that core (measured:
# a GEMM 250 times slower, 1.0 cores busy).
from pyscf import lib as _pyscf_lib      # noqa: E402, F401


# the PySCF-style front end (zccsdt.ccsdt), imported on first use
_FRONT = ('CCSDT', 'RCCSDT', 'UCCSDT', 'GCCSDT', 'CCSDTQ', 'GCCSDTQ')
# the approximate models and the rest of the ladder (zccsdt.ccn; docs/ccn.md)
_CCN = ('CC2', 'CCSD', 'CC3', 'CC4', 'GCC2', 'GCCSD', 'GCC3', 'GCC4')


def __getattr__(name):
    if name in _CCN:
        from zccsdt import ccn
        return getattr(ccn, name)
    if name in _FRONT:
        from zccsdt import ccsdt
        return getattr(ccsdt, name)
    raise AttributeError("module 'zccsdt' has no attribute %r" % name)
