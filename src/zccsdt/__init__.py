'''zccsdt: spin-orbital / spinor CCSDT, CCSDT(Q), CCSDTQ and CCSDTQ(P).'''

import os

# OpenMP threads of the C kernels must not spin while BLAS threads run (and
# vice versa); this only takes effect if libgomp is not loaded yet.
os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')


# the PySCF-style front end (zccsdt.ccsdt), imported on first use
_FRONT = ('CCSDT', 'RCCSDT', 'UCCSDT', 'GCCSDT', 'CCSDTQ', 'GCCSDTQ')


def __getattr__(name):
    if name in _FRONT:
        from zccsdt import ccsdt
        return getattr(ccsdt, name)
    raise AttributeError("module 'zccsdt' has no attribute %r" % name)
