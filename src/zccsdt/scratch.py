'''Scratch files for large arrays (DIIS vectors, T3, the packed vvvv operator).

A Scratch is an HDF5 file whose datasets are contiguous and are accessed
through numpy.memmap instead of explicit reads and writes: the kernels work
directly on the page cache, so nothing is copied while the arrays fit in
memory, and the operating system writes them out and pages them back in when
they do not.

The directory is ZCCSDT_TMPDIR, else PySCF's lib.param.TMPDIR (which follows
PYSCF_TMPDIR / TMPDIR).
'''

import ctypes
import mmap
import os
import tempfile

import numpy as np


def scratch_dir():
    '''Directory of the temporary scratch files.'''
    d = os.environ.get('ZCCSDT_TMPDIR')
    if d:
        return d
    try:
        from pyscf import lib
        return lib.param.TMPDIR
    except ImportError:
        return os.environ.get('TMPDIR') or tempfile.gettempdir()


class Scratch:
    '''HDF5 file holding large arrays as contiguous datasets, mapped into
    memory.  A temporary file is unlinked right after it is opened, so nothing
    is left behind however the process ends; a named file is kept and stays a
    valid HDF5 file.  ``what`` names the contents in error messages and
    ``hint`` lists the ways out of a full disk.'''

    def __init__(self, filename=None, prefix='zccsdt_', what='an array', hint=''):
        import h5py
        self.keep = filename is not None
        self.what, self.hint = what, hint
        if filename is None:
            fd, filename = tempfile.mkstemp(prefix=prefix, suffix='.h5', dir=scratch_dir())
            os.close(fd)
        self.filename = filename
        try:
            self.h5 = h5py.File(filename, 'w', locking=False)    # private to this object
        except TypeError:                                         # h5py < 3.5
            self.h5 = h5py.File(filename, 'w')
        self.raw = open(filename, 'r+b')
        if not self.keep:
            os.unlink(filename)

    def create(self, key, shape, dtype=np.float64):
        '''New dataset as a numpy.memmap of the given shape (an int for a
        vector).  The space is allocated at once: a full disk raises here,
        not as a bus error on a later write.  New space reads as zeros.'''
        import h5py
        dtype = np.dtype(dtype)
        shape = (int(shape),) if np.isscalar(shape) else tuple(int(x) for x in shape)
        n = int(np.prod(shape, dtype=np.int64))
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_layout(h5py.h5d.CONTIGUOUS)
        dcpl.set_alloc_time(h5py.h5d.ALLOC_TIME_EARLY)
        dcpl.set_fill_time(h5py.h5d.FILL_TIME_NEVER)
        ds = h5py.h5d.create(self.h5.id, key.encode(), h5py.h5t.py_create(dtype),
                             h5py.h5s.create_simple(shape), dcpl)
        offset = ds.get_offset()
        self.h5.flush()
        try:
            os.posix_fallocate(self.raw.fileno(), offset, n * dtype.itemsize)
        except OSError as err:
            raise OSError(err.errno, 'cannot allocate %.1f GB for %s in %s (%s); choose another '
                          'scratch directory (ZCCSDT_TMPDIR / PYSCF_TMPDIR)%s'
                          % (n * dtype.itemsize / 1e9, self.what, self.filename, err.strerror,
                             self.hint)) from err
        return np.memmap(self.raw, dtype=dtype, mode='r+', offset=offset, shape=shape)

    def close(self):
        if self.h5 is not None:
            try:
                self.h5.close()
                self.raw.close()
            finally:
                self.h5 = self.raw = None


_libc = None
_PAGE = mmap.PAGESIZE


def advise(a, advice='willneed', start=0, stop=None):
    '''madvise() on the elements start:stop (flat C order) of a file-backed
    array; nothing happens for an array in ordinary memory or if the call is
    not available.  ``advice``: 'willneed' (read ahead now), 'sequential',
    'random', 'dontneed' (drop the pages; they are written out first if
    dirty).'''
    global _libc
    if not isinstance(a, np.memmap) and not isinstance(getattr(a, 'base', None), np.memmap):
        return
    code = {'willneed': mmap.MADV_WILLNEED, 'sequential': mmap.MADV_SEQUENTIAL,
            'random': mmap.MADV_RANDOM, 'dontneed': mmap.MADV_DONTNEED}[advice]
    if _libc is None:
        try:
            _libc = ctypes.CDLL(None, use_errno=True)
            _libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
        except (OSError, AttributeError):
            _libc = False
    if not _libc:
        return
    n = a.size if stop is None else stop
    lo = a.ctypes.data + start * a.itemsize
    hi = a.ctypes.data + n * a.itemsize
    lo_al = lo - lo % _PAGE
    if hi > lo_al:
        _libc.madvise(lo_al, hi - lo_al, code)
