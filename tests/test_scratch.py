'''Mapped scratch arrays and the I/O hints on them (zccsdt.scratch).'''
import gc

import numpy as np
import pytest

from zccsdt import scratch
from zccsdt.scratch import Scratch


@pytest.fixture(autouse=True)
def _tmp(monkeypatch, tmp_path):
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    monkeypatch.setattr(scratch, 'IO_MODE', 1)  # ZCCSDT_IO=1 ...
    prev = scratch.hints(True)                  # ... in a run that is short of memory
    yield
    scratch.hints(prev)
    assert list(tmp_path.iterdir()) == []       # temporary files are unlinked when opened


def test_registry_follows_the_mappings():
    # mappings that earlier tests of the process left to the garbage collector go now, not in
    # the middle of this test; and what is counted below are the entries of this file - those
    # of other files may still come and go
    gc.collect()
    n0 = len(scratch._MAPS)
    s = Scratch(prefix='zccsdt_test_')
    fd0 = s.raw.fileno()

    def mine():
        return [m for m in scratch._MAPS if m[2] == fd0]
    assert mine() == []
    a = s.create('a', (100, 1000))
    b = s.create('b', 5000, np.complex128)
    assert len(mine()) == 2 and len(scratch._MAPS) == n0 + 2 and scratch.is_mapped(a)
    assert sorted(m[1] - m[0] for m in mine()) == [80000, 800000]
    v = np.asarray(a)[10:20]                    # a plain ndarray view
    fd, off = scratch.file_range(a)
    assert scratch.file_range(v) == (fd, off + 10 * 8000)
    assert scratch.file_range(np.ones(10)) is None
    d = scratch.io_desc(a, np.ones(10))
    assert d.dtype == np.int64 and list(d) == [1, -1, 0]
    assert list(scratch.io_desc(None, b)) == [0, fd, scratch.file_range(b)[1]]
    assert scratch.io_desc(np.ones(3), None) is None
    # an entry goes with its mapping (the address range may be reused) ...
    del a, v
    gc.collect()
    assert [m[1] - m[0] for m in mine()] == [80000]
    # ... or with its file
    s.close()
    assert mine() == [] and len(scratch._MAPS) <= n0
    b[:] = 2                                    # the mapping itself stays valid
    scratch.written(b)
    scratch.unmap(b)
    assert abs(b).sum() == 10000


@pytest.mark.parametrize('mode', [1, 0])
def test_hints_keep_the_data(mode, monkeypatch):
    monkeypatch.setattr(scratch, 'IO_MODE', mode)
    s = Scratch(prefix='zccsdt_test_')
    a = s.create('a', (300, 1000))
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(a.shape)
    a[:] = ref
    scratch.written(a, 1000, 250000)
    scratch.unmap(a, 1000, 250000)              # pages leave the page tables, not the file
    scratch.advise(a, 'willneed', 0, 5000)
    np.testing.assert_array_equal(a, ref)
    a[7] += 1.0                                 # written again after the unmapping
    scratch.written(a)
    scratch.unmap(a)
    ref[7] += 1.0
    np.testing.assert_array_equal(a, ref)
    # arrays in ordinary memory are left alone (unmapping them would lose the data)
    x = np.ones(10 ** 6)
    scratch.written(x)
    scratch.unmap(x)
    assert x.sum() == 10 ** 6
    assert scratch.io_desc(a, a) is None if mode == 0 else scratch.io_desc(a, a)[0] == 1
    assert scratch.hints() == bool(mode)
    scratch.hints(False)                        # enough memory: nothing is said
    assert scratch.io_desc(a, a) is None and not scratch.hints()
    scratch.unmap(a)
    np.testing.assert_array_equal(a, ref)
    s.close()


def test_mapped_blocks_spans():
    from zccsdt.sa.t3blocks import _spans, mapped_blocks
    assert _spans([]) == [] and _spans([-1, -1]) == []
    assert _spans([3, 1, 2, -1, 7, 8, 3]) == [(1, 4), (7, 9)]
    s = Scratch(prefix='zccsdt_test_')
    a = s.create('a', (6, 4, 4, 4))
    a[:] = np.arange(a.size).reshape(a.shape)
    mapped_blocks(a, [0, 1, 4], need=True)
    mapped_blocks(a, [0, 1, 4], wrote=True, drop=True)
    mapped_blocks(a, wrote=True, drop=True)
    mapped_blocks(np.zeros((2, 3, 3, 3)), [0], need=True, wrote=True, drop=True)
    np.testing.assert_array_equal(a.ravel(), np.arange(a.size))
    s.close()
