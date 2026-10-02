'''Memory budget (zccsdt.memory) and scratch arrays (zccsdt.scratch).'''

import numpy as np

from zccsdt import memory, scratch


def _cgroup_tree(tmp_path, v2=False):
    '''A fake cgroup hierarchy: job limit 8 GiB (2 GiB used, 1 GiB of it file
    pages), step limit 3 GiB (1 GiB used, 512 MiB file pages), task unlimited.'''
    G = 1 << 30
    root = tmp_path / 'cg'
    top = root if v2 else root / 'memory'
    names = ('memory.max', 'memory.current') if v2 else \
        ('memory.limit_in_bytes', 'memory.usage_in_bytes')
    keys = ('inactive_file', 'active_file') if v2 else \
        ('total_inactive_file', 'total_active_file')
    spec = {'slurm/job': (8 * G, 2 * G, G), 'slurm/job/step': (3 * G, G, G // 2),
            'slurm/job/step/task': (None, G, G // 2), 'slurm': (None, 5 * G, G), '': (None, 9 * G, G)}
    for rel, (limit, usage, cache) in spec.items():
        d = top / rel
        d.mkdir(parents=True, exist_ok=True)
        unlimited = 'max' if v2 else str(9223372036854771712)
        (d / names[0]).write_text('%s\n' % (unlimited if limit is None else limit))
        (d / names[1]).write_text('%d\n' % usage)
        (d / 'memory.stat').write_text('%s %d\n%s %d\nrss 5\n' % (keys[0], cache // 2, keys[1],
                                                                  cache - cache // 2))
    f = tmp_path / 'cgroup'
    f.write_text('0::/slurm/job/step/task\n' if v2 else
                 '12:devices:/x\n9:memory:/slurm/job/step/task\n5:cpu,cpuacct:/y\n')
    return str(f), str(root)


def test_cgroup_limit_is_the_tightest_ancestor(tmp_path):
    G = 1 << 30
    for v2 in (False, True):
        f, root = _cgroup_tree(tmp_path / ('v2' if v2 else 'v1'), v2)
        # step: 3 - 1 + 0.5 = 2.5 GiB;  job: 8 - 2 + 1 = 7 GiB
        assert memory.cgroup_available(f, root) == 5 * G // 2
    assert memory.cgroup_available(str(tmp_path / 'missing'), str(tmp_path)) is None


def test_budget(monkeypatch):
    monkeypatch.setattr(memory, 'available', lambda: 10_000_000_000)
    monkeypatch.setattr(memory, 'resident', lambda: 1_000_000_000)
    assert memory.budget() == 8_000_000_000
    assert memory.budget(max_memory=4000) == 3_000_000_000
    assert memory.budget(max_memory=500) == 0
    monkeypatch.setattr(memory, 'available', lambda: None)
    assert memory.budget() is None
    assert memory.available is not None and isinstance(memory.node_available(), (int, type(None)))


def test_scratch_arrays(monkeypatch, tmp_path):
    '''Mapped datasets: shaped, zero on creation, written through to the file,
    advise() harmless on mapped and ordinary arrays.'''
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    s = scratch.Scratch(prefix='zccsdt_test_')
    a = s.create('a', (3, 4, 5))
    b = s.create('b', 7, dtype=np.complex128)
    assert a.shape == (3, 4, 5) and b.shape == (7,) and b.dtype == np.complex128
    assert not a.any() and not b.any()
    a[...] = np.arange(60.0).reshape(3, 4, 5)
    for adv in ('willneed', 'sequential', 'random', 'dontneed'):
        scratch.advise(a, adv)
        scratch.advise(a[1], adv, 2, 9)
        scratch.advise(np.zeros(10), adv)
    assert a[2, 3, 4] == 59.0 and a.sum() == 59 * 60 / 2
    assert list(tmp_path.iterdir()) == []            # the temporary file is unlinked
    s.close()
    name = str(tmp_path / 'keep.h5')
    s = scratch.Scratch(name)
    x = s.create('x', (2, 3))
    x[...] = 1.5
    x.flush()
    s.close()
    import h5py
    with h5py.File(name, 'r') as f:
        assert f['x'].shape == (2, 3) and (f['x'][...] == 1.5).all()
