import numpy as np
import pytest
from pyscf.lib import diis as pdiis

from zccsdt.cc.diis import DIIS


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_matches_pyscf(dtype):
    rng = np.random.default_rng(0)
    ref = pdiis.DIIS(incore=True)
    ref.space = 4
    ours = DIIS(space=4)
    x = rng.standard_normal(1000).astype(dtype)
    for it in range(9):
        x = 0.9 * x + 0.01 * rng.standard_normal(1000)
        if dtype == np.complex128:
            x = x + 0.01j * rng.standard_normal(1000)
        a = ref.update(x.copy())
        b = ours.update([x[:300].copy(), x[300:].copy()])
        b = np.concatenate(b)
        np.testing.assert_allclose(b, a, rtol=1e-7, atol=1e-9)
        x = b


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_small_errors_are_extrapolated(dtype):
    '''Error vectors far below 1e-7 (overlaps below 1e-14, the absolute
    threshold of pyscf.lib.diis for dependent vectors, with which the
    extrapolation turns into the mean of the stored vectors): a linear map
    with five eigenvalues is solved by a subspace of six vectors.'''
    rng = np.random.default_rng(1)
    n = 300
    q = np.linalg.qr(rng.standard_normal((n, n)))[0].astype(dtype)
    if dtype == np.complex128:
        q = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))[0]
    A = (q * rng.choice([0.9, 0.5, -0.3, 0.7, -0.8], n)) @ q.conj().T
    b = rng.standard_normal(n).astype(dtype)
    exact = np.linalg.solve(np.eye(n) - A, b)
    x = exact + 1e-8 * rng.standard_normal(n)
    plain = x.copy()
    d = DIIS(space=6)
    for it in range(9):
        x = np.concatenate(d.update([(A @ x + b)[:100], (A @ x + b)[100:]]))
        plain = A @ plain + b
    assert abs(plain - exact).max() > 1e-10             # the iteration alone: 0.9 per step
    assert abs(x - exact).max() < 1e-13


def _run(d, dtype, nit, api, seed=3):
    '''nit DIIS steps on a two-part vector (sizes 50 and 700); returns the
    extrapolated vectors.  api: 'classic' (update only), 'errors' (explicit
    error vectors) or 'buffers' (vector and error built in the storage).'''
    rng = np.random.default_rng(seed)
    n = (50, 700)
    t = [rng.standard_normal(k).astype(dtype) for k in n]
    out = []
    for it in range(nit):
        q = [0.1 * rng.standard_normal(k).astype(dtype) - 0.1 * ti for k, ti in zip(n, t)]
        if dtype == np.complex128:
            q = [qi + 0.05j * rng.standard_normal(qi.size) for qi in q]
        if api == 'classic':
            t = d.update([ti + qi for ti, qi in zip(t, q)])
        elif api == 'errors':
            t = d.update([ti + qi for ti, qi in zip(t, q)], errors=[2 * qi for qi in q])
        else:
            xs, es = d.buffers(t)
            for x, e, ti, qi in zip(xs, es, t, q):
                x[:] = ti + qi
                e[:] = 2 * qi
            big = np.empty(n[1], dtype=dtype)
            t = d.update(xs, errors=es, out=[None, big])
            assert t[1] is big or np.shares_memory(t[1], big)
        t = [np.array(ti) for ti in t]
        out.append(np.concatenate(t))
    return out


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('api', ['classic', 'errors', 'buffers'])
def test_file_backed_equals_incore(monkeypatch, tmp_path, dtype, api):
    '''Large parts in the HDF5 scratch file give the same vectors as in memory,
    through and beyond a full subspace; the temporary file leaves nothing.'''
    from zccsdt.cc import diis
    monkeypatch.setattr(diis, 'INCORE_SIZE', 100)
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    ref = _run(DIIS(space=4, incore=True), dtype, 11, api)
    d = DIIS(space=4)
    got = _run(d, dtype, 11, api)
    assert d.on_disk()
    assert all(isinstance(x[1], np.memmap) and not isinstance(x[0], np.memmap) for x in d.xs)
    assert all(isinstance(e[1], np.memmap) for e in d.es)
    assert list(tmp_path.iterdir()) == []              # unlinked right after opening
    for a, b in zip(ref, got):
        assert np.array_equal(a, b)
    d.close()
    assert not d.on_disk()


def test_named_diis_file_is_hdf5(monkeypatch, tmp_path):
    '''A named DIIS file is kept and holds the stored vectors as datasets.'''
    import h5py
    from zccsdt.cc import diis
    monkeypatch.setattr(diis, 'INCORE_SIZE', 100)
    name = str(tmp_path / 'diis.h5')
    d = DIIS(space=3, filename=name)
    _run(d, np.float64, 7, 'buffers')
    stored = [(np.array(x[1]), np.array(e[1])) for x, e in zip(d.xs, d.es)]
    d.close()
    with h5py.File(name, 'r') as f:
        keys = sorted(f.keys())
        assert keys == sorted('%s%d.1' % (c, k) for c in 'xe' for k in range(3))
        data = [f[k][()] for k in keys]
    for x, e in stored:
        assert any(np.array_equal(x, v) for v in data)
        assert any(np.array_equal(e, v) for v in data)


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_mdot(dtype):
    from zccsdt.cc.diis import _dot, _mdot
    rng = np.random.default_rng(1)
    n = 100003
    mk = lambda: (rng.standard_normal(n) + (1j * rng.standard_normal(n) if dtype == np.complex128 else 0)).astype(dtype)  # noqa: E731
    a, xs = mk(), [mk() for _ in range(5)]
    got = _mdot(a, xs)
    ref = np.array([np.vdot(a, x) for x in xs])
    assert abs(got - ref).max() < 1e-9 * abs(ref).max()
    assert abs(got - np.array([_dot(a, x) for x in xs])).max() < 1e-9 * abs(ref).max()


@pytest.mark.parametrize('kind', ['real', 'complex'])
def test_spin_orbital_solver_diis_on_disk(monkeypatch, tmp_path, kind):
    '''Packed-T3 CCSDT (real and complex amplitudes) with the T3 DIIS vectors
    in the scratch file converges like the run in memory (the storage itself
    is compared exactly in test_file_backed_equals_incore).'''
    from pyscf import gto, scf
    from zccsdt import eris as E
    from zccsdt.cc import diis, solver
    mol = gto.M(atom='F 0 0 0; H 0 0 0.917', basis='6-31g', verbose=0)
    ghf = scf.addons.convert_to_ghf(scf.RHF(mol).run(conv_tol=1e-12))
    C = E.rotate_occ_vir(ghf.mo_coeff, ghf.mo_occ, seed=4, frozen=4) if kind == 'complex' else None
    eris = E.from_scf(ghf, mo_coeff=C, frozen=4)
    ref = solver.kernel(eris, 'ccsdt', solver.Options(conv_tol=1e-9, conv_tol_normt=1e-6,
                                                      incore_complete=True))
    monkeypatch.setattr(diis, 'INCORE_SIZE', 1000)
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    r = solver.kernel(eris, 'ccsdt', solver.Options(conv_tol=1e-9, conv_tol_normt=1e-6))
    # two runs differ by threaded summation order, at the level of conv_tol
    assert ref.converged and r.converged and abs(r.ncycle - ref.ncycle) <= 1
    assert abs(r.e_tot - ref.e_tot) < 1e-8
    assert r.t3.size >= 1000 and list(tmp_path.iterdir()) == []


def test_scratch_allocation_error_is_explicit(monkeypatch, tmp_path):
    '''A scratch file that cannot be allocated fails with a message naming the
    file and the options, when the vector is created.'''
    import errno
    import os
    from zccsdt.cc import diis
    monkeypatch.setattr(diis, 'INCORE_SIZE', 100)
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))

    def full(fd, offset, size):
        raise OSError(errno.ENOSPC, 'No space left on device')
    monkeypatch.setattr(os, 'posix_fallocate', full)
    d = DIIS(space=3)
    with pytest.raises(OSError, match='ZCCSDT_TMPDIR') as info:
        d.buffers([np.zeros(10), np.zeros(500)])
    assert info.value.errno == errno.ENOSPC
    d.close()
