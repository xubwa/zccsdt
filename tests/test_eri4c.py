'''zccsdt.eri4c and zccsdt.hblocks: the four-component integrals in label
blocks against the dense route of socutils.tools.fcidump_rel
(x2c_active_eri_4c: eight pyscf.ao2mo.general calls), for spinors of Li2 on
the z axis with random coefficients that respect m_j and the parity; the
plans with small budgets (slabs, H in a scratch file), the block file, its
J/K, rotation and Hartree-Fock.'''
import numpy as np
import pytest
from pyscf import ao2mo, gto, lib

from zccsdt import eri4c, hblocks


def _system(seed=3):
    mol = gto.M(atom='Li 0 0 0; Li 0 0 2.6', basis='6-31g', verbose=0)
    m2 = eri4c.ao_m2(mol)
    rng = np.random.default_rng(seed)
    N = mol.nao_2c()
    # inversion through the midpoint (as bench/dea/o2_scf.parities)
    sign = []
    for b in range(mol.nbas):
        l, kappa = mol.bas_angular(b), mol.bas_kappa(b)
        dim = (4 * l + 2) if kappa == 0 else (2 * l if kappa > 0 else 2 * l + 2)
        sign += [(-1) ** l] * (dim * mol.bas_nctr(b))
    sign = np.array(sign, dtype=float)
    perm = np.concatenate((np.arange(N // 2, N), np.arange(N // 2)))

    def inv(c):
        return (sign[:, None] * c)[perm]
    cols, labs, pars = [], [], []
    for x in sorted(set(m2.tolist())):
        a = np.flatnonzero(m2 == x)
        for k in range(min(3, len(a) // 2)):
            for p in (0, 1):
                c = np.zeros(N)
                c[a] = rng.standard_normal(len(a))
                c = c + (1 - 2 * p) * inv(c[:, None])[:, 0]
                cols.append(c)
                labs.append(x)
                pars.append(p)
    order = rng.permutation(len(cols))
    cl = 0.3 * np.array(cols).T[:, order]
    cs = np.zeros_like(cl)
    for j in range(cl.shape[1]):
        a = np.flatnonzero(m2 == labs[order[j]])
        c = np.zeros(N)
        c[a] = rng.standard_normal(len(a))
        cs[:, j] = 0.3 * c + 0.3 * (1 - 2 * pars[order[j]]) * inv(c[:, None])[:, 0]
    return mol, cl, cs, np.array(labs)[order], np.array(pars)[order]


def _dense(mol, cl, cs, breit=True):
    '''x2c_active_eri_4c of socutils (with_full_coulomb, with_gaunt, with_breit) on these spinors.'''
    ms = cs * (0.5 / lib.param.LIGHT_SPEED)
    n = cl.shape[1]
    out = ao2mo.general(mol, (cl, cl, cl, cl), intor='int2e_spinor')
    out = out + ao2mo.general(mol, (cl, cl, ms, ms), intor='int2e_spsp2_spinor')
    out += ao2mo.general(mol, (ms, ms, cl, cl), intor='int2e_spsp1_spinor')
    out += ao2mo.general(mol, (ms, ms, ms, ms), intor='int2e_spsp1spsp2_spinor')
    p = 'int2e_breit_' if breit else 'int2e_'
    f = 1.0 if breit else -1.0
    for mo, intor in zip([[cl, ms, ms, cl], [cl, ms, cl, ms], [ms, cl, ms, cl], [ms, cl, cl, ms]],
                         ['ssp1sps2_spinor', 'ssp1ssp2_spinor', 'sps1sps2_spinor', 'sps1ssp2_spinor']):
        out = out + f * ao2mo.general(mol, mo, intor=p + intor, aosym='s1', comp=1)
    return np.asarray(out).reshape(n, n, n, n)


@pytest.fixture(scope='module')
def system():
    mol, cl, cs, lab, par = _system()
    return mol, cl, cs, lab, par, _dense(mol, cl, cs)


def _blockfile(tmp_path, mol, cl, cs, lab, par, name='a.h5', **kw):
    bf = None
    for buf, layout in eri4c.make4c(mol, cl, cs, lab, par, verbose=False, **kw):
        if bf is None:
            groups, members = hblocks.layout_members(layout)
            n = len(lab)
            bf = hblocks.BlockFile.create(str(tmp_path / name), groups, members, 2, 0.0, np.zeros((n, n)), buf.dtype)
        forb = hblocks.write_label_buffer(bf, buf, layout)
        assert forb < 1e-10
    return bf


def test_labels_of_the_dense_integrals(system):
    mol, cl, cs, lab, par, ref = system
    assert abs(ref.imag).max() < 1e-12
    def axes(x):
        return x[:, None, None, None], x[None, :, None, None], x[None, None, :, None], x[None, None, None, :]
    a, b, c, d = axes(lab)
    ok = a - b + c - d == 0
    a, b, c, d = axes(par)
    ok &= (a + b + c + d) % 2 == 0
    assert abs(ref[~ok]).max() < 1e-10 * abs(ref).max()


@pytest.mark.parametrize('direct', [False, True])
def test_blocks_against_the_dense_route(system, tmp_path, direct):
    mol, cl, cs, lab, par, ref = system
    bf = _blockfile(tmp_path, mol, cl, cs, lab, par, direct=direct)
    x = bf.to_dense()
    assert abs(x - ref.real).max() < 1e-11 * abs(ref).max()
    bf.close()


def test_complex_arithmetic(system, tmp_path):
    mol, cl, cs, lab, par, ref = system
    bf = _blockfile(tmp_path, mol, cl, cs, lab, par, real=False)
    assert bf.dtype.kind == 'c'
    assert abs(bf.to_dense() - ref).max() < 1e-11 * abs(ref).max()
    bf.close()


def test_small_budgets(system, tmp_path):
    '''H in a scratch file, then slabs of label blocks: the same integrals.'''
    mol, cl, cs, lab, par, ref = system
    layout = eri4c.Layout(lab, par, eri4c.ao_m2(mol))
    full = eri4c.plan(layout, mol)
    assert len(full.slabs) == 1 and not full.h_disk
    b1 = full.buf_bytes + full.h_bytes // 2 + full.small + full.min_chunk
    p1 = eri4c.plan(layout, mol, budget=b1)
    assert p1.h_disk and len(p1.slabs) == 1 and p1.peak <= b1
    b2 = full.buf_bytes // 2 + full.small + full.min_chunk
    p2 = eri4c.plan(layout, mol, budget=b2)
    assert len(p2.slabs) > 1 and p2.peak <= b2
    import tracemalloc
    for b in (full, p1, p2):
        tracemalloc.start()
        bufs = []
        for buf, layout in eri4c.make4c(mol, cl, cs, lab, par, verbose=False, budget=b.budget):
            bufs.append((buf, layout))
            peak = tracemalloc.get_traced_memory()[1]
            # (the plan counts the label buffer of one slab; the earlier ones are kept by this test)
            assert peak - sum(x[0].buf.nbytes for x in bufs[:-1]) <= b.peak
            tracemalloc.reset_peak()
        tracemalloc.stop()
        groups, members = hblocks.layout_members(bufs[0][1])
        n = len(lab)
        bf = hblocks.BlockFile.create(str(tmp_path / ('b%d.h5' % len(b.slabs))), groups, members, 2, 0.0,
                                      np.zeros((n, n)))
        for buf, layout in bufs:
            hblocks.write_label_buffer(bf, buf, layout)
        assert abs(bf.to_dense() - ref.real).max() < 1e-11 * abs(ref).max()
        bf.close()
    with pytest.raises(MemoryError):
        eri4c.plan(layout, mol, budget=1000)


def test_kshells_add_up(system, tmp_path):
    mol, cl, cs, lab, par, ref = system
    parts = []
    for ks in (range(0, 5), range(5, mol.nbas)):
        bufs = list(eri4c.make4c(mol, cl, cs, lab, par, kshells=ks, verbose=False))
        assert len(bufs) == 1
        parts.append(bufs[0])
    p = str(tmp_path / 'part.npy')
    parts[1][0].save(p)
    parts[0][0].add_file(p)
    buf, layout = parts[0]
    groups, members = hblocks.layout_members(layout)
    n = len(lab)
    bf = hblocks.BlockFile.create(str(tmp_path / 'k.h5'), groups, members, 2, 0.0, np.zeros((n, n)))
    hblocks.write_label_buffer(bf, buf, layout)
    assert abs(bf.to_dense() - ref.real).max() < 1e-11 * abs(ref).max()
    bf.close()


def test_jk_rotation_and_hf(system, tmp_path):
    mol, cl, cs, lab, par, ref = system
    eri = ref.real
    n = len(lab)
    no = 4
    rng = np.random.default_rng(1)
    h1 = 0.05 * rng.standard_normal((n, n))
    h1 = h1 + h1.T
    # (h1 respects the labels and the parities)
    h1 *= (lab[:, None] == lab[None, :]) & (par[:, None] == par[None, :])
    h1 += np.diag(np.linspace(-5, 5, n))
    bf = None
    for buf, layout in eri4c.make4c(mol, cl, cs, lab, par, verbose=False):
        groups, members = hblocks.layout_members(layout)
        bf = hblocks.BlockFile.create(str(tmp_path / 'h.h5'), groups, members, no, 1.5, h1, buf.dtype)
        hblocks.write_label_buffer(bf, buf, layout)
    # Fock matrix and reference energy of the first no spinors
    o = slice(0, no)
    f = h1 + np.einsum('pqii->pq', eri[:, :, o, o]) - np.einsum('piiq->pq', eri[:, o, o, :])
    assert abs(bf.fock() - f).max() < 1e-11
    e = 1.5 + np.trace(h1[o, o]) + 0.5 * (np.einsum('iijj->', eri[o, o, o, o]) - np.einsum('ijji->', eri[o, o, o, o]))
    assert abs(bf.reference_energy() - e) < 1e-10
    # Hartree-Fock: a lower energy than the start, unitaries of the groups, a diagonal Fock matrix
    U, members, eps, e_hf, e0 = hblocks.hartree_fock(bf, h1, no, verbose=False)
    assert abs(e0 - e) < 1e-10 and e_hf <= e0 + 1e-12
    Uf = np.zeros((n, n))
    for g, m in enumerate(bf.members):
        Uf[np.ix_(m, members[g])] = U[g]
    assert abs(Uf.T @ Uf - np.eye(n)).max() < 1e-12
    h1n = Uf.T @ h1 @ Uf
    out = hblocks.rotate(bf, U, members, str(tmp_path / 'r.h5'), h1n)
    ref_rot = np.einsum('ap,bq,cr,ds,abcd->pqrs', Uf, Uf, Uf, Uf, eri, optimize=True)
    assert abs(out.to_dense() - ref_rot).max() < 1e-11
    fn = out.fock(h1n)
    assert abs(fn - np.diag(fn.diagonal())).max() < 1e-8
    assert abs(fn.diagonal() - eps).max() < 1e-8
    assert abs(out.reference_energy(h1n) - e_hf) < 1e-10
    sub = hblocks.subset(out, 9, str(tmp_path / 's.h5'))
    assert abs(sub.to_dense() - ref_rot[:9, :9, :9, :9]).max() < 1e-11
    for x in (bf, out, sub):
        x.close()
