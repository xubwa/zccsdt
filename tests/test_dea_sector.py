'''EOM-DEA on the sector itself (zccsdt.cc.dea_sector, zccsdt.cc.dea_ref;
docs/eom.md, section 11.9)

- the terms that survive on the rows with both ghosts (dea_ref, derived
  from the table of the amplitude equations) against the ghost route, real
  and complex, T of CCSDT and of CCSDTQ;
- the sigma vector of the sector in label blocks, element by element,
  against those terms and against the ghost route in label blocks;
- the roots of the two routes, through the front end;
- the memory of a sigma vector against its plan (tracemalloc).'''
import tracemalloc

import numpy as np
import pytest
from pyscf import gto, scf, x2c

from test_ccsdtq_ref import _spin_mixed
from test_dea import H4, TIGHT, _ghf
from zccsdt import eris as E
from zccsdt.cc import ccsdtq, ccsdtq_ref as R, ccsdtq_terms as Q, dea, dea_ref, dea_sector as D, solver


@pytest.mark.parametrize('cplx', [False, True])
@pytest.mark.parametrize('level', ['ccsd', 'ccsdt', 'ccsdtq'])
def test_terms_against_the_ghost_route(cplx, level):
    '''H4 / STO-3G: the terms of dea_ref on dense sector vectors against the
    sigma vector of the ghost route (dea.Problem: the residual at T + R on
    the padded spin orbitals).'''
    mf = scf.addons.convert_to_ghf(_ghf(H4))
    eris = E.from_scf(mf, mo_coeff=_spin_mixed(mf)[0] if cplx else mf.mo_coeff)
    no, nv = 4, 4
    t4 = None
    if level == 'ccsd':
        cc = solver.kernel(eris, 'ccsd', TIGHT)
        t3 = None
    elif level == 'ccsdt':
        cc = solver.kernel(eris, 'ccsdt', TIGHT)
        t3 = cc.t3
    else:
        cc = ccsdtq.kernel(eris, ccsdtq.Options(conv_tol=1e-13, conv_tol_normt=1e-11))
        t3, t4 = cc.t3, cc.t4
    pb = dea.Problem(eris, cc.t1, cc.t2, t3, t4, rank=4)
    fock, g, _, _ = R.integrals(eris)
    dt = pb.dtype
    amps = [np.asarray(cc.t1, dt), np.asarray(cc.t2, dt),
            None if t3 is None else Q.expand(np.asarray(t3, dt).reshape(4, 4), no, nv, 3),
            None if t4 is None else Q.expand(np.asarray(t4, dt).reshape(1, 1), no, nv, 4)]
    rng = np.random.default_rng(1)
    x = [rng.standard_normal(s) + (1j * rng.standard_normal(s) if cplx else 0) for s in pb.sizes]
    y = pb.sigma(x, pb.new())
    rd = [dea_ref.expand(p, no, nv, k + 2) for k, p in enumerate(x)]
    ref = dea_ref.sigma(rd, amps, fock if cplx else fock.real, g if cplx else g.real, no)
    scale = max(abs(p).max() for p in y)
    for k in range(3):
        assert abs(dea_ref.unique(ref[k], k + 2) - y[k]).max() < 1e-11 * scale


def _bh(sym=True):
    import zccsdt
    mf = x2c.UHF(gto.M(atom='B 0 0 0; H 0 0 1.2324', basis='6-31g', charge=2, verbose=0)).run(conv_tol=1e-12)
    c = zccsdt.CCSDT(mf)
    c.symmetry, c.verbose = sym, 0
    c.conv_tol, c.conv_tol_normt = 1e-12, 1e-10
    c.kernel()
    return c


@pytest.fixture(scope='module')
def bh():
    return _bh()


@pytest.fixture(scope='module')
def bh_t4(bh):
    '''CCSDTQ on the integrals of bh, in the blocks of the labels.'''
    from zccsdt.cc import ccsdtq_sym
    r = ccsdtq_sym.kernel(bh._eris, ccsdtq.Options(conv_tol=1e-12, conv_tol_normt=1e-10), t1=bh.t1, t2=bh.t2, t3=bh.t3)
    assert r.converged
    return r


@pytest.mark.parametrize('shift', [0, 2])
def test_sector_with_t4(bh, bh_t4, shift):
    '''The T of CCSDTQ (two more terms of the 4p2h row): the sigma vector of
    the sector against the ghost route, element by element.'''
    eris, r = bh._eris, bh_t4
    sp = D.ProblemSector(eris, r.t1, r.t2, r.t3, rank=4, target=shift, t4=r.t4)
    gp = dea.ProblemSym(eris, r.t1, r.t2, r.t3, r.t4, rank=4, target=shift)
    _compare(sp, gp, eris)


def _compare(sp, gp, eris):
    no = eris.nocc
    nv = np.asarray(eris.fock).shape[0] - no
    rank = len(sp.sizes) + 1
    dt = sp.dtype
    rng = np.random.default_rng(7)
    x = [rng.standard_normal(n) + (1j * rng.standard_normal(n) if dt.kind == 'c' else 0) for n in sp.sizes]
    y = sp.sigma(x, sp.new())
    dpos = sp.dense_positions()
    full = []
    for k in range(rank - 1):
        row = k + 2
        n = (1 if row == 2 else (no if row == 3 else no * (no - 1) // 2)) * len(Q.tuples(nv, row))
        f = np.zeros(n, dtype=dt)
        f[dpos[k]] = x[k]
        full.append(f)
    xg = gp.new()
    xg[0][:] = full[0]
    for k in range(1, rank - 1):
        xg[k][:] = full[k][gp.dpos[k]]
    yg = gp.sigma(xg, gp.new())
    scale = max(abs(p).max() for p in y)
    assert scale > 0.1
    for k in range(rank - 1):
        got = np.zeros_like(full[k])
        got[dpos[k]] = y[k]
        if k == 0:
            ref = yg[0]
        else:
            ref = np.zeros_like(full[k])
            ref[gp.dpos[k]] = yg[k]
        assert abs(got - ref).max() < 1e-11 * scale


@pytest.mark.parametrize('shift', [0, 2, -2])
def test_sector_against_reference_and_ghosts(bh, shift):
    '''BH / 6-31G from BH(2+), X2C spinors in the blocks of 2 m_j (4 + 18):
    the sigma vector of the sector, every element, against the ghost route
    in label blocks (dea.ProblemSym), for random vectors; R to 3p1h and to
    4p2h.  (Against the terms of dea_ref on dense arrays: bench/dea, O2.)'''
    eris = bh._eris
    no = eris.nocc
    nv = np.asarray(eris.fock).shape[0] - no
    for rank in (3, 4):
        sp = D.ProblemSector(eris, bh.t1, bh.t2, bh.t3, rank=rank, target=shift)
        gp = dea.ProblemSym(eris, bh.t1, bh.t2, bh.t3, rank=rank, target=shift)
        dt = sp.dtype
        rng = np.random.default_rng(7)
        x = [rng.standard_normal(n) + (1j * rng.standard_normal(n) if dt.kind == 'c' else 0) for n in sp.sizes]
        y = sp.sigma(x, sp.new())
        # the same vector on the ghost route: by the places in the dense arrays
        dpos = sp.dense_positions()
        full = []
        for k in range(rank - 1):
            row = k + 2
            n = (1 if row == 2 else (no if row == 3 else no * (no - 1) // 2)) * len(Q.tuples(nv, row))
            f = np.zeros(n, dtype=dt)
            f[dpos[k]] = x[k]
            full.append(f)
        xg = gp.new()
        xg[0][:] = full[0]
        for k in range(1, rank - 1):
            xg[k][:] = full[k][gp.dpos[k]]
        yg = gp.sigma(xg, gp.new())
        scale = max(abs(p).max() for p in y)
        assert scale > 0.1
        for k in range(rank - 1):
            got = np.zeros_like(full[k])
            got[dpos[k]] = y[k]
            if k == 0:
                ref = yg[0]
            else:
                ref = np.zeros_like(full[k])
                ref[gp.dpos[k]] = yg[k]
            assert abs(got - ref).max() < 1e-11 * scale


def test_roots_of_the_two_routes(bh):
    '''The front end: eom_dea() on the sector and through the ghosts give the
    same roots (R to 3p1h and to 4p2h, 2 m_j of the pair 0 and 2).'''
    for rank in (3, 4):
        for shift in (0, 2):
            e1, r1 = bh.eom_dea(nroots=1, shift=shift, rank=rank, conv_tol=1e-9, nguess=6, backend='sector')
            e2, r2 = bh.eom_dea(nroots=1, shift=shift, rank=rank, conv_tol=1e-9, nguess=6, backend='ghost')
            assert r1.converged.all() and r2.converged.all()
            assert abs(e1 - e2).max() < 1e-9


def test_memory_of_a_sigma_vector(bh):
    '''tracemalloc: what one sigma vector allocates stays below the plan
    (memory(): per sigma vector), with batches small enough that the passes
    over r4 take several.'''
    eris = bh._eris
    sp = D.ProblemSector(eris, bh.t1, bh.t2, bh.t3, rank=4, target=0, batch_bytes=1 << 14)
    x = [np.random.default_rng(1).standard_normal(n) for n in sp.sizes]
    y = sp.new()
    sp.sigma(x, y)
    tracemalloc.start()
    tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    sp.sigma(x, y)
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    plan = sp.mem[1]
    assert peak <= plan, (peak, plan)


def test_small_budget(bh):
    '''A budget that the sector does not fit in: Xv in a scratch file, small
    batches; the same sigma vector.'''
    eris = bh._eris
    a = D.ProblemSector(eris, bh.t1, bh.t2, bh.t3, rank=4, target=0)
    b = D.ProblemSector(eris, bh.t1, bh.t2, bh.t3, rank=4, target=0, mem_mb=0.01, batch_bytes=1 << 12)
    assert b.xv_disk and not a.xv_disk
    x = [np.random.default_rng(2).standard_normal(n) for n in a.sizes]
    ya, yb = a.sigma(x, a.new()), b.sigma(x, b.new())
    assert max(abs(p - q).max() for p, q in zip(ya, yb)) < 1e-12
