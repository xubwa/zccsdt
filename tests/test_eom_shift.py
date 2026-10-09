'''EOM-CCSDT states with a label shift in the blocks of the symmetry labels
(zccsdt.cc.t3shift, eom_opt.sigma(shift=), eom.kernel(shift=),
GCCSDT.eom_ee(shift=); docs/eom.md, section 6): the excitation operator
changes the sum of the labels (2 m_j) by a fixed amount

- the layout; the sigma vector of every shift against the packed block, real
  and with a phase on every spinor (complex integrals, the same labels);
- the roots of every shift are roots of the packed block (symmetry = False)
  to 1e-10, and their union is its low spectrum with multiplicities: LiH and
  HBr (frozen core) with X2C spinors; the states of +D and -D;
- the memory plan with a shift, and what is allocated against it.'''
import copy
import gc
import tracemalloc

import numpy as np
import pytest
from pyscf import gto, x2c

import zccsdt
from zccsdt.cc import ccsdt_opt, eom, eom_opt, solver, t3engine, t3pack, t3shift


def _cc(mf, sym, **kw):
    cc = zccsdt.CCSDT(mf, **kw)
    cc.symmetry, cc.verbose = sym, 0
    cc.conv_tol, cc.conv_tol_normt = 1e-12, 1e-10
    cc.kernel()
    return cc


@pytest.fixture(scope='module')
def lih():
    '''CCSDT of LiH / 6-31G with X2C spinors (4 + 18; labels -3 .. 3), in
    the blocks of the labels and on the packed block.'''
    mf = x2c.UHF(gto.M(atom='Li 0 0 0; H 0 0 1.5957', basis='6-31g', verbose=0)).run(conv_tol=1e-12)
    return {sym: _cc(mf, sym) for sym in (True, False)}


@pytest.fixture(scope='module')
def hbr():
    '''HBr / cc-pVDZ-DK, X2C, 28 spinors frozen, 8 + 16 active.'''
    mol = gto.M(atom='Br 0 0 0; H 0 0 1.4145', basis='cc-pvdz-dk', verbose=0)
    mf = x2c.UHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-8
    mf.kernel()
    frozen = list(range(28)) + list(range(36 + 16, mol.nao_2c()))
    return {sym: _cc(mf, sym, frozen=frozen) for sym in (True, False)}


def _p2(a, om, vm):
    return np.ascontiguousarray(a[om.pairs[:, 0][:, None], om.pairs[:, 1][:, None], vm.pairs[:, 0], vm.pairs[:, 1]])


def _lex(L, x):
    '''A flat array of the layout L on the packed block in lexicographic order.'''
    out = np.zeros((L.om.ntri, L.vm.ntri), dtype=x.dtype)
    out[np.ix_(ccsdt_opt._lex(L.om), ccsdt_opt._lex(L.vm))] = L.to_dense(x)
    return out


def _random_vector(om, vm, s, seed):
    rng = np.random.default_rng(seed)
    lo, lv = np.asarray(om.spins), np.asarray(vm.spins)
    r1 = rng.standard_normal((om.n, vm.n)) * (lv[None, :] - lo[:, None] == s)
    x = rng.standard_normal((om.n,) * 2 + (vm.n,) * 2)
    x = x - x.transpose(1, 0, 2, 3)
    x = x - x.transpose(0, 1, 3, 2)
    r2 = x * ((lv[None, None, :, None] + lv[None, None, None, :]) - (lo[:, None, None, None] + lo[None, :, None, None]) == s)
    return r1, r2, rng


def test_layout(lih):
    '''The blocks of all shifts are the packed block; shift 0 is the layout
    of T3; from_dense and to_dense.'''
    eb = lih[True]._eris
    eng = ccsdt_opt.engine_for(eb, np.float64)
    sh = t3shift.ShiftT3(eng)
    om, vm = eng.om, eng.vm
    assert sum(sh.size(s) for s in sh.shifts()) == om.ntri * vm.ntri == 3264
    assert sh.size(0) == eng.size == 868 and sh.size(1) == sh.size(-1) == 704 and sh.size(3 * om.ncls) == 0
    L0 = sh.layout(0)
    assert np.array_equal(L0.obase, eng.L.obase) and np.array_equal(L0.vcol, eng.L.vcol)
    rng = np.random.default_rng(0)
    seen = np.zeros((om.ntri, vm.ntri), int)
    for s in sh.shifts():
        L = sh.layout(s)
        x = rng.standard_normal(L.size)
        D = L.to_dense(x)
        assert np.array_equal(L.from_dense(D), x)
        grp = vm.tri_grp[None, :] - om.tri_grp[:, None]
        assert not D[grp != s].any() and (D[grp == s] != 0).all()
        seen += D != 0
    assert (seen == 1).all()


def _phased(eris, u):
    '''The integrals with the spinor p multiplied by the phase u[p]; the
    labels stay.'''
    no = eris.nocc
    uo, uv = u[:no], u[no:]
    ec = copy.copy(eris)
    for key in ('_buffers', '_vvvv_blocks', '_bare_Q', '_bare_vvvo'):       # (the caches of the real integrals)
        ec.__dict__.pop(key, None)
    ec.fock = np.asarray(eris.fock) * u.conj()[:, None] * u[None, :]
    for key in ('oooo', 'ooov', 'oovv', 'ovov', 'ovvo', 'ovvv'):
        x = np.asarray(getattr(eris, key)).astype(complex)
        for ax, c in enumerate(key):
            f = uo if c == 'o' else uv
            x = x * (f.conj() if ax < 2 else f).reshape([-1 if a == ax else 1 for a in range(4)])
        setattr(ec, key, x)
    a, b = np.triu_indices(len(uv), 1)
    ec._vvvv_pp = np.asarray(eris._vvvv_pp) * (uv[a] * uv[b]).conj()[:, None] * (uv[a] * uv[b])[None, :]
    return ec


def _phases(om, vm, u):
    '''The phases of r1 [i,a], of r2 [i,j,a,b] and of the occupied and the
    virtual triples in the order of the maps.'''
    uo, uv = u[:om.n], u[om.n:].conj()
    return (uo[:, None] * uv[None, :],
            uo[:, None, None, None] * uo[None, :, None, None] * uv[None, None, :, None] * uv[None, None, None, :],
            uo[om.triples].prod(axis=1), uv[vm.triples].prod(axis=1))


def _phase3(L, x, po, pv):
    out = np.array(x, dtype=complex)
    for where, shape, I, A in L.blocks():
        out[where] = (x[where].reshape(shape) * po[I][:, None] * pv[A][None, :]).ravel()
    return out


def test_sigma_with_a_label_shift(lih):
    '''LiH: the sigma vector of a random vector of every shift (-4 .. 4
    classes; 14 to 868 of the 3264 triples) in the blocks is the one on the
    packed block; shift 0 by the shifted passes is the engine's; the
    singles-doubles block; and with a phase on every spinor the sigma
    vector has the phases.'''
    cc = lih[True]
    eb = cc._eris
    ed, t3d = cc._dense_view()
    omb, vmb = t3pack.maps_for(eb)
    omd, vmd = t3pack.maps_for(ed)
    gb = eom_opt.prepare(eb, cc.t1, cc.t2, cc.t3)
    gd = eom_opt.prepare(ed, cc.t1, cc.t2, t3d)
    sh = gb.shifter()
    u = np.exp(2j * np.pi * np.random.default_rng(5).random(omb.n + vmb.n))
    f1, f2, po, pv = _phases(omb, vmb, u)
    gc_ = eom_opt.prepare(_phased(eb, u), cc.t1 * f1, cc.t2 * f2, _phase3(sh.layout(0), np.asarray(cc.t3), po, pv))
    assert gc_.dtype == np.complex128
    assert sh.shifts() == list(range(-4, 5))
    for s in sh.shifts():
        L = sh.layout(s)
        r1, r2, rng = _random_vector(omb, vmb, s, 20 + s)
        r3 = rng.standard_normal(L.size)
        sb = eom_opt.sigma(gb, r1, _p2(r2, omb, vmb), r3, shift=s)
        sd = eom_opt.sigma(gd, r1, _p2(r2, omd, vmd), _lex(L, r3).ravel())
        s3 = sd[2].reshape(omd.ntri, vmd.ntri)
        s2 = t3pack.unpack_pairs(sd[1], omd, vmd)
        assert abs(sb[0] - sd[0]).max() < 1e-10 * abs(s2).max()
        assert abs(t3pack.unpack_pairs(sb[1], omb, vmb) - s2).max() < 1e-10 * abs(s2).max()
        assert abs(_lex(L, sb[2]) - s3).max() < 1e-10 * abs(s3).max() and abs(s3).max() > 1e-3
        # (the packed block has nothing outside the blocks of the shift)
        m1, m2 = gb.masks(s)
        assert abs(sd[0][~m1]).max(initial=0) < 1e-10 and not sb[0][~m1].any() and not sb[1][~m2].any()
        # the singles-doubles block
        a = eom_opt.sigma(gb, r1, _p2(r2, omb, vmb), None, sd=True, shift=s)
        b = eom_opt.sigma(gd, r1, _p2(r2, omd, vmd), None, sd=True)
        assert a[2] is None and abs(a[0] - b[0]).max() < 1e-10 and abs(a[1] - _p2(t3pack.unpack_pairs(b[1], omd, vmd), omb, vmb)).max() < 1e-10
        if s == 0:
            e = eom_opt.sigma(gb, r1, _p2(r2, omb, vmb), r3)
            assert abs(e[2] - sb[2]).max() < 1e-12 and abs(e[0] - sb[0]).max() < 1e-10 and abs(e[1] - sb[1]).max() < 1e-10
            # the totally symmetric sigma vector has nothing in other shifts, whatever the rounding
            # of the integrals and whatever the vector has there (else the lower states of other
            # shifts grow in the Davidson iterations)
            assert not e[0][~m1].any() and not e[1][~m2].any()
            bad = eom_opt.sigma(gb, r1 + 1e-3 * ~m1, _p2(r2, omb, vmb), None, sd=True)
            assert not bad[0][~m1].any() and not bad[1][~m2].any()
        # complex spinors: every element with its phases
        sc = eom_opt.sigma(gc_, r1 * f1, _p2(r2 * f2, omb, vmb), _phase3(L, r3, po, pv), shift=s)
        assert sc[2].dtype == np.complex128
        assert abs(sc[0] - sb[0] * f1).max() < 1e-11 * abs(s2).max()
        assert abs(sc[1] - sb[1] * _p2(f2, omb, vmb)).max() < 1e-11 * abs(s2).max()
        assert abs(sc[2] - _phase3(L, sb[2], po, pv)).max() < 1e-11 * abs(s3).max()


def _in(e, values):
    return max(abs(np.asarray(values) - x).min() for x in np.atleast_1d(e))


def test_roots_of_every_shift_lih(lih):
    '''LiH: the roots of the shifts 0, +-2, +-4 (labels 2 m_j) are roots of
    the packed block to 1e-10, and below 0.2 Eh their union is the spectrum
    of the packed block with its multiplicities (the triplet at 0.1036 is
    one root each of the shifts -2, 0, 2); +D and -D have the same roots;
    without ``partners`` only one of them is computed; EOM-CCSD.'''
    a, b = lih[True], lih[False]
    assert eom.label_shifts(a._eris) == [-4, -2, 0, 2, 4] and eom.time_reversal_symmetric(a._eris)
    with pytest.raises(ValueError, match='multiples of 2'):
        eom.class_shift(a._eris, 1)
    eb, rb = a.eom_ee(nroots=[4, 4, 4, 2, 2], conv_tol=1e-9, shift=[0, 2, -2, 4, -4], partners=True, nguess=6)
    assert a.eom is rb and a.e_ee is eb and all(r.converged.all() and r.partner_of is None for r in rb.values())
    assert [rb[d].r3[0].size for d in (0, 2, -2, 4, -4)] == [868, 704, 704, 368, 368]
    assert rb[2].shift == 1 and rb[2].label_shift == 2 and rb[0].shift is None
    ep, rp = b.eom_ee(nroots=12, conv_tol=1e-9, nguess=18)
    assert rp.converged.all()
    cut = 0.2
    union = np.sort(np.concatenate([e[e < cut] for e in eb.values()]))
    assert len(union) == 12 and abs(union - ep).max() < 1e-10
    assert abs(eb[0][0] - ep[0]) < 1e-10 and abs(eb[2][0] - ep[0]) < 1e-10 and abs(eb[-2][0] - ep[0]) < 1e-10
    for d in (2, 4):
        assert abs(eb[d] - eb[-d]).max() < 1e-10
    # the vector of a shift has its elements there only
    g = rb[2].ground
    m1, m2 = g.masks(1)
    assert not rb[2].r1[0][~m1].any() and not rb[2].r2[0][~m2].any() and abs(rb[2].r1[0]).max() > 0.3
    # one of each pair +-D only; the other gets the energies
    e2, r2 = a.eom_ee(nroots=2, conv_tol=1e-8, shift=[-2, 0, 2])
    assert r2[2].partner_of == -2 and r2[2].r1 is None and r2[-2].partner_of is None and r2[2].label_shift == 2
    assert np.array_equal(e2[2], e2[-2]) and abs(e2[-2] - eb[-2][:2]).max() < 1e-8
    e1, r1 = a.eom_ee(nroots=2, conv_tol=1e-8, shift=-4)
    assert e1 is r1.e and abs(e1 - eb[4]).max() < 1e-8 and a.eom is r1
    ea, ra = a.eom_ee(nroots=1, conv_tol=1e-7, shift='all')
    assert sorted(ea) == [-4, -2, 0, 2, 4] and ra[-2].partner_of == 2 and ra[4].partner_of is None
    # on the packed block (symmetry off) a shift selects the states by masks
    es, rs = b.eom_ee(nroots=2, conv_tol=1e-9, shift=2)
    assert rs.converged.all() and rs.r3[0].size == 3264 and abs(es - eb[2][:2]).max() < 1e-10
    assert np.count_nonzero(rs.r3[0]) <= 704 and rs.shift is None and rs.label_shift == 2
    # EOM-CCSD (the same amplitudes in both layouts) with a shift
    eb_ = a._eris
    ed, t3d = a._dense_view()
    x = eom.kernel(eb_, a.t1, a.t2, None, nroots=3, conv_tol=1e-9, shift=1)
    y = eom.kernel(ed, a.t1, a.t2, None, nroots=14, nguess=18, conv_tol=1e-9)
    assert x.converged.all() and x.r3 is None and _in(x.e, y.e) < 1e-10


def test_roots_of_every_shift_hbr(hbr):
    '''HBr, 28 frozen, 8 + 16: the eight lowest roots of the packed block
    are the lowest roots of the shifts +-4 (0.2475, the state MRCC finds
    first), +-2 (0.2521, 0.2769) and 0 (0.2602, 0.2607), to 1e-10; with a
    phase on every spinor (complex integrals and amplitudes, the same
    labels) the same roots.'''
    a, b = hbr[True], hbr[False]
    assert eom.label_shifts(a._eris) == [-6, -4, -2, 0, 2, 4, 6]
    eb, rb = a.eom_ee(nroots=[2, 2, 1, 2, 1], conv_tol=1e-9, shift=[0, 2, 4, -2, -4])
    assert all(r.converged.all() for r in rb.values()) and rb[-4].partner_of == 4
    ep, rp = b.eom_ee(nroots=8, conv_tol=1e-9, nguess=12)
    assert rp.converged.all()
    union = np.sort(np.concatenate(list(eb.values())))
    assert abs(union - ep).max() < 1e-10
    assert abs(eb[4][0] - 0.2475010526) < 1e-9 and abs(eb[0] - [0.2601782223, 0.2607020264]).max() < 1e-9
    # complex spinors
    eris = a._eris
    om, vm = t3pack.maps_for(eris)
    sh = eom._shifter(ccsdt_opt.engine_for(eris, np.float64))
    u = np.exp(2j * np.pi * np.random.default_rng(7).random(om.n + vm.n))
    f1, f2, po, pv = _phases(om, vm, u)
    ec = _phased(eris, u)
    t3c = _phase3(sh.layout(0), np.asarray(a.t3), po, pv)
    for d in (2, 4):
        res = eom.kernel(ec, a.t1 * f1, a.t2 * f2, t3c, nroots=len(eb[d]), conv_tol=1e-9, shift=eom.class_shift(ec, d))
        assert res.converged.all() and res.r3[0].dtype == np.complex128 and abs(res.r3[0].imag).max() > 1e-6
        assert abs(res.e - eb[d]).max() < 1e-10 and abs(res.e_imag).max() < 1e-10


def test_shifted_vectors_in_a_scratch_file(hbr, monkeypatch, tmp_path):
    '''The vectors of a shift (EE) and the short vectors of IP as mapped
    arrays of a scratch file, with a subspace of 7 vectors: the same roots
    (the collapse in place and the passes take the mapped arrays).'''
    from zccsdt.scratch import is_mapped
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    cc = hbr[True]
    eris = cc._eris
    sh = eom.ghost_short(eris, cc.t1, cc.t2, cc.t3, 'ip')
    for args, s in (((eris, cc.t1, cc.t2, cc.t3), eom.class_shift(eris, 2)),
                    ((sh.eris, sh.t1, sh.t2, sh.t3), sh.class_shift(1))):
        p = eom.plan_for(args[0], np.float64, 2, opts=solver.Options(incore_complete=True), shift=s)
        ref = eom.kernel(*args, nroots=2, conv_tol=1e-8, plan=p, shift=s)
        pm = eom.Plan(p.nroots, 7, True, p.need, p.fixed, p.vectors, p.vectors, p.max_space)
        res = eom.kernel(*args, nroots=2, conv_tol=1e-8, plan=pm, shift=s)
        assert is_mapped(res.r3[0]) and not is_mapped(ref.r3[0]) and res.converged.all()
        assert abs(res.e - ref.e).max() < 1e-8 and res.ncollapse > ref.ncollapse


def test_plan_and_memory_with_a_shift():
    '''HBr / cc-pVDZ-DK, 8 + 28 in label blocks: the plan of a shift counts
    the vectors of that shift and the blocks (not the packed block and not
    the quarter of the buffer budget that the plan of the ground state sets
    aside for the occupied operator); what a run allocates (tracemalloc:
    numpy's arrays, the work array of the engine among them) is below the
    plan less the allowance for the threads, and above a third of it.'''
    mol = gto.M(atom='Br 0 0 0; H 0 0 1.4145', basis='cc-pvdz-dk', verbose=0)
    mf = x2c.UHF(mol).run(conv_tol=1e-10)
    cc = zccsdt.CCSDT(mf, frozen=28)
    cc.verbose, cc.max_cycle = 0, 3
    cc.kernel()
    eris = cc._eris
    om, vm = t3pack.maps_for(eris)
    eng = ccsdt_opt.engine_for(eris, np.float64)
    assert (om.n, vm.n) == (8, 28) and om.spins is not None
    sh = eom._shifter(eng)
    opts = solver.Options(incore_complete=True)
    packed = om.ntri * vm.ntri
    for d in (0, 2, 4):
        s = eom.class_shift(eris, d)
        plan = eom.plan_for(eris, np.float64, 3, max_space=8, opts=opts, shift=s if s else None)
        n3 = sh.size(s)
        assert plan.vectors == 16 * 8 * (om.n * vm.n + om.npair * vm.npair + n3) and n3 < packed / 5
        assert plan.fixed == eom.fixed_bytes(eris, np.float64, shift=s if s else None)
        # (far below the plan of the same integrals on the packed block)
        assert plan.fixed - ccsdt_opt.overhead() < 8 * (2 * vm.npair ** 2 + 7 * om.n * vm.n ** 3 + 22 * om.n ** 2 * vm.n ** 2 + packed)
        if d == 0:
            continue
        arrays = plan.need - ccsdt_opt.overhead()
        t3engine.release_workspace()
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        res = eom.kernel(eris, cc.t1, cc.t2, cc.t3, nroots=3, conv_tol=1e-3, max_cycle=4, plan=plan, shift=s)
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        assert res.nsigma >= 6 and res.r3[0].size == n3
        assert arrays / 3 < peak <= arrays, (d, peak, arrays, plan)
