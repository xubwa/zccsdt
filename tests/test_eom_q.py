'''The one-shot quadruples correction of EOM-CCSDT excitation energies
(zccsdt.cc.eom_q; docs/eom.md, section 14):

- its two terms of the state are the left vector times what the folded
  Jacobian of CC4 (cc4fly.sigma) adds to the sigma vector of CCSDT;
- the term of the ground-state amplitudes against the excitation energies
  of the Jacobian of CCSDT at the CC4 amplitudes, Lambda . rho against
  E(CC4) - E(CCSDT), and the estimate against converged EOM-CC4: H2O /
  STO-3G;
- in the blocks of the labels with the sectors of the (Q) code, and a state
  with a label shift, against the packed block: LiH with X2C spinors.'''
import warnings

import numpy as np
import pytest
from pyscf import scf

import zccsdt
from conftest import water_ghf
from test_ccsdtq_ref import _scf, _spin_mixed
from test_eom_shift import lih    # noqa: F401  (fixture)
from zccsdt import eris as E
from zccsdt.cc import cc4fly, eom, eom_left as EL, eom_opt, eom_q, solver, t4engine


@pytest.mark.parametrize('cplx', [False, True])
def test_terms_are_those_of_the_folded_jacobian(cplx):
    '''LiH / STO-3G (4 + 8), real and complex spin orbitals: fold + t4 =
    L (sigma of the folded CC4 Jacobian - sigma of CCSDT) at omega, to
    1e-12; the Rayleigh quotient of the right-hand vector the same with
    conj(R).'''
    mf = scf.addons.convert_to_ghf(_scf('lih'))
    eris = E.from_scf(mf, mo_coeff=_spin_mixed(mf)[0] if cplx else mf.mo_coeff, packed_vvvv=True)
    rt = solver.kernel(eris, 'ccsdt', solver.Options(conv_tol=1e-12, conv_tol_normt=1e-10))
    res = eom.kernel(eris, rt.t1, rt.t2, rt.t3, nroots=4, conv_tol=1e-8)
    left = EL.kernel(eris, rt.t1, rt.t2, rt.t3, res)
    q = eom_q.kernel(eris, rt.t1, rt.t2, rt.t3, res, left)
    assert q.e_ground is None and len(q.states) == 4 and len(q.text()) == 4
    g = res.ground
    ints = t4engine.Dressed.from_eris(eris, g.t1)
    e = np.asarray(eris.fock).diagonal().real
    for k, c in enumerate(q.states):
        s0 = eom_opt.sigma(g, res.r1[k], res.r2[k], res.r3[k])
        s0 = [np.array(x) for x in s0]
        s = cc4fly.sigma(g, ints, e, res.r1[k], res.r2[k], res.r3[k], res.e[k])
        d2, d3 = np.ravel(s[1] - s0[1]), np.ravel(s[2] - s0[2])
        assert abs(np.ravel(s[0] - s0[0])).max() < 1e-13
        want = np.dot(left.l2[k].ravel(), d2) + np.dot(left.l3[k].ravel(), d3)
        assert c.ground is None and abs(c.fold + c.t4 - want.real) < 1e-12 and abs(want) > 1e-8
        r = np.concatenate([res.r1[k].ravel(), res.r2[k].ravel(), res.r3[k].ravel()])
        ray = (np.vdot(res.r2[k].ravel(), d2) + np.vdot(res.r3[k].ravel(), d3)) / np.vdot(r, r)
        assert abs(c.rayleigh - ray.real) < 1e-12 and c.total == c.fold + c.t4 and c.estimate == c.omega + c.total


def test_h2o_against_converged_cc4():
    '''H2O / STO-3G, spin orbitals of RHF (10 + 4), the states of M_S = 0:
    Lambda . rho is E(CC4) - E(CCSDT) and the term of the ground-state
    amplitudes the change of omega of the CCSDT Jacobian between the CCSDT
    and the CC4 amplitudes, both to a few percent (first order); the
    estimate is closer to converged EOM-CC4 than EOM-CCSDT and than the
    Rayleigh quotient of the right-hand vector.'''
    mf = water_ghf('sto-3g')

    def make(cls):
        cc = cls(mf)
        cc.verbose, cc.conv_tol, cc.conv_tol_normt = 0, 1e-12, 1e-10
        return cc
    cc = make(zccsdt.CCSDT)
    cc.kernel()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        e, res = cc.eom_ee(nroots=2, shift=0, conv_tol=1e-8)
        q = cc.eom_quadruples()
        assert res.quadruples is q and res.left.ops.sector is not None
        c4 = make(zccsdt.CC4)
        c4.kernel()
        with pytest.raises(NotImplementedError, match='CCSDT amplitudes'):
            c4.eom_quadruples()
        de = c4.e_tot - cc.e_tot
        assert abs(q.e_ground - de) < 0.05 * abs(de) and abs(de) > 1e-6
        # the Jacobian of CCSDT at the CC4 amplitudes
        ci = make(zccsdt.CCSDT)
        ci._eris, ci.t1, ci.t2, ci.t3 = c4._eris, c4.t1, c4.t2, c4.t3
        ei, ri = ci.eom_ee(nroots=2, shift=0, conv_tol=1e-8)
        qb = ci.eom_quadruples(ground=False)
        e4, sol = c4.eom_ee(nroots=6, conv_tol=1e-8)
    for k in range(2):
        a, b = q.states[k], qb.states[k]
        conv4 = e4[np.argmin(abs(e4 - a.estimate))]
        shift_i = ei[k] - e[k]
        print('H2O / STO-3G root %d: omega %.9f, (i) %.9f, converged CC4 %.9f; A fold %+.3e t4 %+.3e ground %+.3e (against %+.3e) '
              'estimate %+.2e off; B %+.2e off; Rayleigh %+.2e off' % (k + 1, a.omega, ei[k], conv4, a.fold, a.t4, a.ground,
                                                                    shift_i, a.estimate - conv4, b.estimate - conv4,
                                                                    b.omega + b.rayleigh - conv4))
        assert abs(a.ground - shift_i) < 0.05 * abs(shift_i) + 1e-8
        assert abs(a.estimate - conv4) < 0.5 * abs(a.omega - conv4)
        assert abs(b.estimate - conv4) < 0.5 * abs(b.omega + b.rayleigh - conv4)
        assert abs(b.estimate - conv4) < 0.5 * abs(ei[k] - conv4)


def test_blocks_sectors_and_a_shift(lih):
    '''LiH with X2C spinors (4 + 18): the correction of the totally
    symmetric states in the blocks of the labels (the quadruples in the
    sectors of the (Q) code) and of a state of shift 2 (the quadruples on
    the packed block, no sectors) against the packed block: the levels of
    the packed block that the two shifts hold, summed over the level, term
    by term.'''
    a, b = lih[True], lih[False]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ea, ra = a.eom_ee(nroots=[4, 2], conv_tol=1e-8, shift=[0, 2], nguess=6)
        qa = {d: a.eom_quadruples(ra[d]) for d in (0, 2)}
        eb, rb = b.eom_ee(nroots=8, conv_tol=1e-8, nguess=12)
        qb = b.eom_quadruples()
    assert abs(qa[0].e_ground - qb.e_ground) < 1e-10 and abs(qa[2].e_ground - qb.e_ground) < 1e-10
    terms = lambda c: np.array([c.fold, c.t4, c.ground])    # noqa: E731
    n = 0
    for lv in rb.left.levels:
        got = sum(terms(qb.states[i]) for i in lv)
        k0 = [k for k in range(len(ea[0])) if abs(ea[0][k] - eb[lv[0]]) < 1e-8]
        k2 = [k for k in range(len(ea[2])) if abs(ea[2][k] - eb[lv[0]]) < 1e-8]
        if len(k0) + 2 * len(k2) != len(lv):
            continue
        # (the states of shift -2 are the time-reversed images of those of shift 2)
        want = sum(terms(qa[0].states[k]) for k in k0) + 2 * sum(terms(qa[2].states[k]) for k in k2)
        assert abs(got - want).max() < 5e-9 and abs(got).max() > 1e-7, (lv, got, want)
        n += 1
    assert n >= 3
