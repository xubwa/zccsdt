'''CCSDT(Q): the spin-orbital terms against a brute-force Fock-space
evaluation, the spin-adapted spatial form against the spin-orbital one.'''

import numpy as np
import pytest

import fockspace as F
from zccsdt.sa import derive, somap
from zccsdt.sa.spinfree import OCC, evaluate


def _eval_so(terms, t2, t3, v, no):
    '''Spin-orbital Terms (kinds t2, t3, v) on spin-orbital arrays.'''
    n = v.shape[0]
    sl = {True: slice(0, no), False: slice(no, n)}
    out = 0.0
    for t in terms:
        ops = [v[tuple(sl[c in OCC] for c in ix)] if kind == 'v' else {'t2': t2, 't3': t3}[kind]
               for kind, ix in t.factors]
        out = out + float(t.coef) * np.einsum(','.join(ix for _, ix in t.factors) + '->ijklabcd',
                                              *ops, optimize=True)
    return out


def test_spin_orbital_terms_equal_fock_space():
    '''r = <Q|[W,T3] + 1/2 [[W,T2],T2]|0> and z = <0|(T2^+ + T3^+) W|Q> from
    second-quantized operators in a 9-spin-orbital Fock space, random
    antisymmetric amplitudes and integrals.'''
    no, nv = 5, 4
    rng = np.random.default_rng(1)
    t2 = F.random_antisym(rng, no, nv, 2)
    t3 = F.random_antisym(rng, no, nv, 3)
    v = F.random_integrals(rng, no + nv)
    fk = F.Fock(no, nv)
    W, T2, T3, ref = fk.two_body(v), fk.excitation(t2), fk.excitation(t3), fk.ref
    x = T2 @ ref
    want = {
        'R4.T3': fk.project(W @ (T3 @ ref) - T3 @ (W @ ref), 4),
        'R4.T2': fk.project(0.5 * (W @ (T2 @ x) - 2 * (T2 @ (W @ x)) + T2 @ (T2 @ (W @ ref))), 4),
        'Z4.T2': fk.project(W.T @ (T2 @ ref), 4),
        'Z4.T3': fk.project(W.T @ (T3 @ ref), 4),
    }
    for name, terms in derive.q_terms().items():
        got = _eval_so(terms, t2, t3, v, no)
        assert abs(got - want[name]).max() < 1e-10 * abs(want[name]).max(), name


def test_energy_metric():
    '''E = 1/(4!)^2 sum Z R / D over spin orbitals equals the spatial formula
    1/24 sum z/D sum_tau sgn(tau) 2^cycles(tau) r^{tau(abcd)} for pair-symmetric
    spatial z and r (zccsdt.sa.derive.q_metric).'''
    from itertools import permutations
    no, nv = 3, 3
    rng = np.random.default_rng(2)

    def pairsym(x):
        return sum(x.transpose(p + tuple(4 + y for y in p)) for p in permutations(range(4)))
    z = pairsym(rng.standard_normal((no,) * 4 + (nv,) * 4))
    r = pairsym(rng.standard_normal((no,) * 4 + (nv,) * 4))
    eo, ev = -1 - rng.random(no), 1 + rng.random(nv)

    def denom(eo, ev):
        return sum(eo.reshape([-1 if x == n else 1 for x in range(8)]) for n in range(4)) \
            - sum(ev.reshape([-1 if x == n else 1 for x in range(8)]) for n in range(4, 8))
    e_so = np.sum(somap.quadruples(z) * somap.quadruples(r) / denom(np.r_[eo, eo], np.r_[ev, ev])) / 576
    rt = sum(w * r.transpose((0, 1, 2, 3) + tuple(4 + x for x in tau)) for tau, w in derive.q_metric())
    e_sp = np.sum(z * rt / denom(eo, ev)) / 24
    assert abs(e_so - e_sp) < 1e-10 * abs(e_so)
    w = dict(derive.q_metric())
    assert w[(0, 1, 2, 3)] == 16 and w[(1, 0, 2, 3)] == -8 and w[(1, 2, 0, 3)] == 4
    assert w[(1, 0, 3, 2)] == 4 and w[(1, 2, 3, 0)] == -2
