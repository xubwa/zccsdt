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


@pytest.fixture(scope='module')
def h2o():
    '''H2O / 6-31G, frozen core: integrals, blocks, amplitudes after a few
    CCSDT iterations, and the dense spatial r / z arrays.'''
    from pyscf import gto, scf
    from zccsdt.sa import rccsdt, rccsdt_ref
    from zccsdt.sa.t3blocks import Blocks
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    ints = rccsdt_ref.from_scf(mf, ncore=1)
    r = rccsdt.kernel(ints, max_cycle=4)
    B = Blocks(ints.nocc)
    t3 = B.to_dense(r.t3)
    arr = {'V': ints.V, 'T2': r.t2, 'T3': t3}
    dims = {'o': ints.nocc, 'v': ints.nvir}
    val = {k: evaluate(terms, arr, 'ijklabcd', dims) for k, terms in derive.q_spatial().items()}
    return ints, B, r.t2, r.t3, t3, val


def test_quadruple_blocks_equal_dense(h2o):
    '''A, B, C of single occupied quadruples (distinct, permuted, repeated
    indices) and the energies of the per-quadruple algorithm against the
    dense term-by-term evaluation.'''
    from zccsdt.sa import rccsdt_q, rccsdt_ref
    ints, B, t2, t3b, t3, val = h2o
    assert abs(val['Z4.T3'] - val['R4.T3']).max() < 1e-14      # real orbitals, bare integrals
    ctx = rccsdt_q.Context(ints, B, t2, t3b)
    for quad in ((0, 1, 2, 3), (2, 0, 3, 1), (0, 0, 1, 2), (1, 3, 3, 0), (1, 1, 2, 2)):
        A, Bq, C = rccsdt_q.quad_arrays(ctx, quad)
        assert abs(A - val['R4.T3'][quad]).max() < 1e-13
        assert abs(Bq - val['R4.T2'][quad]).max() < 1e-13
        assert abs(C - val['Z4.T2'][quad]).max() < 1e-13
    ref = rccsdt_ref.q_correction(t2, t3, ints)
    got = rccsdt_q.q_correction_simple(ints, B, t2, t3b)
    assert abs(got[0] - ref[0]) < 1e-13 and abs(got[1] - ref[1]) < 1e-13
    quads = rccsdt_q.unique_quadruples(ints.nocc)
    assert sum(m for _, m in quads) == 4 ** 4 - 4 * (1 + 4 * 3)   # no index three or four times


def test_quad_energy_kernel(monkeypatch):
    '''Tiled C kernel (cyclic accumulator layouts, partial and repeated
    tiles) against the numpy evaluation.'''
    from zccsdt.cc import t3pack
    from zccsdt.sa import rccsdt_q
    if not t3pack.USE_C:
        pytest.skip('no C library')
    v = 13
    rng = np.random.default_rng(4)
    S = [rng.standard_normal((v,) * 4) for _ in range(6)]
    tc, gc = rng.standard_normal((6, v, v)), rng.standard_normal((6, v, v))
    ev = 1 + rng.random(v)
    got = rccsdt_q.quad_energy(S, tc, gc, -3.0, ev)
    monkeypatch.setattr(t3pack, 'USE_C', False)
    ref = rccsdt_q.quad_energy(S, tc, gc, -3.0, ev)
    assert abs(got[0] - ref[0]) < 1e-10 * abs(ref[0])
    assert abs(got[1] - ref[1]) < 1e-10 * abs(ref[1])


def test_block_algorithm_equals_simple(h2o, monkeypatch):
    '''Accumulator / stacked-GEMM algorithm against the plain per-quadruple
    version, with the C kernels and with the numpy fallbacks.'''
    from zccsdt.cc import t3pack
    from zccsdt.sa import rccsdt_q
    ints, B, t2, t3b, t3, val = h2o
    ref = rccsdt_q.q_correction_simple(ints, B, t2, t3b)
    for use_c in (True, False):
        monkeypatch.setattr(t3pack, 'USE_C', use_c and t3pack._lib is not None)
        got = rccsdt_q.q_correction(ints, B, t2, t3b)
        assert abs(got[0] - ref[0]) < 1e-13 and abs(got[1] - ref[1]) < 1e-13
