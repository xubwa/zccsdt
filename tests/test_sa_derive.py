'''The closed-shell spin-adapted equations (zccsdt.sa.derive) reproduce the
spin-orbital equations on random spin-adapted amplitudes and integrals.'''

import numpy as np
import pytest

from zccsdt.sa import derive, somap
from zccsdt.sa.spinfree import evaluate

NO, NV = 2, 3


@pytest.fixture(scope='module')
def data():
    d = somap.random_spatial(NO, NV, seed=7)
    dims = {'o': NO, 'v': NV}
    imds = derive.spatial_intermediates()
    arr = dict(d)
    slots = {name: s for name, (s, _) in derive.intermediate_terms().items()}
    for key, terms in imds.items():
        name = key.split('.')[0]
        arr[key] = evaluate(terms, d, slots[name], dims)
    return d, arr, dims, slots, derive.spatial_residuals()


def test_intermediates(data):
    d, arr, dims, slots, _ = data
    for name, (s, terms) in derive.intermediate_terms().items():
        so = somap.evaluate_so(terms, d, s, NO)
        if len(s) == 2:
            ref = somap.one_body(arr[name])
        else:
            ref = somap.two_body(arr[name + '.J'], arr[name + '.K'])
        assert abs(so - ref).max() < 1e-12 * max(1, abs(so).max()), name


def test_t1_t2_residuals(data):
    d, arr, dims, slots, res = data
    r1 = evaluate(res['R1'], d, 'ia', dims)
    r1so = somap.evaluate_so(derive.t1_residual_terms(), d, 'ia', NO)
    assert abs(r1so[:NO, :NV] - r1).max() < 1e-11
    assert abs(r1so[NO:, NV:] - r1).max() < 1e-11
    r2 = evaluate(res['R2'], d, 'ijab', dims)
    r2so = somap.evaluate_so(derive.t2_residual_terms(), d, 'ijab', NO)
    # spin-orbital R2 has the structure of t2: r(ij,ab) d d - r(ij,ba) d d
    assert abs(r2so - somap.two_body(r2)).max() < 1e-10 * abs(r2so).max()


def test_t3_residual(data):
    d, arr, dims, slots, res = data
    r3 = evaluate(res['R3'], arr, 'ijkabc', dims)
    spatial = dict(arr)
    r3so = somap.evaluate_so(derive.t3_residual_terms(), spatial, 'ijkabc', NO)
    # the full spin-orbital tensor, all spin blocks (the redundant component
    # of the spatial residual maps to zero)
    assert abs(r3so - somap.triples(r3)).max() < 1e-10 * abs(r3so).max()


def test_t3_residual_against_reference_code(data):
    '''Against zccsdt.cc.ccsdt._t3_residual (spin-orbital, occupied-first
    ordering), with intermediates built from the same spatial V, F, T2, T3.'''
    from zccsdt.cc import ccsdt
    d, arr, dims, slots, res = data
    n = NO + NV
    # spin orbitals: occ alpha, occ beta, vir alpha, vir beta
    sp = np.r_[np.arange(NO), np.arange(NO), NO + np.arange(NV), NO + np.arange(NV)]
    sg = np.r_[np.zeros(NO), np.ones(NO), np.zeros(NV), np.ones(NV)].astype(int)
    V = d['V']
    Vs = V[np.ix_(sp, sp, sp, sp)]
    Vx = V.transpose(0, 1, 3, 2)[np.ix_(sp, sp, sp, sp)]
    e = sg[:, None] == sg[None, :]
    ge = Vs * e[:, None, :, None] * e[None, :, None, :] \
        - Vx * e[:, None, None, :] * e[None, :, :, None]
    fdr = d['F'][np.ix_(sp, sp)] * e
    t2so = somap.two_body(d['T2'])
    t3so = somap.triples(d['T3'])
    ref = ccsdt._t3_residual(t2so, t3so, ge.astype(complex), fdr.astype(complex),
                             2 * NO, 2 * n)
    r3 = evaluate(res['R3'], arr, 'ijkabc', dims)
    assert abs(ref - somap.triples(r3)).max() < 1e-10 * abs(ref).max()
