'''EOM-CCSDT on packed amplitudes (zccsdt.cc.eom_opt) against the dense
reference (zccsdt.cc.eom_ref): the sigma vector for random vectors at
amplitudes that solve nothing (the Jacobian of the amplitude equations at
any point), with real and with complex spin orbitals, all electrons and a
frozen core, with and without triples.'''
import numpy as np
import pytest
from pyscf import scf

import fockspace as F
from conftest import water_ghf
from test_ccsdtq_ref import _scf, _spin_mixed
from zccsdt import eris as E
from zccsdt.cc import ccsdtq_ref as R, eom_opt as EO, eom_ref as EOM, t3pack


def pack2(x, om, vm):
    return x[om.pairs[:, 0][:, None], om.pairs[:, 1][:, None], vm.pairs[:, 0], vm.pairs[:, 1]]


def pack3(x, om, vm):
    o, v = om.triples, vm.triples
    return x[o[:, 0][:, None], o[:, 1][:, None], o[:, 2][:, None], v[:, 0], v[:, 1], v[:, 2]]


def random_case(name, cplx, with3, seed=1):
    '''(eris, amplitudes, vector) dense, random: LiH / STO-3G (4 + 8 spin
    orbitals) or H2O / STO-3G with the 1s pair frozen (8 + 4).'''
    if name == 'lih':
        mf, frozen = scf.addons.convert_to_ghf(_scf('lih')), 0
    else:
        mf, frozen = water_ghf('sto-3g'), 2
    C = _spin_mixed(mf)[0] if cplx else mf.mo_coeff
    eris = E.from_scf(mf, mo_coeff=C, frozen=frozen)
    no, nv = eris.nocc, eris.fock.shape[0] - eris.nocc
    rng = np.random.default_rng(seed)
    mk = F.complex_antisym if cplx else (lambda rng, no, nv, k, s=1.0: s * F.random_antisym(rng, no, nv, k))
    n = 3 if with3 else 2
    amps = [mk(rng, no, nv, k, s) for k, s in ((1, 0.1), (2, 0.05), (3, 0.02))[:n]]
    r = [mk(rng, no, nv, k, 1.0) for k in (1, 2, 3)[:n]]
    return eris, amps, r


@pytest.mark.parametrize('name', ['lih', 'h2o'])
@pytest.mark.parametrize('cplx', [False, True])
@pytest.mark.parametrize('with3', [True, False])
def test_sigma_against_the_reference(name, cplx, with3):
    eris, amps, r = random_case(name, cplx, with3)
    fock, g, mo_e, no = R.integrals(eris)
    om, vm = t3pack.maps_for(eris)
    gs = EO.prepare(eris, amps[0], amps[1], pack3(amps[2], om, vm).ravel() if with3 else None)
    assert gs.dtype == (np.complex128 if cplx else np.float64) and gs.change > 1e-3
    s = EO.sigma(gs, r[0], pack2(r[1], om, vm), pack3(r[2], om, vm).ravel() if with3 else None)
    w = EOM.sigma(r, amps, fock, g, no)
    assert abs(s[0] - w[0]).max() < 1e-12 * abs(w[0]).max()
    assert abs(s[1] - pack2(w[1], om, vm)).max() < 1e-12 * abs(w[1]).max()
    if with3:
        assert s[2].dtype == gs.dtype
        assert abs(s[2].reshape(om.ntri, vm.ntri) - pack3(w[2], om, vm)).max() < 1e-12 * abs(w[2]).max()
    else:
        assert s[2] is None
