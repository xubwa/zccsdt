'''Block-stored spin-adapted RCCSDT (zccsdt.sa.rccsdt) against the reference
implementation (zccsdt.sa.rccsdt_ref).'''

import numpy as np
import pytest
from pyscf import gto, scf

from zccsdt.sa import rccsdt as E, rccsdt_ref as R, somap
from zccsdt.sa.spinfree import evaluate
from zccsdt.sa.t3blocks import Blocks


@pytest.fixture(scope='module')
def setup():
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    ints = R.from_scf(mf)
    no, nv = ints.nocc, ints.nvir
    rng = np.random.default_rng(0)
    t1 = 0.05 * rng.standard_normal((no, nv))
    t2 = 0.05 * rng.standard_normal((no, no, nv, nv))
    t2 = t2 + t2.transpose(1, 0, 3, 2)
    t3 = R.project_null(0.02 * somap.pair_symmetrize(rng.standard_normal((no,) * 3 + (nv,) * 3)))
    return ints, t1, t2, t3


def test_blocks_roundtrip(setup):
    ints, t1, t2, t3 = setup
    B = Blocks(ints.nocc)
    assert abs(B.to_dense(B.from_dense(t3)) - t3).max() < 1e-15


def test_residuals_equal_reference(setup):
    ints, t1, t2, t3 = setup
    B = Blocks(ints.nocc)
    r1, r2, r3b = E.residuals(ints, B, t1, t2, B.from_dense(t3))
    a1, a2, a3 = R.residuals(t1, t2, t3, ints)
    assert abs(r1 - a1).max() < 1e-12
    assert abs(r2 - a2).max() < 1e-12
    # equal up to the redundant component
    assert abs(B.to_dense(r3b) - R.project_null(a3)).max() < 1e-12


def test_energy_equals_reference():
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    ints = R.from_scf(mf, ncore=1)
    a = R.kernel(ints, 'ccsdt', conv_tol=1e-11, conv_tol_normt=1e-8)
    b = E.kernel(ints, conv_tol=1e-11, conv_tol_normt=1e-8)
    assert a.converged and b.converged and abs(a.e_tot - b.e_tot) < 1e-10
