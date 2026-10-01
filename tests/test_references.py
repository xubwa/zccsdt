'''CCSD and CCSDT total energies vs MRCC and CFOUR (NCC/ECC/VCC agree to
2e-12 among themselves).  Inputs and outputs of the reference runs are in
bench/refs; the geometries there are given in bohr, converted with PySCF's
factor, so the nuclei coincide with PySCF's Angstrom input below.'''
import json
import os

import pytest
from pyscf import gto, scf

from zccsdt import eris as E
from zccsdt.cc import solver
from zccsdt.genbas import read_genbas

ROOT = os.path.join(os.path.dirname(__file__), '..', 'bench', 'refs')
REFS = json.load(open(os.path.join(ROOT, 'reference_energies.json')))
GEOM = {
    'h2o': 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
    'n2': 'N 0 0 0; N 0 0 1.0977',
    'hf': 'F 0 0 0; H 0 0 0.917',
}
SYSTEMS = {                        # name: (molecule, basis, frozen spin orbitals)
    'h2o_631g': ('h2o', '6-31G', 0),
    'h2o_ccpvdz': ('h2o', 'cc-pvdz', 0),
    'h2o_ccpvdz_fc': ('h2o', 'cc-pvdz', 2),
    'n2_631g': ('n2', '6-31G', 0),
    'hf_ccpvdz': ('hf', 'cc-pvdz', 0),
}


def _basis(mol_atoms, name):
    if name != '6-31G':
        return name
    # the GENBAS 6-31G differs from PySCF's built-in one at ~1e-7
    gb = os.path.join(ROOT, 'GENBAS')
    return {el: read_genbas(gb, '%s:6-31G' % el.upper()) for el in mol_atoms}


@pytest.mark.parametrize('name', list(SYSTEMS))
def test_ccsdt_vs_mrcc_cfour(name):
    molname, basis, frozen = SYSTEMS[name]
    atoms = sorted({a.split()[0] for a in GEOM[molname].split(';')})
    mol = gto.M(atom=GEOM[molname], basis=_basis(atoms, basis), verbose=0)
    rhf = scf.RHF(mol).run(conv_tol=1e-12)
    ref = REFS[name]['recommended']
    assert abs(rhf.e_tot - ref['e_scf']) < 1e-9
    eris = E.from_scf(scf.addons.convert_to_ghf(rhf), frozen=frozen)
    opts = solver.Options(conv_tol=1e-11, conv_tol_normt=1e-8, max_cycle=300)
    rd = solver.kernel(eris, 'ccsd', opts)
    rt = solver.kernel(eris, 'ccsdt', opts)
    assert rd.converged and rt.converged
    assert abs(rd.e_tot - ref['e_ccsd']) < 1e-8
    assert abs(rt.e_tot - ref['e_ccsdt']) < 1e-8


@pytest.mark.parametrize('name', ['h2o_631g', 'h2o_ccpvdz_fc', 'n2_631g'])
def test_spin_adapted_ccsdt_vs_mrcc_cfour(name):
    '''Closed-shell spin-adapted CCSD / CCSDT (spatial orbitals) against the
    same references.'''
    from zccsdt.sa import rccsdt_ref
    molname, basis, frozen = SYSTEMS[name]
    atoms = sorted({a.split()[0] for a in GEOM[molname].split(';')})
    mol = gto.M(atom=GEOM[molname], basis=_basis(atoms, basis), verbose=0)
    rhf = scf.RHF(mol).run(conv_tol=1e-12)
    ref = REFS[name]['recommended']
    ints = rccsdt_ref.from_scf(rhf, ncore=frozen // 2)
    rd = rccsdt_ref.kernel(ints, 'ccsd', conv_tol=1e-11, conv_tol_normt=1e-8)
    rt = rccsdt_ref.kernel(ints, 'ccsdt', conv_tol=1e-11, conv_tol_normt=1e-8)
    assert rd.converged and rt.converged
    assert abs(rd.e_tot - ref['e_ccsd']) < 1e-8
    assert abs(rt.e_tot - ref['e_ccsdt']) < 1e-8


@pytest.mark.parametrize('name', ['h2o_ccpvdz_fc', 'hf_ccpvdz', 'n2_631g'])
def test_spin_adapted_block_ccsdt_vs_mrcc_cfour(name):
    '''Block-stored spin-adapted CCSDT (zccsdt.sa.rccsdt) against the references.'''
    from zccsdt.sa import rccsdt, rccsdt_ref
    molname, basis, frozen = SYSTEMS[name]
    atoms = sorted({a.split()[0] for a in GEOM[molname].split(';')})
    mol = gto.M(atom=GEOM[molname], basis=_basis(atoms, basis), verbose=0)
    rhf = scf.RHF(mol).run(conv_tol=1e-12)
    r = rccsdt.kernel(rccsdt_ref.from_scf(rhf, ncore=frozen // 2), conv_tol=1e-11,
                      conv_tol_normt=1e-8)
    assert r.converged and abs(r.e_tot - REFS[name]['recommended']['e_ccsdt']) < 1e-8


def test_spin_adapted_ccsdt_q_dense_vs_mrcc_cfour():
    '''CCSDT[Q] and CCSDT(Q) from the dense spin-adapted reference
    (rccsdt_ref.q_correction) against MRCC (CFOUR ncc agrees on (Q)).'''
    from zccsdt.sa import rccsdt, rccsdt_ref
    from zccsdt.sa.t3blocks import Blocks
    name = 'h2o_631g'
    molname, basis, frozen = SYSTEMS[name]
    atoms = sorted({a.split()[0] for a in GEOM[molname].split(';')})
    mol = gto.M(atom=GEOM[molname], basis=_basis(atoms, basis), verbose=0)
    rhf = scf.RHF(mol).run(conv_tol=1e-12)
    ints = rccsdt_ref.from_scf(rhf, ncore=frozen // 2)
    r = rccsdt.kernel(ints, conv_tol=1e-11, conv_tol_normt=1e-8)
    e5, e6 = rccsdt_ref.q_correction(r.t2, Blocks(ints.nocc).to_dense(r.t3), ints)
    ref = REFS[name]['recommended']
    assert r.converged
    assert abs(r.e_tot + e5 - ref['e_ccsdt[q]']) < 1e-9
    assert abs(r.e_tot + e6 - ref['e_ccsdt(q)']) < 1e-9
