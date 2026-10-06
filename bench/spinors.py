'''The molecule and the spinor SCF of the bench scripts for heavy elements:
basis sets by row of the periodic table, the nuclear model, the Hamiltonian
(X2C-1e, or with the atomic mean-field two-electron spin-orbit integrals of
socutils: X2CAMF) and the SCF class (PySCF's x2c.UHF or socutils' SpinorSCF,
whose J and K come from the scalar AO integrals).

environment: ZC_GEOMETRY: a file with 'symbol x y z' per atom in bohr, in
             the place of the geometry of the table (the frame and the order
             of the atoms of another program, for operators taken from it);
             ZC_DM0=FILE.npz:KEY: the density the SCF starts from (that of
             the SCF of another program: the same state of a doublet);
             ZC_AXIAL=1: a linear molecule on the z axis: the SCF with the
             symmetry of the axis (a doublet: the spin on the axis);
             ZC_NUCMOD=G: Gaussian nuclei (default: point nuclei);
             ZC_HAM: x2c1e (default) | x2camf[+gaunt][+breit][+aoc][+pcc]
             (socutils.somf.amf.SpinorX2CAMFHelper; the flags as CFOUR's
             %amfiMethod: average of configurations, Gaunt, gauge term,
             two-electron picture change of the scalar part);
             ZC_SCF: pyscf (default) | socutils
'''
import os
import sys

import numpy as np
from pyscf import gto, x2c

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_iter import GEOM, SPIN                                     # noqa: E402

# diatomics with a heavy atom at the origin, the other one on the z axis
# (Angstrom; near the minima of the Hartree-Fock curves)
GEOM.update({
    'hbr': 'Br 0 0 0; H 0 0 1.414',
    'hi': 'I 0 0 0; H 0 0 1.60',
    'hat': 'At 0 0 0; H 0 0 1.71',
    'agh': 'Ag 0 0 0; H 0 0 1.70',
    'auh': 'Au 0 0 0; H 0 0 1.57',
    'agf': 'Ag 0 0 0; F 0 0 2.02',
    'auf': 'Au 0 0 0; F 0 0 1.96',
    'i2': 'I 0 0 0; I 0 0 2.68',
    'at2': 'At 0 0 0; At 0 0 2.97',
    'au2': 'Au 0 0 0; Au 0 0 2.59',
    # doublets with one electron outside closed shells (sigma), for the
    # properties that are odd under time reversal
    'beh': 'Be 0 0 0; H 0 0 1.343',
    'mgf': 'Mg 0 0 0; F 0 0 1.750',
    'srf': 'Sr 0 0 0; F 0 0 2.075',
    'baf': 'Ba 0 0 0; F 0 0 2.159',
    'ybf': 'Yb 0 0 0; F 0 0 2.016',
})
SPIN.update({'beh': 1, 'mgf': 1, 'srf': 1, 'baf': 1, 'ybf': 1})


def atoms_of(mol_name):
    '''[(symbol, (x, y, z))] of GEOM[mol_name] (Angstrom).'''
    out = []
    for line in GEOM[mol_name].replace('\n', ';').split(';'):
        f = line.split()
        if f:
            out.append((f[0], tuple(float(x) for x in f[1:4])))
    return out


def basis_of(mol_name, basis):
    '''``basis`` as gto.M takes it: 'heavy,light' is a dictionary with the
    first name for the atoms beyond argon and the second for the others.'''
    if ',' not in basis:
        return basis
    heavy, light = basis.split(',')
    return {s: heavy if gto.charge(s) > 18 else light for s, _ in atoms_of(mol_name)}


def molecule(mol_name, basis, **kwargs):
    nucmod = 'G' if os.environ.get('ZC_NUCMOD', '').upper().startswith('G') else None
    atom, unit = GEOM[mol_name], 'Angstrom'
    if os.environ.get('ZC_GEOMETRY'):
        with open(os.environ['ZC_GEOMETRY']) as f:
            atom, unit = f.read(), 'Bohr'
    return gto.M(atom=atom, unit=unit, basis=basis_of(mol_name, basis), nucmod=nucmod,
                 spin=SPIN.get(mol_name, 0), verbose=0, **kwargs)


def axial(mol, dm):
    '''``dm`` without the blocks between spinor AOs of different m_j, for a
    linear molecule on the z axis: the part that has the symmetry of the
    axis.  The two Kramers partners of a doublet and every state between
    them solve the SCF (with weak spin-orbit coupling any direction of the
    spin); the one with the spin on the axis has a definite Omega - a
    property that is odd under time reversal is defined for it, and it
    stays a solution in a field along the axis.'''
    if abs(mol.atom_coords()[:, :2]).max() > 1e-8:
        raise ValueError('not on the z axis')
    mj = np.array([lab[3] for lab in mol.spinor_labels(fmt=False)])
    return np.where(mj[:, None] == mj[None, :], dm, 0)


def hamiltonian():
    '''(name, flags) of ZC_HAM.'''
    name, *flags = os.environ.get('ZC_HAM', 'x2c1e').lower().split('+')
    if name not in ('x2c1e', 'x2camf') or set(flags) - {'gaunt', 'breit', 'aoc', 'pcc'}:
        raise ValueError('ZC_HAM=%s' % os.environ['ZC_HAM'])
    return name, flags


def scf(mol, conv_tol=1e-10, conv_tol_grad=1e-8, field=None, dm0=None, max_cycle=200):
    '''The converged spinor SCF of ``mol``.  ``field``: a matrix in the basis
    of the spinor AOs added to the one-electron Hamiltonian (finite
    differences of a property).  The energy of a heavy molecule is not
    defined to 1e-12 - its last digits change from one iteration to the next -,
    so ``conv_tol`` has a floor and the orbital gradient tells the
    convergence.'''
    name, flags = hamiltonian()
    if os.environ.get('ZC_SCF', 'pyscf') == 'socutils':
        from socutils.scf import spinor_hf
        mf = spinor_hf.SpinorSCF(mol)
        # (on its own it has the nonrelativistic one-electron Hamiltonian)
        mf.with_x2c = x2c.x2c.SpinorX2CHelper(mol)
    else:
        mf = x2c.UHF(mol)
    if name == 'x2camf':
        from socutils.somf import amf
        mf.with_x2c = amf.SpinorX2CAMFHelper(
            mol, with_gaunt='gaunt' in flags, with_breit='breit' in flags,
            with_aoc='aoc' in flags, with_pcc='pcc' in flags)
    if field is not None:
        h0 = mf.get_hcore()
        mf.get_hcore = lambda *args: h0 + field
    if dm0 is None and os.environ.get('ZC_DM0'):
        path, key = os.environ['ZC_DM0'].rsplit(':', 1)
        dm0 = np.load(path)[key]
    # (about 1e-14 of the energy, from the nuclear charges: 1.5e-12 for water,
    # 1.4e-10 for an iodine atom)
    floor = 1e-14 * sum(float(z) ** 2.4 for z in mol.atom_charges())
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = max(conv_tol, floor), conv_tol_grad, max_cycle
    mf.kernel(dm0=dm0)
    if os.environ.get('ZC_AXIAL') == '1':
        mf.kernel(dm0=axial(mol, mf.make_rdm1()))
    return mf
