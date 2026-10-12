'''The ladder CC2 < CCSD < CC3 < CCSDT < CC4 < CCSDTQ (< FCI): ground-state
energies and the lowest excitation energies (docs/ccn.md, section 12).

    python bench/ccn/ladder.py h2o        H2O / 6-31G, RHF as spin orbitals, all electrons
                                          (the basis entries and the geometry of section 7),
                                          with FCI; lowest triplet and singlet
    python bench/ccn/ladder.py hbr        HBr / cc-pVDZ-DK, X2C spinors, 28 frozen (8 + 28);
                                          the lowest totally symmetric state (label shift 0)

CCSDTQ has no excitation energies in this program (no Jacobian route):
those of MRCC (LR-CCSDTQ, same basis and geometry) for H2O.
'''
import os
import sys
import time

import numpy as np
from pyscf import fci, gto, scf, x2c

import zccsdt
from zccsdt.cc import cc4fly
from zccsdt.genbas import read_genbas

HARTREE_EV = 27.211386245988
which = sys.argv[1]
TIGHT = dict(conv_tol=1e-11, conv_tol_normt=1e-8)


def run(name, mf, frozen=None, **kw):
    cls = getattr(zccsdt, name)
    cc = cls(mf, frozen=frozen) if frozen is not None else cls(mf)
    cc.verbose = 0
    cc.conv_tol, cc.conv_tol_normt = TIGHT['conv_tol'], TIGHT['conv_tol_normt']
    for k, v in kw.items():
        setattr(cc, k, v)
    t0 = time.time()
    cc.kernel()
    return cc, time.time() - t0


def levels(e, tol=1e-6):
    '''The distinct levels of sorted energies with their multiplicities.'''
    out = []
    for x in np.sort(e):
        if out and abs(x - out[-1][0]) < tol:
            out[-1][1] += 1
        else:
            out.append([x, 1])
    return out


if which == 'h2o':
    genbas = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'refs', 'GENBAS')
    mol = gto.M(atom='O 0 0 0; H 0 1.430522676295752 1.109269235119691; H 0 -1.430522676295752 1.109269235119691',
                basis={x: read_genbas(genbas, x + ':6-31G') for x in ('O', 'H')}, unit='Bohr', verbose=0)
    rhf = scf.RHF(mol).run(conv_tol=1e-13, conv_tol_grad=1e-9)
    print('E(SCF) = %.12f' % rhf.e_tot)
    # FCI: the ground state, the lowest triplet and singlet (spin of the states)
    sol = fci.FCI(rhf)
    sol.conv_tol = 1e-12
    e_fci, ci = sol.kernel(nroots=8)
    S2 = [fci.spin_op.spin_square0(c, mol.nao, mol.nelectron)[0] for c in ci]
    trip = min(x for x, s in zip(e_fci[1:], S2[1:]) if abs(s - 2) < 1e-3)
    sing = min(x for x, s in zip(e_fci[1:], S2[1:]) if abs(s) < 1e-3)
    print('FCI     E = %.12f   triplet %.6f eV   singlet %.6f eV' % (e_fci[0], (trip - e_fci[0]) * HARTREE_EV,
                                                                   (sing - e_fci[0]) * HARTREE_EV), flush=True)
    for name in ('CC2', 'CCSD', 'CC3', 'CCSDT', 'CC4', 'CCSDTQ'):
        cc, t = run(name, scf.addons.convert_to_ghf(rhf))
        line = '%-7s E = %.12f   (%.1f s)' % (name, cc.e_tot, t)
        if name == 'CCSDTQ':
            # MRCC LR-CCSDTQ (bench/ccn/mrcc_ground.sh with nsing / ntrip): totals
            e0, et, es = -76.120856115034, -75.835882067845, -75.808993052699
            line += '   MRCC LR-CCSDTQ: triplet %.6f eV   singlet %.6f eV   (MRCC E %.12f)' % (
                (et - e0) * HARTREE_EV, (es - e0) * HARTREE_EV, e0)
        else:
            if name == 'CC4':
                e, _ = cc.eom_ee(nroots=5, conv_tol=1e-7)
            else:
                e, _ = cc.eom_ee(nroots=5, shift=0, conv_tol=1e-7)
            lv = levels(e)
            # shift 0 of 2 m_s (CC2 .. CCSDT) or all states (CC4): the triplet is the level whose
            # M_S = 0 component is there (CC4: threefold), the singlet the first single level
            t_ = next(x for x, n in lv if n >= (3 if name == 'CC4' else 1))
            s_ = next(x for x, n in lv if n == 1 and abs(x - t_) > 1e-6)
            line += '   triplet %.6f eV   singlet %.6f eV' % (t_ * HARTREE_EV, s_ * HARTREE_EV)
        print(line, flush=True)
else:
    mol = gto.M(atom='Br 0 0 0; H 0 0 1.4145', basis='cc-pvdz-dk', verbose=0)
    mf = x2c.UHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-8
    mf.kernel()
    print('E(SCF) = %.12f' % mf.e_tot)
    for name in ('CC2', 'CCSD', 'CC3', 'CCSDT', 'CC4', 'CCSDTQ'):
        cc, t = run(name, mf, frozen=28)
        line = '%-7s E = %.12f   (%.1f s)' % (name, cc.e_tot, t)
        if name != 'CCSDTQ':
            if name == 'CC4':
                e, _ = cc.eom_ee(nroots=2, conv_tol=1e-7)
            else:
                e, _ = cc.eom_ee(nroots=2, shift=0, conv_tol=1e-7)
            line += '   lowest totally symmetric state %.6f eV' % (np.min(e) * HARTREE_EV)
        print(line, flush=True)
