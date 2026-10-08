'''Total energies of zccsdt next to those of the reference programs
(bench/refs/reference_energies.json: MRCC, CFOUR ncc/ecc/vcc at the bohr
geometry) for the small nonrelativistic molecules of bench/refs, every
method that has a reference: CCSD, CCSDT, CCSDT[Q], CCSDT(Q), CCSDT(Q)_L,
CCSDTQ, CCSDTQ(P) on spin orbitals (zccsdt.CCSDTQ and its CCSDT step) and
the closed-shell spin-adapted CCSDT and (Q) (zccsdt.CCSDT of an RHF object).

usage: python bench/energy_table.py [system ...]
'''
import json
import os
import sys
import time

from pyscf import gto, scf

import zccsdt
from zccsdt import eris as E
from zccsdt.cc import solver
from zccsdt.genbas import read_genbas

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'refs')
REFS = json.load(open(os.path.join(ROOT, 'reference_energies.json')))
SYSTEMS = {         # name: (geometry in Angstrom, basis, frozen spatial orbitals)
    'bh_sto3g': ('B 0 0 0; H 0 0 1.2324', 'sto-3g', 0),
    'lih_sto3g': ('Li 0 0 0; H 0 0 1.5957', 'sto-3g', 0),
    'ch2_sto3g': ('C 0 0 0; H 0 0.87 0.67; H 0 -0.87 0.67', 'sto-3g', 0),
    'h2o_sto3g': ('O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', 'sto-3g', 0),
    'n2_sto3g': ('N 0 0 0; N 0 0 1.0977', 'sto-3g', 0),
    'h2o_631g': ('O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', '6-31G', 0),
    'n2_631g': ('N 0 0 0; N 0 0 1.0977', '6-31G', 0),
    'hf_ccpvdz': ('F 0 0 0; H 0 0 0.917', 'cc-pvdz', 0),
    'h2o_ccpvdz_fc': ('O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', 'cc-pvdz', 1),
    'h2o_ccpvdz': ('O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', 'cc-pvdz', 0),
}


def basis_of(geom, name):
    if name != '6-31G':             # the GENBAS 6-31G differs from PySCF's built-in one at 1e-7
        return name
    atoms = sorted({a.split()[0] for a in geom.split(';')})
    return {el: read_genbas(os.path.join(ROOT, 'GENBAS'), '%s:6-31G' % el.upper()) for el in atoms}


def references(name):
    '''{method: {program: energy}} at the bohr geometry.'''
    d, out = REFS[name], {}
    for prog, v in (('MRCC', d.get('mrcc', {}).get('bohr', {})),
                    ('CFOUR ncc', d.get('cfour', {}).get('NCC', {}).get('bohr', {})),
                    ('CFOUR ecc', d.get('cfour', {}).get('ECC', {}).get('bohr', {})),
                    ('CFOUR vcc', d.get('cfour', {}).get('VCC', {}).get('bohr', {}))):
        for key, val in v.items():
            if key.startswith('e_') and isinstance(val, float) and not key.endswith('_corr') and key != 'e_nuc':
                out.setdefault(key[2:], {})[prog] = val
    return out


def run(name):
    geom, basis, ncore = SYSTEMS[name]
    ref = references(name)
    mol = gto.M(atom=geom, basis=basis_of(geom, basis), verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-9
    mf.kernel()
    ours = {'scf': {'': mf.e_tot}}
    t0 = time.time()
    eris = E.from_scf(scf.addons.convert_to_ghf(mf), frozen=2 * ncore)
    r = solver.kernel(eris, 'ccsd', solver.Options(conv_tol=1e-12, conv_tol_normt=1e-10, max_cycle=300))
    ours['ccsd'] = {'spin orbitals': r.e_tot}
    full = 'ccsdtq' in ref
    cc = (zccsdt.CCSDTQ if full else zccsdt.CCSDT)(mf.to_ghf() if not full else mf, frozen=ncore if full else 2 * ncore)
    cc.verbose = 0
    cc.conv_tol, cc.conv_tol_normt, cc.max_cycle = 1e-12, 1e-10, 300
    cc.kernel()
    c3 = cc.ccsdt_run if full else cc
    ours['ccsdt'] = {'spin orbitals': c3.e_tot}
    eq = c3.ccsdt_q()
    ours['ccsdt[q]'] = {'spin orbitals': c3.e_tot + c3.e_q5}
    ours['ccsdt(q)'] = {'spin orbitals': c3.e_tot + eq}
    if 'ccsdt(q)_l' in ref:
        ours['ccsdt(q)_l'] = {'spin orbitals': c3.e_tot + c3.ccsdt_q_lambda()}
    if full:
        ours['ccsdtq'] = {'spin orbitals': cc.e_tot}
        if 'ccsdtq(p)' in ref:
            ours['ccsdtq(p)'] = {'spin orbitals': cc.e_tot + cc.ccsdtq_p()}
    rc = zccsdt.CCSDT(mf, frozen=ncore)           # closed shell, spin adapted
    rc.verbose = 0
    rc.conv_tol, rc.conv_tol_normt, rc.max_cycle = 1e-12, 1e-10, 300
    rc.kernel()
    ours['ccsdt']['closed shell'] = rc.e_tot
    try:
        eq = rc.ccsdt_q()
        ours['ccsdt(q)']['closed shell'] = rc.e_tot + eq
    except Exception as err:                      # noqa: BLE001
        print('   (closed-shell (Q) not run: %s)' % err)
    print('== %s: %s; %d electrons, %d orbitals, %d frozen; %.0f s' % (name, REFS[name]['description'][:60], mol.nelectron,
                                                                         mol.nao, ncore, time.time() - t0), flush=True)
    for method in ('scf', 'ccsd', 'ccsdt', 'ccsdt[q]', 'ccsdt(q)', 'ccsdt(q)_l', 'ccsdtq', 'ccsdtq(p)'):
        if method not in ours:
            continue
        for prog, val in ref.get(method, {}).items():
            print('  %-11s %-10s %19.12f' % (method, prog, val))
        base = ref.get(method, {}).get('MRCC')
        for how, val in ours[method].items():
            print('  %-11s zccsdt %-14s %19.12f%s' % (method, how, val, '' if base is None else '   - MRCC %9.1e' % (val - base)))
    sys.stdout.flush()


if __name__ == '__main__':
    for name in (sys.argv[1:] or list(SYSTEMS)):
        run(name)
