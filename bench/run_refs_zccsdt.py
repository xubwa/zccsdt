'''zccsdt CCSDT energies for the systems in bench/refs (RHF -> GHF reference).'''
import json
import sys
import time
from pyscf import gto, scf

from zccsdt import eris as E
from zccsdt.cc import solver

GEOM = {
    'h2o': 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
    'n2': 'N 0 0 0; N 0 0 1.0977',
    'hf': 'F 0 0 0; H 0 0 0.917',
}
SYSTEMS = {                       # name: (molecule, basis, frozen spin orbitals)
    'h2o_631g': ('h2o', '6-31g', 0),
    'h2o_ccpvdz': ('h2o', 'cc-pvdz', 0),
    'h2o_ccpvdz_fc': ('h2o', 'cc-pvdz', 2),
    'n2_631g': ('n2', '6-31g', 0),
    'hf_ccpvdz': ('hf', 'cc-pvdz', 0),
}


def run(name):
    molname, basis, frozen = SYSTEMS[name]
    mol = gto.M(atom=GEOM[molname], basis=basis, verbose=0)
    rhf = scf.RHF(mol).run(conv_tol=1e-12)
    eris = E.from_scf(scf.addons.convert_to_ghf(rhf), frozen=frozen)
    opts = solver.Options(conv_tol=1e-11, conv_tol_normt=1e-8, max_cycle=300)
    t = time.time()
    rd = solver.kernel(eris, 'ccsd', opts)
    rt = solver.kernel(eris, 'ccsdt', opts)
    return dict(e_scf=rhf.e_tot, e_ccsd=rd.e_tot, e_ccsdt=rt.e_tot,
                converged=bool(rd.converged and rt.converged), ncycle=rt.ncycle,
                seconds=time.time() - t)


if __name__ == '__main__':
    names = sys.argv[1:] or list(SYSTEMS)
    out = {n: run(n) for n in names}
    for n, r in out.items():
        print('%-14s scf %.10f ccsd %.10f ccsdt %.10f conv %s (%d cyc, %.0fs)'
              % (n, r['e_scf'], r['e_ccsd'], r['e_ccsdt'], r['converged'],
                 r['ncycle'], r['seconds']))
    with open('bench/refs/zccsdt_energies.json', 'w') as f:
        json.dump(out, f, indent=1)
