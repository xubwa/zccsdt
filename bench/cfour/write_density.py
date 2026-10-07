'''The relaxed one-particle density of zccsdt for CFOUR's property program:
in the directory of a CFOUR run (xjoda, xsox2catm, xsox2chf, xx2cvtran), the
relaxed density of the spinor CCSD (or CCSD(T)) of zccsdt.zccsd.DirectZCCSD
on PySCF's SCF of the same molecule, in the basis of CFOUR's spinors
(scf2ccmo), written as the file ``denfinal`` that xsox2cprop evaluates all
its operators with - what xx2czvec writes at the end of CFOUR's own chain.

The density is carried through the AO basis: D(CFOUR's spinors) =
C+ S D(AO) S C with CFOUR's coefficients C in PySCF's spinor basis (socutils
maps the basis), so that the two SCF calculations only have to give the same
state, not the same spinors.  If CFOUR's own denfinal is there it is kept as
denfinal.cfour and compared element by element.

usage: python bench/cfour/write_density.py RUNDIR PYSCF_BASIS [ccsd|ccsd(t)] [--run]
       --run: xsox2cprop afterwards (from $CFOUR_BIN or ~/apps/cfour-x2cso/bin),
       its output as xsox2cprop.zccsdt.out
socutils has to be on the path.
'''
import os
import shutil
import subprocess
import sys

import numpy as np
from pyscf import gto, x2c

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import props                                                            # noqa: E402
from socutils.tools.label import Labels                                 # noqa: E402

from zccsdt import cfour                                                # noqa: E402
from zccsdt.zccsd import DirectZCCSD, zvector                           # noqa: E402


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    d, basis = args[0], args[1]
    with_t = len(args) > 2 and args[2].lower() == 'ccsd(t)'
    x = cfour.read(d, pppp=False)
    ntot = x.nfrozen + x.nocc + x.nvir + x.ndropped
    mol = gto.M(atom=props.read_mol(os.path.join(d, 'MOL')), unit='Bohr', basis=basis, verbose=0)
    n2c = mol.nao_2c()
    assert ntot == n2c, (ntot, n2c)
    mf = x2c.UHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-9
    mf.kernel()
    print('E(SCF): PySCF %.12f  CFOUR %.12f  difference %.1e' % (mf.e_tot, x.e_scf, mf.e_tot - x.e_scf))
    order = np.argsort(mf.mo_energy)
    nocc = int((mf.mo_occ > 0).sum())
    frozen = list(order[:x.nfrozen]) + list(order[nocc + x.nvir:])
    mycc = DirectZCCSD(mf, frozen=frozen if x.ndropped else x.nfrozen)
    mycc.verbose = 0
    mycc.conv_tol, mycc.conv_tol_normt = 1e-11, 1e-9
    mycc.kernel()
    if with_t:
        mycc.solve_lambda(with_t='gradient')
    else:
        mycc.solve_lambda()
    relaxed, unrelaxed, z = zvector.kernel(mycc, tol=1e-10, with_t=with_t)
    C = zvector.full_space(mycc)[0]
    S = mol.intor('int1e_ovlp_spinor')
    Ccf = Labels.spinor_cfour2pyscf(mol).conj().T @ cfour._complex(os.path.join(d, 'scf2ccmo'), (n2c, n2c))
    print('E(CCSD) correlation %.12f; CFOUR spinors orthonormal in PySCF basis to %.1e'
          % (mycc.e_corr, abs(Ccf.conj().T @ S @ Ccf - np.eye(n2c)).max()))
    out = {}
    for name, dm in (('relaxed', relaxed), ('unrelaxed', unrelaxed)):
        ao = C @ dm @ C.conj().T
        out[name] = Ccf.conj().T @ S @ ao @ S @ Ccf
        dip = [-np.einsum('pq,qp->', mol.intor('int1e_r_spinor')[k], ao).real for k in range(3)]
        print('%-9s trace %.10f  electronic dipole (PySCF operator, no picture change) %s'
              % (name, np.trace(out[name]).real, np.array2string(np.array(dip), precision=9)))
    own = out['relaxed']
    path = os.path.join(d, 'denfinal')
    if os.path.exists(path) and not os.path.exists(path + '.cfour'):
        shutil.copy(path, path + '.cfour')
    if os.path.exists(path + '.cfour'):
        ref = cfour._complex(path + '.cfour', (n2c, n2c))
        print('CFOUR denfinal: trace %.10f, Hermitian to %.1e' % (np.trace(ref).real, abs(ref - ref.conj().T).max()))
        for label, cand in (('as it is', own), ('transposed', own.T), ('minus the SCF density', own - np.diag(
                (np.arange(n2c) < x.nfrozen + x.nocc).astype(float))), ('transposed, minus SCF', own.T - np.diag(
                    (np.arange(n2c) < x.nfrozen + x.nocc).astype(float)))):
            print('  zccsdt %-24s - CFOUR: max %.2e' % (label, abs(cand - ref).max()))
    cfour._write_complex(os.path.join(d, 'denfinal.zccsdt'), own)
    np.save(os.path.join(d, 'density_zccsdt.npy'), np.stack([out['relaxed'], out['unrelaxed']]))
    if '--run' in sys.argv:
        exe = os.path.join(os.environ.get('CFOUR_BIN', os.path.expanduser('~/apps/cfour-x2cso/bin')), 'xsox2cprop')
        shutil.copy(os.path.join(d, 'denfinal.zccsdt'), path)
        with open(os.path.join(d, 'xsox2cprop.zccsdt.out'), 'w') as f:
            subprocess.run([exe], cwd=d, stdout=f, stderr=subprocess.STDOUT, check=False)
        if os.path.exists(path + '.cfour'):
            shutil.copy(path + '.cfour', path)
        print('xsox2cprop with the density of zccsdt: xsox2cprop.zccsdt.out')


if __name__ == '__main__':
    main()
