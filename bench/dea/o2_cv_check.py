'''The 1s contribution of O2 (all electrons minus 1s frozen): checks that the
two calculations of a basis share the orbital basis and the spinors.

usage: python o2_cv_check.py check <dir> <basis>        (basis: pwcvtz_x2c_tp, ...)
       python o2_cv_check.py scf <dir> <basis>          the all-electron SCF with output (checkpoint
                                                        <dir>/o2_2plus_krhf_x2cmp_<basis>_ae.chk)

Prints: AO functions and AO spinors of the basis (contracted, and of the
uncontracted basis of the X2C step), the smallest eigenvalue of the overlap
of each, the number of spinors of each checkpoint (all AO spinors are kept
when it equals the AO spinors: no linear-dependence removal), E(SCF) and the
largest difference of the coefficients and energies of the two checkpoints
(<basis>_ae and <basis>, the frozen-core tag), and for the block files in
<dir> their spinors, occupied spinors, ecore, E(HF) in the correlated space.'''
import os
import sys

import numpy as np

sys.argv, _args = sys.argv[:1], sys.argv[1:]
mode, d, basis = _args[0], _args[1], _args[2]
sys.argv += ['--basis', basis]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import o2_scf  # noqa: E402
from pyscf import gto  # noqa: E402
from pyscf.lib import chkfile  # noqa: E402


def main():
    mol = gto.M(atom='O 0 0 0; O 0 0 1.20752', basis={'O': o2_scf.basis_of()}, verbose=0, nucmod='G', charge=2,
                spin=0)
    s = mol.intor('int1e_ovlp_spinor')
    w = np.linalg.eigvalsh(s)
    umol, _ = mol.decontract_basis() if hasattr(mol, 'decontract_basis') else (None, None)
    print('%s: %d AO functions (spherical), %d AO spinors; smallest eigenvalue of the spinor overlap %.3e'
          % (basis, mol.nao_nr(), mol.nao_2c(), w.min()))
    if umol is not None:
        wu = np.linalg.eigvalsh(umol.intor('int1e_ovlp_spinor'))
        print('uncontracted (X2C step): %d AO spinors; smallest eigenvalue %.3e' % (umol.nao_2c(), wu.min()))
    chks = {tag: os.path.join(d, 'o2_2plus_krhf_x2cmp_%s%s.chk' % (basis, tag)) for tag in ('_ae', '')}
    data = {}
    for tag, path in chks.items():
        if not os.path.exists(path):
            print('%s: missing' % path)
            continue
        x = chkfile.load(path, 'scf')
        data[tag] = x
        print('%s: %d x %d coefficients (spinors kept %d of %d AO spinors), occupied %d, E(SCF) %.12f'
              % (os.path.basename(path), x['mo_coeff'].shape[0], x['mo_coeff'].shape[1], x['mo_coeff'].shape[1],
                 mol.nao_2c(), int((np.asarray(x['mo_occ']) > 0).sum()), float(x['e_tot'])))
    if len(data) == 2:
        a, b = data['_ae'], data['']
        print('the two checkpoints: max |dC| %.1e, max |d eps| %.1e, dE(SCF) %.1e'
              % (abs(a['mo_coeff'] - b['mo_coeff']).max(), abs(a['mo_energy'] - b['mo_energy']).max(),
                 float(a['e_tot']) - float(b['e_tot'])))
    from zccsdt import hblocks
    for name in sorted(os.listdir(d)):
        if name.endswith('_x2ccorr.h5'):
            bf = hblocks.BlockFile(os.path.join(d, name))
            print('%s: %d spinors, %d occupied, %d frozen spinors folded in (mo_frozen), ecore %.12f, E(HF) %.12f, '
                  'E(reference) %.12f' % (name, bf.n, bf.ncore, bf['mo_frozen'].shape[1], bf.ecore, bf.attr('e_hf'),
                                          bf.reference_energy(bf.h1)))
            bf.close()


if __name__ == '__main__':
    from pyscf import lib
    lib.num_threads(int(os.environ.get('OMP_NUM_THREADS', 1)))
    if mode == 'scf':
        mf = o2_scf.run_scf(os.path.join(d, 'o2_2plus_krhf_x2cmp_%s_ae.chk' % basis), verbose=4)
        print('E(SCF) %.12f, %d spinors of %d AO spinors' % (mf.e_tot, mf.mo_coeff.shape[1], mf.mol.nao_2c()))
    else:
        main()
