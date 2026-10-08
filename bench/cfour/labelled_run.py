'''A CFOUR run directory with symmetry labels, made here: the spinors of a
linear molecule adapted to m_j (zccsdt.eris.axial_spinors: real integrals), in
the order of their labels, their MO integrals in CFOUR's file formats
(zccsdt.cfour.write) and the labels 2 m_j in the file xzlabels - test data
for the native program cfour/xzccsdt and for the Python program on the same
files, without a symmetry-adapted SCF in CFOUR.

usage: python bench/cfour/labelled_run.py CHAINDIR OUTDIR [phases]
  CHAINDIR: a directory in which xjoda (at least) has run: its MOL, GENBAS and
            ZMAT give the molecule, its JOBARC the frozen spinors and joda's
            flags.  The SCF is PySCF's X2C-1e here (not CFOUR's spinors).
  OUTDIR:   gets a copy of JOBARC, JAINDX, ZMAT, ... of CHAINDIR and the
            integral files, scf2ceig, nmototal and xzlabels written here;
            SCFENEG of its JOBARC is the SCF energy of this SCF.
  phases:   multiply every active spinor by a phase exp(i x), x random with
            this seed: complex integrals with the same labels.

and: python bench/cfour/labelled_run.py --run OUTDIR METHOD [dense]
  the Python program on the files of OUTDIR, in the blocks of the labels
  (default) or with the labels not used (dense); prints the energies.
'''
import os
import shutil
import sys

import numpy as np


def make(chain, out, seed=None):
    from pyscf import x2c
    import zccsdt
    from zccsdt import cfour
    mol = cfour.molecule(chain)
    mf = x2c.UHF(mol)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-8, 200
    mf.kernel()
    assert mf.converged
    nfrozen = 2 * int(cfour.JobArc(chain).get('NUMDROPA')[0])
    cc = zccsdt.GCCSDT(mf, frozen=nfrozen)
    cc.real_spinors = cc.symmetry = True
    eris = cc.ao2mo()
    if getattr(eris, 'labels', None) is None:
        raise SystemExit('the spinors of this SCF are not eigenfunctions of j_z')
    no, n = eris.nocc, eris.nmo
    print('%d occupied, %d virtual spinors, %d frozen; labels (2 m_j): occupied %s, virtual %s'
          % (no, n - no, nfrozen, eris.labels[:no], eris.labels[no:]))
    if seed is not None:
        ph = np.exp(1j * np.random.default_rng(int(seed)).uniform(0, 2 * np.pi, n))
        sl = {'o': slice(0, no), 'v': slice(no, n)}
        for key in ('oooo', 'ooov', 'oovv', 'ovov', 'ovvo', 'ovvv'):
            f = [ph[sl[c]] for c in key]
            x = np.asarray(getattr(eris, key)).astype(np.complex128)
            x *= (f[0].conj()[:, None, None, None] * f[1].conj()[None, :, None, None]
                  * f[2][None, None, :, None] * f[3][None, None, None, :])
            setattr(eris, key, x)
        a, b = np.triu_indices(n - no, 1)
        pp = ph[no:][a] * ph[no:][b]
        eris._vvvv_pp = np.asarray(eris._vvvv_pp) * (pp.conj()[:, None] * pp[None, :])
        eris.fock = np.asarray(eris.fock).astype(np.complex128)
    os.makedirs(out, exist_ok=True)
    for name in os.listdir(chain):
        if os.path.isfile(os.path.join(chain, name)) and not name.startswith(('moin', 'PPPH', 'XZ')):
            shutil.copy(os.path.join(chain, name), out)
    tmp = os.path.join(out, 'written')
    cfour.write(tmp, eris, nfrozen=nfrozen)
    for name in os.listdir(tmp):
        if name not in ('JOBARC', 'JAINDX'):
            shutil.move(os.path.join(tmp, name), os.path.join(out, name))
    shutil.rmtree(tmp)
    cfour.JobArc(out).put('SCFENEG', np.array([eris.e_hf], dtype='<f8'))
    print('E(SCF) = %.12f' % eris.e_hf)


def run(directory, method, dense=False):
    import zccsdt
    from zccsdt import cfour
    eris = cfour.from_cfour(directory, labels=True)
    j = cfour.job(directory, method)
    if dense:
        eris.use_labels = False
    cc = zccsdt.GCCSDT.from_integrals(eris, verbose=4)
    cc.symmetry = not dense
    cc.conv_tol, cc.conv_tol_normt, cc.max_cycle = j.conv_tol, j.conv_tol_normt, j.max_cycle
    if j.max_memory:
        cc.max_memory = j.max_memory
    cc.kernel()
    print('E(CCSDT)      = %.12f' % cc.e_tot)
    if j.method in ('ccsdt[q]', 'ccsdt(q)'):
        e_q = cc.ccsdt_q()
        print('E(CCSDT[Q])   = %.12f' % (cc.e_tot + cc.e_q5))
        print('E(CCSDT(Q))   = %.12f' % (cc.e_tot + e_q))


if __name__ == '__main__':
    if sys.argv[1] == '--run':
        run(sys.argv[2], sys.argv[3], len(sys.argv) > 4 and sys.argv[4] == 'dense')
    else:
        make(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
