'''A run directory of the native program (cfour/xzccsdt) for EOM-DEA on a
Hamiltonian file of O2(2+) (bench/dea/common.py): the integrals of the
reference determinant in CFOUR's file formats, the labels 2 Omega
(xzlabels) and the parities (xzparity), and a JOBARC of CFOUR's xjoda for a
ZMAT that asks for CALC=CCSDT and the double electron attachment (section
%xzdea; docs/native.md, section 20).

The X2Ccorr Hamiltonians of O2 are not made by CFOUR's chain: xjoda runs on
a stand-in molecule (HF / 6-31G of cfour/merge/test, whose GENBAS is
copied) only for JOBARC - joda's flags (CALC, CC_CONV, ESTATE_CONV, ...,
MEMORY_SIZE); the integral files, scf2ceig and nmototal are written here
(zccsdt.cfour.write) for the spinors of the Hamiltonian file, in the order
of their labels, and E(SCF) (SCFENEG) and the frozen pairs (NUMDROPA, 0)
are put into that JOBARC.  Nothing of the stand-in molecule is read by the
native program.

usage: python bench/dea/native_rundir.py <file.npz> <outdir> [--freeze n] [--targets 0,2] [--roots 1]
           [--cc-conv 10] [--estate-conv 9] [--mem-gb 8] [--section text] [--cfour-bin dir]
  --freeze n      the first n spinors of the reference frozen (their mean field folded in; common.load)
  --targets       2 Omega of the two attached electrons, one problem each (default 0,2)
  --roots         roots per target (default 1)
  --cc-conv       CC_CONV (default 10)
  --estate-conv   ESTATE_CONV: the largest element of a residual below 10^-N (default 9)
  --mem-gb        MEMORY_SIZE in GB (default 8)
  --xz line       a line of the section %xzccsdt (repeatable)
'''
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = os.path.join(HERE, '..', '..', 'cfour', 'merge', 'test')
BIN = '/home/xwang405/apps/cfour-x2cso/bin'

ZMAT = '''stand-in molecule (the integrals are those of {src})
F 0.000000000000 0.000000000000 0.000000000000
H 0.000000000000 0.000000000000 1.732800000000

*CFOUR(CALC=CCSDT,BASIS=SPECIAL
SYM=OFF,COORDINATES=CARTESIAN,UNITS=BOHR,CHARGE=0,MULT=1
RELATIVISTIC=X2C1E,X2CKEY=SOATMX,ABCDTYPE=STANDARD
SCF_CONV=10,SCF_MAXCYC=200
CC_CONV={cc},CC_MAXCYC=100
ESTATE_CONV={es},ESTATE_MAXCYC=120
MEMORY_SIZE={mem},MEM_UNIT=GB)

F:6-31G
H:6-31G

%xzdea
{targets}
parity g
{extra}
'''


def write_parity(directory, par):
    '''The file xzparity: one parity per spinor of the SCF (0 g, 1 u), the
    order of scf2ceig, the format of xzlabels without the modulus.'''
    with open(os.path.join(directory, 'xzparity'), 'w') as f:
        f.write('# parity per spinor (0 g, 1 u; order of scf2ceig); first line: spinors\n')
        f.write('%d\n' % len(par))
        for k in range(0, len(par), 16):
            f.write(' '.join('%d' % x for x in par[k:k + 16]) + '\n')


def main():
    args = sys.argv[1:]
    opt = {'--freeze': '0', '--targets': '0,2', '--roots': '1', '--cc-conv': '10', '--estate-conv': '9',
           '--mem-gb': '8', '--cfour-bin': BIN}
    xz = []
    while '--xz' in args:
        k = args.index('--xz')
        xz.append(args[k + 1])
        del args[k:k + 2]
    for key in list(opt):
        if key in args:
            k = args.index(key)
            opt[key] = args[k + 1]
            del args[k:k + 2]
    path, out = args[0], args[1]
    from zccsdt import cfour
    eris, lab, par, info = common.load(path, sort=True, freeze=int(opt['--freeze']))
    no, n = eris.nocc, info['n']
    print('%s: %d spinors, %d occupied, frozen %s; E(reference) %.12f; largest |imag| %.1e, |f_ov| %.1e, '
          'forbidden integral %.1e' % (path, n, no, opt['--freeze'], info['e_ref'], info['imag'], info['fock_ov'],
                                       info['forbidden']))
    # (the files have the spinor energies only: canonical spinors; f_ov of the SCF is left out)
    f = np.asarray(eris.fock)
    eris.fock = np.diag(np.diag(f))
    os.makedirs(out, exist_ok=True)
    targets = '\n'.join('%s %s' % (t, opt['--roots']) for t in opt['--targets'].split(','))
    extra = ('%xzccsdt\n' + '\n'.join(xz) + '\n') if xz else ''
    with open(os.path.join(out, 'ZMAT'), 'w') as fz:
        fz.write(ZMAT.format(src=os.path.basename(path), cc=opt['--cc-conv'], es=opt['--estate-conv'],
                             mem=opt['--mem-gb'], targets=targets, extra=extra))
    shutil.copy(os.path.join(TEST, 'GENBAS'), out)
    with open(os.path.join(out, 'xjoda.out'), 'w') as log:
        subprocess.run([os.path.join(opt['--cfour-bin'], 'xjoda')], cwd=out, stdout=log, stderr=subprocess.STDOUT,
                       check=True)
    tmp = os.path.join(out, 'written')
    cfour.write(tmp, eris, nfrozen=0)
    for name in os.listdir(tmp):
        if name not in ('JOBARC', 'JAINDX'):
            shutil.move(os.path.join(tmp, name), os.path.join(out, name))
    shutil.rmtree(tmp)
    arc = cfour.JobArc(out)
    arc.put('SCFENEG', np.array([eris.e_hf], dtype='<f8'))
    arc.put('NUMDROPA', np.array([0], dtype='<i8'))
    if par is not None:
        write_parity(out, np.asarray(par, dtype=np.int64))
    print('written: %s (labels 2 Omega: occupied %s, virtual %d spinors from %d to %d)'
          % (out, lab[:no], n - no, lab[no:].min(), lab[no:].max()))


if __name__ == '__main__':
    main()
