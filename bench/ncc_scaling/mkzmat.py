'''ZMAT and GENBAS entries of a CFOUR ncc run at the nuclei of a geometry
file, with the basis from PySCF's data (the entries of bench/cfour/genbas.py),
so that CFOUR and zccsdt (zc_run.py, same file, same basis name) run the same
nuclei and the same primitives.

usage: python bench/ncc_scaling/mkzmat.py GEOMETRY BASIS GENBAS ZMAT KEY=VALUE ...
  GEOMETRY: 'symbol x y z' per atom in bohr;  BASIS: PySCF name
  GENBAS:   file to append the entries '<ATOM>:ZC<NAME>' to
  ZMAT:     file to write
  keys: calc (CCSDT | CCSDT(Q)), sym (ON | OFF), frozen (ON | OFF),
        conv (CC_CONV), maxcyc (CC_MAXCYC), mem (MEMORY_SIZE in GB)
'''
import os
import re
import sys

from pyscf import gto

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cfour'))
from genbas import entry                                                # noqa: E402


def main():
    geometry, basis, genbas, zmat = sys.argv[1:5]
    opt = dict(calc='CCSDT', sym='ON', frozen='OFF', conv='10', maxcyc='200', mem='24')
    opt.update(a.split('=', 1) for a in sys.argv[5:])
    with open(geometry) as f:
        text = f.read()
    mol = gto.M(atom=text, unit='Bohr', basis=basis, verbose=0)
    name = 'ZC' + re.sub('[^A-Z0-9]', '', basis.upper())
    symbols = [mol.atom_pure_symbol(i) for i in range(mol.natm)]
    entries = []
    for s in dict.fromkeys(symbols):
        entries += entry(s, name, mol._basis[s])
    with open(genbas, 'a') as f:
        f.write('\n' + '\n'.join(entries) + '\n')
    lines = ['%s %s %s' % (os.path.basename(geometry), opt['calc'], basis)]
    lines += ['%s %.12f %.12f %.12f' % (s.upper(), *xyz) for s, xyz in zip(symbols, mol.atom_coords())]
    lines += ['',
              '*CFOUR(CALC=%s,BASIS=SPECIAL,REF=RHF,CC_PROG=NCC,MULT=1' % opt['calc'],
              'COORDINATES=CARTESIAN,UNITS=BOHR,SYMMETRY=%s' % opt['sym'],
              'FROZEN_CORE=%s,SCF_CONV=10,CC_CONV=%s,CC_MAXCYC=%s' % (opt['frozen'], opt['conv'], opt['maxcyc']),
              'SCF_MAXCYC=300,MEMORY_SIZE=%s,MEM_UNIT=GB)' % opt['mem'],
              '']
    lines += ['%s:%s' % (s.upper(), name) for s in symbols]
    with open(zmat, 'w') as f:
        f.write('\n'.join(lines) + '\n\n')
    print('%s %s: %d basis functions, %d electrons' % (geometry, basis, mol.nao_nr(), mol.nelectron))


if __name__ == '__main__':
    main()
