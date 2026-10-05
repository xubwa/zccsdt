'''The geometry and the basis of a molecule of the bench scripts for CFOUR,
from PySCF's data, so that both programs run the same primitives: entries
of a GENBAS file and the lines of the ZMAT.

usage: python bench/cfour/genbas.py MOLECULE BASIS GENBAS GEOMETRY LINES
  MOLECULE: a key of the GEOM of bench/spinors.py; BASIS: a PySCF name, or
  'heavy,light' (bench/spinors.basis_of)
appends the entries '<ATOM>:ZC<NAME>' to the file GENBAS and writes the files
GEOMETRY ('ATOM x y z' per atom, in bohr - UNITS=BOHR -: joda converts
Angstrom with 0.5291771, PySCF with 0.52917721092, a geometry 2e-7 larger,
which is 1e-8 in the energy and 5e-8 in a dipole moment) and LINES
('ATOM:ZC<NAME>' per atom, for BASIS=SPECIAL).
'''
import os
import re
import sys

import numpy as np
from pyscf import gto

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import spinors                                                          # noqa: E402


def entry(symbol, name, shells):
    '''The GENBAS entry (lines) of the shells [l, [exponent, c1, c2, ..], ..]
    of PySCF: per angular momentum all primitives and the matrix of the
    contraction coefficients (primitives by contracted functions).'''
    by_l = {}
    for shell in shells:
        l, rows = shell[0], shell[1:]
        if not isinstance(rows[0], (list, tuple)):      # [l, kappa, ..]: 0 is both j
            if rows[0] != 0:
                raise NotImplementedError('shells of one kappa')
            rows = rows[1:]
        by_l.setdefault(l, []).append(np.array(rows, dtype=float))
    blocks = []
    for l in sorted(by_l):
        exps = sorted({e for sh in by_l[l] for e in sh[:, 0]}, reverse=True)
        where = {e: i for i, e in enumerate(exps)}
        cols = []
        for sh in by_l[l]:
            for k in range(1, sh.shape[1]):
                col = np.zeros(len(exps))
                for e, c in zip(sh[:, 0], sh[:, k]):
                    col[where[e]] += c
                cols.append(col)
        blocks.append((exps, np.array(cols).T))
    out = ['%s:%s' % (symbol.upper(), name), 'from PySCF (bench/cfour/genbas.py)', '',
           '%3d' % len(blocks), ''.join('%5d' % l for l in sorted(by_l)),
           ''.join('%5d' % c.shape[1] for _, c in blocks),
           ''.join('%5d' % len(e) for e, _ in blocks), '']
    for exps, coef in blocks:
        out += [' '.join('%.12g' % e for e in exps[i:i + 5]) for i in range(0, len(exps), 5)]
        out.append('')
        for row in coef:
            out += [' '.join('%.10f' % c for c in row[i:i + 7]) for i in range(0, len(row), 7)]
        out.append('')
    return out


def main():
    mol_name, basis, genbas, geometry, lines = sys.argv[1:6]
    atoms = spinors.atoms_of(mol_name)
    basis = spinors.basis_of(mol_name, basis)
    mol = gto.M(atom=spinors.GEOM[mol_name], basis=basis, verbose=0,
                spin=spinors.SPIN.get(mol_name, 0))
    names, entries = {}, []
    for symbol, _ in atoms:
        if symbol not in names:
            b = basis[symbol] if isinstance(basis, dict) else basis
            names[symbol] = 'ZC' + re.sub('[^A-Z0-9]', '', b.upper())
            entries += entry(symbol, names[symbol], mol._basis[symbol])
    with open(genbas, 'a') as f:
        f.write('\n' + '\n'.join(entries) + '\n')
    with open(geometry, 'w') as f:
        f.write('\n'.join('%s %.12f %.12f %.12f' % (s.upper(), *xyz)
                          for (s, _), xyz in zip(atoms, mol.atom_coords())) + '\n')
    with open(lines, 'w') as f:
        f.write('\n'.join('%s:%s' % (s.upper(), names[s]) for s, _ in atoms) + '\n')
    print('%s: %d basis functions (%d spinors), %d electrons; GENBAS entries %s'
          % (mol_name, mol.nao_nr(), mol.nao_2c(), mol.nelectron,
             ', '.join('%s:%s' % (s.upper(), n) for s, n in names.items())))


if __name__ == '__main__':
    main()
