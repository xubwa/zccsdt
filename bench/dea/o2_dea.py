'''EOM-DEA of O2 on a Hamiltonian file (bench/dea/common.py): the lowest root
of the blocks 2 Omega = 0 and 2 Omega = 2 (parity g) at every level, total
energies and their difference - the zero-field splitting of X 3Sigma_g-.

usage: python o2_dea.py <file.npz> <out.json> [level ...] [--exact E0 E1] [--drop] [--tol 1e-9] [--blocks]
                        [--ghost] [--mem MB] [--freeze n]
--blocks: in the blocks of the labels 2 Omega: the levels ccsd, ccsd+t3, ccsdt, ccsdt+t4 on the sector
          itself (zccsdt.cc.dea_sector), the others through two ghost spinors (zccsdt.cc.dea.ProblemSym)
--ghost:  with --blocks, every level through the ghost spinors
--mem: the memory of the EOM-DEA problem (MB; default 2000)
levels: ccsd ccsd+t3 cc3 ccsdt cc4 ccsdt+t4cc4 ccsdt+t4 (default: all)'''
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def main():
    args = sys.argv[1:]
    exact, drop, tol = None, False, 1e-9
    if '--exact' in args:
        k = args.index('--exact')
        exact = (float(args[k + 1]), float(args[k + 2]))
        del args[k:k + 3]
    if '--drop' in args:
        args.remove('--drop')
        drop = True
    sort = '--blocks' in args
    if sort:
        args.remove('--blocks')
    if '--tol' in args:
        k = args.index('--tol')
        tol = float(args[k + 1])
        del args[k:k + 2]
    backend = 'sector'
    if '--ghost' in args:
        args.remove('--ghost')
        backend = 'ghost'
    if '--sector' in args:          # (the default)
        args.remove('--sector')
    mem = 2000.0
    if '--mem' in args:
        k = args.index('--mem')
        mem = float(args[k + 1])
        del args[k:k + 2]
    freeze = 0
    if '--freeze' in args:
        k = args.index('--freeze')
        freeze = int(args[k + 1])
        del args[k:k + 2]
    path, out = args[0], args[1]
    levels = args[2:] or list(common.LEVELS)
    t0 = time.time()
    eris, lab, par, info = common.load(path, drop_forbidden=drop, sort=sort, freeze=freeze)
    no = eris.nocc
    print('file %s: %d spinors, %d occupied; E(reference) = %.12f; %s integrals (max |imag| %.1e)'
          % (path, info['n'], no, info['e_ref'], 'real' if np.asarray(eris.oovv).dtype.kind == 'f' else 'complex',
             info['imag']))
    print('  Fock matrix: max |f_ov| %.2e, max off-diagonal |f_oo|, |f_vv| %.2e; largest integral that the labels '
          'forbid %.2e%s' % (info['fock_ov'], info['fock_offdiag_oo_vv'], info['forbidden'],
                             ' (set to zero)' if drop else ''))
    print('  labels 2 Omega: occupied %s' % lab[:no])
    print('                  virtual  %s' % lab[no:])
    if par is not None:
        print('  parity (0 g, 1 u): occupied %s virtual %s' % (par[:no], par[no:]))
    print('  thresholds: ground state |dT| < 1e-11, EOM residual norm < %.0e' % tol, flush=True)
    res = common.run_levels(eris, lab, par, levels, conv_tol=tol, out=out, backend=backend, mem_mb=mem)
    print()
    for line in common.table(res, exact):
        print(line)
    print('per level: sigma vectors (2 Omega = 0, 2), seconds per sigma vector, ground state seconds, elements in the masks')
    for lev, r in res.items():
        print('  %-12s %s  %s  %.1f  %s' % (lev, list(r['nsigma'].values()),
                                            ['%.2f' % x for x in r['sigma_seconds'].values()], r['ground_seconds'],
                                            list(r['sizes'].values())))
    print('total %.0f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
