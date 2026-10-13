'''EOM-DEA of O2 on a Hamiltonian file by the reference route on determinants
(zccsdt.cc.dea_strings): any number of holes of the operator, any T.

usage: python o2_strings.py <file.npz> <out.json> <T: ccsd | ccsdt | ccsdtq> <hmax> [hmax ...] [--tol 1e-9] [--drop]
hmax: 1 (R to 3p1h), 2 (4p2h), 3 (5p3h), 4 (6p4h), ..'''
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

NAMES = {1: '2p + 3p1h', 2: '.. + 4p2h', 3: '.. + 5p3h', 4: '.. + 6p4h', 5: '.. + 7p5h', 6: '.. + 8p6h'}


def main():
    from zccsdt.cc import ccsdtq_ref as R, ccsdtq_terms as Q, dea_strings as DS
    args = sys.argv[1:]
    tol = 1e-9
    if '--tol' in args:
        k = args.index('--tol')
        tol = float(args[k + 1])
        del args[k:k + 2]
    drop = '--drop' in args
    if drop:
        args.remove('--drop')
    path, out, model = args[0], args[1], args[2]
    hmaxs = [int(x) for x in args[3:]]
    t0 = time.time()
    eris, lab, par, info = common.load(path, drop_forbidden=drop)
    print('  largest integral that the labels forbid %.1e%s; max |f_ov| %.1e' % (info['forbidden'], ' (set to zero)' if drop else '', info['fock_ov']))
    no = eris.nocc
    nv = info['n'] - no
    print('file %s: %d + %d spinors; E(reference) = %.12f; T of %s' % (path, no, nv, info['e_ref'], model.upper()), flush=True)
    st = None
    for m in {'ccsd': ['ccsd'], 'ccsdt': ['ccsd', 'ccsdt'], 'ccsdtq': ['ccsd', 'ccsdt', 'ccsdtq']}[model]:
        st = common.ground(eris, m, start=st)
    e_cc = eris.e_hf + st['e_corr']
    print('ground state %s: E = %.11f (%.1f s)' % (model.upper(), e_cc, st['seconds']), flush=True)
    amps = [st['t1'], Q.unique(st['t2'], 2)]
    if st['t3'] is not None:
        amps.append(np.asarray(st['t3']))
    if st['t4'] is not None:
        amps.append(np.asarray(st['t4']))
    fock, g, _, _ = R.integrals(eris)
    fock, g = np.asarray(fock).real, np.asarray(g).real
    results = {}
    for hmax in hmaxs:
        rec = {'E': {}, 'residual': {}, 'nsigma': {}, 'sigma_seconds': {}, 'sizes': {}, 'weights': {}, 'T': model,
               'operator': NAMES.get(hmax, str(hmax)), 'e_cc': e_cc}
        for tgt in (0, 2):
            pb = DS.Problem(fock, g, no, amps, hmax, labels=lab, parity=par, target=tgt, e_ref=eris.e_hf)
            e, res = DS.kernel(pb, nroots=1, nguess=4, conv_tol=tol, verbose=True, max_cycle=150)
            v = res.vectors[0]
            w = np.array([float(np.vdot(p, p)) for p in v])
            rec['E'][tgt], rec['residual'][tgt] = float(e[0]), float(res.residuals[0])
            rec['nsigma'][tgt], rec['sigma_seconds'][tgt] = pb.nsigma, pb.seconds / max(pb.nsigma, 1)
            rec['sizes'][tgt], rec['weights'][tgt] = pb.sizes, [float(x) for x in w / w.sum()]
            print('  2 Omega = %d: E = %.11f  residual %.1e%s  %d sigma vectors, %.1f s each; sizes %s; weights %s'
                  % (tgt, e[0], res.residuals[0], '' if res.converged.all() else ' NOT CONVERGED', pb.nsigma,
                     rec['sigma_seconds'][tgt], pb.sizes, ' '.join('%.5f' % x for x in rec['weights'][tgt])), flush=True)
            del pb
        rec['split'] = rec['E'][2] - rec['E'][0]
        print('LEVEL T %-6s R %-10s E(2 Omega = 0) = %.11f  E(2 Omega = 2) = %.11f  split = %.4e Eh = %.4f cm-1'
              % (model, rec['operator'], rec['E'][0], rec['E'][2], rec['split'], rec['split'] * common.H2CM), flush=True)
        results[hmax] = rec
        with open(out, 'w') as f:
            json.dump(results, f, indent=1)
    print('total %.0f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
