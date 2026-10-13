'''EOM-DEA of O2 on a Hamiltonian file with the 5p3h operators at lowest
order, folded (zccsdt.cc.dea5), on top of the complete rows to 4p2h.

usage: python o2_fold5.py <file.npz> <out.json> <T: ccsdt | ccsdtq> [--blocks] [--tol 1e-9]'''
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def main():
    from zccsdt.cc import dea, dea5
    args = sys.argv[1:]
    tol = 1e-9
    if '--tol' in args:
        k = args.index('--tol')
        tol = float(args[k + 1])
        del args[k:k + 2]
    blocks = '--blocks' in args
    if blocks:
        args.remove('--blocks')
    path, out, model = args[0], args[1], args[2]
    t0 = time.time()
    eris, lab, par, info = common.load(path, sort=blocks)
    print('file %s: %d spinors; E(reference) = %.12f; T of %s; %s' % (path, info['n'], info['e_ref'], model.upper(),
                                                                    'label blocks' if blocks else 'packed layout'), flush=True)
    st = None
    for m in {'ccsdt': ['ccsd', 'ccsdt'], 'ccsdtq': ['ccsd', 'ccsdt', 'ccsdtq']}[model]:
        st = common.ground(eris, m, start=st)
    e_cc = eris.e_hf + st['e_corr']
    print('ground state %s: E = %.11f (%.1f s)' % (model.upper(), e_cc, st['seconds']), flush=True)
    rec = {'T': model, 'e_cc': e_cc, 'E4': {}, 'E5': {}, 'nsigma4': {}, 'nsigma5': {}, 'sigma5_seconds': {}, 'timing': {},
           'omega_steps': {}, 'residual5': {}}
    for tgt in (0, 2):
        if blocks:
            pb = dea.ProblemSym(st['eris'], st['t1'], st['t2'], st['t3'], st['t4'], rank=4, target=tgt, parity=par,
                                target_parity=0)
        else:
            pb = dea.Problem(st['eris'], st['t1'], st['t2'], st['t3'], st['t4'], rank=4, labels=lab, target=tgt, parity=par,
                             target_parity=0)
        r4 = dea.kernel(None, None, None, problem=pb, nroots=1, nguess=4, conv_tol=tol, max_cycle=120, verbose=True)
        rec['E4'][tgt], rec['nsigma4'][tgt] = e_cc + float(r4.e[0]), r4.nsigma
        omega, sol, f5 = dea5.kernel(pb, (float(r4.e[0]), r4.vectors[0]), conv_tol=tol, verbose=True)
        rec['E5'][tgt], rec['nsigma5'][tgt] = e_cc + omega, f5.nsigma
        rec['sigma5_seconds'][tgt], rec['timing'][tgt] = f5.seconds / max(f5.nsigma, 1), f5.timing
        rec['omega_steps'][tgt], rec['residual5'][tgt] = [float(x) for x in sol.history[0]], float(sol.residuals[0])
        print('  2 Omega = %d: E(4p2h) = %.11f  E(5p3h folded) = %.11f  %s; %d sigma vectors of %.1f s (%s)'
              % (tgt, rec['E4'][tgt], rec['E5'][tgt], 'converged' if sol.converged.all() else 'NOT CONVERGED', f5.nsigma,
                 rec['sigma5_seconds'][tgt], ', '.join('%s %.0f s' % kv for kv in f5.timing.items())), flush=True)
        del pb, f5
        with open(out, 'w') as f:
            json.dump(rec, f, indent=1)
    s4, s5 = rec['E4'][2] - rec['E4'][0], rec['E5'][2] - rec['E5'][0]
    rec['split4'], rec['split5'] = s4, s5
    print('LEVEL T %-6s 4p2h complete     E(0) = %.11f  E(2) = %.11f  split = %.4e Eh = %.4f cm-1'
          % (model, rec['E4'][0], rec['E4'][2], s4, s4 * common.H2CM))
    print('LEVEL T %-6s + 5p3h lowest order, folded  E(0) = %.11f  E(2) = %.11f  split = %.4e Eh = %.4f cm-1'
          % (model, rec['E5'][0], rec['E5'][2], s5, s5 * common.H2CM))
    with open(out, 'w') as f:
        json.dump(rec, f, indent=1)
    print('total %.0f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
