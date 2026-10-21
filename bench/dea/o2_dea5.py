'''EOM-DEA of O2 with complete 5p3h operators on the sector
(zccsdt.cc.dea5_sector): R = 2p + 3p1h + 4p2h + 5p3h, T of CCSDT, the lowest
root of the blocks 2 Omega = 0 and 2 (parity g), total energies and the
splitting.

usage: python o2_dea5.py <file.npz | file.h5> <out.json> [--tol 1e-9] [--mem MB] [--targets 0 2]
                         [--sigmas n] [--max-space n] [--freeze n]
--sigmas n: only n sigma vectors of random vectors per target (timings, memory), no roots'''
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def rss_gb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS'):
                    return float(line.split()[1]) / 1e6
    except OSError:
        pass
    return 0.0


def main():
    from zccsdt import timing
    from zccsdt.cc import dea5_sector as D5
    args = sys.argv[1:]

    def opt(name, default, conv):
        if name in args:
            k = args.index(name)
            v = conv(args[k + 1])
            del args[k:k + 2]
            return v
        return default
    tol = opt('--tol', 1e-9, float)
    mem = opt('--mem', 100000.0, float)
    nsig = opt('--sigmas', 0, int)
    max_space = opt('--max-space', None, int)
    freeze = opt('--freeze', 0, int)
    targets = (0, 2)
    if '--targets' in args:
        k = args.index('--targets')
        targets = tuple(dict.fromkeys(int(x) for x in args[k + 1:k + 3]))      # (one target: --targets t t)
        del args[k:k + 3]
    path, out = args[0], args[1]
    t0 = time.time()
    eris, lab, par, info = common.load(path, sort=True, freeze=freeze)
    no = eris.nocc
    print('file %s: %d + %d spinors; E(reference) = %.12f; max |f_ov| %.1e; RSS %.1f GB'
          % (path, no, info['n'] - no, info['e_ref'], info['fock_ov'], rss_gb()), flush=True)
    st = common.ground(eris, 'ccsd', 1e-11)
    st = common.ground(eris, 'ccsdt', 1e-11, start=st)
    e_cc = eris.e_hf + st['e_corr']
    print('ground state CCSDT: E = %.12f (%d iterations, %.1f s)' % (e_cc, st['ncycle'], st['seconds']), flush=True)
    rec = {'T': 'CCSDT', 'operator': '2p + 3p1h + 4p2h + 5p3h (sector)', 'e_cc': e_cc, 'file': path, 'E': {},
           'omega': {}, 'residual': {}, 'nsigma': {}, 'sigma_seconds': {}, 'sizes': {}, 'weights': {},
           'prepare_seconds': {}, 'timers': {}}
    for tgt in targets:
        tp = time.time()
        pb = D5.ProblemSector5(eris, st['t1'], st['t2'], st['t3'], target=tgt, parity=par, mem_mb=mem, verbose=True)
        rec['prepare_seconds'][tgt] = time.time() - tp
        rec['sizes'][tgt] = list(pb.counts())
        print('2 Omega = %d: elements %s, prepared in %.1f s, RSS %.1f GB' % (tgt, pb.counts(), time.time() - tp, rss_gb()),
              flush=True)
        if nsig:
            rng = np.random.default_rng(1)
            x = pb.project([rng.standard_normal(n) for n in pb.sizes])
            y = pb.new()
            for it in range(nsig):
                timing.reset()
                ts = time.time()
                pb.sigma(x, y)
                print('  sigma vector %d: %.1f s, RSS %.1f GB' % (it + 1, time.time() - ts, rss_gb()), flush=True)
                print('    ' + ', '.join('%s %.1f' % (name, sec) for name, sec, _ in timing.report()[:40]), flush=True)
            rec['sigma_seconds'][tgt] = pb.seconds / max(pb.nsigma, 1)
            del pb, x, y
            continue
        e, res = D5.kernel(pb, nroots=1, nguess=4, conv_tol=tol, max_space=max_space, max_cycle=150, mem_mb=mem,
                           verbose=True)
        v = res.vectors[0]
        w = np.array([float(np.vdot(p, p).real) for p in v])
        rec['omega'][tgt] = float(e[0])
        rec['E'][tgt] = e_cc + float(e[0])
        rec['residual'][tgt] = float(res.residuals[0])
        rec['nsigma'][tgt] = int(res.nsigma)
        rec['sigma_seconds'][tgt] = pb.seconds / max(pb.nsigma, 1)
        rec['weights'][tgt] = [float(x) for x in w / w.sum()]
        print('  2 Omega = %d: E = %.11f  residual %.1e%s  %d sigma vectors, %.1f s each; weights %s'
              % (tgt, rec['E'][tgt], res.residuals[0], '' if res.converged.all() else ' NOT CONVERGED', res.nsigma,
                 rec['sigma_seconds'][tgt], ' '.join('%.5f' % x for x in rec['weights'][tgt])), flush=True)
        del pb, res, v
        with open(out, 'w') as f:
            json.dump(rec, f, indent=1)
    if len(rec['E']) == 2:
        e0, e2 = (rec['E'][t] for t in targets)
        rec['split'] = e2 - e0
        print('LEVEL T CCSDT R 5p3h (sector) E(2 Omega = %d) = %.11f  E(2 Omega = %d) = %.11f  split = %.10e Eh = %.5f cm-1'
              % (targets[0], e0, targets[1], e2, rec['split'], rec['split'] * common.H2CM), flush=True)
    rec['seconds'] = time.time() - t0
    with open(out, 'w') as f:
        json.dump(rec, f, indent=1)
    print('total %.0f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
