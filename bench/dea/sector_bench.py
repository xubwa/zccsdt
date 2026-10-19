'''EOM-DEA-CCSDT: the sigma vector of the ghost route (zccsdt.cc.dea.ProblemSym)
against that of the sector (zccsdt.cc.dea_sector.ProblemSector) on one
Hamiltonian file, in one process (same node, same threads): the time of a
sigma vector, its pieces, and the peak memory by phase (RssAnon sampled,
tracemalloc of the sector's sigma against its plan); then, optionally, the
roots of the sector in the blocks 2 Omega = 0 and 2 (parity g) - the
zero-field splitting - in the same process.

usage: python sector_bench.py <file.npz> <out.json> [--freeze n] [--nsigma 3] [--mem MB]
                              [--backends ghost,sector] [--roots] [--tol 1e-9]'''
import json
import os
import sys
import threading
import time
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


class MemWatch(threading.Thread):
    '''Peak RssAnon of this process (sampled), resettable by phase.'''

    def __init__(self, dt=0.05):
        super().__init__(daemon=True)
        self.dt, self.peak, self._done = dt, 0, threading.Event()

    @staticmethod
    def now():
        with open('/proc/self/status') as f:
            return {l.split(':')[0]: int(l.split()[1]) for l in f if l.startswith('Rss')}.get('RssAnon', 0) * 1024

    def run(self):
        while not self._done.wait(self.dt):
            self.peak = max(self.peak, self.now())

    def phase(self):
        '''(peak since the last call, now); starts a new phase.'''
        p, n = max(self.peak, self.now()), self.now()
        self.peak = n
        return p, n

    def stop(self):
        self._done.set()
        self.join()


def opt(args, name, default, cast=str):
    if name in args:
        k = args.index(name)
        v = cast(args[k + 1])
        del args[k:k + 2]
        return v
    return default


def main():
    from zccsdt import timing
    from zccsdt.cc import dea, dea_sector
    args = sys.argv[1:]
    freeze = opt(args, '--freeze', 0, int)
    nsig = opt(args, '--nsigma', 3, int)
    mem = opt(args, '--mem', 100000.0, float)
    backends = opt(args, '--backends', 'ghost,sector').split(',')
    tol = opt(args, '--tol', 1e-9, float)
    roots = '--roots' in args
    if roots:
        args.remove('--roots')
    path, out = args[0], args[1]
    watch = MemWatch()
    watch.start()
    rec = {'file': path, 'freeze': freeze, 'threads': os.environ.get('OMP_NUM_THREADS'), 'host': os.uname()[1],
           'phases': {}}

    def phase(name):
        p, n = watch.phase()
        rec['phases'][name] = {'peak_GB': p / 1e9, 'after_GB': n / 1e9}
        print('  [memory] %-28s peak %.2f GB, after %.2f GB' % (name, p / 1e9, n / 1e9), flush=True)

    def dump():
        with open(out, 'w') as f:
            json.dump(rec, f, indent=1)
    t0 = time.time()
    eris, lab, par, info = common.load(path, sort=True, freeze=freeze)
    no = eris.nocc
    nv = np.asarray(eris.fock).shape[0] - no
    print('file %s: %d occupied + %d virtual spinors (frozen %d)' % (path, no, nv, freeze), flush=True)
    phase('integrals')
    st = common.ground(eris, 'ccsdt', 1e-11)
    print('ground state CCSDT: E = %.11f (%.0f s)' % (eris.e_hf + st['e_corr'], st['seconds']), flush=True)
    rec['e_cc'] = eris.e_hf + st['e_corr']
    phase('ground state')
    rng = np.random.default_rng(11)
    for be in backends:
        timing.reset()
        t1 = time.time()
        if be == 'ghost':
            pb = dea.ProblemSym(st['eris'], st['t1'], st['t2'], st['t3'], None, rank=4, target=0, parity=par,
                                target_parity=0, mem_mb=mem)
        else:
            pb = dea_sector.ProblemSector(st['eris'], st['t1'], st['t2'], st['t3'], rank=4, target=0, parity=par,
                                          target_parity=0, mem_mb=mem, verbose=True)
        prep = time.time() - t1
        print('%s: prepared in %.1f s; vector %s elements' % (be, prep, pb.sizes), flush=True)
        phase(be + ': prepared')
        x = pb.project([rng.standard_normal(n) for n in pb.sizes])
        y = pb.new()
        phase(be + ': vectors')
        ts = []
        for k in range(nsig):
            t1 = time.time()
            pb.sigma(x, y)
            ts.append(time.time() - t1)
            print('  sigma %d: %.1f s' % (k + 1, ts[-1]), flush=True)
        pieces = timing.report()
        tm_peak = None
        if be == 'sector':
            # one more, under tracemalloc (not timed): what a sigma vector allocates
            tracemalloc.start()
            base = tracemalloc.get_traced_memory()[0]
            pb.sigma(x, y)
            tm_peak = tracemalloc.get_traced_memory()[1] - base
            tracemalloc.stop()
        phase(be + ': sigma vectors')
        r = {'prepare_s': prep, 'sigma_s': ts, 'sizes': [int(n) for n in pb.sizes],
             'pieces': [(n_, s_ / nsig) for n_, s_, _ in pieces[:25]]}
        if be == 'sector':
            r['plan_GB'] = [m / 1e9 for m in pb.mem]
            r['tracemalloc_sigma_GB'] = (tm_peak or 0) / 1e9
            r['batch_MB'] = pb.batch_bytes / 1e6
            print('  plan: %.2f GB fixed + %.2f GB per sigma vector; tracemalloc of a sigma vector %.2f GB'
                  % (pb.mem[0] / 1e9, pb.mem[1] / 1e9, (tm_peak or 0) / 1e9), flush=True)
        for n_, s_, c_ in pieces[:16]:
            print('    %-20s %8.2f s  %6.0f calls' % (n_, s_ / nsig, c_ / nsig), flush=True)
        rec[be] = r
        del pb, x, y
        phase(be + ': released')
        dump()
    if roots:
        res = common.run_levels(eris, lab, par, ['ccsdt'], conv_tol=tol, backend='sector', mem_mb=mem,
                                start_states={'ccsdt': st})
        rec['roots'] = res['ccsdt']
        phase('roots')
    rec['total_s'] = time.time() - t0
    dump()
    watch.stop()


if __name__ == '__main__':
    main()
