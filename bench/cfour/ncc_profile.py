'''Summarize the timer tables that an ncc built with -DPROFILE (see
rebuild_ncc.sh) prints after the CC iterations and after the non-iterative
correction.

usage: python ncc_profile.py OUT NTHREADS

The timers are in thread-seconds: inside a parallel region each thread adds
its own time, outside the elapsed time is multiplied by the number of threads.
Listed are the layer-1 timers (the primitives: BLAS levels, transposition,
spin adaptation, ...) as shares of the top-level timer.  They are nested here
and there (BLAS 1 calls inside spin adaptation), so the shares of everything
but BLAS 3 are approximate and can add up to more than 100% on one thread.
'''
import re
import sys

TOP = {'iter': 'layer4_ccsdt_iter_rhf', 'CCSDT(Q)': 'ccsdt_q_rhf'}
EXTRA = ('z_blas1_in_layer2', 'z_q_energy', 'z_allocate_hunk')


def tables(path):
    out, cur = {}, None
    for line in open(path, errors='replace'):
        m = re.match(r'(CCSDT iterations (?:converged|did not converge)'
                     r'|Non-iterative calculation of (\S+))', line)
        if m:
            cur = out.setdefault('iter' if m.group(2) is None else m.group(2), {})
            cur['_line'] = line.strip()
            continue
        m = re.match(r'(\S+):\s+([0-9.]+) s\s+(\d+) x\s+([0-9.]+) gflops/sec', line)
        if m and cur is not None:
            cur[m.group(1)] = (float(m.group(2)), int(m.group(3)), float(m.group(4)))
    return out


def report(path, nthreads):
    for sec, t in tables(path).items():
        top = TOP.get(sec)
        if top not in t:
            continue
        tot = t[top][0]
        print(t['_line'])
        print('  %s: %.1f thread-s = %.2f s per thread' % (top, tot, tot / nthreads))
        prim = [(k, v) for k, v in t.items()
                if k.startswith('layer1_') or k in EXTRA]
        acc = 0.0
        for k, v in sorted(prim, key=lambda kv: -kv[1][0]):
            if v[0] < 0.002 * tot:
                continue
            acc += v[0]
            print('    %-20s %9.2f s %6.1f%%  %7.2f GFLOP/s per thread  %8d calls'
                  % (k, v[0], 100 * v[0] / tot, v[2], v[1]))
        print('    %-20s %9.2f s %6.1f%%' % ('(in no such timer)', tot - acc,
                                             100 * (tot - acc) / tot))


if __name__ == '__main__':
    report(sys.argv[1], int(sys.argv[2]))
