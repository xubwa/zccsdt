'''The hh ladder + F_oo of the T3 equation in its two forms (t3kern.OccOperator,
docs/factorization.md 2.3) on random data: the matrix on packed occupied
triples (one GEMM) against the pair-driven form, for a range of occupied
spinors and a packed T3 of a given size (dense layout: one block with all
triples).  For each form: the time to build its operator and to apply it.

usage: python bench/hh_ladder.py OMIN OMAX [GB of T3, default 2] [step in o, default 2]
The number of threads is that of the environment (OMP_NUM_THREADS).
'''
import sys
import time

from pyscf import lib as _lib       # noqa: F401  (PySCF first: one OpenMP runtime, see zccsdt/__init__.py)
import numpy as np

from zccsdt.cc import t3kern, t3pack


def main(omin, omax, gb=2.0, step=2):
    rng = np.random.default_rng(1)
    print('threshold of the code: pair-driven from %d occupied spinors on; %d columns per chunk'
          % (t3kern.HH_PAIR_MIN_OCC, t3kern.HH_PAIR_COLS))
    print('  o  triples  columns | matrix: build   apply  cores | pair: build   apply  cores | matrix / pair')
    for o in range(omin, omax + 1, step):
        om = t3pack.TripleMap(o)
        ncol = int(gb * 1e9 / 8 / om.ntri)
        W = rng.standard_normal((o,) * 4)
        W = W - W.transpose(1, 0, 2, 3)
        W = W - W.transpose(0, 1, 3, 2)
        F = rng.standard_normal((o, o))
        T = rng.standard_normal((om.ntri, ncol))
        R = np.zeros_like(T)
        res = []
        for mode in ('matrix', 'pair'):
            best = (1e9, 1e9, 0.0)
            for rep in range(3):
                op = t3kern.OccOperator(om, W, F, np.float64, mode=mode)
                t0 = time.perf_counter()
                op.matrix() if mode == 'matrix' else op.pair_matrices()
                t1, c1 = time.perf_counter(), time.process_time()
                op.apply(T, R)
                t2, c2 = time.perf_counter(), time.process_time()
                if t2 - t0 < best[0] + best[1]:
                    best = (t1 - t0, t2 - t1, (c2 - c1) / (t2 - t1))
            res.append(best)
        print('%3d %8d %8d |      %8.4f %7.3f %6.1f |    %8.4f %7.3f %6.1f | %6.2f'
              % ((o, om.ntri, ncol) + res[0] + res[1] + (sum(res[0][:2]) / sum(res[1][:2]),)), flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    main(int(a[0]), int(a[1]), float(a[2]) if len(a) > 2 else 2.0, int(a[3]) if len(a) > 3 else 2)
