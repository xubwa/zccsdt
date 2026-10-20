'''The integral container of common.load from a block file against the one
from the .npz of the dense route, in the order of the labels (--blocks of
o2_dea.py), block by block; the time and memory of both.

usage: python x2ccorr_loadcheck.py <dense.npz> <blocks.h5> [--freeze n]'''
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from zccsdt import memory  # noqa: E402
from zccsdt.eris import BLOCKS  # noqa: E402


def main():
    args = sys.argv[1:]
    freeze = 0
    if '--freeze' in args:
        k = args.index('--freeze')
        freeze = int(args[k + 1])
        del args[k:k + 2]
    old, new = args
    t0 = time.time()
    eb, lb, pb, ib = common.load(new, sort=True, freeze=freeze)
    t1 = time.time()
    print('block file: %.1f s, RssAnon %.2f GB' % (t1 - t0, memory.resident() / 1e9), flush=True)
    ea, la, pa, ia = common.load(old, sort=True, freeze=freeze, drop_forbidden=True)
    print('dense file: %.1f s' % (time.time() - t1), flush=True)
    print('labels identical %s, parities identical %s' % (np.array_equal(la, lb), np.array_equal(pa, pb)))
    print('e_hf %.12f %.12f difference %.1e' % (ea.e_hf, eb.e_hf, eb.e_hf - ea.e_hf))
    print('fock max difference %.1e' % abs(ea.fock - eb.fock).max())
    for k in BLOCKS:
        a, b = np.asarray(getattr(ea, k)), np.asarray(getattr(eb, k))
        print('  %s %s max difference %.1e (largest %.1e)' % (k, a.shape, abs(a - b).max(initial=0.0),
                                                            abs(a).max(initial=0.0)))


if __name__ == '__main__':
    main()
