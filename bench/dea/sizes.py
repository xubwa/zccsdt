'''Sizes of the EOM-DEA problem of a Hamiltonian file without computing
anything: the vectors of the sector by rank (all elements; those of the
labels 2 Omega of a target; those of 2 Omega and parity), and the label
blocks of the ghost route (T3, T4 with two ghosts) with the memory of the
T4 engine.

usage: python sizes.py <file.npz> [target ...] [--freeze k]'''
import sys
from math import comb

import numpy as np


def count(lo, lv, po, pv, nh, target):
    '''Elements r[(nh holes), (nh + 2 particles)] in all, with the labels
    adding to ``target``, and with even parity besides - by convolution of
    the label distributions.'''
    def dist(lab, par, k):
        # {(sum of labels, parity): number of k-subsets}
        cur = [{(0, 0): 1}] + [dict() for _ in range(k)]
        for l, p in zip(lab, par):
            for m in range(k - 1, -1, -1):
                for (s, q), c in cur[m].items():
                    key = (s + int(l), (q + int(p)) % 2)
                    cur[m + 1][key] = cur[m + 1].get(key, 0) + c
        return cur[k]
    do, dv = dist(lo, po, nh), dist(lv, pv, nh + 2)
    total = comb(len(lo), nh) * comb(len(lv), nh + 2)
    by_label = sum(c * d for (s, q), c in dv.items() for (t, u), d in do.items() if s - t == target)
    both = sum(c * d for (s, q), c in dv.items() for (t, u), d in do.items() if s - t == target and (q - u) % 2 == 0)
    return total, by_label, both


def main():
    freeze = 0
    if '--freeze' in sys.argv:
        k = sys.argv.index('--freeze')
        freeze = int(sys.argv[k + 1])
        del sys.argv[k:k + 2]
    z = np.load(sys.argv[1])
    targets = [int(x) for x in sys.argv[2:]] or [0, 2]
    no = int(z['ncore']) - freeze
    lab = np.asarray(z['omega2'], dtype=np.int64)[freeze:]
    par = (np.asarray(z['parity'], dtype=np.int64) if 'parity' in z.files else np.zeros(len(lab) + freeze, dtype=np.int64))[freeze:]
    lo, lv, po, pv = lab[:no], lab[no:], par[:no], par[no:]
    nv = len(lv)
    print('%s: %d occupied + %d virtual spinors' % (sys.argv[1], no, nv))
    for tgt in targets:
        print('target 2 Omega = %d:  rank: all elements / with the labels 2 Omega / and parity g  (GB as real numbers)' % tgt)
        for nh in range(0, min(no, 5)):
            a, b, c = count(lo, lv, po, pv, nh, tgt)
            print('  %dp%dh  %.3e (%.2f GB)  %.3e (%.2f GB)  %.3e (%.2f GB)' % (nh + 2, nh, a, a * 8e-9, b, b * 8e-9, c, c * 8e-9))
    try:
        from zccsdt.cc import t4engine_sym as S
    except Exception as e:
        print('no label-block engine:', e)
        return
    for tgt in targets:
        plo = np.sort(np.concatenate(([1, tgt - 1], lo)))
        lvs = np.sort(lv)
        lo_min = min(plo.min(), lvs.min())
        ko, kv = (plo - lo_min) // 2, (lvs - lo_min) // 2
        lay = S.Layout(ko, kv, int(max(ko.max(), kv.max())) + 1)
        print('target %d, ghost route in label blocks (two ghosts): T3 %.3e, T4 %.3e elements (%.2f GB); packed T4 %.3e; '
              'memory of one T4 residual %.1f GB (batches of 1 GB)'
              % (tgt, lay.t3off[-1], lay.size, lay.size * 8e-9, comb(no + 2, 4) * comb(nv, 4), S.memory(lay, 8) / 1e9))
    lo_min = min(lo.min(), lv.min())
    lay = S.Layout((np.sort(lo) - lo_min) // 2, (np.sort(lv) - lo_min) // 2, int((max(lo.max(), lv.max()) - lo_min) // 2) + 1)
    print('ground state: T4 in label blocks %.3e elements (%.2f GB), memory of one T4 residual %.1f GB'
          % (lay.size, lay.size * 8e-9, S.memory(lay, 8) / 1e9))


if __name__ == '__main__':
    main()
