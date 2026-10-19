'''The EOM-DEA-CCSDT problem on the sector (zccsdt.cc.dea_sector) and on the
ghost route (zccsdt.cc.dea.ProblemSym) counted from the labels alone -
nothing computed: the elements of the vectors, the multiply-adds of the
pieces of a sigma vector of the sector, its memory plan; with timings of
sector_bench.py runs (--calib) the time of a sigma vector predicted per
piece from the rates measured there.

usage: python sector_cost.py <file.npz | scf log> [--freeze n] [--target 0] [--calib run.json:file.npz[:freeze] ...]

A log of o2_scf.py (its tables "occupied spinors (energy, 2 m_j, parity)" and
"virtual spinors") stands for a Hamiltonian file that is not there yet.'''
import json
import re
import sys
from math import comb

import numpy as np


def labels_of(path, freeze=0):
    '''(2 m_j of the occupied, of the virtual spinors; parities), the frozen
    spinors (the lowest ``freeze`` occupied) left out.'''
    if path.endswith('.npz'):
        z = np.load(path)
        no = int(z['ncore'])
        lab = np.asarray(z['omega2'], dtype=np.int64)
        par = np.asarray(z['parity'], dtype=np.int64) if 'parity' in z.files else np.zeros(len(lab), dtype=np.int64)
        eps = np.asarray(z['eps']).real if 'eps' in z.files else np.arange(len(lab), dtype=float)
        o = np.argsort(eps[:no], kind='stable')[freeze:]
        return lab[:no][o], lab[no:], par[:no][o], par[no:]
    occ, vir, part = [], [], None
    for line in open(path):
        if line.startswith('occupied spinors'):
            part = occ
            continue
        if line.startswith('virtual spinors'):
            part = vir
            continue
        m = re.match(r'^\s+(?:\d+\s+)?(-?\d+\.\d+)\s+(-?\d+) ([gu])', line)
        if part is not None and m:
            part.append((float(m.group(1)), int(m.group(2)), 0 if m.group(3) == 'g' else 1))
        elif part is vir and vir and not m:
            part = None
    occ = sorted(occ)[freeze:]
    return (np.array([x[1] for x in occ]), np.array([x[1] for x in vir]),
            np.array([x[2] for x in occ]), np.array([x[2] for x in vir]))


class _Eris:
    '''Just the labels, as zccsdt.cc.t3pack.classes_of() reads them.'''

    def __init__(self, lo, lv):
        o, v = np.argsort(lo, kind='stable'), np.argsort(lv, kind='stable')
        self.labels = np.concatenate((lo[o], lv[v]))
        self.nocc, self.modulus, self.use_labels = len(lo), 0, True


def count(lo, lv, target=0, isz=8):
    from zccsdt.cc import dea_sector as D, t3pack, t4engine_sym as S
    eris = _Eris(lo, lv)
    ko, kv, ncls = t3pack.classes_of(eris)
    lay = S.Layout(ko, kv, ncls)
    sec = D.Sector(lay, D.shift_of(eris, target))
    s, nQ, nV, nO, nK = sec.s, lay.nQ, lay.nV, lay.nO, lay.nK
    no = lay.no
    fl = {}
    fl['pp'] = sum(len(sec.rows4[M]) * sum(int(nQ[q]) ** 2 * int(nQ[M - q]) for q in range(lay.n2)
                                             if sec.off22[M, q] >= 0) for M in sec.blocks4)
    ring = 0
    for j in range(no):
        kj = ko[j]
        for d in range(-(ncls - 1), ncls):
            kK = kj + s + d
            if not (0 <= kK < lay.n3 and nK[kK]):
                continue
            nc = sum(int(nV[ko[m] - d]) for m in range(no) if m != j and 0 <= ko[m] - d < ncls)
            nr = sum(int(nV[ko[i] - d]) for i in range(no) if i != j and 0 <= ko[i] - d < ncls)
            nr += (d == 0) + (int(nO[kj + d] + nV[kj + d]) if 0 <= kj + d < ncls else 0)
            ring += nc * nr * int(nK[kK])
    fl['ring'] = ring
    r22 = r13 = 0
    for M in sec.blocks4:
        for P_ in sec.rows4[M]:
            i, j = lay.o2[P_]
            for q in range(lay.n2):
                if sec.off22[M, q] < 0:
                    continue
                k = 0
                for x in (i, j):
                    e = q - ko[x]
                    if 0 <= e < ncls:
                        k += int(nV[e] + nO[e])
                k += sum(int(nV[q - s + ko[n]]) for n in range(no) if n not in (i, j) and 0 <= q - s + ko[n] < ncls)
                r22 += int(nQ[q]) * k * int(nQ[M - q])
            kP = ko[i] + ko[j]
            for p in range(ncls):
                if sec.off13[M, p] < 0:
                    continue
                k = 0
                for mu in (kP - p, s - p):
                    if 0 <= mu < ncls:
                        k += int(nO[mu] + nV[mu])
                for x in (i, j):
                    c = M - p - ko[x]
                    if 0 <= c < lay.n2:
                        k += int(lay.nO2[c])
                r13 += int(nV[p]) * k * int(nK[M - p])
    fl['rows'] = r22 + r13
    fl['hh'] = sum(len(sec.rows4[M]) ** 2 * int(sec.nA[M]) for M in sec.blocks4)
    # memory traffic of the expansions and reductions, in elements: pp 6 splits there and back,
    # the ring 4 splits there and back (every row for two spectators), the rows 6 + 4
    fl['move'] = sec.size4 * (2 * 6 + 2 * 2 * 4 + 10)
    fixed, per = D.memory(sec, 4, isz, D.BATCH_BYTES)
    ghost = {}
    try:
        plo = np.sort(np.concatenate(([1, target - 1], lo)))
        lmin = min(plo.min(), lv.min())
        step = int(np.gcd.reduce(np.diff(np.unique(np.concatenate((plo, lv))))))
        lg = S.Layout((plo - lmin) // step, (np.sort(lv) - lmin) // step, int((max(plo.max(), lv.max()) - lmin) // step) + 1)
        ghost = {'T4_blocks': int(lg.size), 'T3_blocks': int(lg.t3off[-1]),
                 'pp': int(sum(int(lg.nI[M]) * sum(int(lg.nQ[q]) ** 2 * int(lg.nQ[M - q]) for q in range(lg.n2)
                                                    if lg.off22[M, q] >= 0) for M in lg.sectors))}
    except Exception as e:      # (the layout of the padded spinors may not exist for odd targets)
        ghost = {'error': str(e)}
    return {'no': no, 'nv': lay.nv, 'ncls': int(ncls), 'sizes': [int(x) for x in sec.sizes], 'flops': fl,
            'plan_GB': [fixed / 1e9, per / 1e9], 'vector_GB': sum(sec.sizes) * isz / 1e9, 'ghost': ghost}


def main():
    args = sys.argv[1:]

    def opt(name, default, cast=str):
        if name in args:
            k = args.index(name)
            v = cast(args[k + 1])
            del args[k:k + 2]
            return v
        return default
    freeze = opt('--freeze', 0, int)
    target = opt('--target', 0, int)
    calib = []
    while '--calib' in args:
        k = args.index('--calib')
        calib.append(args[k + 1])
        del args[k:k + 2]
    lo, lv, po, pv = labels_of(args[0], freeze)
    c = count(lo, lv, target)
    print('%s: %d + %d spinors, %d classes; target %d' % (args[0], c['no'], c['nv'], c['ncls'], target))
    print('  sector: 2p %d, 3p1h %d, 4p2h %d elements (parity is a mask: about half of them are zeros); '
          'a vector %.2f GB' % (*c['sizes'], c['vector_GB']))
    print('  plan: %.2f GB fixed + %.2f GB per sigma vector' % tuple(c['plan_GB']))
    print('  multiply-adds: ' + ', '.join('%s %.3e' % kv for kv in c['flops'].items()))
    if 'T4_blocks' in c['ghost']:
        print('  ghost route: T4 blocks %.3e, T3 blocks %.3e, pp ladder %.3e multiply-adds'
              % (c['ghost']['T4_blocks'], c['ghost']['T3_blocks'], c['ghost']['pp']))
    rates = {}
    for spec in calib:
        run, f, *fz = spec.split(':')
        r = json.load(open(run))
        cc = count(*labels_of(f, int(fz[0]) if fz else 0)[:2], target)
        pieces = dict(r['sector']['pieces'])
        t = {'pp': pieces.get('dea.pp', 0), 'ring': pieces.get('dea.ring', 0), 'rows': pieces.get('dea.rows', 0),
             'hh': pieces.get('dea.hh', 0)}
        rest = float(np.mean(r['sector']['sigma_s'][1:] or r['sector']['sigma_s'])) - sum(t.values())
        print('  calibration %s (%d + %d): sigma %.1f s; ' % (run, cc['no'], cc['nv'], rest + sum(t.values()))
              + ', '.join('%s %.1f s = %.1f GF/s' % (k, v, cc['flops'][k] / max(v, 1e-9) / 1e9) for k, v in t.items())
              + ', the rest %.1f s' % rest)
        rates = {k: cc['flops'][k] / max(v, 1e-9) for k, v in t.items()}
        rates['rest'] = (rest, cc['sizes'][2])
    if rates:
        pred = {k: c['flops'][k] / rates[k] for k in ('pp', 'ring', 'rows', 'hh')}
        # the rest (the intermediates of the sector, the 3p1h row, the thin products) as the vector grows
        pred['rest'] = rates['rest'][0] * c['sizes'][2] / rates['rest'][1]
        print('  predicted sigma vector (rates of the last calibration): %.0f s (%s)'
              % (sum(pred.values()), ', '.join('%s %.0f' % kv for kv in pred.items())))
        c['predicted_s'] = pred
    out = opt('--json', None)
    if out:
        json.dump(c, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
