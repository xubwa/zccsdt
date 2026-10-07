'''(P) in the blocks of the symmetry labels (zccsdt.cc.pentuples_sym) against
the dense code (zccsdt.cc.pentuples), one after the other in one process on
the same integrals and amplitudes: a linear molecule on the z axis with X2C
spinors (labels 2 m_j).  The cost of (P) does not depend on the amplitudes:
T2 is that of second order, T3 and T4 are random numbers on the elements
that the labels allow.

usage: python bench/sym_p.py GEOM BASIS FROZEN [NVIR] [CODES]
       e.g.  python bench/sym_p.py 'Br 0 0 0; H 0 0 1.4145' cc-pvdz-dk 28 38
  CODES: dense,sectors (default) or one of them; mod4 and one for the
         sectors with the labels modulo 4 and with one label
  ZC_TRACE=1: the peak of the allocations of each code (tracemalloc)
              against its plan
'''
import copy
import os
import sys
import time
import tracemalloc

import numpy as np
from pyscf import gto, x2c

import zccsdt
from zccsdt import timing
from zccsdt.cc import pentuples, pentuples_sym, t4pack as P


def status(key):
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith(key):
                return int(line.split()[1]) / 1024.0            # MB
    return float('nan')


def reset_peak():
    try:
        with open('/proc/self/clear_refs', 'w') as f:
            f.write('5')
    except OSError:
        pass


def allowed(lab, no, k):
    occ = lab[:no][P.tuples(no, k)].sum(axis=1)
    vir = np.rint(P.tuple_sums(lab[no:].astype(float), k)).astype(np.int64)
    return occ[:, None] == vir[None, :]


geom, basis, frozen = sys.argv[1], sys.argv[2], int(sys.argv[3])
nvir = int(sys.argv[4]) if len(sys.argv) > 4 and int(sys.argv[4]) else None
codes = (sys.argv[5] if len(sys.argv) > 5 else 'dense,sectors').split(',')
mol = gto.M(atom=geom, basis=basis, verbose=0)
mf = x2c.UHF(mol)
mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-8
mf.kernel()
order = np.argsort(mf.mo_energy)
nocc = int((mf.mo_occ > 0).sum())
drop = []
if nvir is not None:
    # (the cut between two levels, not inside a pair +m_j, -m_j)
    e = mf.mo_energy[order]
    while nocc + nvir < len(e) and e[nocc + nvir] - e[nocc + nvir - 1] < 1e-6:
        nvir += 1
    drop = list(order[nocc + nvir:])
cc = zccsdt.GCCSDT(mf, frozen=list(order[:frozen]) + drop)
cc.verbose = 0
eris = cc.ao2mo()
no = eris.nocc
lab = np.asarray(eris.labels)
nv = len(lab) - no
mo_e = np.asarray(eris.mo_energy)
eo, ev = mo_e[:no], mo_e[no:]
t2 = np.conj(np.asarray(eris.oovv)) / (eo[:, None, None, None] + eo[None, :, None, None]
                                     - ev[None, None, :, None] - ev[None, None, None, :])
rng = np.random.default_rng(3)
t3, t4 = (s * rng.standard_normal(ok.shape) * ok
          for s, ok in ((3e-3, allowed(lab, no, 3)), (1e-3, allowed(lab, no, 4))))
print('%d occupied + %d virtual spinors, labels of the virtual ones: %s;  T4 %.0f MB;  %s threads'
      % (no, nv, ' '.join('%d:%d' % (k, (lab[no:] == k).sum()) for k in np.unique(lab[no:])), t4.nbytes / 1e6,
         os.environ.get('OMP_NUM_THREADS')), flush=True)
out = {}
for code in codes:
    er = eris
    if code == 'mod4':
        er = copy.copy(eris)
        er.modulus = 4
    elif code == 'one':
        er = copy.copy(eris)
        er.labels = np.zeros_like(lab)
    if code == 'dense':
        pl = pentuples.plan(no, nv, 8)
        run = lambda: pentuples.kernel(er, t2, t3, t4, plan_=pl)
    else:
        pl = pentuples_sym.plan(er, 8)
        run = lambda: pentuples_sym.kernel(er, t2, t3, t4, plan_=pl)
    timing.reset()
    reset_peak()
    rss0 = status('VmRSS')
    if os.environ.get('ZC_TRACE'):
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
    w0, c0 = time.perf_counter(), time.process_time()
    out[code] = run()
    w, c = time.perf_counter() - w0, time.process_time() - c0
    line = ''
    if os.environ.get('ZC_TRACE'):
        line = '  allocated at the peak %.0f MB' % ((tracemalloc.get_traced_memory()[1] - base) / 1e6)
        tracemalloc.stop()
    print('%-8s E(P) = %.15e   %.2f s wall, %.1f cores busy   plan %.0f MB, resident at the peak %.0f MB '
          '(%.0f MB before)%s' % (code, out[code], w, c / w, pl.need / 1e6, status('VmHWM'), rss0, line), flush=True)
    for name, sec, calls in timing.report():
        if sec > 0.005 * w:
            print('    %-14s %9.2f s  %6d calls' % (name, sec, calls))
    if code != 'dense':
        print('    plan: ' + ', '.join('%s %.0f MB' % (k, v / 1e6) for k, v in pl.parts.items()), flush=True)
for code in codes[1:]:
    print('%s - %s: %.1e' % (code, codes[0], out[code] - out[codes[0]]))
