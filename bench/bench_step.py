'''One measurement of a spinor CCSDT(Q) benchmark, appended to a results file
as one JSON line: CCSDT iterations, or the (Q) step (whole, or a sample of
the occupied quadruples with the time of the whole step estimated from it),
through the front end, dense or in the blocks of the symmetry labels.

A benchmark of a large case is a list of such measurements, one process
each, ordered by cost (bench/bench_plan.sh): what is cheap and tells most
comes first, every result is on disk as soon as it exists, a measurement
that is already in the file is skipped, and the spinors are those of one
SCF kept in a file, so that all measurements are the same calculation.

usage: python bench/bench_step.py --geom GEOM --basis BASIS --frozen N --mode dense|blocks
                                  --what iter|q [--niter 4] [--fraction 0.05] [--real 1|0]
                                  --out RESULTS.jsonl [--spinors FILE.npz] [--tag TAG]
  iter: NITER iterations; the first, with its setup, is reported apart
  q:    2 iterations, then (Q): --fraction 0 the whole step, else a sample
The number of threads is that of the environment (OMP_NUM_THREADS).
'''
import argparse
import json
import os
CORES = sorted(os.sched_getaffinity(0))      # (before a threading runtime binds this thread to one of them)
import socket
import sys
import time

import numpy as np
from pyscf import gto, x2c

import zccsdt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_sa import MemWatch                                         # noqa: E402


def reference(mol, path):
    '''The X2C SCF of ``mol``; with ``path`` the spinors of the first run are
    kept there and read by the later ones (every SCF mixes the two spinors
    of a Kramers pair in its own way).'''
    mf = x2c.UHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-8
    if path and os.path.exists(path):
        d = np.load(path)
        mf.mo_coeff, mf.mo_energy, mf.mo_occ, mf.e_tot = d['mo_coeff'], d['mo_energy'], d['mo_occ'], float(d['e_tot'])
        mf.converged = True
        return mf
    mf.kernel()
    if path:
        tmp = path + '.%d.npz' % os.getpid()
        np.savez(tmp, mo_coeff=mf.mo_coeff, mo_energy=mf.mo_energy, mo_occ=mf.mo_occ, e_tot=mf.e_tot)
        os.replace(tmp, path)
    return mf


def done(path, key):
    if not os.path.exists(path):
        return False
    with open(path) as f:
        return any(json.loads(line).get('key') == key for line in f if line.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--geom', required=True)
    ap.add_argument('--basis', required=True)
    ap.add_argument('--frozen', type=int, default=0)
    ap.add_argument('--mode', choices=('dense', 'blocks'), required=True)
    ap.add_argument('--what', choices=('iter', 'q'), required=True)
    ap.add_argument('--niter', type=int, default=4)
    ap.add_argument('--fraction', type=float, default=0.05)
    ap.add_argument('--real', type=int, default=1, help='0: the spinors as the SCF left them (complex)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--spinors')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    nthr = int(os.environ.get('OMP_NUM_THREADS', 0))
    key = '|'.join(str(x) for x in (a.tag, a.geom, a.basis, a.frozen, a.mode, a.what, a.real, nthr,
                                    a.niter if a.what == 'iter' else a.fraction))
    if done(a.out, key):
        print('already in %s: %s' % (a.out, key))
        return
    load0 = open('/proc/loadavg').read().split()[:3]
    mol = gto.M(atom=a.geom, basis=a.basis, verbose=0)
    mf = reference(mol, a.spinors)
    cc = zccsdt.CCSDT(mf, frozen=a.frozen)
    cc.verbose, cc.symmetry, cc.real_spinors = 3, a.mode == 'blocks', bool(a.real)
    cc.conv_tol, cc.conv_tol_normt = 1e-12, 1e-10
    cc.max_cycle = a.niter if a.what == 'iter' else 2
    marks = [(time.perf_counter(), time.process_time())]
    cc.callback = lambda info: marks.append((time.perf_counter(), time.process_time()))
    watch = MemWatch()
    watch.start()
    cc.kernel()
    watch.stop()
    wall, cpu = np.diff([m[0] for m in marks]), np.diff([m[1] for m in marks])
    rest = slice(1, None) if len(wall) > 1 else slice(0, None)
    eris = cc._eris
    no, nv = eris.nocc, eris.nmo - eris.nocc
    rec = {'key': key, 'tag': a.tag, 'geom': a.geom, 'basis': a.basis, 'frozen': a.frozen, 'mode': a.mode,
           'what': a.what, 'threads': nthr, 'cores': CORES, 'host': socket.gethostname(),
           'job': os.environ.get('SLURM_JOB_ID'), 'commit': os.environ.get('ZC_COMMIT'),
           'nocc': no, 'nvir': nv, 'dtype': str(np.asarray(cc.t2).dtype), 't3_elements': int(np.size(cc.t3)),
           'e_scf': float(mf.e_tot), 'iterations': [float(w) for w in wall],
           'iteration': float(wall[rest].mean()), 'iteration_cores': float(cpu[rest].sum() / wall[rest].sum()),
           'iteration_peak_gb': watch.anon / 1e6, 'e_corr': float(cc.e_corr), 'load_before': load0}
    if a.what == 'q':
        watch = MemWatch()
        watch.start()
        t0, c0 = time.perf_counter(), time.process_time()
        if a.fraction:      # (a sample: the time of the whole step from the time per quadruple)
            est = cc.ccsdt_q(fraction=a.fraction)
            rec.update(q_fraction=a.fraction, q_estimate=float(est), q_sampled=cc.q_estimate['sampled'],
                       q_quadruples=cc.q_estimate['quadruples'], q_sectors=cc.q_estimate['sectors'])
        else:
            eq = cc.ccsdt_q()
            rec.update(q_fraction=0.0, e_q5=float(cc.e_q5), e_q=float(eq))
        tq, cq = time.perf_counter() - t0, time.process_time() - c0
        watch.stop()
        rec.update(q_seconds=tq, q_cores=cq / tq, q_peak_gb=watch.anon / 1e6)
    rec['load_after'] = open('/proc/loadavg').read().split()[:3]
    with open(a.out, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print('RESULT ' + json.dumps({k: rec[k] for k in rec if k not in ('cores', 'geom', 'key')}), flush=True)


if __name__ == '__main__':
    main()
