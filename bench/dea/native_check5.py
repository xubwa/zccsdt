'''The sigma vector of EOM-DEA with complete 5p3h of the native program
(cfour/xzccsdt, %xzdea level 5p3h, option deasig of %xzccsdt) against
zccsdt.cc.dea5_sector.Engine5, element by element, on the same integral
files and the same amplitudes (the file XZAMPS that the native program
wrote): a random vector of the first target of %xzdea.

usage: python native_check5.py <rundir> <xzccsdt binary> [--batch n] [--seed n] [--threads n] [--target t]
                                [--complex] [--mpi n] [--piece p] [--xz 'line of %xzccsdt' ...]
  rundir: made by native_rundir.py --level 5p3h, with XZAMPS of a native CCSDT run in it
  --batch: option deabatch of the native run (elements of an operand of a batch; small: many batches)
  --mpi n: the binary through "mpirun -np n" (the MPI binary)
  --piece: only this piece of the 5p3h row (pp ring hh x23 x14 e1x; the native option deaonly,
           Engine5.sigma(only=...)); the rows to 4p2h are not compared then
The 5p3h part of the vector of the native program is stored as the parity
keeps it: per block of r5, per row (occupied triple) in their order, the
quintuples of the parity of the row (all of them without a parity).
'''
import os
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from native_check import read_amps, read_parity, section  # noqa: E402


def main():
    args = sys.argv[1:]
    opt = {'--batch': None, '--seed': '7', '--threads': '8', '--target': None, '--mpi': None, '--piece': None}
    cplx = '--complex' in args
    if cplx:
        args.remove('--complex')
    extra = []
    while '--xz' in args:
        k = args.index('--xz')
        extra.append(args[k + 1])
        del args[k:k + 2]
    for key in list(opt):
        if key in args:
            k = args.index(key)
            opt[key] = args[k + 1]
            del args[k:k + 2]
    rundir, binary = args[0], args[1]
    from zccsdt import cfour
    from zccsdt.cc import ccsdt_opt, dea_sector as D, dea5_sector as D5, t4engine, t4engine_sym as S
    eris = cfour.from_cfour(rundir, labels=True)
    imag = max(abs(np.asarray(getattr(eris, k)).imag).max() for k in ('oooo', 'ooov', 'oovv', 'ovvo', 'ovvv'))
    imag = max(imag, abs(np.asarray(eris._vvvv_pp).imag).max())
    real = imag < 1e-10 and not cplx
    if real:
        for k in ('oooo', 'ooov', 'oovv', 'ovov', 'ovvo', 'ovvv', 'fock', '_vvvv_pp'):
            setattr(eris, k, np.ascontiguousarray(np.asarray(getattr(eris, k)).real))
    eris.use_labels = True
    no = eris.nocc
    nv = np.asarray(eris.fock).shape[0] - no
    t1, t2, T3, ecc, hdr = read_amps(os.path.join(rundir, 'XZAMPS'), no, nv, 2)
    print('integrals: %d + %d, %s; XZAMPS: E_corr %.12f, %s, %d classes, %d elements of T3'
          % (no, nv, 'real' if real else 'complex', ecc, 'complex' if hdr[2] == 2 else 'real', hdr[3], hdr[4]))
    dtype = T3.dtype
    lay = S.layout_for(eris)
    assert lay is not None and int(lay.t3off[-1]) == len(T3), (lay.t3off[-1], len(T3))
    zm = open(os.path.join(rundir, 'ZMAT')).read()
    body = zm.split('%xzdea')[1].split('\n\n')[0]
    target = int([l for l in body.splitlines() if l.strip() and l.split()[0].lstrip('-').isdigit()][0].split()[0])
    if opt['--target'] is not None:
        target = int(opt['--target'])
    sh = D.shift_of(eris, target)
    sec = D.Sector(lay, sh)
    sec5 = D5.Sector5(sec)
    t0 = time.perf_counter()
    ints = t4engine.Dressed.from_eris(eris, t1)
    g = D.Ground(lay, t2, T3, ints, xv=True)
    T3p = np.asarray(lay.t3_to_packed(T3))
    g5 = D5.Ground5(sec5, g, T3p, ints.ovvo)
    eng4 = D.Engine(sec, g, 4)
    eng = D5.Engine5(sec5, g, g5, eng4)
    print('Python: ground state of the sector %.1f s; target %d, shift %d: 2p %d, 3p1h %d, 4p2h %d, 5p3h %d'
          % (time.perf_counter() - t0, target, sh, sec.n2, sec.size3, sec.size4, sec5.size5))
    # the parity: masks of r2, r3, r4, the stored columns of r5
    par = read_parity(rundir)
    m = re.search(r'parity\s+([gu])', body)
    tp = None
    if par is not None and m:
        tp = 0 if m.group(1) == 'g' else 1
    pv, po = (par[no:], par[:no]) if tp is not None else (None, None)
    masks = [np.ones(n, bool) for n in sec.sizes]
    if tp is not None:
        aq, bq = lay.va, lay.vb
        qs = slice(lay.qoff[sh], lay.qoff[sh + 1])
        masks[0] = (pv[aq[qs]] + pv[bq[qs]] - tp) % 2 == 0
        for K in sec.blocks3:
            tri = lay.tri[K].astype(np.int64)
            mk = (pv[tri].sum(axis=1)[None, :] - po[sec.rows3[K]][:, None] - tp) % 2 == 0
            masks[1][sec.off3[K]:sec.off3[K + 1]] = mk.ravel()
        for M in sec.blocks4:
            qd = sec.quads[M].astype(np.int64)
            pr = lay.o2[sec.rows4[M]]
            mk = (pv[qd].sum(axis=1)[None, :] - po[pr].sum(axis=1)[:, None] - tp) % 2 == 0
            masks[2][sec.off4[M]:sec.off4[M + 1]] = mk.ravel()
    # the stored r5: per block, per row, the columns of the parity of the row
    sel5 = []                 # (start in the full r5, column indices) per row, in the order of the stored part
    for M in sec5.blocks5:
        qn = sec5.quints[M].astype(np.int64)
        pc = pv[qn].sum(axis=1) % 2 if tp is not None else None
        for r, I in enumerate(sec5.rows5[M]):
            if tp is None:
                cols = np.arange(len(qn))
            else:
                prow = po[lay.o3[I]].sum() % 2
                cols = np.flatnonzero((pc - prow - tp) % 2 == 0)
            sel5.append((sec5.off5[M] + r * len(qn), cols))
    n5c = sum(len(c) for _, c in sel5)

    def pack5(full):
        return np.concatenate([full[o + c] for o, c in sel5]) if sel5 else np.zeros(0, dtype=full.dtype)

    def unpack5(st):
        full = np.zeros(sec5.size5, dtype=st.dtype)
        k = 0
        for o, c in sel5:
            full[o + c] = st[k:k + len(c)]
            k += len(c)
        return full
    rng = np.random.default_rng(int(opt['--seed']))
    x = [rng.standard_normal(n) for n in sec.sizes + [n5c]]
    if dtype == np.complex128:
        x = [a + 1j * rng.standard_normal(len(a)) for a in x]
    x = [np.ascontiguousarray(a, dtype=dtype) for a in x]
    for k in range(3):
        x[k] *= masks[k]
    t0 = time.perf_counter()
    pieces = {'pp': (('pp',), 'pp'), 'ring': (('ring',), 'ring'), 'hh': (('hh',), 'hh'),
              'x23': (('D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8'), 'x23'), 'x14': (('E1', 'E2'), 'x14'),
              'e1x': (('E1x',), 'e1x xv5')}
    for k in range(1, 9):
        pieces['d%d' % k] = (('D%d' % k,), 'x23 d%d' % k)
    only, nonly = (None, None) if opt['--piece'] is None else pieces[opt['--piece']]
    s2, s3, s4, s5 = eng.sigma(x[0], x[1], x[2], unpack5(x[3]), only=only)
    y = [np.array(s2) * masks[0], np.array(s3) * masks[1], np.array(s4) * masks[2], pack5(np.asarray(s5))]
    print('Python: a sigma vector %.2f s' % (time.perf_counter() - t0))
    vec = os.path.join(rundir, 'deavec.bin')
    np.concatenate(x).astype(dtype).tofile(vec)
    lines = ['deasig deavec.bin'] + extra
    if cplx:
        lines.append('real 0')
    if opt['--batch']:
        lines.append('deabatch ' + opt['--batch'])
    if nonly:
        lines.append('deaonly ' + nonly)
    zpath = os.path.join(rundir, 'ZMAT')
    keep = open(zpath).read()
    try:
        section(zpath, lines)
        if opt['--target'] is not None:
            text = open(zpath).read()
            head, rest = text.split('%xzdea\n', 1)
            sect, tail = (rest.split('\n\n', 1) + [''])[:2]
            kept = [l for l in sect.splitlines() if not (l.strip() and l.split()[0].lstrip('-').isdigit())]
            open(zpath, 'w').write(head + '%xzdea\n' + '\n'.join([opt['--target'] + ' 1'] + kept) + '\n\n' + tail)
        env = dict(os.environ, OMP_NUM_THREADS=opt['--threads'])
        cmd = [binary] if opt['--mpi'] is None else \
            ['bash', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mpirun.sh'), '-np', opt['--mpi'], binary]
        with open(os.path.join(rundir, 'xzccsdt.deasig.out'), 'w') as out:
            r = subprocess.run(cmd, cwd=rundir, stdout=out, stderr=subprocess.STDOUT, env=env)
    finally:
        open(zpath, 'w').write(keep)
    log = open(os.path.join(rundir, 'xzccsdt.deasig.out')).read()
    for line in log.splitlines():
        if 'deasig' in line or 'F,' in line or '5p3h part' in line or '(of them' in line:
            print('native: ' + line.strip())
    if r.returncode != 0:
        raise SystemExit('the native program stopped (%d), see %s/xzccsdt.deasig.out' % (r.returncode, rundir))
    got = np.fromfile(vec + '.out', dtype=dtype)
    n2, n3, n4 = sec.sizes
    parts = [got[:n2], got[n2:n2 + n3], got[n2 + n3:n2 + n3 + n4], got[n2 + n3 + n4:]]
    assert len(parts[3]) == n5c, (len(parts[3]), n5c)
    for k, name in enumerate(('2p', '3p1h', '4p2h', '5p3h')):
        d = abs(parts[k] - y[k])
        i = int(np.argmax(d)) if d.size else 0
        print('%-5s elements %12d   max |native - Python| %.2e (at %d)   max |sigma| %.2e'
              % (name, len(y[k]), d.max() if d.size else 0, i, abs(y[k]).max() if d.size else 0))


if __name__ == '__main__':
    main()
