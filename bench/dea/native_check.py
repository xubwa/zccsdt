'''The sigma vector of EOM-DEA of the native program (cfour/xzccsdt, option
deasig of %xzccsdt) against zccsdt.cc.dea_sector, element by element, on
the same integral files and the same amplitudes (the file XZAMPS that the
native program wrote): a random vector of the first target of %xzdea.

usage: python native_check.py <rundir> <xzccsdt binary> [--only pieces] [--batch n] [--seed n] [--threads n]
                               [--target t] [--complex] [--xz 'line of %xzccsdt' ...]
  rundir: made by native_rundir.py, with XZAMPS of a native CCSDT run in it
  --only: pieces of the sigma vector (pp hh ring x22 x13 row3 row2), as Engine.sigma(only=...)
  --batch: option deabatch of the native run (elements of an operand of a batch; small: many batches)
  --target: this target instead of the first of %xzdea (the section is rewritten for the run)
  --complex: the native run in complex arithmetic (option real 0; its XZAMPS must be complex too),
             the Python sigma vector on the complex integrals of the files
'''
import os
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def read_amps(path, no, nv, ncplx):
    with open(path, 'rb') as f:
        tag = f.read(8)
        assert tag == b'XZAMPS01', tag
        hdr = np.frombuffer(f.read(40), dtype='<i8')
        e = np.frombuffer(f.read(8), dtype='<f8')[0]
        dt = np.complex128 if hdr[2] == 2 else np.float64
        n3 = int(hdr[4])
        t1 = np.frombuffer(f.read(no * nv * np.dtype(dt).itemsize), dtype=dt).reshape(no, nv)
        t2 = np.frombuffer(f.read(no * no * nv * nv * np.dtype(dt).itemsize), dtype=dt).reshape(no, no, nv, nv)
        t3 = np.frombuffer(f.read(n3 * np.dtype(dt).itemsize), dtype=dt)
    assert hdr[0] == no and hdr[1] == nv, hdr
    return t1.copy(), t2.copy(), t3.copy(), e, hdr


def read_parity(rundir):
    path = os.path.join(rundir, 'xzparity')
    if not os.path.isfile(path):
        return None
    nums = [int(x) for line in open(path) if not line.lstrip().startswith('#') for x in line.split()]
    return np.array(nums[1:], dtype=np.int64)


def section(zmat, opts):
    '''The ZMAT with the section %xzccsdt replaced by the lines opts.'''
    text = open(zmat).read()
    text = re.sub(r'\n%xzccsdt\n(.*\n)*?(\n|$)', '\n', text)
    if opts:
        text = text.rstrip('\n') + '\n\n%xzccsdt\n' + '\n'.join(opts) + '\n\n'
    open(zmat, 'w').write(text)


def main():
    args = sys.argv[1:]
    opt = {'--only': None, '--batch': None, '--seed': '7', '--threads': '8', '--target': None}
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
    from zccsdt.cc import dea_sector as D, t4engine, t4engine_sym as S
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
    t0 = time.perf_counter()
    ints = t4engine.Dressed.from_eris(eris, t1)
    g = D.Ground(lay, t2, T3, ints, xv=True)
    eng = D.Engine(sec, g, 4)
    print('Python: ground state of the sector %.1f s; target %d, shift %d: 2p %d, 3p1h %d, 4p2h %d'
          % (time.perf_counter() - t0, target, sh, sec.n2, sec.size3, sec.size4))
    rng = np.random.default_rng(int(opt['--seed']))
    x = [rng.standard_normal(n) for n in sec.sizes]
    if dtype == np.complex128:
        x = [a + 1j * rng.standard_normal(len(a)) for a in x]
    x = [np.ascontiguousarray(a, dtype=dtype) for a in x]
    only = None if opt['--only'] is None else tuple(opt['--only'].split(','))
    t0 = time.perf_counter()
    y = list(eng.sigma(x[0], x[1], x[2], only=only))
    print('Python: a sigma vector %.2f s' % (time.perf_counter() - t0))
    par = read_parity(rundir)
    m = re.search(r'parity\s+([gu])', body)
    if par is not None and m:
        tp = 0 if m.group(1) == 'g' else 1
        pv, po = par[no:], par[:no]
        aq, bq = lay.va, lay.vb
        qs = slice(lay.qoff[sh], lay.qoff[sh + 1])
        y[0] *= (pv[aq[qs]] + pv[bq[qs]] - tp) % 2 == 0
        for K in sec.blocks3:
            tri = lay.tri[K].astype(np.int64)
            mk = (pv[tri].sum(axis=1)[None, :] - po[sec.rows3[K]][:, None] - tp) % 2 == 0
            y[1][sec.off3[K]:sec.off3[K + 1]] *= mk.ravel()
        for M in sec.blocks4:
            qd = sec.quads[M].astype(np.int64)
            pr = lay.o2[sec.rows4[M]]
            mk = (pv[qd].sum(axis=1)[None, :] - po[pr].sum(axis=1)[:, None] - tp) % 2 == 0
            y[2][sec.off4[M]:sec.off4[M + 1]] *= mk.ravel()
        print('masks of the parity %s applied' % m.group(1))
    vec = os.path.join(rundir, 'deavec.bin')
    np.concatenate(x).astype(dtype).tofile(vec)
    lines = ['deasig deavec.bin'] + extra
    if cplx:
        lines.append('real 0')
    if only:
        lines.append('deaonly ' + ' '.join(only))
    if opt['--batch']:
        lines.append('deabatch ' + opt['--batch'])
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
        with open(os.path.join(rundir, 'xzccsdt.deasig.out'), 'w') as out:
            r = subprocess.run([binary], cwd=rundir, stdout=out, stderr=subprocess.STDOUT, env=env)
    finally:
        open(zpath, 'w').write(keep)
    log = open(os.path.join(rundir, 'xzccsdt.deasig.out')).read()
    for line in log.splitlines():
        if 'deasig' in line or 'F,' in line:
            print('native: ' + line.strip())
    if r.returncode != 0:
        raise SystemExit('the native program stopped (%d), see %s/xzccsdt.deasig.out' % (r.returncode, rundir))
    got = np.fromfile(vec + '.out', dtype=dtype)
    n2, n3 = sec.n2, sec.size3
    parts = [got[:n2], got[n2:n2 + n3], got[n2 + n3:]]
    for k, name in enumerate(('2p', '3p1h', '4p2h')):
        d = abs(parts[k] - y[k])
        i = int(np.argmax(d)) if d.size else 0
        print('%-5s elements %10d   max |native - Python| %.2e (at %d)   max |sigma| %.2e'
              % (name, len(y[k]), d.max() if d.size else 0, i, abs(y[k]).max() if d.size else 0))


if __name__ == '__main__':
    main()
