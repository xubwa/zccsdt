'''Tables of a job of bench/ncc_scaling/slurm_scaling.sh, from the files it
kept (one directory per run: command.txt, and CFOUR's out with time.txt and
ncc_time.txt, or the log of zc_run.py with time.txt), in Markdown.

usage: python bench/ncc_scaling/summary.py DIRECTORY [FIRST LAST]
  FIRST LAST: the iterations whose mean is "seconds per iteration" (default:
  the line '# window FIRST LAST' of the list of runs, else 2 5); iteration 1
  of both programs is not a full one (T3 = 0; integral blocks)

A run belongs to the series named by its tag without the trailing _t<threads>;
the series ncc_on, ncc_off and zc_rhf are compared with each other
(iterations), and ncc_onq with zc_rhfq ((Q)); see mkruns.py.
'''
import glob
import os
import re
import sys


def read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ''


def time_v(text):
    '''(user + system CPU seconds, wall seconds, peak resident GB) of /usr/bin/time -v'''
    def num(pattern):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None
    user, system = num(r'User time \(seconds\): (\S+)'), num(r'System time \(seconds\): (\S+)')
    m = re.search(r'Elapsed \(wall clock\) time .*: ([\d:.]+)\s*$', text, re.M)
    wall = None
    if m:
        wall = 0.0
        for part in m.group(1).split(':'):
            wall = 60 * wall + float(part)
    rss = num(r'Maximum resident set size \(kbytes\): (\d+)')
    return (None if user is None else user + system, wall, None if rss is None else rss / 1e6)


def mean(x):
    return sum(x) / len(x) if x else None


def ncc(d, first, last):
    text = read(os.path.join(d, 'out'))

    def lastof(pattern, cast=str):
        m = re.findall(pattern, text)
        return cast(m[-1]) if m else None

    r = {'program': 'ncc'}
    r['group'] = lastof(r'computational point group is (\S+)')
    r['nbf'] = lastof(r'There are\s+(\d+) functions in the AO basis', int)
    rows = re.findall(r'^\s*(\d+)\s+(-?\d+\.\d{10,})\s+\S+\s+\S+\s+\S+\s+(\d+\.\d+)\s+(\d+\.\d+)\s*$', text, re.M)
    r['energies'] = [float(x[1]) for x in rows]
    cpu, wall = [float(x[2]) for x in rows], [float(x[3]) for x in rows]
    r['n_it'] = len(rows)
    r['it'] = mean(wall[first - 1:last])
    r['it_cpu'] = mean(cpu[first - 1:last])
    r['n_window'] = len(wall[first - 1:last])
    r['converged'] = lastof(r'iterations converged in (\d+) cycles', int)
    m = re.search(r'calculation of CCSDT\(Q\) took (\S+) cpu seconds and (\S+) walltime', text)
    if m:
        r['q_cpu'], r['q'] = float(m.group(1)), float(m.group(2))
    r['e_scf'] = lastof(r'E\(SCF\)=\s+(-?\d+\.\d+)', float)
    r['e_ccsdt'] = lastof(r'Total CCSDT energy:\s+(-?\d+\.\d+)', float)
    r['e_q'] = lastof(r'CCSDT\(Q\) correlation energy:\s+(-?\d+\.\d+)', float)
    r['e_ccsdtq'] = lastof(r'Total CCSDT\(Q\) energy:\s+(-?\d+\.\d+)', float)
    r['mem_limit'] = lastof(r'Memory limit:\s+(.+)')
    r['mem_optimal'] = lastof(r'Optimal memory requirement:\s+(.+)')
    r['threads_ncc'] = lastof(r'Running with (\d+) threads', int)
    r['cpu'], r['wall'], r['rss'] = time_v(read(os.path.join(d, 'ncc_time.txt')))
    r['nlwp'] = read(os.path.join(d, 'nlwp.txt')).strip() or None
    return r


def zc(d, first, last):
    text = read(os.path.join(d, 'log'))
    v = dict(re.findall(r'^ZC (\w+) = (.*)$', text, re.M))
    r = {'program': 'zccsdt ' + v.get('class', '?')}
    wall = [float(x) for x in v.get('iterations', '').split()]
    cpu = [float(x) for x in v.get('iteration_cpu', '').split()]
    r['n_it'] = len(wall)
    r['it'] = mean(wall[first - 1:last])
    r['it_cpu'] = mean(cpu[first - 1:last])
    r['n_window'] = len(wall[first - 1:last])
    r['converged'] = int(v['cycles']) if v.get('converged') == 'True' else None
    for key, name in (('q_wall', 'q'), ('q_cpu', 'q_cpu'), ('e_scf', 'e_scf'), ('e_ccsdt', 'e_ccsdt'),
                      ('e_q', 'e_q'), ('e_ccsdtq', 'e_ccsdtq'), ('peak_anon_gb', 'anon')):
        if key in v:
            r[name] = float(v[key])
    m = re.search(r'orbitals\s+(.*)', text)
    r['orbitals'] = m.group(1).strip() if m else None
    m = re.search(r'n_file=(\d+), n_memory=(\d+)', v.get('diis', ''))
    r['diis'] = 'file %s, memory %s' % m.groups() if m else None
    r['cpu'], r['wall'], r['rss'] = time_v(read(os.path.join(d, 'time.txt')))
    if 'Traceback' in text:
        r['error'] = text.strip().splitlines()[-1]
    return r


def f(x, fmt='%.2f'):
    return '' if x is None else fmt % x


def ratio(a, b, fmt='%.2f'):
    return '' if a is None or b is None or b == 0 else fmt % (a / b)


def table(head, rows):
    print('| ' + ' | '.join(head) + ' |')
    print('|' + '---|' * len(head))
    for row in rows:
        print('| ' + ' | '.join(str(x) for x in row) + ' |')
    print()


def main():
    top = sys.argv[1].rstrip('/')
    window = (2, 5)
    m = re.search(r'^#\s*window\s+(\d+)\s+(\d+)', read(os.path.join(top, 'runs')), re.M)
    if m:
        window = (int(m.group(1)), int(m.group(2)))
    if len(sys.argv) > 3:
        window = (int(sys.argv[2]), int(sys.argv[3]))
    runs = []
    for d in sorted(glob.glob(os.path.join(top, '*', 'command.txt'))):
        d = os.path.dirname(d)
        tag = os.path.basename(d)
        r = ncc(d, *window) if os.path.exists(os.path.join(d, 'out')) else zc(d, *window)
        cmd = read(os.path.join(d, 'command.txt'))
        r['tag'] = tag
        r['series'] = re.sub(r'_t\d+$', '', tag)
        r['threads'] = int(re.search(r'OMP_NUM_THREADS=(\d+)', cmd).group(1))
        r['mkl'] = int(re.search(r'MKL_NUM_THREADS=(\d+)', cmd).group(1))
        b = re.search(r'--(membind=0|preferred=0|interleave=all)', cmd)
        r['bind'] = b.group(1) if b else 'first touch'
        r['wait'] = re.search(r'OMP_WAIT_POLICY=(\w+)', cmd).group(1)
        runs.append(r)
    runs.sort(key=lambda r: (r['series'], r['threads']))
    print('%s: seconds per CCSDT iteration = mean of iterations %d-%d\n' % (os.path.basename(top), *window))

    print('All runs (CPU / wall: of the iterations of the window and of (Q), by the timers of the programs; '
          '"process": xncc or python as a whole, /usr/bin/time):\n')
    table(['run', 'threads', 'memory policy', 'iterations', 's / iteration', 'cores busy', '(Q) s', 'cores busy',
           'process wall s', 'process cores busy', 'peak RSS GB', 'note'],
          [[r['tag'], r['threads'], r['bind'], r['n_it'], f(r['it'], '%.3f'), ratio(r['it_cpu'], r['it']),
            f(r.get('q')), ratio(r.get('q_cpu'), r.get('q')), f(r['wall'], '%.0f'), ratio(r['cpu'], r['wall']),
            f(r['rss']),
            '; '.join(x for x in (
                'converged in %d' % r['converged'] if r['converged'] else '',
                'MKL threads %d' % r['mkl'] if r['mkl'] != r['threads'] else '',
                'OMP_WAIT_POLICY=%s' % r['wait'] if r['wait'] != 'default' else '',
                'threads seen %s' % r['nlwp'] if r.get('nlwp') else '',
                'memory limit %s, optimal %s' % (r['mem_limit'], r['mem_optimal']) if r.get('mem_limit') else '',
                'window of %d' % r['n_window'] if r['n_window'] != window[1] - window[0] + 1 else '',
                r.get('diis') and 'DIIS pairs: ' + r['diis'] or '', r.get('error', '')) if x)]
           for r in runs])

    series = {}
    for r in runs:
        series.setdefault(r['series'], {})[r['threads']] = r
    for what, key in (('seconds per CCSDT iteration', 'it'), ('seconds of (Q)', 'q')):
        for name, s in series.items():
            have = {n: r[key] for n, r in s.items() if r.get(key) is not None}
            if len(have) < 2:
                continue
            print('%s, %s (%s):\n' % (name, what, next(iter(s.values()))['program']))
            table(['threads'] + list(have),
                  [[what] + [f(t, '%.3f' if t < 10 else '%.2f') for t in have.values()],
                   ['speedup from 1 thread'] + [ratio(have.get(1), t) for t in have.values()],
                   ['speedup from 8 threads'] + [ratio(have.get(8), t) for t in have.values()],
                   ['efficiency (speedup from 1 / threads)']
                   + [ratio(have.get(1), t * n) for n, t in have.items()]])
    for what, key, sfx in (('seconds per CCSDT iteration', 'it', ''), ('seconds of (Q)', 'q', 'q')):
        ref = series.get('zc_rhf' + sfx, {})
        rows = []
        for n in sorted(ref):
            a, b, z = (series.get(k + sfx, {}).get(n, {}).get(key) for k in ('ncc_on', 'ncc_off', 'zc_rhf'))
            if z is None:
                continue
            rows.append([n, f(a, '%.3f'), f(b, '%.3f'), f(z, '%.3f'), ratio(a, z), ratio(b, z)])
        if rows:
            print('ncc against zccsdt RCCSDT, %s:\n' % what)
            table(['threads', 'ncc, symmetry', 'ncc, no symmetry', 'zccsdt', 'ncc (symmetry) / zccsdt',
                   'ncc (no symmetry) / zccsdt'], rows)

    conv = [r for r in runs if r['converged'] and r.get('e_ccsdt') is not None]
    if conv:
        print('Energies of the converged runs:\n')
        table(['run', 'E(SCF)', 'E(CCSDT)', 'E(Q)', 'E(CCSDT(Q))'],
              [[r['tag'], f(r.get('e_scf'), '%.12f'), f(r['e_ccsdt'], '%.12f'), f(r.get('e_q'), '%.12f'),
                f(r.get('e_ccsdtq'), '%.12f')] for r in conv])
        z = [r for r in conv if r['program'].startswith('zccsdt')]
        if z:
            z = max(z, key=lambda r: r.get('e_ccsdtq') is not None)
            for r in conv:
                if r is not z:
                    print('%s - %s: CCSDT %s  CCSDT(Q) %s' % (
                        r['tag'], z['tag'], f(r['e_ccsdt'] - z['e_ccsdt'], '%.1e'),
                        f(r['e_ccsdtq'] - z['e_ccsdtq'], '%.1e')
                        if r.get('e_ccsdtq') is not None and z.get('e_ccsdtq') is not None else ''))
            print()
    # ncc with and without symmetry is one program: the correlation energies of
    # the iterations of two runs must be the same numbers
    on, off = series.get('ncc_on', {}), series.get('ncc_off', {})
    pairs = [(on[n], off[n]) for n in sorted(on) if n in off and on[n]['energies'] and off[n]['energies']]
    if pairs:
        a, b = pairs[-1]
        k = min(len(a['energies']), len(b['energies']))
        print('ncc with and without symmetry, largest difference of the correlation energies of iterations '
              '1-%d (%s, %s): %.1e\n' % (k, a['tag'], b['tag'],
                                        max(abs(x - y) for x, y in zip(a['energies'][:k], b['energies'][:k]))))


if __name__ == '__main__':
    main()
