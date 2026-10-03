'''Summary of an xx2cccsd output (CFOUR two-component CCSD(T)): orbital
counts, memory, the mean wall time of its timers per residual evaluation
(the first left out), the CCSD and (T) totals.

usage: python bench/cfour/x2ccc_summary.py xx2cccsd.out
'''
import re
import sys
from collections import defaultdict

text = open(sys.argv[1]).read()
STAGES = ('Intermediate', 'PPPP ladder', 'HHHH ladder', 'Ring Diagram')


def first(pattern, cast=str):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else None


no = first(r'no \(occupied orbital\)\s+(\d+)', int)
nu = first(r'nu \(virtual orbital\)\s+(\d+)', int)
mem = re.findall(r'Memory used for (CCSD(?:\(T\) energy)?) \(in GB\)\s+(\S+)', text)
print('xx2cccsd: no = %s, nu = %s spinors; %s'
      % (no, nu, ', '.join('%s %.2f GB' % (k, float(v)) for k, v in mem)))

# One residual evaluation (hbarso) prints I1-I9 (the parts of
# "Intermediate"), "Intermediate", t2abcdao* (the parts of the PPPP
# ladder), "PPPP ladder", "HHHH ladder", "Ring Diagram" -- the last one
# closes it.  The final evaluation (the convergence test) has no
# "CCSD cycle" line after it.
evals, cur = [], defaultdict(float)
for line in text.splitlines():
    m = re.match(r'\s*(.+?) Elapsed wall clock time\s*=\s*(\S+)', line)
    if not m:
        continue
    name, sec = m.group(1).strip(), float(m.group(2))
    if name in ('CCSD', '(T)', 'MO Integral sorting', 'AO Integral'):
        continue
    cur[name] += sec
    if name == 'Ring Diagram':
        evals.append(cur)
        cur = defaultdict(float)
n = len(evals)
ncyc = len(re.findall(r'CCSD cycle\s+\d+', text))
ccsd = first(r'CCSD Elapsed wall clock time =\s*(\S+)', float) or 0.0
t3 = first(r'\(T\) Elapsed wall clock time =\s*(\S+)', float) or 0.0
print('%d CCSD cycles, %d residual evaluations, %.1f s in all (%.2f s per evaluation); '
      '(T) %.1f s' % (ncyc, n, ccsd, ccsd / max(n, 1), t3))
if n > 1:
    rest = evals[1:]
    mean = {k: sum(e[k] for e in rest) / len(rest) for k in {k for e in rest for k in e}}
    print('mean per residual evaluation (evaluations 2-%d):' % n)
    for k in STAGES:
        print('  %-24s %9.3f s' % (k, mean.get(k, 0.0)))
        parts = (sorted([p for p in mean if re.fullmatch(r'I\d+', p)], key=lambda p: (p[1], p))
                 if k == 'Intermediate' else
                 sorted(p for p in mean if p.startswith('t2abcdao')) if k == 'PPPP ladder'
                 else [])
        for p in parts:
            print('    %-22s %9.3f s' % (p, mean[p]))
    print('  %-24s %9.3f s' % ('not timed (T1/T2 terms, DIIS, update)',
                               ccsd / n - sum(mean.get(k, 0.0) for k in STAGES)))
e = first(r'final CCSD correlation energy:\s*(\S+)') or 'not converged'
et = first(r'Total triples correction in CCSD\(T\)\s*\(\s*([^,]+),')
print('E_corr(CCSD) %s; (T) correction %s; %s'
      % (e, et, first(r'(?m)^(CCSD\(T\) energy\s+\S+)') or 'no CCSD(T) energy'))
