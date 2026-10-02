'''Anonymous memory by stage of an iteration of the spin-adapted CCSDT
(zccsdt.sa.rccsdt): peak resident anonymous memory inside each timer, the
data behind the memory model of rccsdt.plan_memory.

usage: python bench/memstages_sa.py MOL BASIS [PLAN-FIELDS] [NCORE] [NITER]
  PLAN-FIELDS: fields of an explicit rccsdt.MemoryPlan, e.g.
  "t3_file=True, pp_file=True, pp_bytes=64<<20, pp_order='batch', keep=False"
  ZC_MAXMEM: max_memory in MB (the plan is then the automatic one)
'''
import os
import sys

from pyscf import gto, scf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memstages_cc import anon, stage_watch                # noqa: E402
from profile_sa import GEOM, CFOUR_BOHR                   # noqa: E402
from zccsdt import timing                                 # noqa: E402
from zccsdt.sa import rccsdt, rccsdt_ref                  # noqa: E402

STAGES = ('sa.dress', 'sa.imds', 'sa.Wpp', 'sa.pp_ladder', 'sa.ccsd', 'sa.t3_pass',
          'sa.drive_hh', 'sa.project', 'sa.diis')


def main():
    peak, cur, stop = stage_watch(STAGES, (rccsdt, rccsdt_ref))
    name, basis = sys.argv[1:3]
    plan = eval('rccsdt.MemoryPlan(%s)' % sys.argv[3]) \
        if len(sys.argv) > 3 and sys.argv[3] else None
    ncore = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    niter = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    mem = {'max_memory': float(os.environ['ZC_MAXMEM'])} if os.environ.get('ZC_MAXMEM') else {}
    mol = gto.M(atom=GEOM[name], basis=basis, unit=CFOUR_BOHR, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-10)
    ints = rccsdt_ref.from_scf(mf, ncore=ncore, full=False)
    del mf
    o, v = ints.nocc, ints.nvir
    g = 8 * o * v ** 3
    print('%s/%s  nocc=%d nvir=%d (spatial)  after SCF: anonymous %.2f GB'
          % (name, basis, o, v, anon()))
    print('sizes: g = o v^3 %.3f GB, packed vvvv %.3f GB, T3 %.3f GB'
          % (g / 1e9, 8 * (v * (v + 1) // 2) ** 2 / 1e9 + 8 * (v * (v - 1) // 2) ** 2 / 1e9,
             8 * o * (o + 1) * (o + 2) // 6 * v ** 3 / 1e9))
    cur[0] = '(between the timers)'
    r = rccsdt.kernel(ints, max_cycle=niter, verbose=True, plan=plan, **mem)
    stop.set()
    print('plan: %s' % r.plan)
    print('kept on the integral object: %.2f GB' % (rccsdt.cached_bytes(ints) / 1e9))
    print('iterations: ' + ' '.join('%.2f' % t for t in r.timings))
    base = peak.get('setup', 0.0)
    for k, val in sorted(peak.items(), key=lambda kv: -kv[1]):
        print('  %-22s peak anonymous %6.2f GB  (%5.1f g above the start)'
              % (k, val, (val - base) * 1e9 / g))
    for k, sec, calls in timing.report(len(r.timings))[:12]:
        print('  %-16s %8.3fs' % (k, sec))


if __name__ == '__main__':
    main()
