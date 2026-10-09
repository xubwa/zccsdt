'''The lists of runs of bench/ncc_scaling/slurm_scaling.sh for the two cases
of docs/ncc_scaling.md: o3_tz.runs and c6h6_dz.runs, next to this file.

usage: python bench/ncc_scaling/mkruns.py

Series (tag without _t<threads>):
  ncc_on, ncc_off   ncc with and without point-group symmetry, CCSDT, a fixed
                    number of iterations; threads 1 2 4 8 16 24 48 and 7 23 47
                    (ncc deals its blocks to the threads by a rule on the block
                    indices that is uneven when the number of threads shares
                    factors with the number of occupied orbitals)
  ncc_onq           ncc with symmetry, CCSDT(Q)
  zc_rhf, zc_rhfq   zccsdt RCCSDT: iterations; 2 iterations and (Q)
  zc_ghf            zccsdt GCCSDT on the spin orbitals of the same RHF
  *_inter           48 threads with the memory interleaved over the sockets
  *_mkl1            ncc with MKL_NUM_THREADS=1
  *_passive, *_active   OMP_WAIT_POLICY set (the other runs: not set)
  *_mem150          ncc with MEMORY_SIZE = 150 GB instead of 70
  *_again           a run of the start of the job once more at its end
'''
import os


def runs(name, geom, basis, frozen, ncore, win, nit_ncc, nit_zc, mem, zmem, qthreads, conv_bind, tail, ghf, again, qpassive):
    lines = []
    add = lines.append
    fz = 'ON' if frozen else 'OFF'

    def bd(n):
        return 'socket' if n <= 24 else 'both'

    def ncc(tag, sym, calc, conv, maxcyc, m, n, b=None, mkl='n', wait=''):
        add(('ncc %s %s %s %s %s %s %s %s %s %d %s %s %s'
             % (tag, geom, basis, fz, sym, calc, conv, maxcyc, m, n, b or bd(n), mkl, wait)).rstrip())

    def zc(tag, kind, niter, q, n, b=None, wait=''):
        add(('zc %s %s %s %s %d %d %d %d %d %s %s'
             % (tag, kind, geom, basis, ncore, niter, q, zmem, n, b or bd(n), wait)).rstrip())

    add('# %s: runs of bench/ncc_scaling/slurm_scaling.sh (written by mkruns.py)' % name)
    add('# window %d %d' % win)
    for sym, s in (('ON', 'on'), ('OFF', 'off')):
        add('# ncc, SYMMETRY=%s: CCSDT, %d iterations' % (sym, nit_ncc))
        for n in (1, 2, 4, 7, 8, 16, 23, 24, 47, 48):
            ncc('ncc_%s_t%d' % (s, n), sym, 'T', 10, nit_ncc, mem, n)
        ncc('ncc_%s_inter_t48' % s, sym, 'T', 10, nit_ncc, mem, 48, 'inter')
        ncc('ncc_%s_mkl1_t8' % s, sym, 'T', 10, nit_ncc, mem, 8, mkl='1')
        ncc('ncc_%s_mkl1_t24' % s, sym, 'T', 10, nit_ncc, mem, 24, mkl='1')
        for n, wait in ((8, 'PASSIVE'), (24, 'PASSIVE'), (24, 'ACTIVE')):
            ncc('ncc_%s_%s_t%d' % (s, wait.lower(), n), sym, 'T', 10, nit_ncc, mem, n, wait=wait)
        if sym == 'ON':
            add('# ncc, SYMMETRY=ON: CCSDT(Q) (CC_CONV=3: (Q) needs converged iterations; '
                '24 threads: CC_CONV=10, the energies)')
            for n in qthreads:
                ncc('ncc_onq_t%d' % n, 'ON', 'Q', 10 if n == 24 else 3, 200, mem, n)
            ncc('ncc_onq_mem150_t48', 'ON', 'Q', 3, 200, 150, 48)
            if qpassive:
                ncc('ncc_onq_passive_t24', 'ON', 'Q', 3, 200, mem, 24, wait='PASSIVE')
    add('# zccsdt RCCSDT: %d iterations' % nit_zc)
    for n in (1, 2, 4, 8, 16, 24, 48):
        zc('zc_rhf_t%d' % n, 'rhf', nit_zc, 0, n)
    zc('zc_rhf_inter_t48', 'rhf', nit_zc, 0, 48, 'inter')
    for n, wait in ((8, 'PASSIVE'), (24, 'PASSIVE'), (24, 'ACTIVE')):
        zc('zc_rhf_%s_t%d' % (wait.lower(), n), 'rhf', nit_zc, 0, n, wait=wait)
    add('# zccsdt RCCSDT: 2 iterations and (Q) (24 threads: converged, the energies)')
    for n in qthreads:
        zc('zc_rhfq_t%d' % n, 'rhf', 0 if n == 24 else 2, 1, n, conv_bind if n == 24 else None)
    if qpassive:
        zc('zc_rhfq_passive_t24', 'rhf', 2, 1, 24, wait='PASSIVE')
    add('# runs of the start once more: drift of the node over the job')
    ncc('ncc_on_again_t%d' % again, 'ON', 'T', 10, nit_ncc, mem, again)
    ncc('ncc_off_again_t%d' % again, 'OFF', 'T', 10, nit_ncc, mem, again)
    zc('zc_rhf_again_t%d' % again, 'rhf', nit_zc, 0, again)
    if ghf:
        add('# zccsdt GCCSDT on the spin orbitals of the same RHF (real): 3 iterations')
        for n in (8, 24):
            zc('zc_ghf_t%d' % n, 'ghf', 3, 0, n)
    if tail:
        add('# (Q) with few threads, last: more than an hour each')
        for n in tail:
            ncc('ncc_onq_t%d' % n, 'ON', 'Q', 3, 200, mem, n)
            zc('zc_rhfq_t%d' % n, 'rhf', 2, 1, n)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), name + '.runs'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


# O3 / cc-pVTZ, all electrons: 12 occupied, 78 virtual orbitals
runs('o3_tz', 'o3.bohr', 'cc-pvtz', False, 0, (2, 5), 5, 5, 70, 70000, [4, 8, 16, 24, 48], 'socket', [2, 1], True, 8, True)
# benzene / cc-pVDZ, frozen core: 15 occupied, 93 virtual orbitals.  The
# converged zccsdt run holds 8 pairs of DIIS vectors (69 GB) in memory: more
# than one socket has with the rest, hence 'pref'
runs('c6h6_dz', 'c6h6.bohr', 'cc-pvdz', True, 6, (2, 3), 3, 3, 70, 70000, [8, 16, 24, 48], 'pref', [], False, 24, False)
