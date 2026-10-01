'''Minimal FMA counts of the dominant CCSDT terms for different spin
representations (closed-shell molecule, o/v = spatial occupied/virtual).

  spin-orbital : all 2o x 2v spin orbitals, antisymmetry exploited, no Sz
                 (what zccsdt does; required for spinors)
  Sz-blocked   : spin-orbital terms restricted to Ms-conserving blocks
                 (what a spin-integrated UHF code such as CFOUR vcc/ecc can do;
                 zccsdt UCCSDT)
  RCCSDT       : Sz-blocked with alpha <-> beta spin flip, only the target
                 blocks with 0 or 1 beta indices (zccsdt RCCSDT)
  spin-adapted : zccsdt.sa.rccsdt (RHF, NOSA): T3 blocks i<=j<=k (not all
                 equal); pp ladder as (d,e)-symmetric and antisymmetric
                 packed GEMMs, 3 pair views of a block with distinct indices
                 (v^5/2 FMA each) and 1.5 views' worth for a repeated index;
                 ring 4 o^2 v^4 FMA per occupied pair p <= q (o v^2 rows of
                 X, contractions over (m, d) with t~ / t, t2-, t3-ordered t)
  ncc model    : CFOUR ncc (RHF, spin-adapted), counted from its source:
                 T3 blocks i<=j<=k (not all equal), each a full abc cube;
                 pp ladder = 3 GEMMs of v^2 x v^2 x v per block (MAGIC_CYCLE
                 states with constraint 1/4/7); rings = 9 (wmaei) plus 13-26
                 (wmaie, 13 states x 2 types, some skipped by an index-order
                 test) contractions of o v^4 per block.  o, v spatial.
'''
import sys
from collections import Counter
from itertools import combinations


def spins(n):
    return [0] * n + [1] * n               # n alpha then n beta


def pair_ms(n):
    c = Counter()
    for p, q in combinations(spins(n), 2):
        c[p + q] += 1
    return c


def triple_ms(n):
    c = Counter()
    for p, q, r in combinations(spins(n), 3):
        c[p + q + r] += 1
    return c


def ladder(o, v):
    '''sum_{d<e} W(ab,de) t(ijk,dec) for all i<j<k, a<b, c.'''
    O, V = 2 * o, 2 * v
    full = (O * (O - 1) * (O - 2) // 6) * (V * (V - 1) // 2) ** 2 * V
    tO, pV, sV = triple_ms(o), pair_ms(v), Counter({0: v, 1: v})
    sz = rf = 0
    for mI, nI in tO.items():
        for mc, nc in sV.items():
            for mab, nab in pV.items():      # W conserves Ms: Ms(de) = Ms(ab)
                if mab + mc == mI:
                    sz += nI * nc * nab * pV[mab]
                    if mI <= 1:
                        rf += nI * nc * nab * pV[mab]
    return full, sz, rf


def ring(o, v):
    '''sum_{m,d} W(ma,di) t(mjk,dbc) for all i,a and pairs j<k, b<c.'''
    O, V = 2 * o, 2 * v
    full = (O * (O - 1) // 2) * (V * (V - 1) // 2) * (O * V) ** 2
    pO, pV, s = pair_ms(o), pair_ms(v), Counter({0: o, 1: o})
    sv = Counter({0: v, 1: v})
    sz = rf = 0
    for mjk, njk in pO.items():
        for mbc, nbc in pV.items():
            for mi, ni in s.items():
                for ma, na in sv.items():
                    if mi + mjk != ma + mbc:
                        continue
                    for mm, nm in s.items():          # summed (m, d)
                        for md, nd in sv.items():
                            # t(mjk,dbc) needs Ms(m)+Ms(jk) = Ms(d)+Ms(bc);
                            # W(ma,di) needs Ms(m)+Ms(a) = Ms(d)+Ms(i)
                            if mm + mjk == md + mbc and mm + ma == md + mi:
                                sz += njk * nbc * ni * na * nm * nd
                                if mi + mjk <= 1:           # target block stored
                                    rf += njk * nbc * ni * na * nm * nd
    return full, sz, rf


def sa_model(o, v):
    '''(ladder, ring) FMA of the spin-adapted engine (zccsdt.sa.rccsdt).'''
    ndist = o * (o - 1) * (o - 2) // 6
    nrep = o * (o - 1)
    ladder = (ndist * 3 + nrep * 1.5) * v**5 / 2
    ring = o * (o + 1) // 2 * 4 * o**2 * v**4
    return ladder, ring


def ncc_model(o, v):
    nblk = (o + 2) * (o + 1) * o // 6 - o
    ladder = nblk * 3 * v**5
    ring_lo = nblk * (9 + 13) * o * v**4
    ring_hi = nblk * (9 + 26) * o * v**4
    return ladder, ring_lo, ring_hi


if __name__ == '__main__':
    for name, o, v in [('H2O/cc-pVTZ', 5, 53), ('N2/cc-pVTZ', 7, 53),
                       ('H2S/cc-pVTZ', 9, 53), ('CH2O/cc-pVTZ', 8, 80),
                       ('H2O/cc-pVQZ', 5, 110)]:
        lf, ls, lr = ladder(o, v)
        rf, rs, rr = ring(o, v)
        nl, nrl, nrh = ncc_model(o, v)
        sl, sr = sa_model(o, v)
        print('%-12s spin-orbital: ladder %.3e ring %.3e | Sz-blocked: ladder %.3e '
              'ring %.3e | RCCSDT: ladder %.3e ring %.3e | spin-adapted: ladder %.3e '
              'ring %.3e | ncc: ladder %.3e ring %.3e-%.3e'
              % (name, lf, rf, ls, rs, lr, rr, sl, sr, nl, nrl, nrh))
        tot, totsz, totr, tots = lf + rf, ls + rs, lr + rr, sl + sr
        print('%-12s ratio spin-orbital/Sz-blocked %.1f   spin-orbital/ncc %.1f-%.1f   '
              'RCCSDT/ncc %.2f-%.2f   spin-adapted/ncc %.2f-%.2f'
              % ('', tot / totsz, tot / (nl + nrh), tot / (nl + nrl),
                 totr / (nl + nrh), totr / (nl + nrl), tots / (nl + nrh), tots / (nl + nrl)))
