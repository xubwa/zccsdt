'''Minimal FMA counts of the dominant CCSDT terms for different spin
representations (closed-shell molecule, o/v = spatial occupied/virtual).

  spin-orbital : all 2o x 2v spin orbitals, antisymmetry exploited, no Sz
                 (what zccsdt does; required for spinors)
  Sz-blocked   : spin-orbital terms restricted to Ms-conserving blocks
                 (what a spin-integrated UHF code such as CFOUR vcc/ecc can do)
The RHF spin-adapted count (CFOUR ncc) is lower still and not modelled.
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
    sz = 0
    for mI, nI in tO.items():
        for mc, nc in sV.items():
            for mab, nab in pV.items():      # W conserves Ms: Ms(de) = Ms(ab)
                if mab + mc == mI:
                    sz += nI * nc * nab * pV[mab]
    return full, sz


def ring(o, v):
    '''sum_{m,d} W(ma,di) t(mjk,dbc) for all i,a and pairs j<k, b<c.'''
    O, V = 2 * o, 2 * v
    full = (O * (O - 1) // 2) * (V * (V - 1) // 2) * (O * V) ** 2
    pO, pV, s = pair_ms(o), pair_ms(v), Counter({0: o, 1: o})
    sv = Counter({0: v, 1: v})
    sz = 0
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
    return full, sz


if __name__ == '__main__':
    for name, o, v in [('H2O/cc-pVTZ', 5, 53), ('N2/cc-pVTZ', 7, 53)]:
        lf, ls = ladder(o, v)
        rf, rs = ring(o, v)
        print('%-12s ladder %.3e / %.3e (x%.1f)   ring %.3e / %.3e (x%.1f)'
              % (name, lf, ls, lf / ls, rf, rs, rf / rs))
