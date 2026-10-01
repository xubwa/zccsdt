'''Closed-shell spin-adapted CCSDT equations, derived from the spin-orbital
equations of zccsdt.cc.ccsdt (T1-dressed formalism) with zccsdt.sa.spinfree.

All quantities are T1-dressed: V = dressed <pq|rs>, F = dressed Fock, and T1
does not appear.  Spin-orbital factor kinds and index conventions are those of
zccsdt.sa.spinfree.  Running this module prints the spatial equations.

Intermediates (spin-orbital definitions, as in ccsdt._t3_residual):

  Wv(ab,e,i) = <ab||ei> + 1/2 <mn||ei> t(mn,ab) - P(ab) <mb||ef> t(mi,af)
               - f(m,e) t(mi,ab) + 1/2 <mn||ef> t(imn,abf)
  Wo(m,a,j,i) = <ma||ji> - <mn||ie> t(jn,ae) + <mn||je> t(in,ae)
               - 1/2 <ma||ef> t(ij,ef) + 1/2 <mn||ef> t(ijn,aef)
  Fv(a,d) = f(a,d) - 1/2 t(mn,af) <mn||df>
  Fo(m,i) = f(m,i) + 1/2 <mn||ef> t(in,ef)
  Wr(m,a,d,i) = <ma||di> + <mn||de> t(in,ae)
  Wpp(ab,de) = <ab||de> + 1/2 t(mn,ab) <mn||de>
  Whh(mn,ij) = <mn||ij> + 1/2 <mn||ef> t(ij,ef)

Every 4-index intermediate N is spin-free:
N(pqrs) = N.J_pqrs d(p,r) d(q,s) - N.K_pqrs d(p,s) d(q,r).
'''

from fractions import Fraction

from zccsdt.sa.spinfree import (Term, antisymmetrize, P_one_two, P_pair, full_asym,
                                trace_all, simplify, show)

EXT1 = [('i', 'a')]
EXT2 = [('i', 'a'), ('j', 'b')]
EXT3 = [('i', 'a'), ('j', 'b'), ('k', 'c')]


def T(coef, *factors):
    return Term(coef, factors)


# ------------------------------------------------------- intermediates

def intermediate_terms():
    '''name -> (output slots, spin-orbital terms).'''
    return {
        'Wv': ('abei', [T(1, ('v', 'abei')),
                        T('1/2', ('v', 'mnei'), ('t2', 'mnab')),
                        *antisymmetrize(T(-1, ('v', 'mbef'), ('t2', 'miaf')), [P_pair('a', 'b')]),
                        T(-1, ('f', 'me'), ('t2', 'miab')),
                        T('1/2', ('v', 'mnef'), ('t3', 'imnabf'))]),
        'Wo': ('maji', [T(1, ('v', 'maji')),
                        T(-1, ('v', 'mnie'), ('t2', 'jnae')),
                        T(1, ('v', 'mnje'), ('t2', 'inae')),
                        T('-1/2', ('v', 'maef'), ('t2', 'ijef')),
                        T('1/2', ('v', 'mnef'), ('t3', 'ijnaef'))]),
        'Fv': ('ad', [T(1, ('f', 'ad')), T('-1/2', ('t2', 'mnaf'), ('v', 'mndf'))]),
        'Fo': ('mi', [T(1, ('f', 'mi')), T('1/2', ('v', 'mnef'), ('t2', 'inef'))]),
        'Wr': ('madi', [T(1, ('v', 'madi')), T(1, ('v', 'mnde'), ('t2', 'inae'))]),
        'Wpp': ('abde', [T(1, ('v', 'abde')), T('1/2', ('t2', 'mnab'), ('v', 'mnde'))]),
        'Whh': ('mnij', [T(1, ('v', 'mnij')), T('1/2', ('v', 'mnef'), ('t2', 'ijef'))]),
    }


def spatial_intermediates():
    '''Spatial components: name.J / name.K (4-index) or name (2-index) ->
    simplified spatial terms (with the slot letters as output indices).'''
    out = {}
    for name, (slots, terms) in intermediate_terms().items():
        if len(slots) == 2:
            out[name] = simplify(trace_all(terms, [(slots[0], slots[1])]), slots)
        else:
            p, q, r, s = slots
            out[name + '.J'] = simplify(trace_all(terms, [(p, r), (q, s)]), slots)
            # N = J d(p,r)d(q,s) - K d(p,s)d(q,r): K is minus the coefficient
            kt = trace_all(terms, [(p, s), (q, r)])
            out[name + '.K'] = simplify([(-c, f) for c, f in kt], slots)
    return out


# ------------------------------------------------------- residuals

def t3_residual_terms():
    '''Spin-orbital T3 residual (ccsdt._t3_residual) with the intermediates as
    spin-free tensors.'''
    Wv, Wo, Wr, Wpp, Whh = (('w', n) for n in ('Wv', 'Wo', 'Wr', 'Wpp', 'Whh'))
    terms = []
    drive = [T('-1/4', (Wv, 'abei'), ('t2', 'jkec')), T('-1/4', (Wo, 'maji'), ('t2', 'mkbc'))]
    for t in drive:
        terms += antisymmetrize(t, full_asym('ijk', 'abc'))
    terms += antisymmetrize(T(1, (('one', 'Fv'), 'ad'), ('t3', 'ijkdbc')), [P_one_two('a', 'b', 'c')])
    terms += antisymmetrize(T(-1, (('one', 'Fo'), 'mi'), ('t3', 'mjkabc')), [P_one_two('i', 'j', 'k')])
    terms += antisymmetrize(T(1, (Wr, 'madi'), ('t3', 'mjkdbc')),
                            [P_one_two('i', 'j', 'k'), P_one_two('a', 'b', 'c')])
    terms += antisymmetrize(T('1/2', (Wpp, 'abde'), ('t3', 'ijkdec')), [P_one_two('c', 'a', 'b')])
    terms += antisymmetrize(T('1/2', (Whh, 'mnij'), ('t3', 'mnkabc')), [P_one_two('k', 'i', 'j')])
    return terms


def t2_residual_terms():
    '''Spin-orbital CCSD T2 residual (ccsdt_opt.ccsd_residual_packed, T1-dressed,
    intermediates expanded) plus the T3 -> T2 terms.'''
    terms = [T(1, ('v', 'abij')),
             T('1/2', ('v', 'abcd'), ('t2', 'ijcd')),
             T('1/2', ('v', 'klij'), ('t2', 'klab')),
             T('1/4', ('v', 'klcd'), ('t2', 'ijcd'), ('t2', 'klab'))]
    # P(ab) F_bc t_ijac, F_bc = f_bc - 1/2 t(mn,bf) <mn||cf>
    terms += antisymmetrize(T(1, ('f', 'bc'), ('t2', 'ijac')), [P_pair('a', 'b')])
    terms += antisymmetrize(T('-1/2', ('t2', 'mnbf'), ('v', 'mncf'), ('t2', 'ijac')), [P_pair('a', 'b')])
    # - P(ij) F_kj t_ikab, F_kj = f_kj + 1/2 <kn||ef> t(jn,ef)
    terms += antisymmetrize(T(-1, ('f', 'kj'), ('t2', 'ikab')), [P_pair('i', 'j')])
    terms += antisymmetrize(T('-1/2', ('v', 'knef'), ('t2', 'jnef'), ('t2', 'ikab')), [P_pair('i', 'j')])
    # P(ij)P(ab) W_kbcj t_ikac, W_kbcj = <kb||cj> + 1/2 <kl||cd> t(jl,bd)
    for t in (T(1, ('v', 'kbcj'), ('t2', 'ikac')),
              T('1/2', ('v', 'klcd'), ('t2', 'jlbd'), ('t2', 'ikac'))):
        terms += antisymmetrize(t, [P_pair('i', 'j'), P_pair('a', 'b')])
    # T3 -> T2
    terms.append(T(1, ('f', 'me'), ('t3', 'ijmabe')))
    terms += antisymmetrize(T('1/2', ('v', 'amef'), ('t3', 'ijmebf')), [P_pair('a', 'b')])
    terms += antisymmetrize(T('1/2', ('v', 'mnej'), ('t3', 'imnabe')), [P_pair('i', 'j')])
    return terms


def t1_residual_terms():
    '''Spin-orbital CCSD T1 residual (T1-dressed) plus the T3 -> T1 term.'''
    return [T(1, ('f', 'ai')),
            T(1, ('f', 'kc'), ('t2', 'ikac')),
            T('1/2', ('v', 'akcd'), ('t2', 'ikcd')),
            T('-1/2', ('v', 'klic'), ('t2', 'klac')),
            T('1/4', ('v', 'mnef'), ('t3', 'imnaef'))]


def spatial_residuals():
    '''Simplified spatial residuals: 'R1' (ia), 'R2' (ijab), 'R3' (ijkabc).'''
    return {'R1': simplify(trace_all(t1_residual_terms(), EXT1), 'ia'),
            'R2': simplify(trace_all(t2_residual_terms(), EXT2), 'ijab'),
            'R3': simplify(trace_all(t3_residual_terms(), EXT3), 'ijkabc')}


# ------------------------------------------------------------ CCSDT(Q)

EXT4 = [('i', 'a'), ('j', 'b'), ('k', 'c'), ('l', 'd')]


def q_terms():
    '''Spin-orbital quantities of the CCSDT(Q) correction (bare integrals,
    canonical HF orbitals; T1 does not enter):

      r(ijkl,abcd) = <ijkl abcd| [W, T3] + 1/2 [[W, T2], T2] |0>,
      z(ijkl,abcd) = <0| (T2^+ + T3^+) W |ijkl abcd>,
      E(Q) = 1/(4!)^2 sum z r / (e_i + e_j + e_k + e_l - e_a - e_b - e_c - e_d),

    each as the full antisymmetrizer over ijkl and abcd applied to one
    representative product.  The coefficients were fixed by a brute-force
    Fock-space evaluation (tests/test_sa_q.py); 'R4.T3' / 'R4.T2' are the two
    parts of r and 'Z4.T2' (disconnected, fifth order: its part of the energy
    is the [Q] correction) / 'Z4.T3' those of z.'''
    A = full_asym('ijkl', 'abcd')

    def asym(*base):
        return [t for b in base for t in antisymmetrize(b, A)]
    f24, f16 = Fraction(1, 24), Fraction(1, 16)
    return {
        'R4.T3': asym(T(f24, ('t3', 'ijkabe'), ('v', 'cdel')),
                      T(-f24, ('t3', 'ijmabc'), ('v', 'mdkl'))),
        'R4.T2': asym(T(f16, ('t2', 'ijae'), ('t2', 'klfd'), ('v', 'bcef')),
                      T(f16, ('t2', 'imab'), ('t2', 'nlcd'), ('v', 'mnjk')),
                      T(Fraction(1, 4), ('t2', 'ijae'), ('t2', 'mlcd'), ('v', 'mbek'))),
        'Z4.T2': asym(T(f16, ('t2', 'ijab'), ('v', 'klcd'))),
        'Z4.T3': asym(T(f24, ('t3', 'ijkabe'), ('v', 'elcd')),
                      T(-f24, ('t3', 'ijmabc'), ('v', 'klmd'))),
    }


_Q_SPATIAL = {}


def q_spatial():
    '''Spatial (identity-pairing) coefficients r_ijkl^abcd and z_ijkl^abcd of
    q_terms(), simplified; cached.'''
    if not _Q_SPATIAL:
        for name, terms in q_terms().items():
            _Q_SPATIAL[name] = simplify(trace_all(terms, EXT4), 'ijklabcd')
    return _Q_SPATIAL


# weights sgn(tau) 2^cycles(tau) of the energy: with spatial z and r,
#   E(Q) = 1/24 sum_{ijkl,abcd} z_ijkl^abcd / D sum_tau w(tau) r_ijkl^{tau(abcd)}
def q_metric():
    '''[(tau, sgn(tau) 2^cycles(tau))] for the 24 permutations of 4 slots.'''
    from itertools import permutations
    out = []
    for tau in permutations(range(4)):
        seen, cyc = set(), 0
        for x in range(4):
            if x not in seen:
                cyc += 1
                while x not in seen:
                    seen.add(x)
                    x = tau[x]
        out.append((tau, (-1) ** (4 - cyc) * 2 ** cyc))
    return out


if __name__ == '__main__':
    for name, terms in spatial_intermediates().items():
        print('== %s (%d terms)\n%s' % (name, len(terms), show(terms)))
    for name, terms in spatial_residuals().items():
        print('== %s (%d terms)\n%s' % (name, len(terms), show(terms)))
