'''The spin-orbital CCSDTQ equations and the quantities of the perturbative
(Q) and (P) corrections, term by term, with a dense evaluation of them.

T1-dressed formalism (as zccsdt.cc.ccsdt): T1 is absorbed in the Fock matrix
and in the antisymmetrized integrals (ccsdt._t1_dress), so the cluster
operator of the equations is T = T2 + T3 + T4 and

    R_k = <Phi_k| F + V + [F + V, T] + 1/2 [[F + V, T], T]
                 + 1/6 [[[V, T], T], T] |0>_connected,   k = 1 .. 4,

with F and V the dressed, normal-ordered one- and two-body operators.

A term is ``coefficient * A(representative product)``: A is the full
antisymmetrizer over the external occupied indices and over the external
virtual indices (all signed permutations, no 1/k!), and letters that occur
twice are summed.  External letters of rank k are the first k of ijklm and of
abcde; the other occupied (m n o) and virtual (e f g) letters are summed.
Factor kinds: 'f' = f[p, q], 'v' = <pq||rs>, 't2' / 't3' / 't4' = amplitudes
t[i.., a..].  No factor is conjugated: the expressions are those of the
commutators, for real and for complex quantities alike.

Where the terms come from - two ways that do not share anything:

- Wick's theorem (zccsdt.cc.wick): for every group all contractions of the
  quasi-particle annihilators of H with the cluster operators, 4061 in all,
  collected by topology.  They give the 5 + 14 + 22 + 34 terms of R1 to R4
  and the 8 of the pentuples below, with these coefficients.
- Second-quantized operators in a Fock space of 6 + 6 spin orbitals
  (tests/fockspace.py): for random amplitudes and integrals every group
  is the commutator it is named after to 1e-10 or better of its size, and
  its representatives are linearly independent there.  (The coefficients
  were first found that way, by a fit; in a space of 5 + 5 spin orbitals
  [[V,T2],T4] and 1/2 [[V,T3],T3] are rank deficient.)

tests/test_ccsdtq_terms.py has both, and the sum of all terms against the
projections of exp(-T) H exp(T) for complex amplitudes including T1.

GROUPS: name -> (rank, 'f' or 'v', kinds of the cluster operators, symmetry
factor of the commutator, [(coefficient, factors)]).
'''

from fractions import Fraction
from itertools import combinations, permutations

import numpy as np

OCC = set('ijklmno')
EXT = {1: 'ia', 2: 'ijab', 3: 'ijkabc', 4: 'ijklabcd', 5: 'ijklmabcde'}


def _t(coef, *factors):
    return Fraction(coef), tuple((f.split('_')[0], f.split('_')[1]) for f in factors)


GROUPS = {
    'R1: F': (1, 'f', (), 1, [_t(1, 'f_ai')]),
    'R1: [F,T2]': (1, 'f', ('t2',), 1, [_t(1, 'f_me', 't2_imae')]),
    'R1: [V,T2]': (1, 'v', ('t2',), 1, [_t('1/2', 'v_amef', 't2_imef'),
                                        _t('1/2', 'v_mnei', 't2_mnae')]),
    'R1: [V,T3]': (1, 'v', ('t3',), 1, [_t('1/4', 'v_mnef', 't3_imnaef')]),

    'R2: V': (2, 'v', (), 1, [_t('1/4', 'v_abij')]),
    'R2: [F,T2]': (2, 'f', ('t2',), 1, [_t('1/2', 'f_be', 't2_ijae'),
                                        _t('-1/2', 'f_mj', 't2_imab')]),
    'R2: [V,T2]': (2, 'v', ('t2',), 1, [_t('1/8', 'v_abef', 't2_ijef'),
                                        _t('1/8', 'v_mnij', 't2_mnab'),
                                        _t(1, 'v_mbej', 't2_imae')]),
    'R2: 1/2 [[V,T2],T2]': (2, 'v', ('t2', 't2'), 2, [
        _t('-1/4', 'v_mnef', 't2_imab', 't2_jnef'),
        _t('-1/4', 'v_mnef', 't2_ijae', 't2_mnbf'),
        _t('1/16', 'v_mnef', 't2_mnab', 't2_ijef'),
        _t('1/2', 'v_mnef', 't2_imae', 't2_jnbf')]),
    'R2: [F,T3]': (2, 'f', ('t3',), 1, [_t('1/4', 'f_me', 't3_ijmabe')]),
    'R2: [V,T3]': (2, 'v', ('t3',), 1, [_t('1/4', 'v_bmef', 't3_ijmaef'),
                                        _t('1/4', 'v_mnej', 't3_imnabe')]),
    'R2: [V,T4]': (2, 'v', ('t4',), 1, [_t('1/16', 'v_mnef', 't4_ijmnabef')]),

    'R3: [V,T2]': (3, 'v', ('t2',), 1, [_t('1/4', 'v_bcek', 't2_ijae'),
                                        _t('-1/4', 'v_mcjk', 't2_imab')]),
    'R3: [F,T3]': (3, 'f', ('t3',), 1, [_t('1/12', 'f_ce', 't3_ijkabe'),
                                        _t('-1/12', 'f_mk', 't3_ijmabc')]),
    'R3: [V,T3]': (3, 'v', ('t3',), 1, [_t('1/24', 'v_bcef', 't3_ijkaef'),
                                        _t('1/24', 'v_mnjk', 't3_imnabc'),
                                        _t('1/4', 'v_mcek', 't3_ijmabe')]),
    'R3: 1/2 [[F,T2],T2]': (3, 'f', ('t2', 't2'), 2, [
        _t('-1/4', 'f_me', 't2_imab', 't2_jkec')]),
    'R3: 1/2 [[V,T2],T2]': (3, 'v', ('t2', 't2'), 2, [
        _t('-1/8', 'v_mnek', 't2_mnab', 't2_ijec'),
        _t('-1/2', 'v_mnek', 't2_imae', 't2_jnbc'),
        _t('-1/8', 'v_cmef', 't2_ijef', 't2_mkab'),
        _t('1/2', 'v_cmef', 't2_imae', 't2_jkfb')]),
    'R3: [[V,T2],T3]': (3, 'v', ('t2', 't3'), 1, [
        _t('1/8', 'v_mnef', 't2_imab', 't3_njkefc'),
        _t('1/8', 'v_mnef', 't2_ijae', 't3_mnkfbc'),
        _t('1/48', 'v_mnef', 't2_mnab', 't3_ijkefc'),
        _t('1/48', 'v_mnef', 't2_ijef', 't3_mnkabc'),
        _t('1/4', 'v_mnef', 't2_imae', 't3_njkfbc'),
        _t('1/24', 'v_mnef', 't2_mnae', 't3_ijkfbc'),
        _t('1/24', 'v_mnef', 't2_imef', 't3_njkabc')]),
    'R3: [F,T4]': (3, 'f', ('t4',), 1, [_t('1/36', 'f_me', 't4_ijkmabce')]),
    'R3: [V,T4]': (3, 'v', ('t4',), 1, [_t('1/24', 'v_mnek', 't4_ijmnabce'),
                                        _t('1/24', 'v_cmef', 't4_ijkmabef')]),

    'R4: [V,T3]': (4, 'v', ('t3',), 1, [_t('1/24', 'v_cdel', 't3_ijkabe'),
                                        _t('-1/24', 'v_mdkl', 't3_ijmabc')]),
    'R4: [F,T4]': (4, 'f', ('t4',), 1, [_t('1/144', 'f_de', 't4_ijklabce'),
                                        _t('-1/144', 'f_ml', 't4_ijkmabcd')]),
    'R4: [V,T4]': (4, 'v', ('t4',), 1, [_t('1/192', 'v_cdef', 't4_ijklabef'),
                                        _t('1/192', 'v_mnkl', 't4_ijmnabcd'),
                                        _t('1/36', 'v_mdel', 't4_ijkmabce')]),
    'R4: 1/2 [[V,T2],T2]': (4, 'v', ('t2', 't2'), 2, [
        _t('1/16', 'v_bcef', 't2_ijae', 't2_klfd'),
        _t('1/16', 'v_mnjk', 't2_imab', 't2_nlcd'),
        _t('1/4', 'v_mbek', 't2_ijae', 't2_mlcd')]),
    'R4: [[F,T2],T3]': (4, 'f', ('t2', 't3'), 1, [
        _t('-1/24', 'f_me', 't2_imab', 't3_jklecd'),
        _t('-1/24', 'f_me', 't2_ijae', 't3_mklbcd')]),
    'R4: [[V,T2],T3]': (4, 'v', ('t2', 't3'), 1, [
        _t('1/8', 'v_mnel', 't2_imab', 't3_njkecd'),
        _t('-1/24', 'v_mnel', 't2_ijae', 't3_mnkbcd'),
        _t('1/48', 'v_mnel', 't2_mnab', 't3_ijkecd'),
        _t('-1/12', 'v_mnel', 't2_imae', 't3_njkbcd'),
        _t('-1/24', 'v_dmef', 't2_imab', 't3_jklefc'),
        _t('1/8', 'v_dmef', 't2_ijae', 't3_mklfbc'),
        _t('1/48', 'v_dmef', 't2_ijef', 't3_mklabc'),
        _t('-1/12', 'v_dmef', 't2_imae', 't3_jklfbc')]),
    'R4: [[V,T2],T4]': (4, 'v', ('t2', 't4'), 1, [
        _t('1/48', 'v_mnef', 't2_imab', 't4_njklefcd'),
        _t('1/48', 'v_mnef', 't2_ijae', 't4_mnklfbcd'),
        _t('1/384', 'v_mnef', 't2_mnab', 't4_ijklefcd'),
        _t('1/384', 'v_mnef', 't2_ijef', 't4_mnklabcd'),
        _t('1/36', 'v_mnef', 't2_imae', 't4_njklfbcd'),
        _t('1/288', 'v_mnef', 't2_mnae', 't4_ijklfbcd'),
        _t('1/288', 'v_mnef', 't2_imef', 't4_njklabcd')]),
    'R4: 1/2 [[V,T3],T3]': (4, 'v', ('t3', 't3'), 2, [
        _t('1/48', 'v_mnef', 't3_mijabc', 't3_nklefd'),
        _t('1/48', 'v_mnef', 't3_ijkeab', 't3_mnlfcd'),
        _t('1/144', 'v_mnef', 't3_mniabc', 't3_jklefd'),
        _t('1/32', 'v_mnef', 't3_mijeab', 't3_nklfcd')]),
    'R4: 1/6 [[[V,T2],T2],T2]': (4, 'v', ('t2', 't2', 't2'), 6, [
        _t('-1/32', 'v_mnef', 't2_mnab', 't2_ijec', 't2_klfd'),
        _t('-1/32', 'v_mnef', 't2_ijef', 't2_mkab', 't2_nlcd'),
        _t('1/4', 'v_mnef', 't2_imae', 't2_njbc', 't2_klfd')]),

    # pentuples, to the lowest order of perturbation theory: the right-hand
    # side of the T5 equation of the (P) correction
    'R5: [V,T4]': (5, 'v', ('t4',), 1, [_t('1/288', 'v_defm', 't4_ijklabcf'),
                                        _t('-1/288', 'v_nelm', 't4_ijknabcd')]),
    'R5: [[V,T2],T3]': (5, 'v', ('t2', 't3'), 1, [
        _t('1/48', 'v_bcfg', 't2_ijaf', 't3_klmgde'),
        _t('1/48', 'v_nojk', 't2_inab', 't3_olmcde'),
        _t('-1/24', 'v_nefm', 't2_inab', 't3_jklfcd'),
        _t('-1/24', 'v_nefm', 't2_ijaf', 't3_nklbcd')]),
    'R5: 1/6 [[[V,T2],T2],T2]': (5, 'v', ('t2', 't2', 't2'), 6, [
        _t('1/16', 'v_nofm', 't2_inab', 't2_jocd', 't2_klfe'),
        _t('-1/16', 'v_enfg', 't2_inab', 't2_jkfc', 't2_lmgd')]),
}

# <rank k| V T_n |0>, n = k - 2 (the disconnected product) and n = k - 1: with
# the amplitudes conjugated these are the left-hand quantities
# <0| (T_{k-2}^+ + T_{k-1}^+) V |rank k> of the perturbative corrections
LEFT = {
    'Z4.T2': (4, 't2', [_t('1/16', 'v_cdkl', 't2_ijab')]),
    'Z4.T3': (4, 't3', [_t('1/24', 'v_cdel', 't3_ijkabe'), _t('-1/24', 'v_mdkl', 't3_ijmabc')]),
    'Z5.T3': (5, 't3', [_t('1/144', 'v_delm', 't3_ijkabc')]),
    'Z5.T4': (5, 't4', [_t('1/288', 'v_defm', 't4_ijklabcf'), _t('-1/288', 'v_nelm', 't4_ijknabcd')]),
}


def residual_terms(rank, with_t4=True, with_t3=True):
    '''All terms of R_rank (CCSDTQ; without T4: CCSDT; without T3: CCSD).'''
    out = []
    for name, (k, _, kinds, _, terms) in GROUPS.items():
        if (k != rank or not name.startswith('R%d' % rank) or rank == 5
                or ('t4' in kinds and not with_t4) or ('t3' in kinds and not with_t3)):
            continue
        out += terms
    return out


# ------------------------------------------------------------ dense evaluation

def _sign(perm):
    s, p = 1, list(perm)
    for x in range(len(p)):
        while p[x] != x:
            y = p[x]
            p[x], p[y] = p[y], p[x]
            s = -s
    return s


_PATHS = {}


def product(factors, arrays, nocc, rank, occ=None, vir=None):
    '''One representative product as a dense array over the external
    indices of ``rank``.  arrays: {'f': f[p,q], 'v': <pq||rs>, 't2': ...}
    over all spin orbitals (f, v) and [i.., a..] (amplitudes).  ``occ`` /
    ``vir``: the external occupied / virtual indices run over these
    occupied / virtual orbitals only (virtual ones counted from 0).'''
    n = arrays['v'].shape[0]
    only = {c: occ for c in EXT[rank][:rank]} if occ is not None else {}
    if vir is not None:
        only.update({c: vir for c in EXT[rank][rank:]})
    ops = []
    for kind, ix in factors:
        a = arrays[kind]
        if kind in ('v', 'f'):
            a = a[tuple(slice(0, nocc) if c in OCC else slice(nocc, n) for c in ix)]
        for axis, c in enumerate(ix):
            if c in only:
                a = np.take(a, only[c], axis=axis)
        ops.append(a)
    sub = ','.join(ix for _, ix in factors) + '->' + EXT[rank]
    key = (sub,) + tuple(a.shape for a in ops)
    if key not in _PATHS:               # (the search of the path takes longer than a small product)
        _PATHS[key] = np.einsum_path(sub, *ops, optimize='optimal')[0]
    return np.einsum(sub, *ops, optimize=_PATHS[key])


def total(terms, arrays, nocc, rank, occ=None, vir=None):
    '''sum of coefficient * representative product (not antisymmetrized).'''
    out = 0
    for coef, factors in terms:
        out = out + float(coef) * product(factors, arrays, nocc, rank, occ, vir)
    return out


def element(terms, arrays, nocc, rank, occ, vir):
    '''The antisymmetrized sum of terms at one occupied and one virtual
    tuple: sum_{po, pv} sgn x[po(occ), pv(vir)].'''
    x = total(terms, arrays, nocc, rank, occ, vir)
    if rank == 1:
        return x[0, 0]
    perms = [(p, _sign(p)) for p in permutations(range(rank))]
    out = 0
    for p, sp in perms:
        y = x[p]
        for q, sq in perms:
            out = out + sp * sq * y[q]
    return out


def asym(x, k):
    '''The antisymmetrizer over the first k and over the last k axes.'''
    out = np.zeros_like(x)
    for p in permutations(range(k)):
        out += _sign(p) * x.transpose(p + tuple(range(k, 2 * k)))
    x, out = out, np.zeros_like(out)
    for p in permutations(range(k)):
        out += _sign(p) * x.transpose(tuple(range(k)) + tuple(k + y for y in p))
    return out


def tuples(n, k):
    '''The sorted k-tuples of range(n), [C(n,k), k].'''
    return np.array(list(combinations(range(n), k)), dtype=np.int64).reshape(-1, k)


def asym_unique(x, k):
    '''The antisymmetrizer of ``x`` at sorted occupied and sorted virtual
    tuples only: [C(no,k), C(nv,k)].'''
    occ, vir = tuples(x.shape[0], k), tuples(x.shape[-1], k)
    perms = list(permutations(range(k)))
    sg = np.array([_sign(p) for p in perms])
    po, pv = occ[:, perms], vir[:, perms]               # [n, k!, k]
    out = np.zeros((len(occ), len(vir)), dtype=x.dtype)
    for a in range(len(perms)):
        io = tuple(po[:, a, s][:, None, None] for s in range(k))
        iv = tuple(pv[:, :, s][None, :, :] for s in range(k))
        out += sg[a] * (x[io + iv] * sg[None, None, :]).sum(axis=2)
    return out


def unique(x, k):
    '''x[i<j<.., a<b<..] of an antisymmetric array: [C(no,k), C(nv,k)].'''
    occ, vir = tuples(x.shape[0], k), tuples(x.shape[-1], k)
    return x[tuple(occ[:, s][:, None] for s in range(k)) + tuple(vir[:, s][None, :] for s in range(k))]


def expand(u, no, nv, k):
    '''The full antisymmetric array of its unique elements (inverse of
    unique).'''
    occ, vir = tuples(no, k), tuples(nv, k)
    out = np.zeros((no,) * k + (nv,) * k, dtype=u.dtype)
    for p in permutations(range(k)):
        for q in permutations(range(k)):
            idx = (tuple(occ[:, p[s]][:, None] for s in range(k))
                   + tuple(vir[:, q[s]][None, :] for s in range(k)))
            out[idx] = _sign(p) * _sign(q) * u
    return out


def residual(rank, arrays, nocc, with_t4=True, with_t3=True):
    '''R_rank as a dense antisymmetric array, from dressed f and v.'''
    x = total(residual_terms(rank, with_t4, with_t3), arrays, nocc, rank)
    return x if rank == 1 else asym(x, rank)
