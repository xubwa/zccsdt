'''The Lambda equations of spin-orbital CCSDT, its densities and its orbital
gradient, term by term, with a dense evaluation of them.

T1-dressed formalism throughout, as the amplitude equations
(zccsdt.cc.ccsdtq_terms): f and v below are the dressed Fock matrix and the
dressed antisymmetrized integrals of zccsdt.cc.ccsdt._t1_dress, T = T2 + T3,
and no term has t1 or a bare integral in it.  The left-hand state is

    <0| (1 + Lambda),   Lambda = Lambda_1 + Lambda_2 + Lambda_3,
    Lambda_k = (1/k!)^2 sum l_k[i.., a..] i+ j+ .. b a,

so that <0| Lambda |Phi_ij^ab> = l2[i,j,a,b] (the lambda1[i,a], lambda2
[i,j,a,b] of pyscf.cc.gccsd_lambda and of zccsdt.zccsd.lam; l_k is conj(t_k)
in first order).  The Lagrangian is

    L = <0| (1 + Lambda) exp(-T) H exp(T) |0>
      = E0 + sum f[p,q] g1[p,q] + 1/4 sum <pq||rs> g2[p,q,r,s]         (*)

with E0 = <0| exp(-T1) H exp(T1) |0>.  Nothing is conjugated: every
expression is that of the operators, for real and for complex quantities.

LAMBDA[k]: the residual of the Lambda equation of rank k,

    dL/dt_k[i.., a..] = <0| (1 + Lambda) [exp(-T) H exp(T), a+ b+ .. j i] |0>,

zero for the solution; for converged t it is <0| (1 + Lambda) (Hbar - E)
|Phi_k>.  A term is ``coefficient * A(representative product)`` as in
ccsdtq_terms (rank 1: the product), external letters i j k and a b c.
The derivative with respect to t1 goes through the dressed integrals,
d/dt1[i,a] h~ = [h~, a+_a a_i]; its terms are in the dressed f and v like
the others.

DENSITY[block]: the coefficient of a block of dressed integrals in L - E0,
by the external letters - 'ai': that of f[a,i]; 'aibc': that of <ai||bc> as
the amplitude equations write it.  A term is ``coefficient * product`` (no
antisymmetrizer).  density() puts them into g1[p,q] and the antisymmetric
g2[p,q,r,s] of (*).  The densities of the bare integrals follow by the
transformation that dresses the integrals (zccsdt.cc.lambda_ref.densities).

GRADIENT: the orbital gradient in the dressed basis,

    G[a,i] = <0| (1 + Lambda) exp(-T) [H~, a+_i a_a] exp(T) |0>,
    H~ = exp(-T1) H exp(T1),   T = T2 + T3,

L with the dressed integrals one-index transformed by the de-excitation
a+_i a_a, as ``coefficient * product``, external letters a i.  For converged
t and lambda it is the derivative of the Lagrangian with respect to the
rotation of the bare orbitals a+_i a_a, Gnh[a,i] of zccsdt.zccsd.zvector
(lambda_ref.orbital_gradient has the relation).  LAMBDA[1] is the same
thing with the excitation a+_a a_i.

Where the terms come from (zccsdt.cc.lambda_derive): the Lambda equations
from Wick's theorem (zccsdt.cc.wick.derive_left: 37 + 35 + 16 terms), and a
second time from the table of the amplitude equations, by differentiating
the Lagrangian; the densities and the gradient from the Lagrangian.
tests/test_lambda_terms.py holds the tables against both derivations and
against second-quantized operators in a small Fock space for complex
amplitudes (tests/fockspace.py).

Summed letters: occupied m n o and, where more are needed, j k l; virtual
e f g h and b c d.
'''

from fractions import Fraction

import numpy as np

from zccsdt.cc import ccsdtq_terms as Q

OCC = set('ijklmno')


def _t(coef, *factors):
    return Fraction(coef), tuple((f.split('_')[0], f.split('_')[1]) for f in factors)


LAMBDA = {
    1: [
        _t(1, 'l1_ie', 'f_ea'),
        _t('-1/2', 'l1_ie', 'v_mnaf', 't2_mnef'),
        _t(-1, 'l1_ma', 'f_im'),
        _t('-1/2', 'l1_ma', 'v_inef', 't2_mnef'),
        _t(-1, 'l1_me', 'v_eiam'),
        _t(1, 'l1_me', 'v_inaf', 't2_mnef'),
        _t('1/2', 'l2_imef', 'f_na', 't2_mnef'),
        _t('1/2', 'l2_imef', 'v_efam'),
        _t(1, 'l2_imef', 'v_enag', 't2_mnfg'),
        _t('1/4', 'l2_imef', 'v_noag', 't3_mnoefg'),
        _t('1/4', 'l2_imef', 'v_noam', 't2_noef'),
        _t('1/2', 'l2_mnae', 'f_if', 't2_mnef'),
        _t('1/4', 'l2_mnae', 'v_eifg', 't2_mnfg'),
        _t('1/2', 'l2_mnae', 'v_eimn'),
        _t('1/4', 'l2_mnae', 'v_iofg', 't3_mnoefg'),
        _t(1, 'l2_mnae', 'v_iofm', 't2_noef'),
        _t('1/2', 'l2_mnef', 'v_eiag', 't2_mnfg'),
        _t('1/4', 'l2_mnef', 'v_ioag', 't3_mnoefg'),
        _t('1/2', 'l2_mnef', 'v_ioam', 't2_noef'),
        _t('-1/12', 'l3_imnefg', 'f_oa', 't3_mnoefg'),
        _t('-1/4', 'l3_imnefg', 'v_efah', 't2_mngh'),
        _t('1/4', 'l3_imnefg', 'v_eoah', 't3_mnofgh'),
        _t('1/2', 'l3_imnefg', 'v_eoam', 't2_nofg'),
        _t('1/2', 'l3_imnefg', 'v_joah', 't2_jnfg', 't2_moeh'),
        _t('-1/8', 'l3_imnefg', 'v_joah', 't2_joef', 't2_mngh'),
        _t('-1/12', 'l3_imnefg', 'v_joam', 't3_jnoefg'),
        _t('-1/12', 'l3_mnoaef', 'f_ig', 't3_mnoefg'),
        _t('1/12', 'l3_mnoaef', 'v_eigh', 't3_mnofgh'),
        _t('1/2', 'l3_mnoaef', 'v_eigm', 't2_nofg'),
        _t('1/8', 'l3_mnoaef', 'v_ijgh', 't2_jmef', 't2_nogh'),
        _t('-1/2', 'l3_mnoaef', 'v_ijgh', 't2_jmeg', 't2_nofh'),
        _t('1/4', 'l3_mnoaef', 'v_ijgm', 't3_jnoefg'),
        _t('1/4', 'l3_mnoaef', 'v_ijmn', 't2_joef'),
        _t('-1/12', 'l3_mnoefg', 'v_eiah', 't3_mnofgh'),
        _t('-1/4', 'l3_mnoefg', 'v_ijah', 't2_jmef', 't2_nogh'),
        _t('-1/12', 'l3_mnoefg', 'v_ijam', 't3_jnoefg'),
        _t(1, 'f_ia'),
    ],
    2: [
        _t(1, 'l1_ia', 'f_jb'),
        _t('1/2', 'l1_ie', 'v_ejab'),
        _t('1/2', 'l1_ma', 'v_ijbm'),
        _t('1/2', 'l2_ijae', 'f_eb'),
        _t('-1/4', 'l2_ijae', 'v_mnbf', 't2_mnef'),
        _t('1/8', 'l2_ijef', 'v_efab'),
        _t('1/16', 'l2_ijef', 'v_mnab', 't2_mnef'),
        _t('-1/2', 'l2_imab', 'f_jm'),
        _t('-1/4', 'l2_imab', 'v_jnef', 't2_mnef'),
        _t(-1, 'l2_imae', 'v_ejbm'),
        _t(1, 'l2_imae', 'v_jnbf', 't2_mnef'),
        _t('-1/4', 'l2_imef', 'v_jnab', 't2_mnef'),
        _t('1/16', 'l2_mnab', 'v_ijef', 't2_mnef'),
        _t('1/8', 'l2_mnab', 'v_ijmn'),
        _t('-1/4', 'l2_mnae', 'v_ijbf', 't2_mnef'),
        _t('1/4', 'l3_ijmaef', 'f_nb', 't2_mnef'),
        _t('1/4', 'l3_ijmaef', 'v_efbm'),
        _t('1/2', 'l3_ijmaef', 'v_enbg', 't2_mnfg'),
        _t('1/8', 'l3_ijmaef', 'v_nobg', 't3_mnoefg'),
        _t('1/8', 'l3_ijmaef', 'v_nobm', 't2_noef'),
        _t('1/8', 'l3_ijmefg', 'v_enab', 't2_mnfg'),
        _t('1/48', 'l3_ijmefg', 'v_noab', 't3_mnoefg'),
        _t('1/4', 'l3_imnabe', 'f_jf', 't2_mnef'),
        _t('1/8', 'l3_imnabe', 'v_ejfg', 't2_mnfg'),
        _t('1/4', 'l3_imnabe', 'v_ejmn'),
        _t('1/8', 'l3_imnabe', 'v_jofg', 't3_mnoefg'),
        _t('1/2', 'l3_imnabe', 'v_jofm', 't2_noef'),
        _t('1/2', 'l3_imnaef', 'v_ejbg', 't2_mnfg'),
        _t('1/4', 'l3_imnaef', 'v_jobg', 't3_mnoefg'),
        _t('1/2', 'l3_imnaef', 'v_jobm', 't2_noef'),
        _t('1/24', 'l3_imnefg', 'v_joab', 't3_mnoefg'),
        _t('1/48', 'l3_mnoabe', 'v_ijfg', 't3_mnoefg'),
        _t('1/8', 'l3_mnoabe', 'v_ijfm', 't2_noef'),
        _t('1/24', 'l3_mnoaef', 'v_ijbg', 't3_mnoefg'),
        _t('1/4', 'v_ijab'),
    ],
    3: [
        _t('1/4', 'l1_ia', 'v_jkbc'),
        _t('1/4', 'l2_ijab', 'f_kc'),
        _t('1/4', 'l2_ijae', 'v_ekbc'),
        _t('1/4', 'l2_imab', 'v_jkcm'),
        _t('1/12', 'l3_ijkabe', 'f_ec'),
        _t('-1/24', 'l3_ijkabe', 'v_mncf', 't2_mnef'),
        _t('1/24', 'l3_ijkaef', 'v_efbc'),
        _t('1/48', 'l3_ijkaef', 'v_mnbc', 't2_mnef'),
        _t('-1/12', 'l3_ijmabc', 'f_km'),
        _t('-1/24', 'l3_ijmabc', 'v_knef', 't2_mnef'),
        _t('-1/4', 'l3_ijmabe', 'v_ekcm'),
        _t('1/4', 'l3_ijmabe', 'v_kncf', 't2_mnef'),
        _t('-1/8', 'l3_ijmaef', 'v_knbc', 't2_mnef'),
        _t('1/48', 'l3_imnabc', 'v_jkef', 't2_mnef'),
        _t('1/24', 'l3_imnabc', 'v_jkmn'),
        _t('-1/8', 'l3_imnabe', 'v_jkcf', 't2_mnef'),
    ],
}

DENSITY = {
    'ab': [
        _t('1/2', 'l2_mnae', 't2_mnbe'),
        _t('1/12', 'l3_mnoaef', 't3_mnobef'),
    ],
    'ai': [
        _t(1, 'l1_ia'),
    ],
    'ia': [
        _t(1, 'l1_me', 't2_imae'),
        _t('1/4', 'l2_mnef', 't3_imnaef'),
        _t('1/4', 'l3_mnoefg', 't2_imef', 't2_noag'),
    ],
    'ij': [
        _t('-1/2', 'l2_jmef', 't2_imef'),
        _t('-1/12', 'l3_jmnefg', 't3_imnefg'),
    ],
    'abcd': [
        _t('1/8', 'l2_mnab', 't2_mncd'),
        _t('1/24', 'l3_mnoabe', 't3_mnocde'),
    ],
    'abci': [
        _t('-1/4', 'l3_imnabe', 't2_mnce'),
    ],
    'abij': [
        _t('1/4', 'l2_ijab'),
    ],
    'aibc': [
        _t('-1/2', 'l1_ma', 't2_imbc'),
        _t('-1/4', 'l2_mnae', 't3_imnbce'),
        _t('1/2', 'l3_mnoaef', 't2_imbe', 't2_nocf'),
        _t('-1/8', 'l3_mnoaef', 't2_imef', 't2_nobc'),
    ],
    'iabj': [
        _t(1, 'l2_jmae', 't2_imbe'),
        _t('1/4', 'l3_jmnaef', 't3_imnbef'),
    ],
    'iajk': [
        _t('1/4', 'l3_jkmaef', 't2_imef'),
    ],
    'ijab': [
        _t('1/4', 'l1_me', 't3_ijmabe'),
        _t('1/4', 'l2_mnef', 't2_ijbe', 't2_mnaf'),
        _t('1/16', 'l2_mnef', 't2_ijef', 't2_mnab'),
        _t('1/2', 'l2_mnef', 't2_imae', 't2_jnbf'),
        _t('-1/4', 'l2_mnef', 't2_imef', 't2_jnab'),
        _t('-1/24', 'l3_mnoefg', 't2_ijae', 't3_mnobfg'),
        _t('1/48', 'l3_mnoefg', 't2_ijef', 't3_mnoabg'),
        _t('-1/24', 'l3_mnoefg', 't2_imab', 't3_jnoefg'),
        _t('1/4', 'l3_mnoefg', 't2_imae', 't3_jnobfg'),
        _t('-1/8', 'l3_mnoefg', 't2_imef', 't3_jnoabg'),
        _t('1/48', 'l3_mnoefg', 't2_mnab', 't3_ijoefg'),
        _t('-1/8', 'l3_mnoefg', 't2_mnae', 't3_ijobfg'),
        _t('1/4', 't2_ijab'),
    ],
    'ijak': [
        _t('-1/2', 'l1_ke', 't2_ijae'),
        _t('-1/4', 'l2_kmef', 't3_ijmaef'),
        _t('-1/8', 'l3_kmnefg', 't2_ijef', 't2_mnag'),
        _t('1/2', 'l3_kmnefg', 't2_imae', 't2_jnfg'),
    ],
    'ijkl': [
        _t('1/8', 'l2_klef', 't2_ijef'),
        _t('1/24', 'l3_klmefg', 't3_ijmefg'),
    ],
}

GRADIENT = [
    _t(1, 'l1_me', 'f_af', 't2_imef'),
    _t(-1, 'l1_me', 'f_ni', 't2_mnae'),
    _t('-1/2', 'l1_me', 'v_aefg', 't2_imfg'),
    _t(-1, 'l1_me', 'v_aeim'),
    _t('1/2', 'l1_me', 'v_anfg', 't3_imnefg'),
    _t(1, 'l1_me', 'v_anfi', 't2_mnef'),
    _t(-1, 'l1_me', 'v_anfm', 't2_inef'),
    _t(-1, 'l1_me', 'v_enfi', 't2_mnaf'),
    _t('1/2', 'l1_me', 'v_nofi', 't3_mnoaef'),
    _t('-1/2', 'l1_me', 'v_noim', 't2_noae'),
    _t('-1/4', 'l2_mnef', 'f_ag', 't3_imnefg'),
    _t('1/2', 'l2_mnef', 'f_am', 't2_inef'),
    _t('1/2', 'l2_mnef', 'f_ei', 't2_mnaf'),
    _t('1/4', 'l2_mnef', 'f_oi', 't3_mnoaef'),
    _t('-1/4', 'l2_mnef', 'v_aegh', 't3_imnfgh'),
    _t('-1/2', 'l2_mnef', 'v_aegi', 't2_mnfg'),
    _t(1, 'l2_mnef', 'v_aegm', 't2_infg'),
    _t('-1/4', 'l2_mnef', 'v_aogh', 't2_imef', 't2_nogh'),
    _t(1, 'l2_mnef', 'v_aogh', 't2_imeg', 't2_nofh'),
    _t('-1/4', 'l2_mnef', 'v_aogh', 't2_imgh', 't2_noef'),
    _t('-1/8', 'l2_mnef', 'v_aogh', 't2_ioef', 't2_mngh'),
    _t('1/2', 'l2_mnef', 'v_aogh', 't2_ioeg', 't2_mnfh'),
    _t('1/4', 'l2_mnef', 'v_aogi', 't3_mnoefg'),
    _t('-1/2', 'l2_mnef', 'v_aogm', 't3_inoefg'),
    _t('-1/2', 'l2_mnef', 'v_aoim', 't2_noef'),
    _t('-1/4', 'l2_mnef', 'v_aomn', 't2_ioef'),
    _t('-1/4', 'l2_mnef', 'v_efgi', 't2_mnag'),
    _t('-1/2', 'l2_mnef', 'v_eogi', 't3_mnoafg'),
    _t(1, 'l2_mnef', 'v_eoim', 't2_noaf'),
    _t(-1, 'l2_mnef', 'v_jogi', 't2_jnaf', 't2_moeg'),
    _t('1/2', 'l2_mnef', 'v_jogi', 't2_jnag', 't2_moef'),
    _t('-1/4', 'l2_mnef', 'v_jogi', 't2_joae', 't2_mnfg'),
    _t('-1/8', 'l2_mnef', 'v_jogi', 't2_joef', 't2_mnag'),
    _t('1/4', 'l2_mnef', 'v_jogi', 't2_joeg', 't2_mnaf'),
    _t('1/4', 'l2_mnef', 'v_joim', 't3_jnoaef'),
    _t('1/4', 'l3_mnoefg', 'f_ah', 't2_imef', 't2_nogh'),
    _t('1/12', 'l3_mnoefg', 'f_am', 't3_inoefg'),
    _t('1/12', 'l3_mnoefg', 'f_ei', 't3_mnoafg'),
    _t('1/4', 'l3_mnoefg', 'f_ji', 't2_jmef', 't2_noag'),
    _t('-1/8', 'l3_mnoefg', 'v_aebh', 't2_imfg', 't2_nobh'),
    _t('1/2', 'l3_mnoefg', 'v_aebh', 't2_imfh', 't2_nobg'),
    _t('1/12', 'l3_mnoefg', 'v_aehi', 't3_mnofgh'),
    _t('-1/4', 'l3_mnoefg', 'v_aehm', 't3_inofgh'),
    _t('-1/4', 'l3_mnoefg', 'v_aemn', 't2_iofg'),
    _t('1/24', 'l3_mnoefg', 'v_ajbh', 't2_ijef', 't3_mnobgh'),
    _t('1/12', 'l3_mnoefg', 'v_ajbh', 't2_ijeh', 't3_mnobfg'),
    _t('1/24', 'l3_mnoefg', 'v_ajbh', 't2_imbh', 't3_jnoefg'),
    _t('-1/8', 'l3_mnoefg', 'v_ajbh', 't2_imef', 't3_jnobgh'),
    _t('-1/4', 'l3_mnoefg', 'v_ajbh', 't2_imeh', 't3_jnobfg'),
    _t('-1/24', 'l3_mnoefg', 'v_ajbh', 't2_jmbh', 't3_inoefg'),
    _t('1/8', 'l3_mnoefg', 'v_ajbh', 't2_jmef', 't3_inobgh'),
    _t('1/4', 'l3_mnoefg', 'v_ajbh', 't2_jmeh', 't3_inobfg'),
    _t('-1/24', 'l3_mnoefg', 'v_ajbh', 't2_mnbh', 't3_ijoefg'),
    _t('1/4', 'l3_mnoefg', 'v_ajbh', 't2_mneh', 't3_ijobfg'),
    _t('-1/4', 'l3_mnoefg', 'v_ajhi', 't2_jmef', 't2_nogh'),
    _t('-1/4', 'l3_mnoefg', 'v_ajhm', 't2_ijef', 't2_nogh'),
    _t('1/2', 'l3_mnoefg', 'v_ajhm', 't2_inef', 't2_jogh'),
    _t('1/2', 'l3_mnoefg', 'v_ajhm', 't2_ineh', 't2_jofg'),
    _t('1/12', 'l3_mnoefg', 'v_ajim', 't3_jnoefg'),
    _t('-1/12', 'l3_mnoefg', 'v_ajmn', 't3_ijoefg'),
    _t('1/12', 'l3_mnoefg', 'v_efhi', 't3_mnoagh'),
    _t('-1/4', 'l3_mnoefg', 'v_efim', 't2_noag'),
    _t('1/2', 'l3_mnoefg', 'v_ejhi', 't2_jmaf', 't2_nogh'),
    _t('1/4', 'l3_mnoefg', 'v_ejhi', 't2_jmfg', 't2_noah'),
    _t('-1/2', 'l3_mnoefg', 'v_ejhi', 't2_jmfh', 't2_noag'),
    _t('-1/4', 'l3_mnoefg', 'v_ejim', 't3_jnoafg'),
    _t('1/24', 'l3_mnoefg', 'v_jkhi', 't2_jkae', 't3_mnofgh'),
    _t('1/24', 'l3_mnoefg', 'v_jkhi', 't2_jkef', 't3_mnoagh'),
    _t('1/24', 'l3_mnoefg', 'v_jkhi', 't2_jkeh', 't3_mnoafg'),
    _t('-1/4', 'l3_mnoefg', 'v_jkhi', 't2_jmae', 't3_knofgh'),
    _t('1/12', 'l3_mnoefg', 'v_jkhi', 't2_jmah', 't3_knoefg'),
    _t('-1/4', 'l3_mnoefg', 'v_jkhi', 't2_jmef', 't3_knoagh'),
    _t('-1/4', 'l3_mnoefg', 'v_jkhi', 't2_jmeh', 't3_knoafg'),
    _t('1/8', 'l3_mnoefg', 'v_jkhi', 't2_mnae', 't3_jkofgh'),
    _t('-1/24', 'l3_mnoefg', 'v_jkhi', 't2_mnah', 't3_jkoefg'),
    _t('1/8', 'l3_mnoefg', 'v_jkhi', 't2_mneh', 't3_jkoafg'),
    _t('-1/8', 'l3_mnoefg', 'v_jkim', 't2_jkef', 't2_noag'),
    _t('1/2', 'l3_mnoefg', 'v_jkim', 't2_jnae', 't2_kofg'),
    _t(-1, 'f_ai'),
    _t('-1/2', 'v_amef', 't2_imef'),
    _t('-1/2', 'v_mnei', 't2_mnae'),
]


# ------------------------------------------------------------ dense evaluation

_PATHS = {}


def product(factors, arrays, nocc, out):
    '''One product as a dense array over the letters ``out``.  arrays:
    {'f': f[p,q], 'v': <pq||rs>} over all spin orbitals (dressed) and
    {'t2', 't3', 'l1', 'l2', 'l3'} as [i.., a..].'''
    n = arrays['v'].shape[0]
    ops = []
    for kind, ix in factors:
        a = arrays[kind]
        if kind in ('v', 'f'):
            a = a[tuple(slice(0, nocc) if c in OCC else slice(nocc, n) for c in ix)]
        ops.append(a)
    sub = ','.join(ix for _, ix in factors) + '->' + out
    key = (sub,) + tuple(a.shape for a in ops)
    if key not in _PATHS:
        _PATHS[key] = np.einsum_path(sub, *ops, optimize='optimal')[0]
    return np.einsum(sub, *ops, optimize=_PATHS[key])


def total(terms, arrays, nocc, out):
    '''sum of coefficient * product.'''
    x = 0
    for coef, factors in terms:
        x = x + float(coef) * product(factors, arrays, nocc, out)
    return x


def residual(rank, arrays, nocc):
    '''The residual of the Lambda equation of ``rank`` as a dense
    antisymmetric array [i.., a..], from the dressed f and v.'''
    x = total(LAMBDA[rank], arrays, nocc, Q.EXT[rank])
    return x if rank == 1 else Q.asym(x, rank)


def density(arrays, nocc):
    '''(g1[p,q], g2[p,q,r,s]) of (*) over all spin orbitals: the
    coefficients of the dressed f and of the dressed <pq||rs> in L - E0, g2
    antisymmetric in (p,q) and in (r,s).  ``arrays``: the amplitudes ('v'
    is asked for its size only).'''
    n = arrays['v'].shape[0]
    dtype = np.result_type(*[arrays[k] for k in ('t2', 't3', 'l1', 'l2', 'l3')])
    g1 = np.zeros((n, n), dtype=dtype)
    g2 = np.zeros((n, n, n, n), dtype=dtype)
    for block, terms in DENSITY.items():
        x = total(terms, arrays, nocc, block)
        sl = tuple(slice(0, nocc) if c in OCC else slice(nocc, n) for c in block)
        if len(block) == 2:
            g1[sl] += x
        else:
            g2[sl] += x
    g2 = g2 - g2.transpose(1, 0, 2, 3)
    return g1, g2 - g2.transpose(0, 1, 3, 2)


def gradient(arrays, nocc):
    '''G[a,i] of the table GRADIENT.'''
    return total(GRADIENT, arrays, nocc, 'ai')
