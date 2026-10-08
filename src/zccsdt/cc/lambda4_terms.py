'''The Lambda equations of spin-orbital CCSDTQ, its densities and its orbital
gradient, term by term, with a dense evaluation of them: the tables of
zccsdt.cc.lambda_terms (CCSDT) with T4 and Lambda_4.

T1-dressed formalism throughout, as the amplitude equations
(zccsdt.cc.ccsdtq_terms): f and v below are the dressed Fock matrix and the
dressed antisymmetrized integrals of zccsdt.cc.ccsdt._t1_dress,
T = T2 + T3 + T4, and no term has t1 or a bare integral in it.  The
left-hand state is

    <0| (1 + Lambda),   Lambda = Lambda_1 + Lambda_2 + Lambda_3 + Lambda_4,
    Lambda_k = (1/k!)^2 sum l_k[i.., a..] i+ j+ .. b a,

and the Lagrangian

    L = <0| (1 + Lambda) exp(-T) H exp(T) |0>
      = E0 + sum f[p,q] g1[p,q] + 1/4 sum <pq||rs> g2[p,q,r,s]         (*)

with E0 = <0| exp(-T1) H exp(T1) |0>.  Nothing is conjugated.  The tables
are those of lambda_terms in every convention:

LAMBDA[k], k = 1 .. 4: the residual of the Lambda equation of rank k,
    dL/dt_k[i.., a..] = <0| (1 + Lambda) [exp(-T) H exp(T), a+ b+ .. j i] |0>,
    a term being ``coefficient * A(representative product)`` (rank 1: the
    product), external letters i j k l and a b c d;
DENSITY[block]: the coefficient of a block of dressed integrals in L - E0,
    ``coefficient * product``, by the letters of the integral;
GRADIENT: the orbital gradient in the dressed basis,
    G[a,i] = <0| (1 + Lambda) exp(-T) [a+_i a_a, H~] exp(T) |0>,
    ``coefficient * product``, external letters a i.

Numbers of terms (of them with t4 or l4):

    LAMBDA[1]   66  (37 of CCSDT + 3 of l3 with t4 + 26 of l4)
    LAMBDA[2]   63  (35 + 28 of l4)
    LAMBDA[3]   35  (16 + 19 of l4)
    LAMBDA[4]   16  (1 of l2, 3 of l3, 12 of l4)
    DENSITY     76 in 13 blocks  (38 + 38: one for every term of L)
    GRADIENT   168 in 11 blocks of integrals  (81 + 87)

Without t4 and l4 every table is that of CCSDT, term by term
(ccsdt_limit(); tests/test_lambda4_terms.py).

Where the terms come from (zccsdt.cc.lambda_derive with nmax = 4,
``python -m zccsdt.cc.lambda_derive 4`` prints the tables below): the
Lambda equations from Wick's theorem (zccsdt.cc.wick.derive_left), and a
second time from the table of the amplitude equations of CCSDTQ, by
differentiating the Lagrangian; the densities and the gradient from the
Lagrangian.  tests/test_lambda4_terms.py holds the tables against both
derivations and against second-quantized operators in a Fock space of
6 + 6 spin orbitals for complex amplitudes (tests/fockspace.py).

Summed letters: occupied m n o and, where more are needed, j k l; virtual
e f g h and b c d.
'''

from fractions import Fraction

import numpy as np

from zccsdt.cc import ccsdtq_terms as Q, lambda_terms as L3
from zccsdt.cc.lambda_terms import OCC, product, total  # noqa: F401


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
        _t('-1/24', 'l3_imnefg', 'v_joah', 't4_jmnoefgh'),
        _t('-1/12', 'l3_imnefg', 'v_joam', 't3_jnoefg'),
        _t('-1/12', 'l3_mnoaef', 'f_ig', 't3_mnoefg'),
        _t('1/12', 'l3_mnoaef', 'v_eigh', 't3_mnofgh'),
        _t('1/2', 'l3_mnoaef', 'v_eigm', 't2_nofg'),
        _t('1/8', 'l3_mnoaef', 'v_ijgh', 't2_jmef', 't2_nogh'),
        _t('-1/2', 'l3_mnoaef', 'v_ijgh', 't2_jmeg', 't2_nofh'),
        _t('1/24', 'l3_mnoaef', 'v_ijgh', 't4_jmnoefgh'),
        _t('1/4', 'l3_mnoaef', 'v_ijgm', 't3_jnoefg'),
        _t('1/4', 'l3_mnoaef', 'v_ijmn', 't2_joef'),
        _t('-1/12', 'l3_mnoefg', 'v_eiah', 't3_mnofgh'),
        _t('-1/4', 'l3_mnoefg', 'v_ijah', 't2_jmef', 't2_nogh'),
        _t('-1/36', 'l3_mnoefg', 'v_ijah', 't4_jmnoefgh'),
        _t('-1/12', 'l3_mnoefg', 'v_ijam', 't3_jnoefg'),
        _t('-1/144', 'l4_imnoefgh', 'f_ja', 't4_jmnoefgh'),
        _t('1/24', 'l4_imnoefgh', 'v_efab', 't3_mnobgh'),
        _t('1/4', 'l4_imnoefgh', 'v_ejab', 't2_jmfg', 't2_nobh'),
        _t('1/36', 'l4_imnoefgh', 'v_ejab', 't4_jmnobfgh'),
        _t('-1/12', 'l4_imnoefgh', 'v_ejam', 't3_jnofgh'),
        _t('1/48', 'l4_imnoefgh', 'v_jkab', 't2_jkef', 't3_mnobgh'),
        _t('-1/12', 'l4_imnoefgh', 'v_jkab', 't2_jmbe', 't3_knofgh'),
        _t('-1/8', 'l4_imnoefgh', 'v_jkab', 't2_jmef', 't3_knobgh'),
        _t('1/24', 'l4_imnoefgh', 'v_jkab', 't2_mnbe', 't3_jkofgh'),
        _t('-1/8', 'l4_imnoefgh', 'v_jkam', 't2_jnef', 't2_kogh'),
        _t('1/96', 'l4_imnoefgh', 'v_jkam', 't4_jknoefgh'),
        _t('1/144', 'l4_jmnoaefg', 'f_ih', 't4_jmnoefgh'),
        _t('-1/8', 'l4_jmnoaefg', 'v_eibh', 't2_jobg', 't2_mnfh'),
        _t('1/96', 'l4_jmnoaefg', 'v_eibh', 't4_jmnobfgh'),
        _t('1/12', 'l4_jmnoaefg', 'v_eihm', 't3_jnofgh'),
        _t('1/24', 'l4_jmnoaefg', 'v_ikbh', 't2_kmef', 't3_jnobgh'),
        _t('1/12', 'l4_jmnoaefg', 'v_ikbh', 't2_kmeh', 't3_jnobfg'),
        _t('-1/48', 'l4_jmnoaefg', 'v_ikbh', 't2_mnbh', 't3_jkoefg'),
        _t('1/8', 'l4_jmnoaefg', 'v_ikbh', 't2_mneh', 't3_jkobfg'),
        _t('-1/4', 'l4_jmnoaefg', 'v_ikhm', 't2_jkfg', 't2_noeh'),
        _t('-1/36', 'l4_jmnoaefg', 'v_ikhm', 't4_jknoefgh'),
        _t('-1/24', 'l4_jmnoaefg', 'v_ikmn', 't3_jkoefg'),
        _t('-1/144', 'l4_jmnoefgh', 'v_eiab', 't4_jmnobfgh'),
        _t('-1/24', 'l4_jmnoefgh', 'v_ikab', 't2_kmef', 't3_jnobgh'),
        _t('-1/24', 'l4_jmnoefgh', 'v_ikab', 't2_mnbe', 't3_jkofgh'),
        _t('-1/144', 'l4_jmnoefgh', 'v_ikam', 't4_jknoefgh'),
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
        _t('-1/24', 'l4_ijmnaefg', 'f_ob', 't3_mnoefg'),
        _t('-1/8', 'l4_ijmnaefg', 'v_efbh', 't2_mngh'),
        _t('1/8', 'l4_ijmnaefg', 'v_eobh', 't3_mnofgh'),
        _t('1/4', 'l4_ijmnaefg', 'v_eobm', 't2_nofg'),
        _t('1/4', 'l4_ijmnaefg', 'v_kobh', 't2_knfg', 't2_moeh'),
        _t('-1/16', 'l4_ijmnaefg', 'v_kobh', 't2_koef', 't2_mngh'),
        _t('-1/48', 'l4_ijmnaefg', 'v_kobh', 't4_kmnoefgh'),
        _t('-1/24', 'l4_ijmnaefg', 'v_kobm', 't3_knoefg'),
        _t('-1/48', 'l4_ijmnefgh', 'v_eoab', 't3_mnofgh'),
        _t('-1/32', 'l4_ijmnefgh', 'v_koab', 't2_kngh', 't2_moef'),
        _t('1/384', 'l4_ijmnefgh', 'v_koab', 't4_kmnoefgh'),
        _t('-1/24', 'l4_imnoabef', 'f_jg', 't3_mnoefg'),
        _t('1/24', 'l4_imnoabef', 'v_ejgh', 't3_mnofgh'),
        _t('1/4', 'l4_imnoabef', 'v_ejgm', 't2_nofg'),
        _t('1/16', 'l4_imnoabef', 'v_jkgh', 't2_kmef', 't2_nogh'),
        _t('-1/4', 'l4_imnoabef', 'v_jkgh', 't2_kmeg', 't2_nofh'),
        _t('1/48', 'l4_imnoabef', 'v_jkgh', 't4_kmnoefgh'),
        _t('1/8', 'l4_imnoabef', 'v_jkgm', 't3_knoefg'),
        _t('1/8', 'l4_imnoabef', 'v_jkmn', 't2_koef'),
        _t('-1/12', 'l4_imnoaefg', 'v_ejbh', 't3_mnofgh'),
        _t('-1/4', 'l4_imnoaefg', 'v_jkbh', 't2_kmef', 't2_nogh'),
        _t('-1/36', 'l4_imnoaefg', 'v_jkbh', 't4_kmnoefgh'),
        _t('-1/12', 'l4_imnoaefg', 'v_jkbm', 't3_knoefg'),
        _t('1/288', 'l4_imnoefgh', 'v_jkab', 't4_kmnoefgh'),
        _t('-1/32', 'l4_kmnoabef', 'v_ijgh', 't2_kofh', 't2_mneg'),
        _t('1/384', 'l4_kmnoabef', 'v_ijgh', 't4_kmnoefgh'),
        _t('1/48', 'l4_kmnoabef', 'v_ijgm', 't3_knoefg'),
        _t('-1/288', 'l4_kmnoaefg', 'v_ijbh', 't4_kmnoefgh'),
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
        _t('1/24', 'l4_ijkmabef', 'f_nc', 't2_mnef'),
        _t('1/24', 'l4_ijkmabef', 'v_efcm'),
        _t('1/12', 'l4_ijkmabef', 'v_encg', 't2_mnfg'),
        _t('1/48', 'l4_ijkmabef', 'v_nocg', 't3_mnoefg'),
        _t('1/48', 'l4_ijkmabef', 'v_nocm', 't2_noef'),
        _t('1/24', 'l4_ijkmaefg', 'v_enbc', 't2_mnfg'),
        _t('1/144', 'l4_ijkmaefg', 'v_nobc', 't3_mnoefg'),
        _t('1/24', 'l4_ijmnabce', 'f_kf', 't2_mnef'),
        _t('1/48', 'l4_ijmnabce', 'v_ekfg', 't2_mnfg'),
        _t('1/24', 'l4_ijmnabce', 'v_ekmn'),
        _t('1/48', 'l4_ijmnabce', 'v_kofg', 't3_mnoefg'),
        _t('1/12', 'l4_ijmnabce', 'v_kofm', 't2_noef'),
        _t('1/8', 'l4_ijmnabef', 'v_ekcg', 't2_mnfg'),
        _t('1/16', 'l4_ijmnabef', 'v_kocg', 't3_mnoefg'),
        _t('1/8', 'l4_ijmnabef', 'v_kocm', 't2_noef'),
        _t('1/48', 'l4_ijmnaefg', 'v_kobc', 't3_mnoefg'),
        _t('1/144', 'l4_imnoabce', 'v_jkfg', 't3_mnoefg'),
        _t('1/24', 'l4_imnoabce', 'v_jkfm', 't2_noef'),
        _t('1/48', 'l4_imnoabef', 'v_jkcg', 't3_mnoefg'),
    ],
    4: [
        _t('1/16', 'l2_ijab', 'v_klcd'),
        _t('1/36', 'l3_ijkabc', 'f_ld'),
        _t('1/24', 'l3_ijkabe', 'v_elcd'),
        _t('1/24', 'l3_ijmabc', 'v_kldm'),
        _t('1/144', 'l4_ijklabce', 'f_ed'),
        _t('-1/288', 'l4_ijklabce', 'v_mndf', 't2_mnef'),
        _t('1/192', 'l4_ijklabef', 'v_efcd'),
        _t('1/384', 'l4_ijklabef', 'v_mncd', 't2_mnef'),
        _t('-1/144', 'l4_ijkmabcd', 'f_lm'),
        _t('-1/288', 'l4_ijkmabcd', 'v_lnef', 't2_mnef'),
        _t('-1/36', 'l4_ijkmabce', 'v_eldm'),
        _t('1/36', 'l4_ijkmabce', 'v_lndf', 't2_mnef'),
        _t('-1/48', 'l4_ijkmabef', 'v_lncd', 't2_mnef'),
        _t('1/384', 'l4_ijmnabcd', 'v_klef', 't2_mnef'),
        _t('1/192', 'l4_ijmnabcd', 'v_klmn'),
        _t('-1/48', 'l4_ijmnabce', 'v_kldf', 't2_mnef'),
    ],
}

DENSITY = {
    'ab': [
        _t('1/2', 'l2_mnae', 't2_mnbe'),
        _t('1/12', 'l3_mnoaef', 't3_mnobef'),
        _t('1/144', 'l4_imnoaefg', 't4_imnobefg'),
    ],
    'ai': [
        _t(1, 'l1_ia'),
    ],
    'ia': [
        _t(1, 'l1_me', 't2_imae'),
        _t('1/4', 'l2_mnef', 't3_imnaef'),
        _t('1/4', 'l3_mnoefg', 't2_imef', 't2_noag'),
        _t('1/36', 'l3_mnoefg', 't4_imnoaefg'),
        _t('-1/24', 'l4_jmnoefgh', 't2_imef', 't3_jnoagh'),
        _t('1/24', 'l4_jmnoefgh', 't2_mnae', 't3_ijofgh'),
    ],
    'ij': [
        _t('-1/2', 'l2_jmef', 't2_imef'),
        _t('-1/12', 'l3_jmnefg', 't3_imnefg'),
        _t('-1/144', 'l4_jmnoefgh', 't4_imnoefgh'),
    ],
    'abcd': [
        _t('1/8', 'l2_mnab', 't2_mncd'),
        _t('1/24', 'l3_mnoabe', 't3_mnocde'),
        _t('1/16', 'l4_imnoabef', 't2_iocf', 't2_mnde'),
        _t('1/192', 'l4_imnoabef', 't4_imnocdef'),
    ],
    'abci': [
        _t('-1/4', 'l3_imnabe', 't2_mnce'),
        _t('-1/24', 'l4_imnoabef', 't3_mnocef'),
    ],
    'abij': [
        _t('1/4', 'l2_ijab'),
    ],
    'aibc': [
        _t('-1/2', 'l1_ma', 't2_imbc'),
        _t('-1/4', 'l2_mnae', 't3_imnbce'),
        _t('1/2', 'l3_mnoaef', 't2_imbe', 't2_nocf'),
        _t('-1/8', 'l3_mnoaef', 't2_imef', 't2_nobc'),
        _t('-1/24', 'l3_mnoaef', 't4_imnobcef'),
        _t('-1/12', 'l4_jmnoaefg', 't2_imbe', 't3_jnocfg'),
        _t('1/24', 'l4_jmnoaefg', 't2_imef', 't3_jnobcg'),
        _t('-1/48', 'l4_jmnoaefg', 't2_mnbc', 't3_ijoefg'),
        _t('1/8', 'l4_jmnoaefg', 't2_mnbe', 't3_ijocfg'),
    ],
    'iabj': [
        _t(1, 'l2_jmae', 't2_imbe'),
        _t('1/4', 'l3_jmnaef', 't3_imnbef'),
        _t('1/4', 'l4_jmnoaefg', 't2_imef', 't2_nobg'),
        _t('1/36', 'l4_jmnoaefg', 't4_imnobefg'),
    ],
    'iajk': [
        _t('1/4', 'l3_jkmaef', 't2_imef'),
        _t('1/24', 'l4_jkmnaefg', 't3_imnefg'),
    ],
    'ijab': [
        _t('1/4', 'l1_me', 't3_ijmabe'),
        _t('1/4', 'l2_mnef', 't2_ijbe', 't2_mnaf'),
        _t('1/16', 'l2_mnef', 't2_ijef', 't2_mnab'),
        _t('1/2', 'l2_mnef', 't2_imae', 't2_jnbf'),
        _t('-1/4', 'l2_mnef', 't2_imef', 't2_jnab'),
        _t('1/16', 'l2_mnef', 't4_ijmnabef'),
        _t('-1/24', 'l3_mnoefg', 't2_ijae', 't3_mnobfg'),
        _t('1/48', 'l3_mnoefg', 't2_ijef', 't3_mnoabg'),
        _t('-1/24', 'l3_mnoefg', 't2_imab', 't3_jnoefg'),
        _t('1/4', 'l3_mnoefg', 't2_imae', 't3_jnobfg'),
        _t('-1/8', 'l3_mnoefg', 't2_imef', 't3_jnoabg'),
        _t('1/48', 'l3_mnoefg', 't2_mnab', 't3_ijoefg'),
        _t('-1/8', 'l3_mnoefg', 't2_mnae', 't3_ijobfg'),
        _t('-1/288', 'l4_kmnoefgh', 't2_ijae', 't4_kmnobfgh'),
        _t('1/32', 'l4_kmnoefgh', 't2_ijef', 't2_koah', 't2_mnbg'),
        _t('1/384', 'l4_kmnoefgh', 't2_ijef', 't4_kmnoabgh'),
        _t('-1/4', 'l4_kmnoefgh', 't2_ikah', 't2_jmef', 't2_nobg'),
        _t('-1/32', 'l4_kmnoefgh', 't2_ikgh', 't2_jmef', 't2_noab'),
        _t('1/288', 'l4_kmnoefgh', 't2_imab', 't4_jknoefgh'),
        _t('-1/36', 'l4_kmnoefgh', 't2_imae', 't4_jknobfgh'),
        _t('1/48', 'l4_kmnoefgh', 't2_imef', 't4_jknoabgh'),
        _t('1/384', 'l4_kmnoefgh', 't2_mnab', 't4_ijkoefgh'),
        _t('-1/48', 'l4_kmnoefgh', 't2_mnae', 't4_ijkobfgh'),
        _t('-1/48', 'l4_kmnoefgh', 't3_ijkbgh', 't3_mnoaef'),
        _t('-1/144', 'l4_kmnoefgh', 't3_ijkfgh', 't3_mnoabe'),
        _t('1/32', 'l4_kmnoefgh', 't3_ikoagh', 't3_jmnbef'),
        _t('-1/48', 'l4_kmnoefgh', 't3_ikofgh', 't3_jmnabe'),
        _t('1/4', 't2_ijab'),
    ],
    'ijak': [
        _t('-1/2', 'l1_ke', 't2_ijae'),
        _t('-1/4', 'l2_kmef', 't3_ijmaef'),
        _t('-1/8', 'l3_kmnefg', 't2_ijef', 't2_mnag'),
        _t('1/2', 'l3_kmnefg', 't2_imae', 't2_jnfg'),
        _t('-1/24', 'l3_kmnefg', 't4_ijmnaefg'),
        _t('-1/48', 'l4_kmnoefgh', 't2_ijef', 't3_mnoagh'),
        _t('1/12', 'l4_kmnoefgh', 't2_imae', 't3_jnofgh'),
        _t('1/8', 'l4_kmnoefgh', 't2_imef', 't3_jnoagh'),
        _t('-1/24', 'l4_kmnoefgh', 't2_mnae', 't3_ijofgh'),
    ],
    'ijkl': [
        _t('1/8', 'l2_klef', 't2_ijef'),
        _t('1/24', 'l3_klmefg', 't3_ijmefg'),
        _t('-1/16', 'l4_klmnefgh', 't2_imef', 't2_jngh'),
        _t('1/192', 'l4_klmnefgh', 't4_ijmnefgh'),
    ],
}

GRADIENT = [
    _t(-1, 'l1_me', 'f_af', 't2_imef'),
    _t(1, 'l1_me', 'f_ni', 't2_mnae'),
    _t('1/2', 'l1_me', 'v_aefg', 't2_imfg'),
    _t(1, 'l1_me', 'v_aeim'),
    _t('-1/2', 'l1_me', 'v_anfg', 't3_imnefg'),
    _t(-1, 'l1_me', 'v_anfi', 't2_mnef'),
    _t(1, 'l1_me', 'v_anfm', 't2_inef'),
    _t(1, 'l1_me', 'v_enfi', 't2_mnaf'),
    _t('-1/2', 'l1_me', 'v_nofi', 't3_mnoaef'),
    _t('1/2', 'l1_me', 'v_noim', 't2_noae'),
    _t('1/4', 'l2_mnef', 'f_ag', 't3_imnefg'),
    _t('-1/2', 'l2_mnef', 'f_am', 't2_inef'),
    _t('-1/2', 'l2_mnef', 'f_ei', 't2_mnaf'),
    _t('-1/4', 'l2_mnef', 'f_oi', 't3_mnoaef'),
    _t('1/4', 'l2_mnef', 'v_aegh', 't3_imnfgh'),
    _t('1/2', 'l2_mnef', 'v_aegi', 't2_mnfg'),
    _t(-1, 'l2_mnef', 'v_aegm', 't2_infg'),
    _t('1/4', 'l2_mnef', 'v_aogh', 't2_imef', 't2_nogh'),
    _t(-1, 'l2_mnef', 'v_aogh', 't2_imeg', 't2_nofh'),
    _t('1/4', 'l2_mnef', 'v_aogh', 't2_imgh', 't2_noef'),
    _t('1/8', 'l2_mnef', 'v_aogh', 't2_ioef', 't2_mngh'),
    _t('-1/2', 'l2_mnef', 'v_aogh', 't2_ioeg', 't2_mnfh'),
    _t('1/8', 'l2_mnef', 'v_aogh', 't4_imnoefgh'),
    _t('-1/4', 'l2_mnef', 'v_aogi', 't3_mnoefg'),
    _t('1/2', 'l2_mnef', 'v_aogm', 't3_inoefg'),
    _t('1/2', 'l2_mnef', 'v_aoim', 't2_noef'),
    _t('1/4', 'l2_mnef', 'v_aomn', 't2_ioef'),
    _t('1/4', 'l2_mnef', 'v_efgi', 't2_mnag'),
    _t('1/2', 'l2_mnef', 'v_eogi', 't3_mnoafg'),
    _t(-1, 'l2_mnef', 'v_eoim', 't2_noaf'),
    _t(1, 'l2_mnef', 'v_jogi', 't2_jnaf', 't2_moeg'),
    _t('-1/2', 'l2_mnef', 'v_jogi', 't2_jnag', 't2_moef'),
    _t('1/4', 'l2_mnef', 'v_jogi', 't2_joae', 't2_mnfg'),
    _t('1/8', 'l2_mnef', 'v_jogi', 't2_joef', 't2_mnag'),
    _t('-1/4', 'l2_mnef', 'v_jogi', 't2_joeg', 't2_mnaf'),
    _t('1/8', 'l2_mnef', 'v_jogi', 't4_jmnoaefg'),
    _t('-1/4', 'l2_mnef', 'v_joim', 't3_jnoaef'),
    _t('-1/4', 'l3_mnoefg', 'f_ah', 't2_imef', 't2_nogh'),
    _t('-1/36', 'l3_mnoefg', 'f_ah', 't4_imnoefgh'),
    _t('-1/12', 'l3_mnoefg', 'f_am', 't3_inoefg'),
    _t('-1/12', 'l3_mnoefg', 'f_ei', 't3_mnoafg'),
    _t('-1/4', 'l3_mnoefg', 'f_ji', 't2_jmef', 't2_noag'),
    _t('-1/36', 'l3_mnoefg', 'f_ji', 't4_jmnoaefg'),
    _t('1/8', 'l3_mnoefg', 'v_aebh', 't2_imfg', 't2_nobh'),
    _t('-1/2', 'l3_mnoefg', 'v_aebh', 't2_imfh', 't2_nobg'),
    _t('1/24', 'l3_mnoefg', 'v_aebh', 't4_imnobfgh'),
    _t('-1/12', 'l3_mnoefg', 'v_aehi', 't3_mnofgh'),
    _t('1/4', 'l3_mnoefg', 'v_aehm', 't3_inofgh'),
    _t('1/4', 'l3_mnoefg', 'v_aemn', 't2_iofg'),
    _t('-1/24', 'l3_mnoefg', 'v_ajbh', 't2_ijef', 't3_mnobgh'),
    _t('-1/12', 'l3_mnoefg', 'v_ajbh', 't2_ijeh', 't3_mnobfg'),
    _t('-1/24', 'l3_mnoefg', 'v_ajbh', 't2_imbh', 't3_jnoefg'),
    _t('1/8', 'l3_mnoefg', 'v_ajbh', 't2_imef', 't3_jnobgh'),
    _t('1/4', 'l3_mnoefg', 'v_ajbh', 't2_imeh', 't3_jnobfg'),
    _t('1/24', 'l3_mnoefg', 'v_ajbh', 't2_jmbh', 't3_inoefg'),
    _t('-1/8', 'l3_mnoefg', 'v_ajbh', 't2_jmef', 't3_inobgh'),
    _t('-1/4', 'l3_mnoefg', 'v_ajbh', 't2_jmeh', 't3_inobfg'),
    _t('1/24', 'l3_mnoefg', 'v_ajbh', 't2_mnbh', 't3_ijoefg'),
    _t('-1/4', 'l3_mnoefg', 'v_ajbh', 't2_mneh', 't3_ijobfg'),
    _t('1/4', 'l3_mnoefg', 'v_ajhi', 't2_jmef', 't2_nogh'),
    _t('1/36', 'l3_mnoefg', 'v_ajhi', 't4_jmnoefgh'),
    _t('1/4', 'l3_mnoefg', 'v_ajhm', 't2_ijef', 't2_nogh'),
    _t('-1/2', 'l3_mnoefg', 'v_ajhm', 't2_inef', 't2_jogh'),
    _t('-1/2', 'l3_mnoefg', 'v_ajhm', 't2_ineh', 't2_jofg'),
    _t('1/12', 'l3_mnoefg', 'v_ajhm', 't4_ijnoefgh'),
    _t('-1/12', 'l3_mnoefg', 'v_ajim', 't3_jnoefg'),
    _t('1/12', 'l3_mnoefg', 'v_ajmn', 't3_ijoefg'),
    _t('-1/12', 'l3_mnoefg', 'v_efhi', 't3_mnoagh'),
    _t('1/4', 'l3_mnoefg', 'v_efim', 't2_noag'),
    _t('-1/2', 'l3_mnoefg', 'v_ejhi', 't2_jmaf', 't2_nogh'),
    _t('-1/4', 'l3_mnoefg', 'v_ejhi', 't2_jmfg', 't2_noah'),
    _t('1/2', 'l3_mnoefg', 'v_ejhi', 't2_jmfh', 't2_noag'),
    _t('-1/12', 'l3_mnoefg', 'v_ejhi', 't4_jmnoafgh'),
    _t('1/4', 'l3_mnoefg', 'v_ejim', 't3_jnoafg'),
    _t('-1/24', 'l3_mnoefg', 'v_jkhi', 't2_jkae', 't3_mnofgh'),
    _t('-1/24', 'l3_mnoefg', 'v_jkhi', 't2_jkef', 't3_mnoagh'),
    _t('-1/24', 'l3_mnoefg', 'v_jkhi', 't2_jkeh', 't3_mnoafg'),
    _t('1/4', 'l3_mnoefg', 'v_jkhi', 't2_jmae', 't3_knofgh'),
    _t('-1/12', 'l3_mnoefg', 'v_jkhi', 't2_jmah', 't3_knoefg'),
    _t('1/4', 'l3_mnoefg', 'v_jkhi', 't2_jmef', 't3_knoagh'),
    _t('1/4', 'l3_mnoefg', 'v_jkhi', 't2_jmeh', 't3_knoafg'),
    _t('-1/8', 'l3_mnoefg', 'v_jkhi', 't2_mnae', 't3_jkofgh'),
    _t('1/24', 'l3_mnoefg', 'v_jkhi', 't2_mnah', 't3_jkoefg'),
    _t('-1/8', 'l3_mnoefg', 'v_jkhi', 't2_mneh', 't3_jkoafg'),
    _t('1/8', 'l3_mnoefg', 'v_jkim', 't2_jkef', 't2_noag'),
    _t('-1/2', 'l3_mnoefg', 'v_jkim', 't2_jnae', 't2_kofg'),
    _t('1/24', 'l3_mnoefg', 'v_jkim', 't4_jknoaefg'),
    _t('-1/24', 'l4_jmnoefgh', 'f_ab', 't2_imef', 't3_jnobgh'),
    _t('1/24', 'l4_jmnoefgh', 'f_ab', 't2_mnbe', 't3_ijofgh'),
    _t('1/144', 'l4_jmnoefgh', 'f_am', 't4_ijnoefgh'),
    _t('-1/144', 'l4_jmnoefgh', 'f_ei', 't4_jmnoafgh'),
    _t('1/24', 'l4_jmnoefgh', 'f_ki', 't2_kmef', 't3_jnoagh'),
    _t('1/24', 'l4_jmnoefgh', 'f_ki', 't2_mnae', 't3_jkofgh'),
    _t('1/12', 'l4_jmnoefgh', 'v_aebc', 't2_imbf', 't3_jnocgh'),
    _t('-1/24', 'l4_jmnoefgh', 'v_aebc', 't2_imfg', 't3_jnobch'),
    _t('1/48', 'l4_jmnoefgh', 'v_aebc', 't2_mnbc', 't3_ijofgh'),
    _t('-1/8', 'l4_jmnoefgh', 'v_aebc', 't2_mnbf', 't3_ijocgh'),
    _t('-1/144', 'l4_jmnoefgh', 'v_aebi', 't4_jmnobfgh'),
    _t('-1/4', 'l4_jmnoefgh', 'v_aebm', 't2_ijgh', 't2_nobf'),
    _t('-1/36', 'l4_jmnoefgh', 'v_aebm', 't4_ijnobfgh'),
    _t('1/24', 'l4_jmnoefgh', 'v_aemn', 't3_ijofgh'),
    _t('1/4', 'l4_jmnoefgh', 'v_akbc', 't2_ijch', 't2_kmef', 't2_nobg'),
    _t('1/4', 'l4_jmnoefgh', 'v_akbc', 't2_ijgh', 't2_kmbe', 't2_nocf'),
    _t('-1/16', 'l4_jmnoefgh', 'v_akbc', 't2_ijgh', 't2_kmef', 't2_nobc'),
    _t('-1/144', 'l4_jmnoefgh', 'v_akbc', 't2_ikbe', 't4_jmnocfgh'),
    _t('-1/16', 'l4_jmnoefgh', 'v_akbc', 't2_ikef', 't2_joch', 't2_mnbg'),
    _t('1/192', 'l4_jmnoefgh', 'v_akbc', 't2_ikef', 't4_jmnobcgh'),
    _t('-1/288', 'l4_jmnoefgh', 'v_akbc', 't2_imbc', 't4_jknoefgh'),
    _t('1/36', 'l4_jmnoefgh', 'v_akbc', 't2_imbe', 't4_jknocfgh'),
    _t('-1/48', 'l4_jmnoefgh', 'v_akbc', 't2_imef', 't4_jknobcgh'),
    _t('-1/288', 'l4_jmnoefgh', 'v_akbc', 't2_kmbc', 't4_ijnoefgh'),
    _t('1/36', 'l4_jmnoefgh', 'v_akbc', 't2_kmbe', 't4_ijnocfgh'),
    _t('-1/48', 'l4_jmnoefgh', 'v_akbc', 't2_kmef', 't4_ijnobcgh'),
    _t('-1/192', 'l4_jmnoefgh', 'v_akbc', 't2_mnbc', 't4_ijkoefgh'),
    _t('1/24', 'l4_jmnoefgh', 'v_akbc', 't2_mnbe', 't4_ijkocfgh'),
    _t('1/24', 'l4_jmnoefgh', 'v_akbc', 't3_ijkcgh', 't3_mnobef'),
    _t('1/72', 'l4_jmnoefgh', 'v_akbc', 't3_ijkfgh', 't3_mnobce'),
    _t('-1/48', 'l4_jmnoefgh', 'v_akbc', 't3_ijobch', 't3_kmnefg'),
    _t('-1/16', 'l4_jmnoefgh', 'v_akbc', 't3_ijocgh', 't3_kmnbef'),
    _t('-1/48', 'l4_jmnoefgh', 'v_akbc', 't3_ijofgh', 't3_kmnbce'),
    _t('1/24', 'l4_jmnoefgh', 'v_akbi', 't2_kmef', 't3_jnobgh'),
    _t('1/24', 'l4_jmnoefgh', 'v_akbi', 't2_mnbe', 't3_jkofgh'),
    _t('1/24', 'l4_jmnoefgh', 'v_akbm', 't2_ikef', 't3_jnobgh'),
    _t('-1/12', 'l4_jmnoefgh', 'v_akbm', 't2_inbe', 't3_jkofgh'),
    _t('-1/8', 'l4_jmnoefgh', 'v_akbm', 't2_inef', 't3_jkobgh'),
    _t('-1/12', 'l4_jmnoefgh', 'v_akbm', 't2_knbe', 't3_ijofgh'),
    _t('-1/8', 'l4_jmnoefgh', 'v_akbm', 't2_knef', 't3_ijobgh'),
    _t('-1/12', 'l4_jmnoefgh', 'v_akbm', 't2_nobe', 't3_ijkfgh'),
    _t('-1/144', 'l4_jmnoefgh', 'v_akim', 't4_jknoefgh'),
    _t('-1/8', 'l4_jmnoefgh', 'v_akmn', 't2_ijgh', 't2_koef'),
    _t('-1/96', 'l4_jmnoefgh', 'v_akmn', 't4_ijkoefgh'),
    _t('1/8', 'l4_jmnoefgh', 'v_efbi', 't2_joah', 't2_mnbg'),
    _t('1/96', 'l4_jmnoefgh', 'v_efbi', 't4_jmnoabgh'),
    _t('-1/24', 'l4_jmnoefgh', 'v_efim', 't3_jnoagh'),
    _t('-1/12', 'l4_jmnoefgh', 'v_ekbi', 't2_kmaf', 't3_jnobgh'),
    _t('1/12', 'l4_jmnoefgh', 'v_ekbi', 't2_kmbf', 't3_jnoagh'),
    _t('1/12', 'l4_jmnoefgh', 'v_ekbi', 't2_kmfg', 't3_jnoabh'),
    _t('1/24', 'l4_jmnoefgh', 'v_ekbi', 't2_mnab', 't3_jkofgh'),
    _t('-1/8', 'l4_jmnoefgh', 'v_ekbi', 't2_mnaf', 't3_jkobgh'),
    _t('1/8', 'l4_jmnoefgh', 'v_ekbi', 't2_mnbf', 't3_jkoagh'),
    _t('1/4', 'l4_jmnoefgh', 'v_ekim', 't2_jkgh', 't2_noaf'),
    _t('1/36', 'l4_jmnoefgh', 'v_ekim', 't4_jknoafgh'),
    _t('-1/4', 'l4_jmnoefgh', 'v_klbi', 't2_jlah', 't2_kmef', 't2_nobg'),
    _t('1/4', 'l4_jmnoefgh', 'v_klbi', 't2_jlbh', 't2_kmef', 't2_noag'),
    _t('-1/16', 'l4_jmnoefgh', 'v_klbi', 't2_jlgh', 't2_kmef', 't2_noab'),
    _t('1/16', 'l4_jmnoefgh', 'v_klbi', 't2_joaf', 't2_klgh', 't2_mnbe'),
    _t('-1/288', 'l4_jmnoefgh', 'v_klbi', 't2_klae', 't4_jmnobfgh'),
    _t('1/288', 'l4_jmnoefgh', 'v_klbi', 't2_klbe', 't4_jmnoafgh'),
    _t('1/192', 'l4_jmnoefgh', 'v_klbi', 't2_klef', 't4_jmnoabgh'),
    _t('-1/144', 'l4_jmnoefgh', 'v_klbi', 't2_kmab', 't4_jlnoefgh'),
    _t('1/36', 'l4_jmnoefgh', 'v_klbi', 't2_kmae', 't4_jlnobfgh'),
    _t('-1/36', 'l4_jmnoefgh', 'v_klbi', 't2_kmbe', 't4_jlnoafgh'),
    _t('-1/24', 'l4_jmnoefgh', 'v_klbi', 't2_kmef', 't4_jlnoabgh'),
    _t('1/192', 'l4_jmnoefgh', 'v_klbi', 't2_mnab', 't4_jkloefgh'),
    _t('-1/48', 'l4_jmnoefgh', 'v_klbi', 't2_mnae', 't4_jklobfgh'),
    _t('1/48', 'l4_jmnoefgh', 'v_klbi', 't2_mnbe', 't4_jkloafgh'),
    _t('1/48', 'l4_jmnoefgh', 'v_klbi', 't3_jklagh', 't3_mnobef'),
    _t('-1/48', 'l4_jmnoefgh', 'v_klbi', 't3_jklbgh', 't3_mnoaef'),
    _t('-1/72', 'l4_jmnoefgh', 'v_klbi', 't3_jklfgh', 't3_mnoabe'),
    _t('-1/24', 'l4_jmnoefgh', 'v_klbi', 't3_jloabh', 't3_kmnefg'),
    _t('1/16', 'l4_jmnoefgh', 'v_klbi', 't3_jloagh', 't3_kmnbef'),
    _t('-1/48', 'l4_jmnoefgh', 'v_klim', 't2_klef', 't3_jnoagh'),
    _t('1/12', 'l4_jmnoefgh', 'v_klim', 't2_knae', 't3_jlofgh'),
    _t('1/8', 'l4_jmnoefgh', 'v_klim', 't2_knef', 't3_jloagh'),
    _t('-1/24', 'l4_jmnoefgh', 'v_klim', 't2_noae', 't3_jklfgh'),
    _t(1, 'f_ai'),
    _t('1/2', 'v_amef', 't2_imef'),
    _t('1/2', 'v_mnei', 't2_mnae'),
]


# ------------------------------------------------------------ dense evaluation

AMPLITUDES = ('t2', 't3', 't4', 'l1', 'l2', 'l3', 'l4')


def residual(rank, arrays, nocc):
    '''The residual of the Lambda equation of ``rank`` (1 .. 4) as a dense
    antisymmetric array [i.., a..], from the dressed f and v.  arrays:
    {'f', 'v'} over all spin orbitals and {'t2', 't3', 't4', 'l1', .., 'l4'}
    as [i.., a..] (lambda_terms.product).'''
    x = total(LAMBDA[rank], arrays, nocc, Q.EXT[rank])
    return x if rank == 1 else Q.asym(x, rank)


def density(arrays, nocc):
    '''(g1[p,q], g2[p,q,r,s]) of (*) over all spin orbitals: the
    coefficients of the dressed f and of the dressed <pq||rs> in L - E0, g2
    antisymmetric in (p,q) and in (r,s).  ``arrays``: the amplitudes ('v'
    is asked for its size only).'''
    n = arrays['v'].shape[0]
    dtype = np.result_type(*[arrays[k] for k in AMPLITUDES])
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


def gradient_blocks():
    '''{block: [(coefficient, factors)]}: the terms of GRADIENT by the
    block of dressed integrals in them (lambda_terms.gradient_blocks).'''
    out = {}
    for coef, factors in GRADIENT:
        kind, ix = next(f for f in factors if f[0] in ('f', 'v'))
        cls = ''.join('o' if c in OCC else 'v' for c in ix)
        key = 'f.' + cls if kind == 'f' else cls[:2] + '.' + cls[2:]
        out.setdefault(key, []).append((coef, factors))
    return out


def quadruple(terms):
    '''The terms that have t4 or l4 in them.'''
    return [(c, f) for c, f in terms if any(k in ('t4', 'l4') for k, _ in f)]


def ccsdt_limit():
    '''(LAMBDA, DENSITY, GRADIENT) without the terms of t4 and l4: the
    tables of zccsdt.cc.lambda_terms (LAMBDA[4] is then
    <0| (Lambda_2 + Lambda_3) H |Q>, which the equations of CCSDT do not
    have).'''
    def rest(terms):
        return [t for t in terms if t not in quadruple(terms)]
    lam = {k: rest(v) for k, v in LAMBDA.items()}
    dens = {b: rest(v) for b, v in DENSITY.items()}
    return lam, {b: v for b, v in dens.items() if v}, rest(GRADIENT)
