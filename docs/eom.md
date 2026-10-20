# Equation-of-motion CCSDT on spinors

Excited (EE), ionized (IP) and electron-attached (EA) states on top of a
converged CCSDT (or CCSD) ground state of spin orbitals or spinors, real or
complex: excitation energies and right-hand vectors; for EE-CCSDT also the
left-hand vectors, the densities of the excited states, transition moments
and a one-shot quadruples correction of the excitation energies (sections
12 to 15).

| What | Where |
|---|---|
| dense reference: sigma vector three ways, matrix, IP / EA by a ghost orbital | `src/zccsdt/cc/eom_ref.py`, `tests/test_eom_ref.py` |
| sigma vector on packed amplitudes | `src/zccsdt/cc/eom_opt.py`, `tests/test_eom_opt.py` |
| Davidson solver, memory plan, driver, ghost spinor | `src/zccsdt/cc/eom.py`, `tests/test_eom.py`, `tests/test_eom_solver.py` |
| states with a label shift in the blocks of the symmetry labels | `src/zccsdt/cc/t3shift.py`, `tests/test_eom_shift.py` |
| IP and EA with short vectors | `eom.ghost_short`, `tests/test_eom_short.py` |
| front end | `GCCSDT.eom_ee`, `eom_ip`, `eom_ea` (`src/zccsdt/ccsdt.py`; `docs/usage.md`) |
| double electron attachment (EOM-DEA; section 11): two ghost spinors, label blocks, the 5p3h operators folded, any rank on determinants | `src/zccsdt/cc/dea.py`, `dea5.py`, `dea_strings.py`, `tests/test_dea*.py`, `GCCSDT.eom_dea`, `bench/dea/` |
| left-hand vectors of EOM-EE-CCSDT (section 12) | `src/zccsdt/cc/eom_left.py`, `lambda_shift.py` (the Lambda passes in the blocks of a label shift), `tests/test_eom_left.py`, `GCCSDT.eom_left` |
| densities of excited states, transition densities, dipole and transition moments (section 13) | `src/zccsdt/cc/eom_density.py`, `tests/test_eom_density.py`, `GCCSDT.eom_density`, `eom_rdm1`, `eom_properties` |
| one-shot quadruples correction of the excitation energies (section 14) | `src/zccsdt/cc/eom_q.py`, `tests/test_eom_q.py`, `GCCSDT.eom_quadruples`, `bench/eom_q_bench.py` |
| reference runs | `bench/refs/reference_energies.json` (`eom_ee`), `bench/dirac/mrcc_eom.sh`, `bench/cfour/eom_ccsd.sh` |
| timing | `bench/eom_bench.py`, `bench/slurm_eom.sh`; label blocks: `bench/eom2_bench.py`, `bench/slurm_eom2.sh`; left-hand vectors and densities: `bench/eom_left_bench.py`, `bench/slurm_eom_left.sh` |
| the native program in CFOUR's chain (`xzccsdt eom`, `ccsdt_eom`): the sigma vector of section 3 and the label shifts of section 6 in Fortran, with the solver of section 9 | `cfour/xzccsdt/xzeom.inc`, `xzeoms.inc`, `xzeomd.F90`; docs/native.md, section 16 |

Stage two (sections 4 to 6, 8): states with a label shift in the blocks of
a linear molecule, IP and EA with short vectors, the threaded solver, the
choice of the roots.

Not done: the response of the spinors in the properties of excited states
(relaxed densities, gradients), left-hand vectors of IP / EA / DEA states
and of the approximate models.  The native Fortran program has
EOM-EE-CCSDT (not IP / EA, right-hand vectors only): docs/native.md,
section 16, with its validation against this code and against MRCC and
CFOUR; what the left-hand vectors would take there: section 15.

Notation as in `docs/factorization.md` and `docs/lambda.md`: `o`, `v`
occupied and virtual spin orbitals or spinors; i j k l m n occupied, a b c d
e f virtual; `P`, `Q` pairs i<j, a<b; `<pq||rs>` antisymmetrized integrals;
`H~`, `<pq||rs>~`, `f~` the T1-dressed Hamiltonian, integrals and Fock
matrix (`docs/lambda.md`, section 2).

## 1. The eigenproblem

With `Hbar = exp(-T) H exp(T)`, T = T1 + T2 + T3 the converged cluster
operator, and an excitation operator R = R1 + R2 + R3 (the same manifold as
T; r0 is not needed for the energies),

    sigma_mu(R) = <mu| [Hbar, R] |0> = omega r_mu,      mu in singles, doubles, triples.

([Hbar, R] |0> = (Hbar R)_connected |0>: R commutes with T, and the
disconnected part is E r_mu for converged amplitudes.)  omega is the
excitation energy; the matrix is not Hermitian, so its right and left
eigenvectors differ; only the right ones are computed here.

**The sigma vector is the Jacobian of the amplitude equations on R.**  With
the residual `Res_mu(T) = <mu| exp(-T) H exp(T) |0>`,

    d/dx Res_mu(T + x R) at x = 0  =  <mu| [Hbar, R] |0>,

for any T, converged or not.  That gives a route to the sigma vector that
needs no new algebra: the directional derivative of an existing residual
code.  `Hbar` ends at the fourth power of T, so the residuals are
polynomials of degree at most 4 in the amplitudes (jointly; the fourth
power only through T1), and

    [8 (Res(T + R) - Res(T - R)) - (Res(T + 2R) - Res(T - 2R))] / 12

is their exact derivative along R - four evaluations of the residual, exact
to the rounding (`eom_ref.sigma(how='stencil')`, the formula of
`qlgrad_ref.directional`, which the zeta equation of CCSDT(Q)_Lambda uses in
the same way, `docs/lambda.md` 12).  In the T1-dressed formalism the
residuals are linear in the dressed integrals and at most quadratic in
(T2, T3) - R1 linear, R2 and R3 quadratic, the only product with T3 being
T2 x T3 - so three evaluations are enough: central differences in (T2, T3)
at fixed dressing, which are exact for a quadratic, plus one evaluation
with the derivative of the dressed integrals (`how='dressed'`).

**Size of a vector.**  o v + C(o,2) C(v,2) + C(o,3) C(v,3) numbers (unique
elements i<j<k, a<b<c): the size of T1, T2 and the packed T3.  A Davidson
subspace of n vectors holds 2 n of them (vectors and sigma vectors).

**EOM-CCSD** is the same with two ranks, on CCSD amplitudes: every formula
below without its T3 and R3 terms.

## 2. The sigma vector in the T1-dressed formalism

`exp(-T1) H exp(T1) = H~` has the integrals `<pq||rs>~` and the Fock matrix
`f~` of `ccsdt._t1_dress`; the equations are those of T1 = 0 with H~
(`zccsdt.cc.ccsdtq_terms`: 5 + 14 + 22 terms for R1, R2, R3 of CCSDT, from
Wick's theorem).  Differentiating along R = (r1, r2, r3):

    sigma = Res[H'; t2, t3]  +  sum over the terms of Res[H~; t2, t3] of the
            term with each amplitude factor replaced in turn by r of its rank,

    H' = [H~, R1] = d/dx exp(-x R1) H~ exp(x R1):

the dressing is a group action, T1 and R1 commute, so the derivative of the
dressed integrals along r1 is a one-index transformation of the *dressed*
integrals, first order only (`eom_ref.dressed_derivative`):

- every virtual bra index a picks up `-sum_m r1[m,a] <m..||..>~`,
- every occupied ket index i picks up `+sum_e r1[i,e] <..||..e>~`,
- the Fock matrix: the same on its two indices, plus `sum_ia <ri||sa>~ r1[i,a]`.

`<mn||ef>' = 0` (no index of that block is dressed), which removes every
term with H' and the T3 part of an intermediate.  The blocks:

    f'[m,e]      = <mn||ef> r1[n,f]
    f'[m,i]      = f~[m,e] r1[i,e] + <mn||ie>~ r1[n,e]
    f'[a,e]      = -r1[m,a] f~[m,e] + <an||ef>~ r1[n,f]
    f'[a,i]      = -r1[m,a] f~[m,i] + f~[a,e] r1[i,e] + <na||fi>~ r1[n,f]
    <mn||ei>'    = <mn||ef> r1[i,f]                       <mb||ef>'  = -r1[n,b] <mn||ef>
    <mn||ij>'    = P(ij) <mn||ej>~ r1[i,e]
    <ma||di>'    = -r1[n,a] <mn||di>~ + <ma||de>~ r1[i,e]
    <ma||ij>'    = -r1[n,a] <mn||ij>~ + P(ij) <ma||ej>~ r1[i,e]
    <ab||de>'    = -P(ab) r1[m,a] <mb||de>~
    <ab||ei>'    = -P(ab) r1[m,a] <mb||ei>~ + <ab||ef>~ r1[i,f]
    <ab||ij>'    = -P(ab) r1[m,a] <mb||ij>~ + P(ij) <ab||ej>~ r1[i,e]

`eom_ref.sigma(how='terms')` evaluates exactly this on dense arrays: the
term table with `v`, `f` replaced by `v'`, `f'`, plus the table
differentiated factor by factor (`eom_ref.linearized_terms`).  It is the
form the code on packed amplitudes takes.

## 3. On the factorization of the CCSDT step

The amplitude step (`ccsdt_opt.update_amps`, `docs/factorization.md`) is

    R3 = residual(T3; W_ring, W_pp, O_hh | W_vvvo x t2, W_vooo x t2)

- the passes over packed T3 with the operators of the ring term, of the pp
ladder (F_vv folded in) and of the hh ladder (F_oo folded in), which act
linearly on T3, and the two drive terms, bilinear in (W, t2) - with the
intermediates W of H~, t2 and (W_vvvo, W_vooo only) T3.  Every W is linear
in H~ and affine in (t2, T3).  Hence

    sigma_3 = residual(R3; W_ring, W_pp, O_hh | W'_vvvo x t2, W'_vooo x t2)
            + residual(T3; W'_ring, W'_pp, O'_hh | W_vvvo x r2, W_vooo x r2),

    W' = W[H'; t2, T3] + (the part of W linear in t2, T3)[H~; r2, R3]:

**two calls of the engine function that the ground state calls once**
(`DenseT3.residual` / `SpinBlockT3.residual`), with other operands.  No
kernel was written for EOM, and the engine of the label blocks serves as it
is.  `ccsdt_opt.update_amps(capture=)` hands the dressed integrals and the
operands of the converged ground state to `eom_opt.prepare` - one amplitude
step; the function is otherwise unchanged (its tests pass unchanged).

| Quantity | Ground state (kept by `prepare`) | Derivative along R (per sigma vector) |
|---|---|---|
| F_oo[m,i] | `f~ + 1/2 <mn\|\|ef> t2[in,ef]` | `f' + 1/2 <mn\|\|ef> r2[in,ef]` |
| F_vv[a,d] | `f~ - 1/2 t2[mn,af] <mn\|\|df>` | `f' - 1/2 r2[mn,af] <mn\|\|df>` |
| W_ovvo (ring operator; CCSD ring) | `<ma\|\|di>~ + <mn\|\|de> t2[in,ae]` | `<ma\|\|di>' + <mn\|\|de> r2[in,ae]` (one (ov)^2 GEMM, `X(r2)`; its partial traces are the r2 parts of F') |
| W_pp[ab,de] (pp ladder, packed pairs) | `<ab\|\|de>~ + t2[mn,ab] <mn\|\|de> + 1/2 A[F_vv delta]` | `-P(ab) r1[m,a] <mb\|\|de>~ + r2[mn,ab] <mn\|\|de> + 1/2 A[F'_vv delta]`: v^4/4 numbers, built from the packed `<mb\|\|de>~` as the T1 dressing of `<ab\|\|de>` is |
| W_oooo, O_hh (hh ladder + F_oo) | `<mn\|\|ij>~ + 1/2 <mn\|\|ef> t2[ij,ef]` | `<mn\|\|ij>' + 1/2 <mn\|\|ef> r2[ij,ef]`, a second `t3kern.OccOperator` |
| W_vvvo[ab,ei] (drive) | `<ab\|\|ei>~ + T2 terms + <mn\|\|ef> T3` | see below |
| W_vooo[m,a,ij] (drive) | `<ma\|\|ij>~ + T2 terms + <mn\|\|ef> T3` | `<ma\|\|ij>'` + the T2 terms with (`<mn\|\|ie>'`, t2) and with (H~, r2) + `r1[n,b] (<mn\|\|ef> t2[ij,ef])` + pass 1 over R3 |
| pass 1: T3 -> T1, T2, W | `eng.pass1(T3; H~)` | `eng.pass1(R3; H~)` - all four outputs - and `eng.pass1(T3; H')`, of which only the T2 part is not zero |

W'_vvvo, the one o v^4 object of a sigma vector, without any new product:

    W'[ab,ei] = -P(ab) r1[m,a] W_ovvo[m,b,e,i]            <ab||ei>' (first part) and -P(ab) <mb||ef>' t2[mi,af]: together one GEMM with the kept W_ovvo
              + sum_f W_pp[ab,ef] r1[i,f]                 <ab||ef>~ r1[i,f] from the ladder operator (`pair_t1_contract`); its T2 part is the term
                                                          1/2 <mn||ei>' t2[mn,ab] exactly, its F_vv part is taken out again (o v^3 work)
              - f'[m,e] t2[mi,ab]
              + (T2 terms of W_vvvo)[H~; r2]              `ccsdt_opt.wvvvo_packed` with r2 in the place of t2
              + <mn||ef> R3[imn,abf]                      pass 1 over R3

sigma_1 and sigma_2 (the linearized CCSD equations with the T3 terms) are
GEMMs on the layouts of `ccsd_residual_packed`; two points are worth
writing down:

- **The pp ladder of sigma_2 uses the T3 ladder operator.**  `W_pp x (r2 +
  dtau1)` = the wanted `1/2 <ab||cd>~ (r2 + dtau1)[ij,cd]` + `t2 <mn||cd>
  (r2 + dtau1)` + half of `P(ab) F_bc r2[ij,ac]`; the second is removed by
  a product in pair space (its r2 part comes back with W'_oooo), the third
  completed.  So the dressed `<ab||cd>~` is not kept besides W_pp: the pp
  operator is in memory twice (W_pp, W'_pp), as in the ground state (bare,
  dressed), not three times.  dtau1 = the derivative of
  `t1[i,a] t1[j,b] - ..` along r1, the ket part of `<ab||ij>'`.
- **`<ab||cd>'` on (t2 + tau1) would count the tau1 part twice**: the
  derivative of the dressed `<ab||ij>~` on its bra is all in
  `-P(ab) r1[m,a] <mb||ij>~` with the fully dressed `<mb||ij>~` (the first
  version had it both ways: an error of 1e-4 of sigma_2, found by the
  reference).

**Cost.**  Two T3 residuals (pass 2, pp ladder, hh ladder each) and two
passes 1, one `t2 x <mb||ef>` product (o^2 v^4, as in W_vvvo of the ground
state), o v^4 for `<ab||de>'` and `W_pp r1`, and (ov)^3: about two
amplitude iterations less the T1 dressing.  Measured: section 8.

**The factorized second residual: what holds and what does not.**  The
second residual - T3 with the derivatives of the operators - was expected
to cost a fraction of the first ("a sigma vector little more than one
iteration").  Term by term:

- *The ring term cannot be reduced.*  `P(i/jk) P(a/bc) sum_md W'[m,d,i,a]
  T3[mjk,dbc]` with `W' = <ma||di>' + <mn||de> r2[in,ae]`: whichever
  factor meets T3 first - `(g r2) T3`, as now, or `(g T3) r2` with a
  six-index intermediate, stored or not - the last product sums two
  single indices against two free ones with the spectators (jk, bc):
  (o v)^2 C(o,2) C(v,2) flops, a ring pass.  W' is an (o v) x (o v)
  matrix of full rank; a low rank in the occupied indices exists only in
  its r1 parts, which are an o / v share of it.  The first version of
  this section said "likewise the ring term": wrong.
- *The pp ladder can be replaced.*  `W'_pp = -P(ab) r1[m,a] <mb||de>~ +
  r2[mn,ab] <mn||de> + 1/2 A[F'_vv delta]`.  With Y[(mn), I, c] = sum_de
  <mn||de> T3[I, dec] and Z[m, b, I, c] = sum_de <mb||de>~ T3[I, dec] -
  both independent of R, made once in `prepare` from one pass over T3,
  (C(o,2) + o v) C(o,3) v numbers, 6 o / v of the size of T3 - the ladder
  becomes `r2[P, (ab)] Y[P, I, c] - P(ab) r1[m,a] Z[m,b,I,c]` followed by
  the reduction P(c/ab) of the ladder: o^5 v^3 / 24 + o^4 v^3 / 6 flops
  (with the r1 part as v x o products per b and a six-term reduction of
  a v^3 cube; as one GEMM with the operator [r2 | r1 delta] it costs
  (o / v)^2 + 2 o / v of the ladder).  The F'_vv part moves into the ring
  operator of the same residual at no cost, as `F'_vv[a,d] delta_im / 3`.
- *Pass 1 over T3 with H'*: two of its three T2 terms follow from the T3
  parts of W_vvvo and W_vooo of the ground state contracted with r1; the
  third, `f'[m,e] T3[ijm,abe]`, still gathers T3.

So the saving is bounded by the pp ladder of the second residual, less
what replaces it (its reduction, which is about a sixth of it, and the
products above).  Its share of a sigma vector, measured (the timer
`eom.t3/pp_ladder`):

| | o + v | pp ladder of the second residual | of a sigma vector | what could be saved |
|---|---|---|---|---|
| H2O / cc-pVTZ, packed block, 8 threads (job 31825453) | 10 + 106 | 1.39 s | 8.18 s: 17% | about 11% |
| HBr / cc-pVTZ-DK, label blocks, 24 threads (job 31828491) | 8 + 78 | 0.0105 s | 0.372 s: 2.8% | - |
| HBr / cc-pVQZ-DK, label blocks, 24 threads | 26 + 120 | 0.78 s | 11.06 s: 7.1% | about 2.5% (o / v = 0.22: the products that replace the ladder cost half of it) |

**Not implemented**: for the linear molecules in label blocks that this
stage is about the gain is a few percent of a sigma vector, against two
stored intermediates of the size of T3, a new reduction kernel and their
block and shift structure.  The ratio of a sigma vector to an iteration
stays near 1.7 to 2; what remains above the two residuals is the two
passes 1 and the dense small terms (section 8).

## 4. The solver (`zccsdt.cc.eom.davidson`)

Block Davidson for a general matrix, right-hand vectors only, real or
complex arithmetic (the arithmetic of the integrals):

1. An orthonormal basis V; sigma vectors S = A V; `G = V^H S`.
2. `scipy.linalg.eig(G)`.  The roots: the ``nroots`` lowest real parts, or
   - when a guess is followed - those with the largest overlap with the
   roots of the iteration before (with the guess at the start).
3. Degenerate roots (closer than 1e-9, or a hundredth of the threshold):
   the eigenvectors that a general eigensolver returns for them can be
   nearly parallel, and in real arithmetic two roots that are degenerate
   but for the rounding come as a complex pair with one vector.  They are
   replaced by an orthonormal basis of their space - the right singular
   vectors of `G - omega` to its smallest singular values, turned as close
   to the given vectors as possible.  Before this was handled, real
   arithmetic lost a component of the triplet of LiH between the guess
   stage and the triples stage in some runs and not in others (the
   rounding of the threads decides whether the pair comes out complex).
4. For every root that is not converged: the residual `S y - omega V y`,
   divided by `omega - D` (D: the differences of the orbital energies,
   `e_a + .. - e_i - ..`; denominators below 1e-3 are replaced),
   orthogonalized twice against V and the other corrections, added to V.
   A correction that vanishes is dropped.
5. Converged: the norm of the residual of the normalized vector below
   `conv_tol` (default 1e-6; the error of omega is of second order in it
   for a nearly Hermitian matrix: 1e-8 gave 2e-10 Eh against MRCC).
6. When V would exceed `max_space` vectors (default 8 + 4 nroots) it is
   replaced by an orthonormal basis of the current roots, and S with it, in
   place and in chunks (`eom.transform`: no array of the size of a vector
   is made).

**Guess.**  For EOM-CCSDT the roots of the singles-doubles block of the
same matrix - the sigma vector without R3 and without the three T3 terms
of sigma_2, with no work of the size of T3 (section 8) - from unit vectors on the
lowest single excitations, converged to 1e-4; they are followed.  (It is
not EOM-CCSD - the amplitudes are those of CCSDT - but as good a start.)
For EOM-CCSD the unit vectors themselves.  `guess=` takes vectors (r1, r2).
Spin is not a symmetry of the code: a triplet is three roots, in spinors
split by spin-orbit coupling.

**Which roots come back** (`eom.kernel`; found with F- and Li, where the
first version lost roots).

- The singles-doubles stage solves for `extra` (default 2) roots more than
  asked for, from as many unit vectors more.  Where the single excitations
  run out - IP of three electrons has three - the guess goes on with unit
  vectors on the lowest doubles of the shift (the first version returned
  three roots for four asked, without a word).  If fewer roots can be
  started than asked for, a warning says so (the ghost route at full
  length knows its singles only).
- The triples stage starts from all of these vectors and takes the
  **lowest** roots; it does not follow the singles-doubles roots one by
  one.  A given `guess=` is followed, by the weight of an eigenvector in
  the space of the previous roots - the first version matched the roots of
  two iterations one to one, and inside a degenerate level, whose
  eigenvectors are rotated at will from one iteration to the next, the
  last root went to a vector far above (F-, 2p^-1: five roots at 3.1402 eV
  and one at 67 eV; now six at 3.1402).
- `nroots` that cut a degenerate level (the next roots of the
  singles-doubles stage within 2e-6 Eh of the last one): a warning and
  `Result.cut_level`; with `complete=True` the rest of the level is
  computed too and more roots than asked for come back.
- What is not guaranteed: a state without a single excitation in its
  sector (the M_S = -1 component of the triplet of Li+ from an alpha
  reference: 2h1p only) is found if a doubles guess of its sector is among
  the vectors started and its singles-doubles root among the nroots +
  extra lowest - for Li with `extra=10`, not with the default, where the
  singlet above the triplet takes the fourth place.  The energies that are
  returned are right in either case (Li: the FCI differences to 1e-8).
  Sectors (below) are the remedy that does not depend on the guess.

**Degenerate levels and sectors.**  The solver finds a degenerate level
whole - asked for six roots of a spectrum that starts with a threefold
level, a single root and a pair it returns those six with orthonormal
vectors (`tests/test_eom_solver.py`) -, so on the packed block the three
components of a triplet are three roots, each with its sigma vectors.
What tells the components apart is a label that adds, and where the
spinors have one the states of one label shift D (section 6) are an
eigenproblem of their own:

- in the blocks of the labels of a linear molecule: `eom_ee(shift=D)`
  (section 6), the cheap way - the vectors are those of the shift only;
- on the packed block, by masks (`eom.Sector`, `eom.kernel(sector=(labels,
  D))`): the guess is taken from the singles with l_a - l_i = D, and every
  sigma vector is projected on the elements of the shift (r1, r2 by masks,
  the packed r3 row by row from the sums of the labels of the triples, one
  pass of numpy over the vector).  The cost per sigma vector is that of
  the packed block; what is saved is the roots that are not asked for.
  The front end uses it for `eom_ee(shift=D)` when the labels are not in
  use in the blocks: 2 m_j with `symmetry = False`, and **2 m_s for spin
  orbitals of pure spin** (an RHF or UHF reference as GHF): shift 2 holds
  the M_S = 1 components of the triplets (and of higher multiplets) and no
  singlet, so each triplet is computed once; shift 0 holds the singlets
  and the M_S = 0 components; shift -2 is the time-reversed image of
  shift 2 for a closed shell and is not computed.  H2O / 6-31G: the two
  lowest triplets as the two lowest roots of shift 2, to 1e-9 of MRCC's.
- Not done, because it is not cheap: telling singlets from the M_S = 0
  components of triplets inside shift 0 (the eigenvalue of the spin flip;
  for Kramers-restricted spinors that of time reversal, which is a linear
  map on real vectors only), and deflating the partners of a converged
  root without a label - that needs the operation that makes one partner
  from another (S+-, a rotation), applied to R3.

**Vectors and memory** (`eom.plan_for`, `eom.Plan`).  The operands of the
ground state and of a sigma vector are in memory: the pp operator twice,
seven arrays of the size of `<ia||bc>` (f = o v^3: the dressed block, its
packed form, the bra-dressed `<ab||ei>`, W_vvvo, the operand of pass 1, and
3.5 f of transients), T3, the batch buffers, 22 arrays of o^2 v^2
(`eom.fixed_bytes`).  The 2 max_space vectors of the subspace are in memory
if they fit in `max_memory` next to that; else their rank-3 parts are
mapped arrays of a scratch file (counted against `max_disk`), which the
passes and the vector operations take as they take a mapped T3; `max_space`
is lowered, to 2 nroots + 2 at least, to what memory or disk hold.
`MemoryError` with the numbers if the fixed part does not fit (the sliced
pp operator and the mapped T3 of the ground-state plan are not used here:
a calculation whose CCSDT needed them cannot run EOM yet);
`DiskFullError` if the smallest subspace fits nowhere.  What numpy
allocates in a run is checked against the plan by `tracemalloc`
(`tests/test_eom.py`: between 50% and 100% of it for 8 + 38 spin
orbitals); the resident memory of the production cases against the
plan is in section 8.

*In the blocks of the symmetry labels* the fixed part is counted with the
sizes of the blocks: T3, the pp operator and its derivative (with a shift:
the rectangular blocks between the pair blocks h and h + s), the work
array of the passes (`SpinBlockT3.buffer_bytes`, and for a shift
`ShiftT3.buffer_bytes`: every pass sizes its batch to two thirds of
`mem_mb` by what one occupied pair or triple takes in that layout), the
blocks of the occupied operator of a shift; the o v^3 arrays are dense
there too.  The amplitude step of `prepare` is counted with at most two
blocks of T3 for the blocks of its occupied operator, where the plan of
the ground state sets aside a quarter of the buffer budget (500 MB by
default) - that reserve was most of what the first version planned above
the use for HBr (1.62 GB planned, 0.93 GB used).  A vector of a shift is
o v + C(o,2) C(v,2) + (the blocks of the shift) numbers.
`tests/test_eom_shift.py`: HBr / cc-pVDZ-DK, 8 + 28, two shifts - what
numpy allocates (the work array of the engine included) is between a
third and all of the plan less the allowance for the threads.

**Vector operations** are threaded (`csrc/t3pack.c`): dot products with
all vectors of the subspace in one pass (`zc_vmdot`: the new column and
row of the projected matrix, the orthogonalization), linear combinations
(`zc_vlincomb`: residuals, corrections), the collapse of the subspace in
place in blocks of 256 elements of all vectors (`zc_vtransform`), and the
division by the denominators block by block (`zc_vprecond`).  The first
version did the last two and the columns of the projected matrix with
numpy on one thread.

## 5. Ionized and attached states

R = 1h + 2h1p + 3h2p (IP), 1p + 2p1h + 3p2h (EA): r[i], r[ij,a],
r[ijk,ab] and r[a], r[i,ab], r[ij,abc].  They are the EE problem with one
more spinor x that no integral touches and whose Fock element is zero - a
virtual one that R always excites into (IP: r[i,x], r[ij,ax], r[ijk,abx]),
an occupied one that R always excites from (EA).  H does not change the
occupation of x, so the EE sigma vector maps the vectors that have x once
to themselves, and among them it is the IP (EA) sigma vector; the
eigenvalues are the ionization energies (EA: E(N+1) - E(N)).

- Reference: `eom_ref.ghost`, `eom_ref.eigenvalues(kind='ip' | 'ea')`,
  checked against the operators `a+ b+ k j i` in a Fock space and against
  PySCF's IP- and EA-EOM-CCSD.
- **The ghost route at full length** (`eom.ghost`; `eom_ip(short=False)`):
  the integral container and the amplitudes padded, the guess taken from
  the singles with x, everything else the EE code on the packed block.
  The vectors have the length of the EE problem with one spinor more, of
  which C(o,3) C(v,2) elements are not zero (IP), and a sigma vector costs
  what an EE one costs.  Kept as the reference of the short vectors.
- **Short vectors** (`eom.ghost_short`; the default of `eom_ip`, `eom_ea`).
  "x once" is a label: give x a symmetry label of its own and the vectors
  with x once are exactly the vectors of one label shift (section 6) of the
  padded problem, and the machinery of the label blocks stores and passes
  over nothing else.
  - *Without symmetry labels* (the packed layout): every spinor gets the
    label 0 and the ghost the label 1 (EA: -1).  T3 has the one block it
    had - the flat array is passed as it is -, and R3 is the block
    [occupied triples, virtual triples with x] = [C(o,3), C(v,2)] (EA:
    [C(o,2), C(v,3)]): the short vector.
  - *With the labels of a linear molecule* (n classes of 2 m_j): the ghost
    is 3 (n - 1) + 1 classes beyond the last one (IP) or before the first
    (EA).  Then no element without the ghost has the shift of an element
    with it, T3 keeps its blocks (the same flat array), and the vectors
    with x once fall into blocks by the label of the electron that is
    removed or attached: the change of the sum of the labels of the state,
    minus the label of the spinor that is emptied (IP), the label of the
    spinor that is filled (EA).  Each label is an eigenproblem of its own
    (`eom_ip(shift=D)`, a list, or `'all'`); the states of +D and -D of a
    closed shell are each other's time-reversed images and only one of
    them is computed (section 6).
  - *What a sigma vector costs.*  `residual(R3; W)` - the passes over R3
    with the operators of the ground state - loses a virtual (occupied)
    index everywhere: the gathers, products and reductions run over the
    blocks of the short layout only.  `residual(T3; W')` still gathers T3
    (pass 2 once, the pp ladder once, pass 1 once), but its products are
    small: of the rows (a i) of the ring operator only (x i) is wanted -
    o instead of o v rows -, of the rows (a<b) of the pp operator only
    (a x), v instead of C(v,2).  The classes without spinors between the
    labels of the molecule and that of the ghost cost loops that find
    nothing.  Measured: section 8.
  - `Short.strip(res, k)` takes the ghost index out of a root: r[i],
    r[(i<j), a], r3 (IP).

## 6. Symmetry labels

With the labels of a linear molecule in use (2 m_j; `docs/usage.md`) T3 is
stored in the blocks of occupied and virtual triples with equal sums of
labels (`t3engine.T3Layout`).  An excitation operator R changes the sum of
the labels of the spinors by a fixed amount D, **the label shift** of the
state:

    r1[i,a]        where l_a - l_i = D,
    r2[ij,ab]      where l_a + l_b - l_i - l_j = D,
    r3[ijk,abc]    where l_a + l_b + l_c - l_i - l_j - l_k = D,

and the sigma vector maps the vectors of one shift to themselves: every
shift is an eigenproblem of its own.  D is counted in the labels of the
run, as the output lists them; nothing is assumed about their meaning (for
2 m_j, D = 2 (Omega' - Omega) of the two states; the labels are only
added).  D = 0: the totally symmetric states, with the block structure of
T.  The front end: `eom_ee(shift=D)`, a list of shifts, or `'all'`.

**Totally symmetric states** (`shift` None or 0): every operand of section
3 has the block structure of its ground-state counterpart and the code
runs unchanged in the blocks of the engine; the guess takes the singles
with equal labels.

**States with a shift** (`zccsdt.cc.t3shift`).

1. *The layout* (`ShiftLayout`): the occupied triples of group g (the sum
   of their classes; a class is a label less the smallest, divided by the
   step) with the virtual triples of group g + s, s the shift in classes.
   The blocks of all shifts together are the packed block (LiH / 6-31G:
   14 + 112 + 368 + 704 + 868 + 704 + 368 + 112 + 14 = 3264).  r1 and r2
   stay dense arrays (o v, C(o,2) C(v,2): small next to r3) with zeros
   outside their shift; the sigma vector is masked to the shift, so the
   rounding of the integrals (1e-12 of them is outside the labels after an
   SCF converged to 1e-12) cannot leak into other shifts.
2. *Every array has a shift*: (the labels of its creators) - (those of its
   annihilators); amplitudes virtual minus occupied, `<pq||rs>` bra minus
   ket.  A product has the sum of the shifts of its factors.  The operands
   of the ground state have shift 0; r1, r2, R3 and with them H' and all
   the derivatives W' have the shift s.
3. *The passes* (`ShiftT3.pass1`, `ShiftT3.residual`) are those of the
   spin-blocked engine with shifts as arguments:

       pass1(T, sT; integrals of shift sG)       outputs of shift sT + sG
       residual(T, sT; ring, pp and hh operators of shift sW;
                drive: t2 of shift sa, W_vvvo and W_vooo of shift sT + sW - sa)
                                                  result of shift sT + sW

   A sigma vector is `residual(R3, s; W of shift 0; t2 with W' of shift s)`
   plus `residual(T3, 0; W' of shift s; r2 of shift s with W)`, and
   `pass1(R3, s; H~)`, `pass1(T3, 0; H' of shift s)`.  In the code: where
   the engine takes the class of an index from the classes of the others
   (the pair group of the gathered `T[P, (d m), Q]`: g_Q = g_P + s_m - s_d
   + sT; the classes of e and m in the drive terms; the two conditions of
   the rank-1 and rank-2 outputs of pass 1, which coincide without a
   shift), the shift enters; the result is reduced into the blocks of the
   output layout.  **No kernel was written**: the gathers and reductions of
   `csrc/t3pack.c` are driven by tables (the row of a block per occupied
   triple, the column per virtual triple), and only the tables know the
   layout.  A block of the product whose operand does not exist (its
   group is out of range) is set to zero, since the reductions read it.
4. *Operators with a shift couple different blocks.*  `W'_pp` is a list of
   rectangular blocks, (pairs of block h + s) x (pairs of block h), built
   as the square ones are (`-P(ab) r1 <mb||de>~`, `r2 <mn||de>`, and the
   antisymmetrized F'_vv between two blocks: `t3shift.one_body_pairs`).
   `W'_ring[(a i), (d m)]` is taken between the classes delta = s_d - s_m
   and delta + s.  The occupied operator `O'[I, J]` (hh ladder and F'_oo)
   couples the triples of group g with those of g + s: its blocks are cut
   from the matrix on the range of triples that holds both groups
   (`t3pack.occ_operator_direct`), one GEMM each - the pair-driven form of
   the hh ladder is for the square blocks only.
5. *The parts of rank one and two* of a sigma vector are dense GEMMs that
   do not look at labels; only the loop over the pair blocks of the pp
   operator knows the shift.
6. *The guess*: unit vectors on the lowest singles with l_a - l_i = D, the
   singles-doubles block of that shift.  A shift without singles has no
   guess (`ValueError`); `eom.label_shifts` lists those that have.
7. *Vectors, preconditioner, plan*: the Davidson solver takes vectors of
   any length; the denominators are those of the blocks of the layout;
   the plan counts the vectors of the shift and, in label blocks, what the
   blocks take (section 4).

**+D and -D.**  If the reference is symmetric under time reversal - a
closed shell: every spinor of label l has a partner of label -l with its
energy, which `eom.time_reversal_symmetric` tests - the states of -D are
the time-reversed states of +D with the same energies.  Given a list that
has both, the front end computes the first of each pair and hands the
other its energies (a `Result` without vectors, `partner_of` set);
`partners=True` computes both.  `'all'` orders the shifts by |D|, +D
first.  For an open-shell reference every shift is computed.

**Labels with a modulus** (the fermion irreps of C2 and its like,
`eris.modulus`): the T3 engine has no blocks for them
(`t3pack.classes_of` returns None and CCSDT runs on the packed block), so
there is no shifted layout either.  On the packed block the states of all
labels are roots.

**On the packed block** (`cc.symmetry = False`, or spin orbitals) the
states of one shift can be selected without blocks, by masks: section 4,
"sectors".

## 7. Validation

**The reference** (`tests/test_eom_ref.py`, 30 s).

| Check | Agreement |
|---|---|
| sigma by the differentiated term table, by the five-point formula and by three evaluations in dressed integrals; complex amplitudes that solve nothing, 3 + 4 spin orbitals; two and three ranks | 1e-12 |
| the same against `<mu\| [Hbar, R] \|0>` of second-quantized operators in the Fock space (`tests/fockspace.py`) | 1e-12 |
| `[H~, R1]` by the one-index transformation against the derivative of the dressing | 1e-13 |
| IP and EA sigma through the ghost orbital against the operators `a+ b+ k j i`, `a+ b+ c+ k j` in the Fock space | 1e-12 |
| three electrons, where CCSDT is exact (Li / STO-3G, 10 spin orbitals; X2C spinors mixed by complex unitary matrices): all 119 eigenvalues of the EOM-CCSDT matrix against the differences of the eigenvalues of H among the 120 determinants | 1e-9 |
| EOM-EE-, IP-, EA-CCSD against PySCF (`pyscf.cc.eom_gccsd`), H2O / STO-3G, 10 + 4 spin orbitals, four roots each; with complex spin orbitals (every pair mixed by an SU(2) matrix) the same eigenvalues from PySCF's complex amplitudes | 1e-8 |

(PySCF's own EOM solver was not used with the complex orbitals: its EA
guess gave four roots of 1e-16 there, and its real-eigenvalue filter
stopped the EE and IP runs in some orders of the calls.)

**Packed amplitudes against the reference** (`tests/test_eom_opt.py`,
`tests/test_eom.py`).

| Check | Agreement |
|---|---|
| sigma for random vectors at random amplitudes: LiH / STO-3G (4 + 8), H2O / STO-3G with the 1s pair frozen (8 + 4); real and complex; with and without triples | 1e-12 of the largest element (observed 1e-14 to 1e-15) |
| the six lowest roots of LiH (EOM-CCSD, EOM-CCSDT; real, complex) against all eigenvalues of the dense matrix (424 x 424) | 1e-10 (observed 2e-12 to 5e-12) |
| the sigma vector in label blocks against the packed block (LiH / 6-31G, X2C, 4 + 18) | 1e-11 |
| roots in label blocks are roots on the packed block | 1e-8 (observed 1e-10; threshold of the residual 1e-8) |
| the vectors of the subspace in a scratch file, a subspace of 11 instead of 20 vectors | the same roots (1e-9, 1e-8) |
| IP and EA of EOM-CCSD and EOM-CCSDT by the ghost spinor against the dense IP / EA matrices (LiH; real with the packed `<ab\|\|cd>`, complex) and PySCF's roots (H2O / STO-3G) | 1e-9, 1e-8 |

**States with a label shift** (`tests/test_eom_shift.py`).

| Check | Agreement |
|---|---|
| the blocks of all shifts are the packed block, each element once; shift 0 is the layout of T3 | exact |
| sigma of a random vector of every shift (-4 .. 4 classes: 14 to 868 of 3264 triples; LiH / 6-31G, X2C, 4 + 18) in the blocks against the packed block; also the singles-doubles block; shift 0 by the shifted passes against the engine | 1e-10 of the largest element (observed 4e-12: the SCF is converged to 1e-12 and so are the zeros of the integrals); 1e-12 |
| the same with a phase on every spinor (complex integrals and amplitudes with the labels they had): sigma carries the phases | 1e-11 |
| LiH: the roots of the shifts 0, +-2, +-4 are roots of the packed block (`symmetry = False`), and below 0.2 Eh their union is its spectrum with multiplicities - 12 roots; the lowest level, a triplet, is one root each of the shifts -2, 0, 2 | 1e-10 (observed 2e-11; threshold of the residual 1e-9) |
| HBr / cc-pVDZ-DK, 28 frozen, 8 + 16: the eight lowest roots of the packed block are the lowest of the shifts +-4 (0.2475010526, the state MRCC finds first: a root with no single excitation of shift 0), +-2 (0.2520595925, 0.2769384419) and 0 (0.2601782223, 0.2607020264) | 1e-10 (observed 2e-11) |
| the same roots with a phase on every spinor (complex arithmetic in the blocks) | 1e-10 |
| +D against -D, computed separately (`partners=True`) | 1e-10 (observed 1e-11) |
| a shift on the packed block by masks (`symmetry = False`): the roots of the blocks | 1e-10 |
| EOM-CCSD with a shift against the packed block | 1e-10 |
| the plan of a shift and what a run allocates (HBr, 8 + 28) | between 1/3 and 1 of the plan |

**IP and EA with short vectors** (`tests/test_eom_short.py`).

| Check | Agreement |
|---|---|
| LiH / STO-3G, CCSD and CCSDT, real and complex: four roots of IP and EA against the ghost route at full length (the same number of sigma vectors) and against the dense IP / EA matrices of the reference | 1e-10 (observed 1e-15); 1e-9 (observed 3e-11) |
| the stripped vector (ghost index taken out) has all of the norm | 1e-12 |
| H2O / STO-3G, EOM-IP- and EA-CCSD against PySCF | 1e-8 |
| LiH / 6-31G, X2C, front end: short vectors on the packed layout against the ghost route; by the label of the electron in the blocks of the labels (`shift='all'`), the union of the roots against the packed layout; +-label | 1e-10 (observed 1e-11) |
| HBr / cc-pVDZ-DK, 8 + 16: the eight valence ionization energies by the labels -+1 (three each) and -+3 (one each) against the packed layout | 1e-10 |

**The solver and the sectors** (`tests/test_eom_solver.py`): the threaded
collapse, preconditioner and multiple dot products against numpy (1e-12);
a model matrix with a threefold and a twofold level; H2O / 6-31G, spin
orbitals: the two lowest triplets of MRCC as the two lowest roots of the
shift 2 of the spin label, the lowest triplet and singlet in shift 0
(1e-9).

**MRCC** (LR-CC, the same eigenvalues; release of February 2018, `cctol=11`;
energies with 12 decimals; `bench/refs/reference_energies.json`, `eom_ee`).

H2O / 6-31G, MRCC's own RHF and integrals (`nsing=4`, `ntrip=3`), zccsdt on
PySCF's integrals of the same basis and geometry, 10 + 16 spin orbitals
(`python`: threshold 1e-8 of the residual; 16 roots, 302 sigma vectors):

| State | MRCC EOM-CCSDT / Eh | zccsdt | Difference |
|---|---|---|---|
| triplet 1 | 0.284831364143 | 0.2848313643 (three roots) | 2e-10 |
| singlet 1 | 0.311687769200 | 0.3116877693 | 1e-10 |
| triplet 2 | 0.366400122961 | 0.3664001230 (three roots) | < 1e-10 |
| triplet 3 | 0.375682797356 | 0.3756827973 (three roots) | < 1e-10 |
| singlet 2 | 0.394170561703 | 0.3941705618 | 1e-10 |
| singlet 3 | 0.404172632463 | 0.4041726323 | 2e-10 |

(The differences are the threshold of the residual.)  In the tests: the
lowest triplet and singlet of EOM-CCSDT with real and with complex spin
orbitals, and EOM-CCSD with the 1s pair frozen, 12 roots against the six
states of MRCC (0.281140765433, 0.308222703511, ..), to 1e-9.

**Relativistic, on identical integrals.**  HBr / cc-pVDZ-DK, X2C spinors of
PySCF in Kramers pairs, 28 frozen, 8 + 16 active, written as MRCONEE and
MDCINT by `zccsdt.dirac`; MRCC through its interface to DIRAC
(`iface=dirac`, `nstate=4`: `bench/dirac/mrcc_eom.sh`) - its relativistic
code does excited states there, which gives EOM-CCSDT on spinors an
independent reference:

| | MRCC / Eh | zccsdt | Difference |
|---|---|---|---|
| EOM-CCSDT, state 1 and 2 (degenerate) | 0.247501052593, ..595 | 0.2475010526 | 2e-11 |
| EOM-CCSDT, state 3 | 0.260178222291 | 0.2601782223 | 2e-11 |
| EOM-CCSD, state 1 and 2 | 0.247699688957, ..959 | (test) | < 1e-9 |
| EOM-CCSD, state 3 | 0.261343263658 | (test) | < 1e-9 |

MRCC looks for its states one after the other from its own guesses; zccsdt
has a degenerate pair at 0.2520595925 between them, and 0.2607020264 just
above.  MRCC takes real integrals only: complex spinors are covered by the
complex spin orbitals above and by CFOUR below.

**CFOUR's `xx2ceomcc`, on CFOUR's spinors** (complex; EOM-CCSD;
`bench/cfour/eom_ccsd.sh`).  Two runs of CFOUR per molecule, since
`xx2ceomcc` wants `ABCDTYPE=AOBASIS` and the `<ab||cd>` file that
`zccsdt.cfour.from_cfour` reads is written with `ABCDTYPE=STANDARD` only:
the full chain for CFOUR's roots, the chain up to `xx2cvtran` for the
integrals, CCSD and EOM-CCSD of zccsdt on them (the excitation energies do
not depend on how the two SCF runs mix a Kramers pair).  `ESTATE_CONV=8`
(the largest element of CFOUR's residual below 1e-8), zccsdt 1e-9 (norm).

| | CCSD correlation energy: CFOUR, difference | Roots of `xx2ceomcc` / Eh | Difference to the nearest root of zccsdt | Cycles of CFOUR (one sigma vector each) | zccsdt |
|---|---|---|---|---|---|
| H2O / 6-31G, X2C, 10 + 16 spinors | -0.135423711709, 1e-11 | 0.280687616382, 0.280688286535 (`%eomalg 0`, four roots asked: each twice); 0.280687329835, 0.280687616614 (`%eomalg 2`, two roots) | 4e-12, 5e-12; 5e-11, 2e-10 | 51, 50, 50, 51; 52, 54 | six roots (0.2806873298, 0.2806876164, 0.2806882865: the triplet split by 1e-6; 0.3076954869; ..) in 129 sigma vectors |
| HBr / cc-pVDZ-DK, X2C, 14 pairs frozen, 8 + 28 spinors | -0.140627327312, 9e-12 | 0.234048257246, 0.238083258259 (each twice) | 1e-10, 5e-11 | 37, 39, 39, 37 | six roots (0.2340482571 twice, 0.2380832583 twice, 0.2445417703, 0.2450180764) in 113 sigma vectors |

The numbers of iterations are not comparable and were not meant to be:
CFOUR adds one vector per cycle to the subspace of one root and converges
on the largest element; with its default algorithm every root has a
subspace of its own and follows its own unit vector, so two of four roots
came out twice (section 9).

## 8. Timing

Job 31825453, 8 cores and 30 GB of c105 (2 x Xeon Gold 6248R; the shared
node - other work ran on its other cores, among it test runs of this
development on 2 to 4 threads), `bench/slurm_eom.sh`, commit 190ddad, MRCC
first, then zccsdt, scratch on the local disk.  MRCC: its release binary
(February 2018; OpenMP, 8 threads, 24 GB allowed).  Thresholds: zccsdt
1e-6 in the norm of the residual, MRCC `cctol=8` (H2O) and 11 (HBr).
Measured once.

**H2O / cc-pVTZ, all electrons** (10 + 106 spin orbitals; the basis of
CFOUR's GENBAS in both programs, SCF energies equal to 1e-12).  MRCC at its
best for a closed shell - its own RHF integrals, C2v, spin adaptation, 2.0
million determinants; the lowest singlet and the lowest triplet (`nsing=2`,
`ntrip=1`, `symm=B2`).  zccsdt: RHF as GHF spin orbitals, no symmetry of
any kind, T3 of 23.2 million elements, real; four roots = the triplet
(three roots) and the singlet.

| | zccsdt | MRCC |
|---|---|---|
| CCSDT iteration | 3.94 s (median of 17) | 5.6 s (17 iterations in 1.59 min) |
| sigma vector | 8.18 s = 2.08 CCSDT iterations (the triples stage by its 56 sigma vectors; 7.2 s in the sigma vectors, 0.9 s in the vector operations of the solver) | 8.1 s per LR iteration (18 in 2.43 min, for each of the two states; 0.11 min at the start, 0.16 at the end) |
| sigma vectors | 56 = 14 per root, 14 iterations, 2 collapses | 18 per state |
| excited states, time | 470 s (guess stage 7.5 s: 32 singles-doubles vectors of 0.2 s; prepare 3.8 s) | 291 s |
| omega / Eh: triplet, singlet | 0.2783768695, 0.2984329001 | 0.278376868212, 0.298432899788 |
| peak resident memory; plan | 12.44 GB; 12.57 GB (24 vectors and their sigma vectors in memory: 8.99 GB; fixed part 3.58 GB) | 24 GB allowed |

The energies differ by 1.3e-9 and 3e-10 at these thresholds.  Per sigma
vector the two programs are level, MRCC with the point group and the spin
adaptation and zccsdt with neither; zccsdt needs fewer vectors per root
but computes the triplet three times, so the excited states took 1.6
times MRCC's time.

A sigma vector of zccsdt (seconds, of 8.18): the residual with R3 and the
operators of the ground state 2.80 (pass 2 1.41, pp ladder 1.37); the
residual with T3 and the derivatives of the operators 2.82 (1.41, 1.39);
the two passes 1 1.04; W'_vvvo 0.26; `<ab||de>'` and the ladders of
sigma_2 0.21; the small terms 0.13; the solver's vector operations 0.89
(residual vectors and orthogonalization on threads, preconditioner and
collapse with numpy on one thread).  A CCSDT iteration (3.94): pass 2
1.46, pp ladder 1.41, pass 1 0.51, intermediates 0.25, DIIS 0.17, dressing
0.11.

**HBr / cc-pVTZ-DK, X2C spinors, 28 frozen** (8 + 78).  MRCC through its
interface to DIRAC on the MRCONEE / MDCINT written from PySCF's spinors
(`nstate=4`; no spin adaptation, the two fermion irreps of the files);
zccsdt through the front end on the same SCF, in the blocks of the
symmetry labels (the totally symmetric states) and on the packed block (all
states), real arithmetic.  CCSDT energies: -2605.165536741802 (MRCC),
-2605.16553674179 (zccsdt).

| | zccsdt, label blocks | zccsdt, packed block | MRCC |
|---|---|---|---|
| CCSDT iteration | 0.21 s | 0.61 s | 6.1 s (first six: 0.61 min) |
| sigma vector | 0.36 s = 1.75 iterations | 1.33 s = 2.19 iterations | about 10 s (91 LR iterations and 22 CCSDT iterations in 1063 s) |
| roots; sigma vectors | 3 symmetric; 43 = 14.3 per root, 2 collapses | 6; 107 = 17.8 per root, 4 collapses | 3 states; 25, 28, 38 iterations |
| excited states, time | 17 s | 150 s | about 930 s (1063 s with the ground state) |
| omega / Eh | 0.2404148156, 0.2408465051, 0.3332902955 | 0.2298116645 (twice), 0.2337441996 (twice), 0.2404148166, 0.2408465143 | 0.229811656650, 0.229811656668, 0.240414821269 |
| peak resident memory; plan | 0.93 GB; 1.62 GB | 3.79 GB; 4.23 GB | 24 GB allowed |

MRCC's three states are roots of zccsdt on the packed block to 8e-9 and
5e-9 (the residuals are 4e-7 to 9e-7; the same comparison with 8 + 16
spinors and a threshold of 1e-8 agrees to 2e-11, section 7).  The
symmetric root at 0.24041 comes out of the blocks and of the packed block
equal to 1e-9.  Per sigma vector zccsdt on the packed block takes an
eighth of MRCC's LR iteration, in the blocks a twenty-eighth - the ratio
of the CCSDT iterations (10 and 29).

In the blocks the plan is far above what was used (1.62 against 0.93 GB):
`fixed_bytes` counts the batch buffers and the o v^3 arrays of the dense
layout.  On the packed block and for H2O the peak is 90% and 99% of the
plan; nothing was above it.

(The numbers of this job are from the first version: 8 cores of the
shared node, shift 0 only in the blocks.)

**Label shifts and IP in label blocks, a node of its own.**

Job 31828491, node c603 (2 x Xeon Gold 6248R), 24 cores - socket 0 and its
memory, no other job on the node -, `bench/slurm_eom2.sh`, commit ae688c1,
scratch on the local disk, real arithmetic, T3 in the blocks of the labels
2 m_j, threshold 1e-6 of the residual, three roots per shift, two per
label of IP.  Measured once.  Of each pair +-D one shift was computed.

| | HBr / cc-pVTZ-DK, 28 frozen, 8 + 78 | HBr / cc-pVQZ-DK, 10 frozen, lowest 120 virtual spinors, 26 + 120 |
|---|---|---|
| T3 in blocks (packed block) | 0.62 million (4.26) | 81.9 million (730) |
| CCSDT iteration (median) | 0.205 s | 6.05 s |
| **EE, shift 0**: triples of a vector; sigma vector; per root | 622 138; 0.372 s = 1.82 iterations; 14.0 | 81 875 732; 11.06 s = 1.83; 15.3 |
| **EE, shift +-2** | 582 637; 0.365 s = 1.78; 14.3 | 78 744 914; 10.47 s = 1.73; 15.7 |
| **EE, shift +-4** | 478 234; 0.330 s = 1.61; 18.7 | 70 043 592; 9.71 s = 1.60; 17.7 |
| EE, time of a shift (guess stage in it) | 20.6, 18.3, 21.4 s (4.7, 2.4, 2.7 s) | 613, 596, 591 s (98, 98, 70 s) |
| lowest roots / Eh: shift 0 | 0.2404148147, 0.2408465048, 0.3332902852 | 0.2418705730, 0.2423160390, 0.3322661699 |
| shift +-2 | 0.2337441859, 0.2560105237, 0.3333697909 | 0.2349495523, 0.2559147563, 0.3323390838 |
| shift +-4 | 0.2298116664, 0.4236483143, 0.4989666136 | 0.2310729220, 0.3776418761, 0.4154702080 |
| **IP, label -+1** (short vectors): triples; sigma vector; per root | 27 713; 0.253 s = 1.23 iterations; 16.0 | 2 341 338; 3.72 s = 0.61; 17.0 |
| **IP, label -+3** | 23 294; 0.246 s = 1.20; 17.5 | 2 116 224; 3.63 s = 0.60; 19.5 |
| IP / Eh: label -+1; -+3 | 0.4317625609, 0.5730642677; 0.4200399244, 0.7022642982 | 0.4374237765, 0.5755092816; 0.4254934673, 0.7124140607 |
| memory, shift 0: growth of the resident memory in the run; plan | 0.37 GB; 2.12 GB (1.5 GB of it the allowance for 24 threads) | 31.3 GB; 36.2 GB (27 GB the 20 + 20 vectors) |
| shift +-2, +-4 | 0.20, 0.03 GB; 2.10, 2.07 GB | 28.4, 25.7 GB; 35.3, 32.5 GB |
| IP | 0.12, 0.35 GB; 1.92 GB | 4.9, 5.9 GB; 10.7, 10.6 GB |

The lowest state of HBr is in shift +-4 (0.22981; MRCC on the same
integrals: 0.229811656650), the next in +-2 - with shift 0 alone, which is
all the first version had in blocks, neither is found.  The second root of
IP label -+3 has no weight in the singles (a 2h1p state).

*Memory.*  The growth is below the plan in every run.  The resident memory
of the process is above it from the second eigenproblem on (38.98 GB at
the peak of shift 2 against a plan of 35.26 GB): every `Result` keeps the
operands of its ground state (`res.ground`, 6 GB here), and the front end
keeps the last result, so a new problem starts 6 GB higher until the old
result is dropped.  Not fixed.

*A sigma vector, 26 + 120, shift 0* (seconds of 11.06; of the same run):
the residual with R3 2.74 (pass 2 1.90, pp ladder 0.77), the residual with
T3 2.73 (1.90, 0.78), the two passes 1 1.82, W'_vvvo 0.86, the dense small terms
with `<ab||de>'` and the ladders of sigma_2 1.4, the solver's vector
operations 1.55.  The guess stage is 12 to 16% of the time
of a shift: 52 to 73 singles-doubles vectors of 1.35 s (dense GEMMs on
o^2 v^2 and o v^3 arrays that do not use the labels).

*IP.*  With 26 + 120 a sigma vector is 0.6 iterations: the passes over the
short R3 0.21 s, the residual with T3 0.58 s (it gathers T3, its products
are small), pass 1 0.47 s - and 2.2 s of dense terms of the size of the
EE problem (small terms, `<ab||de>'`, W'_vvvo), which do not know that
only the elements with the ghost are wanted.  With 8 + 78, where a CCSDT
iteration is 0.2 s of mostly such terms, an IP sigma vector is 1.2
iterations.  The vectors are 35 times shorter than those of EE (26 + 120).

*The vector operations of the solver* (8 + 78, the same job, the three
shifts): 2.34, 1.93, 2.40 s with numpy on one thread for collapse,
preconditioner and projected matrix (`ZCCSDT_EOM_NUMPY=1`), 0.94, 0.78,
1.00 s with the threaded kernels; a sigma vector 0.390, 0.383, 0.365 s
against 0.372, 0.365, 0.330 s.  With 26 + 120 they are 1.55 s of 11.06
per sigma vector with the kernels: every residual and every
orthogonalization reads the whole subspace (up to 40 vectors of 0.65 GB)
once per root, which is bound by the memory bandwidth.  Forming the
residuals of all roots in one pass would cut that by the number of roots;
not done.

What was not measured: a run with the vectors in the scratch file at
production size (the tests do it at 8 + 16), complex arithmetic at
production size, EA at production size, the packed block and MRCC on
this node.

## 9. CFOUR's spinor EOM-CCSD: the solver

A study of the sources of `xx2ceomcc` (`~/apps/cfour-x2cso-rebuild/x2ceomcc`,
helpers in `libx2ccc`), for the port of EOM-CCSDT to the native program:
what the solver does, what it reads, and which of its routines take a
vector of any length.  The CIS guess was left aside.  Routine `foo` is in
`x2ceomcc/foo.f90` unless a directory is given.  Statements about behaviour
that were only read, not run, are marked.

**Running it.**  `xjoda; [xx2camf;] xsox2catm; xsox2chf; xx2cvtran;
xx2cccsd; xx2chbar; xx2ceomcc` - not chained by `xcfour`; no Lambda program
is needed for energies.  It reads the SCF files of `xsox2chf`, the integral
files of `xx2cvtran` (`moinhhhh`, `moinphhp`, `moinpphh`, `moinhhpp`,
`moinhhhp`, `PPPHnnnn`), `t1guess` / `t2guess` and `TOTENERG` of
`xx2cccsd`, and the files of `xx2chbar` (`hbarftia`, `hbarftii`,
`hbarftaa`, `hbarphhp`, `hbarhhhh`, `hbarhhhp`, `hbahhhp2`, `WPH1nnnn`).
`ABCDTYPE=AOBASIS` is mandatory (`ccsdgs` stops otherwise); the AO
integrals are held in memory.  A file `l1guess` left by a Lambda run makes
every converged root compute a density besides.

**Input** (`setsocc`: JOBARC flags and private `%sections` of the ZMAT - a
line with the tag, the values on the next line):

| What | Keyword | Default |
|---|---|---|
| number of roots | `%eomnroot` | 1 |
| iterations | `ESTATE_MAXCYC` | 40 (per root with algorithm 0, in all with 1 and 2) |
| threshold | `ESTATE_CONV=N` -> 1e-N; `%eomthrs` (a real number) overrides | 1e-5 |
| largest subspace | `%eomndav0` | `ESTATE_MAXCYC` |
| algorithm | `%eomalg`: 0 `eomr`, 1 `eomene`, 2 `eomene2` | 0 |
| root selection (algorithm 0) | `EOMFOLLOW=ENERGY` or `OVERLAP` | ENERGY |
| guess given | `%eomroot` (per root: unit vectors on singles with coefficients); `mreomwfr` from an earlier run | - |
| restart | `%irestart` (pairs of vector files to reuse) | 0 |
| core-valence separation | `%cvsinf`, `%cvs2inf` | off |
| IP / EA | `CONTINUUM=VIRTUAL` / `OCCUPIED`: masks on the EE vector for excitations into continuum virtual spinors / out of occupied spinors with zero energy | NONE |
| left vectors, properties, triples correction | `DERIV_LEV`, `ESTATE_PROP=EXPECTATION`, `EOM_NONIT` | off |

`EXCITE`, `EOM_NSTATES`, `ESTATE_SYM` are not consulted.  IP and EA are not
separate programs: the EE program with a continuum orbital, as in section
5 here.  No example input exists in the tree.

**Algorithm.**  One new vector per iteration - a single-vector Davidson,
never a block.  One orthonormal basis (right vectors; left ones in a
second solve through `ccsdhcl`), Galerkin projection (`eomsubmoc`), the
projected matrix diagonalized by `zgeev` (`eomgdiag`; `eomeiggdrv` is its
wrapper - `eomeighdrv` is `zheev` for the CIS, `eomeigdrv` a non-LAPACK
solver without a caller), the eigenpairs sorted by their real parts.

- `eomr` (algorithm 0): the roots one after the other, each in a new
  subspace from its own guess, no deflation against the roots found.  With
  `EOMFOLLOW=ENERGY` every root takes the lowest eigenvalue, so guesses of
  one symmetry end in the same state; with `OVERLAP` the root with the
  largest component on the first subspace vector, the guess (`eompkvec`).
  Run here: four roots of H2O gave two states twice, with OVERLAP.
- `eomene` (1): one growing subspace for all roots, started from the first
  guess only; when a root has converged its subspace coefficients are
  kept, the eigenvectors with the largest overlap with the kept ones are
  skipped, and the lowest of the rest is the next target (`eompkvec3`).
  No compaction.
- `eomene2` (2): the same with compaction: at 20 + nroot + 2 vectors
  (hard-wired) the subspace is replaced by the Ritz vectors of the
  converged roots and the lowest others, nroot + 2 in all, C and HC
  transformed alike and written to new files (`eomcompact`).  Run here: two
  roots of H2O in 52 and 54 cycles.

**Guess** (not studied further): unit vectors on the singles with the
smallest `e_a - e_i`, one per root (`ccsdgs`), or what `%eomroot` gives.

**Preconditioner** (`ccsddenomao`, `eomnewc2oc`): singles with the diagonal
of Hbar, `F_aa - F_ii + W_aiai`; doubles with the differences of the
orbital energies; the correction is `r / (D - omega)`, zero where
`|D - omega| < 1e-4`.

**Orthogonalization** (`eomothgoc`): Gram-Schmidt over all stored vectors,
forward and backward, normalizing in between; a warning if an overlap is
above 1e-9 afterwards; no test for a vanishing vector.

**Convergence** (`eomconv`): the largest element of the residual of the
normalized vector at most the threshold; no test on the energy.  It prints
per cycle `EOM cycle`, the cycle, the size of the subspace, the real part
of the eigenvalue, the largest element of the residual and the threshold;
per root `root .. converged in .. cycles`, the ten largest elements of the
vector (`libx2ccc/eomprtwf`), `eigenvalue in a.u.` and `eomccsd eigenvalue
in eV` with real and imaginary part.  Files of the result: `mreomwfr`,
`mreomeer`; JOBARC `TOTENER2`.

**Compaction with algorithm 0** (`eomcptoc`, by reading): at `%eomndav0`
vectors the subspace collapses to the current Ritz vector and its sigma
vector.  After that the 1 x 1 projected matrix appears to be set from the
new vector and the sigma vector of the last old one (`cissubm` on the work
arrays): a subspace smaller than `ESTATE_MAXCYC` should not be trusted with
algorithm 0 until that has been checked by a run.

**Storage.**  One sequential unformatted file per vector, one record of
the whole vector: `EOMCnnnn` (vectors), `EOMHnnnn` (sigma vectors), names
from `libx2ccc/davhcfname` (at most 9999); `hcread` / `hcwrt` open, read or
write and close the file at every access.  Six vectors are in memory, and
the converged roots; an iteration reads the whole subspace about seven
times (matrix, Ritz vector and residual, orthogonalization).  The vector
(`libx2ccc/eomindexnosym`): singles `a + (i-1) nv`, then doubles on
triangles *with* their diagonals i = j, a = b, which a mask (`psmasksd`)
sets to zero before and after every sigma vector.

**Which routines take a vector of any length.**

| Routine | Does | Generic in the length? |
|---|---|---|
| `eomr`, `eomene`, `eomene2`, `eomr2` | the drivers | no: offsets of the work array of `soccmem`, the sigma routine and the masks called directly |
| `ccsdhc`, `ccsdhcl` | sigma vector, right / left: work(davc0) -> work(davhc0), no arguments | no (it is the EOM-CCSD) |
| `ccsdgs`, `soreadgs1` | guess | no (singles-doubles length in the declarations) |
| `ccsddenomao` | diagonal | no |
| `psmasksd`, `cvsmasksd`, `conmasksd`, `convirmaskeom` | projectors | no |
| `eomcpchcoc`, `hcread`, `hcwrt`, `libx2ccc/davhcfname` | vectors to and from their files | yes |
| `eomsubmoc` | new row and column of the projected matrix | yes |
| `eommatcp`, `eomgdiag`, `eomreorder`, `eompkvec`, `eompkvec2`, `eompkvec3`, `updatecroot`, `eomcompactpkvec` | the projected problem and the choice of the root | yes (they see the subspace only; `eomgdiag` takes offsets of the work array) |
| `eomnewroc`, `eomcompactnewroc` | Ritz vector and residual from the files | yes |
| `eomconv` | test and print | yes |
| `eomnewc2oc`, `eomothgoc` | precondition with a given diagonal, orthogonalize against the files | yes |
| `eomcptoc`, `eomcompact`, `restartcheck` | collapse, compaction, restart | yes in the algebra; bound to the work offsets |
| `cpeomv`, `cpgsc0`, `norm`, `norml`, `bothg2` | copy, normalize, biorthogonalize | yes |
| `libx2ccc/eomprtwf`, `eomprtwfd` | print the largest elements | no (decodes the singles-doubles index) |

So the file-based kernel of the solver - store, project, diagonalize,
pick, residual, test, precondition, orthogonalize, compact - takes the
length as an argument and can be called with vectors of singles + doubles
+ triples.  What the native program has to supply: a driver in the place
of `eomr` / `eomene2` (about 60 lines, with the work offsets sized for the
longer vector as `eomccini` sizes them); a sigma routine with the implicit
interface "vector in work(davc0), result in work(davhc0)"; the diagonal of
that length; a guess of that length (zero in the triples); its own masks
and a printing routine.  Two things to decide in the port rather than
inherit: the vectors of EOM-CCSDT are too long to be read seven times per
iteration from sequential files (the packed T3 kernels want them mapped or
in memory, as here), and the default algorithm does not keep the roots
apart; `eomene2` with its compaction is the one to start from.  The
complex arithmetic and the layout of singles and doubles with their zero
diagonals are CFOUR's; the native CCSDT has its own packed layouts.

The Python solver of section 4 was not modelled on this one: it is a block
method with one subspace for all roots.

**The port** (docs/native.md, section 16.3): the native program has this
solver in the design of `eomene2` - one subspace for all roots of a label
shift, one correction vector per unconverged root and iteration, the roots
verified at the end - with `eomreorder` and `eomconv` linked in complex
arithmetic and everything that touches a vector written for its own
layouts; the subspace is in the work array if it fits and in the files
`EOMCnnnn` / `EOMHnnnn` otherwise.  Its EOM-CCSD on the amplitudes of
`xx2cccsd` reproduces the roots of `xx2ceomcc` to 1e-10.

## 10. Not done

- (Left-hand vectors, r0, transition moments and the densities of excited
  states of EOM-EE-CCSDT: done since, sections 12 to 14; what is not done
  of them is in 12.5, 13.6 and 14.4.)  Excited-state gradients.
- The pp ladder of the second residual by stored intermediates (section
  3): a few percent in label blocks, not implemented; the ring term
  cannot be reduced.
- IP / EA: the dense terms of a sigma vector are those of the EE problem
  (section 8); a label for IP / EA on the packed block (sectors by masks
  are wired for EE only).
- Label shifts for labels with a modulus (CCSDT has no blocks for them).
- Singlets apart from the M_S = 0 components of triplets; partners of a
  degenerate level without a label; a state without a single excitation
  in its sector needs a larger `extra` (section 4).
- A `Result` keeps the operands of its ground state: successive
  eigenproblems of one object add them up until the result is dropped.
- The residuals of all roots in one pass over the subspace (section 8).
- EOM when the ground state needed the sliced pp operator or a mapped T3:
  `prepare` keeps the operands in memory.  U / R-form integrals (UCCSDT,
  RCCSDT) are not served: `capture` is for dense integrals.
- The small terms of a sigma vector are einsum and GEMMs on dense o^2 v^2
  arrays without the labels (15% of a sigma vector and all of the guess
  stage with 26 + 120).
- Core-valence separation, an energy window: the solver takes the lowest
  roots of a shift (or follows a guess).
- In the native Fortran program: IP / EA, the block solver of section 4
  (it has the solver of section 9: docs/native.md, section 16.3).

## 11. Double electron attachment (EOM-DEA)

States with two electrons more than a closed-shell reference: the three
components of a triplet, and the singlets beside it, come from one
determinant on the same footing (O2 from O2(2+); the application below is
the zero-field splitting of X 3Sigma_g-).

| What | Where |
|---|---|
| the route through two ghost spinors, packed layout and label blocks | `src/zccsdt/cc/dea.py` (`Problem`, `ProblemSym`, `kernel`), `tests/test_dea.py` |
| the sector itself in label blocks, no ghosts (the default in label blocks: R to 4p2h, T of CCSD, CCSDT, CCSDTQ; 11.9) | `src/zccsdt/cc/dea_sector.py`, the surviving terms `src/zccsdt/cc/dea_ref.py`, `tests/test_dea_sector.py` |
| the 5p3h operators at lowest order, folded | `src/zccsdt/cc/dea5.py`, `tests/test_dea5.py` |
| any rank on determinants (reference route) | `src/zccsdt/cc/dea_strings.py`, `tests/test_dea_strings.py` |
| front end | `GCCSDT.eom_dea` (`docs/usage.md`) |
| O2: SCF of O2(2+), Hamiltonian files, levels | `bench/dea/` (`o2_scf.py`, `o2_x2ccorr.py`, `common.py`, `o2_dea.py`, `o2_strings.py`, `o2_fold5.py`, `sizes.py`; the sector: `sector_bench.py`, `sector_check.py`, `sector_cost.py`) |

### 11.1 Operators and levels

R = 2p + 3p1h + 4p2h + ..: `r[(a<b)]`, `r[i, (a<b<c)]`,
`r[(i<j), (a<b<c<d)]`, `r[(i<j<k), (a<b<c<d<e)]`.  The matrix is
`<mu| [Hbar, R] |0>`, Hbar of the cluster operator T of the reference; its
eigenvalues are E(N+2) - E_CC(N).

For the **target** states the operators count as excitations of the two
attached electrons: 2p is the model space, 3p1h the single excitations,
4p2h the doubles, 5p3h the triples, 6p4h the quadruples.  A complete
4p2h truncation therefore describes the targets as CCSD describes a ground
state, complete 5p3h as CCSDT does.

| Level (`rank` / hmax) | R | T | For the target states |
|---|---|---|---|
| DEA-EOM-CCSD | 2p + 3p1h | CCSD | singles only |
| the same with the T of CCSDT | 2p + 3p1h | CCSDT | singles only |
| "DEA-EOM-CC3" (`row3='cc3'`) | 3p1h row at lowest order, `<3p1h\| [V~, R2] + [F, R3] \|0>` | CC3 | singles at lowest order (CC2-like; a negative control) |
| DEA-EOM-CCSDT | .. + 4p2h | CCSDT | CCSD-like |
| the same with the T of CCSDTQ or CC4 | .. + 4p2h | T1 .. T4 | CCSD-like |
| "DEA-EOM-CC4" (`row4='cc4'`) | 4p2h row at lowest order | CC4 | doubles at lowest order (CC2-like one rank up) |
| + 5p3h at lowest order, folded (`dea5`) | complete to 4p2h; `(omega - D5) r5 = <5p3h\| [V, R4] + [[V, T2], R3] + [[V, R2], T3] + 1/2 [[[V, T2], T2], R2] \|0>` | CCSDT or CCSDTQ | triples at lowest order on complete doubles (CC3-like) |
| complete 5p3h (`dea_strings`, hmax 3) | .. + 5p3h | any | CCSDT-like |
| 6p4h (`dea_strings`, hmax 4) | .. + 6p4h | any | CCSDTQ-like |

### 11.2 Two ghost spinors: one residual per sigma vector

The reference gets two more occupied spinors g1, g2 that no integral
touches and whose Fock elements are zero.  The EE operators that empty
both - `r[g1 g2, ab]`, `r[g1 g2 i, abc]`, `r[g1 g2 ij, abcd]` - are the DEA
operators, and the EE Jacobian maps them to themselves (section 5 has the
same for one ghost).  There are no singles in this sector.

Two operators of the sector have the product zero (each empties g1 and
g2), T has no ghost index and no term of the amplitude equations without R
has one.  So on the rows with both ghosts the residual of the amplitude
equations at T + R is **exactly linear in R**,

    sigma_mu(R) = Res_mu(T + R),

one evaluation of the residual code of the ground state - not the four of
the stencil of section 1, and no code of its own: the CCSDT step
(`ccsdt_opt.update_amps`, its Jacobi step undone) for the rows of 2p and
3p1h, the T4 equation and the T4 terms of CCSDTQ (`t4engine.residual`) for
4p2h; for the lowest-order rows the residuals of CC3 and CC4.  The ghosts
have an orbital energy far below the others (it enters the Jacobi steps
only).  With T3 = 0 and the amplitudes of CCSD the rows of 2p and 3p1h are
those of Hbar(CCSD): DEA-EOM-CCSD.

Vectors are stored short (the rows with both ghosts).  The residual code
works on the padded arrays: a sigma vector costs an iteration of the
ground state with o + 2 occupied spinors, C(o + 2, 4) / C(o, 2) times what
the sector alone costs in the pp ladder of T4 (11 for o = 10).  Section
11.9 evaluates the sector itself; in label blocks it is the default for R
to 4p2h.

**States** are selected by masks on the packed layout (`Problem`: a label
that adds with its `target`, and a parity), or by blocks:

**Label blocks** (`ProblemSym`).  The ghosts get labels *inside* the range
of the others, 1 and target - 1.  The operators of the target - the sum
of the labels of the two electrons, 2 Omega of the state - are then
totally symmetric on the padded spinors, T + R has the block structure of
T, and the label-block engines of the ground state (`t3engine`,
`t4engine_sym`) are called as they are; the short vector is the elements
with both ghosts, by index arrays into the flat block arrays.  (Section 5
puts its one ghost far outside the labels and uses shifted layouts; a
shifted T4 layout does not exist, and is not needed this way.)  The
blocks also hold amplitudes with one ghost that happen to be symmetric:
zeros that are carried along.  Parity is a mask on the short vector.

**Guess**: unit vectors on the lowest pairs (a<b) of the block; the
Davidson solver of section 4.

### 11.3 The 5p3h operators at lowest order, folded (`dea5`)

    (omega - D5) r5 = <5p3h| [V, R4] + [[V, T2], R3] + [[V, R2], T3] + 1/2 [[[V, T2], T2], R2] |0>
    sigma(3p1h) += <3p1h| V r5 |0>,      sigma(4p2h) += <4p2h| (F~ + V~) r5 |0>

D5 the differences of the orbital energies (canonical spinors), V bare in
the 5p3h row, the rows below complete in r5 (`f~[m,e]`, `<dm||ef>~`,
`<mn||le>~` T1-dressed, `<mn||ef>`).  The right-hand side is the quantity
r of (P) (`pentuples.Pentuples`: `<P| [V, T4] + [[V, T2], T3] + 1/6 [[[V,
T2], T2], T2] |0>`) at the amplitudes T + R, on the rows with both ghosts
- linear in R again.  For one occupied triple (i<j<k) at a time:
r5[(abcde)] from the class of (P), divided by omega - D5, expanded on
(e | abcd) and (ef | abc) and contracted into the rows of 3p1h and 4p2h;
nothing of the size of the 5p3h vector is kept.  The products with two
and three virtual indices are done in the blocks that the labels of the
triple allow (2 Omega and parity), with the summed indices for which the
operand is not zero there.  The eigenproblem depends on omega and is
solved to self-consistency from the root of the rows to 4p2h
(`ccn.folded_roots`; the slope is the weight of 5p3h in the state, 0.2%).

### 11.4 Any rank on determinants (`dea_strings`): the reference

`P exp(-T) H exp(T) P` on the determinants `|K, B>` - K the occupied
spinors still filled, B the h + 2 virtual ones - with at most hmax holes,
evaluated as it stands: exp(T) on the vector up to hmax + 2 holes, H,
exp(-T).  Every operator is `sum coef[X, Yc, Ya] v+(Yc) v(Ya) O(X)`; for a
determinant of the result the strings are gathered, multiplied with the
coefficients in one matrix product and scattered, label by label (2 Omega
and parity).  No diagrams, any T, any rank; with all holes it is full CI.
Its intermediates are the determinants with hmax + 2 holes, which is why
it is a reference for small spaces only (section 11.7).

### 11.5 Validation

| Check | Agreement |
|---|---|
| complete operator: LiH from LiH(2+) (STO-3G; ghost route with R to 4p2h, determinant route with two holes); LiH(2-) from LiH with T of CCSDTQ and four holes - against the full CI | 1e-9 |
| sigma of every level of the ghost route (CCSD, CCSDT, CC3 row, 4p2h with T3 / T4, CC4 row; real and complex) against the dense term tables (`ccsdtq_ref`, `ccn_ref`) at T + R, and against the dense EE Jacobian on R (`eom_ref.sigma`); linear in R | 1e-11 |
| the determinant matrix with one and two holes against the matrix of the ghost route (T to T4), eigenvalues; the two routes on one vector element by element | 1e-9 |
| the determinant matrix with three holes (2p .. 5p3h, 896 determinants, random T1 .. T4) against `P exp(-T) H exp(T) P` of second-quantized operators in the Fock space | 1e-9 |
| `dea5`: the terms of r5 in the rows of 3p1h and 4p2h against the blocks of the determinant matrix; the right-hand side term by term against `<5p3h\| [Hbar, R] \|0>` of second-quantized operators (no T: [V, R4]; odd in T2: [[V, T2], R3]; T3 alone; T2 alone); the folded root against the lowest eigenvalue of the model matrix with the 5p3h rows written out | 1e-10, 1e-10, 1e-8 |
| label blocks against masks on the packed layout: BH from BH(2+) / 6-31G, X2C (front end; 3p1h, 4p2h, folded 5p3h); O2 / 26 spinors, every level | 1e-9; all 11 printed digits |
| O2 / 26 spinors: the 4p2h roots of the ghost route and of the determinant route | 1e-11 |

### 11.6 O2: the zero-field splitting of X 3Sigma_g-

**Setup** (`bench/dea/`).  O2(2+), closed shell, r = 1.20752 Angstrom on
the z axis, Gaussian nuclei; Kramers-restricted HF with the X2CMP
Hamiltonian (Gaunt and Breit in the mean field, `socutils.somf.x2cmp`) and
Coulomb two-electron integrals; basis DYVTZ-SOG (Dyall TZ primitives,
contracted per kappa: 120 spinors) - E(SCF) = -148.313899397728 - and two
complete double-zeta bases: cc-pVDZ-DK (56 spinors; E(SCF) =
-148.242607249637) and DYVDZ-SOG (the j-adapted Dyall double zeta of the
other calculations, 56 spinors; E(SCF) = -148.285428295875, theirs
-148.285428247507).  For the correlation of O 1s: all electrons in
DYVTZ-SOG (120 correlated spinors; a valence set: no tight s / p, d or f),
and DYVTZ-SOG plus the core-valence functions of cc-pCVTZ (the shells of
cc-pCVTZ that cc-pVTZ has not: s 21.032, 7.845; p 57.437, 15.159; d
15.858, uncontracted), 172 spinors, E(SCF) = -148.314814148623, with 1s
frozen and with all electrons.  The added primitives are written into the
shells contracted per kappa (every shell of an l lists them, with zero
coefficients, and the shell with kappa = 0 gets a contraction for each);
as shells of their own the X2CMP SCF did not converge.  With them the SCF
gradient stays at 4e-9 while the energy is stationary to 2e-13
(conv_tol_grad 1e-8 for this basis).  The frozen-1s runs of an
all-electron file fold the 1s spinors into h1 and ecore (`common.load(...,
freeze=4)`; on DYVTZ-SOG it reproduces the roots of the frozen-core file
to 1e-11).  The X2Ccorr operator with nothing frozen is the bare
four-component one-electron operator.  The canonical spinors are adapted to m_j (real integrals) and carry a
parity (computed, not assumed); O 1s frozen: 10 + 106 (10 + 42) correlated
spinors.  The states: the lowest root of the blocks 2 Omega = 0 and
2 Omega = 2 of parity g; splitting = E(2) - E(0).

Two Hamiltonians:

- **spin-orbit only**: the X2CMP one-electron operator, Coulomb
  two-electron integrals;
- **X2Ccorr** (contains the spin-spin interaction; `o2_x2ccorr.py`): the
  X2C spinors taken back to the four-component picture, the no-pair
  Dirac-Coulomb-Breit two-electron integrals between them, the bare
  four-component one-electron operator plus the four-component mean field
  of the frozen 1s spinors.  The assembly reproduces a file of the group
  made for other spinors (ecore 1e-11, h1 1e-12, integrals 1e-13).  The
  spinors of the X2CMP calculation are not Hartree-Fock spinors of this
  operator (max |f_ov| 1.1e-5): a closed-shell HF for the 10 electrons
  inside the correlated space, block by block of (2 Omega, parity), lowers
  the energy by 7.5e-10 Eh (largest rotation angle 3.5e-5 rad; TZ:
  -148.322050765318 -> -148.322050766066; DZ: -148.250738032661 ->
  -148.250738033179) and gives the canonical spinors used here.  The same
  Hamiltonian in label blocks, without any n^4 array (`run-blocks`, needed
  from cc-pV5Z-X2C-TP on): 11.10.

Small spaces for comparison with full CI: the n lowest canonical spinors,
n = 26, 30, 32 (cut between shells; 28 would cut the pi_g(1/2, 3/2) pair of
virtual spinors in two).  The Hamiltonian of every space is written as
`o2_hf2plus[_dz]_n<n>[_x2ccorr].npz` (h1, (pq|rs), ecore, labels; the
reference energy recomputed from the file equals the SCF to 5e-13).

Thresholds: ground state |dT| < 1e-11; EOM residual norm < 1e-9 (folded:
|lambda(omega) - omega| < 1e-10).

**Splitting / cm-1** (experiment 3.96; it corresponds to X2Ccorr).

*Spin-orbit only*

| R | T of | 26 | 30 | 32 | cc-pVDZ-DK (52) | DYVDZ-SOG (52) | DYVTZ-SOG (116) | DYVTZ-SOG, all el. (120) | + CV, 1s frozen (168) | + CV, all el. (172) |
|---|---|---|---|---|---|---|---|---|---|---|
| 3p1h | CCSD | 2.6150 | 2.9682 | 2.9942 | 2.2098 | 2.6159 | 2.5618 | 2.5582 | 2.5861 | 2.5925 |
| 3p1h | CCSDT | 2.6235 | 2.9898 | 3.0190 | 2.3928 | 2.8031 | 2.7787 | 2.7796 | 2.8093 | 2.8357 |
| 3p1h, row at lowest order | CC3 | 3.6557 | 4.2506 | 4.3181 |  |  |  |  |  |  |
| 4p2h | CCSDT | 2.1260 | 2.3324 | 2.3538 | 1.9537 | 2.2637 | 2.3269 | 2.3265 | 2.3553 |  |
| 4p2h | CCSDTQ | 2.1414 | 2.3526 | 2.3753 | 1.9772 | 2.2897 |  |  |  |  |
| 4p2h, row at lowest order | CC4 | 2.0107 |  |  |  |  |  |  |  |  |
| 5p3h at lowest order, folded | CCSDT | 2.0807 | 2.2657 | 2.2822 | 1.8273 | 2.1500 | 2.1829 |  |  |  |
| 5p3h at lowest order, folded | CCSDTQ | 2.0953 |  |  |  |  |  |  |  |  |
| 5p3h | CCSDT | 2.0698 | 2.2540 | 2.2718 |  |  |  |  |  |  |
| 5p3h | CCSDTQ | 2.0825 | 2.2700 | 2.2888 |  |  |  |  |  |  |
| 6p4h | CCSDTQ | 2.0677 | 2.2525 |  |  |  |  |  |  |  |

*X2Ccorr*

| R | T of | 26 | 30 | 32 | cc-pVDZ-DK (52) | DYVDZ-SOG (52) | DYVTZ-SOG (116) | DYVTZ-SOG, all el. (120) | + CV, 1s frozen (168) | + CV, all el. (172) |
|---|---|---|---|---|---|---|---|---|---|---|
| 3p1h | CCSD | 3.1639 | 3.2682 | 3.2933 | 2.8872 | 3.2203 | 3.3575 | 3.3591 | 3.3732 | 3.4317 |
| 3p1h | CCSDT | 3.1088 | 3.2207 | 3.2462 | 2.6530 | 3.0389 | 3.1421 | 3.1414 | 3.1606 | 3.2095 |
| 4p2h | CCSDT | 3.4605 | 3.4396 | 3.4540 | 3.1699 | 3.4865 | 3.6026 | 3.5998 | 3.6101 |  |
| 4p2h | CCSDTQ | 3.4415 | 3.4219 | 3.4362 | 3.1442 | 3.4664 |  |  |  |  |
| 5p3h at lowest order, folded | CCSDT | 3.5439 | 3.5261 | 3.5455 | 3.4354 | 3.6886 | 3.8445 |  |  |  |
| 5p3h | CCSDT | 3.5410 | 3.5180 | 3.5357 |  |  |  |  |  |  |
| 5p3h | CCSDTQ | 3.5239 | 3.5006 | 3.5180 |  |  |  |  |  |  |
| 6p4h | CCSDT | 3.5376 |  |  |  |  |  |  |  |  |
| 6p4h | CCSDTQ | 3.5369 | 3.5107 |  |  |  |  |  |  |  |

Total energies of the 116-spinor space (Eh):

| Hamiltonian | R, T | E(2 Omega = 0) | E(2 Omega = 2) | splitting / Eh |
|---|---|---|---|---|
| spin-orbit only | 3p1h, CCSD | -150.16748498689 | -150.16747331453 | 1.1672e-05 |
| | 3p1h, CCSDT | -150.17621137304 | -150.17619871233 | 1.2661e-05 |
| | 4p2h, CCSDT | -150.20882789470 | -150.20881729275 | 1.0602e-05 |
| X2Ccorr | 3p1h, CCSD | -150.17574146051 | -150.17572616260 | 1.5298e-05 |
| | 3p1h, CCSDT | -150.18447008413 | -150.18445576778 | 1.4316e-05 |
| | 4p2h, CCSDT | -150.21709561323 | -150.21707919864 | 1.6415e-05 |

(The energies of every space and level are in the json files of the runs;
`bench/dea/tables.py` prints them.)

**The full space at the 5p3h level** (folded lowest-order 5p3h on
complete 4p2h, T of CCSDT): spin-orbit only E(0) = -150.21774021155,
E(2) = -150.21773026555, splitting 9.9460e-06 Eh = 2.1829 cm-1; X2Ccorr
-150.22601117439, -150.22599365737, 1.7517e-05 Eh = 3.8445 cm-1
(experiment 3.96).  From the truncated spaces the folded row errs by
about +0.01, the 6p4h step is about -0.015 / +0.01 and T4 about +0.02 /
-0.02 cm-1.

**Correlation of O 1s** (all electrons minus 1s frozen, cm-1):

| | basis | 3p1h, T of CCSD | 3p1h, T of CCSDT | 4p2h, T of CCSDT |
|---|---|---|---|---|
| spin-orbit only | DYVTZ-SOG (valence) | -0.0036 | +0.0009 | -0.0004 |
| | DYVTZ-SOG + CV | +0.0064 | +0.0264 | not finished |
| X2Ccorr | DYVTZ-SOG (valence) | +0.0016 | -0.0007 | -0.0028 |
| | DYVTZ-SOG + CV | +0.0585 | +0.0489 | not finished |

**DYVDZ-SOG against the CI of the other calculations** (CI up to
quadruple excitations relative to the CAS, 12 electrons: 2.1662 and
3.6570 cm-1): the folded 5p3h level gives 2.1500 and 3.6886.

**What the numbers say.**

- The series in the rank of R is not monotone in quality per rank: 3p1h
  alone (singles for the target) is off by 0.3 to 0.6 cm-1, complete 4p2h
  changes it by -0.5 (spin-orbit only) and +0.3 (X2Ccorr), 5p3h by
  another -0.06 .. -0.08 and +0.08, 6p4h by -0.015 and +0.013 (n = 26),
  -0.018 and +0.010 (n = 30).
  The lowest-order rows are no substitute one rank down (the 3p1h row at
  lowest order: +1.0 to +1.3 cm-1; the 4p2h row at lowest order: 2.011
  against 2.126 complete).
- **The folded lowest-order 5p3h row** against the complete one, same T:
  +0.0109, +0.0117, +0.0104 cm-1 (spin-orbit only, n = 26, 30, 32),
  +0.0029, +0.0081, +0.0098 (X2Ccorr): it carries the 5p3h step to about
  0.01 cm-1, always from above.
- In the complete double-zeta basis the 5p3h step (folded) is larger than
  in the truncated spaces: -0.126 (spin-orbit only: 1.9537 -> 1.8273),
  +0.266 (X2Ccorr: 3.1699 -> 3.4354); the complete 5p3h row had not
  finished there when this was written (section 11.7), so the error of
  the folded row is calibrated on the truncated spaces only.
- T4 of the ground state at fixed R: +0.015 .. +0.02 cm-1 (spin-orbit
  only), -0.02 (X2Ccorr).
- **Against the exact full CI.**  On the Hamiltonian files of another
  calculation - state-averaged CASSCF spinors of O2, 26 to 32 spinors,
  spin-orbit only, where the full CI is known - with the O2(2+)
  determinant of those spinors as the reference (f_ov is not zero there;
  the complete rows take it as it is), T of CCSDTQ, determinant route:

  | n | 3p1h | 4p2h | 5p3h | 6p4h | 7p5h | full CI |
  |---|---|---|---|---|---|---|
  | 26 | 4.5256 | 3.2397 | 3.2683 | 3.2684 | 3.2684 | 3.2683 |
  | 28 | 4.0050 | 2.9130 | 2.9044 | 2.8857 | | 2.8864 |
  | 30 | 3.9157 | 2.8833 | 2.8710 | | | 2.8492 |
  | 32 | 3.8848 | 2.8763 | 2.8584 | | | 2.8355 |

  n = 26, E(2 Omega = 0): -149.81654719716 (4p2h), -149.81654231670
  (5p3h), -149.81655060092 (6p4h), -149.81655022231 (7p5h); full CI
  -149.8165502235 (given to ten decimals): the errors are +3.0e-6,
  +7.9e-6, -3.8e-7 and +1e-9 Eh, the same for the two components to
  1e-9 from 5p3h on.  The top of the sequence is the full CI; at 5p3h the
  splitting is still 0.02 cm-1 off for n = 28, 30, 32 (+0.018, +0.022,
  +0.023), at 6p4h 0.001 (n = 28: -0.0007).

### 11.7 Sizes and cost

Counted (`bench/dea/sizes.py`), block 2 Omega = 0 (2 Omega = 2: 4% less):

| | 10 + 22 | 10 + 42 (DZ) | 10 + 106 (TZ) |
|---|---|---|---|
| 3p1h: all / 2 Omega / 2 Omega and parity | 1.5e4 / 3.6e3 / 1.8e3 | 1.1e5 / 2.0e4 / 1.0e4 | 1.9e6 / 2.8e5 / 1.4e5 |
| 4p2h | 3.3e5 / 6.5e4 / 3.3e4 | 5.0e6 / 7.6e5 / 3.8e5 | 2.2e8 / 2.8e7 / 1.4e7 |
| 5p3h | 3.2e6 / 5.7e5 / 2.8e5 | 1.0e8 / 1.4e7 / 7.0e6 | 1.2e10 / 1.4e9 / 6.8e8 (5.5 GB) |
| 6p4h | 1.6e7 / 2.6e6 / 1.3e6 | 1.1e9 / 1.4e8 / 7.0e7 | 3.6e11 / 3.7e10 / 1.9e10 |
| T4 of the ghost route, label blocks (packed) | 6.7e5 (3.6e6) | 8.1e6 (5.5e7) | 3.0e8 = 2.4 GB (2.5e9) |
| T4 of O2(2+), label blocks | 2.8e5 | 3.4e6 | 1.3e8 |

Measured (one run each; the node of the first three columns was shared
with other steps of this work):

| | n = 26 | DZ (52) | TZ (116) |
|---|---|---|---|
| 3p1h sigma vector, label blocks | 0.15 s | 0.47 s | 2.5 s (12 threads); 0.7 s (24 cores of a node) |
| the same on the packed layout | 0.07 s | | 39 s |
| 4p2h sigma vector, label blocks | 0.6 s (packed: 2.7 s) | 8.7 s | 98 - 104 s (12 threads); 43 s (24 cores) |
| sigma vectors per root, 4p2h | 30 - 42 | 30 - 37 | 38 - 62 |
| folded 5p3h sigma vector | 7 - 9 s | 66 - 117 s (8 threads) | 1167 s, 1591 s (the first of two runs, 48 cores: right-hand side 402 - 536 s, rows below 472 - 735 s, intermediates of (P) 248 - 270 s, rows to 4p2h 44 - 48 s); 64 GB |
| folded 5p3h: sigma vectors after the 4p2h root | 28 - 36 | 32 - 36 | |
| determinant route, 5p3h sigma vector | 4 s | more than 45 min (not finished) | not possible |
| determinant route, 6p4h | 27 - 32 s | | |

The determinant route needs the determinants with hmax + 2 holes as
intermediates: for 5p3h with 106 virtual spinors 3.4e12 before the labels.
It is the reference of the small spaces.

These are the times of the ghost route; on the sector itself the 4p2h
sigma vector of 116 spinors takes 3.2 s on 32 cores (ghost route on the same
node: 39 s), 11.9.

### 11.8 Not done

- The complete 5p3h row for large spaces: the R-linear terms of the
  pentuples equation of CCSDTQP beyond the lowest order - the ladders and
  the ring term on r5, the terms with one de-excitation line (V T2 R4,
  V T3 R4, V T3 R3, V T4 R3, V T2 T2 R3, V T4 R2, V T2 T3 R2, V T2 R5) -
  as tensor contractions in the sector.  The folded lowest-order row is
  what exists (0.01 cm-1 of the complete one for O2).
- A sigma vector of the ghost route costs an iteration of the ground state
  with o + 2 occupied spinors: the engines work on the padded arrays, not
  on the sector (the passes over rows of T4 that hold a ghost and those
  that hold none are both done).  For R to 4p2h in label blocks the sector
  itself (11.9) has replaced it; the lowest-order rows (CC3, CC4) and the
  folded 5p3h operators still go through the ghosts.  The folded 5p3h row makes its
  intermediates of (P) for all occupied pairs and quadruples of the padded
  spinors at every sigma vector, although T2 + R2 differs from T2 in one
  occupied pair.
- DIP, and the sectors with other numbers of ghosts; left-hand vectors,
  properties.
- Complex arithmetic in `dea5` and `dea_strings` (real integrals: axial
  spinors); the ghost route of `dea.Problem` takes both.
- Out-of-core vectors on the ghost route: its Davidson subspace is in
  memory (116 spinors: 2.8e7 numbers, 224 MB, per vector); on the sector
  (11.9) the 4p2h part goes to a scratch file when it does not fit.
- The guess is unit vectors on the lowest pairs (a<b): a block whose
  lowest states have little weight there needs `nguess`.

### 11.9 The sector itself, without ghosts (`dea_sector`)

The ghost route of 11.2 evaluates the residual of the ground state on the
padded spinors and keeps the rows with both ghosts: the T4 equation runs
over all C(o + 2, 4) occupied quadruples (495 for o = 10) where the sector
has C(o, 2) occupied pairs (45), and its blocks of T4 are those of the
padded spinors (O2(2+), 168 spinors: 1.60e9 numbers in the ghost blocks for
1.5e8 of the 4p2h vector).  `zccsdt.cc.dea_sector` evaluates the same sigma
vector on the short arrays r2[(a<b)], r3[i, (a<b<c)], r4[(i<j), (a<b<c<d)]
in the blocks of the labels, with nothing padded.

**Which terms survive** (`zccsdt.cc.dea_ref`).  No integral and no T has
a ghost index and R has both, so in a product that survives on the rows
(g1 g2 ..) R is one factor and g1, g2 are two of its external occupied
letters.  From the table of the amplitude equations (`ccsdtq_terms`,
T1-dressed, T = T2 + T3 (+ T4)), for every term `coef A(product)` of the
residual of rank k and every amplitude factor with two external occupied
letters (L1, L2): that factor becomes r of the sector (with the sign that
brings L1, L2 to the front of its occupied letters), the antisymmetrizer
keeps the permutations that put g1 on L1 and g2 on L2 (their sign times
the antisymmetrizer over the k - 2 others).  Merged, this gives 7 terms in
the 2p row, 19 in the 3p1h row and 40 in the 4p2h row (42 with the T4 of
CCSDTQ: r2(ae) <mn||ef> t4(mnij; fbcd) and r2(ef) <mn||ef> t4(mnij; abcd));
`python -m zccsdt.cc.dea_ref [--t4]` prints them.  Evaluated on dense
arrays they are the ghost route to 1e-15 (H4, T of CCSD, CCSDT, CCSDTQ,
real and complex).

**Factorized** as the T4 equation of `t4engine` (the intermediates of the
ground state are those of the label-block engine, `t4engine_sym`), with i < j
real, the ghosts implied, P the antisymmetrizers over the splits:

4p2h, sigma4[ij; abcd]:

    pp     P(ab/cd) sum_{e<f} Wq(ab,ef) r4(ij; efcd)                Wq = Wpp + F_vv on pairs / 3
    hh     sum_{m<n} Wh(mn,ij) r4(mn; abcd)                          Wh = Whh - F_oo on pairs
    ring   P(ij) P(a/bcd) sum_{me} Wr(m,a,e,i) r4(mj; ebcd)
    22     P(ab/cd) { P(ij) [ - sum_e Wv(ab,e,i) r3(j; cde) + sum_m t(im,ab) XoR(m; j; cd) ]
                      + sum_{nf} ZR(nf; ab) t(nij; fcd) }
    13     P(a/bcd) { sum_m Wo(m,a,ij) r3(m; bcd) + sum_e t(ij,ae) XvR(e; bcd)
                      + sum_m WoR(m,a) t(mij; bcd) + sum_e r2(ae) Xv(e; ij; bcd)
                      - P(ij) sum_{m<n} YR(mn; j; a) t(mni; bcd) }
    T4     + sum_{m<n} WhR(mn) t4(mnij; abcd) + P(a/bcd) sum_{m<n} sum_f [sum_e r2(ae) <mn||ef>] t4(mnij; fbcd)

with the intermediates of the sector (linear in R, made per sigma vector)

    XoR(m; j; cd) = sum_{ne} <mn||ej> r3(n; ecd) + 1/2 sum_n WhR(mn) t(nj, cd)
                    - P(cd) sum_{e<f} <dm||ef> r3(j; efc) + sum_n sum_{e<f} <mn||ef> r4(nj; efcd)
    XvR(e; bcd)   = P(d/bc) [ sum_{mf} <dm||ef> r3(m; fbc) + 1/2 sum_f Wpp(bc,ef) r2(fd) ]
                    + 1/2 sum_{mnf} <mn||ef> r4(mn; fbcd)
    WoR(m, d)     = - sum_{e<f} <dm||ef> r2(ef) - sum_e f_me r2(de) - sum_n YR(mn; n; d)
    YR(mn; j; d)  = sum_{e<f} <mn||ef> r3(j; efd),   WhR(mn) = sum_{e<f} <mn||ef> r2(ef)
    ZR(nf; ab)    = sum_{me} <mn||ef> r3(m; eab)

and those of the ground state (once per problem): F_vv, F_oo, Wpp, Whh, Wr;
Wv(ab,e,i) and Wo(m,a,ij) with their T3 terms; the six-index Xv(e; kl; bcd)
of the T4 equation without T4.  The 22 / 13 lines are the products with Wv,
Wo, Xv, Xo, Y, Z of the T4 equation (`t4engine`) in which R takes one place:
every term of the 4p2h row of `dea_ref` is in exactly one of them (the list
is in the docstring of `dea_sector`; the two terms of 1/2 [[V,T3],T3] with
r3 in one place and in the other are one product, ZR t3, by the symmetry of
<mn||ef> under (m e) <-> (n f)).

3p1h, sigma3[i; abc]:

    P(a/bc) [ sum_{e<f} Wq3(bc,ef) r3(i; aef) + sum_e Wv(bc,e,i) r2(ae) ]    Wq3 = Wpp + F_vv on pairs / 2
    + P(c/ab) [ sum_{me} Wr(m,c,e,i) r3(m; abe) + sum_m t(im,ab) (sum_n YR(mn; n; c) + hR(c, m)) ]
    - sum_m Fo(m,i) r3(m; abc) + sum_{m<n} WhR(mn) t(mni; abc)
    + sum_{me} f_me r4(im; abce) + 1/2 sum_{mne} <mn||ei> r4(mn; abce) + P(c/ab) sum_m sum_{e<f} <cm||ef> r4(im; abef)

with hR(c, m) = sum_{e<f} <cm||ef> r2(ef); 2p, sigma2[ab]:

    sum_{e<f} Wq2(ab,ef) r2(ef) + sum_{me} f_me r3(m; eab) + P(ab) sum_m sum_{e<f} <bm||ef> r3(m; aef)
    + sum_{m<n, e<f} <mn||ef> r4(mn; efab)                            Wq2 = Wpp + F_vv on pairs

**Operation counts** (multiply-adds, dense, before the labels; o, v the
correlated occupied and virtual spinors):

| Piece | Sector | Ghost route (o + 2 occupied, the T4 equation) |
|---|---|---|
| pp ladder | C(o,2) C(v,2)^3 ~ o^2 v^6 / 16 | C(o+2,4) C(v,2)^3 ~ (o+2)^4 v^6 / 192 |
| ring | o^3 v^5 / 6 | (o+2)^5 v^5 / 36 |
| 22 (Wv r3, t2 XoR, ZR t3) | o^2 v^5 / 4 + o^3 v^4 / 4 + o^3 v^5 / 8 | the products with Wv, Xo, Z on all quadruples |
| 13 (Wo r3, t2 XvR, WoR t3, r2 Xv, YR t3) | o^2 v^5 (o + v) / 12 + o^4 v^4 / 12 | as above |
| hh | o^4 v^4 / 96 | (o+2)^6 v^4 / 1152 |
| XvR, ZR, XoR (per sigma) | o v^5 / 2, o^2 v^4 / 2, o^3 v^3 | - |
| 3p1h row | o v^5 / 4 (its pp ladder) | an iteration of CCSDT with o + 2 |

For o = 10 the pp ladder is 11 times smaller on the sector (495 / 45); the
labels cut both by about the same factor.

**Storage** (`Sector`): the labels of `t3pack.classes_of()`; `shift` the
class of a pair of two attached electrons ((target - 2 l_min) / step).
r2: the pairs of class shift; r3: one block per class K of the virtual
triples, rows the occupied i of class K - shift; r4: one block per class M
of the virtual quadruples (a list, `zc_t4s_quads`), rows the occupied pairs
of class M - shift.  Parity is a mask, as in 11.2.  The products go
through the kernels of the label-block T4 engine (`csrc/t4sym.c`, no new
kernel): a block of rows expanded on (ab | cd), (a | bcd) or (a | bc) and
reduced back antisymmetrized.

- pp ladder: per block M the rows in batches (an operand and a product of
  at most `batch_bytes`), r4 on [(ef), (row, (cd))], one product with the
  block of Wq per class of (ef); from the same view the 4p2h terms of
  sigma2, of sigma3 (<cm||ef>) and of XoR (thin products per row).
- ring: per spectator j the rows (m, j) on [(m e), (bcd)], grouped by
  d = k_m - k_e (the triples (bcd) then have one class), times the stacked
  operator [Wr rows (i a) | f_ov | -1/2 <mj||ei> | 1/2 <mj||ef>]: the ring,
  and the 4p2h terms of sigma3 and of XvR as rows of the same product;
  the blocks d in batches of `batch_bytes`.
- hh ladder: per block one product with Wh on the pairs of the block.
- the rows of sigma4 one at a time: X[(ab), (cd)] per class of (ab) and
  X[a, (bcd)] per class of a, each one product of stacked operands
  (gathered on all threads, `tn_rows`), reduced once.

**Memory** (`memory()`, the plan of `ProblemSector`): fixed - the dressed
integrals (<vv||vv> on pairs, as the ground state has them), the
intermediates of the ground state (Xv with the rows of T3 is the largest:
5.2 GB for 168 spinors, 11.9 GB for DYVQZ; in a scratch file when the
budget does not hold it with four vectors), Wq for the 4p2h and the 3p1h rows, the
tables and masks of the elements; per sigma vector - the result, r3 and
sigma3 on (a | bc), the intermediates of the sector, the largest of the
passes (the pp batches, a ring batch with the stacked operator, the
operands of one row).  `batch_bytes` is what the budget leaves, at most
1 GB.  The Davidson subspace goes into a scratch file (its 4p2h part) when
2 max_space + nroots + 2 vectors do not fit next to the plan
(`dea.kernel`).

**Validation** (`tests/test_dea_sector.py`; `bench/dea/`):

| Check | Agreement |
|---|---|
| `dea_ref` (the surviving terms on dense arrays) against the ghost route `dea.Problem`, H4 / STO-3G, random vectors, T of CCSD, CCSDT, CCSDTQ, real and complex | 1e-15 (test: 1e-11) |
| the sector against the ghost route in label blocks (`ProblemSym`), every element: BH from BH(2+) / 6-31G, X2C, 2 m_j of the pair 0, +-2, R to 3p1h and to 4p2h, T of CCSDT and of CCSDTQ (test: 1e-11); O2 / 26 spinors, T of CCSDT and CCSDTQ, and DYVDZ (52), 2 Omega = 0 and 2, parity g (`sector_check.py`) | 2e-14 of 10 - 60 |
| the sector against `dea_ref` on dense arrays: O2 / 26 spinors (2 Omega = 0) and 32 spinors (2 Omega = 2), X2Ccorr; elements outside the blocks of the reference: 1e-14 | 2e-14 of 15 |
| roots through the front end, both routes (BH) | 1e-9 |
| O2 X2Ccorr, 26 spinors: DEA-EOM-CCSD, -CCSD with T3, -CCSDT | all 11 printed digits |
| O2 DYVDZ-SOG (52) X2Ccorr: E(2 Omega = 0), E(2 Omega = 2), split | equal to the last digit (split 1.588555954867843e-05 Eh = 3.4865 cm-1) |
| O2 TZ (116) X2Ccorr, DEA-EOM-CCSDT | E(2 Omega = 0), E(2 Omega = 2) all 11 printed digits; split 1.641458334e-05 against 1.641458340e-05 Eh (6e-14: the residual threshold 1e-9), 3.6026 cm-1 |
| O2 sogcv, 1s frozen (168) X2Ccorr, DEA-EOM-CCSDT | energies 3e-14, 9e-14; split 1.644903156e-05 against 1.644903168e-05 Eh (1.1e-13), 3.6101 cm-1; the same under a budget of 14 GB (batches of 374 MB, the subspace in a scratch file): the same energies and split to all digits, 64 + 40 sigma vectors of 21.5, 20.2 s; RssAnon +15.5 GB over the roots against the budget of 14 GB - the preconditioner of that run divided by temporaries of the size of a block (now rows of at most 2e6 elements) and the root vectors are copied out of the scratch file at the end (1.2 GB each) |
| a sigma vector against the plan (tracemalloc; RssAnon over the phase) | BH (batches of 16 kB) in the test; 116 spinors 1.39 / 1.54 GB against 1.47; 168 spinors 3.53 / 4.3 GB against 3.92 (the RssAnon of the phase includes what the allocator keeps) |

**Measured** (`bench/dea/sector_bench.py`: one Slurm job per comparison,
the ghost route first, then the sector, in one process - same node, 32
cores of a Xeon Gold 6248R, 120 GB; the time of a sigma vector of the
block 2 Omega = 0 for a random vector):

| | ghost route | sector | |
|---|---|---|---|
| **116 spinors** (10 + 106), job 32222499 | | | |
| 4p2h sigma vector | 42.8, 38.7, 39.6 s | 3.4, 3.3, 3.2 s | 12x |
| pieces of a sigma vector | T4 ring 16.0, rows 10.8, pp 7.1 s; the CCSDT step on o + 2 | rows 1.2, pp 0.7, ring 0.7, intermediates 0.5 s | |
| memory of the sigma vectors (RssAnon peak over the phase) | +10.5 GB | +1.5 GB (plan 1.47) | |
| prepared (fixed part) | 2.6 s | 3.2 s, +1.6 GB (plan 1.98) | |
| a run, 2 Omega = 0 and 2 | 62 + 38 sigma vectors, 43 s each (24 cores, the run of 11.6) | 62 + 38, 4.0 s each, 460 s | |
| **168 spinors** (sogcv, 1s frozen: 10 + 158), job 32222500 | | | |
| 4p2h sigma vector | 393.5, 392.3, 389.0 s | 18.0, 17.4, 17.4 s (another node: 13.9, 14.3 s) | 22x |
| pieces | T4 ring 191, rows 103, pp 54, Xv 17 s | ring 4.9, pp 4.6, rows 4.2, intermediates 2.5 s | |
| memory of the sigma vectors | +43 GB (the run of 11.6: MaxRSS 85 GB) | +4.3 GB (plan 3.92) | |
| prepared | 21.5 s | 28.5 s, +8.5 GB (plan 9.3; Xv with the rows of T3 5.2 GB of it) | |
| a run, 2 Omega = 0 and 2 | 64 + 40 sigma vectors, 311 s each on 48 cores: 9.2 h | 64 + 40, 26 s each (the code before the stacked rows), 51 min | |

The ghost route spends its time in the T4 ring and the products of one
occupied quadruple, both over the C(12, 4) quadruples of the padded spinors;
on the sector the pp ladder, the ring and the rows are of one size, the pp
ladder at 400 GF/s (168 spinors) and the ring and the rows at 75 - 77
GF/s (smaller products: the rows per occupied pair, the ring per
spectator and label).

**DYVQZ** (o = 10, v = 206), counted from the labels (`sector_cost.py` on the
SCF of the DYVQZ-SOG session; the Hamiltonian file was not there yet),
2 Omega = 0 (2 Omega = 2: 4% less):

| | sector | ghost route |
|---|---|---|
| 4p2h elements (the blocks; half of them parity u) | 3.48e8 = 2.8 GB per vector | T4 blocks 3.76e9 = 30 GB per array |
| pp ladder, multiply-adds | 5.6e12 | 6.0e13 |
| ring, rows, hh | 8.9e11, 7.8e11, 4e9 | |
| fixed memory (plan) | 24 GB: Xv with the rows of T3 11.9, <vv\|\|vv> on pairs 3.6, the rest of the integrals and intermediates | |
| per sigma vector (plan) | 6.1 GB | |
| a sigma vector, 32 cores | about 45 s (the rates of the 168-spinor run per piece: pp 14, ring 12, rows 10, the rest 8 s) | 35 - 45 h per run were estimated |

A run of both blocks: about 100 sigma vectors, 1.5 h of sigma vectors on
32 cores, with the subspace (2 max_space + 3 = 27 vectors, 76 GB) in a
scratch file on a node of 120 GB - a few hours in all, against 35 - 45 h
and 250 - 300 GB through the ghosts.

**For the native program** (cfour/xzccsdt, which has EOM-EE in label
blocks with real arithmetic, its Davidson solver and MPI over the rows of R3,
docs/native.md 16), what this needs:

- the vector: r2 on the pairs of one class, r3 in blocks [i, K], r4 in
  blocks [(ij), quadruples of class M] (the quadruple lists and the (a | bcd),
  (ab | cd) expansions of `t4sym.c`, which the CC4 code of the native
  program has in its own form); parity a mask; a label shift = the class of
  the pair, as `%xzeom` gives shifts;
- the intermediates of the ground state: those of an EOM-EE sigma vector
  (F, Wpp, Whh, Wr, Wv and Wo with T3) plus the six-index Xv of the T4
  equation (the largest array; on disk when it does not fit);
- per sigma vector: the small intermediates (XoR, XvR, ZR, YR, WoR, WhR;
  o v^3 at most), then three passes over r4 - the pp ladder by batches of
  rows of a block, the ring by spectator and label, the rows of sigma4 one
  occupied pair at a time - and the 3p1h and 2p rows; all of it products of
  blocks (dgemm) with the expansions and reductions of `t4sym.c`;
- MPI: the rows of r4 (occupied pairs) are the natural unit, 45 for o = 10
  - the pp ladder and the rows by pairs, the ring by spectator (its products
  write rows (i, j) of other owners: a reduction of sigma4 by rows, as the
  EOM-EE code reduces sigma_3); the small intermediates on every rank
  (summed after the passes that make them from r4);
- the reference: `dea_sector.Engine.sigma` (pieces selectable by `only`),
  element by element, and the roots of 11.6.

### 11.10 The X2Ccorr Hamiltonian in label blocks (`eri4c`, `hblocks`)

The dense route of `o2_x2ccorr.py run` (11.6) holds n^4 complex numbers
several times: socutils' `x2c_active_eri_4c` makes eight classes of
integrals by `pyscf.ao2mo.general`, each a full complex (n^2, n^2) array,
and sums them; then a complex -> real copy, a Hartree-Fock with dense Fock
builds, four tensordot rotations and `np.savez` of the complex array.
Profile for cc-pVQZ-X2C-TP (216 correlated spinors, 284 uncontracted AO
spinors, 48 cores, `bench/dea/x2ccorr_profile.py`):

| step | s | RssAnon, peak in the step |
|---|---|---|
| frozen-core mean field (four-component J, K) | 130 | 3.1 GB |
| `ao2mo.general` LLLL, LLSS, SSLL, SSSS | 171, 194, 197, 224 | 37 GB, then 73 GB (the sum and one class) |
| `ao2mo.general` four Breit classes | 470, 474, 483, 464 | 73 GB |
| complex -> real copy | 11 | 55 GB |
| HF in the correlated space (83 dense Fock builds) | 988 | 21 GB |
| four tensordot rotations | 175 | 55 GB |

(job MaxRSS 71 GB without the `np.savez` of the 35 GB complex array; the
whole step was 2901 s in job 32267426, on another node).  The AO integrals
are a small part of the time: libcint needs 33 s (LLLL) to 298 s (one Breit
class) for all of them on 8 cores.  For DYVQZ-SOG the step was 5456 s and
126 GB (job 32222135); for cc-pV5Z-X2C-TP (360 correlated spinors, 444 AO)
one array is 16 n^4 = 269 GB.

**What replaces it.**  `zccsdt.eri4c` (adapted from socutils'
`x2c_4c_spinors` and `x2c_active_eri_4c`: the same spinors, classes and
signs) never makes an n^4 array:

- *labels*: on the axis the AO spinors of PySCF carry m_j and the spinors of
  the X2CMP calculation 2 Omega; their coefficients vanish on the AO of other
  m_j (to 1e-11 for the large components of cc-pV5Z-X2C-TP: set to zero), so
  (pq|rs) vanishes unless -l_p + l_q - l_r + l_s = 0 and the parities add to
  an even number.  Only those blocks are made and stored: 5.6 % of n^4 for
  cc-pVQZ-X2C-TP, real (the imaginary parts of the AO integrals are exactly
  zero on the axis; the first and the last (k, l) of every chunk are checked);
- *classes*: five instead of eight - (SS|LL) = (LL|SS)^(rspq),
  (SL|SL) = conj (LS|LS)^(qpsr), (SL|LS) = conj (LS|SL)^(qpsr) (the AO
  integrals obey these to 6e-13 of the largest Breit integral) - and of the
  classes that are symmetric under (ij|kl) -> (kl|ij) themselves (LLLL, SSSS,
  LSLS) only the AO shells I <= K (I = K with weight 1/2):
  (pq|rs) = Y + Y^(rspq), Y = (LS|LS)' + (LS|SL)/2, then its mirror
  conj Y^(qpsr) added, then (LL|LL)' + (SS|SS)' + (LL|SS) (the primes: the
  halves).  About 40 % of the AO integrals of the eight classes (by
  libcint's times) are computed, each once;
- *two passes per class*: chunks of AO integrals (all i, j; one k shell; a
  range of l shells; libcint with OpenMP over the shell pairs ij), the bra by
  two dense GEMMs (the first on the real view of the complex chunk, no
  copy), scattered into the half-transformed integrals
  H[(l1, l2, rho)][k, l, p, q] that the labels allow (k of label rho, l of
  label l1 - l2 + rho; 12 - 15 % of n^2 N^2); then the ket for every label
  triple by two GEMMs into the label block (l1, l2, rho, l1 - l2 + rho);
- *memory plan* (`eri4c.plan`; budget = max_memory less what is resident):
  the label buffer (13.5 GB for 364 spinors), H of one class (21.1 GB) and
  one chunk (6.4 GB at most); H goes to a scratch file when it does not fit,
  and when the label buffer does not fit either, the blocks are made in slabs
  closed under the two mirrors (one pass over the AO integrals per slab).
  Measured: peak 13.4 GB for cc-pVQZ-X2C-TP (plan 12.2 GB plus 2 GB resident
  before), 42.7 GB for cc-pV5Z-X2C-TP (plan 41.6 GB plus 1.3 GB);
- *over nodes*: `kshells` restricts the k shells; the label buffers of
  several processes add up (`LabelBuffer.save / add_file`; the mirrors are
  linear, so each part may apply them).  Not needed so far - one shared node
  does cc-pV5Z-X2C-TP - and there is no driver for it yet;
- the mean field of the frozen 1s spinors: by default the four-component J
  and K of their density as in `run` (`frozen_field`; 113 s for
  cc-pVQZ-X2C-TP on 48 cores, 1521 s for cc-pV5Z-X2C-TP on 32 cores); with
  `--fold-core` the four frozen spinors are transformed with the correlated
  ones (n + 4) and folded (`hblocks.fold`: h1 += sum_c <pc||qc>, ecore +=
  sum_c h_cc + 1/2 sum_cd <cd||cd>; 20 s).  The two differ by the rounding of
  the dense route: its J and K contain the coefficients of the 1s spinors off
  the labels (1e-13) times the large small-component integrals - h1 2.6e-10
  (DYVDZ-SOG), 7.2e-10 (DYVTZ-SOG), the splits 6e-14 Eh, 1.4e-13 Eh;
- elements of h1 between groups (6.6e-11 for DYVDZ-SOG, 1.4e-8 for
  DYVTZ-SOG: rounding of the same kind) are set to zero, as the integrals
  between groups are not stored (kept, they left f_ov = 5e-9 in the
  canonical spinors; set to zero, 1e-12).

`zccsdt.hblocks`: the block file (HDF5; the groups (2 Omega, parity), their
members, a table of the allowed blocks (g1 g2 | g3 g4) with offsets into one
contiguous dataset, each block [p, q, r, s]; h1, ecore, labels, energies,
coefficients) - readable through `numpy.memmap` at the dataset offset, or by
any HDF5 reader; J and K of a density that is block diagonal in the groups
(the blocks (g g | h h) and (g h | h g) only), the closed-shell HF in the
correlated space group by group (the procedure of `o2_x2ccorr.hartree_fock`;
the two spinors of a Kramers pair, of equal energy, ordered by label and not
by their last digits), the rotation to the canonical spinors block by block
(U is block diagonal in the groups: every block maps onto itself, four small
GEMMs - no n^4 tensordot), subsets, folding of frozen spinors.
`common.load` reads a block file (`.h5`) straight into the antisymmetrized
blocks of the integral container (the large ones into scratch files when
they take more than half of the budget).

Generating the integrals once in the canonical spinors (HF first) would need
the blocks (g g | h h) and (g h | h g) of every group h with an occupied
spinor - itself a pass over all AO integrals; the rotation of the stored
blocks costs 10 s (216 spinors) to 56 s (360).

**Validation** (`bench/dea/x2ccorr_compare.py`: element by element against
the .npz of the dense route, the spinors matched by label, parity and energy;
`x2ccorr_loadcheck.py`: the integral containers of `common.load`; default
options):

| | DYVDZ-SOG (52) | DYVTZ-SOG (116) | cc-pVQZ-X2C-TP (216) |
|---|---|---|---|
| (pq\|rs), max difference (largest element) | 3.9e-13 (0.82) | 2.3e-12 (0.82) | 1.5e-11 (2.13) |
| the same with `--fold-core` | 3.4e-11 | 1.2e-10 | |
| h1 within the groups | 2.9e-12 | 1.2e-11 | 1.9e-11 |
| ecore, E(reference) | 5e-12 | 6e-12 | 5e-12 |
| elements of the .npz that the labels forbid | 1.1e-11 | 3.4e-9 | 1.7e-10 |
| integral container (all seven blocks) | 4.2e-13 | 3.4e-12 | |
| split, DEA-EOM-CCSDT (4p2h) / Eh | 1.588555960552185e-05 (dense 1.588555954867843e-05) | 1.6414583399182447e-05 (dense: the same digits) | 1.682436874261839e-05 (dense 1.682436871419668e-05) |
| split, DEA-EOM-CCSD / Eh | 1.4672877995280942e-05 (same digits) | 1.5297908504408042e-05 (dense 1.529790853282975e-05) | 1.579624952796621e-05 (dense 1.57962494995445e-05) |

(DYVDZ-SOG: the splits with the T of CCSDT and of CCSDTQ also to 6e-14 Eh;
cc-pVQZ-X2C-TP: the DEA on the file of the first run - the frozen-core field
of the default, LSLS, LLLL, SSSS in full, h1 between groups kept; its
integrals agree with the dense file to 9.9e-12, as those of the default to
1.5e-11; total energies of the
ground and the DEA states to 1.4e-12 Eh.)  The tests
(`tests/test_eri4c.py`, Li2 with random coefficients that respect m_j and
parity) compare with the eight `ao2mo.general` calls of socutils in real and
complex arithmetic, with budgets that put H into a scratch file and that
force slabs (tracemalloc peak within each plan), kshells in two parts, and
the J/K, HF, rotation and subsets of the block file against dense arrays.

**Time and memory** of the integral step (SCF from the checkpoint, frozen
core, integrals, HF, rotation, file):

| | cc-pVQZ-X2C-TP (216), 48 cores | cc-pV5Z-X2C-TP (360), 32 cores |
|---|---|---|
| dense route | 2901 s, > 71 GB | not possible on a node (269 GB per array) |
| blocks, default | 370 s (frozen core 113, integrals 242, HF 4, rotation 10), 13.4 GB | about 4100 s (estimated: frozen core 1521 s + the row below), 43 GB |
| blocks, `--fold-core` | 319 s (integrals of 220 spinors 270, fold 4, HF 9, rotation 28), 13.6 GB | 2603 s (integrals of 364 spinors 2460, fold 20, HF 42, rotation 48), 42.7 GB |
| file | 0.98 GB (dense .npz: 35 GB) | 6.4 GB (dense: 269 GB) |

For cc-pV5Z-X2C-TP by class (32 cores, the bra GEMMs in parentheses):
LSLS 614 s (132), LSSL 951 s (148), LLLL 233 s (125), SSSS 319 s (125),
LLSS 304 s (145); AO integrals 1500 s of 2444 s, the dense bra GEMMs 675 s.
A first run with LSLS, LLLL, SSSS in full (no halves) and the
four-component frozen-core field took 3001 s for the integrals and 4656 s
for the step, 41.6 GB; E(reference) of the two runs agrees to 3e-11 Eh (the
difference of the frozen-core fields).

**Usage**

    python bench/dea/o2_x2ccorr.py run-blocks <dir with the SCF checkpoint> --basis pv5z_x2c_tp \
        [--out <dir>] [--max-memory MB] [--npz] [--direct] [--fold-core]
    python bench/dea/o2_dea.py <dir>/o2_hf2plus_pv5z_x2c_tp_n360_x2ccorr.h5 out.json ccsd ccsdt --blocks

Not done:
- the bra GEMMs are dense in the AO index (27 % of the time for
  cc-pV5Z-X2C-TP); the coefficients are block diagonal in m_j (a factor of
  about 5 in the flops of the first two quarters);
- the four-component J and K of the frozen 1s density (`frozen_field`,
  PySCF's DHF) take 1521 s for cc-pV5Z-X2C-TP; `--fold-core` replaces them
  at the price of 1e-10 differences from the dense route;
- the split over nodes by k shells has no driver;
- `common.load` still builds the dense antisymmetrized blocks of the
  container (vvvv 120 GB for 350 virtual spinors, in a scratch file) - the
  native port reads the block file instead.

## 12. Left-hand eigenvectors (`zccsdt.cc.eom_left`)

    e, res = mycc.eom_ee(nroots=3, conv_tol=1e-8)
    left = mycc.eom_left()              # eom_left.Result: left.l1[k], l2[k], l3[k], left.e, left.overlap

The Jacobian A of section 1 is not Hermitian; the left-hand eigenvector of
the state k is the row vector with

    sum_nu l_nu A[nu, mu] = omega_k l_mu,        A[nu, mu] = <nu| [Hbar, tau_mu] |0>,

over the unique elements of the three ranks.  Nothing is conjugated: L =
sum_mu l_mu <mu|, as Lambda is held (docs/lambda.md, section 1), so that
`<0| L exp(-T)` is the bra of the state.  `l1[i,a]`, `l2[(i<j),(a<b)]`,
`l3` have the layouts of `r1`, `r2`, `r3` of the state.

### 12.1 The left product is the Lambda residual without its constant term

The Lambda equations are the derivatives of the Lagrangian with respect to
the amplitudes (docs/lambda.md, section 1),

    dL/dt_mu = <0| (1 + Lambda) [Hbar, tau_mu] |0> = eta_mu + sum_nu lambda_nu A[nu, mu],
    eta_mu   = <0| [Hbar, tau_mu] |0>:    eta_1 = f~[i,a],  eta_2 = <ij||ab>,  eta_3 = 0,

**for any amplitudes** - the part that is linear in Lambda is Lambda times
the Jacobian, with no use of the amplitude equations.  So

    (l A)_1 = r1[lambda := l] - f~[i,a],   (l A)_2 = r2[lambda := l] - <ij||ab>,   (l A)_3 = R3[lambda := l]

with (r1, r2, R3) = `lambda_opt.residuals`: **no term was written for the
left product**, and both engines of the Lambda equations serve as they are
- `lambda_opt.DenseL3` on the packed block, `lambda_blk.BlockL3` in the
blocks of the symmetry labels (for a vector with a shift: its passes with
shift arguments, 12.2) -, with the intermediates of `make_intermediates`
made once (`eom_left.LeftOps`).  Its cost is that of a
Lambda iteration less the Jacobi step and DIIS (section 12.4).

`tests/test_eom_left.py`: for random amplitudes that solve nothing and a
random l, `l A` equals the vector times the dense Jacobian of `eom_ref`
(`matrix()`, 424 x 424: LiH / STO-3G, 4 + 8, and H2O with the 1s pair
frozen, 8 + 4) to 1e-11 of its size, rank by rank, with real and with
complex spin orbitals.

**No component on the reference.**  In the space with the reference, the
left eigenvalue equation `<0| L (Hbar - E0) = omega <0| L` has the column
of the reference `l0 omega = 0` (the residuals `<mu| Hbar |0>` vanish), so
an excited state has l0 = 0 and there is no inhomogeneous term: the left
problem is the eigenproblem of the transposed Jacobian, nothing else.  The
row of the reference gives the other side: `<0| Hbar |mu> = eta_mu` is the
row of Hbar - E0 from the reference into the excited determinants, so the
right-hand eigenvector has the component

    r0(k) = eta . r(k) / omega_k = - Lambda . r(k)

on the reference (the second form: the left ground state `<0| (1 +
Lambda)` is orthogonal to the right vector of an excited state; `Lambda A =
- eta`).  r0 is zero by symmetry unless the state is totally symmetric; it
is not needed for the energies, and it enters the densities (section 13).
This is the whole "projection against the reference": the left vectors
that the solver returns are orthogonal to the right ground state (1, 0)
because they have no l0, and the right vectors are made orthogonal to the
left ground state by r0.

### 12.2 Label shifts and layouts

A left vector of a state with the label shift D (section 6) has its
elements where those of the right vector are: `l1[i,a]` where l_a - l_i =
D, the rank-3 part in the blocks of `t3shift.ShiftLayout(D)` - the matrix A
maps the elements of one shift to themselves from either side.  (As an
operator, `L = sum l i+ j+ .. b a` lowers the sum of the labels by D: the
left state carries the shift -D, and `L R` is totally symmetric.)

| where the right-hand vectors are | the left product |
|---|---|
| the packed block, no labels | `DenseL3` on the same layout |
| the packed block, the states of one shift by masks (`sector=`: 2 m_s of spin orbitals, 2 m_j with `symmetry = False`) | the same, the product projected on the shift (`eom.Sector`) |
| the blocks of the labels, totally symmetric states | `BlockL3` on the layout of T3; the parts of rank one and two masked to the shift, as the sigma vector is |
| the blocks of the labels, a state with a shift | **in the blocks of that shift** (`lambda_shift.ShiftL3`): the passes of `BlockL3` with the shift of l |
| labels in use but no engine of the blocks (no C kernels), or `dense=True` | through the packed block (`LeftOps.mode == 'dense'`): the rank-3 part is put on the packed block in lexicographic order, the product is taken there with `DenseL3` and projected on the shift, and what the layout of the shift holds of it is read back - right, at the cost and with the memory of the packed block; the reference of the tests |

**The passes with a shift** (`lambda_shift`; T3 and every operand of the
ground state have no shift, l1, l2, l3 and the three residuals have the
shift s of the vector; no kernel was written - the gathers and reductions
of `csrc/t3pack.c` are driven by the tables of `t3shift.ShiftLayout`):

- the half-expanded `L[P, D_delta, Q]` (delta = class(d) - class(m)) has
  its pairs Q in the group gP - delta + s;
- pass 1: the density `Dr[(a i), (g o)]` of l3 with T3 couples the index
  pairs of delta + s with those of delta (a rectangular block per delta);
  `D1` (l2 with T3) lives in delta = -s; the two stacked products have
  operands of different classes - x of class gP - class(d), y of class
  gQ - class(m), which coincide without a shift;
- the Gram matrix of l3 and T3 is between the occupied triples of group g
  and g + s;
- pass 2: the outer product with l1 goes to the blocks gQ = gP (rows of
  delta = s), that with l2 to gQ = gP + s (rows of delta = 0); of the four
  parts of the summed index of the drive products the two whose left
  operand is made of l (l2 and `Y1`) have the class of the other two plus
  s; the reduction takes the targets of the shifted layout;
- the ladder: the pairs of the gathered `Lv[Q, I, c]` are in the group
  g + s - class(c); `Xq` meets the occupied pairs of group g - class(c),
  `X2` the occupied index of class class(c) - s;
- the occupied operator does not change the group: block by block;
- of the dense residuals of rank one and two only the product of l2 with
  the diagonal pair blocks of the pp operator knows the labels (the
  occupied pairs of block h - s; `lambda_opt.residuals`).

`tests/test_eom_left.py`: LiH with X2C spinors (4 + 18, labels -3 .. 3) -
for every shift (-4 .. 4 classes; 14 to 868 of the 3264 triples) the
product of a random vector in the blocks of the shift is the one of the
packed block without labels to 1e-10 and the one that goes through the
packed block to 1e-11, **`l (A r) = (l A) r` with the sigma vector of the
shifted blocks** (`t3shift`: code that shares nothing with the Lambda
engines) to 1e-10, and with a phase on every spinor (complex integrals)
the product has the conjugate phases to 1e-11.

### 12.3 The solver

`eom.davidson` - the block Davidson solver of section 4, unchanged - on
the transposed matrix (its `apply` is the left product), **started from
the complex conjugates of the converged right-hand vectors, which it
follows** (for a Hermitian matrix they would be the left vectors), with
the preconditioner of the right-hand side.

*An eigenvalue problem, not a linear equation at the known omega.*  The
equation `(A^T - omega) l = 0` at the eigenvalue is singular; solving it
would need the projection against the solution that is being looked for,
or a shift, and gives nothing that the eigensolver does not: started from
conj(R), the Rayleigh quotient of the first iteration is omega to second
order, the preconditioner uses that value from the first correction on,
and the convergence test is the residual `l A - omega l` with the solver's
own omega - which is then a check: the eigenvalues of the two sides agree
to the accuracy of the vectors (below).  Measured (12.4): as many
iterations as the right-hand side takes from its singles-doubles guess
(0.97 to 1.15 times the products).

*Normalization.*  The solver returns vectors of norm one.  With
`M[k,m] = sum_mu l_mu(k) r_mu(m)`:

- between different levels M vanishes by the eigenvalue equations; its
  largest element there, relative to the diagonal, is reported
  (`Result.biorthogonality`) and nothing is done about it;
- inside a degenerate level (roots within `level_tol` = 1e-6 Eh: the
  components of a multiplet on the packed block) the left vectors are any
  basis of the left eigenspace; they are replaced by `M_level^-1 L`, the
  combinations that are biorthogonal to the right vectors as they are;
- a single root is scaled by 1 / M[k,k].

Then `L_k R_m = delta_km` (`Result.overlap`).  Only the sums over a level
of anything made of `L_k` and `R_k` are defined inside a degenerate level.

*Accuracy.*  The two eigenvalues differ by the product of the residuals
over the gap, times the departure from normality: 4e-8 to 1e-7 Eh with
both sides at 1e-6, 1e-9 to 7e-9 at 1e-7, below 1e-9 at 1e-8 (the tests ask
for 1e-9 there).  **A density has the errors of both vectors to first
order**: `eom_left` takes the threshold of the right-hand vectors by
default (kept in `eom.Result.conv_tol`), and properties to 1e-7 want
`conv_tol=1e-8` in `eom_ee` (LiH: with the left vectors at 1e-6 the
dipoles of the components of a triplet differed by 2e-6 between the blocks
and the packed block; at 1e-8 by 5e-8).

`tests/test_eom_left.py`:

| check | found |
|---|---|
| eigenvalues of the left problem against those of the right one, thresholds 1e-8 to 1e-10 (LiH / STO-3G real and complex, LiH / 6-31G X2C: shifts 0 and 2 in blocks, 8 roots of the packed block, 2 m_s sectors) | below 1e-9 (7e-11 for the packed block of LiH) |
| the left vector of the lowest singlet of LiH / STO-3G against the left eigenvector of the dense matrix (`scipy.linalg.eig(left=True)`) scaled to L R = 1; the projector `sum_k R_k L_k` of the triplet against that of the dense eigenvectors of the level | 1e-6 of the largest element asked (the vectors are at 1e-9) |
| biorthogonality between different levels before the normalization | 8e-12 to 6e-11 at 1e-8; 9e-7 at 1e-6 and 1e-7 at 1e-7 for HBr with a shift (12.4) |
| the blocks against the packed block: the weights of the ranks in `L R` of a level that lies in shift 0 with all its roots | 1e-6 asked |
| vectors in a scratch file: the same roots | 1e-10 |

### 12.4 Memory and cost

**Plan** (`eom_left.plan_for`, before anything is allocated): the plan of
the Lambda equations for the product (`lambda_opt.LambdaPlan`: T3, which
is there; l3 and its residual, which are a vector of the subspace and its
product; the pp operator, the kept operands, the batch buffers, which
shrink first when memory is short) and the 2 `max_space` vectors of the
subspace (default 4 + 3 nroots; at least 2 nroots + 2) - in memory if
`max_memory` holds them, else their rank-3 parts in a scratch file
(`max_disk`), the subspace lowered one vector at a time to what fits.
With a shift in the blocks the work array of the shifted passes is
counted where it is larger; through the packed block ('dense') the packed
T3 and the two copies of a conversion besides.  `MemoryError` / `DiskFullError` with the
numbers.  The operands of the right-hand side (`Result.ground`) stay
allocated and are resident memory that the budget sees.
`tests/test_eom_left.py`: H2O / cc-pVDZ, 8 + 38 - what numpy allocates
(tracemalloc) is between half and all of the plan less T3 and the
allowance for the threads; budgets that give a smaller subspace, the
scratch file, the error.  HBr / cc-pVDZ-DK, 8 + 28 in blocks: planned
21.7 MB and allocated 21.2 MB for shift 0, 21.4 and 20.7 MB for shift 2,
where the route through the packed block plans 66.5 MB.  (The first
version combined the vectors of a degenerate level in new arrays - one
more vector per root of the level, not in the plan - and did not count
the dense l2 of a product: 21.2 MB allocated against 20.9 planned; the
level is combined in place now and l2 is counted.)

**Measured** (`bench/slurm_eom_left.sh`, job 31939788 - the column of
shift 2: job 31945925 -, Xeon Gold 6248R, 8 threads, `max_memory` 26 GB,
both sides converged to 1e-7; the right-hand side from its singles-doubles
guess, the left from conj(R)):

| | H2O / cc-pVTZ, packed block | HBr / cc-pVTZ-DK, blocks, shift 0 | blocks, shift 2 | HBr, packed block |
|---|---|---|---|---|
| occupied + virtual, roots | 10 + 106, 2 | 8 + 78, 2 | 8 + 78, 2 | 8 + 78, 3 |
| CCSDT iteration | 4.63 s | 0.24 s | 0.24 s | 0.74 s |
| sigma vector (right) | 9.16 s = 1.98 iterations | 0.45 s = 1.85 | 0.42 s = 1.75 | 1.50 s = 2.01 |
| **left product** | **6.32 s = 1.36 iterations** | **0.28 s = 1.16** | **0.28 s = 1.15** | **1.05 s = 1.41** |
| left product / sigma vector | 0.69 | 0.63 | 0.66 | 0.70 |
| sigma vectors, iterations (right) | 34, 16 | 33, 16 | 33, 16 | 60, 28 |
| left products, iterations | 38, 19 | 37, 21 | 38, 21 | 58, 24 |
| right solve in all (guess stage included) | 330.9 s | 18.3 s | 16.9 s | 97.1 s |
| **left solve in all** | **244.6 s = 0.74** | **10.9 s = 0.59** | **11.0 s = 0.65** | **61.4 s = 0.63** |
| omega(left) - omega(right), largest | 6.6e-9 | 6.5e-9 | 2.4e-9 | 2.7e-9 |
| biorthogonality between levels | (one level) | 2.4e-12 | 1.2e-7 | 7e-14 |
| resident memory above the start of the left solve, peak | 5.87 GB | 0.36 GB | 0.35 GB | 1.98 GB |
| plan less T3 | 6.54 GB | 1.06 GB | 1.06 GB | 2.69 GB |

A left product costs 1.2 to 1.4 CCSDT iterations - a Lambda iteration -
against 1.8 to 2.0 for a sigma vector (two residuals of the T3 engine and
two passes 1), and the left solve as a whole 0.6 to 0.75 of the right one,
with as many products.  A state with a shift costs what a totally
symmetric one costs.  Through the packed block - the first version of the
states with a shift, job 31939788 - the same two roots of shift 2 took
1.02 s per product (2.4 times the sigma vector of the shifted blocks),
39.6 s in all and 1.32 GB, with the same eigenvalues to all ten digits
printed.
**26 + 120 spinors, 24 threads: the left product is slower than the sigma
vector.**  HBr / cc-pVQZ-DK, 10 frozen, the lowest 120 virtual spinors,
label blocks (T3: 8.2e7 elements), two roots, thresholds 1e-6,
`max_memory` 80 GB (jobs 31946736: shift 2, and 31947000: shift 0):

| | shift 0 (`BlockL3`) | shift 2 (`ShiftL3`) |
|---|---|---|
| CCSDT iteration | 6.13 s | 6.08 s |
| sigma vector | 10.74 s = 1.75 iterations | 10.15 s = 1.67 |
| **left product** | **13.19 s = 2.15 iterations** | **13.07 s = 2.15** |
| left product / sigma vector | 1.23 | 1.29 |
| sigma vectors / left products | 30 / 36 | 31 / 37 |
| right solve / left solve in all | 401.6 s / 484.4 s = 1.21 | 393.7 s / 493.4 s = 1.25 |
| omega(left) - omega(right) | -4.5e-8, -3.6e-8 | -3.7e-8, 1.9e-8 |
| resident memory above the start, peak; plan less T3 | 20.20 GB; 25.45 GB | 19.71 GB; 24.95 GB |
| of a left product: pass 1 | 5.1 s (stacked products 2.6 s on 4.1 cores, gathers 1.3 s, density 1.1 s) | 5.1 s (2.6 s on 3.9 cores, 1.3 s, 1.0 s) |
| pass 2 | 3.7 s (drive 1.5 s on 3.7 cores) | 3.7 s (1.5 s on 3.7 cores) |
| ladder | 1.5 s (`Xq`, `X2` 0.7 s on 3.5 cores) | 1.5 s (0.7 s on 3.4 cores) |
| residuals of rank one and two | 1.6 s | 1.6 s |

The shifted passes cost what the passes without a shift cost; the plan
holds at this size.  But a left product is 2.15 CCSDT iterations here,
where it is 1.15 with 8 occupied spinors and 8 threads: the Lambda passes
in blocks do their products per class of the indices with the threads of
the BLAS (`gemm` on blocks that are small with six classes of 26 occupied
spinors), where the T3 engine hands lists of small products to threads
(`gemm_tasks`) - 6.4 s of the 13 s run on four to seven of 24 cores.  This is the
engine of the ground-state Lambda equations as it is (measured before
with 8 occupied spinors only: docs/lambda.md 9.7), not something of the
left vectors; what it costs them is in 12.5.

**Under a budget** (job 31942452, H2O / cc-pVTZ, thresholds 1e-6;
`EB_LEFTMEM`: `max_memory` = the resident memory at the start of the left
solve + 3600 MB, where the unlimited run takes 5.87 GB more): the plan
shrinks the buffers of the passes to 329 MB and the subspace to 7
vectors, in memory; the peak is 3.52 GB above the start; a left product
takes 6.47 s (6.32 s with the large buffers), the solve 40 products in 20
iterations.

(The first run of the benchmark compared the peak of the resident memory
of the process with the plan - 7.7 GB against 6.7 GB for H2O - which
counts the operands of the right-hand side that were there before; the
table has the peak above the start.)

### 12.5 Not done

- EOM-CCSD without the T3 machinery (the product runs through the Lambda
  equations of CCSDT; `zccsdt.zccsd` has its own Lambda of CCSD), the
  approximate models (CC2, CC3, CC4: their Jacobians are not derivatives
  of a Lagrangian that is implemented), IP / EA / DEA.
- The right-hand operands are not released for the left solve.
- U / R-form integrals.
- **The threads of the Lambda passes in blocks** (12.4): with 26 occupied
  spinors and 24 threads the stacked products of pass 1, the drive
  products of pass 2, the products with `Xq` and `X2` and the dense
  residuals keep four to seven cores busy - a left product takes 13 s;
  with these parts on all cores it would be about 8 s (an estimate from
  their share, not measured).  The remedy is the task lists of the
  T3 engine (`t3kern.gemm_tasks`, as `ShiftT3.pass1` and `_pass2` use
  them) in `BlockL3` and `ShiftL3`; it would speed up the ground-state
  Lambda equations of such molecules by the same amount.
- Not measured: complex spinors with labels at a production size (tested
  on LiH with phases).  The densities of a state with a shift still go
  through the packed block (13.2).

## 13. Densities of excited states and transition moments (`zccsdt.cc.eom_density`)

    states = mycc.eom_density(response=True, transition=True)   # eom_density.State per root
    D = mycc.eom_rdm1(states[0], ao_repr=True)                  # <q+ p>, Hermitian, as make_rdm1(relaxed=False)
    pr = mycc.eom_properties(response=True)                     # dipole and transition moments; pr.text()

### 13.1 Definitions

With `Obar[p,q] = exp(-T) p+ q exp(T)`, the right vector `R_k = r0 + R1 +
R2 + R3` (r0 of 12.1), the left vector `L_k = L1 + L2 + L3`, `L_k R_k = 1`:

    g(k)[p,q]       = <0| L_k Obar[p,q] R_k |0>                         the density of the EOM-CC state
    <k~| p+ q |0>   = <0| L_k Obar[p,q] |0>                             left transition density
    <0~| p+ q |k>   = <0| (1 + Lambda) Obar[p,q] R_k |0>                right transition density
    gr(k)[p,q]      = g(k)[p,q] + <0| Z_k Obar[p,q] |0>,                with the response of the amplitudes
                      Z_k A = - xi,   xi_kappa = <0| L_k [Hbar, tau_kappa] R_k |0>

- **g(k)** is the unrelaxed EOM-CC density: the expectation value of the
  biorthogonal states at fixed amplitudes and spinors.  The reference
  contributes through r0 and through the part of `L R` that is the unit
  operator: trace g = N; no separate reference term is added.
- **The transition densities** have trace zero; the transition strength of
  an operator is the product of its two moments, `<0~| O |k> <k~| O |0>`
  (for the exact states |<0| O |k>|^2; it does not depend on the scale of
  R_k, since L_k R_k = 1), the oscillator strength 2/3 omega times the sum
  over the three components of the dipole.  This is the EOM-CC expression,
  not the linear-response one (which has the response of the ground-state
  amplitudes to the operator besides and is size-intensive).
- **gr(k)** is the derivative of the energy of the state, `E0 + omega_k =
  <0| L_k Hbar R_k |0>`, with respect to a one-electron perturbation **at
  fixed spinors**: E is stationary in L and R, and Z makes it stationary in
  the amplitudes, as Lambda does for the ground state.  It is what a
  finite field in the Hamiltonian gives, and gr(k) minus the ground-state
  density of t and Lambda is the derivative of omega_k.  It differs from
  g(k) at any truncation (the difference vanishes for the exact method:
  xi = 0 for eigenvectors of the full Hbar): 3e-6 a.u. in the dipole of
  LiH / STO-3G, 7e-3 for H2O / cc-pVTZ and 3e-3 for HBr / cc-pVTZ-DK (13.5).

**What a relaxed density would need besides** (not implemented): the
response of the spinors.  The orbital gradient of the state is the
generalized Fock contraction of its one- and two-particle densities with
the integrals, as for the ground state (`Gnh[a,i]` of
`density_opt.kernel(gradient=True)`, docs/lambda.md, section 10), with the
densities of the state in the place of those of t and Lambda: in the
notation of 13.2, `Gnh(T; Zt) + d/dx Gnh(T + x R; L)` at x = 0.  The
gradient is linear in the left vector and of fourth degree in T (a
two-particle operator: four commutators), so the derivative is a
five-point formula where 13.2 has a central difference - four more
evaluations of the gradient code per state - provided that code takes
amplitudes and left vectors that solve nothing, as the density code does
(to be checked: the table GRADIENT is written for converged t and lambda).
Then the Z-vector equations of the SCF and the frozen and dropped spinors
exactly as in `density_opt.relaxed_rdm1`.

### 13.2 Nothing is contracted that the ground-state codes do not contract

`d1(T; l)[p,q] = <0| (1 + l) exp(-T) p+ q exp(T) |0>` is
`density_opt.kernel(gradient=False)` for **any** amplitudes T and **any**
left vector l (the code never uses the equations they solve; in the blocks
of the labels `density_blk`).  It is linear in l and of second degree in T
(`Obar` ends at the double commutator), and `d1_ref = d1(T; 0)` is the
unit matrix of the occupied spinors with t1 in the block [i,a].  With R
the excitation part of R_k, which commutes with T:

    <0| L [Obar, R] |0>   = d/dx d1(T + x R; L) - (the part of d1_ref: r1 in [i,a])
                          = [d1(T + R; L) - d1(T - R; L)] / 2 - r1[i,a]            =: F      exact: second degree
    <0| L R Obar |0>      = d1(T; L'),      L' = the part of L R that still de-excites:
                            l'1[i,a]  = sum l2[ij,ab] r1[j,b] + 1/4 sum l3[ijk,abc] r2[jk,bc]
                            l'2[ij,ab] = sum l3[ijk,abc] r1[k,c]
                            (the fully contracted part of L R is L . R = 1: it gives d1_ref)

    g(k)           = d1(T; L') + r0 (d1(T; L) - d1_ref) + F
    <k~| p+ q |0>  = d1(T; L) - d1_ref
    <0~| p+ q |k>  = d1(T; r0 Lambda + Lambda') - d1_ref + [d1(T + R; Lambda) - d1(T - R; Lambda)] / 2
                     + (r0 + Lambda . R) d1_ref           Lambda' = the same contraction of Lambda with R;
                                                          the last term is zero for converged vectors

`L'` is one pass of the T3 engine over l3 - its pass 1 with r2 in the place
of `<mn||ef>` and r1 in that of `f[m,e]`, the outputs that the amplitude
equations call the T3 terms of the singles and of the doubles equation
(`eom_density.contract`; with a shift: `ShiftT3.pass1(l3, D; integrals of
shift -D)`, the output totally symmetric).

For the response, with `xi = r0 omega l + eta + L' A + xi''` (the same
three pieces of `L [Hbar, tau] R`):

    xi''_kappa = <0| L [[Hbar, tau_kappa], R] |0> = d/dx (L A(T + x R))_kappa at x = 0
               = [8 (y(1) - y(-1)) - (y(2) - y(-2))] / 12,    y(x) = the left product of 12.1 at the amplitudes T + x R

(A is a polynomial of third degree in T: the five-point formula is exact),
and `Z = Zt - L' - r0 L` with

    Zt A = - eta - xi'':        the Lambda equations of the ground state with xi'' as a source
                                (`lambda_opt.kernel(sources=)`, started from Lambda)
    gr(k) = d1(T; Zt) + F

Zt = Lambda + W, `W A = - xi''`; W is the response vector that section 14
uses as well.

**Cost per state**, in evaluations of the density code (each 0.8 CCSDT
iterations for H2O / cc-pVTZ): three for g(k), four more for the two
transition densities; for the response the left product at four displaced
amplitudes with their intermediates (6.6 iterations), one solution of the
Lambda equations (19 cycles: 25 iterations) and one density.  Measured,
all of it for one state: 38 CCSDT iterations (H2O / cc-pVTZ, packed), 42
(HBr / cc-pVTZ-DK in blocks), 119 for a state with a shift (through the
packed block; job 31939788).  An implementation with intermediates of its
own - the one-particle density needs none of the passes with `<vv||vv>` -
would take one pass for g(k); this one took no new contraction.

**Layouts.**  Where the engines of the blocks can take the arrays - labels
in use and a totally symmetric state: T + x R and L have the layout of T3
- everything runs in the blocks (`eom_density.View`); a state with a shift
is put on the packed block (T + x R is then not totally symmetric: the
Lambda and density engines of the blocks cannot take it: the left solve of
such a state runs in its blocks, 12.2, its densities do not).  Memory: the density code and the Lambda equations
plan themselves under `max_memory`; the arrays of the size of T3 that are
made besides (the displaced T3, a rank-3 part of zeros; on the packed
block the copies of T3, R3, L3, Lambda3) are checked against the budget
with the plan of the density code before they are allocated.

### 13.3 Front end and conventions

- `eom_density(right=None, roots=None, response=False, transition=False)`:
  a list of `eom_density.State` (`d1`, `d1_response`, `left`, `right`:
  [p,q] = <p+ q> over the active spinors, not Hermitian, as
  `density_opt.Result.d1`; `r0`; `zeta` = Zt), kept as `right.states`.
  The left vectors and Lambda are solved for if they are not there.
- `eom_rdm1(state, response=False, ao_repr=False)`: the density of a state
  in the convention of `make_rdm1(relaxed=False)` - `D[p,q] = <q+ p>`,
  made Hermitian, over all spinors [frozen occupied, active occupied,
  active virtual, dropped virtual], the frozen ones occupied -, so that
  the code that takes a ground-state density takes this one:
  `dip = -numpy.einsum('xpq,qp->x', mol.intor('int1e_r_spinor'), D_ao)`,
  and the operators of docs/usage.md in the same way.
- `eom_properties(right=None, roots=None, response=False)`: an
  `eom_density.Properties` - the dipole moment of the ground state (t and
  Lambda at fixed spinors) and of every state (with the picture change of
  the X2C Hamiltonian, as `zccsd.rdm.dipole_integrals`), the two
  transition moments, the dipole strength and the oscillator strength;
  for a degenerate level the sums over the level.

### 13.4 Validation (`tests/test_eom_density.py`)

| check | found |
|---|---|
| g(k), both transition densities, L' and xi'' against second-quantized operators in a Fock space (`tests/fockspace.py`): 4 + 5 spin orbitals, random Hermitian integrals, amplitudes, Lambda, L and R that solve nothing (L R = 1), real and complex | 1e-11 of their size asked |
| **gr(k) against finite fields**: LiH / STO-3G, real and complex spin orbitals; the derivative of E(CCSDT) + omega (all eigenvalues of the dense matrix) with respect to a field z at fixed orbitals, central differences of sixth order with 5e-4 a.u. - the lowest singlet, and the triplet as a level | singlet: finite field 1.3009719076, gr 1.3009719077 (1e-10; 1e-8 asked), g 1.3009684683; the ground state 1e-10 |
| two electrons (HeH+ / 6-31G, 2 + 6; the code runs without triples): densities of eight excited states against the eigenvectors of H among the determinants, level by level; g = gr; the strengths against `|<0| r |k>|^2`; the ground-state dipole | 2e-7 asked (vectors at 1e-9) |
| the blocks of the labels against the packed block (LiH X2C, 4 + 18): dipoles with and without the response and strengths of the levels that lie in shift 0; a state of shift 2 (through the packed block) - its density conserves the labels, its transition densities change them by -+2, and with the M = 0 component it adds up to the level of the packed block | 2e-6 asked; the dipoles of the components of the triplet within 5e-8 at 1e-8 |
| frozen core, dropped virtual spinors, the picture-changed dipole (HBr / cc-pVDZ-DK, 8 + 16 of 64 spinors): blocks against the packed block; `eom_rdm1` over all spinors, its trace and its dipole | 2e-6 / 1e-9 asked |

(The finite-field test first used the fourth-order formula: the fifth
derivative of the energy of LiH is large, and the ground state alone was
off by 2e-7 at 1e-3 a.u. and 4e-6 at 2e-3.)

### 13.5 Numbers

Job 31939788 (thresholds 1e-7; dipole moments in atomic units along the
axis, the spinors of the SCF fixed):

| | omega / Eh | dipole, g(k) | dipole, gr(k) |
|---|---|---|---|
| H2O / cc-pVTZ, lowest triplet (two of its components) | 0.27837687 | -0.23807394 | -0.24466639 |
| HBr / cc-pVTZ-DK, 28 frozen, shift 0, root 1 | 0.24041482 | -0.56305597 | -0.56561215 |
| shift 0, root 2 | 0.24084651 | -0.56534101 | -0.56788451 |
| shift 2, root 1 (the same level on the packed block: -0.57071141, -0.57325366) | 0.23374419 | -0.57071153 | -0.57325377 |
| shift 2, root 2 | 0.25601053 | -0.61366489 | -0.61649011 |

### 13.6 Not done

- The response of the spinors (13.1); excited-state gradients.
- The linear-response transition moments (with the response of the
  amplitudes to the operator); transition densities between two excited
  states (`<0| L_k Obar R_m |0>`: the formulas of 13.2 with L_k and R_m,
  and r0(m) d1(T; L_k) - small, not wired).
- One pass per density instead of three to seven evaluations of the
  ground-state code.
- A state with a shift in the blocks of the labels (13.2).
- Two-particle densities of the excited states.

## 14. A one-shot quadruples correction of the excitation energies (`zccsdt.cc.eom_q`)

    q = mycc.eom_quadruples()            # eom_q.Result; q.states[k].estimate, q.text()

The analogue of (Q) for an excitation energy: the first-order change of
omega_k of EOM-CCSDT under everything that CC4 adds to CCSDT, with both
eigenvectors,

    d omega_k = L_k dA(omega_k) R_k,        L_k R_k = 1.

### 14.1 What is included

The folded Jacobian of CC4 in the space of singles, doubles and triples
(docs/ccn.md, sections 4 and 11; `cc4fly.sigma`) differs from the Jacobian
of CCSDT at the CCSDT amplitudes by three things, all of first order in
the coupling to the quadruples, and **all three are included**:

| term | is | costs |
|---|---|---|
| `fold` | `L A_PQ (omega - D_Q)^-1 A_QP R`: the quadruples of the state, `r4 = ([V', T3] + [V~, R3] + [[V', T2], T2] / 2 + [[V~, R2], T2]) / (omega + e_ijkl - e_abcd)`, V' = [V~, R1], contracted with the integrals of H~ into the doubles and triples rows (`cc4fly.Back`) and then with l2, l3 | a pass over the quadruples with three drives |
| `t4` | L times the T4 of the ground state, `([V~, T3] + [[V~, T2], T2] / 2) / (e_ijkl - e_abcd)`, contracted with the integrals of H' = [H~, R1]: the derivative along R1 of the T4 terms of the doubles and triples equations | a pass with one drive |
| `ground` | the change of the Jacobian of CCSDT with the amplitudes, which the T4 terms of the amplitude equations move.  `Res(T) + rho(T) = 0` with rho the sums of T4 in the doubles and triples equations, so `dT = - A^-1 rho` to first order and `L (dA/dT . dT) R = xi'' . dT = W . rho` with `W A = - xi''`: **the response vector of the density of the state** (13.2: W = Zt - Lambda) | one pass for rho, once for all states, and a dot product per state - given the response vector (a Lambda-like equation per state) |

`omega_k + fold + t4 + ground` estimates the excitation energy of CC4.  In
the same way `Lambda . rho` is the first-order change of the ground-state
energy (a (Q)-like correction with the denominators and the dressed
integrals of CC4) and `Zt . rho` that of the energy of the state; the
correction of omega is their difference plus the two terms of the state.
Without `ground` (`eom_quadruples(ground=False)`: no response vector, no
Lambda) the correction belongs to the Jacobian at the amplitudes that were
given - for the CC4 amplitudes it estimates what the folded terms add to
the excitation energy "without the T4 terms", (i) of docs/native.md 19.3.

Requirements: canonical (semicanonical) spinors, as CC4.  The quadruples
are made on the packed block by the passes of `cc4fly` - in the sectors of
the (Q) code for the ground state and for the totally symmetric states of
a calculation in the blocks of the labels, without sectors (one process)
for a state with a shift and for the states of the packed block, whose
degenerate levels hold components of several shifts (the first version
used the sectors there and returned a third of the correction of a
triplet: the sector code keeps the totally symmetric part of its drives).

### 14.2 The table: one shot against converged EOM-CC4

`bench/eom_q_bench.py` (the cases of docs/ccn.md, section 12; converged
EOM-CC4 by `cc4fly.eom` on the same integrals, all thresholds 1e-8;
Hartree).  **A**: from CCSDT - omega(EOM-CCSDT) + fold + t4 + ground, all
at the CCSDT amplitudes.  **B**: from the CC4 ground state - the Jacobian
of CCSDT at the CC4 amplitudes ((i): the excitation energies without the
T4 terms), its left and right vectors, + fold + t4.  **Rayleigh**: B with
conj(R) / |R|^2 in the place of L, the estimate (ii) of docs/native.md
19.3.

| state | omega(CCSDT) | (i) | converged CC4 | A: fold, t4, ground | error of A | error of B | error of Rayleigh | error of (i) | error of CCSDT |
|---|---|---|---|---|---|---|---|---|---|
| H2O / 6-31G (10 + 16), lowest triplet (M_S = 0) | 0.284831364 | 0.285023404 | 0.284907411 | -2.457e-4, +1.124e-4, +1.901e-4 | **-1.9e-5** | **-2.2e-5** | -2.07e-4 | +1.16e-4 | -7.6e-5 |
| lowest singlet | 0.311687769 | 0.311883821 | 0.311781371 | -2.313e-4, +1.099e-4, +1.941e-4 | **-2.1e-5** | **-2.5e-5** | -2.08e-4 | +1.02e-4 | -9.4e-5 |
| HBr / cc-pVDZ-DK (8 + 28), shift 0, root 1 | 0.243377127 | 0.243481482 | 0.243411260 | -1.280e-4, +5.30e-5, +1.042e-4 | **-4.9e-6** | **-5.2e-6** | -7.3e-5 | +7.0e-5 | -3.4e-5 |
| shift 0, root 2 | 0.243830886 | 0.243935303 | 0.243864221 | -1.289e-4, +5.31e-5, +1.043e-4 | **-4.9e-6** | **-5.3e-6** | -7.4e-5 | +7.1e-5 | -3.3e-5 |
| shift 2, root 1 | 0.236883112 | 0.236987832 | 0.236916891 | -1.288e-4, +5.31e-5, +1.046e-4 | **-4.9e-6** | **-5.2e-6** | -7.3e-5 | +7.1e-5 | -3.4e-5 |
| shift 2, root 2 | 0.260664345 | 0.260771466 | 0.260672107 | -1.590e-4, +5.45e-5, +1.069e-4 | **-5.3e-6** | **-5.8e-6** | -8.4e-5 | +9.9e-5 | -7.8e-6 |
| shift 4, root 1 | 0.232875883 | 0.232980520 | 0.232911505 | -1.268e-4, +5.31e-5, +1.045e-4 | **-4.9e-6** | **-5.1e-6** | -7.2e-5 | +6.9e-5 | -3.6e-5 |
| shift 4, root 2 (doubles) | 0.544490906 | 0.544575972 | 0.544613194 | -3.25e-5, +5.03e-5, +8.51e-5 | **-1.95e-5** | **-1.98e-5** | -2.48e-4 | -3.7e-5 | -1.22e-4 |

What it says:

- **The left-right expression fixes the overshoot of the Rayleigh
  quotient.**  The effect of the T4 terms beyond (i) is, converged,
  -1.16e-4 (H2O triplet), -7.0e-5 (HBr, shift 0); the Rayleigh quotient
  gives -3.23e-4 and -1.43e-4 - 278 % and 204 %, the factor of two to
  three of docs/native.md -, the left-right expression -1.38e-4 and
  -7.5e-5: 119 % and 107 % (124 % for the singlet of H2O; 106 to 107 %
  for the other states of HBr that are dominated by single excitations).
  For the doubles-dominated root of shift 4 the Rayleigh quotient has the
  wrong sign (-2.11e-4 against +3.7e-5); the left-right expression has
  47 % of the effect.
- **How close**: 5e-6 Eh (0.13 meV, 1.1 cm-1) for the singly excited
  states of HBr, 2e-5 Eh (0.6 meV, 5 cm-1) for H2O and for the
  doubles-dominated state - an order of magnitude better than the Rayleigh
  quotient, 4 to 7 times better than EOM-CCSDT (not for root 2 of shift 2,
  where CCSDT happens to be within 8e-6), and the same as the estimate
  (iii) of docs/native.md 19.3 - one evaluation of the T4 terms followed
  by the relaxation of the vector: its errors there are -2.2e-5, -2.5e-5
  (H2O) and -5.2e-6 (shift 0), -5.2e-6, -5.8e-6 (shift 2), -5.1e-6,
  -2.0e-5 (shift 4) for HBr, on the spinors of CFOUR's SCF (other
  integrals than here: its excitation energies differ in the fourth
  digit), against B here: -2.2e-5, -2.5e-5, -5.2e-6, -5.2e-6, -5.8e-6,
  -5.1e-6, -1.98e-5.  One evaluation with the left
  vector and one evaluation with a relaxed right vector carry the same
  information; neither is the converged answer to 1e-5.
- **The term of the ground-state amplitudes is first order and accurate**:
  `ground` is +1.901e-4 against (i) - omega(CCSDT) = +1.920e-4 for H2O
  (99 %), +1.0419e-4 against +1.0436e-4 for HBr (99.8 %), and `Lambda .
  rho` is -4.599e-4 against E(CC4) - E(CCSDT) = -4.650e-4 for H2O (98.9 %),
  -2.1247e-4 against -2.1268e-4 for HBr (99.9 %).  So A, which needs no
  CC4 ground state, is as good as B (its errors are slightly smaller in
  this table).
- The three terms cancel to a large part (H2O: -2.46e-4 + 1.12e-4 +
  1.90e-4 = +5.7e-5 for a converged shift of +7.6e-5): what CC4 changes in
  an excitation energy is small next to its parts, and a part of them
  alone - the folded term, or the ground-state shift - is worse than
  nothing.
- For H2O the reference beyond CC4 exists (docs/ccn.md, section 12):
  the triplet is at 0.284974 (CCSDTQ, MRCC) and 0.285007 (FCI).  EOM-CCSDT
  is 1.76e-4 below FCI, converged CC4 1.00e-4, the one-shot estimate
  1.19e-4: the estimate inherits what CC4 lacks.

Time (H2O / 6-31G, 8 threads, a shared node: for orientation): right
vectors 2.0 s, left vectors 0.9 s, response vectors and densities 2.0 s,
the quadruples correction 3.8 s (rho 0.5 s, 1.6 s per state); the CC4
ground state 11 s and converged EOM-CC4 of the five lowest roots 189 s.
HBr, shift 0: 4.9, 2.9, 6.2 and 3.4 s against 12 s and 130 s.

### 14.3 Validation (`tests/test_eom_q.py`)

- LiH / STO-3G, real and complex spin orbitals: `fold + t4` is the left
  vector times (`cc4fly.sigma` - `eom_opt.sigma`) at omega, to 1e-12; the
  Rayleigh quotient likewise.
- H2O / STO-3G (10 + 4), the states of M_S = 0: `Lambda . rho` against
  E(CC4) - E(CCSDT) and `ground` against (i) - omega(CCSDT) within 5 %;
  the estimate against converged EOM-CC4: 1.1e-7 and 2.6e-8 Eh off, where
  the Rayleigh quotient is 4.8e-6 and 1.1e-6 off and EOM-CCSDT 2.7e-5 and
  1.1e-5.
- LiH with X2C spinors (4 + 18): the three terms in the blocks of the
  labels with the sectors of the (Q) code (shift 0) and for a state of
  shift 2 (no sectors) against the packed block, summed over each level
  of the packed block that the two shifts hold, to 5e-9.

### 14.4 Not done

- The correction is of first order: the relaxation of the vectors in the
  field of the T4 terms (the second evaluation of the native program:
  2e-6 or better) is not in it, and no second-order term.
- Large cases: the passes are those of `cc4fly` on the packed block (the
  memory plan of the (Q) objects; processes over the quadruples only in
  the sectors of totally symmetric states); the timing above is of the
  small cases.  The sectors for a drive with a shift (`run_sectors` with a
  label shift of the drive) would give the states of a linear molecule
  what the totally symmetric ones have.
- A (Q)-like form with the full perturbative triples-quadruples structure
  of CCSDTQ (the terms that CC4 leaves out of its quadruples equation).
- The estimate is not size-intensive beyond what CC4 is, and it has the
  denominators of the canonical spinors only.

## 15. Porting the left vectors to the native program

No native code was written; this is the list of what the port takes,
against what `xzccsdt` has (docs/native.md, sections 11, 15 and 16).

**What the native Lambda step already provides.**  `xzlamim` and
`xzlamstep` (`xzlam.inc`, `xzlblk.inc`) are `lambda_opt.make_intermediates`
and `lambda_opt.residuals` routine by routine - dense and in the blocks of
the labels, real and complex from the same `.inc` files, with the memory
plan of the work array and (section 15 there) over MPI ranks.  The left
product of section 12.1 is that step with three changes:

1. **the residuals before the Jacobi step**: `xzlamstep` ends in `xzlupd`
   (l += R / D); the left product stops before it and subtracts the
   constant terms - `f~[i,a]` from the residual of rank one, `<ij||ab>`
   from that of rank two (they are where `xzlr1` and `xzlr2a` start: two
   flags);
2. **vector in, vector out**: l1, l2, l3 and the three residuals are
   regions of the work array already (the residual of rank three is a DIIS
   slot or its own region); the solver hands in the offsets of a subspace
   vector and of its product instead of the regions of Lambda;
3. **the masks of the shift on the parts of rank one and two** in blocks
   (the sigma vector has them: `xzeom.inc`).

Nothing of the T3-sized passes changes, and the intermediates are those of
a Lambda run: a job that has run `xzccsdt lambda` has made them once
already (they are not kept on file: `xzlamim` is run again, a quarter to a
third of an iteration).  **One caveat of the memory layout**: the native
Lambda step makes the pp operator in the place of the bare `<ab||cd>`, and
the right-hand sigma vector keeps its own operators (W_pp and its
derivative); a run that does right vectors, then left vectors, reads the
integrals again between the two, as `ccsdt(q)_l` does after Lambda (0.5 s
for 10 + 106 spinors).

**What the solver needs** (`xzeoms.inc`, docs/native.md 16.3):

- the routine that applies the matrix as an argument (sigma vector or
  left product): the solver is CFOUR's algorithm 2 on one orthonormal
  basis and never looks into the product;
- **the start from `XZEOMR`**: the right-hand vectors of the roots are on
  that file (16.1) in the layout of the vectors of the solver; the left
  solve reads them, conjugates, orthonormalizes and uses them in the place
  of the singles-doubles guess - no guess stage;
- **following the roots** instead of taking the lowest: the targets are
  the eigenvectors of the projected matrix with the largest weight in the
  space of the start vectors (the rule of `eom.davidson(follow=True)`,
  section 4; CFOUR's `eompkvec3` recognizes converged roots by overlap in
  the same way, so the machinery is there, only its use for the targets is
  new);
- after convergence: the matrix `M[k,m] = sum l(k) r(m)` (no conjugation:
  a second dot-product kernel next to `xzevmdot`, or its conjugating flag
  off), the inverse of its blocks by level, the transformed left vectors
  written as a file `XZEOML` in the format of `XZEOMR`; the output prints
  omega of both sides and the largest off-diagonal element of M between
  levels;
- memory: the left solve is a Lambda run with a subspace in the place of
  the DIIS vectors - the plan of section 11 there with `2 x` (subspace)
  vectors where it has 8 DIIS vectors, their rank-3 parts in the work
  array or on the DIIS file exactly as now; no new region;
- over MPI ranks: the Lambda step is distributed already (section 15
  there); the vector operations of the solver are those of the right-hand
  side (16.7).

**A state with a shift** needs the Lambda step in blocks with shift
arguments; the native one takes arrays without a shift (`xzlblk.inc` =
`lambda_blk`).  The model is `lambda_shift.ShiftL3` (12.2), 400 lines
that list every place where the shift enters, validated against the route
through the packed block and against the shifted sigma vector.  The block
passes that the native Lambda step shares with the T3 equation take the
shifts of their operands from the module variables of `xzshctx` already
(docs/native.md 16.2: pass 2, the stacked products of pass 1, the operator
on occupied triples); the parts that are the Lambda step's own do not -
the density `Dr` and `D1` of pass 1 (rectangular blocks), the Gram matrix
between two groups, the ladder with `Xq` and `X2`, the two classes of the
outer products and of the four parts of the drive index in pass 2, and
the loop over the pair blocks of the pp operator in the residual of rank
two.  Totally symmetric states need none of it.

**Then, in this order** (each small once the left vectors are there):

1. *The densities* (section 13): the native density stage (`xzden.inc`:
   the passes over l3 and T3, the orbital gradient, the files of
   `xx2czvec`) is the code that 13.2 calls three to seven times per state,
   with other amplitudes and other left vectors - it has to take both as
   arguments (now: `XZAMPS`, `XZLAMB`) and must not use the equations
   they solve.  The contraction L' is pass 1 of the T3 engine with other
   operands, and the source xi'' is the left product at four displaced
   amplitudes, i.e. `xzlamim` and the left product four times; the
   response vector is a Lambda run with a source.  With the orbital
   gradient of that stage the relaxed density of an excited state would
   go through `xx2czvec` as the ground state's does (13.1).
2. *The one-shot quadruples correction* (section 14): `fold` and `t4` are
   the two passes of a folded sigma vector of the native EOM-CC4
   (`xze4`: the T4 terms of `xzt4` with three drives, and with the
   derivative integrals) followed by two dot products with l2 and l3
   instead of the addition to the sigma vector; `ground` needs the
   response vector W (a Lambda run with a constant term added to the
   three residuals: a source argument of `xzlam`, as `lambda_opt.kernel`
   has it) and one ground-state pass `xz4run` for rho.  Without `ground` - on the
   CC4 amplitudes that a native CC4 job has anyway, variant B of 14.2 -
   the correction is the first macro iteration's evaluation contracted
   with the left vector: no Lambda, no response, and the same accuracy as
   the relaxed estimate that the program prints now.
