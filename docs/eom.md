# Equation-of-motion CCSDT on spinors

Excited (EE), ionized (IP) and electron-attached (EA) states on top of a
converged CCSDT (or CCSD) ground state of spin orbitals or spinors, real or
complex: excitation energies and right-hand vectors.

| What | Where |
|---|---|
| dense reference: sigma vector three ways, matrix, IP / EA by a ghost orbital | `src/zccsdt/cc/eom_ref.py`, `tests/test_eom_ref.py` |
| sigma vector on packed amplitudes | `src/zccsdt/cc/eom_opt.py`, `tests/test_eom_opt.py` |
| Davidson solver, memory plan, driver, ghost spinor | `src/zccsdt/cc/eom.py`, `tests/test_eom.py`, `tests/test_eom_solver.py` |
| states with a label shift in the blocks of the symmetry labels | `src/zccsdt/cc/t3shift.py`, `tests/test_eom_shift.py` |
| IP and EA with short vectors | `eom.ghost_short`, `tests/test_eom_short.py` |
| front end | `GCCSDT.eom_ee`, `eom_ip`, `eom_ea` (`src/zccsdt/ccsdt.py`; `docs/usage.md`) |
| double electron attachment (EOM-DEA; section 11): two ghost spinors, label blocks, the 5p3h operators folded, any rank on determinants | `src/zccsdt/cc/dea.py`, `dea5.py`, `dea_strings.py`, `tests/test_dea*.py`, `GCCSDT.eom_dea`, `bench/dea/` |
| reference runs | `bench/refs/reference_energies.json` (`eom_ee`), `bench/dirac/mrcc_eom.sh`, `bench/cfour/eom_ccsd.sh` |
| timing | `bench/eom_bench.py`, `bench/slurm_eom.sh`; label blocks: `bench/eom2_bench.py`, `bench/slurm_eom2.sh` |
| the native program in CFOUR's chain (`xzccsdt eom`, `ccsdt_eom`): the sigma vector of section 3 and the label shifts of section 6 in Fortran, with the solver of section 9 | `cfour/xzccsdt/xzeom.inc`, `xzeoms.inc`, `xzeomd.F90`; docs/native.md, section 16 |

Stage two (sections 4 to 6, 8): states with a label shift in the blocks of
a linear molecule, IP and EA with short vectors, the threaded solver, the
choice of the roots.

Not in this document, and not done: left-hand vectors, transition
properties, excited-state gradients.  The native Fortran program has
EOM-EE-CCSDT (not IP / EA): docs/native.md, section 16, with its
validation against this code and against MRCC and CFOUR.

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

- Left-hand vectors, r0, transition moments, excited-state densities and
  gradients.
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
| the 5p3h operators at lowest order, folded | `src/zccsdt/cc/dea5.py`, `tests/test_dea5.py` |
| any rank on determinants (reference route) | `src/zccsdt/cc/dea_strings.py`, `tests/test_dea_strings.py` |
| front end | `GCCSDT.eom_dea` (`docs/usage.md`) |
| O2: SCF of O2(2+), Hamiltonian files, levels | `bench/dea/` (`o2_scf.py`, `o2_x2ccorr.py`, `common.py`, `o2_dea.py`, `o2_strings.py`, `o2_fold5.py`, `sizes.py`) |

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
ground state with o + 2 occupied spinors, about (o + 2)^4 / (6 o^2) times
what the sector alone would cost in the T4 terms (5.8 for o = 10).

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
contracted per kappa: 120 spinors) - E(SCF) = -148.313899397728 - and, as
a complete double-zeta basis, cc-pVDZ-DK (56 spinors; E(SCF) =
-148.242607249637; the directory of the SOG sets has no double zeta for
O).  The canonical spinors are adapted to m_j (real integrals) and carry a
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
  -148.250738033179) and gives the canonical spinors used here.

Small spaces for comparison with full CI: the n lowest canonical spinors,
n = 26, 30, 32 (cut between shells; 28 would cut the pi_g(1/2, 3/2) pair of
virtual spinors in two).  The Hamiltonian of every space is written as
`o2_hf2plus[_dz]_n<n>[_x2ccorr].npz` (h1, (pq|rs), ecore, labels; the
reference energy recomputed from the file equals the SCF to 5e-13).

Thresholds: ground state |dT| < 1e-11; EOM residual norm < 1e-9 (folded:
|lambda(omega) - omega| < 1e-10).

**Splitting / cm-1** (experiment 3.96; it corresponds to X2Ccorr).

*Spin-orbit only*

| R | T of | 26 | 30 | 32 | DZ (52) | TZ (116) |
|---|---|---|---|---|---|---|
| 3p1h | CCSD | 2.6150 | 2.9682 | 2.9942 | 2.2098 | 2.5618 |
| 3p1h | CCSDT | 2.6235 | 2.9898 | 3.0190 | 2.3928 | 2.7787 |
| 3p1h, row at lowest order | CC3 | 3.6557 | 4.2506 | 4.3181 |  |  |
| 4p2h | CCSDT | 2.1260 | 2.3324 | 2.3538 | 1.9537 | 2.3269 |
| 4p2h | CCSDTQ | 2.1414 | 2.3526 | 2.3753 | 1.9772 |  |
| 4p2h, row at lowest order | CC4 | 2.0107 |  |  |  |  |
| 5p3h at lowest order, folded | CCSDT | 2.0807 | 2.2657 | 2.2822 | 1.8273 |  |
| 5p3h at lowest order, folded | CCSDTQ | 2.0953 |  |  |  |  |
| 5p3h | CCSDT | 2.0698 | 2.2540 | 2.2718 |  |  |
| 5p3h | CCSDTQ | 2.0825 | 2.2700 | 2.2888 |  |  |
| 6p4h | CCSDTQ | 2.0677 |  |  |  |  |

*X2Ccorr*

| R | T of | 26 | 30 | 32 | DZ (52) | TZ (116) |
|---|---|---|---|---|---|---|
| 3p1h | CCSD | 3.1639 | 3.2682 | 3.2933 | 2.8872 | 3.3575 |
| 3p1h | CCSDT | 3.1088 | 3.2207 | 3.2462 | 2.6530 | 3.1421 |
| 4p2h | CCSDT | 3.4605 | 3.4396 | 3.4540 | 3.1699 | 3.6026 |
| 4p2h | CCSDTQ | 3.4415 | 3.4219 | 3.4362 | 3.1442 |  |
| 5p3h at lowest order, folded | CCSDT | 3.5439 | 3.5261 | 3.5455 | 3.4354 |  |
| 5p3h | CCSDT | 3.5410 | 3.5180 | 3.5357 |  |  |
| 5p3h | CCSDTQ | 3.5239 | 3.5006 | 3.5180 |  |  |
| 6p4h | CCSDT | 3.5376 |  |  |  |  |
| 6p4h | CCSDTQ | 3.5369 |  |  |  |  |

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

**What the numbers say.**

- The series in the rank of R is not monotone in quality per rank: 3p1h
  alone (singles for the target) is off by 0.3 to 0.6 cm-1, complete 4p2h
  changes it by -0.5 (spin-orbit only) and +0.3 (X2Ccorr), 5p3h by
  another -0.06 .. -0.08 and +0.08, 6p4h by -0.015 and +0.013 (n = 26).
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
  | 28 | 4.0050 | 2.9130 | 2.9044 | | | 2.8864 |
  | 30 | 3.9157 | 2.8833 | 2.8710 | | | 2.8492 |
  | 32 | 3.8848 | 2.8763 | | | | 2.8355 |

  n = 26, E(2 Omega = 0): -149.81654719716 (4p2h), -149.81654231670
  (5p3h), -149.81655060092 (6p4h), -149.81655022231 (7p5h); full CI
  -149.8165502235 (given to ten decimals): the errors are +3.0e-6,
  +7.9e-6, -3.8e-7 and +1e-9 Eh, the same for the two components to
  1e-9 from 5p3h on.  The top of the sequence is the full CI; at 5p3h the
  splitting can still be 0.02 cm-1 off (n = 28, 30).

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
  that hold none are both done).  The folded 5p3h row makes its
  intermediates of (P) for all occupied pairs and quadruples of the padded
  spinors at every sigma vector, although T2 + R2 differs from T2 in one
  occupied pair.
- DIP, and the sectors with other numbers of ghosts; left-hand vectors,
  properties.
- Complex arithmetic in `dea5` and `dea_strings` (real integrals: axial
  spinors); the ghost route of `dea.Problem` takes both.
- Out-of-core vectors: the Davidson subspace of the 4p2h problem is in
  memory (116 spinors: 2.8e7 numbers, 224 MB, per vector).
- The guess is unit vectors on the lowest pairs (a<b): a block whose
  lowest states have little weight there needs `nguess`.
