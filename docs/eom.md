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
| reference runs | `bench/refs/reference_energies.json` (`eom_ee`), `bench/dirac/mrcc_eom.sh`, `bench/cfour/eom_ccsd.sh` |
| timing | `bench/eom_bench.py`, `bench/slurm_eom.sh`; label blocks: `bench/eom2_bench.py`, `bench/slurm_eom2.sh` |

Not in this document, and not done: left-hand vectors, transition
properties, excited-state gradients, the native Fortran program.

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

**What would make it cheaper, not done.**  The second residual - T3 with
the derivatives of the operators - is the part of `Hbar` with three-body
character.  Its pp ladder costs o^3 v^5 / 24 as a ladder, but `W'_pp` has
low rank in the occupied indices: `r2[mn,ab] (<mn||de> T3[ijk,dec])` is
o^5 v^3 through the intermediate in brackets, and `-P(ab) r1[m,a]
(<mb||de>~ T3[ijk,dec])` is o^4 v^4; likewise the ring term with
`<mn||de> r2`.  With these the second residual would cost a fraction of
the first for o << v, and a sigma vector little more than one iteration.
It needs kernels for T3 contracted with integrals to a six-index
intermediate; the three T2 terms of `pass1(T3; H')` also follow from the
T3 parts of W_vvvo and W_vooo of the ground state contracted with r1,
except `f'[m,e] T3[ijm,abe]`.

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

What was not measured: a run with the vectors in the scratch file at
production size (the tests do it at 8 + 16), complex arithmetic at
production size, more than 8 threads, IP and EA.

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

## 10. Not done

- Left-hand vectors, r0, transition moments, excited-state densities and
  gradients.
- The cheap IP / EA (section 5) and states with a label shift in the
  blocks (section 6): both written down, neither implemented.
- The factorized second residual (section 3) that would bring a sigma
  vector from two iterations to little more than one.
- EOM when the ground state needed the sliced pp operator or a mapped T3:
  `prepare` keeps the operands in memory.  U / R-form integrals (UCCSDT,
  RCCSDT) are not served: `capture` is for dense integrals.
- The small terms of a sigma vector are einsum and GEMMs on dense o^2 v^2
  arrays without the labels; the preconditioner and the subspace
  transformations are numpy on one thread.
- Core-valence separation, an energy window or a target label for the
  roots: the solver takes the lowest roots (or follows a guess).
- The native Fortran program.
