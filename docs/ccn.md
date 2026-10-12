# The approximate iterative models: CC2, CC3, CC4

The hierarchy CC2 < CCSD < CC3 < CCSDT < CC4 < CCSDTQ for spin orbitals and
spinors, real or complex, ground states and excitation energies (linear
response: the eigenvalues of the Jacobian of the amplitude equations, the
EOM analogue), with one front end.

| What | Where |
|---|---|
| definitions, dense reference of all six models, their Jacobians, the folded eigenproblem | `src/zccsdt/cc/ccn_ref.py`, `tests/test_ccn_ref.py` |
| CC2 on packed amplitudes; CC3 with T3 stored: a switch of the CCSDT step | `src/zccsdt/cc/ccn.py`; `ccsdt_opt.update_amps(model='cc3')`, `solver.kernel(eris, 'cc2' / 'cc3')` |
| Jacobians of CC2 and CC3 on packed amplitudes (label shifts, IP / EA as in EOM-CCSDT) | `eom_opt.prepare(model=)`, `eom_opt.sigma`, `eom.kernel(model=)` |
| front end | `zccsdt.CC2`, `zccsdt.CCSD`, `zccsdt.CC3` (`src/zccsdt/ccn.py`; `docs/usage.md`) |
| tests | `tests/test_ccn_ref.py`, `tests/test_ccn.py` |

Notation as in `docs/factorization.md` and `docs/eom.md`: `o`, `v` occupied
and virtual spin orbitals or spinors; i j k l m n occupied, a b c d e f
virtual; `P`, `Q` pairs i<j, a<b; `<pq||rs>` antisymmetrized integrals;
`H~ = exp(-T1) H exp(T1) = F~ + V~` the T1-dressed Hamiltonian with the
integrals `<pq||rs>~` and the Fock matrix `f~` of `ccsdt._t1_dress`.  In
that formalism the cluster operator of the equations is T2 + T3 + T4.

## 1. Definitions

n is the highest rank of a model: 2, 3, 4 for CC2, CC3, CC4.  **The
equations of the ranks below n are those of the full model of rank n**
(CCSD, CCSDT, CCSDTQ) - with every term of T_n in them.  The equation of
rank n is kept to the lowest order in the fluctuation potential, T1 counted
as of order zero (Christiansen, Koch, Jorgensen for CC2 and CC3; Kallay and
Gauss for CC4, the model MRCC calls CC4):

    CC2:  <D| V~ + [F, T2] |0> = 0
    CC3:  <T| [V~, T2] + [F, T3] |0> = 0
    CC4:  <Q| [V~, T3] + 1/2 [[V~, T2], T2] + [F, T4] |0> = 0

V~ is the two-body part of H~.  (The one-body part of H~ cannot reach
these projections with the amplitudes that are left: `[F~, T2]` ends at
doubles, `[[F~, T2], T2]` and `[F~, T3]` at triples.)

**What F is.**  F is the Fock operator of the reference determinant **as
the integral container has it**: `sum_pq f[p,q] p+ q` with the matrix
`eris.fock`.

- It is not dressed with T1.  `[F, T_n]` projected on rank n takes the
  occupied-occupied and the virtual-virtual block of f only:

      <mu_n| [F, T_n] |0>  =  P f[a,e] t[..e..] - P f[m,i] t[..m..],

  the terms `'R2: [F,T2]'`, `'R3: [F,T3]'`, `'R4: [F,T4]'` of
  `ccsdtq_terms.GROUPS` with f in the place of f~.
- It is not assumed diagonal: with the full blocks the models are
  invariant under rotations among the occupied and among the virtual
  spinors (the dense reference keeps the full blocks; `tests/test_ccn.py`,
  `test_spinors_that_are_not_canonical`: energies and roots to 1e-10).
- **The production code runs on semicanonical spinors**, where F_oo and
  F_vv are exactly diagonal and [F, T_n] is the orbital-energy
  denominator.  Whenever an off-diagonal element of F_oo or F_vv of the
  active spinors is not exactly zero, they are rotated - F_oo and F_vv
  diagonalized group by group (the symmetry labels of the iterations, the
  spin blocks of spin orbitals of pure spin: the connected components of
  the nonzero elements), so labels and real arithmetic stay; frozen
  spinors are not touched (`zccsdt.cc.ccn.semicanonical_rotation`):
  - the front end (`GCC2`, `GCC3`) rotates the spinors before the
    integrals are made (`ccn.semicanonical_columns`: one Fock build), and
    `cc.semicanonical` holds the rotation; amplitudes and vectors refer to
    the rotated spinors, `cc._eris` to their integrals;
  - `solver.kernel(eris, 'cc2' | 'cc3')` on given integrals turns the
    container (`ccn.semicanonicalize`: every block, `<ab||cd>` on pairs
    unpacked to v^4 numbers) and returns `Result.eris` and
    `Result.rotation`; `ccn.back_t1`, `back_t2` turn amplitudes back.
  The noise this removes in a converged SCF is small but not zero: max
  |F_oo|, |F_vv| off the diagonal 3e-10 for HBr / cc-pVDZ with
  conv_tol 1e-10 and conv_tol_grad 1e-8, both for socutils' SpinorSCF with
  X2CAMF and for PySCF's x2c.UHF (the numbers came out the same); up to
  1e-5 for looser SCFs.  CFOUR's spinors (`zccsdt.cfour`) are read with
  the Fock matrix as the diagonal of their energies: exactly canonical,
  nothing is rotated.  A converged SCF and the same SCF semicanonicalized
  give the same energies to 1e-12 (`tests/test_cc3fly.py`).
- **Hartree-Fock reference** (f[i,a] = 0, every case of the front end):
  nothing else can be meant.  `exp(-T1) F exp(T1) = F + [F, T1]` then, and
  `[F, T1]` is a pure excitation operator, which commutes with T_n.
- **Another reference** (f[i,a] not zero; ROHF, Brueckner or
  Kohn-Sham orbitals): two definitions are in use for the triples of CC3
  and their analogues.  (a) F as above, f[i,a] counted as a perturbation
  of first order like V - the terms `f[m,e] t1[i,e] t3[mjk,abc]` of
  `[[F, T1], T3]` are then of third order and dropped; (b) F~1 =
  exp(-T1) F exp(T1), which keeps them (blocks `f[m,i] + f[m,e] t1[i,e]`
  and `f[a,e] - t1[m,a] f[m,e]`).  Implemented in production: (a).  The
  dense reference has both and a third for comparison
  (`ccn_ref.residuals(fock_top='bare' | 'one' | 'dressed')`; 'dressed' is
  the Fock matrix f~ of H~, which also contains `[[V, T1], T_n]` and is no
  one's definition), and `tests/test_ccn_ref.py` checks (a) and (b) as
  commutators of second-quantized operators for a Fock matrix with all
  blocks.  **What MRCC does for such references could not be established:
  its release of 2018 refuses them** - "CCn methods are not implemented
  for ROHF reference! Use UHF orbitals!" - and with UHF orbitals f[i,a] is
  zero.  The same message stops every CCn calculation through MRCC's
  interface to DIRAC (`iface=dirac`), whose reference it takes for an
  open-shell one: section 7.

**Three remarks.**

- The energy is that of CCSD, `f[i,a] t1 + 1/4 <ij||ab> (t2 + t1 t1)`.
- CC3 does not drop the T3 terms of the singles and doubles equations and
  CC4 not the T4 terms of the doubles and triples equations; what is
  dropped is everything of the rank-n equation beyond the lowest order.
- With three electrons CC3 is not exact in general (CCSDT is): H3 / 6-31G
  misses 1e-6 Eh.  In a basis where no triple excitation conserves S_z
  (H3 / STO-3G: 2 alpha + 1 beta occupied, 1 alpha + 2 beta virtual spin
  orbitals; Li / STO-3G by the angular momentum besides) T3 is zero and
  the two agree - a trap for a test.

## 2. The equations, term by term

The terms of `ccsdtq_terms.GROUPS` (coefficient times the antisymmetrizer
of a representative product; 5 + 14 + 22 + 34 terms for R1 .. R4 of CCSDTQ)
that the rank-n equation of each model keeps, and what the existing
amplitude step computes for them:

**CC2, doubles** (3 of 14 terms).

    R2[ij,ab] = <ab||ij>~ + P(ab) f[b,e] t2[ij,ae] - P(ij) f[m,j] t2[im,ab]

dropped: `[V~, T2]` (pp ladder, hh ladder, ring), `1/2 [[V~, T2], T2]`,
the T2 parts of F_vv and F_oo, the dressing of f.  In the layouts of
`ccsd_residual_packed`: the fully dressed `<ab||ij>~` is `dressed_vvoo_pp`
plus the ladder of the packed dressed `<ab||de>~` with
`tau1 = t1 t1 - t1 t1` alone (not with t2 + tau1); the two Fock terms act
with the bare blocks (`ccn.cc2_doubles`).

**CC3, triples** (4 of 22 terms).

    R3[ijk,abc] = A[1/4 <bc||ek>~ t2[ij,ae] - 1/4 <mc||jk>~ t2[im,ab]]              the drive terms, [V~, T2]
                + P(c/ab) f[c,e] t3[ijk,abe] - P(k/ij) f[m,k] t3[ijm,abc]           [F, T3]

(A: the antisymmetrizer over ijk and over abc.)

In the factorization of the CCSDT step (`docs/factorization.md`, section 2):

| Term of the CCSDT triples equation | CC3 |
|---|---|
| drive `W_vvvo x t2`, `W_vooo x t2` | kept, with `W_vvvo = <ab\|\|ei>~` and `W_vooo = <ma\|\|ij>~`: the dressed integrals alone, without their T2 terms (`1/2 [[V~,T2],T2]`, `1/2 [[F~,T2],T2]`) and without their T3 dressing from pass 1 (`[[V,T2],T3]`) |
| ring (`W_ovvo` on T3) | dropped |
| pp ladder with F_vv folded in | the ladder is dropped; of F_vv the bare f[a,e] stays |
| hh ladder with F_oo folded in | the ladder is dropped; of F_oo the bare f[m,i] stays |
| pass 1: T3 -> T1, T2 | kept (it belongs to the singles and doubles equations) |
| pass 1: T3 -> W_vvvo, W_vooo | dropped |

**CC4, quadruples** (7 of 34 terms).

    R4 = A[1/24 <cd||el>~ t3[ijk,abe] - 1/24 <md||kl>~ t3[ijm,abc]]                    [V~, T3]
       + A[1/16 <bc||ef>~ t2[ij,ae] t2[kl,fd] + 1/16 <mn||jk>~ t2[im,ab] t2[nl,cd]
           + 1/4 <mb||ek>~ t2[ij,ae] t2[ml,cd]]                                          1/2 [[V~, T2], T2]
       + A[1/144 f[d,e] t4[ijkl,abce] - 1/144 f[m,l] t4[ijkm,abcd]]                      [F, T4]

dropped: `[V~, T4]` (three terms), `[[F~, T2], T3]`, `[[V~, T2], T3]`,
`[[V~, T2], T4]`, `1/2 [[V~, T3], T3]`, `1/6 [[[V~, T2], T2], T2]`.  The
right-hand side is the quantity of the (Q) correction - `[V, T3] +
1/2 [[V, T2], T2]` projected on quadruples (`zccsdt.cc.quadruples`: Y4 and
Z4 per occupied quadruple) - with the dressed integrals in the place of
the bare ones.  The T4 it gives enters the doubles equation
(`1/16 <mn||ef> t4[ijmn,abef]`) and the triples equation (`[F~, T4]`,
`[V~, T4]`: three terms), exactly as in CCSDTQ.

## 3. The Jacobian

The excitation energies of linear response are the eigenvalues of the
Jacobian A of the amplitude equations at the converged amplitudes (for a
full model: the EOM-CC matrix, `docs/eom.md`, section 1),

    sigma_mu(R) = d/dx Res_mu(T + x R) at x = 0,     R = R1 + .. + Rn.

With `H' = [H~, R1]` (the dressed integrals with one index transformed by
r1, `eom_ref.dressed_derivative`) it is: the terms of the residual with
the integrals of H', plus the terms with every amplitude factor replaced
in turn by r of its rank.  The bare F of the rank-n equation has no
derivative.  For the rank-n equations:

    CC2:  sigma_2 = <D| V' + [F, R2] |0>
    CC3:  sigma_3 = <T| [V', T2] + [V~, R2] + [F, R3] |0>
    CC4:  sigma_4 = <Q| [V', T3] + [V~, R3] + [[V', T2], T2]/2 + [[V~, R2], T2] + [F, R4] |0>

and the lower ranks are the rows of the EOM-CCSD / CCSDT / CCSDTQ matrix,
at the amplitudes of the model.  The matrix has the block structure
(P: the ranks below n, Q: rank n)

    A = | A_PP   A_PQ |        A_QQ = <mu_n| [F, tau_nu_n] |0>:
        | A_QP   A_QQ |        the differences of the orbital energies for canonical spinors,

so it is an ordinary matrix on the full space and the Davidson solver of
EOM-CCSDT applies as it is, with R_n stored.

**On the factorization of the CCSDT step** (`eom_opt.sigma`, `g.model =
'cc3'`): sigma_1 and sigma_2 are those of EOM-CCSDT (`docs/eom.md`,
section 3; pass 1 over R3 with H~, over T3 with H').  sigma_3 is two calls
of the engine's residual without its ring and ladder operators,

    sigma_3 = residual(R3; -, f_vv, f_oo | <ab||ei>' x t2, <ma||ij>' x t2)
            + residual( - ; -, -, -      | <ab||ei>~ x r2, <ma||ij>~ x r2),

the second of which does not read T3 at all.  `<ab||ei>'` =
`-P(ab) r1[m,a] <mb||ei>~ + <ab||ef>~ r1[i,f]`; the second part is taken
from the kept ladder operator of the ground state (as in EOM-CCSDT), whose
T2 part `t2[mn,ab] <mn||ef> r1[i,f]` and F_vv part are taken out again.
The engine functions got a switch for this and nothing else: an operator
passed as `None` is left out (`DenseT3.residual`, `SpinBlockT3`,
`t3shift.ShiftT3.residual`); the drive terms without the ring term, which
never gather T3, are the kernel `zc_t3_drive` (dense layout) and the
existing drive products (blocks); `[F, T3]` is a pass
`R += (e_abc - e_ijk) T3` (`zc_t3_fdiag`): the spinors are semicanonical
(section 1), and the sigma vector likewise.

CC2 (`g.model = 'cc2'`): sigma_1 as EOM-CCSD; sigma_2 = `<ab||ij>'` -
the bra part `-P(ab) r1[m,a] <mb||ij>~`, the ket part with the bra-dressed
`<ab||ei>` and the ladder of `<ab||de>~` on the derivative of tau1 - plus
the two bare Fock terms on r2.

## 4. The folded eigenproblem

Rank n can be eliminated: from the Q rows, `R_Q = (omega - A_QQ)^-1 A_QP
R_P`, and

    [A_PP + A_PQ (omega - A_QQ)^-1 A_QP] R_P = omega R_P.

The matrix depends on omega; an eigenvalue of the full A is a fixed point
omega = lambda(omega) of an eigenvalue of the folded matrix, found by
iteration per root (`ccn_ref.folded`, `folded_root`; the rate of
convergence is the weight of rank n in the state, a few percent for a
state dominated by single excitations).  `A_QQ` is diagonal for canonical
spinors, so `R_Q` is made element by element, like the amplitudes T_n of
the ground state,

    CC3:  t3[ijk,abc] = drive[t2; V~] / (e_ijk - e_abc),
          r3[ijk,abc] = (drive[t2; V'] + drive[r2; V~]) / (omega - (e_abc - e_ijk)),

per occupied triple (quadruple), contracted into the equations of the
lower ranks and discarded: neither T_n nor R_n is stored.  The two forms
give the same numbers (`tests/test_ccn_ref.py`, `test_folded_eigenproblem`:
1e-10 for CC2, CC3 and CC4).  The spinors are semicanonical (section 1),
so A_QQ is diagonal.

## 5. Scaling and storage

| | iteration | sigma vector | stored |
|---|---|---|---|
| CC2, this code | as CCSD (o^2 v^4 from the dressing and the ladder of `<ab\|\|de>~` on tau1) | as EOM-CCSD | the integrals of CCSD with `<ab\|\|cd>` on pairs, T2 |
| CC2, AO-direct / Cholesky (step 2b) | o^2 v^2 n_chol | the same | o v n_chol |
| CC3, T3 stored | o^3 v^4 (drive, pass 1) + o^3 v^3 passes; no o^4 v^4, o^3 v^5 | 2 drive passes, 2 passes 1 | T3 (C(o,3) C(v,3)), the operators of CCSDT |
| CC3, folded | o^3 v^4 | o^3 v^4, per omega | as CCSD |
| CCSDT | o^4 v^4 / 4 + o^3 v^5 / 24 + .. | two iterations | T3 |
| CC4, T4 stored | o^4 v^5 (drive) + the T4 terms of the lower equations | | T4 |
| CC4, folded | o^4 v^5, as (Q) with the back-contraction | | T3 |
| CCSDTQ | o^5 v^5, o^4 v^6 | | T4 |

## 6. Symmetry labels and real arithmetic

Nothing of the models touches the label structure: every term kept is a
term of the full model, with the same operands.

- **CC3 with T3 stored** runs in the blocks of the labels of a linear
  molecule (`t3engine.SpinBlockT3`) and in real arithmetic for m_j-adapted
  spinors, as CCSDT does; the excitation energies in the blocks of a label
  shift (`eom_ee(shift=)`, `t3shift`), IP and EA with short vectors.
- **CC2** (and the CCSD class of the ladder) has no rank-3 array; the
  integrals and T2 are dense, as in the CCSDT step.  A label shift selects
  states by masks (`eom.Sector`), IP and EA use the ghost spinor at full
  length, which is cheap without triples.
- The Fock operator of the rank-n equation is block diagonal in the labels
  like every operator of shift 0.

## 7. Validation

**Reference programs: what they have.**

| Program | Ground state | Excitation energies |
|---|---|---|
| MRCC (release of February 2018), its own closed-shell and UHF code | `calc=CC3`, `CC4` (also CCSDT-1a/b, CCSDT-3). `calc=CC2` stops in its integral program here (density fitting); through the interface it runs | none: "Approximate CC methods are not implemented for excited states!" |
| MRCC through its interface to DIRAC (`iface=dirac`: the spinor reference of `docs/benchmarks.md`) | CC2 runs. **CC3, CC4: refused** - "CCn methods are not implemented for ROHF reference! Use UHF orbitals!" - whatever the flags of `fort.56` (HF = 1; "spatial" = 0 ends in a crash) | none |
| CFOUR (`xvcc`, `xvee`; the build of `~/apps/cfour-x2cso`), nonrelativistic closed shell | CC2, CC3 | EOMEE-CC2, EOMEE-CC3 ("CC3-LR") |

So for spinors there is no reference program for CC3 and CC4: the spinor
code is checked against the dense reference - whose definitions are
checked as commutators in a Fock space with a complex Hamiltonian - and by
the equality of the real code in label blocks with the packed block.

**Definitions** (`tests/test_ccn_ref.py`).

| Check | Agreement |
|---|---|
| the rank-n equation of CC2, CC3, CC4 against `<mu_n\| [F, T_n] + [V~, T_(n-1)] (+ ..) \|0>` of second-quantized operators (complex Hamiltonian with a full Fock matrix, complex amplitudes that solve nothing; 3 + 4 and 4 + 4 spin orbitals); the variant with exp(-T1) F exp(T1) likewise; the lower equations against the projections of exp(-T) H exp(T) | 1e-12 |
| the Jacobian against the exact directional derivative of the residuals (five-point formula) | 1e-10 |
| two electrons (He / 6-31G): CCSD = FCI, CC2 not; three electrons (H3 / 6-31G): CCSDT = FCI for the ground state and all excitation energies, CC3 not | 1e-10, 1e-8 |
| the folded eigenproblem iterated to self-consistency against the eigenvalues of the full Jacobian (H2O / STO-3G for CC2, CC3; a model Hamiltonian with a non-diagonal Fock matrix for CC4) | 1e-10 |
| H2O / STO-3G, all electrons: MRCC's closed-shell CC2 (through the interface), CC3, CC4 and the full models | 1e-10 Eh (observed 1e-12) |

**Production code** (`tests/test_ccn.py`).

| Check | Agreement |
|---|---|
| CC2, CC3 ground states on packed amplitudes against the dense reference; LiH / STO-3G, real and complex (spin orbitals mixed by SU(2) matrices) | 1e-11 Eh |
| sigma vectors of CC2 and CC3 at random amplitudes against the reference; LiH (4 + 8), H2O frozen core (8 + 4); real, complex | 1e-12 of the largest element (observed 1e-16) |
| spin orbitals rotated among the occupied and among the virtual ones: energy, excitation energies | 1e-10, 1e-9 |
| H2O / 6-31G, RHF as spin orbitals (10 + 16): total energies of CC2, CCSD, CC3 against CFOUR; CC3 against MRCC | 1e-12 Eh (table below) |
| the same: the five lowest singlet excitation energies of CC2, CCSD, CC3 against CFOUR's `xvee` (ESTATE_CONV=9; zccsdt residual 1e-8) | 5e-11 Eh (table below) |
| LiH / 6-31G, X2C spinors (4 + 18): CC3 in label blocks (868 of 3264 triples, real) against the packed block: energy; roots of the shifts 0, 2, 4; IP and EA with short vectors against the ghost spinor at full length | 1e-11 Eh; 1e-8 (threshold of the residual) |

H2O / 6-31G at the geometry of `bench/refs/mrcc/h2o_631g/MINP.eom.bohr`,
basis of `bench/refs/GENBAS` (CFOUR's entries, and the data MRCC has
itself: the SCF energy is -75.983948495280 in the three programs), all
electrons:

| | CFOUR | MRCC | zccsdt | difference |
|---|---|---|---|---|
| E(CC2) | -76.113595476238 | (-76.113595478917 through the interface, at the geometry in Angstrom) | -76.113595476238 | 4e-13 |
| E(CCSD) | -76.119346380612 | -76.119346380614 | -76.119346380613 | 1e-12 |
| E(CC3) | -76.120380732032 | -76.120380732033 | -76.120380732033 | 7e-13 |

| singlet | CC2: CFOUR | zccsdt - CFOUR | CCSD: CFOUR | zccsdt - CFOUR | CC3: CFOUR | zccsdt - CFOUR |
|---|---|---|---|---|---|---|
| 1 | 0.307623345946 | -1.1e-11 | 0.308197284210 | -1.9e-11 | 0.311525887921 | -1.3e-11 |
| 2 | 0.392332664219 | -8.2e-12 | 0.391939281735 | -1.5e-11 | 0.394096015139 | 1.8e-11 |
| 3 | 0.399064742585 | -8.6e-12 | 0.401631424467 | 1.6e-11 | 0.404075925693 | -1.5e-11 |
| 4 | 0.491210789055 | -4.0e-12 | 0.491507715334 | -5.4e-11 | 0.492790540247 | -3.4e-11 |
| 5 | 0.566560552152 | 1.3e-12 | 0.568321212557 | 1.2e-11 | 0.568464049362 | -3.7e-11 |

(Hartree.  zccsdt: `zccsdt.CC2 / CCSD / CC3(mf)`, `eom_ee(nroots=16,
shift=0)` - the singlets and the M_S = 0 components of the triplets of the
spin orbitals; the roots matched to CFOUR's, which are singlets.  H2O /
STO-3G with CFOUR's basis entries: the same to 2e-10, the CC3 ground state
to 1e-12.)

## 8. CC2 of spinors on Cholesky vectors (prototype)

`zccsdt.zccsd.cc2` (`DirectZCCSD(ladder='chol').cc2()`, `.cc2_ee()`): T1 in
the coefficients, the dressed three-index integrals `B[c,i,a] = L~[c,a,i]`
from one half-transformation of the scalar AO Cholesky vectors per
iteration, the doubles `t2 = antisym[B B] / D` per occupied index, used for
`Y[c,i,a] = sum_jb t2[ij,ab] L[c,j,b]` and discarded; excitation energies
from the folded sigma vector with one-index transformed vectors.  The
equations and the layouts are in the module.  **State: validated, not
profiled.**  With vectors that reproduce the AO integrals to rounding it is
the CC2 of sections 1 to 3: energy to 1e-11 Eh, sigma vectors to 1e-9 of
their size, roots to 1e-9 (`tests/test_zccsd_cc2.py`; LiH / 6-31G, X2C);
at the default `max_error = 1e-5` of the class the decomposition costs
2e-7 Eh in the energy and 5e-7 Eh in the excitation energies there.  **Decided for the production route** (the user, 10-12): the SCF and the
CC use the same Cholesky vectors - a CD-HF reference, as CFOUR's chain
`xsox2chfcd` -> `x2ccccd` does (socutils' SCF with those vectors, or
CFOUR's CD-HF spinors), so that the reference Fock matrix is that of the
Hamiltonian the correlation sees - with the semicanonical spinors of
section 1.  The prototype's mix, the Fock matrix of the exact SCF with
Cholesky integrals for the correlation, is not to be kept.  Not
done: the production standard (profile at 30 + 400 spinors, Kramers pairs,
three real products per complex one, the memory plan checked on a large
case), an integral-direct variant without Cholesky vectors, IP and EA.
CFOUR's spinor programs (`x2ccccd`, `x2cccsd`, `x2ceomcc`, `x2ceomcccd`,
`x2cvtrancd`, `x2chbarcd`) have no CC2: the string does not occur in their
sources.

## 9. CC3 without stored triples

`zccsdt.cc.cc3fly` (scheduler) and `csrc/cc3_fly.c` (kernel);
`zccsdt.CC3(mf)` with `store_t3 = False`,
`solver.kernel(eris, 'cc3', Options(store_t3=False))`,
`ccsdt_opt.update_amps(opts, t1, t2, None, eris, model='cc3')`,
`eom_opt.prepare(.., model='cc3', fly=True)` and `sigma(.., omega=)`,
`ccn.eom_folded`.

**The equations.**  For semicanonical spinors (section 1) the triples equation of CC3 is
solved in closed form, with the operands of the drive terms of the CCSDT
step (`docs/factorization.md` 2.2),

    Z[P, c, f, Q]  = - sum_e t2[P, e c] <Q||e f>~ - sum_m <m c||P>~ t2[m f, Q]        P = (i<j), Q = (a<b)
    t3[ijk, abc]   = P(i/jk) P(a/bc) Z / (e_i + e_j + e_k - e_a - e_b - e_c)

and enters the singles and doubles equations through the four sums of
pass 1 (`docs/factorization.md` 2.4), with T[P, d, m, Q] the half-expanded
triples (the triple (m; P) in its three ways, signs +, -, +):

    Y[P, d, y]  += T[P, d, m, Q] <y m||Q>~         -> the term 1/2 P(ab) <bm||ef>~ t3[ijm, aef] of R2
    X[x, m, Q]  += <P||d x>~ T[P, d, m, Q]         -> the term -1/2 P(ij) <mn||je>~ t3[imn, abe]
    r1[m, d]    += T[P, d, m, Q] <P||Q>            -> R1
    rA[P, Q]    += f~[m, d] T[P, d, m, Q]          -> R2

(the two other sums of pass 1 - the T3 parts of W_vvvo and W_vooo - are
not in CC3).  Everything else of the step is the CCSD part of the CCSDT
step, unchanged.

**The batch.**  The occupied triples (i, j, k) with i0 <= i < i1 <= j < k:
one pair (j < k) and a range of the third index below it, all of one
symmetry class.  For such a batch the three pieces of Z that its triples
need - Z[(jk), c, i, Q], Z[(ik), c, j, Q], Z[(ij), c, k, Q] - are exactly
three products per i and virtual class, with operands that are sub-blocks
of four arrays made once per iteration (no copy per batch), and nothing is
computed twice: 3 (c, Q)-slices per triple.  The batch then is

1. the drive products into three pieces `[i][H]` of the work array, H the
   number of (c, Q) of the group of the batch;
2. the amplitudes `t3[i][A]`: the antisymmetrized sum of the three pieces
   over the three ways of the virtual triple, over the denominator;
3. the gather `T[i][d, Q] = +- t3[i][A(d, Q)]` over the first piece;
4. the contractions of the four sums, one round per i.

The products of a stage are cut into pieces of 96 columns (or 16 rows) and
shared by the threads from one queue, a sequential GEMM per piece: one
code for the few large products of the dense layout and the many small
ones of the blocks.

**Symmetry labels.**  The operands are dense arrays in the order of the
classes; the triples exist only in the blocks the labels allow: for a batch
of group g (the sum of the classes of i, j, k) the virtual triples of group
g, the (c, Q) with class(c) + group(Q) = g, and in every product the
ranges of the contracted index that the labels leave.  The layout without
labels is the case of one class.  Real arithmetic where the integrals are
real.

**Work and memory.**  3 o^3 v^4 / 6 multiply-adds for the drive products
(e part; o^4 v^3 / 2 for the m part) and as much for the contractions:
o^3 v^4 in all, in blocks less by the fraction of allowed blocks.  Memory:
four drive operands (2 o^2 v^2 + o^2 v o + (v + o) o v^2 / 2), four
contraction operands (o v^3 / 2 + ..), the four sums (o^2 v^2 + 2 o^2 v^2 /
2 + o v), and the work array, `ni (3 H + n_A)` elements for a batch of ni
occupied triples - at least one triple, `10 C(v,3)` elements without
labels, as many as `mem_mb` (2000 MB) holds otherwise.  No array of the
size of T3: with 26 + 120 spinors T3 is 5.8 GB (real, packed) and one
triple of work 23 MB.

**Excitation energies.**  The folded eigenproblem of section 4 in the
space of singles and doubles.  A sigma vector at the frequency omega is the
EOM-CCSD-like part (the code of `eom_opt.sigma` for a Ground without T3)
plus two passes over the batches:

    r3 = (drive[t2; <ab||ei>', <ma||ij>'] + drive[r2; <ab||ei>~, <ma||ij>~]) / (omega + e_ijk - e_abc),
         contracted with the integrals of H~ (the four sums): sigma_1, sigma_2;
    t3 of the ground state, made again, contracted with the integrals of H' = [H~, R1]
         (<mn||ei>', <am||ef>', f'[m,e]): sigma_2.

`ccn.eom_folded`: the roots of the singles-doubles block (no triples work)
to 1e-4 as the start, then root by root `ccn.folded_roots`: Davidson
iterations at a fixed frequency (`eom.davidson`, started from the vector
of the step before, with a threshold that follows |lambda(omega) - omega|
down to `conv_tol`), the frequency moved by the secant step for
lambda(omega) = omega, until |lambda(omega) - omega| < `omega_tol`
(default conv_tol / 100) with the residual below `conv_tol`.  The totally
symmetric states (label shift 0); states with a shift, IP and EA are with
T3 stored.

**Validation** (`tests/test_cc3fly.py`).

| Check | Agreement |
|---|---|
| the step at amplitudes that solve nothing against the step with the stored T3 of the closed form; H2O (8 + 4), LiH (4 + 8); real, complex; one occupied triple per batch and all in one | 1e-12 |
| ground state against T3 stored: the same, and LiH / 6-31G with X2C spinors (4 + 18) on the packed layout and in the blocks of 6 classes | 1e-12 Eh (observed 8e-15) |
| the folded sigma vector against the Jacobian with R3 stored, folded by hand (two of its sigma vectors and a division), at two frequencies; all of the cases above | 1e-11 of the largest element (observed 1e-14) |
| the three lowest roots (totally symmetric states in blocks) against `eom_ee` with R3 stored; conv_tol 1e-9 | 1e-9 Eh (observed 1e-11) |
| spinors rotated among the occupied and among the virtual ones (the given integrals made semicanonical by solver.kernel): energy, roots of the folded problem | 1e-10, 1e-9 |
| an SCF converged loosely (noise of 1e-5 in F_oo, F_vv): the front end (spinors rotated before the integrals), solver.kernel on the integrals of the SCF (container turned), label blocks and packed, T3 stored and not | 1e-12 Eh |

**Timings** (job 31901611, node c521, 2 x 24 cores Xeon Gold 6248R, 24
threads, MKL; `bench/ccn/cc3_bench.py`, commit 63ee1d2; seconds per
iteration after the first, peak memory of the iterations; a sigma vector:
with R3 stored one sigma vector of EOM-CC3 / EOM-CCSDT, without stored
triples one folded sigma vector at a frequency):

| | o + v | CC3, no T3 | CC3, T3 stored | CCSDT | CCSD of DirectZCCSD; (T) |
|---|---|---|---|---|---|
| HBr / cc-pVQZ-DK, X2C, 10 frozen, lowest 120 virtual; real, label blocks | 26 + 120 | 2.54 s, 4.2 GB | 3.63 s, 6.5 GB | 6.28 s, 6.9 GB | 0.58 s (complex, Kramers); (T) 4.6 s |
| sigma vector | | 4.2 s | 5.4 s | 9.8 s | |
| the same on the packed layout (real) | 26 + 120 | 11.6 s, 5.0 GB | 27.1 s, 21.6 GB | 84.2 s, 22.7 GB | |
| sigma vector | | 23.9 s | 35.7 s | 152 s | |
| H2O / cc-pVTZ, X2C, all electrons; complex, dense | 10 + 106 | 1.42 s, 3.4 GB | 2.43 s, 4.2 GB | 6.58 s, 5.0 GB | 0.04 s; (T) 0.24 s |
| sigma vector | | 2.65 s | 3.97 s | 11.7 s | |

- Without stored triples an iteration of 26 + 120 in blocks is 1.31 s of
  batches (11.7 cores busy on 24) and 1.2 s of the CCSD part of the CCSDT
  step (dressing 0.50, W_vvvo 0.32, the CCSD residual 0.16).  In a folded
  sigma vector (two passes over the batches, three drive terms) the
  batches take 2.38 s: drive products 0.73, contractions 0.71, gather
  0.56, amplitudes 0.35 (the times the kernel reports).  The gather and the
  amplitudes are memory passes of a third of the time: the next thing to
  make faster.
  Against DirectZCCSD, the production CCSD(T) of spinors (complex, Kramers
  pairs, AO ladder), a CC3 iteration costs 4.4 CCSD iterations; CC3 to
  convergence (about 15 iterations) is 8 times CCSD(T) here.
- Threads (whole iteration, CC3 without T3): HBr in blocks 13.6 / 3.28 /
  2.54 s on 1 / 8 / 24 threads (5.4 times faster on 24: the small products
  of the blocks and the CCSD part); H2O dense complex 18.6 / 2.91 / 1.42 s
  (13 times).
- Without labels the batches are the large products of the dense layout
  and CC3 without T3 is 2.3 times faster than with T3 stored and 7.3 times
  faster than CCSDT, at a quarter of their memory; in the blocks of the
  labels the stored T3 is small (82 million elements) and the gain is 1.4
  and 2.5.

## 10. Porting CC3 to the native program

What a Fortran driver needs for CC3 without stored triples; nothing here
uses Python objects, and the kernel is called by `zccsdt.cc.cc3fly.Fly.run`
exactly as a driver would call it.

**Kernel entry points** (`csrc/cc3_fly.c`, compiled like `ccsdt_t3.c`,
which it needs for `zc_gemm_d` / `zc_gemm_z`; add `cc3_fly` to `KERNELS`
of `cfour/xzccsdt/Makefile`).  All integers are `int64_t` passed by value,
all arrays row-major (the last index runs fastest: as Fortran arrays, the
indices reversed), `T` = `double` (`_d`) or `double complex` (`_z`):

    int64_t zc_cc3_work(ncls, g, ni, vrng, qoff, aoff)
        elements of work for a batch of ni occupied triples of group g

    void zc_cc3_batch_{d,z}(
        no, nv, nQ, nA, ncls,                  sizes: spinors, virtual pairs and triples, classes
        orng[ncls+1], vrng[ncls+1],            first spinor of every class (occupied, virtual)
        qoff[2 ncls], aoff[3 ncls - 1],        first virtual pair / triple of every group
        vtri[nv*nQ], vsgn[nv*nQ] (int8),       (d, Q) -> triple and sign       (TripleMap.tri_c, sgn_c)
        vfree[3*nA], vpair[3*nA],              triple -> its three (free; pair) (TripleMap.free_c, pair_c)
        eo[no], ev3[nA], omega,                denominator: eo[i] + eo[j] + eo[k] - ev3[A] + omega
        i0, i1, j, k,                          the batch: i0 <= i < i1 <= j < k, i of one class
        nset,                                  1 or 2 drive terms (summed)
        At1, Ao1, Bv1, Bo1,  At2, Ao2, Bv2, Bo2,   the drive operands (set 2 is not read if nset = 1)
        Gv, Gsq, gsq, fov,                     the contraction operands
        Ysq, Xo, r1, rAsq,                     the four sums, accumulated
        work, stats[4])                        work array; seconds: products, amplitudes, gather, contractions

The tables are those the native program has for its T3 blocks (the maps of
`xzmaps.F90` correspond to `TripleMap`); without labels ncls = 1, orng =
(0, no), vrng = (0, nv), qoff = (0, nQ), aoff = (0, nA).

| Operand | Shape (row-major) | Content | From |
|---|---|---|---|
| `At` | [no][no][nv][nv] | `At[hi][lo][c][e] = a2[lo hi, e c]` (hi > lo read) | T2 (or r2) |
| `Ao` | [no][no][nv][no] | `Ao[hi][lo][c][m] = <m c\|\|lo hi>~` | dressed `<ma\|\|ij>` |
| `Bv` | [nv][no][nQ] | `Bv[e][i][Q] = <Q\|\|e i>~` | dressed `<ab\|\|ei>` on pairs a<b |
| `Bo` | [no][no][nQ] | `Bo[m][i][Q] = a2[m i, Q]` | T2 (or r2) on pairs a<b |
| `Gv` | [no][nQ][nv] | `Gv[m][Q][y] = <y m\|\|Q>~` | dressed `<am\|\|ef>` on pairs e<f |
| `Gsq` | [no][no][no][nv] | `Gsq[hi][lo][x][d] = <lo hi\|\|d x>~` | dressed `<mn\|\|ei>` |
| `gsq` | [no][no][nQ] | `gsq[hi][lo][Q] = <lo hi\|\|Q>` | `<mn\|\|ef>` |
| `fov` | [no][nv] | `f~[m][d]` | dressed Fock matrix |
| `Ysq` | [no][no][nv][nv] | `Ysq[hi][lo][d][y]` | out |
| `Xo` | [no][no][nQ] | `Xo[x][m][Q]` | out |
| `r1` | [no][nv] | R1 from T3 | out |
| `rAsq` | [no][no][nQ] | `rAsq[hi][lo][Q]` | out |

**Which integrals.**  Classes with at most three virtual indices in the
triples part: the dressed `<ab||ei>` (v^3 o / 2: built from `<ab||cd>` with
T1, as W_vvvo of the CCSDT step without its T2 and T3 terms), `<am||ef>~`
(v^3 o / 2), `<ma||ij>~`, `<mn||ei>~`, `<mn||ef>`, f~.  `<ab||cd>` enters
only through the T1 dressing of `<ab||ei>` and the CCSD part of the step.

**The batch loop** (`cc3fly.Fly.batches`, `run`):

    zero Ysq, Xo, r1, rAsq
    for k = 0 .. no-1, j = 0 .. k-1, class s:
        the i of class s below j, in ranges of at most nwork / per_i(g) (g = s + class(j) + class(k)):
            call zc_cc3_batch(.., i0, i1, j, k, ..)
    fold (o^2 v^2 work; P = (lo < hi), Q = (a < b)):
        R1[i, a]  += r1[i][a]
        R2[P, Q]  += rAsq[hi][lo][Q] + Ysq[hi][lo][a][b] - Ysq[hi][lo][b][a] + Xo[hi][lo][Q] - Xo[lo][hi][Q]

The batches write overlapping parts of the four sums, so the loop over
batches is sequential and the threads work inside a batch (ranks of an MPI
run: each rank its own four sums over its share of the pairs (j, k),
summed at the end - the sums are of the size of T2).

**Memory plan in terms of a work array.**  Fixed per iteration: the eight
operands and four sums of the table (2 (o^2 v^2 + o^3 v) + (v + o) o nQ +
o nQ v + o^3 v + o^2 nQ + o v, and 2 o^2 nQ + o^2 v^2 + o v for the sums).
Per batch: `ni (3 H_g + nA_g)` elements, H_g = sum over the virtual classes
s of n_s nQ(g - s), nA_g the virtual triples of group g - `zc_cc3_work`.
The smallest plan is one triple of the largest group; any larger work
array is used for longer ranges of i (fewer, larger products).  The kernel
allocates only its task list and one integer per (triple, way) of the
group (`3 nA_g` int64) per call.

**What the folded EOM needs from the solver.**

- vectors of singles and doubles only; a sigma routine with one more
  argument, the frequency;
- per sigma vector: `<ab||ei>'` and `<ma||ij>'` (the derivatives of the
  two drive integrals along r1: section 3; `eom_opt.sigma`, the block
  `if cc3:` of the derivative of W_vvvo, and `d_ovoo`), then
  `zc_cc3_batch` over all batches twice: nset = 2 with (t2, `<ma||ij>'`,
  `<ab||ei>'`, t2) and (r2, `<ma||ij>~`, `<ab||ei>~`, r2), omega in the
  denominator, the contraction operands of the ground state -> sigma_1,
  sigma_2; and nset = 1 with the operands of the ground state, omega = 0,
  and the contraction operands of H' (`Gv = -r1[n,y] <n m||Q>`, `Gsq` from
  `<mn||ei>' = <mn||ef> r1[i,f]`, `gsq = 0`, `fov = f'[m,e]`) -> sigma_2
  only;
- an outer loop per root over the frequency (`ccn.folded_roots`): the
  eigensolver at fixed omega started from the last vector, omega replaced
  by the secant step, convergence when the eigenvalue reproduces omega.
  The solver of `xzeoms.inc` serves as the inner solver if it can be
  restarted from one vector with a new matrix; the singles-doubles stage
  it has is the start.
- With T3 stored (the route of `xzccsdt ccsdt_eom`) nothing is folded and
  the existing solver applies as it is: the sigma vector of section 3,
  i.e. the EOM-CCSDT one with the ring and ladder operators left out and
  the drive operators without their T2 and T3 parts.

**Semicanonical spinors.**  The kernel takes the denominators from eo
and ev3 (the diagonal of F) and nothing else of F: the driver must give it
semicanonical spinors.  The native program reads CFOUR's spinors, which
are canonical (the Fock matrix of the files is the diagonal of the SCF
energies; nothing to rotate); a driver that takes spinors from elsewhere
rotates them first, as section 1 describes (diagonalize F_oo and F_vv of
the active spinors per label block, before the integral transformation).

**Checks for the port**: the numbers of sections 7 and 9; for one batch,
the four sums against pass 1 over the stored T3 of the closed form
(`tests/test_cc3fly.py::test_spin_orbitals`).
