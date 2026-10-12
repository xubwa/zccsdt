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
- It is not assumed diagonal.  For canonical spinors the term is the
  orbital-energy denominator; for spinors rotated among the occupied or
  among the virtual ones the off-diagonal elements act on the amplitudes,
  and the energies do not change (`tests/test_ccn.py`,
  `test_spinors_that_are_not_canonical`: 1e-10).
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
`R += (e_abc - e_ijk) T3` (`zc_t3_fdiag`), and what the Fock matrix has
off its diagonal goes through the pp and the occupied operator
(`ccsdt_opt.fock_offdiag`: `None` for canonical spinors).

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
1e-10 for CC2, CC3 and CC4).  For spinors that are not canonical in the
occupied or in the virtual block the folded form first rotates them there
(the models are invariant).

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
