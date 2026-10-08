# The left-hand side of CCSDT: Lambda, densities, orbital gradient, (Q)_Lambda

Equations as implemented in the dense reference (`src/zccsdt/cc/lambda_ref.py`)
from the tables of terms (`src/zccsdt/cc/lambda_terms.py`), for spin orbitals
or spinors, real or complex, with antisymmetrized integrals and no spin
symmetry. This is the ground truth for an implementation on packed
amplitudes; nothing here is fast.

| File | What |
|---|---|
| `src/zccsdt/cc/lambda_terms.py` | the tables `LAMBDA`, `DENSITY`, `GRADIENT` and their dense evaluation |
| `src/zccsdt/cc/lambda_derive.py` | their derivation (symbolic): Wick's theorem and derivatives of the Lagrangian |
| `src/zccsdt/cc/wick.py` | `derive_left`: the left-hand projections |
| `src/zccsdt/cc/lambda_ref.py` | Lambda by Jacobi steps with DIIS, densities, orbital gradient, CCSDT(Q)_Lambda |
| `tests/fockspace.py` (`Left`) | the exact left-hand state in a small Fock space |
| `tests/test_lambda_terms.py`, `tests/test_lambda_ref.py` | the checks listed in section 7 |
| `bench/lambda_ccsdt.py`, `bench/slurm_lambda_refs.sh` | molecules, and the reference runs of MRCC and CFOUR |
| `src/zccsdt/cc/lambda_opt.py`, `csrc/lambda_t3.c` | the Lambda equations on packed Lambda_3 (section 9); `tests/test_lambda_opt.py`, `bench/profile_lambda_ccsdt.py` |

## 1. Definitions and conventions

Notation as in `docs/factorization.md` and `zccsdt.cc.ccsdtq_terms`: i j k ..
occupied, a b c .. virtual, p q r s any; `<pq||rs>` antisymmetrized;
amplitudes `t_k[i.., a..]` antisymmetric in both groups,
`T_k = (1/k!)^2 sum t_k[i.., a..] a+ b+ .. j i`.

The left-hand state is `<0| (1 + Lambda) exp(-T)` with

    Lambda = Lambda_1 + Lambda_2 + Lambda_3,
    Lambda_k = (1/k!)^2 sum l_k[i.., a..] i+ j+ .. b a,

so `<0| Lambda |Phi_ij^ab> = l2[i,j,a,b]`. These are `lambda1[i,a]`,
`lambda2[i,j,a,b]` of `pyscf.cc.gccsd_lambda` and of `zccsdt.zccsd.lam`;
`l_k` is `conj(t_k)` in first order. Nothing is conjugated in any
expression: the Lagrangian

    L = <0| (1 + Lambda) exp(-T) H exp(T) |0>

is holomorphic in the integrals, in t and in lambda. (For a complex
Hermitian Hamiltonian without a symmetry that makes it real the CCSDT
energy has a small imaginary part; the model of the tests has 2e-6 of it.)

The Lambda equations are `dL/dt_k = 0`,

    dL/dt_k[i.., a..] = <0| (1 + Lambda) [Hbar, a+ b+ .. j i] |0>,   Hbar = exp(-T) H exp(T),

which for converged t is `<0| (1 + Lambda) (Hbar - E) |Phi_k> = 0`.
A Jacobi step is `l_k += (dL/dt_k) / D_k`, `D = e_i + .. - e_a - ..`.

Densities: the response (unrelaxed) densities at fixed orbitals, the
reference included, not Hermitian,

    d1[p,q]     = <0| (1 + Lambda) exp(-T) p+ q exp(T) |0>,
    d2[p,q,r,s] = <0| (1 + Lambda) exp(-T) p+ q+ s r exp(T) |0>,

    L = sum h[p,q] d1[p,q] + 1/4 sum <pq||rs> d2[p,q,r,s]   (= E for converged t),
    dE/dx = sum (dh/dx)[p,q] d1[p,q] + 1/4 sum (d<pq||rs>/dx) d2[p,q,r,s].

`h = f - sum_i <pi||qi>` is the one-electron Hamiltonian.

Mapping onto the CCSD code (`zccsdt.zccsd.rdm`, `zvector`, PySCF's
`gccsd_rdm`), which has `D[p,q] = <q+ p>`, `d[p,q,r,s] = <p+ r+ s q>`, both
made Hermitian, and `E = sum h D^T + 1/2 sum (pq|rs) d`:

    D[p,q] = d1[q,p],   d[p,q,r,s] = d2[p,r,q,s],   then (X + X^+)/2

(`lambda_ref.pyscf_rdms`). Without T3 and Lambda_3 the residuals of rank 1
and 2 vanish for PySCF's lambda1 and lambda2, `pyscf_rdms` gives PySCF's
`make_rdm1` and `make_rdm2`, and the blocks of `d1` before it is made
Hermitian are `gamma1` of `zccsd.rdm`:
`doo = d1[o,o]^T - 1`, `dov = d1[v,o]^T (= l1)`, `dvo = d1[o,v]^T`, `dvv = d1[v,v]^T`.

## 2. T1-dressed formalism

As in the amplitude equations, T1 is absorbed in the integrals:
`H~ = exp(-T1) H exp(T1)` has the integrals `h~ = x h y^T` and
`<pq||rs>~` transformed with `x` twice and `y` twice
(`x = 1 - t1^T` in the virtual-occupied block, `y = 1 + t1` in the
occupied-virtual block: `zccsdt.cc.ccsdt._t1_dress`, which also gives the
dressed Fock matrix `f~ = h~ + sum_i <pi||qi>~`). With `T = T2 + T3`:

    L = E0 + 1/4 sum <mn||ef>~ t2[m,n,e,f] + sum_k (1/k!)^2 sum l_k R_k
      = E0 + sum f~[p,q] g1[p,q] + 1/4 sum <pq||rs>~ g2[p,q,r,s],

`R_k` the residuals of `ccsdtq_terms`, `E0 = <0| H~ |0>`. All tables are in
`f~`, `<pq||rs>~`, `t2`, `t3`, `l1`, `l2`, `l3`: **no table has t1 or a bare
integral in it, the Lambda_1 equation included.**

- t2 and t3 are differentiated directly.
- t1 is in the dressed integrals only, and
  `d/dt1[i,a] H~ = [H~, a+_a a_i]`: the derivative is L with the dressed
  integrals one-index transformed (a virtual bra index takes minus the
  values of i, an occupied ket index those of a; `f~` gets `<pi||qa>~`
  besides, `E0` gives `f~[i,a]`). So the Lambda_1 residual is the
  "generalized Fock" contraction of the dressed densities with the dressed
  integrals,

      dL/dt1[i,a] = sum_p h~[p,a] d1~[p,i] - sum_q h~[i,q] d1~[a,q]
                  + 1/2 sum <pq||as>~ d2~[p,q,i,s] - 1/2 sum <iq||rs>~ d2~[a,q,r,s]

  (`lambda_ref.one_index` with `lambda_ref.dressed_densities`), written out
  term by term in `LAMBDA[1]`.
- The densities of the bare integrals are the dressed ones transformed
  back: `d1 = x^T d1~ y`, and `d2` the same on its four indices
  (`lambda_ref.densities`). `d1~ = g1 + (1 on the occupied diagonal)`;
  `d2~ = g2 +` the antisymmetrized products of `g1` with the occupied unit
  matrix (the Fock matrix) `+` the reference.

## 3. The tables

A term is `coefficient * A(product)` for the Lambda equations of rank 2 and
3 (A: all signed permutations of the external occupied and of the external
virtual letters, as in `ccsdtq_terms`) and `coefficient * product`
elsewhere. Numbers of terms:

| Table | Terms | Factors |
|---|---|---|
| `LAMBDA[1]` | 37 | 1 without lambda (f~[i,a]), 6 with l1, 13 with l2, 17 with l3 |
| `LAMBDA[2]` | 35 | 1 without lambda (<ij||ab>~), 3 with l1, 12 with l2, 19 with l3 |
| `LAMBDA[3]` | 16 | 1 with l1, 3 with l2, 12 with l3 |
| `DENSITY` | 38 in 13 blocks | one for every term of the Lagrangian |
| `GRADIENT` | 81 in 11 blocks of integrals | 3 without lambda, 10 with l1, 25 with l2, 43 with l3 |

`DENSITY` by block (external letters = those of the integral):
f~: `ai` 1 (= l1), `ia` 3, `ij` 2, `ab` 2; <pq||rs>~: `abij` 1 (= l2/4),
`ijab` 13, `ijkl` 2, `abcd` 2, `iabj` 2, `ijak` 4, `iajk` 1, `aibc` 4,
`abci` 1.

Two derivations, compared coefficient by coefficient as rational numbers
(`tests/test_lambda_terms.py::test_tables_equal_derivations`):

1. **Wick's theorem** (`wick.derive_left`, `lambda_derive.lambda_by_wick`):
   `[Hbar, tau]` is the connected product with the excitation `tau` as one
   more cluster operator of unit amplitude, and Lambda closes its open
   lines. All contractions of `<0| (1 + Lambda) (H~ tau T ..)_C |0>` for
   tau of rank 1, 2, 3 (rank 1 with the dressed H is the derivative with
   respect to t1) and every product of T2 and T3 that can be connected.
   What was added to `wick.py`: `derive_left`, a few lines on top of
   `derive` (tau as the last cluster operator; Lambda takes the place of
   the projection).
2. **Derivatives of the Lagrangian** built from the table of the amplitude
   equations (`lambda_derive.lambda_by_differentiation`, `one_index`,
   `densities`): a factor t2 or t3 taken out, the integrals one-index
   transformed, or the integral taken out.

Terms are compared in a canonical form (`lambda_derive.canonical`: summed
letters renamed, indices sorted within antisymmetric groups, factors
sorted, external letters permuted under A; the smallest form with its sign).

## 4. The orbital gradient: which contractions of densities with integrals

For the Z-vector equations (`zccsdt.zccsd.zvector`) the orbital gradient is
`G[a,i] = 1/2 Gnh[a,i]` with

    Gnh[a,i] = <0| (1 + Lambda) exp(-T) [a+_i a_a, H] exp(T) |0>,

L with the integrals one-index transformed by the de-excitation (a bra
index i takes the values of a, a ket index a minus those of i). The other
half `Gnh[i,a] = -dL/dt1[i,a]` and the blocks among occupied and among
virtual spin orbitals vanish for converged t and lambda. With the dressed
quantities `[a+_i a_a, H]` becomes `[a+_i a_a, H~]` plus one-index
transformations among occupied, among virtual spin orbitals and by
`a+_a a_i`, each contracted with t1 - and those vanish too. So **the
orbital gradient is the de-excitation one-index transformation in the
dressed basis**:

    Gnh[a,i] = sum_q h~[a,q] d1~[i,q] - sum_p h~[p,i] d1~[p,a]
             + 1/2 sum <aq||rs>~ d2~[i,q,r,s] - 1/2 sum <pq||is>~ d2~[p,q,a,s],

the table `GRADIENT` (`lambda_ref.orbital_gradient(dressed=True)`; from the
full bare densities: `dressed=False`; both agree with
`zvector.orbital_gradient` on the Hermitian densities). Which block of
the density meets which block of dressed integrals:

| Integrals (block of `gradient_blocks`) | Density met (block of `DENSITY`) | made of | terms |
|---|---|---|---|
| `<ae||fg>` (vv.vv) | `aibc`, the coefficient of `<ai||bc>` (o v^3 elements) | l1 t2, l2 t3, l3 t2 t2 | 4 |
| `<ef||gi>`, `<ae||fm>`, `<ea||fi>` (vv.vo) | `abcd` (v^4); `iabj`; `ab` of f~ (the Fock matrix `sum_m <pm||qm>`) | l2 t2, l3 t3 | 2 + 2 + 2 |
| `<ef||im>`, `<ae||mn>`, `<ea||mi>` (vv.oo) | `abci` (v^3 o); `iajk`; `ai` of f~ | l3 t2; l3 t2; l1 | 1 + 1 + 1 |
| `<am||ef>` (vo.vv) | `ijab` | t2, l1 t3, l2 t2 t2, l3 t2 t3 | 17 |
| `<am||ej>`, `<em||fi>`, `<ma||ei>` (vo.vo) | `ijak`; `aibc`; `ia` of f~ | l1 t2, l2 t3, l3 t2 t2 | 13 |
| `<mn||ie>` (oo.vo) | `ijab` | t2, l1 t3, l2 t2 t2, l3 t2 t3 | 17 |
| `<am||jk>`, `<em||ij>`, `<ma||ji>` (vo.oo) | `ijkl`; `iabj`; `ij` of f~ | l2 t2, l3 t3 | 2 + 2 + 2 |
| `<mn||ij>` (oo.oo) | `ijak` | l1 t2, l2 t3, l3 t2 t2 | 4 |
| `f[a,e]`, `f[m,i]` (f.vv, f.oo) | `ia` of f~ | l1 t2, l2 t3, l3 t2 t2 | 3 + 3 |
| `f[a,m]`, `f[e,i]`, `f[a,i]` (f.vo) | `ij`; `ab` of f~; the reference | l2 t2, l3 t3 | 2 + 2 + 1 |

81 terms in all (a term of `DENSITY` that is not symmetric in two indices of
the transformed kind gives two).  The density `abij` (= l2 / 4) does not
enter: it has no occupied bra index and no virtual ket index.

(`lambda_terms.gradient_blocks()` has the terms of each block.) No density
with three or four virtual indices has to be stored: in the terms it is a
product, and the contraction is done through its factors. With the best
order of pairwise contractions (at o = 10, v = 40):

- the largest intermediates are o v^3 (`sum l2[m,n,e,f] t3[i,m,n,f,g,h]`
  and its l1 t2 and l3 t2 t2 analogues, contracted with `<ae||gh>`: the
  size of the integrals `<ia||bc>`), o^4 v^2 and o^5 v - all smaller than
  T3;
- the most expensive steps are o^4 v^4 (four terms: `l3 <aj||bh> t2 t3`
  twice, `l3 <ae||hm> t3`, `l3 <ef||hi> t3`), o^5 v^3 and o^3 v^4. There is
  no o^3 v^5 step: the gradient is cheaper than a CCSDT iteration;
- `<ab||cd>` is met by the o v^3 density `aibc` only (four terms), as in
  the CCSD code, and `<ab||ci>` by the v^4 density `l2 t2 + l3 t3` and by
  `iabj`.

The Lambda equations: `LAMBDA[3]` has the pp ladder (o^3 v^5: `l3
<ef||bc>`), o^4 v^4 (ring) and o^5 v^3; `LAMBDA[2]` and `LAMBDA[1]` go up to
o^4 v^4 and o^5 v^3.

## 5. CCSDT(Q)_Lambda

Canonical Hartree-Fock orbitals, bare integrals (t1 and l1 do not enter),
sums over the unique quadruples, D4 the differences of orbital energies:

    E(Q)_L = sum z4 r4 / D4,
    r4 = <Q| [V, T3] + 1/2 [[V, T2], T2] |0>          (as in CCSDT(Q)),
    z4 = <0| (Lambda_2 + Lambda_3) V |Q>,

`Lambda_2 V` the disconnected product (`l2[i,j,a,b] <kl||cd>`,
antisymmetrized) and `Lambda_3 V` the connected one (`l3` with `<el||cd>`
and with `<kl||md>`). It is CCSDT(Q) of `ccsdtq_ref.quadruples` with
`conj(l2)`, `conj(l3)` in the place of t2, t3 on the left
(`lambda_ref.quadruples_lambda`); the part of Lambda_2 alone is
CCSDT[Q]_L.

Pinned against MRCC (`calc=CCSDT(Q)_L`) and CFOUR (`CALC=CCSDT(Q)_L`,
`CC_PROG=NCC`), RHF, all electrons, bohr geometries, job 31735245; zccsdt
with real and with complex spin orbitals (every pair of spin orbitals mixed
by its own SU(2) matrix) gives the same numbers:

| | MRCC | CFOUR ncc | zccsdt (dense) |
|---|---|---|---|
| BH / STO-3G, E(Q)_L total | -24.809914657338 | (other SCF state) | -24.809914657338 |
| H2O / STO-3G, E(Q)_L total | -75.012639366162 | -75.012639366158 | -75.012639366162 |
| H2O / STO-3G, [Q]_L correction | | -1.5131522e-05 | -1.5131521e-05 |
| H2O / STO-3G, (Q)_L correction | -1.5604671e-05 | -1.5604666e-05 | -1.5604671e-05 |
| H2O / 6-31G, E(Q)_L total | -76.120882822615 | -76.120882822605 | -76.120882822616 |
| H2O / 6-31G, [Q]_L correction | | -4.49301234e-04 | -4.49301219e-04 |
| H2O / 6-31G, (Q)_L correction | -4.57583683e-04 | -4.57583673e-04 | -4.57583683e-04 |

The first definition tried is the one that matches both programs. What it
is not: with `conj(t)` for lambda it is CCSDT(Q) (BH: -5.9230e-05 against
-5.5747e-05 for (Q)_L; H2O / STO-3G: -1.5790e-05 against -1.5605e-05), and
MRCC's CCSDT(Q)/A and CCSDT(Q)/B are different numbers again (in
`bench/refs/reference_energies.json`). CFOUR with `SYMMETRY=OFF` converges
the SCF of BH / STO-3G to another state (E = -24.4622854), so there is no
CFOUR number for BH.

## 6. Exact ground truth

`tests/fockspace.py::Left`: operators as matrices in the Fock space of a
few spin orbitals, vectors over the whole space, the bra of a vector its
transpose (never its conjugate). For given amplitudes: `Hbar x`, the matrix
of `Hbar` among the reference and the singly, doubly and triply excited
determinants, Lambda from the linear system
`<0| (1 + Lambda) (Hbar - E) |mu> = 0` (`solve`), the residual
`<0| (1 + Lambda) [Hbar, tau_mu] |0>` for any amplitudes
(`lambda_residual`), and `d1`, `d2` as expectation values (`rdm1`, `rdm2`).
No formula of the tables is used.

## 7. Validation

Tables (`tests/test_lambda_terms.py`, 6 s):

- the tables equal both derivations (rational coefficients);
- for complex t1, t2, t3, l1, l2, l3 that solve nothing and complex f and
  <pq||rs> (Hermitian and not), in 4+5, 5+4 and 5+5 spin orbitals: the three
  Lambda residuals, d1, d2, the Lagrangian from the densities, the orbital
  gradient (bare, and of the dressed basis) and the Lambda_1 residual as a
  one-index transformation all equal the operator expressions to 1e-12 of
  their size;
- the terms of each table are linearly independent as vectors (none written
  twice, none vanishing).

Reference implementation (`tests/test_lambda_ref.py`, 25 s):

- model Hamiltonian, 4+5 spin orbitals, complex Hermitian, f not diagonal
  (t1 up to 0.08), converged CCSDT: the iterative Lambda equals the solution
  of the linear system to 4e-14 (20 iterations; lambda differs from conj(t)
  by 5e-3); d1, d2 equal the expectation values to 4e-14; the energy from the
  densities; trace(d1) = N;
- the same model, derivatives of the re-converged CCSDT energy at fixed
  orbitals by central differences of fourth order: one-electron
  perturbation 3e-12 off the trace with d1 (value -5.197), two-electron
  perturbation 3.9e-10 off 1/4 trace with d2 (value 5.308) at step 1e-4,
  sixteen times that at step 2e-4;
- the blocks of the one-index transformed Lagrangian among occupied, among
  virtual and for the excitations vanish (to 4e-9 of the de-excitation
  block, which is 1.7e-5 in this model: rounding);
  the dressed table equals the gradient from the bare densities; half of it
  is `zvector.orbital_gradient` with the Hermitian densities;
- the CCSD limit against PySCF (`gccsd_lambda`, `gccsd_rdm`) and
  `zccsd.rdm.gamma1`, H2O / STO-3G, real and complex spin orbitals;
- molecules: CCSDT(Q)_L as in section 5; the derivative of the CCSDT energy
  with respect to a field along z in the Hamiltonian at fixed orbitals
  against trace(z d1):

| | trace(z d1) | four-point difference, step 1e-3 | difference |
|---|---|---|---|
| BH / STO-3G, real | 2.087019107908 | 2.087019107987 | 7.9e-11 |
| BH / STO-3G, complex | 2.087019107908 | 2.087019107989 | 8.2e-11 |
| H2O / STO-3G, real | 1.582638016242 | 1.582638016253 | 1.2e-11 |
| H2O / STO-3G, complex | 1.582638016242 | 1.582638016240 | 1.4e-12 |
| H2O / 6-31G, real | 1.227823968433 | 1.227823968377 | 5.7e-11 |

  (at step 2e-3 the differences are 1.2e-9 for BH, 7.5e-10 for H2O / 6-31G
  and 2.4e-11 to 3.9e-11 for H2O / STO-3G: the truncation error of the
  stencil.  STO-3G: in the tests; 6-31G: `bench/lambda_ccsdt.py` in job
  31735245, where the dense Lambda equations take 55 s for 17 iterations.)

- orbital relaxation through the code of the CCSD line: `GRADIENT` halved
  is the right-hand side of `zccsd.zvector.solve`, and `relaxed_rdm1` on
  the Hermitian density gives the relaxed CCSDT dipole derivative, against
  E(SCF) + E_corr(CCSDT) with the SCF repeated in the field (four-point
  differences, step 1e-3):

| | unrelaxed | relaxed | finite field | difference |
|---|---|---|---|---|
| H2O / STO-3G, real | 1.582638016243 | 1.582604399041 | 1.582604399045 | 4.1e-12 |
| BH / STO-3G, complex | 2.087019107914 | 2.087012339690 | 2.087012339757 | 6.7e-11 |

## 8. Not done here

- Frozen orbitals: with frozen occupied or virtual spin orbitals the blocks
  of the gradient between frozen and active ones do not vanish; they are
  one-index transformations of the same Lagrangian with integrals that
  carry one frozen index (as `zccsd.orbgrad_frozen` does for CCSD).
- Non-canonical or non-Hartree-Fock orbitals in (Q)_Lambda (terms with
  f[i,a] on the left).
- Any factorization for packed amplitudes in the dense reference: its
  tables are flat lists of terms, each a product of at most four tensors.
  The Lambda equations on packed Lambda_3 are section 9; densities and
  the orbital gradient are dense still.

## 9. The equations on packed Lambda_3 (`zccsdt.cc.lambda_opt`)

`lambda_opt.kernel(eris, t1, t2, t3, opts)` solves the Lambda equations with
l1 and l2 dense and l3 on the unique block `[i<j<k, a<b<c]`, as
`ccsdt_opt` holds T3 (`docs/factorization.md`): Jacobi steps with DIIS (the
T3-sized vectors in the scratch file) from conj(t), converged on the norm
of the change.  No array of o^3 v^3 or v^4 elements is formed.  Notation of
`docs/factorization.md`: P, Q occupied and virtual pairs, I, A triples,
`L[P, d, m, Q]` the half-expanded l3 (`l3[m, P, d, Q]`), `Lv[Q, I, c]` the
one gathered for the ladder.

### 9.1 What is made once

One amplitude iteration's worth of intermediates and two contractions of T3
(`make_intermediates`, a quarter to a third of a Lambda iteration):

| Intermediate | Is | Elements |
|---|---|---|
| `Fvv`, `Foo`, `Wov`, `Woooo` | `Feff_vv`, `Feff_oo`, `Weff_ovvo`, `Weff_oooo` of the T3 equation | o^2 v^2 |
| `Wpp` | its pp ladder operator `~<ab\|\|de> + sum_{m<n} t2[mn,ab] <mn\|\|de>` with `1/2 A[Fvv]` folded in, made in the dressed packed `<ab\|\|de>` | v^4 / 4 |
| `O6T` | the transpose of its occupied operator (hh ladder and Foo) | C(o,3)^2 |
| `W_vvvo[Q, e, i]` | the drive intermediate of the T3 equation with its T3 part: the matrix element `<ab\|\|ei>` of Hbar | o v^3 / 2 |
| `W_vooo[j, P, e]` | `<ej\|\|mn>` of Hbar: `f[j,f] t2[mn,ef] + ~<ej\|\|mn> + sum_{f<g} ~<ej\|\|fg> t2[mn,fg] + P(mn) ~<jo\|\|fm> t2[no,ef] + sum_{o,f<g} <jo\|\|fg> t3[mno,efg]` (the last from pass 1 over T3) | o^3 v / 2 |
| `Xq[P, I, e]` | `sum_{f<g} <ij\|\|fg> t3[I, e, fg]` | o^5 v / 12 |
| `X2[Q, I, i]` | `A(ef) sum_{g<h} ~<ei\|\|gh> t3[I, f, gh]`, Q = (e<f) | o^4 v^2 / 12 |
| `Vx[g, e, j, b]` | `~<ej\|\|bg>` | o v^3 |

All integrals are the T1-dressed ones (`~` where a bare one could be meant).
`W_vvvo` with `t2` is the operand of pass 1 (`GmQ`), `W_vooo` with `t2` the
other one (`GPd`); `~<ei||Q>` and `<kn||Q>` are rows of the drive operand
`Bdrv`, `<P||cf>` and `<P||cm>` columns of `Adrv`.

### 9.2 A Lambda iteration

Three passes over the packed l3, as an amplitude iteration has them over T3,
and one product of the two packed arrays:

1. **Pass 1** (`DenseL3.pass1`): batches of occupied pairs of the
   half-expanded l3, and of T3 next to them.
   - Two stacked GEMMs, those of pass 1 of the amplitude equations:
     over (P, d), `Y2[f, i, Q] = sum_{m<n, e} t2[mn, e f] l3[i mn, e Q]` and
     `sum W_vooo[j, P, e] L[P, e, i, Q]`; over (m, Q),
     `Y1[P, a, n] = sum_{m, e<f} l3[m P, a ef] t2[mn, ef]` and
     `sum L[P, a, m, Q] W_vvvo[Q, b, m]`.
   - One GEMM per occupied pair for the density
     `Dr[(a,i), (g,o)] = sum_{m<n, e<f} l3[i mn, a ef] t3[o mn, g ef]`
     (o^4 v^4 / 4: the cost of the ring), and a matrix-vector product for
     `D1[g, o] = sum_{P,Q} l2[P, Q] T[P, g, o, Q]`.
2. **Gram matrix** `G[I, J] = sum_A l3[I, A] t3[J, A]`, one GEMM of the packed
   arrays (C(o,3)^2 C(v,3)); `sum_m` of its elements gives
   `l3[ijm,efg] t3[nom,efg]` on pairs and `Goo3[i,o] = l3[imn,efg]
   t3[omn,efg]`.  `Gvv3[a,g] = l3[mno,aef] t3[mno,gef]` is a trace of `Dr`.
3. **Pass 2** (`DenseL3.pass2`): the half-expanded residual of a batch,
   `Z[P, (a,i), Q] = WL L[P] + U V[P] - Adrv Bdrv`, reduced once
   (`P(i/P) P(a/Q)`): ring, the two outer products, four drive products.
4. **Ladder pass** (`DenseL3.ladder`): `Wpp^T Lv` reduced with `P(c/Q)`, and
   on the same gathered batch the products of `Lv` with `Xq` and `X2`.
5. **hh**: `R += O6T l3` on the packed arrays.
6. The residuals of rank 1 and 2 from the small arrays; `l += R / D`.

`LAMBDA[3]`, term by term (numbering of the table, from 1):

| Terms | | Evaluated as |
|---|---|---|
| 1 | `l1[i,a] <jk\|\|bc>` | pass 2, outer product `Z[P,(a,i),Q] += l1[i,a] <P\|\|Q>` |
| 2 | `l2[ij,ab] f[k,c]` | pass 2, outer product `f[i,a] l2[P,Q]` |
| 3 | `l2[ij,ae] <ek\|\|bc>` | pass 2, drive: `Adrv = l2[P,e,a]`, `Bdrv = ~<ek\|\|Q>` |
| 4 | `l2[im,ab] <jk\|\|cm>` | pass 2, drive: `Adrv = <P\|\|cm>`, `Bdrv = l2[m,i,Q]` |
| 5, 6 | `l3 Fvv` | folded into the ladder operator, as in the T3 equation |
| 7, 8 | `l3[ijk,aef] W(ef,bc)` | ladder pass with `Wpp^T`: the same operator, GEMM with the transposed flag (no second copy) |
| 9, 10, 14, 15 | `l3 Foo`, `l3[imn,abc] Woooo[jk,mn]` | `O6T`, the transposed occupied operator |
| 11, 12 | `l3[ijm,abe] Wov[k,e,c,m]` | pass 2, ring with `WL = WringT^T` |
| 13 | `-(l3 t2)[ij,a,n] <kn\|\|bc>` | pass 2, drive: `Adrv = Y1[P,a,n]`, `Bdrv = <kn\|\|Q>` |
| 16 | `-(l3 t2)[i,ab,f] <jk\|\|cf>` | pass 2, drive: `Adrv = <P\|\|cf>`, `Bdrv = Y2[f,i,Q]` |

The operators of the terms linear in l3 are those of the T3 equation,
transposed - checked term by term, not assumed.  The drive products are not
the transposes of the drive terms of T3 (`t2 W_vvvo`, `W_vooo t2`): two have
l2 with bare-shaped integrals, two the intermediates `Y1`, `Y2` of l3 with
`<mn||ef>`; the drive GEMM has K = 2 (o + v) instead of o + v.

The terms with l3 of `LAMBDA[2]` (19) and `LAMBDA[1]` (17), by what they
take of l3:

| Of l3 | `LAMBDA[2]` | `LAMBDA[1]` |
|---|---|---|
| pass 1 with `W_vvvo` (5 terms: `l3[ijm,aef] <ef\|\|bm>`-bar) | `P(ab)` of it | (l2 with `W_vvvo`) |
| pass 1 with `W_vooo` (5 terms: `l3[imn,abe] <ej\|\|mn>`-bar) | `P(ij)` of it | (l2 with `W_vooo`) |
| `Y1[P,a,n]` | with `~<en\|\|ab>`; with `<jo\|\|bm>` (`P(ij) P(ab)`) | with `Wov` (2 terms); with `Woooo` (2 terms) |
| `Y2[f,i,Q]` | with `<ij\|\|fm>`; with `~<ej\|\|bg>` (`P(ij) P(ab)`, o^2 v^4) | with the columns of the ladder operator (2 terms, `pair_matvec`); with `Wov` (2 terms); with `t2` into the density `Dov` |
| `Dr` | with `<jo\|\|bg>` (`P(ij) P(ab)`) | with `~<eo\|\|ah>`; with `<ij\|\|gm>` |
| `G` (Gram) | `l3 t3` on pairs with `<no\|\|ab>`; `Goo3` with `<jo\|\|ab>` | the same with `<jo\|\|am>`; `Goo3` with f and with `<ij\|\|am>` |
| `Gvv3` | with `<ij\|\|bg>` | with f and with `~<ei\|\|ah>` |
| ladder pass | `Xq Lv` | `Lv X2` |

and of l2 with T3: `D1` (one term of `LAMBDA[1]`, with `<io||ag>`), and the
T3 parts of `W_vvvo`, `W_vooo`.  `Goo3`, `Gvv3`, `D1` enter next to the
CCSD-like `l2 t2` contractions with the same integrals.  The terms of l1
and l2 alone are the CCSD Lambda equations with the dressed integrals and
`Fvv`, `Foo`, `Wov`, `Woooo`, `Wpp` (the folded `Fvv` taken out where l2 or
`Y2` meets the ladder operator).

Cost of an iteration, against the amplitude iteration: the same ring
(o^4 v^4 / 4), pp ladder (o^3 v^5 / 24) and occupied operator; the drive
GEMM twice; and **one more product of the size of the ring**, the density
`Dr` of l3 with T3, which `LAMBDA[2]` needs with four indices free
(`1/4 l3[imn,aef] <jo||bg> t3[mno,efg]`) and which has no cheaper order of
contraction.  That, the second gather in pass 1 and the Gram matrix are
what a Lambda iteration costs more than an amplitude iteration.

### 9.3 Memory

`lambda_opt.plan_memory` (`LambdaPlan`; `Result.plan`), with f = o v^3
elements, s = o^2 v^2, r = 1 + o / v:

| Part | Elements | |
|---|---|---|
| T3, l3, the residual | 3 C(o,3) C(v,3) | in memory, all three |
| pp operator | v^4 / 4 | next to the bare one of the integral object |
| kept operands | f (1 + 1.5 r) + 9 s + o^4 v^2 / 12 | `Vx`, `Bdrv`, `GmQ`, small ones, `X2` |
| setup | at most f (2 + 1.5 r), or the dressing of the pp operator | the dressed `<ie\|\|bc>` goes when `Vx` is made |
| iteration | the batch buffers (2/3 `mem_mb`, at least f), or 1.5 f after the passes; 8 s | |
| DIIS | in the scratch file (in memory below `diis.INCORE_SIZE` elements: counted) | |

With `max_memory` the buffers shrink to what is left (bisection: the
largest that fit), down to one occupied pair; below that `MemoryError`.
**There is no scratch fallback** for T3, l3, the residual or the pp
operator in this version (the amplitude solver has one).

### 9.4 Validation (`tests/test_lambda_opt.py`, 30 s)

- Random complex and real amplitudes and Lambda that solve nothing, random
  Hermitian integrals with a full Fock matrix, 4+5 to 5+6 spin orbitals:
  each of the six groups of terms of the Lambda_3 residual (the rows of the
  table above), the three residuals, their parts by the rank of Lambda, and
  a Jacobi step equal the dense evaluation of the tables to 1e-12 of their
  size (1e-15 found) - with the C kernels and with the numpy forms, with all
  occupied pairs in a batch and with one.
- Converged Lambda against `lambda_ref`: the complex model of section 7
  (1e-10), H2O / STO-3G with real and with
  SU(2)-mixed complex spin orbitals (1e-9 asked, 2e-12 found; real orbitals
  give real Lambda), X2C spinors of HBr / cc-pVDZ-DK turned off the axis,
  28 frozen, 12 virtual spinors (1e-9).
- H2O / STO-3G, real and complex: `trace(z d1)` with the densities of
  `lambda_ref.densities` and this Lambda equals the four-point finite-field
  derivative of the CCSDT energy at fixed orbitals to 2e-10.
- Memory: what the intermediates and an iteration allocate (tracemalloc)
  is within the plan and above 0.8 of it, stage by stage, for 4+36 and 6+30
  spin orbitals with the largest buffers, the smallest, and a size that
  leaves a short last batch.
- No triples (two occupied or two virtual spin orbitals); the kernels of
  `csrc/lambda_t3.c` against their numpy forms; integrals and amplitudes
  are not modified.

### 9.5 Timing

`bench/profile_lambda_ccsdt.py`: amplitude iterations and Lambda iterations
in one process (three of each after a warm-up iteration), 8 cores of a
Xeon Gold 6248R, MKL, batch buffers 2000 MB; job 31737372 on c105 (a node
that other jobs were using: the first run of the same script, job 31737112
on other cores of it, gave the amplitude iterations within 2% and ratios of
1.26, 1.28 and 1.17 before the outer products were put on the ring GEMM).
Seconds per iteration:

| | H2O / cc-pVTZ, spin orbitals of RHF | H2O / cc-pVTZ, X2C | HBr / cc-pVTZ-DK, X2C, 28 frozen |
|---|---|---|---|
| occupied + virtual, arithmetic | 10 + 106, real | 10 + 106, complex | 8 + 78, complex |
| amplitude iteration | 4.18 | 13.74 | 1.90 |
| **Lambda iteration** | **5.05** | **17.28** | **2.19** |
| **ratio** | **1.21** | **1.26** | **1.15** |
| intermediates, once | 1.65 | 4.57 | 0.82 |
| cores busy in a Lambda iteration | 7.6 | 7.7 | 7.6 |
| Lambda: pass 1 | 1.69 | 5.81 | 0.72 |
| - of it the density `Dr` | 1.12 | 3.94 | 0.46 |
| - the two stacked GEMMs | 0.30 | 1.41 | 0.16 |
| - gathers (l3 and T3) | 0.25 | 0.44 | 0.09 |
| Lambda: pass 2 | 1.53 | 5.14 | 0.72 |
| - ring GEMM (with the outer products) | 1.06 | 3.89 | 0.52 |
| - drive GEMM | 0.23 | 0.86 | 0.12 |
| - gather, reduction | 0.23 | 0.38 | 0.07 |
| Lambda: ladder pass | 1.42 | 5.31 | 0.55 |
| Lambda: residuals of rank 1 and 2 | 0.24 | 0.61 | 0.14 |
| Lambda: hh, Gram matrix, update | 0.04 | 0.16 | 0.02 |
| Lambda: DIIS | 0.11 | 0.21 | 0.02 |
| amplitudes: pass 1 | 0.52 | 1.83 | 0.24 |
| amplitudes: pass 2 | 1.48 | 5.00 | 0.70 |
| amplitudes: pp ladder | 1.37 | 5.13 | 0.53 |
| amplitudes: dressing, intermediates, CCSD part | 0.61 | 1.37 | 0.36 |
| peak anonymous memory above the start of Lambda | 2.59 GB | 3.86 GB | 2.11 GB |
| planned (`LambdaPlan.need` less T3, which is there at the start) | 3.11 GB | 4.19 GB | 3.69 GB |

The excess over the amplitude iteration is pass 1: the density of l3 with T3
(as much as the ring), the second gather and the wider drive GEMM (+0.09,
+0.31, +0.03 s); against it, a Lambda iteration dresses no integrals
(0.37, 0.76, 0.21 s less than the amplitude iteration's dressing,
intermediates and CCSD part).  Pass 2 and the ladder are those of the
amplitudes to 4%.  The first of the two jobs showed 4.77 GB above the start
for the X2C water run against 4.19 GB planned: a short last batch of the
ladder gathered in new arrays next to the buffers; fixed, and in the tests
since (`test_memory_plan` has a buffer size with a short last batch).

### 9.6 Not done

- **Symmetry labels**: the dense layout only.  Integrals with labels in use
  are taken as they are (T3 is put on the lexicographic unique block,
  `ccsdt_opt.packed_t3`, and Lambda_3 is returned on it).  Everything that
  touches the packed arrays is a method of `DenseL3` (`pass1`, `gram`,
  `pass2`, `ladder`, `occ`, `update`): an engine on the blocks of a label
  has these to provide, on `t3engine.SpinBlockT3`'s gathers and reductions.
- **Scratch fallback** (9.3).
- U / R-form integrals (`eris_spin`): spin orbitals or spinors only.
- Densities and the orbital gradient from the packed arrays (the next
  stage; `lambda_ref.densities` takes dense ones), frozen orbitals.
- A front end: `GCCSDT` has no `solve_lambda` (its file was not to be
  touched); the hook would be `lambda_opt.kernel(self.ao2mo(), self.t1,
  self.t2, self.t3, options)`.
