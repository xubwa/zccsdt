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
- Any factorization for packed amplitudes: the tables are flat lists of
  terms, each a product of at most four tensors.

## 9. Densities, orbital gradient and relaxed density from packed amplitudes

`src/zccsdt/cc/density_opt.py` (`tests/test_density_opt.py`,
`bench/density_ccsdt.py`, `bench/cfour/write_density_ccsdt.py`): the
one-particle density, the orbital gradient and - through
`zccsd.zvector` - the relaxed density from t3 and lambda3 as the amplitude
code stores them, `[C(o,3), C(v,3)]` (or the symmetry blocks of the engine
of the front end). It takes the place of the first and the last item of
section 8 for the densities and the gradient; Lambda itself is still that of
the dense reference.

```python
from zccsdt.cc import density_opt
res = density_opt.kernel(eris, t1, t2, t3, l1, l2, l3)       # res.d1[p,q] = <p+ q>, res.gnh[a,i]
rel = density_opt.relaxed_rdm1(mf, eris, t1, t2, t3, l1, l2, l3, mo_coeff,
                               frozen_occ=None, dropped_vir=None)
# rel.relaxed, rel.unrelaxed: D[p,q] = <q+ p>, Hermitian, in the spinors rel.mo_coeff
# [frozen occupied, active occupied, active virtual, dropped virtual]; C D C^+ in the AO basis
```

### 9.1 Factorization of the gradient

The 81 terms are not evaluated one by one. With g1 and g2 the coefficients
of the dressed f and <pq||rs> in the Lagrangian (g2: `DENSITY`,
antisymmetrized),

    Gnh[a,i] = f[a,i] + sum_q f[a,q] g1[i,q] - sum_p f[p,i] g1[p,a] + sum_pq <pa||qi> g1[p,q]
             + 1/2 sum <aq||rs> g2[i,q,r,s] - 1/2 sum <pq||is> g2[p,q,a,s]

and every block of g2 but one is made once and contracted with its blocks
of dressed integrals:

| Block of g2 | Elements | Made of | Meets |
|---|---|---|---|
| `ijkl` | o^4 / 4 | l2 t2; l3 t3 (`O4`) | `<am\|\|jk>` |
| `ijak` | o^3 v | l1 t2; l2 t3; l3 t2 t2 (`C3`, `Z`) | `<am\|\|ej>`, `<mn\|\|ij>` |
| `iajk` | o^3 v / 2 | l3 t2 (`Z`) | `<ae\|\|jk>` |
| `ijab` | o^2 v^2 / 4 | t2; l1 t3; l2 t2 t2; l3 t2 t3 (7 terms: through g1, `R3`, `O4`, `Z`, `C3` and one o^5 v intermediate) | `<am\|\|ef>`, `<mn\|\|ie>` |
| `iabj` | o^2 v^2 | l2 t2; l3 t3 (`R3`) | `<ae\|\|fm>`, `<em\|\|ij>` |
| `abci` | o v^3 / 2 | l3 t2 (`C3`) | `<ef\|\|im>` |
| `aibc` | o v^3 / 2 | l1 t2; l2 t3; l3 t2 t2 (`C3`, `Z`) | `<ae\|\|fg>`, `<em\|\|if>` |
| `abcd` | not made | l2 t2; l3 t3 | `<ef\|\|gi>`: through its factors |

The intermediates of lambda3 and t3 (P, Q: occupied and virtual pairs):

| Name | Definition | Elements | Product |
|---|---|---|---|
| `Z[P,g,i]` | 1/2 sum_(m,ef) l3[m,P,e,f,g] t2[i,m,e,f] | o^3 v / 2 | o^4 v^3 / 4 |
| `C3[Q,c,i]` | sum_(mn,e) l3[i,m,n,Q,e] t2[m,n,c,e] | o v^3 / 2 | o^3 v^4 / 4 |
| `R3[(a j),(b i)]` | 1/4 sum l3[j,m,n,a,e,f] t3[i,m,n,b,e,f]: the ring density | o^2 v^2 | o^4 v^4 / 4 |
| `O4[P,P']` | sum_(m, a<b<c) l3[m,P,abc] t3[m,P',abc] | o^4 / 4 | C(o,3)^2 C(v,3) |
| `Y[P,g,m,h,i]` | sum_(e<f) l3[m,P,g,e,f] `<ef\|\|hi>` for a batch of P | batch | o^4 v^4 / 4 |
| `H[P,g,m,P']` | sum_(e<f) l3[m,P,g,e,f] t2[P',e,f] for a batch of P | batch | o^5 v^3 / 4 |

`R3` also gives the l3 t3 parts of the one-particle blocks (its traces over
the occupied and over the virtual pair of indices), and the terms of `ijab`
with one-particle densities collapse: `l2 t2 t2` and `l3 t2 t3` with a
two-index contraction of lambda and t are `t2` times the block `ab` or `ij`
of g1.

Passes over the packed arrays, both through the half-expanded batches of the
amplitude code (`t3pack.gather_half`: `X[P, d, m, Q]`, a matrix `(P d) x (m
Q)`):

1. lambda3 alone: `Z` and `C3`, one GEMM each per batch;
2. t3 and lambda3 together: t3 with (l2, `Z`) stacked over (P, d) and with
   (l2, `C3`) stacked over (m, Q) - the l2 t3 parts of `aibc` and `ijak` and
   two terms of `ijab` -, t3 with l1 and l2 (matrix-vector); `R3`, one GEMM
   per occupied pair; `Y`, antisymmetrized in (g, h) and contracted with t3
   of the same batch; `H` and its product with t3;
3. the Gram matrix of the rows of the two packed arrays, one GEMM without a
   gather: `O4`.

t3 is gathered once, lambda3 twice. The two o^4 v^4 / 4 products (`R3`, `Y`)
are the cost - a CCSDT iteration has one of them and the o^3 v^5 / 24
ladder. The packed dressed `<ab||de>` is used twice in slabs of rows, for
the dressed `<ef||gi>` and `<ef||ij>` before the passes and with `aibc`
after them; if all of it fits next to the batches it is dressed once.

Memory (`density_opt.plan_memory`, a `Plan`): five arrays of o v^3 / 2
elements (the dressed `<ef||gi>`, `C3` with l2, `aibc` with a term of
`ijab`, the bare packed `<ma||ef>` of the amplitude code, one transient),
about twenty of o^2 v^2, the batch buffers (two half-expanded batches and
`Y`: 2 o v^3 / 2 + o^2 v^2 elements per occupied pair) and the slab of
`<ab||de>`. The batches take what the budget leaves up to 4 GB, then the
operator; `MemoryError` below one pair and the rows of one first index.

### 9.2 Frozen and dropped spinors

The orbital gradient between a frozen spinor f (occupied or virtual) and an
active one x is, in the convention of `zccsd.zvector`,

    G[f,x] = (B[f,x] + conj(K[f,x])) / 2 + (n_x - n_f) V[f,x],
    B[f,x] = 1/2 sum_qrs <fq||rs> gamma2[x,q,r,s],   K[f,x] = 1/2 sum_pqs <pq||fs> gamma2[p,q,x,s],

gamma2 the two-particle density of the bare integrals without its
one-particle terms and V the Fock matrix of the Hermitian part of gamma1 =
d1 - 1(occupied). In the dressed basis

    gamma2 = (g2 transformed back) + A[(gamma1 - tau / 2) x tau],   tau = t1 in the occupied-virtual block,

so B and K are the contractions of the blocks of g2 above with the
integrals `<fq||rs>` dressed in their three active indices (as a bra index:
32 block contractions; the block `abcd` through two more products of the
kind of `Y` in pass 2, one with t3 first), transformed back in the fourth
index, plus Fock matrices of gamma1 - tau / 2 and of tau
(`density_opt.frozen_terms`). The integrals come from a transformation with
the frozen spinors as the first index (`frozen_integrals`: (o + v)^3 numbers
for each, in batches of 2 GB; it is done twice, before and after the
passes). `relaxed_rdm1` then hands the gradient of all spinors to
`zvector._relax`: the rotations between frozen and active spinors of a kind
follow from the Fock matrix staying diagonal.

`density_opt.all_blocks` takes every spinor as the index outside: all four
blocks of the one-index transformed Lagrangian from the same contractions
(n^4 integrals: a check for small cases).

### 9.3 Validation

`tests/test_density_opt.py` (34 tests, 64 s):

- g1, d1 and the table `GRADIENT` for complex amplitudes that solve nothing,
  4+5 to 6+3 spin orbitals, also without triples (o = 2, v = 2), real
  arithmetic, without the C kernels, and in every plan of the memory: 1e-13
  of their size (1e-15 found);
- B and K of three "frozen spinors" with random integrals against the dense
  gamma2, and all four blocks of the one-index transformed Lagrangian
  against `lambda_ref.one_index` (1e-15);
- H2O / STO-3G, converged CCSDT (the packed solver) and the Lambda of the
  dense reference: d1 and Gnh of the dense tables (5e-18); the blocks among
  occupied, among virtual spin orbitals and the excitation block vanish
  (4e-9 of the de-excitation block);
- the CCSD limit (t3 = lambda3 = None, amplitudes and lambda of
  `zccsd.DirectZCCSD` in its spinors): the relaxed density of
  `zccsd.zvector.kernel`, 1.5e-11 (all spinors active) and 2.4e-11 (H2O /
  6-31G, two frozen, four dropped, and both);
- the allocations of `kernel` (tracemalloc) within the plan, for one pair
  per batch with the thinnest slab and for everything at once (5 + 40 spin
  orbitals: 0.76 and 0.85 of the plan);
- the relaxed dipole derivative against E(SCF) + E_corr(CCSDT) with the SCF
  repeated in the field (four-point differences, step 1e-3), tests and
  `bench/density_ccsdt.py validate` (job 31736990):

| | unrelaxed | relaxed | finite field | difference |
|---|---|---|---|---|
| H2O / STO-3G, spin orbitals | 1.582638016243 | 1.582604399041 | 1.582604399056 | 1.5e-11 |
| H2O / STO-3G, X2C spinors | 1.583149382876 | 1.583115567033 | 1.583115567046 | 1.3e-11 |
| H2O / STO-3G, X2C, 2 frozen | 1.583089763541 | 1.583062831139 | 1.583062831172 | 3.4e-11 |
| H2O / 6-31G, spin orbitals | 1.227823998609 | 1.227708663950 | 1.227708663909 | 4.1e-11 |
| H2O / 6-31G, complex spin orbitals | 1.227823998609 | 1.227708663950 | 1.227708663916 | 3.4e-11 |
| H2O / 6-31G, X2C spinors | 1.229477115815 | 1.229360128032 | 1.229360127966 | 6.6e-11 |
| H2O / 6-31G, X2C, 2 frozen, 4 dropped | 1.256218427307 | 1.265120598104 | 1.265120598041 | 6.4e-11 |
| HF / 6-31G, X2C, 2 frozen, through `GCCSDT` (real spinors, T3 in blocks) | 16.451171123641 | 16.451317738981 | 16.451317738986 | 5.3e-12 |

  (at step 2e-3 the differences of the 6-31G rows are 7e-10 to 1e-9: the
  truncation error of the stencil.)

- CFOUR (`bench/cfour/write_density_ccsdt.py`, H2O / 6-31G, X2C-1e, CFOUR's
  own spinors through `cfour.reference`): the CCSD limit of the code path
  gives CFOUR's own `denfinal` element by element to 3.2e-12; with the
  relaxed CCSDT density as `denfinal`, `xsox2cprop` prints for every
  operator the trace of CFOUR's operator with the density made here (2e-13
  of the size of the operator; electronic dipole 0.0120771406 against
  0.0158031375 for CCSD and 0.0570680934 for the SCF; job 31737542). There
  is no CCSDT property of CFOUR to compare with.

### 9.4 Timing

H2O / cc-pVTZ, 8 cores of a Xeon Gold 6248R (`bench/density_ccsdt.py time`,
jobs 31737824 and 31737656 on c105, a node that other jobs shared: load 5 to
11). Lambda is the conjugate of t in these runs - the cost does not depend
on the numbers, and the dense reference cannot make Lambda at this size.
Second of two runs; "density" is everything of `relaxed_rdm1`: densities,
orbital gradient and the Z-vector equations.

| | o + v | CCSDT iteration | density | ratio | cores busy | peak memory of the call (plan) |
|---|---|---|---|---|---|---|
| spin orbitals, real | 10 + 106 | 4.10 s | 4.43 s | 1.08 | 7.0 | 4.7 GB (5.7) |
| the same, 3 pairs per batch, slabs of 60 MB | 10 + 106 | 4.07 s | 4.62 s | 1.14 | 6.7 | 0.55 GB (1.5) |
| X2C spinors, complex | 10 + 106 | 13.42 s | 13.37 s | 1.00 | 7.3 | 5.0 GB (6.3) |
| spin orbitals, 2 frozen, 20 dropped | 8 + 86 | 0.94 s | 6.68 s | 7.1 | 3.5 | 1.7 GB (3.7) |
| X2C spinors, 2 frozen, 20 dropped | 8 + 86 | 2.76 s | 13.82 s | 5.0 | 4.9 | 3.6 GB (6.6) |

(the plan counts 0.7 GB for the threaded BLAS.)  Where the time goes, real
spin orbitals, all active: pass 2 3.15 s (`Y` 1.19, `R3` 1.09, the products
of t3 alone 0.39, gathers 0.31, `H` 0.16), pass 1 0.38, the blocks 0.24, the
contractions with the integrals 0.24, dressing 0.29, the Z-vector equations
0.08, the Gram matrix 0.01.

The frozen spinors are not at that level: for 22 of them the integrals with
one frozen index and their contractions take 4.4 s at 1.8 cores (the
transformation, done twice, 2.7 s; the contractions are einsum calls on
(o + v)^3 arrays per frozen spinor), and the two products of pass 2 with
their integrals 1.0 s - together more than the rest.
