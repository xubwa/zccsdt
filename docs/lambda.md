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
| `src/zccsdt/cc/quadruples_lambda.py`, `quadruples_lambda_sym.py`, `csrc/qlambda.c` | CCSDT(Q)_Lambda on packed amplitudes, per occupied quadruple and in symmetry sectors (section 5.1); `tests/test_quadruples_lambda.py`, `tests/test_quadruples_lambda_sym.py`, `bench/profile_q_lambda.py` |
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

### 5.1 On packed amplitudes (`zccsdt.cc.quadruples_lambda`, `quadruples_lambda_sym`)

`quadruples_lambda.kernel(eris, t2, T3, l2, L3)` returns (E[Q]_L, E(Q)_L)
from t2, l2 dense and T3, Lambda_3 on the unique block in lexicographic
order (what `lambda_opt.kernel` returns), for one occupied quadruple at a
time like `zccsdt.cc.quadruples`; `quadruples_lambda_sym.kernel` is the same
in the blocks of a symmetry label like `quadruples_sym`.

r4 is formed by the code of (Q), unchanged. The left-hand side is where the
two differ: (Q) has `z = <Q| V T2 |0>` and takes the T3 part of its left-hand
side as the conjugate of the `[V,T3]` part of r4, which is there anyway;
with Lambda_3 that part is a product of its own. Per occupied quadruple,
with the splits stacked along the summed index and each product followed by
the antisymmetrizing reduction onto a<b<c<d:

    X[(ab),(cd)]  + l(ij,ab) <kl||cd>                     (ij|kl), 6 splits     z2
                  + sum_e <ie||ab> l(jkl,cde)             (i|jkl), 4 splits     z3
    X[a,(bcd)]    + sum_m <ij||ma> l(mkl,bcd)             (ij|kl), 6 splits     z3

These are the two products of `[V,T3]` with l3 in the place of T3 and the
integrals the other way round: `<ab||ei> = -<ie||ab>*` and
`<ma||ij> = <ij||ma>*`, so the stacked integral operands of z3 are the
complex conjugates of those of `[V,T3]`, with the same signs of the splits,
and `<kl||cd>` is the conjugate of `<cd||kl>` of (Q). Nothing else is
conjugated: z4 is `<0| Lambda V |Q>` itself and the energy is
`Re sum z4 r4 / D4` (`zc_ql_energy`, and `zc_ql_sector_energy` for the sum
over the splits of the virtual quadruples of a sector, in `csrc/qlambda.c`).
With `l2 = conj(t2)`, `l3 = conj(T3)` the two energies are [Q] and (Q).

Cost: the two products of `[V,T3]` once more, `o^4 v^4 (v + o) / 24`
multiplications next to the `o^4 v^4 (v + o) / 12` of (Q) and its
`o^2 v^5 / 4` for the intermediate of T2 with `<vv||vv>`.

Memory: the plan of (Q) (`quadruples.plan`: batches of columns of a product,
the intermediate for a few occupied indices at a time) for what the arrays
of the left-hand side leave - l2 and `<oo||vv>` on pairs, the stacked
operands of the two products of Lambda_3 (`8 v C(v,2) + 6 o (v + C(v,3))`
elements) and z3 (`C(v,4)`); the products themselves go through the work
arrays of (Q), one at a time. In sectors one block more of each of the two
kinds. Lambda_3 is given, like T3. `tests/test_quadruples_lambda.py` and
`tests/test_quadruples_lambda_sym.py` hold the peak of the allocations
against the plan (tracemalloc).

Checked: z2 and z3 against the terms `Z4.T2` and `Z4.T3` of the table and
each product of Lambda_3 through its own integrals, random real and complex
amplitudes; the energies against `lambda_ref.quadruples_lambda`; (Q) for
lambda = conj(t); every plan against the plain one; H2O and BH / STO-3G with
CCSDT and Lambda of the production codes, real and with complex spin
orbitals, against the dense reference (1e-13) and the numbers of the table
above (MRCC to 1e-10, CFOUR's two corrections to 2e-11), H2O / 6-31G
against MRCC (1e-9) and CFOUR (5e-11); HBr / cc-pVDZ-DK with X2C spinors
(8 + 28, converged Lambda): the sectors against the dense code to 1e-13 with
the labels 2 m_j, modulo 4 and with one label, and with a phase on every
spinor (complex amplitudes; Lambda takes the conjugate phases of t).

Timing, 8 cores of a Xeon Gold 6248R (`bench/profile_q_lambda.py`, job
31739048 on c105, a node that other jobs shared: load 9 to 10 when the steps
started), the four codes one after the other on the same amplitudes. HBr:
X2C spinors, 28 frozen, converged CCSDT and Lambda (the integrals of a
linear molecule on the axis are real), labels 2 m_j; H2O: RHF orbitals as
spin orbitals, three CCSDT iterations and lambda = conj(t) (the cost does
not depend on the numbers), labels 2 m_s. Peak memory is the anonymous
memory of the process, with the amplitudes and integrals that were there
before (0.50 and 1.05 GB); the plan is what the code counts besides them.

| | o + v | (Q) | (Q)_L | ratio | cores busy | peak memory (plan) of (Q)_L |
|---|---|---|---|---|---|---|
| HBr / cc-pVTZ-DK, dense | 8 + 78 | 4.47 s | 6.11 s | 1.37 | 6.9 | 1.45 GB (0.50 + 0.92) |
| the same, sectors (8 labels) | 8 + 78 | 1.23 s | 1.56 s | 1.27 | 4.1 | 0.80 GB (0.50 + 0.91) |
| H2O / cc-pVTZ, dense | 10 + 106 | 44.1 s | 62.3 s | 1.41 | 7.5 | 4.43 GB (1.05 + 3.46) |
| the same, sectors (2 labels) | 10 + 106 | 18.4 s | 24.7 s | 1.35 | 6.4 | 2.47 GB (1.05 + 3.99) |

Sectors and dense code agree to 3e-19 Eh in all four. HBr: E[Q] =
-2.7035e-05, E(Q) = -2.24571e-04, E[Q]_L = -2.6049e-05, E(Q)_L =
-2.27608e-04 Eh. Where the time of the dense (Q)_L of H2O goes: the
products of r 22.1 s and their reductions 8.7 s as in (Q), the two products
of Lambda_3 14.7 s and their reductions (with that of z2) 7.3 s, the
intermediate of T2 with `<vv||vv>` 4.0 s, the stacked operands 1.9 + 2.0 s
at 4 cores.

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

- **Symmetry labels**: done since, section 9.7 (`kernel(..., blocks=True)`).
  Without `blocks`, integrals with labels in use are taken as they are (T3
  is put on the lexicographic unique block, `ccsdt_opt.packed_t3`, and
  Lambda_3 is returned on it).
- **Scratch fallback** (9.3).
- U / R-form integrals (`eris_spin`): spin orbitals or spinors only.
- Densities and the orbital gradient from the packed arrays (the next
  stage; `lambda_ref.densities` takes dense ones), frozen orbitals.
- (the front end came later: `GCCSDT.solve_lambda()`, section 9.7)

- Any factorization for packed amplitudes: the tables are flat lists of
  terms, each a product of at most four tensors.

### 9.7 In the blocks of the symmetry labels (`zccsdt.cc.lambda_blk`)

`lambda_opt.kernel(eris, t1, t2, t3, opts, blocks=True)` takes T3 as the
amplitude equations leave it for integrals with labels in use - the blocks
of `t3engine.T3Layout`, block g the triples whose classes add up to g - and
returns l3 in the same layout (`Result.blocks`); `lambda_opt.packed_l3(eris,
l3)` puts it on the unique block in lexicographic order, `blocked_l3` is the
inverse (`ccsdt_opt.packed_t3`, `blocked_t3`).  l3 has the labels of T3, so
nothing else is stored.  The equations, the intermediates and the code of
the residuals of rank 1 and 2 are those of 9.1 and 9.2; the engine is
`lambda_blk.BlockL3` instead of `DenseL3`, with the same methods, on the
gathers and reductions of the block engine of the T3 equation
(`t3engine.SpinBlockT3`, `csrc/t3pack.c`: `zc_bgather2`, `zc_breduce2`,
`zc_bgather_vir`, `zc_breduce_ppl`).  One class of labels is no symmetry:
the blocks are then the packed block.

Classes: the labels shifted and divided by their step (`t3pack.classes_of`),
pairs in groups gP, gQ = the sum of two classes, triples in groups of three.

| Step | In blocks |
|---|---|
| pass 1 | per batch of occupied pairs of a group gP and per group gQ of virtual pairs: l3 and T3 gathered as `[P, D, Q]`, D the (d, m) with class(d) - class(m) = gP - gQ.  The density `Dr` is block diagonal in this difference: one GEMM per pair into its block.  The two stacked GEMMs run per class of d and of m, on the whole batch (`[(P,d), (m,Q)]`; a copy of the gathered block where D has more than one block of classes); x and y of the operands `[x, P, d]`, `[(m,Q), y]` are of the class gP - class(d).  `D1` where gP = gQ |
| Gram matrix | per group of triples; put into the C(o,3)^2 array that `LAMBDA[1,2]` read |
| pass 2 | as pass 2 of the T3 equation in blocks: ring with the blocks `[D, D]` of the transposed operator, the outer products where gP = gQ, the drive products per class of the summed index (the four parts of K, each a range per class; the rows of `Bdrv` of a class copied next to each other once per iteration), one reduction per batch (`zc_breduce2`) |
| ladder | per group of triples and class of the free index c: `Wpp[h]^T Lv` with the diagonal pair block h = g - class(c) of the operator (GEMM with the transposed flag), `zc_breduce_ppl`; `Xq` and `X2` on the same gathered batches (P of group h, i of the class of c) |
| hh, update | per group of triples, with the diagonal blocks of `O6T` |
| once: `Xq`, `X2` | `BlockL3.contract_x`: per group of triples and class of the free index; the T3 parts of `W_vvvo`, `W_vooo` from pass 1 of the block engine of the T3 equation, as an amplitude iteration |

**What stays dense**: l1, l2, every operand (`GmQ`, `GPd`, `Adrv`, `Bdrv`,
`Vx`, `Xq`, `X2`; pairs in the order of their groups) and the residuals of
rank 1 and 2 - o v^3-sized arrays and products that do not use the labels.
`Wpp` is the list of its diagonal pair blocks (`pair_matvec` and the ladder
of l2 take the list; `csrc/lambda_t3.c` has `zc_lam_pair_expand_tab`,
`zc_lam_pair_matvec_tab` for pairs in any order).

**Memory** (`plan_memory_blocks`, `LambdaPlan.blocks`): three times the
blocks of T3, the pair blocks of the operator, the dense operands of 9.3
and the blocks of the ring operator, the work array of the block engine
(kept: `SpinBlockT3.buffer_bytes(mem_mb)`; pass 1 here takes two gathered
blocks of a batch, less than pass 2), and per iteration the copies in blocks
of `Bdrv` and `GmQ` (the elements whose labels add up).  With `max_memory`
the buffers shrink (bisection on a need that grows with `mem_mb`: the
largest that fit) to one occupied pair and one triple per batch;
`MemoryError` with the numbers below that.  `plan_for` (both layouts) adds
the bare `<my||Q>`, `<mn||Q>` on pairs and, for the blocks, the pair blocks
of the bare `<ab||de>` where the integral object does not hold them for
these maps: the packed block had 4.39 GB above the start against 4.27 GB of
planned arrays for 8 + 160 spinors without it (job 31740301), inside the
0.7 GB that the plan allows for the BLAS and the allocator.

**Front end**: `GCCSDT.solve_lambda()` solves in blocks when `symmetry` is
set and the iterations ran in blocks, on the packed block otherwise or when
the blocks do not fit in `max_memory` (`lambda_blocks` says which).
`self.l3` has the layout of `self.t3` in every case (after the packed block
its blocks are taken out); `make_rdm1` and `ccsdt_q_lambda` read
`_packed_l3()`, which puts the blocks on the packed block once.

**Validation** (`tests/test_lambda_blk.py`, 45 s), against the equations on
the packed block:

- Random integrals that conserve the labels, amplitudes and Lambda with the
  labels that solve nothing, 5 + 7 spin orbitals with one label and with
  four (an occupied class empty), real and complex, with whole batches and
  with one occupied pair and one triple per batch: each of the six groups
  of terms of the Lambda_3 residual, the three residuals, their parts by
  the rank of Lambda and a Jacobi step to 1e-12 of their size.
- HBr / cc-pVDZ-DK, X2C, 8 + 28 spinors, labels 2 m_j (six of the virtual
  spinors; 31820 of 183456 elements of l3): converged l1, l2, l3 and the
  pseudo-energy `sum f[a,i] l1[i,a] + 1/4 sum <ab||ij> l2[ij,ab]` to 1e-10
  (2e-11, 1e-11, 3e-12 found; the same number of cycles), also with the
  smallest buffers; with one label; with a phase on every spinor (complex
  integrals and amplitudes: against the packed block with these integrals,
  and against the real Lambda with the conjugate phases of t - the Fock
  matrix takes the phases too: without them its 4e-10 between spinors of
  one label left 1.5e-9 in l1).
- The front end: l3 in the layout of T3, the same Lambda without
  `symmetry` and when the plan of the blocks raises.
- Memory: the allocations of the intermediates and of an iteration
  (tracemalloc) within the plan, those of the iteration above 0.85 of it,
  the work array as planned, for 6 + 30 spin orbitals with one label
  (complex) and with six (real and complex); budgets from the floor to all
  buffers: within the budget,
  growing with it, never below a buffer size that fits; `plan_for` with and
  without the bare integrals on the integral object.

**Timing**: `bench/profile_lambda_blocks.py`, HBr on the z axis, X2C, 28
frozen, real integrals (spinors adapted to m_j): amplitudes converged in
blocks, then three amplitude iterations after a warm-up iteration and the
Lambda equations to convergence (17 cycles in both layouts), the packed
block first and the blocks after it in the same process on the same
integrals and amplitudes.  Job 31741024 on c659 (not the login node;
commit df267f3), 8 cores of a Xeon Gold 6248R, MKL, `mem_mb` 2000.
Seconds per iteration (cores busy):

| | cc-pVTZ-DK | cc-pVQZ-DK |
|---|---|---|
| occupied + virtual spinors | 8 + 78 | 8 + 160 |
| labels of the virtual spinors | 8 | 10 |
| elements of l3: packed block, blocks | 4 260 256, 622 138 (14.6%) | 37 515 520, 4 699 202 (12.5%) |
| amplitude iteration, packed block | 0.843 (6.2) | 16.19 (6.5) |
| amplitude iteration, blocks | 0.407 (2.7) | 3.545 (4.4) |
| - ratio | 2.07 | 4.57 |
| **Lambda iteration, packed block** | **0.991** (7.2) | **17.35** (7.7) |
| **Lambda iteration, blocks** | **0.297** (3.2) | **2.040** (5.1) |
| **- ratio** | **3.34** | **8.50** |
| Lambda intermediates, once: packed block, blocks | 0.561, 0.291 | 6.92, 3.13 |
| blocks: pass 1 (density, stacked GEMMs, gathers) | 0.107 (0.020, 0.059, 0.022) | 0.546 (0.129, 0.207, 0.162) |
| blocks: pass 2 | 0.066 | 0.396 |
| blocks: ladder pass | 0.026 | 0.249 |
| blocks: residuals of rank 1 and 2 (dense) | 0.075 | 0.697 |
| blocks: DIIS | 0.011 | 0.082 |
| packed block: pass 1, pass 2, ladder, rank 1 and 2 | 0.345, 0.296, 0.184, 0.109 | 4.61, 4.31, 6.13, 1.66 |
| peak anonymous memory above the start of Lambda: packed block | 1.75 GB | 4.39 GB |
| - planned (`need` less T3) | 2.40 GB | 5.11 GB |
| peak above the start: blocks | 0.34 GB | 2.36 GB |
| - planned | 1.04 GB | 3.11 GB |
| blocks under a budget a quarter of the way from the floor of the plan to all buffers: s per iteration, peak, planned | 0.408, 0.24 GB, 1.00 GB | 2.33, 1.79 GB, 2.91 GB |
| blocks against the packed block: max \|dl1\|, \|dl2\|, \|dl3\| | 4.8e-11, 3.0e-11, 5.0e-12 | 2.0e-11, 7.5e-12, 2.8e-12 |

(The plans contain 0.7 GB for the BLAS threads and the allocator, and for
the blocks the DIIS vectors in memory, 0.08 and 0.60 GB: the blocks are
below `diis.INCORE_SIZE`, the packed block of cc-pVQZ-DK is not.)  The
first run of the script, job 31740301 on the same node with commit 6c6c581
- before `plan_for` counted the bare integrals and before the copies of the
block passes were threaded - gave 0.827, 0.374, 0.892, 0.270 s for
cc-pVTZ-DK and 16.33, 3.516, 17.13, 1.937 s for cc-pVQZ-DK (ratios of the
Lambda iteration 3.31 and 8.84): the two jobs differ by up to 10% in either
direction, in the packed-block numbers too, so the threaded copies bought
nothing that this measurement resolves.

In blocks the three passes over l3 take 0.2 and 1.2 s where the packed
block takes 0.8 and 15 s (the ladder of cc-pVQZ-DK: 25 times less), and
what the labels do not touch is what is left: the residuals of rank 1 and
2 are a quarter and a third of the iteration (0.075 and 0.70 s; 0.109 and
1.66 s on the packed block, where l2 and `Y2` meet the whole pp operator
instead of its diagonal blocks - their parts were not timed one by one).
The cores are less busy than on the packed block (3.2 and 5.1 of 8): many
small products.  A Lambda iteration in blocks costs 0.73 and 0.58 of an
amplitude iteration in blocks, which dresses the integrals and makes the
intermediates with dense operands every time (2.3 of its 3.5 s for
cc-pVQZ-DK).

**Not done**: the residuals of rank 1 and 2 and the o v^3-sized operands in
blocks (the next third of the iteration, and most of its memory: the
dense operands and transients of an iteration are 1.3 GB for 8 + 160
spinors, `LambdaPlan.iteration`, where the blocks of l3 are 38 MB);
labels that add modulo something (`t3pack.classes_of` gives no classes for
them: the packed block); the closed-shell flip of the block engine and
U / R-form integrals; a numpy form of the reductions (without the C kernels
`block_engine` returns None and the front end uses the packed block); a
scratch fallback.

## 10. Densities, orbital gradient and relaxed density from packed amplitudes

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

### 10.1 Factorization of the gradient

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

### 10.2 Frozen and dropped spinors

The orbital gradient between a frozen spinor f (occupied or virtual) and an
active one x is, in the convention of `zccsd.zvector`,

    G[f,x] = (B[f,x] + conj(K[f,x])) / 2 + (n_x - n_f) V[f,x],
    B[f,x] = 1/2 sum_qrs <fq||rs> gamma2[x,q,r,s],   K[f,x] = 1/2 sum_pqs <pq||fs> gamma2[p,q,x,s],

gamma2 the two-particle density of the bare integrals without its
one-particle terms and V the Fock matrix of the Hermitian part of gamma1 =
d1 - 1(occupied). In the dressed basis

    gamma2 = (g2 transformed back) + A[(gamma1 - tau / 2) x tau],   tau = t1 in the occupied-virtual block,

so B and K are the contractions of the blocks of g2 above with the
integrals `<fq||rs>` dressed in their three active indices, transformed back
in the fourth index, plus Fock matrices of gamma1 - tau / 2 and of tau
(`density_opt.frozen_terms`). `relaxed_rdm1` then hands the gradient of all
spinors to `zvector._relax`: the rotations between frozen and active spinors
of a kind follow from the Fock matrix staying diagonal.

**The integrals** come once, from the transformation of the spinor CCSD
with the frozen spinors as the first index (`FrozenIntegrals`:
`zccsd.ao2mo.slabs` on the scalar AO integrals - spin orbitals of a GHF,
real or complex, and j-adapted spinors alike), in slabs (f q|r s) of as many
frozen spinors as the plan allows: (o + v)^3 numbers for each, never all in
memory. A slab is copied, T1-dressed in place (the last index by one GEMM
for the slab, the first by one for each frozen spinor, the middle one by
small products on threads) and cut into six blocks of T[f,a,b,c] = X[f,b,a,c]
- X[f,c,a,b]: `ooo`, `ovo`, `voo`, `vvo`, and the two with a virtual pair,
packed, with the pair first (`T[Q, q, f]`). That is done twice: for B
(`<fq||rs>`: q a bra index, r and s ket indices) and, from the conjugate of
the slab, for K (`<pq||fs>`). Every block of g2 then meets a block of T in
one GEMM, or in one for each value of an index (the packed blocks of g2 with
three virtual indices): the blocks of g2 are laid out for that once, before
the first slab.

**The block `abcd`** is not made, and with frozen spinors there are two
ways to its l3 t3 part (`density_opt.frozen_route`):

| route | what is formed | multiply-adds | memory |
|---|---|---|---|
| `y` | `Y[P,g,m,h,x]` for the pairs of a batch: lambda3 with `<ef\|\|hx>` (and t3 with `<xe\|\|gh>`), one column x for every occupied spinor and two for every frozen one | o^3 v^4 / 4 for each column | a batch |
| `gamma` | `Gamma[(e<f),(g<h)]` = 3 sum_(m<n<o, d) l3[mno, d, (ef)] t3[mno, d, (gh)], one GEMM per batch of occupied triples on `t3pack.gather_vir_half` | o^3 v^5 / 24: a particle-particle ladder | v^4 / 4 elements |

`Gamma` serves the active spinors (in the place of `Y`) and any number of
frozen ones - `Gamma` times a packed block, the pair of `Gamma` that holds
the free index unpacked by ranges of columns (or rows) and by gathers - and
is taken where 6 (2 nfrozen + o) > v and it fits; `y` is then the fallback
of a small budget, as a pass over t3 and lambda3 for every batch of frozen
spinors. With many frozen spinors the frozen part of the T3 work is so one
ladder, whatever their number (in the first version it was two columns of
`Y` for each: 302 of 585 s for HI below).

**Kramers pairs** (`density_opt.kramers_pairs`, the default of
`relaxed_rdm1` where they are found): the frozen spinors are rotated into
pairs f, K f level by level (`zccsd.kramers.pair`), the slabs are made for
the unbarred half and for one barred spinor, and with U[y,x] = <y|K x> of the
active spinors

    B[K f, :] = conj(B[f, :]) U^+,    K[K f, :] = conj(K[f, :]) U^T

if t and lambda are symmetric under time reversal - which the one barred
spinor tells (1e-7 of the row); if it does not agree, the other half is done
as well. The rows are rotated back to the frozen spinors that were given.
Not done for real integrals of complex spinors (`eris.axial_spinors`: the
partner that `kramers.pair` makes need not have real integrals).

**Memory** (`plan_memory(..., nfrozen, source)`): `Gamma` and the buffers
of its pass (2 v C(v,2) elements per occupied triple), eight more arrays of
o^2 v^2 for the layouts of g2, and for a batch of nk frozen spinors the slab,
its dressed copy, the blocks (together about three times (o + v)^3 numbers
for each) and what
the transformation needs for them (`zccsd.ao2mo.plan_slabs`). nk is the
largest number that fits in what the budget leaves (bisection, as
`orbgrad_frozen.slab_plan`); the batches of the passes, the slab of
`<ab||de>` and the frozen stage are not alive together. `MemoryError` if
the integrals of one frozen spinor do not fit.

`density_opt.all_blocks` takes every spinor as the index outside: all four
blocks of the one-index transformed Lagrangian from the same contractions
(n^4 integrals: a check for small cases).

### 10.3 Validation

`tests/test_density_opt.py` (51 tests, 95 s):

- g1, d1 and the table `GRADIENT` for complex amplitudes that solve nothing,
  4+5 to 6+3 spin orbitals, also without triples (o = 2, v = 2), real
  arithmetic, without the C kernels, and in every plan of the memory: 1e-13
  of their size (1e-15 found);
- B and K of three "frozen spinors" with random integrals against the dense
  gamma2 - both routes, batches of one, two and three frozen spinors, the
  integrals asked for once -, and all four blocks of the one-index
  transformed Lagrangian against `lambda_ref.one_index` (1e-15);
- `FrozenIntegrals` against all AO integrals transformed at once, for real
  and complex spin orbitals and for j-adapted spinors (1e-12);
- the Kramers pairs: the same B, K, gradient and relaxed density with half
  of the frozen spinors as with all (H2O / 6-31G, X2C and real spin
  orbitals, CCSDT, 1e-9 of their size), and for a lambda1 that is not
  symmetric under time reversal the check finds it and all are done;
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
  orbitals: 0.76 and 0.85 of the plan), and with frozen spinors in both
  routes, one frozen spinor at a time and all at once (0.62 to 0.83);
- the plan with frozen spinors over 150 budgets (never above the budget;
  `Gamma` where it fits; the batches of frozen spinors grow with the budget
  and one more does not fit), and the slabs of `FrozenIntegrals` over 60;
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

### 10.4 Timing

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

(frozen and dropped spinors: the next table.)

(the plan counts 0.7 GB for the threaded BLAS.)  Where the time goes, real
spin orbitals, all active: pass 2 3.15 s (`Y` 1.19, `R3` 1.09, the products
of t3 alone 0.39, gathers 0.31, `H` 0.16), pass 1 0.38, the blocks 0.24, the
contractions with the integrals 0.24, dressing 0.29, the Z-vector equations
0.08, the Gram matrix 0.01.

**With frozen and dropped spinors.** The first version (6de131c) made the
integrals with one frozen index twice with PySCF's transformation,
contracted them by einsum calls on (o + v)^3 arrays, one frozen spinor
after the other, and had two columns of `Y` in pass 2 for every frozen
spinor. Each of the first three rows is one job with the tree before and
the trees after in turn on one node (8 cores; second of two runs; the ratio is to the CCSDT
iteration of the same process; the checksums of the relaxed density and of
the orbital gradient over all spinors - Frobenius norms and occupation
numbers, which do not depend on how the SCF mixed degenerate spinors -
agree to 12 and 11 digits between the trees):

| | o + v, frozen + dropped | CCSDT iteration | before | after | cores busy | peak memory of the call (plan) |
|---|---|---|---|---|---|---|
| H2O / cc-pVTZ, spin orbitals, real (job 31740284, c659) | 8 + 86, 2 + 20 | 1.15 s | 8.04 s = 7.0 | 2.26 s = 1.96 | 5.6 | 1.3 GB (2.4) |
| H2O / cc-pVTZ, X2C spinors (job 31740285, c561) | 8 + 86, 2 + 20 | 3.31 s | 15.64 s = 4.7 | 4.90 s = 1.48 | 6.3 | 2.6 GB (4.1) |
| HI / dyall.v2z (H: cc-pVDZ, uncontracted), X2C spinors (job 31740283, c659) | 18 + 86, 36 + 116 | 53.3 s, 48.6 s | 717.9 s = 13.5 | 64.0 s = 1.32 | 7.5 | 7.7 GB (14.5) |
| the same, no limit and under `max_memory` 9000 and 8700 MB with the spinors of one SCF (job 31741942, c561; one run each) | 18 + 86, 36 + 116 | 48.6 s | | 63.6 s = 1.31; 69.7 s = 1.43; 72.0 s = 1.48 | 7.5; 7.3; 7.2 | 8.4 GB (10.7); 2.4 GB (3.6), 7.79 GB with what was there; 2.0 GB (3.3), 7.42 GB |

(c659 and c561 were shared with jobs of other users: load 44 to 51 during
the runs of HI before, 7 to 18 during those after - hence the two iteration
times, each measured next to its density. Under 9000 MB the plan has 3
occupied pairs per batch, the dressed `<ab||de>` in slabs and 8 frozen
spinors at a time, under 8700 MB 1 pair and 5; the three runs of the last
row give the same checksums in all 13 digits. On c105, in
separate jobs: 588.3 s = 13.5 iterations before (31738966) and 57.9 s =
1.33 after (31739579), iteration 43.5 s in both.)

Step by step (the trees in turn in one job: H2O with real spin orbitals
31740187 and with X2C spinors 31740201 on c105, second runs; HI in separate
jobs on c105, 31738966, 31739169, 31739579):

| | H2O real | H2O X2C | HI |
|---|---|---|---|
| CCSDT iteration | 0.94 s | 2.76 s | 43.5 s |
| before | 6.83 s | 13.90 s | 588.3 s |
| the integrals once, from `zccsd.ao2mo.slabs`, in batches from the plan; blocks and GEMMs; `Gamma` (as a sum over occupied pairs) | 2.94 s | 7.16 s | 90.7 s |
| `Gamma` as a ladder, the dressing on threads | 2.22 s | 4.50 s | |
| Kramers pairs | 1.98 s | 4.30 s | 57.9 s |
| the gathers on threads (other nodes, see above: 2.31 to 2.26 s, 5.08 to 4.90 s) | | | |

Where the time went for HI before (c105, the first run, 584.5 s): the columns of `Y` of
the frozen spinors 302 s, the integral transformation (twice) 176 s, the
dressing 20 s, the contractions 25 s; everything else 61 s. After (64.0 s,
job 31740283): pass 2 30.6 s (`R3` 18.2), `Gamma` 13.3 s - it replaces the
`Y` of the active spinors, 18.5 s on c105 before -, the Z-vector
equations 6.2 s, and the frozen stage 5.5 s: the transformation of 77
frozen spinors 2.5 s, dressing 1.1 s, the contractions 1.0 s, the blocks 0.6
s. The T3 work does not depend on the number of frozen spinors any more,
and what does - the frozen stage - is a tenth of the whole for 152 of
them.

Under 9000 MB the frozen stage takes 9.7 s in 14 slabs: the gathers of
`Gamma` and of the packed blocks of g2 are done for every slab (their
contractions 3.8 s, against 1.0 s in one slab), and the AO integrals are
made in 5 bra passes (2.8 s against 1.2 s).

What is left for the frozen spinors:

- real spin orbitals go through the complex transformation and take the
  real part; the relaxed density of H2O with real spin orbitals is at 2.0
  iterations because its iteration is short (the densities do not use the
  spin symmetry either);
- no Kramers pairs for the real integrals of `eris.axial_spinors`;
- `Gamma` needs v^4 / 4 elements in memory; without them route `y` costs
  two columns of `Y` for every frozen spinor again (a mapped `Gamma`, made
  in slabs of its columns, would remove that);
- with slabs of few frozen spinors, the gathers per slab (above);
- with no frozen spinors `Y` is kept although `Gamma` would be cheaper
  where v < 6 o (HI: about 13 s against 18 s).
