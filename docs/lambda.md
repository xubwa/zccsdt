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
| `src/zccsdt/cc/qgrad_ref.py`, `quadruples_grad.py` | the first derivatives of CCSDT(Q): sources of Lambda, densities of (Q), relaxed density (section 11); `tests/test_qgrad_ref.py`, `tests/test_quadruples_grad.py` |
| `src/zccsdt/cc/density_opt.py`, `density_blk.py` | densities, orbital gradient and relaxed density from packed amplitudes (section 10) and in the blocks of the symmetry labels (10.5); `tests/test_density_opt.py`, `tests/test_density_blk.py`, `bench/density_ccsdt.py`, `bench/profile_density_blocks.py` |

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
set and the iterations ran in blocks, on the packed block otherwise, when
the blocks do not fit in `max_memory`, or when the packed block has fewer
than `blocks_min` = 800 000 elements - it is the faster one then (10.5:
the measurements) - (`lambda_blocks` says which).
`self.l3` has the layout of `self.t3` in every case (after the packed block
its blocks are taken out); `ccsdt_q_lambda`, and `make_rdm1` where it runs
on the packed block (10.5), read `_packed_l3()`, which puts the blocks on
the packed block once.

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
code stores them: `[C(o,3), C(v,3)]` (10.1 to 10.4), or the blocks of the
symmetry labels, in which the passes over them then run (10.5,
`density_blk.py`). It takes the place of the first and the last item of
section 8 for the densities and the gradient; Lambda is section 9.

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

### 10.5 In the blocks of the symmetry labels (`zccsdt.cc.density_blk`)

`density_opt.kernel(..., blocks=True)` and `relaxed_rdm1(..., blocks=True)`
take t3 and l3 as the amplitude and the Lambda equations in blocks leave
them for integrals with labels in use (9.7: the blocks of
`t3engine.T3Layout`, block g the triples whose classes add up to g) and run
every pass over them in these blocks: `density_blk.BlockDensity`, on the
gathers of the block engines (`lambda_blk.BlockL3._gather`, `zc_bgather2`,
`zc_bgather_vir`). The dense arrays - the blocks of g2 of 10.1, the
integrals, everything with at most three virtual indices - are those of the
packed block, with pairs in the order of their groups, and the code that
makes and contracts them is the same (`kernel` branches where it touches t3
or l3). One class of labels is no symmetry: the blocks are then the packed
block, and the packed block (`blocks=False`) stays the reference and the
fallback.

| Step | In blocks |
|---|---|
| pass 1 (`pass_l`) | per batch of occupied pairs of a group gP and per group gQ of virtual pairs: l3 gathered as `[P, D, Q]`, D the (d, m) with class(d) - class(m) = gP - gQ. `Z` and `C3` are one GEMM each per class of d and of m, on the whole batch (the free index of the class gP - class(d)); the operand `t2[i,m,Q]` is copied once per (class of m, gQ, class of i) |
| pass 2 (`pass_tl`) | t3 and l3 gathered next to each other. The two stacked GEMMs of t3 with (l2, `Z`) and (l2, `C3`) per class of d and of m; `R3` one GEMM per occupied pair into its block of the difference of classes (block diagonal: sum n_D^2 elements, scattered into the o^2 v^2 array at the end); the products with l1 and l2 where gP = gQ; `H` for the occupied pairs of the group gQ only (t2 of a pair P' and a pair Q vanishes otherwise) and its product with t3 into the block [gQ, gQ] of the o^5 v intermediate |
| `Gamma` (`gamma`) | always - the l3 t3 part of the block `abcd` goes through `Gamma` in blocks, never through `Y` -: its diagonal pair blocks, sum n_h^2 elements for blocks of n_h pairs (as the pp ladder operator of the amplitude equations), per group of occupied triples and class of the free virtual index one GEMM on the arrays with that index apart |
| `Gamma` times `<ef\|\|gi>` and the integrals of a frozen index (`gamma_pairs`) | per pair block and per pair of classes of the two indices of the unpacked pair: these pairs are a range of rows of the block, so each of the two terms is one GEMM (one with the antisymmetric square where the classes are equal); copies of at most 64 MB of rows of `Gamma` at a time |
| Gram matrix, `O4` | per group of triples (`BlockL3.gram`) |
| the dressed `<ab\|\|de>` (`Slabs`) | its diagonal pair blocks (`ccsdt_opt.dressed_vvvv_rows(..., block=h)`): all kept if they fit, else dressed twice in slabs of the rows of a block; `aibc` meets a block with the columns of that block |

**What stays dense**, and what it costs (HBr / cc-pVQZ-DK, the profile
below): the integrals with one frozen index (their transformation, the
dressing, the six blocks and their products with the blocks of g2 - the
frozen spinors carry no labels here), the Z-vector equations of the SCF,
and the blocks of g2 with their contractions (the o^2 v^4 product of
`aibc`, t2 with `C3`; the loops over a virtual index). The labels are not
used in any of them.

**Memory** (`plan_memory_blocks`, `Plan.blocks`): the dense arrays of
`plan_memory` (`fixed_bytes`), the copies in blocks of the operand of the
stacked GEMMs and the blocks of `R3` (`BlockDensity.operand_bytes`), the
blocks of `Gamma`, the work array of the block engine - l3 and t3 gathered
for a batch of occupied pairs of one group and one group of virtual pairs,
with `H`; t3 and l3 with one class of the free virtual index apart for a
batch of occupied triples; as many pairs and triples as fit, at most 4 GB;
given back after the passes -, the pair blocks of the dressed `<ab||de>`
(kept if they all fit next to the batches, else slabs of the rows of a
block), and the frozen stage as in 10.2. `MemoryError` with the numbers
below one pair and one triple per batch and the rows of one first virtual
index of the widest pair block, or below the integrals of one frozen
spinor. `kernel` takes the pair blocks of the bare `<ab||de>` from the
budget where the integral object does not hold them for these maps
(`bare_bytes`). One class: the whole `Gamma` and at least `v` rows of
`<ab||de>` - the packed block with route `y` needs less.

**Front end**: `GCCSDT.make_rdm1()` runs in blocks when `symmetry` is set,
T3 and l3 are in blocks, the plan fits (`relaxed_rdm1(..., plan_only=True)`
inside `try`, the run outside it; what was prepared for the frozen spinors
is kept with the plan, `Plan.held`) and the packed block has at least
`blocks_min` elements (below); else on the packed block (`_dense_view`,
`_packed_l3`). `rdm1_blocks` says which.

**Validation** (`tests/test_density_blk.py`, 26 tests, 65 s), against the
code on the packed block:

- Every product of the passes one by one - `Z`, `C3`, the two stacked
  GEMMs of t3, the terms with l1 and l2, `R3`, the o^5 v intermediate - as
  the sum over the half-expanded arrays, `Gamma` as the diagonal blocks of
  the full matrix (nothing outside them) and its product with random
  three-index integrals: 5 + 7 spin orbitals with one label and with four,
  real and complex, whole batches and one pair and one triple per batch
  (1e-13 of their size).
- d1, g1, the table `GRADIENT` and the contractions B and K with the
  integrals of three "frozen spinors" (random, without labels) for
  amplitudes and Lambda that solve nothing, the same models, in three
  plans (everything at once; one pair, one triple, one row of a pair block
  and one frozen spinor at a time; between them): 1e-13 of their size
  (1e-15 found); the density alone; CCSD (no t3) and two occupied spinors.
- HBr / cc-pVDZ-DK, X2C, 28 frozen and 8 + 28 active spinors, labels 2 m_j
  (31820 of 183456 elements of t3 and l3), converged amplitudes and
  Lambda: the relaxed and the unrelaxed density, d1, `Gnh[a,i]`, the
  orbital gradient over all 64 spinors and the solution of the Z-vector
  equations to 1e-10 (5e-14, 4e-14, 8e-14, 1e-13, 6e-14, 5e-14 found), also
  with the smallest plan and with the plan made by a call of its own; with
  one label (3e-18: the same operations); with a phase on every active
  spinor - complex integrals, amplitudes and Lambda, the Fock matrix with
  the phases too - against the packed block with these integrals (1e-13
  found) and, in the AO basis, against the real densities (1e-9 asserted);
  with 4 dropped virtual spinors (28 frozen, 8 + 24 active; amplitudes
  after three iterations, their conjugates as Lambda: 3e-13 found).
- The front end: in blocks with `symmetry`, on the packed block without it,
  when the plan of the blocks raises, and below `blocks_min` - the same
  density (1e-10).
- Memory: the allocations of `kernel` (tracemalloc, the work array of the
  block engine among them) within the plan, for the smallest plan and with
  no limit, 6 + 30 spin orbitals with one label and with six and 4 + 44
  with four, with and without frozen spinors (0.49 to 0.73 of the plan
  without its 0.7 GB for the BLAS threads: the o^2 v^2 arrays that
  `fixed_bytes` counts are most of a plan this small), the work array given
  back; the plan over 120 budgets from its floor to everything: never above
  the budget, the need growing with it, one more pair, triple or frozen
  spinor per batch does not fit, the pair blocks kept where they fit next
  to the batches, `MemoryError` below the floor.
- At production size, in the timing job below: blocks against the packed
  block on the same amplitudes and Lambda, max differences of the relaxed
  density, the unrelaxed one, `Gnh[a,i]`, the gradient over all spinors and
  z: 3.2e-13, 2.1e-13, 2.6e-13, 1.3e-13, 3.2e-13 (cc-pVTZ-DK) and 1.1e-13,
  1.1e-13, 1.0e-13, 5.1e-14, 1.1e-13 (cc-pVQZ-DK); the checksums of 10.4
  are the same in all printed digits, also under the budget. Through the
  front end, with and without `symmetry` (two CCSDT, Lambda and density
  calculations from the spinors of one SCF): the relaxed density in the AO
  basis to 1.1e-11 and 5.8e-12.

**Timing**: `bench/profile_density_blocks.py stages`, HBr on the z axis,
X2C, 28 frozen spinors, real integrals (spinors adapted to m_j): amplitudes
converged in blocks, then - the packed block first and the blocks after it,
in one process on the same integrals, amplitudes and Lambda - three
amplitude iterations after a warm-up iteration, the Lambda equations to
convergence (17 cycles in both layouts), and everything of `relaxed_rdm1`
(densities, orbital gradient with the 28 frozen spinors, Z-vector
equations) twice, the second run in the table. Job 31744031 on c028 (not
the login node; load 0.6 before it; snapshot of commit 584fc20), 8 cores
of a Xeon Gold 6248R, MKL. Seconds (cores busy):

| | cc-pVTZ-DK | cc-pVQZ-DK |
|---|---|---|
| occupied + virtual spinors, frozen | 8 + 78, 28 | 8 + 160, 28 |
| labels of the virtual spinors | 8 | 10 |
| elements of t3 and of l3: packed block, blocks | 4 260 256, 622 138 (14.6%) | 37 515 520, 4 699 202 (12.5%) |
| `Gamma`, 8 bytes an element: v^4 / 4, its pair blocks | 0.072 GB, 0.010 GB | 1.29 GB, 0.15 GB |
| amplitude iteration, packed block | 0.813 (6.2) | 13.92 (6.8) |
| amplitude iteration, blocks | 0.356 (2.6) | 2.566 (4.4) |
| - ratio | 2.28 | 5.43 |
| Lambda iteration, packed block | 0.895 (7.2) | 15.51 (7.7) |
| Lambda iteration, blocks | 0.277 (3.1) | 1.641 (5.2) |
| - ratio | 3.23 | 9.45 |
| **relaxed density, packed block** | **2.265** (5.6) | **22.98** (6.8) |
| **relaxed density, blocks** | **1.743** (4.4) | **10.92** (5.5) |
| **- ratio** | **1.30** | **2.10** |
| - in amplitude iterations of its layout: packed block, blocks | 2.79, 4.89 | 1.65, 4.26 |
| the passes over t3 and l3 (pass 1, pass 2, `Gamma`, Gram): packed block | 0.67 (0.071, 0.351, 0.242, 0.004) | 11.30 (0.917, 4.305, 6.054, 0.025) |
| - blocks | 0.18 (0.050, 0.099, 0.032, 0.000) | 0.87 (0.157, 0.365, 0.346, 0.001) |
| - ratio | 3.7 | 13.0 |
| blocks: the frozen stage (transformation, dressing, cutting, contraction, product with `Gamma`) | 0.654 (0.360, 0.083, 0.046, 0.086, 0.059) | 4.322 (1.896, 0.894, 0.488, 0.410, 0.503) |
| blocks: Z-vector equations | 0.335 | 2.113 |
| blocks: not under a timer - J - K of the SCF density with the integrals of the SCF, the core Hamiltonian, small steps | 0.25 | 1.35 |
| blocks: the blocks of g2 (of which the term of `aibc` with t2 and `C3`, o^2 v^4) | 0.104 (0.066) | 0.861 (0.623) |
| blocks: their contraction with the integrals | 0.150 | 0.776 |
| blocks: dressing, with `<ab\|\|de>` | 0.071 | 0.618 |
| peak anonymous memory above the start of the call: packed block (plan) | 0.86 GB (1.98) | 8.65 GB (10.33) |
| - blocks (plan) | 0.84 GB (1.75) | 6.39 GB (8.13) |
| blocks under a budget a quarter of the way from the floor of the plan to all of it: s, peak, plan | 2.173, 0.25 GB, 1.15 GB | 12.95, 2.07 GB, 3.82 GB |

(The plans contain 0.7 GB for the BLAS threads and the allocator; most of
the peak is the frozen stage with all 28 frozen spinors in one slab. Under
the budget: 5 frozen spinors at a time, the pair blocks of `<ab||de>`
dressed twice.) The same script in job 31743861 on c116 (snapshot of
993e812: before the Fock matrix of the frozen spinors and the check of
`scf_veff` shared their J - K, which took 0.87 s out of the Z-vector stage
for cc-pVQZ-DK) gave 0.685, 0.286, 0.757, 0.242, 2.138, 1.714 s for
cc-pVTZ-DK and 11.45, 1.979, 12.52, 1.224, 21.00, 10.22 s for cc-pVQZ-DK
(ratios of the relaxed density 1.25 and 2.05): that node was a sixth to a
fifth faster in everything, the ratios are those of the table.

The passes over t3 and l3 take 0.18 and 0.87 s in blocks where the packed
block takes 0.67 and 11.3 s, and they are a tenth and a twelfth of the
call then. What the labels do not touch is the rest: for cc-pVQZ-DK the
integrals with a frozen index 4.3 s (their transformation 1.9 s), the
Z-vector equations 2.1 s, one build of J - K with the spinor integrals of
the SCF 1.3 s, the dense blocks of g2 and their contractions 1.6 s. The
relaxed density in blocks costs 4.3 to 4.9 amplitude iterations in blocks
(1.7 to 2.8 on the packed block, where the iteration is the slow part).

The whole front end (`bench/profile_density_blocks.py front`, the same
job; `GCCSDT` from the spinors of one SCF, `symmetry` False and then True;
conv_tol 1e-10, |dT| 1e-9; wall seconds):

| | cc-pVTZ-DK, no symmetry | with | cc-pVQZ-DK, no symmetry | with |
|---|---|---|---|---|
| SCF (X2C-UHF, once) | 3.5 | | 18.9 | |
| integrals and CCSDT (19 iterations) | 24.82 | 15.78 | 368.5 | 153.0 |
| Lambda (17 iterations) | 15.25 | 4.98 | 252.9 | 29.90 |
| relaxed density | 2.53 | 1.78 | 23.59 | 11.04 |
| all three | 42.60 | 22.53 | 645.0 | 193.9 |
| - ratio | | 1.89 | | 3.33 |

(E(CCSDT) is the same in all 16 digits. The integral transformation is in
the second row: 8.4 and 102 s where the first run of the job timed it by
itself - more than half of the run with symmetry for cc-pVQZ-DK. On c116,
job 31743861: 35.61 and 19.32 s, 546.8 and 158.2 s, ratios 1.84 and 3.46.)

**Small molecules**: in blocks every pass is many small products, and
below a size the few large ones of the packed block are faster. Measured
with the same script (`stages`: five amplitude iterations, Lambda to
convergence, the relaxed density three times), each case the packed block
and the blocks in one process, the cases of a sample one after the other
in one job: job 31743587 on c703 (snapshot of 81b1f05) and job 31743890 on
c547 (993e812), nodes that other jobs shared (load 38 falling to 6, and 35
to 6, during the jobs; the two samples differ by up to 30% in the times of
the packed block). Time on the packed block over time in blocks, first and
second sample:

| | o + v | packed block, blocks (elements) | amplitude iteration | Lambda iteration | relaxed density |
|---|---|---|---|---|---|
| HBr / cc-pVDZ-DK, 28 frozen | 8 + 28 | 183 456, 31 820 | 0.46, 0.53 | 0.29, 0.34 | 0.78, 0.82 |
| HBr / cc-pVTZ-DK, 28 frozen, 38 dropped | 8 + 40 | 553 280, 75 378 | 0.66, 0.78 | 0.67, 0.83 | 0.93, 0.96 |
| - 32 dropped | 8 + 46 | 850 080, 119 866 | -, 1.00 | -, 1.11 | -, 1.00 |
| - 26 dropped | 8 + 52 | 1 237 600, 170 662 | 1.06, 1.20 | 1.16, 1.37 | 0.98, 1.04 |
| N2 / cc-pVDZ, 4 frozen | 10 + 42 | 1 377 600, 224 144 | 1.08, 1.19 | 1.38, 1.58 | 0.94, 1.15 |
| HBr / cc-pVTZ-DK, 28 frozen, 14 dropped | 8 + 64 | 2 333 184, 324 190 | 1.64, 1.75 | 1.95, 2.20 | 1.05, 1.10 |
| HF / cc-pVTZ, 2 frozen | 8 + 78 | 4 260 256, 622 138 | 2.33, 2.30 | 3.04, 3.26 | 1.47, 1.74 |

(Lambda iteration, packed block and blocks, second sample: 0.033 and 0.098
s, 0.120 and 0.144, 0.177 and 0.159, 0.240 and 0.175, 0.253 and 0.161,
0.466 and 0.212, 0.905 and 0.277 s. The relaxed density of the first
sample is that of the first commit of the blocks.) The Lambda iteration
crosses between 0.55 and 0.85 million elements of the packed block in both
samples and the amplitude iteration at the same place. The relaxed density,
whose passes over t3 and l3 are a small part of it at these sizes, is
within 7% of even from 0.55 to 1.2 million elements and ahead in blocks
from 2.3 million on (N2 with 1.4 million: 6% behind in one sample, 15%
ahead in the other). The rule of the front end, for `solve_lambda` and
`make_rdm1` alike: in blocks where the packed block [C(o,3), C(v,3)] has at
least `GCCSDT.blocks_min` = 800 000 elements, on the packed block below
(`_blocks_pay`; `self.l3` has the layout of `self.t3` in either case). The
amplitude iterations are not under the rule: they run in blocks whenever
the labels are in use, at half the speed of the packed block for 8 + 28
spinors (0.08 against 0.04 s per iteration).

**Not done**:

- the frozen stage, the blocks of g2 and their contractions in the blocks
  of the labels (above: what they cost; the integrals with a frozen index
  would need the labels of the frozen spinors, which `relaxed_rdm1` does
  not have - it takes any frozen spinors, and rotates them into Kramers
  pairs where it can);
- route `y` in blocks: `Gamma` is always made, sum n_h^2 elements; with
  one label that is the whole v^4 / 4 matrix, which the packed block
  avoids where it does not fit (one label in blocks takes the time of the
  packed block: 3.3 s both for cc-pVTZ-DK on the login node with 4
  threads - not a benchmark);
- `gamma_pairs` is bound by its copies of rows of `Gamma` (0.5 s at 1.7
  cores for cc-pVQZ-DK, 28 frozen spinors twice), and the stacked GEMMs of
  the passes by their number (2 to 5 cores busy);
- the pair blocks of the bare `<ab||de>` are taken in memory
  (`ccsdt_opt.vvvv_packed` on grouped maps), also where the packed
  integrals are a mapped file;
- as in 9.7: labels that add modulo something, the closed-shell flip of
  the block engine, U / R-form integrals, a numpy form of the gathers
  (without the C kernels `density_blk.engine` returns None and the front
  end uses the packed block).

## 11. First derivatives of CCSDT(Q)

| File | What |
|---|---|
| `src/zccsdt/cc/qgrad_ref.py` | the dense reference: the derivatives of E(Q) from the term table, the relaxed density from full arrays |
| `src/zccsdt/cc/quadruples_grad.py` | the same from packed amplitudes, one occupied quadruple at a time; the orbital gradient of E(Q); the terms of the frozen spinors |
| `src/zccsdt/cc/quadruples_grad_sym.py`, `csrc/tnpack.c` (`zc_qg_sector_vectors`, `zc_xv_sector_adj`) | the pass in the sectors of the symmetry labels |
| `lambda_ref.kernel(sources=)`, `lambda_opt.kernel(sources=)` | the Lambda equations with constant terms |
| `density_opt.relaxed_rdm1(q=)` | the density and the Z-vector equations of CCSDT(Q) |
| `GCCSDT.solve_lambda(with_q=True)`, `make_rdm1(with_q=True)` | the front end (`GCCSDTQ`: through its CCSDT step) |
| `tests/test_qgrad_ref.py`, `tests/test_quadruples_grad.py`, `tests/test_quadruples_grad_sym.py` | the checks of 11.5 |
| `bench/qgrad_ccsdt_q.py`, `bench/profile_q_gradient.py`, `bench/slurm_q_gradient.sh`, `bench/slurm_q_gradient_sectors.sh` | finite fields at several steps; timing |

### 11.1 The functional

(Q) is not variational and not holomorphic. With canonical Hartree-Fock
spin orbitals and bare integrals (`ccsdtq_ref.quadruples`),

    E(Q) = Re sum conj(z) r / D                  (i<j<k<l, a<b<c<d),
    r = A(g, t3) + B(g, t2, t2),   A = <Q| [V,T3] |0>,   B = <Q| 1/2 [[V,T2],T2] |0>,
    z = Z(g, t2) + A(g, t3),       Z = <Q| V T2 |0>  (disconnected),
    D = e_i + e_j + e_k + e_l - e_a - e_b - e_c - e_d:

a real function of t2, t3, of g = `<pq||rs>` and of the orbital energies.
**t1 does not enter** (no source for Lambda_1), and neither does the
off-diagonal Fock matrix. The Lagrangian of CCSDT(Q) is the real function

    F = Re L(t, lambda; f, g) + E(Q)(t2, t3; g, e),

L the holomorphic Lagrangian of CCSDT of section 1. **Complex amplitudes**:
t and conj(t) are taken as independent, as the CCSD(T) code does
(`zccsd.lam_t`). With y = r / D and w = z / D,

    dE(Q) = Re sum [ conj(y) dz + conj(w) dr ] - sum Re(conj(w) y) dD,

and dz, dr are linear in dt and dg with nothing conjugated, so

    dE(Q) = Re [ sum_(i<j,a<b) s2 dt2 + sum_(i<j<k,a<b<c) s3 dt3 + sum d2[p,q,r,s] dg[p,q,r,s] ]
            + sum_p de[p] d e_p

with conj(y) and conj(w) as left-hand quadruple amplitudes L(.):

| | made of | goes to |
|---|---|---|
| s3 | `<0| L(y+w) [V, tau3] |0>` | the Lambda_3 equation |
| s2 | `<0| L(y) (V tau2)_disc |0> + <0| L(w) [[V,T2], tau2] |0>` | the Lambda_2 equation |
| d2 of `<cd||el>`, `<md||kl>` | conj(y+w) t3 | the orbital gradient (o v^3 / 2 and o^3 v / 2 elements) |
| d2 of `<cd||kl>` | conj(y) t2 | the orbital gradient |
| d2 of `<bc||ef>`, `<mn||jk>`, `<mb||ek>` | conj(w) t2 t2 | the orbital gradient (the first: v^4 / 4 elements) |
| de[i], de[a] | -/+ the sum of Re(conj(w) y) over the quadruples with i, with a | the density, and the gradient through the Fock operator |

F is stationary in t when

    dL/dt_k + s_k = 0,   s1 = 0:

**the Lambda equations of CCSDT with s2 and s3 as constant terms of the
residuals.** For real quantities s is the derivative of E(Q), for complex
ones twice its Wirtinger derivative; the equations stay holomorphic in
lambda and the solver is the one of section 9 (`sources=`). This lambda is
not the one of CCSDT(Q)_Lambda (section 5), which is that of CCSDT.

### 11.2 Densities and the response of the spin orbitals

With that lambda the derivative of F with respect to a perturbation is the
derivative of its explicit dependence on the integrals:

- the densities of the CCSDT Lagrangian (section 1), made Hermitian - the
  real part;
- `de` on the diagonal of the one-particle density (e_p = f[p,p] has the
  perturbation in it);
- the spin orbitals follow the perturbation as canonical Hartree-Fock
  spin orbitals, and F depends on all their rotations:
  - occupied-virtual: the Z-vector equations with the orbital gradient
    G[a,i] = (Gnh[a,i] - conj(Gnh[i,a])) / 2. For the CCSDT part Gnh[i,a] is
    minus the Lambda_1 residual without its source, zero; for E(Q) both
    halves are there (the function is not holomorphic);
  - **occupied-occupied and virtual-virtual: not redundant**, as for (T).
    The canonical condition fixes them, y_pq = G_pq / (e_q - e_p), which
    goes into the density and, as a Fock matrix, to the right-hand side of
    the Z-vector equations (`zccsd.zvector._relax`, the code of CCSD(T) and
    of the frozen spinors); none between degenerate spin orbitals, where
    G vanishes.

The one-index transformation of E(Q) is that of `zvector.gradient_nh`
(a bra index y of an integral taking the values of x, a ket index x minus
those of y, in all four positions of d2; the Fock operator with its
occupied spin orbitals for `de`).

Three identities carry the implementation, each checked against the dense
densities (`tests/test_qgrad_ref.py`, `tests/test_quadruples_grad.py`):

1. *The Lagrangian among occupied and among virtual spin orbitals.* L does
   not change when t, lambda and the integrals are rotated together, and
   dL/dt = -s now, so (bare and T1-dressed basis alike, s1 = 0)

       Gnh[j,i] = -1/2 sum s2[j,k,a,b] t2[i,k,a,b] - 1/12 sum s3[j,k,l,a,b,c] t3[i,k,l,a,b,c]
       Gnh[b,a] =  1/2 sum s2[i,j,a,c] t2[i,j,b,c] + 1/12 sum s3[i,j,k,a,c,d] t3[i,j,k,b,c,d].

   (BH and LiH / STO-3G: 1e-13 off the one-index transformation of the
   dense densities, blocks of 1e-4 and 3e-7.)

2. *The table `GRADIENT` is no longer the bare gradient.* It is the
   de-excitation one-index transformation in the dressed basis (section 4),
   equal to the bare one when the other blocks vanish. With sources

       Gnh[a,i] = GRADIENT[a,i] + sum_j t1[j,a] Gnh[j,i] - sum_b t1[i,b] Gnh[a,b]

   (from exp(-T1) a+_i a_a exp(T1); the term with the excitation block has
   s1 = 0). BH / STO-3G: the table alone is off by 2.8e-6 in a gradient of
   1.1e-5, the formula by 8e-16.

3. *The whole gradient among occupied and among virtual spin orbitals is a
   one-particle density of (Q).* Written with y the solution of D y = r for
   the whole occupied-occupied and virtual-virtual Fock matrix, (Q) needs no
   canonical orbitals and does not change under these rotations (with t);
   it differs from E(Q) by the off-diagonal f alone. So

       G[p,q] = (e_q - e_p) (Gamma[q,p] + conj(Gamma[p,q])) / 2,
       Gamma[a,e] =  sum conj(w[ijkl,abcd]) y[ijkl,ebcd]     (i<j<k<l, b<c<d),
       Gamma[m,i] = -sum conj(w[ijkl,abcd]) y[mjkl,abcd]     (j<k<l, a<b<c<d),

   Gamma the derivative of that form with respect to f (its diagonal is
   `de`): the rotation y_pq is the Hermitian part of Gamma, transposed. The
   production code uses it for the virtual block (one product per
   quadruple); the occupied block would need two quadruples at a time and
   is made from identity 1 and the densities instead. (LiH and BH /
   STO-3G: 3e-15 and 8e-14 off for blocks of 6e-9 and 4e-6.)

Frozen occupied and dropped virtual spinors f: E(Q) is that of the active
spinors, and the gradient between f and an active spinor x has the
densities of (Q) with the integrals of one frozen index next to those of
the Lagrangian (section 10.2): b[f,x] with f in a bra place, k[f,x] in a ket
place, G[f,x] = (b + conj(k)) / 2, added to `bf` and `kf` of
`density_opt.kernel` batch by batch (`quadruples_grad.FrozenTerms`; the
integrals are made once, and the Kramers pairs of the frozen spinors are
used as before). `de` is in the density whose Fock matrix is the term V of
10.2.

### 11.3 The dense reference (`zccsdt.cc.qgrad_ref`)

`quadruples_gradient(t2, t3, g, mo_energy, nocc)`: for one occupied
quadruple at a time, r and z from the term table, then for every term
`coefficient * A(product)` and every factor of it the product with the
left-hand quadruple in the place of that factor (the sum of L A(product)
over the unique quadruples is the sum of L product over all), added to s2,
s3 or d2. `relaxed_rdm1(mf, mo_coeff, nfrozen, ndrop)`: CCSDT, the
derivative of E(Q), `lambda_ref.kernel(sources=)`, the densities of the
Lagrangian over all spin orbitals (the frozen ones occupied),
`zvector.orbital_gradient` on them, `zvector.gradient_nh` for E(Q),
`zvector._relax`. None of the identities of 11.2 is used.

### 11.4 From packed amplitudes (`zccsdt.cc.quadruples_grad`)

One pass over the occupied quadruples like (Q) (`QuadruplesGradient` is a
`Quadruples`): after r and z of a quadruple, conj(y), conj(w) and their sum
are expanded to the layouts of the five products (`t4pack.tn_expand`, the
adjoint of the antisymmetrizing reduction), and every product X = A B is
followed by X_ B^T for the operand A and A^T X_ for the operand B:

| product of (Q) | with | its two adjoints |
|---|---|---|
| `t(ij,ab) <cd||kl>` | conj(y) | s2; the density of `<cd||kl>` |
| `- sum_e <ab||ei> t(jkl,cde)` | conj(y) + conj(w) | the density of `<ab||ei>`; s3 (reduced onto c<d<e) |
| `sum_m <ma||ij> t(mkl,bcd)` | conj(y) + conj(w) | the density of `<ma||ij>`; s3 (rows) |
| `sum_m t(im,ab) Xo(m;jkl;cd)` | conj(w) | s2; the adjoint of Xo |
| `2 sum_e t(ij,ae) Xv(e;kl;bcd)` | conj(w) | s2; the adjoint of Xv |

The adjoint of Xo (o^4 v^2 / 12) is kept for all quadruples and goes back
through the definition of Xo at the end: s2 and the densities of
`<mn||jk>`, `<md||el>`. The adjoint of Xv lives next to Xv, for the same
pairs (x, l), and goes back when Xv of the pair leaves: s2, and the
density of `<bc||ef>` on pairs, with the packed `<vv||vv>` in the slabs of
rows that made Xv. Gamma of the virtual spin orbitals is one more product
per quadruple, on the expanded conj(w) and y.

The disconnected product z has rank-six operands. They are six more rows
under the stacked operands of the two other products on pairs, and its
adjoints are the differences of theirs, conj(y) = (conj(y) + conj(w)) -
conj(w): two expansions on pairs per quadruple instead of three, no sum of
two expanded arrays, no products with six rows that only stream through
the expanded array.

**In the sectors of the symmetry labels** (`quadruples_grad_sym`, the pass
of `quadruples_sym` with its adjoints; labels 2 m_j, a modulus, one label =
no symmetry). For an occupied quadruple of total label M the products are
formed in their blocks as for the energy; `zc_qg_sector_vectors` walks
through the virtual quadruples of the sector, sums the splits of each (r
and z), and writes conj(y), conj(w), y back with their signs to the places
that the splits were read from - every place of a block belongs to one
split of one quadruple, so the blocks of the expanded left-hand quadruples
are filled without tables and without races (the places of groups that
share an index stay zero). Then, block by block, X_ B^T and A^T X_ on the
rows and columns that were picked for X = A B; s3 is scattered into the
packed rows of the occupied triples. The adjoint of Xv is in the blocks
of Xv and goes back through the blocks `<bc||ef> t(xl,fd)`
(`zc_xv_sector_adj`); the density of `<bc||ef>` has the label of (bc) on
(ef) and is kept in the diagonal blocks of the pair labels
(`quadruples_grad.PairBlocks`: sum |Q2(q)|^2 elements, a seventh of
C(v,2)^2 for HBr / cc-pVTZ-DK). The adjoint of Xo goes back through the
dense code (o^4 v^2 (o + v) multiplications). The dense pass keeps the
symmetry-breaking elements of the amplitudes (1e-10 of the others after
the iterations) and the sectors do not have them; the derivatives are
linear in them, so the two passes agree to 1e-10 on converged amplitudes
and to 2e-15 on their label-conserving parts.

Then (`orbital_gradient`, `active_terms`): the densities with at most two
virtual indices as dense blocks, every one with the blocks of integrals
that one changed index gives - matrix products; all integrals with an
occupied index are kept by the amplitude code. The density of `<ab||ei>`
stays on its pairs [i, e, (a<b)]: with `<kb||ei>` (one occupied index of
the density at a time, the pair unpacked: v^3 elements), `<ab||ki>` and
`<ab||ek>` on pairs, and with the packed `<cd||ea>` in slabs of rows. The
density of `<bc||ef>` meets `<ie||bc>` on pairs (`_vvvv_terms`): it is read
in slabs of rows - an array in memory, a mapped one, or the blocks of the
pair labels. In the virtual block the density of `<bc||ef>` would meet
`<vv||vv>`; identity 3 replaces that block. The parts of identity 1 with
s3 and t3 are made at the end of the pass, where T3 is on the packed
block (`Gradient.src3`): the density step needs no packed copy of T3 when
it runs in the blocks of the labels.

Frozen spinors (`FrozenTerms`): a batch at a time, as the frozen stage of
`density_opt` has its own terms - the integrals `<fu||vw>` of the batch
once, every density with at most two virtual indices laid out for each of
its four indices and one matrix product with the block of the integrals of
the batch; the densities of `<ab||ei>` and `<bc||ef>` with `<fe||ab>` on
pairs.

Cost: every product of (Q) twice more (three times its products in all:
o^4 v^4 (v + o) / 4 multiplications), five expansions (two on pairs, three
on an index and a triple), Gamma (o^4 v^5 / 144), Xv twice more (o^2 v^5 / 2); the
contractions with the integrals are o v^5 / 2 and less.

Memory. Nothing of it only raises where a smaller batch or a scratch file
is possible:

- the pass on the packed block (`plan`): what is returned - s3 (the size
  of T3), the density of `<bc||ef>` (C(v,2)^2 elements: the size of the
  packed `<vv||vv>`), that of `<ab||ei>` (o v C(v,2)) and the small ones -,
  the adjoint of Xv next to Xv, three work arrays per batch of columns for
  one of (Q), and the adjoints of the stacked operands. The batches are
  those of the plan of (Q) with the largest budget of its own for which
  all of this fits (bisection). **Where the largest batches do not fit, s3
  and the density of `<bc||ef>` go to a scratch file** (`Plan.mapped`,
  `zccsdt.scratch`; the density is written in the slabs of rows of Xv and
  read in slabs, s3 by rows of occupied triples) and the batches take the
  memory; with the largest batches the density alone is mapped if s3
  fits, and nothing if both do - so the plan is monotone in the budget
  (the batches never shrink, the mapped set never grows) and uses it (0.9
  of every budget while the batches grow, in the tests). MemoryError with
  the numbers below the smallest batches with both arrays mapped;
- the pass in sectors (`quadruples_grad_sym.plan`): the plan of (Q) in
  sectors, what is returned (the density of `<bc||ef>` in its blocks), five
  expanded blocks and the stacked operands of a quadruple, the adjoint of
  Xv with the work of its way back. The front end plans the sectors first
  and takes the packed block if they do not fit (`q_gradient_sectors`);
- Lambda: s3 in memory in the layout of the residual is used as it is;
  the copy in the blocks of the labels, or of a mapped s3, is in the plan
  (`lambda_opt.plan_for(sources=)`);
- the density step (`relaxed_rdm1(q=)`): the work of the orbital gradient
  (`active_bytes`: the largest of its three stages, 2 o v^3 elements and
  less) is checked against the budget before anything is done and is the
  least of `Plan.need`; the work of a batch of frozen spinors
  (`FrozenTerms.own(nk)`: n^3 + n v^2 elements per frozen spinor) is in the
  plan of the batches (`plan_memory(extra_own=)`), which takes as many
  frozen spinors per batch as fit.

Front end: `GCCSDT.solve_lambda(with_q=True)` makes the pass
(`_q_terms`, kept for `make_rdm1`) and solves with the sources - in the
blocks of the symmetry labels or on the packed block by the rule of 10.5;
`make_rdm1(with_q=True)` solves again if the lambda that is there is the
other one, and so do `make_rdm1()` and `ccsdt_q_lambda()`.

**Not done**:

- the adjoint of Xo and the orbital gradient of E(Q) among the active
  spinors do not use the labels (small: o^4 v^2 and o v^5 / 2);
- s3 of the pass in sectors is on the packed block (the pass reads T3
  there), and the plan of (Q) in sectors counts Xv of an occupied index
  by its dense size (v C(v,3)) where its blocks have a part: with little
  memory it holds fewer indices at a time than would fit;
- the two-particle density of CCSDT(Q) as an array, geometric gradients,
  non-canonical or non-Hartree-Fock references;
- the unrelaxed density of CCSDT(Q) is returned but is not the derivative
  of anything at fixed orbitals;
- between degenerate spinors no rotation is made (Gamma does not vanish
  there by itself; a perturbation that couples the spinors of a degenerate
  level would need it).

### 11.5 Validation

Derivatives of E(Q) (5 + 5 spin orbitals, random amplitudes and integrals,
four-point differences of the dense E(Q) along random directions, step
1e-3): t2, t3, the integrals and the orbital energies to 5e-11 of
derivatives of 10 to 1000, real and complex.

The pass on packed amplitudes against the dense reference (5+6, 6+5, 4+7
and 7+5 spin orbitals, real and complex, in every plan, with and without
the C kernels): E(Q), s2, s3, the six blocks of densities, `de` and Gamma
to 3e-15 of their size.

The pass in sectors against the pass on the packed block
(`tests/test_quadruples_grad_sym.py`; HBr / cc-pVDZ-DK with X2C spinors, 28
frozen, 8 + 28, labels 2 m_j; integrals and amplitudes projected on their
label-conserving parts): E(Q), s2, s3, all densities, `de`, Gamma and the
parts of the gradient with s3 and t3 to 2e-15 of their size (1e-12 asked) -
with the labels as they are, modulo 4, with one label, and with a phase on
every spinor (complex amplitudes); 3e-15 for HBr / cc-pVTZ-DK (8 + 78, on
the login node). On the amplitudes as the iterations leave them the two
passes differ by 1e-10 of the size of each array (2e-9 asked): the
symmetry-breaking elements, which the packed block keeps. The orbital
gradient among the active spinors and the terms of frozen spinors from the
density of `<bc||ef>` in blocks are those from the array (1e-12), and the
front end gives the same Gradient from the sectors, from the packed block
(`symmetry` off) and from the packed block with s3 and the density of
`<bc||ef>` in a scratch file (a budget between the two floors).

The orbital gradient of E(Q) and the terms of frozen spinors (matrix
products on the packed densities, a batch of frozen spinors) against the
dense densities contracted index by index with dense integrals
(`test_gradient_terms_equal_einsums`, 5 + 8 complex and 4 + 9 real spin
orbitals, random): 1e-12 asked.

Relaxed dipole derivative, trace(z D) of the dense reference against
E(SCF) + E_corr(CCSDT) + E(Q) with the SCF repeated in the field
(`bench/qgrad_ccsdt_q.py`; energies of the production codes; four-point
differences at steps 4e-3, 2e-3, 1e-3 and the extrapolation of two
consecutive steps, (16 f(h) - f(2h)) / 15; SCF converged to 1e-13 and
1e-10 in the gradient, CCSDT to 1e-13 and 1e-11, Lambda and the Z-vector
equations to 1e-11). Differences analytic - finite field:

| | o + v | trace(z D), CCSDT(Q) | its (Q) part | step 4e-3 | 2e-3 | 1e-3 | 4e-3, 2e-3 extrapolated | 2e-3, 1e-3 extrapolated |
|---|---|---|---|---|---|---|---|---|
| H2O / STO-3G | 10 + 4 | 1.582688302954 | 8.39e-5 | 6.0e-10 | 1.7e-11 | -1.5e-11 | -2.1e-11 | -1.7e-11 |
| H2O / STO-3G, complex spin orbitals | 10 + 4 | 1.582688302954 | 8.39e-5 | 5.9e-10 | 2.3e-11 | 4.2e-11 | -1.4e-11 | 4.3e-11 |
| H2O / STO-3G, X2C spinors | 10 + 4 | 1.583200002586 | 8.44e-5 | 5.8e-10 | 4.2e-11 | -1.9e-11 | 6.3e-12 | -2.4e-11 |
| H2O / STO-3G, X2C, 2 frozen | 8 + 4 | 1.583147291350 | 8.45e-5 | 5.8e-10 | 2.0e-11 | -2.6e-11 | -1.7e-11 | -2.9e-11 |
| BH / STO-3G | 6 + 6 | 2.087338743535 | 3.26e-4 | -1.9e-8 | -1.2e-9 | -7.0e-11 | -1.2e-12 | 5.2e-12 |
| BH / STO-3G, complex | 6 + 6 | 2.087338743531 | 3.26e-4 | -1.9e-8 | -1.2e-9 | -8.3e-11 | -3.6e-12 | -9.2e-12 |
| BH / STO-3G, complex, 2 frozen, 2 dropped | 4 + 4 | 2.094223049398 | 3.08e-5 | -1.7e-8 | -1.1e-9 | -7.7e-11 | 7.0e-12 | -1.1e-11 |
| LiH / STO-3G | 4 + 8 | 4.833109829421 | -4.83e-6 | -6.4e-5 | -3.9e-6 | -2.4e-7 | 1.3e-7 | 1.9e-9 |
| H2O / 6-31G | 10 + 16 | 1.227852046041 | 1.43e-4 | 1.2e-8 | 7.6e-10 | 1.1e-11 | 2.8e-11 | -3.9e-11 |
| H2O / 6-31G, complex | 10 + 16 | 1.227852046041 | 1.43e-4 | 1.2e-8 | 7.3e-10 | 2.9e-11 | -3.8e-12 | -1.7e-11 |
| HF / 6-31G, X2C | 10 + 12 | 16.451076063601 | -3.12e-4 | -5.8e-9 | -3.5e-10 | -1.5e-11 | 1.1e-11 | 7.7e-12 |
| H2O / 6-31G, X2C, 2 frozen, 4 dropped | 8 + 12 | 1.265709070026 | 5.88e-4 | 1.6e-8 | 1.0e-9 | 1.1e-10 | -6.8e-12 | 4.9e-11 |

(STO-3G rows but LiH: on the login node; LiH and 6-31G: job 31764120, c105;
the dense derivative of E(Q) takes 15 minutes for 10 + 16 spin orbitals.)
CCSDT alone has the same differences in every row, to 5e-12 where they are
small: what is left is the error of the differences (energies to 1e-13 over
steps of 1e-3), not of the (Q) terms. LiH: the stencil has a truncation
error of 2.4e-7 at step 1e-3; its test extrapolates steps 1e-3 and 5e-4
(3e-11). With STO-3G the gradient among the virtual spin orbitals of H2O
and BH vanishes by symmetry; it is 6e-9 for LiH and 1e-5 to 2e-5 in the
6-31G rows, 4e-6 to 1e-5 among the occupied ones.

CFOUR (`CALC=CCSDT(Q)`, `CC_PROG=NCC`, `PROPS=FIRST_ORDER`; RHF, all
electrons, `SYMMETRY=OFF`, the bohr geometry and the basis of
`bench/refs`; inputs `bench/refs/cfour/*/ZMAT.qprops.bohr`, run on the
login node, 2 s each): the dipole moment of the relaxed density,
sum Z_A z_A - trace(z D), ten decimals printed:

| | CFOUR ncc | zccsdt (dense reference) |
|---|---|---|
| H2O / STO-3G, CCSDT | 0.6359340712 | 0.635934071198 |
| H2O / STO-3G, CCSDT(Q) | 0.6358501673 | 0.635850167285 |
| H2O / 6-31G, CCSDT | 0.9908298365 | 0.990829836453 |
| H2O / 6-31G, CCSDT(Q) | 0.9906864242 | 0.990686424198 |

CFOUR gave it with the first input that was tried; MRCC was not run.

Production against the dense reference (converged CCSDT, the pass,
`lambda_opt.kernel(sources=)`, `relaxed_rdm1(q=)`): lambda, the density
without the response, the orbital gradient between all pairs of spinors
and the relaxed density, largest differences.

| | lambda | density | gradient (its size) | relaxed density |
|---|---|---|---|---|
| BH / STO-3G | 3e-13 | 9e-14 | 1e-13 (2e-5) | 2e-13 |
| BH / STO-3G, complex | 2e-13 | 7e-14 | 8e-14 (1e-5) | 2e-13 |
| LiH / STO-3G, complex | 8e-14 | 4e-14 | 2e-14 (4e-7) | 3e-14 |
| H2O / STO-3G, X2C | 2e-13 | 3e-14 | 3e-14 (7e-6) | 4e-14 |
| H2O / STO-3G, X2C, 2 frozen | 3e-13 | 1e-13 | 6e-14 (9e-4) | 7e-14 |
| BH / STO-3G, complex, 2 frozen, 2 dropped | 4e-16 | 1e-16 | 1e-13 (8e-3) | 2e-13 |
| BH / STO-3G, X2C, 2 frozen, 2 dropped | 2e-12 | 6e-13 | 2e-12 (7e-3) | 3e-12 |
| H2O / 6-31G, X2C, 2 frozen, 4 dropped | 3e-13 | 7e-14 | 5e-12 (1e-2) | 3e-12 |

(BH, LiH and the rows with frozen spinors but the last are in
`tests/test_quadruples_grad.py`, with 1e-10 asked; all rows on the login
node, the last: 0.8 s for the production codes, 100 s for the dense
reference.) With the frozen spinors of the X2C rows the integrals with a
frozen index are made for the Kramers pairs.

At the size of the timings below: H2O / cc-pVTZ, RHF orbitals as spin
orbitals (10 + 106), through the front end, against CFOUR's ncc with the
primitives and the bohr geometry of PySCF (`bench/cfour/genbas.py`;
`SYMMETRY=ON`; job 31765428, input and output kept with the logs of the
job). E(SCF) -76.057114083120 and -76.057114083117, E(Q) -3.676439958e-4
and -3.67643995767e-4, and the dipole moment of the relaxed density:

| | CFOUR ncc | zccsdt (`GCCSDT.make_rdm1`) |
|---|---|---|
| H2O / cc-pVTZ, CCSDT(Q) | 0.7492563653 | 0.7492563653 |

(CCSDT alone: 0.7498653312 here; CFOUR's run printed the SCF and the
CCSDT(Q) density only.)

Front end: HF / 6-31G with X2C spinors, two frozen (real spinors adapted
to m_j): `make_rdm1(with_q=True)` against the finite field of
`e_tot + ccsdt_q()`, 1.4e-10 off (0.874002176879); Lambda and the
densities in the blocks of the labels give the density of the packed block
(5e-9 asked, the convergence of lambda).

Memory (tracemalloc; the mapped arrays are not allocations):

| | cases | peak of the allocations / plan |
|---|---|---|
| the pass on the packed block (`test_memory_plan`) | 5 + 20 complex, 6 + 26 real; the smallest batches with s3 and the density of `<bc||ef>` mapped, a plan in between, the largest batches with the density mapped, everything in memory | 0.86 to 0.94 of `Plan.need` (0.8 asked); the plans of 40 budgets from the floor up are monotone and take 0.90 to 1.00 of the budget while the batches grow |
| the pass in sectors (`test_quadruples_grad_sym.py::test_memory_plan`) | HBr / cc-pVDZ-DK, 8 + 28 | 0.56 (the plan of (Q) in sectors counts Xv by its dense size) |
| the orbital gradient of E(Q) (`test_gradient_terms_within_their_plan`) | 5 + 22, real and complex | 0.60 of `active_bytes` |
| the terms of 1 and of 4 frozen spinors | 5 + 22, real and complex | 0.83 to 0.90 of `FrozenTerms.own` |

### 11.6 Timing of the first version

(The pass on the packed block only, as of commits bd2c483 and 2774438; the
state after the sectors, the plans and the changes to the pass: 11.7.)

`bench/profile_q_gradient.py` through the front end with its defaults, 8
cores of a Xeon Gold 6248R, `max_memory` 26 GB, the steps one after the
other on the same amplitudes (`bench/slurm_q_gradient.sh`). Two jobs: 31765428
on c105 (commit bd2c483; the node shared, load 2 to 10 while it ran - two
other jobs of this work among it) and 31766046 on c320 (the tree of this
commit: the arithmetic on the vectors of a quadruple on threads; load 29
at the start, 7 to 12 at the end). H2O / cc-pVTZ: RHF orbitals as spin
orbitals, 10 + 106, no labels - everything on the packed block. HBr /
cc-pVTZ-DK: X2C spinors, 28 frozen, 8 + 78, real spinors with the labels
2 m_j - (Q) in sectors, Lambda and the densities in blocks, the pass over
the quadruples on the packed block. CCSDT converged to 1e-11 in the energy
and 1e-9 in the amplitudes, Lambda to 1e-9.

| | H2O, 31765428 | H2O, 31766046 | HBr, 31765428 | HBr, 31766046 |
|---|---|---|---|---|
| CCSDT iteration | 4.41 s | 5.06 s | 0.30 s | 0.36 s |
| (Q), front end | 46.2 s | 50.4 s | 1.49 s (sectors) | 1.83 s (sectors) |
| (Q), dense code | 45.4 s | 50.1 s | 4.41 s | 5.23 s |
| (Q) gradient pass | 162.3 s | 170.5 s | 15.3 s | 18.0 s |
| ... over (Q) dense | 3.58 | 3.41 | 3.47 | 3.43 |
| ... over (Q) of the front end | 3.51 | 3.38 | 10.3 | 9.8 |
| Lambda iteration, CCSDT | 5.23 s (17) | 5.99 s (17) | 0.24 s (17) | 0.28 s (17) |
| Lambda iteration, with the sources | 5.25 s (17) | 6.08 s (17) | 0.24 s (16) | 0.28 s (16) |
| relaxed density after Lambda, CCSDT | 10.8 s | 6.3 s | 1.74 s | 1.95 s |
| relaxed density after Lambda, CCSDT(Q) | 5.7 s | 6.8 s | 3.40 s | 4.23 s |
| all after the CCSDT energy, CCSDT | 101.7 s | 110.5 s | 6.0 s | 7.0 s |
| all after the CCSDT energy, CCSDT(Q) | 259.2 s | 283.0 s | 22.8 s | 27.0 s |
| ... over (Q) dense | 5.7 | 5.7 | 5.2 | 5.2 |
| ... over (Q) of the front end | 5.6 | 5.6 | 15.3 | 14.8 |

(In 31765428 the CCSDT density came first and has the setup of the Coulomb
operator of the Z-vector equations in it; 6.3 against 6.8 s is the pair
to compare.) The sources cost nothing in a Lambda iteration and do not
change the number of iterations. The pass is 3.4 to 3.6 times the dense
(Q); where (Q) runs in sectors (HBr: three times faster than dense) the
pass, which does not, is ten times (Q) and three quarters of the whole
property. With frozen spinors the (Q) terms double the time of the
density (HBr: the contractions with the integrals of 28 frozen spinors,
einsums one spinor at a time).

Where the pass of H2O goes (31766046): the adjoint products 71.8 s at 6.5
cores (three times the 24.2 s of the products of (Q), for twice their
multiplications: the two of the T2 Xv product write into blocks of columns
of the adjoint of Xv, and one result array per quadruple is allocated),
the expansions 19.7 s, the products and reductions of (Q) itself 24.2 and
9 s, the adjoint of Xv 12 s, the vectors of a quadruple 7.1 s (20.9 s on
one core in 31765428), Gamma 2.9 s.

Memory: the plan of the pass and the peak of the anonymous memory of the
process, with what was there before (amplitudes, integrals, DIIS released):

| | plan of the pass | of it the derivatives | before | peak |
|---|---|---|---|---|
| H2O / cc-pVTZ | 6.86 GB | 0.50 GB | 0.95 GB (1.05) | 7.67 GB (7.84) |
| HBr / cc-pVTZ-DK | 1.79 GB | 0.13 GB | 0.55 GB | 2.20 GB (2.31) |

(in brackets job 31766046; (Q) itself plans 3.29 and 0.87 GB.) Everything
fits here: one batch of columns, Xv for all occupied indices. The density
step with the (Q) terms peaks at 6.6 and 1.6 GB.

CFOUR's ncc for the same H2O calculation (job 31765428, same node, after
the runs above; RHF, spin-adapted, `SYMMETRY=ON`, 8 threads): 66 s for all
of it - CCSDT 9.0 s (22 iterations), (Q) as it does it for a density
39.1 s, Lambda 15.5 s (21 iterations), the densities 0.7 s.
zccsdt has no spin adaptation and no point group on this path (GCCSDT on
116 spin orbitals: 84 s for CCSDT, 46 s for (Q), 259 s for the property):
the numbers agree, the times are not a comparison of like with like.

### 11.7 Sectors, memory and speed of the pass: before and after

One job, 31769367 on c105 (8 cores of a Xeon Gold 6248R, `--mem=30G`,
`bench/slurm_q_gradient_sectors.sh`): the tree of commit 8059a77 (before)
and that of 129de08 (after), one after the other, each with its own
`bench/profile_q_gradient.py`, `max_memory` 26 GB; then, after only, HBr /
cc-pVQZ-DK and H2O again with `max_memory` 4 GB. The node is shared: load
1.4 at the start, 6 to 9 from the second run on; a test run of this work
(4 threads) overlapped the first minute and a half of the job - the SCF and
the first CCSDT iterations of the first run. Molecules and convergence as
in 11.6.

| | H2O / cc-pVTZ, 10 + 106, before | after | HBr / cc-pVTZ-DK, 8 + 78, before | after | HBr / cc-pVQZ-DK, 8 + 160, after |
|---|---|---|---|---|---|
| CCSDT iteration | 4.28 s | 4.28 s | 0.30 s | 0.30 s | 1.98 s |
| (Q), front end | 44.7 s | 44.7 s | 1.50 s (sectors) | 1.49 s (sectors) | 14.5 s (sectors) |
| (Q), dense code | 43.8 s | 44.1 s | 4.46 s | 4.39 s | 94.7 s |
| (Q) gradient pass, front end | 141.8 s | 129.9 s | 14.43 s (packed block) | 3.19 s (sectors) | 27.4 s (sectors) |
| ... over (Q) of the front end | 3.17 | 2.90 | 9.59 | 2.14 | 1.89 |
| (Q) gradient pass, packed block | 141.8 s | 129.9 s | 14.43 s | 13.65 s | 283.1 s (density of `<bc||ef>` mapped, 10551 of 12720 columns per batch) |
| ... over (Q) dense | 3.23 | 2.94 | 3.23 | 3.11 | 2.99 |
| Lambda iteration, CCSDT / with the sources | 5.15 / 5.17 s | 5.15 / 5.16 s | 0.24 / 0.24 s | 0.24 / 0.24 s | 1.23 / 1.23 s |
| relaxed density after Lambda, CCSDT | 4.56 s | 5.62 s | 1.79 s | 1.63 s | 9.54 s |
| relaxed density after Lambda, CCSDT(Q) | 5.67 s | 5.25 s | 3.43 s | 1.99 s | 14.73 s |
| all after the CCSDT energy, CCSDT | 93.9 s | 95.2 s | 6.15 s | 5.96 s | 32.6 s |
| all after the CCSDT energy, CCSDT(Q) | 237.2 s | 224.8 s | 22.0 s | 9.31 s | 65.3 s |
| ... over (Q) of the front end | 5.30 | 5.02 | 14.6 | 6.25 | 4.49 |
| trace(z D), CCSDT(Q) | 1.469282104952 | 1.469282104952 | 2.335750517122 | 2.335750517122 | 2.347398874663 |

The numbers of the two trees are the same to the twelve decimals printed
(E(Q), trace(z D) of CCSDT and of CCSDT(Q)); HBr after has the pass in
sectors, before on the packed block. The densities of H2O (4.6 to 5.7 s)
scatter by a second between runs of the same code (the CCSDT one is not
changed).

Where the time goes now:

- the pass on the packed block, H2O: the adjoint products 46.2 s (58.3 s
  before) for 21.9 s of the products of (Q) - 2.11 times, for 2.09 times
  the multiplications; the expansions 18.2 s (five per quadruple, for
  8.7 s of the five reductions of the energy: the expansion sets its array
  to zero and writes through the index tables); the adjoint of Xv 11.4 s
  at 5.3 cores for 4.0 s of Xv (two products, an expansion and a
  reduction per slab where Xv has one product and a reduction); the
  vectors of a quadruple 4.3 s; Gamma 3.0 s. These three are what is
  left above three times (Q);
- the pass in sectors, HBr / cc-pVQZ-DK: the kernel that sums the splits
  and writes the left-hand quadruples 6.7 s (8 cores), the adjoints on
  pairs 5.4 s at 3.4 cores and the products on pairs 3.4 s at 3.7 cores
  (many small blocks: numpy picks rows and columns on one core), the
  adjoints on an index and a triple 3.2 s, the adjoint of Xv 3.0 s for
  1.9 s of Xv. For 8 + 78: 3.19 s at 3.4 cores - Python between small
  products;
- the density step with frozen spinors: the (Q) terms of the 28 frozen
  spinors 0.24 s (HBr / cc-pVTZ-DK; the whole (Q) part of the step was
  1.6 s before, 0.4 s now) and 3.0 s at 2.5 cores for cc-pVQZ-DK, the
  orbital gradient among the active spinors 2.3 s at 2 cores there:
  numpy unpacks the pairs of the density of `<ab||ei>` and gathers the
  slices of the integrals on one core.

Memory, anonymous memory of the process (sampled) against the plans:

| | plan of the pass | before the pass | peak in the pass | `max_memory` |
|---|---|---|---|---|
| H2O, packed block | 6.86 GB, nothing mapped | 1.27 GB | 7.92 GB | 26 GB |
| HBr / cc-pVTZ-DK, sectors | 1.04 GB (the density of `<bc||ef>`: 0.010 GB in blocks, 0.072 as an array) | 0.94 GB | 0.85 GB | 26 GB |
| HBr / cc-pVQZ-DK, sectors | 12.01 GB (0.153 GB of 1.294) | 3.19 GB | 6.77 GB | 26 GB |
| HBr / cc-pVQZ-DK, packed block | 22.13 GB, the density of `<bc||ef>` (1.29 GB) mapped | 3.00 GB | 24.72 GB | 26 GB |
| H2O, packed block, tight | 2.79 GB, s3 and the density mapped; 16 batches of columns, Xv for 6 of 9 occupied indices | 1.26 GB | 3.98 GB | 4 GB |

The plan of the sectors is far above what they take (it counts Xv of an
occupied index by its dense size, see "Not done"); "before the pass"
includes arrays of the step before that are released as the pass starts
(HBr / cc-pVTZ-DK ends below it). The run under 4 GB: every step stays
below `max_memory` in anonymous memory - (Q) 3.94 GB, Lambda 3.29 and
3.70 GB, the densities 2.91 and 2.81 GB, the pass 3.98 GB -, the pass
takes 234 s instead of 130 s (16 batches: the expansions 50 s, the
reductions 34 s, the adjoint of Xv 33 s), and the numbers are those of the
run with 26 GB (E(Q) -3.676439958136e-4 against ...137e-4, trace(z D)
1.469282104952). The maximum resident size of that process was 7.08 GB:
with the pages of the mapped scratch files (DIIS vectors, s3, the
density), which are not anonymous memory and were not sampled apart here.
The density step with the (Q) terms peaks at 6.69 GB (H2O), 1.63 GB (HBr
/ cc-pVTZ-DK) and 10.44 GB (cc-pVQZ-DK) with 26 GB.

## 12. First derivatives of CCSDT(Q)_Lambda

The theory (12.1 to 12.3), the dense reference (12.4), and the production
code on packed amplitudes behind `GCCSDT.make_rdm1(with_q='lambda')` (12.5
to 12.7).

| File | What |
|---|---|
| `src/zccsdt/cc/qlgrad_ref.py` | the dense reference: the derivatives of E(Q)_L, the two linear equations, the densities, the relaxed density from full arrays |
| `tests/test_qlgrad_ref.py` | the checks of the reference in 12.8 |
| `bench/qlgrad_ccsdt_ql.py` | finite fields at several steps (a dipole field, or a point charge as MRCC can have it) |

(The files of the production code: 12.6.)

### 12.1 The functional

With canonical Hartree-Fock spin orbitals and bare integrals (section 5),

    E(Q)_L = Re e,   e = sum z r / D               (i<j<k<l, a<b<c<d),
    r = A(g, t3) + B(g, t2, t2),                   A = <Q| [V,T3] |0>,  B = <Q| 1/2 [[V,T2],T2] |0>,
    z = Z(g^T, l2) + A(g^T, l3) = <0| (Lambda_2 + Lambda_3) V |Q>,     g^T[p,q,r,s] = g[r,s,p,q],

Z the disconnected product `<Q| V T2 |0>` of the term table. **Nothing is
conjugated**: `<0| Lambda V |Q>` has lambda as it is and the integrals with
bra and ket exchanged, so e is a holomorphic function of t2, t3, l2, l3 and
g, like the Lagrangian L of CCSDT (section 1), and E(Q)_L is the real part
of it as E(CCSDT) is the real part of L. (`lambda_ref.quadruples_lambda`
and the production code write z as `conj(Z(g, conj(lambda)))`, conjugating
twice: the same number for Hermitian integrals. `qlgrad_ref.energy` is e
for any g, which the derivative with respect to g needs. For a Hamiltonian
with a symmetry that makes the energy real - every molecule here - Im e
vanishes, 1e-16 and less in the X2C runs of 12.8; the model of the tests
has 1.2e-4 of it next to Re e = -2.6e-4.)

This is what makes (Q)_L simpler than (Q) of section 11 in one respect and
harder in another:

- no Wirtinger derivatives. (Q) has `conj(z)` and is a real, non-holomorphic
  function, t and conj(t) independent; here every derivative is an ordinary
  complex one and the real part is taken once, at the end, by making the
  densities Hermitian;
- e depends on lambda, and lambda is not a variational parameter of
  anything: it is the solution of the Lambda equations of CCSDT. So the
  Lagrangian needs those equations as a second constraint, with a second
  set of multipliers.

t1 and l1 do not enter e, and neither does the off-diagonal Fock matrix.

### 12.2 The Lagrangian and its two linear equations

Constraints: the amplitude equations `R_k(t) = 0` and the Lambda equations
`W_k(t, lambda) = dL/dt_k = 0`, `L(t, lambda) = E(t) + sum lambda_mu R_mu(t)`
(sums over the unique elements, as everywhere below). Multipliers: zeta
for W - a vector like t, `[z1[i,a], z2[i,j,a,b], z3[i,j,k,a,b,c]]` - and
lambda-bar for R:

    F = L(t, lambda) + e(t, lambda; g, e_p) + sum zeta_mu W_mu(t, lambda) + sum lambda-bar_mu R_mu(t)
      = L(t, lambda') + e(t, lambda) + sum zeta_mu W_mu(t, lambda),        lambda' = lambda + lambda-bar,

since L is linear in lambda. F is holomorphic in everything; the energy
is Re F.

**Stationary in lambda** (at fixed lambda'). W is linear in lambda with
`dW_mu/dl_nu = dR_nu/dt_mu`, so

    sum_mu (dR_nu/dt_mu) zeta_mu + q_nu = 0,      q_nu = de/dl_nu                (1)

**the Jacobian of the CCSDT amplitude equations on zeta**: a right-hand
linear response equation, the transpose of the matrix of the Lambda
equations. Its source is

    q_nu = sum_Q (dz_Q/dl_nu) y_Q = <nu| V Y4 |0>,     Y4 = r / D,

what T4 = Y4 adds to the doubles and triples equations of CCSDTQ (the
groups `R2: [V,T4]` and `R3: [V,T4]` of `ccsdtq_terms`): **zeta is the first
order response of the CCSDT amplitudes to the perturbative quadruples.**
q1 = 0; zeta_1 is not zero (the Jacobian couples the ranks). A Jacobi step
is `zeta += (J zeta + q) / D`, as for the amplitudes.

**Stationary in t.**

    W_k(t, lambda') + s_k = 0,
    s_k = de/dt_k + sum_mu zeta_mu (dW_mu/dt_k)(t, lambda)                      (2)

**the Lambda equations of CCSDT for lambda' with constant sources**
(`lambda_ref.kernel(sources=)`, as in section 11). The first part of s is
the analogue of the sources of (Q), with w = z / D as the left-hand
quadruples: `<0| L(w) [V, tau3] |0>` for rank 3, `<0| L(w) [[V,T2], tau2] |0>`
for rank 2 - no term of rank 1, and not the disconnected one of (Q): that
one is q2 here. The second part is the second derivative of the CCSDT
Lagrangian with respect to t, contracted with zeta - the term bilinear in
zeta and lambda (and one without lambda, from the energy); it is the
directional derivative of the Lambda residuals along zeta in t at fixed
lambda, and it has a rank-1 part: **s1 is not zero** (4.8e-7 for H2O /
STO-3G, 1.9e-6 for BH).

Order of the calculation:

1. CCSDT: t. The Lambda equations of CCSDT: lambda. E(Q)_L and, in the
   same pass over the quadruples, its derivatives q2, q3 (with respect to
   l2, l3), the part of s2, s3 from e, the densities d2 of e, and `de`.
2. Equation (1) for zeta.
3. The directional derivative of W along zeta: s.
4. Equation (2) for lambda'.
5. Densities, orbital gradient, Z-vector equations (12.3).

CFOUR's ncc has the same structure for CCSD(T)_Lambda (two more iterative
solvers after Lambda, `upsilon_it` and `pi_it` in its task list), and
refuses CCSDT(Q)_Lambda (12.8).

### 12.3 Densities and the response of the spin orbitals

With zeta and lambda' the derivative of F is that of its explicit
dependence on the integrals:

- `L(t, lambda')`: the densities d1, d2 of the CCSDT Lagrangian with
  lambda' in the place of lambda (section 1);
- `sum zeta_mu W_mu(t, lambda)`: W is the derivative of L with respect to t,
  so its coefficients of the integrals are the **directional derivative of
  the densities d1(t, lambda), d2(t, lambda) along zeta in t** (the
  reference drops out);
- e: its own d2 (`de = sum d2 d<pq||rs>`: y times l2 and l3 on the
  integrals of z, `<kl||cd>`, `<ie||ab>`, `<ij||ma>`; w times t3 and t2 t2 on
  those of r) and `de[p]` on the orbital energies, -/+ the sum of Re(w y)
  over the quadruples with p.

The first two are added and made Hermitian (`lambda_ref.pyscf_rdms`: the
real part); from there on everything is as for CCSDT(Q) (section 11.2):
`de` on the diagonal of the one-particle density, the spin orbitals follow
as canonical Hartree-Fock spin orbitals - occupied-virtual rotations
through the Z-vector equations with G[a,i] = (Gnh[a,i] - conj(Gnh[i,a])) / 2
of all the densities, occupied-occupied and virtual-virtual ones through
y_pq = G_pq / (e_q - e_p) (`zccsd.zvector._relax`), none between degenerate
spin orbitals.

Frozen occupied and dropped virtual spinors: t, lambda, zeta and lambda'
are those of the active spinors, E(Q)_L too. The frozen ones are occupied
in the densities (`qgrad_ref.full_densities`; the sum of the two Lagrangian
densities goes in, the second has no reference of its own), and the
gradient between a frozen and an active spinor of the same kind gives a
rotation by the canonical condition, as before.

### 12.4 The dense reference (`zccsdt.cc.qlgrad_ref`)

`quadruples_gradient(t2, t3, l2, l3, g, mo_energy, nocc)`: one occupied
quadruple at a time as `qgrad_ref`, r from the terms of `R4` on (g, t2, t3)
and z from those of `Z4` on (g^T, l2, l3); the derivative of every term
with respect to every factor with w (terms of r) or y (terms of z) as the
left-hand quadruples: s2, s3 (from e), q2, q3, d2, de.

**The directional derivatives** (`directional`): exp(-T) H exp(T) ends at
the fourth power of T, so the amplitude residuals, the Lambda residuals
and the densities are polynomials of degree <= 4 in the amplitudes (t1
included), and

    [8 (f(t + zeta) - f(t - zeta)) - (f(t + 2 zeta) - f(t - 2 zeta))] / 12

is their derivative along zeta exactly, not a numerical approximation:
`jacobian` (on `ccsdtq_ref.residuals`), `lambda_response` (on
`lambda_ref.residuals`), `density_response` (on `lambda_ref.densities`).
No new table of terms was derived. `jacobian_dressed` is the same Jacobian
from three evaluations in the T1-dressed formalism, where the residuals
P(H~; t2, t3) are linear in the dressed integrals and quadratic in (t2, t3):

    J zeta = [P(H~; t2 + z2, t3 + z3) - P(H~; t2 - z2, t3 - z3)] / 2 + P(dH~; t2, t3),

dH~ = [H~, z1] the dressed integrals one-index transformed with zeta_1.

`solve_zeta` (Jacobi steps with DIIS), `kernel` (steps 1 to 4 and the
densities of the Lagrangian at fixed orbitals), `lagrangian` (F for any
arguments), `relaxed_rdm1(mf, mo_coeff, nfrozen, ndrop)` (everything, from
full arrays over all spin orbitals; `zvector.orbital_gradient` on the
densities, `zvector.gradient_nh` for those of e, `zvector._relax`). None of
the identities of 12.5 is used.

### 12.5 Identities for an implementation on packed amplitudes

The three of section 11.2 carry over; each is checked against the dense
densities (`tests/test_qlgrad_ref.py::test_gradient_among_occupied_and_among_virtual`,
LiH and BH with complex spin orbitals).

1. *The Lagrangian among occupied and among virtual spin orbitals.* L and
   zeta . W do not change when t, lambda, zeta, lambda' and the integrals
   are rotated together, and at the solution dF/dt and dF/dlambda leave
   only the derivatives of e. So the part of the gradient that comes from
   the densities of lambda' and of zeta is made of the sources of the two
   equations - s **from e only**, without the zeta term (the two parts
   with the directional derivative of W cancel, s1 with them) - and q:

       Gnh[j,i] = -1/2 sum s2[j,k,a,b] t2[i,k,a,b] - 1/12 sum s3[j,k,l,a,b,c] t3[i,k,l,a,b,c]
                  +1/2 sum q2[i,k,a,b] l2[j,k,a,b] + 1/12 sum q3[i,k,l,a,b,c] l3[j,k,l,a,b,c]
       Gnh[b,a] =  1/2 sum s2[i,j,a,c] t2[i,j,b,c] + 1/12 sum s3[i,j,k,a,c,d] t3[i,j,k,b,c,d]
                  -1/2 sum q2[i,j,b,c] l2[i,j,a,c] - 1/12 sum q3[i,j,k,b,c,d] l3[i,j,k,a,c,d]

   with lambda of CCSDT (lambda rotates the other way round). (LiH and BH:
   1e-14 and 8e-14 off, blocks of 1.3e-7 and 7e-6; with the other sign of
   the q terms 5e-8 and 5e-6 off.)

2. *The table `GRADIENT`* for the occupied-virtual block: applied to
   (t, lambda'), plus its directional derivative along zeta at (t, lambda),
   plus the two blocks of identity 1 with t1,

       Gnh[a,i] = GRADIENT(t, lambda')[a,i] + d_zeta GRADIENT(t, lambda)[a,i]
                  + sum_j t1[j,a] Gnh[j,i] - sum_b t1[i,b] Gnh[a,b].

   The terms with the excitation block (t1 t1 times the Lambda_1 residual,
   which does not vanish any more: s1) cancel between the two pieces.
   (LiH, BH, H2O / STO-3G: 4e-16 to 3e-15 off the one-index transformation
   of the full densities; the tables alone 2e-9, 9e-7, 1e-7 off.)

3. *The whole gradient among occupied and among virtual spin orbitals is a
   one-particle density of (Q)_L*: with y the solution of D y = r for the
   whole Fock matrix, e does not change under these rotations (with t and
   lambda), so

       G[p,q] = (e_q - e_p) (Gamma[q,p] + conj(Gamma[p,q])) / 2,
       Gamma[a,e] =  sum w[ijkl,abcd] y[ijkl,ebcd]     (i<j<k<l, b<c<d),
       Gamma[m,i] = -sum w[ijkl,abcd] y[mjkl,abcd]     (j<k<l, a<b<c<d)

   (`qlgrad_ref.fock_density`; Re of its diagonal is `de`): Gamma of 11.2
   without the conjugation of w. (LiH, BH: 1e-14 and 8e-14 off.)

### 12.6 On packed amplitudes: what was built

| File | What |
|---|---|
| `src/zccsdt/cc/quadruples_lambda_grad.py` | the pass over the occupied quadruples on the packed block: q, s, the densities of e, `de`, Gamma; `source_terms`, `orbital_gradient` |
| `src/zccsdt/cc/quadruples_lambda_grad_sym.py`, `csrc/qlambda.c` (`zc_qlg_sector_vectors`) | the same pass in the sectors of the symmetry labels |
| `src/zccsdt/cc/zeta_opt.py` | equation (1) for zeta (`kernel`), the sources of equation (2) (`lambda_sources`), their memory plans |
| `src/zccsdt/cc/lambda_opt.py` | `ResidualSources`: sources in the layouts of the residuals, for `kernel(sources=)` |
| `src/zccsdt/cc/density_opt.py` | `relaxed_rdm1(zeta=, lams=, stencil=)`: the densities of zeta, the terms of e; `_Replay` |
| `src/zccsdt/ccsdt.py` | `GCCSDT.solve_lambda(with_q='lambda')`, `make_rdm1(with_q='lambda')`, `_ql_terms` |
| `tests/test_quadruples_lambda_grad.py`, `tests/test_quadruples_lambda_grad_sym.py` | the checks of 12.8 |
| `bench/profile_ql_gradient.py`, `bench/slurm_ql_gradient.sh` | the timing of 12.7 |

**The pass over the quadruples** (`quadruples_lambda_grad.kernel`). After r
(four products, `quadruples.right`) and z (three, `quadruples_lambda.left_lambda`)
of an occupied quadruple: y = r / D and w = z / D, and every product
backwards with its two adjoints as in 11.4 - with w for the four of r, with y
for the three of z, nothing conjugated:

| product | left-hand vector | its two adjoints go to |
|---|---|---|
| `- sum_e <ab||ei> t(jkl,cde)` | w | the density of `<ab||ei>`; s3 |
| `sum_m <ma||ij> t(mkl,bcd)` | w | the density of `<ma||ij>`; s3 |
| `sum_m t(im,ab) Xo(m;jkl;cd)` | w | s2; the adjoint of Xo (at the end: s2, the densities of `<mn||jk>`, `<md||el>`) |
| `2 sum_e t(ij,ae) Xv(e;kl;bcd)` | w | s2; the adjoint of Xv (when Xv leaves: s2, the density of `<bc||ef>`) |
| `l(ij,ab) <kl||cd>` | y | q2; the density of `<kl||cd>` |
| `sum_e <ie||ab> l(jkl,cde)` | y | the density of `<ie||ab>`; q3 |
| `sum_m <ij||ma> l(mkl,bcd)` | y | the density of `<ij||ma>`; q3 |

Four expansions of a quadruple (w and y in the two layouts) where (Q) has
five, seven products with two adjoints each where (Q) has five, and
Gamma[a,e] = sum w[a,(bcd)] y[e,(bcd)] on the expanded vectors. E(Q)_L is
the real part of e and the integrals of a rotated basis are Hermitian, so
the coefficients of `<kl||cd>`, `<ie||ab>` and `<ij||ma>` are added,
conjugated, to those of `<cd||kl>`, `-<ab||ei>` and `<ma||ij>`: the result
is a `quadruples_grad.Gradient` (dE = Re sum density x d integral on the
six blocks of 11.4) with q2, q3 and Im e besides, and
`quadruples_grad.active_terms` and `FrozenTerms` - the orbital gradient of
the correction among the active spinors and with a frozen index - are used
as they are. What is new in the gradient is identity 1 with its q terms
(`source_terms`; the parts with the packed arrays, s3 with t3 and q3 with
l3, are made at the end of the pass) and Gamma without the conjugation.
In sectors (`quadruples_lambda_grad_sym`) the kernel
`zc_qlg_sector_vectors` writes y and w to the blocks of the seven products
(two blocks on two pairs, two on an index and a triple), and the adjoints
are taken block by block as in 11.4.

Memory (`plan`): that of `quadruples_grad` with q3 (one more array of the
size of T3), q2, the arrays of the left-hand side of 5.1 and the adjoints
of the operands of z; where the largest batches do not fit, s3, q3 and the
density of `<bc||ef>` are mapped arrays of a scratch file
(`MAPPED`), MemoryError with the numbers below the floor. In sectors the
blocks must fit (`quadruples_lambda_grad_sym.plan`; the front end plans
inside `try` and runs the packed block otherwise).

**Equation (1)** (`zeta_opt.kernel`). No linearized amplitude equations are
coded. `ccsdt_opt.update_amps` is the Jacobi step N(t) = t + R(t) / D of the
amplitude equations, for T3 on the packed block or in the blocks of the
labels, in memory or mapped; with a stencil whose weights sum to zero and
reproduce the first derivative,

    zeta + (J zeta + q) / D = sum_k w_k N(t + x_k zeta) + q / D,

so a Jacobi step for zeta costs as many amplitude iterations as the stencil
has points, and DIIS runs on it as on the amplitudes (`diis.DIIS`, the
vectors in the scratch file). Three stencils:

| stencil | points | error of J zeta |
|---|---|---|
| exact | t +- zeta, t +- 2 zeta | none: R is a polynomial of degree <= 4 |
| central | t +- h zeta, h max\|zeta\| = 1e-5 | h^2 times the cubic term: 1e-10 relative, times a number of order one |
| forward | t + h zeta (and N(t), once), h max\|zeta\| = 3e-8 | h times the quadratic term: 1e-8 relative |

The solver goes up this ladder: forward differences until the change is
1e-6 of zeta (or below the threshold), central differences until it is
below the threshold, and then the last iterate once more with the points
+-2h - with those at +-h the exact formula at step h. If the exact change
is below the threshold as well, the iterations end; if not, they go on
with the exact stencil. So the result does not rest on the error estimates
of the table: the last Jacobi step is an exact one, and its size is the
convergence criterion. `Result.stencil_error` is the difference between
the central and the exact step at the last iterate (1e-16 to 1e-14 of the
full tensors in every run so far).

Memory (`zeta_opt.plan_for`): the amplitude steps have the `MemoryPlan` of
the amplitude equations, whose two arrays of the size of T3 are the shifted
amplitudes and the new vector. The equation has five more (zeta_3, q3 / D,
N(t), the accumulator of a stencil, a copy for the last check) and its DIIS
vectors. If an amplitude step does not fit in memory next to them, they are
mapped arrays of a scratch file, which every use passes over once from
front to back, and the amplitude step has the whole budget.

**Equation (2)** (`zeta_opt.lambda_sources`, `lambda_opt.kernel(sources=)`).
The directional derivative of the Lambda residuals: `lambda_opt.make_intermediates`
and `residuals` at the four amplitude sets of the exact stencil (two with
`stencil='central'`), in the layout of the Lambda equations (the packed
block, or the blocks of the labels), plus s2 and s3 of the pass - a
`lambda_opt.ResidualSources`, which the Lambda solver takes as it is. The
solver starts from lambda.

**Densities** (`density_opt.relaxed_rdm1(zeta=, lams=)`). `density_opt.kernel`
for (t, lambda') and at the amplitudes of the stencil for (t, lambda) -
four more passes with the exact stencil (the default), two with the central
one -; d1, the table GRADIENT and, with frozen or dropped spinors, B and K
of `frozen_terms` are combined with the weights (each is a polynomial in t;
B and K after their terms with t1). The integrals with one frozen index are
transformed once: `_Replay` writes the batches of the first pass to a
scratch file and the other passes read them. Then as for CCSDT(Q):
`quadruples_lambda_grad.orbital_gradient` (identities 1 to 3),
`FrozenTerms` in the first pass, `zvector._relax`.

**Front end.** `GCCSDT.solve_lambda(with_q='lambda')` and
`make_rdm1(with_q='lambda')`, next to `with_q=False` (CCSDT) and `True`
(CCSDT(Q)); `GCCSDTQ` hands both to its CCSDT step. `self.l1, l2, l3` are
and stay lambda of CCSDT (what `ccsdt_q_lambda()` takes; it is solved again
if the object holds that of CCSDT(Q)); zeta is `zeta1, zeta2, zeta3`,
lambda' is `l1_ql, l2_ql, l3_ql`, in the layouts of t and lambda;
`converged_zeta`, `converged_lambda_ql`, `ql_gradient_sectors`, `ql_info`
(cycles, amplitude steps, seconds, plans). The pass, zeta and lambda' are
kept until `t3` or `l3` is another array. `zeta_stencil` ('forward',
'central', 'exact': where the ladder starts) and `response_stencil`
('exact', 'central': the derivatives that are taken once). Labels: the
pass in sectors, the amplitude steps of zeta in the layout of the
iterations, lambda' and the densities in the blocks of the labels under
the conditions of `solve_lambda()` and `make_rdm1()` (`blocks_min`, the
plan inside `try`), on the packed block otherwise.

### 12.7 Timing

One job, 31773728 on c388 (8 cores of a Xeon Gold 6248R, `--mem=30G`,
`bench/slurm_ql_gradient.sh`, `bench/profile_ql_gradient.py`), the tree of
commit c268dd2: H2O / cc-pVTZ as spin orbitals (10 + 106, real, no labels:
everything on the packed block) and HBr / cc-pVTZ-DK with X2C spinors, 28
frozen (8 + 78, real, labels 2 m_j: sectors and blocks), `max_memory` 26
GB; then H2O again with `max_memory` 4 GB. Thresholds 1e-11 (energy) and
1e-9 (amplitudes, Lambda, zeta, lambda'). The node is shared: load 0.15 at
the start, 10 to 14 from the middle of the first run on (other jobs came),
and the times of one and the same step scatter by 20% for it - a CCSDT
iteration of H2O took 4.27 s in the first run and 4.62 s in the third, an
amplitude step inside zeta 5.04 s (the default ladder), 4.31 s (the exact
stencil, twenty minutes later, vector operations included) and 4.56 s (the
third run).

| | H2O / cc-pVTZ, 10 + 106 | HBr / cc-pVTZ-DK, 8 + 78, 28 frozen | H2O, `max_memory` 4 GB |
|---|---|---|---|
| CCSDT iteration | 4.27 s (19) | 0.33 s (19) | 4.62 s (19) |
| Lambda iteration, CCSDT | 5.16 s (17) | 0.26 s (17, blocks) | 5.45 s (17) |
| ... with the sources of (Q) / lambda' | 5.20 s (17) / 5.65 s (16) | 0.28 s (16) / 0.26 s (16) | 5.44 s (17) / 5.47 s (16) |
| (Q) energy | 43.9 s | 1.66 s (sectors) | 48.8 s |
| (Q)_L energy | 61.6 s | 1.98 s (sectors) | 69.4 s |
| (Q) gradient pass | 128.4 s | 3.42 s (sectors) | 260.7 s (s3, `<bc\|\|ef>` mapped) |
| (Q)_L gradient pass | 217.0 s | 4.00 s (sectors) | 436.0 s (s3, q3, `<bc\|\|ef>` mapped) |
| ... over the (Q)_L energy / over (Q) | 3.52 / 4.94 | 2.02 / 2.41 | 6.28 / 8.94 |
| zeta: iterations (forward + central), amplitude steps | 15 (13 + 2), 20 | 15 (13 + 2), 20 | 15 (13 + 2), 20 |
| zeta: seconds, per iteration | 109.2 s, 7.28 s | 6.51 s, 0.43 s | 91.1 s, 6.08 s (vectors mapped) |
| ... per iteration in CCSDT iterations | 1.71 | 1.32 | 1.31 |
| ... a forward iteration / the last one (central + check) | 6.0 to 7.2 s / 18.1 s | 0.32 s / 1.25 s | 4.5 to 5.1 s / 17.2 s |
| zeta with central differences from the start | 15, 32 steps, 165.5 s | 15, 32 steps, 10.3 s | |
| zeta with the exact formula throughout | 15, 60 steps, 258.7 s | 15, 60 steps, 19.1 s | |
| central against exact step, last iterate | 2.6e-13 | 2.5e-14 | 2.6e-13 |
| sources of lambda' (exact stencil) | 29.2 s | 2.05 s | 28.0 s |
| lambda' | 92.6 s | 4.43 s | 89.3 s |
| relaxed density after the equations, CCSDT | 4.75 s | 1.78 s | 5.11 s |
| ... CCSDT(Q) | 5.32 s | 2.29 s | 5.74 s |
| ... CCSDT(Q)_L (five passes) | 30.95 s | 5.59 s | 25.5 s |
| all after the CCSDT energy, CCSDT | 94.3 s | 6.49 s | 99.9 s |
| all after the CCSDT energy, CCSDT(Q) | 223.9 s | 10.4 s | 361.1 s |
| all after the CCSDT energy, CCSDT(Q)_L | 568.4 s | 27.3 s | 764.7 s |
| ... in CCSDT iterations | 133 | 83 | 165 |
| ... over CCSDT / over CCSDT(Q) | 6.0 / 2.5 | 4.2 / 2.6 | 7.7 / 2.1 |
| ... without Lambda of CCSDT (which E(Q)_L needs) | 478.9 s | 22.6 s | 669.9 s |
| dipole moment, CCSDT | 0.7498653312 | 0.3366355793 | 0.7498653312 |
| dipole moment, CCSDT(Q) | 0.7492563653 | 0.3363222230 | 0.7492563653 |
| dipole moment, CCSDT(Q)_L | 0.7492938320 | 0.3363471027 | 0.7492938320 |

("All after the CCSDT energy" of CCSDT(Q)_L: Lambda of CCSDT, the gradient
pass - which gives E(Q)_L as well -, zeta, the sources, lambda' and the
density. In the script Lambda of CCSDT is solved twice, because the
CCSDT(Q) part in between replaces it; the second time is not counted.)

**The cost of zeta in amplitude iterations.** A Jacobi step of the
equation is one amplitude step per point of the stencil, and an amplitude
step is an amplitude iteration without its DIIS: 20 steps for 15
iterations with the default ladder (1 for N(t), 13 forward iterations, one
central one, and the last iterate with the four points of the exact
formula), 32 with central differences from the start, 60 with the exact
formula throughout - 1.33, 2.13 and 4 amplitude steps per iteration, the
same zeta to 2e-11 of the full tensors (threshold 1e-9). Measured per
iteration: 1.32 (HBr) and 1.31 (H2O, third run) CCSDT iterations; 1.71 in
the first H2O run, whose amplitude steps took 5.04 s against 4.27 s of the
CCSDT iterations half an hour before, on a node that had filled up in
between. The whole solve: 20 to 26 CCSDT iterations, where CCSDT itself
took 19.

**Where the time goes** (H2O, first run, 568 s): the pass 38%, zeta 19%,
lambda' 16%, Lambda of CCSDT 16%, the densities 5%, the sources of lambda'
5%. The pass is 3.5 times the (Q)_L energy and 4.9 times (Q) on the packed
block (estimated before it was written: 3 and 4.5 to 5), 2.0 and 2.4 times
in sectors; the two derivatives that are taken once cost 5.7 (HBr: 7.9)
Lambda iterations (the sources: four builds of the intermediates and four
residuals) and four density passes. With frozen spinors the integrals with one frozen index
are transformed once (HBr: 0.6 s of the 5.6 s).

**Memory.** Peak of the anonymous resident memory of the process during a
stage, less what was resident before it, against the plan of the stage
(H2O; T3 is 0.185 GB, the packed `<vv||vv>` 0.25 GB):

| | 26 GB: plan | peak - before | 4 GB: plan | peak - before | peak |
|---|---|---|---|---|---|
| (Q)_L gradient pass | 7.25 GB | 6.86 GB | 2.34 GB (0.62 GB more in the scratch file) | 2.01 GB | 3.90 GB |
| zeta | 3.82 GB (0.93 GB of it the vectors) | | 2.17 GB (the vectors mapped) | | |
| lambda' | 3.30 GB | | 2.72 GB | | |
| zeta, sources, lambda' together | | 3.10 GB | | 1.93 GB | 3.93 GB |
| relaxed density, CCSDT(Q)_L | | 4.67 GB | | 0.65 GB | 2.83 GB |

With 4 GB every stage stays below `max_memory` (the largest peak of the
run: 3.98 GB, in the pass of CCSDT(Q)) and gives the same numbers to the
twelve decimals printed; the pass takes twice as long with its three large
arrays in the scratch file (as that of (Q) does), zeta is not slower with
its vectors there. The plans of zeta and of the Lambda equations are upper
bounds that count 0.7 GB for what the threaded BLAS keeps (section 9.3):
not tight.

### 12.8 Validation

#### The production code against the dense reference

`tests/test_quadruples_lambda_grad.py`, `tests/test_quadruples_lambda_grad_sym.py`
(13 + 5 tests, 100 s with four threads on the login node).

**The pass** (random amplitudes and lambda that solve nothing, 5 + 6 and
4 + 7 complex, 6 + 5 real; every memory plan; with and without the C
kernels): e (real and imaginary part), s2, s3, q2, q3, `de`, Gamma and the
densities - as an antisymmetric array over all spin orbitals in the form
that Hermitian integrals see - are those of `qlgrad_ref` to 4e-16 to 2e-15
relative (asserted: 1e-12). Sectors against the packed block (HBr /
cc-pVDZ-DK, 8 + 28, label-conserving parts; labels 2 m_j, modulo 4, one
label; complex amplitudes by a phase on every spinor): 2e-16 to 2e-15
relative for every array.

**Molecules** (converged CCSDT and Lambda to 1e-11; `zeta_opt.kernel` with
the default ladder; the sources and the densities with the exact stencil).
Largest absolute differences to `qlgrad_ref.relaxed_rdm1`:

| | o + v | E(Q)_L | zeta | sources of lambda' | lambda' | unrelaxed density | orbital gradient (largest element) | relaxed density |
|---|---|---|---|---|---|---|---|---|
| BH / STO-3G | 6 + 6 | -5.57e-5 | 1e-15 | 6e-17 | 2e-12 | 4.0e-13 | 1.1e-13 (1.1e-5) | 4.0e-13 |
| BH / STO-3G, complex | 6 + 6 | -5.57e-5 | | | | 4.0e-13 | 8e-14 (8.6e-6) | 4.0e-13 |
| LiH / STO-3G, complex | 4 + 8 | -1.35e-7 | 2e-17 | 2e-17 | 2e-13 | 8e-14 | 2e-14 (2.4e-7) | 6e-14 |
| H2O / STO-3G, X2C, 2 frozen | 8 + 4 | -1.57e-5 | | | | 2.7e-13 | 6e-14 (8.8e-4) | 3.5e-13 |
| BH / STO-3G, X2C, 2 frozen, 2 dropped | 4 + 4 | -1.45e-5 | 1e-14 | 4e-16 | 4e-12 | 2.0e-12 | 1.8e-12 (7.4e-3) | 2.6e-12 |

(asserted: 1e-10. The tests have four of the rows; BH with complex spin
orbitals was run by hand with the same code; empty cells: not printed.) What is left is the
convergence of the iterations: with the reference's own zeta the sources
agree to 6e-17 (exact stencil) and 1e-15 (central). Every block of the
gradient agrees by itself - frozen-active 2e-14 of 5e-3, dropped-active
2e-12 of 7e-3, among the occupied 1e-13 of 4e-6, among the virtual 4e-15 of
6e-9 (LiH). The density with the central stencil is that of the exact one
to 1e-12 (asserted), and the density of CCSDT is 1e-6 to 2e-4 away.

**The stencils of zeta** (BH with complex spin orbitals, 1e-11): the three
starting points of the ladder end at the same zeta (asserted 1e-11) with 48
(exact), 26 (central) and 18 (forward) amplitude steps for 12 iterations
each; H2O / STO-3G with X2C: 44, 24, 16 for 11; at a threshold of 1e-8: 32,
18, 13 and 28, 16, 12. The central step against the exact one at the last
iterate: 1e-16 to 8e-14 of the full tensors in all runs, 12.7 included.
With the vectors of the equation in a scratch file: the same zeta.

**Front end.** HF / 6-31G with X2C spinors, 2 frozen (8 + 12; real spinors,
T3 in the blocks of the labels): trace(z D) of
`make_rdm1(with_q='lambda')` = 0.873992419899 against the four-point
finite field of E(SCF) + E_corr(CCSDT) + E(Q)_L (step 1e-3, energies of
the front end, thresholds 1e-12 and 1e-10): 0.873992419985, difference
-9e-11 (asserted 2e-9; CCSDT 0.873691371042, CCSDT(Q) 0.874002176879).
Lambda, lambda' and the densities in the blocks of the labels (`blocks_min
= 0`) against the packed block: 1e-12. After `make_rdm1()` and
`make_rdm1(with_q=True)` the same density again, and `ccsdt_q_lambda()` the
same energy. HBr / cc-pVDZ-DK, 28 frozen (Kramers pairs), everything in
blocks and sectors against `symmetry = False`, everything on the packed
block: the densities in the AO basis agree within the thresholds of the
iterations (1e-9; asserted 2e-8), norms of zeta_2, zeta_3 and lambda'_2 -
lambda_2 to 1e-8.

**Memory** (tracemalloc; 12.7 has the resident memory of the large runs).
The pass on the packed block: allocations within `Plan.need` and above 0.8
of it for four plans from the floor up, 5 + 20 complex and 6 + 26 real;
in sectors 0.5 to 1 of the plan. zeta (H2O / 6-31G with X2C spinors, T3 of
1.08 MB, DIIS space 4): 36.1 MB allocated with the vectors in memory, 22.2
MB with them in the scratch file - the difference is the 13 vectors of the
plan (14.0 MB); the amplitude plan itself is the upper bound of 12.7.

#### The dense reference

**Derivatives of e** (5 + 5 spin orbitals, random amplitudes, lambda and
integrals that are antisymmetric and nothing else - not Hermitian -,
four-point differences of the dense e along random directions, step 1e-3):
t2, t3, l2, l3, g and the orbital energies to 1.2e-12 of derivatives of
0.4 to 1100, real and complex (complex numbers compared); q2 and q3 are the terms
`R2: [V,T4]`, `R3: [V,T4]` with T4 = r / D (1e-12); e is
`lambda_ref.quadruples_lambda` for Hermitian integrals.

**Directional derivatives** (4 + 5 model with a complex Hermitian
Hamiltonian, not diagonal in f, random zeta): the five-point formula with
step 1 against the same with step 1e-2 - J zeta 6e-15 (of 14), the Lambda
response 1e-14 (of 0.3) -, `jacobian_dressed` 5e-15, and
zeta . (J^T lambda) = lambda . (J zeta) with J^T lambda the part of the
Lambda residuals linear in lambda (2e-16 of 0.19).

**The Lagrangian** (the same model with more correlation: e = -2.64e-4 +
1.18e-4 i, zeta up to 5.5e-4, lambda' - lambda up to 1.2e-3; 35 and 42
iterations). Derivatives of F along random directions, four-point
differences, step 1e-3:

| with respect to | at the solution | zeta left out | lambda for lambda' |
|---|---|---|---|
| t | 3.7e-12 | 1.1e-3 | 1.0e-1 |
| lambda | 1.9e-12 | 1.4e-1 | 1.9e-12 |
| zeta | 2.3e-12 | 2.7e-12 | 2.3e-12 |
| lambda' | 1.1e-12 | 1.1e-12 | 3.6e-12 |

Derivatives of E(CCSDT) + e at fixed orbitals, the amplitude and the
Lambda equations solved again for every Hamiltonian (to 1e-12), against
the traces with the densities; steps 4e-4, 2e-4 and their extrapolation,
differences finite difference - analytic of the (Q)_L part (what e adds
to the derivative of CCSDT) and of CCSDT:

| perturbation | (Q)_L part, analytic | 4e-4 | 2e-4 | extrapolated | CCSDT, extrapolated |
|---|---|---|---|---|---|
| one-electron, complex Hermitian | -2.969e-4 - 1.448e-3 i | 7.9e-13 | 1.4e-13 | 2.0e-13 | 1.2e-12 (of 5.0) |
| two-electron, complex Hermitian | -2.652e-3 - 4.390e-3 i | 7.3e-9 | 4.6e-10 | 2.0e-12 | 6.4e-12 (of 5.0) |
| orbital energies of D | 1.878e-4 | | | 1.6e-16 (step 1e-3) | |

For the one-electron perturbation the explicit part of e is zero: all of
the first row is zeta and lambda-bar. For the two-electron one the
explicit part (d2 of e) is 2.06e-3 + 1.43e-3 i.

**Relaxed dipole derivative**, trace(z D) of the dense reference against
E(SCF) + E_corr(CCSDT) + E(Q)_L with the SCF repeated and Lambda solved
again in the field (`bench/qlgrad_ccsdt_ql.py`; energies, amplitudes and
Lambda of the production codes; four-point differences at steps 4e-3,
2e-3, 1e-3 and the extrapolation of two consecutive steps; SCF converged
to 1e-13 and 1e-10 in the gradient, CCSDT to 1e-13 and 1e-11, Lambda,
zeta, lambda' and the Z-vector equations to 1e-11). Differences analytic -
finite field:

| | o + v | trace(z D), CCSDT(Q)_L | its (Q)_L part | step 4e-3 | 2e-3 | 1e-3 | 4e-3, 2e-3 extrapolated | 2e-3, 1e-3 extrapolated | CCSDT alone, last column |
|---|---|---|---|---|---|---|---|---|---|
| H2O / STO-3G | 10 + 4 | 1.582686756789 | 8.24e-5 | 6.0e-10 | 2.1e-11 | -2.2e-11 | -1.7e-11 | -2.5e-11 | -2.5e-11 |
| H2O / STO-3G, complex spin orbitals | 10 + 4 | 1.582686756789 | 8.24e-5 | 6.0e-10 | 5.2e-11 | -2.0e-11 | 1.6e-11 | -2.4e-11 | -2.4e-11 |
| H2O / STO-3G, X2C spinors | 10 + 4 | 1.583198446106 | 8.29e-5 | 5.8e-10 | 2.5e-11 | -5.0e-12 | -1.2e-11 | -7.1e-12 | -7.5e-12 |
| H2O / STO-3G, X2C, 2 frozen | 8 + 4 | 1.583145734039 | 8.29e-5 | 5.9e-10 | 3.5e-11 | -2.5e-11 | -2.3e-12 | -2.9e-11 | -2.9e-11 |
| BH / STO-3G | 6 + 6 | 2.087292537268 | 2.80e-4 | -1.9e-8 | -1.2e-9 | -7.8e-11 | 5.3e-12 | -4.3e-12 | -1.1e-11 |
| BH / STO-3G, complex | 6 + 6 | 2.087292537268 | 2.80e-4 | -1.9e-8 | -1.2e-9 | -7.5e-11 | -2.0e-12 | -1.7e-13 | -6.6e-12 |
| BH / STO-3G, complex, 2 frozen, 2 dropped | 4 + 4 | 2.094213650605 | 2.15e-5 | -1.7e-8 | -1.1e-9 | -8.0e-11 | 2.7e-12 | -1.4e-11 | -1.4e-11 |
| BH / STO-3G, X2C, 2 frozen, 2 dropped | 4 + 4 | 2.094153685064 | 2.13e-5 | -1.7e-8 | -1.1e-9 | -7.8e-11 | -3.0e-12 | -1.2e-11 | -1.2e-11 |
| LiH / STO-3G | 4 + 8 | 4.833110085756 | -4.58e-6 | -6.4e-5 | -3.9e-6 | -2.4e-7 | 3.8e-11 (1e-3, 5e-4) | 4.0e-12 (5e-4, 2.5e-4) | 4.5e-12 |
| LiH / STO-3G, complex | 4 + 8 | 4.833110085756 | -4.58e-6 | -6.4e-5 | -3.9e-6 | -2.4e-7 | 2.2e-11 (1e-3, 5e-4) | 6.9e-12 (5e-4, 2.5e-4) | 7.4e-12 |
| H2O / 6-31G | 10 + 16 | 1.227832006988 | 1.23e-4 | 1.2e-8 | 7.9e-10 | 2.6e-11 | 5.6e-11 | -2.4e-11 | -2.1e-11 |
| H2O / 6-31G, complex | 10 + 16 | 1.227832006988 | 1.23e-4 | 1.2e-8 | 7.1e-10 | -2.2e-12 | -3.3e-11 | -4.9e-11 | -4.6e-11 |
| HF / 6-31G, X2C | 10 + 12 | 16.451085826812 | -3.02e-4 | -5.9e-9 | -3.5e-10 | -4.2e-11 | 2.0e-11 | -2.1e-11 | -2.7e-11 |
| H2O / 6-31G, X2C, 2 frozen, 4 dropped | 8 + 12 | 1.265705021477 | 5.84e-4 | 1.6e-8 | 9.8e-10 | 6.9e-11 | -2.9e-11 | 7.8e-12 | 8.3e-12 |

(STO-3G rows: on the login node; 6-31G rows: jobs 31770509 to 31770512,
c158; the dense reference takes 22 minutes for 10 + 16 spin orbitals.)
CCSDT alone has the same differences in
every row, to 7e-12: what is left is the error of the differences
(energies to 1e-13 over steps of 1e-3), as in 11.5, not of the terms of
(Q)_L - the point-charge runs below, with a hundred times larger energy
differences, show them to 2e-13. LiH has the truncation error it has in
11.5; its extrapolations are those of the steps in brackets. Lambda' and
zeta took 13 to 19 and 10 to 16 iterations. The gradient among the
occupied spin orbitals is 1e-6 to 1e-5 (LiH: 1e-14), among the virtual
ones 7e-9 for LiH, 1e-5 to 2e-5 in the 6-31G rows and zero by symmetry in
the other STO-3G ones.

**External references.**

- CFOUR's ncc (`CALC=CCSDT(Q)_L`, `PROPS=FIRST_ORDER`, the input of
  `bench/refs/cfour/h2o_sto3g/ZMAT.qprops.bohr` with that `CALC`) stops in
  `xncc` with `ncc.cxx(134): error: CCSDT(Q)_L derivatives are not supported
  in NCC`. Its source has a density for CCSD(T)_Lambda
  (`cc_density/ccsdt/ccsd_t_l_density.cxx`) and none for CCSDT(Q)_Lambda
  (`cc_density/ccsdtq/`: CCSDT(Q), CCSDTQ and the approximate iterative
  models).
- MRCC (`calc=CCSDT(Q)_L`, `dens=1`): `Derivatives are not implemented for
  approximate CC methods!`
- So the only number another program gives is a finite difference of its
  energies. MRCC takes point charges (`qmmm=Amber`, `pointcharges`): H2O
  with a charge q at (0, 0, 10) bohr, q = +-0.1, +-0.2, +-0.4 (RHF, all
  electrons, no symmetry; four-point differences at steps 0.2 and 0.1 and
  their extrapolation), against trace(V D) + the repulsion of the charge
  and the nuclei, V = -1 / |r - R|
  (`bench/qlgrad_ccsdt_ql.py CASE charge=0,0,10 steps=0.2,0.1`):

  | | zccsdt, analytic | zccsdt, differences (0.2, 0.1 extrapolated) | MRCC, differences, step 0.2 | step 0.1 | extrapolated |
  |---|---|---|---|---|---|
  | H2O / STO-3G, CCSDT | 0.006434729263 | 0.006434729263 | 0.006434729265 | 0.006434729264 | 0.006434729264 |
  | H2O / STO-3G, CCSDT(Q)_L | 0.006433921675 | 0.006433921675 | 0.006433921676 | 0.006433921673 | 0.006433921673 |
  | ... its (Q)_L part | -8.07588e-7 | -8.07588e-7 | -8.07589e-7 | -8.07591e-7 | -8.07591e-7 |
  | H2O / 6-31G, CCSDT | 0.009869336169 | 0.009869336169 | 0.009869336174 | 0.009869336171 | 0.009869336171 |
  | H2O / 6-31G, CCSDT(Q)_L | 0.009868191549 | 0.009868191549 | 0.009868191557 | 0.009868191550 | 0.009868191550 |
  | ... its (Q)_L part | -1.144620e-6 | -1.144620e-6 | -1.144617e-6 | -1.144621e-6 | -1.144621e-6 |

  MRCC's SCF, CCSDT and CCSDT(Q)_L energies with the charge are those of
  zccsdt to all twelve decimals printed (q = 0.1, STO-3G); with twelve
  decimals its differences are good to 2e-11 at step 0.1, and they are
  within 2e-12 of the analytic numbers. zccsdt's own differences for this
  perturbation are 4e-14 (STO-3G) and 2e-13 (6-31G) off the analytic
  derivative, of a (Q)_L part of 1e-6: six digits of what (Q)_L adds.
  (STO-3G on the login node, 6-31G: job 31770606 on c158, MRCC first.)

### 12.9 Not done

- linearized amplitude equations, Lambda residuals and densities of their
  own: all three derivatives along zeta are differences of the production
  codes (12.6). For zeta that is one amplitude iteration per iteration in
  the forward phase; for the Lambda residuals and the densities four
  evaluations each, once;
- the Lambda equations hold T3, l3 and the residual in memory (section 9),
  so lambda' does too, and its sources are one more array of that size;
  only the pass and zeta go to scratch files;
- E[Q]_L (`e_ql5`) is not made by the gradient pass (`ccsdt_q_lambda()`
  has it);
- the two-particle density as an array, geometric gradients,
  non-canonical or non-Hartree-Fock references;
- CCSDT[Q]_Lambda (the part of Lambda_2 in z);
- between degenerate spinors no rotation is made, as in 11.4;
- no comparison with another program's analytic number: none has one
  (12.8).
