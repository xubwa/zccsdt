# CCSDT factorization and contraction schemes

This document records how each part of the CCSDT iteration is evaluated in
`src/zccsdt/cc/ccsdt_opt.py`, the alternatives considered and why the chosen
scheme was picked.

- The equations are those of the ported reference implementation
  (`src/zccsdt/cc/ccsdt.py`, T1-dressed formalism), which is validated
  element-wise against a determinant-space brute-force oracle.
- Every optimized step is tested against the reference step to 1e-12
  (`tests/test_ccsdt_opt.py`).

Notation:
- `o`, `v`: numbers of occupied and virtual spin orbitals.
- `P`, `Q`: occupied pairs i<j and virtual pairs a<b, with
  `n_P = o(o-1)/2` and `n_Q = v(v-1)/2`.
- `I`, `A`: occupied triples and virtual triples, with `C(o,3)` and `C(v,3)`
  of them.

Flop counts are fused multiply-adds (FMA).

## 1. T3 storage and access (see also `src/zccsdt/cc/t3pack.py`)

T3 is stored only as its unique block `T3p[I, A]` (i<j<k, a<b<c). This is
the minimum: 1/36 of the full array. Nothing in the iteration ever forms
the full o^3 v^3 array.

A triple is a free index plus a pair in three ways, with signs +, -, +:
(p<q<s) = (p; qs) = -(q; ps) = (s; pq). Every T3 contraction uses one of
two *views*:

| View | Shape | Built by | Used for |
|---|---|---|---|
| half-expanded | `T[P, d, m, Q]` (batch of occupied pairs) | `gather_half` (C, OpenMP) | ring, drive, T3 -> T1/T2, T3 -> W |
| vir-half-expanded | `T[Q, I, c]` (batch of occupied triples) | `gather_vir_half` (C) | pp ladder |

Results come back through fused "antisymmetrize + pack" reductions:
- `add_reduce_split`: computes `P(i/jk) P(a/bc)` of `Z[P, a, i, Q]`. It is
  parallel over target triples, so threads never write the same element.
- `add_reduce_pair_first`: computes `P(c/ab)` of `X[Q, I, c]`.

No 6-index transpose exists anywhere. The gather *is* the transpose, and it
writes exactly the layout the following GEMM needs. Gather and reduction
traffic is O(o^3 v^3) per pass, against O(o^4 v^4) or O(o^3 v^5) flops.

Alternatives considered:

| Storage | Memory | Comment |
|---|---|---|
| full o^3 v^3 | 36x | ported starting point; 52 GB per copy for N2/cc-pVTZ complex |
| i<j<k with full abc cubes (CFOUR `ncc`) | 6x | each triple's abc cube is a natural GEMM operand, but the ladder costs 2x flops unless pairs are packed |
| i<j<k with (a<b, c) (CFOUR `vcc`) | 3x | ladder-friendly, but needs re-packing after every term |
| **fully packed (chosen)** | **1x** | minimal memory and flops; the views are built on the fly by cheap gathers |

## 2. Per-iteration work

| Step | Scheme | FMA executed | Minimum (antisymmetry exploited) |
|---|---|---|---|
| pp ladder + F_vv | per batch of triples I: `X[Q',(I,c)] = W(Q',Q) T[Q,(I,c)]`, then P(c/ab) reduction | C(o,3) n_Q^2 v ~ o^3 v^5/24 | o^3 v^5/24 |
| ring | per occupied pair P: `Z[P] = W[(a,i),(d,m)] T[P](d m, Q)` | n_P (ov)^2 n_Q ~ o^4 v^4/4 | o^4 v^4/4 |
| drive (T2 x W_vvvo, T2 x W_vooo) | GEMMs accumulated into the ring buffer Z (BLAS beta=1) | o^3 v^4/4 + o^4 v^3/4 | same |
| hh ladder + F_oo | o < 16: one operator O6[I,J] on packed T3, `R += O6 @ T3p`; else pair-driven, one GEMM with W'(ij,mn) per spectator index (2.3) | C(o,3)^2 C(v,3) ~ o^6 v^3/216; pair-driven o C(o-1,2)^2 C(v,3) ~ o^5 v^3/24 | o^5 v^3/24 (ratio o/9) |
| pass 1: T3 -> W_vvvo, W_vooo, T1, T2 | two stacked GEMMs per batch of pairs, plus two matrix-vector passes | about o^3 v^4 | same order |
| T1 dressing | slot by slot; packed vvvv dressed directly | o v^4/2 (vvvv) | -- |
| CCSD part (T1/T2 residual) and the two- and four-index intermediates | T1-dressed; ladder on packed pairs with the dressed packed vvvv; ring layout of t2 for the o^3 v^3 products (2.6) | o^2 v^4/4 (ladder) + 3 (ov)^3 (W_ovvo, ring, t2 x ovvv) + o^2 v^4 (t2 x ovvv of W_vvvo) | -- |

### 2.1 pp ladder (o^3 v^5), F_vv folded

The term is `R += P(c/ab) sum_{d<e} W(ab,de) t(ijk,dec)`. Its operator is
`W = ~<ab||de> + sum_{m<n} t2(mn,ab) <mn||de>`, kept packed over both pairs
(`n_Q x n_Q`, v^4/4).

The F_vv term `P(a/bc) F_ad t(ijk,dbc)` has the same structure. Adding
`1/2 (F_ad d_be - F_bd d_ae - F_ae d_bd + F_be d_ad)` to W reproduces it
exactly (see `add_one_body_pairs`). The term therefore costs no extra GEMM
and no extra pass over T3.

T3 is gathered pair-major as `T[Q, I, c]`. The GEMM output `X[Q', I, c]` is
reduced directly by the C kernel, so the batch involves no transpose at all.

Packed vvvv dressing (`dressed_vvvv_pp`) uses
`~<ab||de> = <ab||de> - Y[a,b] + Y[b,a] + Z[ab]`, where
`Y = t1^T <ov||vv>` and Z is a pair-space GEMM with
`tau(mn,ab) = t(m,a) t(n,b) - t(n,a) t(m,b)`. No full v^4 temporary is
needed. The undressed packed vvvv is cached on the integral object.

### 2.2 Ring + drive (o^4 v^4)

Both terms carry the antisymmetrizer `P(i/jk) P(a/bc)` (the drive terms
after renaming indices). They are accumulated into one half-expanded buffer
`Z[P, a, i, Q]` per batch of occupied pairs and reduced once.

- **Ring.** `W[(a,i),(d,m)] = ~<ma||di> + sum_{ne} <mn||de> t2(in,ae)`. The
  half-expanded `T[P]` is exactly the `(d m) x Q` matrix this GEMM needs:
  M = K = ov, N = n_Q, one GEMM per pair.
- **Drive 1.** `Z[P,c,i,Q] -= sum_e t2(P,e,c) W_vvvo(Q,e,i)` is a single
  GEMM, (n_b v) x v x (o n_Q).
- **Drive 2.** `Z[P,a,k,Q] -= sum_m W_vooo(m,a,P) t2(m,k,Q)` is a single
  GEMM, (n_b v) x o x (o n_Q).

The drive GEMMs accumulate directly into Z through SciPy BLAS with beta = 1
(`zccsdt.blas.gemm`). The numpy form `Z -= A @ B` was 18x slower because of
the output-sized temporary and its page faults.

### 2.3 hh ladder and F_oo (o^5 v^3)

Both act only on the occupied indices of T3. Together they form a linear
operator on antisymmetric 3-index tensors,

    (O t)[ijk] = P(k/ij) 1/2 sum_mn W[mn,ij] t[mnk] - P(i/jk) sum_m F[m,i] t[mjk],

which `t3kern.OccOperator` applies to the packed T3 (or to the block of a
symmetry label: O does not couple the blocks) in one of two forms.

**The matrix on packed triples** (fewer than 16 occupied spinors, and
arrays in a mapped file). `O6[I, J]`, C(o,3) x C(o,3), and one GEMM,
`R += O6 @ T3p`, with N = C(v,3). This does o/9 times the minimal flops
(1.1x for o = 10, 1.6x for o = 14), but needs no gather, no reduction and no
transpose. CFOUR evaluates this term with level-1 DAXPY loops.

The matrix is written down element by element
(`t3pack.occ_operator_direct`, kernel `zc_occ_operator`): with the triple
I = (k; P) written as (free index; pair) in its three ways,
`O6[I, (k; P')] += s s' W[P', P]` for the pairs P' without k, and
`O6[I, (m; P)] -= s s' F[m, k]` for the m outside P, s and s' being the
signs of the two decompositions: 3 (C(o,2) + o) terms per row. Until
October 2026 it was built by applying the operator to the packed basis with
einsum (`t3pack.occ_operator`, kept as the reference of the tests): o^8
work and temporaries of C(o,3) o^3 elements that the memory plan did not
count. That never showed with 8 to 10 occupied spinors; with 26 it was
1.4 s per iteration on two cores, whatever the number of virtual spinors
(more than half of an iteration of 26 + 28 spinors in label blocks, a tenth
at 26 + 120), and it took the peak memory of 26 + 120 in blocks to 6.6 GB
where the plan counted 5.4 GB. Now: 5 ms, within the plan (4.9 GB at that
commit).

**Pair-driven** (16 occupied spinors and more). With the triples written as
(spectator k; pair P),

    R[(k; P), A] += sum_P' W'[P, P'] T[(k; P'), A],
    W'[(i<j), (m<n)] = W[mn,ij] - 1/2 (F[m,i] d_nj - F[n,i] d_mj - F[m,j] d_ni + F[n,j] d_mi)

(each F term is met from two spectators, hence the half; the same folding
as F_vv into the pp operator). The kernel `zc_t3_hhpair` shares the columns
of T3 among the threads in chunks of 96; for a chunk and one spectator after
the other it copies the rows that contain the spectator (C(o-1,2) x 96, in
cache), multiplies with the matrix of that spectator (a sequential GEMM)
and adds the result to the rows of R. The matrices of all spectators
(`zc_t3_hhpair_w`, signs folded in) take 9/o of the memory of O6. That is
o/9 of the flops of the matrix form, for reading T3 and writing R three
times instead of once. The transposed operator (Lambda equations) is the
same with W' transposed.

Measured (`bench/hh_ladder.py`, random data, real, T3 of 2 GB; one node with
2 x 24 cores Xeon Gold 6248R, job 31780775; seconds for one application,
matrix / pair-driven):

| o | 8 threads | 16 | 24 | 48 |
|---|---|---|---|---|
| 8  | 0.12 / 0.27 | 0.09 / 0.15 | 0.08 / 0.11 | 0.15 / 0.16 |
| 12 | 0.26 / 0.33 | 0.13 / 0.17 | 0.10 / 0.12 | 0.11 / 0.14 |
| 14 | 0.37 / 0.40 | 0.20 / 0.21 | 0.14 / 0.14 | 0.12 / 0.15 |
| 16 | 0.57 / 0.48 | 0.32 / 0.25 | 0.22 / 0.18 | 0.18 / 0.17 |
| 20 | 1.02 / 0.67 | 0.57 / 0.40 | 0.39 / 0.32 | 0.29 / 0.23 |
| 26 | 2.10 / 1.19 | 1.06 / 0.65 | 0.80 / 0.48 | 0.61 / 0.36 |
| 30 | 3.25 / 1.52 | 1.63 / 0.85 | 1.15 / 0.58 | 0.80 / 0.42 |

The crossover is between 14 and 16 occupied spinors at every thread count
(`t3kern.HH_PAIR_MIN_OCC = 16`, environment `ZCCSDT_HH_PAIR_MIN_OCC`), and
the gain at 26 is 1.6 to 1.8 where the flop count says 2.9: the rest is
the three passes over T3 and R. With the T3 of 26 + 120 spinors (5.8 GB,
dense layout): 6.1 / 3.5 s on 8 threads, 3.1 / 1.9 on 16, 2.3 / 1.5 on 24,
1.8 / 1.0 on 48. In the blocks of the labels of HBr both forms take the
same time (0.13 to 0.05 s per iteration at 26 + 120: the blocks are small);
the pair-driven form is used there too, for its memory.

### 2.4 Pass 1: T3 -> T1, T2 and the T3 dressing of W_vvvo, W_vooo

All six contractions read the same half-expanded batch `T[P, d, m, Q]`:

| Quantity | Contraction | Summed over |
|---|---|---|
| dW_vvvo[i,e,Q] | `sum g(P,e,f) T[P,f,i,Q]` | (P, d) |
| C0[i,j,Q] (T3 -> T2) | `sum g(P,e,j) T[P,e,i,Q]` | (P, d) |
| dW_vooo[m,a,P] | `sum g(m,n,R) T[P,a,n,R]` | (m, Q) |
| B0[P,a,b] (T3 -> T2) | `sum g(a,m,R) T[P,b,m,R]` | (m, Q) |
| r1[a,i] (T3 -> T1) | `sum g(P,Q) T[P,a,i,Q]` | (P, Q) |
| rA[P,Q] (T3 -> T2) | `sum f(m,e) T[P,e,m,Q]` | (d, m) |

- The two (P, d) contractions share the matrix view `(P d) x (m Q)` of T.
  With their integral operands stacked, they are **one GEMM**.
- The two (m, Q) contractions likewise form **one GEMM**.
- The last two are matrix-vector passes.

In an earlier version, the layout `T[P, m, d, Q]` forced a loop of many
medium GEMMs. That was 25x slower because of BLAS threading overhead.

### 2.5 T1 dressing

The dressing is a product of one-index transformations:
- a bra virtual index picks up `-t1(m,a)` from occupied;
- a ket occupied index picks up `+t1(i,e)` from virtual.

It is applied slot by slot, ket slots first (`dress_block`). Each step is a
contiguous GEMM on blocks that differ only in that slot's source space.
The full n^4 tensor and the n^5 transformation of the starting point are
gone.

The GEMMs accumulate in the dressed block (`_add_contract_slot`,
`zccsdt.blas.gemm` with beta = 1): the last slot and the first are one
product, the second slot is one product per leading index, and the third is
moved to the end and back by the threaded transposing copy
(`ccsdt_opt.transposed`, kernel `zc_axpby4`). The numpy form
(`blk + np.matmul(...)`) made a temporary of the size of every result, added
it on one thread and, for a slot in the middle, ran a stack of small
products one after the other. `dress_fock` takes `sum_ia <ri||sa> t1[i,a]`
from the stored blocks as they are; it used to make a negated, transposed
copy of the whole ovvv block for the vv part.

### 2.6 CCSD part and the intermediates

The T1/T2 residuals (`ccsd_residual_packed`) and W_vooo, W_ovvo, F_oo, F_vv,
W_oooo are small next to the T3 terms - 5 % of an iteration in the dense
layout - but they were written with einsum and fancy indexing, which run on
one thread wherever they copy, transpose, gather or scatter an o^2 v^2
array. In the blocks of a symmetry label, where the T3 terms shrink by an
order of magnitude and these do not, they were 30 to 40 % of an iteration
and did not get faster with more cores. They now follow the patterns of the
packed spinor CCSD (`zccsdt.zccsd.amps`): every layout is written by an
OpenMP kernel, every product is a GEMM of the kernel library, one-body
terms act on packed pairs. (The kernels of `zccsd` themselves do not fit:
they are complex only, for pairs i>j in tril order - the pairs here follow
the symmetry labels - and for the bare-integral equations with an AO-driven
ladder, while the CCSDT step has T1-dressed integrals and its ladder
operator in memory. The kernels are in `csrc/t3pack.c`, real and complex.)

| Term | Before | Now |
|---|---|---|
| t2 gathers `t2[:, :, (a<b)]`, `<mn\|\|(e<f)>`, packed ovvv of the pass-1 operand | fancy indexing | `t3pack.pack_last2` (`zc_pack_last2`) |
| W_ovvo += `<mn\|\|de> t2[i,n,a,e]` | einsum (transposed copies of both operands on one thread) | t2 in the ring layout `A[(i a),(k c)]` once per step (`transposed`), one GEMM `X = G_ring A^T` |
| F_oo, F_vv (T2 parts) | two einsums | partial traces of X: `F_oo[m,i] += 1/2 sum_d X[(m d),(i d)]`, `F_vv[a,d] -= 1/2 sum_m X[(m d),(m a)]` |
| operator of the T3 ring term `[(a i),(d m)]`, of the CCSD ring term `1/2 (g + W)[(k c),(j b)]` | numpy transposes of W_ovvo `[m,a,d,i]` | both from X in the ring layout by `zc_axpby4`; W_ovvo in its own order is not formed |
| W_vooo: `<mn\|\|ie> t2[j,n,b,e]`, `<mb\|\|ef> t2[i,j,e,f]` | einsum | GEMM with A; GEMM on the arrays as stored |
| CCSD ring term, P(ij) P(ab) of `X[(i a),(j b)]` | four fancy gathers of an (ov)^2 array | `t3pack.add_asym_pack4` (`zc_asym_pack4`) |
| P(ab) F_bc t_ijac | two fancy gathers | `t3pack.add_asym_last2` |
| -P(ij) F_kj t_ikab | einsum on `t2[:, :, (a<b)]` and two fancy gathers | a matrix on the packed occupied pairs (`occ_pair_one_body`) added to W_klij: no extra product |
| R1: `f_kc t_ikac`, `g_klic t_klac` | einsum | matrix-vector product and o small GEMMs with A |
| `<ab\|\|ij>` dressed in the ket with Bv | stacked `np.matmul(t1, Bv)` and fancy gathers | `zc_vvoo_ket`, one virtual pair per thread |
| new T2 = t2 + A(R2) / D | four fancy scatters, an o^2 v^2 denominator array, two temporaries | `t3pack.t2_update` (`zc_t2_update`), one pass |
| DIIS: pack and unpack of T2, first \|dT3\| | fancy indexing, numpy in chunks | `pack_last2`, `unpack_pairs`, `diff_norm2` |

The o^2 v^4 product of W_vvvo (`sum_mf <mb||ef> t2[m,i,a,f]`, a GEMM per
occupied index m on ovvv as stored) was threaded before and is unchanged;
the stage of W_vvvo, of which it is the largest part, is what is left of
the intermediates (0.6 to 0.75 s of 1.1 to 1.2 s at 26 + 120). It, like
every array of this section, does not use the symmetry labels: the
integrals and T2 are dense, so the label blocks gain nothing here.

### 2.7 Label blocks: products too small for a threaded BLAS

In the blocks of the labels (`t3engine.SpinBlockT3`) the passes run over
groups of occupied pairs, of virtual pairs and of blocks of rows, and what
is one large GEMM in the dense layout becomes thousands of products of a
millisecond. Called one after the other, each with the threads of the BLAS,
they do not scale: with 26 + 120 spinors pass 2 took 4.8, 4.7 and 7.8 s on
16, 24 and 48 cores. Two changes:

- `zccsdt.blas.gemm_tasks` (`zc_gemm_tasks`): a list of products of parts
  of three arrays in one parallel region. Every product is split into
  pieces of 96 columns, a piece is one sequential GEMM, and the threads take
  the pieces of all products from one queue. The ring products of a pair
  group, the drive products of a block (their operand for all pairs of the
  batch is built at once) and the products of pass 1 go through it.
- The batch of pass 2 is sized by the memory it is allowed
  (`_pass2_batch`): as many occupied pairs as two thirds of `mem_mb` hold in
  the block layout. The batch of the dense layout - three pairs at 26 + 120,
  43 MB of the 2000 MB allowed - made 1861 groups of gathers, products and
  reductions per iteration; now 192, in a work array of 1.3 GB, which the
  memory plan counts (and shrinks with `max_memory`: 473 MB under 5.5 GB,
  T3 then in the scratch file).

### 2.8 Measured: 26 + 120 spinors, before and after

HBr, X2C, cc-pVQZ-DK, 10 spinors frozen, the lowest 120 virtual spinors,
real arithmetic; `bench/bench_plan.sh`, 4 iterations, one node with 2 x 24
cores (Xeon Gold 6248R), threads bound to cores; seconds per iteration
(cores busy); the stage timers are averages over all four iterations.
Before: 940d92a; after: d81db42 (both in one job, 31781047, node c360).

Label blocks:

| Stage | | 8 threads | 16 | 24 | 48 |
|---|---|---|---|---|---|
| whole iteration | before | 16.4 (4.7) | 13.2 (7.1) | 13.1 (9.4) | 17.0 (15.3) |
| | after | 10.9 (6.7) | 7.1 (11.9) | 6.8 (16.9) | 6.6 (26.5) |
| pass 2 | before | 5.95 (5.2) | 4.77 (8.2) | 4.75 (10.6) | 7.91 (16.5) |
| | after | 4.43 (7.4) | 2.74 (13.8) | 2.21 (20.1) | 2.31 (33.5) |
| intermediates | before | 4.86 (3.4) | 4.16 (4.5) | 4.21 (6.2) | 4.43 (10.2) |
| | after | 1.95 (6.5) | 1.21 (10.6) | 1.13 (14.9) | 1.07 (22.3) |
| T1 dressing | before | 1.20 (2.1) | 1.24 (3.9) | 1.37 (7.4) | 1.51 (11.4) |
| | after | 0.47 (4.4) | 0.43 (8.4) | 0.53 (14.2) | 0.64 (17.9) |
| pass 1 | before | 1.21 (5.5) | 0.92 (8.8) | 0.88 (11.6) | 1.22 (16.1) |
| | after | 1.25 (5.6) | 0.91 (9.3) | 0.82 (12.2) | 1.01 (17.0) |
| pp ladder | before | 1.65 (7.5) | 0.96 (14.5) | 0.76 (20.6) | 0.69 (30.4) |
| | after | 1.65 (7.5) | 0.97 (14.4) | 1.24 (19.8) | 0.67 (31.5) |
| DIIS | before | 0.67 (5.0) | 0.61 (8.0) | 0.56 (11.7) | 0.86 (21.3) |
| | after | 0.74 (4.7) | 0.57 (8.4) | 0.58 (11.4) | 0.82 (21.5) |
| peak memory / GB | before | 6.6 | 7.0 | 7.3 | 8.0 |
| | after | 6.0 | 6.5 | 6.6 | 7.1 |

(The pp ladder is unchanged; its 1.24 s on 24 threads comes from one slow
iteration of that run, 8.0 s after 6.2 and 6.4.)

With the lowest 160 virtual spinors (26 + 160, same job), whole iteration
and its three stages that changed:

| Stage | | 8 threads | 16 | 24 | 48 |
|---|---|---|---|---|---|
| whole iteration | before | 39.7 (5.4) | 30.1 (8.3) | 28.7 (11.1) | 37.5 (17.6) |
| | after | 28.8 (7.1) | 17.8 (13.1) | 15.5 (18.1) | 16.3 (29.3) |
| pass 2 | before | 15.4 (5.8) | 11.5 (9.0) | 11.3 (11.7) | 18.2 (17.3) |
| | after | 11.5 (7.5) | 6.7 (14.3) | 5.3 (20.4) | 5.5 (31.6) |
| intermediates | before | 9.30 (4.2) | 7.75 (6.0) | 7.17 (8.1) | 8.01 (14.9) |
| | after | 4.86 (6.8) | 3.16 (12.3) | 2.80 (16.8) | 3.06 (28.7) |
| T1 dressing | before | 2.76 (2.2) | 2.63 (3.6) | 2.87 (6.4) | 2.97 (9.1) |
| | after | 1.11 (4.6) | 1.03 (8.1) | 1.23 (13.6) | 1.28 (20.0) |

The parts of pass 2 after the change, on 24 / 48 threads: ring products
0.98 / 0.78 s (23 / 44 cores busy), reduction 0.59 / 0.75, gathers
0.29 / 0.34, drive terms 0.29 / 0.37. Of pass 1: products 0.33 / 0.45
(13 / 18 cores busy), gathers 0.25 / 0.27, accumulation 0.16 / 0.18 on two
cores. The DIIS step of these runs still had its serial parts (unpacking
T2 0.15 s, the first norm 0.11 to 0.16 s averaged over the iterations),
which fb95bd3 gave to the threads; that is not in the table.

Dense layout (same job; the passes and the pp ladder are unchanged, and
their times on 48 threads differ by a quarter between two runs of the same
code - 36.6 and 46.4 s for pass 2 in job 31780775):

| Stage | | 24 threads | 48 |
|---|---|---|---|
| hh ladder + F_oo | before (job 31780775, c197) | 2.39 | 1.57 |
| | after (same job) | 1.38 | 0.92 |
| intermediates | before | 4.53 (6.7) | 3.85 (12.2) |
| | after | 1.16 (15.4) | 0.92 (24.7) |
| T1 dressing | before | 1.40 (7.9) | 1.32 (14.4) |
| | after | 0.35 (15.1) | 0.39 (19.5) |
| whole iteration | before | 94.1 | 85.9 |
| | after | 87.9 | 86.5 |

What the dense iteration spends on 48 threads (job 31781047): the ring
GEMMs of pass 2 30.8 s of its 40 (one product of a 3120 x 3120 operator
with a 3120 x 7140 block per occupied pair: 30 Gflop/s per core where 16
threads reach 60, so it is the product itself, not the gathers around it,
that stops scaling), reduction 5.3, gathers 2.6, drive 1.6; pass 1: its
two stacked GEMMs 8.8 s for about 4 10^12 flops (a third of the rate of
the ring GEMMs: one of them sums over 185640 elements for a 360 x 146
result), gathers 2.8, matrix-vector passes 2.5. The DIIS step of a dense
iteration (6.4 to 6.7 s there) was 1.7 s the norm of the first iteration on
one core, averaged over four iterations (threaded since fb95bd3), 1.5 s
the error vector and about 3 s the copy of the new T3 into the vector
storage: both write a T3-sized array into pages of the scratch file that
did not exist before, which does not get faster with more threads (the step
took 5.1, 4.7, 4.8 and 6.4 s on 8, 16, 24 and 48 threads before) and
ends when the subspace is full (measured apart, vectors of 1 GB, 8 threads:
0.55 to 0.93 s per update while the subspace fills against 0.14 to 0.38 s
in memory, 0.38 s for both afterwards).

**The DIIS vectors in the scratch file slow the other stages of a dense
iteration down.** The same job ran the dense case once more with the
vectors in memory (`incore_complete`; `BP_INCORE=1` of the benchmark):

| | 24 threads | 48 |
|---|---|---|
| whole iteration, vectors in the scratch file | 87.9 | 86.5 |
| whole iteration, vectors in memory | 76.4 | 53.9 |
| pass 2, file / memory | 42.7 / 37.9 | 46.4 / 26.9 |
| DIIS, file / memory | 4.75 / 1.17 | 6.65 / 1.65 |
| peak memory / GB after four iterations, file / memory | 22.1 / 42.5 | 22.6 / 43.0 |

With the file the iterations of the 48-thread runs took 65.6, 97.6 and
96.3 s (69.6, 100.7, 87.5 before the changes): the second is fast, the
later ones are not. In memory: 54.4, 53.5, 53.9 s. So most of what looked
like a dense pass 2 that does not scale from 24 to 48 cores is an effect
of the file-backed vectors on everything that runs after them - 11.6 GB of
new page-cache pages per iteration while the subspace fills. Why (the
buffers of the passes no longer getting huge pages once the page cache has
taken the free memory apart is the guess) was not looked into, and neither
was a run long enough to fill the subspace. The default is unchanged: the
vectors go to the scratch file first (`zccsdt.disk.plan_diis`), and eight
of them with their error vectors are 93 GB here. In label blocks the
effect is small (26 + 120, same commit, job 31781047: 6.84 / 6.64 s per
iteration on 24 / 48 threads with the file, 5.85 / 6.20 in memory; pass 2
the same).

## 3. Memory, and when it is short

With everything in memory an iteration of the spin-orbital / spinor code
takes, with f = o v^3 elements (8 bytes real, 16 complex):

| Part | Elements | |
|---|---|---|
| T3 and its residual | 2 C(o,3) C(v,3) | fully packed |
| pp operator | 2 n_Q^2 ~ v^4 / 2 | the bare packed `<ab\|\|de>` and its dressed form (the T3 ladder operator is made from the dressed form in place) |
| ovvv-sized arrays | at most 3.5 f at a time | packed bare ovvv (f / 2, kept), dressed ovvv (f), pass-1 operand and result, `W_vvvo` (f / 2 each); 1.75 f while the batch buffers are alive |
| batch buffers | 2/3 `mem_mb` | half-expanded T3 of a batch of occupied pairs (pass 1, pass 2) or triples (ladder) and its product |
| DIIS | 2 `diis_space` T3-sized vectors | in the scratch file by default |
| overhead | 0.3 GB + 0.05 GB per thread | buffers of the BLAS threads, allocator |

The integral object holds the full ovvv block (f) besides.  The ovvv-sized
arrays are dropped as soon as they have been used, because they are what a
run needs at least: the dressed vovv is never formed (it is minus the bra
transpose of the dressed ovvv), the pass-1 operand goes when pass 1 has
returned and its result as soon as it has been added to `W_vvvo` (first of
all terms), the `t2 x ovvv` term of `W_vvvo` runs in batches of the first
virtual index on ovvv as stored, and pass 2 gets its drive operand written
directly.

`solver.Options(max_memory=)` (MB for the whole process, as in PySCF) plans
the rest (`ccsdt_opt.plan_memory`, `MemoryPlan`; the plan used is
`Result.plan`).  The default budget is 80% of what the node and the memory
cgroups of the process leave (`zccsdt/memory.py`), so the limit of a Slurm
job or step is seen.  If the sum above fits, nothing changes.  Otherwise:

1. The batch buffers shrink (`mem_mb`, down to one occupied pair).
2. **The pp operator is sliced** (`_pp_sweep`).  The bare packed
   `<ab||de>` is a mapped array of a scratch file: `eris.from_scf(...,
   packed_vvvv=True, max_memory=)` builds it there when it does not fit in
   memory twice (for spinors from the rows that PySCF's out-of-core
   transformation writes for a few ranges of the first virtual index: two
   passes over the AO integrals, v^3 elements in memory), and
   `place_vvvv` moves it there if the plan of the iterations says so.  Its
   dressed rows are produced `pp_bytes` at a time (`dressed_vvvv_rows`: one
   pass of a C kernel for the two one-index terms, one GEMM for the pair
   term).  Each slab feeds `sum_f ~<ab||ef> t1[i,f]` of `W_vvvo`, the CCSD
   ladder and, turned into rows of the ladder operator, the T3 ladder
   (`pp_ladder_rows`), before the next one is made: the dressed operator
   is never held as a whole.  The reduction of a slab adds, to each packed
   element of the residual, the one to three terms whose pair lies in the
   slab (`zc_reduce_pair_first_rows`).
3. **T3 goes to the scratch file, then its residual**, as mapped arrays like
   the DIIS vectors, when they do not fit next to an eighth of the dressed
   operator.  T3 goes first and the residual stays in memory if it fits
   alone (`MemoryPlan.r3_file = False`): the passes only read T3, so its
   pages can be given up at no cost, while pages of the residual, to which
   every pass adds, would have to be written to the file and read back.  If
   the residual does not fit either, it is accumulated directly in the
   storage of the next DIIS vector, the Jacobi update is written over it,
   and the DIIS extrapolation is written over the old T3: no T3-sized array
   exists in ordinary memory.  The occupied operator (hh ladder and F_oo) is
   applied in blocks of columns, so that one GEMM touches a block of T3 and
   of the residual instead of all of both.
4. With T3 in the file the sliced ladder runs batch-wise: for each batch of
   occupied triples all slabs, so that T3 and the residual are passed over
   once.  The rows of the ladder operator are then kept in a second scratch
   array by the first sweep and read back for the other batches.  That pass
   over the operator file is what a batch costs, so the memory that is left
   goes to the batch buffers (up to three times `mem_mb`) and the slabs stay
   thin.  With T3 in memory the ladder runs slab-wise (T3 is gathered again
   for each slab, which is cheap) unless the operator is small.
5. If the ovvv-sized part and the smallest buffers do not fit,
   `MemoryError` is raised before the iterations start.

The operator, which an iteration passes over once, goes out before T3, which
every pass reads.  `incore_complete=True` keeps everything in memory whatever
the budget, and `plan=MemoryPlan(...)` sets the plan explicitly.

Mapped arrays and the operating system (`zccsdt/scratch.py`): by default
paging is left to the kernel.  With `ZCCSDT_IO=1`, and a plan that has put
anything into a scratch file, the kernels announce the rows they are about
to read (`madvise(WILLNEED)`), drop the rows of T3 they have read from the
page tables (`madvise(DONTNEED)`: the data stays in the page cache while
there is room and is cheap to take back when there is not), and start
writing data that is final for the iteration - the new T3, the DIIS
vectors - as soon as it is written (`sync_file_range`).  On a local NVMe
that the job has to itself these hints changed the iteration time by -3% to
+7% (`docs/benchmarks.md`), which is why they are optional; writing the
residual out after every pass, or unmapping it, cost 6-40% and is not done.

U / R-form integrals (`eris_spin`: UCCSDT and the spin-flip RCCSDT) are
handled the same way with the spin-blocked T3 engine: T3 and its residual are
mapped (stored spin blocks), `eris_spin.from_scf(..., max_memory=)` builds
the alpha-alpha, alpha-beta and beta-beta pair blocks of the packed vvvv in
batches into a scratch file, and `_pp_sweep_spin` dresses and applies each
block in slabs of rows (`spin_imds.dressed_vvvv_rows`,
`SpinBlockT3.pp_ladder_rows`, `zc_breduce_ppl_rows`).  For RHF the
alpha-alpha rows serve the beta-beta block as well, as in memory.  Batch-wise,
a batch of triples reads back only the pair blocks its spin group contracts
with (alpha-alpha-alpha: the alpha-alpha block alone).

The memory of this engine differs from the table in three ways, which the
engine tells the planner (`buffer_bytes`, `kept_buffers`, `ovvv`,
`ovvv_cached`; f still counts spin orbitals):

- **The batch buffers are one work array** that stays allocated between the
  passes and the iterations (`t3engine.workspace`; pass 1 takes half of
  `mem_mb` from it, the ladder by rows up to two thirds, pass 2 a third of f
  per occupied pair).  It therefore counts next to every stage of the
  iteration, not only next to the passes, and `plan_memory` asks the engine
  for its exact size.  It is released when the buffer budget changes; there
  is one engine per set of orbital maps whatever the budget.
- **The ovvv-sized arrays** are spin-blocked where the algebra is (dressed
  ovvv 3/8 f, packed 3/16 f) and dense where the engine takes dense operands
  (pass-1 operand and result, `W_vvvo`: f / 2 each): 2 f up to the end of
  pass 1 and 1.25 f afterwards.
- **A run that is not short of memory keeps one more f** between the
  iterations: the bare `<ab||ei>`, from which the bra-dressed block starts,
  and the array of `W_vvvo`.  A plan with anything in a scratch file keeps
  neither (the bare block is then written again in every iteration).

Any buffer size works (one pair or one triple per batch), so the least this
engine needs is 2 f plus a third of f for the work array plus the overhead.

Limits of this version:

- **Dense integrals on spin-grouped maps** (`eris.from_scf(...,
  spin_order=True)`, a validation path): T3 can be mapped, the pp operator
  stays in memory.
- **The ovvv-sized arrays** (3.5 f, plus the full ovvv block of the integral
  object) must fit: 0.7 GB for X2C H2O/cc-pVTZ (o = 10, v = 106), 6 GB for
  o = 10 and v = 220, 30 GB for o = 20 and v = 300 (complex; plus the
  overhead of the table).  With U / R-form integrals it is 2.3 f of real
  numbers: 1.8 GB for NH2/cc-pVQZ (o = 9, v = 221).

Timings under memory limits are in `docs/benchmarks.md`.

## 4. Open items

- **Intermediates without the symmetry labels.** The CCSD part and the
  intermediates (2.6) work on dense integrals and a dense T2 whatever the
  layout of T3. In label blocks the o^2 v^4 product of W_vvvo and the
  (ov)^3 products are then a tenth of an iteration that the labels could
  cut by the number of classes.
- **The Lambda equations and the density** still build their small
  intermediates with einsum (`lambda_opt`, timer `small`; only the hh
  operator was changed there).
- **Pass 1 matrix-vector passes.** r1 and rA could be fused into the C
  gather.
- **Memory.** Without `packed_vvvv=True` the integral container holds the
  full vvvv (v^4); the packed build keeps only `<ab||cd>` with a<b, c<d.
  The full ovvv block (o v^3) is still held next to its packed form.
- **Real/complex.** Both are covered by the same code. Complex runs about
  4x the real flops, as expected.
- **Memory floor of the U/R form.** The pass-1 operand and result and
  `W_vvvo` are still dense arrays (f / 2 each) of which spin conservation
  leaves 3/8; with spin-blocked operands the 2 f of this form would be
  about 1.5 f.
- **Both T3 arrays in the file.** Each amplitude is then read, and each
  element of the residual read and written, once per pass that touches it
  (pass 1, pass 2, the ladders, the update): the iteration is I/O-bound.
  Fusing passes over batches of occupied triples would cut that; so would a
  DIIS that reads its stored vectors once per iteration instead of twice
  (the error vectors for the overlaps, the amplitude vectors for the
  extrapolation).
