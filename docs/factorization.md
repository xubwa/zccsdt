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
| hh ladder + F_oo | one operator O6[I,J] on packed T3: `R += O6 @ T3p` | C(o,3)^2 C(v,3) ~ o^6 v^3/216 | o^5 v^3/24 (ratio o/9) |
| pass 1: T3 -> W_vvvo, W_vooo, T1, T2 | two stacked GEMMs per batch of pairs, plus two matrix-vector passes | about o^3 v^4 | same order |
| T1 dressing | slot by slot; packed vvvv dressed directly | o v^4/2 (vvvv) | -- |
| CCSD part (T1/T2 residual) | reference code (Gauss-Stanton intermediates) | o^2 v^4 (ladder) | o^2 v^4/4 with packed pairs (not yet done) |

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
operator on antisymmetric 3-index tensors. Its matrix
`O6[I, J]` (C(o,3) x C(o,3)) is built by applying the operator to the packed
basis (`occ_operator`). The terms then reduce to one GEMM on the packed
T3, `R += O6 @ T3p`, with N = C(v,3).

This does o/9 times the minimal flops (1.1x for o = 10, 1.6x for o = 14),
but needs no gather, no reduction and no transpose. CFOUR evaluates this
term with level-1 DAXPY loops. For o larger than about 25, a pair-driven
GEMM would be cheaper.

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
3. **T3 and its residual go to the scratch file**, as mapped arrays like the
   DIIS vectors, when they do not fit next to an eighth of the dressed
   operator.  The residual is then accumulated directly in the storage of
   the next DIIS vector, the Jacobi update is written over it, and the DIIS
   extrapolation is written over the old T3: no T3-sized array exists in
   ordinary memory.  The occupied operator (hh ladder and F_oo) is applied
   in blocks of columns, so that one GEMM touches a block of T3 and of the
   residual instead of all of both.
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

- **CCSD part.** The T2 ladder could reuse the packed W(ab,de)
  (o^2 v^4/4 instead of o^2 v^4). Some intermediates are still built with
  generic einsum.
- **Pass 1 matrix-vector passes.** r1 and rA could be fused into the C
  gather.
- **Memory.** Without `packed_vvvv=True` the integral container holds the
  full vvvv (v^4); the packed build keeps only `<ab||cd>` with a<b, c<d.
  The full ovvv block (o v^3) is still held next to its packed form.
- **Real/complex.** Both are covered by the same code. Complex runs about
  4x the real flops, as expected.
