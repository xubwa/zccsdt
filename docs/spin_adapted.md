# Closed-shell spin-adapted CCSDT (`zccsdt.sa`)

Nonrelativistic closed-shell CCSDT on an RHF reference, with spatial orbitals
and spatial integrals throughout: fully spin-adapted amplitudes, no spin
orbitals at run time.  This is the RCCSDT path; the spin-orbital/spinor code
(`zccsdt.cc`) stays the general one (X2C, open shells via UCCSDT).

## Formalism

Non-orthogonal spin adaptation (NOSA).  The spatial amplitudes t_ijk^abc are
the coefficients of E_ai E_bj E_ck in T3.  They are invariant under
simultaneous permutations of the pairs (ia), (jb), (kc), and they relate to the
spin-orbital amplitudes by

    t(ijk, abc) = sum_pi sgn(pi) t_ijk^{pi(abc)} prod_n delta(s_occ(n), s_vir(pi(n)))

(for example t(IJk, ABc) = t_ijk^abc - t_jik^abc in the alpha-alpha-beta
block).  The spatial residual Omega_ijk^abc is the coefficient of the identity
spin pairing (lines i-a, j-b, k-c) in the spin-orbital residual.  T1 and T2 are
the usual closed-shell t_i^a and t_ij^ab = t_ji^ba.

The component of t_ijk^abc that is constant over the permutations of abc (for
fixed ijk) does not enter the wavefunction.  It is projected out of every
residual, r <- r - (1/6) sum_sigma r^{sigma(abc)}, so that the redundant
directions never grow; this also removes t_iii^abc and t_ijk^aaa.

The T1-dressed formalism is used as in the spin-orbital code: bra virtual
slots of <pq|rs> transform with X = 1 - t1^T, ket occupied slots with
Y = 1 + t1, and T1 does not appear in the T2 and T3 equations otherwise.

## Equations

The spatial equations are derived mechanically from the spin-orbital ones,
not by hand:

- `zccsdt.sa.spinfree` expands every spin-orbital factor into spatial
  components with spin deltas, traces the spin lines through each product
  (each closed loop of internal indices gives a factor 2), keeps the products
  whose lines realize the identity pairing, and merges identical terms.
- `zccsdt.sa.derive` applies it to the T1-dressed spin-orbital CCSDT
  equations of `zccsdt.cc.ccsdt` (intermediates Wv, Wo, Fv, Fo, Wr, Wpp, Whh;
  every 4-index intermediate is spin-free with J and K components).
- `zccsdt.sa.somap` maps spatial tensors back to spin orbitals; the tests
  check the derived intermediates and residuals against the spin-orbital
  expressions and against `ccsdt._t3_residual` on random spin-adapted data
  (`tests/test_sa_derive.py`).

The resulting T3 residual has the form

    Omega = P[D + E] + sum_positions X_ring + sum_pair_positions (pp + hh ladder)

with P the six-term pair symmetrizer and

    D_ijk^abc = sum_e Wv_bcek t_ij^ae,     E_ijk^abc = -sum_m Wo_maji t_mk^bc,
    X_{i;jk}^{a;bc} = sum_md J_madi t~_mjk^dbc - K_madi t_mjk^dbc
                      - K_mbdi t_mjk^adc - K_mcdi t_mjk^abd,
    t~_mjk^dbc = 2 t_mjk^dbc - t_mjk^bdc - t_mjk^cbd,
    pp: sum_de W_ab^de t_ijk^dec,   hh: sum_mn W_mn^ij t_mnk^abc,

where J, K are the components of the ring intermediate Wr.  F_vv is part
of the attached ring term and F_oo of the hh operator.

## Code

| Module | Content |
|---|---|
| `sa/rccsdt_ref.py` | Dense reference: all residuals evaluated term by term from the derived equations on full spatial arrays (validation, small systems). `RHFIntegrals` holds the MO Fock matrix and either the full `V` or the MO coefficients (blocks transformed on demand). |
| `sa/t3blocks.py` | T3 storage: one v^3 cube per sorted occupied triple i <= j <= k (not all equal), cube views of any ordered triple, placement rules, null projection, Jacobi step, tiled gather and scatter helpers. |
| `sa/rccsdt.py` | The block engine (`kernel`). |
| `sa/rccsdt_q.py` | The (Q) correction (`kernel`; `rccsdt.run(mf, q=True)`). |

### Storage and placement

A block stores t_{s0 s1 s2}^{xyz} for its sorted triple; any ordered triple is
a block plus an axis permutation.  Residual contributions come as

- **attached** terms X_{r;pq}^{x;yz} (p <= q, symmetric under
  (p,y) <-> (q,z)), summed over the three positions of a block, and
- **ordered** terms G_t of the six-term symmetrizer.

`Blocks.uses_attached` / `uses_ordered` list the (block, axes) placements so
that blocks with a repeated occupied index receive each contribution once.
The ordered term P[D + E] is rewritten in attached form,
X_{k;ij}^{x;yz} = G_{ijk}^{yzx} + G_{jik}^{zyx}, so all T3 contributions are
attached or block-local.

### One iteration (`residuals`)

1. **Dressing.**  Each T1-dressed block is its bare block plus t1
   corrections from blocks with one slot switched (bra virtual -> occupied,
   ket occupied -> virtual), built from cached contiguous bare blocks.
   <am|fe> and <am|ji> follow from ovvv and ovoo by pair symmetry.
2. **pp operator and block pass.**  W_ab^de = ~<ab|de> + sum_mn tau_mn^ab
   <mn|de> as its (d,e)-symmetric and antisymmetric parts packed over
   a <= b, d <= e (a < b, d < e), in one pass over the packed bare integrals;
   all rows at once, or in slabs of a when memory is short (below).  The
   rows at hand feed
   - the vvvo intermediate sum_f W_ab^ef t1_i^f and the CCSD ladder (tau,
     so the t1 t1 vvvv term rides along);
   - the T3 ladder (`ladder_pp`): for every block, its three pair views are
     packed into symmetric / antisymmetric pair matrices and multiplied by
     the packed W.  This takes half the arithmetic of a full v^2 x v^2 GEMM.
     Blocks with a repeated index need only the symmetric part of one view,
     and one of the other two views follows from the other by symmetry.
3. **Intermediates and CCSD part.**  Wv is built directly in the layout of the
   drive GEMM.
4. **Pair pass** (`t3_pass`), one loop over occupied pairs p <= q.  The cubes
   t_mpq of all m are gathered once into five layouts with the
   contraction index (x, m) leading: Y = t~, T, T2 = t^{yxz}, T3 = t^{yzx},
   Z = Y with y <-> z.  From them:
   - the ring X_{i;pq} for all i as three GEMMs (F_vv rides in the K
     operator), scattered into the blocks;
   - T3 -> R1, R2, Wv, Wo as two stacked GEMMs and one product each for the
     ovvv / F_ov and Wo terms of both pair orders.
5. **k pass** (`drive_hh`), one loop over k: the drive P[D + E] in attached
   form (one GEMM over all pairs) and the hh ladder (one GEMM over the gathered
   t_mnk), with one scatter of their sum; virtual slabs bound the memory.
6. Null projection; Jacobi step with the DIIS vectors in one pass (the DIIS
   metric weights a block by the number of ordered triples it stands for).
   The new vector and its error vector are written directly into the DIIS
   storage.

### Memory, and when it is short

With everything in memory an iteration takes (8-byte elements, n = o + v):

| Part | Elements | |
|---|---|---|
| T3 and its residual | 2 o^3 v^3 / 6 | |
| pp operator | v^4 | the bare packed `<vv\|vv>` and its dressed form, v^4 / 2 each |
| ovvv-sized arrays | about 23 o v^3 | integral blocks, their dressed forms, intermediates, the buffers of the pair pass |
| k pass | 2.6 `slab_bytes` | slabs of the virtual index of k |
| pp ladder | 6 `batch` v^3 | packed views of `batch` blocks and their products |
| DIIS | 2 `diis_space` T3-sized vectors | in the scratch file by default |

The DIIS vectors are stored as in PySCF: in an HDF5 scratch file unless
`incore_complete=True` (`diis_file` names the file; the default is a
temporary file in `ZCCSDT_TMPDIR` or PySCF's `lib.param.TMPDIR`).  The file
is memory-mapped (`zccsdt/scratch.py`): with enough free memory the vectors
stay in the page cache and the DIIS step runs as fast as in memory, and
under a memory limit the kernel pages them.

`kernel(..., max_memory=)` (MB, as in PySCF) plans the rest (`plan_memory`,
`MemoryPlan`).  The default budget is 80% of what the node and the memory
cgroups of the process leave (`zccsdt/memory.py`), so the limit of a Slurm
job or step is seen.  If the sum above fits, nothing changes.  Otherwise:

1. The work buffers of the k pass and of the ladder shrink.
2. **The pp operator is sliced.**  The bare packed `<vv|vv>` goes to the
   scratch file, built from PySCF's out-of-core transformation with the rows
   of one first index at a time (`vvvv_row_blocks`).  If its dressed form
   does not fit either, the dressed rows are produced in slabs of the first
   virtual index (`pp_slabs`); each slab feeds the vvvo intermediate, the
   CCSD ladder and the T3 ladder of all blocks before the next one is made,
   so the dressed operator is never stored.
3. **T3 and its residual go to the scratch file**, as mapped arrays like the
   DIIS vectors, when they do not fit next to a few rows of the operator.
   The passes work on them unchanged and the operating system keeps as much
   of them in memory as there is room for.  With T3 in the file the sliced
   ladder runs batch-wise (all slabs for one batch of blocks), so that T3 and
   the residual are passed over once.
4. If the ovvv-sized part alone does not fit, `MemoryError` is raised before
   anything is allocated.

The operator, which an iteration passes over once, goes out before T3, which
every pass reads.  `incore_complete=True` keeps everything in memory whatever
the budget, and `plan=MemoryPlan(...)` sets the plan explicitly.  Timings
under memory limits are in `docs/benchmarks.md`.

### Leading arithmetic per iteration (FMA, o / v spatial)

| Term | Spin-adapted engine | Note |
|---|---|---|
| pp ladder | ~ o^3 v^5 / 4 (+ 0.75 o^2 v^5) | symmetric / antisymmetric packing, repeated-index savings |
| ring | 2 o^4 v^4 | 4 o^2 v^4 per pair p <= q |
| T3 -> R1/R2/Wv/Wo | ~ 3 o^3 v^4 | pair pass |
| drive (D + E) | o^3 v^4 | k pass |

`bench/flop_counts.py` compares the ladder + ring arithmetic of the
spin-orbital, Sz-blocked, spin-orbital RCCSDT (spin flip) and spin-adapted
representations with a model of CFOUR `ncc` counted from its source.  The
spin-adapted engine needs 0.31-0.43 of the `ncc` model (H2O, N2, H2S cc-pVTZ;
H2O cc-pVQZ); the spin-orbital RCCSDT needs 0.57-0.85.  That model
overstates `ncc` about twofold: by `ncc`'s own flop counter the engine does
0.6-0.9 of its GEMM arithmetic (`docs/benchmarks.md`, "Where the difference
comes from").

## CCSDT(Q) (`sa/rccsdt_q.py`)

The perturbative quadruples correction of Bomble, Stanton, Kallay and Gauss
for an RHF reference with canonical orbitals: with converged T2 and T3 and
bare integrals W (T1 does not enter),

    r(ijkl,abcd) = <ijkl abcd| [W, T3] + 1/2 [[W, T2], T2] |0>,
    z(ijkl,abcd) = <0| (T2^+ + T3^+) W |ijkl abcd>,
    E(Q) = 1/(4!)^2 sum z r / (e_i + e_j + e_k + e_l - e_a - e_b - e_c - e_d).

The T2^+ part of z is disconnected and of fifth order; alone it gives the
[Q] correction.

### Equations

- **Spin-orbital terms** (`derive.q_terms`): eight products, each under the
  full antisymmetrizer of ijkl and abcd.  Their signs and factors were not
  taken from diagram rules: the operators were built as Jordan-Wigner
  matrices in a small Fock space (`tests/fockspace.py`), r and z were
  evaluated from the commutators above for random amplitudes and integrals,
  and the coefficients were fitted (exactly 1/24, -1/24, 1/16, 1/16, 1/4 for
  r and 1/16, 1/24, -1/24 for z).  The test repeats this comparison.
- **Spatial form** (`derive.q_spatial`): spin-line tracing gives the
  identity-pairing coefficients r_ijkl^abcd and z_ijkl^abcd (pairs i-a, j-b,
  k-c, l-d), invariant under simultaneous permutations of the four pairs.
  All coefficients are +1 or -1: 24 terms from T3 (with `<vv|vo>` and
  `<ov|oo>`), 72 from T2 T2 (12 `<vv|vv>`, 12 `<oo|oo>`, 48 ring), 6 + 24
  for z.  For real orbitals the T3 part of z equals the T3 part of r.
- **Energy.**  Summing the spin-orbital product over spins gives

      E = 1/24 sum_{ijkl,abcd} z_ijkl^abcd / D sum_tau w(tau) r_ijkl^{tau(abcd)},
      w(tau) = sgn(tau) 2^cycles(tau)

  (16 for the identity, -8 for transpositions, 4 for 3-cycles and double
  transpositions, -2 for 4-cycles).  The operator M = sum_tau w(tau) P_tau
  factorizes into Jucys-Murphy elements,

      M = 2 (2 - (01)) (2 - (02) - (12)) (2 - (03) - (13) - (23)),

  so it is applied with 9 terms instead of 24.

### Per-quadruple form

Only sorted occupied quadruples i <= j <= k <= l with no index three times
are needed, weighted by their number of orderings.  With the four pairs
P_n = (o_n, x_n) of a quadruple, r = A + B and z = A + C are sums over
arrangements (p, q, r, s) of the pairs:

| Part | Term | Count | GEMM shape |
|---|---|---|---|
| A | + sum_e t_{o_p o_q o_r}^{x_p x_q e} <x_r x_s\|e o_s> | 12 | (x_p x_q) x e x (x_r x_s) |
| A | - sum_m t_{o_p o_q m}^{x_p x_q x_r} <m x_s\|o_r o_s> | 12 | (x_p x_q x_r) x m x x_s |
| B | + sum_f I_{pq}[x_q, x_p, x_r; f] t_{o_r o_s}^{f x_s} | 12 | (x_q x_p x_r) x f x x_s |
| B | + sum_m t_{o_p m}^{x_p x_q} G(o_q, o_r, o_s)[m; x_r, x_s] | 24 | (x_p x_q) x m x (x_r x_s) |
| C | t_{o_p o_q}^{x_p x_q} <o_r o_s\|x_r x_s> | 6 | outer product |

I_{pq}[x_q, x_p, y; f] = sum_e t_{o_p o_q}^{e x_q} <x_p y|e f> is an o^2 v^4
intermediate (the `<vv|vv>` terms of T2 T2); G collects the `<oo|oo>` and
ring integrals with one T2 into an o^4 v^2 array, so all T2 T2 terms with
those integrals cost only o v^4 per quadruple.

### Algorithm (`q_correction`)

The outputs of the GEMMs are v^4 arrays and a pass over one of them costs
about as much as its GEMM, so nothing is transposed:

- **Cyclic layouts.**  Each term is stored as [x_w, x_x, x_y, x_3] for one
  of the three rotations (w, x, y) of the first three pairs.  These three
  layouts host every term as a GEMM output: a 2|2 product for the pair
  partition {wx|y3}, a 3|1 product when pair 3 is the special one, a 1|3
  product when pair w is.  (A search over all layout triples found this
  family; it also lets one array per intermediate serve both of its uses.)
- **Stacked products.**  The four T3 x `<vv|vo>` terms of a rotation are one
  [(x_w x_x), 4v] x [4v, (x_y x_3)] product: T3 cubes and integral slices are
  copied side by side along the contraction index (v^3 copies).  The
  o-contracted terms are stacked in the same way (K = 3o or 4o).
- **Symmetrized intermediates.**  The two orders of a pair enter with the
  same T2 factor, so Ibar_c = I_wx + I_xw is formed once and the 12 I x T2
  terms become 6 products.  Ibar is kept for the pairs (i,j), (j,k), (k,i)
  while l runs; with l innermost these are all the pairs that do not
  contain l.
- **Energy kernel** (`zc_sa_q_energy`).  One pass over the six accumulators
  (three for A, three for B).  The work unit is a sorted quadruple of tiles
  (6 indices each) with its 24 permuted tiles; each tile is loaded into a
  local buffer indexed by the base element, so the 24 permutations of an
  element sit at one offset of 24 buffers and M and the energy are
  unit-stride row operations.  C is formed on the fly from v x v matrices.
  The rows of a tile are scattered over the accumulators, so they are
  prefetched before use; coinciding tiles are copied in cache.
- **Repeated occupied index.**  If two of the first three pairs have the same
  occupied index (i = j or j = k; the pairs are relabeled so that these are
  pairs 0 and 1), r and z are invariant under the exchange of the two pairs,
  and the exchange maps every term to a term.  One term of each mirror pair
  is accumulated (terms that are their own image at half weight), rotation 2
  is not computed at all (it is the image of rotation 1), and the kernel adds
  the image of its tile buffers in cache: 7 + 4 instead of 12 + 6 products
  of v^5.  Quadruples with k = l only are computed in full (the exchange
  would move the contiguous index of the accumulators).
- **BLAS.**  All products, including the small ones for G, go through MKL
  (`zccsdt.blas.gemm`): calls into numpy's own BLAS between the threaded
  GEMMs made those twice as slow.

Leading cost per quadruple: 12 + 6 products of v^5 FMA (7 + 4 with a
repeated index among the first three), at most 0.75 o^4 v^5 in total, against
o^3 v^5 / 4 + 2 o^4 v^4 for a CCSDT iteration.  Memory: 10 v^4 (6
accumulators, 3 intermediates, the `<vv|vv>` integrals) besides T3.

### When 10 v^4 does not fit (`q_correction_slabs`)

`rccsdt_q.kernel` takes the block algorithm if its work arrays fit in the
memory budget (`max_memory` as for the iterations) and otherwise cuts them
into slabs of six virtual indices, one tile of the energy kernel:

- **Slabs of the largest index.**  The energy is a sum over sorted virtual
  quadruples, each of which needs r and a at its 24 permutations.  With the
  quadruples grouped by the slab [d0, d1) of their largest index, a slab
  needs r only where one index is in the slab and the other three are below
  d1: four *faces* per occupied quadruple, one for each pair that can carry
  the slab index.  Face p is what the block algorithm computes for the
  quadruple with pair p moved to the end, with the last index restricted to
  the slab and the others to [0, d1): six arrays (d1, d1, d1, 6) instead of
  six v^4 arrays.  Summed over the slabs every element of r is computed
  once.  Equal occupied indices give equal faces, which are computed once.
- **Kernel** (`zc_sa_q_energy_faces`): the tile quadruples whose largest
  tile is in the slab; a permuted tile is read from the face of the position
  at which the slab tile sits.  M r and the energy are shared with the full
  kernel.
- **`<vv|vv>` terms.**  Restricted to a slab the I x T2 terms would still
  need the v^4 intermediate I, so the two contractions are done in the other
  order.  For every slab and ordered occupied pair (a, b)

      N[y, d, (e x)] = sum_f (y f|e x) t_ab^{f d},   y < d1, d in the slab,
      M[d, w, (e x)] = sum_f (d f|e x) t_ab^{f w},   d in the slab, w < d1,

  packed over e >= x, are formed in one pass over the rows E[(y f), (e x)] of
  the integrals (`rccsdt.vvvv_rows`: 4 v^4 bytes, mapped from the scratch
  file when they do not fit).  A face contracts them with the other T2 of
  each term; where the result would land transposed in its first two
  indices, the product is done slice by slice (`blas.gemm_slices`).
- **Memory** per slab index: 24 v^3 (faces), o^2 v^3 (N and M) and 2 v^3.
  H2O/cc-pVQZ: 3.6 GB instead of 11.7 GB; v = 282 with o = 5: 60 GB instead
  of 506 GB.  If that does not fit, `MemoryError` is raised.
- **Time**: about twice the block algorithm.  The 12 I x T2 terms take four
  restricted products per face and layout, twice the arithmetic of the six
  symmetrized products, and quadruples with a repeated index are not
  shortened.  N and M are unpacked to dense arrays for every use unless
  memory is left to keep the dense forms of the pairs used last.

## Validation

- `tests/test_sa_derive.py`: derived spatial equations = spin-orbital
  equations (intermediates, R1, R2, R3, `ccsdt._t3_residual`).
- `tests/test_sa_rccsdt_ref.py`: dressing, null projection, RCCSD vs PySCF,
  RCCSDT vs the spin-orbital code.
- `tests/test_sa_rccsdt.py`: block engine residuals and energy vs the
  reference, C kernels vs numpy, the merged drive / hh pass vs the separate
  terms, integral blocks from AO integrals vs the full V.
- `tests/test_references.py`: converged energies vs MRCC / CFOUR (H2O
  6-31G and cc-pVDZ fc, HF cc-pVDZ, N2 6-31G) to better than 1e-10 Eh.
- `tests/test_sa_outcore.py`: the sliced pp operator (both loop orders, in
  memory and in the scratch file), mapped T3, the out-of-core `<vv|vv>`
  build and whole calculations planned from `max_memory` against the run in
  memory; `tests/test_memory.py`: the cgroup-aware budget, scratch arrays.
- `tests/test_sa_q.py`: (Q) spin-orbital terms vs the Fock-space evaluation,
  the energy metric vs the spin-orbital sum, per-quadruple arrays vs the
  dense term-by-term evaluation, the block algorithm and its C kernel vs the
  plain versions, the sliced algorithm vs the block algorithm.  `tests/test_references.py`: CCSDT[Q] and CCSDT(Q) vs MRCC
  (CFOUR ncc gives the same (Q) to 3e-12 Eh).  With the block algorithm the
  CCSDT(Q) total energies of all five reference systems agree with MRCC to
  8e-11 Eh, the corrections themselves to 1e-11 Eh.

Timings against CFOUR `ncc` are in `docs/benchmarks.md`.
