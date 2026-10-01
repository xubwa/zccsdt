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
2. **pp operator.**  W_ab^de = ~<ab|de> + sum_mn tau_mn^ab <mn|de> as its
   (d,e)-symmetric and antisymmetric parts packed over a <= b, d <= e
   (a < b, d < e), in one pass over the packed bare integrals.
3. **Intermediates and CCSD part.**  Wv is built directly in the layout of the
   drive GEMM; the t1 t1 vvvv term rides with the CCSD ladder (tau).
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
6. **Block pass** (`ladder_pp`): for every block, its three pair views are
   packed into symmetric / antisymmetric pair matrices and multiplied by the
   packed W.  This takes half the arithmetic of a full v^2 x v^2 GEMM.  Blocks
   with a repeated index need only the symmetric part of one view, and one of
   the other two views follows from the other by symmetry.
7. Null projection; Jacobi step with the DIIS vectors in one pass (the DIIS
   metric weights a block by the number of ordered triples it stands for).
   The new vector and its error vector are written directly into the DIIS
   storage.

### Memory and DIIS storage

In memory: T3 and its residual, the packed pp operator and bare vvvv (v^4
elements together), ovvv-sized blocks, and the work arrays of the passes
(a few T3- or o v^3-sized buffers, bounded by slabs where they would grow
with o^2 v^3).

The DIIS subspace needs 2 x `diis_space` T3-sized vectors, more than
everything else for larger molecules.  As in PySCF they are stored in an HDF5
scratch file unless `incore_complete=True` (`diis_file` names the file; the
default is a temporary file in `ZCCSDT_TMPDIR` or PySCF's `lib.param.TMPDIR`).
The file is memory-mapped: with enough free memory the vectors stay in the
page cache and the DIIS step runs as fast as in memory, and under a memory
limit the kernel pages them (`docs/benchmarks.md`, "DIIS storage").

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
H2O cc-pVQZ); the spin-orbital RCCSDT needs 0.57-0.85.

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

Timings against CFOUR `ncc` are in `docs/benchmarks.md`.
