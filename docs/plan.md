# zccsdt implementation plan (draft for review)

Status: **confirmed (2026-09-30)**.
Decisions: Python implementation first (no C++ until profiling says where it
pays); contraction schemes are hand-optimized term by term; benchmark
substitutes of Section 8 accepted; default git branch (`master`); T1-dressed
Hamiltonian accepted. Order of work: first make the Python code *correct*
(einsum prototype from the generated equations, M2, validated against all
references), and only then optimize (M3-M5).
Background on CFOUR's design is in `docs/cfour_notes.md`.

Notation: `o`, `v` = number of occupied / virtual **spin orbitals (spinors)**.
Flop counts are given in fused multiply-adds (FMA); one complex FMA = 4 real
FMA = 8 real flops.

---

## Starting point (decided 2026-09-30): code ported from socutils

socutils (same author) already contains a validated spin-orbital CCSDT
(`cc/zccsdt.py`, commit ffbe691), which is T1-dressed and checked
element-wise against a determinant-space brute-force oracle. It was ported
here as the starting point; the einsum prototype of M2 is replaced by it.

| zccsdt file | origin | change |
|---|---|---|
| `src/zccsdt/cc/ccsdt.py` | `cc/zccsdt.py` | imports; class wrapper removed |
| `src/zccsdt/cc/ccsd.py` | `update_amps` from `cc/zccsd.py` | imports only |
| `src/zccsdt/cc/gintermediates.py` | `cc/gintermediates.py` (PySCF, Apache-2.0) | none |
| `src/zccsdt/cc/bruteforce.py` | `cc/_ccsdt_bruteforce.py` | none |
| `src/zccsdt/lib/_clib.py`, `csrc/*.c` | `cc/_ccsdt_clib.py`, `lib/ccsdt/*.c` | library path, standalone CMake |
| `src/zccsdt/eris.py`, `src/zccsdt/cc/solver.py` | new | integrals from any GHF/spinor SCF; driver without the PySCF CC classes |

Known limitations of the starting point:
- The integrals are held as a full n^4 tensor `g`, and T1 dressing costs
  n^5 per iteration.
- T3 is unpacked to the full o^3 v^3 array every iteration, and several full
  o^3 v^3 intermediates exist at once.
- Real input is computed in complex arithmetic (C kernels are
  complex-only).

Measured baseline: H2O/cc-pVDZ X2CAMF, about 8 s/iteration, 5.4 GB peak,
48 threads. These limitations are exactly what the optimization milestones
remove.

The `zwick` engine (Section 3) is deferred. The brute-force oracle now
provides the ground truth for every residual; the engine is still wanted
later for zmr.

---

## 0. Environment facts that shape the plan

- Node: 2 x Xeon Gold 6248R (48 cores, AVX-512, 2 FMA units/core), 188 GB RAM.
  Theoretical DP peak is roughly 32 flop/cycle/core; the practical peak will
  be *measured* with a large DGEMM/ZGEMM (milestone M4) and used as the
  denominator for all "% of peak" numbers.
- Toolchain: g++ 9.3 and icpx 2022.1, CMake 3.27, Intel MKL 2022.1 (the same
  MKL CFOUR links against), HDF5 (h5py 3.x, h5cc).
- Python 3.11, numpy 2.4.6 (OpenBLAS), PySCF 2.13.1, pytest.
- Spinor FCI: `socutils-git/fci/zfci.py` (`kernel(h1e, eri, norb, nelec)`,
  pyscf `fci_dhf_slow` conventions); reached via
  `PYTHONPATH=~/apps/socutils-git-env`.
- MRCC: `dmrcc`/`minp` on PATH (`/home/czhan119/code/cfour-git/bin`).
- **cfour-x2cso has no spinor CCSDT** (spinor CC stops at CCSD(T)). CFOUR's
  iterative CCSDT codes (`ncc` RHF, `vcc`/`ecc` UHF) are nonrelativistic and
  real. This affects the benchmark plan; see Section 8 and open question Q1.

---

## 1. Repository layout

```
zccsdt/
  pyproject.toml            # pip install -e . ; builds nothing by itself
  README.md
  docs/
    plan.md  cfour_notes.md  equations.md  factorization.md
    t3_storage.md  data_format.md  benchmarks.md
  src/
    zwick/                  # Wick engine: standalone, no dependency on zccsdt
      operators.py          #   index spaces, second-quantized strings
      contract.py           #   full contractions relative to a reference
      reference.py          #   contraction rules (HF now; pluggable for zmr)
      bch.py                #   e^{-T} H e^{T} expansion, projections
      canonical.py          #   term canonicalization, merging, P(ij) detection
      emit.py               #   emit einsum / term lists (JSON)
    zccsdt/
      io/                   # HDF5 reader/writer, integral generators
        h5format.py  pyscf_ints.py  socutils_ints.py  cfour_ints.py
      equations/            # generated equations (checked-in JSON + generator)
      opt/                  # contraction-order search, CSE, flop model
      proto/                # numpy einsum reference CCSD / CCSDT (correctness only)
      packed/               # packed T3 storage, gather/scatter, sign tables
      terms/                # one module per hand-optimized contraction scheme
      core/                 # optimized driver (packed T3 + BLAS through numpy)
      solver/               # DIIS, convergence, amplitude update
  (csrc/                   # only after profiling, see Section 6)
  tests/                    # pytest; one test module per feature
  bench/                    # benchmark scripts + CFOUR/MRCC input templates
```

`zwick` lives in `src/zwick` with its own tests (`tests/zwick/`) and no import
of `zccsdt`. It can later be moved to its own repository unchanged and shared
with zmr.

---

## 2. Data exchange format (HDF5), `docs/data_format.md`

One file per calculation, written by any front end (PySCF, socutils, CFOUR
converter) and read by zccsdt:

```
/                     attrs: format_version, dtype ("float64"|"complex128"),
                             nocc, nvir, nfrozen, e_nuc, e_core, e_ref,
                             source ("pyscf-ghf", "socutils-x2c", "cfour-x2c", ...)
/mo_energy            (n,)   real
/fock                 (n,n)  MO Fock matrix (active space; allows non-HF/non-canonical)
/eri/oooo /ooov /oovv /ovov /ovvv /vvvv
/eri/ovoo /vvoo /vvov  (bra-ket partners; stored explicitly, no hermiticity assumed)
```
- ERIs are antisymmetrized physicist-notation `<pq||rs>`. Like-space pairs
  (oo, vv) are stored triangle-packed with the strict `p<q` convention
  (`/eri/vvvv` has shape `(v(v-1)/2, v(v-1)/2)`); mixed pairs are stored full.
- Frozen core is applied by the front end (core energy folded into `e_core`,
  Fock dressed), so the CC code only sees the active space.
- CFOUR converter (Python, `scipy.io.FortranFile`): reads the `moin*`,
  `PPPH*`, `scf2ceig`, `scf2cfck` files written by `xx2cvtran` / `xsox2chf`
  and repacks them. It requires `ABCDTYPE=STANDARD` so that `moinpppp` exists.
  No CFOUR source is modified.

---

## 3. Wick engine (`zwick`)

Scope for zccsdt: single-reference normal ordering relative to a determinant
(HF), used to derive spin-orbital CC equations. Scope reserved for zmr:
generalized normal ordering (density/cumulant contractions), which enters only
through the `Reference` object, so no engine rewrite is needed later.

Design:
1. **Operators.** Index spaces are objects (`occ`, `vir`, later `act`, ...).
   A tensor operator is a coefficient tensor times a normal-ordered string.
   H is `f + (1/4) <pq||rs> {p+ q+ s r}`; `T_n` are excitation operators.
2. **Contractions.** For products of normal-ordered groups, enumerate full
   contractions between different groups only (recursive pairing with
   space-compatibility pruning); sign from the permutation parity. A
   `Reference` supplies which pairs contract and to what: HF gives
   `delta_ij` for occ and `delta_ab` for vir.
3. **Similarity transform.** `<mu| e^{-T} H e^{T} |0>`, using the
   connected-diagram form (BCH terminates at 4 nested commutators for a
   two-body H). Optionally T1-dressed H (Section 4).
4. **Canonicalization.** Each term is a coefficient times a product of
   tensors with antisymmetric slot groups. A canonical form is found by
   relabeling dummies and sorting antisymmetric slots (brute force over the
   small symmetry group is fine for CCSDT sizes). Equal terms are merged, and
   terms related by external-index permutations are collected into `P(ij/k)`
   operators.
5. **Output.** A JSON term list (the checked-in source of truth for equations)
   plus an einsum emitter for the prototype.

Validation of the engine itself:
- Unit tests for small identities, e.g. `<0|{a+ i}{i+ a}|0>` or the known
  number of terms in the CCSD equations.
- The CCSD residual reproduces PySCF GCCSD numerically.
- CCSDT equals FCI for three electrons.

---

## 4. Equations and factorization

### 4.1 T1-dressed Hamiltonian (proposed; see Q2)
Per iteration the integrals are dressed by T1 through a non-unitary orbital
transformation. Only the bra-virtual and ket-occupied indices change, so every
block costs at most `o v^4` (the vvvv block):

```
~<ab||ef> = <ab||ef> - P(ab) t_m^a <mb||ef> + t_m^a t_n^b <mn||ef>
```
The CCSDT equations then contain only T2 and T3 with dressed integrals. That
means far fewer terms, and the factorization problem becomes much smaller.

Cost:
- Time: the dressing is negligible (`o v^4`, against `o^3 v^5` for one
  iteration).
- Symmetry: the dressed integrals lose bra-ket hermiticity, but we never
  assume it anyway, so nothing is lost.
- Memory: one dressed copy of each integral block. For N2/cc-pVTZ complex
  (o = 14, v = 106):

  | block | packed entries | complex size |
  |---|---|---|
  | vvvv | C(v,2)^2 = 3.1e7 | 0.50 GB |
  | vvov + ovvv | 2 o v C(v,2) = 1.7e7 | 0.26 GB |
  | all o^2 v^2 and smaller blocks | about 1e6 | 0.02 GB |

  The total is about 0.8 GB, compared with T3 + R3 = 2.2 GB and several
  T3-sized DIIS vectors.
- CFOUR stores a dressed W(ab,ef) list too, so against CFOUR the extra cost
  is essentially zero.
- If memory ever matters, the dressed vvvv need not be stored. The ladder
  GEMM consumes W in row batches of (b<c), so each batch can be dressed just
  before use: `o v^4` extra flops in total, zero extra v^4 memory. The same
  holds for vvov.

### 4.2 Factorization workflow (`zccsdt/opt`)
Per term, the steps are:
1. Take the term from the generated term list.
2. The flop tool enumerates the binary contraction orders. It reports the
   flops under the packed-storage model (Section 5) and the intermediate
   sizes, flagging any intermediate larger than T3.
3. The tool lists terms that share a subexpression or the same outer
   contraction with T3 or T2. These are the candidates for shared
   intermediates such as CFOUR's W(abej) and W(mbij), dressed with T2 and T3.
4. The scheme is chosen by hand and implemented as one function; the
   decision is recorded in the design notes.
5. A test checks the function against the raw equations.

Division of labour (decided): the tools are for *derivation, bookkeeping
and verification*; the factorization itself is **designed by hand, term by
term**. Automatic rules can only reproduce known patterns, while the purpose
of this project is to find the best choice for every term: contraction order,
intermediates, storage layout, batching and GEMM shape. Concretely:
- The engine derives the equations. The flop tool reports for each term the
  minimum cost and the cost of the chosen scheme, and it lists all
  candidate binary contraction orders as input for the manual decision.
- Each production term is a hand-written function in `zccsdt/terms/`,
  with its own design note in `docs/factorization.md`: the options
  considered, the choice made and why.
- Every hand-written term, and every intermediate, is checked automatically
  against the generated raw equations on random amplitudes.

### 4.3 T3 equation: draft contraction scheme

Schematic form. Exact signs and prefactors will come from the engine.
W-bar intermediates are T1-dressed and T2/T3-dressed as in CFOUR.

| # | Term | Scaling | Min. FMA (antisym. exploited) | Draft scheme |
|---|---|---|---|---|
| T3-1 | `1/2 P(a/bc) W_bcef t_ijk^aef` | o3v5 | o^3 v^5 / 24 | Per batch of ijk triples: gather T3 into `[e<f][c,ijk]`. One GEMM with W `[(b<c),(e<f)]`, giving `[(b<c)][a,ijk]`. Antisymmetrize P(a/bc) into packed R3. Optimal flops; GEMM with M = K = v^2/2 and large N. |
| T3-2 | `P(i/jk)P(a/bc) W_amie t_mjk^ebc` (ring) | o4v4 | o^4 v^4 / 4 | Loop over pairs j<k: gather `T[(b<c),(m,e)]`, GEMM with `W[(m,e),(i,a)]` (K = ov), then scatter with P(i/jk)P(a/bc) into R3. OpenMP over target triples, so the scatter is conflict-free. Optimal flops. |
| T3-3 | `1/2 P(k/ij) W_mnij t_mnk^abc` + `-P(k/ij) F_mk t_ijm^abc` | o5v3 | o^5 v^3 / 24 | Build the o^3 x o^3 operator `W6[(i<j<k),(l<m<n)]`: antisymmetrized W_mnij plus F_oo (at most 1.2e5 x 1.2e5 entries is fine for o <= 14; it is (C(o,3))^2). Then **one GEMM directly on packed T3**: `R3[ijk][abc] += W6 . T3`. This costs o/9 times the minimum (1.1x for o = 10, 1.6x for o = 14), but needs zero gathers, zero transposes, and N = C(v,3). For o > ~25, switch to the pair-driven GEMM. CFOUR uses DAXPY here. |
| T3-4 | `P(c/ab) F_ce t_ijk^abe` | o3v4 | o^3 v^4 / 12 | Folded into W_bcef as a one-body piece: `W'_bcef = W_bcef + (delta x F)` antisymmetrized. It then rides along with T3-1 at zero extra GEMM cost. |
| T3-5 | `P(k/ij)P(a/bc) W_bcek t_ij^ae` | o3v4 | o^3 v^4 / 4 | Per k: GEMM `W_k[(b<c), e] x T2[e, (a, i<j)]` (M = v^2/2, K = v, N = v o^2/2), then the same P(k/ij)P(a/bc) scatter kernel as the ring. |
| T3-6 | `-P(i/jk)P(c/ab) W_mcjk t_im^ab` | o4v3 | o^4 v^3 / 4 | GEMM with K = o. Cheap; generic kernel. |

- **Folded into the W intermediates, as in CFOUR:**
  - quadratic [[V,T2],T3] terms go into ring W, W_mnij, W_abef and F;
  - T3·<mn||ef> goes into W_bcek and W_mcjk;
  - T2^2 and T2^3 go into W_bcek and W_mcjk.
- **T2 dressing of W_abef (term T3-1):** `+1/2 t_mn^ab <mn||ef>` costs
  o^2 v^4 and is stored in the dressed vvvv buffer. The T2 equation's ladder
  can reuse the same buffer, with the matching W_mnij variant for the rest.
  This avoids a third v^4 array.

### 4.4 T3 -> T1/T2 and the CCSD part

| Term | Scaling | Scheme |
|---|---|---|
| `W_bmef t_ijm^aef` -> R2 | o3v4 | Per pair i<j: gather `T[a,(m,e<f)]` (K = o v^2/2), then GEMM with `W[(m,e<f), b]` |
| `W_mnje t_imn^abe` -> R2 | o4v3 | Per abc batch on packed T3 (analogous to T3-3) |
| `F_me t_ijm^abe` -> R2, `<mn\|\|ef> t_imn^aef` -> R1 | o3v3 | Single passes over T3 |
| CCSD terms | at most o2v4 | Generic permute + GEMM on unpacked T1/T2 (at most o^2 v^2 in size, cheap) |

The o2v4 CCSD ladder uses the packed vvvv with packed tau.

---

## 5. T3 storage and antisymmetry strategy (`docs/t3_storage.md`)

Decision (proposed): **store T3 and R3 fully packed**, `T3[i<j<k][a<b<c]`
with abc contiguous. Every access to T3 is a fused *gather* ("unpack +
permute + sign") that produces exactly the matrix the next GEMM needs. Every
write is a fused *scatter* ("antisymmetrize + pack"). There are no standalone
transposes of 6-index tensors.

| Option | Memory (entries) | Ladder FMA | Ring FMA | GEMM friendliness | Extra traffic |
|---|---|---|---|---|---|
| A. full o^3 v^3 | o^3 v^3 (36x) | o^3v^5/2 without tricks | o^4v^4 | trivial reshapes, but needs full transposes | full 6-index transposes |
| B. i<j<k, full abc (ncc style) | o^3v^3/6 (6x) | o^3v^5/12 if (e,f) not packed | o^4v^4/2 | large v^3 blocks per triple | abc permutation copies |
| C. i<j<k, (a<b, c) half-packed | o^3v^3/12 (3x) | o^3v^5/24 | o^4v^4/4 | natural for ladder | re-pack after each term |
| **D. fully packed (chosen)** | **o^3v^3/36 (1x)** | **o^3v^5/24** | **o^4v^4/4** | GEMM on gathered panels | gather/scatter O(o^3v^3) per term vs O(o^4v^4) flops |

Rationale:
- D is the minimum in both memory and flops. For N2/cc-pVTZ complex, that
  is 1.1 GB per T3-sized vector, against 6.7 GB for B.
- Gather/scatter traffic is O(o^3 v^3) per term against O(o^4 v^4) or
  O(o^3 v^5) flops, a ratio of about ov (about 1500 at TZ). It is affordable
  if the gathers are blocked for cache locality: for a fixed triple, all the
  entries needed come from one contiguous abc block.
- The hole-ladder operator trick (T3-3) works *directly* on D with no gathers.
- DIIS/extrapolation vectors of T3 size are cheapest in D.

Measured cost of each gather/scatter will be reported separately from GEMM
time. If gathers turn out to dominate, the fallback is C, which only changes
the kernels, not the equations.

Denominators: the diagonal Fock is used in the Jacobi update. Off-diagonal
Fock pieces live in F and W, so non-canonical orbitals work, which is
important for the invariance tests.

---

## 6. Implementation language: Python first

Decided: everything is first written in Python/numpy. The decision on what to
translate, and into what, is made **after** profiling the Python version.

The optimized Python version is not the einsum prototype. It already uses the
production design:
- packed T3 (Section 5);
- the hand-designed term schemes (Section 4);
- all heavy work in BLAS, through `numpy.matmul` / `scipy.linalg.blas`
  `dgemm` / `zgemm` on gathered panels;
- gathers and scatters as precomputed index/sign tables applied with numpy
  fancy indexing (`np.take`, `np.add.at`-free scatter via sorted unique
  targets).

It therefore measures the *algorithm* faithfully. The GEMM part already runs
at BLAS speed. What Python costs is the gather/scatter and loop overhead,
which the profile will show per term. Likely outcomes:
- (a) GEMM dominates, so only the BLAS backend matters: OpenBLAS (current
  numpy) vs MKL;
- (b) gather/scatter dominates, so only those few kernels move to compiled
  code (C with OpenMP, or numba), keeping the Python driver;
- (c) loop overhead over triples dominates, so batching is increased first.

A clean C interface for later Fortran use is designed at that point, around
whatever turned out to be the hot kernels.

Everything is dtype-generic (`float64` / `complex128`) from the start. The
same code path serves both; no function branches on dtype except BLAS
dispatch.

## 7. Milestones, commits and tests

Each milestone is several small commits, and every feature ships with its test.

| M | Content | Tests (pytest) |
|---|---|---|
| M0 | Skeleton: pyproject, pytest config; HDF5 format + writer/reader; PySCF GHF integral generator (real), RHF->GHF->complex-unitary generator, socutils X2C spinor generator | Round-trip HDF5; antisymmetry of stored blocks; E_HF recomputed from the file equals the SCF energy (real, rotated-complex, X2C) |
| M1 | `zwick` engine | Contraction identities; term counts; CCSD energy expression |
| M2 | (Done by the port) CCSD/CCSDT reference implementation + DIIS solver | **CCSD vs PySCF GCCSD, real and complex, 1e-10**; **CCSDT = FCI for 3 electrons** (real: PySCF FCI; spinor: zfci); **RHF->GHF->complex-unitary invariance** for CCSD and CCSDT (1e-10); other invariance tests (below) |
| M3 | Flop/bookkeeping tool; T1 dressing; hand-designed factorization for each term, with design notes in `docs/factorization.md` | Each term and intermediate vs raw equations on random amplitudes (1e-12) |
| M4 | Packed T3 storage, gather/scatter tables, measured BLAS peak (OpenBLAS vs MKL) | Pack/unpack round trip; gather/scatter vs dense einsum, real and complex |
| M5 | Optimized Python CCSDT driver (packed T3 + BLAS), one term at a time | Residual equals prototype to 1e-12 per iteration; energies equal to prototype to 1e-10 |
| M6 | External references: MRCC and CFOUR CCSDT on small molecules; CFOUR x2c integral converter | **CCSDT vs CFOUR / MRCC to 1e-8** (H2O/STO-3G, H2O/6-31G, HF, N2 at small basis, all C1); CCSD from CFOUR spinor integrals vs `xx2cccsd` CCSD energy (1e-8) |
| M7 | Profiling per term; decide what to translate and into what (Section 6); tuning; benchmarks | Benchmarks are not pytest pass/fail; results go in `docs/benchmarks.md` |

Additional correctness tests, cheap and strong for the complex code path:
- **RHF -> GHF -> complex unitary (primary complex test).**
  1. Take an RHF reference and expand it to GHF spin orbitals.
  2. Apply a random complex unitary U = U_occ (+) U_vir that mixes alpha and
     beta within the occupied space and within the virtual space.
  3. Transform the integrals with these orbitals. They become genuinely
     complex, with no alpha/beta block sparsity, no 8-fold symmetry and a
     non-diagonal Fock matrix.
  4. CCSD and CCSDT are invariant under occ-occ and vir-vir rotations, so the
     energy must equal the real RHF-based result. That result is itself
     checked against PySCF (CCSD), FCI, CFOUR and MRCC (CCSDT).

  This exercises every spin block, every complex-conjugation path and the
  off-diagonal Fock handling, using only real reference programs. The same
  construction with phases only (diagonal U) and with a UHF/ROHF-like open
  shell GHF reference gives two cheaper variants.
- **Phase invariance.** Multiply real GHF orbitals by random phases e^{i theta_p}
  (the diagonal special case of the test above).
- **Spin-free limit.** Spinor code on a nonrelativistic GHF reference equals
  the real code.
- **Hermiticity.** The imaginary part of the energy vanishes (below 1e-12) for
  Hermitian H.

---

## 8. Benchmarks and efficiency metrics

Per dominant term (T3-1, T3-2, T3-3, T3-5, T3->T2) we will record:
- theoretical minimum FMA (Section 4.3);
- FMA actually executed;
- time split into gather, GEMM and scatter;
- achieved flop rate as a fraction of the measured ZGEMM/DGEMM peak.

Per iteration we will record total wall time and peak RSS (from
`/proc/self/status` VmHWM, plus the internal allocator count). The thread
count is fixed, with `OMP_NUM_THREADS` and `MKL_NUM_THREADS` set equal for
zccsdt and CFOUR.

Systems: H2O and N2 with cc-pVDZ and cc-pVTZ, all C1, all-electron (or the
same frozen core in both codes).

Comparison targets (see Q1). Because CFOUR has no spinor CCSDT:
1. **Nonrelativistic CCSDT:** CFOUR `vcc`/`ecc` UHF (spin-integrated) and
   `ncc` RHF (spin-adapted) per-iteration time against zccsdt real
   spin-orbital. We will also compute the "representation factor" exactly:
   the ratio of minimal FMA counts of spin-orbital vs spin-integrated vs
   spin-adapted formulations for each dominant term, obtained by counting over
   the actual spin blocks. We report the raw ratio and the ratio divided by
   this factor.
2. **Spinor, same conditions, CCSD level:** zccsdt complex CCSD iteration
   (optimized code path) against `xx2cccsd` on the same CFOUR integrals.
3. **Spinor CCSDT:** complex against real zccsdt (the expected ratio is about
   4x flops), plus absolute % of peak. Optionally, DIRAC+MRCC
   Kramers-unrestricted CCSDT as an external spinor reference, if a working
   DIRAC build exists.

The target "within 2x of CFOUR" is applied to (1) after removing the
representation factor, and to (2) directly.

---

## 9. Open questions

- Resolved on 2026-09-30:
  - Q1: benchmark substitutes of Section 8 accepted.
  - Q3: factorization is manual, term by term.
  - Q4: MRCC and CFOUR may be run on this node.
  - Q5: default branch `master`, no remote.
  - Q2: T1-dressed Hamiltonian accepted.
  - C++ is deferred until the Python version has been profiled.
