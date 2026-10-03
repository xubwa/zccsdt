# Notes on the CFOUR (cfour-x2cso) CC implementations

Design notes only. Nothing in zccsdt is copied from CFOUR; these notes record
what we learned from reading the source so that design decisions in
`docs/plan.md` can refer to them. Paths are relative to `~/apps/cfour-x2cso`.

## 1. Iterative CCSDT in CFOUR

| Module | Spin treatment | Notes |
|---|---|---|
| `ncc/` (C++) | RHF, spin-adapted only | Cleanest design. T3 code in `ncc/cc_it/ccsdt/ccsdt.cxx`, diagram kernels in `ncc/core/diagrams.cxx`. MPI + DPD-like list layer. |
| `vcc/` (`trps*.f`) | Spin-integrated UHF (AAA/AAB/ABB/BBB) | Watts-Bartlett code. `trps14.f` is the per-ijk driver for the AAA case. |
| `ecc/` + `libecctrip/` | RHF spin-adapted (`trp1rhf.f`) and UHF (`trp1aaa/aab/abb.f`) | Gauss-style; also used by `lcc/` (Lambda) and `rsp/`. |
| `ccsdtincore/` | Spin-integrated UHF, full 6-index arrays, no BLAS | Debug/reference code; useful as an equation oracle only. |

None of these is relativistic or complex.

### T3 storage
- `ncc`: blocks indexed by i<=j<=k (i=j=k skipped); each block is the full abc
  cube (per irrep). Permutations of abc are handled as views resolved at
  contraction time. Blocks grouped into disk "hunks" distributed over MPI ranks.
- `vcc`: one direct-access record per i<j<k triple, packed a<b<c (AAA). For the
  ladder/ring terms the packed block is expanded to half-packed (a<b, c) with
  signs (`symexpt3.f`); the result Z(a<b, c) is folded back to a<b<c by
  `symcontw.f`: T(abc) = Z(bc,a) - Z(ac,b) + Z(ab,c).
- `libecctrip` RHF CCSDT reads the whole old T3 into core; new blocks are
  streamed to a sequential file.

### Organization of one CCSDT iteration (common to all three)
1. Fold the old T3 into the Hbar-like intermediates through <mn||ef>:
   W(abej) += T3.<mn||ef>, W(mbij) += T3.<mn||ef>. These dressed intermediates
   are shared by the T2 equation and by the T2 -> T3 term.
2. Rebuild CCSD-type intermediates (F(mi), F(ae), F(me), W(mnij), W(abef),
   ring W(mbej)).
3. Loop over ijk blocks. For each block: build Z3(abc) from
   T2.W(abej), T2.W(mbij), F.T3, W(abef).T3 (o3v5), W(mnij).T3 (o5v3), rings
   (o4v4); divide by the denominator; immediately accumulate this block's
   contribution to R1/R2 (T3 -> T1/T2) while it is in memory.
4. DIIS (or CROP in ncc) on T1/T2, optional extrapolation of T3.

### Dominant contractions
- o3v5 ladder: per ijk, W(a<b, e<f) x T3(e<f, c) by one GEMM (vcc
  `t3wt314p.f`, libecctrip `t3int31.f`, ncc `c3_into_z3_wabef_rhf`). W(ab,ef)
  is a stored dressed MO list, streamed in column batches if memory is short.
- o5v3 hole ladder: vcc/libecctrip loop over m<n and DAXPY the T3(mnk) block
  (level-1 BLAS). ncc switches between GEMM and AXPY depending on the pattern.
- Rings: per m, expand T3(mjk) and GEMM T3(ab,e).W(e,c).
- Diagonal (i=j, j=k) blocks are handled with scale factors rather than stored.

### Antisymmetry and transposes
No antisymmetric GEMM anywhere: always unpack/half-unpack, plain DGEMM, then
re-antisymmetrize/re-pack. Explicit out-of-place transposes (`TRANT3`,
`trans_321`, `sym_213`, `SYMTR*`, `TRANSP`).

### Parallelism
Essentially threaded BLAS only. `ncc` uses OpenMP macros in transposes and
packing; MPI distributes ijk hunks.

## 2. X2C spinor CC (`x2cccsd/`, `libx2ccc/`, `x2cvtran/`)

Read in detail on 2026-10-02 (cfour-x2cso at 35b1f40e3): the spinor CCSD and
CCSD(T) of Liu, Shen, Asthana and Cheng, J. Chem. Phys. 148, 034106 (2018),
equations (A1)-(A8) (Gauss-Stanton intermediates).

- Levels: CCSD and CCSD(T), plus Lambda, EOM, densities and gradients, and
  Cholesky (`x2ccccd/`) and PNO (`x2cccpno/`) variants. **There is no spinor
  or complex iterative CCSDT.** `ccsdso/eccsdt.f` is in fact a CCSD(T)
  routine of the older, Kramers-blocked `ccsdso` code.
- Run sequence (no xcfour driver): `xjoda; xsox2catm; xsox2chf; xx2cvtran;
  xx2cccsd` (X2CAMF: `xx2camf` first). `xsox2chf` reads the basis from ZMAT
  itself, so `BASIS=SPECIAL` with one line per atom. Threads from
  `OMP_NUM_THREADS` / `MKL_NUM_THREADS`. `bench/cfour/slurm_x2ccc.sh` runs
  the chain; `bench/cfour/x2ccc_summary.py` sums up the timers that
  `xx2cccsd` prints for every iteration.
- Build: ifort, `-i8`, MKL ILP64 `mkl_intel_thread`, `-qopenmp`.

### Representation and storage
- Kramers-unrestricted spinors, complex arithmetic. No Kramers symmetry and
  no savings from the double group: its irreps (`dg_cosym`) only zero
  symmetry-forbidden amplitudes (`pgsymmasksd0`), and only with `%ipgmask`
  in ZMAT, which the EOM programs read and `xx2cccsd` does not.
- Amplitudes: `t1(nv,no)`, `t2(nv(nv+1)/2, no(no+1)/2)` complex, triangle
  packed with the (zero) diagonal included, in core. DIIS (`CC_EXPORDER`,
  default 5 vectors) keeps the amplitudes and the Jacobi step of each
  iteration in core; once the space is full all vectors are shifted by one.
- MO integrals: antisymmetrized <pq||rs> blocks written by `xx2cvtran` as
  unformatted sequential Fortran files and read into core at the start:
  `moinhhhh`, `moinpphh`, `moinhhhp` (packed pairs), `moinphhp`, `moinhhpp`
  (full o^2 v^2). <ab||ci> (a<=b) stays on disk, one file per occupied
  index (`PPPH000k`), and is read back slab by slab by every term that
  needs it. <ab||cd> exists only when `ABCDTYPE` is not `AOBASIS`
  (`moinpppp`, v^4/4 complex, in core). Complex numbers as interleaved
  re/im; only the 4-fold complex symmetry (pq|rs) = (rs|pq) = (qp|sr)* is
  assumed.
- AO integrals: the scalar (spin-free) two-electron integrals are real.
  `xx2cccsd` keeps a screened list (packed indices and value) and, by
  default (`flagspa` 0), two dense copies of the 4-fold packed matrix
  (N(N+1)/2)^2 for the ladder, 2 N^4 bytes each (N: scalar AO functions).
  With `%flagspa 1` the ladder runs over the list instead, the target rows
  distributed over the OpenMP threads (`loading`, `balancing`,
  `sparse_t2aa_lbal`).
- Transformation (`x2cvtran`, AO-ladder case): the sparse AO list is
  transformed with the real and imaginary parts of the occupied spinor
  coefficients, then to all spinors, alpha and beta components summed,
  into `vmo(nmo,nmo,nmo,no)` -- every integral with at least one occupied
  index, nmo^3 o complex numbers in core -- and sorted into the blocks above.
- Memory: one work array partitioned up front (`eomccini`), among others
  six scratch arrays of max(o^4, o^2 v^2, v^3, N^3, N^2 o^2/4) complex
  numbers each. No memory planning and no out-of-core path apart from the
  PPPH files.

### One CCSD iteration (`hbarso`)
Costs in complex multiply-adds, o and v counting spinors:

| Term | Routine (timer) | How | Cost |
|---|---|---|---|
| W(mbej) += T2 <mn\|\|ef> | `socccon1` (I7) | T2 unpacked to (a,i,b,j), one ZGEMM | o^3 v^3 |
| ring P(ij)P(ab) t_im^ae W(mbej) | `t2rngso` (Ring Diagram) | T2 unpacked, one ZGEMM, antisymmetrized, repacked | o^3 v^3 |
| -1/2 P(ab) t_m^a tau_ij^ef <mb\|\|ef> | `socccon5oc` | per m: one ZGEMM over packed ef | o^3 v^3 / 4 |
| pppp ladder | `t2abcdaoso` (PPPP ladder) | in the AO basis, see below | 1.5 N^4 o^2 real + 3 o^2 (N v^2 + N^2 v) |
| hhhh ladder; W(mnij) | `t2ijklso`; `socccon2` (I5) | packed ZGEMM | o^4 v^2 / 8 each |
| other terms with <ab\|\|ci> | `socccon6oc`, `soccct17oc`, `soccct15eoc` (I8), `soccct11oc` (I22) | per occupied index: read the slab, then ZGEMMs; I8 one (v x o x v) ZGEMM per (m, b), I22 matrix-vector products | o^2 v^3 / 2 to o^2 v^3 |
| F(ae), F(mi), Fock terms | `socccon3/4`, `t2ftaaso2`, `t2ftiiso2` | ZGEMM | o^2 v^3 / 2, o^3 v^2 / 2 |

- **AO-driven pppp ladder** (default with `ABCDTYPE=AOBASIS`): for every
  pair i<=j, tau_ij is back-transformed to AO pairs with the alpha and beta
  components of the virtual spinors (`vtransub1`, two ZGEMMs per pair):
  alpha-alpha and beta-beta (antisymmetric, packed mu>=nu) and alpha-beta
  (full). The complex tau is split into real and imaginary parts and
  multiplied by the real AO integrals with DGEMMs (alpha-alpha and
  beta-beta: (N(N+1)/2)^2 x o^2/2 each, `rmulcpmat`; alpha-beta: one slab
  per AO index, `rmulcpmatoc`), and the result is transformed back. v^4 is
  never stored. Against the packed MO ladder (o^2 v^4 / 8 complex) the
  floating-point work is about 1/3 for H2O/cc-pVQZ (o = 10, v = 220,
  N = 115) and about 1/2 for benzene/cc-pVDZ (o = 42, v = 186, N = 114),
  the transformations included (about 48/N of the DGEMM work when v ~ 2N).
- **I/O**: five terms read all PPPH files in every iteration (5 o v^3 / 2
  complex numbers, single-threaded `rfile`); every iteration also writes
  t1 and t2 to files (restart).
- **Update and convergence**: Jacobi step r = R / D, t <- t + r; DIIS
  extrapolates t and r; converged when max |r| (`r1max`, `r2max`) <
  10^-`CC_CONV`. The energy change is not tested.
- **Parallelism**: OpenMP in the reordering loops, threaded MKL in the
  GEMMs; the GEMMs of the I8 / I22 terms and the slab reads are small or
  serial.

### (T) (`t3drvsooc`)
- Loop over i>j>k. For each triplet W(b<=c, a) is built by 3 ZGEMMs of
  (v(v+1)/2 x v x v) with the PPPH slabs of i, j and k and 3 of
  (v(v+1)/2 x v x o) with <jk||ma>; the disconnected part by ZAXPYs; the
  energy by an OpenMP reduction over a>b>c (a precomputed list of triples).
  Work: o^3 v^4 / 4 complex multiply-adds.
- The three PPPH slabs (v^3/2 complex numbers each) are read from disk for
  every triplet: o^3 v^3 / 4 complex numbers, 4 o^3 v^3 bytes, read in
  all -- from the page cache at best, serially between the threaded GEMMs.
- Memory: two W arrays of v^2 (v+1)/2, T2 unpacked twice, <jk||ma>
  conjugated: O(o^2 v^2), nothing of size o v^3 in core.

Measured profile (per term, three molecules, against zccsdt's CCSD):
`docs/benchmarks.md`, "Two-component CCSD(T): CFOUR `xx2cccsd`".

### Cholesky variant (`x2ccccd/`, `x2cvtrancd/`)
The same equations; the <ab||ci> terms are built from Cholesky vectors
(`ivab`, `ivia`, `ivai`) instead of the PPPH files, `moinhhpp` and
`moinphhp` are read from disk in every iteration instead of being kept,
and the pppp ladder can also run on the Cholesky vectors (`t2abcdcd*`).
