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

- Levels: CCSD and CCSD(T) (plus Lambda/EOM/PNO variants). **There is no
  spinor or complex iterative CCSDT.** `ccsdso/eccsdt.f` is in fact a CCSD(T)
  routine of the older, Kramers-blocked `ccsdso` code.
- Amplitudes: `t1(nv,no)`, `t2(nv(nv+1)/2, no(no+1)/2)` complex, triangle
  packed (diagonal included), all in core.
- Integrals: antisymmetrized <pq||rs> blocks written as unformatted sequential
  Fortran files `moinhhhh`, `moinpphh`, `moinphhp`, `moinhhpp`, `moinhhhp`,
  `PPPH000k` (one per occupied index), and `moinpppp` only when
  `ABCDTYPE` is not `AOBASIS`. Complex stored as interleaved re/im. Only the
  4-fold complex symmetry (pq|rs) = (rs|pq) = (qp|sr)* is assumed.
- Orbital energies / coefficients / Fock: `scf2ceig`, `scf2ccmo`, `scf2cfck`
  (plain files and JOBARC records) written by `xsox2chf`.
- Contractions: ring term by unpacking T2 to (a,i,b,j) and one (nv*no)^3 ZGEMM;
  hhhh and pppp ladders by packed ZGEMM; AO-driven pppp ladder (default
  `ABCDTYPE=AOBASIS`) splits complex tau into re/im and uses two real DGEMMs
  with the real AO integrals. OpenMP for reordering loops, threaded MKL.
- (T): loop over i<j<k, W(a<=b, c) blocks via six ZGEMMs.
- Run sequence (no xcfour driver): `xjoda; xx2camf|xsox2catm; xsox2chf;
  xx2cvtran; xx2cccsd`. Threads from `OMP_NUM_THREADS` / `MKL_NUM_THREADS`.
  Per-iteration output line `CCSD cycle ...` plus per-term timings.
- Build: ifort, `-i8`, MKL ILP64 `mkl_intel_thread`, `-qopenmp`.
