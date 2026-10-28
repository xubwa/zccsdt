# The native program and CFOUR's spinor CC library (libx2ccc)

The question: should the CCSD part of the native CCSDT iteration
(`cfour/xzccsdt`: the T1 and T2 residuals from T1 and T2, the one- and
two-body intermediates, the T1 dressing) be done by CFOUR's own spinor CC
routines - `libx2ccc.a` and the CCSD program `xx2cccsd` - instead of this
repository's code?  Reasons given for it: reuse, CFOUR's data structures and
files, an easier merge.

This document has the map of CFOUR's code (section 1), the correspondence
term by term (2), a probe program that calls CFOUR's CCSD residual routine
and compares it with the native code (3), their timing in one job (4), the
state of the CCSD start `XZ_START=ccsd` (5) and a recommendation (6).
CFOUR's tree (`~/apps/cfour-x2cso-rebuild`) was read and linked, not changed
and not copied; the few lines quoted are marked.  "Read" below means: taken
from the source, not run; "measured" gives the job.

**In short.**  CFOUR's CCSD iteration is one routine, `hbarso`, a sequence
of calls of 24 routines of `libx2ccc` on arrays in the work array; it can be called from
another program without xx2cccsd's initialisation (a dozen module variables
and the offsets of one module set by hand).  The probe does that: the new T1
and T2 of one step agree with the native CCSD part to 1e-15 element-wise
(H2O 10 + 16, HBr 26 + 78, 26 + 60 with dropped virtuals; 3e-14 at 26 + 120).  It is
not slower than the native code, as was expected from the comparison of
whole CCSD runs; for the same result it takes 0.74 s against 1.00 s (8
threads) and 0.51 s against 0.48 s (24 threads) at 26 + 78.  And still the
native iteration should not call it: the CCSDT step needs the T1-dressed
integrals and intermediates for the T3 terms whatever computes R1 and R2, so
`hbarso` would replace only 0.2 to 0.4 s of native work per iteration, cost
0.5 to 0.7 s, and bring complex dense arithmetic without labels into an
iteration that runs real and in blocks.  The whole CCSD part is 0.3 % of a
dense CCSDT iteration at this size.

## 1. How xx2cccsd organises a CCSD iteration

### 1.1 The program

`x2cccsd/` (76 sources) is the program; `libx2ccc/` (355 sources) the
library it shares with `xx2clcc`, `xx2chbar`, `xx2cccden*`, `xx2czvec`,
`xx2ceom*`.  The routines of the library are named by contraction, not by
index pattern: `socccon1` ... `socccon46tt` (contractions with T2),
`soccct11` ... `soccct21` (with T1), `t2rngso`, `t2ijklso`, `t2tau`, `r1ft`,
`iint*` (Hbar elements for EOM), `sparse_*` and `aoreord*` (the AO-driven
ladder), `expnd2*`, `vsort*` (packing).  Variants by suffix: `oc` reads
`<ab||ci>` from one file per occupied spinor, `cd` works on Cholesky vectors,
`l` belongs to the Lambda equations, `t` / `tt` to triples and EOM.

The chain of calls (read):

    X2CSOCCSD (x2csoccsd.f)   MCHPRM, CRAPSI, READIN -> the work array ICORE(I0)
     soccsd                   int_itrf0/1/3/fg (basis, molecule), setsocc
      setsocc                 flags of JOBARC and sections of the ZMAT into the
                              module soccmod: no, nu, nmo, ndrop (NUMDROPA),
                              ndropvir (%vircut: NUMVIRTA), inonhf, ndiis, conv,
                              ppppao (ABCDTYPE), ...; irreps of the spinors (file dgcosym)
      x2c_ccsddrv
       eomccini               the layout: every array an offset (module soccmem)
                              into the work array; index tables (settrioff); the
                              AO integrals (file IIII or x2c_int2e) and, unless
                              %flagspa, two AO arrays of n^2 (n+1)^2 / 4 words
       wppphfname             names of the files PPPH0001 ...
       getrec scf2ccmo, scf2cfck, scf2ceig; frmfck2c3, itrf_frkso (Fock matrix
                              in the spinor basis: f_ii, f_aa, f_ia)
       vreadf, vreadp         the MO integral files into the work array
       x2c_ccsdsoao           the iterations:
        t12iniso0             start: files t1guess, t2guess if there, else second order
        do icyc = 1, ncycle
          hbarso              intermediates, R1, R2, the update R / D   <- the CCSD step
          diisccso            DIIS (vectors in the work array)
          t12updso            t = t + r, norms, energy, files t1update, t2update
        dmpccwf               files t1guess, t2guess
        t3drvso0              (T) if asked for
        dmpccene              TOTENERG

`hbarso` is in the program directory, not in the library; so are the masks
it applies, `t2ftaaso2`, `t2ftiiso2`, the updates `r1updso`, `r2updso`, the
readers and the ladder drivers.  Everything it calls for a contraction is in
`libx2ccc.a`.

### 1.2 Data: files, arrays, index order, packing

All amplitudes and integrals are `complex(kind=dcp)` arrays in Fortran order
with indices from 1, placed in the one work array of `CRAPSI` at offsets that
are module variables (`use soccmem`: `t1`, `t2`, `r1`, `r2`, `hhhh`, `pphh`,
`phhp`, `hhpp`, `ppph`, `hhhp`, `pppp`, `ftaa`, `ftii`, `ftia`, `whhhh`,
`wphhp`, `tau`, `taut`, `tmp1` ... `tmp11`, the DIIS vectors, ...).  A pair
index is triangular **with the diagonal**: `ab = a + b (b - 1) / 2` for
`a <= b`, `nu (nu + 1) / 2` values, the diagonal elements zero (`psmasksd`
sets them to zero in every iteration).

| array | file | layout | content |
|---|---|---|---|
| `t1`, `r1` | `t1guess` (end), `t1update` (every cycle) | `(a, i)` | T1, its update |
| `t2`, `r2`, `tau`, `taut` | `t2guess`, `t2update` | `(ab, ij)` | T2 on `a < b`, `i < j`; tau = t2 + t1 t1, taut with 1/2 |
| `hhhh` | `moinhhhh` | `(mn, ij)` | `<mn\|\|ij>` |
| `pphh` | `moinpphh` | `(ab, ij)` | `<ab\|\|ij>` |
| `hhpp` | `moinhhpp` | `(e, m, f, n)` | `<mn\|\|ef>` as an `(o v)^2` matrix |
| `phhp` | `moinphhp` | `(e, m, b, j)` | `<mb\|\|ej>` |
| `hhhp` | `moinhhhp` | `(mn, i, e)` | `<mn\|\|ie>` |
| `ppph` | `PPPH0001` ... | `(ab, c)`, the file of the occupied spinor i | `<ab\|\|ci>`: **never in core as a whole** |
| `pppp` | `moinpppp` | `(ab, cd)` | `<ab\|\|cd>` with `ABCDTYPE=STANDARD`; otherwise the ladder is AO-driven |
| `oe` | JOBARC `scf2ceig` | complex, active occupied then virtual | spinor energies (`oe = ieig + 2 ndrop`) |

These are the files the native reader takes (`xzread.f90`, docs/native.md
section 2); it reads `moinhhpp` where CFOUR also reads `moinpphh`.  Frozen
and dropped spinors: the files hold the active spinors only, `no` and `nu`
are the active numbers, nothing else in the CC code knows about them.  The
native layouts differ in three ways: C order with indices from 0 (declared
reversed in Fortran), pairs **without** the diagonal in lexicographic order
(first index slowest), and `<ov||vv>` and the dressed arrays in core.
`T1[i][a]` is the same memory layout as `t1(a, i)`; T2 needs a gather
(`cfput1` of the probe: `t2(ab, ij) = T2[i][j][a][b]`, 24 lines).

### 1.3 One iteration: `hbarso`, term by term

The equations are (A1) to (A8) of J. Chem. Phys. 148, 034106 (2018), the
spin-orbital CCSD equations with tau and the F and W intermediates of
Stanton, Gauss, Watts and Bartlett; `W_abef` is not formed (the ladder runs
on tau and one extra term takes its T1 part).  Read from `hbarso.f90` and
the routines; the times are those `hbarso` prints itself and the probe's
timer around the call, HBr 26 + 78, 8 / 24 threads (job 31781783, c261).

| step | routine | how it is done | s (8 / 24 thr) |
|---|---|---|---|
| masks on T (core-valence, constrained occupied, old gradient, point group) | `cvsmasksd1`, `psmasksd`, `conmasksd`, `oldgradmask` | loops; `conmasksd` reads the files CONTOCCN, CONTOCCP and `oldgradmask` the ZMAT in every call | |
| tau, taut (A8) | `t2tau` twice | OpenMP loop over pairs | 0.004 / 0.002 |
| F_ae (A3): `-1/2 taut <mn\|\|ef>` | `socccon3` | OpenMP expansion of both operands to `(nu, nu, pairs)`, one ZGEMM | 0.008 / 0.005 |
| F_ae: `t_m^f <ma\|\|fe>` | `soccct11oc` | per occupied m: read PPPH file; per a: expand, ZGEMM with one column - a serial loop of o v small products | 0.073 / 0.082 |
| F_mi (A4) | `socccon4`, `soccct12` | expansion (OpenMP inside a serial loop over pairs), ZGEMM | 0.010 / 0.006 |
| F_me (A5) | `soccct13` | ZGEMM, matrix times vector | 0.002 / 0.001 |
| W_mnij (A6): `1/2 tau <mn\|\|ef>` | `socccon2` | one ZGEMM on packed pairs | 0.007 / 0.003 |
| W_mnij: `<mn\|\|ij>`, `P(ij) t_j^e <mn\|\|ie>` | `daxpy`, `soccct14` | ZGEMM, OpenMP antisymmetrisation | 0.001 / 0.001 |
| W_mbej (A7): `-(1/2 t2 + t1 t1) <mn\|\|ef>` | `socccon1` | OpenMP expansion of T2 to `(nu, no, nu, no)`, one ZGEMM `(o v)^3` | 0.118 / 0.046 |
| W_mbej: `t_j^f <mb\|\|ef>` | `soccct15eoc` | per m: read PPPH file; per b: expand, ZGEMM, o ZAXPYs - serial loop | 0.082 / 0.089 |
| W_mbej: `-t_n^b <mn\|\|ej>` | `soccct16` | OpenMP copies, ZGEMM | 0.013 / 0.009 |
| R2 (A2): `<ab\|\|ij>` | `dcopy` | | |
| pp ladder `1/2 tau <ab\|\|ef>` | `t2abcdmoso` -> `t2abcdso` (or `t2abcdaoso`: AO-driven) | one ZGEMM on packed pairs, bare integrals | 0.052 / 0.020 |
| hh ladder `1/2 tau W_mnij` | `t2ijklso` | one ZGEMM | 0.006 / 0.003 |
| ring `P(ij) P(ab) t_im^ae W_mbej` | `t2rngso` | OpenMP expansion, one ZGEMM `(o v)^3`, OpenMP antisymmetrisation, ZAXPY | 0.131 / 0.053 |
| `-t D` | `t2oeso` | OpenMP loop | |
| `P(ab) t_ij^ae (F_be - 1/2 t_m^b F_me)` | `t2ftaaso2` | OpenMP expansion, ZGEMM, OpenMP reduction | |
| `-P(ij) t_im^ab (F_mj + 1/2 t_j^e F_me)` | `t2ftiiso2` | the same | |
| `P(ij) t_i^e <ab\|\|ej>` | `soccct17oc` | per j: read PPPH file, ZGEMM; OpenMP antisymmetrisation | |
| `-P(ab) t_m^a <mb\|\|ij>` | `soccct18` | OpenMP copy, ZGEMM, OpenMP reduction | |
| `-P(ij) P(ab) t_i^e t_m^a <mb\|\|ej>` | `soccct19` | two ZGEMMs, OpenMP loops | |
| `-1/2 P(ab) t_m^a tau_ij^ef <mb\|\|ef>` | `socccon5oc` | per i: read PPPH file, ZGEMM; then ZGEMM and OpenMP reduction | |
| R1 (A1): `-t D`, `-1/2 t_im^ef <ma\|\|ef>`, `-1/2 t_mn^ae <nm\|\|ei>`, `-t_n^f <na\|\|if>`, the F terms | `t1oeso`, `socccon6oc` (PPPH files), `socccon7`, `t1rngso`, `r1ft` | ZGEMMs; `r1ft` has serial loops | |
| masks on R, `R / D`, norms | `r1updso` (serial), `r2updso` (OpenMP), `t1normso`, `t2normso` (serial) | | |
| all of R2 after the ring, R1, masks, update | | not timed by `hbarso`: the rest of the call | 0.233 / 0.185 |
| **`hbarso`, whole** | | | **0.743 / 0.506** |

The `<ab||ci>` files are read five times per iteration, once per occupied
spinor each (`soccct11oc`, `soccct15eoc`, `soccct17oc`, `socccon5oc`,
`socccon6oc`: 5 o sequential reads of `nu^2 (nu + 1) / 2` complex numbers;
26 + 120: 1.8 GB per iteration, from the page cache in the runs here).

### 1.4 Threading and compile flags

From `make.out` of the rebuild, the library and the program:

    ifort -O3 -cpp -assume byterecl -assume buffered_io -extend_source -i8 -mcmodel=medium -DINT64 ... -qopenmp -I ../libx2c -c ../libx2ccc/socccon1.f90

(`-qopenmp`, no `-xHost`: the loops are compiled for the baseline
instruction set), linked with threaded MKL, 8-byte integers
(`-lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5`) - the same MKL
and OpenMP runtime as the native program.

Of the 355 sources of `libx2ccc`, 191 call ZGEMM and 93 have OpenMP
directives; there is no DGEMM and no real variant.  The pattern of a routine
is: OpenMP loop that unpacks an operand to a rectangular array, one ZGEMM
over the whole contraction, OpenMP loop that antisymmetrises and packs.  So
the large products are threaded by MKL and the packing by OpenMP.  What is
not threaded: the loops over occupied spinors and virtual indices around the
PPPH files (`soccct11oc`, `soccct15eoc`: o v products of v^2 work each, with
the file read in between), `r1ft`, `r1updso`, the norms, and the pair loops
of `socccon4`, `t2ftiiso2`, `socccon6oc`, where the parallel region is
inside a serial loop.  Their shares show in the table: the two PPPH loops of
the intermediates take 0.155 s of 0.743 s at 8 threads and 0.171 s of 0.506 s
at 24.

### 1.5 Memory

One work array (`MEMORY_SIZE`), every array a module offset set once in
`eomccini`; nothing is allocated per iteration.  In core for CCSD: all MO
integral classes except `<ab||ci>` (one record of `nu^2 (nu + 1) / 2` at a
time), `<ab||cd>` on pairs with `ABCDTYPE=STANDARD`, the intermediates of
1.3, two tau arrays, six work arrays of `2 max(o^4, o^2 v^2, v^3, n^3, ...)`
words, `ndiis` DIIS vectors and residuals in core, and - for every
`ABCDTYPE` - the AO integral list and (unless `%flagspa`) two arrays of
`n^2 (n + 1)^2 / 4` words for n basis functions, which the MO-driven
iteration does not use.  There is no plan and no check of the total against
MAXCOR was seen in `eomccini` (it prints "Memory used (in GB)").  The part that
`hbarso` itself needs is 655 MB at 26 + 78 and 2.0 GB at 26 + 120 (the
probe's region, section 3).

### 1.6 Symmetry and real arithmetic

Confirmed from the source: every amplitude, integral and intermediate of
`libx2ccc` is `complex(kind=dcp)`; there are no real routines.  Symmetry
labels do not exist as blocks: the arrays are dense.  What exists is a mask,
`pgsymmasksd0`: with `ipgmask /= 0` it sets to zero the elements of R1 and
R2 whose irreps (file `dgcosym`, `gpvalue`) do not add up - after the dense
residual has been computed.  No source of `x2cccsd/` or `libx2ccc/` assigns
`ipgmask`, so in xx2cccsd the mask is off.

### 1.7 Related programs

`xx2chbar` builds the CCSD Hbar after convergence (`eomhbar`: `hiaccsd`,
`hijccsd`, `habccsd`, `hbariabjccsd`, `hbarijklccsd`, `hbarabciccsd`,
`hbarijkaccsd`, `hbarciabccsdoc`, `hbarkaijccsd`), with `W_abci` in files
per occupied spinor (`WPH1nnnn`, `WPH2nnnn`) and its `<ab||ef> t1` part
through the AO integrals (`t1abciaosooc`).  `xx2clcc` solves the CCSD Lambda
equations on these, `xx2cccden*` and `xx2czvec` make the densities and the
Z-vector.  These are the places where intermediates of the kind the T3
equation needs (W_abei, W_mbij) exist in CFOUR; they are made once per
calculation, for CCSD, not per iteration (read; not run here).

## 2. The native CCSD part against libx2ccc, term by term

The native step (`xzstep.inc`, `xzimd.inc`; docs/native.md section 3,
docs/factorization.md) is written in **T1-dressed** integrals: R1 and R2 are
the short dressed expressions of the header of `xzccsd`, and the same
dressed arrays and intermediates are the operands of the T3 terms.  CFOUR's
`hbarso` is written in bare integrals with tau.  The two give the same R1
and R2 (section 3), but they do not share intermediates, with two
exceptions.

| native (routine) | what it makes | needed by T3 terms | CFOUR in `hbarso` | conversion / remark |
|---|---|---|---|---|
| `xzfock` | dressed f_oo, f_ov, f_vo, f_vv | f_ov (pass 1), f_oo, f_vv (through F) | f_ov = F_me (`soccct13`): **identical**; the t1 parts of f_vv, f_oo are in `soccct11oc`, `soccct12`; f_vo: none | `ftia(m, e)` -> `fov[m][e]`: a transpose.  F_ae, F_mi of CFOUR are not the dressed f: they carry `-1/2 taut <mn\|\|ef>` and leave `-1/2 t F_me` to `t2ftaaso2`, `t2ftiiso2` |
| `xzdroo` | dressed `<mn\|\|ie>`, `<mn\|\|ij>` | yes (W_vvvo, W_vooo, pass 1) | none as arrays; `soccct14` adds `P(ij) t_j^e <mn\|\|ie>` into W_mnij | no counterpart |
| `xzdrovvq` | dressed `<mb\|\|ef>` on e<f, in core | yes (pass 1, W_vvvo, W_vooo) | none: CFOUR keeps the bare `<ab\|\|ci>` on disk | no counterpart |
| `xzdrov` | dressed `<mb\|\|ej>`, `<mb\|\|ij>` | yes (W_ovvo, W_vooo) | the two singly dressed terms of `<mb\|\|ej>` are `soccct15eoc`, `soccct16`, added into W_mbej; the `t1 t1` term is inside `socccon1` together with t2 | terms exist, grouped differently |
| `xzt2forms` | T2 on pairs, tau1 = `A(t1 t1)`, t2 + tau1 | T2 on pairs yes | `t2tau` (tau = t2 + tau1) | pairs with / without diagonal |
| `xzp1ops` + GEMM | R1: `1/2 ~<ak\|\|cd> t_ikcd` | the operand is pass 1's | `socccon6oc` (bare, from the PPPH files) | - |
| `xzbv` | bra-dressed `<ab\|\|ei>`: the start of W_vvvo | yes | none in xx2cccsd (xx2chbar: `hbarabciccsd`) | no counterpart |
| `xzvvoo` | dressed `<ab\|\|ij>` | no | spread over `pphh`, `soccct17oc`, `soccct18`, `soccct19`, `socccon5oc`, the tau of the ladder | CCSD only |
| `xzweff` | W_ovvo = `~<ma\|\|di> + <mn\|\|de> t2`; F_vv, F_oo | yes: the ring operator of pass 2, the pp and hh operators | W_mbej (`socccon1` + `phhp` + `soccct15eoc` + `soccct16`) = `~<mb\|\|ej> + 1/2 <mn\|\|ef> t2`: **half** the T2 term | W_ovvo = 2 W_mbej - `~<mb\|\|ej>`: needs the dressed integral again, so nothing is saved |
| sweep: `xzdrrows`, `zc_vvvv_dress`, GEMM | rows of the dressed `<ab\|\|cd>`; CCSD ladder with t2 + tau1 | yes: W_vvvo (`~<ab\|\|ef> t1`), the pp ladder of T3 | `t2abcdso`: bare `<ab\|\|cd>`, one ZGEMM with tau; the T1 dressing of the ladder is `socccon5oc` and part of `soccct17oc` | CFOUR never has the dressed `<ab\|\|cd>` |
| `xzccsd`: W_klij | `~<kl\|\|ij> + 1/2 <kl\|\|ef> t2` | yes: the hh ladder of T3 | W_mnij (`socccon2` + `hhhh` + `soccct14`): **identical** | `whhhh(mn, ij)` with diagonal -> `wkl[K][P]` without: a gather of o^4 / 4 elements |
| `xzccsd`: hh term, F terms, ring, R1, the update | R2, R1, new T1 and T2 | no | `t2ijklso`, `t2ftaaso2`, `t2ftiiso2`, `t2rngso`, `r1ft`, `socccon7`, `t1rngso`, `r1updso`, `r2updso` | CCSD only |
| `xzwvoovo`, `xzwvx`, `zc_pair_t1_contract` | the rest of W_vvvo | T3 only | none in xx2cccsd | - |
| `xzwo`, `xzp2ops`, `xzo6` / `xzwpair` | W_vooo, operands of pass 2, the occupied operator | T3 only | none in xx2cccsd | - |

**What the structural difference implies.**  Of everything `hbarso` makes,
the T3 terms can use W_mnij and F_me as they are; W_mbej, F_ae and F_mi are
related to the native intermediates only through dressed integrals that the
native code must then form anyway; W_vvvo, W_vooo, the dressed `<ov||vv>`
and the dressed rows of `<ab||cd>` do not exist in xx2cccsd at all.  So a
CCSDT iteration that called `hbarso` for R1 and R2 would still do the T1
dressing, Bv, W_ovvo with F_vv and F_oo, W_klij and the dressed rows itself.
What `hbarso` could replace is what the "needed by T3 terms" column marks
"no": `xzvvoo`, the CCSD ladder GEMM, the R1 GEMM and the body of `xzccsd`
without W_klij - and the T3 parts of R1 and R2 (pass 1) would have to be
added to its result afterwards (`hbarso` returns the update R / D with its
masks applied; the T3 parts divided by D can be added, the equations are
linear in R).

## 3. The probe: `cfour/x2ccc_probe`

`x2cprobe` is a small CFOUR-style program (CRAPSI, the work array) that runs
where `xzccsdt` runs.  It links

- the native program without its main program (objects and modules of
  `build/xzccsdt`, read-only);
- from CFOUR's **program directory** `x2cccsd/`: `hbarso.o` and the 22
  objects it needs that are not in the library (`soccmem.o`, the six masks
  and `pgmaskcount0.o`, `t2abcdmoso.o`, `t2abcdso.o`, `t2ftaaso2.o`,
  `t2ftiiso2.o`, `fia2r1.o`, `r1updso.o`, `r2updso.o`, `t1normso.o`,
  `t2normso.o`, `ppphreadp.o`, `wppphfname.o`, `vreadp.o`, `vreadf.o`,
  `ztranspiv.o`);
- `libx2ccc.a`, `libx2cint.a` (for one module variable, `iconti`),
  `liblibr.a`, MKL.

`hbarso(work)` takes the work array and everything else from modules; this
is the complete list (the undefined symbols of `hbarso.o`, and what the
routines below it use):

- `soccmod`: `no`, `nu`, `nmo`, `ndrop`, `ndimt1`, `ndimt2`, `inonhf`,
  `ipgmask`, `ncvs`, `icvs`, `ppppao`, `iofftvv`; the tables `too`, `tvv`
  (allocated and filled by `settrioff`); the names of the PPPH files
  (allocated by `wppphfname`); the norms it returns;
- `soccmem`: the offsets `hhhh`, `pphh`, `phhp`, `hhpp`, `ppph`, `hhhp`,
  `pppp`, `oe`, `t1`, `t2`, `r1`, `r2`, `ftaa`, `ftii`, `ftia`, `ftaa2`,
  `ftii2`, `ftia2`, `whhhh`, `wphhp`, `tau`, `taut`, `tmp1`, `tmp2`, `tmp3`;
- `cvwint_module` (libx2c): `iconti`;
- files of the run directory read in every call: `PPPHnnnn`, `CONTOCCN`,
  `CONTOCCP`, `ZMAT`.

No COMMON block is involved.  xx2cccsd sets all of this in `setsocc`,
`eomccini` and `x2c_ccsddrv`, which could be called too, but they also read
the basis, compute or read the AO integrals and place the AO arrays of 1.5;
the probe sets the variables itself (`cfsetup`: 60 lines), places the arrays
in one region taken from the native stack, and calls CFOUR's `settrioff`,
`wppphfname`, `vreadf`, `vreadp` for the tables and the integrals
(`cfload`).  The AO-driven ladder (`t2abcdaoso`) is a stub: `ppppao` is
false, as `xzccsdt` requires `ABCDTYPE=STANDARD`.

The native side is `nccstep`: the calls of `xzstep` that the T1 and T2
equations need when there is no T3, the same routines of `xzimd.inc` and the
same kernels in the same order, on the integrals as `xzlayout` places them
(dense, complex).  With `PROBE_T3OPS=1` it also forms the intermediates that
only the T3 equation needs, timed apart.  Two builds: against the native
program of this branch ("base"), and - `PDEFS=-DXZ_NEWIMD` - against the one
of commit 85a86b0 of the branch `worktree-agent-af28b5132d71d4c7c` ("new":
its commit 6d73752 turned the o^4 v^2 and o^2 v^3 loops of `xzimd.inc` into
GEMMs and changed three argument lists), which is the native code that the
comparison should be made with.

What a run does: second-order T2, three Jacobi steps of the native CCSD
(amplitudes with T1 /= 0), then one step of each code on these amplitudes -
T1 and T2 copied into CFOUR's layout, `call hbarso(work)`, its `t + r`
against the native new T1 and T2 over all elements - and then the two steps
alternating, timed.

**Agreement** (largest absolute difference of the new amplitudes; the step
itself changes them by 3e-4 to 7e-4):

| case | o + v | T1 | T2 | where |
|---|---|---|---|---|
| H2O / 6-31G, X2C-1e | 10 + 16 | 1.6e-17 | 1.1e-16 | login node |
| HBr / cc-pVTZ-DK, 10 frozen | 26 + 78 | 9.6e-17 | 4.9e-16 | job 31781783, both native builds, 8 and 24 threads |
| the same, `%vircut` (18 dropped) | 26 + 60 | 1.4e-16 | 3.9e-16 | login node |
| HBr / cc-pVQZ-DK, 10 frozen, lowest 120 virtual spinors | 26 + 120 | 6.1e-17 | 2.9e-14 | job 31781783 |

The correlation energies of the two new sets of amplitudes agree to all 14
digits printed (26 + 78: -0.29262136870079).

    source /data/lcheng24/intel/oneapi/setvars.sh --force
    make -C cfour/xzccsdt && make -C cfour/x2ccc_probe      # -> build/x2ccc_probe/x2cprobe
    cd RUNDIR && OMP_NUM_THREADS=8 PROBE_REP=5 .../x2cprobe  # after xjoda ... xx2cvtran

## 4. Timing

One job (31781783, node c261, 2 x 24 cores Xeon Gold 6248R, `-c 48
--mem=180G`), run directories in the node-local `/tmp`, integral files of
CFOUR's chain (`ABCDTYPE=STANDARD`), threads bound to cores
(`OMP_PLACES=cores OMP_PROC_BIND=close`); HBr, X2C-1e, 10 spinors frozen;
complex, dense.  In each process the two codes alternate, five repetitions;
means (the least differs by less than 1 % everywhere).  `hbarso` was timed
in both processes of a thread count, next to each native build: the two
numbers agree to 0.1 %.

**For the same result** - new T1 and T2 from T1, T2 and the integrals:

| seconds per step | 26 + 78, 8 thr | 24 thr | 26 + 120, 8 thr | 24 thr |
|---|---|---|---|---|
| CFOUR `hbarso` (whole call) | 0.743 | 0.506 | 2.584 | 1.645 |
| native CCSD part, new build (85a86b0) | 1.003 | 0.485 | 3.625 | 1.698 |
| native CCSD part, this branch | 1.782 | 0.907 | 5.681 | 2.922 |
| `hbarso` / native new | 0.74 | 1.04 | 0.71 | 0.97 |
| native, the T3-only intermediates in addition (new) | 0.806 | 0.355 | 4.099 | 1.753 |

CFOUR's routine is faster at 8 threads (by 1.35x and 1.40x) and equal at 24;
it scales worse (1.47x and 1.57x from 8 to 24 threads; the native part 2.07x
and 2.13x) because of its serial loops around the PPPH files.  The native
routines of this branch, before 6d73752, are 1.7 to 1.8x slower than the
new ones.  Memory: the region of `hbarso` (its own copies of all integral
classes in its packing, amplitudes, intermediates, work) is 655 MB and
2025 MB, next to 2052 MB and 6096 MB of the native side of the probe.

The stages of the native CCSD part, new build, and what each is to the T3
terms (seconds):

| stage | for the T3 terms | 26 + 78, 8 thr | 24 thr | 26 + 120, 8 thr | 24 thr |
|---|---|---|---|---|---|
| T1 dressing | needed | 0.240 | 0.101 | 0.739 | 0.343 |
| pass-1 operands and the R1 GEMM | operands needed, GEMM not | 0.030 | 0.025 | 0.080 | 0.062 |
| bra-dressed `<ab\|\|ei>` (Bv) | needed | 0.099 | 0.070 | 0.310 | 0.179 |
| dressed `<ab\|\|ij>` | **CCSD only** | 0.027 | 0.011 | 0.077 | 0.030 |
| W_ovvo, F_vv, F_oo | needed | 0.162 | 0.071 | 0.542 | 0.233 |
| dressed `<ab\|\|cd>` rows | needed | 0.140 | 0.069 | 0.878 | 0.414 |
| CCSD pp ladder GEMM | **CCSD only** | 0.046 | 0.018 | 0.245 | 0.093 |
| `xzccsd` | **CCSD only** but W_klij | 0.260 | 0.121 | 0.754 | 0.344 |
| sum | | 1.003 | 0.485 | 3.625 | 1.698 |
| CCSD only, at most (rows 2, 4, 7, 8) | | 0.363 | 0.175 | 1.156 | 0.529 |
| `hbarso` in their place | | 0.743 | 0.506 | 2.584 | 1.645 |
| per iteration, if `hbarso` replaced them | | +0.38 | +0.33 | +1.43 | +1.12 |

(The conversions of T1, T2 and of the result between the layouts, and the
T3 parts of R1 and R2, are not in the last row.)

**Against a whole iteration.**  `xzccsdt ccsdt` (the new build) on the same
files in the same job, 26 + 78, dense, complex, three cycles without DIIS
(`XZ_NCYC=3`): 110.3 s per cycle with 8 threads, 45.0 s with 24 (pass 2 58 %, the pp
ladder of T3 22 %, pass 1 13 %).  The stages that only CCSD needs (dressed
`<ab||ij>`, the CCSD ladder GEMM, `xzccsd`) are 0.34 s and 0.15 s of that,
0.3 %; everything that is not a pass over T3 is 1.7 %.  26 + 120 was not
iterated here.  In blocks of the labels and real arithmetic an iteration is
much shorter (docs/factorization.md 2.8: the Python program, 26 + 120, 10.9 s
with 8 threads and 6.8 s with 24, of which intermediates 1.95 and 1.13 s),
and the dense part is then a real share of it; that mode was not run here.

**Why this is not the 2.7 - 6x of docs/benchmarks.md.**  That comparison is
of whole CCSD iterations of `zccsdt.zccsd.DirectZCCSD` and xx2cccsd with
`ABCDTYPE=AOBASIS`, where the AO-driven ladder dominates CFOUR's iteration.
`xzccsdt` requires `ABCDTYPE=STANDARD`; with `<ab||cd>` in core the ladder
is one ZGEMM (0.05 s of 0.74 s here) and the rest of `hbarso` is GEMM-bound
as the native code is.  Not measured here: xx2cccsd as a program (its
initialisation, DIIS, the files it writes every cycle).

## 5. The CCSD start: `XZ_START=ccsd`

Commit 85a86b0 ("xzccsdt: XZ_START=ccsd, the T1 and T2 of CFOUR's xx2cccsd
as the start") on the branch `worktree-agent-af28b5132d71d4c7c`, not merged
into this branch; read with `git show`, one file changed (`xzcc.inc`, +88).

- It reads the files `t1guess` and `t2guess` of the run directory, which
  xx2cccsd's `dmpccwf` writes after its last cycle (one record each,
  complex: `t1(a, i)` and `t2(ab, ij)` on pairs with the diagonal) - not
  `t1update` / `t2update`, which every cycle writes.  This is the layout of
  1.2; the gather is that of the probe's `cfput1`, in the other direction.
- Frozen and dropped spinors: nothing to do, the files hold the active
  spinors; the record lengths are checked against `no`, `nv` and a mismatch
  falls back to the second-order start with a warning.  The commit message
  lists a frozen + `%vircut` case as run.
- Labels: the amplitudes go through `operm`, `vperm` with the signs of the
  pairs, so the order of the classes is handled in the code.  Real
  arithmetic: an imaginary part above 1e-8 falls back to second order with
  a warning.  The memory of the two buffers is taken in the dry run, so the
  plan counts it.
- Tested, by the commit message, on the login node: the correlation energy
  of the start equals the CCSD energy that xx2cccsd prints (H2O / 6-31G, the
  frozen + vircut case, HBr / cc-pVDZ-DK); CCSDT then takes 17, 15, 17
  cycles instead of 17, 17, 18.  There is no regression test, no entry in
  docs/native.md, and no run with labels or in real arithmetic is reported.
  An inference from the source, not a run: on the labelled directories of
  `bench/cfour/labelled_run.py` the path cannot be exercised as it is,
  because xx2cccsd takes its Fock matrix and spinor energies from the
  JOBARC records of CFOUR's own SCF (`scf2ccmo`, `scf2cfck`, `scf2ceig` in
  `x2c_ccsddrv`), which belong to other spinors than the files written
  there.

So: complete for what it claims (an option, off by default), tested by hand
on three small unlabelled cases, and worth 0 to 2 of 17 cycles there.

## 6. Recommendation

**In the iteration: none of the CCSD part should call libx2ccc.**

1. *Time.*  For the same R1 and R2 `hbarso` is as fast as the native code
   or up to 1.4x faster (section 4) - but the native step cannot drop its
   dressing and intermediates, the T3 terms need them.  Replacing what is
   CCSD-only costs 0.33 to 0.38 s more per iteration at 26 + 78 and 1.1 to
   1.4 s more at 26 + 120.  Against a dense complex iteration (110 / 45 s
   at 26 + 78) that is nothing either way: there is no time to win, a little
   to lose.
2. *Labels and real arithmetic.*  CFOUR's routines are complex and dense
   (1.6).  Calling them from a real run in blocks means: T1 and T2 expanded
   to complex in CFOUR's packing every iteration, a second copy of every
   integral class, complex and dense (2.0 GB at 26 + 120, of which
   `<ab||cd>` 843 MB, where the native program in blocks holds an allowed
   part of it in real numbers), and a complex dense product where the
   native code does a real one.  In that mode an iteration is seconds and
   `hbarso`'s 1.6 to 2.6 s at 26 + 120 would be a quarter of it (against the
   Python program's 6.8 to 10.9 s; unmeasured for the native program).
3. *Code.*  What would go: `xzvvoo` (33 lines), `xzccsd` (125) and the
   ladder call, about 170 of the 927 lines of `xzimd.inc`.  What would come:
   the set-up and the conversions (the probe's `cfside.F90`: 226 lines with
   comments), the T3 parts added after the fact, and a link dependence on
   23 object files of a program directory (`hbarso` is not in the library),
   on module state made for xx2cccsd, on `libx2cint`, and on three files
   read in every call.  No saving.
4. *The merge.*  The native program already lives in CFOUR's world where it
   matters for a merge: its integral files, JOBARC, the work array of CRAPSI,
   `liblibr`.  Its internal layouts are those of its kernels; adopting
   CFOUR's packed-with-diagonal arrays for T2 would touch every kernel and
   buy nothing in the T3 terms, where the time is.

**Single terms: no.**  Both codes are GEMM-bound in the same products
(CFOUR's ring 0.13 s and W_mbej product 0.12 s against `xzccsd` 0.26 s and
W_ovvo with F 0.16 s at 26 + 78, 8 threads), and each call would need its
operands converted.

**What the measurement does say about the native code.**  (a) The routines
of `xzimd.inc` on this branch are 1.7 to 1.8x slower than those of 85a86b0:
that branch should be merged.  (b) With it, the native CCSD part is still
1.35 to 1.4x behind `hbarso` at 8 threads.  The stages to look at are the T1
dressing (0.24 s, of which `<ov||vo>`, `<ov||oo>` 0.18 s in the production
table) and `xzccsd` (0.26 s), which CFOUR does with ZGEMMs; in the dense
mode this is 0.3 % of an iteration, in blocks it may matter.

**The CCSD start.**  `XZ_START=ccsd` (section 5) is the one use of xx2cccsd
that costs nothing in the iteration; it is done, optional, and saved 0 to 2
of 17 cycles in the three small cases.  It needs a run of xx2cccsd and
CFOUR's own spinors.  The same start can be had without CFOUR: the probe's
`nccstep` is a CCSD step made of the native routines (about 130 lines of
calls), 1.0 / 0.5 s per step at 26 + 78 where a CCSDT step is 110 / 45 s,
and it would work with labels and in real arithmetic as those routines do.
Twenty such steps cost less than a fifth of one CCSDT cycle.  Whether it
saves cycles at this size was not measured; it is not in the production
program (`cfour/xzccsdt` was not to be changed here).

**What libx2ccc is good for here.**  As an independent reference: the probe
checks the native CCSD part against code that shares nothing with it but
the integral files, to 1e-15, at any size, in seconds.

## 7. Not done

- Single libx2ccc routines called on native arrays (for instance `t2rngso`
  on a converted T2 and W): only the whole residual was prototyped.  The
  per-term times of 1.3 and 4 are the basis of the statement that no term
  would gain.
- xx2chbar's W_abci / W_mbij against the native W_vvvo / W_vooo: read, not
  compared; their definitions (CCSD Hbar, in files per occupied spinor,
  AO-driven `<ab||ef>` part) were not checked against the T3 equation.
- Real arithmetic and label blocks in the probe: CFOUR's routines have
  neither; the native numbers for them are those of docs/native.md, not
  remeasured here.
- xx2cccsd as a whole program was not run or timed.
