# Using zccsdt

zccsdt follows the conventions of PySCF's coupled-cluster classes: build the
SCF with PySCF, hand it to a CCSDT class, call `kernel()` or `run()`.

```python
from pyscf import gto, scf
import zccsdt

mol = gto.M(atom='O 0 0 0; H 0 .757 .587; H 0 -.757 .587', basis='cc-pvtz', verbose=4)
mf = scf.RHF(mol).run()

mycc = zccsdt.CCSDT(mf)         # RCCSDT for RHF
mycc.set_frozen()               # chemical core
mycc.kernel()                   # -> e_corr, t1, t2, t3
mycc.ccsdt_q()                  # (Q), closed shell
print(mycc.e_tot, mycc.e_tot_q)
```

| Class | Reference | Engine |
|---|---|---|
| `zccsdt.RCCSDT` | RHF | spin-adapted closed-shell CCSDT on spatial orbitals (`zccsdt.sa.rccsdt`), (Q) |
| `zccsdt.UCCSDT` | UHF, ROHF | spin orbitals in spin blocks, U-form integrals (`zccsdt.eris_spin`) |
| `zccsdt.GCCSDT` | GHF, X2C spinors | spin orbitals or spinors without spin blocks, real or complex (`zccsdt.eris`), (Q) |
| `zccsdt.GCCSDTQ` | GHF, X2C spinors | CCSDTQ on spin orbitals or spinors, real or complex (`zccsdt.cc.ccsdtq`), (P) |

`zccsdt.CCSDT(mf)` and `zccsdt.CCSDTQ(mf)` pick the class from the type of
`mf`.  More examples are in `examples/`.

## Options

As in PySCF: `conv_tol` (energy, default 1e-10), `conv_tol_normt` (1e-7
for the residual norm of RCCSDT, 1e-8 for the step norm of the others),
`max_cycle` (100), `diis_space` (8, a maximum: see `max_disk`), `diis_file`, `incore_complete`,
`frozen` (core orbitals: spatial for RCCSDT, per spin for UCCSDT, spin
orbitals or spinors - or a list of them - for GCCSDT and GCCSDTQ), `verbose`,
`stdout`, `callback`.  In
addition:

- `max_memory`: MB for the whole process.  Unlike PySCF's objects the
  default is `None`, which means 80% of what the node and the memory limit
  of the job (the Slurm cgroup) leave, not 4000 MB.  What does not fit goes
  to scratch files (`ZCCSDT_TMPDIR`, else PySCF's TMPDIR); see Section 3 of
  `docs/factorization.md` and `docs/spin_adapted.md`.
- `max_disk`: MB for the scratch files of the process - the disk that the
  run may use in its scratch location, the counterpart of `max_memory`.
  The default `None` means the environment variable `ZCCSDT_MAX_DISK` (MB)
  if it is set (for tests, and for a disk that is shared), else the free
  space of the scratch directory (`statvfs` of `ZCCSDT_TMPDIR`, else
  PySCF's TMPDIR; of the directory of `diis_file` for the DIIS vectors)
  less a reserve of 2% (at most 1 GB).  It is never taken to be more than
  that free space.  See "Disk" below for what is done with it.
- `diis_space` is the number of DIIS vectors that is asked for: a maximum.
- `diis_t3`: T3 in the DIIS vectors.  The default `None` means yes, unless
  neither the disk nor the memory holds three vectors of it (then DIIS on T1
  and T2, T3 by its Jacobi step: a few more iterations, about a quarter for
  N2 and H2S / cc-pVTZ); `True` insists (an error then); `False` is DIIS on
  T1 and T2 only.  `diis_t4` of GCCSDTQ is the same for T4.
- `plan`: an explicit memory plan instead of the automatic one.

### Disk

Every scratch file takes its whole space when it is created
(`posix_fallocate`), so a disk that is too small is an `OSError` with the
numbers at that point and never a bus error (SIGBUS) hours later on a write
to a mapped array.  The solvers do not let it come to that: before the
first iteration they plan the disk (`zccsdt/disk.py`), as they plan the
memory.

1. The arrays that the memory plan maps to scratch files come first (T3
   or T4 and their residuals, the packed <ab||cd>, the kept rows of the
   ladder operator).  If `max_disk` does not hold them the run stops with
   a `DiskFullError` (an `OSError`, ENOSPC) before anything is allocated.
2. The DIIS vectors get the rest.  One stored vector is a pair - the
   amplitudes and their error vector - of p bytes for its large parts
   (2 T3 for CCSDT and its Lambda equations, 2 (T3 + T4) for CCSDTQ).  In
   this order of preference:
   - (a) all `diis_space` pairs in the scratch file (the default
     placement), if the disk budget holds them;
   - (b) a split: as many pairs in the file as the disk holds, the others
     in memory - in what `max_memory` leaves beyond the memory plan of the
     solver (nothing, if that plan had to map its own T3);
   - (c) a smaller subspace: n = floor(disk / p) + floor(memory / p) pairs,
     if n is at least 3;
   - (d) DIIS without T3 (T4 for CCSDTQ, l3 for the Lambda equations),
     which then takes its Jacobi step, if not even 3 pairs fit;
   - (e) an error before the first iteration, with what was needed and what
     is there - only with `diis_t3 = True` (no (d)), or when the mapped
     arrays of step 1 do not fit.
3. (c) and (d) are reductions that the log states with the numbers (the
   line `DIIS vectors` of the memory plan: asked for, used, bytes per pair,
   the two budgets); they are not errors, because `diis_space` is a
   maximum.  `incore_complete` is kept as it is asked for: all pairs in
   memory, whatever `max_memory`, and a `MemoryError` if the node does not
   hold them.  The plan that was used is `mycc.diis_plan_used`.

All file-backed DIIS vectors are created in the first iteration that stores
one, so a disk that another job has filled since the plan was made shows
then.  `DirectZCCSD` (CCSD and its Lambda equations) has `max_disk` too, in
a smaller form: where its memory plan has the DIIS vectors in the scratch
file, as many of `diis_space` are kept as the disk holds, at least 3 (T2
cannot be left out; below that the run stops before the iterations).  Not
planned ahead: its other scratch files (integrals, intermediates, (T)) and
those of the perturbative corrections (Q), (P) and their gradients - they
are checked when they are created (free space, `ZCCSDT_MAX_DISK`, the
`max_disk` of a CCSDT or CCSDTQ run that is under way), with the same kind
of message.  The integral files that PySCF writes (`lib.H5TmpFile` of an
out-of-core transformation) are neither counted nor preallocated; a full
disk is an error of HDF5 there, not a bus error.  A scratch directory on
tmpfs (`/dev/shm`) takes memory, not disk: its pages count against the
memory limit of the job, which this plan does not know.

`zccsdt.disk.dry_run(nocc, nvir, dtype, max_memory, max_disk)` gives the two
plans of a spin-orbital or spinor CCSDT calculation without integrals (T3 on
its unique block, no symmetry labels: the upper bound), for choosing the
memory and the local disk of a job before it is submitted.  Benzene /
cc-pVDZ with X2C, 30 occupied and 186 virtual spinors: the complex T3 takes
68.5 GB, a pair 137.1 GB, the 8 pairs of the default 1.1 TB.

```python
from zccsdt import disk
print(disk.describe(*disk.dry_run(30, 186, complex, max_memory=320000, max_disk=900000)))
print(disk.describe(*disk.dry_run(30, 186, complex, max_memory=1400000, max_disk=900000)))
```

```
memory plan  157.7 GB of memory, 0 bytes of disk: T3 in memory, its residual in memory
DIIS         7 vectors with T3, 137.1 GB per pair: 6 in a scratch file (822.6 GB of disk), 1 in memory (137.1 GB); reduced from the 8 asked for (diis_space): disk 900.0 GB, memory 162.3 GB; memory 137.1 GB, disk 822.6 GB
memory plan  157.7 GB of memory, 0 bytes of disk: T3 in memory, its residual in memory
DIIS         8 vectors with T3, 137.1 GB per pair: 6 in a scratch file (822.6 GB of disk), 2 in memory (274.2 GB); disk budget 900.0 GB; memory 274.2 GB, disk 822.6 GB
```

With 320 GB of memory and 900 GB of disk the run keeps 7 of the 8 vectors
(6 in the file, 1 in memory); with 1.4 TB of memory all 8 (6 in the file, 2
in memory).  These are plans, computed from the sizes; neither calculation
has been run.

Results: `converged`, `e_hf`, `e_corr`, `e_tot`, `cycles`, `t1`, `t2`,
`t3`, `plan_used`, `diis_plan_used`, `timings` (wall time per iteration), `peak_memory`, and
after `ccsdt_q()` `e_q`, `e_q5`, `e_tot_q`; for GCCSDTQ in addition `t4`,
`e_ccsdt` and `ccsdt_run`, `e_corr` and `e_tot` are those of CCSDTQ, and
after `ccsdtq_p()` `e_p`, `e_tot_p`.

## Output

Output goes through `pyscf.lib.logger` to `mycc.stdout` at the level
`mycc.verbose` (both from the `Mole` object, as in PySCF):

| verbose | printed |
|---|---|
| 3 (NOTE, PySCF's default) | header, input, memory plan, a table of the iterations, energies, diagnostics, largest amplitudes, timings by stage, peak memory |
| 4 (INFO) | in addition the scratch directory and the parts of each timing stage |
| 5 (DEBUG) | in addition the time of each stage in every iteration, and PySCF's timer line |
| 0 | nothing |

For H2O/cc-pVTZ with one frozen core orbital (4 threads; abridged):

```
==========================================================================================
 zccsdt RCCSDT: closed-shell CCSDT, spin-adapted, on spatial orbitals
==========================================================================================
 zccsdt 0.1.0 (5224e90)   PySCF 2.13.1   numpy 2.4.6   4 threads   c233   2026-10-02 19:46:40
------------------------------------------------------------------------------------------
 reference         RHF   E(SCF) = -76.057114083119   converged
 molecule          H2O   10 electrons   charge 0   2S = 0   basis cc-pvtz
 orbitals          58 functions; 4 occupied, 53 virtual spatial orbitals, 1 frozen
 convergence       |dE| < 1e-10 Eh and |R| < 1e-07; at most 100 iterations
 DIIS              8 vectors, with T3; large parts in a scratch file
 memory            budget 136.5 GB = 80% of 170.7 GB available (node and job limit)
 amplitudes        T1 212   T2 44,944   T3 2,382,032 (19.1 MB: 16 blocks of 53^3, i <= j <= k)
 memory plan       needs 433.3 MB next to the integrals: everything in memory
   T3, residual      memory, memory (19.1 MB each)
   <vv|vv>           bare form in memory, dressed form in memory (31.6 MB each)
   buffers           ladder 8 blocks per batch, k pass 19.1 MB slabs; kept between iterations
------------------------------------------------------------------------------------------
 iter           E_corr / Eh       dE / Eh       |R1|       |R2|       |R3|  DIIS   time / s  cores
 ----  --------------------  ------------  ---------  ---------  ---------  ----  ---------  -----
    1    -0.261889917538158   -3.6654e-04   5.46e-02   1.25e-01   1.29e-01     0       1.14    2.7
    2    -0.274166496575756   -1.2277e-02   8.86e-03   5.40e-02   1.76e-02     1       0.44    3.5
  ...
   16    -0.275159407616241    9.0714e-11   6.14e-09   2.97e-08   1.74e-08     8       0.45    3.5
------------------------------------------------------------------------------------------
 RCCSDT converged in 16 iterations
   E(SCF)                -76.057114083119
   E_corr(CCSDT)          -0.275159407616
   E(CCSDT)              -76.332273490735
   E_corr(same-spin)      -0.058919236440
   E_corr(oppo-spin)      -0.216240174128
 diagnostics       T1 0.0063   D1 0.0108   D2 0.1308
 largest amplitudes
   T1                0.00706381   5 -> 11
  ...
   T3                0.00167537   4 5 5 -> 10 11 11
------------------------------------------------------------------------------------------
 timings                                              wall / s    CPU / s  cores    share
 integrals                                                0.01       0.02    2.0
 RCCSDT iterations (16)                                   7.97      26.96    3.4  100.0 %
   pair pass: ring, T3 -> T1, T2, W                       2.41       8.65    3.6   30.3 %
   Jacobi step and DIIS                                   1.73       6.19    3.6   21.8 %
   pp ladder                                              1.21       4.39    3.6   15.2 %
   ...
 total                                                    7.98      26.98    3.4
 per iteration 0.45 s on average, the first 1.14 s (with the integral blocks)
 peak memory 801.6 MB anonymous (arrays; the plan counted 433.3 MB next to the integrals)
==========================================================================================
 (Q) correction: 19 unique occupied quadruples; work arrays 664.6 MB in the block algorithm, ...
   E[Q]                   -0.000153511684   (fifth order)
   E(Q)                   -0.000355455044
   E(CCSDT(Q))           -76.332628945779
   time              4.22 s wall, 15.19 s CPU (3.6 cores); peak memory 954.6 MB anonymous
==========================================================================================
```

- **The iteration table**: correlation energy, its change, the norms that
  decide convergence (residual norms for RCCSDT, step norms of the full
  T1, T2, T3 for the others), the number of DIIS vectors, the wall time of
  the iteration and the cores it kept busy (CPU time / wall time: well
  below the number of threads means waiting, e.g. for the scratch disk
  when memory is short).
- **Timings**: wall and CPU time of each stage of the iterations, booked by
  nested timers (`zccsdt.timing`), so that the stages add up to the
  iterations and their parts (verbose 4) to the stage.  "other" is what no
  stage timer covers.
- **Memory**: the plan says where T3, its residual and the `<vv|vv>`
  operator live and how the passes are sliced; the peak anonymous memory
  (sampled during the iterations) can be held against what the plan
  counted.
- **Amplitudes**: MO numbers count from 1 over all orbitals, frozen ones
  included; UCCSDT labels spin orbitals with a (alpha) and b (beta).

The functions underneath (`zccsdt.sa.rccsdt.kernel`, `zccsdt.cc.solver.kernel`)
keep their own interface for scripts and the benchmarks.

## CCSDT(Q), CCSDTQ and CCSDTQ(P) on spinors

```python
from pyscf import gto, x2c
import zccsdt

mol = gto.M(atom='Br 0 0 0; H 0 0 1.4145', basis='cc-pvdz-dk', verbose=4)
mf = x2c.UHF(mol).run()                 # spinors of the X2C Hamiltonian

mycc = zccsdt.CCSDT(mf, frozen=28)      # GCCSDT; frozen counts spinors
mycc.kernel()
e_q = mycc.ccsdt_q()                    # (Q): e_q, e_q5 ([Q]), e_tot_q

myq = zccsdt.CCSDTQ(mf, frozen=28)      # GCCSDTQ
myq.kernel()                            # CCSDT first, then CCSDTQ
print(myq.e_ccsdt, myq.e_tot)           # myq.ccsdt_run is the GCCSDT object of the first step
e_p = myq.ccsdtq_p()                    # (P): e_p, e_tot_p = E(CCSDTQ(P))
```

All of it works on spin orbitals or spinors without spin blocks, with real
or complex integrals.  T3 and T4 are stored on their unique blocks
(i<j<k(<l), a<b<c(<d)); the symmetry that is used is that of a linear
molecule (below).

`frozen` is a number - the lowest occupied spinors - or, as in PySCF, a list
of spinor indices: the occupied ones in it are frozen and the virtual ones
are left out of the calculation (and of the integral transformation).  For
a truncated virtual space:

```python
import numpy as np
order = np.argsort(mf.mo_energy)
nocc = int((mf.mo_occ > 0).sum())
frozen = list(order[:28]) + list(order[nocc + 16:])     # the core; all but the 16 lowest virtual spinors
myq = zccsdt.CCSDTQ(mf, frozen=frozen)
```

(cut between Kramers pairs, not through one).

**Linear molecules.**  For a linear molecule on the z axis the spinors can
be chosen as eigenfunctions of j_z with real coefficients, and then every
integral and amplitude is real.  The classes do that by themselves
(`real_spinors`, default `True`; `zccsdt.eris.axial_spinors`): each
degenerate level of the reference is rotated into its m_j components - a
Kramers pair into m_j and -m_j - and the report says so.  The energies are
the same and the arithmetic is two to three times faster.  It applies to
closed shells and to open shells whose SCF kept the symmetry of the axis
(the doublet of BeH from `pyscf.x2c.UHF` on the axis did not); for every
other molecule and orientation, or with `real_spinors = False`, the
spinors are taken as they are and the calculation is complex - the general
case, and the one most timings of `docs/benchmarks.md` are about.

With these spinors every one of them has a label that adds, 2 m_j, and an
integral or an amplitude vanishes unless the labels of its upper indices sum
to those of its lower ones.  The classes use that (`symmetry`, default
`True`): the spinors are put in the order of their labels, T3 and T4 are
stored and the equations of CCSDT and CCSDTQ are worked in the blocks that
conserve the label (`zccsdt.cc.t3engine`, `t4engine_sym`, `ccsdtq_sym`), and
(Q) and (P) form their products in those blocks (`quadruples_sym`,
`pentuples_sym`).  The report says so (`T4 ... in the blocks of the
labels`, `symmetry ... the products in blocks`); the energies are those of
the dense code, which `symmetry = False` runs.  Where the blocks do not fit
the memory, the dense code with its plans is taken.  Integrals in Kramers
pairs from `zccsdt.dirac` carry the two labels of C2 (modulo 4): (Q) and (P)
use them, the iterations do not.  What is not used: the pairing of the
amplitudes of a closed shell under time reversal, and the point groups of
other molecules (their spinors would have to come from an SCF that keeps the
symmetry).

- **(Q)** (`zccsdt.cc.quadruples`) is the correction of Bomble, Stanton,
  Kallay and Gauss for canonical Hartree-Fock orbitals: `e_q5` is its
  fifth-order part, CCSDT[Q].  It is computed one occupied quadruple at a
  time; `max_memory` decides how many columns of the products are formed
  at once.
- **CCSDTQ** (`zccsdt.cc.ccsdtq`, `zccsdt.cc.t4engine`): `kernel()` runs
  CCSDT and continues from its amplitudes; `kernel(t1, t2, t3, t4)` starts
  from the amplitudes given.  The report has the size of T4, the memory
  plan (the batches of the products of the T4 equation; the DIIS vectors
  of T4 in memory or in the scratch file), a column `|dT4|` in the table
  of the iterations and the stages of the T4 equation in the timings.
- **(P)** (`zccsdt.cc.pentuples`) is the correction of CCSDTQ(P) as MRCC
  computes it for a Hartree-Fock reference (its CCSDTQ[P] is the same
  number), for canonical orbitals: one occupied pentuple at a time, nothing
  of the size of T5 is kept.  It needs the second-order quadruples (an
  array as large as T4) and less memory than the CCSDTQ iterations; with
  too little memory that array and the six-index intermediate go to a
  scratch file.
- An RHF, UHF or ROHF reference given to `zccsdt.CCSDTQ` is taken as GHF
  spin orbitals (there is no spin-adapted CCSDTQ here); `frozen` then
  counts orbitals per spin, as in the CCSDT class of that reference.
- What does not fit: the plans of the three steps follow `max_memory` (the
  products in more batches, DIIS vectors in the scratch file,
  intermediates for a few indices at a time).  If T4, its residual and the
  six-index intermediate of the T4 equation do not fit either, they become
  arrays of a scratch file - T4 first, then the intermediate, then the
  residual - which the operating system pages in and out: slow, but the
  calculation runs; the report says which.  Below the floor (the batches
  of the products at their smallest) it stops with a MemoryError that says
  how much is missing.

### Lambda, relaxed densities, CCSDT(Q)_Lambda

```python
mycc = zccsdt.CCSDT(mf, frozen=28)      # GCCSDT, as above
mycc.kernel()
mycc.solve_lambda()                     # mycc.l1, l2, l3; mycc.converged_lambda
dm = mycc.make_rdm1(ao_repr=True)       # relaxed density, <q+ p>, frozen spinors included
dip = -numpy.einsum('xpq,qp->x', mol.intor('int1e_r_spinor'), dm).real
e_ql = mycc.ccsdt_q_lambda()            # (Q)_Lambda: e_ql, e_ql5 ([Q]_Lambda), e_tot_ql
```

- `solve_lambda()` solves the left-hand equations of CCSDT on packed
  amplitudes (`zccsdt.cc.lambda_opt`): Lambda_3 is stored like T3, an
  iteration takes 1.15 to 1.26 times an amplitude iteration, and the memory
  follows `max_memory` like the amplitude equations.  Where the iterations
  ran in the blocks of the symmetry labels (a linear molecule on the z
  axis, `symmetry`), Lambda is solved in the blocks too
  (`zccsdt.cc.lambda_blk`; `mycc.lambda_blocks`): 3.3 and 8.5 times faster
  per iteration than on the packed block for HBr with 8 + 78 and 8 + 160
  spinors (docs/lambda.md section 9.7).  `mycc.l3` has the layout of
  `mycc.t3` - the blocks then;
  `zccsdt.cc.lambda_opt.packed_l3(mycc._eris, mycc.l3)` puts it on the
  packed block `[i<j<k, a<b<c]`.
- `make_rdm1(relaxed=True)` is the density whose trace with a one-electron
  operator is the derivative of the CCSDT energy: the densities of t and
  Lambda, the orbital gradient and the Z-vector equations for the response of
  the spinors (`zccsdt.cc.density_opt`), with frozen occupied and left-out
  virtual spinors.  All of it costs one to two CCSDT iterations (HI with 36 frozen and 116
  left-out spinors: 1.3; docs/lambda.md section 10).
  `relaxed=False` gives the density of t and Lambda at fixed spinors;
  without `ao_repr` the matrix is in the spinors `mycc.rdm1_basis`.  It
  needs the SCF object (not integrals that were given) and a tightly
  converged SCF (`mf.conv_tol_grad` 1e-8 or below).  `tol` (1e-11) is the
  threshold of the Z-vector equations, whose error enters the property to
  first order: with 1e-9 the dipole moment of HF / 6-31G was off by 7e-10
  (five-point finite fields at three step sizes, extrapolated, against the
  analytic value at 1e-12: 1e-11).
- `ccsdt_q_lambda()` is (Q) with Lambda_2 and Lambda_3 on the left
  (`zccsdt.cc.quadruples_lambda`; in the sectors of the labels,
  `quadruples_lambda_sym`, where the integrals have them): 1.3 to 1.4 times
  the time of (Q).  `zccsdt.CCSDTQ` passes it on to its CCSDT step.
- For a linear molecule with `symmetry` the Lambda equations and the
  densities run in the blocks of the labels (`lambda_blk`, `density_blk`)
  where C(o,3) C(v,3) is at least `mycc.blocks_min` (800,000: below that
  the blocks are slower), else on the packed block; `mycc.l3` has the
  layout of `mycc.t3` either way, and `lambda_blocks`, `rdm1_blocks` say
  what ran.  HBr / cc-pVQZ-DK, 8 + 160 spinors, from the spinors to the
  relaxed density: 194 s with the labels, 645 s without.  (Q)_Lambda takes
  T3 and Lambda_3 packed and forms its products in the sectors.
- **Properties of CCSDT(Q)**: `mycc.solve_lambda(with_q=True)` and
  `mycc.make_rdm1(with_q=True)` - the Lambda equations of CCSDT with the
  derivatives of E(Q) as sources, and the relaxed density with the parts
  of (Q) (`zccsdt.cc.quadruples_grad`; docs/lambda.md section 11), frozen
  and left-out spinors included.  The pass over the quadruples takes about
  three times the (Q) energy on the packed block and about twice in the
  sectors of the labels (`quadruples_grad_sym`, `mycc.q_gradient_sectors`);
  the Lambda iterations cost what they cost without (Q).  With little
  memory the source of Lambda_3 and the vvvv density of (Q) go to a
  scratch file.
  H2O / cc-pVTZ, the dipole moment against CFOUR's `ncc`
  (`CALC=CCSDT(Q)`, `PROPS=FIRST_ORDER`): 0.7492563653 in both.
- An RHF reference makes `zccsdt.CCSDT` the closed-shell class, which has
  none of this: `zccsdt.CCSDT(mf.to_ghf())` for these methods.

### Integrals from another program, and for another program

The classes for spin orbitals and spinors also run on an integral container
that was not made from a PySCF object:

```python
from zccsdt import dirac
eris = dirac.from_dirac('MRCONEE', 'MDCINT')     # the MO integrals of a DIRAC calculation
myq = zccsdt.GCCSDTQ.from_integrals(eris).run()
```

`zccsdt.dirac` reads the two files that DIRAC's integral transformation
leaves (`**MOLTRA`, `.4INDEX`; real and complex groups) and writes them:

```python
x = dirac.integrals(mf, frozen=28, nvir=16)      # PySCF spinors in Kramers pairs (psi, T psi)
dirac.write('run', x)                            # run/MRCONEE, run/MDCINT
eris = dirac.container(x)                        # the same integrals for the classes here
```

That is the route to an independent reference for the high-order methods
on spinors: MRCC reads DIRAC's files (`iface=dirac`) and does CCSDT(Q),
CCSDTQ and CCSDTQ(P) on them, for real integrals.  `dirac.integrals()`
gives real integrals for an RHF reference taken as GHF spin orbitals and
for the X2C spinors of a linear molecule on the z axis (it adapts them to
m_j); `bench/dirac/write_integrals.py`, `bench/dirac/mrcc_rel.sh` and
`bench/dirac/run_integrals.py` run the two programs on the same files.
The energies agree to 1e-12 Eh, and the same molecule turned away from the
axis - complex spinors, which MRCC cannot take - gives the same correlation
energies (Section "CCSDT(Q), CCSDTQ and CCSDTQ(P) on spinors" of
`docs/benchmarks.md`).

`zccsdt.eris.drop_virtuals(eris, nvir)` cuts a container down to its lowest
virtual spinors.

#### CFOUR's two-component programs (X2CSO)

`zccsdt.cfour` runs the high-order methods on the spinors of CFOUR's
spin-orbit X2C programs, whose own coupled cluster ends at CCSD(T).  CFOUR
does the SCF and the integral transformation; the ZMAT of the run says what
is done here.

```python
from zccsdt import cfour
eris = cfour.from_cfour('rundir')                # the MO integrals xx2cvtran left there
myq = zccsdt.GCCSDTQ.from_integrals(eris).run()
```

or, without writing Python:

```
python -m zccsdt.cfour RUNDIR                    # the CC step on a finished CFOUR run
python -m zccsdt.cfour --chain [--scratch /tmp] RUNDIR     # CFOUR's programs first
```

**The CFOUR side.**  The two-component programs are not chained by
`xcfour`; in a directory with `ZMAT` and `GENBAS` they are called one after
the other:

```
xjoda; [xx2camf;] xsox2catm; xsox2chf; xx2cvtran
```

(`xx2camf` only for X2CAMF, i.e. a `%amfiMethod` section in the ZMAT).
`--chain` does exactly that - the list comes from the ZMAT, `--programs`
overrides it, the executables are taken from `--bin`, else `$CFOUR_BIN`,
else `~/apps/cfour-x2cso/bin` - and checks every program before the next
one starts: exit status, the last line of a CFOUR program (`@CHECKOUT-I`;
`xx2camf FINISHED!`), no "Something very wrong", and the files it has to
leave.  A failure stops the chain with the name of the program and the end
of its output.  Each program writes `<program>.out` and `<program>.time`.
With `--scratch DIR` the input files are copied to a new directory under
`DIR` (node-local disk for real runs), everything runs there, the outputs
come back to `RUNDIR` and the directory is removed (`--keep`: not).
`bench/cfour/slurm_x2ccc.sh` runs CFOUR through the same code.

Nothing has to be told to CFOUR to keep the files: its programs leave
everything in the directory they run in.  What is read: `JOBARC` and
`JAINDX` (SCF energy, frozen pairs, dropped virtual spinors, joda's flags),
`nmototal`, `scf2ceig`, `dgcosym`, `moinhhhh`, `moinhhhp`, `moinhhpp`,
`moinpphh`, `moinphhp`, `PPPH0001` ... (one per occupied spinor) and
`moinpppp`.  The ZMAT needs

- `ABCDTYPE=STANDARD`: only then does `xx2cvtran` transform and write
  `<ab||cd>` (`moinpppp`); with `ABCDTYPE=AOBASIS` CFOUR's CCSD contracts
  the AO integrals and the file is never made.  It needs an `xx2cvtran`
  built from sources that contain `x2cvtran/vsort.f90` and `vtranso2.f90`
  (the x2cso branch since August 2026; `~/apps/cfour-x2cso/bin` was rebuilt
  from it on 2026-10-07): an older binary takes the AO path whatever
  `ABCDTYPE` says, and `from_cfour()` then stops with a message that says
  so.  Checked with the rebuilt programs for H2O / STO-3G: `<ab||cd>` of the
  file against PySCF's integrals over CFOUR's spinors to 6e-15, and CCSD
  solved here on the files equal to `xx2cccsd` (-0.049502919581 Eh).
- `SYM=OFF`, `RELATIVISTIC=X2C1E`, `X2CKEY=SOATMX`, `BASIS=SPECIAL` with
  one line per atom after the keywords, as for CFOUR's own CCSD.
- frozen core as `DROPMO=1>n` (n Kramers pairs), virtual spinors above an
  energy dropped with a section `%vircut`: both are CFOUR's business - the
  integral files hold the active spinors only, and how many were frozen
  and dropped is read from what CFOUR wrote, not from the ZMAT.

**What the ZMAT says to the CC step.**

| ZMAT | here |
|---|---|
| `CALC=CCSD`, `CCSDT`, `CCSDT[Q]`, `CCSDT(Q)`, `CCSDTQ` | the method (joda accepts all of these in a two-component input, and the chain up to `xx2cvtran` runs with each of them) |
| `CALC=CC(n-1)(n)` with `EXCITATION=4`, `5`; `CALC=CC(n)` with `EXCITATION=2`, `3`, `4` | CCSDT(Q), CCSDTQ(P); CCSD, CCSDT, CCSDTQ (CFOUR's general names, `EXCITATION` as joda stored it) |
| `CC_CONV=N` | `conv_tol_normt = 1e-N`, `conv_tol = 1e-(N+2)` |
| `CC_MAXCYC` | `max_cycle` |
| `MEMORY_SIZE`, `MEM_UNIT` | `max_memory` (CFOUR counts 1 GB as 1e9 bytes) |
| section `%zccsdt` | `method=`, `max_memory=` (MB), `max_disk=` (MB, CCSDT and beyond), `conv_tol=`, `conv_tol_normt=`, `max_cycle=`: one per line, they override the keywords |

The thresholds and the memory are taken as joda understood them (the
flags in `JOBARC`), so abbreviated keywords and defaults are CFOUR's.
`CCSDTQ(P)` is not a `CALC` of CFOUR by that name: joda warns and falls back
to `CALC=SCF` (the chain still makes the integrals).  It is
`CALC=CC(n-1)(n),EXCITATION=5`, or it is asked for in the section, which
CFOUR ignores like every section it does not know - and which is the only
place for `method=ccsdt(q)_l`, CCSDT(Q)_Lambda, that this joda has no
`CALC` for:

```
*CFOUR(CALC=CCSDTQ,BASIS=SPECIAL
...
MEMORY_SIZE=24,MEM_UNIT=GB)

O:PVDZ
H:PVDZ
H:PVDZ

%zccsdt
method=ccsdtq(p)
max_memory=20000

```

(a section ends at an empty line).  `--method` and `--max-memory` on the
command line override both.  `--jobarc` writes the final total energy into
`JOBARC` as `TOTENERG`, the record that CFOUR's own `xx2cccsd` leaves
there for joda.

**CCSD, CCSD(T) and properties on CFOUR's spinors.**  CCSD and CCSD(T) do
not read the integral files: where the directory has `scf2ccmo`, `MOL`,
`GENBAS` and `ZMAT` (after `xsox2chf`) the AO-driven code of
`zccsdt.zccsd` runs on CFOUR's spinors, so that `xx2cvtran` is not needed:

```
python -m zccsdt.cfour --chain --programs "xjoda xsox2catm xsox2chf" RUNDIR
```

(`--mo-integrals`: CCSD on the files of `xx2cvtran` instead).  Everything
is CFOUR's - geometry, basis, spinors, their energies, occupation, frozen
and dropped spinors - except the two-electron AO integrals, which libcint
makes for the same basis; the one-electron Hamiltonian is taken back from
CFOUR's Fock matrix and spinors (h = S C e C^+ S - G(D)), so that X2C-1e and
X2CAMF both come out as CFOUR has them.  No SCF is done here.

With `--properties` (or `--density`, or `PROPS` in the ZMAT) the Lambda
equations and the Z-vector equations follow, the relaxed density is written
in CFOUR's spinors as the file `denfinal` - the full density, the SCF part
included, `D[p,q] = <q+ p>` - and `xsox2cprop` evaluates all its operators
with it:

```
python -m zccsdt.cfour --chain --programs "xjoda xsox2catm xsox2chf" --properties RUNDIR
```

for `CALC=CCSD`, `CCSD(T)`, `CCSDT` and `CCSDT(Q)` (the last two through
`zccsdt.CCSDT` on integrals transformed here).  H2O / 6-31G, X2C-1e, against
CFOUR's own chain (`xx2cccsd`, `xx2cccdenocbp`, `xx2czvec`, `xsox2cprop`):
the density differs from CFOUR's `denfinal` by at most 1.7e-10 for CCSD and
1.6e-11 for CCSD(T), the dipole moment that `xsox2cprop` prints is
0.01580313752067 against 0.01580313752033 (CCSD) and 0.01227486493503
against 0.01227486493556 (CCSD(T)).  CFOUR has no CCSDT density to compare
with: that one (0.012077140) rests on finite fields.  With a frozen core
(`DROPMO=1>1`, the same molecule): the energies of CCSD and CCSD(T) equal
CFOUR's to all digits printed, the densities differ by at most 4.3e-12 and
8.8e-12.  Not checked through this route: `%vircut` (the density code has
it; with most virtual spinors dropped CFOUR's own relaxed density is not
the derivative of its energy, see below).

**Conventions.**  The spinors are those of CFOUR's list `scf2ceig` without
the frozen and the dropped ones, in its order (occupied first, each set by
energy); `eris.spinors` are their numbers in that list.  The files hold
antisymmetrized integrals `<pq||rs> = (pr|qs) - (ps|qr)`, complex, each
block once (`<ij||kl>`, `<ij||ka>`, `<ab||ij>`, `<ab||ci>`, `<ab||cd>` on
pairs p<=q; `<ij||ab>` and `<ia||bj>` in full); `<ia||bc>` and the other
triangles are completed from the antisymmetry and
`<pq||rs> = <rs||pq>^*`.  There is no one-electron matrix: the spinors are
canonical, the Fock matrix is the diagonal of their energies, and the SCF
energy is the one in `JOBARC`.

**Checks.**  `bench/cfour/check_integrals.py RUNDIR BASIS` compares every
block that is in the files, element by element, with the integrals PySCF
makes for CFOUR's own spinors (`scf2ccmo`), and - with the amplitudes
`xx2cccsd` converged to - the CCSD energy, the (T) correction and the T1
equation with CFOUR's numbers; none of this needs `<ab||cd>`.  For H2O
(STO-3G and 6-31G; all electrons, and with a frozen pair and two dropped
virtual spinors; X2C-1e and X2CAMF) and HBr / cc-pVDZ-DK with 28 frozen
spinors: integrals to 1e-14, the energies to 1e-15 Eh, the T1 residual
below 1e-12.  `moinpppp` has not been read from a file CFOUR wrote (see
above): its layout is the one of `vsort.f90`, and is tested only against
`cfour.write()`, the writer of the same files in `tests/test_cfour.py`.

## Spinor CCSD(T): `zccsdt.zccsd.DirectZCCSD`

The two-component CCSD(T) that started from socutils' `DirectZCCSD` (pp
ladder in the scalar AO basis, no `<vv||vv>`). It is a PySCF CCSD object:

```python
from pyscf import gto, x2c
from zccsdt.zccsd import DirectZCCSD

mol = gto.M(atom='O 0 0 0; H 0 .757 .587; H 0 -.757 .587', basis='cc-pvtz',
            verbose=4, max_memory=24000)
mf = x2c.UHF(mol).run()         # any spinor SCF of PySCF or socutils

mycc = DirectZCCSD(mf, frozen=2)
mycc.kernel()                   # -> e_corr, t1, t2
e_t = mycc.ccsd_t()             # (T)
```

PySCF's settings apply (`conv_tol`, `conv_tol_normt`, `max_cycle`,
`diis_space`, `diis_file`, `incore_complete`, `level_shift`,
`iterative_damping`, `frozen`, `verbose`). In addition:

- `max_memory` is PySCF's: MB for the whole process, **4000 by default**
  (from the `Mole` object). The integrals, the CCSD step and (T) are sliced
  to stay within it; what does not fit goes to scratch files (`ZCCSDT_TMPDIR`,
  else PySCF's TMPDIR). A calculation that has the memory should be given
  it: benzene/cc-pVDZ takes 9.4 s per iteration with 24000 MB and 13.0 s
  with 4000 MB on the same node.
- `kramers` (default `None`): for a time-reversal symmetric reference (a
  closed shell without a magnetic field) the spinors are rotated within
  their degenerate levels into Kramers pairs and the symmetry of the
  amplitudes halves the expensive terms of CCSD and the triples of (T).
  **`t1` and `t2` then refer to the paired spinors `mycc.eris.mo_coeff`,
  not to `mf.mo_coeff`**; the energies do not depend on it. `False`: the
  spinors of the SCF as they are. `True`: an error if the reference has no
  Kramers pairs (with `None` it runs without them and says so in the log).
- `three_m` (default `None`): three real GEMMs instead of a ZGEMM for the
  large complex products where that pays and the memory is there; `False`:
  ZGEMM.
- `ao_ladder` (default `None`): the two AO operators of the pp ladder, 2 N^4
  bytes each for N scalar AO functions as square matrices and N^4 as
  triangles (0.7 GB and 0.35 GB together at N = 115, 250 and 125 GB at
  N = 500). `None`: kept in memory if they fit into the share of
  `max_memory` for stored integrals and the CCSD step fits next to them,
  as squares or in thinner parts of their rows; else `'direct'`: made from
  the AO integrals in every iteration, about N^4/8 integrals for every
  block of pairs (integral-direct; nothing of size N^4 is stored).
  `'memory'` and `'direct'` force one or the other; `'file'` keeps the
  triangles in a scratch file.
- `socutils_amps = True`, `ovvv_direct = True`, `ladder = 'chol'`: the
  routes copied from socutils (einsums on full arrays, PySCF's driver).

Lambda and densities:

```python
l1, l2 = mycc.solve_lambda()            # CCSD Lambda: mycc.l1[i,a], mycc.l2[i,j,a,b]
dm1 = mycc.make_rdm1()                  # density of t and lambda at fixed spinors, <q+ p>
mu = mycc.dip_moment()                  # its dipole moment, Debye (unit='AU')

mycc.solve_lambda(with_t=True)          # CCSD(T): with the (T) terms; E(T) is in mycc.e_t
mycc.make_rdm1(with_t=True)
mycc.dip_moment(relaxed=True)           # with orbital relaxation (CCSD; Z-vector equations)
mycc.dip_moment(with_t=True, relaxed=True)      # CCSD(T) with orbital relaxation
```

The Lambda equations run on the integrals and within the memory plan of the
CCSD, with Kramers pairs where the CCSD has them; an iteration takes 1.3
times a CCSD iteration. Like `t1` and `t2`, `l1`, `l2` and `make_rdm1()`
refer to the spinors `mycc.eris.mo_coeff`: an expectation value is
`einsum('pq,qp', V, dm1)` with V in that basis, or with
`make_rdm1(ao_repr=True)` (frozen spinors included) and V in the 2c AO
basis. The dipole operator has the picture-change transformation of X2C
(`picture_change=False`: the bare operator). With `with_t` the (T) terms of
the equations and of the density come from the batches of the (T) code, in
2.5 times the time of the (T) energy and within `max_memory`; that run gives
E(T) as well, so `ccsd_t()` need not be called before it.

`relaxed=True` adds the response of the spinors (`zccsd/zvector.py`): the
orbital gradient of the CCSD Lagrangian is evaluated in the layouts of the
amplitude step and within `max_memory` (`zccsd/orbgrad.py`: about three
CCSD iterations), the Z-vector equations take a few Fock builds. The result
is the density in the basis of all spinors of the SCF (`ao_repr=True`: in
the 2c AO basis). Frozen spinors, occupied and virtual, are allowed: the
blocks of the gradient between a frozen and an active spinor come from the
integrals with one frozen index (`zccsd/orbgrad_frozen.py`: an integral
transformation with the frozen spinors as the first index, contracted slab
by slab within `max_memory`; with Kramers pairs half of them).
A relaxed density is the derivative of the energy only for the spinors of
a converged SCF, and it has what is left of the orbital gradient in first
order: PySCF stops at `conv_tol_grad`, by default sqrt(conv_tol), which left
3e-8 in the Fock matrix of H2O/cc-pVDZ and 5e-8 in its dipole moments. Set
`mf.conv_tol_grad` (1e-8 or below) for such properties; `relaxed=True`
warns if the occupied-virtual block of the Fock matrix exceeds 1e-7.

`relaxed=True` together with `with_t=True` is the relaxed density of
CCSD(T), the derivative of E(SCF) + E(CCSD) + E(T). Call it on the CCSD
object as it is after `kernel()`: one pass over the triples then gives E(T),
the sources of Lambda and the densities of (T) for the orbital gradient
(`triples.kernel` with `terms='gradient'`), the Lambda equations of CCSD(T)
are solved, and the gradient gets its (T) terms (`zccsd/orbgrad_t.py`).
After `solve_lambda(with_t=True)` lambda is reused, but the pass over the
triples is done once more for the densities; `solve_lambda(with_t='gradient')`
keeps them. They are released when the relaxed density is made: the density
of <bc||ek> has 8 o v^3 bytes (in memory if it takes no more than a quarter
of what is left, else in a scratch file). Frozen occupied spinors are
allowed. The relaxed density also has elements between different occupied
and between different virtual spinors - (T) depends on the orbital
energies -, except between spinors closer in energy than 1e-9
(`zvector.DEGENERATE`: degenerate spinors, between which no rotation
changes the energy).

**Heavy elements, other properties, doublets.** The SCF objects of
socutils work as they are, e.g. X2CAMF with Gaussian nuclei:

```python
from socutils.scf import spinor_hf

mol = gto.M(atom='I 0 0 0; H 0 0 1.60', basis={'I': 'dyallv3z', 'H': 'unc-ccpvtz'},
            nucmod='G', max_memory=48000)
mf = spinor_hf.SCF(mol).x2camf(with_gaunt=True, with_breit=True)
# (older socutils: mf = spinor_hf.SpinorSCF(mol);
#  mf.with_x2c = socutils.somf.amf.SpinorX2CAMFHelper(mol, with_gaunt=True, with_breit=True))
mf.conv_tol_grad = 1e-8
mf.kernel()
mycc = DirectZCCSD(mf, frozen=36)                       # the 36 lowest spinors
mycc.kernel()
dm = mycc.make_rdm1(with_t=True, relaxed=True, ao_repr=True)    # relaxed CCSD(T)
value = numpy.einsum('pq,qp', operator, dm).real        # any one-electron operator
```

- `frozen` may be a list of spinors, also of virtual ones (those above an
  orbital energy, say: CFOUR's `%vircut`), for the energies and for the
  relaxed density alike. The frozen spinors cost the relaxed density an
  integral transformation with them as the first index: for HI (54 occupied
  spinors, 18 of them correlated, 378 spinors) 26 s of its 72 s, and 434 s
  after the CCSD in all, on 16 cores. Freeze whole levels: a frozen spinor
  that is degenerate with an active one of its kind leaves the relaxed
  density undefined (a warning).
- With most virtual spinors dropped, CFOUR's own relaxed properties are not
  the derivative of its energy (`docs/benchmarks.md`, "Frozen virtual
  spinors"): a difference from CFOUR with an aggressive `%vircut` is to be
  checked with finite differences before it is taken for an error here.
- An `operator` is a matrix in the 2c AO basis of the SCF (the j-adapted
  spinors of PySCF). `dip_moment` has the dipole operator with PySCF's
  picture change; an operator at a heavy nucleus should also have the
  response of the X2C decoupling (0.4 % of an electric field gradient),
  which socutils' property code has and CFOUR's files have:
  `bench/cfour/props.py` reads the latter with `socutils.prop.cfour` and
  compares all properties of CFOUR's `xsox2cprop` (docs/benchmarks.md,
  "Heavy elements").
- A doublet: the properties that are odd under time reversal (effective
  electric field, hyperfine and g operators, ...) are those of the state
  with the spin on the axis of a linear molecule. The SCF has to be in it:
  a state with the spin off the axis is, with spin-orbit coupling, not a
  solution, the Z-vector equations are nearly singular there, and the
  relaxed density means nothing (the g operator shows the direction of the
  spin). `bench/spinors.axial` takes the off-axis part out of a density
  (the molecule on the z axis), from which the SCF then converges on the
  axis.

At `verbose >= 4` the log says where the integrals are (memory or scratch
file), how the step is sliced, which BLAS does the matrix products and
whether Kramers pairs are used.

**Libraries.** The compiled parts are built with `cmake -S csrc -B build &&
cmake --build build`:

- `libnrr_opt.so`, the kernels of the integral transformation, is needed
  (an `ImportError` says how to build it). It links the libraries of the
  PySCF it was built against, with whatever BLAS they use.
- The matrix products go through `zccsdt.blas.gemm`: the MKL build of the
  kernel library if it is there (`-DZC_MKL_DIR=<MKL lib directory>`; one
  pool of threads with the kernels as long as the process holds one OpenMP
  runtime, which the log names), else its default build
  (OpenBLAS, or `-DZC_BLAS_LIB=<a CBLAS library>`), else SciPy's BLAS.
  `ZCCSDT_T3_LIB=<path>` picks one. All three give the same results; with
  OpenBLAS H2O/cc-pVQZ takes the same time and benzene/cc-pVDZ 1.4 times
  as long per CCSD iteration and 1.2 times in (T) (docs/benchmarks.md). A
  warning is logged when the products do not run on MKL.
- Without `libzc_zccsd.so` (the OpenMP kernels of the CCSD step and of (T))
  numpy does the same on one thread: for tests and small molecules.
