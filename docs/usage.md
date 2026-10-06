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
| `zccsdt.GCCSDT` | GHF, X2C spinors | spin orbitals or spinors without spin blocks, real or complex (`zccsdt.eris`) |

`zccsdt.CCSDT(mf)` picks the class from the type of `mf`.  More examples are
in `examples/`.

## Options

As in PySCF: `conv_tol` (energy, default 1e-10), `conv_tol_normt` (1e-7
for the residual norm of RCCSDT, 1e-8 for the step norm of the others),
`max_cycle` (100), `diis_space` (8), `diis_file`, `incore_complete`,
`frozen` (core orbitals: spatial for RCCSDT, per spin for UCCSDT, spin
orbitals or spinors for GCCSDT), `verbose`, `stdout`, `callback`.  In
addition:

- `max_memory`: MB for the whole process.  Unlike PySCF's objects the
  default is `None`, which means 80% of what the node and the memory limit
  of the job (the Slurm cgroup) leave, not 4000 MB.  What does not fit goes
  to scratch files (`ZCCSDT_TMPDIR`, else PySCF's TMPDIR); see Section 3 of
  `docs/factorization.md` and `docs/spin_adapted.md`.
- `diis_t3`: T3 in the DIIS vectors (default); `False` is an option for DIIS
  on T1 and T2 only.
- `plan`: an explicit memory plan instead of the automatic one.

Results: `converged`, `e_hf`, `e_corr`, `e_tot`, `cycles`, `t1`, `t2`,
`t3`, `plan_used`, `timings` (wall time per iteration), `peak_memory`, and
after `ccsdt_q()` `e_q`, `e_q5`, `e_tot_q`.

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
