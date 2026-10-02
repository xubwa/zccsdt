'''Closed-shell spin-adapted (RHF) coupled cluster on spatial orbitals
(design notes: docs/spin_adapted.md).

- rccsdt: the CCSDT solver with block-stored T3; rccsdt.run(mf) for a PySCF
  RHF object, rccsdt.kernel(ints) for prepared integrals.  max_memory (MB)
  bounds the memory: what does not fit (T3, the <vv|vv> operator, the DIIS
  vectors) goes to the scratch directory.
- rccsdt_q: the perturbative (Q) correction (rccsdt.run(mf, q=True)), per
  occupied quadruple in memory or in slabs of the virtual space.
- rccsdt_ref: dense reference CCSD / CCSDT and the integral container.
- t3blocks: T3 block storage, placement rules and kernels.
- derive: the spin-adapted equations, derived from the spin-orbital ones
  with spinfree (spin-line tracing); somap maps spatial tensors back to
  spin orbitals for the checks.
'''
