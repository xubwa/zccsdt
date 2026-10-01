'''Closed-shell spin-adapted (RHF) coupled cluster on spatial orbitals
(design notes: docs/spin_adapted.md).

- rccsdt: the CCSDT solver with block-stored T3; rccsdt.run(mf) for a PySCF
  RHF object, rccsdt.kernel(ints) for prepared integrals.
- rccsdt_ref: dense reference CCSD / CCSDT and the integral container.
- t3blocks: T3 block storage, placement rules and kernels.
- derive: the spin-adapted equations, derived from the spin-orbital ones
  with spinfree (spin-line tracing); somap maps spatial tensors back to
  spin orbitals for the checks.
'''
