Plain MD Protocol
=================

The plain MD protocol enables the user to run an MD simulation of a `ChemicalSystem`, which can contain e.g. a solvated protein-ligand complex, ora ligand and water. 
The protocol applies a LangevinMiddleIntegrator which uses Langevin dynamics, with the LFMiddle discretization (J. Phys. Chem. A 2019, 123, 28, 6056-6079). 
Before running the production MD simulation in the NPT ensemble, the protocol performs a minimization of the system, followed by an equilibration in the canonical ensemble as well as an equilibration in the NPT ensemble. A MonteCarloBarostat is used in the NPT ensemble to maintain constant pressure.
