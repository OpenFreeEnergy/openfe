Relative Hybrid Topology Protocol
=================================

The relative free energy calculation approach calculates the difference in 
free energy between two similar ligands. Depending on the ChemicalSystem 
provided, the protocol either calculates the relative binding free energy 
(RBFE), or the relative hydration free energy (RHFE). In a thermodynamic 
cycle, one ligand is converted into the other ligand by alchemically 
transforming the atoms that vary between the two ligands. The 
transformation is carried out in both environments, meaning both in the 
solvent (ΔG\ :sub:`solv`\) and in the binding site (ΔG\ :sub:`site`\) for RBFE calculations 
and in the solvent (ΔG\ :sub:`solv`\) and vacuum (ΔG\ :sub:`vacuum`\) for RHFE calculations.

.. _label: Thermodynamic cycle for the relative binding free energy protocol
.. figure:: img/rbfe_thermocycle.png
   :scale: 50%

   Thermodynamic cycle for the relative binding free energy protocol.
   
The :class:`.RelativeHybridTopologyProtocol` uses a hybrid topology approach to represent the two
ligands, meaning that a single set of coordinates is used to represent the
common core of the two ligands while the atoms that differ between the two
ligands are represented separately. An atom map defines which atoms belong
to the core (mapped atoms) and which atoms are unmapped and represented
separately. During the alchemical transformation, mapped atoms are switched
from the type in one ligand to the type in the other ligands, while unmapped
atoms are switched on or off, depending on which ligand they belong to.

