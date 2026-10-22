.. _userguide_preparing_inputs:

Preparing input structures for OpenFE
=====================================

Before running a free energy calculation, input structures must be
prepared with the correct chemistry, coordinates, and connectivity.
**openfe** does not perform structure preparation, so tasks such as
assigning protonation states, modelling missing residues, and
generating ligand binding poses must be completed beforehand.

This page describes what **openfe** expects its
inputs to look like, and how to handle common edge cases.

Input files
-----------

The table below lists the input files typically used for protein--ligand calculations with OpenFE.
The filenames shown are examples.

.. list-table:: Input files
   :header-rows: 1
   :widths: 23 20 57

   * - File
     - Component
     - Contents
   * - :ref:`protein.pdb <prep_protein>`
     - ``ProteinComponent``
     - Prepared protein, including retained crystallographic waters
       and structural ions, but excluding ligands and cofactors.
   * - :ref:`ligands.sdf <prep_ligands>`
     - ``SmallMoleculeComponent`` for each ligand
     - Ligands with correct chemistry, explicit hydrogens, and 3D coordinates of their binding poses.
   * - :ref:`cofactors.sdf <prep_cofactors>` (optional)
     - ``SmallMoleculeComponent`` for each cofactor
     - Noncovalent cofactors with correct chemistry and coordinates.

Bulk solvent is added during simulation setup using a ``SolventComponent``.

.. note::

   Membrane and pre-solvated systems instead use ``ProteinMembraneComponent`` or ``SolvatedPDBComponent``
   with explicit solvent and valid periodic box vectors.
   See :doc:`the membrane RBFE tutorial  </tutorials/rbfe_membrane_protein>`.

.. _prep_protein:

Preparing the protein
---------------------

**Missing atoms, residues and chain termini**

Missing atoms need to be reconstructed to ensure residues match
their force field templates. Missing residues should ideally be
modelled, especially near the binding site. If gaps remain,
the exposed termini may need ``ACE`` (N-terminal) or ``NME``
(C-terminal) caps.

.. note::

   Some tools, such as PyMOL, may misplace caps or insert ``TER``
   records between a cap and its terminal residue. Ensure that caps are correctly positioned and that
   no ``TER`` records separate a cap from the residue it caps.

**Alternative locations (altLoc)**

Resolve alternative atom positions (``altLoc``) by selecting a
consistent conformation and removing unused alternatives.

**Protonation states and hydrogen atoms**

Assign the intended protonation states and include all required
hydrogens. Ensure that atom names and protonation
states are compatible with the selected force fields.

**Disulfide bonds**

Disulfide bonds may be detected automatically from the distance
between cysteine sulfur (``SG``) atoms, but including ``CONECT``
records is recommended. Ensure bonded cysteines are not
protonated at the sulfur atoms.

**Non-standard and modified residues**

Non-standard residues must be compatible with the selected force
fields. Phosphoserine (``SEP``) and phosphothreonine (``TPO``)
are supported with the required ``CONECT`` records. Other
modifications are not currently supported. For now, our recommendation is to replace these
with canonical residues.

Crystallographic waters and ions
--------------------------------

**Crystallographic waters**

Retain binding-site and structurally important waters, but check
for clashes with all ligands in the series and remove
conflicting waters where appropriate. Waters can
remain in the protein PDB but must include explicit hydrogens.
During solvation, the protocols add bulk water without replacing existing
waters.

**Ions and additives**

Retain structurally important ions in the protein PDB and ensure
they are compatible with the selected force fields. Remove
crystallisation additives, such as sulfate, glycerol or polyethylene
glycol, unless they are relevant to the system. If retained, extract
them into a separate SDF file and treat them as
:ref:`cofactors <prep_cofactors>`.

.. _prep_ligands:

Preparing ligands
-----------------

**Chemical structure**

Provide ligands in SDF format, as PDB files do not reliably
preserve bond orders. Each ligand must have:

* Correct connectivity, bond orders, formal charges and stereochemistry.
* The intended protonation and tautomeric state, with all hydrogens present.
* A unique name in the SDF title line, used to label results.

The protocols assign partial charges during setup, but these can
also be assigned in advance (using e.g. ``openfe charge-molecules``) for
reproducibility and to avoid repeated charge calculations.

**Binding poses**

Ligands must have 3D coordinates representing their intended binding
poses in the same coordinate frame as the prepared protein. Check for
clashes with the protein, retained waters and cofactors.

For Relative Hybrid Topology, ligands should be well aligned on their
common core. The atom mapping should also be inspected separately.

.. _prep_cofactors:

Cofactors and other small molecules
-----------------------------------

Extract noncovalently bound cofactors and any crystallisation
additives that need to be retained from the protein PDB into
separate SDF files.

The :ref:`ligand preparation requirements <prep_ligands>` also
apply to cofactors, including correct bonding, protonation,
explicit hydrogens and coordinates.
Metal-containing cofactors, such as heme, are not supported.

Checking prepared structures
----------------------------

Before planning or running a calculation, verify that:

* The protein structure is complete, with correct protonation,
  termini, disulfide bonds and connectivity.
* Waters, ions and cofactors are appropriately retained or removed,
  without duplicate components.
* Ligands have correct chemistry, explicit hydrogens and binding
  poses aligned with the prepared protein.
* All components are compatible with the selected force fields.
* OpenFE can load, parameterize and initialize the system without
  errors or unexpected structural changes.

See also
--------

* :doc:`Loading Molecules and Chemical Systems
  </cookbook/loading_molecules>`
* :doc:`Components, Chemical Systems and Thermodynamic Cycles
  <chemical_systems_and_thermodynamic_cycles>`
* :doc:`Membrane RBFE tutorial </tutorials/rbfe_membrane_protein>`
* :ref:`Protocols and domain of applicability <userguide_protocols>`