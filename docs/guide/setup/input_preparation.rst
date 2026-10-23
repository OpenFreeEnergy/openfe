.. _userguide_preparing_inputs:

Preparing input structures
==========================

Before running a free energy calculation, input structures must be
prepared with the correct chemistry, coordinates, and connectivity.
**openfe** does not perform structure preparation, so tasks such as
assigning protonation states, modelling missing residues, and
generating ligand binding poses must be completed beforehand.

This page describes what **openfe** expects its
inputs to look like, and how to handle common edge cases.

Input files
-----------

The table below lists the input files typically used for protein--ligand calculations (filenames are examples).

In **openfe**, *components* represent parts of a molecular system,
such as proteins, ligands or solvent. They are combined into a
``ChemicalSystem``. See :doc:`Components and Chemical Systems
<chemical_systems_and_thermodynamic_cycles>` for details.

.. list-table::
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
     - Noncovalent cofactors with correct chemistry, explicit hydrogens, and 3D coordinates of their binding poses.

Bulk solvent is added during simulation setup using a ``SolventComponent``.

.. note::

   Membrane and pre-solvated systems use ``ProteinMembraneComponent`` or ``SolvatedPDBComponent``
   instead. These must include explicit solvent and valid periodic box vectors, so no additional
   ``SolventComponent`` is needed for the complex leg.
   See :doc:`the membrane RBFE tutorial </tutorials/rbfe_membrane_protein>`.

.. _prep_protein:

Preparing the protein
---------------------

.. rubric:: Missing atoms, residues and chain termini

* **Missing atoms:** Reconstruct missing atoms and check that PDB
  atom names and element symbols are correct for force field assignment.
* **Naming conventions:** Use standard PDB or AMBER-compatible
  residue and atom names, and ensure element symbols are correct.
* **Missing residues:** Model missing residues, especially near
  the binding site.
* **Chain termini:** If gaps remain, the exposed termini may
  require ``ACE`` (N-terminal) or ``NME`` (C-terminal) caps.

.. note::

   Some structure preparation tools require special care when
   adding terminal caps:

   * **Maestro:** Hydrogen-only caps (H-caps) are not supported. Use ``ACE``/``NME`` caps instead.
   * **PyMOL:** Caps may be incorrectly positioned or separated
     from their terminal residues by ``TER`` records. Check the
     cap placement and ensure no ``TER`` record separates a cap
     from the residue it caps.

.. rubric:: Alternative locations (altLoc)

Resolve alternative atom positions (``altLoc``) by selecting a consistent conformation and removing unused alternatives.

.. rubric:: Protonation states and hydrogen atoms

Assign the intended protonation states and include all required hydrogens.
Ensure that atom and residue names, and protonation states are compatible with the selected force fields.

.. rubric:: Disulfide bonds

Disulfide bonds may be detected automatically from the distance between cysteine sulfur (``SG``) atoms, but including
``CONECT`` records is recommended.
Ensure bonded cysteines are not protonated at the sulfur atoms.

.. rubric:: Non-standard and modified residues

Only phosphoserine (``SEP``) and phosphothreonine (``TPO``) are supported, and require ``CONECT`` records.
Other modified residues cannot currently be read; replace them with their canonical residues.

Crystallographic waters and ions
--------------------------------

.. rubric:: Crystallographic waters

* Retain binding-site and structurally important waters, but check for clashes with all ligands in the series and
  remove conflicting waters where appropriate.
* Waters can remain in the protein PDB but must have explicit hydrogens and recognised residue names, such as ``HOH`` or ``WAT``.
* During solvation, the protocols add bulk water without replacing existing waters.

.. rubric:: Ions and additives

* Retain structurally important ions in the protein PDB and ensure
  they are compatible with the selected force fields.
* Remove crystallisation additives, such as sulfate, glycerol or polyethylene
  glycol, unless they are relevant to the system.
* If additives are retained, extract them into a separate SDF file and treat them as :ref:`cofactors <prep_cofactors>`.

.. _prep_ligands:

Preparing ligands
-----------------

.. rubric:: Chemical structure

Provide ligands in SDF format, as PDB files do not reliably
preserve bond orders. Each ligand must have:

* Correct connectivity, bond orders, formal charges and stereochemistry.
* The intended protonation and tautomeric state, with explicit hydrogens.
* A unique name in the SDF title line, used to label results.

Partial charges are assigned during protocol setup but can also
be assigned in advance for reproducibility and to avoid repeated calculations.
See :doc:`Generating Partial Charges with the OpenFE CLI </tutorials/charge_molecules_cli_tutorial>`.

.. rubric:: Binding poses

* Ligands must have 3D coordinates representing their intended binding
  poses in the same coordinate frame as the prepared protein.
* Check for clashes with the protein, retained waters and cofactors.
* For :ref:`Relative Hybrid Topology
  <userguide_relative_hybrid_topology_protocol>`, ligands should be
  aligned on their common core, and the atom mapping should be
  inspected separately.

.. _prep_cofactors:

Cofactors and other small molecules
-----------------------------------

* Extract noncovalently bound cofactors and any crystallisation
  additives that need to be retained from the protein PDB into
  separate SDF files.
* The :ref:`ligand preparation requirements <prep_ligands>` also
  apply to cofactors, including correct bonding, protonation,
  explicit hydrogens and coordinates.
* Metal-containing cofactors, such as heme, are not supported by the standard preparation workflow.

Checking prepared structures
----------------------------

.. card:: :octicon:`checklist;1em;sd-text-success` Before running a calculation
   :class-card: sd-shadow-sm sd-mb-3

   Verify that:

   * The :ref:`protein structure <prep_protein>` is complete, with
     correct protonation, termini, disulfide bonds and connectivity.
   * Waters, ions and :ref:`cofactors <prep_cofactors>` are
     appropriately retained or removed, without duplicate components.
   * :ref:`Ligands <prep_ligands>` have correct chemistry, explicit
     hydrogens and binding poses aligned with the prepared protein.
   * All components are compatible with the selected force fields.
   * **openfe** can load, parameterize and initialize the system
     without errors or unexpected structural changes.

See also
--------

* :doc:`Loading Molecules and Chemical Systems
  </cookbook/loading_molecules>`
* :doc:`Components, Chemical Systems and Thermodynamic Cycles
  <chemical_systems_and_thermodynamic_cycles>`
* :doc:`Membrane RBFE tutorial </tutorials/rbfe_membrane_protein>`
* :ref:`Protocols and domain of applicability <userguide_protocols>`