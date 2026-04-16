.. _userguide_chemicalsystems_and_components:

Chemical Systems, Components and Thermodynamic Cycles
=====================================================

.. _userguide_chemical_systems:

Chemical Systems
----------------

A :class:`.ChemicalSystem` represents the end state of an alchemical transformation,
which can then be input to a :class:`.Protocol`. 

A :class:`.ChemicalSystem` **does** contain the following information (when present):

* exact atomic information (including protonation state) of protein, ligands, co-factors, and any crystallographic
  waters
* atomic positions of all explicitly defined components such as ligands or proteins
* the abstract definition of the solvation environment, if present

A :class:`.ChemicalSystem` does **NOT** include the following:

* forcefield applied to any component, including details on water model or virtual particles
* thermodynamic conditions (e.g. temperature or pressure)

.. note::

   For some protocols, such as :class:`.SepTopProtocol` and :class:`.AbsoluteBindingProtocol`, a single
   :class:`.ChemicalSystem` is used to represent both legs of the thermodynamic cycle (complex and solvent).

   This differs from the :class:`.RelativeHybridTopologyProtocol`, where each leg is defined by
   separate :class:`.ChemicalSystem`\s. This behaviour is expected to change in future versions.

.. _userguide_components:

Components
----------

A :class:`.ChemicalSystem` is defined by a set of component objects that together
describe the full simulated system.

Components are composable building blocks that are combined additively, defining
the chemical composition of the system.

For a conventional protein–ligand system in water, a :class:`.ChemicalSystem` is typically composed of:

* :class:`.ProteinComponent`: an entire biological assembly, typically the contents of a PDB file.

  It may include crystallographic waters or ions, and can define
  disulfide bonds via CONECT records.

* one or more :class:`.SmallMoleculeComponent`\s: Ligands and cofactors

* a :class:`.SolventComponent`: Solvent conditions

Splitting the total system into components serves three purposes:

1. alchemical transformations can be easily understood by comparing the differences in components.
2. components can be reused to compose different systems.
3. :class:`.Protocol`\s can have component-specific behavior. E.g. different force fields for each component.

For protein-membrane systems, the protein is represented using a solvated
:class:`.ProteinMembraneComponent` instead of a :class:`.ProteinComponent`.
The :class:`.ProteinMembraneComponent` is an explicitly solvated protein-membrane system, including box vectors.

The :class:`.ProteinMembraneComponent` requires periodic box vectors to define the simulation box.
These can be provided in several ways:

1. CRYST record in the PDB file

   If the PDB file includes a CRYST record, OpenMM can automatically read the box vectors from it.

2. Manually specifying box vectors

   Box vectors can be provided explicitly in OpenMM format.

3. Inferring from atomic positions

   Box vectors can be estimated from the atomic coordinates in the PDB file.

.. warning::

   Inferring box vectors from atomic positions can be inaccurate,
   if the PDB originates from a previous simulation where atoms may be distributed across periodic images.

In protein-membrane systems, no further solvation of the complex system is performed by the protocol. This means:

* In the :class:`.RelativeHybridTopologyProtocol`, a separate :class:`.SolventComponent` for the complex leg is not
  required. The :class:`.ChemicalSystem` for the complex end states consists of the :class:`.ProteinMembraneComponent`
  and the ligand (and optionally cofactors) as :class:`.SmallMoleculeComponent`\s.
* In contrast, in the :class:`.SepTopProtocol` or :class:`.AbsoluteBindingProtocol`, a :class:`.SolventComponent` is still
  needed because the :class:`.ChemicalSystem` represents both complex and solvent legs. The :class:`.SolventComponent`
  is then only used to solvate the solvent leg of the transformation.


Thermodynamic Cycles
--------------------

We can now describe a thermodynamic cycle as a set of :class:`.ChemicalSystem`\s. 
The exact end states to construct are detailed in the :ref:`pages for each specific Protocol <userguide_protocols>`.

As an example, we can construct the classic relative binding free energy cycle by defining four components: two ligands,
a protein, and a solvent: 

.. figure:: ../protocols/img/rbfe_thermocycle.png
   :scale: 40%
   :alt: RBFE thermodynamic cycle

   Illustration of the relative binding free energy thermodynamic cycles and the chemical systems at each end state.

::

  import openfe

  # two small molecules defined in a molfile format
  ligand_A = openfe.SmallMoleculeComponent.from_sdf_file('./ligand_A.sdf')
  ligand_B = openfe.SmallMoleculeComponent.from_sdf_file('./ligand_B.sdf')
  # a complete biological assembly
  protein = openfe.ProteinComponent.from_pdb_file('./protein.pdb')
  # defines an aqueous solvent environment, with a concentration of ions
  solvent = openfe.SolventComponent(smiles='O')

  # ligand_A + protein + solvent
  ligand_A_complex = openfe.ChemicalSystem(components={'ligand': ligand_A, 'protein': protein, 'solvent': solvent})
  # ligand_B + protein + solvent
  ligand_B_complex = openfe.ChemicalSystem(components={'ligand': ligand_B, 'protein': protein, 'solvent': solvent})
  # ligand_A + solvent
  ligand_A_solvent = openfe.ChemicalSystem(components={'ligand': ligand_A, 'solvent': solvent})
  # ligand_B + solvent
  ligand_B_solvent = openfe.ChemicalSystem(components={'ligand': ligand_B, 'solvent': solvent})

If the system components are defined explicitly (e.g. with a solvated protein-membrane system),
the `ligand_A_complex` would instead use an :class:`.ExplicitSolventComponent` or :class:`.ProteinMembraneComponent`.
IN this case, a separate :class:`.SolventComponent` is not required.
Here is an example using a :class:`.ProteinMembraneComponent`:

::

  # explicitly solvated protein-membrane complex, including box vectors (here read from CRYST1 records in the PDB)
  protein_membrane = openfe.ProteinMembraneComponent.from_pdb_file('./protein_membrane.pdb')
  # ligand_A + explicitly solvated protein-membrane
  ligand_A_complex = openfe.ChemicalSystem(components={'ligand': ligand_A, 'protein_membrane': protein_membrane})


See Also
--------

* To see how to construct a :class:`.ChemicalSystem` from your files, see :ref:`the cookbook entry on loading molecules <Loading Molecules>`
* For details of what thermodynamic cycles to construct, consult the :ref:`pages for each specific Protocol <userguide_protocols>`
