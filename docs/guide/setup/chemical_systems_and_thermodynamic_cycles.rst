.. _userguide_chemicalsystems_and_components:

ChemicalSystems, Components and Thermodynamic Cycles
====================================================

Chemical Systems
----------------

In order to define the input systems to a Protocol,
which correspond as the end states of an alchemical transformation,
we need an object model to represent their chemical composition.
In ``openfe`` a :class:`.ChemicalSystem` is used to capture this information,
and represents the chemical models that are present in each end state.

A :class:`.ChemicalSystem` **does** include information, where present, on:

* exact atomic information (including protonation state) of protein, ligands, co-factors, and any crystallographic
  waters
* atomic positions of all explicitly  ligand or conformation of a protein
* the abstract definition of the solvation environment, if present

It **does not** include any information on:

* forcefield applied to any component, including details on water model or virtual particles
* thermodynamic conditions, i.e. temperature and pressure

Components
----------

A :class:`.ChemicalSystem` is composed of many :class:`.Component` objects,
each representing a single ''piece'' of the overall system.
Examples of components include:

* :class:`.ProteinComponent` to represent an entire biological assembly, typically the contents of a PDB file
* :class:`.SmallMoleculeComponent` to represent ligands and cofactors
* :class:`.SolventComponent` to represent the solvent conditions

Splitting the total system into components serves two purposes:

* alchemical transformations can be easily understood by comparing the differences in Components
* ``Protocol`` \s can know to treat different components differently, for example applying different force fields

Thermodynamic Cycles
--------------------

With a language to express chemical systems piecewise, we can now also construct thermodynamic cycles based on these.
The exact end states to construct are detailed in the :ref:`pages for each specific Protocol <userguide_protocols>`.
For example to  construct the classic relative binding free energy cycle, we will need four components, two ligands,
a protein, and a solvent.  These four ingredients can then be combined into the four point on the thermodynamic cycle
that we wish to sample:

.. todo image of RBFE cycle taken from HREX docs

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
  # ligand_A + solvent
  ligand_B_solvent = openfe.ChemicalSystem(components={'ligand': ligand_B, 'solvent': solvent})


See Also
--------

* To see how to construct a :class:`.ChemicalSystem` \s from your files, see :ref:`the cookbook entry on loading molecules <Loading Molecules>`
* For details of what thermodynamic cycles to construct, consult the :ref:`pages for each specific Protocol <userguide_protocols>`
