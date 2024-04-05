.. _userguide_chemicalsystems_and_components:

ChemicalSystems, Components and Thermodynamic Cycles
====================================================

Chemical Systems
----------------

In order to define the input systems to a Protocol, so as the end states of an alchemical transformation,
we need an object model to represent their chemical composition.
In ``openfe`` a :class:`.ChemicalSystem` is used to capture this information,
and represents the chemical models that are present in each end state.

A ``ChemicalSystem`` **does** include information, where present, on:

* the chemical models to be simulated, i.e. which atoms are present, their bonds
* exact protonation state of both the protein and ligand
* the exact pose of a ligand or conformation of a protein
* the presence of any crystallographic waters

It **does not** include any information on:

* forcefield applied to any component, including details on water model or virtual particles
* thermodynamic conditions, i.e. temperature and pressure

Components
----------

A ``ChemicalSystem`` is composed of many ``Component`` objects,
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
For example to  construct the classic relative binding free energy cycle, we will need four components, two ligands,
a protein, and a solvent.  These four ingredients can then be combined into the four point on the thermodynamic cycle
that we wish to sample:

.. todo image of RBFE cycle taken from HREX docs

::

  import openfe

  ligand1 = openfe.SmallMoleculeComponent.from_sdf_file('./ligand1.sdf')
  ligand2 = openfe.SmallMoleculeComponent.from_sdf_file('./ligand2.sdf')
  protein = openfe.ProteinComponent.from_pdb_file('./protein.pdb')
  solvent = openfe.SolventComponent()

  ligand1_complex = openfe.ChemicalSystem(components={'ligand': ligand1, 'protein': protein, 'solvent': solvent})
  ligand2_complex = openfe.ChemicalSystem(components={'ligand': ligand2, 'protein': protein, 'solvent': solvent})
  ligand1_solvent = openfe.ChemicalSystem(components={'ligand': ligand1, 'solvent': solvent})
  ligand2_solvent = openfe.ChemicalSystem(components={'ligand': ligand2, 'solvent': solvent})


See Also
--------

* To see how to construct ``ChemicalSystem`` \s from your files, see :ref:`the cookbook entry on loading molecules <Loading Molecules>`
* For details of what thermodynamic cycles to construct, consult the :ref:`pages for each specific Protocol <userguide_protocols>`