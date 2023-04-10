.. _define-rbfe:

Defining RBFE Calculations
==========================

An :class:`.AlchemicalNetwork` for relative binding free energy calculations
can be easily created with a :class:`.RBFEAlchemicalNetworkPlanner`.

Creating the :class:`.AlchemicalNetwork` basically involves the following
three steps:

1. Creating a :class:`.LigandNetwork` to represent the planned ligand
   transformations.
2. Creating a :class:`.ChemicalSystem` (which combines protein, solvent, and
   ligand) for each ligand.
3. Using the :class:`.LigandNetwork` to create the
   :class:`.AlchemicalNetwork` (where each node is a
   :class:`.ChemicalSystem` and the edges also carry information about the
   :class:`.Protocol`).

Each aspect of this can be performed manually. For details on customizing
the :class:`.LigandNetwork`, see :ref:`define_ligand_network`.
