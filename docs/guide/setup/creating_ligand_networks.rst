.. _userguide_ligand_network:

Defining the Ligand Network
===========================
A :class:`.LigandNetwork` is a network where nodes are :class:`.SmallMoleculeComponent`\ s and edges are :class:`.LigandAtomMapping`\ s.
For example, a :class:`.LigandNetwork` with drug candidates as nodes can be used to conduct a free energy campaign and compute ligand rankings.

**openfe** includes an interface to common :any:`Ligand Network Planners`, which are implemented in OpenFE's `konnektor <https://github.com/OpenFreeEnergy/konnektor>`_ package.
(See `konnektor's documentation <https://konnektor.openfree.energy/en/latest/>`_ for more information on network generators.)

Because each edge in a :class:`.LigandNetwork` is a :class:`.LigandAtomMapping`, each edge can be assigned a score that indicates the mapping's quality.
Some network generators use these scores to construct more efficient network topologies.


Below is an example of a ``LigandNetwork`` with scores assigned to each atom mapping:

.. image:: img/ligand_network.png
   :width: 80%
   :align: center
   :alt: Concept of a simple MST ligand network


Generating Ligand Networks
--------------------------

:class:`.LigandNetwork` generation can typically described by three steps:

1. Generate the :ref:`Atom Mappings<Creating Atom Mappings>`  of all pairwise combinations of :class:`.SmallMoleculeComponent`\ s
2. :ref:`Calculate scores<Scoring Atom Mappings>` for each :class:`.LigandAtomMapping`
3. Build a :class:`.LigandNetwork` with all possible mappings directed by their scores.

.. code:: python

   import openfe
   from openfe import setup

   # load a set of ligands
   mols = [SmallMoleculeComponent.from_rdkit(x) for x in rdmols]

   # generate the required mapper, scorer, and planner objects
   mapper = setup.KartografAtomMapper()
   scorer = setup.lomap_scorers.default_lomap_score
   network_planner =  setup.ligand_network_planning.generate_minimal_spanning_network

   # plan the ligand network
   ligand_network = network_planner(ligands=mols, mappers=[mapper], scorer=scorer)

Practical information on generating ligand networks can be found in our :ref:`cookbook for ligand network generation <cookbook/generate_ligand_network.nblink>`.

.. note::
   Like the Component objects, a ``LigandNetwork`` object is immutable once created!
