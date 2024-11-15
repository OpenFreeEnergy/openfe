.. _userguide_ligand_network:

Defining the Ligand Network
===========================
A :class:`.LigandNetwork` is a set of small molecules connected by mappings of two small molecules.
One example for such a network could be a set of drug candidates, that should be ranked using alchemical transformations.
The :class:`.LigandNetwork` is a tool that is used to orchestrate the free energy calculations to efficiently
compute a ligand ranking.
It is of course possible to calculate all possible :class:`.Transformation` defined by all possible :class:`.AtomMappings`
connecting all :class:`.SmallMoleculeComponent` with a ''maximal network'' (using :func:`.generate_maximal_network`),
but it is much more efficient to use a network with less transformations like a ''radial network'' (also known as a star map, using :func:`.generate_radial_network`)
or a ''minimimal spanning network'' (using :func:`.generate_minimal_spanning_network`).

Any :class:`.LigandNetwork`` generation can be generally conceptualized into three steps:

* Generate the :ref:`Atom Mappings<Creating Atom Mappings>`  of all pairwise combinations of ligands
* :ref:`Score all resulting Atom Mappings<Creating Atom Mappings>`
* Build a :class:`.LigandNetwork` with all possible mappings directed by their scores.

.. image:: img/ligand_network.png
   :width: 90%
   :align: center
   :alt: Concept of a simple MST ligand network


Generating Ligand Networks
--------------------------

The ''LigandNetwork'' can be generated with OpenFE employing a :class:`.LigandAtomMapper` and a atom mapping scorer,
like the :func:`.default_lomap_score` together with a ``LigandNetworkPlanner``, like e.g. the :func:`.generate_radial_network`.
In the following code, we will show how a ``LigandNetwork`` can be planned:

.. code::

   import openfe
   from openfe import setup

   # as previously detailed, load a set of ligands
   mols = [SmallMoleculeComponent.from_rdkit(x) for x in rdmols]

   # first let's generate the required objs
   mapper = setup.KartografAtomMapper()
   scorer = setup.lomap_scorers.default_lomap_score
   network_planner =  setup.ligand_network_planning.generate_minimal_spanning_network

   # Now let's plan the Network
   ligand_network = network_planner(ligands=mols, mappers=[mapper], scorer=scorer)

Practical information on generating ligand networks can be found in our :ref:`cookbook for ligand network generation <cookbook/generate_ligand_network.nblink>` .

.. note::
   Like the Component objects, a ``LigandNetwork`` object is immutable once created!
