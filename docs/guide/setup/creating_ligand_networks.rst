.. _define_ligand_network:

Defining the Ligand Network
===========================

A ligand network is a planning unit that we need to plan how we want to calculaute the final free energy network graph.

A simple ligand network generation could be generally conceptualized into three steps for N ligands:
* Generate the :ref:`Atom Mappings<Creating Atom Mappings>`  of all pairwise combinations of ligands (for atom mappings see here: )
* :ref:`Score all resulting Atom Mappings<Scoring Mappings>
* Build a ``LigandNetwork`` with all possible mappings directed by their scores.

.. image:: img/ligand_network.png
   :width: 90%
   :align: center
   :alt: Concept of a simple MST ligand network


Generating Ligand Networks
--------------------------

The ''LigandNetwork'' can be generated with OpenFE employing a :class:`.LigandAtomMapper` and a atom mapping scorer,
like the :func:`default_lomap_score` together with a ``LigandNetworkPlanner``, like e.g. the :func:`generate_radial_network`.
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

This network already



.. note::
   Like the Component objects, a ``LigandNetwork`` object is immutable once created!



Visualising LigandNetwork
-------------------------

.. note::
   This functionality currently lives in Konnektor our future network package, which will be integrated to OpenFE soon.

It is possible to visualize the ``LigandNetwork``. This can be done as follows:
.. code::
   from konnektor.visualization.visualization import draw_ligand_network

   fig = draw_ligand_network(ligand_network, title="Radial Graph");
   fig.show()


.. image:: img/radial_network.png
   :width: 90%
   :align: center
   :alt: Concept of a simple MST ligand network
