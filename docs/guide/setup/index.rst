.. _userguide_setup:

Simulation Setup
================

This section provides details on how to set up a free energy calculation or MD simulations.

Setting up a calculation generally involves:

* :ref:`Preparing the input structures <userguide_preparing_inputs>`.
* Loading the structures and creating :class:`.ChemicalSystem` objects.
* Defining the :class:`.Protocol` with specific `ProtocolSettings`.
* Creating :class:`.LigandAtomMapping` objects for relative free energy calculation `Protocols`.

.. image:: img/setup_1x.png
   :width: 70%
   :align: center
   :alt: Concept of a ChemicalSystems and Transformations

The image below demonstrates how, for relative free energy calculations, you plan a
network of ligand transformations starting from input SDF / MOL2 / PDB files:

.. image:: img/setup_2x.png
   :width: 70%
   :align: center
   :alt: Concept of a LigandNetwork and AlchemicalNetwork

The procedure for setting up a simulation depends on the
type of free energy calculation you are running.
More detailed guidance is available in the following sections:

.. toctree::
   :maxdepth: 1

   input_preparation
   chemical_systems_and_thermodynamic_cycles
   creating_atom_mappings_and_scores
   defining_protocols
   creating_ligand_networks
   alchemical_network_model

To set up your alchemical network using the Python interface, but run it using the CLI,
you will need to export the network in the same format used by the CLI.
See :ref:`dumping transformations <dumping_transformations>` for more details.
