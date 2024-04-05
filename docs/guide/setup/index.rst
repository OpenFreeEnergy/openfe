Simulation Setup
================

This section provides details on how to set up free energy calculation or MD simulations.

All protocols in OpenFE follow the same general structure: 

* Reading in input structures and Creating ``ChemicalSystem`` \s
* Defining the Protocol with specific ProtocolSettings
* Creating ``LigandAtomMapping`` \s for relative free energy calculations Protocols


This will get you from your SDF/MOL2/PDB files to an
:class:`.AlchemicalNetwork`, which contains all the information to run a
simulation campaign.


.. image:: img/setup_1x.png
   :width: 60%
   :align: center
   :alt: Concept of a ChemicalSystems and Transformations

For relative free energy calculations, you can plan an entire network of transformations,
starting from your SDF/MOL2/PDB files to an
:class:`.AlchemicalNetwork`, which contains all the information to run a
simulation campaign. 

.. image:: img/setup_2x.png
   :width: 60%
   :align: center
   :alt: Concept of a LigandNetwork and AlchemicalNetwork

The procedure for setting up a simulation depends somewhat on the on the
type of free energy calculation you are running. See more detailed
instructions can be found under:

.. toctree::

   chemical_systems_and_thermodynamic_cycles
   creating_atom_mappings_and_scores
   defining_protocols
   creating_ligand_networks
   alchemical_network
   define_rbfe
   define_rhfe

If you intend to set up your alchemical network using the Python interface,
but to run it using the CLI, you will want to export the network in the same
format used by the CLI. See :ref:`dumping transformations <dumping_transformations>`
for more details.
