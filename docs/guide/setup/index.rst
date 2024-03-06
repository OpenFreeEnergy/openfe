Simulation Setup
================

This section provides details on how to set up a free energy calculation.
This will get you from your SDF/MOL2/PDB files to an
:class:`.AlchemicalNetwork`, which contains all the information to run a
simulation campaign.

The first thing you will need to do will be to load in your molecules.
OpenFE can small molecules from anything that RDKit can handle, and can load
proteins from PDB or PDBx. For details of this, see
:ref:`loading molecules <Loading Molecules>`.

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

.. toctree (hidden): chemical components and chemical systems

If you intend to set up your alchemical network using the Python interface,
but to run it using the CLI, you will want to export the network in the same
format used by the CLI. See :ref:`dumping transformations <dumping_transformations>`
for more details.
