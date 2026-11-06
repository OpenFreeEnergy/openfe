.. _how_to_guides:

How-To Guides
=============

These guides show how to carry out specific tasks with **openfe**.
For the concepts behind each step, see the :doc:`User Guide </guide/index>`.

.. module:: openfe
    :noindex:

.. toctree::
   :maxdepth: 1
   :caption: Setting up chemical systems

   loading_molecules
   user_charges
   bespoke_parameters
   packmol_memgen_setup

.. TODO: merge cofactors_rbfe into loading_molecules,
   "Create a ChemicalSystem", "Specify custom force fields", OpenFold3 notebooks.

.. toctree::
   :maxdepth: 1
   :caption: Planning: Ligand networks

   generate_ligand_network
   hand_write_ligand_network
   network_from_orion_fepp
   ligandnetwork_vis

.. TODO: add howto on scorers

.. toctree::
   :maxdepth: 1
   :caption: Planning: Protocols and alchemical networks

   choose_protocol
   rfe_alchemical_planners
   create_alchemical_network

.. TODO: add "Customize CLI planning with YAML" (from guide/cli/cli_yaml) and move the
   how-to parts of guide/setup/defining_protocols into choose_protocol.

.. toctree::
   :maxdepth: 1
   :caption: Running calculations

   dumping_transformation
   run_quickrun
   resume_quickrun
   run_on_slurm
   run_repeats_in_parallel
   run_with_mps
   run_task_based

.. toctree::
   :maxdepth: 1
   :caption: Working with results

   jq_inspection

.. TODO: add "Gather free energy results" and "Load results in Python" (from
   guide/results/working_with_results), "Analyze an RBFE network with cinnabar" (from
   guide/results/working_with_networks) and "Visualize trajectories with openfe_analysis".
.. TODO: merge dumping_transformation and jq_inspection into a single
   "Save and inspect OpenFE JSON files" how-to.

.. toctree::
   :maxdepth: 1
   :caption: Repairing and extending networks

   repairing_broken_networks
