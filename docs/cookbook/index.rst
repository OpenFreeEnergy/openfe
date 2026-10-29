.. _cookbooks:

How-to guides
=============

These guides show how to carry out specific tasks with **openfe**, mostly using the Python API.
For the concepts behind each step, see the :doc:`User Guide </guide/index>`.
To run a transformation with ``openfe quickrun``, see :ref:`userguide_quickrun`.

.. TODO: remove the quickrun sentence above once moved to "Running calculations"
.. TODO: remove the work-in-progress note and rename the "Cookbooks" labels in the
   protocol pages once all content is moved.

.. note:: This section is a work-in-progress.

.. module:: openfe
    :noindex:

.. toctree::
   :maxdepth: 1
   :caption: Setup

   loading_molecules
   user_charges
   bespoke_parameters

.. TODO: add "Load protein, ligands and cofactors" (merge cofactors_rbfe into loading_molecules),
   "Create a ChemicalSystem", "Specify custom force fields", membrane packmol-memgen and OpenFold3 notebooks.

.. toctree::
   :maxdepth: 1
   :caption: Ligand networks

   generate_ligand_network
   hand_write_ligand_network
   network_from_orion_fepp
   ligandnetwork_vis
   repairing_broken_networks

.. TODO: add howto on scorers

.. toctree::
   :maxdepth: 1
   :caption: Protocols and alchemical networks

   choose_protocol
   rfe_alchemical_planners
   create_alchemical_network

.. TODO: add "Customize CLI planning with YAML" (from guide/cli/cli_yaml) and move the
   how-to parts of guide/setup/defining_protocols into choose_protocol.

.. toctree::
   :maxdepth: 1
   :caption: Running calculations

.. TODO: add: run a transformation with quickrun,
   run repeats in parallel, run on a SLURM cluster, resume interrupted calculations and
   share a GPU using NVIDIA MPS (from guide/execution/quickrun_execution), and task-based
   execution.

.. toctree::
   :maxdepth: 1
   :caption: Working with results

   dumping_transformation
   jq_inspection

.. TODO: add "Gather free energy results" and "Load results in Python" (from
   guide/results/working_with_results), "Analyze an RBFE network with cinnabar" (from
   guide/results/working_with_networks) and "Visualize trajectories with openfe_analysis".
