.. _cookbooks:

How-to guides
=============

These guides show how to carry out specific tasks with **openfe**, mostly using the Python API.
For the concepts behind each step, see the :doc:`User Guide </guide/index>`.
To run a transformation with ``openfe quickrun``, see :ref:`userguide_quickrun`.

.. TODO: remove the quickrun sentence above once the "Running calculations" group exists.
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

The Basic Workflow
------------------

The typical way to use the Python API is to load a number of molecules you want to calculate free energies of, construct a :class:`LigandNetwork` connecting them in an efficient way, and then combine that with information for how each ligand should be simulated to construct an :class:`AlchemicalNetwork`, which specifies the entire simulation campaign. This provides a lot of flexibility in how molecules are specified, mapped, connected, and simulated, without exposing a great deal of complexity. OpenFE recommends this workflow for most users.

.. container:: deflist-flowchart

    * Setup
        - .. container:: flowchart-sidebyside

            -   -
                    .. rst-class:: flowchart-spacer
                -

                    .. rst-class:: arrow-down arrow-from-nothing
                - :any:`choose_protocol`

                - :class:`Protocol`
                    Simulation procedure for an alchemic mutation.

                    .. rst-class:: arrow-down arrow-tail arrow-combine-right
                -

            -   -
                    .. rst-class:: width-8
                -  Chemical component definition
                    SDF, PDB, RDKit, OpenFF Molecule, solvent spec, etc.

                - .. container:: flowchart-sidebyside

                    - .. rst-class:: width-3

                        -

                            .. rst-class:: arrow-down arrow-multiple
                        - :any:`Loading proteins`, :any:`Defining solvents`

                        - :class:`SolventComponent` and :class:`ProteinComponent`
                            Other chemical components needed to simulate the ligand.

                            .. rst-class:: arrow-down arrow-multiple arrow-tail arrow-combine
                        -

                    -   - .. container:: flowchart-sidebyside

                            - .. rst-class:: width-5

                                -
                                    .. rst-class:: arrow-down arrow-multiple
                                - :any:`Loading small molecules`


                                - :class:`SmallMoleculeComponent`
                                    The ligands that will be mutated.

                            - .. rst-class:: width-3

                                -
                                    .. rst-class:: flowchart-spacer
                                -

                                - Orion/FEP+
                                    Network from another tool.


                        - .. container:: flowchart-sidebyside

                            - .. rst-class:: width-2

                                -
                                    .. rst-class:: arrow-down arrow-multiple
                                - :any:`generate_ligand_network`

                            - .. rst-class:: width-2

                                -
                                    .. rst-class:: arrow-down arrow-multiple
                                - :any:`hand_write_ligand_network`

                            - .. rst-class:: width-1

                                -
                                    .. rst-class:: arrow-down arrow-tail arrow-multiple arrow-combine-right
                                -

                                    .. rst-class:: flowchart-spacer
                                -

                            - .. rst-class:: width-3

                                -
                                    .. rst-class:: arrow-down arrow-tail arrow-combine-left
                                -

                                    .. rst-class:: arrow-down arrow-head flowchart-spacer
                                - :any:`network_from_orion_fepp`

                        - :class:`LigandNetwork <openfe.setup.LigandNetwork>`
                            A network of ligand transformations.

                        - .. container:: flowchart-sidebyside

                            -   -
                                    .. rst-class:: arrow-down arrow-tail arrow-combine-left width-4
                                -

                            -   -
                                    .. rst-class:: arrow-cycle width-4
                                -

                                - :any:`ligandnetwork_vis`


            .. rst-class:: arrow-down arrow-head
        - :any:`create_alchemical_network`

        - :class:`AlchemicalNetwork`
            A complete simulation campaign.

      .. rst-class:: arrow-down
    * :any:`dumping_transformations`

    * Run
        - :any:`openfe quickrun <cli_quickrun>`
            OpenFE recommends using the ``openfe quickrun`` CLI command to execute a transformation.

      .. rst-class:: arrow-down
    *

    * Gather
        - :any:`openfe gather <cli_gather>`
            OpenFE recommends using the ``openfe gather`` CLI command to collect the results of a transformation.
