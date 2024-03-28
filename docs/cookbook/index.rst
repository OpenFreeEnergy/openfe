.. _cookbooks:

Cookbook
========

This section describes common tasks involving the OpenFE Python API.

The :any:`OpenFE CLI<cli-reference>` provides a simple way to perform the most common procedures for free energy calculations, but does not provide much flexibility for fine-tuning your approach or combining OpenFE with other tools. The :any:`Python API<api>` allows that flexibility, but using it is more complex. This cookbook breaks down common steps that would be implemented in Python to help navigate that complexity.

.. note:: This section is a work-in-progress.

.. module:: openfe
    :noindex:

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

List of Cookbooks
-----------------

.. toctree::
    :maxdepth: 1

    loading_molecules
    dumping_transformation
    choose_protocol
    generate_ligand_network
    network_from_orion_fepp
    hand_write_ligand_network
    ligandnetwork_vis
    create_alchemical_network
    under_the_hood
    user_charges

    
