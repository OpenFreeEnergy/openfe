.. _guide-introduction:

Introduction
============

Here we present an overview of the workflow for calculating free energies in
OpenFE in the broadest strokes possible. This workflow is reflected in both
the Python API and in the command line interface, and so we have a section
for each.

The Basic Workflow
------------------

The overall workflow of OpenFE involves three stages:
The typical way to use the Python API is to load a number of molecules you want to calculate free energies of, construct a :class:`LigandNetwork` connecting them in an efficient way, and then combine that with information for how each ligand should be simulated to construct an :class:`AlchemicalNetwork`, which specifies the entire simulation campaign. This provides a lot of flexibility in how molecules are specified, mapped, connected, and simulated, without exposing a great deal of complexity. OpenFE recommends this workflow for most users.

1. :ref:`Simulation setup <userguide_setup>`: Defining the simulation campaign you are going to run.
2. :ref:`Execution <userguide_execution>`: Running and performing initial analysis of your
   simulation campaign.
3. :ref:`Gather results <userguide_results>`: Assembling the results from the simulation
   campaign for further analysis.





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

In many use cases, these stages may be done on different machines. For
example, you are likely to make use of HPC or cloud computing resources to
run the simulation campaign. Because of this, each stage has a defined output which
is then the input for the next stage:

.. TODO make figure
.. .. figure:: ???
    :alt: Setup -> (AlchemicalNetwork) -> Execution -> (ProtocolResults) -> Gather

    The main stages of a free energy calculation in OpenFE, and the intermediates between them.

The output of the :ref:`simulation setup <userguide_setup>` stage is an :class:`.AlchemicalNetwork`. This contains all
the information about what is being simulated (e.g., what ligands, host proteins, solvation details, etc.) and the
information about how to perform the simulation (the Protocol).

The output of the :ref:`execution <userguide_execution>` stage is the basic results from each edge.
This can depend of the specific analysis intended, but will either involve a
:class:`.ProtocolResult` representing the calculated :math:`\Delta G` for
each edge or the :class:`.ProtocolDAGResult` linked to the data needed to
calculate that :math:`\Delta G`.

The :ref:`gather results <userguide_results>` stage aggregates the individual results for further analysis. For example, the CLI's ``gather`` command will create a
table of the :math:`\Delta G` for each leg.

For more workflow details, see :ref:`under-the-hood`.

.. TODO: Should the CLI workflow be moved to under "CLI Interface"?
