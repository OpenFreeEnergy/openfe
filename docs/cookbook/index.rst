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
        - .. container:: width-7

            -  Chemical component definition
                SDF, PDB, RDKit, OpenFF Molecule, solvent spec, etc.

        - .. container:: flowchart-sidebyside

            -   -

                    .. rst-class:: arrow-down arrow-multiple
                - :any:`Loading proteins`, :any:`Defining solvents`

                    .. rst-class:: width-3
                - :class:`SolventComponent` and :class:`ProteinComponent`
                    Other chemical components needed to simulate the ligand.

                    .. rst-class:: arrow-down arrow-multiple arrow-tail arrow-combine-right
                -

            -   - .. container:: flowchart-sidebyside

                    -   -

                            .. rst-class:: arrow-down arrow-multiple
                        - :any:`Loading small molecules`

                            .. rst-class:: width-4
                        - :class:`SmallMoleculeComponent`
                            The ligands that will be mutated.


                            .. rst-class:: arrow-down arrow-multiple arrow-tail arrow-combine-right
                        -

                    -   -

                        - :class:`LigandAtomMapper`
                            Generates atom maps between one molecule and another.

                            .. rst-class:: arrow-down arrow-tail arrow-combine
                        -

                    -   -
                            .. rst-class:: flowchart-spacer
                        -

                        - :any:`Atom Map Scorers`
                            Objective function for optimization of a ligand network.

                            .. rst-class:: arrow-down arrow-tail arrow-combine-left
                        -

                    .. rst-class:: arrow-down arrow-head
                -

                - :class:`LigandNetwork`
                    A network of ligand transformations.

                    .. rst-class:: arrow-down arrow-tail arrow-combine
                -

            -
                -
                    .. rst-class:: flowchart-spacer
                -

                - :class:`Protocol`
                    Simulation procedure for an alchemic mutation.

                    .. rst-class:: arrow-down arrow-tail arrow-combine-left
                -

            .. rst-class:: arrow-down arrow-head
        -

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
    loading_molecules
    creating_atom_mappings
    dumping_transformation
    under_the_hood

    
