Cookbook
========

This will include various how-to guides.

.. module:: openfe
    :noindex:

Network-First Workflow
-----------------------------

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
                - :class:`SolventComponent`, :class:`ProteinComponent`
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
    *

    * Run
        - term
            def

      .. rst-class:: arrow-down
    *

    * Gather
        - term
            def


Transformation-First Workflow
-----------------------------

.. container:: deflist-flowchart

    * Setup
        - .. container:: flowchart-sidebyside

            -   -
                  .. rst-class:: flowchart-spacer
                -

                - :class:`Protocol`
                    Simulation procedure for an alchemic mutation.

                    .. rst-class:: arrow-down arrow-tail arrow-combine-right
                -

            -   - Chemical component definition
                    SDF, PDB, RDKit, OpenFF Molecule, solvent spec, etc.

                  .. rst-class:: arrow-down arrow-tail
                - :any:`Loading Molecules`

                - .. container:: flowchart-sidebyside

                    -   -
                            .. rst-class:: arrow-down arrow-head arrow-combine-right
                        -

                        - :class:`SmallMoleculeComponent`
                            Small molecule components describe the ligands that will be mutated.

                        - .. container:: flowchart-sidebyside

                            -   -
                                    .. rst-class:: arrow-down arrow-multiple
                                -

                                - :class:`LigandNetwork`
                                    A network of ligand transformations.

                                    .. rst-class:: arrow-down arrow-multiple
                                -

                            -   -
                                    .. rst-class:: arrow-down
                                - :any:`Creating Atom Mappings`

                        - :class:`LigandAtomMapping`
                            Corresponds atoms in one small molecule to those in another.

                            .. rst-class:: arrow-down arrow-tail arrow-combine
                        -

                    -   -
                            .. rst-class:: arrow-down arrow-head arrow-combine-left
                        -

                        - :class:`SmallMoleculeComponent`, :class:`SolventComponent`, :class:`ProteinComponent`
                            All components are included in the ChemicalSystem

                            .. rst-class:: arrow-down arrow-multiple
                        - :any:`Assembling into ChemicalSystems`

                        - :class:`ChemicalSystem`
                            The complete system of chemical components required to simulate a given transformation target.

                            .. rst-class:: arrow-down arrow-tail arrow-combine-left
                        -

            .. rst-class:: arrow-down arrow-head
        -

        - :class:`Transformation`
            A single transformation

            .. rst-class:: arrow-down
        -


        - :class:`AlchemicalNetwork`
            A complete description of a simulation campaign, consisting of a collection of :class:`Transformation` objects and a :class:`ChemicalSystem` for each ligand the transformations connect.

        .. rst-class:: arrow-down
    *

    * Run
        - term
            def

        .. rst-class:: arrow-down
    *

    * Gather
        - term
            def


.. toctree::

    loading_molecules
    creating_atom_mappings
    dumping_transformation

    
