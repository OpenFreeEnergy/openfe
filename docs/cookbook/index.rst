Cookbook
========

This will include various how-to guides.

.. module:: openfe
    :noindex:

Basic Workflow
--------------


.. container:: deflist-flowchart

    * Setup
        - .. container:: flowchart-sidebyside

            -

              .. rst-class:: flowchart-narrow
            -   -
                  .. rst-class:: flowchart-spacer
                -

                - :class:`Protocol`
                    Simulation procedure for an alchemic mutation.

                    .. rst-class:: arrow-down
                -

            -   - Chemical component definition
                    SDF, PDB, RDKit, OpenFF Molecule, solvent spec, etc.

                - .. container:: flowchart-sidebyside

                    -
                      .. rst-class:: flowchart-narrow
                    -   -

                            .. rst-class:: arrow-down
                        - :any:`Loading small molecules`

                        - :class:`SmallMoleculeComponent`
                            Small molecule components describe the ligands that will be mutated.

                            .. rst-class:: arrow-down arrow-downright arrow-multiple
                        -

                    -   - .. container:: flowchart-sidebyside

                            -   -

                                    .. rst-class:: arrow-down
                                - :any:`Defining solvents`

                                - :class:`SolventComponent`
                                    Small molecule components describe the ligands that will be mutated.

                                    .. rst-class:: arrow-down
                                -

                            -   -

                                    .. rst-class:: arrow-down
                                - :any:`Loading proteins`

                                - :class:`ProteinComponent`
                                    Small molecule components describe the ligands that will be mutated.

                                    .. rst-class:: arrow-down
                                -

                        - :class:`ChemicalSystem`
                            The complete system of chemical components required to simulate a given transformation target.

                          .. rst-class:: arrow-down
                        -

        - :any:`Alchemical Network Planning`
            Classes to plan a simulation campaign.

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

Network-First Workflow
-----------------------------

.. container:: deflist-flowchart

    * Setup
        - .. container:: flowchart-sidebyside

            -
              .. rst-class:: flowchart-narrow
            -
                - :class:`SmallMoleculeComponent`, :class:`SolventComponent`, :class:`ProteinComponent`
                    All components are included in the ChemicalSystem

                  .. rst-class:: arrow-down
                - :any:`Assembling into ChemicalSystems`

                - :class:`ChemicalSystem`
                    The complete system of chemical components required to simulate a given transformation target.

                  .. rst-class:: arrow-down arrow-multiple
                -

            -   - .. container:: flowchart-sidebyside

                    - - :class:`SmallMoleculeComponent`
                          Small molecule components describe the ligands that     will be mutated.


                          .. rst-class:: arrow-down arrow-multiple
                      -

                    - - :class:`LigandAtomMapper`
                          Generates atom maps between one molecule and another.

                        .. rst-class:: arrow-down
                      -

                    - -
                        .. rst-class:: flowchart-spacer
                      -
                      - :any:`Atom Map Scorers`
                          Objective function for optimization of a ligand network.

                        .. rst-class:: arrow-down
                      -

                - :mod:`openfe.setup.ligand_network_planning`
                      Functions to plan a network of ligand mutations given a     :class:`LigandAtomMapper` and a scorer.

                        .. rst-class:: arrow-down
                -

                - :class:`LigandNetwork`
                      A network of ligand transformations.

                        .. rst-class:: arrow-down
                -

            -
                -
                  .. rst-class:: flowchart-spacer
                -

                - :class:`Protocol`
                    Simulation procedure for an alchemic mutation.

                  .. rst-class:: arrow-down
                -

        - :class:`AlchemicalNetwork`
            A complete simulation campaign.


Transformation-First Workflow
-----------------------------

.. container:: deflist-flowchart

    * Setup
        - .. container:: flowchart-sidebyside

            -

              .. rst-class:: flowchart-narrow
            -   -
                  .. rst-class:: flowchart-spacer
                -

                - :class:`Protocol`
                    Simulation procedure for an alchemic mutation.

                    .. rst-class:: arrow-down
                -

            -   - Chemical component definition
                    SDF, PDB, RDKit, OpenFF Molecule, solvent spec, etc.

                  .. rst-class:: arrow-down
                - :any:`Loading Molecules`

                - Component
                    A component of a chemical system.

                - .. container:: flowchart-sidebyside

                    -   - :class:`SmallMoleculeComponent`
                            Small molecule components describe the ligands that will be mutated.

                        - .. container:: flowchart-sidebyside

                            -   -
                                    .. rst-class:: arrow-down arrow-multiple
                                -

                                - :class:`LigandNetwork`

                                    .. rst-class:: arrow-down arrow-multiple
                                -

                            -   -
                                    .. rst-class:: arrow-down
                                - :any:`Creating Atom Mappings`

                        - :class:`LigandAtomMapping`
                            Corresponds atoms in one small molecule to those in another.

                            .. rst-class:: arrow-down
                        -

                    -
                        - :class:`SmallMoleculeComponent`, :class:`SolventComponent`, :class:`ProteinComponent`
                            All components are included in the ChemicalSystem

                          .. rst-class:: arrow-down
                        - :any:`Assembling into ChemicalSystems`

                        - :class:`ChemicalSystem`
                            The complete system of chemical components required to simulate a given transformation target.

                          .. rst-class:: arrow-down
                        -

        - :class:`Transformation`
            A single transformation

          .. rst-class:: arrow-down
        -


        - :class:`AlchemicalNetwork`
            A complete description of a simulation campaign, consisting of a collection of :class:`Transformation` objects and a :class:`ChemicalSystem` for each ligand the transformations connect.


.. toctree::

    loading_molecules
    creating_atom_mappings
    dumping_transformation

    
