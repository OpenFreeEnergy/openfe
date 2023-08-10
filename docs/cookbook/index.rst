Cookbook
========

This will include various how-to guides.

.. module:: openfe
    :noindex:

.. container:: deflist-flowchart

    * Setup
        - .. container:: flowchart-sidebyside

            -

              .. rst-class:: flowchart-narrow
            -   - :class:`Protocol`
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
            A complete description of a simulation campaign, including the :class:`LigandNetwork` and a :class:`ChemicalSystem` for each ligand.

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

    
