Cookbook
========

This will include various how-to guides.

.. container:: deflist-flowchart

    * Setup
        - Chemical component definition
            SDF, PDB, RDKit, OpenFF Molecule, solvent spec, etc.

          .. rst-class:: arrow-down
        - :any:`Loading Molecules`

        - Component
            A component of a chemical system.

        - .. container:: flowchart-sidebyside

            -
                - :class:`openfe.SmallMoleculeComponent`
                    Small molecule components describe the ligands that will be mutated.

                - .. container:: flowchart-sidebyside

                    -
                        -
                            .. rst-class:: arrow-down
                        -

                    -
                        -
                            .. rst-class:: arrow-down
                        - :any:`Creating Atom Mappings`

                        - Mapping
                            Mapping between atoms in a mutation.

                            .. rst-class:: arrow-down
                        -

                - LigandNetwork
                    A network of mutations between the provided ligands.

                    .. rst-class:: arrow-down
                -

            -
                - :class:`openfe.SmallMoleculeComponent`, :class:`openfe.SolventComponent`, :class:`openfe.ProteinComponent`
                    All components are included in the ChemicalSystem

                  .. rst-class:: arrow-down
                -

                - ChemicalSystem
                    A description of the system that each ligand lives in.

                  .. rst-class:: arrow-down
                -

        - AlchemicNetwork
            Combining chemical systems for each ligand with a network that describes how to mutate between them yields a complete description of a simulation campaign.

      .. rst-class:: arrow-down
    *

    * Run
        Def

      .. rst-class:: arrow-down
    *

    * Gather
        Def


.. toctree::

    loading_molecules
    creating_atom_mappings
    dumping_transformation

    
