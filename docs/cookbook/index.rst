Cookbook
========

This will include various how-to guides.

.. module:: openfe
    :noindex:

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
                - :class:`SmallMoleculeComponent`
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

                        - :class:`LigandAtomMapping`
                            Mapping between atoms in one small molecule and those in another for a single transformation (network edge).

                            .. rst-class:: arrow-down
                        -

                - :class:`LigandNetwork`
                    A network of mutations between the provided ligands.

                    .. rst-class:: arrow-down
                -

            -
                - :class:`SmallMoleculeComponent`, :class:`SolventComponent`, :class:`ProteinComponent`
                    All components are included in the ChemicalSystem

                  .. rst-class:: arrow-down
                -

                - :class:`ChemicalSystem`
                    A description of the system that each ligand lives in.

                  .. rst-class:: arrow-down
                -

        - :class:`AlchemicalNetwork`
            A complete description of a simulation campaign, including the :class:`LigandNetwork` and a :class:`ChemicalSystem` for each ligand.

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

    
