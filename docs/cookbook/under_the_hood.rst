.. _under-the-hood:

Under the Hood
==============

.. module:: openfe
    :noindex:

If you want to implement your own atom mapper or free energy procedure, or you want to do something a bit more bespoke, it's helpful to understand how OpenFE thinks about individual alchemic mutation specifications. A :class:`Transformation` stores all the information needed to run an alchemic mutation from one chemical system to another and is the basic unit of an OpenFE simulation campaign. Indeed, :class:`Transformation` objects describe the edges of the graph in the :class:`AlchemicalNetwork` class.

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
                            The ligands that will be mutated.

                            .. rst-class:: arrow-down
                        - :any:`Creating Atom Mappings`

                        - :class:`LigandAtomMapping`
                            Corresponds atoms in one small molecule to those in another.

                            .. rst-class:: arrow-down arrow-tail arrow-combine
                        -

                    -   -
                            .. rst-class:: arrow-down arrow-head arrow-combine-left
                        -

                        - :class:`SmallMoleculeComponent`, :class:`SolventComponent` and :class:`ProteinComponent`
                            The components that make up the chemical system.

                            .. rst-class:: arrow-down arrow-multiple-combine
                        - :any:`Assembling into ChemicalSystems`

                        - :class:`ChemicalSystem`
                            Each of the chemical systems, composed of components, that the :class:`Transformation` mutates between.

                            .. rst-class:: arrow-down arrow-tail arrow-combine-left arrow-multiple
                        -

            .. rst-class:: arrow-down arrow-head
        -

        - :class:`Transformation`
            A single alchemic mutation from one chemical system to another.

      .. rst-class:: arrow-down
    *

    * Run
        - :class:`Transformation`
            A single alchemic mutation from one chemical system to another.

            .. rst-class:: arrow-down
        -

        - :class:`ProtocolDAG`
            A directed acyclic graph describing how to compute a :class:`Transformation`.

        - .. container:: flowchart-sidebyside

            -
                -
                    .. rst-class:: arrow-down arrow-multiple
                -

                - :class:`ProtocolUnit`
                    A single unit of computation within a :class:`ProtocolDAG`

                    .. rst-class:: arrow-down
                -

                - :class:`ProtocolUnitResult`
                    The result of a completed :class:`ProtocolUnit`

                    .. rst-class:: arrow-down arrow-multiple-combine
                -

            -
                -
                    .. rst-class:: arrow-down
                - :any:`openfe.orchestration`

        - :class:`ProtocolDAGResult`
            A completed transformation.

      .. rst-class:: arrow-down
    *

    * Gather
        - .. container:: flowchart-sidebyside

            -

                - :class:`Transformation`
                    The specification for the alchemic mutation.

                    .. rst-class:: arrow-down
                -

                - :class:`Protocol`
                    A completed single run of a transformation.

                    .. rst-class:: arrow-down arrow-combine-right arrow-tail
                -

            -

                - :class:`ProtocolResult`
                    A completed single run of a transformation.

                    .. rst-class:: arrow-down arrow-combine-left arrow-multiple arrow-tail
                -

            .. rst-class:: arrow-down arrow-head
        -

        - :class:`ProtocolDAGResult`
                A completed transformation with multiple user-defined replicas.

