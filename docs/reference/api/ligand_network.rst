Ligand Network Tools
====================

.. module:: openfe.setup
    :noindex:

Ligand Network
--------------

A network of mutations between ligands.

.. autosummary::
    :nosignatures:
    :toctree: generated/

    LigandNetwork


.. _Ligand Network Planners:

Network Planners
~~~~~~~~~~~~~~~~

.. module:: openfe.setup.ligand_network_planning

Functions that build a :class:`.LigandNetwork` from a collection of :class:`SmallMoleculeComponents` by optimizing over a `scoring function <Atom Map Scorers>`_.

.. autosummary::
    :nosignatures:
    :toctree: generated/

    generate_radial_network
    generate_maximal_network
    generate_minimal_spanning_network


.. _Ligand Network Loaders:

Network Loaders
~~~~~~~~~~~~~~~

Functions to load a :class:`.LigandNetwork` from equivalent classes in other packages, or to specify one by hand.

.. autosummary::
    :nosignatures:
    :toctree: generated/

    generate_network_from_names
    generate_network_from_indices
    load_orion_network
    load_fepplus_network

.. _Atom Mappers:

Atom Mappings
-------------

Tools for mapping atoms in one molecule to those in another. Used to generate efficient ligand networks.

.. module:: openfe.setup.atom_mapping

.. rubric:: Abstract Base Class

.. autosummary::
    :nosignatures:
    :toctree: generated/

    LigandAtomMapper

.. rubric:: Implementations

.. autosummary::
    :nosignatures:
    :toctree: generated/

    LomapAtomMapper
    PersesAtomMapper

.. rubric:: Data Types

.. autosummary::
    :nosignatures:
    :toctree: generated/

    LigandAtomMapping

.. _Atom Map Scorers:

Atom Map Scorers
----------------

Scoring functions for a mapping between ligands. These are used as objective functions for :any:`Ligand Network Planners`.


Lomap Scorers
~~~~~~~~~~~~~

Scorers implemented by the `LOMAP <https://github.com/OpenFreeEnergy/Lomap>`_ package.

.. apparently we need the atom_mapping because internally autofunction is
    trying ``import openfe.setup.lomap_scorers``, which doesn't work (whereas
    ``from openfe.setup import lomap_scorers`` does)
.. module:: openfe.setup.atom_mapping.lomap_scorers

.. autosummary::
    :nosignatures:
    :toctree: generated/

    default_lomap_score
    ecr_score
    mcsr_score
    mncar_score
    atomic_number_score
    hybridization_score
    sulfonamides_score
    heterocycles_score
    transmuting_methyl_into_ring_score
    transmuting_ring_sizes_score


Perses Scorers
~~~~~~~~~~~~~~

Scorers implemented by the `Perses <https://github.com/choderalab/perses>`_ package.

.. module:: openfe.setup.atom_mapping.perses_scorers

.. autosummary::
    :nosignatures:
    :toctree: generated/

    default_perses_scorer
