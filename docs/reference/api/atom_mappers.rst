.. _Atom Mappers:

Atom Mappings
=============

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
