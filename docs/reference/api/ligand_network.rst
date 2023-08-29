Ligand Network Tools
====================


Atom Mappers
------------

.. module:: openfe.setup
   :noindex:

.. autosummary::
   :nosignatures:
   :toctree: generated/

   LomapAtomMapper
   PersesAtomMapper

.. _Atom Map Scorers:

Atom Map Scorers
----------------

LOMAP Scorers
~~~~~~~~~~~~~

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
   tmcsr_score
   atomic_number_score
   hybridization_score
   sulfonamides_score
   heterocycles_score
   transmuting_methyl_into_ring_score
   transmuting_ring_sizes_score




PersesScorers
~~~~~~~~~~~~~

.. module:: openfe.setup.atom_mapping.perses_scorers

.. autosummary::
   :nosignatures:
   :toctree: generated/

   default_perses_scorer

.. _Ligand Network Planners:

Network Planners
----------------

.. module:: openfe.setup.ligand_network_planning

.. autosummary::
   :nosignatures:
   :toctree: generated/

   generate_radial_network
   generate_maximal_network
   generate_minimal_spanning_network
