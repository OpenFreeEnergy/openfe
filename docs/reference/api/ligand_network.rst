Ligand Network Tools
====================

These tools 

Atom Mappers
------------

.. autoclass:: openfe.setup.LomapAtomMapper

.. autoclass:: openfe.setup.PersesAtomMapper


Scorers
-------

LOMAP Scorers
~~~~~~~~~~~~~

.. apparently we need the atom_mapping because internally autofunction is
   trying ``import openfe.setup.lomap_scorers``, which doesn't work (whereas
   ``from openfe.setup import lomap_scorers`` does)

.. autofunction:: openfe.setup.atom_mapping.lomap_scorers.default_lomap_score
.. autofunction:: openfe.setup.atom_mapping.lomap_scorers.ecr_score
.. autofunction:: openfe.setup.atom_mapping.lomap_scorers.mcsr_score
.. autofunction:: openfe.setup.atom_mapping.lomap_scorers.mncar_score
.. autofunction:: openfe.setup.atom_mapping.lomap_scorers.tmcsr_score
.. autofunction:: openfe.setup.atom_mapping.lomap_scorers.atomic_number_score
.. autofunction:: openfe.setup.atom_mapping.lomap_scorers.hybridization_score
.. autofunction:: openfe.setup.atom_mapping.lomap_scorers.sulfonamides_score
.. autofunction:: openfe.setup.atom_mapping.lomap_scorers.heterocycles_score
.. autofunction:: openfe.setup.atom_mapping.lomap_scorers.transmuting_methyl_into_ring_score
.. autofunction:: openfe.setup.atom_mapping.lomap_scorers.transmuting_ring_sizes_score




PersesScorers
~~~~~~~~~~~~~

.. autofunction:: openfe.setup.atom_mapping.perses_scorers.default_perses_scorer

Network Planners
----------------

.. autofunction:: openfe.setup.ligand_network_planning.generate_radial_network

.. autofunction:: openfe.setup.ligand_network_planning.generate_maximal_network

.. autofunction:: openfe.setup.ligand_network_planning.generate_minimal_spanning_network
