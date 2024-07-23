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
    generate_minimal_redundant_network
    generate_lomap_network


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

