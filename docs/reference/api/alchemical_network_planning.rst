.. _Alchemical Network Planning:

Simulation Campaign Planning
============================

While a :class:`LigandNetwork` describes a network of ligands and their atom
mappings, a :class:`AlchemicalNetwork` describes a single replicate of a
simulation campaign. It includes all the information needed to perform the
simulation, and so implicitly includes the :class:`LigandNetwork`.

Alchemical Simulations
~~~~~~~~~~~~~~~~~~~~~~

Descriptions of anticipated alchemical simulation campaigns.

.. module:: openfe
    :noindex:

.. autosummary::
    :nosignatures:
    :toctree: generated/

    Transformation
    AlchemicalNetwork

Alchemical Network Planners
---------------------------
Alchemical network planners are objects that pull all the ideas in OpenFE
into a quick setup for simulation. The goal is to create the
:class:`.AlchemicalNetwork` that represents an entire simulation campaign,
starting from a bare amount of user input.

.. module:: openfe.setup
    :noindex:

.. autosummary::
    :nosignatures:
    :toctree: generated/

    RBFEAlchemicalNetworkPlanner
    RHFEAlchemicalNetworkPlanner
