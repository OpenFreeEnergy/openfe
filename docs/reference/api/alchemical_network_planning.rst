Alchemical Network Planning
===========================

Alchemical network planners are objects that pull all the ideas in OpenFE
into a quick setup for simulation. The goal is to create the
:class:`.AlchemicalNetwork` that represents an entire simulation campaign,
starting from a bare amount of user input. This also requries several helper
classes along the way.

Alchemical Network Planners
---------------------------

.. autoclass:: openfe.setup.RBFEAlchemicalNetworkPlanner
.. autoclass:: openfe.setup.RHFEAlchemicalNetworkPlanner


Chemical System Generators
--------------------------

.. autoclass:: openfe.setup.chemicalsystem_generator.EasyChemicalSystemGenerator
