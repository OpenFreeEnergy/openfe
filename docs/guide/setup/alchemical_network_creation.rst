.. _alchemical_network_creation:

Alchemical Networks: Creation
=============================

In :ref:`the previous section <alchemical_network_model>` we detail the
theory of :class:`.AlchemicalNetwork` objects and what they contain. Here
we explain how you can go about creating a :class:`.AlchemicalNetwork`
by combining its various components.

Python API
----------

You can manually create a :class:`.AlchemicalNetwork` by creating a list
of :class:`.Transformation` objects.
The :ref:`cookbook on creating alchemical networks <cookbook/create_alchemical_network.nblink>`
demonstrates how to do this.

Alchemical Network Planners
---------------------------

OpenFE provides convenience classes for creating :class:`.AlchemicalNetwork`.
These currently include;

* :class:`.RBFEAlchemicalNetworkPlanner`: creating relative binding free energy networks using the :class:`.RelativeHybridTopologyProtocol`
* :class:`.RHFEAlchemicalNetworkPlanner`: creating relative hydration free energy networks using the :class:`.RelativeHybridTopologyProtocol`

The :ref:`Relative Alchemical Network Planners cookbook <cookbook/rfe_alchemical_planners.nblink>`
demonstrates how to use these.


.. note::
   The Network Planners are provided for user convenience. Whilst they cover
   majority of use cases, they may not currently offer the complete range
   of options available through the Python API.


Command-line interface
----------------------

The Alchemical Network Planners can be used directly through the
:ref:`command line interface <userguide_cli_interface>`.

For example, you can create a relative binding free energy (RBFE) network
using:

.. code:: bash

    $ openfe plan-rbfe-network -p protein.pdb -M dir_with_sdfs/

Similarly you can create a relative hydration free energy (RHFE) network
using:

.. code:: bash

    $ openfe plan-rhfe network -M dir_with_sdfs/

Please see the :ref:`RBFE CLI tutorial <rbfe_cli_tutorial>`
for an example on how to use the CLI to run an RBFE campaign.

.. todo: link to appropriate CLI page in the userguide?
