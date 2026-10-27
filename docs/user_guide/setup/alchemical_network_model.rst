.. _alchemical_network_model:

Alchemical Networks: Planning a Simulation Campaign
===================================================

The ultimate goal of the setup stage is to create an :class:`.AlchemicalNetwork`,
which contains all the information needed for a campaign of simulations, including the 
``openfe`` objects that define the chemical systems and alchemical transformations.

.. TODO provide a written or image based comparison between alchemical and thermodynamic cycles

Like any network, an :class:`.AlchemicalNetwork` can be described in terms
of nodes and edges between nodes. The nodes are :class:`.ChemicalSystem`\s,
which describe the specific molecules involved. The edges are
:class:`.Transformation` objects, which carry all the information about how
the simulation is to be performed.


.. figure:: img/AlchemicalNetwork.png

In practice, nodes must be associated with a transformation in order to be
relevant in an alchemical network; that is, there are no disconnected nodes.
This means that the alchemical network can be fully described by just the
edges (which contain information on the nodes they connect). Note that this
does not mean that the entire network must be fully connected -- just that
there are no solitary nodes.

Each :class:`.Transformation` represents everything that is needed to
calculate the free energy differences between the two
:class:`.ChemicalSystem`\ s that are the nodes for that edge. In addition to
containing the information for each :class:`.ChemicalSystem`, the
:class:`.Transformation` also contains a :class:`.Protocol` and, when
relevant, atom mapping information for alchemical transformations. The latter
is often done through a :class:`.LigandNetwork`.

.. _alchemical_network_creation:

3 Ways to Create an Alchemical Network
--------------------------------------

1. Python API 
^^^^^^^^^^^^^

You can manually create a :class:`.AlchemicalNetwork` by creating a list
of :class:`.Transformation` objects. For examples using the Python API,
see :ref:`cookbook on creating alchemical networks <cookbook/create_alchemical_network.nblink>`.

2. Python ``NetworkPlanner`` Convenience Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OpenFE also provides the convenience classes :class:`.RBFEAlchemicalNetworkPlanner` and :class:`.RHFEAlchemicalNetworkPlanner`,
which use the :class:`.RelativeHybridTopologyProtocol` for creating :class:`.AlchemicalNetwork`\s.
For example usage of these convenience classes, see :ref:`Relative Alchemical Network Planners cookbook <cookbook/rfe_alchemical_planners.nblink>`.

.. note::
   The Network Planners are provided for user convenience. While they cover
   majority of use cases, they may not currently offer the complete range
   of options available through the Python API.

3. Command Line ``NetworkPlanner`` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Alchemical Network Planners can also be called directly from the 
:ref:`command line interface <userguide_cli_interface>`.

For example, you can create a Relative Hydration Free Energy (RHFE) network
using:

.. code:: bash

    $ openfe plan-rhfe-network -M dir_with_sdfs/

or a Relative Binding Free Energy (RBFE) network using:

.. code:: bash

    $ openfe plan-rbfe-network -p protein.pdb -M dir_with_sdfs/


For more CLI details, see :ref:`RBFE CLI tutorial <rbfe_cli_tutorial>` and the :ref:`userguide_cli_interface`.

See Also
--------
* :ref:`Alchemical Network API reference <Alchemical Network Planning>`
* :ref:`Chemical Systems UserGuide entry <userguide_chemicalsystems_and_components>`