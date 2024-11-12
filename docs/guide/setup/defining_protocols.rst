.. _defining-protocols:

Creating and using Protocols
============================

A :class:`.Protocol` is a computational method for estimating the free energy difference between two chemical systems.

Included in the ``openfe`` are the following common protocols:
   * :class:`.RelativeHybridTopologyProtocol`
   * :class:`.AbsoluteSolvationProtocol`
   * :class:`.PlainMDProtocol`

For more information, see  :ref:`userguide_protocols`.

The :class:`.Protocol` class is intended to be built upon to create new, custom protocols,
and several are in development by the openfe development team as well as  external academic groups

Because :class:`.Protocol`\s share a common interface for how they are created and executed,
it is relatively straightforward to try out a new method,
or benchmark several to choose the best for a particular project.

Defining Settings and Creating Protocols
----------------------------------------

The ``settings`` for a given ``Protocol`` allow for user-defined behavior.
Each ``Protocol`` has a ``.default_settings()`` method, which will provide a sensible default
starting point and relevant documentation.

.. TODO: print what a settings object looks like, or how you might define custom settings

For example, to create an instance of the OpenMM RFE Protocol with default settings::

   from openfe.protocols import openmm_rfe

   settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
   protocol = openmm_rfe.RelativeHybridTopologyProtocol(settings)

``Protocol`` objects **cannot be modified once created**. This is crucial for data provenance.
Therefore, the ``Settings`` objects must be customised *before* the ``Protocol`` object is created.
For example, to customise the production run length of the RFE Protocol::

   from openfe.protocols import openmm_rfe

   settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
   settings.simulation_settings.production_length = '10 ns'

   protocol = openmm_rfe.RelativeHybridTopologyProtocol(settings)


Creating Transformations from Protocols
-----------------------------------------

With only ``settings`` defined, a ``Protocol`` contains no chemistry-specific information.
This means that a single ``Protocol`` object can be applied to multiple pairs of ``ChemicalSystem`` objects
to measure each free energy difference.

The :class:`.Transformation` class connects two ``ChemicalSystem`` objects with a ``Protocol``, and
often a :ref:`AtomMapping <userguide_mappings>` (depending on the system).

A ``Transformation`` object is then capable of creating computational work via the :func:`.Transformation.create()` method.
For further details on this, refer to the :ref:`userguide_execution` section.

Finally, a ``Protocol`` is responsible for using the data generated in this process to perform further analysis,
such as generating an estimate of the free energy difference.
For further details on this refer to the :ref:`userguide_results` section,
or the details of each method in :ref:`userguide_protocols`.
