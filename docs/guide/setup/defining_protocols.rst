.. _defining-protocols:

Creating and using Protocols
============================

With a thermodynamic cycle of interest identified and represented using `ChemicalSystem` objects,
the next step is to choose a method for estimating the free energy differences.
This is done using a :class:`.Protocol`,
a computational method for estimating the free energy difference between two chemical systems.

Just as there are multiple possible methods for estimating free energy differences,
there are multiple available Protocols to choose from.
For example, included in the ``openfe`` package are the
:class:`.RelativeHybridTopologyProtocol`,
:class:`.AbsoluteSolvationProtocol`,
and :class:`.PlainMDProtocol`;
for a full list see :ref:`userguide_protocols`.
This selection is being built upon,
both by the openfe development team as well as with external academic groups.

What all these Protocols share is a common interface for how they are created and executed,
therefore it is relatively simple to
try out a new method,
or benchmark several to choose the best for a particular project.

Settings and creating Protocols
-------------------------------

A ``Protocol`` has a variety of options which control the details of how it behaves,
these are referred to as ``Settings``.
While methods which share common foundations and approaches will share many of these options in common,
ultimately the best resource for understanding these will be the documentation for each particular ``Protocol``.
Each Protocol will define a ``.default_settings()`` method,
which will return a sensible default starting point.
For example, to create an instance of the OpenMM RFE Protocol with default settings::

   from openfe.protocols import openmm_rfe

   settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
   prot = openmm_rfe.RelativeHybridTopologyProtocol(settings)

A key point of ``Protocol`` objects is that they cannot be modified once created,
this is important for later tracking the provenance of data,
therefore the ``Settings`` objects must be customised before the ``Protocol`` object is created.
For example to customise the production run length of the RFE Protocol::

   from openfe.protocols import openmm_rfe

   settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
   settings.simulation_settings.production_length = '10 ns'

   prot = openmm_rfe.RelativeHybridTopologyProtocol(settings)

Using Protocols
---------------

Up until this point, the ``Protocol`` has not involved any of the specific chemistry that is of interest.
This means that a single defined ``Protocol`` can be applied to multiple pairs of ``ChemicalSystem`` objects
to measure each difference.
The :class:`.Transformation` object is a handy container for connecting two ``ChemicalSystem`` objects
and the ``Protocol`` together.
Often a :class:`.LigandAtomMapping` object is also required to define the correspondence of atoms,
for further details refer to the :ref:`userguide_mappings` section.

The ``Transformation`` object is then capable of creating computational work via the :func:`.Transformation.create()` method.
For further details on this, refer to the :ref:`userguide_execution` section.
Finaly, a ``Protocol`` is responsible for using the data generated in this process to perform further analysis,
such as generating an estimate of the free energy difference.
For further details on this refer to the :ref:`userguide_results` section,
or the details of each method in :ref:`userguide_protocols`.
