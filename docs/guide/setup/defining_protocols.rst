.. _defining-protocols:

Protocols in OpenFE
============================

A :class:`.Protocol` is a computational method for estimating the free energy difference between two chemical systems.

Just as there are multiple possible methods for estimating free energy differences,
there are multiple available ``Protocol``\s to choose from.

For example, included in the ``openfe`` package are the following:
   * :class:`.RelativeHybridTopologyProtocol`
   * :class:`.AbsoluteBindingProtocol`
   * :class:`.SepTopProtocol`
   * :class:`.AbsoluteSolvationProtocol`
   * :class:`.PlainMDProtocol`

More protocols are in development, and a full list of available protocols
can be found at :ref:`userguide_protocols`.

Because :class:`.Protocol`\s share a common interface for how they are created and executed,
it is relatively straightforward to try out a new method,
or benchmark several to choose the best for a particular project.

Defining Settings and Creating Protocols
----------------------------------------

A ``Settings`` object contains all the parameters needed by a ``Protocol``.
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

Adaptive Settings
~~~~~~~~~~~~~~~~~

.. warning::

  The ``_adaptive_settings()`` method is experimental and subject to change.

In addition to the ``.default_settings()`` method, some protocols
provide an ``_adaptive_settings`` method. This method generates recommended settings
based on properties of the input :class:`.ChemicalSystem`\s and, where required, the :class:`.AtomMapping`.

For example::

   from openfe.protocols import openmm_rfe

   settings = openmm_rfe.RelativeHybridTopologyProtocol._adaptive_settings(
       stateA=stateA,
       stateB=stateB,
       mapping=mapping,
   )

   protocol = openmm_rfe.RelativeHybridTopologyProtocol(settings)

The adaptive settings may modify parameters based on properties of the input systems.
For example (:class:`.RelativeHybridTopologyProtocol`):

* Transformations involving a change in net charge use a larger number of lambda windows and longer production simulations.
* If both states contain a :class:`.ProteinComponent`, the solvation padding is set to 1 nm.

Optionally, you can pass a preexisting settings object to the ``_adaptive_settings`` method via the ``initial_settings`` argument. If provided, an adapted copy of these settings will be returned instead
of using the default settings.

In systems containing membrane-protein complexes (i.e. using a
:class:`.ProteinMembraneComponent`), adaptive settings select a membrane-appropriate barostat, the ``MonteCarloMembraneBarostat``.

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
