Protocols and the Execution Model
=================================

Protocols in OpenFE are built on a flexible execution model. 
Result objects are shaped by this model, and therefore some basic
background on it can be useful when looking into the details of simulation
results, even if most users don't need to work with the details of the
execution model.

Each protocol involves a number of steps (called ``ProtocolUnit``\ s) which occur in
some order. Formally, this is described by a directed acyclic graph (DAG),
so the collection of steps to run is called a ``ProtocolDAG``. A
:class:`.Protocol` creates the ``ProtocolDAG``, and a single ``ProtocolDAG``
should give information necessary to obtain an estimate (perhaps a very poor
estimate) of the :math:`\Delta G`. Over the course of a campaign, a single
:class:`.Protocol` may create multiple ``ProtocolDAG``\ s, e.g., to extend a
simulation. NB: While independent runs can be created as separate
``ProtocolDAG``\ s, the recommend way to do independent runs is as a
``repeats`` part of the settings for the protocol, which puts the
independent runs in a single ``ProtocolDAG``.

There are results objects at each level of this: so the
:class:`.ProtocolResult` is associated the :class:`.Protocol`, and may be
made from multiple :class:`.ProtocolDAGResult`\ s. Similarly, each
:class:`.ProtocolDAGResult` may carry information about multiple
:class:`.ProtocolUnitResult`\ s.

.. TODO FUTURE: add information about scratch/shared/permanent storage
   once that becomes relevant
