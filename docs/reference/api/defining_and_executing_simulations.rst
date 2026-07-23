.. _reference_execution:

Defining and Executing Simulations
==================================

.. _executors:

Executing Simulations
---------------------

.. module:: openfe
    :noindex:

.. autosummary::
    :toctree: generated/
    :recursive:

    execute_DAG
    storage

General classes
---------------

.. module:: openfe
    :noindex:

.. autosummary::
    :nosignatures:
    :toctree: generated/

    ProtocolDAG
    ProtocolUnitResult
    ProtocolUnitFailure
    ProtocolDAGResult

Specialized classes
-------------------

These classes are abstract classes that are specialized (subclassed) for an individual Protocol.

.. module:: openfe
    :noindex:

.. autosummary::
    :nosignatures:
    :toctree: generated/

    Protocol
    ProtocolUnit
    ProtocolResult
