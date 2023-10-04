# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Descriptions and implementations of free energy calculation protocols.

A :class:`.Protocol` describes a methodology for computing a free energy from a
pair of chemical systems. It produces a number of :class:`ProtocolDAG` objects,
which are directed acyclic graphs describing the dependencies between
individual units of work that can otherwise be run in parallel. Each unit of
work is described and implemented by a :class:`ProtocolUnit`. The results of a
``ProtocolUnit`` are described by a :class:`ProtocolUnitResult` when successful
or else a :class:`ProtocolUnitFailure`; these types are then collected into a
:class:`ProtocolDAGResult` to describe the result of a protocol DAG, and
multiple DAGs from different transformations in the same campaign can be
collected into a :class:`ProtocolResult`.

``Protocol`` instances are configured by Pydantic models describing their
settings.. Base classes and a number of implemented building blocks for complex
configurations are found in the :class:`settings` module.

Other submodules in this module represent individual implementations of
protocols.
"""

from gufe.protocols import (
    Protocol,
    ProtocolDAG,
    ProtocolUnit,
    ProtocolUnitResult,
    ProtocolUnitFailure,
    ProtocolDAGResult,
    ProtocolResult,
)

from . import openmm_rfe, openmm_utils, settings

__all__ = [
    "settings",
    "openmm_rfe",
    "Protocol",
    "ProtocolDAG",
    "ProtocolUnit",
    "ProtocolUnitResult",
    "ProtocolUnitFailure",
    "ProtocolDAGResult",
    "ProtocolResult",
]
