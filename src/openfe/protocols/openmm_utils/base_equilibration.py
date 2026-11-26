# This code is part of OpenFE and is licensed under the MIT license.
"""Minimal GUFE-facing infrastructure for protocol-specific equilibration units.

This module deliberately does not define an equilibration schedule. Protocol
subclasses interpret their upstream results and implement their scientific methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gufe
import openmm
from openmm import app, unit as omm_unit


@dataclass(frozen=True)
class EquilibrationInputs:
    """In-memory inputs resolved by a protocol-specific subclass.

    ``states`` is a nonempty list/tuple. Each of ``systems``, ``positions`` and
    ``topologies`` is either a shared single value or a list/tuple with exactly
    one value per state. One-element lists are *not* broadcast to many states.

    Positions must be OpenMM Quantity objects carrying length units, not
    bare NumPy arrays. PlainMD converts positions.npy (stored in nm) to a
    Quantity in its ``_execute()``. These are execution-time objects, not
    serialized GUFE ProtocolUnit inputs.

    A topology must include every particle in its corresponding OpenMM System
    in the same order. A PDB subset intended for visualization is insufficient.
    """

    states: list[Any] | tuple[Any, ...]
    systems: openmm.System | list[openmm.System] | tuple[openmm.System, ...]
    positions: (
        omm_unit.Quantity
        | list[omm_unit.Quantity]
        | tuple[omm_unit.Quantity, ...]
    )
    topologies: app.Topology | list[app.Topology] | tuple[app.Topology, ...]


@dataclass(frozen=True)
class EquilibrationEntry:
    """Inputs for exactly one thermodynamic state, used only during execution."""

    index: int
    state: Any  # None for PlainMD; future protocols may provide alchemical states.
    system: openmm.System
    positions: omm_unit.Quantity
    topology: app.Topology


class BaseEquilibrationUnit(gufe.ProtocolUnit, ABC):
    """Shared iteration and file layout for protocol-specific equilibration.

    Subclasses implement GUFE ``_execute`` to interpret upstream results and
    construct ``EquilibrationInputs``. They implement ``_equilibrate_state``
    to run the appropriate protocol-specific physics. This base class only
    normalizes and iterates over state inputs, organizes paths and indexes
    outputs; it does not define an MD schedule or alchemical Hamiltonian.

    Broadcasting shares references; it does not copy OpenMM objects. A
    subclass must not modify shared input Systems or positions in place.
    """

    @staticmethod
    def _broadcast(value: Any, n_states: int, name: str) -> tuple[Any, ...]:
        """Broadcast a scalar or require exactly one list/tuple value per state."""
        if isinstance(value, (list, tuple)):
            if len(value) != n_states:
                raise ValueError(
                    f"{name} must contain {n_states} entries, received {len(value)}"
                )
            return tuple(value)
        return (value,) * n_states

    @classmethod
    def _normalise_inputs(
        cls, inputs: EquilibrationInputs
    ) -> tuple[EquilibrationEntry, ...]:
        """Create aligned entries without copying or mutating input objects."""
        if not isinstance(inputs, EquilibrationInputs):
            raise TypeError("equilibration_inputs must be EquilibrationInputs")
        if not isinstance(inputs.states, (list, tuple)) or not inputs.states:
            raise ValueError("states must be a non-empty list or tuple")

        n_states = len(inputs.states)
        systems = cls._broadcast(inputs.systems, n_states, "systems")
        positions = cls._broadcast(inputs.positions, n_states, "positions")
        topologies = cls._broadcast(inputs.topologies, n_states, "topologies")
        return tuple(
            EquilibrationEntry(
                index=index,
                state=state,
                system=system,
                positions=position,
                topology=topology,
            )
            for index, (state, system, position, topology) in enumerate(
                zip(inputs.states, systems, positions, topologies)
            )
        )

    @abstractmethod
    def _equilibrate_state(
        self,
        *,
        state_input: EquilibrationEntry,
        output_dir: Path,
        scratch_dir: Path,
        dry: bool,
        verbose: bool,
        **state_kwargs: Any,
    ) -> dict[str, Any]:
        """Equilibrate one state using protocol-specific physics and settings.

        ``run()`` forwards protocol-specific keyword arguments to this hook.
        Concrete subclasses should declare those arguments explicitly so
        misspellings raise TypeError. The base adds the authoritative index.
        """

    def _output_dir(
        self, *, state_input: EquilibrationEntry, n_entries: int, basepath: Path
    ) -> Path:
        """Use legacy root for PlainMD; isolate files for multiple states."""
        return basepath if n_entries == 1 else basepath / f"state_{state_input.index:03d}"

    def run(
        self,
        *,
        equilibration_inputs: EquilibrationInputs,
        dry: bool = False,
        verbose: bool = True,
        scratch_basepath: Path | None = None,
        shared_basepath: Path | None = None,
        **state_kwargs: Any,
    ) -> dict[str, Any]:
        """Equilibrate resolved in-memory inputs.

        For direct calls, pass ``EquilibrationInputs`` holding OpenMM objects,
        not upstream GUFE ``ProtocolUnitResult`` objects. Each protocol
        subclass's ``_execute()`` loads its own upstream artifacts, checks
        the execution environment, and calls this method.

        Additional keyword arguments are forwarded unchanged to the concrete
        ``_equilibrate_state()``. For PlainMD these are the validated NVT and
        NPT step counts supplied by the setup unit. Concrete hooks should
        declare these explicitly rather than swallowing arbitrary keywords.
        """
        scratch_basepath = Path(".") if scratch_basepath is None else Path(scratch_basepath)
        shared_basepath = Path(".") if shared_basepath is None else Path(shared_basepath)
        entries = self._normalise_inputs(equilibration_inputs)

        outputs = []
        for state_input in entries:
            output_dir = self._output_dir(
                state_input=state_input, n_entries=len(entries), basepath=shared_basepath
            )
            scratch_dir = self._output_dir(
                state_input=state_input, n_entries=len(entries), basepath=scratch_basepath
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            scratch_dir.mkdir(parents=True, exist_ok=True)
            result = self._equilibrate_state(
                state_input=state_input,
                output_dir=output_dir,
                scratch_dir=scratch_dir,
                dry=dry,
                verbose=verbose,
                **state_kwargs,
            )
            outputs.append({**result, "index": state_input.index})

        return {"equilibrated_states": outputs}
