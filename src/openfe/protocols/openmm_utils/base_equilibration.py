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
from openmm import app
from openmm import unit as omm_unit


@dataclass(frozen=True)
class EquilibrationInput:
    """Inputs required to equilibrate a single thermodynamic state."""

    index: int
    state: Any  # None for PlainMD; future protocols may provide alchemical states.
    system: openmm.System
    positions: omm_unit.Quantity
    topology: app.Topology


class BaseEquilibrationUnit(gufe.ProtocolUnit, ABC):
    """Base class for protocol-specific equilibration units.

    Subclasses resolve the upstream results into equilibration
    entries and implement the protocol-specific equilibration
    procedure.

    This class manages iteration, directory layout and collection
    of equilibration results.
    """

    @abstractmethod
    def _equilibrate_state(
        self,
        *,
        state_input: EquilibrationInput,
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
        self, *, state_input: EquilibrationInput, n_entries: int, basepath: Path
    ) -> Path:
        """Use legacy root for PlainMD; isolate files for multiple states."""
        return basepath if n_entries == 1 else basepath / f"state_{state_input.index:03d}"

    def run(
        self,
        *,
        equilibration_inputs: list[EquilibrationInput],
        dry: bool = False,
        verbose: bool = True,
        scratch_basepath: Path | None = None,
        shared_basepath: Path | None = None,
        **state_kwargs: Any,
    ) -> dict[str, Any]:
        """Equilibrate each supplied thermodynamic state."""

        if not equilibration_inputs:
            raise ValueError("At least one equilibration input is required")

        indices = [input.index for input in equilibration_inputs]

        if len(indices) != len(set(indices)):
            raise ValueError("Equilibration input indices must be unique")

        scratch_basepath = Path(".") if scratch_basepath is None else Path(scratch_basepath)
        shared_basepath = Path(".") if shared_basepath is None else Path(shared_basepath)

        outputs = []

        for state_input in equilibration_inputs:
            output_dir = self._output_dir(
                state_input=state_input,
                n_entries=len(equilibration_inputs),
                basepath=shared_basepath,
            )
            scratch_dir = self._output_dir(
                state_input=state_input,
                n_entries=len(equilibration_inputs),
                basepath=scratch_basepath,
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
