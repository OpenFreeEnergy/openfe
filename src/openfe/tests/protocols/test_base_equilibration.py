# This code is part of OpenFE and is licensed under the MIT license.
"""Tests for the shared equilibration runner (no OpenMM integration needed)."""

from pathlib import Path

import pytest

from openfe.protocols.openmm_utils.base_equilibration import (
    BaseEquilibrationUnit,
    EquilibrationEntry,
)


class RecordingUnit:
    """Exercise concrete BaseEquilibrationUnit methods without a GUFE DAG.

    The real GUFE constructor requires serializable ProtocolUnit inputs; this
    harness tests the base runner's in-memory inputs independently.
    """

    _output_dir = BaseEquilibrationUnit._output_dir
    run = BaseEquilibrationUnit.run

    def __init__(self):
        self.calls = []

    def _equilibrate_state(
        self,
        *,
        state_input,
        output_dir,
        scratch_dir,
        dry,
        verbose,
        equil_steps_nvt,
        equil_steps_npt,
    ):
        self.calls.append(
            {
                "state_input": state_input,
                "output_dir": output_dir,
                "scratch_dir": scratch_dir,
                "dry": dry,
                "verbose": verbose,
                "nvt": equil_steps_nvt,
                "npt": equil_steps_npt,
            }
        )
        return {"index": 999, "state_path": output_dir / "equilibrated_state.xml"}


def _inputs(n=1):
    return [
        EquilibrationEntry(
            index=i,
            state=i,
            system=object(),
            positions=object(),
            topology=object(),
        )
        for i in range(n)
    ]


def test_one_state_preserves_plainmd_root_directories(tmp_path):
    unit = RecordingUnit()
    result = unit.run(
        equilibration_inputs=_inputs(),
        equil_steps_nvt=0,
        equil_steps_npt=2,
        dry=True,
        verbose=False,
        scratch_basepath=tmp_path / "scratch",
        shared_basepath=tmp_path / "shared",
    )
    assert unit.calls[0]["output_dir"] == tmp_path / "shared"
    assert unit.calls[0]["scratch_dir"] == tmp_path / "scratch"
    assert unit.calls[0]["dry"] is True
    assert unit.calls[0]["verbose"] is False
    assert (tmp_path / "scratch").is_dir()
    assert (tmp_path / "shared").is_dir()
    assert result["equilibrated_states"] == [
        {
            "index": 0,
            "state_path": tmp_path / "shared" / "equilibrated_state.xml",
        }
    ]


def test_multiple_states_have_separate_directories(tmp_path):
    unit = RecordingUnit()
    inputs = _inputs(n=3)

    result = unit.run(
        equilibration_inputs=inputs,
        equil_steps_nvt=2,
        equil_steps_npt=4,
        dry=True,
        verbose=False,
        scratch_basepath=tmp_path / "scratch",
        shared_basepath=tmp_path / "shared",
    )

    assert len(unit.calls) == 3
    assert len(result["equilibrated_states"]) == 3

    output_dirs = set()
    scratch_dirs = set()

    for entry, call, output in zip(
        inputs,
        unit.calls,
        result["equilibrated_states"],
    ):
        assert call["state_input"] is entry
        assert call["nvt"] == 2
        assert call["npt"] == 4

        assert call["output_dir"].is_dir()
        assert call["scratch_dir"].is_dir()

        assert call["output_dir"].parent == tmp_path / "shared"
        assert call["scratch_dir"].parent == tmp_path / "scratch"

        assert output["index"] == entry.index
        assert output["state_path"] == (
            call["output_dir"] / "equilibrated_state.xml"
        )

        output_dirs.add(call["output_dir"])
        scratch_dirs.add(call["scratch_dir"])

    assert len(output_dirs) == 3
    assert len(scratch_dirs) == 3


def test_empty_inputs_are_rejected(tmp_path):
    unit = RecordingUnit()

    with pytest.raises(
        ValueError,
        match="At least one equilibration entry is required",
    ):
        unit.run(
            equilibration_inputs=[],
            equil_steps_nvt=2,
            equil_steps_npt=4,
            scratch_basepath=tmp_path / "scratch",
            shared_basepath=tmp_path / "shared",
        )

    assert not unit.calls
    assert not (tmp_path / "scratch").exists()
    assert not (tmp_path / "shared").exists()


def test_duplicate_indices_are_rejected(tmp_path):
    unit = RecordingUnit()
    entry = _inputs()[0]

    with pytest.raises(
        ValueError,
        match="Equilibration entry indices must be unique",
    ):
        unit.run(
            equilibration_inputs=[entry, entry],
            equil_steps_nvt=2,
            equil_steps_npt=4,
            scratch_basepath=tmp_path / "scratch",
            shared_basepath=tmp_path / "shared",
        )

    assert not unit.calls
    assert not (tmp_path / "scratch").exists()
    assert not (tmp_path / "shared").exists()


def test_unknown_protocol_specific_keyword_is_rejected(tmp_path):
    unit = RecordingUnit()

    with pytest.raises(TypeError, match="equil_step_nvt"):
        unit.run(
            equilibration_inputs=_inputs(),
            equil_step_nvt=2,  # typo
            equil_steps_npt=4,
            scratch_basepath=tmp_path / "scratch",
            shared_basepath=tmp_path / "shared",
        )