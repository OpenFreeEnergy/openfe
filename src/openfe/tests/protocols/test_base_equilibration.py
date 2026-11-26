# This code is part of OpenFE and is licensed under the MIT license.
"""Tests for the shared equilibration runner (no OpenMM integration needed)."""

from pathlib import Path

import pytest

from openfe.protocols.openmm_utils.base_equilibration import (
    BaseEquilibrationUnit,
    EquilibrationInputs,
)


class RecordingUnit:
    """Exercise concrete BaseEquilibrationUnit methods without a GUFE DAG.

    The real GUFE constructor requires serializable ProtocolUnit inputs; this
    harness tests the base runner's in-memory inputs independently.
    """

    _normalise_inputs = BaseEquilibrationUnit._normalise_inputs
    _output_dir = BaseEquilibrationUnit._output_dir
    run = BaseEquilibrationUnit.run

    def __init__(self):
        self.calls = []

    def _equilibrate_state(
        self, *, state_input, output_dir, scratch_dir, dry, verbose,
        equil_steps_nvt, equil_steps_npt,
    ):
        self.calls.append({
            "state_input": state_input,
            "output_dir": output_dir,
            "scratch_dir": scratch_dir,
            "dry": dry,
            "verbose": verbose,
            "nvt": equil_steps_nvt,
            "npt": equil_steps_npt,
        })
        return {"index": 999, "state_path": output_dir / "equilibrated_state.xml"}


def _inputs(n=1, **overrides):
    values = dict(
        states=list(range(n)),
        systems=object(),
        positions=object(),
        topologies=object(),
    )
    values.update(overrides)
    return EquilibrationInputs(**values)


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
    assert result["equilibrated_states"][0]["index"] == 0  # authoritative


def test_multiple_states_use_separate_paths_and_broadcast_scalars(tmp_path):
    system, positions, topology = object(), object(), object()
    unit = RecordingUnit()
    result = unit.run(
        equilibration_inputs=_inputs(
            2, systems=system, positions=positions, topologies=topology,
        ),
        equil_steps_nvt=3,
        equil_steps_npt=5,
        shared_basepath=tmp_path / "shared",
        scratch_basepath=tmp_path / "scratch",
    )
    assert [x["index"] for x in result["equilibrated_states"]] == [0, 1]
    for i, call in enumerate(unit.calls):
        assert call["state_input"].system is system
        assert call["state_input"].positions is positions
        assert call["state_input"].topology is topology
        assert call["output_dir"] == tmp_path / "shared" / f"state_{i:03d}"
        assert call["scratch_dir"] == tmp_path / "scratch" / f"state_{i:03d}"
        assert call["nvt"] == 3 and call["npt"] == 5
        assert call["output_dir"].is_dir() and call["scratch_dir"].is_dir()


def test_per_state_sequence_inputs_are_aligned():
    systems, positions, topologies = [object(), object()], [object(), object()], [object(), object()]
    entries = BaseEquilibrationUnit._normalise_inputs(
        _inputs(2, systems=systems, positions=positions, topologies=topologies)
    )
    for i, entry in enumerate(entries):
        assert entry.index == i
        assert entry.system is systems[i]
        assert entry.positions is positions[i]
        assert entry.topology is topologies[i]


@pytest.mark.parametrize("field", ["systems", "positions", "topologies"])
@pytest.mark.parametrize("n_values", [0, 1, 3])
def test_sequence_length_must_match_states(field, n_values):
    with pytest.raises(ValueError, match=rf"{field} must contain 2 entries"):
        BaseEquilibrationUnit._normalise_inputs(
            _inputs(2, **{field: [object() for _ in range(n_values)]})
        )


@pytest.mark.parametrize("states", [[], (), "one", None])
def test_states_must_be_nonempty_list_or_tuple(states):
    with pytest.raises(ValueError, match="non-empty list or tuple"):
        BaseEquilibrationUnit._normalise_inputs(_inputs(states=states))


def test_normalization_requires_equilibration_inputs():
    with pytest.raises(TypeError, match="EquilibrationInputs"):
        BaseEquilibrationUnit._normalise_inputs({"states": [None]})


def test_unknown_protocol_specific_keyword_is_rejected(tmp_path):
    unit = RecordingUnit()
    with pytest.raises(TypeError, match="equil_step_nvt"):
        unit.run(
            equilibration_inputs=_inputs(),
            equil_step_nvt=2,  # typo
            equil_steps_npt=4,
            shared_basepath=tmp_path,
        )
