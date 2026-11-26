# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import logging
import os
import pathlib
import shutil

import gufe
import openmm
import openmm.unit as openmm_unit
import pytest
from gufe.protocols.errors import ProtocolUnitExecutionError
from openff.units import unit

import openfe
from openfe.data._registry import POOCH_CACHE
from openfe.protocols.openmm_md.plain_md_methods import (
    PlainMDProtocol,
    PlainMDSetupUnit,
    PlainMDEquilibrationUnit,
    PlainMDSimulationUnit,
)
from openfe.protocols.openmm_utils import serialization
from openfe.protocols.openmm_utils.base_equilibration import EquilibrationInputs

from ...conftest import HAS_INTERNET


@pytest.fixture()
def vacuum_protocol_settings():
    # setup a cheap vacuum md protocol
    settings = PlainMDProtocol.default_settings()
    settings.protocol_repeats = 1
    settings.forcefield_settings.nonbonded_method = "nocutoff"
    settings.engine_settings.compute_platform = None
    settings.simulation_settings.equilibration_length_nvt = 1 * unit.picoseconds
    settings.simulation_settings.equilibration_length = 1 * unit.picoseconds
    settings.simulation_settings.production_length = 1 * unit.picoseconds
    settings.output_settings.checkpoint_interval = 0.5 * unit.picoseconds
    settings.output_settings.trajectory_write_interval = 0.5 * unit.picoseconds
    return settings


def test_verify_execution_environment():
    # verify using the current versions of the software
    PlainMDSimulationUnit._verify_execution_environment(
        setup_outputs={
            "gufe_version": gufe.__version__,
            "openfe_version": openfe.__version__,
            "openmm_version": openmm.__version__,
        }
    )


def test_verify_execution_environment_fail():
    # pass in different versions to force failure
    with pytest.raises(ProtocolUnitExecutionError, match="Python environment"):
        PlainMDSimulationUnit._verify_execution_environment(
            setup_outputs={
                "gufe_version": 0.1,
                "openfe_version": openmm.__version__,
                "openmm_version": openmm.__version__,
            }
        )


def test_verify_execution_env_missing_key():
    errmsg = "Missing environment information from setup outputs."
    with pytest.raises(ProtocolUnitExecutionError, match=errmsg):
        PlainMDSimulationUnit._verify_execution_environment(
            setup_outputs={
                "foo_version": 0.1,
                "openfe_version": openfe.__version__,
                "openmm_version": openmm.__version__,
            },
        )


@pytest.mark.skipif(
    not os.path.exists(POOCH_CACHE) and not HAS_INTERNET,
    reason="Internet unavailable and test data is not cached locally",
)
def test_check_restart(vacuum_protocol_settings, plain_md_checkpoint_path):
    # test we can correctly detect when we should be restarting
    assert PlainMDSimulationUnit._check_restart(
        output_settings=vacuum_protocol_settings.output_settings,
        shared_path=plain_md_checkpoint_path.parent,
    )

    # make sure it does not try and restart if inputs are missing
    assert not PlainMDSimulationUnit._check_restart(
        output_settings=vacuum_protocol_settings.output_settings,
        shared_path=pathlib.Path("."),
    )


@pytest.mark.parametrize(
    "step, remaining",
    [
        (0, (250, 250)),
        (125, (125, 250)),  # partial NVT
        (250, (0, 250)),  # exactly at NVT/NPT boundary
        (375, (0, 125)),  # partial NPT
        (500, (0, 0)),  # equilibration complete
    ],
)
def test_remaining_equilibration_steps(step, remaining):
    assert PlainMDEquilibrationUnit._get_remaining_equilibration_steps(
        current_step_count=step,
        equil_steps_nvt=250,
        equil_steps_npt=250,
    ) == remaining


@pytest.mark.parametrize("step", [-1, 501, 625])
def test_invalid_equilibration_checkpoint_step(step):
    with pytest.raises(ValueError, match="outside the equilibration interval"):
        PlainMDEquilibrationUnit._get_remaining_equilibration_steps(
            current_step_count=step,
            equil_steps_nvt=250,
            equil_steps_npt=250,
        )


def test_remaining_equilibration_without_nvt():
    assert PlainMDEquilibrationUnit._get_remaining_equilibration_steps(
        current_step_count=125,
        equil_steps_nvt=0,
        equil_steps_npt=250,
    ) == (0, 125)


@pytest.mark.skipif(
    not os.path.exists(POOCH_CACHE) and not HAS_INTERNET,
    reason="Internet unavailable and test data is not cached locally",
)
class TestPlainMDResume:
    @pytest.fixture
    def protocol_dag(self, vacuum_protocol_settings, benzene_vacuum_system):
        protocol = PlainMDProtocol(
            settings=vacuum_protocol_settings,
        )
        return protocol.create(
            stateA=benzene_vacuum_system, stateB=benzene_vacuum_system, mapping=None
        )

    def test_resume(
        self, protocol_dag, tmp_path, caplog, vacuum_protocol_settings, plain_md_checkpoint_path
    ):
        setup_unit, equil_unit, simulation_unit = list(protocol_dag.protocol_units)
        assert isinstance(setup_unit, PlainMDSetupUnit)
        assert isinstance(equil_unit, PlainMDEquilibrationUnit)
        assert isinstance(simulation_unit, PlainMDSimulationUnit)

        setup_path = tmp_path / "setup"
        equil_path = tmp_path / "equilibration"
        prod_path = tmp_path / "production"
        for path in (setup_path, equil_path, prod_path):
            path.mkdir()

        # dry run the setup unit
        setup_results = setup_unit.run(
            dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path
        )
        equil_results = equil_unit.run(
            equilibration_inputs=EquilibrationInputs(
                states=[None],
                systems=serialization.deserialize(setup_results["system"]),
                positions=setup_results["debug"]["positions"],
                topologies=setup_results["debug"]["topology"],
            ),
            equil_steps_nvt=setup_results["equil_steps_nvt"] or 0,
            equil_steps_npt=setup_results["equil_steps_npt"],
            scratch_basepath=equil_path,
            shared_basepath=equil_path,
        )
        equilibrated = equil_results["equilibrated_states"][0]
        assert equilibrated["equilibration_end_step"] == 500
        # Create a genuine production checkpoint at 2.5 ps (625 steps):
        # the equilibrated XML State contributes the first 500 steps, then
        # we simulate 125 of the 250 production steps.
        checkpoint = prod_path / vacuum_protocol_settings.output_settings.checkpoint_storage_filename
        partial = openmm.app.Simulation(
            setup_results["debug"]["topology"],
            serialization.deserialize(setup_results["system"]),
            openmm.LangevinMiddleIntegrator(
                298.15 * openmm_unit.kelvin,
                1.0 / openmm_unit.picosecond,
                4 * openmm_unit.femtoseconds,
            ),
        )
        partial.loadState(str(equilibrated["equilibrated_state"]))
        partial.step(125)
        partial.saveState(str(checkpoint))
        assert partial.context.getStepCount() == 625
        del partial

        assert PlainMDSimulationUnit._check_restart(
            output_settings=vacuum_protocol_settings.output_settings,
            shared_path=prod_path,
        )

        caplog.set_level(logging.INFO)
        sim_results = simulation_unit.run(
            system=serialization.deserialize(setup_results["system"]),
            topology=setup_results["debug"]["topology"],
            equilibrated_state=equilibrated["equilibrated_state"],
            equilibration_end_step=equilibrated["equilibration_end_step"],
            prod_steps=setup_results["prod_steps"],
            reinitialize_velocities_for_production=equilibrated[
                "reinitialize_velocities_for_production"
            ],
            verbose=True,
            scratch_basepath=prod_path,
            shared_basepath=prod_path,
        )
        assert "Restarting simulation from checkpoint state" in caplog.text
        assert "Running production phase for 125 steps" in caplog.text
        assert "Running NPT equilibration" not in caplog.text
        assert sim_results["nc"].exists()
        assert sim_results["last_checkpoint"] is not None
        assert sim_results["last_checkpoint"].exists()

        # The final checkpoint should contain the total 3 ps, 750 steps.
        final = openmm.app.Simulation(
            setup_results["debug"]["topology"],
            serialization.deserialize(setup_results["system"]),
            openmm.LangevinMiddleIntegrator(
                298.15 * openmm_unit.kelvin,
                1.0 / openmm_unit.picosecond,
                4 * openmm_unit.femtoseconds,
            ),
        )
        final.loadState(str(sim_results["last_checkpoint"]))
        assert final.context.getTime().value_in_unit(openmm_unit.picoseconds) == pytest.approx(3)
        assert final.context.getStepCount() == 750
        del final


@pytest.mark.skipif(
    not os.path.exists(POOCH_CACHE) and not HAS_INTERNET,
    reason="Internet unavailable and test data is not cached locally",
)
def test_resume_equilibration_from_npt_checkpoint(
    vacuum_protocol_settings, benzene_vacuum_system,
    plain_md_checkpoint_path, tmp_path, caplog,
):
    """The legacy 1.5-ps checkpoint resumes NPT and ends at 2 ps.

    Unlike the production test, this intentionally uses the original
    mid-equilibration checkpoint fixture.
    """
    protocol = PlainMDProtocol(settings=vacuum_protocol_settings)
    dag = protocol.create(
        stateA=benzene_vacuum_system,
        stateB=benzene_vacuum_system,
        mapping=None,
    )
    setup_unit, equil_unit, _ = list(dag.protocol_units)
    setup_dir = tmp_path / "setup"
    equil_dir = tmp_path / "equilibration"
    setup_dir.mkdir()
    equil_dir.mkdir()
    setup = setup_unit.run(
        dry=True,
        scratch_basepath=setup_dir,
        shared_basepath=setup_dir,
    )

    checkpoint_name = vacuum_protocol_settings.output_settings.checkpoint_storage_filename
    shutil.copyfile(plain_md_checkpoint_path, equil_dir / checkpoint_name)

    caplog.set_level(logging.INFO)
    result = equil_unit.run(
        equilibration_inputs=EquilibrationInputs(
            states=[None],
            systems=serialization.deserialize(setup["system"]),
            positions=setup["debug"]["positions"],
            topologies=setup["debug"]["topology"],
        ),
        equil_steps_nvt=setup["equil_steps_nvt"] or 0,
        equil_steps_npt=setup["equil_steps_npt"],
        shared_basepath=equil_dir,
        scratch_basepath=equil_dir,
    )
    state = result["equilibrated_states"][0]

    assert "Restarting simulation from checkpoint state" in caplog.text
    assert "Minimizing systems" not in caplog.text
    assert "Running NVT equilibration" not in caplog.text
    assert "Running NPT equilibration for 125 steps" in caplog.text
    assert state["equilibration_end_step"] == 500
    assert state["equilibrated_state"].is_file()
    assert state["reinitialize_velocities_for_production"] is False

    final = openmm.app.Simulation(
        setup["debug"]["topology"],
        serialization.deserialize(setup["system"]),
        openmm.LangevinMiddleIntegrator(
            298.15 * openmm_unit.kelvin,
            1.0 / openmm_unit.picosecond,
            4 * openmm_unit.femtoseconds,
        ),
    )
    final.loadState(str(state["equilibrated_state"]))
    assert final.context.getStepCount() == 500
    assert final.context.getTime().value_in_unit(
        openmm_unit.picoseconds
    ) == pytest.approx(2)
    del final
