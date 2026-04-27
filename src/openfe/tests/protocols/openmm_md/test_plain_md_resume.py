# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import os
import pathlib
import shutil

import gufe
import openmm
import openmm.unit as openmm_unit
import pytest
from gufe import ChemicalSystem, SmallMoleculeComponent
from gufe.protocols.errors import ProtocolUnitExecutionError
from openff.units import unit

import openfe
from openfe.data._registry import POOCH_CACHE
from openfe.protocols.openmm_md.plain_md_methods import (
    PlainMDProtocol,
    PlainMDSetupUnit,
    PlainMDSimulationUnit,
)

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
        # test that we can resume a simulation from a checkpoint
        protocol_units = list(protocol_dag.protocol_units)
        setup_unit: PlainMDSetupUnit = protocol_units[0]
        simulation_unit: PlainMDSimulationUnit = protocol_units[1]
        # copy the files over
        shutil.copyfile(plain_md_checkpoint_path, tmp_path / "checkpoint.xml")
        # dry run the setup unit
        setup_results = setup_unit.run(
            dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path
        )
        # make sure the protocol thinks it can restart
        assert PlainMDSimulationUnit._check_restart(
            output_settings=vacuum_protocol_settings.output_settings,
            shared_path=tmp_path,
        )
        # now run the simulation unit in resume mode this should be 0.5 ps of equilibration and 1 ps of production
        sim_results = simulation_unit.run(
            system=setup_results["debug"]["system"],
            positions=setup_results["debug"]["positions"],
            topology=setup_results["debug"]["topology"],
            equil_steps_nvt=setup_results["equil_steps_nvt"],
            equil_steps_npt=setup_results["equil_steps_npt"],
            prod_steps=setup_results["prod_steps"],
            verbose=True,
            scratch_basepath=tmp_path,
            shared_basepath=tmp_path,
        )
        # make sure it prints that its restarting
        assert "Restarting simulation from checkpoint state" in caplog.text
        # check the number of npt steps to run is correct, this should be 0.5 ps at 4fs timestep
        assert "Running NPT equilibration for 125 steps" in caplog.text
        # make sure the production phase steps are correct, this should be the full 1ps at 4fs timestep
        assert "Running production phase for 250 steps" in caplog.text

        # check the outputs of the simulation unit
        assert sim_results["system_pdb"].exists()
        assert sim_results["nc"].exists()
        assert sim_results["last_checkpoint"]

        # load the final checkpoint and check the simulation time is correct, this should be 3 ps
        # also check the total step count
        simulation = openmm.app.Simulation(
            setup_results["debug"]["topology"],
            setup_results["debug"]["system"],
            openmm.LangevinMiddleIntegrator(
                298.15 * openmm_unit.kelvin,
                1.0 / openmm_unit.picosecond,
                4 * openmm_unit.femtoseconds,
            ),
        )
        simulation.context.setPositions(setup_results["debug"]["positions"])
        simulation.loadState(str(sim_results["last_checkpoint"]))
        total_sim_time = simulation.context.getTime()
        # check the time is 3 ps
        assert total_sim_time.value_in_unit(openmm_unit.picoseconds) == pytest.approx(3)
        # check the step count has been extended
        assert simulation.context.getStepCount() == 750
