# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import copy
import logging
import os
import pathlib
import shutil

import gufe
import numpy as np
import openmm
import pooch
import pytest
from gufe.protocols import execute_DAG
from gufe.protocols.errors import ProtocolUnitExecutionError
from numpy.testing import assert_allclose
from openfe_analysis.utils.multistate import _determine_position_indices
from openff.units import unit as offunit
from openff.units.openmm import from_openmm
from openmmtools.multistate import MultiStateReporter

import openfe
from openfe.data._registry import POOCH_CACHE
from openfe.protocols import openmm_rfe
from openfe.protocols.openmm_rfe._rfe_utils.multistate import HybridRepexSampler
from openfe.protocols.openmm_rfe.hybridtop_units import (
    HybridTopologyMultiStateAnalysisUnit,
    HybridTopologyMultiStateSimulationUnit,
    HybridTopologySetupUnit,
)

from ...conftest import HAS_INTERNET
from .test_hybrid_top_protocol import _get_units


@pytest.fixture()
def protocol_settings():
    settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
    settings.solvation_settings.solvent_padding = None
    settings.solvation_settings.number_of_solvent_molecules = 750
    settings.solvation_settings.box_shape = "dodecahedron"
    settings.protocol_repeats = 1
    settings.simulation_settings.equilibration_length = 100 * offunit.picosecond
    settings.simulation_settings.production_length = 200 * offunit.picosecond
    settings.simulation_settings.time_per_iteration = 2.5 * offunit.picosecond
    settings.output_settings.checkpoint_interval = 100 * offunit.picosecond
    settings.engine_settings.compute_platform = None
    return settings


def test_verify_execution_environment():
    # Verification should pass
    openmm_rfe.HybridTopologyMultiStateSimulationUnit._verify_execution_environment(
        setup_outputs={
            "gufe_version": gufe.__version__,
            "openfe_version": openfe.__version__,
            "openmm_version": openmm.__version__,
        },
    )


def test_verify_execution_environment_fail():
    # Passing a bad version should fail
    with pytest.raises(ProtocolUnitExecutionError, match="Python environment"):
        openmm_rfe.HybridTopologyMultiStateSimulationUnit._verify_execution_environment(
            setup_outputs={
                "gufe_version": 0.1,
                "openfe_version": openfe.__version__,
                "openmm_version": openmm.__version__,
            },
        )


def test_verify_execution_env_missing_key():
    errmsg = "Missing environment information from setup outputs."
    with pytest.raises(ProtocolUnitExecutionError, match=errmsg):
        openmm_rfe.HybridTopologyMultiStateSimulationUnit._verify_execution_environment(
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
def test_check_restart(protocol_settings, htop_trajectory_path):
    assert openmm_rfe.HybridTopologyMultiStateSimulationUnit._check_restart(
        output_settings=protocol_settings.output_settings,
        shared_path=htop_trajectory_path.parent,
    )

    assert not openmm_rfe.HybridTopologyMultiStateSimulationUnit._check_restart(
        output_settings=protocol_settings.output_settings,
        shared_path=pathlib.Path("."),
    )


@pytest.mark.skipif(
    not os.path.exists(POOCH_CACHE) and not HAS_INTERNET,
    reason="Internet unavailable and test data is not cached locally",
)
class TestCheckpointResuming:
    @pytest.fixture()
    def protocol_dag(
        self, protocol_settings, benzene_system, toluene_system, benzene_to_toluene_mapping
    ):
        protocol = openmm_rfe.RelativeHybridTopologyProtocol(settings=protocol_settings)

        return protocol.create(
            stateA=benzene_system, stateB=toluene_system, mapping=benzene_to_toluene_mapping
        )

    @staticmethod
    def _check_sampler(sampler, num_iterations: int):
        # Helper method to do some checks on the sampler
        assert sampler._iteration == num_iterations
        assert sampler.number_of_iterations == 80
        assert sampler.is_completed is (num_iterations == 80)
        assert sampler.n_states == sampler.n_replicas == 11
        assert sampler.is_periodic
        assert sampler.mcmc_moves[0].n_steps == 625
        assert from_openmm(sampler.mcmc_moves[0].timestep) == 4 * offunit.fs

    @staticmethod
    def _get_positions(dataset):
        frame_list = _determine_position_indices(dataset)
        positions = []
        for frame in frame_list:
            positions.append(copy.deepcopy(dataset.variables["positions"][frame].data))
        return positions

    @staticmethod
    def _copy_simfiles(basedir: pathlib.Path, filepath):
        shutil.copyfile(filepath, f"{basedir}/{filepath.name}")

    @pytest.mark.integration
    def test_resume(self, protocol_dag, htop_trajectory_path, htop_checkpoint_path, tmp_path):
        """
        Attempt to resume a simulation unit with pre-existing checkpoint &
        trajectory files.
        """
        # copy files
        self._copy_simfiles(tmp_path, htop_trajectory_path)
        self._copy_simfiles(tmp_path, htop_checkpoint_path)

        # 1. Check that the trajectory / checkpoint contain what we expect
        reporter = MultiStateReporter(
            tmp_path / "simulation.nc",
            checkpoint_storage="checkpoint.chk",
        )
        sampler = HybridRepexSampler.from_storage(reporter)

        self._check_sampler(sampler, num_iterations=40)
        # Deep copy energies & positions for later tests
        init_energies = copy.deepcopy(reporter.read_energies())[0]
        assert init_energies.shape == (41, 11, 11)
        init_positions = self._get_positions(reporter._storage[0])
        assert len(init_positions) == 2

        reporter.close()
        del sampler

        # 2. get & run the units
        pus = list(protocol_dag.protocol_units)
        setup_unit = _get_units(pus, HybridTopologySetupUnit)[0]
        simulation_unit = _get_units(pus, HybridTopologyMultiStateSimulationUnit)[0]
        analysis_unit = _get_units(pus, HybridTopologyMultiStateAnalysisUnit)[0]

        # Dry run the setup since it'll be easier to use the objects directly
        setup_results = setup_unit.run(
            dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path
        )

        # Now we run the simulation in resume mode
        sim_results = simulation_unit.run(
            system=setup_results["hybrid_system"],
            positions=setup_results["hybrid_positions"],
            selection_indices=setup_results["selection_indices"],
            scratch_basepath=tmp_path,
            shared_basepath=tmp_path,
        )

        # Finally we analyze the results
        _ = analysis_unit.run(
            pdb_file=setup_results["pdb_structure"],
            trajectory=sim_results["nc"],
            checkpoint=sim_results["checkpoint"],
            scratch_basepath=tmp_path,
            shared_basepath=tmp_path,
        )

        # 3. Analyze the trajectory/checkpoint again
        reporter = MultiStateReporter(
            tmp_path / "simulation.nc",
            checkpoint_storage="checkpoint.chk",
        )
        sampler = HybridRepexSampler.from_storage(reporter)

        self._check_sampler(sampler, num_iterations=80)

        # Check the energies and positions
        energies = reporter.read_energies()[0]
        assert energies.shape == (81, 11, 11)
        assert_allclose(init_energies, energies[:41])

        positions = self._get_positions(reporter._storage[0])
        assert len(positions) == 3
        for i in range(2):
            assert_allclose(positions[i], init_positions[i])

        reporter.close()
        del sampler

        # Check the openfe-analysis outputs are there
        structural_analysis_file = tmp_path / "structural_analysis.npz"
        assert (structural_analysis_file).exists()

    @pytest.mark.slow
    def test_resume_fail_particles(
        self, protocol_dag, htop_trajectory_path, htop_checkpoint_path, tmp_path
    ):
        """
        Test that the run unit will fail with a system incompatible
        to the one present in the trajectory/checkpoint files.

        Here we check that we don't have the same particles / mass.
        """
        # copy files
        self._copy_simfiles(tmp_path, htop_trajectory_path)
        self._copy_simfiles(tmp_path, htop_checkpoint_path)

        pus = list(protocol_dag.protocol_units)
        setup_unit = _get_units(pus, HybridTopologySetupUnit)[0]
        simulation_unit = _get_units(pus, HybridTopologyMultiStateSimulationUnit)[0]

        # Dry run the setup since it'll be easier to use the objects directly
        setup_results = setup_unit.run(
            dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path
        )

        # Fake system should trigger a mismatch
        errmsg = "Stored checkpoint System particles do not"
        with pytest.raises(ValueError, match=errmsg):
            _ = simulation_unit.run(
                system=openmm.System(),
                positions=setup_results["hybrid_positions"],
                selection_indices=setup_results["selection_indices"],
                scratch_basepath=tmp_path,
                shared_basepath=tmp_path,
            )

    @pytest.mark.slow
    def test_resume_fail_constraints(
        self, protocol_dag, htop_trajectory_path, htop_checkpoint_path, tmp_path
    ):
        """
        Test that the run unit will fail with a system incompatible
        to the one present in the trajectory/checkpoint files.

        Here we check that we don't have the same constraints.
        """
        # copy files
        self._copy_simfiles(tmp_path, htop_trajectory_path)
        self._copy_simfiles(tmp_path, htop_checkpoint_path)

        pus = list(protocol_dag.protocol_units)
        setup_unit = _get_units(pus, HybridTopologySetupUnit)[0]
        simulation_unit = _get_units(pus, HybridTopologyMultiStateSimulationUnit)[0]

        # Dry run the setup since it'll be easier to use the objects directly
        setup_results = setup_unit.run(
            dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path
        )

        # Create a fake system without constraints
        fake_system = copy.deepcopy(setup_results["hybrid_system"])

        for i in reversed(range(fake_system.getNumConstraints())):
            fake_system.removeConstraint(i)

        # Fake system should trigger a mismatch
        errmsg = "Stored checkpoint System constraints do not"
        with pytest.raises(ValueError, match=errmsg):
            _ = simulation_unit.run(
                system=fake_system,
                positions=setup_results["hybrid_positions"],
                selection_indices=setup_results["selection_indices"],
                scratch_basepath=tmp_path,
                shared_basepath=tmp_path,
            )

    @pytest.mark.slow
    def test_resume_fail_forces(
        self, protocol_dag, htop_trajectory_path, htop_checkpoint_path, tmp_path
    ):
        """
        Test that the run unit will fail with a system incompatible
        to the one present in the trajectory/checkpoint files.

        Here we check we don't have the same forces.
        """
        # copy files
        self._copy_simfiles(tmp_path, htop_trajectory_path)
        self._copy_simfiles(tmp_path, htop_checkpoint_path)

        pus = list(protocol_dag.protocol_units)
        setup_unit = _get_units(pus, HybridTopologySetupUnit)[0]
        simulation_unit = _get_units(pus, HybridTopologyMultiStateSimulationUnit)[0]

        # Dry run the setup since it'll be easier to use the objects directly
        setup_results = setup_unit.run(
            dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path
        )

        # Create a fake system without the last force
        fake_system = copy.deepcopy(setup_results["hybrid_system"])
        fake_system.removeForce(fake_system.getNumForces() - 1)

        # Fake system should trigger a mismatch
        errmsg = "Number of forces stored in checkpoint System"
        with pytest.raises(ValueError, match=errmsg):
            _ = simulation_unit.run(
                system=fake_system,
                positions=setup_results["hybrid_positions"],
                selection_indices=setup_results["selection_indices"],
                scratch_basepath=tmp_path,
                shared_basepath=tmp_path,
            )

    @pytest.mark.slow
    def test_resume_differ_barostat(
        self, protocol_dag, htop_trajectory_path, htop_checkpoint_path, tmp_path
    ):
        """
        Test that the run unit will fail if the barostat differs.
        """
        # copy files
        self._copy_simfiles(tmp_path, htop_trajectory_path)
        self._copy_simfiles(tmp_path, htop_checkpoint_path)

        pus = list(protocol_dag.protocol_units)
        setup_unit = _get_units(pus, HybridTopologySetupUnit)[0]
        simulation_unit = _get_units(pus, HybridTopologyMultiStateSimulationUnit)[0]

        # Dry run the setup since it'll be easier to use the objects directly
        setup_results = setup_unit.run(
            dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path
        )

        # Create a fake system with the fake forcetype
        fake_system = copy.deepcopy(setup_results["hybrid_system"])

        # Loop through forces and remove the force matching force type
        for i, f in enumerate(fake_system.getForces()):
            if isinstance(f, openmm.MonteCarloBarostat):
                findex = i

        fake_system.removeForce(findex)
        new_force = openmm.MonteCarloBarostat(
            1 * openmm.unit.atmosphere, 300 * openmm.unit.kelvin, 100
        )
        fake_system.addForce(new_force)

        # Fake system should trigger a mismatch
        errmsg = "stored checkpoint System does not match the same force"
        with pytest.raises(ValueError, match=errmsg):
            _ = simulation_unit.run(
                system=fake_system,
                positions=setup_results["hybrid_positions"],
                selection_indices=setup_results["selection_indices"],
                scratch_basepath=tmp_path,
                shared_basepath=tmp_path,
            )

    @pytest.mark.slow
    def test_resume_differ_forces(
        self, protocol_dag, htop_trajectory_path, htop_checkpoint_path, tmp_path, caplog
    ):
        """
        Test that the run unit will warn if forces don't match
        to the one present in the trajectory/checkpoint files.
        """
        # copy files
        self._copy_simfiles(tmp_path, htop_trajectory_path)
        self._copy_simfiles(tmp_path, htop_checkpoint_path)

        pus = list(protocol_dag.protocol_units)
        setup_unit = _get_units(pus, HybridTopologySetupUnit)[0]
        simulation_unit = _get_units(pus, HybridTopologyMultiStateSimulationUnit)[0]

        # Dry run the setup since it'll be easier to use the objects directly
        setup_results = setup_unit.run(
            dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path
        )

        # Create a fake system with the fake forcetype
        fake_system = copy.deepcopy(setup_results["hybrid_system"])

        # Loop through forces and remove the force matching force type
        for i, f in enumerate(fake_system.getForces()):
            if isinstance(f, openmm.NonbondedForce):
                findex = i

        fake_system.removeForce(findex)
        fake_system.addForce(openmm.NonbondedForce())

        # Mismatching force should trigger a warning
        wmsg = "does not exactly match one of the forces in the simulated System"
        caplog.set_level(logging.INFO)

        _ = simulation_unit.run(
            system=fake_system,
            positions=setup_results["hybrid_positions"],
            selection_indices=setup_results["selection_indices"],
            scratch_basepath=tmp_path,
            shared_basepath=tmp_path,
            dry=True,
        )

        assert wmsg in caplog.text

    @pytest.mark.slow
    @pytest.mark.parametrize("bad_file", ["trajectory", "checkpoint"])
    def test_resume_bad_files(
        self, protocol_dag, htop_trajectory_path, htop_checkpoint_path, bad_file, tmp_path
    ):
        """
        Test what happens when you have a bad trajectory and/or checkpoint
        files.
        """
        # copy files
        if bad_file == "trajectory":
            with open(tmp_path / "simulation.nc", "w") as f:
                f.write("foo")
        else:
            self._copy_simfiles(tmp_path, htop_trajectory_path)

        if bad_file == "checkpoint":
            with open(tmp_path / "checkpoint.chk", "w") as f:
                f.write("bar")
        else:
            self._copy_simfiles(tmp_path, htop_checkpoint_path)

        pus = list(protocol_dag.protocol_units)
        setup_unit = _get_units(pus, HybridTopologySetupUnit)[0]
        simulation_unit = _get_units(pus, HybridTopologyMultiStateSimulationUnit)[0]

        # Dry run the setup since it'll be easier to use the objects directly
        setup_results = setup_unit.run(
            dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path
        )

        with pytest.raises(OSError, match="Unknown file format"):
            _ = simulation_unit.run(
                system=setup_results["hybrid_system"],
                positions=setup_results["hybrid_positions"],
                selection_indices=setup_results["selection_indices"],
                scratch_basepath=tmp_path,
                shared_basepath=tmp_path,
            )

    @pytest.mark.slow
    @pytest.mark.parametrize("missing_file", ["trajectory", "checkpoint"])
    def test_missing_file(
        self, protocol_dag, htop_trajectory_path, htop_checkpoint_path, missing_file, tmp_path
    ):
        """
        Test that an error is thrown if either file is missing but the other isn't.
        """
        # copy files
        if missing_file == "trajectory":
            pass
        else:
            self._copy_simfiles(tmp_path, htop_trajectory_path)

        if missing_file == "checkpoint":
            pass
        else:
            self._copy_simfiles(tmp_path, htop_checkpoint_path)

        pus = list(protocol_dag.protocol_units)
        setup_unit = _get_units(pus, HybridTopologySetupUnit)[0]
        simulation_unit = _get_units(pus, HybridTopologyMultiStateSimulationUnit)[0]

        # Dry run the setup since it'll be easier to use the objects directly
        setup_results = setup_unit.run(
            dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path
        )

        errmsg = f"file is present but not the {missing_file} file."
        with pytest.raises(IOError, match=errmsg):
            _ = simulation_unit.run(
                system=setup_results["hybrid_system"],
                positions=setup_results["hybrid_positions"],
                selection_indices=setup_results["selection_indices"],
                scratch_basepath=tmp_path,
                shared_basepath=tmp_path,
            )
