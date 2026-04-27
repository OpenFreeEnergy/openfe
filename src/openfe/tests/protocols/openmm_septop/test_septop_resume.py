# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import copy
import logging
import os
import pathlib
import shutil

import gufe
import openmm
import pytest
from gufe.protocols.errors import ProtocolUnitExecutionError
from numpy.testing import assert_allclose
from openfe_analysis.utils.multistate import _determine_position_indices
from openff.units import unit as offunit
from openff.units.openmm import from_openmm
from openmmtools.multistate import MultiStateReporter, ReplicaExchangeSampler

import openfe
from openfe.data._registry import POOCH_CACHE
from openfe.protocols.openmm_septop import (
    SepTopComplexRunUnit,
    SepTopProtocol,
    SepTopSolventAnalysisUnit,
    SepTopSolventRunUnit,
    SepTopSolventSetupUnit,
)

from ...conftest import HAS_INTERNET
from .utils import _get_units


@pytest.fixture()
def protocol_settings():
    settings = SepTopProtocol.default_settings()
    settings.protocol_repeats = 1
    settings.solvent_output_settings.output_indices = "resname UNK"
    settings.complex_solvation_settings.solvent_padding = None
    settings.complex_solvation_settings.number_of_solvent_molecules = 50000
    settings.complex_solvation_settings.box_shape = "dodecahedron"
    settings.solvent_solvation_settings.solvent_padding = None
    settings.solvent_solvation_settings.number_of_solvent_molecules = 1000
    settings.solvent_solvation_settings.box_shape = "dodecahedron"
    settings.complex_simulation_settings.equilibration_length = 50 * offunit.picosecond
    settings.complex_simulation_settings.production_length = 50 * offunit.picosecond
    settings.solvent_simulation_settings.equilibration_length = 50 * offunit.picosecond
    settings.solvent_simulation_settings.production_length = 50 * offunit.picosecond
    settings.complex_simulation_settings.time_per_iteration = 2.5 * offunit.picosecond
    settings.solvent_simulation_settings.time_per_iteration = 2.5 * offunit.picosecond
    settings.complex_output_settings.checkpoint_interval = 25 * offunit.picosecond
    settings.solvent_output_settings.checkpoint_interval = 25 * offunit.picosecond
    settings.complex_output_settings.positions_write_frequency = 25 * offunit.picosecond
    settings.solvent_output_settings.positions_write_frequency = 25 * offunit.picosecond
    settings.engine_settings.compute_platform = None
    return settings


def test_verify_execution_environment():
    # Verification should pass
    SepTopComplexRunUnit._verify_execution_environment(
        setup_outputs={
            "gufe_version": gufe.__version__,
            "openfe_version": openfe.__version__,
            "openmm_version": openmm.__version__,
        },
    )


def test_verify_execution_environment_fail():
    # Passing a bad version should fail
    with pytest.raises(ProtocolUnitExecutionError, match="Python environment"):
        SepTopComplexRunUnit._verify_execution_environment(
            setup_outputs={
                "gufe_version": 0.1,
                "openfe_version": openfe.__version__,
                "openmm_version": openmm.__version__,
            },
        )


def test_verify_execution_env_missing_key():
    errmsg = "Missing environment information from setup outputs."
    with pytest.raises(ProtocolUnitExecutionError, match=errmsg):
        SepTopComplexRunUnit._verify_execution_environment(
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
def test_check_restart(protocol_settings, septop_solv_trajectory_path):
    assert SepTopSolventRunUnit._check_restart(
        output_settings=protocol_settings.solvent_output_settings,
        shared_path=septop_solv_trajectory_path.parent,
    )

    assert not SepTopSolventRunUnit._check_restart(
        output_settings=protocol_settings.solvent_output_settings,
        shared_path=pathlib.Path("."),
    )


@pytest.mark.skipif(
    not os.path.exists(POOCH_CACHE) and not HAS_INTERNET,
    reason="Internet unavailable and test data is not cached locally",
)
def test_check_restart_one_file_missing(protocol_settings, septop_solv_trajectory_path):
    protocol_settings.solvent_output_settings.checkpoint_storage_filename = "foo.nc"

    errmsg = "the trajectory file is present but not the checkpoint file."
    with pytest.raises(IOError, match=errmsg):
        SepTopSolventRunUnit._check_restart(
            output_settings=protocol_settings.solvent_output_settings,
            shared_path=septop_solv_trajectory_path.parent,
        )


class TestCheckpointResuming:
    @pytest.fixture()
    def protocol_dag(
        self,
        protocol_settings,
        benzene_complex_system,
        toluene_complex_system,
    ):
        protocol = SepTopProtocol(settings=protocol_settings)

        return protocol.create(
            stateA=benzene_complex_system,
            stateB=toluene_complex_system,
            mapping=None,
        )

    @pytest.fixture()
    def protocol_units(self, protocol_dag):
        pus = list(protocol_dag.protocol_units)
        setup_unit = _get_units(pus, SepTopSolventSetupUnit)[0]
        sim_unit = _get_units(pus, SepTopSolventRunUnit)[0]
        analysis_unit = _get_units(pus, SepTopSolventAnalysisUnit)[0]
        return setup_unit, sim_unit, analysis_unit

    @pytest.fixture()
    def setup_results(self, protocol_units, tmp_path):
        setup_unit, _, _ = protocol_units

        return setup_unit.run(
            dry=True,
            scratch_basepath=tmp_path,
            shared_basepath=tmp_path,
        )

    @pytest.fixture()
    def pdb_file(self, setup_results):
        return openmm.app.pdbfile.PDBFile(str(setup_results["topology"]))

    @staticmethod
    def _check_sampler(sampler, num_iterations: int):
        # Helper method to do some checks on the sampler
        assert sampler._iteration == num_iterations
        assert sampler.number_of_iterations == 20
        assert sampler.is_completed is (num_iterations == 20)
        assert sampler.n_states == sampler.n_replicas == 27
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
    def _copy_simfiles(cwd: pathlib.Path, filepath):
        shutil.copyfile(filepath, f"{cwd}/{filepath.name}")

    @pytest.mark.integration
    def test_resume(
        self,
        protocol_dag,
        protocol_units,
        setup_results,
        pdb_file,
        septop_solv_trajectory_path,
        septop_solv_checkpoint_path,
        tmp_path,
    ):
        """
        Attempt to resume a simulation unit with pre-existing checkpoint &
        trajectory files.
        """
        self._copy_simfiles(tmp_path, septop_solv_trajectory_path)
        self._copy_simfiles(tmp_path, septop_solv_checkpoint_path)

        # 1. Check that the trajectory / checkpoint contain what we expect
        reporter = MultiStateReporter(
            tmp_path / "solvent.nc",
            checkpoint_storage="solvent_checkpoint.nc",
        )
        sampler = ReplicaExchangeSampler.from_storage(reporter)

        self._check_sampler(sampler, num_iterations=10)

        # Deep copy energies & positions for later comparison
        init_energies = copy.deepcopy(reporter.read_energies())[0]
        assert init_energies.shape == (11, 27, 27)
        init_positions = self._get_positions(reporter._storage[0])
        assert len(init_positions) == 2

        reporter.close()
        del sampler

        # 2. get & run the units
        _, sim_unit, analysis_unit = protocol_units

        # Now we run the simulation in resume mode
        sim_results = sim_unit.run(
            setup_results["alchem_restrained_system"],
            pdb_file,
            setup_results["selection_indices"],
            scratch_basepath=tmp_path,
            shared_basepath=tmp_path,
        )

        # Finally we analyze the results
        _ = analysis_unit.run(
            trajectory=sim_results["trajectory"],
            checkpoint=sim_results["checkpoint"],
            scratch_basepath=tmp_path,
            shared_basepath=tmp_path,
        )

        # Analyze the trajectory / checkpoint again
        reporter = MultiStateReporter(
            tmp_path / "solvent.nc",
            checkpoint_storage="solvent_checkpoint.nc",
        )

        sampler = ReplicaExchangeSampler.from_storage(reporter)

        self._check_sampler(sampler, num_iterations=20)

        # Check the energies and positions
        energies = reporter.read_energies()[0]
        assert energies.shape == (21, 27, 27)
        assert_allclose(init_energies, energies[:11])

        positions = self._get_positions(reporter._storage[0])
        assert len(positions) == 3
        for i in range(2):
            assert_allclose(positions[i], init_positions[i])

        reporter.close()
        del sampler

        # Check the free energy plots are there
        mbar_overlap_file = tmp_path / "mbar_overlap_matrix.png"
        assert (mbar_overlap_file).exists()

    @pytest.mark.slow
    def test_resume_fail_particles(
        self,
        protocol_dag,
        protocol_units,
        setup_results,
        pdb_file,
        septop_solv_trajectory_path,
        septop_solv_checkpoint_path,
        tmp_path,
    ):
        """
        Test that the run unit will fail with a system incompatible
        to the one present in the trajectory/checkpoint files.

        Here we check that we don't have the same particles / mass.
        """
        # copy files
        self._copy_simfiles(tmp_path, septop_solv_trajectory_path)
        self._copy_simfiles(tmp_path, septop_solv_checkpoint_path)

        _, sim_unit, _ = protocol_units

        # Create a fake system where we will add a particle
        fake_system = copy.deepcopy(setup_results["alchem_restrained_system"])
        fake_system.addParticle(42)

        # Fake system should trigger a mismatch
        errmsg = "Stored checkpoint System particles do not"
        with pytest.raises(ValueError, match=errmsg):
            _ = sim_unit.run(
                fake_system,
                pdb_file,
                setup_results["selection_indices"],
                scratch_basepath=tmp_path,
                shared_basepath=tmp_path,
            )

    @pytest.mark.slow
    def test_resume_fail_constraints(
        self,
        protocol_dag,
        protocol_units,
        setup_results,
        pdb_file,
        septop_solv_trajectory_path,
        septop_solv_checkpoint_path,
        tmp_path,
    ):
        """
        Test that the run unit will fail with a system incompatible
        to the one present in the trajectory/checkpoint files.

        Here we check that we don't have the same constraints.
        """
        # copy files
        self._copy_simfiles(tmp_path, septop_solv_trajectory_path)
        self._copy_simfiles(tmp_path, septop_solv_checkpoint_path)

        _, sim_unit, _ = protocol_units

        # Create a fake system without constraints
        fake_system = copy.deepcopy(setup_results["alchem_restrained_system"])

        for i in reversed(range(fake_system.getNumConstraints())):
            fake_system.removeConstraint(i)

        # Fake system should trigger a mismatch
        errmsg = "Stored checkpoint System constraints do not"
        with pytest.raises(ValueError, match=errmsg):
            _ = sim_unit.run(
                fake_system,
                pdb_file,
                setup_results["selection_indices"],
                scratch_basepath=tmp_path,
                shared_basepath=tmp_path,
            )

    @pytest.mark.slow
    def test_resume_fail_forces(
        self,
        protocol_dag,
        protocol_units,
        setup_results,
        pdb_file,
        septop_solv_trajectory_path,
        septop_solv_checkpoint_path,
        tmp_path,
    ):
        """
        Test that the run unit will fail with a system incompatible
        to the one present in the trajectory/checkpoint files.

        Here we check we don't have the same forces.
        """
        # copy files
        self._copy_simfiles(tmp_path, septop_solv_trajectory_path)
        self._copy_simfiles(tmp_path, septop_solv_checkpoint_path)

        _, sim_unit, _ = protocol_units

        # Create a fake system without the last force
        fake_system = copy.deepcopy(setup_results["alchem_restrained_system"])
        fake_system.removeForce(fake_system.getNumForces() - 1)

        # Fake system should trigger a mismatch
        errmsg = "Number of forces stored in checkpoint System"
        with pytest.raises(ValueError, match=errmsg):
            _ = sim_unit.run(
                fake_system,
                pdb_file,
                setup_results["selection_indices"],
                scratch_basepath=tmp_path,
                shared_basepath=tmp_path,
            )

    @pytest.mark.slow
    def test_resume_differ_barostat(
        self,
        protocol_dag,
        protocol_units,
        setup_results,
        pdb_file,
        septop_solv_trajectory_path,
        septop_solv_checkpoint_path,
        tmp_path,
    ):
        """
        Test that the run unit will fail with a system incompatible
        to the one present in the trajectory/checkpoint files.

        Here we check what happens if you have a different barostat
        """
        # copy files
        self._copy_simfiles(tmp_path, septop_solv_trajectory_path)
        self._copy_simfiles(tmp_path, septop_solv_checkpoint_path)

        _, sim_unit, _ = protocol_units

        # Create a fake system with the fake force type
        fake_system = copy.deepcopy(setup_results["alchem_restrained_system"])

        # Loop through forces and remove the force matching force type
        for i, f in enumerate(fake_system.getForces()):
            if isinstance(f, openmm.MonteCarloBarostat):
                findex = i

        fake_system.removeForce(findex)

        # Now add the new barostat
        new_force = openmm.MonteCarloBarostat(
            1 * openmm.unit.atmosphere, 300 * openmm.unit.kelvin, 100
        )
        fake_system.addForce(new_force)

        # Fake system should trigger a mismatch
        errmsg = "stored checkpoint System does not match the same force"
        with pytest.raises(ValueError, match=errmsg):
            _ = sim_unit.run(
                fake_system,
                pdb_file,
                setup_results["selection_indices"],
                scratch_basepath=tmp_path,
                shared_basepath=tmp_path,
            )

    @pytest.mark.slow
    def test_resume_differ_forces(
        self,
        protocol_dag,
        protocol_units,
        setup_results,
        pdb_file,
        septop_solv_trajectory_path,
        septop_solv_checkpoint_path,
        tmp_path,
        caplog,
    ):
        """
        Test that the run unit will fail with a system incompatible
        to the one present in the trajectory/checkpoint files.

        Here we check we have a different force
        """
        # copy files
        self._copy_simfiles(tmp_path, septop_solv_trajectory_path)
        self._copy_simfiles(tmp_path, septop_solv_checkpoint_path)

        _, sim_unit, _ = protocol_units

        # Create a fake system with the fake force type
        fake_system = copy.deepcopy(setup_results["alchem_system"])

        # Loop through forces and remove the force matching force type
        for i, f in enumerate(fake_system.getForces()):
            if isinstance(f, openmm.NonbondedForce):
                findex = i

        fake_system.removeForce(findex)

        # Now add a fake force
        new_force = openmm.NonbondedForce()
        new_force.setNonbondedMethod(openmm.NonbondedForce.PME)
        new_force.addGlobalParameter("lambda_electrostatics_A", 1.0)
        new_force.addGlobalParameter("lambda_electrostatics_B", 0.0)

        fake_system.addForce(new_force)

        # Mismatching force should trigger a warning
        wmsg = "does not exactly match one of the forces in the simulated System"
        caplog.set_level(logging.INFO)

        _ = sim_unit.run(
            fake_system,
            pdb_file,
            setup_results["selection_indices"],
            scratch_basepath=tmp_path,
            shared_basepath=tmp_path,
        )

        assert wmsg in caplog.text

    @pytest.mark.slow
    @pytest.mark.parametrize("bad_file", ["trajectory", "checkpoint"])
    def test_resume_bad_files(
        self,
        protocol_dag,
        protocol_units,
        setup_results,
        pdb_file,
        septop_solv_trajectory_path,
        septop_solv_checkpoint_path,
        bad_file,
        tmp_path,
    ):
        """
        Test what happens when you have a bad trajectory and/or checkpoint
        files.
        """
        # copy files

        if bad_file == "trajectory":
            with open(tmp_path / "solvent.nc", "w") as f:
                f.write("foo")
        else:
            self._copy_simfiles(tmp_path, septop_solv_trajectory_path)

        if bad_file == "checkpoint":
            with open(tmp_path / "solvent_checkpoint.nc", "w") as f:
                f.write("bar")
        else:
            self._copy_simfiles(tmp_path, septop_solv_checkpoint_path)

        _, sim_unit, _ = protocol_units

        with pytest.raises(OSError, match="Unknown file format"):
            _ = sim_unit.run(
                setup_results["alchem_restrained_system"],
                pdb_file,
                setup_results["selection_indices"],
                scratch_basepath=tmp_path,
                shared_basepath=tmp_path,
            )
