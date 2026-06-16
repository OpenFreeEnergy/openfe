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
from openfe.protocols import openmm_afe

from ...conftest import HAS_INTERNET
from .utils import _get_units


@pytest.fixture()
def protocol_settings():
    settings = openmm_afe.AbsoluteSolvationProtocol.default_settings()
    settings.protocol_repeats = 1
    settings.solvent_output_settings.output_indices = "resname UNK"
    settings.solvation_settings.solvent_padding = None
    settings.solvation_settings.number_of_solvent_molecules = 750
    settings.solvation_settings.box_shape = "dodecahedron"
    settings.vacuum_simulation_settings.equilibration_length = 100 * offunit.picosecond
    settings.vacuum_simulation_settings.production_length = 200 * offunit.picosecond
    settings.solvent_simulation_settings.equilibration_length = 100 * offunit.picosecond
    settings.solvent_simulation_settings.production_length = 200 * offunit.picosecond
    settings.vacuum_engine_settings.compute_platform = "CUDA"
    settings.solvent_engine_settings.compute_platform = "CUDA"
    settings.vacuum_simulation_settings.time_per_iteration = 2.5 * offunit.picosecond
    settings.solvent_simulation_settings.time_per_iteration = 2.5 * offunit.picosecond
    settings.vacuum_output_settings.checkpoint_interval = 100 * offunit.picosecond
    settings.solvent_output_settings.checkpoint_interval = 100 * offunit.picosecond
    settings.vacuum_engine_settings.compute_platform = None
    settings.solvent_engine_settings.compute_platform = None
    return settings


def test_verify_execution_environment():
    # Verification should pass
    openmm_afe.AHFESolventSimUnit._verify_execution_environment(
        setup_outputs={
            "gufe_version": gufe.__version__,
            "openfe_version": openfe.__version__,
            "openmm_version": openmm.__version__,
        },
    )


def test_verify_execution_environment_fail():
    # Passing a bad version should fail
    with pytest.raises(ProtocolUnitExecutionError, match="Python environment"):
        openmm_afe.AHFESolventSimUnit._verify_execution_environment(
            setup_outputs={
                "gufe_version": 0.1,
                "openfe_version": openfe.__version__,
                "openmm_version": openmm.__version__,
            },
        )


def test_verify_execution_env_missing_key():
    errmsg = "Missing environment information from setup outputs."
    with pytest.raises(ProtocolUnitExecutionError, match=errmsg):
        openmm_afe.AHFESolventSimUnit._verify_execution_environment(
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
def test_solvent_check_restart(protocol_settings, ahfe_solv_trajectory_path):
    assert openmm_afe.AHFESolventSimUnit._check_restart(
        output_settings=protocol_settings.solvent_output_settings,
        shared_path=ahfe_solv_trajectory_path.parent,
    )

    assert not openmm_afe.AHFESolventSimUnit._check_restart(
        output_settings=protocol_settings.solvent_output_settings,
        shared_path=pathlib.Path("."),
    )


@pytest.mark.skipif(
    not os.path.exists(POOCH_CACHE) and not HAS_INTERNET,
    reason="Internet unavailable and test data is not cached locally",
)
def test_vacuum_check_restart(protocol_settings, ahfe_vac_trajectory_path):
    assert openmm_afe.AHFEVacuumSimUnit._check_restart(
        output_settings=protocol_settings.vacuum_output_settings,
        shared_path=ahfe_vac_trajectory_path.parent,
    )

    assert not openmm_afe.AHFEVacuumSimUnit._check_restart(
        output_settings=protocol_settings.vacuum_output_settings,
        shared_path=pathlib.Path("."),
    )


@pytest.mark.skipif(
    not os.path.exists(POOCH_CACHE) and not HAS_INTERNET,
    reason="Internet unavailable and test data is not cached locally",
)
def test_check_restart_one_file_missing(protocol_settings, ahfe_vac_trajectory_path):
    protocol_settings.vacuum_output_settings.checkpoint_storage_filename = "foo.nc"

    errmsg = "the trajectory file is present but not the checkpoint file."
    with pytest.raises(IOError, match=errmsg):
        openmm_afe.AHFEVacuumSimUnit._check_restart(
            output_settings=protocol_settings.vacuum_output_settings,
            shared_path=ahfe_vac_trajectory_path.parent,
        )


class TestCheckpointResuming:
    @pytest.fixture()
    def protocol_dag(
        self,
        protocol_settings,
        benzene_modifications,
    ):
        stateA = openfe.ChemicalSystem(
            {
                "benzene": benzene_modifications["benzene"],
                "solvent": openfe.SolventComponent(),
            }
        )

        stateB = openfe.ChemicalSystem({"solvent": openfe.SolventComponent()})

        protocol = openmm_afe.AbsoluteSolvationProtocol(settings=protocol_settings)

        # Create DAG from protocol, get the vacuum and solvent units
        # and eventually dry run the first solvent unit
        return protocol.create(
            stateA=stateA,
            stateB=stateB,
            mapping=None,
        )

    @staticmethod
    def _check_sampler(sampler, num_iterations: int):
        # Helper method to do some checks on the sampler
        assert sampler._iteration == num_iterations
        assert sampler.number_of_iterations == 80
        assert sampler.is_completed is (num_iterations == 80)
        assert sampler.n_states == sampler.n_replicas == 14
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
        self, protocol_dag, ahfe_solv_trajectory_path, ahfe_solv_checkpoint_path, tmp_path
    ):
        """
        Attempt to resume a simulation unit with pre-existing checkpoint &
        trajectory files.
        """
        self._copy_simfiles(tmp_path, ahfe_solv_trajectory_path)
        self._copy_simfiles(tmp_path, ahfe_solv_checkpoint_path)

        # 1. Check that the trajectory / checkpoint contain what we expect
        reporter = MultiStateReporter(
            tmp_path / "solvent.nc",
            checkpoint_storage="solvent_checkpoint.nc",
        )
        sampler = ReplicaExchangeSampler.from_storage(reporter)

        self._check_sampler(sampler, num_iterations=40)

        # Deep copy energies & positions for later comparison
        init_energies = copy.deepcopy(reporter.read_energies())[0]
        assert init_energies.shape == (41, 14, 14)
        init_positions = self._get_positions(reporter._storage[0])
        assert len(init_positions) == 2

        reporter.close()
        del sampler

        # 2. get & run the units
        pus = list(protocol_dag.protocol_units)
        setup_unit = _get_units(pus, openmm_afe.AHFESolventSetupUnit)[0]
        sim_unit = _get_units(pus, openmm_afe.AHFESolventSimUnit)[0]
        analysis_unit = _get_units(pus, openmm_afe.AHFESolventAnalysisUnit)[0]

        # Dry run the setup since it'll be easier to use the objects directly
        setup_results = setup_unit.run(
            dry=True,
            scratch_basepath=tmp_path,
            shared_basepath=tmp_path,
        )

        # Now we run the simultion in resume mode
        sim_results = sim_unit.run(
            system=setup_results["alchem_system"],
            positions=setup_results["debug_positions"],
            selection_indices=setup_results["selection_indices"],
            box_vectors=setup_results["box_vectors"],
            alchemical_restraints=False,
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

        self._check_sampler(sampler, num_iterations=80)

        # Check the energies and positions
        energies = reporter.read_energies()[0]
        assert energies.shape == (81, 14, 14)
        assert_allclose(init_energies, energies[:41])

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
        self, protocol_dag, ahfe_solv_trajectory_path, ahfe_solv_checkpoint_path, tmp_path
    ):
        """
        Test that the run unit will fail with a system incompatible
        to the one present in the trajectory/checkpoint files.

        Here we check that we don't have the same particles / mass.
        """
        # copy files
        self._copy_simfiles(tmp_path, ahfe_solv_trajectory_path)
        self._copy_simfiles(tmp_path, ahfe_solv_checkpoint_path)

        pus = list(protocol_dag.protocol_units)
        setup_unit = _get_units(pus, openmm_afe.AHFESolventSetupUnit)[0]
        sim_unit = _get_units(pus, openmm_afe.AHFESolventSimUnit)[0]

        # Dry run the setup since it'll be easier to use the objects directly
        setup_results = setup_unit.run(
            dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path
        )

        # Create a fake system where we will add a particle
        fake_system = copy.deepcopy(setup_results["alchem_system"])
        fake_system.addParticle(42)

        # Fake system should trigger a mismatch
        errmsg = "Stored checkpoint System particles do not"
        with pytest.raises(ValueError, match=errmsg):
            _ = sim_unit.run(
                system=fake_system,
                positions=setup_results["debug_positions"],
                selection_indices=setup_results["selection_indices"],
                box_vectors=setup_results["box_vectors"],
                alchemical_restraints=False,
                scratch_basepath=tmp_path,
                shared_basepath=tmp_path,
            )

    @pytest.mark.slow
    def test_resume_fail_constraints(
        self, protocol_dag, ahfe_solv_trajectory_path, ahfe_solv_checkpoint_path, tmp_path
    ):
        """
        Test that the run unit will fail with a system incompatible
        to the one present in the trajectory/checkpoint files.

        Here we check that we don't have the same constraints.
        """
        # copy files
        self._copy_simfiles(tmp_path, ahfe_solv_trajectory_path)
        self._copy_simfiles(tmp_path, ahfe_solv_checkpoint_path)

        pus = list(protocol_dag.protocol_units)
        setup_unit = _get_units(pus, openmm_afe.AHFESolventSetupUnit)[0]
        sim_unit = _get_units(pus, openmm_afe.AHFESolventSimUnit)[0]

        # Dry run the setup since it'll be easier to use the objects directly
        setup_results = setup_unit.run(
            dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path
        )

        # Create a fake system without constraints
        fake_system = copy.deepcopy(setup_results["alchem_system"])

        for i in reversed(range(fake_system.getNumConstraints())):
            fake_system.removeConstraint(i)

        # Fake system should trigger a mismatch
        errmsg = "Stored checkpoint System constraints do not"
        with pytest.raises(ValueError, match=errmsg):
            _ = sim_unit.run(
                system=fake_system,
                positions=setup_results["debug_positions"],
                selection_indices=setup_results["selection_indices"],
                box_vectors=setup_results["box_vectors"],
                alchemical_restraints=False,
                scratch_basepath=tmp_path,
                shared_basepath=tmp_path,
            )

    @pytest.mark.slow
    def test_resume_fail_forces(
        self, protocol_dag, ahfe_solv_trajectory_path, ahfe_solv_checkpoint_path, tmp_path
    ):
        """
        Test that the run unit will fail with a system incompatible
        to the one present in the trajectory/checkpoint files.

        Here we check we don't have the same forces.
        """
        # copy files
        self._copy_simfiles(tmp_path, ahfe_solv_trajectory_path)
        self._copy_simfiles(tmp_path, ahfe_solv_checkpoint_path)

        pus = list(protocol_dag.protocol_units)
        setup_unit = _get_units(pus, openmm_afe.AHFESolventSetupUnit)[0]
        sim_unit = _get_units(pus, openmm_afe.AHFESolventSimUnit)[0]

        # Dry run the setup since it'll be easier to use the objects directly
        setup_results = setup_unit.run(
            dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path
        )

        # Create a fake system without the last force
        fake_system = copy.deepcopy(setup_results["alchem_system"])
        fake_system.removeForce(fake_system.getNumForces() - 1)

        # Fake system should trigger a mismatch
        errmsg = "Number of forces stored in checkpoint System"
        with pytest.raises(ValueError, match=errmsg):
            _ = sim_unit.run(
                system=fake_system,
                positions=setup_results["debug_positions"],
                selection_indices=setup_results["selection_indices"],
                box_vectors=setup_results["box_vectors"],
                alchemical_restraints=False,
                scratch_basepath=tmp_path,
                shared_basepath=tmp_path,
            )

    @pytest.mark.slow
    def test_resume_differ_barostat(
        self,
        protocol_dag,
        ahfe_solv_trajectory_path,
        ahfe_solv_checkpoint_path,
        tmp_path,
    ):
        """
        Test that the run unit will fail with a system incompatible
        to the one present in the trajectory/checkpoint files.

        Here we check what happens if you have a different barostat
        """
        # copy files
        self._copy_simfiles(tmp_path, ahfe_solv_trajectory_path)
        self._copy_simfiles(tmp_path, ahfe_solv_checkpoint_path)

        pus = list(protocol_dag.protocol_units)
        setup_unit = _get_units(pus, openmm_afe.AHFESolventSetupUnit)[0]
        sim_unit = _get_units(pus, openmm_afe.AHFESolventSimUnit)[0]

        # Dry run the setup since it'll be easier to use the objects directly
        setup_results = setup_unit.run(
            dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path
        )

        # Create a fake system with the fake force type
        fake_system = copy.deepcopy(setup_results["alchem_system"])

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
                system=fake_system,
                positions=setup_results["debug_positions"],
                selection_indices=setup_results["selection_indices"],
                box_vectors=setup_results["box_vectors"],
                alchemical_restraints=False,
                scratch_basepath=tmp_path,
                shared_basepath=tmp_path,
            )

    @pytest.mark.slow
    def test_resume_differ_forces(
        self,
        protocol_dag,
        ahfe_solv_trajectory_path,
        ahfe_solv_checkpoint_path,
        tmp_path,
        caplog,
    ):
        """
        Test that the run unit will fail with a system incompatible
        to the one present in the trajectory/checkpoint files.

        Here we check we have a different force
        """
        # copy files
        self._copy_simfiles(tmp_path, ahfe_solv_trajectory_path)
        self._copy_simfiles(tmp_path, ahfe_solv_checkpoint_path)

        pus = list(protocol_dag.protocol_units)
        setup_unit = _get_units(pus, openmm_afe.AHFESolventSetupUnit)[0]
        sim_unit = _get_units(pus, openmm_afe.AHFESolventSimUnit)[0]

        # Dry run the setup since it'll be easier to use the objects directly
        setup_results = setup_unit.run(
            dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path
        )

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
        new_force.addGlobalParameter("lambda_electrostatics", 1.0)

        fake_system.addForce(new_force)

        # Mismatching force should trigger a warning
        wmsg = "does not exactly match one of the forces in the simulated System"
        caplog.set_level(logging.INFO)

        _ = sim_unit.run(
            system=fake_system,
            positions=setup_results["debug_positions"],
            selection_indices=setup_results["selection_indices"],
            box_vectors=setup_results["box_vectors"],
            alchemical_restraints=False,
            scratch_basepath=tmp_path,
            shared_basepath=tmp_path,
            dry=True,
        )

        assert wmsg in caplog.text

    @pytest.mark.slow
    @pytest.mark.parametrize("bad_file", ["trajectory", "checkpoint"])
    def test_resume_bad_files(
        self, protocol_dag, ahfe_solv_trajectory_path, ahfe_solv_checkpoint_path, bad_file, tmp_path
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
            self._copy_simfiles(tmp_path, ahfe_solv_trajectory_path)

        if bad_file == "checkpoint":
            with open(tmp_path / "solvent_checkpoint.nc", "w") as f:
                f.write("bar")
        else:
            self._copy_simfiles(tmp_path, ahfe_solv_checkpoint_path)

        pus = list(protocol_dag.protocol_units)
        setup_unit = _get_units(pus, openmm_afe.AHFESolventSetupUnit)[0]
        sim_unit = _get_units(pus, openmm_afe.AHFESolventSimUnit)[0]

        # Dry run the setup since it'll be easier to use the objects directly
        setup_results = setup_unit.run(
            dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path
        )

        with pytest.raises(OSError, match="Unknown file format"):
            _ = sim_unit.run(
                system=setup_results["alchem_system"],
                positions=setup_results["debug_positions"],
                selection_indices=setup_results["selection_indices"],
                box_vectors=setup_results["box_vectors"],
                alchemical_restraints=False,
                scratch_basepath=tmp_path,
                shared_basepath=tmp_path,
            )
