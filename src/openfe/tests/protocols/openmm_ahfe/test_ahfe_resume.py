# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import copy
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
        self, protocol_dag, ahfe_solv_trajectory_path, ahfe_solv_checkpoint_path, tmpdir
    ):
        """
        Attempt to resume a simulation unit with pre-existing checkpoint &
        trajectory files.
        """
        cwd = pathlib.Path(str(tmpdir))
        self._copy_simfiles(cwd, ahfe_solv_trajectory_path)
        self._copy_simfiles(cwd, ahfe_solv_checkpoint_path)

        # 1. Check that the trajectory / checkpoint contain what we expect
        reporter = MultiStateReporter(
            f"{cwd}/solvent.nc",
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
            scratch_basepath=cwd,
            shared_basepath=cwd,
        )

        # Now we run the simultion in resume mode
        sim_results = sim_unit.run(
            system=setup_results["alchem_system"],
            positions=setup_results["debug_positions"],
            selection_indices=setup_results["selection_indices"],
            box_vectors=setup_results["box_vectors"],
            alchemical_restraints=False,
            scratch_basepath=cwd,
            shared_basepath=cwd,
        )

        # Finally we analyze the results
        analysis_results = analysis_unit.run(
            trajectory=sim_results["trajectory"],
            checkpoint=sim_results["checkpoint"],
            scratch_basepath=cwd,
            shared_basepath=cwd,
        )

        # Analyze the trajectory / checkpoint again
        reporter = MultiStateReporter(
            f"{cwd}/solvent.nc",
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
        mbar_overlap_file = cwd / "mbar_overlap_matrix.png"
        assert (mbar_overlap_file).exists()
