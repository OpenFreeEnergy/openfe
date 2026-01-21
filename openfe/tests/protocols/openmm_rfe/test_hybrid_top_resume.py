# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import os
import pathlib
import shutil
import copy

import numpy as np
import pooch
import pytest
from gufe.protocols import execute_DAG
from numpy.testing import assert_allclose
from openff.units import unit as offunit
from openff.units.openmm import from_openmm
from openfe_analysis.utils.multistate import _determine_position_indices

import openfe
from openfe.protocols import openmm_rfe
from openfe.protocols.openmm_rfe.hybridtop_units import (
    HybridTopologyMultiStateAnalysisUnit,
    HybridTopologyMultiStateSimulationUnit,
    HybridTopologySetupUnit,
)
from openmmtools.multistate import MultiStateReporter
from openfe.protocols.openmm_rfe._rfe_utils.multistate import HybridRepexSampler

from .test_hybrid_top_protocol import _get_units
from ...conftest import HAS_INTERNET


POOCH_CACHE = pooch.os_cache("openfe")
zenodo_resume_data = pooch.create(
    path=POOCH_CACHE,
    base_url="doi:10.5281/zenodo.18331259",
    registry={
        "multistate_checkpoints.zip": "md5:2cf8aa417ac8311aca1551d4abf3b3ed"
    },
)


@pytest.fixture(scope='module')
def trajectory_path():
    zenodo_resume_data.fetch("multistate_checkpoints.zip", processor=pooch.Unzip())
    topdir = "multistate_checkpoints.zip.unzip/multistate_checkpoints"
    subdir = "hybrid_top"
    filename = "simulation.nc"
    return pathlib.Path(pooch.os_cache("openfe") / f"{topdir}/{subdir}/{filename}")


@pytest.fixture(scope='module')
def checkpoint_path():
    zenodo_resume_data.fetch("multistate_checkpoints.zip", processor=pooch.Unzip())
    topdir = "multistate_checkpoints.zip.unzip/multistate_checkpoints"
    subdir = "hybrid_top"
    filename = "checkpoint.chk"
    return pathlib.Path(pooch.os_cache("openfe") / f"{topdir}/{subdir}/{filename}")


@pytest.mark.skipif(
    not os.path.exists(POOCH_CACHE) and not HAS_INTERNET,
    reason="Internet unavailable and test data is not cached locally",
)
class TestCheckpointResuming:
    @pytest.fixture(scope='class')
    def protocol_dag(self, benzene_system, toluene_system, benzene_to_toluene_mapping):
        settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
        settings.solvation_settings.solvent_padding = None
        settings.solvation_settings.number_of_solvent_molecules = 750
        settings.solvation_settings.box_shape = "dodecahedron"
        settings.protocol_repeats = 1
        settings.simulation_settings.equilibration_length = 100 * offunit.picosecond
        settings.simulation_settings.production_length = 200 * offunit.picosecond
        settings.simulation_settings.time_per_iteration = 2.5 * offunit.picosecond
        settings.output_settings.checkpoint_interval = 100 * offunit.picosecond
        protocol = openmm_rfe.RelativeHybridTopologyProtocol(settings=settings)

        return protocol.create(
            stateA=benzene_system,
            stateB=toluene_system,
            mapping=benzene_to_toluene_mapping
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
            positions.append(
                copy.deepcopy(dataset.variables["positions"][frame].data)
            )
        return positions

    def test_resume(self, protocol_dag, trajectory_path, checkpoint_path, tmpdir):
        """
        Attempt to resume a simulation unit with pre-existing checkpoint &
        trajectory files.
        """
        # define a temp directory path
        cwd = pathlib.Path(str(tmpdir))

        shutil.copyfile(trajectory_path, f"{cwd}/{trajectory_path.name}")
        shutil.copyfile(checkpoint_path, f"{cwd}/{checkpoint_path.name}")

        # 1. Check that the trajectory / checkpoint contain what we expect
        reporter = MultiStateReporter(
            f"{cwd}/{trajectory_path.name}",
            checkpoint_storage=checkpoint_path.name,
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
        setup_results = setup_unit.run(dry=True, scratch_basepath=cwd, shared_basepath=cwd)
    
        # Now we run the simulation in resume mode
        sim_results = simulation_unit.run(
            system=setup_results["hybrid_system"],
            positions=setup_results["hybrid_positions"],
            selection_indices=setup_results["selection_indices"],
            scratch_basepath=cwd,
            shared_basepath=cwd,
        )

        # TODO: can't do this right now: openfe-analysis isn't closing
        # netcdf files properly, so we can't do any follow-up operations
        # Once openfe-analysis is released, add tests for this.
        # # Finally we analyze the results
        # analysis_results = analysis_unit.run(
        #     pdb_file=setup_results["pdb_structure"],
        #     trajectory=sim_results["nc"],
        #     checkpoint=sim_results["checkpoint"],
        #     scratch_basepath=cwd,
        #     shared_basepath=cwd,
        # )

        # 3. Analyze the trajectory/checkpoint again
        reporter = MultiStateReporter(
            f"{cwd}/{trajectory_path.name}",
            checkpoint_storage=checkpoint_path.name,
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
