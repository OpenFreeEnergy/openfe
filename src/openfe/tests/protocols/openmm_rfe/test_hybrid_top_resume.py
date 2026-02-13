# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import copy
import os
import pathlib
import shutil

import numpy as np
import openmm
import pooch
import pytest
from gufe.protocols import execute_DAG
from numpy.testing import assert_allclose
from openfe_analysis.utils.multistate import _determine_position_indices
from openff.units import unit as offunit
from openff.units.openmm import from_openmm
from openmmtools.multistate import MultiStateReporter

import openfe
from openfe.protocols import openmm_rfe
from openfe.protocols.openmm_rfe._rfe_utils.multistate import HybridRepexSampler
from openfe.protocols.openmm_rfe.hybridtop_units import (
    HybridTopologyMultiStateAnalysisUnit,
    HybridTopologyMultiStateSimulationUnit,
    HybridTopologySetupUnit,
)

from ...conftest import HAS_INTERNET
from .test_hybrid_top_protocol import _get_units

POOCH_CACHE = pooch.os_cache("openfe")
zenodo_resume_data = pooch.create(
    path=POOCH_CACHE,
    base_url="doi:10.5281/zenodo.18331259",
    registry={"multistate_checkpoints.zip": "md5:2cf8aa417ac8311aca1551d4abf3b3ed"},
)


@pytest.fixture(scope="module")
def trajectory_path():
    zenodo_resume_data.fetch("multistate_checkpoints.zip", processor=pooch.Unzip())
    topdir = "multistate_checkpoints.zip.unzip/multistate_checkpoints"
    subdir = "hybrid_top"
    filename = "simulation.nc"
    return pathlib.Path(pooch.os_cache("openfe") / f"{topdir}/{subdir}/{filename}")


@pytest.fixture(scope="module")
def checkpoint_path():
    zenodo_resume_data.fetch("multistate_checkpoints.zip", processor=pooch.Unzip())
    topdir = "multistate_checkpoints.zip.unzip/multistate_checkpoints"
    subdir = "hybrid_top"
    filename = "checkpoint.chk"
    return pathlib.Path(pooch.os_cache("openfe") / f"{topdir}/{subdir}/{filename}")


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
    return settings


@pytest.mark.skipif(
    not os.path.exists(POOCH_CACHE) and not HAS_INTERNET,
    reason="Internet unavailable and test data is not cached locally",
)
def test_check_restart(protocol_settings, trajectory_path):
    assert openmm_rfe.HybridTopologyMultiStateSimulationUnit._check_restart(
        output_settings=protocol_settings.output_settings,
        shared_path=trajectory_path.parent,
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
    def _copy_simfiles(cwd: pathlib.Path, filepath):
        shutil.copyfile(filepath, f"{cwd}/{filepath.name}")

    @pytest.mark.integration
    def test_resume(self, protocol_dag, trajectory_path, checkpoint_path, tmpdir):
        """
        Attempt to resume a simulation unit with pre-existing checkpoint &
        trajectory files.
        """
        # define a temp directory path & copy files
        cwd = pathlib.Path(str(tmpdir))
        self._copy_simfiles(cwd, trajectory_path)
        self._copy_simfiles(cwd, checkpoint_path)

        # 1. Check that the trajectory / checkpoint contain what we expect
        reporter = MultiStateReporter(
            f"{cwd}/simulation.nc",
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
            f"{cwd}/simulation.nc",
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

    def test_resume_fail(self, protocol_dag, trajectory_path, checkpoint_path, tmpdir):
        """
        Test that the run unit will fail with a system incompatible
        to the one present in the trajectory/checkpoint files.
        """
        # define a temp directory path & copy files
        cwd = pathlib.Path(str(tmpdir))
        self._copy_simfiles(cwd, trajectory_path)
        self._copy_simfiles(cwd, checkpoint_path)

        pus = list(protocol_dag.protocol_units)
        setup_unit = _get_units(pus, HybridTopologySetupUnit)[0]
        simulation_unit = _get_units(pus, HybridTopologyMultiStateSimulationUnit)[0]
        analysis_unit = _get_units(pus, HybridTopologyMultiStateAnalysisUnit)[0]

        # Dry run the setup since it'll be easier to use the objects directly
        setup_results = setup_unit.run(dry=True, scratch_basepath=cwd, shared_basepath=cwd)

        # Fake system should trigger a mismatch
        with pytest.raises(ValueError, match="System in checkpoint does not"):
            sim_results = simulation_unit.run(
                system=openmm.System(),
                positions=setup_results["hybrid_positions"],
                selection_indices=setup_results["selection_indices"],
                scratch_basepath=cwd,
                shared_basepath=cwd,
            )

    @pytest.mark.parametrize("bad_file", ["trajectory", "checkpoint"])
    def test_resume_bad_files(
        self, protocol_dag, trajectory_path, checkpoint_path, bad_file, tmpdir
    ):
        """
        Test what happens when you have a bad trajectory and/or checkpoint
        files.
        """
        # define a temp directory path & copy files
        cwd = pathlib.Path(str(tmpdir))

        if bad_file == "trajectory":
            with open(f"{cwd}/simulation.nc", "w") as f:
                f.write("foo")
        else:
            self._copy_simfiles(cwd, trajectory_path)

        if bad_file == "checkpoint":
            with open(f"{cwd}/checkpoint.chk", "w") as f:
                f.write("bar")
        else:
            self._copy_simfiles(cwd, checkpoint_path)

        pus = list(protocol_dag.protocol_units)
        setup_unit = _get_units(pus, HybridTopologySetupUnit)[0]
        simulation_unit = _get_units(pus, HybridTopologyMultiStateSimulationUnit)[0]
        analysis_unit = _get_units(pus, HybridTopologyMultiStateAnalysisUnit)[0]

        # Dry run the setup since it'll be easier to use the objects directly
        setup_results = setup_unit.run(dry=True, scratch_basepath=cwd, shared_basepath=cwd)

        with pytest.raises(OSError, match="Unknown file format"):
            sim_results = simulation_unit.run(
                system=setup_results["hybrid_system"],
                positions=setup_results["hybrid_positions"],
                selection_indices=setup_results["selection_indices"],
                scratch_basepath=cwd,
                shared_basepath=cwd,
            )

    @pytest.mark.parametrize("missing_file", ["trajectory", "checkpoint"])
    def test_missing_file(
        self, protocol_dag, trajectory_path, checkpoint_path, missing_file, tmpdir
    ):
        """
        Test that an error is thrown if either file is missing but the other isn't.
        """
        # define a temp directory path & copy files
        cwd = pathlib.Path(str(tmpdir))

        if missing_file == "trajectory":
            pass
        else:
            self._copy_simfiles(cwd, trajectory_path)

        if missing_file == "checkpoint":
            pass
        else:
            self._copy_simfiles(cwd, checkpoint_path)

        pus = list(protocol_dag.protocol_units)
        setup_unit = _get_units(pus, HybridTopologySetupUnit)[0]
        simulation_unit = _get_units(pus, HybridTopologyMultiStateSimulationUnit)[0]
        analysis_unit = _get_units(pus, HybridTopologyMultiStateAnalysisUnit)[0]

        # Dry run the setup since it'll be easier to use the objects directly
        setup_results = setup_unit.run(dry=True, scratch_basepath=cwd, shared_basepath=cwd)

        errmsg = "One of either the trajectory or checkpoint files are missing"
        with pytest.raises(IOError, match=errmsg):
            sim_results = simulation_unit.run(
                system=setup_results["hybrid_system"],
                positions=setup_results["hybrid_positions"],
                selection_indices=setup_results["selection_indices"],
                scratch_basepath=cwd,
                shared_basepath=cwd,
            )
