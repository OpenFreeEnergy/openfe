# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import pathlib
import pooch

import pytest
from openff.units import unit as offunit

import openfe
from openfe.protocols import openmm_afe

from ...conftest import HAS_INTERNET
from utils import _get_units


POOCH_CACHE = pooch.os_cache("openfe")
zenodo_resume_data = pooch.create(
    path=POOCH_CACHE,
    base_url="doi:10.5281/zenodo.18331259",
    registry={"multistate_checkpoints.zip": "md5:2cf8aa417ac8311aca1551d4abf3b3ed"},
)

@pytest.fixture(scope="module")
def vac_trajectory_path():
    zenodo_resume_data.fetch("multistate_checkpoints.zip", processor=pooch.Unzip())
    topdir = "multistate_checkpoints.zip.unzip/multistate_checkpoints"
    subdir = "ahfes"
    filename = "vacuum.nc"
    return pathlib.Path(pooch.os_cache("openfe") / f"{topdir}/{subdir}/{filename}")


@pytest.fixture(scope="module")
def vac_checkpoint_path():
    zenodo_resume_data.fetch("multistate_checkpoints.zip", processor=pooch.Unzip())
    topdir = "multistate_checkpoints.zip.unzip/multistate_checkpoints"
    subdir = "ahfes"
    filename = "vacuum_checkpoint.chk"
    return pathlib.Path(pooch.os_cache("openfe") / f"{topdir}/{subdir}/{filename}")


@pytest.fixture(scope="module")
def sol_trajectory_path():
    zenodo_resume_data.fetch("multistate_checkpoints.zip", processor=pooch.Unzip())
    topdir = "multistate_checkpoints.zip.unzip/multistate_checkpoints"
    subdir = "ahfes"
    filename = "solvent.nc"
    return pathlib.Path(pooch.os_cache("openfe") / f"{topdir}/{subdir}/{filename}")


@pytest.fixture(scope="module")
def sol_checkpoint_path():
    zenodo_resume_data.fetch("multistate_checkpoints.zip", processor=pooch.Unzip())
    topdir = "multistate_checkpoints.zip.unzip/multistate_checkpoints"
    subdir = "ahfes"
    filename = "solvent_checkpoint.chk"
    return pathlib.Path(pooch.os_cache("openfe") / f"{topdir}/{subdir}/{filename}")


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


@pytest.mark.skipif(
    not os.path.exists(POOCH_CACHE) and not HAS_INTERNET,
    reason="Internet unavailable and test data is not cached locally",
)
def test_solvent_check_restart(protocol_settings, sol_trajectory_path):
    assert openmm_afe.ABFESolventSimUnit._check_restart(
        output_settings=protocol_settings.solvent_output_settings,
        shared_path=sol_trajectory_path.parent,
    )

    assert not openmm_afe.ABFESolventSimUnit._check_restart(
        output_settings=protocol_settings.solvent_output_settings,
        shared_path=pathlib.Path("."),
    )


@pytest.mark.skipif(
    not os.path.exists(POOCH_CACHE) and not HAS_INTERNET,
    reason="Internet unavailable and test data is not cached locally",
)
def test_vacuum_check_restart(protocol_settings, vac_trajectory_path):
    assert openmm_afe.ABFEVacuumSimUnit._check_restart(
        output_settings=protocol_settings.vacuum_output_settings,
        shared_path=vac_trajectory_path.parent,
    )

    assert not openmm_afe.ABFEVacuumSimUnit._check_restart(
        output_settings=protocol_settings.vacuum_output_settings,
        shared_path=pathlib.Path("."),
    )



class TestCheckpointResuming:
    @pytest.fixture()
    def protocol_dag(
        self, protocol_settings, benzene_modifications,
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

    def test_resume(self, protocol_dag, tmpdir):
        """
        Attempt to resume a simulation unit with pre-existing checkpoint &
        trajectory files.
        """
        cwd = pathlib.Path("resume_files")
        r = openfe.execute_DAG(protocol_dag, shared_basedir=cwd, scratch_basedir=cwd, keep_shared=True)





# @pytest.mark.integration  # takes too long to be a slow test ~ 4 mins locally
# def test_openmm_run_engine(
#     platform,
#     get_available_openmm_platforms,
#     benzene_modifications,
#     tmpdir,
# ):
#     cwd = pathlib.Path(str(tmpdir))
#     r = execute_DAG(dag, shared_basedir=cwd, scratch_basedir=cwd, keep_shared=True)
# 
#     assert r.ok()
# 
#     # Check outputs of solvent & vacuum results
#     for phase in ["solvent", "vacuum"]:
#         purs = [pur for pur in r.protocol_unit_results if pur.outputs["simtype"] == phase]
# 
#         # get the path to the simulation unit shared dict
#         for pur in purs:
#             if "Simulation" in pur.name:
#                 sim_shared = tmpdir / f"shared_{pur.source_key}_attempt_0"
#                 assert sim_shared.exists()
#                 assert pathlib.Path(sim_shared).is_dir()
# 
#         # check the analysis outputs
#         for pur in purs:
#             if "Analysis" not in pur.name:
#                 continue
# 
#             unit_shared = tmpdir / f"shared_{pur.source_key}_attempt_0"
#             assert unit_shared.exists()
#             assert pathlib.Path(unit_shared).is_dir()
# 
#             # Does the checkpoint file exist?
#             checkpoint = pur.outputs["checkpoint"]
#             assert checkpoint == sim_shared / f"{pur.outputs['simtype']}_checkpoint.nc"
#             assert checkpoint.exists()
# 
#             # Does the trajectory file exist?
#             nc = pur.outputs["trajectory"]
#             assert nc == sim_shared / f"{pur.outputs['simtype']}.nc"
#             assert nc.exists()
# 
#     # Test results methods that need files present
#     results = protocol.gather([r])
#     states = results.get_replica_states()
#     assert len(states.items()) == 2
#     assert len(states["solvent"]) == 1
#     assert states["solvent"][0].shape[1] == 20
