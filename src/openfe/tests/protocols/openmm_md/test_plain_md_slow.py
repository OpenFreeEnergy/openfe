# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import pathlib

import pytest
from gufe.protocols import execute_DAG
from openff.units import unit

from openfe.protocols import openmm_md


@pytest.mark.integration
@pytest.mark.parametrize("platform", ["CPU", "CUDA"])
def test_vacuum_sim(
    benzene_vacuum_system,
    platform,
    available_platforms,
    tmp_path,
):
    if platform not in available_platforms:
        pytest.skip(f"OpenMM Platform: {platform} is not available")

    # Run a vacuum MD simulation and check what files we get.
    settings = openmm_md.PlainMDProtocol.default_settings()
    settings.simulation_settings.equilibration_length_nvt = None
    settings.simulation_settings.equilibration_length = 10 * unit.picosecond
    settings.simulation_settings.production_length = 20 * unit.picosecond
    settings.output_settings.checkpoint_interval = 5 * unit.picosecond
    settings.forcefield_settings.nonbonded_method = "nocutoff"
    settings.engine_settings.compute_platform = platform

    prot = openmm_md.PlainMDProtocol(settings)

    dag = prot.create(
        stateA=benzene_vacuum_system,
        stateB=benzene_vacuum_system,
        mapping=None,
    )

    r = execute_DAG(
        dag,
        shared_basedir=tmp_path,
        scratch_basedir=tmp_path,
        keep_shared=True,
    )

    assert r.ok()

    assert len(r.protocol_unit_results) == 2

    pur = r.protocol_unit_results[1]
    unit_shared = tmp_path / f"shared_{pur.source_key}_attempt_0"
    assert unit_shared.exists()
    assert pathlib.Path(unit_shared).is_dir()

    # check the files
    files = [
        "equil_npt.pdb",
        "minimized.pdb",
        "simulation.xtc",
        "simulation.log",
        "checkpoint.xml",
    ]
    for file in files:
        assert (unit_shared / file).exists()

    # NVT PDB should not exist
    assert not (unit_shared / "equil_nvt.pdb").exists()

    # check that the output file paths are correct
    assert pur.outputs["system_pdb"] == unit_shared / "system.pdb"
    assert pur.outputs["minimized_pdb"] == unit_shared / "minimized.pdb"
    assert pur.outputs["nc"] == unit_shared / "simulation.xtc"
    assert pur.outputs["last_checkpoint"] == unit_shared / "checkpoint.xml"
    assert pur.outputs["npt_equil_pdb"] == unit_shared / "equil_npt.pdb"
    assert pur.outputs["nvt_equil_pdb"] is None


@pytest.mark.integration
@pytest.mark.parametrize("platform", ["CUDA"])
def test_complex_solvent_sim_gpu(
    benzene_complex_system,
    platform,
    available_platforms,
    tmp_path,
):
    if platform not in available_platforms:
        pytest.skip(f"OpenMM Platform: {platform} is not available")

    # Run an MD simulation and check what files we get.
    settings = openmm_md.PlainMDProtocol.default_settings()
    settings.simulation_settings.equilibration_length_nvt = 50 * unit.picosecond
    settings.simulation_settings.equilibration_length = 50 * unit.picosecond
    settings.simulation_settings.production_length = 100 * unit.picosecond
    settings.output_settings.checkpoint_interval = 10 * unit.picosecond
    settings.engine_settings.compute_platform = platform

    prot = openmm_md.PlainMDProtocol(settings)

    dag = prot.create(
        stateA=benzene_complex_system,
        stateB=benzene_complex_system,
        mapping=None,
    )

    r = execute_DAG(
        dag,
        shared_basedir=tmp_path,
        scratch_basedir=tmp_path,
        keep_shared=True,
    )

    assert r.ok()

    assert len(r.protocol_unit_results) == 2

    pur = r.protocol_unit_results[1]
    unit_shared = tmp_path / f"shared_{pur.source_key}_attempt_0"
    assert unit_shared.exists()
    assert pathlib.Path(unit_shared).is_dir()

    # check the files
    files = [
        "checkpoint.xml",
        "equil_nvt.pdb",
        "equil_npt.pdb",
        "minimized.pdb",
        "simulation.xtc",
        "simulation.log",
    ]
    for file in files:
        assert (unit_shared / file).exists()

    # check that the output file paths are correct
    assert pur.outputs["system_pdb"] == unit_shared / "system.pdb"
    assert pur.outputs["minimized_pdb"] == unit_shared / "minimized.pdb"
    assert pur.outputs["nc"] == unit_shared / "simulation.xtc"
    assert pur.outputs["last_checkpoint"] == unit_shared / "checkpoint.xml"
    assert pur.outputs["nvt_equil_pdb"] == unit_shared / "equil_nvt.pdb"
    assert pur.outputs["npt_equil_pdb"] == unit_shared / "equil_npt.pdb"
