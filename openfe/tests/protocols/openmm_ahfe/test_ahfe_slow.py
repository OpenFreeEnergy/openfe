# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import pathlib

import pytest
from gufe.protocols import execute_DAG
from openff.units import unit

import openfe
from openfe.protocols import openmm_afe


@pytest.mark.integration  # takes too long to be a slow test ~ 4 mins locally
@pytest.mark.flaky(reruns=3)  # pytest-rerunfailures; we can get bad minimisation
@pytest.mark.parametrize("platform", ["CPU", "CUDA"])
def test_openmm_run_engine(
    platform,
    get_available_openmm_platforms,
    benzene_modifications,
    tmpdir,
):
    if platform not in get_available_openmm_platforms:
        pytest.skip(f"OpenMM Platform: {platform} not available")

    # Run a really short calculation to check everything is going well
    s = openmm_afe.AbsoluteSolvationProtocol.default_settings()
    s.protocol_repeats = 1
    s.solvent_output_settings.output_indices = "resname UNK"
    s.vacuum_equil_simulation_settings.equilibration_length = 0.1 * unit.picosecond
    s.vacuum_equil_simulation_settings.production_length = 0.1 * unit.picosecond
    s.vacuum_simulation_settings.equilibration_length = 0.1 * unit.picosecond
    s.vacuum_simulation_settings.production_length = 0.1 * unit.picosecond
    s.solvent_equil_simulation_settings.equilibration_length_nvt = 0.1 * unit.picosecond
    s.solvent_equil_simulation_settings.equilibration_length = 0.1 * unit.picosecond
    s.solvent_equil_simulation_settings.production_length = 0.1 * unit.picosecond
    s.solvent_simulation_settings.equilibration_length = 0.1 * unit.picosecond
    s.solvent_simulation_settings.production_length = 0.1 * unit.picosecond
    s.vacuum_engine_settings.compute_platform = platform
    s.solvent_engine_settings.compute_platform = platform
    s.vacuum_simulation_settings.time_per_iteration = 20 * unit.femtosecond
    s.solvent_simulation_settings.time_per_iteration = 20 * unit.femtosecond
    s.vacuum_output_settings.checkpoint_interval = 20 * unit.femtosecond
    s.solvent_output_settings.checkpoint_interval = 20 * unit.femtosecond
    s.vacuum_simulation_settings.n_replicas = 20
    s.solvent_simulation_settings.n_replicas = 20
    s.lambda_settings.lambda_elec = [
        0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    ]  # fmt: skip
    s.lambda_settings.lambda_vdw = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0
    ]  # fmt: skip
    s.lambda_settings.lambda_restraints = [1.0 for i in range(20)]

    protocol = openmm_afe.AbsoluteSolvationProtocol(
        settings=s,
    )

    stateA = openfe.ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "solvent": openfe.SolventComponent(),
        }
    )

    stateB = openfe.ChemicalSystem({"solvent": openfe.SolventComponent()})

    # Create DAG from protocol, get the vacuum and solvent units
    # and eventually dry run the first solvent unit
    dag = protocol.create(
        stateA=stateA,
        stateB=stateB,
        mapping=None,
    )

    cwd = pathlib.Path(str(tmpdir))
    r = execute_DAG(dag, shared_basedir=cwd, scratch_basedir=cwd, keep_shared=True)

    assert r.ok()

    # get the path to the simulation unit shared dict
    for pur in r.protocol_unit_results:
        if "Simulation" in pur.name:
            sim_shared = tmpdir / f"shared_{pur.source_key}_attempt_0"
            assert sim_shared.exists()
            assert pathlib.Path(sim_shared).is_dir()

    # check the analysis outputs
    for pur in r.protocol_unit_results:
        if "Analysis" not in pur.name:
            continue

        unit_shared = tmpdir / f"shared_{pur.source_key}_attempt_0"
        assert unit_shared.exists()
        assert pathlib.Path(unit_shared).is_dir()

        # Does the checkpoint file exist?
        checkpoint = pur.outputs["checkpoint"]
        assert checkpoint == sim_shared / f"{pur.outputs['simtype']}_checkpoint.nc"
        assert checkpoint.exists()

        # Does the trajectory file exist?
        nc = pur.outputs["trajectory"]
        assert nc == sim_shared / f"{pur.outputs['simtype']}.nc"
        assert nc.exists()

    # Test results methods that need files present
    results = protocol.gather([r])
    states = results.get_replica_states()
    assert len(states.items()) == 2
    assert len(states["solvent"]) == 1
    assert states["solvent"][0].shape[1] == 20
