# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from gufe.protocols import execute_DAG
import pytest
from openff.units import unit
from openmm import Platform
import os
import pathlib

import openfe
from openfe.protocols import openmm_afe


@pytest.fixture
def available_platforms() -> set[str]:
    return {Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())}


@pytest.fixture
def set_openmm_threads_1():
    # for vacuum sims, we want to limit threads to one
    # this fixture sets OPENMM_CPU_THREADS='1' for a single test, then reverts to previously held value
    previous: str | None = os.environ.get('OPENMM_CPU_THREADS')

    try:
        os.environ['OPENMM_CPU_THREADS'] = '1'
        yield
    finally:
        if previous is None:
            del os.environ['OPENMM_CPU_THREADS']
        else:
            os.environ['OPENMM_CPU_THREADS'] = previous


@pytest.mark.integration  # takes too long to be a slow test ~ 4 mins locally
@pytest.mark.flaky(reruns=3)  # pytest-rerunfailures; we can get bad minimisation
@pytest.mark.parametrize('platform', ['CPU', 'CUDA'])
def test_openmm_run_engine(platform,
                           available_platforms,
                           benzene_modifications,
                           set_openmm_threads_1, tmpdir):
    if platform not in available_platforms:
        pytest.skip(f"OpenMM Platform: {platform} not available")

    # Run a really short calculation to check everything is going well
    s = openmm_afe.AbsoluteSolvationProtocol.default_settings()
    s.alchemsampler_settings.n_repeats = 1
    s.solvent_output_settings.output_indices = "resname UNK"
    s.vacuum_simulation_settings.equilibration_length = 0.1 * unit.picosecond
    s.vacuum_simulation_settings.production_length = 0.1 * unit.picosecond
    s.solvent_simulation_settings.equilibration_length = 0.1 * unit.picosecond
    s.solvent_simulation_settings.production_length = 0.1 * unit.picosecond
    s.vacuum_engine_settings.compute_platform = platform
    s.solvent_engine_settings.compute_platform = platform
    s.alchemsampler_settings.steps_per_iteration = 5 * unit.timestep
    s.vacuum_output_settings.checkpoint_interval = 5 * unit.timestep
    s.solvent_output_settings.checkpoint_interval = 5 * unit.timestep
    s.alchemsampler_settings.n_replicas = 20
    s.lambda_settings.lambda_elec = \
        [0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    s.lambda_settings.lambda_vdw = \
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]


    protocol = openmm_afe.AbsoluteSolvationProtocol(
            settings=s,
    )

    stateA = openfe.ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'solvent': openfe.SolventComponent()
    })

    stateB = openfe.ChemicalSystem({
        'solvent': openfe.SolventComponent(),
    })

    # Create DAG from protocol, get the vacuum and solvent units
    # and eventually dry run the first solvent unit
    dag = protocol.create(
        stateA=stateA,
        stateB=stateB,
        mapping=None,
    )


    cwd = pathlib.Path(str(tmpdir))
    r = execute_DAG(dag, shared_basedir=cwd, scratch_basedir=cwd,
                    keep_shared=True)

    assert r.ok()
    for pur in r.protocol_unit_results:
        unit_shared = tmpdir / f"shared_{pur.source_key}_attempt_0"
        assert unit_shared.exists()
        assert pathlib.Path(unit_shared).is_dir()
        checkpoint = pur.outputs['last_checkpoint']
        assert checkpoint == f"{pur.outputs['simtype']}_checkpoint.nc"
        assert (unit_shared / checkpoint).exists()
        nc = pur.outputs['nc']
        assert nc == unit_shared / f"{pur.outputs['simtype']}.nc"
        assert nc.exists()

    # Test results methods that need files present
    results = protocol.gather([r])
    states = results.get_replica_states()
    assert len(states.items()) == 2
    assert len(states['solvent']) == 1
    assert states['solvent'][0].shape[1] == 20
