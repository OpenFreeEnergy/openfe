# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from gufe.protocols import execute_DAG
import pytest
from openff.units import unit
from openmm import Platform
import os
import pathlib

import openfe
from openfe.protocols import openmm_rfe


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


@pytest.mark.slow
@pytest.mark.flaky(reruns=3)  # pytest-rerunfailures; we can get bad minimisation
@pytest.mark.parametrize('platform', ['CPU', 'CUDA'])
def test_openmm_run_engine(benzene_vacuum_system, platform,
                           available_platforms, benzene_modifications,
                           set_openmm_threads_1, tmpdir):
    if platform not in available_platforms:
        pytest.skip(f"OpenMM Platform: {platform} not available")
    # this test actually runs MD
    # if this passes, you're 99% likely to have a good time
    # these settings are a small self to self sim, that has enough eq that it doesn't occasionally crash
    s = openfe.protocols.openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
    s.simulation_settings.equilibration_length = 0.1 * unit.picosecond
    s.simulation_settings.production_length = 0.1 * unit.picosecond
    s.integrator_settings.n_steps = 5 * unit.timestep
    s.system_settings.nonbonded_method = 'nocutoff'
    s.alchemical_sampler_settings.n_repeats = 1
    s.engine_settings.compute_platform = platform
    s.simulation_settings.checkpoint_interval = 5 * unit.timestep

    p = openmm_rfe.RelativeHybridTopologyProtocol(s)

    b = benzene_vacuum_system['ligand']

    # make a copy with a different name
    rdmol = benzene_modifications['benzene'].to_rdkit()
    b_alt = openfe.SmallMoleculeComponent.from_rdkit(rdmol, name='alt')
    benzene_vacuum_alt_system = openfe.ChemicalSystem({
        'ligand': b_alt
    })

    m = openfe.LigandAtomMapping(componentA=b, componentB=b_alt,
                                 componentA_to_componentB={i: i for i in range(12)})
    dag = p.create(stateA=benzene_vacuum_system, stateB=benzene_vacuum_alt_system,
                   mapping={'ligand': m})

    cwd = pathlib.Path(str(tmpdir))
    r = execute_DAG(dag, shared_basedir=cwd, scratch_basedir=cwd,
                    keep_shared=True)

    assert r.ok()
    for pur in r.protocol_unit_results:
        unit_shared = tmpdir / f"shared_{pur.source_key}_attempt_0"
        assert unit_shared.exists()
        assert pathlib.Path(unit_shared).is_dir()
        checkpoint = pur.outputs['last_checkpoint']
        assert checkpoint == unit_shared / "checkpoint.nc"
        assert checkpoint.exists()
        nc = pur.outputs['nc']
        assert nc == unit_shared / "simulation.nc"
        assert nc.exists()


@pytest.mark.slow  # takes ~7 minutes to run
@pytest.mark.flaky(reruns=3)
def test_run_eg5_sim(eg5_protein, eg5_ligands, eg5_cofactor, tmpdir):
    # this runs a very short eg5 complex leg
    # different to previous test:
    # - has a cofactor
    # - has an alchemical swap present
    # - runs in solvated protein
    # if this passes 99.9% chance of a good time
    s = openfe.protocols.openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
    s.simulation_settings.equilibration_length = 0.1 * unit.picosecond
    s.simulation_settings.production_length = 0.1 * unit.picosecond
    s.integrator_settings.n_steps = 5 * unit.timestep
    s.alchemical_sampler_settings.n_repeats = 1
    s.simulation_settings.checkpoint_interval = 5 * unit.timestep

    p = openmm_rfe.RelativeHybridTopologyProtocol(s)

    base_sys = {
        'protein': eg5_protein,
        'cofactor': eg5_cofactor,
        'solvent': openfe.SolventComponent(),
    }
    # this is just a simple (unmapped) *-H -> *-F switch
    l1, l2 = eg5_ligands[0], eg5_ligands[1]
    m = openfe.LigandAtomMapping(
        componentA=l1, componentB=l2,
        # a bit lucky, first 51 atoms map to each other, H->F swap is at 52
        componentA_to_componentB={i: i for i in range(51)}
    )

    sys1 = openfe.ChemicalSystem(components={**base_sys, 'ligand': l1})
    sys2 = openfe.ChemicalSystem(components={**base_sys, 'ligand': l2})

    dag = p.create(stateA=sys1, stateB=sys2,
                   mapping={'ligand': m})

    cwd = pathlib.Path(str(tmpdir))
    r = execute_DAG(dag, shared_basedir=cwd, scratch_basedir=cwd,
                    keep_shared=True)

    assert r.ok()
