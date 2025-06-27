# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import pytest
from openff.units import unit
from gufe.protocols import execute_DAG
import openfe
import simtk
from openfe import ChemicalSystem, SolventComponent
from openfe.protocols.openmm_septop import (
    SepTopSolventSetupUnit,
    SepTopProtocol,
)
import numpy as np
from numpy.testing import assert_allclose
from openff.units.openmm import from_openmm
from openfe.protocols.openmm_septop.utils import deserialize, SepTopParameterState

from openmm import Platform
import os
import pathlib
import mdtraj as md

from openfecli.tests.commands.test_atommapping import print_test


@pytest.fixture()
def default_settings():
    return SepTopProtocol.default_settings()


def compare_energies(alchemical_system, positions):

    alchemical_state = SepTopParameterState.from_system(alchemical_system)

    from openmmtools.alchemy import AbsoluteAlchemicalFactory

    energy = AbsoluteAlchemicalFactory.get_energy_components(
        alchemical_system, alchemical_state, positions
    )
    na_A = "alchemically modified NonbondedForce for non-alchemical/alchemical sterics for region A"
    na_B = "alchemically modified NonbondedForce for non-alchemical/alchemical sterics for region B"
    nonbonded = "unmodified NonbondedForce"

    # Lambda 0: LigandA sterics on, elec on, ligand B sterics off, elec off
    alchemical_state.lambda_sterics_A = 1
    alchemical_state.lambda_sterics_B = 0
    alchemical_state.lambda_electrostatics_A = 1
    alchemical_state.lambda_electrostatics_B = 0
    energy_0 = AbsoluteAlchemicalFactory.get_energy_components(
        alchemical_system, alchemical_state, positions
    )

    # Lambda 7: LigandA sterics on, elec on, ligand B sterics on, elec off
    alchemical_state.lambda_sterics_A = 1
    alchemical_state.lambda_sterics_B = 1
    alchemical_state.lambda_electrostatics_A = 1
    alchemical_state.lambda_electrostatics_B = 0
    energy_7 = AbsoluteAlchemicalFactory.get_energy_components(
        alchemical_system, alchemical_state, positions
    )

    # Lambda 8: LigandA sterics on, elec partially on,
    # ligand B sterics on, elec partially on
    alchemical_state.lambda_sterics_A = 1
    alchemical_state.lambda_sterics_B = 1
    alchemical_state.lambda_electrostatics_A = 0.75
    alchemical_state.lambda_electrostatics_B = 0.25
    energy_8 = AbsoluteAlchemicalFactory.get_energy_components(
        alchemical_system, alchemical_state, positions
    )

    # Lambda 12: LigandA sterics on, elec off, ligand B sterics on, elec on
    alchemical_state.lambda_sterics_A = 1
    alchemical_state.lambda_sterics_B = 1
    alchemical_state.lambda_electrostatics_A = 0
    alchemical_state.lambda_electrostatics_B = 1
    energy_12 = AbsoluteAlchemicalFactory.get_energy_components(
        alchemical_system, alchemical_state, positions
    )

    # Lambda 13: LigandA sterics partially on, elec off, ligand B sterics on, elec on
    alchemical_state.lambda_sterics_A = 0.857142857
    alchemical_state.lambda_sterics_B = 1
    alchemical_state.lambda_electrostatics_A = 0
    alchemical_state.lambda_electrostatics_B = 1
    energy_13 = AbsoluteAlchemicalFactory.get_energy_components(
        alchemical_system, alchemical_state, positions
    )

    return (
        na_A,
        na_B,
        nonbonded,
        energy,
        energy_0,
        energy_7,
        energy_8,
        energy_12,
        energy_13,
    )


@pytest.mark.flaky(reruns=3)  # pytest-rerunfailures; we can get bad minimisation
def test_lambda_energies(
    eg5_ligands,
    eg5_protein,
    eg5_cofactor,
    tmpdir,
    default_settings,
):
    # check system parametrisation works even if confgen fails
    default_settings.protocol_repeats = 1
    default_settings.solvent_equil_simulation_settings.minimization_steps = 100
    default_settings.solvent_equil_simulation_settings.equilibration_length_nvt = (
        10 * unit.picosecond
    )
    default_settings.solvent_equil_simulation_settings.equilibration_length = (
        10 * unit.picosecond
    )
    default_settings.solvent_equil_simulation_settings.production_length = (
        1 * unit.picosecond
    )
    default_settings.solvent_solvation_settings.box_shape = "dodecahedron"
    default_settings.solvent_solvation_settings.solvent_padding = 1.8 * unit.nanometer

    protocol = SepTopProtocol(
        settings=default_settings,
    )

    stateA = ChemicalSystem(
        {
            "lig_02": eg5_ligands[0],
            "protein": eg5_protein,
            "cofactor": eg5_cofactor,
            "solvent": SolventComponent(),
        }
    )

    stateB = ChemicalSystem(
        {
            "lig_03": eg5_ligands[1],
            "protein": eg5_protein,
            "cofactor": eg5_cofactor,
            "solvent": SolventComponent(),
        }
    )

    # Create DAG from protocol, get the vacuum and solvent units
    # and eventually dry run the first vacuum unit
    dag = protocol.create(
        stateA=stateA,
        stateB=stateB,
        mapping=None,
    )
    prot_units = list(dag.protocol_units)
    solv_setup_unit = [u for u in prot_units if isinstance(u, SepTopSolventSetupUnit)]

    with tmpdir.as_cwd():
        output = solv_setup_unit[0].run()
        system = output["system"]
        alchemical_system = deserialize(system)
        topology = output["topology"]
        pdb = simtk.openmm.app.pdbfile.PDBFile(str(topology))
        positions = pdb.getPositions(asNumpy=True)

        # Remove Harmonic restraint force solvent
        alchemical_system.removeForce(13)

        (
            na_A,
            na_B,
            nonbonded,
            energy,
            energy_0,
            energy_7,
            energy_8,
            energy_12,
            energy_13,
        ) = compare_energies(alchemical_system, positions)

        for key, value in energy.items():
            if key == na_A:
                assert_allclose(from_openmm(value), from_openmm(energy_0[key]))
                assert_allclose(from_openmm(value), from_openmm(energy_7[key]))
                assert_allclose(from_openmm(value), from_openmm(energy_8[key]))
                assert_allclose(from_openmm(value), from_openmm(energy_12[key]))
                assert not np.allclose(from_openmm(value), from_openmm(energy_13[key]))

            elif key == na_B:
                assert not np.allclose(from_openmm(value), from_openmm(energy_0[key]))
                assert_allclose(from_openmm(energy_0[key]), 0)
                assert_allclose(from_openmm(value), from_openmm(energy_7[key]))
                assert_allclose(from_openmm(value), from_openmm(energy_8[key]))
                assert_allclose(from_openmm(value), from_openmm(energy_12[key]))
                assert_allclose(from_openmm(value), from_openmm(energy_13[key]))

            elif key == nonbonded:
                assert not np.allclose(from_openmm(value), from_openmm(energy_0[key]))
                assert_allclose(
                    from_openmm(energy_0[key]),
                    from_openmm(energy_7[key]),
                    rtol=1e-05,
                )
                assert not np.allclose(
                    from_openmm(energy_0[key]), from_openmm(energy_8[key])
                )
                assert not np.allclose(
                    from_openmm(energy_0[key]), from_openmm(energy_12[key])
                )
                assert not np.allclose(
                    from_openmm(energy_0[key]), from_openmm(energy_13[key])
                )

            else:
                assert_allclose(from_openmm(value), from_openmm(energy_0[key]))
                assert_allclose(from_openmm(value), from_openmm(energy_7[key]))
                assert_allclose(from_openmm(value), from_openmm(energy_8[key]))
                assert_allclose(from_openmm(value), from_openmm(energy_12[key]))
                assert_allclose(from_openmm(value), from_openmm(energy_13[key]))


@pytest.fixture
def available_platforms() -> set[str]:
    return {
        Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())
    }


@pytest.fixture
def set_openmm_threads_1():
    # for vacuum sims, we want to limit threads to one
    # this fixture sets OPENMM_CPU_THREADS='1' for a single test, then reverts to previously held value
    previous: str | None = os.environ.get("OPENMM_CPU_THREADS")

    try:
        os.environ["OPENMM_CPU_THREADS"] = "1"
        yield
    finally:
        if previous is None:
            del os.environ["OPENMM_CPU_THREADS"]
        else:
            os.environ["OPENMM_CPU_THREADS"] = previous


@pytest.mark.integration
@pytest.mark.flaky(reruns=3)  # pytest-rerunfailures; we can get bad minimisation
@pytest.mark.parametrize("platform", ["CPU", "CUDA"])
def test_openmm_run_engine(
    platform,
    available_platforms,
    benzene_modifications,
    T4_protein_component,
    set_openmm_threads_1,
    tmpdir,
    default_settings,
):
    if platform not in available_platforms:
        pytest.skip(f"OpenMM Platform: {platform} not available")

    # Run a really short calculation to check everything is going well
    default_settings.protocol_repeats = 1
    default_settings.solvent_output_settings.output_indices = "resname UNK"
    default_settings.complex_equil_simulation_settings.equilibration_length = (
        0.1 * unit.picosecond
    )
    default_settings.complex_equil_simulation_settings.production_length = (
        0.1 * unit.picosecond
    )
    default_settings.complex_simulation_settings.equilibration_length = (
        0.1 * unit.picosecond
    )
    default_settings.complex_simulation_settings.production_length = (
        0.1 * unit.picosecond
    )
    default_settings.solvent_equil_simulation_settings.equilibration_length_nvt = (
        0.1 * unit.picosecond
    )
    default_settings.solvent_equil_simulation_settings.equilibration_length = (
        0.1 * unit.picosecond
    )
    default_settings.solvent_equil_simulation_settings.production_length = (
        0.1 * unit.picosecond
    )
    default_settings.solvent_simulation_settings.equilibration_length = (
        0.1 * unit.picosecond
    )
    default_settings.solvent_simulation_settings.production_length = (
        0.1 * unit.picosecond
    )
    default_settings.engine_settings.compute_platform = platform
    default_settings.complex_simulation_settings.time_per_iteration = (
        20 * unit.femtosecond
    )
    default_settings.solvent_simulation_settings.time_per_iteration = (
        20 * unit.femtosecond
    )
    default_settings.complex_output_settings.checkpoint_interval = 20 * unit.femtosecond
    default_settings.solvent_output_settings.checkpoint_interval = 20 * unit.femtosecond

    protocol = SepTopProtocol(
        settings=default_settings,
    )

    stateA = openfe.ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "T4L": T4_protein_component,
            "solvent": openfe.SolventComponent(),
        }
    )

    stateB = openfe.ChemicalSystem(
        {
            "toluene": benzene_modifications["toluene"],
            "T4L": T4_protein_component,
            "solvent": openfe.SolventComponent(),
        }
    )

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
    for pur in r.protocol_unit_results:
        unit_shared = tmpdir / f"shared_{pur.source_key}_attempt_0"
        assert unit_shared.exists()
        assert pathlib.Path(unit_shared).is_dir()
        checkpoint = pur.outputs["last_checkpoint"]
        assert checkpoint == f"{pur.outputs['simtype']}_checkpoint.nc"
        assert (unit_shared / checkpoint).exists()
        nc = pur.outputs["nc"]
        assert nc == unit_shared / f"{pur.outputs['simtype']}.nc"
        assert nc.exists()

    # Test results methods that need files present
    results = protocol.gather([r])
    states = results.get_replica_states()
    assert len(states.items()) == 2
    assert len(states["solvent"]) == 1
    assert states["solvent"][0].shape[1] == 19


@pytest.mark.flaky(reruns=1)  # pytest-rerunfailures; we can get bad minimisation
@pytest.mark.parametrize("platform", ["CPU", "CUDA"])
def test_restraints_solvent(
    platform,
    available_platforms,
    benzene_complex_system,
    toluene_complex_system,
    set_openmm_threads_1,
    tmpdir,
    default_settings,
):
    if platform not in available_platforms:
        pytest.skip(f"OpenMM Platform: {platform} not available")

    # Run a really short calculation to check everything is going well
    default_settings.protocol_repeats = 1
    default_settings.solvent_equil_simulation_settings.equilibration_length_nvt = (
        10 * unit.picosecond
    )
    default_settings.solvent_equil_simulation_settings.equilibration_length = (
        10 * unit.picosecond
    )
    default_settings.solvent_equil_simulation_settings.production_length = (
        10 * unit.picosecond
    )
    default_settings.engine_settings.compute_platform = platform

    protocol = SepTopProtocol(
        settings=default_settings,
    )

    # Create DAG from protocol, get the vacuum and solvent units
    # and eventually dry run the first solvent unit
    dag = protocol.create(
        stateA=benzene_complex_system,
        stateB=toluene_complex_system,
        mapping=None,
    )
    prot_units = list(dag.protocol_units)
    solv_setup_unit = [u for u in prot_units if isinstance(u, SepTopSolventSetupUnit)]
    solv_setup_output = solv_setup_unit[0].run()
    pdb = md.load_pdb("topology.pdb")
    assert pdb.n_atoms == 1346
    central_atoms = np.array([[2, 19]], dtype=np.int32)
    distance = md.compute_distances(pdb, central_atoms)[0][0]
    # For right now just checking that ligands at least somewhat apart
    assert distance > 0.5
