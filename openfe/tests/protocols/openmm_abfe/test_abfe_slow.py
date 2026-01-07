# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import pathlib

import pytest
from openff.units import unit as offunit

import openfe
from openfe.protocols.openmm_afe import AbsoluteBindingProtocol
from openfe.protocols.openmm_utils.charge_generation import HAS_NAGL, HAS_OPENEYE


@pytest.mark.integration
#@pytest.mark.flaky(reruns=3)  # pytest-rerunfailures; we can get bad minimisation
@pytest.mark.skipif(not HAS_NAGL, reason="need NAGL")
@pytest.mark.xfail(
    HAS_OPENEYE and HAS_NAGL,
    reason="NAGL/openeye incompatibility. See https://github.com/openforcefield/openff-nagl/issues/177",
)
@pytest.mark.parametrize("platform", ["CUDA"])
def test_openmm_run_engine(
    platform,
    available_platforms,
    eg5_protein,
    eg5_ligands,
    eg5_cofactor,
    tmpdir,
):
    if platform not in available_platforms:
        pytest.skip(f"OpenMM Platform: {platform} not available")

    settings = AbsoluteBindingProtocol.default_settings()

    # Run a really short calculation to check everything is going well
    settings.protocol_repeats = 1
    settings.engine_settings.compute_platform = "CUDA"

    # Solvent
    settings.solvent_equil_simulation_settings.equilibration_length_nvt = 10 * offunit.picosecond
    settings.solvent_equil_simulation_settings.equilibration_length = 10 * offunit.picosecond
    settings.solvent_equil_simulation_settings.production_length = 10 * offunit.picosecond
    settings.solvent_simulation_settings.equilibration_length = 50 * offunit.picosecond
    settings.solvent_simulation_settings.production_length = 125 * offunit.picosecond
    settings.solvent_simulation_settings.early_termination_target_error = 0.12 * offunit.kilocalorie_per_mole  # fmt: skip
    settings.solvent_simulation_settings.time_per_iteration = 2.5 * offunit.ps
    settings.solvent_solvation_settings.box_shape = "dodecahedron"
    settings.solvent_solvation_settings.solvent_padding = 1.5 * offunit.nanometer

    # Complex
    settings.complex_equil_simulation_settings.equilibration_length_nvt = 10 * offunit.picosecond
    settings.complex_equil_simulation_settings.equilibration_length = 10 * offunit.picosecond
    settings.complex_equil_simulation_settings.production_length = 100 * offunit.picosecond
    settings.complex_simulation_settings.equilibration_length = 50 * offunit.picosecond
    settings.complex_simulation_settings.production_length = 125 * offunit.picosecond
    settings.complex_simulation_settings.early_termination_target_error = 0.12 * offunit.kilocalorie_per_mole  # fmt: skip
    settings.complex_simulation_settings.time_per_iteration = 2.5 * offunit.ps
    settings.complex_solvation_settings.box_shape = "dodecahedron"
    settings.complex_solvation_settings.solvent_padding = 0.9 * offunit.nanometer

    # General FF things
    settings.forcefield_settings.nonbonded_cutoff = 0.8 * offunit.nanometer
    settings.partial_charge_settings.partial_charge_method = "nagl"
    settings.partial_charge_settings.nagl_model = "openff-gnn-am1bcc-0.1.0-rc.3.pt"

    protocol = AbsoluteBindingProtocol(settings=settings)

    # unpack ligands
    ligand, _ = eg5_ligands

    stateA = openfe.ChemicalSystem(
        {
            "protein": eg5_protein,
            "cofactor": eg5_cofactor,
            "ligand": ligand,
            "solvent": openfe.SolventComponent(),
        }
    )

    stateB = openfe.ChemicalSystem(
        {
            "protein": eg5_protein,
            "cofactor": eg5_cofactor,
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
    r = openfe.execute_DAG(dag, shared_basedir=cwd, scratch_basedir=cwd, keep_shared=True)

    assert r.ok()

    # Check outputs of solvent & complex results
    for phase in ["solvent", "complex"]:
        purs = [
            pur for pur in r.protocol_unit_results
            if pur.outputs["simtype"] == phase
        ]

        # get the path to the simulation unit shared dict
        for pur in purs:
            if "Simulation" in pur.name:
                sim_shared = tmpdir / f"shared_{pur.source_key}_attempt_0"
                assert sim_shared.exists()
                assert pathlib.Path(sim_shared).is_dir()

        # check the analysis outputs
        for pur in purs:
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
    assert len(states["complex"]) == 1
    assert states["solvent"][0].shape[1] == 14
    assert states["complex"][0].shape[1] == 30
