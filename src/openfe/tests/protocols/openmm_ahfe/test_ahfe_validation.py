# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import pytest
from gufe import LigandAtomMapping, ProtocolDAGResult
from gufe.settings import OpenMMSystemGeneratorFFSettings, ThermoSettings
from openff.units import unit as offunit

from openfe import ChemicalSystem, SolventComponent
from openfe.protocols import openmm_afe
from openfe.protocols.openmm_afe import AbsoluteSolvationProtocol, AbsoluteSolvationSettings
from openfe.protocols.openmm_afe.equil_afe_settings import (
    AbsoluteSolvationSettings,
    AlchemicalSettings,
    IntegratorSettings,
    LambdaSettings,
    MDOutputSettings,
    MDSimulationSettings,
    MultiStateOutputSettings,
    MultiStateSimulationSettings,
    OpenFFPartialChargeSettings,
    OpenMMEngineSettings,
    OpenMMSolvationSettings,
)
from openfe.protocols.openmm_utils import system_validation


@pytest.fixture()
def default_settings():
    return AbsoluteSolvationProtocol.default_settings()


@pytest.fixture()
def stateA(benzene_modifications):
    return ChemicalSystem(
        {"benzene": benzene_modifications["benzene"], "solvent": SolventComponent()}
    )


@pytest.fixture()
def stateB():
    return ChemicalSystem({"solvent": SolventComponent()})


@pytest.mark.parametrize(
    "val",
    [
        {"elec": [0.0, 1.0], "vdw": [1.0, 1.0], "restraints": [0.0, 0.0]},
    ],
)
def test_validate_lambda_schedule_naked_charge(val, default_settings):
    errmsg = (
        "There are states along this lambda schedule "
        "where there are atoms with charges but no LJ "
        f"interactions: lambda 0: "
        f"elec {val['elec'][0]} vdW {val['vdw'][0]}"
    )
    default_settings.lambda_settings.lambda_elec = val["elec"]
    default_settings.lambda_settings.lambda_vdw = val["vdw"]
    default_settings.lambda_settings.lambda_restraints = val["restraints"]
    default_settings.vacuum_simulation_settings.n_replicas = 2
    default_settings.solvent_simulation_settings.n_replicas = 2
    with pytest.raises(ValueError, match=errmsg):
        AbsoluteSolvationProtocol._validate_lambda_schedule(
            default_settings.lambda_settings,
            default_settings.vacuum_simulation_settings,
        )
    with pytest.raises(ValueError, match=errmsg):
        AbsoluteSolvationProtocol._validate_lambda_schedule(
            default_settings.lambda_settings,
            default_settings.solvent_simulation_settings,
        )


@pytest.mark.parametrize(
    "val",
    [
        {"elec": [1.0, 1.0], "vdw": [0.0, 1.0], "restraints": [0.0, 0.0]},
    ],
)
def test_validate_lambda_schedule_nreplicas(val, default_settings):
    default_settings.lambda_settings.lambda_elec = val["elec"]
    default_settings.lambda_settings.lambda_vdw = val["vdw"]
    default_settings.lambda_settings.lambda_restraints = val["restraints"]
    n_replicas = 3
    default_settings.vacuum_simulation_settings.n_replicas = n_replicas
    errmsg = (
        f"Number of replicas {n_replicas} does not equal the"
        f" number of lambda windows {len(val['vdw'])}"
    )
    with pytest.raises(ValueError, match=errmsg):
        AbsoluteSolvationProtocol._validate_lambda_schedule(
            default_settings.lambda_settings,
            default_settings.vacuum_simulation_settings,
        )


@pytest.mark.parametrize(
    "val",
    [
        {"elec": [1.0, 1.0, 1.0], "vdw": [0.0, 1.0], "restraints": [0.0, 0.0]},
    ],
)
def test_validate_lambda_schedule_nwindows(val, default_settings):
    default_settings.lambda_settings.lambda_elec = val["elec"]
    default_settings.lambda_settings.lambda_vdw = val["vdw"]
    default_settings.lambda_settings.lambda_restraints = val["restraints"]
    n_replicas = 3
    default_settings.vacuum_simulation_settings.n_replicas = n_replicas
    errmsg = (
        "Components elec, vdw, and restraints must have equal amount"
        f" of lambda windows. Got {len(val['elec'])} elec lambda"
        f" windows, {len(val['vdw'])} vdw lambda windows, and"
        f"{len(val['restraints'])} restraints lambda windows."
    )
    with pytest.raises(ValueError, match=errmsg):
        AbsoluteSolvationProtocol._validate_lambda_schedule(
            default_settings.lambda_settings,
            default_settings.vacuum_simulation_settings,
        )


@pytest.mark.parametrize(
    "val",
    [
        {"elec": [1.0, 1.0], "vdw": [1.0, 1.0], "restraints": [0.0, 1.0]},
    ],
)
def test_validate_lambda_schedule_nonzero_restraints(val, default_settings):
    wmsg = (
        "Non-zero restraint lambdas applied. The absolute "
        "solvation protocol doesn't apply restraints, "
        "therefore restraints won't be applied."
    )
    default_settings.lambda_settings.lambda_elec = val["elec"]
    default_settings.lambda_settings.lambda_vdw = val["vdw"]
    default_settings.lambda_settings.lambda_restraints = val["restraints"]
    default_settings.vacuum_simulation_settings.n_replicas = 2
    with pytest.warns(UserWarning, match=wmsg):
        AbsoluteSolvationProtocol._validate_lambda_schedule(
            default_settings.lambda_settings,
            default_settings.vacuum_simulation_settings,
        )


def test_validate_endstates_protcomp(benzene_modifications, T4_protein_component):
    stateA = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "protein": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )

    stateB = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "phenol": benzene_modifications["phenol"],
            "solvent": SolventComponent(),
        }
    )

    with pytest.raises(ValueError, match="Protein components are not allowed"):
        AbsoluteSolvationProtocol._validate_endstates(stateA, stateB)


def test_validate_endstates_nosolvcomp_stateA(benzene_modifications, T4_protein_component):
    stateA = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
        }
    )

    stateB = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "phenol": benzene_modifications["phenol"],
            "solvent": SolventComponent(),
        }
    )

    with pytest.raises(ValueError, match="No SolventComponent found in stateA"):
        AbsoluteSolvationProtocol._validate_endstates(stateA, stateB)


def test_validate_endstates_nosolvcomp_stateB(benzene_modifications, T4_protein_component):
    stateA = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "solvent": SolventComponent(),
        }
    )

    stateB = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "phenol": benzene_modifications["phenol"],
        }
    )

    with pytest.raises(ValueError, match="No SolventComponent found in stateA and/or stateB"):
        AbsoluteSolvationProtocol._validate_endstates(stateA, stateB)


def test_validate_alchem_comps_appearingB(benzene_modifications):
    stateA = ChemicalSystem(
        {
            "solvent": SolventComponent(),
            "toluene": benzene_modifications["toluene"],
        }
    )

    stateB = ChemicalSystem(
        {"benzene": benzene_modifications["benzene"], "solvent": SolventComponent()}
    )

    with pytest.raises(ValueError, match="Components appearing in state B"):
        AbsoluteSolvationProtocol._validate_endstates(stateA, stateB)


def test_validate_alchem_comps_multi(benzene_modifications):
    stateA = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "toluene": benzene_modifications["toluene"],
            "solvent": SolventComponent(),
        }
    )

    stateB = ChemicalSystem({"solvent": SolventComponent()})

    alchem_comps = system_validation.get_alchemical_components(stateA, stateB)

    assert len(alchem_comps["stateA"]) == 2

    with pytest.raises(ValueError, match="Only one alchemical species"):
        AbsoluteSolvationProtocol._validate_endstates(stateA, stateB)


def test_validate_alchem_nonsmc(benzene_modifications):
    stateA = ChemicalSystem(
        {
            "solvent": SolventComponent(),
            "solvent2": SolventComponent(smiles="C"),
        }
    )

    stateB = ChemicalSystem(
        {
            "solvent": SolventComponent(),
        }
    )

    errmsg = "Only disappearing SmallMoleculeComponents"
    with pytest.raises(ValueError, match=errmsg):
        AbsoluteSolvationProtocol._validate_endstates(stateA, stateB)


def test_charged_alchem_comp(charged_benzene_modifications):
    stateA = ChemicalSystem(
        {
            "solute": charged_benzene_modifications["benzoic_acid"],
            "solvent": SolventComponent(),
        }
    )

    stateB = ChemicalSystem(
        {
            "solvent": SolventComponent(),
        }
    )

    assert charged_benzene_modifications["benzoic_acid"].total_charge == -1

    with pytest.raises(ValueError, match="Charged alchemical molecules"):
        AbsoluteSolvationProtocol._validate_endstates(stateA, stateB)


def test_extends_error(default_settings, stateA, stateB):
    fake_results = ProtocolDAGResult(
        protocol_units=[], protocol_unit_results=[], transformation_key="foo"
    )
    protocol = openmm_afe.AbsoluteSolvationProtocol(settings=default_settings)

    with pytest.raises(ValueError, match="Can't extend simulation"):
        protocol.validate(stateA=stateA, stateB=stateB, mapping=None, extends=fake_results)


def test_vac_bad_nonbonded(stateA, stateB):
    settings = openmm_afe.AbsoluteSolvationProtocol.default_settings()
    settings.vacuum_forcefield_settings.nonbonded_method = "pme"
    protocol = openmm_afe.AbsoluteSolvationProtocol(settings=settings)

    with pytest.raises(ValueError, match="Only the nocutoff"):
        protocol.validate(stateA=stateA, stateB=stateB, mapping=None)


def test_vac_nvt_error(stateA, stateB):
    settings = openmm_afe.AbsoluteSolvationProtocol.default_settings()
    settings.vacuum_equil_simulation_settings.equilibration_length_nvt = 1 * offunit.nanosecond
    protocol = openmm_afe.AbsoluteSolvationProtocol(settings=settings)

    with pytest.raises(ValueError, match="cannot be run in vacuum"):
        protocol.validate(stateA=stateA, stateB=stateB, mapping=None)


def test_mapping_warning(benzene_modifications, default_settings, stateA, stateB):
    # Pass in a fake mapping and expect a warning it won't be used
    protocol = openmm_afe.AbsoluteSolvationProtocol(settings=default_settings)
    mapping = LigandAtomMapping(
        componentA=benzene_modifications["benzene"],
        componentB=benzene_modifications["benzene"],
        componentA_to_componentB={},
    )

    with pytest.warns(UserWarning, match="mapping was passed"):
        protocol.validate(stateA=stateA, stateB=stateB, mapping=mapping)


@pytest.mark.parametrize("phase", ["solvent", "vacuum"])
def test_high_timestep(phase, stateA, stateB):
    s = AbsoluteSolvationProtocol.default_settings()
    if phase == "solvent":
        s.solvent_forcefield_settings.hydrogen_mass = 1.0
    else:
        s.vacuum_forcefield_settings.hydrogen_mass = 1.0

    protocol = openmm_afe.AbsoluteSolvationProtocol(
        settings=s,
    )

    with pytest.raises(ValueError, match="too large for hydrogen"):
        protocol.validate(stateA=stateA, stateB=stateB, mapping=None)


def test_validate_forcefield_settings(stateA, stateB):
    # make sure the default settings with a different nonbonded method still works
    settings = AbsoluteSolvationProtocol.default_settings()
    assert settings.vacuum_forcefield_settings.nonbonded_method == "nocutoff"

    protocol = AbsoluteSolvationProtocol(settings=settings)
    protocol.validate(stateA=stateA, stateB=stateB, mapping=None)

    # change some other forcefield settings and make sure an error is raised
    settings.solvent_forcefield_settings.small_molecule_forcefield = "gaff-2.11"
    protocol = AbsoluteSolvationProtocol(settings=settings)
    with pytest.raises(
        ValueError,
        match="The following settings differ:\n  small_molecule_forcefield: vacuum=openff-2.2.1, solvent=gaff-2.11",
    ):
        protocol.validate(stateA=stateA, stateB=stateB, mapping=None)


def test_settings_validation():
    # make sure an error is raised if invalid settings are initialized
    with pytest.raises(
        ValueError,
        match="The following settings differ:\n  small_molecule_forcefield: vacuum=openff-2.2.1, solvent=gaff-2.11",
    ):
        _ = AbsoluteSolvationSettings(
            protocol_repeats=3,
            solvent_forcefield_settings=OpenMMSystemGeneratorFFSettings(
                small_molecule_forcefield="gaff-2.11",
            ),
            vacuum_forcefield_settings=OpenMMSystemGeneratorFFSettings(
                nonbonded_method="nocutoff",
            ),
            thermo_settings=ThermoSettings(
                temperature=298.15 * offunit.kelvin,
                pressure=1 * offunit.bar,
            ),
            alchemical_settings=AlchemicalSettings(),
            lambda_settings=LambdaSettings(
                lambda_elec=[
                    0.0,
                    0.25,
                    0.5,
                    0.75,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                lambda_vdw=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.12,
                    0.24,
                    0.36,
                    0.48,
                    0.6,
                    0.7,
                    0.77,
                    0.85,
                    1.0,
                ],
                lambda_restraints=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ),
            partial_charge_settings=OpenFFPartialChargeSettings(),
            solvation_settings=OpenMMSolvationSettings(),
            vacuum_engine_settings=OpenMMEngineSettings(),
            solvent_engine_settings=OpenMMEngineSettings(),
            integrator_settings=IntegratorSettings(),
            solvent_equil_simulation_settings=MDSimulationSettings(
                equilibration_length_nvt=0.1 * offunit.nanosecond,
                equilibration_length=0.2 * offunit.nanosecond,
                production_length=0.5 * offunit.nanosecond,
            ),
            solvent_equil_output_settings=MDOutputSettings(
                equil_nvt_structure="equil_nvt_structure.pdb",
                equil_npt_structure="equil_npt_structure.pdb",
                production_trajectory_filename="production_equil.xtc",
                log_output="equil_simulation.log",
            ),
            solvent_simulation_settings=MultiStateSimulationSettings(
                n_replicas=14,
                equilibration_length=1.0 * offunit.nanosecond,
                production_length=10.0 * offunit.nanosecond,
            ),
            solvent_output_settings=MultiStateOutputSettings(
                output_filename="solvent.nc",
                checkpoint_storage_filename="solvent_checkpoint.nc",
            ),
            vacuum_equil_simulation_settings=MDSimulationSettings(
                equilibration_length_nvt=None,
                equilibration_length=0.2 * offunit.nanosecond,
                production_length=0.5 * offunit.nanosecond,
            ),
            vacuum_equil_output_settings=MDOutputSettings(
                equil_nvt_structure=None,
                equil_npt_structure="equil_structure.pdb",
                production_trajectory_filename="production_equil.xtc",
                log_output="equil_simulation.log",
            ),
            vacuum_simulation_settings=MultiStateSimulationSettings(
                n_replicas=14,
                equilibration_length=0.5 * offunit.nanosecond,
                production_length=2.0 * offunit.nanosecond,
            ),
            vacuum_output_settings=MultiStateOutputSettings(
                output_filename="vacuum.nc",
                checkpoint_storage_filename="vacuum_checkpoint.nc",
            ),
        )
