# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import pytest
from openff.units import unit as offunit
from gufe import ProtocolDAGResult, LigandAtomMapping
from openfe import ChemicalSystem, SolventComponent
from openfe.protocols import openmm_afe
from openfe.protocols.openmm_afe import (
    AbsoluteSolvationSolventUnit,
    AbsoluteSolvationVacuumUnit,
    AbsoluteSolvationProtocol,
)

from openfe.protocols.openmm_utils import system_validation
from openfe.protocols.openmm_utils.charge_generation import (
    HAS_NAGL,
    HAS_OPENEYE,
    HAS_ESPALOMA_CHARGE,
)


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


def test_validate_endstates_nosolvcomp_stateA(
    benzene_modifications, T4_protein_component
):
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


def test_validate_endstates_nosolvcomp_stateB(
    benzene_modifications, T4_protein_component
):
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

    with pytest.raises(
        ValueError, match="No SolventComponent found in stateA and/or stateB"
    ):
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

    errmsg = "Only dissapearing SmallMoleculeComponents"
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
        protocol.validate(
            stateA=stateA, stateB=stateB, mapping=None, extends=fake_results
        )


def test_vac_bad_nonbonded(stateA, stateB):
    settings = openmm_afe.AbsoluteSolvationProtocol.default_settings()
    settings.vacuum_forcefield_settings.nonbonded_method = "pme"
    protocol = openmm_afe.AbsoluteSolvationProtocol(settings=settings)

    with pytest.raises(ValueError, match="Only the nocutoff"):
        protocol.validate(stateA=stateA, stateB=stateB, mapping=None)


def test_vac_nvt_error(stateA, stateB):
    settings = openmm_afe.AbsoluteSolvationProtocol.default_settings()
    settings.vacuum_equil_simulation_settings.equilibration_length_nvt = (
        1 * offunit.nanosecond
    )
    protocol = openmm_afe.AbsoluteSolvationProtocol(settings=settings)

    with pytest.raises(ValueError, match="cannot be run in vacuum"):
        protocol.validate(stateA=stateA, stateB=stateB, mapping=None)


def test_mapping_warning(benzene_modifications, default_settings, stateA, stateB):
    # Pass in a fake mapping and expect a warning it won't be used
    protocol = openmm_afe.AbsoluteSolvationProtocol(settings=default_settings)
    mapping = LigandAtomMapping(componentA=benzene_modifications['benzene'], componentB=benzene_modifications['benzene'], componentA_to_componentB={})

    with pytest.warns(UserWarning, match='mapping was passed'):
        protocol.validate(stateA=stateA, stateB=stateB, mapping=mapping)
