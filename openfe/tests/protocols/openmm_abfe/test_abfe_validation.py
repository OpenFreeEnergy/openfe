# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import pytest
from gufe import ProtocolDAGResult, LigandAtomMapping
from openfe import ChemicalSystem, SolventComponent
from openfe.protocols.openmm_afe import (
    AbsoluteBindingProtocol,
)


@pytest.fixture()
def default_settings():
    return AbsoluteBindingProtocol.default_settings()


@pytest.mark.parametrize(
    "val",
    [
        {"elec": [0.0, 1.0], "vdw": [1.0, 1.0], "restraints": [0.0, 0.0]},
    ],
)
def test_validate_complex_lambda_schedule_naked_charge(val, default_settings):
    errmsg = (
        "There are states along this lambda schedule "
        "where there are atoms with charges but no LJ "
        f"interactions: lambda 0: "
        f"elec {val['elec'][0]} vdW {val['vdw'][0]}"
    )
    default_settings.complex_lambda_settings.lambda_elec = val["elec"]
    default_settings.complex_lambda_settings.lambda_vdw = val["vdw"]
    default_settings.complex_lambda_settings.lambda_restraints = val["restraints"]
    default_settings.complex_simulation_settings.n_replicas = 2
    with pytest.raises(ValueError, match=errmsg):
        AbsoluteBindingProtocol._validate_lambda_schedule(
            default_settings.complex_lambda_settings,
            default_settings.complex_simulation_settings,
        )


@pytest.mark.parametrize(
    "val",
    [
        {"elec": [0.0, 1.0], "vdw": [1.0, 1.0], "restraints": [0.0, 0.0]},
    ],
)
def test_validate_solvent_lambda_schedule_naked_charge(val, default_settings):
    errmsg = (
        "There are states along this lambda schedule "
        "where there are atoms with charges but no LJ "
        f"interactions: lambda 0: "
        f"elec {val['elec'][0]} vdW {val['vdw'][0]}"
    )
    default_settings.solvent_lambda_settings.lambda_elec = val["elec"]
    default_settings.solvent_lambda_settings.lambda_vdw = val["vdw"]
    default_settings.solvent_lambda_settings.lambda_restraints = val["restraints"]
    default_settings.solvent_simulation_settings.n_replicas = 2
    with pytest.raises(ValueError, match=errmsg):
        AbsoluteBindingProtocol._validate_lambda_schedule(
            default_settings.solvent_lambda_settings,
            default_settings.solvent_simulation_settings,
        )


@pytest.mark.parametrize(
    "val",
    [
        {"elec": [1.0, 1.0], "vdw": [0.0, 1.0], "restraints": [0.0, 1.0]},
    ],
)
def test_validate_lambda_schedule_nreplicas(val, default_settings):
    # Only testing complex since it'll be the same for solvent
    default_settings.complex_lambda_settings.lambda_elec = val["elec"]
    default_settings.complex_lambda_settings.lambda_vdw = val["vdw"]
    default_settings.complex_lambda_settings.lambda_restraints = val["restraints"]
    n_replicas = 3
    default_settings.complex_simulation_settings.n_replicas = n_replicas
    errmsg = (
        f"Number of replicas {n_replicas} does not equal the"
        f" number of lambda windows {len(val['vdw'])}"
    )
    with pytest.raises(ValueError, match=errmsg):
        AbsoluteBindingProtocol._validate_lambda_schedule(
            default_settings.complex_lambda_settings,
            default_settings.complex_simulation_settings,
        )


@pytest.mark.parametrize(
    "val",
    [
        {"elec": [1.0, 1.0, 1.0], "vdw": [0.0, 1.0], "restraints": [1.0, 1.0]},
    ],
)
def test_validate_lambda_schedule_nwindows(val, default_settings):
    # Only testing complex since it'll be the same for solvent
    default_settings.complex_lambda_settings.lambda_elec = val["elec"]
    default_settings.complex_lambda_settings.lambda_vdw = val["vdw"]
    default_settings.complex_lambda_settings.lambda_restraints = val["restraints"]
    n_replicas = 3
    default_settings.complex_simulation_settings.n_replicas = n_replicas
    errmsg = (
        "Components elec, vdw, and restraints must have equal amount"
        f" of lambda windows. Got {len(val['elec'])} elec lambda"
        f" windows, {len(val['vdw'])} vdw lambda windows, and"
        f"{len(val['restraints'])} restraints lambda windows."
    )
    with pytest.raises(ValueError, match=errmsg):
        AbsoluteBindingProtocol._validate_lambda_schedule(
            default_settings.complex_lambda_settings,
            default_settings.complex_simulation_settings,
        )


def test_validate_no_protcomp(
    benzene_modifications,
):
    stateA = ChemicalSystem(
        {"benzene": benzene_modifications["benzene"], "solvent": SolventComponent()}
    )

    stateB = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "solvent": SolventComponent(),
        }
    )

    errmsg = "No ProteinComponent found"
    with pytest.raises(ValueError, match=errmsg):
        AbsoluteBindingProtocol._validate_endstates(stateA, stateB)


def test_validate_endstates_nosolvcomp_stateA(
    benzene_modifications, T4_protein_component
):
    stateA = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "protein": T4_protein_component,
        }
    )

    stateB = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "protein": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )

    with pytest.raises(ValueError, match="No SolventComponent found"):
        AbsoluteBindingProtocol._validate_endstates(stateA, stateB)


def test_validate_endstates_nosolvcomp_stateB(
    benzene_modifications, T4_protein_component
):
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
            "protein": T4_protein_component,
        }
    )

    with pytest.raises(ValueError, match="No SolventComponent"):
        AbsoluteBindingProtocol._validate_endstates(stateA, stateB)


def test_validate_endstates_multiple_uniqueA(
    benzene_modifications, T4_protein_component
):
    stateA = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "toluene": benzene_modifications["toluene"],
            "protein": T4_protein_component,
            "solvlent": SolventComponent(),
        }
    )

    stateB = ChemicalSystem(
        {
            "protein": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )

    with pytest.raises(ValueError, match="Only one alchemical species"):
        AbsoluteBindingProtocol._validate_endstates(stateA, stateB)


def test_validate_solvent_endstates_solvent_dissapearing(
    benzene_modifications, T4_protein_component
):
    stateA = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "protein": T4_protein_component,
            "foo": SolventComponent(smiles="C"),
            "solvent": SolventComponent(),
        }
    )

    stateB = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "protein": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )

    errmsg = "Only dissapearing small molecule components"
    with pytest.raises(ValueError, match=errmsg):
        AbsoluteBindingProtocol._validate_endstates(stateA, stateB)


def test_validate_endstates_unique_stateB(benzene_modifications, T4_protein_component):
    stateA = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "protein": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )

    stateB = ChemicalSystem(
        {
            "toluene": benzene_modifications["toluene"],
            "protein": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )

    with pytest.raises(ValueError, match="appearing in state B"):
        AbsoluteBindingProtocol._validate_endstates(stateA, stateB)


def test_charged_endstate(charged_benzene_modifications, T4_protein_component):

    stateA = ChemicalSystem(
        {
            "benzene": charged_benzene_modifications["benzoic_acid"],
            "protein": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )

    stateB = ChemicalSystem(
        {
            "protein": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )

    errmsg = "Charged alchemical molecules are not currently supported"
    with pytest.raises(ValueError, match=errmsg):
        AbsoluteBindingProtocol._validate_endstates(stateA, stateB)


def test_validate_fail_extends(
    benzene_modifications, T4_protein_component, default_settings
):
    stateA = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "protein": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )

    stateB = ChemicalSystem(
        {
            "protein": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )

    prot = AbsoluteBindingProtocol(settings=default_settings)
    fake_result = ProtocolDAGResult(
        protocol_units=[], protocol_unit_results=[], transformation_key="foo"
    )

    with pytest.raises(ValueError, match="extend simulations"):
        prot.validate(stateA=stateA, stateB=stateB, mapping=None, extends=fake_result)


def test_high_timestep(benzene_modifications, T4_protein_component):
    s = AbsoluteBindingProtocol.default_settings()
    s.forcefield_settings.hydrogen_mass = 1.0

    stateA = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "protein": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )

    stateB = ChemicalSystem(
        {
            "protein": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )

    protocol = AbsoluteBindingProtocol(settings=s)

    with pytest.raises(ValueError, match="too large for hydrogen"):
        protocol.validate(stateA=stateA, stateB=stateB, mapping=None)


def test_validate_warnings(
    benzene_modifications, T4_protein_component, default_settings
):
    stateA = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "protein": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )

    stateB = ChemicalSystem(
        {
            "protein": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )

    mapping = LigandAtomMapping(
        componentA=benzene_modifications["benzene"],
        componentB=benzene_modifications["benzene"],
        componentA_to_componentB={},
    )

    # one the complex restraints
    default_settings.complex_lambda_settings.lambda_restraints = [
        0 for _ in default_settings.complex_lambda_settings.lambda_restraints
    ]

    # add a restraint in solvent
    default_settings.solvent_lambda_settings.lambda_restraints[-1] = 0.9

    prot = AbsoluteBindingProtocol(settings=default_settings)

    with pytest.warns(UserWarning) as record:
        prot.validate(stateA=stateA, stateB=stateB, mapping=mapping, extends=None)

    assert len(record) == 3
    assert "mapping was passed" in str(record[0].message)
    assert "being applied in the complex" in str(record[1].message)
    assert "add restraints in the solvent" in str(record[2].message)
