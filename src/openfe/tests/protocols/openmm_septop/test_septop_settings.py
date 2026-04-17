# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import pytest


from openfe import ChemicalSystem, SolventComponent
from openfe.protocols.openmm_septop import (
    SepTopProtocol,
)
from openfe.protocols.openmm_septop.equil_septop_settings import SepTopSettings


@pytest.fixture()
def protocol_dry_settings():
    # a set of settings for dry run tests
    s = SepTopProtocol.default_settings()
    s.engine_settings.compute_platform = None
    s.protocol_repeats = 1
    return s


@pytest.fixture()
def default_settings():
    s = SepTopProtocol.default_settings()
    return s


def test_create_default_settings():
    settings = SepTopProtocol.default_settings()
    assert settings


@pytest.mark.parametrize(
    "val",
    [
        {"elec": [0.0, -1], "vdw": [0.0, 1.0], "restraints": [0.0, 1.0]},
        {"elec": [0.0, 1], "vdw": [0.0, 1.5], "restraints": [0.0, 1.0]},
        {"elec": [0.0, 1], "vdw": [0.0, 1], "restraints": [-0.1, 1.0]},
    ],
)
def test_incorrect_window_settings(val, default_settings):
    errmsg = "Lambda windows must be between 0 and 1."
    lambda_settings = default_settings.complex_lambda_settings
    with pytest.raises(ValueError, match=errmsg):
        lambda_settings.lambda_elec_A = val["elec"]
        lambda_settings.lambda_vdw_A = val["vdw"]
        lambda_settings.lambda_restraints_A = val["restraints"]


@pytest.mark.parametrize(
    "val",
    [
        {
            "elec": [0.0, 0.1, 0.0],
            "vdw": [0.0, 1.0, 1.0],
            "restraints": [0.0, 1.0, 1.0],
        },
        {
            "elec": [0.0, 0.0, 0.0],
            "vdw": [0.0, 1.0, 0.0],
            "restraints": [0.0, 1.0, 1.0],
        },
        {
            "elec": [0.0, 0.0, 0.0],
            "vdw": [0.0, 1.0, 1.0],
            "restraints": [0.0, 1.0, 0.0],
        },
    ],
)
def test_monotonic_lambda_windows_A(val, default_settings):
    errmsg = "The lambda schedule for ligand A"
    lambda_settings = default_settings.complex_lambda_settings

    with pytest.raises(ValueError, match=errmsg):
        lambda_settings.lambda_elec_A = val["elec"]
        lambda_settings.lambda_vdw_A = val["vdw"]
        lambda_settings.lambda_restraints_A = val["restraints"]


@pytest.mark.parametrize(
    "val",
    [
        {
            "elec": [1.0, 0.1, 1.0],
            "vdw": [1.0, 1.0, 1.0],
            "restraints": [1.0, 1.0, 1.0],
        },
        {
            "elec": [1.0, 1.0, 1.0],
            "vdw": [1.0, 0.0, 1.0],
            "restraints": [1.0, 1.0, 1.0],
        },
        {
            "elec": [1.0, 1.0, 1.0],
            "vdw": [1.0, 1.0, 1.0],
            "restraints": [1.0, 0.0, 1.0],
        },
    ],
)
def test_monotonic_lambda_windows_B(val, default_settings):
    errmsg = "The lambda schedule for ligand B"
    lambda_settings = default_settings.complex_lambda_settings

    with pytest.raises(ValueError, match=errmsg):
        lambda_settings.lambda_elec_B = val["elec"]
        lambda_settings.lambda_vdw_B = val["vdw"]
        lambda_settings.lambda_restraints_B = val["restraints"]


def test_output_induces_not_all(default_settings):
    errmsg = "Equilibration simulations need to output the full system"

    with pytest.raises(ValueError, match=errmsg):
        default_settings.complex_equil_output_settings.output_indices = "no water"


def test_adaptive_settings_no_protein_membrane(toluene_complex_system, default_settings):
    settings = SepTopProtocol._adaptive_settings(
        toluene_complex_system, toluene_complex_system, default_settings
    )

    assert isinstance(settings, SepTopSettings)
    # Should use default barostat since no ProteinMembraneComponent
    assert settings.complex_integrator_settings.barostat == "MonteCarloBarostat"


def test_adaptive_settings_with_protein_membrane(a2a_protein_membrane_component, a2a_ligands):
    stateA = ChemicalSystem(
        {
            "ligandA": a2a_ligands[0],
            "protein": a2a_protein_membrane_component,
            "solvent": SolventComponent(),
        }
    )

    settings = SepTopProtocol._adaptive_settings(stateA, stateA)
    assert isinstance(settings, SepTopSettings)
    # Barostat should have been updated
    assert settings.complex_integrator_settings.barostat == "MonteCarloMembraneBarostat"
