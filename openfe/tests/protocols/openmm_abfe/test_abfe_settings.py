# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import pytest
from openfe.protocols.openmm_afe import (
    AbsoluteBindingProtocol,
)


@pytest.fixture()
def default_settings():
    return AbsoluteBindingProtocol.default_settings()


def test_create_default_settings():
    settings = AbsoluteBindingProtocol.default_settings()
    assert settings


def test_negative_repeats_settings(default_settings):
    with pytest.raises(ValueError, match="protocol_repeats must be a positive"):
        default_settings.protocol_repeats = -1


@pytest.mark.parametrize(
    "val",
    [
        {"elec": [0.0, -1], "vdw": [0.0, 1.0], "restraints": [0.0, 1.0]},
        {"elec": [0.0, 1.5], "vdw": [0.0, 1.5], "restraints": [-0.1, 1.0]},
    ],
)
def test_incorrect_window_settings(val, default_settings):
    errmsg = "Lambda windows must be between 0 and 1."
    lambda_settings = default_settings.complex_lambda_settings
    with pytest.raises(ValueError, match=errmsg):
        lambda_settings.lambda_elec = val["elec"]
        lambda_settings.lambda_vdw = val["vdw"]
        lambda_settings.lambda_restraints = val["restraints"]


@pytest.mark.parametrize(
    "val",
    [
        {
            "elec": [0.0, 0.1, 0.0],
            "vdw": [0.0, 1.0, 1.0],
            "restraints": [0.0, 1.0, 1.0],
        },
        {
            "elec": [0.0, 0.1, 0.2],
            "vdw": [0.0, 1.0, 0.2],
            "restraints": [0.0, 1.0, 1.0],
        },
        {
            "elec": [0.0, 0.1, 0.2],
            "vdw": [0.0, 1.0, 1.0],
            "restraints": [0.0, 1.0, 0.0],
        },
    ],
)
def test_monotonic_lambda_windows(val, default_settings):
    errmsg = "The lambda schedule is not monotonic."
    lambda_settings = default_settings.complex_lambda_settings

    with pytest.raises(ValueError, match=errmsg):
        lambda_settings.lambda_elec = val["elec"]
        lambda_settings.lambda_vdw = val["vdw"]
        lambda_settings.lambda_restraints = val["restraints"]


def test_equil_not_all_complex(default_settings):
    with pytest.raises(ValueError, match="output_indices must be all"):
        default_settings.complex_equil_output_settings.output_indices = "not water"


def test_equil_not_all_solvent(default_settings):
    with pytest.raises(ValueError, match="output_indices must be all"):
        default_settings.solvent_equil_output_settings.output_indices = "not water"
