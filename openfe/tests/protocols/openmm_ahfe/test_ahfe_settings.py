# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import pytest

from openfe.protocols import openmm_afe
from openfe.protocols.openmm_afe import (
    AbsoluteSolvationProtocol,
)


@pytest.fixture()
def default_settings():
    return AbsoluteSolvationProtocol.default_settings()


def test_create_default_settings():
    settings = AbsoluteSolvationProtocol.default_settings()
    assert settings


def test_invalid_protocol_repeats():
    settings = AbsoluteSolvationProtocol.default_settings()
    with pytest.raises(ValueError, match="must be a positive value"):
        settings.protocol_repeats = -1


@pytest.mark.parametrize(
    "val",
    [
        {"elec": [0.0, -1], "vdw": [0.0, 1.0], "restraints": [0.0, 1.0]},
        {"elec": [0.0, 1.5], "vdw": [0.0, 1.5], "restraints": [-0.1, 1.0]},
    ],
)
def test_incorrect_window_settings(val, default_settings):
    errmsg = "Lambda windows must be between 0 and 1."
    lambda_settings = default_settings.lambda_settings
    with pytest.raises(ValueError, match=errmsg):
        lambda_settings.lambda_elec = val["elec"]
        lambda_settings.lambda_vdw = val["vdw"]
        lambda_settings.lambda_restraints = val["restraints"]


@pytest.mark.parametrize(
    "val",
    [
        {"elec": [0.0, 0.1, 0.0], "vdw": [0.0, 1.0, 1.0], "restraints": [0.0, 1.0, 1.0]},
    ],
)
def test_monotonic_lambda_windows(val, default_settings):
    errmsg = "The lambda schedule is not monotonically increasing"
    lambda_settings = default_settings.lambda_settings

    with pytest.raises(ValueError, match=errmsg):
        lambda_settings.lambda_elec = val["elec"]
        lambda_settings.lambda_vdw = val["vdw"]
        lambda_settings.lambda_restraints = val["restraints"]
