# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import pytest

from openfe.protocols import openmm_afe
from openfe.protocols.openmm_afe import (
    AbsoluteSolvationProtocol,
)
from openfe.protocols.openmm_afe.equil_afe_settings import (
    AbsoluteSolvationSettings,
    LambdaSettings,
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
def test_incorrect_window_settings(val):
    errmsg = "Lambda windows must be between 0 and 1."
    with pytest.raises(ValueError, match=errmsg):
        _ = LambdaSettings(
            lambda_elec=val["elec"],
            lambda_vdw=val["vdw"],
            lambda_restraints=val["restraints"],
        )


@pytest.mark.parametrize(
    "val",
    [
        {"elec": [0.0, 0.1, 0.0], "vdw": [0.0, 1.0, 1.0], "restraints": [0.0, 1.0, 1.0]},
    ],
)
def test_monotonic_lambda_windows(val):
    errmsg = "The lambda schedule is not monotonically increasing"

    with pytest.raises(ValueError, match=errmsg):
        _ = LambdaSettings(
            lambda_elec=val["elec"],
            lambda_vdw=val["vdw"],
            lambda_restraints=val["restraints"],
        )


def test_legacy_lambda_settings(default_settings):
    """
    Check that we can load ``lambda_settings`` from pre-openfe v1.13 settings.

    TODO
    ----
    Remove in openfe v1.14. See Issue #2247
    """
    legacy = dict(default_settings)
    legacy.pop("solvent_lambda_settings")
    legacy.pop("vacuum_lambda_settings")
    legacy["lambda_settings"] = LambdaSettings(
        lambda_elec=[0.0, 0.5, 1.0, 1.0],
        lambda_vdw=[0.0, 0.0, 0.5, 1.0],
        lambda_restraints=[0.0, 0.0, 0.0, 0.0],
    )

    with pytest.warns(FutureWarning, match="lambda_settings"):
        settings = AbsoluteSolvationSettings(**legacy)

    for lambda_settings in (settings.solvent_lambda_settings, settings.vacuum_lambda_settings):
        assert lambda_settings.lambda_elec == [0.0, 0.5, 1.0, 1.0]
        assert lambda_settings.lambda_vdw == [0.0, 0.0, 0.5, 1.0]
        assert lambda_settings.lambda_restraints == [0.0, 0.0, 0.0, 0.0]
    # solvent and vacuum must not share the same object
    assert settings.solvent_lambda_settings is not settings.vacuum_lambda_settings


def test_legacy_lambda_settings_mixed(default_settings):
    # Mixing the legacy field with the new ones is not allowed
    mixed = dict(default_settings)
    mixed["lambda_settings"] = LambdaSettings()

    with pytest.raises(ValueError, match="lambda_settings"):
        AbsoluteSolvationSettings(**mixed)
