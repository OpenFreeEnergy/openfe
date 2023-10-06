# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import pytest
from openff.units import unit as offunit
from openfe.protocols import openmm_afe


@pytest.fixture()
def default_settings():
    return openmm_afe.AbsoluteSolvationProtocol.default_settings()


def test_create_default_settings():
    settings = openmm_afe.AbsoluteSolvationProtocol.default_settings()
    assert settings


@pytest.mark.parametrize('val', [
    {'elec': 0, 'vdw': 5},
    {'elec': -2, 'vdw': 5},
    {'elec': 5, 'vdw': -2},
    {'elec': 5, 'vdw': 0},
])
def test_incorrect_window_settings(val, default_settings):
    errmsg = "lambda steps must be positive"
    alchem_settings = default_settings.alchemical_settings
    with pytest.raises(ValueError, match=errmsg):
        alchem_settings.lambda_elec_windows = val['elec']
        alchem_settings.lambda_vdw_windows = val['vdw']

