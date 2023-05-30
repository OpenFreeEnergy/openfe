# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from openff.units import unit
from openfe.protocols.openmm_utils import settings_validation
import pytest


def test_validate_timestep():
    with pytest.raises(ValueError, match="too large for hydrogen mass"):
        settings_validation.validate_timestep(2.0, 4.0 * unit.femtoseconds)


@pytest.mark.parametrize('nametype, timelengths', [
    ['Equilibration', [1.003 * unit.picoseconds, 1 * unit.picoseconds]],
    ['Production', [1 * unit.picoseconds, 1.003 * unit.picoseconds]],
])
def test_get_simsteps_indivisible_simtime(nametype, timelengths):
    errmsg = f"{nametype} time not divisible by timestep"
    with pytest.raises(ValueError, match=errmsg):
        settings_validation.get_simsteps(
                timelengths[0],
                timelengths[1],
                2 * unit.femtoseconds,
                100)

@pytest.mark.parametrize('nametype, timelengths', [
    ['Equilibration', [1 * unit.picoseconds, 10 * unit.picoseconds]],
    ['Production', [10 * unit.picoseconds,  1 * unit.picoseconds]],
])
def test_mc_indivisible(nametype, timelengths):
    errmsg = f"{nametype} time 1.0 ps should contain"
    with pytest.raises(ValueError, match=errmsg):
        settings_validation.get_simsteps(
                timelengths[0], timelengths[1],
                2 * unit.femtoseconds, 1000)
