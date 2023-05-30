# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import pytest
from openff.units import unit
import openfe
from openfe.protocols.openmm_utils import (
        settings_validation, system_validation,
)


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


def test_duplicate_chemical_components(benzene_modifications):
    stateA = openfe.ChemicalSystem({'A': benzene_modifications['toluene'],
                                    'B': benzene_modifications['toluene'],})
    stateB = openfe.ChemicalSystem({'A': benzene_modifications['toluene']})

    errmsg = "state A components B:"

    with pytest.raises(ValueError, match=errmsg):
        system_validation.get_alchemical_components(stateA, stateB)
