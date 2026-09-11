# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Test the restraint settings.
"""

import pytest
from openff.units import unit

from openfe.protocols.restraint_utils.settings import (
    BoreschRestraintSettings,
    DistanceRestraintSettings,
    FlatBottomRestraintSettings,
)


def test_distance_restraint_settings_default():
    """
    Basic settings regression test
    """
    settings = DistanceRestraintSettings(
        spring_constant=10 * unit.kilojoule_per_mole / unit.nm**2,
    )
    assert settings.central_atoms_only is False
    assert isinstance(settings, DistanceRestraintSettings)


def test_distance_restraint_negative_idxs():
    """
    Check that an error is raised if you have negative
    atom indices in host atoms.
    """
    with pytest.raises(ValueError, match="negative indices passed"):
        _ = DistanceRestraintSettings(
            spring_constant=10 * unit.kilojoule_per_mole / unit.nm**2,
            host_atoms=[-1, 0, 2],
            guest_atoms=[0, 1, 2],
        )


def test_flatbottom_restraint_settings_default():
    """
    Basic settings regression test
    """
    settings = FlatBottomRestraintSettings(
        spring_constant=10 * unit.kilojoule_per_mole / unit.nm**2,
        well_radius=1 * unit.nanometer,
    )
    assert isinstance(settings, FlatBottomRestraintSettings)


def test_flatbottom_restraint_negative_well():
    """
    Check that an error is raised if you have a negative
    well radius.
    """
    with pytest.raises(ValueError, match="well radius cannot be negative"):
        _ = FlatBottomRestraintSettings(
            spring_constant=10 * unit.kilojoule_per_mole / unit.nm**2,
            well_radius=-1 * unit.nm,
        )


def test_boresch_restraint_settings_default():
    """
    Basic settings regression test
    """
    settings = BoreschRestraintSettings(
        K_r=10 * unit.kilojoule_per_mole / unit.nm**2,
        K_thetaA=10 * unit.kilojoule_per_mole / unit.radians**2,
        K_thetaB=10 * unit.kilojoule_per_mole / unit.radians**2,
        K_phiA=10 * unit.kilojoule_per_mole / unit.radians**2,
        K_phiB=10 * unit.kilojoule_per_mole / unit.radians**2,
        K_phiC=10 * unit.kilojoule_per_mole / unit.radians**2,
    )
    assert isinstance(settings, BoreschRestraintSettings)


@pytest.mark.parametrize("parameter", ["host_restraint_ids", "guest_restraint_ids"])
def test_boresch_restraints_restraint_negative_ids(parameter):
    setting = BoreschRestraintSettings()

    errmsg = "``guest_atoms`` and ``host_atoms`` cannot have negative indices."
    with pytest.raises(ValueError, match=errmsg):
        setattr(setting, parameter, [1, 2, -3])


@pytest.mark.parametrize("parameter", ["host_restraint_ids", "guest_restraint_ids"])
def test_boresch_restraints_too_many_ids(parameter):
    setting = BoreschRestraintSettings()

    errmsg = "Tuple should have at most 3 items after validation, not 4"
    with pytest.raises(ValueError, match=errmsg):
        setattr(setting, parameter, [1, 2, 3, 4])


def test_boresch_restraint_partially_defined_ids():
    setting = BoreschRestraintSettings()

    with pytest.raises(ValueError, match="must both either be defined or undefined"):
        setting.host_restraint_ids = [1, 2, 3]
