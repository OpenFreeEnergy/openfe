# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Test the restraint settings.
"""
import pytest
from openfe.protocols.restraint_utils.settings import (
    BoreschRestraintSettings,
    DistanceRestraintSettings,
    FlatBottomRestraintSettings,
)
from openff.units import unit


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


# TODO: re-enable this Issue #1556
# def test_boresch_restraint_negative_idxs():
#     """
#     Check that the positive_idxs_list validator is
#     working as expected.
#     """
#     with pytest.raises(ValueError, match="negative indices"):
#         _ = BoreschRestraintSettings(
#             K_r=10 * unit.kilojoule_per_mole / unit.nm**2,
#             K_thetaA=10 * unit.kilojoule_per_mole / unit.radians**2,
#             K_thetaB=10 * unit.kilojoule_per_mole / unit.radians**2,
#             phi_A0=10 * unit.kilojoule_per_mole / unit.radians**2,
#             phi_B0=10 * unit.kilojoule_per_mole / unit.radians**2,
#             phi_C0=10 * unit.kilojoule_per_mole / unit.radians**2,
#             host_atoms=[-1, 0],
#             guest_atoms=[0, 1],
#         )
