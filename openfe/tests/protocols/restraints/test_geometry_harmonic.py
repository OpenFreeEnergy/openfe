# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import pytest

from openfe.protocols.restraint_utils.geometry.harmonic import (
    DistanceRestraintGeometry
)


def test_hostguest_geometry():
    """
    A very basic will it build test.
    """
    geom = DistanceRestraintGeometry(guest_atoms=[1, 2, 3], host_atoms=[4])

    assert isinstance(geom, DistanceRestraintGeometry)


def test_get_distance_restraint():
    """
    Check that you get a distance restraint.
    """
