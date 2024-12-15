# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import pytest

from openfe.protocols.restraint_utils.geometry.base import (
    HostGuestRestraintGeometry
)


def test_hostguest_geometry():
    """
    A very basic will it build test.
    """
    geom = HostGuestRestraintGeometry(guest_atoms=[1, 2, 3], host_atoms=[4])

    assert isinstance(geom, HostGuestRestraintGeometry)


def test_hostguest_positiveidxs_validator():
    """
    Check that the validator is working as intended.
    """
    with pytest.raises(ValueError, match="negative indices passed"):
        geom = HostGuestRestraintGeometry(guest_atoms=[-1, 1], host_atoms=[0])
