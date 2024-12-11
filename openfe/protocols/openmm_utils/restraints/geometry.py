# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Restraint Geometry classes

TODO
----
* Add relevant duecredit entries.
"""
import abc
from pydantic.v1 import BaseModel, validator

from openff.units import unit
import MDAnalysis as mda
from MDAnalysis.lib.distances import calc_bonds


class BaseRestraintGeometry(BaseModel, abc.ABC):
    class Config:
        arbitrary_types_allowed = True


class HostGuestRestraintGeometry(BaseRestraintGeometry):
    """
    An ordered list of guest atoms to restrain.

    Note
    ----
    The order matters! It will be used to define the underlying
    force.
    """

    guest_atoms: list[int]
    """
    An ordered list of host atoms to restrain.

    Note
    ----
    The order matters! It will be used to define the underlying
    force.
    """
    host_atoms: list[int]

    @validator("guest_atoms", "host_atoms")
    def positive_idxs(cls, v):
        if any([i < 0 for i in v]):
            errmsg = "negative indices passed"
            raise ValueError(errmsg)
        return v


class CentroidDistanceMixin:
    def get_distance(self, topology, coordinates) -> unit.Quantity:
        u = mda.Universe(topology, coordinates)
        ag1 = u.atoms[self.host_atoms]
        ag2 = u.atoms[self.guest_atoms]
        bond = calc_bonds(
            ag1.center_of_mass(), ag2.center_of_mass(), u.atoms.dimensions
        )
        # convert to float so we avoid having a np.float64
        return float(bond) * unit.angstrom


def _check_single_atoms(value):
    if len(value) != 1:
        errmsg = (
            "Host and guest atom lists must only include a single atom, "
            f"got {len(value)} atoms."
        )
        raise ValueError(errmsg)
    return value


class BondDistanceMixin:
    def get_distance(self, topology, coordinates) -> unit.Quantity:
        u = mda.Universe(topology, coordinates)
        at1 = u.atoms[self.host_atoms[0]]
        at2 = u.atoms[self.guest_atoms[0]]
        bond = calc_bonds(at1.position, at2.position, u.atoms.dimensions)
        # convert to float so we avoid having a np.float64 value
        return float(bond) * unit.angstrom


class CentroidDistanceRestraintGeometry(HostGuestRestraintGeometry, CentroidDistanceMixin):
    pass


class BondDistanceRestraintGeoemtry(HostGuestRestraintGeometry, BondDistanceMixin):
    _check_host_atoms: classmethod = validator("host_atoms", allow_reuse=True)(_check_single_atoms)
    _check_guest_atoms: classmethod = validator("guest_atoms", allow_reuse=True)(_check_single_atoms)
