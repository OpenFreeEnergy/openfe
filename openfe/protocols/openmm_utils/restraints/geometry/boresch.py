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
from MDAnalysis.lib.distances import calc_bonds, calc_angles

from .base import HostGuestRestraintGeometry


class BoreschRestraintGeometry(HostGuestRestraintGeometry):
    """
    A class that defines the restraint geometry for a Boresch restraint.

    The restraint is defined by the following:

      H0                         G2
       -                        -
        -                      -
         H1 - - H2 -- G0 - - G1

    Where HX represents the X index of ``host_atoms`` and GX
    the X index of ``guest_atoms``.
    """
    def get_bond_distance(self, topology, coordinates) -> unit.Quantity:
        u = mda.Universe(topology, coordinates)
        at1 = u.atoms[host_atoms[2]]
        at2 = u.atoms[guest_atoms[0]]
        bond = calc_bonds(at1.position, at2.position, u.atoms.dimensions)
        # convert to float so we avoid having a np.float64
        return float(bond) * unit.angstrom

    def get_angles(self, topology, coordinates) -> unit.Quantity:
        u = mda.Universe(topology, coordinates)
        at1 = u.atoms[host_atoms[1]]
        at2 = u.atoms[host_atoms[2]]
        at3 = u.atoms[guest_atoms[0]]
        at4 = u.atoms[guest_atoms[1]]

        angleA = calc_angles(at1.position, at2.position, at3.position, u.atoms.dimensions)
        angleB = calc_angles(at2.position, at3.position, at4.position, u.atoms.dimensions)
        return angleA, angleB

    def get_dihedrals(self, topology, coordinates) -> unit.Quantity:
        u = mda.Universe(topology, coordinates)
        at1 = u.atoms[host_atoms[0]]
        at2 = u.atoms[host_atoms[1]]
        at3 = u.atoms[host_atoms[2]]
        at4 = u.atoms[guest_atoms[0]]
        at5 = u.atoms[guest_atoms[1]]
        at6 = u.atoms[guest_atoms[2]]

        dihA = calc_dihedrals(at1.position, at2.position, at3.position, at4.position, u.atoms.dimensions)
        dihB = calc_dihedrals(at2.position, at3.position, at4.position, at5.position, u.atoms.dimensions)
        dihC = calc_dihedrals(at3.position, at4.position, at5.position, at6.position, u.atoms.dimensions)

        return dihA, dihB, dihC
