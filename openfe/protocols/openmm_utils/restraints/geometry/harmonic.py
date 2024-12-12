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
from rdkit import Chem

from .base import HostGuestRestraintGeometry
from .utils import _get_central_atom_idx


class DistanceRestraintGeometry(HostGuestRestraintGeometry):
    """
    A geometry class for a distance restraint between two groups of atoms.
    """

    def get_distance(self, topology, coordinates) -> unit.Quantity:
        u = mda.Universe(topology, coordinates)
        ag1 = u.atoms[self.host_atoms]
        ag2 = u.atoms[self.guest_atoms]
        bond = calc_bonds(
            ag1.center_of_mass(), ag2.center_of_mass(), box=u.atoms.dimensions
        )
        # convert to float so we avoid having a np.float64
        return float(bond) * unit.angstrom


def _get_selection(universe, atom_list, selection):
    if atom_list is None:
        if selection is None:
            raise ValueError(
                "one of either the atom lists or selections must be defined"
            )

        ag = universe.select_atoms(selection)
    else:
        ag = universe.atoms[atom_list]

    return ag


def get_distance_restraint(
    topology: Union[str, openmm.app.Topology],
    trajectory: pathlib.Path,
    topology_format: Optional[str] = None,
    host_atoms: Optional[list[int]] = None,
    guest_atoms: Optional[list[int]] = None,
    host_selection: Optional[str] = None,
    guest_selection: Optional[str] = None,
) -> DistanceRestraintGeometry:
    u = mda.Universe(topology, trajectory, topology_format=topology_format)

    guest_ag = _get_selection(u, guest_atoms, guest_selection)
    host_ag = _get_selection(u, host_atoms, host_selection)

    return DistanceRestraintGeometry(guest_atoms=guest_atoms, host_atoms=host_atoms)


def get_molecule_centers_restraint(
    topology: Union[str, openmm.app.Topology],
    trajectory: pathlib.Path,
    molA_rdmol: Chem.Mol,
    molB_rdmol: Chem.Mol,
    molA_idxs: list[int],
    molB_idxs: list[int],
    topology_format: Optional[str] = None,
):
    # We assume that the mol idxs are ordered
    centerA = molA_idxs[_get_central_atom_idx(molA_rdmol)]
    centerB = molB_idxs[_get_central_atom_idx(molB_rdmol)]

    u = mda.Universe(topology, trajectory, topology_format=topology_format)
    guest_ag = _get_selection(
        u,
        [centerA],
        None,
    )
    guest_ag = _get_selection(
        u,
        [centerB],
        None,
    )

    return DistsanceRestraintGeometry(guest_atoms=guest_atoms, host_atoms=host_atoms)
