# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Restraint Geometry classes

TODO
----
* Add relevant duecredit entries.
"""
from typing import Optional
import MDAnalysis as mda
from rdkit import Chem

from .base import HostGuestRestraintGeometry
from .utils import (
    get_central_atom_idx,
    _get_mda_selection,
)


class DistanceRestraintGeometry(HostGuestRestraintGeometry):
    """
    A geometry class for a distance restraint between two groups of atoms.
    """


def get_distance_restraint(
    universe: mda.Universe,
    host_atoms: Optional[list[int]] = None,
    guest_atoms: Optional[list[int]] = None,
    host_selection: Optional[str] = None,
    guest_selection: Optional[str] = None,
) -> DistanceRestraintGeometry:
    """
    Get a DistanceRestraintGeometry between two groups of atoms.

    You can either select the groups by passing through a set of indices
    or an MDAnalysis selection.

    Parameters
    ----------
    universe : mda.Universe
      An MDAnalysis Universe defining the system and its coordinates.
    host_atoms : Optional[list[int]]
      A list of host atoms indices. Either ``host_atoms`` or
      ``host_selection`` must be defined.
    guest_atoms : Optional[list[int]]
      A list of guest atoms indices. Either ``guest_atoms`` or
      ``guest_selection`` must be defined.
    host_selection : Optional[str]
      An MDAnalysis selection string to define the host atoms.
      Either ``host_atoms`` or ``host_selection`` must be defined.
    guest_selection : Optional[str]
      An MDAnalysis selection string to define the guest atoms.
      Either ``guest_atoms`` or ``guest_selection`` must be defined.

    Returns
    -------
    DistanceRestraintGeometry
      An object that defines a distance restraint geometry.
    """
    guest_ag = _get_mda_selection(universe, guest_atoms, guest_selection)
    guest_atoms = [a.ix for a in guest_ag]
    host_ag = _get_mda_selection(universe, host_atoms, host_selection)
    host_atoms = [a.ix for a in host_ag]

    return DistanceRestraintGeometry(
        guest_atoms=guest_atoms, host_atoms=host_atoms
    )


def get_molecule_centers_restraint(
    molA_rdmol: Chem.Mol,
    molB_rdmol: Chem.Mol,
    molA_idxs: list[int],
    molB_idxs: list[int],
):
    """
    Get a DistanceRestraintGeometry between the central atoms of
    two molecules.

    Parameters
    ----------
    molA_rdmol : Chem.Mol
      An RDKit Molecule for the first molecule.
    molB_rdmol : Chem.Mol
      An RDKit Molecule for the second molecule.
    molA_idxs : list[int]
      The indices of the first molecule in the system. Note we assume these
      to be sorted in the same order as the input rdmol.
    molB_idxs : list[int]
      The indices of the second molecule in the system. Note we assume these
      to be sorted in the same order as the input rdmol.

    Returns
    -------
    DistanceRestraintGeometry
      An object that defines a distance restraint geometry.
    """
    # We assume that the mol idxs are ordered
    centerA = molA_idxs[get_central_atom_idx(molA_rdmol)]
    centerB = molB_idxs[get_central_atom_idx(molB_rdmol)]

    return DistanceRestraintGeometry(
        guest_atoms=[centerA], host_atoms=[centerB]
    )
