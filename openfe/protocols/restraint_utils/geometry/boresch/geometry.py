# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Restraint Geometry classes

TODO
----
* Add relevant duecredit entries.
"""
from typing import Optional

from rdkit import Chem

from openff.units import unit
from openff.models.types import FloatQuantity
import MDAnalysis as mda
from MDAnalysis.lib.distances import calc_bonds, calc_angles, calc_dihedrals

from openfe.protocols.restraint_utils.geometry.base import (
    HostGuestRestraintGeometry
)
from .guest import find_guest_atom_candidates
from .host import find_host_atom_candidates, find_host_anchor


class BoreschRestraintGeometry(HostGuestRestraintGeometry):
    """
    A class that defines the restraint geometry for a Boresch restraint.

    The restraint is defined by the following:

      H2                         G2
       -                        -
        -                      -
         H1 - - H0 -- G0 - - G1

    Where HX represents the X index of ``host_atoms`` and GX
    the X index of ``guest_atoms``.
    """
    r_aA0: FloatQuantity['nanometer']
    """
    The equilibrium distance between H0 and G0.
    """
    theta_A0: FloatQuantity['radians']
    """
    The equilibrium angle value between H1, H0, and G0.
    """
    theta_B0: FloatQuantity['radians']
    """
    The equilibrium angle value between H0, G0, and G1.
    """
    phi_A0: FloatQuantity['radians']
    """
    The equilibrium dihedral value between H2, H1, H0, and G0.
    """
    phi_B0: FloatQuantity['radians']

    """
    The equilibrium dihedral value between H1, H0, G0, and G1.
    """
    phi_C0: FloatQuantity['radians']

    """
    The equilibrium dihedral value between H0, G0, G1, and G2.
    """


def _get_restraint_distances(
    atomgroup: mda.AtomGroup
) -> tuple[unit.Quantity]:
    """
    Get the bond, angle, and dihedral distances for an input atomgroup
    defining the six atoms for a Boresch-like restraint.

    The atoms must be in the order of H0, H1, H2, G0, G1, G2.

    Parameters
    ----------
    atomgroup : mda.AtomGroup
      An AtomGroup defining the restrained atoms in order.

    Returns
    -------
    bond : unit.Quantity
      The H0-G0 bond value.
    angle1 : unit.Quantity
      The H1-H0-G0 angle value.
    angle2 : unit.Quantity
      The H0-G0-G1 angle value.
    dihed1 : unit.Quantity
      The H2-H1-H0-G0 dihedral value.
    dihed2 : unit.Quantity
      The H1-H0-G0-G1 dihedral value.
    dihed3 : unit.Quantity
      The H0-G0-G1-G2 dihedral value.
    """
    bond = calc_bonds(
        atomgroup.atoms[0].position,
        atomgroup.atoms[3].position,
        box=atomgroup.dimensions
    ) * unit.angstroms

    angles = []
    for idx_set in [[1, 0, 3], [0, 3, 4]]:
        angle = calc_angles(
            atomgroup.atoms[idx_set[0]].position,
            atomgroup.atoms[idx_set[1]].position,
            atomgroup.atoms[idx_set[2]].position,
            box=atomgroup.dimensions,
        )
        angles.append(angle * unit.radians)

    dihedrals = []
    for idx_set in [[2, 1, 0, 3], [1, 0, 3, 4], [0, 3, 4, 5]]:
        dihed = calc_dihedrals(
            atomgroup.atoms[idx_set[0]].position,
            atomgroup.atoms[idx_set[1]].position,
            atomgroup.atoms[idx_set[2]].position,
            atomgroup.atoms[idx_set[3]].position,
            box=atomgroup.dimensions,
        )
        dihedrals.append(dihed * unit.radians)

    return bond, angles[0], angles[1], dihedrals[0], dihedrals[1], dihedrals[2]


def find_boresch_restraint(
    universe: mda.Universe,
    guest_rdmol: Chem.Mol,
    guest_idxs: list[int],
    host_idxs: list[int],
    guest_restraint_atoms_idxs: Optional[list[int]] = None,
    host_restraint_atoms_idxs: Optional[list[int]] = None,
    host_selection: str = "all",
    dssp_filter: bool = False,
    rmsf_cutoff: unit.Quantity = 0.1 * unit.nanometer,
    host_min_distance: unit.Quantity = 1 * unit.nanometer,
    host_max_distance: unit.Quantity = 3 * unit.nanometer,
    angle_force_constant: unit.Quantity = (
        83.68 * unit.kilojoule_per_mole / unit.radians**2
    ),
    temperature: unit.Quantity = 298.15 * unit.kelvin,
) -> BoreschRestraintGeometry:
    """
    Find suitable Boresch-style restraints between a host and guest entity
    based on the approach of Baumann et al. [1] with some modifications.

    Parameters
    ----------
    universe : mda.Universe
      An MDAnalysis Universe defining the system and its coordinates.
    guest_rdmol : Chem.Mol
      An RDKit Mol for the guest molecule.
    guest_idxs : list[int]
      Indices in the topology for the guest molecule.
    host_idxs : list[int]
      Indices in the topology for the host molecule.
    guest_restraint_atoms_idxs : Optional[list[int]]
      User selected indices of the guest molecule itself (i.e. indexed
      starting a 0 for the guest molecule). This overrides the
      restraint search and a restraint using these indices will
      be retruned. Must be defined alongside ``host_restraint_atoms_idxs``.
    host_restraint_atoms_idxs : Optional[list[int]]
      User selected indices of the host molecule itself (i.e. indexed
      starting a 0 for the hosts molecule). This overrides the
      restraint search and a restraint using these indices will
      be returnned. Must be defined alongside ``guest_restraint_atoms_idxs``.
    host_selection : str
      An MDAnalysis selection string to sub-select the host atoms.
    dssp_filter : bool
      Whether or not to filter the host atoms by their secondary structure.
    rmsf_cutoff : unit.Quantity
      The cutoff value for atom root mean square fluction. Atoms with RMSF
      values above this cutoff will be disregarded.
      Must be in units compatible with nanometer.
    host_min_distance : unit.Quantity
      The minimum distance between any host atom and the guest G0 atom.
      Must be in units compatible with nanometer.
    host_max_distance : unit.Quantity
      The maximum distance between any host atom and the guest G0 atom.
      Must be in units compatible with nanometer.
    angle_force_constant : unit.Quantity
      The force constant for the G1-G0-H0 and G0-H0-H1 angles. Must be
      in units compatible with kilojoule / mole / radians ** 2.
    temperature : unit.Quantity
      The system temperature in units compatible with Kelvin.

    Returns
    -------
    BoreschRestraintGeometry
      An object defining the parameters of the Boresch-like restraint.

    References
    ----------
    [1] Baumann, Hannah M., et al. "Broadening the scope of binding free energy
        calculations using a Separated Topologies approach." (2023).
    """
    if (guest_restraint_atoms_idxs is not None) and (host_restraint_atoms_idxs is not None):  # fmt: skip
        # In this case assume the picked atoms were intentional /
        # representative of the input and go with it
        guest_ag = universe.select_atoms[guest_idxs]
        guest_anchor = [
            at.ix for at in guest_ag.atoms[guest_restraint_atoms_idxs]
        ]
        host_ag = universe.select_atoms[host_idxs]
        host_anchor = [
            at.ix for at in host_ag.atoms[host_restraint_atoms_idxs]
        ]

        # Set the equilibrium values as those of the final frame
        universe.trajectory[-1]
        atomgroup = universe.atoms[host_anchor + guest_anchor]
        bond, ang1, ang2, dih1, dih2, dih3 = _get_restraint_distances(
            atomgroup
        )

        # TODO: add checks to warn if this is a badly picked
        # set of atoms.
        return BoreschRestraintGeometry(
            host_atoms=host_anchor,
            guest_atoms=guest_anchor,
            r_aA0=bond,
            theta_A0=ang1,
            theta_B0=ang2,
            phi_A0=dih1,
            phi_B0=dih2,
            phi_C0=dih3
        )

    if (guest_restraint_atoms_idxs is not None) ^ (host_restraint_atoms_idxs is not None):  # fmt: skip
        # This is not an intended outcome, crash out here
        errmsg = (
            "both ``guest_restraints_atoms_idxs`` and "
            "``host_restraint_atoms_idxs`` "
            "must be set or both must be None. "
            f"Got {guest_restraint_atoms_idxs} and {host_restraint_atoms_idxs}"
        )
        raise ValueError(errmsg)

    # 1. Fetch the guest anchors
    guest_anchors = find_guest_atom_candidates(
        universe=universe,
        rdmol=guest_rdmol,
        guest_idxs=guest_idxs,
        rmsf_cutoff=rmsf_cutoff,
    )

    if len(guest_anchors) == 0:
        errmsg = "No suitable ligand atoms found for the restraint."
        raise ValueError(errmsg)

    # 2. We then loop through the guest anchors to find suitable host atoms
    for guest_anchor in guest_anchors:
        # We next fetch the host atom pool
        # Note: return is a set, so need to convert it later on
        host_pool = find_host_atom_candidates(
            universe=universe,
            host_idxs=host_idxs,
            l1_idx=guest_anchor[0],
            host_selection=host_selection,
            dssp_filter=dssp_filter,
            rmsf_cutoff=rmsf_cutoff,
            min_distance=host_min_distance,
            max_distance=host_max_distance,
        )

        host_anchor = find_host_anchor(
            guest_atoms=universe.atoms[list(guest_anchor)],
            host_atom_pool=universe.atoms[list(host_pool)],
            minimum_distance=0.5 * unit.nanometer,
            angle_force_constant=angle_force_constant,
            temperature=temperature,
        )
        # continue if it's empty, otherwise stop
        if host_anchor is not None:
            break

    if host_anchor is None:
        errmsg = "No suitable host atoms could be found"
        raise ValueError(errmsg)

    # Set the equilibrium values as those of the final frame
    universe.trajectory[-1]
    atomgroup = universe.atoms[list(host_anchor) + list(guest_anchor)]
    bond, ang1, ang2, dih1, dih2, dih3 = _get_restraint_distances(
        atomgroup
    )

    return BoreschRestraintGeometry(
        host_atoms=host_anchor,
        guest_atoms=guest_anchor,
        r_aA0=bond,
        theta_A0=ang1,
        theta_B0=ang2,
        phi_A0=dih1,
        phi_B0=dih2,
        phi_C0=dih3
    )
