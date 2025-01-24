# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Restraint Geometry classes

TODO
----
* Add relevant duecredit entries.
"""
from typing import Optional, Iterable

from rdkit import Chem

from openff.units import unit
import MDAnalysis as mda
import numpy.typing as npt

from openfe.protocols.restraint_utils.geometry.utils import (
    get_aromatic_rings,
    get_heavy_atom_idxs,
    get_central_atom_idx,
    is_collinear,
    get_local_rmsf,
)


def _sort_by_distance_from_atom(
    rdmol: Chem.Mol, target_idx: int, atom_idxs: Iterable[int]
) -> list[int]:
    """
    Sort a list of RDMol atoms by their distance from a target atom.

    Parameters
    ----------
    target_idx : int
      The idx of the atom to measure from.
    atom_idxs : list[int]
      The idx values of the atoms to sort.
    rdmol : Chem.Mol
      RDKit Molecule the atoms belong to

    Returns
    -------
    list[int]
      The input atom idxs sorted by their distance from the target atom.
    """
    distances = []

    conformer = rdmol.GetConformer()
    # Get the target atom position
    target_pos = conformer.GetAtomPosition(target_idx)

    for idx in atom_idxs:
        pos = conformer.GetAtomPosition(idx)
        distances.append(((target_pos - pos).Length(), idx))

    return [i[1] for i in sorted(distances)]


def _bonded_angles_from_pool(
    rdmol: Chem.Mol,
    atom_idx: int,
    atom_pool: list[int],
    aromatic_only: bool,
) -> list[tuple[int, int, int]]:
    """
    Get all bonded angles starting from ``atom_idx`` from a pool of atoms.

    Parameters
    ----------
    rdmol : Chem.Mol
      The RDKit Molecule
    atom_idx : int
      The index of the atom to search angles from.
    atom_pool : list[int]
      The list of indices to pick possible angle partners from.
    aromatic_only : bool
      Prune any angles that include non-aromatic bonds.

    Returns
    -------
    list[tuple[int, int, int]]
      A list of tuples containing all the angles.

    Notes
    -----
    * In the original SepTop code at3 is picked as directly bonded to at1.
      By comparison here we instead follow the case that at3 is bonded to
      at2 but not bonded to at1.
    """
    angles = []

    # Get the base atom and its neighbors
    at1 = rdmol.GetAtomWithIdx(atom_idx)
    at1_neighbors = [at.GetIdx() for at in at1.GetNeighbors()]

    # We loop at2 and at3 through the sorted atom_pool in order to get
    # a list of angles in the branch that are sorted by how close the atoms
    # are from the central atom
    for at2 in atom_pool:
        if at2 in at1_neighbors:
            at2_neighbors = [
                at.GetIdx() for at in rdmol.GetAtomWithIdx(at2).GetNeighbors()
            ]
            for at3 in atom_pool:
                if at3 != atom_idx and at3 in at2_neighbors:
                    angles.append((atom_idx, at2, at3))

    if aromatic_only:  # TODO: move this to its own method?
        aromatic_rings = get_aromatic_rings(rdmol)

        def _belongs_to_ring(angle, aromatic_rings):
            for ring in aromatic_rings:
                if all(a in ring for a in angle):
                    return True
            return False

        for angle in angles:
            if not _belongs_to_ring(angle, aromatic_rings):
                angles.remove(angle)

    return angles


def _get_guest_atom_pool(
    rdmol: Chem.Mol,
    rmsf: npt.NDArray,
    rmsf_cutoff: unit.Quantity
) -> tuple[Optional[set[int]], bool]:
    """
    Filter atoms based on rmsf & rings, defaulting to heavy atoms if
    there are not enough.

    Parameters
    ----------
    rdmol : Chem.Mol
      The RDKit Molecule to search through
    rmsf : npt.NDArray
      A 1-D array of RMSF values for each atom.
    rmsf_cutoff : unit.Quantity
      The rmsf cutoff value for selecting atoms in units compatible with
      nanometer.

    Returns
    -------
    atom_pool : Optional[set[int]]
      A pool of candidate atoms.
    ring_atoms_only : bool
      True if only ring atoms were selected.
    """
    # Get a list of all the aromatic rings
    # Note: no need to keep track of rings because we'll filter by
    # bonded terms after, so if we only keep rings then all the bonded
    # atoms should be within the same ring system.
    atom_pool: set[tuple[int]] = set()
    ring_atoms_only: bool = True
    for ring in get_aromatic_rings(rdmol):
        max_rmsf = rmsf[list(ring)].max()
        if max_rmsf < rmsf_cutoff:
            atom_pool.update(ring)

    # if we don't have enough atoms just get all the heavy atoms
    if len(atom_pool) < 3:
        ring_atoms_only = False
        heavy_atoms = get_heavy_atom_idxs(rdmol)
        atom_pool = set(heavy_atoms[rmsf[heavy_atoms] < rmsf_cutoff])
        if len(atom_pool) < 3:
            return None, False

    return atom_pool, ring_atoms_only


def find_guest_atom_candidates(
    universe: mda.Universe,
    rdmol: Chem.Mol,
    guest_idxs: list[int],
    rmsf_cutoff: unit.Quantity = 1 * unit.nanometer,
) -> list[tuple[int]]:
    """
    Get a list of potential ligand atom choices for a Boresch restraint
    being applied to a given small molecule.

    Parameters
    ----------
    universe : mda.Universe
      An MDAnalysis Universe defining the system and its coordinates.
    rdmol : Chem.Mol
      An RDKit Molecule representing the small molecule ordered in
      the same way as it is listed in the topology.
    guest_idxs : list[int]
      The ligand indices in the topology.
    rmsf_cutoff : unit.Quantity
      The RMSF filter cut-off.

    Returns
    -------
    angle_list : list[tuple[int]]
      A list of tuples for each valid G0, G1, G2 angle. If ``None``, no
      angles could be found.

    Raises
    ------
    ValueError
      If no suitable ligand atoms could be found.

    TODO
    ----
    Should the RDMol have a specific frame position?
    """
    ligand_ag = universe.atoms[guest_idxs]

    # 0. Get the ligand RMSF
    rmsf = get_local_rmsf(ligand_ag)
    universe.trajectory[-1]  # forward to the last frame

    # 1. Get the pool of atoms to work with
    atom_pool, rings_only = _get_guest_atom_pool(rdmol, rmsf, rmsf_cutoff)

    if atom_pool is None:
        # We don't have enough atoms so we raise an error
        errmsg = "No suitable ligand atoms were found for the restraint"
        raise ValueError(errmsg)

    # 2. Get the central atom
    center = get_central_atom_idx(rdmol)

    # 3. Sort the atom pool based on their distance from the center
    sorted_atom_pool = _sort_by_distance_from_atom(rdmol, center, atom_pool)

    # 4. Get a list of probable angles
    angles_list = []
    for atom in sorted_atom_pool:
        angles = _bonded_angles_from_pool(
            rdmol=rdmol,
            atom_idx=atom,
            atom_pool=sorted_atom_pool,
            aromatic_only=rings_only,
        )
        for angle in angles:
            # Check that the angle is at least not collinear
            angle_ag = ligand_ag.atoms[list(angle)]
            if not is_collinear(ligand_ag.positions, angle, u.dimensions):
                angles_list.append(
                    (
                        angle_ag.atoms[0].ix,
                        angle_ag.atoms[1].ix,
                        angle_ag.atoms[2].ix
                    )
                )

    return angles_list
