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

from rdkit import Chem

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

        angleA = calc_angles(
            at1.position, at2.position, at3.position, u.atoms.dimensions
        )
        angleB = calc_angles(
            at2.position, at3.position, at4.position, u.atoms.dimensions
        )
        return angleA, angleB

    def get_dihedrals(self, topology, coordinates) -> unit.Quantity:
        u = mda.Universe(topology, coordinates)
        at1 = u.atoms[host_atoms[0]]
        at2 = u.atoms[host_atoms[1]]
        at3 = u.atoms[host_atoms[2]]
        at4 = u.atoms[guest_atoms[0]]
        at5 = u.atoms[guest_atoms[1]]
        at6 = u.atoms[guest_atoms[2]]

        dihA = calc_dihedrals(
            at1.position, at2.position, at3.position, at4.position, u.atoms.dimensions
        )
        dihB = calc_dihedrals(
            at2.position, at3.position, at4.position, at5.position, u.atoms.dimensions
        )
        dihC = calc_dihedrals(
            at3.position, at4.position, at5.position, at6.position, u.atoms.dimensions
        )

        return dihA, dihB, dihC


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


def _get_bonded_angles_from_pool(
    rdmol: Chem.Mol, atom_idx: int, atom_pool: list[int]
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

    Returns
    -------
    list[tuple[int, int, int]]
      A list of tuples containing all the angles.
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
    return angles


def get_small_molecule_atom_candidates(
    topology: Union[str, openmm.app.Topology],
    trajectory: Union[str, pathlib.Path],
    rdmol: Chem.Mol,
    ligand_idxs: list[int],
    rmsf_cutoff: unit.Quantity = 1 * unit.angstrom,
    angle_force_constant=83.68 * unit.kilojoule_per_mole / unit.radians**2,
):
    """
    Get a list of potential ligand atom choices for a Boresch restraint
    being applied to a given small molecule.

    TODO: remember to update the RDMol with the last frame positions
    """
    if isinstance(topology, openmm.app.Topology):
        topology_format = "OPENMMTOPOLOGY"
    else:
        topology_format = None

    u = mda.Universe(topology, trajectory, topology_format=topology_format)
    ligand_ag = u.atoms[ligand_idxs]

    # 0. Get the ligand RMSF
    rmsf = get_local_rmsf(ligand_ag)
    u.trajectory[-1]  # forward to the last frame

    # 1. Get the pool of atoms to work with
    # TODO: move to a helper function to make it easier to test
    # Get a list of all the aromatic rings
    # Note: no need to keep track of rings because we'll filter by
    # bonded terms after, so if we only keep rings then all the bonded
    # atoms should be within the same ring system.
    atom_pool = set()
    for ring in get_aromatic_rings(rdmol):
        max_rmsf = rmsf[list(ring)].max()
        if max_rmsf < rmsf_cutoff:
            atom_pool.update(ring)

    # if we don't have enough atoms just get all the heavy atoms
    if len(atom_pool) < 3:
        heavy_atoms = get_heavy_atom_idxs(rdmol)
        atom_pool = set(heavy_atoms[rmsf[heavy_atoms] < rmsf_cutoff])
        if len(atom_pool) < 3:
            errmsg = (
                "No suitable ligand atoms for " "the boresch restraint could be found"
            )
            raise ValueError(errmsg)

    # 2. Get the central atom
    center = get_central_atom_idx(rdmol)

    # 3. Sort the atom pool based on their distance from the center
    sorted_anchor_pool = _sort_by_distance_from_atom(rdmol, center, anchor_pool)

    # 4. Get a list of probable angles
    angles_list = []
    for atom in sorted_anchor_pool:
        angles = _get_bonded_angles_from_pool(rdmol, atom, sorted_anchor_pool)
        for angle in _angles:
            angle_ag = ligand_ag.atoms[angle]
            collinear = is_collinear(ligand_ag.positions, angle)
            angle_value = (
                calc_angle(
                    angle_ag.atoms[0].position,
                    angle_ag.atoms[1].position,
                    angle_ag.atoms[2].position,
                    box=angle_ag.universe.dimensions,
                )
                * unit.radians
            )
            energy = check_angle_energy(
                angle_value, angle_force_constant, 298.15 * unit.kelvin
            )
            if not collinear and energy:
                angles_list.append(angle)

    return angles_list


def get_host_atom_candidates(
    topology: Union[str, openmm.app.Topology],
    trajectory: Union[str, pathlib.Path],
    host_idxs: list[int],
    l1_idx: int,
    host_selection: str,
    dssp_filter: bool = False,
    rmsf_cutoff: unit.Quantity = 0.1 * unit.nanometer,
    min_distance: unit.Quantity = 10 * unit.nanometer,
    max_distance: unit.Quantity = 30 * unit.nanometer,
    angle_force_constant=83.68 * unit.kilojoule_per_mole / unit.radians**2,
):
    if isinstance(topology, openmm.app.Topology):
        topology_format = "OPENMMTOPOLOGY"
    else:
        topology_format = None

    u = mda.Universe(topology, trajectory, topology_format=topology_format)
    protein_ag1 = u.atoms[host_idxs]
    protein_ag2 = protein_ag.select_atoms(protein_selection)

    # 0. TODO: implement DSSP filter
    # Should be able to just call MDA's DSSP method, but will need to catch an exception
    if dssp_filter:
        raise NotImplementedError("DSSP filtering is not currently implemented")

    # 1. Get the RMSF & filter
    rmsf = get_local_rmsf(sub_protein_ag)
    protein_ag3 = sub_protein_ag.atoms[rmsf[heavy_atoms] < rmsf_cutoff]

    # 2. Search of atoms within the min/max cutoff
    atom_finder = FindHostAtoms(
        protein_ag3, u.atoms[l1_idx], min_search_distance, max_search_distance
    )
    atom_finder.run()
    return atom_finder.results.host_idxs
