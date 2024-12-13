# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Restraint Geometry classes

TODO
----
* Add relevant duecredit entries.
"""
import abc
import pathlib
from pydantic.v1 import BaseModel, validator

from rdkit import Chem

from openff.units import unit
import MDAnalysis as mda
from MDANalysis.analysis.base import AnalysisBase
from MDAnalysis.lib.distances import calc_bonds, calc_angles, calc_dihedrals
import numpy as np
import numpy.typing as npt

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
    def get_bond_distance(
        self,
        topology: Union[str, pathlib.Path, openmm.app.Topology], 
        coordinates: Union[str, pathlib.Path, npt.NDArray],
    ) -> unit.Quantity:
        """
        Get the H2 - G0 distance.

        Parameters
        ----------
        topology : Union[str, openmm.app.Topology]
        coordinates : Union[str, npt.NDArray]
          A coordinate file or NDArray in frame-atom-coordinate
          order in Angstrom.
        """
        u = mda.Universe(
            topology,
            coordinates,
            format=_get_mda_coord_format(coordinates),
            topology_format=_get_mda_topology_format(topology)
        )
        at1 = u.atoms[host_atoms[2]]
        at2 = u.atoms[guest_atoms[0]]
        bond = calc_bonds(at1.position, at2.position, u.atoms.dimensions)
        # convert to float so we avoid having a np.float64
        return float(bond) * unit.angstrom

    def get_angles(
        self,
        topology: Union[str, pathlib.Path, openmm.app.Topology],
        coordinates: Union[str, pathlib.Path, npt.NDArray],
    ) -> unit.Quantity:
        """
        Get the H1-H2-G0, and H2-G0-G1 angles.

        Parameters
        ----------
        topology : Union[str, openmm.app.Topology]
        coordinates : Union[str, npt.NDArray]
          A coordinate file or NDArray in frame-atom-coordinate
          order in Angstrom.
        """
        u = mda.Universe(
            topology,
            coordinates,
            format=_get_mda_coord_format(coordinates),
            topology_format=_get_mda_topology_format(topology)
        )
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

    def get_dihedrals(
        self,
        topology: Union[str, pathlib.Path, openmm.app.Topology],
        coordinates: Union[str, pathlib.Path, npt.NDArray],
    ) -> unit.Quantity:
        """
        Get the H0-H1-H2-G0, H1-H2-G0-G1, and H2-G0-G1-G2 dihedrals.

        Parameters
        ----------
        topology : Union[str, openmm.app.Topology]
        coordinates : Union[str, npt.NDArray]
          A coordinate file or NDArray in frame-atom-coordinate
          order in Angstrom.
        """
        u = mda.Universe(
            topology,
            coordinates,
            format=_get_mda_coord_format(coordinates),
            topology_format=_get_mda_topology_format(topology)
        )
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


def _get_atom_pool(rdmol: Chem.Mol, rmsf: npt.NDArray) -> Optional[set[int]]:
    """
    Filter atoms based on rmsf & rings, defaulting to heavy atoms if
    there are not enough.

    Parameters
    ----------
    rdmol : Chem.Mol
      The RDKit Molecule to search through
    rmsf : npt.NDArray
      A 1-D array of RMSF values for each atom.

    Returns
    -------
    atom_pool : Optional[set[int]]
    """
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
            return None

    return atom_pool


def get_guest_atom_candidates(
    topology: Union[str, pathlib.Path, openmm.app.Topology],
    trajectory: Union[str, pathlib.Path],
    rdmol: Chem.Mol,
    guest_idxs: list[int],
    rmsf_cutoff: unit.Quantity = 1 * unit.nanometer,
) -> list[tuple[int]]:
    """
    Get a list of potential ligand atom choices for a Boresch restraint
    being applied to a given small molecule.

    Parameters
    ----------
    topology : Union[str, openmm.app.Topology]
      The topology of the system.
    trajectory : Union[str, pathlib.Path]
      A path to the system's coordinate trajectory.
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
      A list of tuples for each valid l1, l2, l3 angle. If ``None``, no
      angles could be found.

    Raises
    ------
    ValueError
      If no suitable ligand atoms could be found.

    TODO
    ----
    Remember to update the RDMol with the last frame positions.
    """
    u = mda.Universe(
        topology,
        coordinates,
        format=_get_mda_coord_format(coordinates),
        topology_format=_get_mda_topology_format(topology)
    )

    ligand_ag = u.atoms[guest_idxs]

    # 0. Get the ligand RMSF
    rmsf = get_local_rmsf(ligand_ag)
    u.trajectory[-1]  # forward to the last frame

    # 1. Get the pool of atoms to work with
    atom_pool = _get_atom_pool(rdmol: Chem.Mol, rmsf: npt.NDArray)

    if atom_pool is None:
        # We don't have enough atoms so we raise an error
        errmsg = "No suitable ligand atoms were found for the restraint"
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


def get_host_atom_candidates(
    topology: Union[str, pathlib.Path, openmm.app.Topology],
    trajectory: Union[str, pathlib.Path],
    host_idxs: list[int],
    l1_idx: int,
    host_selection: str,
    dssp_filter: bool = False,
    rmsf_cutoff: unit.Quantity = 0.1 * unit.nanometer,
    min_distance: unit.Quantity = 1 * unit.nanometer,
    max_distance: unit.Quantity = 3 * unit.nanometer,
):
    """
    Get a list of suitable host atoms.

    Parameters
    ----------
    topology : Union[str, openmm.app.Topology]
      The topology of the system.
    trajectory : Union[str, pathlib.Path]
      A path to the system's coordinate trajectory.
    host_idxs : list[int]
      A list of the host indices in the system topology.
    l1_idx : int
      The index of the proposed l1 binding atom.
    host_selection : str
      An MDAnalysis selection string to fileter the host by.
    dssp_filter : bool
      Whether or not to apply a DSSP filter on the host selection.
    rmsf_cutoff : uni.Quantity
      The maximum RMSF value allowwed for any candidate host atom.
    min_distance : unit.Quantity
      The minimum search distance around l1 for suitable candidate atoms.
    max_distance : unit.Quantity
      The maximum search distance around l1 for suitable candidate atoms.
    """
    u = mda.Universe(
        topology,
        coordinates,
        format=_get_mda_coord_format(coordinates),
        topology_format=_get_mda_topology_format(topology)
    )

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


class EvaluateH2Atoms(AnalysisBase):
    """
    Class to evaluate the suitability of a set of host atoms
    as a H2 atom (i.e. bonded to the guest G0 atom).

    Parameters
    ----------
    guest_atoms: MDAnalysis.AtomGroup
      The guest atoms representing G0-G1-G2.
    host_atom_pool: MDAnalysis.AtomGroup
      The pool of atoms to pick a H2 from.
    angle_force_constant : unit.Quantity
      The force constant for the H2-G0-G1 angle.
    """


def find_boresch_restraint(
    topology: Union[str, pathlib.Path, openmm.app.Topology],
    trajectory: Union[str, pathlib.Path],
    guest_rdmol: Chem.Mol,
    guest_idxs: list[int],
    host_idxs: list[int],
    guest_restraint_atom_idxs: Optional[list[int]] = None,
    host_restraint_atoms_idxs Optional[list[int]] = None,
    host_selection: str = 'all',
    dssp_filter: bool = False,
    rmsf_custoff: unit.Quantity = 0.1 * unit.nanometer,
    host_min_distance: unit.Quantity = 1 * unit.nanometer,
    host_max_distance: unit.Quantity = 3 * unit.nanometer,
) -> BoreschRestraintGeometry:
    """
    Find suitable Boresch-style restraints between a host and guest entity.

    Parameters
    ----------
    ...

    Returns
    -------
    ...
    """
    u = mda.Universe(
        topology,
        coordinates,
        format=_get_mda_coord_format(coordinates),
        topology_format=_get_mda_topology_format(topology)
    )
    u.trajectory[-1]  # Work with the final frame

    if (guest_restraint_atoms_idxs is not None) and (host_restraint_atoms_idxs is not None):
        # In this case assume the picked atoms were intentional / representative
        # of the input and go with it
        guest_ag = u.select_atoms[guest_idxs]
        guest_angle = (at.ix for at in guest_ag.atoms[guest_restraint_atom_idxs])
        host_ag = u.select_atoms[host_idxs]
        host_angle = (at.ix for at in host_ag.atoms[host_restraint_atoms_idxs])
        # TODO sort out the return on this
        return BoreschRestraintGeometry(...)

    if (guest_restraint_atoms_idxs is not None) ^ (host_restraint_atoms_idxs is not None):
        # This is not an intended outcome, crash out here
        errmsg = (
            "both ``guest_restraints_atoms_idxs`` and ``host_restraint_atoms_idxs`` "
            "must be set or both must be None. "
            f"Got {guest_restraint_atoms_idxs} and {host_atoms_restraint_atoms_idxs}"
        )
        raise ValueError(errmsg)

    # Fetch the guest angles
    guest_angles = get_guest_atom_candidates(
        topology=topology,
        trajectory=trajectory,
        rdmol=guest_rdmol,
        guest_idxs=guest_idxs,
        rmsf_cutoff=rmsf_cutoff,
    )

    guest_angle = guest_angles[0]

    # Fetch the host atom pool
    host_pool = get_host_atom_candidates(
        topology=topology,
        trajectory=trajectory,
        host_idxs=host_idxs,
        l1_idx=guest_angle[0],
        host_selection=host_selection,
        dssp_filter=dssp_filter,
        rmsf_cutoff=rmsf_custoff,
        min_distance=host_min_distance,
        max_distance=host_max_distance,
    )

    # Get the guest angle atomgroup
    guest_ag = u.atoms[list(guest_angle)]

    # Find all suitable H2 idxs
    h2_idxs = []
    for i in host_pool:
        host2_at = u.atoms[i]
        pos = np.vstack((at.position, guest_ag.positions))
        angle = calc_angles(pos[0], pos[1], pos[2], box=u.dimensions) * unit.radians
        dihed = calc_dihedrals(pos[0], pos[1], pos[2], pos[3], box=u.dimensions) * unit.radians
        collinear = is_collinear(positions, [0, 1, 2, 3])

