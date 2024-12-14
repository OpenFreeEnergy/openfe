# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Restraint Geometry classes

TODO
----
* Add relevant duecredit entries.
"""
import pathlib
from typing import Union, Optional, Iterable

from rdkit import Chem

import openmm
from openff.units import unit
import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.lib.distances import calc_bonds, calc_angles, calc_dihedrals
import numpy as np
import numpy.typing as npt
from scipy.stats import circmean

from .base import HostGuestRestraintGeometry
from .utils import (
    _get_mda_coord_format,
    _get_mda_topology_format,
    get_aromatic_rings,
    get_heavy_atom_idxs,
    get_central_atom_idx,
    is_collinear,
    check_angular_variance,
    check_dihedral_bounds,
    check_angle_not_flat,
    FindHostAtoms,
    get_local_rmsf
)


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

    def get_bond_distance(
        self,
        topology: Union[str, pathlib.Path, openmm.app.Topology],
        coordinates: Union[str, pathlib.Path, npt.NDArray],
    ) -> unit.Quantity:
        """
        Get the H0 - G0 distance.

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
            topology_format=_get_mda_topology_format(topology),
        )
        at1 = u.atoms[self.host_atoms[0]]
        at2 = u.atoms[self.guest_atoms[0]]
        bond = calc_bonds(at1.position, at2.position, u.atoms.dimensions)
        # convert to float so we avoid having a np.float64
        return float(bond) * unit.angstrom

    def get_angles(
        self,
        topology: Union[str, pathlib.Path, openmm.app.Topology],
        coordinates: Union[str, pathlib.Path, npt.NDArray],
    ) -> unit.Quantity:
        """
        Get the H1-H0-G0, and H0-G0-G1 angles.

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
            topology_format=_get_mda_topology_format(topology),
        )
        at1 = u.atoms[self.host_atoms[1]]
        at2 = u.atoms[self.host_atoms[0]]
        at3 = u.atoms[self.guest_atoms[0]]
        at4 = u.atoms[self.guest_atoms[1]]

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
        Get the H2-H1-H0-G0, H1-H0-G0-G1, and H0-G0-G1-G2 dihedrals.

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
            topology_format=_get_mda_topology_format(topology),
        )
        at1 = u.atoms[self.host_atoms[2]]
        at2 = u.atoms[self.host_atoms[1]]
        at3 = u.atoms[self.host_atoms[0]]
        at4 = u.atoms[self.guest_atoms[0]]
        at5 = u.atoms[self.guest_atoms[1]]
        at6 = u.atoms[self.guest_atoms[2]]

        dihA = calc_dihedrals(
            at1.position, at2.position, at3.position, at4.position,
            box=u.dimensions
        )
        dihB = calc_dihedrals(
            at2.position, at3.position, at4.position, at5.position,
            box=u.dimensions
        )
        dihC = calc_dihedrals(
            at3.position, at4.position, at5.position, at6.position,
            box=u.dimensions
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


def _get_atom_pool(
    rdmol: Chem.Mol,
    rmsf: npt.NDArray,
    rmsf_cutoff: unit.Quantity
) -> Optional[set[int]]:
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
      A list of tuples for each valid G0, G1, G2 angle. If ``None``, no
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
        trajectory,
        format=_get_mda_coord_format(trajectory),
        topology_format=_get_mda_topology_format(topology),
    )

    ligand_ag = u.atoms[guest_idxs]

    # 0. Get the ligand RMSF
    rmsf = get_local_rmsf(ligand_ag)
    u.trajectory[-1]  # forward to the last frame

    # 1. Get the pool of atoms to work with
    atom_pool = _get_atom_pool(rdmol, rmsf)

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
        angles = _get_bonded_angles_from_pool(rdmol, atom, sorted_atom_pool)
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
) -> npt.NDArray:
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

    Return
    ------
    NDArray
      Array of host atom indexes
    """
    u = mda.Universe(
        topology,
        trajectory,
        format=_get_mda_coord_format(trajectory),
        topology_format=_get_mda_topology_format(topology),
    )

    host_ag1 = u.atoms[host_idxs]
    host_ag2 = host_ag1.select_atoms(host_selection)

    # 0. TODO: implement DSSP filter
    # Should be able to just call MDA's DSSP method
    # but will need to catch an exception
    if dssp_filter:
        raise NotImplementedError(
            "DSSP filtering is not currently implemented"
        )

    # 1. Get the RMSF & filter
    rmsf = get_local_rmsf(host_ag2)
    protein_ag3 = host_ag2.atoms[rmsf < rmsf_cutoff]

    # 2. Search of atoms within the min/max cutoff
    atom_finder = FindHostAtoms(
        protein_ag3, u.atoms[l1_idx], min_distance, max_distance
    )
    atom_finder.run()
    return atom_finder.results.host_idxs


class EvaluateHostAtoms1(AnalysisBase):
    """
    Class to evaluate the suitability of a set of host atoms
    as H1 atoms (i.e. the second host atom).

    Parameters
    ----------
    reference : MDAnalysis.AtomGroup
      The reference preceeding three atoms.
    host_atom_pool : MDAnalysis.AtomGroup
      The pool of atoms to pick an atom from.
    minimum_distance : unit.Quantity
      The minimum distance from the bound reference atom.
    angle_force_constant : unit.Quantity
      The force constant for the angle.
    temperature : unit.Quantity
      The system temperature in Kelvin
    """

    def __init__(
        self,
        reference,
        host_atom_pool,
        minimum_distance,
        angle_force_constant,
        temperature,
        **kwargs,
    ):
        super().__init__(reference.universe.trajectory, **kwargs)

        if len(reference) != 3:
            errmsg = "Incorrect number of reference atoms passed"
            raise ValueError(errmsg)

        self.reference = reference
        self.host_atom_pool = host_atom_pool
        self.minimum_distance = minimum_distance.to("angstrom").m
        self.angle_force_constant = angle_force_constant
        self.temperature = temperature

    def _prepare(self):
        self.results.distances = np.zeros(
            (len(self.host_atom_pool), self.n_frames)
        )
        self.results.angles = np.zeros(
            (len(self.host_atom_pool), self.n_frames)
        )
        self.results.dihedrals = np.zeros(
            (len(self.host_atom_pool), self.n_frames)
        )
        self.results.collinear = np.empty(
            (len(self.host_atom_pool), self.n_frames),
            dtype=bool,
        )
        self.results.valid = np.empty(
            len(self.host_atom_pool),
            dtype=bool,
        )

    def _single_frame(self):
        for i, at in enumerate(self.host_atom_pool):
            distance = calc_bonds(
                at.position,
                self.reference.atoms[0].position,
                box=self.reference.dimensions,
            )
            angle = calc_angles(
                at.position,
                self.reference.atoms[0].position,
                self.reference.atoms[1].position,
                box=self.reference.dimensions,
            )
            dihedral = calc_dihedrals(
                at.position,
                self.reference.atoms[0].position,
                self.reference.atoms[1].position,
                self.reference.atoms[2].position,
                box=self.reference.dimensions,
            )
            collinear = is_collinear(
                positions=np.vstack((at.position, self.reference.positions)),
                dimensions=self.reference.dimensions,
            )
            self.results.distances[i][self._frame_index] = distance
            self.results.angles[i][self._frame_index] = angle
            self.results.dihedrals[i][self._frame_index] = dihedral
            self.results.collinear[i][self._frame_index] = collinear

    def _conclude(self):
        for i, at in enumerate(self.host_atom_pool):
            distance_bounds = all(self.results.distances[i] > self.minimum_distance)
            mean_angle = circmean(self.results.angles[i], high=np.pi, low=0)
            angle_bounds = check_angle_not_flat(
                angle=mean_angle * unit.radians,
                force_constant=self.angle_force_constant,
                temperature=self.temperature,
            )
            angle_variance = check_angular_variance(
                self.results.angles[i] * unit.radians,
                upper_bound=np.pi * unit.radians,
                lower_bound=0 * unit.radians,
                width=1.745 * unit.radians,
            )
            mean_dihed = circmean(self.results.dihedrals[i], high=np.pi, low=-np.pi)
            dihed_bounds = check_dihedral_bounds(mean_dihed)
            dihed_variance = check_angular_variance(
                self.results.dihedrals[i] * unit.radians,
                upper_bound=np.pi * unit.radians,
                lower_bound=-np.pi * unit.radians,
                width=5.23 * unit.radians,
            )
            not_collinear = not all(self.results.collinear[i])
            if all(
                [
                    distance_bounds,
                    angle_bounds,
                    angle_variance,
                    dihed_bounds,
                    dihed_variance,
                    not_collinear,
                ]
            ):
                self.results.valid[i] = True


class EvaluateHostAtoms2(EvaluateHostAtoms1):
    def _prepare(self):
        self.results.distances1 = np.zeros((len(self.host_atom_pool), self.n_frames))
        self.results.ditances2 = np.zeros((len(self.host_atom_pool), self.n_frames))
        self.results.dihedrals = np.zeros((len(self.host_atom_pool), self.n_frames))
        self.results.collinear = np.empty(
            (len(self.host_atom_pool), self.n_frames),
            dtype=bool,
        )
        self.results.valid = np.empty(
            len(self.host_atom_pool),
            dtype=bool,
        )

    def _single_frame(self):
        for i, at in enumerate(self.host_atom_pool):
            distance1 = calc_bonds(
                at.position,
                self.reference.atoms[0].position,
                box=self.reference.dimensions,
            )
            distance2 = calc_bonds(
                at.position,
                self.reference.atoms[1].position,
                box=self.reference.dimensions,
            )
            dihedral = calc_dihedrals(
                at.position,
                self.reference.atoms[0].position,
                self.reference.atoms[1].position,
                self.reference.atoms[2].position,
                box=self.reference.dimensions,
            )
            collinear = is_collinear(
                positions=np.vstack((at.position, self.reference.positions)),
                dimensions=self.reference.dimensions,
            )
            self.results.distances1[i][self._frame_index] = distance1
            self.results.distances2[i][self._frame_index] = distance2
            self.results.dihedrals[i][self._frame_index] = dihedral
            self.results.collinear[i][self._frame_index] = collinear

    def _conclude(self):
        for i, at in enumerate(self.host_atom_pool):
            distance1_bounds = all(self.results.distances1[i] > self.minimum_distance)
            distance2_bounds = all(self.results.distances2[i] > self.minimum_distance)
            mean_dihed = circmean(self.results.dihedrals[i], high=np.pi, low=-np.pi)
            dihed_bounds = check_dihedral_bounds(mean_dihed)
            dihed_variance = check_angular_variance(
                self.results.dihedrals[i] * unit.radians,
                upper_bound=np.pi * unit.radians,
                lower_bound=-np.pi * unit.radians,
                width=5.23 * unit.radians,
            )
            not_collinear = not all(self.results.collinear[i])
            if all(
                [
                    distance1_bounds,
                    distance2_bounds,
                    dihed_bounds,
                    dihed_variance,
                    not_collinear,
                ]
            ):
                self.results.valid[i] = True


def _find_host_angle(
    g0g1g2_atoms,
    host_atom_pool,
    minimum_distance,
    angle_force_constant,
    temperature
):
    h0_eval = EvaluateHostAtoms1(
        g0g1g2_atoms,
        host_atom_pool,
        minimum_distance,
        angle_force_constant,
        temperature,
    )
    h0_eval.run()

    for i, valid_h0 in enumerate(h0_eval.results.valid):
        if valid_h0:
            g1g2h0_atoms = g0g1g2_atoms.atoms[1:] + host_atom_pool.atoms[i]
            h1_eval = EvaluateHostAtoms1(
                g1g2h0_atoms,
                host_atom_pool,
                minimum_distance,
                angle_force_constant,
                temperature,
            )
            for j, valid_h1 in enumerate(h1_eval.results.valid):
                g2h0h1_atoms = g1g2h0_atoms.atoms[1:] + host_atom_pool.atoms[j]
                h2_eval = EvaluateHostAtoms2(
                    g2h0h1_atoms,
                    host_atom_pool,
                    minimum_distance,
                    angle_force_constant,
                    temperature,
                )

                if any(h2_eval.ressults.valid):
                    d1_avgs = [d.mean() for d in h2_eval.results.distances1]
                    d2_avgs = [d.mean() for d in h2_eval.results.distances2]
                    dsum_avgs = d1_avgs + d2_avgs
                    k = dsum_avgs.argmin()

                    return host_atom_pool.atoms[[i, j, k]].ix
    return None


def find_boresch_restraint(
    topology: Union[str, pathlib.Path, openmm.app.Topology],
    trajectory: Union[str, pathlib.Path],
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
        trajectory,
        format=_get_mda_coord_format(trajectory),
        topology_format=_get_mda_topology_format(topology),
    )
    u.trajectory[-1]  # Work with the final frame

    if (guest_restraint_atoms_idxs is not None) and (host_restraint_atoms_idxs is not None):  # fmt: skip
        # In this case assume the picked atoms were intentional /
        # representative of the input and go with it
        guest_ag = u.select_atoms[guest_idxs]
        guest_angle = [
            at.ix for at in guest_ag.atoms[guest_restraint_atoms_idxs]
        ]
        host_ag = u.select_atoms[host_idxs]
        host_angle = [
            at.ix for at in host_ag.atoms[host_restraint_atoms_idxs]
        ]
        # TODO sort out the return on this
        return BoreschRestraintGeometry(
            host_atoms=host_angle, guest_atoms=guest_angle
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

    # 1. Fetch the guest angles
    guest_angles = get_guest_atom_candidates(
        topology=topology,
        trajectory=trajectory,
        rdmol=guest_rdmol,
        guest_idxs=guest_idxs,
        rmsf_cutoff=rmsf_cutoff,
    )

    if len(guest_angles) != 0:
        errmsg = "No suitable ligand atoms found for the restraint."
        raise ValueError(errmsg)

    # We pick the first angle / ligand atom set as the one to use
    guest_angle = guest_angles[0]

    # 2. We next fetch the host atom pool
    host_pool = get_host_atom_candidates(
        topology=topology,
        trajectory=trajectory,
        host_idxs=host_idxs,
        l1_idx=guest_angle[0],
        host_selection=host_selection,
        dssp_filter=dssp_filter,
        rmsf_cutoff=rmsf_cutoff,
        min_distance=host_min_distance,
        max_distance=host_max_distance,
    )

    # 3. We then loop through the guest angles to find suitable host atoms
    for guest_angle in guest_angles:
        host_angle = _find_host_angle(
            g0g1g2_atoms=u.atoms[list(guest_angle)],
            host_atom_pool=u.atoms[host_pool],
            minimum_distance=0.5 * unit.nanometer,
            angle_force_constant=angle_force_constant,
            temperature=temperature,
        )
        # continue if it's empty, otherwise stop
        if host_angle is not None:
            break

    if host_angle is None:
        errmsg = "No suitable host atoms could be found"
        raise ValueError(errmsg)

    return BoreschRestraintGeometry(
        host_atoms=host_angle, guest_atoms=guest_angle
    )
