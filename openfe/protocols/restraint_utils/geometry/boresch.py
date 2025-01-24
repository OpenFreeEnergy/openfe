# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Restraint Geometry classes

TODO
----
* Add relevant duecredit entries.
"""
import itertools
import pathlib
from typing import Union, Optional, Iterable

from rdkit import Chem

import openmm
from openff.units import unit
from openff.models.types import FloatQuantity
import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.dssp import DSSP
from MDAnalysis.lib.distances import calc_bonds, calc_angles, calc_dihedrals
import numpy as np
import numpy.typing as npt

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
    topology: Union[str, pathlib.Path, openmm.app.Topology],
    trajectory: Union[str, pathlib.Path, npt.NDArray],
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
    trajectory : Union[str, pathlib.Path, npt.NDArray]
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
    Should the RDMol have a specific frame position?
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


def find_host_atom_candidates(
    topology: Union[str, pathlib.Path, openmm.app.Topology],
    trajectory: Union[str, pathlib.Path, npt.NDArray],
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
    trajectory : Union[str, pathlib.Path, npt.NDArray]
      The system's coordinate trajectory.
    host_idxs : list[int]
      A list of the host indices in the system topology.
    l1_idx : int
      The index of the proposed l1 binding atom.
    host_selection : str
      An MDAnalysis selection string to filter the host by.
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

    # Get an AtomGroup for the host based on the input host indices
    host_ag = u.atoms[host_idxs]
    # Filter the host AtomGroup based on ``host_selection`
    selected_host_ag = host_ag.select_atoms(host_selection)

    # 0. TODO: implement DSSP filter
    # Should be able to just call MDA's DSSP method
    # but will need to catch an exception
    if dssp_filter:
        # TODO: make this a method
        # We use "host_ag" to get the entire host
        protein_ag = host_ag.select_atoms('protein')
        if len(protein_ag) < 50:
            # TODO: make this not fail but warn?
            errmsg = "Insufficient protein residues were found - cannot run DSSP filter"
            raise ValueError(errmsg)

        # Split by fragments
        if not hasattr(protein_ag, 'bonds'):
            protein_ag.guess_bonds()

        fragments = protein_ag.fragments

        structure = []
        helix_count = 0
        sheet_count = 0
        for frag in fragments:
            # Note: will want to always skip the first and last residues because that trips up DSSP
            # TODO: make the skip a user-supplied thing
            chain = frag.residues[10:-10].atoms
            # Run on the last frame
            dssp = DSSP(chain).run(start=-1)

            # Tag each residue motif by its resindex
            dssp_results = [
                (motif, resid) for motif, resid in
                zip(dssp.results.dssp[0], chain.residues.resindices)
            ]

            helix_count += list(dssp.results.dssp[0]).count('H')
            sheet_count += list(dssp.results.dssp[0]).count('E')

            for _, group in itertools.groupby(dssp_results, lambda x: x[0]):
                structure.append(list(group))

        allowed_motifs = ['H']
        if helix_count < sheet_count:
            allowed_motifs.append('E')

        allowed_residxs = []
        for motif_chain in structure:
            # TODO: make the value of a "stable sheet/helix" user selectable
            if motif_chain[0][0] in allowed_motifs and len(motif_chain) > 7:
                # TODO: make the amount of residues to remove at the edges user
                # selectable?
                allowed_residxs.extend(
                    [residue[1] for residue in motif_chain[3:-3]]
                )

        # Resindexes at key at the Universe scale not atomgroup
        allowed_atoms = protein_ag.universe.residues[allowed_residxs].atoms

        # Pick up all the atoms that intersect the initial selection and
        # those allowed.
        selected_host_ag = selected_host_ag.intersection(allowed_atoms)

    # 1. Get the RMSF & filter to create a new AtomGroup
    rmsf = get_local_rmsf(selected_host_ag)
    filtered_host_ag = selected_host_ag.atoms[rmsf < rmsf_cutoff]

    # 2. Search of atoms within the min/max cutoff
    atom_finder = FindHostAtoms(
        host_atoms=filtered_host_ag,
        guest_atoms=u.atoms[l1_idx],
        min_search_distance=min_distance,
        max_search_distance=max_distance,
    )
    atom_finder.run()
    return atom_finder.results.host_idxs


class EvaluateHostAtoms1(AnalysisBase):
    """
    Class to evaluate the suitability of a set of host atoms
    as either H0 or H1 atoms (i.e. the first and second host atoms).

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
        # Set everything to False to begin with
        self.results.valid[:] = False

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
                atoms=[0, 1, 2, 3],
                dimensions=self.reference.dimensions,
            )
            self.results.distances[i][self._frame_index] = distance
            self.results.angles[i][self._frame_index] = angle
            self.results.dihedrals[i][self._frame_index] = dihedral
            self.results.collinear[i][self._frame_index] = collinear

    def _conclude(self):
        for i, at in enumerate(self.host_atom_pool):
            # Check distances
            distance_bounds = all(
                self.results.distances[i] > self.minimum_distance
            )
            # Check angles
            angle_bounds = all(
                check_angle_not_flat(
                    angle=angle * unit.radians,
                    force_constant=self.angle_force_constant,
                    temperature=self.temperature
                )
                for angle in self.results.angles[i]
            )
            angle_variance = check_angular_variance(
                self.results.angles[i] * unit.radians,
                upper_bound=np.pi * unit.radians,
                lower_bound=0 * unit.radians,
                width=1.745 * unit.radians,
            )
            # Check dihedrals
            dihed_bounds = all(
                check_dihedral_bounds(dihed * unit.radians)
                for dihed in self.results.dihedrals[i]
            )
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
    """
    Class to evaluate the suitability of a set of host atoms
    as H2 atoms (i.e. the third host atoms).

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
    def _prepare(self):
        self.results.distances1 = np.zeros((len(self.host_atom_pool), self.n_frames))
        self.results.distances2 = np.zeros((len(self.host_atom_pool), self.n_frames))
        self.results.dihedrals = np.zeros((len(self.host_atom_pool), self.n_frames))
        self.results.collinear = np.empty(
            (len(self.host_atom_pool), self.n_frames),
            dtype=bool,
        )
        self.results.valid = np.empty(
            len(self.host_atom_pool),
            dtype=bool,
        )
        # Default to valid == False
        self.results.valid[:] = False

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
                atoms=[0, 1, 2, 3],
                dimensions=self.reference.dimensions,
            )
            self.results.distances1[i][self._frame_index] = distance1
            self.results.distances2[i][self._frame_index] = distance2
            self.results.dihedrals[i][self._frame_index] = dihedral
            self.results.collinear[i][self._frame_index] = collinear

    def _conclude(self):
        for i, at in enumerate(self.host_atom_pool):
            distance1_bounds = all(
                self.results.distances1[i] > self.minimum_distance
            )
            distance2_bounds = all(
                self.results.distances2[i] > self.minimum_distance
            )
            dihed_bounds = all(
                check_dihedral_bounds(dihed * unit.radians)
                for dihed in self.results.dihedrals[i]
            )
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


def _find_host_anchor(
    guest_atoms: mda.AtomGroup,
    host_atom_pool: mda.AtomGroup,
    minimum_distance: unit.Quantity,
    angle_force_constant: unit.Quantity,
    temperature: unit.Quantity
) -> Optional[list[int]]:
    """
    Find suitable atoms for the H0-H1-H2 portion of the restraint.

    Parameters
    ----------
    guest_atoms : mda.AtomGroup
      The guest anchor atoms for G0-G1-G2
    host_atom_pool : mda.AtomGroup
      The host atoms to search from.
    minimum_distance : unit.Quantity
      The minimum distance to pick host atoms from each other.
    angle_force_constant : unit.Quantity
      The force constant for the G1-G0-H0 and G0-H0-H1 angles.
    temperature : unit.Quantity
      The target system temperature.

    Returns
    -------
    Optional[list[int]]
      A list of indices for a selected combination of H0, H1, and H2.
    """
    # Evalulate the host_atom_pool for suitability as H0 atoms
    h0_eval = EvaluateHostAtoms1(
        guest_atoms,
        host_atom_pool,
        minimum_distance,
        angle_force_constant,
        temperature,
    )
    h0_eval.run()

    for i, valid_h0 in enumerate(h0_eval.results.valid):
        # If valid H0 atom, evaluate rest of host_atom_pool for suitability
        # as H1 atoms.
        if valid_h0:
            h0g0g1_atoms = host_atom_pool.atoms[i] + guest_atoms.atoms[:2]
            h1_eval = EvaluateHostAtoms1(
                h0g0g1_atoms,
                host_atom_pool,
                minimum_distance,
                angle_force_constant,
                temperature,
            )
            h1_eval.run()
            for j, valid_h1 in enumerate(h1_eval.results.valid):
                # If valid H1 atom, evaluate rest of host_atom_pool for
                # suitability as H2 atoms
                if valid_h1:
                    h1h0g0_atoms = host_atom_pool.atoms[j] + h0g0g1_atoms.atoms[:2]
                    h2_eval = EvaluateHostAtoms2(
                        h1h0g0_atoms,
                        host_atom_pool,
                        minimum_distance,
                        angle_force_constant,
                        temperature,
                    )
                    h2_eval.run()

                    if any(h2_eval.results.valid):
                        # Get the sum of the average distances (dsum_avgs)
                        # for all the host_atom_pool atoms
                        distance1_avgs = np.array(
                            [d.mean() for d in h2_eval.results.distances1]
                        )
                        distance2_avgs = np.array(
                            [d.mean() for d in h2_eval.results.distances2]
                        )
                        dsum_avgs = distance1_avgs + distance2_avgs

                        # Now filter by validity as H2 atom
                        h2_dsum_avgs = [
                            (idx, val) for idx, val in enumerate(dsum_avgs)
                            if h2_eval.results.valid[idx]
                        ]

                        # Get the index of the H2 atom with the lowest
                        # average distance
                        k = sorted(h2_dsum_avgs, key=lambda x: x[1])[0][0]

                        return list(host_atom_pool.atoms[[i, j, k]].ix)
    return None


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
    Find suitable Boresch-style restraints between a host and guest entity
    based on the approach of Baumann et al. [1] with some modifications.

    Parameters
    ----------
    topology : Union[str, pathlib.Path, openmm.app.Topology]
      A topology of the system.
    trajectory : Union[str, pathlib.Path]
      A path to a coordinate trajectory file.
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
    u = mda.Universe(
        topology,
        trajectory,
        format=_get_mda_coord_format(trajectory),
        topology_format=_get_mda_topology_format(topology),
    )

    if (guest_restraint_atoms_idxs is not None) and (host_restraint_atoms_idxs is not None):  # fmt: skip
        # In this case assume the picked atoms were intentional /
        # representative of the input and go with it
        guest_ag = u.select_atoms[guest_idxs]
        guest_anchor = [
            at.ix for at in guest_ag.atoms[guest_restraint_atoms_idxs]
        ]
        host_ag = u.select_atoms[host_idxs]
        host_anchor = [
            at.ix for at in host_ag.atoms[host_restraint_atoms_idxs]
        ]

        # Set the equilibrium values as those of the final frame
        u.trajectory[-1]
        atomgroup = u.atoms[host_anchor + guest_anchor]
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
        topology=topology,
        trajectory=trajectory,
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
            topology=topology,
            trajectory=trajectory,
            host_idxs=host_idxs,
            l1_idx=guest_anchor[0],
            host_selection=host_selection,
            dssp_filter=dssp_filter,
            rmsf_cutoff=rmsf_cutoff,
            min_distance=host_min_distance,
            max_distance=host_max_distance,
        )

        host_anchor = _find_host_anchor(
            guest_atoms=u.atoms[list(guest_anchor)],
            host_atom_pool=u.atoms[list(host_pool)],
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
    u.trajectory[-1]
    atomgroup = u.atoms[list(host_anchor) + list(guest_anchor)]
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
