# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Restraint Geometry classes

TODO
----
* Add relevant duecredit entries.
"""
from typing import Optional
import warnings

from openff.units import unit
import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.lib.distances import calc_bonds, calc_angles, calc_dihedrals
import numpy as np
import numpy.typing as npt

from openfe.protocols.restraint_utils.geometry.utils import (
    is_collinear,
    check_angular_variance,
    check_dihedral_bounds,
    check_angle_not_flat,
    FindHostAtoms,
    get_local_rmsf,
    stable_secondary_structure_selection
)


def find_host_atom_candidates(
    universe: mda.Universe,
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
    universe : mda.Universe
      An MDAnalysis Universe defining the system and its coordinates.
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
    # Get an AtomGroup for the host based on the input host indices
    host_ag = universe.atoms[host_idxs]

    # Filter the host AtomGroup based on ``host_selection`
    selected_host_ag = host_ag.select_atoms(host_selection)

    # If requested, filter the host atoms based on if their residues exist
    # within stable secondary structures.
    if dssp_filter:
        # TODO: allow user-supplied kwargs here
        stable_ag = stable_secondary_structure_selection(selected_host_ag)

        if len(stable_ag) < 20:
            wmsg = (
                "Secondary structure filtering: "
                "Too few atoms found via secondary strcuture filtering will "
                "try to only select all residues in protein chains instead."
            )
            warnings.warn(wmsg)
            stable_ag = protein_chain_selection(selected_host_ag)

        if len(stable_ag) < 20:
            wmsg = (
                "Secondary structure filtering: "
                "Too few atoms found in protein residue chains, will just "
                "use all atoms."
            )
            warnings.warn(wmsg)
        else:
            selected_host_ag = stable_ag

    # 1. Get the RMSF & filter to create a new AtomGroup
    rmsf = get_local_rmsf(selected_host_ag)
    filtered_host_ag = selected_host_ag.atoms[rmsf < rmsf_cutoff]

    # 2. Search of atoms within the min/max cutoff
    atom_finder = FindHostAtoms(
        host_atoms=filtered_host_ag,
        guest_atoms=universe.atoms[l1_idx],
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


def find_host_anchor(
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
