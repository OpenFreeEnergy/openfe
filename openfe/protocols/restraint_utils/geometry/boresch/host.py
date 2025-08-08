# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Restraint Geometry classes

TODO
----
* Add relevant duecredit entries.
"""
import warnings
from typing import Optional

import MDAnalysis as mda
import numpy as np
import numpy.typing as npt
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.lib.distances import calc_angles, calc_bonds, calc_dihedrals
from openfe.protocols.restraint_utils.geometry.utils import (
    CentroidDistanceSort,
    FindHostAtoms,
    check_angle_not_flat,
    check_angular_variance,
    check_dihedral_bounds,
    get_local_rmsf,
    is_collinear,
    protein_chain_selection,
    stable_secondary_structure_selection,
)
from openff.units import Quantity, unit


def _host_atoms_search(
    atomgroup: mda.AtomGroup,
    guest_anchor_idx: int,
    rmsf_cutoff: Quantity,
    min_search_distance: Quantity,
    max_search_distance: Quantity,
) -> npt.NDArray:
    """
    Helper method to get a set of host atoms with minimal RMSF
    within a given distance of a guest anchor.

    Parameters
    ----------
    atomgroup : mda.AtomGroup
      An AtomGroup to find host atoms in.
    guest_anchor_idx : int
      The index of the proposed guest anchor binding atom.
    rmsf_cutoff : Quantity
      The maximum allowed RMSF value for any candidate host atom.
    min_search_distance : Quantity
      The minimum host atom search distance around the guest anchor.
    max_search_distance : Quantity
      The maximum host atom search distance around the guest anchor.

    Return
    ------
    NDArray
      Array of host atom indexes
    """
    # 0 Deal with the empty case
    if len(atomgroup) == 0:
        return np.array([], dtype=int)

    # 1 Get the RMSF & filter to create a new AtomGroup
    rmsf = get_local_rmsf(atomgroup)
    filtered_atomgroup = atomgroup.atoms[rmsf < rmsf_cutoff]

    # 2. Search for atoms within the min/max cutoff of the guest anchor
    atom_finder = FindHostAtoms(
        host_atoms=filtered_atomgroup,
        guest_atoms=atomgroup.universe.atoms[guest_anchor_idx],
        min_search_distance=min_search_distance,
        max_search_distance=max_search_distance,
    )
    atom_finder.run()

    return atom_finder.results.host_idxs


def find_host_atom_candidates(
    universe: mda.Universe,
    host_idxs: list[int],
    guest_anchor_idx: int,
    host_selection: str,
    dssp_filter: bool = False,
    rmsf_cutoff: Quantity = 0.1 * unit.nanometer,
    min_search_distance: Quantity = 0.5 * unit.nanometer,
    max_search_distance: Quantity = 1.5 * unit.nanometer,
) -> npt.NDArray:
    """
    Get a list of suitable host atoms.

    Parameters
    ----------
    universe : mda.Universe
      An MDAnalysis Universe defining the system and its coordinates.
    host_idxs : list[int]
      A list of the host indices in the system topology.
    guest_anchor_idx : int
      The index of the proposed l1 binding atom.
    host_selection : str
      An MDAnalysis selection string to filter the host by.
    dssp_filter : bool
      Whether or not to apply a DSSP filter on the host selection.
    rmsf_cutoff : openff.units.Quantity
      The maximum RMSF value allowed for any candidate host atom.
    min_search_distance : openff.units.Quantity
      The minimum search distance around l1 for suitable candidate atoms.
    max_search_distance : openff.units.Quantity
      The maximum search distance around l1 for suitable candidate atoms.

    Return
    ------
    NDArray
      Array of host atom indexes sorted by distance from `guest_anchor_idx`
    """
    # Get an AtomGroup for the host based on the input host indices
    host_ag = universe.atoms[host_idxs]

    # Filter the host AtomGroup based on ``host_selection`
    selected_host_ag = host_ag.select_atoms(host_selection)

    # If the host_selection does not work, raise an error
    if len(selected_host_ag) < 3:
        errmsg = (
            "Boresch-like restraint generation: "
            f"too few atoms selected by ``host_selection``: {host_selection}"
        )
        raise ValueError(errmsg)

    # None filtered_host_ixs for condition checking later
    filtered_host_idxs = None

    # If requested, filter the host atoms based on if their residues exist
    # within stable secondary structures.
    if dssp_filter:
        # TODO: allow more user-supplied kwargs here
        filtered_host_idxs = _host_atoms_search(
            atomgroup=stable_secondary_structure_selection(selected_host_ag),
            guest_anchor_idx=guest_anchor_idx,
            rmsf_cutoff=rmsf_cutoff,
            min_search_distance=min_search_distance,
            max_search_distance=max_search_distance,
        )

        if len(filtered_host_idxs) < 20:
            wmsg = (
                "Restraint generation: DSSP filter found too few host atoms "
                f"({len(filtered_host_idxs)} found). Will attempt to use all protein chains."
            )
            warnings.warn(wmsg)
            filtered_host_idxs = _host_atoms_search(
                atomgroup=protein_chain_selection(selected_host_ag),
                guest_anchor_idx=guest_anchor_idx,
                rmsf_cutoff=rmsf_cutoff,
                min_search_distance=min_search_distance,
                max_search_distance=max_search_distance,
            )

        if len(filtered_host_idxs) < 20:
            wmsg = (
                "Restraint generation: protein chain filter found too few "
               f"host atoms ({len(filtered_host_idxs)} found). Will attempt to use all host atoms in "
                f"selection: {host_selection}."
            )
            warnings.warn(wmsg)
            filtered_host_idxs = None

    if filtered_host_idxs is None:
        filtered_host_idxs = _host_atoms_search(
            atomgroup=selected_host_ag,
            guest_anchor_idx=guest_anchor_idx,
            rmsf_cutoff=rmsf_cutoff,
            min_search_distance=min_search_distance,
            max_search_distance=max_search_distance,
        )

    # Crash out if no atoms were found
    if len(filtered_host_idxs) == 0:
        errmsg = (
            f"No host atoms found within the search distance "
            f"{min_search_distance}-{max_search_distance}. Consider widening the search window."
        )
        raise ValueError(errmsg)

    # Now we sort them by their distance from the guest anchor
    atom_sorter = CentroidDistanceSort(
        sortable_atoms=universe.atoms[filtered_host_idxs],
        reference_atoms=universe.atoms[guest_anchor_idx],
    )
    atom_sorter.run()

    return atom_sorter.results.sorted_atomgroup.ix


class EvaluateBoreschAtoms(AnalysisBase):
    """
    Class to evaluate the suitability of the atoms in a Boresch
    restraint.

    Parameters
    ----------
    restraint : MDAnalysis.AtomGroup
      An AtomGroup defining the H2-H1-H0-G0-G1-G2 atoms, in that
      order.
    angle_force_constant : openff.units.Quantity
      The force constant for the angle.
    temperature : openff.units.Quanity
      The system temperature in units compatible with Kelvin.
    """
    def __init__(
        self,
        restraint: mda.AtomGroup,
        angle_force_constant: Quantity,
        temperature: Quantity,
        **kwargs,
    ):
        super().__init__(restraint.universe.trajectory, **kwargs)

        if len(restraint) != 6:
            errmsg = "Incorrect number of restraint atoms passed"
            raise ValueError(errmsg)

        self.restraint = restraint
        self.angle_force_constant = angle_force_constant
        self.temperature = temperature

    def _prepare(self):
        # Whether or not the restraint is valid
        self.results.valid = True
        # Containers
        self.results.collinear = np.empty(self.n_frames, dtype=bool)
        self.results.angles = np.zeros((2, self.n_frames))
        self.results.dihedrals = np.zeros((3, self.n_frames))

    def _single_frame(self):

        self.results.collinear[self._frame_index] = is_collinear(
            positions=self.restraint.positions,
            atoms=[0, 1, 2, 3, 4, 5],
            dimensions=self.restraint.dimensions,
        )

        # angles
        for i in range(2):
            self.results.angles[i, self._frame_index] = calc_angles(
                self.restraint.atoms[i].position,
                self.restraint.atoms[i+1].position,
                self.restraint.atoms[i+2].position,
                box=self.restraint.dimensions,
            )

        # dihedrals
        for i in range(3):
            self.results.dihedrals[i, self._frame_index] = calc_dihedrals(
                self.restraint.atoms[i].position,
                self.restraint.atoms[i+1].position,
                self.restraint.atoms[i+2].position,
                self.restraint.atoms[i+3].position,
                box=self.restraint.dimensions,
            )

    def _conclude(self):
        # Check angles
        angle_bounds = True
        angle_variance = True
        for i in range(2):
            bounds = all(
                check_angle_not_flat(
                    angle=angle * unit.radians,
                    force_constant=self.angle_force_constant,
                    temperature=self.temperature,
                )
                for angle in self.results.angles[i]
            )
            variance = check_angular_variance(
                self.results.angles[i] * unit.radians,
                upper_bound=np.pi * unit.radians,
                lower_bound=0 * unit.radians,
                width=1.745 * unit.radians,
            )
            angle_bounds &= bounds
            angle_variance &= variance

        # Check dihedrals
        dihed_bounds = True
        dihed_variance = True
        for i in range(3):
            bounds = all(
                check_dihedral_bounds(dihed * unit.radians)
                for dihed in self.results.dihedrals[i]
            )
            variance = check_angular_variance(
                self.results.dihedrals[i] * unit.radians,
                upper_bound=np.pi * unit.radians,
                lower_bound=-np.pi * unit.radians,
                width=5.23 * unit.radians,
            )

            dihed_bounds &= bounds
            dihed_variance &= variance

        not_collinear = not all(self.results.collinear)

        self.results.valid = all(
            [
                angle_bounds,
                angle_variance,
                dihed_bounds,
                dihed_variance,
                not_collinear,
            ]
        )


class EvaluateHostAtoms1(AnalysisBase):
    """
    Class to evaluate the suitability of a set of host atoms
    as either H0 or H1 atoms (i.e. the first and second host atoms).

    Parameters
    ----------
    reference : MDAnalysis.AtomGroup
      The reference preceding three atoms.
    host_atom_pool : MDAnalysis.AtomGroup
      The pool of atoms to pick an atom from.
    minimum_distance : openff.units.Quantity
      The minimum distance from the bound reference atom.
    angle_force_constant : openff.units.Quantity
      The force constant for the angle.
    temperature : openff.units.Quantity
      The system temperature in units compatible with Kelvin
    """

    def __init__(
        self,
        reference: mda.AtomGroup,
        host_atom_pool: mda.AtomGroup,
        minimum_distance: Quantity,
        angle_force_constant: Quantity,
        temperature: Quantity,
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
        self.results.distances = np.zeros((len(self.host_atom_pool), self.n_frames))
        self.results.angles = np.zeros((len(self.host_atom_pool), self.n_frames))
        self.results.dihedrals = np.zeros((len(self.host_atom_pool), self.n_frames))
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
            distance_bounds = all(self.results.distances[i] > self.minimum_distance)
            # Check angles
            angle_bounds = all(
                check_angle_not_flat(
                    angle=angle * unit.radians,
                    force_constant=self.angle_force_constant,
                    temperature=self.temperature,
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
      The reference preceding three atoms.
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
            distance1_bounds = all(self.results.distances1[i] > self.minimum_distance)
            distance2_bounds = all(self.results.distances2[i] > self.minimum_distance)
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


def find_host_anchor_bonded(
    guest_atoms: mda.AtomGroup,
    host_atom_pool: mda.AtomGroup,
    guest_minimum_distance: Quantity,
    angle_force_constant: Quantity,
    temperature: Quantity,
) -> list[int] | None:
    """
    Find suitable atoms for the H0-H1-H2 portion of the restraint
    where all host atoms are bonded to each other.

    Parameters
    ----------
    guest_atoms : mda.AtomGroup
      The guest anchor atoms for G0-G1-G2
    host_atom_pool : mda.AtomGroup
      The host atoms to search from.
    guest_minimum_distance: openff.units.Quantity
      The minimum distance between host atoms and the guest anchor.
    angle_force_constant : openff.units.Quantity
      The force constant for the G1-G0-H0 and G0-H0-H1 angles.
    temperature : openff.units.Quantity
      The target system temperature.

    Returns
    -------
    Optional[list[int]]
      A list of indices for a selected combination of H0, H1, and H2.
    """
    if not hasattr(guest_atoms, 'angles'):
        warnings.warn("no angles found - will attempt to guess")
        guest_atoms.universe.guess_TopologyAttrs(context='default', to_guess=['angles'])

    # Evaluate the host_atom_pool for suitability as H0 atoms
    h0_eval = EvaluateHostAtoms1(
        reference=guest_atoms,
        host_atom_pool=host_atom_pool,
        minimum_distance=guest_minimum_distance,
        angle_force_constant=angle_force_constant,
        temperature=temperature,
    ).run()

    for i, valid_h0 in enumerate(h0_eval.results.valid):
        # If valid H0 atom, get all the angles it's involved in.
        if valid_h0:
            # note: i indexes host_atom_pool but not the universe!
            # from here on, we will switch to using atom.ix instead of i
            atom = host_atom_pool.atoms[i]
            angles = atom.angles.atomgroup_intersection(host_atom_pool, strict=True)
            for indices in angles.indices:
                if atom.ix == indices[0] or atom.ix == indices[-1]:
                    if atom.ix == indices[0]:
                        indices = indices[::-1]
                else:
                    continue

                restraint_atoms = host_atom_pool.universe.atoms[indices] + guest_atoms

                restraint_eval = EvaluateBoreschAtoms(
                    restraint=restraint_atoms,
                    angle_force_constant=angle_force_constant,
                    temperature=temperature,
                ).run()

                if restraint_eval.results.valid:
                    # reverse indices to get H0, H1, H2
                    return [i for i in indices[::-1]]


def find_host_anchor_multi(
    guest_atoms: mda.AtomGroup,
    host_atom_pool: mda.AtomGroup,
    host_minimum_distance: Quantity,
    guest_minimum_distance: Quantity,
    angle_force_constant: Quantity,
    temperature: Quantity,
) -> Optional[list[int]]:
    """
    Find suitable atoms for the H0-H1-H2 portion of the restraint.

    Parameters
    ----------
    guest_atoms : mda.AtomGroup
      The guest anchor atoms for G0-G1-G2
    host_atom_pool : mda.AtomGroup
      The host atoms to search from.
    host_minimum_distance : openff.units.Quantity
      The minimum distance to pick host atoms from each other.
    guest_minimum_distance: openff.units.Quantity
      The minimum distance between host atoms and the guest anchor.
    angle_force_constant : openff.units.Quantity
      The force constant for the G1-G0-H0 and G0-H0-H1 angles.
    temperature : openff.units.Quantity
      The target system temperature.

    Returns
    -------
    Optional[list[int]]
      A list of indices for a selected combination of H0, H1, and H2.
    """
    # Evaluate the host_atom_pool for suitability as H0 atoms
    h0_eval = EvaluateHostAtoms1(
        reference=guest_atoms,
        host_atom_pool=host_atom_pool,
        minimum_distance=guest_minimum_distance,
        angle_force_constant=angle_force_constant,
        temperature=temperature,
    )
    h0_eval.run()

    for i, valid_h0 in enumerate(h0_eval.results.valid):
        # If valid H0 atom, evaluate rest of host_atom_pool for suitability
        # as H1 atoms.
        if valid_h0:
            h0g0g1_atoms = host_atom_pool.atoms[i] + guest_atoms.atoms[:2]
            h1_eval = EvaluateHostAtoms1(
                reference=h0g0g1_atoms,
                host_atom_pool=host_atom_pool,
                minimum_distance=host_minimum_distance,
                angle_force_constant=angle_force_constant,
                temperature=temperature,
            )
            h1_eval.run()
            for j, valid_h1 in enumerate(h1_eval.results.valid):
                # If valid H1 atom, evaluate rest of host_atom_pool for
                # suitability as H2 atoms
                if valid_h1:
                    h1h0g0_atoms = host_atom_pool.atoms[j] + h0g0g1_atoms.atoms[:2]
                    h2_eval = EvaluateHostAtoms2(
                        reference=h1h0g0_atoms,
                        host_atom_pool=host_atom_pool,
                        minimum_distance=host_minimum_distance,
                        angle_force_constant=angle_force_constant,
                        temperature=temperature,
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
                            (idx, val)
                            for idx, val in enumerate(dsum_avgs)
                            if h2_eval.results.valid[idx]
                        ]

                        # Get the index of the H2 atom with the lowest
                        # average distance
                        k = sorted(h2_dsum_avgs, key=lambda x: x[1])[0][0]

                        return list(host_atom_pool.atoms[[i, j, k]].ix)
    return None
