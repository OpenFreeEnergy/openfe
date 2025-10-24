# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Search methods for generating Geometry objects

TODO
----
* Add relevant duecredit entries.
"""

import warnings
from itertools import combinations, groupby
from typing import Optional, Union

import MDAnalysis as mda
import networkx as nx
import numpy as np
import numpy.typing as npt

# from gufe.vendor.openff.models.types import ArrayQuantity  TODO: write a custom quantity to replace this in pydantic v2
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.dssp import DSSP
from MDAnalysis.analysis.rms import RMSF
from MDAnalysis.lib.distances import capped_distance, distance_array, minimize_vectors
from MDAnalysis.transformations.nojump import NoJump
from openfe_analysis.transformations import Aligner
from openff.toolkit import Molecule as OFFMol
from openff.units import Quantity, unit
from rdkit import Chem
from scipy.stats import circvar

DEFAULT_ANGLE_FRC_CONSTANT = 83.68 * unit.kilojoule_per_mole / unit.radians**2


def _get_mda_selection(
    universe: Union[mda.Universe, mda.AtomGroup],
    atom_list: Optional[list[int]] = None,
    selection: Optional[str] = None,
) -> mda.AtomGroup:
    """
    Return an AtomGroup based on either a list of atom indices or an
    mdanalysis string selection.

    Parameters
    ----------
    universe : Union[mda.Universe, mda.AtomGroup]
      The MDAnalysis Universe or AtomGroup to get the AtomGroup from.
    atom_list : Optional[list[int]]
      A list of atom indices.
    selection : Optional[str]
      An MDAnalysis selection string.

    Returns
    -------
    ag : mda.AtomGroup
      An atom group selected from the inputs.

    Raises
    ------
    ValueError
      If both ``atom_list`` and ``selection`` are ``None``
      or are defined.
    """
    if atom_list is None:
        if selection is None:
            raise ValueError("one of either the atom lists or selections must be defined")

        ag = universe.select_atoms(selection)
    else:
        if selection is not None:
            raise ValueError("both atom_list and selection cannot be defined together")
        ag = universe.atoms[atom_list]
    return ag


def get_aromatic_rings(rdmol: Chem.Mol) -> list[set[int]]:
    """
    Get a list of tuples with the indices for each ring in an rdkit Molecule.

    Parameters
    ----------
    rdmol : Chem.Mol
      RDKit Molecule

    Returns
    -------
    list[set[[int]]
      List of tuples for each ring.
    """

    ringinfo = rdmol.GetRingInfo()
    arom_idxs = get_aromatic_atom_idxs(rdmol)

    aromatic_rings = []

    # Add to the aromatic_rings list if all the atoms in a ring are aromatic
    for ring in ringinfo.AtomRings():
        if all(a in arom_idxs for a in ring):
            aromatic_rings.append(set(ring))

    # Reduce the ring list by merging any rings that have colliding atoms
    for x, y in combinations(aromatic_rings, 2):
        if not x.isdisjoint(y):
            x.update(y)
            aromatic_rings.remove(y)

    return aromatic_rings


def get_aromatic_atom_idxs(rdmol: Chem.Mol) -> list[int]:
    """
    Helper method to get aromatic atoms idxs
    in a RDKit Molecule

    Parameters
    ----------
    rdmol : Chem.Mol
      RDKit Molecule

    Returns
    -------
    list[int]
      A list of the aromatic atom idxs
    """
    idxs = [at.GetIdx() for at in rdmol.GetAtoms() if at.GetIsAromatic()]
    return idxs


def get_heavy_atom_idxs(rdmol: Chem.Mol) -> list[int]:
    """
    Get idxs of heavy atoms in an RDKit Molecule

    Parameters
    ----------
    rmdol : Chem.Mol

    Returns
    -------
    list[int]
      A list of heavy atom idxs
    """
    idxs = [at.GetIdx() for at in rdmol.GetAtoms() if at.GetAtomicNum() > 1]
    return idxs


def get_central_atom_idx(rdmol: Chem.Mol) -> int:
    """
    Get the central atom in an rdkit Molecule.

    Parameters
    ----------
    rdmol : Chem.Mol
      RDKit Molecule to query

    Returns
    -------
    int
      Index of central atom in Molecule

    Note
    ----
    If there are equal likelihood centers, will return
    the first entry.
    """
    # TODO: switch to a manual conversion to avoid an OpenFF dependency
    offmol = OFFMol(rdmol, allow_undefined_stereo=True)
    nx_mol = offmol.to_networkx()

    if not nx.is_weakly_connected(nx_mol.to_directed()):
        errmsg = "A disconnected molecule was passed, cannot find the center"
        raise ValueError(errmsg)

    # Get a list of all shortest paths
    # Note: we call dict on shortest_path to support py3.10 which doesn't
    # support networkx 3.5
    shortest_paths = [
        path
        for node_paths in dict(nx.shortest_path(nx_mol)).values()
        for path in node_paths.values()
    ]

    # Get the longest of these paths (returns first instance)
    longest_path = max(shortest_paths, key=len)

    # Return the index of the central atom
    return longest_path[len(longest_path) // 2]


def is_collinear(
    positions: npt.NDArray,
    atoms: Union[list[int], tuple[int, ...]],
    dimensions=None,
    threshold=0.9,
):
    """
    Check whether any sequential vectors in a sequence of atoms are collinear.

    Approach: for each sequential set of 3 atoms (defined as A, B, and C),
    calculates the normalized inner product (i.e. cos^-1(angle)) between
    vectors AB and BC. If the absolute value  of this inner product is
    close to 1 (i.e. an angle of 0 radians), then the three atoms are
    considered as collinear. You can use ``threshold`` to define how close
    to 1 is considered "flat".

    Parameters
    ----------
    positions : npt.NDArray
      System positions.
    atoms : list[int]
      The indices of the atoms to test.
    dimensions : Optional[npt.NDArray]
      The dimensions of the system to minimize vectors.
    threshold : float
      Atoms are not collinear if their sequential vector separation dot
      products are less than ``threshold``. Default 0.9.

    Returns
    -------
    result : bool
        Returns True if any sequential pair of vectors is collinear;
        False otherwise.

    Notes
    -----
    Originally from Yank.
    """
    if len(atoms) < 3:
        raise ValueError("Too few atoms passed for co-linearity test")
    if len(positions) < len(atoms) or len(positions) < max(atoms) + 1:
        errmsg = "atoms indices do not match the positions array passed"
        raise ValueError(errmsg)
    if not all(isinstance(x, int) for x in atoms):
        errmsg = "atoms is not a list of index integers"
        raise ValueError(errmsg)

    result = False
    for i in range(len(atoms) - 2):
        v1 = positions[atoms[i + 1], :] - positions[atoms[i], :]
        v2 = positions[atoms[i + 2], :] - positions[atoms[i + 1], :]
        if dimensions is not None:
            v1 = minimize_vectors(v1, box=dimensions)
            v2 = minimize_vectors(v2, box=dimensions)

        normalized_inner_product = np.dot(v1, v2) / np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))
        result = result or (np.abs(normalized_inner_product) > threshold)
    return result


def _wrap_angle(angle: Quantity) -> Quantity:
    """
    Wrap an angle to -pi to pi radians.

    Parameters
    ----------
    angle : openff.units.Quantity
      An angle in radians compatible units.

    Returns
    -------
    openff.units.Quantity
      The angle in units of radians wrapped.

    Notes
    -----
    Pint automatically converts the angle to radians
    as it passes it through arctan2.
    """
    return np.arctan2(np.sin(angle), np.cos(angle))  # type: ignore


def check_angle_not_flat(
    angle: Quantity,
    force_constant: Quantity = DEFAULT_ANGLE_FRC_CONSTANT,
    temperature: Quantity = 298.15 * unit.kelvin,
) -> bool:
    """
    Check whether the chosen angle is less than 10 kT from 0 or pi radians

    Parameters
    ----------
    angle : openff.units.Quantity
      The angle to check in units compatible with radians.
    force_constant : openff.units.Quantity
      Force constant of the angle in units compatible with
      kilojoule_per_mole / radians ** 2.
    temperature : openff.units.Quantity
      The system temperature in units compatible with Kelvin.

    Returns
    -------
    bool
      False if the angle is less than 10 kT from 0 or pi radians

    Note
    ----
    We assume the temperature to be 298.15 Kelvin.

    Acknowledgements
    ----------------
    This code was initially contributed by Vytautas Gapsys.
    """
    # Convert things
    angle_rads = _wrap_angle(angle)
    frc_const = force_constant.to("unit.kilojoule_per_mole / unit.radians**2")
    temp_kelvin = temperature.to("kelvin")
    RT = 8.31445985 * 0.001 * temp_kelvin  # type: ignore[operator]

    # check if angle is <10kT from 0 or 180
    check1 = 0.5 * frc_const * np.power((angle_rads - 0.0), 2)  # type: ignore[operator]
    check2 = 0.5 * frc_const * np.power((angle_rads - np.pi), 2)  # type: ignore[operator]
    ang_check_1 = check1 / RT
    ang_check_2 = check2 / RT
    if ang_check_1.m < 10.0 or ang_check_2.m < 10.0:
        return False
    return True


def check_dihedral_bounds(
    dihedral: Quantity,
    lower_cutoff: Quantity = -2.618 * unit.radians,
    upper_cutoff: Quantity = 2.618 * unit.radians,
) -> bool:
    """
    Check that a dihedral does not exceed the bounds set by
    lower_cutoff and upper_cutoff on a -pi to pi range.

    All angles and cutoffs are wrapped to -pi to pi before
    applying the check.

    Parameters
    ----------
    dihedral : openff.units.Quantity
      Dihedral in units compatible with radians.
    lower_cutoff : openff.units.Quantity
      Dihedral lower cutoff in units compatible with radians.
    upper_cutoff : openff.units.Quantity
      Dihedral upper cutoff in units compatible with radians.

    Returns
    -------
    bool
      ``True`` if the dihedral is within the upper and lower
      cutoff bounds.
    """
    dihed = _wrap_angle(dihedral)
    lower = _wrap_angle(lower_cutoff)
    upper = _wrap_angle(upper_cutoff)
    if (dihed < lower) or (dihed > upper):  # type: ignore[operator]
        return False
    return True


def check_angular_variance(
    angles: Quantity,
    upper_bound: Quantity,
    lower_bound: Quantity,
    width: Quantity,
) -> bool:
    """
    Check that the variance of a list of ``angles`` does not exceed
    a given ``width``

    Parameters
    ----------
    angles : ArrayLike openff.units.Quantity
      An array of angles in units compatible with radians.
    upper_bound: openff.units.Quantity
      The upper bound in the angle range in radians compatible units.
    lower_bound: openff.units.Quantity
      The lower bound in the angle range in radians compatible units.
    width : openff.units.Quantity
      The width to check the variance against, in units compatible with
      radians.

    Returns
    -------
    bool
      ``True`` if the variance of the angles is less than the width.

    """
    # scipy circ methods already recasts internally so we shouldn't
    # need to wrap the angles
    variance = circvar(
        angles.to("radians").m,
        high=upper_bound.to("radians").m,
        low=lower_bound.to("radians").m,
    )
    return not (variance * unit.radians > width)


class CentroidDistanceSort(AnalysisBase):
    """
    Sort (from shortest to longest) an input AtomGroup
    based on their distance from the center of geometry
    of another AtomGroup.

    Parameters
    ----------
    sortable_atoms : MDAnalysis.AtomGroup
      AtomGroup to sort based on distance to center of geometry of
      ``reference_atoms``.
    reference_atoms : MDAnalysis.AtomGroup
      AtomGroup who's center of geometry will be used to distance sort
      ``sortable_atoms`` with.

    Attributes
    ----------
    results.distances : np.array
      A numpy array of the distances from the centroid of
      ``reference_atoms`` for each frame.
    results.sorted_atomgroup : MDAnalysis.AtomGroup
      A copy of ``sortable_atoms`` where the atoms are sorted by
      their distance from the centroid of ``reference_atoms``.
    """

    _analysis_algorithm_is_parallelizable = False

    def __init__(
        self,
        sortable_atoms,
        reference_atoms,
        **kwargs,
    ):
        super().__init__(sortable_atoms.universe.trajectory, **kwargs)

        def get_atomgroup(ag):
            """
            We need this in case someone passes an Atom not an AG
            """
            if ag._is_group:
                return ag
            return mda.AtomGroup([ag])

        self.sortable_ag = get_atomgroup(sortable_atoms)
        self.reference_ag = get_atomgroup(reference_atoms)

    def _prepare(self):
        self.results.distances = np.zeros((self.n_frames, len(self.sortable_ag)))

    def _single_frame(self):
        self.results.distances[self._frame_index] = distance_array(
            self.reference_ag.center_of_geometry(),
            self.sortable_ag.atoms.positions,
            box=self.reference_ag.dimensions,
        )

    def _conclude(self):
        idxs = np.argsort(np.mean(self.results.distances, axis=0))
        self.results.sorted_atomgroup = self.sortable_ag.atoms[idxs]


class FindHostAtoms(AnalysisBase):
    """
    Class filter host atoms based on their distance
    from a set of guest atoms.

    Parameters
    ----------
    host_atoms : MDAnalysis.AtomGroup
      Initial selection of host atoms to filter from.
    guest_atoms : MDAnalysis.AtomGroup
      Selection of guest atoms to search around.
    min_search_distance: openff.units.Quantity
      Minimum distance to filter atoms within.
    max_search_distance: openff.units.Quantity
      Maximum distance to filter atoms within.

    Attributes
    ----------
    results.host_idxs : np.ndarray
      A NumPy array of host indices in the Universe.
    """

    _analysis_algorithm_is_parallelizable = False

    def __init__(
        self,
        host_atoms,
        guest_atoms,
        min_search_distance,
        max_search_distance,
        **kwargs,
    ):
        super().__init__(host_atoms.universe.trajectory, **kwargs)

        def get_atomgroup(ag):
            if ag._is_group:
                return ag
            return mda.AtomGroup([ag])

        self.host_ag = get_atomgroup(host_atoms)
        self.guest_ag = get_atomgroup(guest_atoms)
        self.min_cutoff = min_search_distance.to("angstrom").m
        self.max_cutoff = max_search_distance.to("angstrom").m

    def _prepare(self):
        self.results.host_idxs = set(self.host_ag.atoms.ix)

    def _single_frame(self):
        pairs = capped_distance(
            reference=self.guest_ag.positions,
            configuration=self.host_ag.positions,
            max_cutoff=self.max_cutoff,
            min_cutoff=self.min_cutoff,
            box=self.guest_ag.universe.dimensions,
            return_distances=False,
        )

        host_idxs = set(self.host_ag.atoms[p].ix for p in pairs[:, 1])

        # We do an intersection as we go along to prune atoms that don't pass
        # the distance selection criteria
        self.results.host_idxs = self.results.host_idxs.intersection(host_idxs)

    def _conclude(self):
        self.results.host_idxs = np.array(list(self.results.host_idxs))


# TODO: needs custom type https://github.com/OpenFreeEnergy/openfe/issues/1569
def get_local_rmsf(atomgroup: mda.AtomGroup):  # -> ArrayQuantity:
    """
    Get the RMSF of an AtomGroup when aligned upon itself.

    Parameters
    ----------
    atomgroup : MDAnalysis.AtomGroup

    Return
    ------
    rmsf : openff.units.Quantity
      ArrayQuantity of RMSF values.
    """
    # The no trajectory case
    if len(atomgroup.universe.trajectory) < 2:
        return np.zeros(atomgroup.n_atoms) * unit.angstrom

    # First let's copy our Universe
    copy_u = atomgroup.universe.copy()
    ag = copy_u.atoms[atomgroup.atoms.ix]

    # Reset the trajectory index, otherwise we'll get in trouble with nojump
    copy_u.trajectory[0]

    nojump = NoJump()
    align = Aligner(ag)
    copy_u.trajectory.add_transformations(nojump, align)

    rmsf = RMSF(ag)
    rmsf.run()
    return rmsf.results.rmsf * unit.angstrom


def _atomgroup_has_bonds(atomgroup: Union[mda.AtomGroup, mda.Universe]) -> bool:
    """
    Check if all residues in an AtomGroup or Universe have bonds.

    Parameters
    ----------
    atomgroup : Union[mda.Atomgroup, mda.Universe]
      Either an MDAnalysis AtomGroup or Universe to check for bonds.

    Returns
    -------
    bool
      True if all residues contain at least one bond, False otherwise.
    """
    if not hasattr(atomgroup, "bonds"):
        return False

    # Assume that any residue with more than one atom should have a bond
    if not all(len(r.atoms.bonds) > 0 for r in atomgroup.residues if len(r.atoms) > 1):
        return False

    return True


def stable_secondary_structure_selection(
    atomgroup: mda.AtomGroup,
    trim_chain_start: int = 10,
    trim_chain_end: int = 10,
    min_structure_size: int = 6,
    trim_structure_ends: int = 2,
) -> mda.AtomGroup:
    """
    Select all atoms in a given AtomGroup which belong to residues with a
    stable secondary structure as defined by Baumann et al.[1]

    The selection algorithm works in the following manner:
      1. Protein residues are selected from the ``atomgroup``.
      2. If there are fewer than 30 protein residues, raise an error.
      3. Split the protein residues by fragment, guessing bonds if necessary.
      4. Discard the first ``trim_chain_start`` and the last
         ``trim_chain_end`` residues per fragment.
      5. Run DSSP using the last trajectory frame on the remaining
         fragment residues.
      6. Extract all contiguous structure units that are longer than
         ``min_structure_size``, removing ``trim_structure_ends``
         residues from each end of the structure.
      7. For all extract structures, if there are more beta-sheet
         residues than there are alpha-helix residues, then allow
         residues to be selected from either structure type. If not,
         then only allow alpha-helix residues.
      8. Select all atoms in the ``atomgroup`` that belong to residues
         from extracted structure units of the selected structure type.

    Parameters
    ----------
    atomgroup : mda.AtomgGroup
      The AtomGroup to select atoms from.
    trim_chain_start: int
      The number of residues to trim from the start of each
      protein chain. Default 10.
    trim_chain_end : int
      The number of residues to trim from the end of each
      protein chain. Default 10.
    min_structure_size : int
      The minimum number of residues needed in a given
      secondary structure unit to be considered stable. Default 8.
    trim_structure_ends : int
      The number of residues to trim from the end of each
      secondary structure units. Default 3.

    Returns
    -------
    AtomGroup : mda.AtomGroup
      An AtomGroup containing all the atoms from the input AtomGroup
      which belong to stable secondary structure residues.

    Raises
    ------
    UserWarning
      If there are no bonds for the protein atoms in the input
      host residue. In this case, the bonds will be guessed
      using a simple distance metric.

    Notes
    -----
    * This selection algorithm assumes contiguous & ordered residues.
    * We recommend always trimming at least one residue at the ends of
      each chain using ``trim_chain_start`` and ``trim_chain_end`` to
      avoid issues with capping residues.
    * DSSP assignment is done on the final frame of the trajectory.

    References
    ----------
    [1] Baumann, Hannah M., et al. "Broadening the scope of binding free energy
        calculations using a Separated Topologies approach." (2023).
    """
    # First let's copy our Universe so we don't overwrite its current state
    copy_u = atomgroup.universe.copy()

    # Create an AtomGroup that contains all the protein residues in the
    # input Universe - we will filter by what matches in the atomgroup later
    copy_protein_ag = copy_u.select_atoms("protein").atoms

    # We need to split by fragments to account for multiple chains
    # To do this, we need bonds!
    if not _atomgroup_has_bonds(copy_protein_ag):
        wmsg = "No bonds found in input Universe, will attempt to guess them."
        warnings.warn(wmsg)
        copy_protein_ag.guess_bonds()

    structures = []  # container for all contiguous secondary structure units
    # Counter for each residue type found
    structure_residue_counts = {"H": 0, "E": 0, "-": 0}
    # THe minimum length any chain must have
    min_chain_length = trim_chain_start + trim_chain_end + min_structure_size

    # Loop over each continually bonded section (i.e. chain) of the protein
    for frag in copy_protein_ag.fragments:
        # If this fragment is too small, skip processing it
        if len(frag.residues) < min_chain_length:
            continue

        # Trim the chain ends
        chain = frag.residues[trim_chain_start:-trim_chain_end].atoms

        try:
            # Run on the last frame
            # TODO: maybe filter out any residue that changes secondary
            # structure during the trajectory
            dssp = DSSP(chain).run(start=-1)
        except ValueError:
            # DSSP may fail if it doesn't recognise the system's atom names
            # or non-canonical residues are included, in this case just skip
            continue

        # Tag each residue structure by its resindex
        dssp_results = [
            (structure, resid)
            for structure, resid in zip(dssp.results.dssp[0], chain.residues.resindices)
        ]

        # Group by contiguous secondary structure
        for _, group_iter in groupby(dssp_results, lambda x: x[0]):
            group = list(group_iter)
            if len(group) >= min_structure_size:
                structures.append(group[trim_structure_ends:-trim_structure_ends])
                num_residues = len(group) - (2 * trim_structure_ends)
                structure_residue_counts[group[0][0]] += num_residues

    # Pick atoms in both helices and beta sheets
    allowed_structures = ["H", "E"]

    allowed_residxs = []
    for structure in structures:
        if structure[0][0] in allowed_structures:
            allowed_residxs.extend([residue[1] for residue in structure])

    # Resindexes are keyed at the Universe scale not AtomGroup
    allowed_atoms = atomgroup.universe.residues[allowed_residxs].atoms

    # Pick up all the atoms that intersect the initial selection and
    # those allowed.
    return atomgroup.intersection(allowed_atoms)


def protein_chain_selection(
    atomgroup: mda.AtomGroup,
    min_chain_length: int = 30,
    trim_chain_start: int = 10,
    trim_chain_end: int = 10,
) -> mda.AtomGroup:
    """
    Return a sub-selection of the input AtomGroup which belongs to protein
    chains trimmed by ``trim_chain_start`` and ``trim_chain_end``.

    Protein chains are defined as any continuously bonded part of system with
    at least ``min_chain_length`` (default: 30) residues which match the
    ``protein`` selection of MDAnalysis.

    Parameters
    ----------
    atomgroup : mda.AtomgGroup
      The AtomGroup to select atoms from.
    min_chain_length : int
      The minimum number of residues to be considered a protein chain. Default 30.
    trim_chain_start : int
      The number of residues to trim from the start of each
      protein chain. Default 10.
    trim_chain_end : int
      The number of residues to trim from the end of each
      protein chain. Default 10.

    Returns
    -------
    atomgroup : mda.AtomGroup
      An AtomGroup containing all the atoms from the input AtomGroup
      which belong to the trimmed protein chains.
    """
    # First let's copy our Universe so we don't overwrite its current state
    copy_u = atomgroup.universe.copy()

    # Create an AtomGroup that contains all the protein residues in the
    # input Universe - we will filter by what matches in the atomgroup later
    copy_protein_ag = copy_u.select_atoms("protein").atoms

    # We need to split by fragments to account for multiple chains
    # To do this, we need bonds!
    if not _atomgroup_has_bonds(copy_protein_ag):
        wmsg = "No bonds found in input Universe, will attempt to guess them."
        warnings.warn(wmsg)
        copy_protein_ag.guess_bonds()

    copy_chains_ags_list = []

    # Loop over each continually bonded section (i.e. chain) of the protein
    for frag in copy_protein_ag.fragments:
        # If this chain is less than min_chain_length residues, it's probably a peptide
        if len(frag.residues) < min_chain_length:
            continue

        chain = frag.residues[trim_chain_start:-trim_chain_end].atoms
        copy_chains_ags_list.append(chain)

    # If the list is empty, return an empty atomgroup
    if not copy_chains_ags_list:
        return atomgroup.atoms[[]]

    # Create a single atomgroup from all chains
    copy_chains_ag = sum(copy_chains_ags_list)

    # Now get a list of all the chain atoms in the original Universe
    # Resindexes are keyed at the Universe scale not AtomGroup
    chain_atoms = atomgroup.universe.atoms[copy_chains_ag.atoms.ix]

    # Return all atoms at the intersection of the input atomgroup and
    # the chains atomgroup
    return atomgroup.intersection(chain_atoms)
