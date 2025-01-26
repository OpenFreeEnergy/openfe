# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Search methods for generating Geometry objects

TODO
----
* Add relevant duecredit entries.
"""
from typing import Union, Optional
from itertools import combinations, groupby
import numpy as np
import numpy.typing as npt
from scipy.stats import circvar
import warnings

from openff.toolkit import Molecule as OFFMol
from openff.units import unit
import networkx as nx
from rdkit import Chem
import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.rms import RMSF
from MDAnalysis.analysis.dssp import DSSP
from MDAnalysis.lib.distances import minimize_vectors, capped_distance
from MDAnalysis.transformations.nojump import NoJump

from openfe_analysis.transformations import Aligner


DEFAULT_ANGLE_FRC_CONSTANT = 83.68 * unit.kilojoule_per_mole / unit.radians**2


def _get_mda_selection(
    universe: mda.Universe,
    atom_list: Optional[list[int]],
    selection: Optional[str]
) -> mda.AtomGroup:
    """
    Return an AtomGroup based on either a list of atom indices or an
    mdanalysis string selection.

    Parameters
    ----------
    universe : mda.Universe
      The MDAnalysis Universe to get the AtomGroup from.
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
            raise ValueError(
                "one of either the atom lists or selections must be defined"
            )

        ag = universe.select_atoms(selection)
    else:
        if selection is not None:
            raise ValueError(
                "both atom_list and selection cannot be defined together"
            )
        ag = universe.atoms[atom_list]
    return ag


def get_aromatic_rings(rdmol: Chem.Mol) -> list[tuple[int, ...]]:
    """
    Get a list of tuples with the indices for each ring in an rdkit Molecule.

    Parameters
    ----------
    rdmol : Chem.Mol
      RDKit Molecule

    Returns
    -------
    list[tuple[int]]
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
      RDKit Molcule to query

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
    shortest_paths = [
        path
        for node_paths in nx.shortest_path(nx_mol).values()
        for path in node_paths.values()
    ]

    # Get the longest of these paths (returns first instance)
    longest_path = max(shortest_paths, key=len)

    # Return the index of the central atom
    return longest_path[len(longest_path) // 2]


def is_collinear(
    positions: npt.ArrayLike,
    atoms: list[int],
    dimensions=None,
    threshold=0.9
):
    """
    Check whether any sequential vectors in a sequence of atoms are collinear.

    Parameters
    ----------
    positions : npt.ArrayLike
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
        v1 = minimize_vectors(
            positions[atoms[i + 1], :] - positions[atoms[i], :],
            box=dimensions,
        )
        v2 = minimize_vectors(
            positions[atoms[i + 2], :] - positions[atoms[i + 1], :],
            box=dimensions,
        )
        normalized_inner_product = np.dot(v1, v2) / np.sqrt(
            np.dot(v1, v1) * np.dot(v2, v2)
        )
        result = result or (np.abs(normalized_inner_product) > threshold)
    return result


def check_angle_not_flat(
    angle: unit.Quantity,
    force_constant: unit.Quantity = DEFAULT_ANGLE_FRC_CONSTANT,
    temperature: unit.Quantity = 298.15 * unit.kelvin,
) -> bool:
    """
    Check whether the chosen angle is less than 10 kT from 0 or pi radians

    Parameters
    ----------
    angle : unit.Quantity
      The angle to check in units compatible with radians.
    force_constant : unit.Quantity
      Force constant of the angle in units compatible with
      kilojoule_per_mole / radians ** 2.
    temperature : unit.Quantity
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
    angle_rads = angle.to("radians")
    frc_const = force_constant.to("unit.kilojoule_per_mole / unit.radians**2")
    temp_kelvin = temperature.to("kelvin")
    RT = 8.31445985 * 0.001 * temp_kelvin

    # check if angle is <10kT from 0 or 180
    check1 = 0.5 * frc_const * np.power((angle_rads - 0.0), 2)
    check2 = 0.5 * frc_const * np.power((angle_rads - np.pi), 2)
    ang_check_1 = check1 / RT
    ang_check_2 = check2 / RT
    if ang_check_1.m < 10.0 or ang_check_2.m < 10.0:
        return False
    return True


def check_dihedral_bounds(
    dihedral: unit.Quantity,
    lower_cutoff: unit.Quantity = -2.618 * unit.radians,
    upper_cutoff: unit.Quantity = 2.618 * unit.radians,
) -> bool:
    """
    Check that a dihedral does not exceed the bounds set by
    lower_cutoff and upper_cutoff.

    Parameters
    ----------
    dihedral : unit.Quantity
      Dihedral in units compatible with radians.
    lower_cutoff : unit.Quantity
      Dihedral lower cutoff in units compatible with radians.
    upper_cutoff : unit.Quantity
      Dihedral upper cutoff in units compatible with radians.

    Returns
    -------
    bool
      ``True`` if the dihedral is within the upper and lower
      cutoff bounds.
    """
    if (dihedral < lower_cutoff) or (dihedral > upper_cutoff):
        return False
    return True


def check_angular_variance(
    angles: unit.Quantity,
    upper_bound: unit.Quantity,
    lower_bound: unit.Quantity,
    width: unit.Quantity,
) -> bool:
    """
    Check that the variance of a list of ``angles`` does not exceed
    a given ``width``

    Parameters
    ----------
    angles : ArrayLike[unit.Quantity]
      An array of angles in units compatible with radians.
    upper_bound: unit.Quantity
      The upper bound in the angle range in radians compatible units.
    lower_bound: unit.Quantity
      The lower bound in the angle range in radians compatible units.
    width : unit.Quantity
      The width to check the variance against, in units compatible with
      radians.

    Returns
    -------
    bool
      ``True`` if the variance of the angles is less than the width.

    """
    variance = circvar(
        angles.to("radians").m,
        high=upper_bound.to("radians").m,
        low=lower_bound.to("radians").m,
    )
    return not (variance * unit.radians > width)


class FindHostAtoms(AnalysisBase):
    """
    Class filter host atoms based on their distance
    from a set of guest atoms.

    Parameters
    ----------
    host_atoms : MDAnalysis.AtomGroup
      Initial selection of host atoms to filter from.
    guest_atoms : MDANalysis.AtomGroup
      Selection of guest atoms to search around.
    min_search_distance: unit.Quantity
      Minimum distance to filter atoms within.
    max_search_distance: unit.Quantity
      Maximum distance to filter atoms within.
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
        self.results.host_idxs = self.results.host_idxs.intersection(
            host_idxs
        )

    def _conclude(self):
        self.results.host_idxs = np.array(list(self.results.host_idxs))


def get_local_rmsf(atomgroup: mda.AtomGroup) -> unit.Quantity:
    """
    Get the RMSF of an AtomGroup when aligned upon itself.

    Parameters
    ----------
    atomgroup : MDAnalysis.AtomGroup

    Return
    ------
    rmsf
      ArrayQuantity of RMSF values.
    """
    # First let's copy our Universe
    copy_u = atomgroup.universe.copy()
    ag = copy_u.atoms[atomgroup.atoms.ix]

    nojump = NoJump()
    align = Aligner(ag)

    copy_u.trajectory.add_transformations(nojump, align)

    rmsf = RMSF(ag)
    rmsf.run()
    return rmsf.results.rmsf * unit.angstrom


def _atomgroup_has_bonds(
    atomgroup: Union[mda.Atomgroup, mda.Universe]
) -> bool:
    """
    Check if all residues in an AtomGroup or Univese has bonds.

    Parameters
    ----------
    atomgroup : Union[mda.Atomgroup, mda.Universe]
      Either an MDAnalysis AtomGroup or Universe to check for bonds.

    Returns
    -------
    bool
      True if all residues contain at least one bond, False otherwise.
    """
    if not hasattr(atomgroup, 'bonds'):
        return False

    if not all(len(r.atoms.bonds) > 0 for r in atomgroup.residues):
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
    * DSSP assignement is done on the final frame of the trajectory.

    References
    ----------
    [1] Baumann, Hannah M., et al. "Broadening the scope of binding free energy
        calculations using a Separated Topologies approach." (2023).
    """
    # First let's copy our Universe so we don't overwrite its current state
    copy_u = atomgroup.universe.copy()

    # Create an AtomGroup that contains all the protein residues in the
    # input Universe - we will filter by what matches in the atomgroup later
    copy_protein_ag = copy_u.select_atoms('protein').atoms

    # We need to split by fragments to account for multiple chains
    # To do this, we need bonds!
    if not _atomgroup_has_bonds(copy_protein_ag, 'bonds'):
        wmsg = "No bonds found in input Universe, will attept to guess them."
        warnings.warn(wmsg)
        protein_ag.guess_bonds()

    structures = []  # container for all contiguous secondary structure units
    # Counter for each residue type found
    structure_residue_counts = {'H': 0, 'E': 0, '-': 0}
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
            (structure, resid) for structure, resid in
            zip(dssp.results.dssp[0], chain.residues.resindices)
        ]

        # Group by contiguous secondary structure
        for _, group_iter in groupby(dssp_results, lambda x: x[0]):
            group = list(group_iter)
            if len(group) >= min_structure_size:
                structures.append(
                    group[trim_structure_ends:-trim_structure_ends]
                )
                num_residues = len(group) - (2 * trim_structure_ends)
                structure_residue_counts[group[0][0]] += num_residues

    # If we have fewer alpha-helix residues than beta-sheet residues
    # then we allow picking from beta-sheets too.
    allowed_structures = ['H']
    if structure_residue_counts['H'] < structure_residue_counts['E']:
        allowed_structures.append('E')

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
    trim_chain_start: int = 10,
    trim_chain_end: int = 10,
) -> mda.AtomGroup:
    """
    Return a sub-selection of the input AtomGroup which belongs to protein
    chains trimmed by ``trim_chain_start`` and ``trim_chain_end``.

    Protein chains are defined as any continuously bonded part of system with
    at least 30 residues which match the ``protein`` selection of MDAnalysis.

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

    Returns
    -------
    atomgroup : mda.AtomGroup
      An AtomGroup containing all the atoms from the input AtomGroup
      which belong to protein chains.
    """
    # First let's copy our Universe so we don't overwrite its current state
    copy_u = atomgroup.universe.copy()

    # Create an AtomGroup that contains all the protein residues in the
    # input Universe - we will filter by what matches in the atomgroup later
    copy_protein_ag = copy_u.select_atoms('protein').atoms

    # We need to split by fragments to account for multiple chains
    # To do this, we need bonds!
    if not _atomgroup_has_bonds(copy_protein_ag, 'bonds'):
        wmsg = (
            "No bonds found in input Universe, will attept to guess them."
        )
        warnings.warn(wmsg)
        protein_ag.guess_bonds()

    copy_chains_ags_list = []

    # Loop over each continually bonded section (i.e. chain) of the protein
    for frag in copy_protein_ag.fragments:
        # If this chain is less than 30 residues, it's probably a peptide
        if len(frag.residues) < 30:
            continue

        chain = frag.residues[trim_chain_start:-trim_chain_end].atoms
        copy_chain_ags_list.append(chain)

    # Create a single atomgroup from all chains
    copy_chains_ag = sum(copy_chain_ags_list)

    # Now get a list of all the chain atoms in the original Universe
    # Resindexes are keyed at the Universe scale not AtomGroup
    chain_atoms = atomgroup.universe.atoms[copy_chains_ag.atoms.ix]

    # Return all atoms at the intersection of the input atomgroup and
    # the chains atomgroup
    return atomgroup.intersection(chain_atoms)
