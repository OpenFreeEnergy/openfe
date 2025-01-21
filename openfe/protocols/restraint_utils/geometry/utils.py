# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Search methods for generating Geometry objects

TODO
----
* Add relevant duecredit entries.
"""
from typing import Union, Optional
from itertools import combinations
import numpy as np
import numpy.typing as npt
from scipy.stats import circvar

import openmm
from openff.toolkit import Molecule as OFFMol
from openff.units import unit
import networkx as nx
from rdkit import Chem
import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.rms import RMSF
from MDAnalysis.lib.distances import minimize_vectors, capped_distance
from MDAnalysis.coordinates.memory import MemoryReader
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


def _get_mda_coord_format(
    coordinates: Union[str, pathlib.Path, npt.NDArray]
) -> Optional[MemoryReader]:
    """
    Helper to set the coordinate format to MemoryReader
    if the coordinates are an NDArray.

    Parameters
    ----------
    coordinates : Union[str, pathlib.Path, npt.NDArray]

    Returns
    -------
    Optional[MemoryReader]
      Either the MemoryReader class or None.
    """
    if isinstance(coordinates, npt.NDArray):
        return MemoryReader
    else:
        return None


def _get_mda_topology_format(
    topology: Union[str, openmm.app.Topology]
) -> Optional[str]:
    """
    Helper to set the topology format to OPENMMTOPOLOGY
    if the topology is an openmm.app.Topology.

    Parameters
    ----------
    topology : Union[str, openmm.app.Topology]


    Returns
    -------
    Optional[str]
      The string `OPENMMTOPOLOGY` or None.
    """
    if isinstance(topology, openmm.app.Topology):
        return "OPENMMTOPOLOGY"
    else:
        return None


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

    if not nx.is_weakly_connected(nx_mol):
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


def is_collinear(positions, atoms, dimensions=None, threshold=0.9):
    """
    Check whether any sequential vectors in a sequence of atoms are collinear.

    Parameters
    ----------
    positions : openmm.unit.Quantity
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
    if ang_check_1 < 10.0 or ang_check_2 < 10.0:
        return False
    return True


def check_dihedral_bounds(
    dihedral: unit.Quantity,
    lower_cutoff: unit.Quantity = 2.618 * unit.radians,
    upper_cutoff: unit.Quantity = -2.618 * unit.radians,
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

        self.host_ag = host_atoms
        self.guest_ag = guest_atoms
        self.min_cutoff = min_search_distance.to("angstrom").m
        self.max_cutoff = max_search_distance.to("angstrom").m

    def _prepare(self):
        self.results.host_idxs = set()

    def _single_frame(self):
        pairs = capped_distance(
            reference=self.host_ag.positions,
            configuration=self.guest_ag.positions,
            max_cutoff=self.max_cutoff,
            min_cutoff=self.min_cutoff,
            box=self.guest_ag.universe.dimensions,
            return_distances=False,
        )

        host_idxs = [self.guest_ag.atoms[p].ix for p in pairs[:, 1]]
        self.results.host_idxs.update(set(host_idxs))

    def _conclude(self):
        self.results.host_idxs = np.array(self.results.host_idxs)


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
