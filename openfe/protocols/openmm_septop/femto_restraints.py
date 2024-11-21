"""Utilities for automatically selecting 'reference' atoms for alignment."""

import copy
import itertools
import logging
import typing

import mdtraj
import networkx
import numpy
import openmm.unit
import parmed
import scipy.spatial
import scipy.spatial.distance
from .femto_utils import compute_angles, compute_dihedrals
from .femto_geometry import compute_distances, compute_angles, compute_dihedrals

_COLLINEAR_THRESHOLD = 0.9  # roughly 25 degrees

# values taken from the SepTop reference implementation at commit 7af0b4d
_ANGLE_CHECK_FORCE_CONSTANT = 20.0 * openmm.unit.kilocalorie_per_mole
_ANGLE_CHECK_T = 298.15 * openmm.unit.kelvin
_ANGLE_CHECK_RT = openmm.unit.MOLAR_GAS_CONSTANT_R * _ANGLE_CHECK_T

_ANGLE_CHECK_FACTOR = 0.5 * _ANGLE_CHECK_FORCE_CONSTANT / _ANGLE_CHECK_RT
_ANGLE_CHECK_CUTOFF = 10.0  # units of kT

_ANGLE_CHECK_MAX_VAR = 100.0  # units of degrees^2

_DIHEDRAL_CHECK_CUTOFF = 150.0  # units of degrees
_DIHEDRAL_CHECK_MAX_VAR = 300.0  # units of degrees^2

_RMSF_CUTOFF = 0.1  # nm

def _is_angle_linear(coords: numpy.ndarray, idxs: tuple[int, int, int]) -> bool:
    """Check if angle is within 10 kT from 0 or 180 following the SepTop reference
    implementation.

    Args:
        coords: The full set of coordinates.
        idxs: The indices of the three atoms that form the angle.

    Returns:
        True if the angle is linear, False otherwise.
    """

    angles = numpy.rad2deg(
        compute_angles(coords, numpy.array([idxs]))
    )

    angle_avg_rad = numpy.deg2rad(scipy.stats.circmean(angles, low=-180.0, high=180.0))
    angle_var_deg = scipy.stats.circvar(angles, low=-180.0, high=180.0)

    check_1 = _ANGLE_CHECK_FACTOR * angle_avg_rad**2
    check_2 = _ANGLE_CHECK_FACTOR * (angle_avg_rad - numpy.pi) ** 2

    return (
        check_1 < _ANGLE_CHECK_CUTOFF
        or check_2 < _ANGLE_CHECK_CUTOFF
        or angle_var_deg > _ANGLE_CHECK_MAX_VAR
    )


def _is_dihedral_trans(coords: numpy.ndarray, idxs: tuple[int, int, int, int]) -> bool:
    """Check if a dihedral angle is within -150 and 150 degrees.

    Args:
        coords: The full set of coordinates.
        idxs: The indices of the four atoms that form the dihedral.

    Returns:
        True if the dihedral is planar.
    """

    dihedrals = numpy.rad2deg(
        compute_dihedrals(coords, numpy.array([idxs]))
    )

    dihedral_avg = scipy.stats.circmean(dihedrals, low=-180.0, high=180.0)
    dihedral_var = scipy.stats.circvar(dihedrals, low=-180.0, high=180.0)

    return (
        numpy.abs(dihedral_avg) > _DIHEDRAL_CHECK_CUTOFF
        or dihedral_var > _DIHEDRAL_CHECK_MAX_VAR
    )


def _are_collinear(
    coords: numpy.ndarray, idxs: typing.Sequence[int] | None = None
) -> bool:
    """Checks whether a sequence of coordinates are collinear.

    Args:
        coords: The full set of coordinates, either with ``shape=(n_coords, 3)`` or
            ``shape=(n_frames, n_coords, 3)``.
        idxs: The sequence of indices of those coordinates to check for collinearity.

    Returns:
        True if any sequential pair of vectors is collinear.
    """

    if coords.ndim == 2:
        coords = coords.reshape(1, *coords.shape)

    idxs = idxs if idxs is not None else list(range(coords.shape[1]))

    for i in range(len(idxs) - 2):
        v_1 = coords[:, idxs[i + 1], :] - coords[:, idxs[i], :]
        v_1 /= numpy.linalg.norm(v_1, axis=-1, keepdims=True)
        v_2 = coords[:, idxs[i + 2], :] - coords[:, idxs[i + 1], :]
        v_2 /= numpy.linalg.norm(v_2, axis=-1, keepdims=True)

        if (numpy.abs((v_1 * v_2).sum(axis=-1)) > _COLLINEAR_THRESHOLD).any():
            return True

    return False


def _create_ligand_queries(
    ligand, snapshots: list[openmm.unit.Quantity] | None
) -> tuple[str, str, str]:
    """Selects three atoms from a ligand for use in
    Boresch-likes restraints using the method described by Baumann et al.

    References:
        [1] Baumann, Hannah M., et al. "Broadening the scope of binding free energy
            calculations using a Separated Topologies approach." (2023).
    """

    ligand_graph = networkx.from_edgelist(
        [(bond.atom1_index, bond.atom2_index)
        for bond in ligand.bonds
        if bond.atom1.atomic_number != 1 and bond.atom2.atomic_number != 1]
    )

    all_paths = [
        path
        for node_paths in networkx.shortest_path(ligand_graph).values()
        for path in node_paths.values()
    ]
    path_lengths = {(path[0], path[-1]): len(path) for path in all_paths}

    longest_path = max(all_paths, key=len)
    center_idx = longest_path[len(longest_path) // 2]

    cycles = networkx.cycle_basis(ligand_graph)

    if len(cycles) >= 1 and snapshots is not None:
        top = mdtraj.Topology.from_openmm(ligand.to_openmm())
        ligand_trajectory = mdtraj.Trajectory(
            snapshots,
            top,
        )
        ligand_trajectory.superpose(ligand_trajectory)

        rmsf = mdtraj.rmsf(ligand_trajectory, ligand_trajectory, 0)
        cycles = [cycle for cycle in cycles if rmsf[cycle].max() < _RMSF_CUTOFF]

    if len(cycles) >= 1:
        open_list = [atom_idx for cycle in cycles for atom_idx in cycle]
    else:
        open_list = [atom.idx for atom in ligand.atoms if atom.atomic_number != 1]

    distances = [path_lengths[(center_idx, atom_idx)] for atom_idx in open_list]
    closest_idx = open_list[numpy.argmin(distances)]
    print(closest_idx)

    if len(cycles) >= 1:
        # restrict the list of reference atoms to select from to those that are in the
        # same cycle as the closest atom.
        cycle_idx = next(
            iter(i for i, cycle in enumerate(cycles) if closest_idx in cycle)
        )
        open_list = cycles[cycle_idx]

        distances = [path_lengths[(closest_idx, atom_idx)] for atom_idx in open_list]

    open_list = [
        idx
        for _, idx in sorted(zip(distances, open_list, strict=True))
        if idx != closest_idx
    ]
    restrain_atoms = (open_list[0], closest_idx, open_list[1])

    # TODO: check if the reference atoms are co-linear
    # TODO: handle the unhappy paths of not enough atoms are found.

    return restrain_atoms


def select_ligand_idxs(
    ligand_1, # OpenFF Topology
    ligand_2, # OpenFF Topology
    ligand_1_queries: tuple[str, str, str] | None = None,
    ligand_2_queries: tuple[str, str, str] | None = None,
) -> tuple[tuple[int, int, int], tuple[int, int, int] | None]:
    """Returns the indices of the reference atoms that may be used to align ligands.

    Args:
        ligand_1: The first ligand.
        ligand_2: The second ligand.
        ligand_1_queries: Three (optional) indices to use to manually
            select atoms from the first ligand.
        ligand_2_queries: Three (optional) indices to use to manually
            select atoms from the second ligand

    Returns:
        The indices of the first and second ligand respectively.
    """
    if ligand_1_queries is None or ligand_2_queries is None:

        if ligand_1_queries is None:
            # Setting frames to None right now
            # ToDo: Enable use of snapshots
            ligand_1_queries = _create_ligand_queries(ligand_1, None)
        if ligand_2_queries is None:
            ligand_2_queries = _create_ligand_queries(ligand_2, None)

    # ligand_1_idxs = queries_to_idxs(ligand_1, ligand_1_queries)

    # ligand_2_idxs = queries_to_idxs(ligand_2, ligand_2_queries)

    return ligand_1_queries, ligand_2_queries


def _filter_receptor_atoms(
        receptor: mdtraj.Trajectory,
        ligand: mdtraj.Trajectory,
        ligand_ref_idx: int,
        min_helix_size: int = 8,
        min_sheet_size: int = 8,
        skip_residues_start: int = 20,
        skip_residues_end: int = 10,
        minimum_distance: openmm.unit.Quantity = 1.0 * openmm.unit.nanometers,
        maximum_distance: openmm.unit.Quantity = 3.0 * openmm.unit.nanometers,
) -> list[int]:
    """Select possible protein atoms for Boresch-style restraints based on
    the criteria
    outlined by Baumann et al.

    Args:
        receptor: The receptor structure.
        ligand: The ligand structure.
        ligand_ref_idx: The index of the first reference ligand atom.
        min_helix_size: The minimum number of residues that have to be in an
        alpha-helix
            for it to be considered stable.
        min_sheet_size: The minimum number of residues that have to be in a
        beta-sheet
            for it to be considered stable.
        skip_residues_start: The number of residues to skip at the start of
        the protein
            as these tend to be more flexible.
        skip_residues_end: The number of residues to skip at the end of the
        protein
            as these tend to be more flexible
        minimum_distance: Discard any protein atoms that are closer than
        this distance
            to the ligand.
        maximum_distance: Discard any protein atoms that are further than
        this distance
            from the ligand.

    Returns:
        The indices of protein atoms that should be considered for use in
        Boresch-style
        restraints.
    """

    assert min_helix_size >= 7, "helices must be at least 7 residues long"
    assert min_sheet_size >= 7, "sheets must be at least 7 residues long"

    backbone_idxs = receptor.top.select("protein and (backbone or name CB)")
    backbone: mdtraj.Trajectory = receptor.atom_slice(backbone_idxs,
                                                      inplace=False)

    structure = mdtraj.compute_dssp(backbone, simplified=True).tolist()[0]
    # following the SepTop reference implementation we prefer to select from
    # alpha
    # helices if they are dominant in the protein, but otherwise select from
    # sheets
    # as well.
    n_helix_residues = structure.count("H")
    n_sheet_residues = structure.count("E")

    allowed_motifs = ["H"] if n_helix_residues >= n_sheet_residues else ["H",
                                                                         "E"]
    min_motif_size = {"H": min_helix_size, "E": min_sheet_size}

    residues_to_keep = []

    structure = structure[skip_residues_start: -(skip_residues_end + 1)]

    for motif, idxs in itertools.groupby(enumerate(structure), lambda x: x[1]):

        idxs = [(idx + skip_residues_start, motif) for idx, motif in idxs]

        if motif not in allowed_motifs or len(idxs) < min_motif_size[motif]:
            continue
        # discard the first and last 3 residues of the helix / sheet
        start_idx, end_idx = idxs[0][0] + 3, idxs[-1][0] - 3

        residues_to_keep.extend(
            f"resid {idx}" for idx in range(start_idx, end_idx + 1))
    rigid_backbone_idxs = backbone.top.select(" ".join(residues_to_keep))

    if len(rigid_backbone_idxs) == 0:
        raise ValueError("no suitable receptor atoms could be found")

    if backbone.n_frames > 1:
        superposed = copy.deepcopy(backbone)
        superposed.superpose(superposed)

        rmsf = mdtraj.rmsf(superposed, superposed, 0)  # nm

        rigid_backbone_idxs = rigid_backbone_idxs[
            rmsf[rigid_backbone_idxs] < _RMSF_CUTOFF
            ]

    distances = scipy.spatial.distance.cdist(
        backbone.xyz[0, rigid_backbone_idxs, :],
        ligand.xyz[0, [ligand_ref_idx], :]
    )

    minimum_distance = minimum_distance.value_in_unit(openmm.unit.nanometer)
    maximum_distance = maximum_distance.value_in_unit(openmm.unit.nanometer)

    distance_mask = (distances > minimum_distance).all(axis=1)
    distance_mask &= (distances <= maximum_distance).any(axis=1)

    return backbone_idxs[rigid_backbone_idxs[distance_mask]].tolist()


def _is_valid_r1(
    receptor: mdtraj.Trajectory,
    receptor_idx: int,
    ligand: mdtraj.Trajectory,
    ligand_ref_idxs: tuple[int, int, int],
) -> bool:
    """Check whether a given receptor atom would be a valid 'R1' atom given the
    following criteria:

    * L2,L1,R1 angle not 'close' to 0 or 180 degrees
    * L3,L2,L1,R1 dihedral between -150 and 150 degrees

    Args:
        receptor: The receptor structure.
        receptor_idx: The index of the receptor atom to check.
        ligand: The ligand structure.
        ligand_ref_idxs: The three reference ligand atoms.
    """

    coords = numpy.concatenate([ligand.xyz, receptor.xyz], axis=1)

    l1, l2, l3 = ligand_ref_idxs
    # r1 = receptor_idx
    r1 = receptor_idx + ligand.n_atoms

    if _are_collinear(coords, (r1, l1, l2, l3)):
        return False

    if _is_angle_linear(coords, (r1, l1, l2)):
        return False

    if _is_dihedral_trans(coords, (r1, l1, l2, l3)):
        return False

    return True


def _is_valid_r2(
    receptor: mdtraj.Trajectory,
    receptor_idx: int,
    receptor_ref_idx_1: int,
    ligand: mdtraj.Trajectory,
    ligand_ref_idxs: tuple[int, int, int],
) -> bool:
    """Check whether a given receptor atom would be a valid 'R2' atom given the
    following criteria:

    * R1,R2 are further apart than 5 Angstroms
    * R2,R1,L1,L2 are not collinear
    * R2,R1,L1 angle not 'close' to 0 or 180 degrees
    * R2,R1,L1,L2 dihedral between -150 and 150 degrees

    Args:
        receptor: The receptor structure.
        receptor_idx: The index of the receptor atom to check.
        receptor_ref_idx_1: The index of the first receptor reference atom.
        ligand: The ligand structure.
        ligand_ref_idxs: The three reference ligand atoms.
    """

    coords = numpy.concatenate([ligand.xyz, receptor.xyz], axis=1)

    l1, l2, l3 = ligand_ref_idxs
    r1, r2 = receptor_ref_idx_1 + ligand.n_atoms, receptor_idx + ligand.n_atoms
    # r1, r2 = receptor_ref_idx_1 , receptor_idx

    if r1 == r2:
        return False

    if numpy.linalg.norm(coords[:, r1, :] - coords[:, r2, :], axis=-1).mean() < 0.5:
        return False

    if _are_collinear(coords, (r2, r1, l1, l2)):
        return False

    if _is_angle_linear(coords, (r2, r1, l1)):
        return False

    if _is_dihedral_trans(coords, (r2, r1, l1, l2)):
        return False

    return True


def _is_valid_r3(
    receptor: mdtraj.Trajectory,
    receptor_idx: int,
    receptor_ref_idx_1: int,
    receptor_ref_idx_2: int,
    ligand: mdtraj.Trajectory,
    ligand_ref_idxs: tuple[int, int, int],
) -> bool:
    """Check whether a given receptor atom would be a valid 'R3' atom given the
    following criteria:

    * R1,R2,R3,L1 are not collinear
    * R3,R2,R1,L1 dihedral between -150 and 150 degrees

    Args:
        receptor: The receptor structure.
        receptor_idx: The index of the receptor atom to check.
        receptor_ref_idx_1: The index of the first receptor reference atom.
        receptor_ref_idx_2: The index of the second receptor reference atom.
        ligand: The ligand structure.
        ligand_ref_idxs: The three reference ligand atoms.
    """

    coords = numpy.concatenate([ligand.xyz, receptor.xyz], axis=1)

    l1, l2, l3 = ligand_ref_idxs
    r1, r2, r3 = (
        receptor_ref_idx_1 + ligand.n_atoms,
        receptor_ref_idx_2 + ligand.n_atoms,
        receptor_idx + ligand.n_atoms,
    )

    if len({r1, r2, r3}) != 3:
        return False

    if _are_collinear(coords[[0]], (r3, r2, r1, l1)):
        return False

    if _is_dihedral_trans(coords, (r3, r2, r1, l1)):
        return False

    return True


def select_receptor_idxs(
    receptor: mdtraj.Trajectory,
    ligand: mdtraj.Trajectory,
    ligand_ref_idxs: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Select possible protein atoms for Boresch-style restraints using the method
    outlined by Baumann et al [1].

    References:
        [1] Baumann, Hannah M., et al. "Broadening the scope of binding free energy
            calculations using a Separated Topologies approach." (2023).

    Args:
        receptor: The receptor structure.
        ligand: The ligand structure.
        ligand_ref_idxs: The indices of the three ligands atoms that will be restrained.

    Returns:
        The indices of the three atoms to use for the restraint
    """
    if not (isinstance(receptor, type(ligand)) or isinstance(ligand, type(receptor))):
        raise ValueError("receptor and ligand must be the same type")

    assert (
        receptor.n_frames == ligand.n_frames
    ), "receptor and ligand must have the same number of frames"

    receptor_idxs = _filter_receptor_atoms(receptor, ligand, ligand_ref_idxs[0])

    valid_r1_idxs = [
        idx
        for idx in receptor_idxs
        if _is_valid_r1(receptor, idx, ligand, ligand_ref_idxs)
    ]

    found_r1, found_r2 = next(
        (
            (r1, r2)
            for r1 in valid_r1_idxs
            for r2 in receptor_idxs
            if _is_valid_r2(receptor, r2, r1, ligand, ligand_ref_idxs)
        ),
        None,
    )

    if found_r1 is None or found_r2 is None:
        raise ValueError("could not find valid R1 / R2 atoms")

    valid_r3_idxs = [
        idx
        for idx in receptor_idxs
        if _is_valid_r3(receptor, idx, found_r1, found_r2, ligand, ligand_ref_idxs)
    ]

    if len(valid_r3_idxs) == 0:
        raise ValueError("could not find a valid R3 atom")

    r3_distances_per_frame = []

    for frame_r, frame_l in zip(receptor.xyz, ligand.xyz, strict=True):
        r3_r_distances = scipy.spatial.distance.cdist(
            frame_r[valid_r3_idxs, :], frame_r[[found_r1, found_r2], :]
        )
        r3_l_distances = scipy.spatial.distance.cdist(
            frame_r[valid_r3_idxs, :], frame_l[[ligand_ref_idxs[0]], :]
        )

        r3_distances_per_frame.append(numpy.hstack([r3_r_distances, r3_l_distances]))

    # chosen to match the SepTop reference implementation at commit 3705ba5
    max_distance = 0.8 * (receptor.unitcell_lengths.mean(axis=0).min(axis=-1) / 2)
    # max_distance = 3

    r3_distances_avg = numpy.stack(r3_distances_per_frame).mean(axis=0)

    max_distance_mask = r3_distances_avg.max(axis=-1) < max_distance
    r3_distances_avg = r3_distances_avg[max_distance_mask]

    valid_r3_idxs = numpy.array(valid_r3_idxs)[max_distance_mask].tolist()

    r3_distances_prod = r3_distances_avg[:, 0] * r3_distances_avg[:, 1]
    found_r3 = valid_r3_idxs[r3_distances_prod.argmax()]

    return found_r1, found_r2, found_r3


def check_receptor_idxs(
    receptor: mdtraj.Trajectory,
    receptor_idxs: tuple[int, int, int],
    ligand: mdtraj.Trajectory,
    ligand_ref_idxs: tuple[int, int, int],
) -> bool:
    """Check if the specified receptor atoms meet the criteria for use in Boresch-style
    restraints as defined by Baumann et al [1].

    References:
        [1] Baumann, Hannah M., et al. "Broadening the scope of binding free energy
            calculations using a Separated Topologies approach." (2023).

    Args:
        receptor: The receptor structure.
        receptor_idxs: The indices of the three receptor atoms that will be restrained.
        ligand: The ligand structure.
        ligand_ref_idxs: The indices of the three ligand atoms that will be restrained.

    Returns:
        True if the atoms meet the criteria, False otherwise.
    """
    if not (isinstance(receptor, type(ligand)) or isinstance(ligand, type(receptor))):
        raise ValueError("receptor and ligand must be the same type")

    assert (
        receptor.n_frames == ligand.n_frames
    ), "receptor and ligand must have the same number of frames"

    r1, r2, r3 = receptor_idxs

    is_valid_r1 = _is_valid_r1(receptor, r1, ligand, ligand_ref_idxs)
    is_valid_r2 = _is_valid_r2(receptor, r2, r1, ligand, ligand_ref_idxs)
    is_valid_r3 = _is_valid_r3(receptor, r3, r1, r2, ligand, ligand_ref_idxs)

    r3_distances_per_frame = [
        scipy.spatial.distance.cdist(frame[[r3], :], frame[[r1, r2], :])
        for frame in receptor.xyz
    ]
    r3_distance_avg = numpy.stack(r3_distances_per_frame).mean(axis=0)

    max_distance = 0.8 * (receptor.unitcell_lengths[-1][0] / 2)
    is_valid_distance = r3_distance_avg.max(axis=-1) < max_distance
    print(is_valid_r1, is_valid_r2, is_valid_r3, is_valid_distance)

    return is_valid_r1 and is_valid_r2 and is_valid_r3 and is_valid_distance


_BORESCH_ENERGY_FN = (
    "0.5 * E;"
    "E = k_dist_a  * (distance(p3,p4) - dist_0)    ^ 2"
    "  + k_theta_a * (angle(p2,p3,p4) - theta_a_0) ^ 2"
    "  + k_theta_b * (angle(p3,p4,p5) - theta_b_0) ^ 2"
    "  + k_phi_a   * (d_phi_a_wrap)                ^ 2"
    "  + k_phi_b   * (d_phi_b_wrap)                ^ 2"
    "  + k_phi_c   * (d_phi_c_wrap)                ^ 2;"
    # compute the periodic dihedral delta (e.g. distance between -180 and 180 is 0)
    "d_phi_a_wrap = d_phi_a - floor(d_phi_a / (2.0 * pi) + 0.5) * (2.0 * pi);"
    "d_phi_a = dihedral(p1,p2,p3,p4) - phi_a_0;"
    "d_phi_b_wrap = d_phi_b - floor(d_phi_b / (2.0 * pi) + 0.5) * (2.0 * pi);"
    "d_phi_b = dihedral(p2,p3,p4,p5) - phi_b_0;"
    "d_phi_c_wrap = d_phi_c - floor(d_phi_c / (2.0 * pi) + 0.5) * (2.0 * pi);"
    "d_phi_c = dihedral(p3,p4,p5,p6) - phi_c_0;"
    f"pi = {numpy.pi}"
).replace(" ", "")


_ANGSTROM = openmm.unit.angstrom
_RADIANS = openmm.unit.radian


class _BoreschGeometry(typing.NamedTuple):
    dist_0: openmm.unit.Quantity

    theta_a_0: openmm.unit.Quantity
    theta_b_0: openmm.unit.Quantity

    phi_a_0: openmm.unit.Quantity
    phi_b_0: openmm.unit.Quantity
    phi_c_0: openmm.unit.Quantity


def _compute_boresch_geometry(
    receptor_atoms: tuple[int, int, int],
    ligand_atoms: tuple[int, int, int],
    coords: openmm.unit.Quantity,
) -> _BoreschGeometry:
    """Computes the equilibrium distances, angles, and dihedrals used by a Boresch
    restraint."""

    r1, r2, r3 = receptor_atoms
    l1, l2, l3 = ligand_atoms

    coords = coords.value_in_unit(openmm.unit.angstrom)

    dist_0 = (
        compute_distances(coords, numpy.array([[r3, l1]]))
        * _ANGSTROM
    )

    theta_a_0 = (
        compute_angles(coords, numpy.array([[r2, r3, l1]]))
        * _RADIANS
    )
    theta_b_0 = (
        compute_angles(coords, numpy.array([[r3, l1, l2]]))
        * _RADIANS
    )

    phi_a_0 = (
        compute_dihedrals(
            coords, numpy.array([[r1, r2, r3, l1]])
        )
        * _RADIANS
    )
    phi_b_0 = (
        compute_dihedrals(
            coords, numpy.array([[r2, r3, l1, l2]])
        )
        * _RADIANS
    )
    phi_c_0 = (
        compute_dihedrals(
            coords, numpy.array([[r3, l1, l2, l3]])
        )
        * _RADIANS
    )

    return _BoreschGeometry(dist_0, theta_a_0, theta_b_0, phi_a_0, phi_b_0, phi_c_0)


def create_boresch_restraint(
    receptor_atoms: tuple[int, int, int],
    ligand_atoms: tuple[int, int, int],
    coords: openmm.unit.Quantity,
    k_distance,
    k_theta,
    ctx_parameter: str | None = None,
) -> openmm.CustomCompoundBondForce:
    """Creates a Boresch restraint force useful in aligning a receptor and ligand.

    Args:
        settings: RestraintSettings
        receptor_atoms: The indices of the receptor atoms to restrain.
        ligand_atoms: The indices of the ligand atoms to restrain.
        coords: The coordinates of the *full* system.
        ctx_parameter: An optional context parameter to use to scale the strength of
            the restraint.

    Returns:
        The restraint force.
    """
    n_particles = 6  # 3 receptor + 3 ligand

    energy_fn = _BORESCH_ENERGY_FN

    # if ctx_parameter is not None:
    #     energy_fn = f"{ctx_parameter} * {energy_fn}"

    force = openmm.CustomCompoundBondForce(n_particles, energy_fn)

    # if ctx_parameter is not None:
    #     force.addGlobalParameter(ctx_parameter, 1.0)

    geometry = _compute_boresch_geometry(receptor_atoms, ligand_atoms, coords)

    # Scale the k_theta_a
    distance_0 = 5.0  # based on original SepTop implementation.
    scale = (geometry.dist_0 / distance_0) ** 2

    parameters = []

    for key, value in [
        ("k_dist_a", k_distance),
        ("k_theta_a", k_theta * scale),
        ("k_theta_b", k_theta),
        ("k_phi_a", k_theta),
        ("k_phi_b", k_theta),
        ("k_phi_c", k_theta),
        ("dist_0", geometry.dist_0),
        ("theta_a_0", geometry.theta_a_0),
        ("theta_b_0", geometry.theta_b_0),
        ("phi_a_0", geometry.phi_a_0),
        ("phi_b_0", geometry.phi_b_0),
        ("phi_c_0", geometry.phi_c_0),
    ]:
        force.addPerBondParameter(key)
        parameters.append(value.value_in_unit_system(openmm.unit.md_unit_system))

    force.addBond(receptor_atoms + ligand_atoms, parameters)
    force.setUsesPeriodicBoundaryConditions(False)
    force.setName("alignment-restraint")
    force.setForceGroup(6)

    return force
