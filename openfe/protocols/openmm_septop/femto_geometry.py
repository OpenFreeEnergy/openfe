"""Common functions for computing the internal coordinates (e.g. bond lengths)."""

import numpy


def compute_bond_vectors(
    coords: numpy.ndarray, idxs: numpy.ndarray
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Computes the vectors between each atom pair specified by the ``idxs`` as
    well as their norms.

    Args:
        coords: The coordinates with ``shape=(n_coords, 3)``.
        idxs: The indices of the coordinates to compute the distances between with
            ``shape=(n_pairs, 2)``.

    Returns:
        A tuple of the vectors with shape=``shape=(n_pairs, 3)`` and norms with
        ``shape=(n_pairs,)``.
    """

    if len(idxs) == 0:
        return numpy.ndarray([]), numpy.ndarray([])

    directions = coords[idxs[:, 1]] - coords[idxs[:, 0]]
    distances = numpy.linalg.norm(directions, axis=1)

    return directions, distances


def compute_distances(coords: numpy.ndarray, idxs: numpy.ndarray) -> numpy.ndarray:
    """Computes the distances between each pair of coordinates.

    Args:
        coords: The coordinates with ``shape=(n_coords, 3)``.
        idxs: The indices of the coordinates to compute the distances between with
            ``shape=(n_pairs, 2)``.

    Returns:
        The distances with ``shape=(n_pairs,)``.
    """

    return compute_bond_vectors(coords, idxs)[1]


def compute_angles(coords: numpy.ndarray, idxs: numpy.ndarray) -> numpy.ndarray:
    """Computes the angles [rad] between each specified triplet of indices.

    Args:
        coords: The coordinates with ``shape=(n_coords, 3)`` or
            ``shape=(n_frames, n_coords, 3)``.
        idxs: The indices of the coordinates to compute the angles between with
            ``shape=(n_pairs, 3)``.

    Returns:
        The angles with ``shape=(n_pairs,)`` or ``shape=(n_frames, n_pairs)``.
    """

    if len(idxs) == 0:
        return numpy.ndarray([])

    is_batched = coords.ndim == 3

    if not is_batched:
        coords = coords[None, :, :]

    vector_ab = coords[:, idxs[:, 1]] - coords[:, idxs[:, 0]]
    vector_ac = coords[:, idxs[:, 1]] - coords[:, idxs[:, 2]]

    # tan theta = sin theta / cos theta
    #
    # ||a x b|| = ||a|| ||b|| sin theta
    #   a . b   = ||a|| ||b|| cos theta
    #
    # => tan theta = (a x b) / (a . b)
    angles = numpy.arctan2(
        numpy.linalg.norm(numpy.cross(vector_ab, vector_ac, axis=-1), axis=-1),
        (vector_ab * vector_ac).sum(axis=-1),
    )

    if not is_batched:
        angles = angles[0]

    return angles


def compute_dihedrals(coords: numpy.ndarray, idxs: numpy.ndarray) -> numpy.ndarray:
    """Computes the angles [rad] between each specified quartet of indices.

    Args:
        coords: The coordinates with ``shape=(n_coords, 3)`` or
            ``shape=(n_frames, n_coords, 3)``.
        idxs: The indices of the coordinates to compute the dihedrals between with
            ``shape=(n_pairs, 4)``.

    Returns:
        The dihedrals with ``shape=(n_pairs,)`` or ``shape=(n_frames, n_pairs)``.
    """

    if len(idxs) == 0:
        return numpy.ndarray([])

    is_batched = coords.ndim == 3

    if not is_batched:
        coords = coords[None, :, :]

    vector_ab = coords[:, idxs[:, 0]] - coords[:, idxs[:, 1]]
    vector_cb = coords[:, idxs[:, 2]] - coords[:, idxs[:, 1]]
    vector_cd = coords[:, idxs[:, 2]] - coords[:, idxs[:, 3]]

    vector_ab_cross_cb = numpy.cross(vector_ab, vector_cb, axis=-1)
    vector_cb_cross_cd = numpy.cross(vector_cb, vector_cd, axis=-1)

    vector_cb_norm = numpy.linalg.norm(vector_cb, axis=-1)[:, :, None]

    y = (
        numpy.cross(vector_ab_cross_cb, vector_cb_cross_cd, axis=-1)
        * vector_cb
        / vector_cb_norm
    ).sum(axis=-1)

    x = (vector_ab_cross_cb * vector_cb_cross_cd).sum(axis=-1)

    phi = numpy.arctan2(y, x)

    if not is_batched:
        phi = phi[0]

    return phi