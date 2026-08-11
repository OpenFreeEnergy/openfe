# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Dummy-atom Boresch restraint geometry for the solvent leg of SepTop.

In the solvent leg there is no host molecule to anchor Boresch restraints to.
Instead, three dummy atoms are placed analytically around each ligand's input
conformer so that all Boresch angles and dihedrals are well-defined (not near
0 or 180 degrees). The dummies carry no nonbonded interactions and have a
very large mass, making them effectively immobile throughout the simulation.

Geometry construction
---------------------
Given three ligand anchor atoms G0, G1, G2 at positions p0, p1, p2:

  D2                            G2
   -                           -
    -                         -
     D1 - - D0 -- G0 - - G1

D0 is placed along the normal n to the G0-G1-G2 plane, at distance r0 from
G0. This guarantees theta_B (D0-G0-G1) = 90 degrees.

D1 is placed so that theta_A (D1-D0-G0) = 90 degrees, in the plane spanned
by n and (p1 - p0).

D2 is placed so that phi_A (D2-D1-D0-G0) = 60 degrees.

All angles are validated to be away from the singular values 0 and 180
degrees.
"""
from __future__ import annotations

import warnings

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Distance (Å) from G0 to D0.
_DUMMY_BOND_LENGTH_A: float = 5.0

#: Minimum safe angle (radians) away from 0 or pi.
_ANGLE_WARN_THRESHOLD_RAD: float = np.deg2rad(10.0)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _normalise(v: np.ndarray) -> np.ndarray:
    """Return the unit vector of *v*. Raises if the norm is zero."""
    n = np.linalg.norm(v)
    if n < 1e-8:
        raise ValueError(f"Cannot normalise a near-zero vector: {v}")
    return v / n


def _perpendicular_in_plane(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """
    Return a unit vector that is perpendicular to *a* and lies in the plane
    spanned by *a* and *b*.

    Parameters
    ----------
    a:
        The primary direction (will be normalised).
    b:
        A second vector that, together with *a*, defines the plane.

    Returns
    -------
    np.ndarray
        Unit vector perpendicular to *a* in the (a, b) plane.
    """
    a_hat = _normalise(a)
    # Remove the a component from b
    b_perp = b - np.dot(b, a_hat) * a_hat
    return _normalise(b_perp)


def _rotate_around_axis(
    v: np.ndarray,
    axis: np.ndarray,
    angle_rad: float,
) -> np.ndarray:
    """
    Rotate vector *v* around *axis* by *angle_rad* using Rodrigues' formula.
    """
    axis = _normalise(axis)
    return (
        v * np.cos(angle_rad)
        + np.cross(axis, v) * np.sin(angle_rad)
        + axis * np.dot(axis, v) * (1 - np.cos(angle_rad))
    )


def _check_angle_safe(angle_rad: float, name: str) -> None:
    """
    Warn if *angle_rad* is within ``_ANGLE_WARN_THRESHOLD_RAD`` of 0 or pi.
    """
    if angle_rad < _ANGLE_WARN_THRESHOLD_RAD or angle_rad > np.pi - _ANGLE_WARN_THRESHOLD_RAD:
        warnings.warn(
            f"Boresch angle {name} = {np.degrees(angle_rad):.1f} deg is close "
            "to a singular value (0 or 180 deg). Consider choosing different "
            "ligand anchor atoms.",
            UserWarning,
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# Core dummy placement
# ---------------------------------------------------------------------------


def find_dummy_atom_positions(
    p_g0: np.ndarray,
    p_g1: np.ndarray,
    p_g2: np.ndarray,
    bond_length_a: float = _DUMMY_BOND_LENGTH_A,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Analytically place three dummy atoms (D0, D1, D2) around three ligand
    anchor atoms (G0, G1, G2) such that the resulting Boresch angles and
    dihedrals are well-defined.

    All input positions must be in **Angstroms**.

    Parameters
    ----------
    p_g0, p_g1, p_g2:
        Positions of ligand anchor atoms G0, G1, G2 in Angstroms.
    bond_length_a:
        Distance from G0 to D0 in Angstroms. Default 5.0 Å.

    Returns
    -------
    p_d0, p_d1, p_d2
        Positions of the three dummy atoms in Angstroms.

    Notes
    -----
    Construction guarantees:

    * theta_B (D0–G0–G1) = 90 deg  (D0 is along the G0/G1/G2 plane normal)
    * theta_A (D1–D0–G0) = 90 deg  (D1 is perpendicular to D0–G0)
    * phi_A   (D2–D1–D0–G0) = 60 deg
    * phi_B   (D1–D0–G0–G1) depends on the in-plane direction chosen for D1;
      by construction this is 0 deg, which *is* a singular value. We therefore
      rotate D1 by 90 deg around the D0→G0 axis so that phi_B = 90 deg.

    The construction is deterministic and rotation-invariant (it only depends
    on the relative geometry of G0/G1/G2).
    """
    p_g0 = np.asarray(p_g0, dtype=float)
    p_g1 = np.asarray(p_g1, dtype=float)
    p_g2 = np.asarray(p_g2, dtype=float)

    # ------------------------------------------------------------------
    # D0: along the normal of the G0/G1/G2 plane, distance r0 from G0.
    # theta_B (D0-G0-G1) = 90 deg by construction.
    # ------------------------------------------------------------------
    v01 = p_g1 - p_g0
    v02 = p_g2 - p_g0

    normal = np.cross(v01, v02)
    if np.linalg.norm(normal) < 1e-8:
        # G0, G1, G2 are collinear — fall back to an arbitrary perpendicular
        warnings.warn(
            "Ligand anchor atoms G0, G1, G2 are (near-)collinear. "
            "Dummy atom placement may produce poorly-defined dihedrals.",
            UserWarning,
            stacklevel=2,
        )
        # Pick an arbitrary vector not parallel to v01
        arb = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(_normalise(v01), arb)) > 0.9:
            arb = np.array([0.0, 1.0, 0.0])
        normal = np.cross(v01, arb)

    n_hat = _normalise(normal)
    p_d0 = p_g0 + bond_length_a * n_hat

    # ------------------------------------------------------------------
    # D1: perpendicular to D0–G0, in the plane spanned by n_hat and v01.
    # This makes theta_A (D1-D0-G0) = 90 deg.
    # We then rotate D1 90 deg around the D0→G0 axis so that
    # phi_B (D1-D0-G0-G1) = 90 deg (away from the singular value 0 deg).
    # ------------------------------------------------------------------
    d0_to_g0 = p_g0 - p_d0  # direction from D0 toward G0

    # In-plane perpendicular to d0_to_g0 using v01 as the in-plane reference
    d1_dir_base = _perpendicular_in_plane(d0_to_g0, v01)

    # Rotate 90 deg around (D0→G0) to set phi_B away from 0
    d1_dir = _rotate_around_axis(d1_dir_base, d0_to_g0, np.pi / 2)

    p_d1 = p_d0 + bond_length_a * d1_dir

    # ------------------------------------------------------------------
    # D2: placed so that phi_A (D2-D1-D0-G0) = 60 deg.
    # D2 lies in a direction perpendicular to D1–D0, rotated 60 deg
    # around the D1→D0 axis from an initial reference direction.
    # ------------------------------------------------------------------
    d1_to_d0 = p_d0 - p_d1

    # Reference direction perpendicular to D1–D0, using D0→G0 as guide
    d2_dir_ref = _perpendicular_in_plane(d1_to_d0, d0_to_g0)

    # Rotate by 60 deg to give phi_A = 60 deg
    d2_dir = _rotate_around_axis(d2_dir_ref, d1_to_d0, np.deg2rad(60.0))

    p_d2 = p_d1 + bond_length_a * d2_dir

    return p_d0, p_d1, p_d2


def _validate_dummy_geometry(
    p_d0: np.ndarray,
    p_d1: np.ndarray,
    p_d2: np.ndarray,
    p_g0: np.ndarray,
    p_g1: np.ndarray,
    p_g2: np.ndarray,
) -> None:
    """
    Compute and warn on any Boresch angles / dihedrals that are close to
    singular values (0 or 180 deg).

    Positions in Angstroms.
    """
    from MDAnalysis.lib.distances import calc_angles, calc_dihedrals

    # theta_A: D1-D0-G0
    theta_A = calc_angles(p_d1, p_d0, p_g0)
    _check_angle_safe(theta_A, "theta_A (D1-D0-G0)")

    # theta_B: D0-G0-G1
    theta_B = calc_angles(p_d0, p_g0, p_g1)
    _check_angle_safe(theta_B, "theta_B (D0-G0-G1)")

    # phi_A: D2-D1-D0-G0
    phi_A = calc_dihedrals(p_d2, p_d1, p_d0, p_g0)
    _check_angle_safe(abs(phi_A) % np.pi, "phi_A (D2-D1-D0-G0)")

    # phi_B: D1-D0-G0-G1
    phi_B = calc_dihedrals(p_d1, p_d0, p_g0, p_g1)
    _check_angle_safe(abs(phi_B) % np.pi, "phi_B (D1-D0-G0-G1)")

    # phi_C: D0-G0-G1-G2
    phi_C = calc_dihedrals(p_d0, p_g0, p_g1, p_g2)
    _check_angle_safe(abs(phi_C) % np.pi, "phi_C (D0-G0-G1-G2)")