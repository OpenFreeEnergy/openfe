# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Utilities for adding dummy (non-interacting, immobile) atoms to an
OpenMM System for use as Boresch restraint anchors in the SepTop solvent leg.

A dummy atom:
* Has zero mass (``DUMMY_MASS_AMU``), so OpenMM treats it as immobile by
  construction -- excluded from velocity initialisation, kinetic energy,
  and the integrator's position update.
* Carries zero charge and zero LJ well-depth (epsilon) in every non-bonded
  force, but a small non-zero length-scale parameter (sigma/radius), since
  a zero length scale can break the analytical long-range dispersion
  correction used by alchemical softcore ``CustomNonbondedForce``
  instances (see ``_dummy_custom_nonbonded_params`` for details).
* Is added to exception/exclusion lists in all relevant forces so it has
  no interactions with the rest of the system regardless of the above
  parameter values.

Usage
-----
::

    system, positions_ang, dummy_idxs = add_dummy_atoms_to_system(
        system, positions_ang, n_dummies=3
    )
    # positions_ang[dummy_idxs[i]] must then be filled in by the caller.
"""
from __future__ import annotations

import numpy as np
import openmm
import openmm.unit as omm_unit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Mass assigned to each dummy atom (Da / amu).
#:
#: Zero mass is used so that OpenMM treats the dummy atoms as immobile by
#: construction: zero-mass particles are excluded from velocity
#: initialisation, kinetic energy, and the integrator's position update
#: entirely, so displacement is exactly zero rather than merely small.
#: This is the pattern documented by the OpenMM developers for tethering
#: real atoms to fixed reference points, see
#: https://github.com/openmm/openmm/issues/2262
DUMMY_MASS_AMU: float = 0.0

#: Lennard-Jones sigma / length-scale value for dummy atoms (nm).
#:
#: Must stay non-zero. A zero length scale, combined with epsilon = 0,
#: causes some alchemical softcore CustomNonbondedForce energy expressions
#: (as produced by openmmtools.alchemy.AbsoluteAlchemicalFactory) to become
#: singular or non-decaying in r for that particle "type", independent of
#: any explicit exclusions -- the analytical long-range dispersion
#: correction is a mean-field integral over particle types, not a pairwise
#: sum, so exclusions alone do not protect against this. The resulting
#: native exception ("CustomNonbondedForce: Long range correction did not
#: converge") is uncatchable from Python and aborts the process. A small
#: arbitrary non-zero sigma (with epsilon = 0, so the actual interaction
#: strength is still zero) avoids the singularity while keeping the dummy
#: energetically inert.
_DUMMY_SIGMA_NM: float = 0.1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _add_dummy_to_nonbonded(
    force: openmm.NonbondedForce,
    new_idx: int,
    existing_indices: list[int],
) -> None:
    """
    Add a dummy particle to a NonbondedForce with zero charge/epsilon
    and add exclusions between it and every other existing particle.

    Parameters
    ----------
    force:
        The NonbondedForce to modify in-place.
    new_idx:
        The particle index of the newly added dummy in the System.
    existing_indices:
        All particle indices that were present *before* this dummy was added.
        Exclusions are created between the dummy and each of them.
    """
    force.addParticle(
        0.0,                    # charge
        _DUMMY_SIGMA_NM,        # sigma (nm)
        0.0,                    # epsilon
    )
    for idx in existing_indices:
        force.addException(new_idx, idx, 0.0, _DUMMY_SIGMA_NM, 0.0)


#: Parameter name substrings that indicate a "well depth" / interaction
#: strength quantity, which should be zeroed for dummy atoms so they have
#: no effective interaction with the rest of the system.
_ZERO_PARAM_NAME_HINTS: tuple[str, ...] = ("epsilon", "charge", "lambda")

#: Parameter name substrings that indicate a length-scale quantity (e.g.
#: sigma, radius). These must stay at a small but non-zero value for dummy
#: atoms: a zero length scale causes some alchemical softcore energy
#: expressions (as used by openmmtools.alchemy.AbsoluteAlchemicalFactory) to
#: become singular or non-decaying in r, which makes OpenMM's analytical
#: long-range dispersion correction fail to converge
#: ("CustomNonbondedForce: Long range correction did not converge").
_NONZERO_PARAM_NAME_HINTS: tuple[str, ...] = ("sigma", "radius", "rmin")


def _dummy_custom_nonbonded_params(force: openmm.CustomNonbondedForce) -> list[float]:
    """
    Build a per-particle parameter list for a dummy atom in a
    CustomNonbondedForce, based on each parameter's name.

    Parameters whose name suggests an interaction-strength quantity
    (``epsilon``, ``charge``, ``lambda``) are set to 0.0, so the dummy has
    no effective interaction strength with any other particle.

    Parameters whose name suggests a length-scale quantity (``sigma``,
    ``radius``, ``rmin``) are set to ``_DUMMY_SIGMA_NM`` instead of 0.0.
    A zero length scale can make alchemical softcore energy expressions
    singular or non-decaying in r for that particle "type", which breaks
    OpenMM's analytical long-range dispersion correction (it requires every
    particle-type combination's energy to decay at least as fast as
    1/r**2 at long range).

    Any parameter not matching either category defaults to 0.0.

    Parameters
    ----------
    force:
        The CustomNonbondedForce to build dummy parameters for.

    Returns
    -------
    list[float]
        Per-particle parameter values, in the force's declared order.
    """
    n_params = force.getNumPerParticleParameters()
    values = []
    for i in range(n_params):
        name = force.getPerParticleParameterName(i).lower()
        if any(hint in name for hint in _NONZERO_PARAM_NAME_HINTS):
            values.append(_DUMMY_SIGMA_NM)
        else:
            # Covers epsilon/charge/lambda hints and any unrecognised
            # parameter name; zero is the safe default for anything that
            # isn't a length scale.
            values.append(0.0)
    return values


def _add_dummy_to_custom_nonbonded(
    force: openmm.CustomNonbondedForce,
    new_idx: int,
    existing_indices: list[int],
) -> None:
    """
    Add a dummy particle to a CustomNonbondedForce, and add exclusions to
    all existing particles.

    Per-particle parameters are chosen via ``_dummy_custom_nonbonded_params``
    rather than zeroed outright: length-scale parameters (sigma, radius)
    are kept at a small non-zero value to avoid breaking the force's
    analytical long-range dispersion correction, while interaction-strength
    parameters (epsilon, charge, lambda) are zeroed so the dummy has no
    effective interaction with any other particle. Explicit exclusions
    additionally guarantee zero pairwise energy regardless of parameter
    values.
    """
    params = _dummy_custom_nonbonded_params(force)
    force.addParticle(params)
    for idx in existing_indices:
        force.addExclusion(new_idx, idx)


def _add_dummy_to_custom_bond(
    force: openmm.CustomBondForce | openmm.HarmonicBondForce,
    new_idx: int,
) -> None:
    """
    No bonds need to be added for dummy atoms; this is a no-op placeholder
    kept here to make the dispatch loop explicit.
    """
    pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_dummy_atoms_to_system(
    system: openmm.System,
    positions_ang: np.ndarray,
    n_dummies: int = 3,
) -> tuple[openmm.System, np.ndarray, list[int]]:
    """
    Append *n_dummies* non-interacting, immobile dummy particles to an
    OpenMM System and a corresponding positions array.

    The positions of the new particles are initialised to the origin (0, 0, 0)
    in Angstroms. The caller is responsible for writing the correct
    coordinates into the returned ``positions_ang`` array before any
    simulation is started.

    Parameters
    ----------
    system:
        The OpenMM System to extend. **Modified in-place.**
    positions_ang:
        Full-system positions in Angstroms, shape ``(N, 3)``.
    n_dummies:
        Number of dummy atoms to add. Default 3.

    Returns
    -------
    system:
        The same System object (modified in-place, returned for convenience).
    positions_ang:
        Extended positions array of shape ``(N + n_dummies, 3)`` in Angstroms.
    dummy_idxs:
        List of the new particle indices, in insertion order.

    Notes
    -----
    The function iterates over all forces in the System and handles:

    * ``NonbondedForce``: zero charge/epsilon, full exclusion list.
    * ``CustomNonbondedForce``: zero per-particle params, full exclusion list.
    * All other force types: no particle entry needed (bond/angle/torsion
      forces only act on explicitly listed atom groups).

    If a ``NonbondedForce`` uses an alchemical lambda parameter (detected by
    the presence of global parameters whose names start with ``"lambda"``),
    the exclusion is still correctly applied because exclusions in
    ``NonbondedForce`` are absolute (not scaled by lambda).
    """
    n_existing = system.getNumParticles()
    existing_indices = list(range(n_existing))
    dummy_idxs: list[int] = []

    for i in range(n_dummies):
        new_idx = system.addParticle(DUMMY_MASS_AMU)
        dummy_idxs.append(new_idx)

        for force in system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                _add_dummy_to_nonbonded(force, new_idx, existing_indices)

            elif isinstance(force, openmm.CustomNonbondedForce):
                _add_dummy_to_custom_nonbonded(force, new_idx, existing_indices)

            # Bond / angle / torsion forces: no entry needed for a particle
            # that is never part of any bonded term. Skip explicitly.
            elif isinstance(force, (
                openmm.HarmonicBondForce,
                openmm.HarmonicAngleForce,
                openmm.PeriodicTorsionForce,
                openmm.CustomBondForce,
                openmm.CustomAngleForce,
                openmm.CustomTorsionForce,
                openmm.CustomCompoundBondForce,
                openmm.CMMotionRemover,
                openmm.MonteCarloBarostat,
                openmm.AndersenThermostat,
            )):
                pass

            else:
                # Unknown force type — log a warning but don't crash.
                # In the worst case the dummy has zero parameters (from
                # addParticle above) and no interaction terms, which is safe.
                import warnings
                warnings.warn(
                    f"Unknown force type {type(force).__name__} encountered "
                    "while adding dummy atoms. The dummy may not be correctly "
                    "excluded from this force.",
                    UserWarning,
                    stacklevel=2,
                )

        # Track this dummy as an existing index for the next iteration's
        # exclusion loop (dummies must also be excluded from each other).
        existing_indices.append(new_idx)

    # Extend the positions array with zeros for the new dummy atoms.
    dummy_positions = np.zeros((n_dummies, 3), dtype=positions_ang.dtype)
    positions_ang = np.vstack([positions_ang, dummy_positions])

    return system, positions_ang, dummy_idxs