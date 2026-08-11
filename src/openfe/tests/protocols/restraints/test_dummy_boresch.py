# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Tests for dummy-atom Boresch restraint geometry and system utilities.
"""
from __future__ import annotations

import numpy as np
import openmm
import pytest
from MDAnalysis.lib.distances import calc_angles, calc_dihedrals, calc_bonds

from openfe.protocols.restraint_utils.geometry.boresch.dummy import (
    _DUMMY_BOND_LENGTH_A,
    find_dummy_atom_positions,
    _validate_dummy_geometry,
)
from openfe.protocols.restraint_utils.openmm.omm_dummy import (
    DUMMY_MASS_AMU,
    add_dummy_atoms_to_system,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _angle_deg(a, b, c):
    """Angle at vertex b in degrees."""
    return np.degrees(calc_angles(
        np.array(a, dtype=float),
        np.array(b, dtype=float),
        np.array(c, dtype=float),
    ))


def _dihedral_deg(a, b, c, d):
    """Dihedral a-b-c-d in degrees."""
    return np.degrees(calc_dihedrals(
        np.array(a, dtype=float),
        np.array(b, dtype=float),
        np.array(c, dtype=float),
        np.array(d, dtype=float),
    ))


def _bond_length(a, b):
    return calc_bonds(np.array(a, dtype=float), np.array(b, dtype=float))


def _simple_ligand_positions():
    """Three non-collinear ligand anchor atoms in Angstroms."""
    return (
        np.array([0.0, 0.0, 0.0]),   # G0
        np.array([1.5, 0.0, 0.0]),   # G1
        np.array([0.75, 1.3, 0.0]),  # G2
    )


class TestFindDummyAtomPositions:

    def test_returns_three_positions(self):
        p_g0, p_g1, p_g2 = _simple_ligand_positions()
        result = find_dummy_atom_positions(p_g0, p_g1, p_g2)
        assert len(result) == 3
        for p in result:
            assert p.shape == (3,)

    def test_d0_bond_length(self):
        """D0 should be exactly bond_length_a from G0."""
        p_g0, p_g1, p_g2 = _simple_ligand_positions()
        p_d0, p_d1, p_d2 = find_dummy_atom_positions(p_g0, p_g1, p_g2)
        dist = _bond_length(p_d0, p_g0)
        assert dist == pytest.approx(_DUMMY_BOND_LENGTH_A, abs=1e-4)

    def test_theta_B_is_90(self):
        """Angle D0-G0-G1 (theta_B) should be 90 degrees."""
        p_g0, p_g1, p_g2 = _simple_ligand_positions()
        p_d0, _, _ = find_dummy_atom_positions(p_g0, p_g1, p_g2)
        angle = _angle_deg(p_d0, p_g0, p_g1)
        assert angle == pytest.approx(90.0, abs=0.1)

    def test_theta_A_is_90(self):
        """Angle D1-D0-G0 (theta_A) should be 90 degrees."""
        p_g0, p_g1, p_g2 = _simple_ligand_positions()
        p_d0, p_d1, _ = find_dummy_atom_positions(p_g0, p_g1, p_g2)
        angle = _angle_deg(p_d1, p_d0, p_g0)
        assert angle == pytest.approx(90.0, abs=0.1)

    def test_phi_B_not_singular(self):
        """phi_B (D1-D0-G0-G1) should not be near 0 or 180 degrees."""
        p_g0, p_g1, p_g2 = _simple_ligand_positions()
        p_d0, p_d1, _ = find_dummy_atom_positions(p_g0, p_g1, p_g2)
        phi = abs(_dihedral_deg(p_d1, p_d0, p_g0, p_g1))
        assert phi > 10.0
        assert phi < 170.0

    def test_phi_A_approximately_60(self):
        """phi_A (D2-D1-D0-G0) should be ~60 degrees."""
        p_g0, p_g1, p_g2 = _simple_ligand_positions()
        p_d0, p_d1, p_d2 = find_dummy_atom_positions(p_g0, p_g1, p_g2)
        phi = abs(_dihedral_deg(p_d2, p_d1, p_d0, p_g0))
        assert phi == pytest.approx(60.0, abs=1.0)

    def test_custom_bond_length(self):
        p_g0, p_g1, p_g2 = _simple_ligand_positions()
        p_d0, _, _ = find_dummy_atom_positions(p_g0, p_g1, p_g2, bond_length_a=3.0)
        dist = _bond_length(p_d0, p_g0)
        assert dist == pytest.approx(3.0, abs=1e-4)

    def test_rotation_invariance(self):
        """Rotating the ligand frame should not change the inter-dummy angles."""
        p_g0, p_g1, p_g2 = _simple_ligand_positions()
        p_d0_orig, p_d1_orig, p_d2_orig = find_dummy_atom_positions(p_g0, p_g1, p_g2)
        theta_B_orig = _angle_deg(p_d0_orig, p_g0, p_g1)

        # Rotate all positions by 45 deg around z
        angle = np.deg2rad(45)
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0,              0,             1],
        ])
        g0r = R @ p_g0
        g1r = R @ p_g1
        g2r = R @ p_g2
        p_d0r, p_d1r, p_d2r = find_dummy_atom_positions(g0r, g1r, g2r)
        theta_B_rot = _angle_deg(p_d0r, g0r, g1r)

        assert theta_B_rot == pytest.approx(theta_B_orig, abs=0.5)

    def test_warns_on_collinear_input(self):
        """Collinear G0/G1/G2 should raise a UserWarning."""
        p_g0 = np.array([0.0, 0.0, 0.0])
        p_g1 = np.array([1.0, 0.0, 0.0])
        p_g2 = np.array([2.0, 0.0, 0.0])  # collinear
        with pytest.warns(UserWarning, match="collinear"):
            find_dummy_atom_positions(p_g0, p_g1, p_g2)

    @pytest.mark.parametrize("translation", [
        np.array([10.0, 0.0, 0.0]),
        np.array([0.0, -5.5, 3.2]),
    ])
    def test_translation_invariance_of_angles(self, translation):
        """Translating the ligand should not change the Boresch angles."""
        p_g0, p_g1, p_g2 = _simple_ligand_positions()
        p_d0, p_d1, _ = find_dummy_atom_positions(p_g0, p_g1, p_g2)
        theta_A_orig = _angle_deg(p_d1, p_d0, p_g0)
        theta_B_orig = _angle_deg(p_d0, p_g0, p_g1)

        g0t = p_g0 + translation
        g1t = p_g1 + translation
        g2t = p_g2 + translation
        p_d0t, p_d1t, _ = find_dummy_atom_positions(g0t, g1t, g2t)
        theta_A_t = _angle_deg(p_d1t, p_d0t, g0t)
        theta_B_t = _angle_deg(p_d0t, g0t, g1t)

        assert theta_A_t == pytest.approx(theta_A_orig, abs=0.1)
        assert theta_B_t == pytest.approx(theta_B_orig, abs=0.1)


def _make_simple_system(n_particles: int = 4) -> tuple[openmm.System, np.ndarray]:
    """
    Build a minimal OpenMM System with a NonbondedForce and HarmonicBondForce
    for testing dummy atom insertion.
    """
    system = openmm.System()
    nb = openmm.NonbondedForce()
    hb = openmm.HarmonicBondForce()

    for i in range(n_particles):
        system.addParticle(12.0)  # carbon mass
        nb.addParticle(float(i) * 0.1, 0.35, 0.5)  # charge, sigma, eps

    # Add one bond between particles 0 and 1
    hb.addBond(0, 1, 0.15, 5000.0)

    system.addForce(nb)
    system.addForce(hb)

    positions = np.random.rand(n_particles, 3).astype(np.float32) * 10.0
    return system, positions


class TestAddDummyAtomsToSystem:

    def test_particle_count_increases(self):
        system, positions = _make_simple_system(4)
        system, new_pos, dummy_idxs = add_dummy_atoms_to_system(system, positions, n_dummies=3)
        assert system.getNumParticles() == 7
        assert len(dummy_idxs) == 3
        assert new_pos.shape == (7, 3)

    def test_dummy_indices_are_correct(self):
        system, positions = _make_simple_system(4)
        system, _, dummy_idxs = add_dummy_atoms_to_system(system, positions, n_dummies=3)
        assert dummy_idxs == [4, 5, 6]

    def test_dummy_mass_is_large(self):
        system, positions = _make_simple_system(4)
        system, _, dummy_idxs = add_dummy_atoms_to_system(system, positions, n_dummies=3)
        for idx in dummy_idxs:
            mass = system.getParticleMass(idx).value_in_unit(openmm.unit.amu)
            assert mass == pytest.approx(DUMMY_MASS_AMU, rel=1e-6)

    def test_dummy_nonbonded_params_are_zero(self):
        system, positions = _make_simple_system(4)
        system, _, dummy_idxs = add_dummy_atoms_to_system(system, positions, n_dummies=3)

        nb = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
        for idx in dummy_idxs:
            charge, sigma, epsilon = nb.getParticleParameters(idx)
            assert charge.value_in_unit(openmm.unit.elementary_charge) == pytest.approx(0.0)
            assert epsilon.value_in_unit(openmm.unit.kilojoule_per_mole) == pytest.approx(0.0)

    def test_exclusions_added_for_all_existing_particles(self):
        n = 4
        system, positions = _make_simple_system(n)
        system, _, dummy_idxs = add_dummy_atoms_to_system(system, positions, n_dummies=1)

        nb = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
        # Collect all exception pairs involving the dummy
        dummy_idx = dummy_idxs[0]
        exception_pairs = set()
        for i in range(nb.getNumExceptions()):
            p1, p2, chargeProd, sigma, epsilon = nb.getExceptionParameters(i)
            pair = frozenset([p1, p2])
            if dummy_idx in pair:
                exception_pairs.add(pair)

        # Dummy must be excluded from all original particles
        for orig_idx in range(n):
            assert frozenset([dummy_idx, orig_idx]) in exception_pairs, (
                f"Missing exclusion between dummy {dummy_idx} and particle {orig_idx}"
            )

    def test_original_particle_count_preserved(self):
        """Original particles should not be modified."""
        n = 4
        system, positions = _make_simple_system(n)
        nb_before = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
        params_before = [nb_before.getParticleParameters(i) for i in range(n)]

        system, _, _ = add_dummy_atoms_to_system(system, positions, n_dummies=3)
        nb_after = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
        params_after = [nb_after.getParticleParameters(i) for i in range(n)]

        for i in range(n):
            assert params_before[i][0] == params_after[i][0]  # charge unchanged
            assert params_before[i][2] == params_after[i][2]  # epsilon unchanged

    def test_new_positions_shape(self):
        system, positions = _make_simple_system(4)
        _, new_pos, _ = add_dummy_atoms_to_system(system, positions, n_dummies=3)
        assert new_pos.shape == (7, 3)

    def test_new_positions_initialised_to_zero(self):
        system, positions = _make_simple_system(4)
        _, new_pos, dummy_idxs = add_dummy_atoms_to_system(system, positions, n_dummies=3)
        for idx in dummy_idxs:
            assert np.allclose(new_pos[idx], 0.0)

    def test_original_positions_preserved(self):
        system, positions = _make_simple_system(4)
        positions_copy = positions.copy()
        _, new_pos, _ = add_dummy_atoms_to_system(system, positions, n_dummies=3)
        np.testing.assert_array_equal(new_pos[:4], positions_copy)

    def test_dummies_excluded_from_each_other(self):
        """Dummy atoms must also be excluded from each other."""
        system, positions = _make_simple_system(2)
        system, _, dummy_idxs = add_dummy_atoms_to_system(system, positions, n_dummies=3)
        assert len(dummy_idxs) == 3

        nb = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
        exception_pairs = set()
        for i in range(nb.getNumExceptions()):
            p1, p2, *_ = nb.getExceptionParameters(i)
            exception_pairs.add(frozenset([p1, p2]))

        d0, d1, d2 = dummy_idxs
        for pair in [(d0, d1), (d0, d2), (d1, d2)]:
            assert frozenset(pair) in exception_pairs, (
                f"Missing exclusion between dummy atoms {pair}"
            )

    def test_harmonic_bond_force_untouched(self):
        """No bonds should be added to the HarmonicBondForce for dummies."""
        system, positions = _make_simple_system(4)
        hb_before = next(f for f in system.getForces() if isinstance(f, openmm.HarmonicBondForce))
        n_bonds_before = hb_before.getNumBonds()

        system, _, _ = add_dummy_atoms_to_system(system, positions, n_dummies=3)
        hb_after = next(f for f in system.getForces() if isinstance(f, openmm.HarmonicBondForce))
        assert hb_after.getNumBonds() == n_bonds_before