# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Additional tests for the dummy-atom Boresch restraints applied to the
SepTop solvent leg.

These tests are dry-run speed (no production MD) and target two things
the existing force-count/sampler-construction tests do not check:

1. Geometry sanity: the Boresch angles/dihedrals returned in
   ``restraint_geometry_A``/``restraint_geometry_B`` (the same
   ``BoreschRestraintGeometry`` objects the complex leg already exposes)
   are not close to the singular values 0 or 180 degrees, and the
   D0-G0 bond length matches the expected construction distance.

2. Energy correctness: evaluating the Boresch CustomCompoundBondForce
   at the reference geometry gives ~0 energy, and perturbing a ligand
   away from that geometry strictly increases the restraint energy.
   This catches atom-index-ordering bugs that a force-count check alone
   cannot.

Add these to test_septop_protocol.py, reusing the existing
`benzene_complex_system`, `toluene_complex_system`, and
`protocol_dry_settings` fixtures.
"""
import numpy as np
import openmm
import openmm.unit
import pytest

from openfe.protocols.openmm_septop import SepTopProtocol
from openfe.protocols.openmm_septop.septop_units import SepTopSolventSetupUnit

# Tolerance (degrees) below which an angle/dihedral is considered
# dangerously close to a singular value (0 or 180 deg).
_SINGULARITY_TOLERANCE_DEG = 5.0

# Expected D0-G0 (and by construction D1-D0, D2-D1) bond length, in
# Angstroms, set by geometry_dummy._DUMMY_BOND_LENGTH_A.
_EXPECTED_DUMMY_BOND_LENGTH_A = 5.0


def _get_solvent_setup_unit(dag):
    units = [u for u in dag.protocol_units if isinstance(u, SepTopSolventSetupUnit)]
    assert len(units) == 1
    return units[0]


def _find_boresch_forces(system: openmm.System) -> list[openmm.CustomCompoundBondForce]:
    """Return all CustomCompoundBondForce instances named 'Boresch-like'."""
    return [
        f
        for f in system.getForces()
        if isinstance(f, openmm.CustomCompoundBondForce) and f.getName() == "Boresch-like"
    ]


@pytest.fixture
def solvent_setup_output(
    benzene_complex_system, toluene_complex_system, tmp_path_factory, protocol_dry_settings
):
    """
    Run the solvent setup unit (dry run) and return its output.

    Function-scoped (the default) because the upstream fixtures
    (benzene_complex_system, toluene_complex_system, protocol_dry_settings)
    are themselves function-scoped — a broader-scoped fixture cannot
    depend on a narrower-scoped one in pytest.
    """
    protocol = SepTopProtocol(settings=protocol_dry_settings)
    dag = protocol.create(
        stateA=benzene_complex_system,
        stateB=toluene_complex_system,
        mapping=None,
    )
    setup_unit = _get_solvent_setup_unit(dag)
    tmp_path = tmp_path_factory.mktemp("solvent_dummy_boresch")
    return setup_unit.run(dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path)


class TestSolventDummyBoreschGeometry:
    """
    Geometry sanity checks for the dummy-atom Boresch restraints applied
    to the SepTop solvent leg, using the BoreschRestraintGeometry objects
    returned directly by the setup unit (mirroring the complex leg's
    restraint_geometry_A/B outputs).
    """

    @pytest.fixture
    def geometries(self, solvent_setup_output):
        geom_A = solvent_setup_output["restraint_geometry_A"]
        geom_B = solvent_setup_output["restraint_geometry_B"]
        return geom_A, geom_B

    def test_both_geometries_present(self, geometries):
        geom_A, geom_B = geometries
        assert geom_A is not None
        assert geom_B is not None

    @pytest.mark.parametrize("idx", [0, 1])
    def test_host_atoms_are_three_dummies(self, geometries, idx):
        """Each ligand's restraint should anchor to exactly 3 dummy atoms."""
        geom = geometries[idx]
        assert len(geom.host_atoms) == 3

    @pytest.mark.parametrize("idx", [0, 1])
    def test_guest_atoms_are_three_ligand_atoms(self, geometries, idx):
        """Each ligand's restraint should use exactly 3 ligand anchor atoms."""
        geom = geometries[idx]
        assert len(geom.guest_atoms) == 3

    @pytest.mark.parametrize("idx", [0, 1])
    def test_bond_length_matches_construction(self, geometries, idx):
        """
        r_aA0 (the D0-G0 equilibrium distance) should match the
        analytical construction distance used by find_dummy_atom_positions.
        """
        geom = geometries[idx]
        r_aA0_ang = geom.r_aA0.to("angstrom").magnitude
        assert r_aA0_ang == pytest.approx(_EXPECTED_DUMMY_BOND_LENGTH_A, abs=0.05)

    @pytest.mark.parametrize("idx", [0, 1])
    def test_angles_not_singular(self, geometries, idx):
        """
        theta_A0 and theta_B0 must be far from 0 or 180 degrees, or the
        restraint becomes numerically unstable.
        """
        geom = geometries[idx]
        theta_A0_deg = geom.theta_A0.to("degrees").magnitude
        theta_B0_deg = geom.theta_B0.to("degrees").magnitude

        for name, angle in [("theta_A0", theta_A0_deg), ("theta_B0", theta_B0_deg)]:
            assert angle > _SINGULARITY_TOLERANCE_DEG, f"{name} too close to 0 deg: {angle}"
            assert (
                angle < 180.0 - _SINGULARITY_TOLERANCE_DEG
            ), f"{name} too close to 180 deg: {angle}"

    @pytest.mark.parametrize("idx", [0, 1])
    def test_dihedrals_not_singular(self, geometries, idx):
        """
        phi_A0, phi_B0, phi_C0 must be far from 0 or 180 degrees.
        """
        geom = geometries[idx]
        phi_A0_deg = abs(geom.phi_A0.to("degrees").magnitude)
        phi_B0_deg = abs(geom.phi_B0.to("degrees").magnitude)
        phi_C0_deg = abs(geom.phi_C0.to("degrees").magnitude)

        for name, angle in [
            ("phi_A0", phi_A0_deg),
            ("phi_B0", phi_B0_deg),
            ("phi_C0", phi_C0_deg),
        ]:
            assert angle > _SINGULARITY_TOLERANCE_DEG, f"{name} too close to 0 deg: {angle}"
            assert (
                angle < 180.0 - _SINGULARITY_TOLERANCE_DEG
            ), f"{name} too close to 180 deg: {angle}"

    def test_dummy_atoms_are_appended_at_end_of_system(self, solvent_setup_output):
        """
        The 6 dummy atoms should be the last 6 particles in the system,
        with zero mass (immobile by construction) and zero nonbonded
        parameters.
        """
        system = solvent_setup_output["alchem_restrained_system"]
        n_total = system.getNumParticles()
        dummy_idxs = list(range(n_total - 6, n_total))

        for idx in dummy_idxs:
            mass = system.getParticleMass(idx).value_in_unit(openmm.unit.amu)
            assert mass == 0.0, f"Dummy particle {idx} should have zero mass, got: {mass}"

        nb_forces = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)]
        assert len(nb_forces) == 1
        nb = nb_forces[0]
        for idx in dummy_idxs:
            charge, sigma, epsilon = nb.getParticleParameters(idx)
            assert charge.value_in_unit(openmm.unit.elementary_charge) == pytest.approx(0.0)
            assert epsilon.value_in_unit(openmm.unit.kilojoule_per_mole) == pytest.approx(0.0)

    def test_host_atoms_match_dummy_particle_range(self, geometries, solvent_setup_output):
        """
        The host_atoms recorded in each geometry should fall within the
        last 6 particle indices of the system (the dummy atoms).
        """
        geom_A, geom_B = geometries
        system = solvent_setup_output["alchem_restrained_system"]
        n_total = system.getNumParticles()
        dummy_range = set(range(n_total - 6, n_total))

        for geom in (geom_A, geom_B):
            for idx in geom.host_atoms:
                assert idx in dummy_range, f"host atom {idx} is not in the dummy particle range"

    def test_custom_nonbonded_dummy_sigma_is_nonzero(self, solvent_setup_output):
        """
        Regression test for a specific crash: dummy atoms must have a
        non-zero sigma (length-scale) parameter in every
        CustomNonbondedForce, even though epsilon (interaction strength)
        is zero. A zero sigma previously caused
        ``CustomNonbondedForce: Long range correction did not converge``
        -- an uncatchable native abort -- because the alchemical softcore
        energy expression became non-decaying in r for that particle type.

        This only checks the built parameters (fast); the corresponding
        slow check that energy evaluation actually succeeds end-to-end is
        ``TestSolventDummyBoreschEnergy.test_dummy_atoms_unperturbed_by_short_dynamics``.
        """
        system = solvent_setup_output["alchem_restrained_system"]
        n_total = system.getNumParticles()
        dummy_idxs = list(range(n_total - 6, n_total))

        custom_nb_forces = [
            f for f in system.getForces() if isinstance(f, openmm.CustomNonbondedForce)
        ]
        assert len(custom_nb_forces) > 0, (
            "Expected at least one CustomNonbondedForce from the alchemical factory"
        )

        for force in custom_nb_forces:
            n_params = force.getNumPerParticleParameters()
            param_names = [
                force.getPerParticleParameterName(i).lower() for i in range(n_params)
            ]
            sigma_like_indices = [
                i
                for i, name in enumerate(param_names)
                if any(hint in name for hint in ("sigma", "radius", "rmin"))
            ]
            if not sigma_like_indices:
                # This force doesn't have a recognisable length-scale
                # parameter at all; nothing to check here.
                continue

            for idx in dummy_idxs:
                params = force.getParticleParameters(idx)
                for sigma_i in sigma_like_indices:
                    assert params[sigma_i] != 0.0, (
                        f"Dummy particle {idx}: parameter "
                        f"'{param_names[sigma_i]}' is zero in "
                        f"{type(force).__name__}, which will break the "
                        "long-range dispersion correction"
                    )


class TestSolventDummyBoreschEnergy:
    """
    Energy correctness checks: the Boresch restraint should evaluate to
    ~0 at the reference geometry and increase when a ligand is moved
    away from it.
    """

    @staticmethod
    def _get_boresch_group_energy(
        system: openmm.System,
        positions: openmm.unit.Quantity,
    ) -> float:
        """
        Evaluate only the force group(s) containing Boresch-like forces
        and return the total potential energy in kJ/mol.
        """
        boresch_forces = _find_boresch_forces(system)
        assert len(boresch_forces) == 2
        groups = {f.getForceGroup() for f in boresch_forces}

        integrator = openmm.VerletIntegrator(1.0 * openmm.unit.femtoseconds)
        platform = openmm.Platform.getPlatformByName("Reference")
        context = openmm.Context(system, integrator, platform)
        context.setPositions(positions)

        total_energy = 0.0
        for group in groups:
            state = context.getState(getEnergy=True, groups={group})
            total_energy += state.getPotentialEnergy().value_in_unit(
                openmm.unit.kilojoule_per_mole
            )

        del context, integrator
        return total_energy

    def test_energy_near_zero_at_reference_geometry(self, solvent_setup_output):
        """
        At the exact positions used to build the restraint, the Boresch
        energy should be ~0, since every term in the energy function is
        centered on its own equilibrium value at those positions.
        """
        system = solvent_setup_output["alchem_restrained_system"]
        positions = solvent_setup_output["positions"]

        energy = self._get_boresch_group_energy(system, positions)

        assert energy == pytest.approx(0.0, abs=1.0e-3)

    def test_energy_increases_when_ligand_a_displaced(self, solvent_setup_output):
        """
        Translating ligand A's guest atoms away from their restrained
        position should strictly increase the Boresch restraint energy.
        """
        system = solvent_setup_output["alchem_restrained_system"]
        positions = solvent_setup_output["positions"]
        geom_A = solvent_setup_output["restraint_geometry_A"]

        baseline_energy = self._get_boresch_group_energy(system, positions)

        perturbed = np.array(positions.value_in_unit(openmm.unit.nanometer), dtype=float)
        for idx in geom_A.guest_atoms:
            perturbed[idx, 0] += 0.5  # 0.5 nm shift along x
        perturbed_positions = perturbed * openmm.unit.nanometer

        perturbed_energy = self._get_boresch_group_energy(system, perturbed_positions)

        assert perturbed_energy > baseline_energy + 1.0, (
            f"Expected restraint energy to increase after displacing ligand A: "
            f"baseline={baseline_energy:.3f} kJ/mol, "
            f"perturbed={perturbed_energy:.3f} kJ/mol"
        )

    def test_energy_increases_when_ligand_b_displaced(self, solvent_setup_output):
        """
        Same check as above, but for ligand B's guest atoms.
        """
        system = solvent_setup_output["alchem_restrained_system"]
        positions = solvent_setup_output["positions"]
        geom_B = solvent_setup_output["restraint_geometry_B"]

        baseline_energy = self._get_boresch_group_energy(system, positions)

        perturbed = np.array(positions.value_in_unit(openmm.unit.nanometer), dtype=float)
        for idx in geom_B.guest_atoms:
            perturbed[idx, 1] += 0.5  # 0.5 nm shift along y
        perturbed_positions = perturbed * openmm.unit.nanometer

        perturbed_energy = self._get_boresch_group_energy(system, perturbed_positions)

        assert perturbed_energy > baseline_energy + 1.0, (
            f"Expected restraint energy to increase after displacing ligand B: "
            f"baseline={baseline_energy:.3f} kJ/mol, "
            f"perturbed={perturbed_energy:.3f} kJ/mol"
        )

    def test_dummy_atoms_unperturbed_by_short_dynamics(self, solvent_setup_output):
        """
        Running a handful of integration steps should leave the dummy
        atoms (last 6 particles) exactly stationary, confirming their zero
        mass excludes them from velocity initialisation and the
        integrator's position update entirely.

        This deliberately uses ``Context.setVelocitiesToTemperature`` and
        full force evaluation (the same operations used during real
        solvent-leg equilibration in ``PlainMDSimulationUnit._run_dynamics``)
        rather than zero-velocity initialisation, since this call sequence
        previously triggered an uncatchable native abort:
        ``CustomNonbondedForce: Long range correction did not converge``.
        That failure was caused by dummy atoms having sigma = 0 in the
        alchemical softcore CustomNonbondedForce instances, which made the
        force's energy expression non-decaying in r for that particle type
        -- independent of mass, and independent of the explicit exclusions
        already in place. Giving dummies a small non-zero sigma (alongside
        epsilon = 0) resolves it; see ``omm_dummy._dummy_custom_nonbonded_params``.
        This test exercises the real call sequence directly to confirm both
        fixes (zero mass, non-zero sigma) hold together.
        """
        system = solvent_setup_output["alchem_restrained_system"]
        positions = solvent_setup_output["positions"]
        n_total = system.getNumParticles()
        dummy_idxs = list(range(n_total - 6, n_total))

        integrator = openmm.LangevinMiddleIntegrator(
            300 * openmm.unit.kelvin,
            1.0 / openmm.unit.picosecond,
            1.0 * openmm.unit.femtoseconds,
        )
        platform = openmm.Platform.getPlatformByName("Reference")
        context = openmm.Context(system, integrator, platform)
        context.setPositions(positions)
        context.setVelocitiesToTemperature(300 * openmm.unit.kelvin)

        try:
            integrator.step(20)

            state = context.getState(getPositions=True)
            new_positions = np.array(
                state.getPositions(asNumpy=True).value_in_unit(openmm.unit.nanometer)
            )
            old_positions = np.array(positions.value_in_unit(openmm.unit.nanometer))

            for idx in dummy_idxs:
                displacement = np.linalg.norm(new_positions[idx] - old_positions[idx])
                assert displacement < 1.0e-8, (
                    f"Dummy atom {idx} moved {displacement:.10f} nm after 20 steps, "
                    "expected exactly 0 given its zero mass"
                )
        finally:
            del context, integrator