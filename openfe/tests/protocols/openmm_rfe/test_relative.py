import pytest
from openfe.tests.helpers import make_htf, _make_system_with_cmap
from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
from openfe.protocols.openmm_rfe._rfe_utils.relative import HybridTopologyFactory
from openmm import app, unit
import openmm
from openff.units import unit as offunit
from openfe.protocols.openmm_rfe import _rfe_utils
import copy


@pytest.mark.parametrize(
    "softcore_alpha", [
        pytest.param(0.5),
        pytest.param(0.75)
    ]
)
def test_softcore_parameters(chloroethane_to_fluoroethane_mapping, softcore_alpha):
    """
    Make sure the softcore parameters are correctly set by the HTF
    """
    settings = RelativeHybridTopologyProtocol.default_settings()
    settings.alchemical_settings.softcore_alpha = softcore_alpha
    htf = make_htf(mapping=chloroethane_to_fluoroethane_mapping, settings=settings)
    forces = {force.getName(): force for force in htf.hybrid_system.getForces()}
    # only the custom nonbonded forces and the custom bond sterics force should have these parameters set
    for force in [forces["CustomNonbondedForce"], forces["CustomBondForce_exceptions"]]:
        num_params = force.getNumGlobalParameters()
        hybrid_soft_core = None
        for i in range(num_params):
            # get the name of the parameter
            param_name = force.getGlobalParameterName(i)
            if param_name == "softcore_alpha":
                hybrid_soft_core = force.getGlobalParameterDefaultValue(i)
                break

        assert hybrid_soft_core == softcore_alpha

def test_particles_mass_no_dummy(htf_chloro_fluoroethane):
    """
    Make sure the number of particles is correct and the masses are as expected when no dummy atoms are in the hybrid system
    """
    hybrid_system = htf_chloro_fluoroethane["hybrid_system"]
    assert hybrid_system.getNumParticles() == 8
    expected_mass = [
        27.225801625000003 * unit.dalton,  # Average of Cl/F
        8.026674 * unit.dalton,  # Carbon with 2Hs and HMR
        6.034621 * unit.dalton,  # Carbon with 3Hs and HMR
        3 * unit.dalton,  # HMR Hydrogen
        3 * unit.dalton,
        3 * unit.dalton,
        3 * unit.dalton,
        3 * unit.dalton,
    ]
    for idx in range(hybrid_system.getNumParticles()):
        assert hybrid_system.getParticleMass(idx) == expected_mass[idx]

def test_particle_mass_dummy(htf_chloro_ethane):
    """
    Make sure the number of particles is correct and the masses are as expected when dummy atoms are in the hybrid system
    """
    hybrid_system = htf_chloro_ethane["hybrid_system"]
    # as we have a single unique atom at each end state there should be 9 particles
    assert hybrid_system.getNumParticles() == 9
    expected_mass = [
        35.4532 * unit.dalton,  # CL mass
        7.0306475 * unit.dalton, # Average mass of Carbon with 2Hs and 3Hs with HMR
        6.034621 * unit.dalton,  # Carbon with 3Hs and HMR
        3 * unit.dalton,  # HMR Hydrogen
        3 * unit.dalton,
        3 * unit.dalton,
        3 * unit.dalton,
        3 * unit.dalton,
        3 * unit.dalton,
    ]
    for idx in range(hybrid_system.getNumParticles()):
        assert hybrid_system.getParticleMass(idx) == expected_mass[idx]

def test_constraints_count_no_dummy(htf_chloro_fluoroethane):
    """The number of hydrogens does not change so 5 constraints in total."""
    assert htf_chloro_fluoroethane["hybrid_system"].getNumConstraints() == 5

def test_constraints_count_dummy(htf_chloro_ethane):
    """The number of hydrogens changes so we have an additional constraint for the unique ethane hydrogen."""
    assert htf_chloro_ethane["hybrid_system"].getNumConstraints() == 6

def test_hybrid_forces_no_dummy(htf_chloro_fluoroethane):
    """
    Test that we only have the expected forces in the hybrid system.
    """
    forces = htf_chloro_fluoroethane["forces"]
    expected_forces = {
        "CustomBondForce",
        "HarmonicBondForce",
        "CustomAngleForce",
        "HarmonicAngleForce",
        "CustomTorsionForce",
        "PeriodicTorsionForce",
        "NonbondedForce",
        "CustomNonbondedForce",
        "CustomBondForce_exceptions",
    }
    assert not set(forces.keys()) - expected_forces

def test_hybrid_forces_dummy(htf_chloro_ethane):
    forces = htf_chloro_ethane["forces"]
    expected_forces = {
        "CustomBondForce",
        "HarmonicBondForce",
        "CustomAngleForce",
        "HarmonicAngleForce",
        "CustomTorsionForce",
        "PeriodicTorsionForce",
        "NonbondedForce",
        "CustomNonbondedForce",
        "CustomBondForce_exceptions",
    }
    assert not set(forces.keys()) - expected_forces


def test_bond_force_no_dummy(htf_chloro_fluoroethane):
    """
    Test the standard and interpolated custom forces are correctly setup when we have no dummy atoms.
    """
    forces = htf_chloro_fluoroethane["forces"]
    mapping = htf_chloro_fluoroethane["mapping"]
    chloro_labels = htf_chloro_fluoroethane["chloro_labels"]
    fluoro_labels = htf_chloro_fluoroethane["fluoro_labels"]

    # there should be no standard bond force terms
    standard_bond_force = forces["HarmonicBondForce"]
    assert standard_bond_force.getNumBonds() == 0
    # there should be two forces in the interpolated bond force
    custom_bond_force = forces["CustomBondForce"]
    # there should be a single global parameter for lambda
    assert custom_bond_force.getNumGlobalParameters() == 1
    # make sure it has the correct name
    assert custom_bond_force.getGlobalParameterName(0) == "lambda_bonds"

    num_bonds = custom_bond_force.getNumBonds()
    assert num_bonds == 2
    # now check the parameters are correctly interpolated
    for i in range(num_bonds):
        p1, p2, params = custom_bond_force.getBondParameters(i)
        # p1, p2 are the index in chloroethane get the expected parameters from the labels
        chloro_bond = chloro_labels["Bonds"][(p1, p2)]
        # make sure the initial parameters match chloroethane
        # this also implicitly checks the per bond parameters have been entered in the expected order
        assert params[0] == chloro_bond.length.m_as(offunit.nanometer)
        assert params[1] == chloro_bond.k.m_as(offunit.kilojoule_per_mole / offunit.nanometer ** 2)
        # then check the fluoro parameters
        # map the index first
        f1 = mapping.componentA_to_componentB[p1]
        f2 = mapping.componentA_to_componentB[p2]
        fluoro_bond = fluoro_labels["Bonds"][(f1, f2)]
        assert params[2] == fluoro_bond.length.m_as(offunit.nanometer)
        assert params[3] == fluoro_bond.k.m_as(offunit.kilojoule_per_mole / offunit.nanometer ** 2)

def test_bond_force_dummy(htf_chloro_ethane):

    forces = htf_chloro_ethane["forces"]
    mapping = htf_chloro_ethane["mapping"]
    chloro_labels = htf_chloro_ethane["chloro_labels"]
    ethane_labels = htf_chloro_ethane["ethane_labels"]

    # there should be one standard bond force term (non-interpolated)
    # as terms with a dummy atom are held fixed
    # there would be 2 if the transformed atom was not H and involved in a constraint
    standard_bond_force = forces["HarmonicBondForce"]
    num_bonds = standard_bond_force.getNumBonds()
    assert num_bonds == 1
    for i in range(num_bonds):
        p1, p2, length, k = standard_bond_force.getBondParameters(i)
        # make sure they correspond to the expected values in stateA chloroethane
        chloro_bond = chloro_labels["Bonds"][(p1, p2)]
        assert length == chloro_bond.length.m_as(offunit.nanometer) * unit.nanometer
        assert k == chloro_bond.k.m_as(offunit.kilojoule_per_mole / offunit.nanometer ** 2) * unit.kilojoule_per_mole / unit.nanometer**2

    # there should then be one interpolated (fully mapped) bond force for the central carbons
    custom_bond_force = forces["CustomBondForce"]
    # there should be a single global parameter for lambda
    assert custom_bond_force.getNumGlobalParameters() == 1
    # make sure it has the correct name
    assert custom_bond_force.getGlobalParameterName(0) == "lambda_bonds"

    num_bonds = custom_bond_force.getNumBonds()
    assert num_bonds == 1
    for i in range(num_bonds):
        p1, p2, params = custom_bond_force.getBondParameters(i)
        # p1, p2 are the index in chloroethane get the expected parameters from the labels
        chloro_bond = chloro_labels["Bonds"][(p1, p2)]
        # make sure the initial parameters match chloroethane
        # this also implicitly checks the per bond parameters have been entered in the expected order
        assert params[0] == chloro_bond.length.m_as(offunit.nanometer)
        assert params[1] == chloro_bond.k.m_as(offunit.kilojoule_per_mole / offunit.nanometer ** 2)
        # then check the ethane parameters
        # map the index first
        e1 = mapping.componentA_to_componentB[p1]
        e2 = mapping.componentA_to_componentB[p2]
        ethane_bond = ethane_labels["Bonds"][(e1, e2)]
        assert params[2] == ethane_bond.length.m_as(offunit.nanometer)
        assert params[3] == ethane_bond.k.m_as(offunit.kilojoule_per_mole / offunit.nanometer ** 2)

def test_angle_force_no_dummy(htf_chloro_fluoroethane):
    """
    Test the standard and interpolated custom angle forces are correctly setup when we have no dummy atoms.
    """
    forces = htf_chloro_fluoroethane["forces"]
    mapping = htf_chloro_fluoroethane["mapping"]
    chloro_labels = htf_chloro_fluoroethane["chloro_labels"]
    fluoro_labels = htf_chloro_fluoroethane["fluoro_labels"]

    # there should be no standard angle force terms
    standard_angle_force = forces["HarmonicAngleForce"]
    assert standard_angle_force.getNumAngles() == 0
    # there should be 12 forces in the interpolated angle force even if the parameters are not interpolated
    custom_angle_force = forces["CustomAngleForce"]
    # there should be a single global parameter for lambda
    assert custom_angle_force.getNumGlobalParameters() == 1
    # make sure it has the correct name
    assert custom_angle_force.getGlobalParameterName(0) == "lambda_angles"

    num_angles = custom_angle_force.getNumAngles()
    assert num_angles == 12
    # now check the parameters are correctly interpolated
    for i in range(num_angles):
        p1, p2, p3, params = custom_angle_force.getAngleParameters(i)
        # p1, p2, p3 are the index in chloroethane get the expected parameters from the labels
        chloro_angle = chloro_labels["Angles"][(p1, p2, p3)]
        # make sure the initial parameters match chloroethane
        # this also implicitly checks the per angle parameters have been entered in the expected order
        assert params[0] == chloro_angle.angle.m_as(offunit.radian)
        assert params[1] == chloro_angle.k.m_as(offunit.kilojoule_per_mole / offunit.radian ** 2)
        # then check the fluoro parameters
        # map the index first
        f1 = mapping.componentA_to_componentB[p1]
        f2 = mapping.componentA_to_componentB[p2]
        f3 = mapping.componentA_to_componentB[p3]
        fluoro_angle = fluoro_labels["Angles"][(f1, f2, f3)]
        assert params[2] == fluoro_angle.angle.m_as(offunit.radian)
        assert params[3] == fluoro_angle.k.m_as(offunit.kilojoule_per_mole / offunit.radian ** 2)

def test_angle_force_dummy(htf_chloro_ethane):
    forces = htf_chloro_ethane["forces"]
    mapping = htf_chloro_ethane["mapping"]
    chloro_labels = htf_chloro_ethane["chloro_labels"]
    ethane_labels = htf_chloro_ethane["ethane_labels"]

    # there should be 6 standard angle force terms (non-interpolated)
    # 3 for chloroethane involving the Cl atom
    # 3 for ethane involving the unique H atom
    standard_angle_force = forces["HarmonicAngleForce"]
    num_angles = standard_angle_force.getNumAngles()
    assert num_angles == 6
    for i in range(num_angles):
        p1, p2, p3, angle, k = standard_angle_force.getAngleParameters(i)
        # if the starting atom index is 0 it is a chloroethane angle else ethane
        if p1 == 0 or p3 ==0:
            chloro_angle = chloro_labels["Angles"][(p1, p2, p3)]
            assert angle == chloro_angle.angle.m_as(offunit.radian) * unit.radian
            assert k == chloro_angle.k.m_as(offunit.kilojoule_per_mole / offunit.radian ** 2) * unit.kilojoule_per_mole / unit.radian**2
        else:
            # manually map the Cl - H
            e1 = 0
            e2 = mapping.componentA_to_componentB[p2]
            e3 = mapping.componentA_to_componentB[p3]
            ethane_angle = ethane_labels["Angles"][(e1, e2, e3)]
            assert angle == ethane_angle.angle.m_as(offunit.radian) * unit.radian
            assert k == ethane_angle.k.m_as(offunit.kilojoule_per_mole / offunit.radian ** 2) * unit.kilojoule_per_mole / unit.radian**2

    # there should then be 9 interpolated (fully mapped) angle terms
    custom_angle_force = forces["CustomAngleForce"]
    # there should be a single global parameter for lambda
    assert custom_angle_force.getNumGlobalParameters() == 1
    # make sure it has the correct name
    assert custom_angle_force.getGlobalParameterName(0) == "lambda_angles"

    num_angles = custom_angle_force.getNumAngles()
    assert num_angles == 9
    for i in range(num_angles):
        p1, p2, p3, params = custom_angle_force.getAngleParameters(i)
        # p1, p2, p3 are the index in chloroethane get the expected parameters from the labels
        chloro_angle = chloro_labels["Angles"][(p1, p2, p3)]
        # make sure the initial parameters match chloroethane
        # this also implicitly checks the per angle parameters have been entered in the expected order
        assert params[0] == chloro_angle.angle.m_as(offunit.radian)
        assert params[1] == chloro_angle.k.m_as(offunit.kilojoule_per_mole / offunit.radian ** 2)
        # then check the ethane parameters
        # map the index first
        e1 = mapping.componentA_to_componentB[p1]
        e2 = mapping.componentA_to_componentB[p2]
        e3 = mapping.componentA_to_componentB[p3]
        ethane_angle = ethane_labels["Angles"][(e1, e2, e3)]
        assert params[2] == ethane_angle.angle.m_as(offunit.radian)
        assert params[3] == ethane_angle.k.m_as(offunit.kilojoule_per_mole / offunit.radian ** 2)

def test_torsion_force_no_dummy(htf_chloro_fluoroethane):
    """
    Test the standard and interpolated custom torsion forces are correctly setup when we have no dummy atoms.
    """
    forces = htf_chloro_fluoroethane["forces"]
    mapping = htf_chloro_fluoroethane["mapping"]
    chloro_labels = htf_chloro_fluoroethane["chloro_labels"]
    fluoro_labels = htf_chloro_fluoroethane["fluoro_labels"]

    # there should be 6 standard torsion force terms (non-interpolated)
    standard_torsion_force = forces["PeriodicTorsionForce"]
    num_torsions = standard_torsion_force.getNumTorsions()
    assert num_torsions == 6
    # now check the parameters are correctly assigned
    for i in range(num_torsions):
        p1, p2, p3, p4, periodicity, phase, k = standard_torsion_force.getTorsionParameters(i)
        # p1, p2, p3, p4 are the index in chloroethane get the expected parameters from the labels
        chloro_torsion = chloro_labels["ProperTorsions"][(p1, p2, p3, p4)]
        assert periodicity == chloro_torsion.periodicity1
        assert phase == chloro_torsion.phase1.m_as(offunit.radian) * unit.radian
        assert k == chloro_torsion.k1.m_as(offunit.kilojoule_per_mole) * unit.kilojoule_per_mole
        # map to fluoroethane
        f1 = mapping.componentA_to_componentB[p1]
        f2 = mapping.componentA_to_componentB[p2]
        f3 = mapping.componentA_to_componentB[p3]
        f4 = mapping.componentA_to_componentB[p4]
        fluoro_torsion = fluoro_labels["ProperTorsions"][(f1, f2, f3, f4)]
        # make sure those parameters also match
        assert periodicity == fluoro_torsion.periodicity1
        assert phase == fluoro_torsion.phase1.m_as(offunit.radian) * unit.radian
        assert k == fluoro_torsion.k1.m_as(offunit.kilojoule_per_mole) * unit.kilojoule_per_mole

    # custom torsion forces
    custom_torsion_force = forces["CustomTorsionForce"]
    # there should be a single global parameter for lambda
    assert custom_torsion_force.getNumGlobalParameters() == 1
    # make sure it has the correct name
    assert custom_torsion_force.getGlobalParameterName(0) == "lambda_torsions"

    num_torsions = custom_torsion_force.getNumTorsions()
    # we have 3 interpolated torsions with two k values each which is 6 terms
    # but the HTF interpolates each torsion from and to zero so 6 * 2 total parameters
    assert num_torsions == 12
    for i in range(num_torsions):
        p1, p2, p3, p4, params = custom_torsion_force.getTorsionParameters(i)
        # check which end state the parameters correspond to
        # if the 3 starting parameters are zero its fluoroethane else chloroethane
        if params[0] == 0.0 and params[1] == 0.0 and params[2] == 0.0:
            # fluoroethane
            f1 = mapping.componentA_to_componentB[p1]
            f2 = mapping.componentA_to_componentB[p2]
            f3 = mapping.componentA_to_componentB[p3]
            f4 = mapping.componentA_to_componentB[p4]
            fluoro_torsion = fluoro_labels["ProperTorsions"][(f1, f2, f3, f4)]
            # now we need to check which interaction this is if we have multiple periodicity's
            periodicity = params[3]
            term_index = fluoro_torsion.periodicity.index(periodicity)
            assert periodicity == fluoro_torsion.periodicity[term_index]
            assert params[4] == fluoro_torsion.phase[term_index].m_as(offunit.radian)
            assert params[5] == fluoro_torsion.k[term_index].m_as(offunit.kilojoule_per_mole)

        else:
            # chloroethane
            chloro_torsion = chloro_labels["ProperTorsions"][(p1, p2, p3, p4)]
            periodicity = params[0]
            term_index = chloro_torsion.periodicity.index(periodicity)
            assert periodicity == chloro_torsion.periodicity[term_index]
            assert params[1] == chloro_torsion.phase[term_index].m_as(offunit.radian)
            assert params[2] == chloro_torsion.k[term_index].m_as(offunit.kilojoule_per_mole)

def test_torsion_force_dummy(htf_chloro_ethane):
    forces = htf_chloro_ethane["forces"]
    mapping = htf_chloro_ethane["mapping"]
    chloro_labels = htf_chloro_ethane["chloro_labels"]
    ethane_labels = htf_chloro_ethane["ethane_labels"]

    # there should be no interpolated torsion force terms
    # as those involving a dummy atom are held fixed
    # and in this example the fully mapped torsions have the same parameters
    custom_torsion_force = forces["CustomTorsionForce"]
    assert custom_torsion_force.getNumTorsions() == 0
    # there should be a single global parameter for lambda
    assert custom_torsion_force.getNumGlobalParameters() == 1
    # make sure it has the correct name
    assert custom_torsion_force.getGlobalParameterName(0) == "lambda_torsions"

    # 15 torsion terms in total in the standard periodic torsion force
    # chloroethane has 3 unique torsions with 2 phases = 6
    # chloroethane and ethane have 6 shared single phase torsions = 6
    # ethane has 3 unique torsions with a single phase = 3
    standard_torsion_force = forces["PeriodicTorsionForce"]
    num_torsions = standard_torsion_force.getNumTorsions()
    assert num_torsions == 15
    # now check the parameters are correctly assigned
    for i in range(num_torsions):
        p1, p2, p3, p4, periodicity, phase, k = standard_torsion_force.getTorsionParameters(i)
        # determine if this is chloroethane or ethane
        if p1 == 0:
            # unique chloroethane torsion
            chloro_torsion = chloro_labels["ProperTorsions"][(p1, p2, p3, p4)]
            assert periodicity in chloro_torsion.periodicity
            term_index = chloro_torsion.periodicity.index(periodicity)
            assert phase == chloro_torsion.phase[term_index].m_as(offunit.radian) * unit.radian
            assert k == chloro_torsion.k[term_index].m_as(offunit.kilojoule_per_mole) * unit.kilojoule_per_mole
        elif p1 == 8:
            # unique ethane torsion
            e1 = 0
            e2 = mapping.componentA_to_componentB[p2]
            e3 = mapping.componentA_to_componentB[p3]
            e4 = mapping.componentA_to_componentB[p4]
            ethane_torsion = ethane_labels["ProperTorsions"][(e1, e2, e3, e4)]
            assert periodicity in ethane_torsion.periodicity
            term_index = ethane_torsion.periodicity.index(periodicity)
            assert phase == ethane_torsion.phase[term_index].m_as(offunit.radian) * unit.radian
            assert k == ethane_torsion.k[term_index].m_as(offunit.kilojoule_per_mole) * unit.kilojoule_per_mole

        else:
            # we have a mapped torsion so check the parameters are the same in both molecules
            chloro_torsion = chloro_labels["ProperTorsions"][(p1, p2, p3, p4)]
            f1 = mapping.componentA_to_componentB[p1]
            f2 = mapping.componentA_to_componentB[p2]
            f3 = mapping.componentA_to_componentB[p3]
            f4 = mapping.componentA_to_componentB[p4]
            ethane_torsion = ethane_labels["ProperTorsions"][(f1, f2, f3, f4)]
            assert periodicity in chloro_torsion.periodicity
            term_index = chloro_torsion.periodicity.index(periodicity)
            assert phase == chloro_torsion.phase[term_index].m_as(offunit.radian) * unit.radian
            assert k == chloro_torsion.k[term_index].m_as(offunit.kilojoule_per_mole) * unit.kilojoule_per_mole
            # check ethane parameters match
            assert periodicity in ethane_torsion.periodicity
            term_index = ethane_torsion.periodicity.index(periodicity)
            assert phase == ethane_torsion.phase[term_index].m_as(offunit.radian) * unit.radian
            assert k == ethane_torsion.k[term_index].m_as(offunit.kilojoule_per_mole) * unit.kilojoule_per_mole


def test_nonbonded_force_no_dummy(htf_chloro_fluoroethane):
    """
    Test the nonbonded particle parameters are correctly set when we have no dummy atoms.
    """
    forces = htf_chloro_fluoroethane["forces"]
    chloro_labels = htf_chloro_fluoroethane["chloro_labels"]

    nonbonded_force = forces["NonbondedForce"]
    # there should be 4 global parameters used to scale the particle offsets
    assert nonbonded_force.getNumGlobalParameters() == 4
    expected_global_params = {
        "lambda_electrostatics_core",
        "lambda_electrostatics_insert",
        "lambda_electrostatics_delete",
        "lambda_sterics_core",
    }
    actual_global_params = {
        nonbonded_force.getGlobalParameterName(i) for i in range(nonbonded_force.getNumGlobalParameters())
    }
    assert actual_global_params == expected_global_params


    # as the particles are fully mapped we should have just 8 particles
    num_atoms = nonbonded_force.getNumParticles()
    assert num_atoms == 8
    # check the input parameters match the chloroethane parameters
    chloro_charges = htf_chloro_fluoroethane["chloro_charges"]
    for i in range(num_atoms):
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(i)
        chloro_vdw = chloro_labels["vdW"][(i,)]
        assert charge == chloro_charges[i] * unit.elementary_charge
        assert sigma == chloro_vdw.sigma.m_as(offunit.nanometer) * unit.nanometer
        # this is always zero as we use the softcore potential for the vdw
        assert epsilon == 0.0 * unit.kilojoule_per_mole

def test_nonbonded_force_dummy(htf_chloro_ethane):
    """Test the nonbonded particle parameters are correctly set when we have dummy atoms."""
    forces = htf_chloro_ethane["forces"]
    chloro_labels = htf_chloro_ethane["chloro_labels"]
    ethane_labels = htf_chloro_ethane["ethane_labels"]

    nonbonded_force = forces["NonbondedForce"]
    # there should be 4 global parameters used to scale the particle offsets
    assert nonbonded_force.getNumGlobalParameters() == 4
    expected_global_params = {
        "lambda_electrostatics_core",
        "lambda_electrostatics_insert",
        "lambda_electrostatics_delete",
        "lambda_sterics_core",
    }
    actual_global_params = {
        nonbonded_force.getGlobalParameterName(i) for i in range(nonbonded_force.getNumGlobalParameters())
    }
    assert actual_global_params == expected_global_params

    # as we have a single unique atom at each end state there should be 9 particles
    num_atoms = nonbonded_force.getNumParticles()
    assert num_atoms == 9
    # check the input parameters match the chloroethane parameters for the mapped atoms
    chloro_charges = htf_chloro_ethane["chloro_charges"]

    for i in range(num_atoms):
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(i)
        if i == 0:
            # unique chloro atom
            chloro_vdw = chloro_labels["vdW"][(i,)]
            assert charge == chloro_charges[i] * unit.elementary_charge
            assert sigma == chloro_vdw.sigma.m_as(offunit.nanometer) * unit.nanometer
            # we use soft core for vdW so epsilon is zero
            assert epsilon == 0.0 * unit.kilojoule_per_mole
        elif i == 8:
            # unique ethane hydrogen atom
            e1 = 0
            ethane_vdw = ethane_labels["vdW"][(e1,)]
            assert charge == 0.0 * unit.elementary_charge  # this should be zero and be scaled to the ethane value via particle offsets
            assert sigma == ethane_vdw.sigma.m_as(offunit.nanometer) * unit.nanometer
            # we use soft core for vdW so epsilon is zero
            assert epsilon == 0.0 * unit.kilojoule_per_mole
        else:
            # mapped atoms should use the chloroethane parameters
            # they will be adjusted using particle offsets
            chloro_vdw = chloro_labels["vdW"][(i,)]
            assert charge == chloro_charges[i] * unit.elementary_charge
            assert sigma == chloro_vdw.sigma.m_as(offunit.nanometer) * unit.nanometer
            #  we use soft core for vdW so epsilon is zero
            assert epsilon == 0.0 * unit.kilojoule_per_mole

def test_nonbonded_offsets_no_dummy(htf_chloro_fluoroethane):
    """Test that the nonbonded particle parameter offsets are correctly set when we have no dummy atoms."""
    forces = htf_chloro_fluoroethane["forces"]
    mapping = htf_chloro_fluoroethane["mapping"]

    nonbonded_force = forces["NonbondedForce"]

    # get the charges
    chloro_charges = htf_chloro_fluoroethane["chloro_charges"]
    fluoro_charges = htf_chloro_fluoroethane["fluoro_charges"]
    # We scale the nonbonded electrostatics with lambda so check the offsets
    # there should be 8 offsets one for each particle
    num_offsets =  nonbonded_force.getNumParticleParameterOffsets()
    assert num_offsets == 8
    for i in range(num_offsets):
        offset_params = nonbonded_force.getParticleParameterOffset(i)
        assert offset_params[0] == "lambda_electrostatics_core"  # Make sure only the electrostatics core lambda is used
        particle_index = offset_params[1]
        # make sure the epsilon and sigma scales are zero
        assert offset_params[3] == offset_params[4] == 0.0
        # calculate the scale for this particle index
        f1 = mapping.componentA_to_componentB[particle_index]
        charge_scale = fluoro_charges[f1] - chloro_charges[particle_index]
        # check the offset value matches
        assert offset_params[2] == charge_scale

def test_nonbonded_offsets_dummy(htf_chloro_ethane):
    forces = htf_chloro_ethane["forces"]
    mapping = htf_chloro_ethane["mapping"]

    nonbonded_force = forces["NonbondedForce"]
    # there should be 9 offsets one for each particle
    num_offsets =  nonbonded_force.getNumParticleParameterOffsets()
    assert num_offsets == 9
    chloro_charges = htf_chloro_ethane["chloro_charges"]
    ethane_charges = htf_chloro_ethane["ethane_charges"]
    for i in range(num_offsets):
        offset_params = nonbonded_force.getParticleParameterOffset(i)
        particle_index = offset_params[1]
        # make sure the epsilon and sigma scales are zero
        assert offset_params[3] == offset_params[4] == 0.0
        # calculate the scale for this particle index
        if particle_index == 0:
            # unique chloro atom
            # make sure this particle is removed at lambda = 1
            assert offset_params[0] == "lambda_electrostatics_delete"
            charge_scale = 0.0 - chloro_charges[particle_index]
        elif particle_index == 8:
            # unique ethane hydrogen atom
            # make sure this particle is inserted at lambda = 1
            assert offset_params[0] == "lambda_electrostatics_insert"
            e1 = 0
            charge_scale = ethane_charges[e1] - 0.0
        else:
            # mapped atoms
            # make sure the charge is scaled with the electrostatics core lambda
            assert offset_params[0] == "lambda_electrostatics_core"
            f1 = mapping.componentA_to_componentB[particle_index]
            charge_scale = ethane_charges[f1] - chloro_charges[particle_index]
        # check the offset value matches
        assert offset_params[2] == charge_scale

def test_nonbonded_exceptions_no_dummy(htf_chloro_fluoroethane):
    """Test that the nonbonded exceptions are correctly set when we have no dummy atoms."""
    forces = htf_chloro_fluoroethane["forces"]
    chloroethane = htf_chloro_fluoroethane["chloroethane"]
    chloro_openff = chloroethane.to_openff()
    chloro_labels = htf_chloro_fluoroethane["chloro_labels"]
    chloro_charges = htf_chloro_fluoroethane["chloro_charges"]
    electro_scale = htf_chloro_fluoroethane["electrostatic_scale"]
    vdw_scale = htf_chloro_fluoroethane["vdW_scale"]

    nonbonded_force = forces["NonbondedForce"]

    num_exceptions = nonbonded_force.getNumExceptions()
    # there should be 28 in total (8 * 7) / 2
    assert num_exceptions == 28

    # there should 9 non-zero exceptions corresponding to the 9 proper torsions
    exception_1_4s = []
    # get all atoms with a minimal path of 3 bonds between them
    for pair_1_4 in chloro_openff.nth_degree_neighbors(3):
        a1 = chloro_openff.atoms.index(pair_1_4[0])
        a2 = chloro_openff.atoms.index(pair_1_4[1])
        exception_1_4s.append((a1, a2))
    assert len(exception_1_4s) == 9
    non_zero_exceptions = 0

    # now check the parameters are correctly assigned
    for i in range(num_exceptions):
        p1, p2, charge_prod, sigma, epsilon = nonbonded_force.getExceptionParameters(i)
        # check if this is a 1-4 interaction, should use the chloroethane parameters
        if (p1, p2) in exception_1_4s or (p2, p1) in exception_1_4s:
            charge1 = chloro_charges[p1]
            charge2 = chloro_charges[p2]
            expected_charge_prod = charge1 * charge2 * electro_scale # get the scaled charge product
            chloro_vdw1 = chloro_labels["vdW"][(p1,)]
            chloro_vdw2 = chloro_labels["vdW"][(p2,)]
            # Lorentz-Berthelot combining rules
            expected_sigma = (chloro_vdw1.sigma + chloro_vdw2.sigma) / 2.0
            expected_epsilon = ((chloro_vdw1.epsilon * chloro_vdw2.epsilon) ** 0.5) * vdw_scale  # scaled epsilon by the 1-4 scale for the ff
            assert expected_charge_prod == pytest.approx(charge_prod.value_in_unit(unit.elementary_charge**2), rel=1e-5)  # charge product
            assert sigma == expected_sigma.m_as(offunit.nanometer) * unit.nanometer  # sigma
            assert expected_epsilon.m_as(offunit.kilojoule_per_mole) == pytest.approx(epsilon.value_in_unit(unit.kilojoule_per_mole), rel=1e-5)  # epsilon
            # track how many non-zero exceptions we have found
            non_zero_exceptions +=1
        # not a 1-4 so this should be set to zero
        else:
            assert charge_prod == 0.0 * unit.elementary_charge**2  # charge product
            assert sigma == 1.0 * unit.nanometer  # sigma, dummy value of 1 used
            assert epsilon == 0.0 * unit.kilojoule_per_mole  # epsilon, should always be zero

    assert non_zero_exceptions == 9

def test_nonbonded_exceptions_dummy(htf_chloro_ethane):
    """Test that the nonbonded exceptions are correctly set when we have dummy atoms, any involving a dummy should be zeroed."""
    forces = htf_chloro_ethane["forces"]
    chloroethane = htf_chloro_ethane["chloroethane"]
    chloro_labels = htf_chloro_ethane["chloro_labels"]
    chloro_charges = htf_chloro_ethane["chloro_charges"]
    chloro_openff = chloroethane.to_openff()
    electro_scale = htf_chloro_ethane["electrostatic_scale"]
    vdw_scale = htf_chloro_ethane["vdW_scale"]

    nonbonded_force = forces["NonbondedForce"]

    num_exceptions = nonbonded_force.getNumExceptions()
    # there should be 36 in total (9 * 8) / 2
    assert num_exceptions == 36

    # get the expected exception atoms
    exception_1_4s = []
    # get all atoms with a minimal path of 3 bonds between them
    for pair_1_4 in chloro_openff.nth_degree_neighbors(3):
        a1 = chloro_openff.atoms.index(pair_1_4[0])
        a2 = chloro_openff.atoms.index(pair_1_4[1])
        exception_1_4s.append((a1, a2))
    assert len(exception_1_4s) == 9
    # manually add the 1-4s involving the unique ethane hydrogen atom (atom 8)
    for i in [5, 6, 7]:
        exception_1_4s.append((8, i))

    # there should 6 non-zero exceptions corresponding to the 6 mapped proper torsions not involving a dummy atom
    non_zero_exceptions = 0
    # there should be 6 zeroed exceptions corresponding to those involving a dummy atom
    zeroed_dummy_exceptions = 0

    # now check the parameters are correctly assigned
    for i in range(num_exceptions):
        p1, p2, charge_prod, sigma, epsilon = nonbonded_force.getExceptionParameters(i)
        # check if this is a 1-4 interaction
        if (p1, p2) in exception_1_4s or (p2, p1) in exception_1_4s:
            if p1 == 0 or p2 == 0 or p1 == 8 or p2 == 8:
                # this is a dummy atom exception which should be interpolated in the custom steric bond force
                # make sure the parameters are set to zero
                assert charge_prod == 0.0 * unit.elementary_charge**2  # charge product should always be zero
                assert epsilon == 0.0 * unit.kilojoule_per_mole  # epsilon, should always be zero
                # sigma will use a dummy value this is not important as epsilon is 0.0
                zeroed_dummy_exceptions += 1
            else:
                # this is a mapped exception so check the parameters match chloroethane
                charge1 = chloro_charges[p1]
                charge2 = chloro_charges[p2]
                expected_charge_prod = charge1 * charge2 * electro_scale  # get the scaled charge product
                chloro_vdw1 = chloro_labels["vdW"][(p1,)]
                chloro_vdw2 = chloro_labels["vdW"][(p2,)]
                # Lorentz-Berthelot combining rules
                expected_sigma = (chloro_vdw1.sigma + chloro_vdw2.sigma) / 2.0
                expected_epsilon = ((chloro_vdw1.epsilon * chloro_vdw2.epsilon) ** 0.5) * vdw_scale # scaled epsilon by the 1-4 scale for the ff
                assert expected_charge_prod ==  pytest.approx(charge_prod.value_in_unit(unit.elementary_charge**2), rel=1e-5)  # charge product
                assert sigma == expected_sigma.m_as(offunit.nanometer) * unit.nanometer  # sigma
                assert expected_epsilon.m_as(offunit.kilojoule_per_mole) ==pytest.approx(epsilon.value_in_unit(unit.kilojoule_per_mole), rel=1e-5)  # epsilon
                # track how many non-zero exceptions we have found
                non_zero_exceptions += 1
        # not a 1-4 so this should be set to zero
        else:
            assert charge_prod == 0.0 * unit.elementary_charge**2  # charge product should always be zero
            assert sigma == 1.0 * unit.nanometer  # sigma, dummy value of 1 used
            assert epsilon == 0.0 * unit.kilojoule_per_mole  # epsilon, should always be zero

    assert non_zero_exceptions == 6
    assert zeroed_dummy_exceptions == 6

def test_nonbonded_exception_offsets_no_dummy(htf_chloro_fluoroethane):
    """Test that the nonbonded exception parameter offsets are correctly set when we have no dummy atoms and interpolate between the end state values."""
    forces = htf_chloro_fluoroethane["forces"]
    mapping = htf_chloro_fluoroethane["mapping"]
    chloroethane = htf_chloro_fluoroethane["chloroethane"]
    chloro_openff = chloroethane.to_openff()
    chloro_labels = htf_chloro_fluoroethane["chloro_labels"]
    fluoro_labels = htf_chloro_fluoroethane["fluoro_labels"]
    chloro_charges = htf_chloro_fluoroethane["chloro_charges"]
    fluoro_charges = htf_chloro_fluoroethane["fluoro_charges"]
    electro_scale = htf_chloro_fluoroethane["electrostatic_scale"]
    vdw_scale = htf_chloro_fluoroethane["vdW_scale"]

    nonbonded_force = forces["NonbondedForce"]

    # there should be 56 exception offsets 2 for each of the 28 exceptions to allow for electrostatics and vdw scaling
    num_offsets = nonbonded_force.getNumExceptionParameterOffsets()
    assert num_offsets == 56

    # get the expected exception atoms
    exception_1_4s = []
    # get all atoms with a minimal path of 3 bonds between them
    for pair_1_4 in chloro_openff.nth_degree_neighbors(3):
        a1 = chloro_openff.atoms.index(pair_1_4[0])
        a2 = chloro_openff.atoms.index(pair_1_4[1])
        exception_1_4s.append((a1, a2))

    # Only 12 of the offsets should be non-zero corresponding to the 1-4 interactions
    # 9 from electrostatics
    # 3 from sterics (3 torsions with Cl/F as all hydrogen vdW remain the same)
    non_zero_offsets = 0
    for i in range(num_offsets):
        parameter, exception_index, charge_prod_scale, sigma_scale, epsilon_scale = nonbonded_force.getExceptionParameterOffset(i)
        # get the index of the particles in the exception
        p1, p2, _, _, _ = nonbonded_force.getExceptionParameters(exception_index)
        # if this is not an expected exception it should not be scaled
        if (p1, p2) not in exception_1_4s and (p2, p1) not in exception_1_4s:
            # should be zero
            assert charge_prod_scale == 0.0
            assert sigma_scale == 0.0
            assert epsilon_scale == 0.0
        else:
            # now check the parameters are correct
            if parameter == "lambda_electrostatics_core":
                # track how many we have found
                if charge_prod_scale != 0.0:
                    non_zero_offsets += 1
                # electrostatics offset
                chloro_charge1 = chloro_charges[p1]
                chloro_charge2 = chloro_charges[p2]
                f1 = mapping.componentA_to_componentB[p1]
                f2 = mapping.componentA_to_componentB[p2]
                fluoro_charge1 = fluoro_charges[f1]
                fluoro_charge2 = fluoro_charges[f2]
                # must use the 1-4 scale factor defined in the force field
                expected_scale = ((fluoro_charge1 * fluoro_charge2) - (chloro_charge1 * chloro_charge2)) * electro_scale
                assert expected_scale == pytest.approx(charge_prod_scale, rel=1e-5)
                # sigma and epsilon should be zero
                assert sigma_scale == 0.0
                assert epsilon_scale == 0.0
            elif parameter == "lambda_sterics_core":
                if sigma_scale != 0.0 or epsilon_scale != 0.0:
                    non_zero_offsets += 1
                # sterics offset
                chloro_vdw1 = chloro_labels["vdW"][(p1,)]
                chloro_vdw2 = chloro_labels["vdW"][(p2,)]
                f1 = mapping.componentA_to_componentB[p1]
                f2 = mapping.componentA_to_componentB[p2]
                fluoro_vdw1 = fluoro_labels["vdW"][(f1,)]
                fluoro_vdw2 = fluoro_labels["vdW"][(f2,)]
                # calculate the LJ parameters using Lorentz-Berthelot mixing
                chloro_sigma = (chloro_vdw1.sigma + chloro_vdw2.sigma) / 2.0
                chloro_epsilon = (chloro_vdw1.epsilon * chloro_vdw2.epsilon) ** 0.5
                fluoro_sigma = (fluoro_vdw1.sigma + fluoro_vdw2.sigma) / 2.0
                fluoro_epsilon = (fluoro_vdw1.epsilon * fluoro_vdw2.epsilon) ** 0.5
                # now get the scales
                expected_sigma_scale = fluoro_sigma.m_as(offunit.nanometer) - chloro_sigma.m_as(offunit.nanometer)
                # must use the 1-4 scale factor defined in the force field
                expected_epsilon_scale = (fluoro_epsilon.m_as(offunit.kilojoule_per_mole) - chloro_epsilon.m_as(offunit.kilojoule_per_mole)) * vdw_scale
                assert sigma_scale == expected_sigma_scale
                assert expected_epsilon_scale == pytest.approx(epsilon_scale, rel=1e-5)
    # make sure we found all non-zero offsets
    assert non_zero_offsets == 12

def test_nonbonded_exception_offsets_dummy(htf_chloro_ethane):
    """Test that the nonbonded exception parameter offsets are correctly set when we have dummy atoms.
    All values should be zero if involving a dummy atom."""
    forces = htf_chloro_ethane["forces"]
    chloroethane = htf_chloro_ethane["chloroethane"]
    chloro_labels = htf_chloro_ethane["chloro_labels"]
    chloro_charges = htf_chloro_ethane["chloro_charges"]
    chloro_openff = chloroethane.to_openff()
    ethane_labels = htf_chloro_ethane["ethane_labels"]
    ethane_charges = htf_chloro_ethane["ethane_charges"]
    mapping = htf_chloro_ethane["mapping"]
    electro_scale = htf_chloro_ethane["electrostatic_scale"]
    vdw_scale = htf_chloro_ethane["vdW_scale"]

    nonbonded_force = forces["NonbondedForce"]

    # there are 36 exceptions so 72 offsets (2 lambda per exception)
    # but the htf removes all offsets involving dummy atoms from the nonbonded force
    # so we remove 2 * 7 offsets involving the unique chloro atom in chloroethane
    # and 2 * 7 offsets involving the unique hydrogen atom in ethane
    # and finally we remove 2 * 2 offsets involving the unique chloro and unique hydrogen atoms
    num_offsets = nonbonded_force.getNumExceptionParameterOffsets()
    assert num_offsets == 42

    # get the expected exception atoms
    exception_1_4s = []
    # get all atoms with a minimal path of 3 bonds between them
    # not involving the dummy atoms
    for pair_1_4 in chloro_openff.nth_degree_neighbors(3):
        a1 = chloro_openff.atoms.index(pair_1_4[0])
        a2 = chloro_openff.atoms.index(pair_1_4[1])
        if a1 != 0 and a2 != 0:
            exception_1_4s.append((a1, a2))

    # we expect only 12 offsets to be non-zero corresponding to the 6 mapped 1-4 interactions
    # each with electrostatics and sterics offsets
    non_zero_offsets = 0

    for i in range(num_offsets):
        parameter, exception_index, charge_prod_scale, sigma_scale, epsilon_scale = nonbonded_force.getExceptionParameterOffset(i)
        # get the index of the particles in the exception
        p1, p2, _, _, _ = nonbonded_force.getExceptionParameters(exception_index)
        # if this is not an expected exception it should not be scaled
        if (p1, p2) not in exception_1_4s and (p2, p1) not in exception_1_4s:
            # should be zero
            assert charge_prod_scale == 0.0
            assert sigma_scale == 0.0
            assert epsilon_scale == 0.0
        else:
            # now check the parameters are correct
            if parameter == "lambda_electrostatics_core":
                # track how many we have found
                if charge_prod_scale != 0.0:
                    non_zero_offsets += 1
                # electrostatics offset
                chloro_charge1 = chloro_charges[p1]
                chloro_charge2 = chloro_charges[p2]
                e1 = mapping.componentA_to_componentB[p1]
                e2 = mapping.componentA_to_componentB[p2]
                ethane_charge1 = ethane_charges[e1]
                ethane_charge2 = ethane_charges[e2]
                # must use the 1-4 scale factor defined in the force field
                expected_scale = ((ethane_charge1 * ethane_charge2) - (chloro_charge1 * chloro_charge2)) * electro_scale
                # we see some rounding issues here so use approx
                assert expected_scale == pytest.approx(charge_prod_scale, rel=1e-5)
                # sigma and epsilon should be zero
                assert sigma_scale == 0.0
                assert epsilon_scale == 0.0
            elif parameter == "lambda_sterics_core":
                if sigma_scale != 0.0 or epsilon_scale != 0.0:
                    non_zero_offsets += 1
                # sterics offset
                chloro_vdw1 = chloro_labels["vdW"][(p1,)]
                chloro_vdw2 = chloro_labels["vdW"][(p2,)]
                e1 = mapping.componentA_to_componentB[p1]
                e2 = mapping.componentA_to_componentB[p2]
                ethane_vdw1 = ethane_labels["vdW"][(e1,)]
                ethane_vdw2 = ethane_labels["vdW"][(e2,)]
                # calculate the LJ parameters using Lorentz-Berthelot mixing
                chloro_sigma = (chloro_vdw1.sigma + chloro_vdw2.sigma) / 2.0
                chloro_epsilon = (chloro_vdw1.epsilon * chloro_vdw2.epsilon) ** 0.5
                ethane_sigma = (ethane_vdw1.sigma + ethane_vdw2.sigma) / 2.0
                ethane_epsilon = (ethane_vdw1.epsilon * ethane_vdw2.epsilon) ** 0.5
                # now get the scales
                expected_sigma_scale = ethane_sigma.m_as(offunit.nanometer) - chloro_sigma.m_as(offunit.nanometer)
                # must use the 1-4 scale factor defined in the force field
                expected_epsilon_scale = (ethane_epsilon.m_as(offunit.kilojoule_per_mole) - chloro_epsilon.m_as(offunit.kilojoule_per_mole)) * vdw_scale
                assert sigma_scale == expected_sigma_scale
                assert epsilon_scale == expected_epsilon_scale
    # make sure we found all non-zero offsets
    assert non_zero_offsets == 12

def test_custom_nb_force_no_dummy(htf_chloro_fluoroethane):
    """
    Test the custom nonbonded force is correctly setup when we have no dummy atoms.
    This test implicitly checks that the global parameters are in the correct order.
    """
    forces = htf_chloro_fluoroethane["forces"]
    mapping = htf_chloro_fluoroethane["mapping"]
    chloro_labels = htf_chloro_fluoroethane["chloro_labels"]
    fluoro_labels = htf_chloro_fluoroethane["fluoro_labels"]

    custom_nb_force = forces["CustomNonbondedForce"]
    # there should be 5 global parameters used to scale the particle parameters
    assert custom_nb_force.getNumGlobalParameters() == 5
    expected_global_params = {
        "lambda_sterics_core",
        "lambda_sterics_insert",
        "lambda_sterics_delete",
        "lambda_electrostatics_core",
        "softcore_alpha",
    }
    actual_global_params = {
        custom_nb_force.getGlobalParameterName(i) for i in range(custom_nb_force.getNumGlobalParameters())
    }
    assert actual_global_params == expected_global_params

    # there should be 8 particles as all atoms are mapped
    num_particles = custom_nb_force.getNumParticles()
    assert num_particles == 8
    # now check the parameters are correctly assigned
    for i in range(num_particles):
        params = custom_nb_force.getParticleParameters(i)
        # check the chloroethane parameters
        chloro_vdw = chloro_labels["vdW"][(i,)]
        assert params[0] == chloro_vdw.sigma.m_as(offunit.nanometer)
        assert params[1] == chloro_vdw.epsilon.m_as(offunit.kilojoule_per_mole)
        # map to fluoroethane
        f1 = mapping.componentA_to_componentB[i]
        fluoro_vdw = fluoro_labels["vdW"][(f1,)]
        assert params[2] == fluoro_vdw.sigma.m_as(offunit.nanometer)
        assert params[3] == fluoro_vdw.epsilon.m_as(offunit.kilojoule_per_mole)
        # as we have no unique parameters both unique_old and unque_new should be zero
        assert params[4] == 0.0
        assert params[5] == 0.0

def test_custom_nb_force_dummy(htf_chloro_ethane):
    forces = htf_chloro_ethane["forces"]
    mapping = htf_chloro_ethane["mapping"]
    chloro_labels = htf_chloro_ethane["chloro_labels"]
    ethane_labels = htf_chloro_ethane["ethane_labels"]

    custom_nb_force = forces["CustomNonbondedForce"]
    # there should be 5 global parameters used to scale the particle parameters
    assert custom_nb_force.getNumGlobalParameters() == 5
    expected_global_params = {
        "lambda_sterics_core",
        "lambda_sterics_insert",
        "lambda_sterics_delete",
        "lambda_electrostatics_core",
        "softcore_alpha",
    }
    actual_global_params = {
        custom_nb_force.getGlobalParameterName(i) for i in range(custom_nb_force.getNumGlobalParameters())
    }
    assert actual_global_params == expected_global_params

    # there should be 9 particles as we have 1 unique atom in each end state
    num_particles = custom_nb_force.getNumParticles()
    assert num_particles == 9
    # now check the parameters are correctly assigned
    for i in range(num_particles):
        params = custom_nb_force.getParticleParameters(i)
        if i == 0:
            # unique chloro atom
            chloro_vdw = chloro_labels["vdW"][(i,)]
            # lambda=0 parameters should be chloroethane values
            assert params[0] == chloro_vdw.sigma.m_as(offunit.nanometer)
            assert params[1] == chloro_vdw.epsilon.m_as(offunit.kilojoule_per_mole)
            # lambda=0 parameters should be zeroed dummy atom values
            assert params[2] == chloro_vdw.sigma.m_as(offunit.nanometer)
            assert params[3] == 0.0
            # check we correctly assign the unique_old and unique_new parameters
            assert params[4] == 1  # unique_old should be true
            assert params[5] == 0  # unique_new should be false
        elif i == 8:
            # unique ethane hydrogen atom
            ethane_vdw = ethane_labels["vdW"][(0,)]
            # lambda=0 parameters should be zeroed dummy atom values
            assert params[0] == ethane_vdw.sigma.m_as(offunit.nanometer)
            assert params[1] == 0.0  # epsilon is zero for dummy atoms
            # lambda=1 parameters should be ethane values
            assert params[2] == ethane_vdw.sigma.m_as(offunit.nanometer)
            assert params[3] == ethane_vdw.epsilon.m_as(offunit.kilojoule_per_mole)
            # check we correctly assign the unique_old and unique_new parameters
            assert params[4] == 0  # unique_old should be false
            assert params[5] == 1  # unique_new should be true
        else:
            # mapped atoms should use the chloro at lambda=0 and ethane at lambda=1
            chloro_vdw = chloro_labels["vdW"][(i,)]
            assert params[0] == chloro_vdw.sigma.m_as(offunit.nanometer)
            assert params[1] == chloro_vdw.epsilon.m_as(offunit.kilojoule_per_mole)
            e1 = mapping.componentA_to_componentB[i]
            ethane_vdw = ethane_labels["vdW"][(e1,)]
            assert params[2] == ethane_vdw.sigma.m_as(offunit.nanometer)
            assert params[3] == ethane_vdw.epsilon.m_as(offunit.kilojoule_per_mole)
            # as we have no unique parameters both unique_old and unique_new should be zero
            assert params[4] == 0
            assert params[5] == 0

def test_custom_nb_exclusions_no_dummy(htf_chloro_fluoroethane):
    """
    Test that the expected exclusions are correctly assigned when we have no dummy atoms.
    This should only be the standard exclusions.
    """
    forces = htf_chloro_fluoroethane["forces"]

    custom_nb_force = forces["CustomNonbondedForce"]

    num_exclusions = custom_nb_force.getNumExclusions()
    # there should be 28 in total (8 * 7) / 2
    assert num_exclusions == 28

    # now check the exclusions are correctly assigned
    exclusion_set = set()
    for i in range(num_exclusions):
        p1, p2 = custom_nb_force.getExclusionParticles(i)
        exclusion_set.add((p1, p2))
    # check we have all expected exclusions
    for a1 in range(8):
        for a2 in range(a1 + 1, 8):
            assert (a1, a2) in exclusion_set

def test_custom_nb_exclusions_dummy(htf_chloro_ethane):
    """
    Test that all exclusions are correctly assigned when we have dummy atoms,
    this should involve an exclusion between the two dummy atoms 0-8.
    """
    forces = htf_chloro_ethane["forces"]

    custom_nb_force = forces["CustomNonbondedForce"]

    num_exclusions = custom_nb_force.getNumExclusions()
    # there should be 36 in total (9 * 8) / 2
    assert num_exclusions == 36

    # now check the exclusions are correctly assigned
    exclusion_set = set()
    for i in range(num_exclusions):
        p1, p2 = custom_nb_force.getExclusionParticles(i)
        exclusion_set.add((p1, p2))
        # due to the way we add the exclusions for the unique_new atoms we need to add the reverse too
        exclusion_set.add((p2, p1))
    # check we have all expected exclusions
    for a1 in range(9):
        for a2 in range(a1 + 1, 9):
            assert (a1, a2) in exclusion_set

def test_custom_nb_interation_groups_no_dummy(htf_chloro_fluoroethane):
    """
    Test the interaction groups are correctly assigned when we have no dummy atoms.

    """
    forces = htf_chloro_fluoroethane["forces"]

    custom_nb_force = forces["CustomNonbondedForce"]

    num_groups = custom_nb_force.getNumInteractionGroups()
    # we should always have 8 groups
    assert num_groups == 8

    # now check the groups are correctly assigned
    particle_set = set(range(8))
    expected_groups = [
        (set(), particle_set),  #  unique_old, core
        (set(), set()),  # unique_old, env
        (set(), particle_set),  # unique_new, core
        (set(), set()),  # unique_new, env
        (particle_set, set()),  # core, env
        (particle_set, particle_set),  # core, core
        (set(), set()),  # unique_new, unique_new
        (set(), set()),  # unique_old, unique_old
    ]
    for i in range(num_groups):
        group1, group2 = custom_nb_force.getInteractionGroupParameters(i)
        expected_g1, expected_g2 = expected_groups[i]
        assert set(group1) == expected_g1
        assert set(group2) == expected_g2

def test_custom_nb_interation_groups_dummy(htf_chloro_ethane):
    """
    Make sure the interaction groups are correctly assigned when we have dummy atoms.
    The unique_old and unique_new atoms should be split to not interact.
    """
    forces = htf_chloro_ethane["forces"]

    custom_nb_force = forces["CustomNonbondedForce"]

    num_groups = custom_nb_force.getNumInteractionGroups()
    # we should always have 8 groups
    assert num_groups == 8

    # now check the groups are correctly assigned
    core_set = set(range(1, 8))
    unique_old_set = {0}
    unique_new_set = {8}
    expected_groups = [
        (unique_old_set, core_set),  #  unique_old, core
        (unique_old_set, set()),  # unique_old, env
        (unique_new_set, core_set),  # unique_new, core
        (unique_new_set, set()),  # unique_new, env
        (core_set, set()),  # core, env
        (core_set, core_set),  # core, core
        (unique_new_set, unique_new_set),  # unique_new, unique_new
        (unique_old_set, unique_old_set),  # unique_old, unique_old
    ]
    for i in range(num_groups):
        group1, group2 = custom_nb_force.getInteractionGroupParameters(i)
        expected_g1, expected_g2 = expected_groups[i]
        assert set(group1) == expected_g1
        assert set(group2) == expected_g2

def test_custom_sterics_force_no_dummy(htf_chloro_fluoroethane):
    """
    Make sure there are no bonds in the custom sterics force when we have no dummy atoms.
    """
    forces = htf_chloro_fluoroethane["forces"]
    custom_sterics_force = forces["CustomBondForce_exceptions"]
    # there should be 5 global parameters used to scale the particle parameters
    assert custom_sterics_force.getNumGlobalParameters() == 5
    expected_global_params = {
        "lambda_sterics_insert",
        "lambda_sterics_delete",
        "lambda_electrostatics_insert",
        "lambda_electrostatics_delete",
        "softcore_alpha",
    }
    actual_global_params = {
        custom_sterics_force.getGlobalParameterName(i) for i in range(custom_sterics_force.getNumGlobalParameters())
    }
    assert actual_global_params == expected_global_params

    # we should have no parameters in this force as there are no dummy atoms
    assert custom_sterics_force.getNumBonds() == 0

def test_custom_sterics_force_dummy(htf_chloro_ethane):
    """
    The custom sterics force should contain only 1-4 interactions involving dummy atoms make sure they are correctly interpolated.
    """
    forces = htf_chloro_ethane["forces"]
    mapping = htf_chloro_ethane["mapping"]
    chloroethane = htf_chloro_ethane["chloroethane"]
    chloro_charges = htf_chloro_ethane["chloro_charges"]
    chloro_labels = htf_chloro_ethane["chloro_labels"]
    ethane_charges = htf_chloro_ethane["ethane_charges"]
    ethane_labels = htf_chloro_ethane["ethane_labels"]
    chloro_openff = chloroethane.to_openff()
    electro_scale = htf_chloro_ethane["electrostatic_scale"]
    vdw_scale = htf_chloro_ethane["vdW_scale"]

    custom_sterics_force = forces["CustomBondForce_exceptions"]
    # there should be 5 global parameters used to scale the particle parameters
    assert custom_sterics_force.getNumGlobalParameters() == 5
    expected_global_params = {
        "lambda_sterics_insert",
        "lambda_sterics_delete",
        "lambda_electrostatics_insert",
        "lambda_electrostatics_delete",
        "softcore_alpha",
    }
    actual_global_params = {
        custom_sterics_force.getGlobalParameterName(i) for i in range(custom_sterics_force.getNumGlobalParameters())
    }
    assert actual_global_params == expected_global_params

    num_bonds = custom_sterics_force.getNumBonds()
    # there should be 6 bonds corresponding to the 1-4s involving a dummy atom
    # 3 involving the unique chloro atom to be interpolated out
    # 3 involving the unique ethane hydrogen atom to be interpolated in
    assert num_bonds == 6

    # now check the bonds are correctly assigned
    expected_bonds = []
    # get all atoms with a minimal path of 3 bonds between them
    for pair_1_4 in chloro_openff.nth_degree_neighbors(3):
        a1 = chloro_openff.atoms.index(pair_1_4[0])
        a2 = chloro_openff.atoms.index(pair_1_4[1])
        if a1 == 0 or a2 == 0:
            expected_bonds.append((a1, a2))
    # manually add the 1-4s involving the unique ethane hydrogen atom (atom 8)
    for i in [5, 6, 7]:
        expected_bonds.append((8, i))

    for i in range(num_bonds):
        p1, p2, params = custom_sterics_force.getBondParameters(i)
        # all parameters here should be 1-4 involving a dummy atom only
        assert (p1, p2) in expected_bonds or (p2, p1) in expected_bonds
        # now check the parameters are correct
        if p1 == 0 or p2 == 0:
            # unique chloro atom being removed
            charge1 = chloro_charges[p1]
            charge2 = chloro_charges[p2]
            # check the charge product at lambda=0
            assert charge1 * charge2 * electro_scale == pytest.approx(params[0], rel=1e-5)
            # this is scaled by the unique flags make sure unique_old is 1 and unique_new is 0
            assert params[5] == 1  # unique_old
            assert params[6] == 0  # unique_new
            # now check the vdw parameters at lambda=0
            chloro_vdw1 = chloro_labels["vdW"][(p1,)]
            chloro_vdw2 = chloro_labels["vdW"][(p2,)]
            expected_sigma = (chloro_vdw1.sigma + chloro_vdw2.sigma) / 2.0
            expected_epsilon = ((chloro_vdw1.epsilon * chloro_vdw2.epsilon) ** 0.5) * vdw_scale
            assert params[1] == expected_sigma.m_as(offunit.nanometer)
            assert expected_epsilon.m_as(offunit.kilojoule_per_mole) == pytest.approx(params[2], rel=1e-5)
            # check the vdw go to zero at lambda=1
            assert params[4] == 0.0  # epsilon at lambda=1
            # sigma doesn't matter if epsilon is zero
        elif p1 == 8 or p2 == 8:
            # unique ethane hydrogen atom being inserted
            e1 = 0 if p1 == 8 else mapping.componentA_to_componentB[p1]
            e2 = 0 if p2 == 8 else mapping.componentA_to_componentB[p2]
            ethane_charge1 = ethane_charges[e1]
            ethane_charge2 = ethane_charges[e2]
            # check the charge product at lambda=1
            assert ethane_charge1 * ethane_charge2 * electro_scale == pytest.approx(params[0], rel=1e-5)
            # this is scaled by the unique flags make sure unique_old is 0 and unique_new is 1
            assert params[5] == 0  # unique_old
            assert params[6] == 1  # unique_new
            # now check the vdw parameters at lambda=1
            ethane_vdw1 = ethane_labels["vdW"][(e1,)]
            ethane_vdw2 = ethane_labels["vdW"][(e2,)]
            expected_sigma = (ethane_vdw1.sigma + ethane_vdw2.sigma) / 2.0
            expected_epsilon = ((ethane_vdw1.epsilon * ethane_vdw2.epsilon) ** 0.5) * vdw_scale
            assert params[3] == expected_sigma.m_as(offunit.nanometer)
            assert  expected_epsilon.m_as(offunit.kilojoule_per_mole) == pytest.approx(params[4], rel=1e-5)
            # check the vdw are zero at lambda=0
            assert params[2] == 0.0  # epsilon at lambda=0

def test_vacuum_system_energy_no_dummy(htf_chloro_fluoroethane):
    """
    Test that the hybrid system energy is the same at lambda=0 and lambda=1 as the pure systems.
    All individual force components should match as there are no dummy atoms.
    """
    integrator = openmm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.002*unit.picoseconds)
    platform = openmm.Platform.getPlatformByName("CPU")
    default_lambda = _rfe_utils.lambdaprotocol.LambdaProtocol()
    htf = htf_chloro_fluoroethane["htf"]
    hybrid_system = htf.hybrid_system
    # set the nonbonded method to NoCutoff to avoid any cutoff issues
    for force in hybrid_system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
        if isinstance(force, openmm.CustomNonbondedForce):
            force.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
    hybrid_simulation = app.Simulation(
        topology=htf.omm_hybrid_topology,
        system=hybrid_system,
        integrator=integrator,
        platform=platform
    )
    for end_state, ref_system, ref_top, pos in [
        (0, htf._old_system, htf._old_topology, htf._old_positions),
        (1, htf._new_system, htf._new_topology, htf._new_positions)
    ]:
        for force in ref_system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
        # set lambda
        # set all lambda values to the current end state
        for name, func in default_lambda.functions.items():
            val = func(end_state)
            hybrid_simulation.context.setParameter(name, val)
        # set positions
        hybrid_simulation.context.setPositions(pos)
        # get the hybrid system energy
        hybrid_state = hybrid_simulation.context.getState(getEnergy=True)
        hybrid_energy = hybrid_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        # now create a reference simulation
        ref_simulation = app.Simulation(
            topology=ref_top,
            system=ref_system,
            integrator=copy.deepcopy(integrator),
            platform=platform
        )
        ref_simulation.context.setPositions(pos)
        ref_state = ref_simulation.context.getState(getEnergy=True)
        ref_energy = ref_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        # energies should be the same
        assert ref_energy == pytest.approx(hybrid_energy, rel=1e-5)
        # check the energy is non-zero to avoid false positives
        assert 0.0 != pytest.approx(hybrid_energy)


def test_vacuum_system_energy_dummy(htf_chloro_ethane):
    """
    Test that the hybrid system nonbonded energy is the same at lambda=0 and lambda=1 as the pure systems.
    All individual force components will not match due to the presence of dummy atoms, only the total nonbonded energy should be the same.
    """
    integrator = openmm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.002*unit.picoseconds)
    platform = openmm.Platform.getPlatformByName("CPU")
    default_lambda = _rfe_utils.lambdaprotocol.LambdaProtocol()
    htf = htf_chloro_ethane["htf"]
    hybrid_system = htf.hybrid_system
    # # set the nonbonded method to NoCutoff to avoid any cutoff issues
    for force in hybrid_system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
        if isinstance(force, openmm.CustomNonbondedForce):
            force.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)

    # set the nonbonded forces to group 1 to easily extract their energies and all others to 0
    for force in hybrid_system.getForces():
        if force.getName() in ["NonbondedForce", "CustomNonbondedForce", "CustomBondForce_exceptions"]:
            force.setForceGroup(1)
        else:
            force.setForceGroup(0)

    hybrid_simulation = app.Simulation(
        topology=htf.omm_hybrid_topology,
        system=hybrid_system,
        integrator=integrator,
        platform=platform
    )

    for end_state, ref_system, ref_top, pos in [
        (0, htf._old_system, htf._old_topology, htf._old_positions),
        (1, htf._new_system, htf._new_topology, htf._new_positions)
    ]:
        # set lambda
        # set all lambda values to the current end state
        for name, func in default_lambda.functions.items():
            val = func(end_state)
            hybrid_simulation.context.setParameter(name, val)
        # set positions as the hybrid positions, as all mapped atoms are in the same place in both end states
        hybrid_simulation.context.setPositions(htf.hybrid_positions)
        # get the hybrid system nonbonded energy
        hybrid_state = hybrid_simulation.context.getState(getEnergy=True, groups={1})
        hybrid_energy = hybrid_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        # set the force group for the reference system nonbonded forces
        for force in ref_system.getForces():
            if force.getName() == "NonbondedForce":
                force.setForceGroup(1)
                force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
            else:
                force.setForceGroup(0)
        # now create a reference simulation
        ref_simulation = app.Simulation(
            topology=ref_top,
            system=ref_system,
            integrator=copy.deepcopy(integrator),
            platform=platform
        )
        ref_simulation.context.setPositions(pos)
        ref_state = ref_simulation.context.getState(getEnergy=True, groups={1})
        ref_energy = ref_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        # energies should be the same
        # this is only true if we correctly interpolate the 1-4 interactions involving dummy atoms
        assert ref_energy == pytest.approx(hybrid_energy, rel=1e-5)
        # check the energy is non-zero to avoid false positives
        assert 0.0 != pytest.approx(hybrid_energy)


def test_system_energy_pme_no_dummy(htf_chlorobenzene_fluorobenzene):
    """
    Test that the hybrid system energy is the same at lambda=0 and lambda=1 as the pure systems using PME,
    for a fully mapped system.
    """
    integrator = openmm.LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
    platform = openmm.Platform.getPlatformByName("CPU")
    default_lambda = _rfe_utils.lambdaprotocol.LambdaProtocol()
    htf = htf_chlorobenzene_fluorobenzene["htf"]
    hybrid_system = htf.hybrid_system

    hybrid_simulation = app.Simulation(
        topology=htf.omm_hybrid_topology,
        system=hybrid_system,
        integrator=integrator,
        platform=platform
    )
    for end_state, ref_system, ref_top, pos in [
        (0, htf._old_system, htf._old_topology, htf._old_positions),
        (1, htf._new_system, htf._new_topology, htf._new_positions)
    ]:
        # set lambda
        # set all lambda values to the current end state
        for name, func in default_lambda.functions.items():
            val = func(end_state)
            hybrid_simulation.context.setParameter(name, val)
        # set positions
        hybrid_simulation.context.setPositions(pos)
        # get the hybrid system energy
        hybrid_state = hybrid_simulation.context.getState(getEnergy=True)
        hybrid_energy = hybrid_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        # now create a reference simulation
        ref_simulation = app.Simulation(
            topology=ref_top,
            system=ref_system,
            integrator=copy.deepcopy(integrator),
            platform=platform
        )
        ref_simulation.context.setPositions(pos)
        ref_state = ref_simulation.context.getState(getEnergy=True)
        ref_energy = ref_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        # energies should be the same
        assert ref_energy == pytest.approx(hybrid_energy, rel=1e-5)
        # make sure the energy is non-zero to avoid false positives
        assert 0.0 != pytest.approx(hybrid_energy)

def test_system_interaction_groups_no_dummy(htf_chlorobenzene_fluorobenzene):
    """Test the interaction groups are correctly assigned when we have environment atoms as well."""
    forces = htf_chlorobenzene_fluorobenzene["forces"]
    htf = htf_chlorobenzene_fluorobenzene["htf"]
    custom_nb_force = forces["CustomNonbondedForce"]
    num_groups = custom_nb_force.getNumInteractionGroups()
    # we should always have 8 groups
    assert num_groups == 8

    # now check the groups are correctly assigned
    # we assume the ligand topology is added to the end of the solvated protein topology which ends on index 21666
    core_set = set(range(21667, 21679))
    unique_old_set = set()
    unique_new_set = set()
    environment_set = set(range(0, 21667))
    expected_groups = [
        (unique_old_set, core_set),  # unique_old, core
        (unique_old_set, environment_set),  # unique_old, env
        (unique_new_set, core_set),  # unique_new, core
        (unique_new_set, environment_set),  # unique_new, env
        (core_set, environment_set),  # core, env
        (core_set, core_set),  # core, core
        (unique_new_set, unique_new_set),  # unique_new, unique_new
        (unique_old_set, unique_old_set),  # unique_old, unique_old
    ]
    for i in range(num_groups):
        group1, group2 = custom_nb_force.getInteractionGroupParameters(i)
        expected_g1, expected_g2 = expected_groups[i]
        assert set(group1) == expected_g1
        assert set(group2) == expected_g2

def test_system_interaction_groups_dummy(htf_chlorobenzene_benzene):
    """Test the interaction groups are correctly assigned when we have environment atoms and dummy atoms."""
    forces = htf_chlorobenzene_benzene["forces"]
    htf = htf_chlorobenzene_benzene["htf"]
    custom_nb_force = forces["CustomNonbondedForce"]
    num_groups = custom_nb_force.getNumInteractionGroups()
    # we should always have 8 groups
    assert num_groups == 8

    # now check the groups are correctly assigned
    # we assume the ligand topology is added to the end of the solvated protein topology which ends on index 21666
    core_set = set(range(21668, 21679))
    unique_old_set = {21667}
    unique_new_set = {21679}
    environment_set = set(range(0, 21667))
    expected_groups = [
        (unique_old_set, core_set),  # unique_old, core
        (unique_old_set, environment_set),  # unique_old, env
        (unique_new_set, core_set),  # unique_new, core
        (unique_new_set, environment_set),  # unique_new, env
        (core_set, environment_set),  # core, env
        (core_set, core_set),  # core, core
        (unique_new_set, unique_new_set),  # unique_new, unique_new
        (unique_old_set, unique_old_set),  # unique_old, unique_old
    ]
    for i in range(num_groups):
        group1, group2 = custom_nb_force.getInteractionGroupParameters(i)
        expected_g1, expected_g2 = expected_groups[i]
        assert set(group1) == expected_g1
        assert set(group2) == expected_g2

def test_system_energy_pme_dummy(htf_chlorobenzene_benzene):
    """
    Test that the hybrid system nonbonded energy is the same at lambda=0 and lambda=1 as the pure systems using PME,
    for a system with dummy atoms.
    """
    integrator = openmm.LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
    platform = openmm.Platform.getPlatformByName("CPU")
    default_lambda = _rfe_utils.lambdaprotocol.LambdaProtocol()
    htf = htf_chlorobenzene_benzene["htf"]
    hybrid_system = htf.hybrid_system

    # set the nonbonded forces to group 1 to easily extract their energies and all others to 0
    for force in hybrid_system.getForces():
        if force.getName() in ["NonbondedForce", "CustomNonbondedForce", "CustomBondForce_exceptions"]:
            force.setForceGroup(1)
        else:
            force.setForceGroup(0)

    hybrid_simulation = app.Simulation(
        topology=htf.omm_hybrid_topology,
        system=hybrid_system,
        integrator=integrator,
        platform=platform
    )

    for end_state, ref_system, ref_top, pos in [
        (0, htf._old_system, htf._old_topology, htf._old_positions),
        (1, htf._new_system, htf._new_topology, htf._new_positions)
    ]:
        # set lambda
        # set all lambda values to the current end state
        for name, func in default_lambda.functions.items():
            val = func(end_state)
            hybrid_simulation.context.setParameter(name, val)
        # set positions as the hybrid positions, as all mapped atoms are in the same place in both end states
        hybrid_simulation.context.setPositions(htf.hybrid_positions)
        # get the hybrid system nonbonded energy
        hybrid_state = hybrid_simulation.context.getState(getEnergy=True, groups={1})
        hybrid_energy = hybrid_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        # set the force group for the reference system nonbonded forces
        for force in ref_system.getForces():
            if force.getName() == "NonbondedForce":
                force.setForceGroup(1)
            else:
                force.setForceGroup(0)
        # now create a reference simulation
        ref_simulation = app.Simulation(
            topology=ref_top,
            system=ref_system,
            integrator=copy.deepcopy(integrator),
            platform=platform
        )
        ref_simulation.context.setPositions(pos)
        ref_state = ref_simulation.context.getState(getEnergy=True, groups={1})
        ref_energy = ref_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        # energies should be the same
        # this is only true if we correctly interpolate the 1-4 interactions involving dummy atoms
        # make sure the energy is non-zero to avoid false positives
        assert  0.0 != pytest.approx(hybrid_energy)
        assert hybrid_energy == pytest.approx(ref_energy, rel=1e-5)

def test_cmap_system_no_dummy_pme_energy(chlorobenzene_to_fluorobenzene_mapping, t4_lysozyme_solvated):
    """
    Test that we can make a hybrid topology for a system with conserved CMAP terms not in the alchemical region and that
    the hybrid energy matches the end state energy.
    """
    settings = RelativeHybridTopologyProtocol.default_settings()
    # update the default force fields to include a force field with CMAP terms
    settings.forcefield_settings.forcefields = [
        "amber/protein.ff19SB.xml",  # cmap amber ff
        "amber/tip3p_standard.xml",  # TIP3P and recommended monovalent ion parameters
        "amber/tip3p_HFE_multivalent.xml",  # for divalent ions
        "amber/phosaa19SB.xml",  # Handles THE TPO
    ]
    htf = make_htf(
        mapping=chlorobenzene_to_fluorobenzene_mapping,
        settings=settings,
        protein=t4_lysozyme_solvated
    )
    # make sure the cmap force was added to the internal store
    assert "cmap_torsion_force" in htf._hybrid_system_forces
    hybrid_system = htf.hybrid_system
    # make sure we can find the force in the system
    forces = {force.getName(): force for force in hybrid_system.getForces()}
    assert isinstance(forces["CMAPTorsionForce"], openmm.CMAPTorsionForce)

    integrator = openmm.LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
    platform = openmm.Platform.getPlatformByName("CPU")
    default_lambda = _rfe_utils.lambdaprotocol.LambdaProtocol()

    hybrid_simulation = app.Simulation(
        topology=htf.omm_hybrid_topology,
        system=hybrid_system,
        integrator=integrator,
        platform=platform
    )
    for end_state, ref_system, ref_top, pos in [
        (0, htf._old_system, htf._old_topology, htf._old_positions),
        (1, htf._new_system, htf._new_topology, htf._new_positions)
    ]:
        # set lambda
        # set all lambda values to the current end state
        for name, func in default_lambda.functions.items():
            val = func(end_state)
            hybrid_simulation.context.setParameter(name, val)
        # set positions
        hybrid_simulation.context.setPositions(pos)
        # get the hybrid system energy
        hybrid_state = hybrid_simulation.context.getState(getEnergy=True)
        hybrid_energy = hybrid_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        # now create a reference simulation
        ref_simulation = app.Simulation(
            topology=ref_top,
            system=ref_system,
            integrator=copy.deepcopy(integrator),
            platform=platform
        )
        ref_simulation.context.setPositions(pos)
        ref_state = ref_simulation.context.getState(getEnergy=True)
        ref_energy = ref_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        # energies should be the same
        assert ref_energy == pytest.approx(hybrid_energy, rel=1e-5)
        # make sure the energy is non-zero to avoid false positives
        assert 0.0 != pytest.approx(hybrid_energy)


def test_verify_cmap_no_cmap():
    """Test that no error is raised if a CMAPTorsionForce is not present in either end state."""
    (
        cmap_old,
        cmap_new,
        old_num_maps,
        new_num_maps,
        old_num_torsions,
        new_num_torsions
    ) = HybridTopologyFactory._verify_cmap_compatibility(
        None, None
    )
    assert cmap_old is None
    assert cmap_new is None
    assert old_num_maps == 0
    assert new_num_maps == 0
    assert old_num_torsions == 0
    assert new_num_torsions == 0

def test_verify_cmap_missing_cmap_error():
    """Test that an error is raised if a CMAPTorsionForce is only present in one of the end states."""
    with pytest.raises(RuntimeError, match="Inconsistent CMAPTorsionForce between end states expected to be present in both"):
        _ = HybridTopologyFactory._verify_cmap_compatibility(
            None, openmm.CMAPTorsionForce()
        )

def test_verify_cmap_incompatible_maps_error():
    """Test that an error is raised if the number of CMAP terms differ between the end states."""
    old_cmap = openmm.CMAPTorsionForce()
    new_cmap = openmm.CMAPTorsionForce()
    old_cmap.addMap(2, [0.0] * 2 * 2)  # add one map
    new_cmap.addMap(2, [0.0] * 2 * 2)  # add one map
    new_cmap.addMap(2, [0.0] * 2 * 2)  # add a second map to make them incompatible
    with pytest.raises(RuntimeError, match="Incompatible CMAPTorsionForce between end states expected to have same number of maps, found old: 1 and new: 2"):
        _ = HybridTopologyFactory._verify_cmap_compatibility(
            old_cmap, new_cmap
        )

def test_verify_cmap_incompatible_torsions_error():
    """Test that an error is raised if the number of CMAP torsions differ between the end states."""
    old_cmap = openmm.CMAPTorsionForce()
    new_cmap = openmm.CMAPTorsionForce()
    old_cmap.addMap(2, [0.0] * 2 * 2)  # add one map
    new_cmap.addMap(2, [0.0] * 2 * 2)  # add one map
    # add torsions
    old_cmap.addTorsion(0, 0, 1, 2, 3, 4, 5, 6, 7)
    new_cmap.addTorsion(0, 0, 1, 2, 3, 4, 5, 6, 7)
    new_cmap.addTorsion(0, 1, 2, 3, 4, 5, 6, 7, 8)  # add a second torsion to make them incompatible
    with pytest.raises(RuntimeError, match="Incompatible CMAPTorsionForce between end states expected to have same number of torsions, found old: 1 and new: 2"):
        _ = HybridTopologyFactory._verify_cmap_compatibility(
            old_cmap, new_cmap
        )

def test_cmap_maps_incompatible_error():
    """Test that an error is raised if the CMAP maps differ between the end states using a dummy system.
    In this case the map parameters differ for map index 0 between the old and new systems
    """
    old_system, old_topology, old_positions = _make_system_with_cmap([4])
    new_system, new_topology, new_positions = _make_system_with_cmap([3])
    with pytest.raises(RuntimeError, match="Incompatible CMAPTorsionForce map parameters found between end states for map 0 expected"):
        _ = HybridTopologyFactory(
            old_system=old_system,
            old_topology=old_topology,
            old_positions=old_positions,
            new_system=new_system,
            new_topology=new_topology,
            new_positions=new_positions,
            # map all atoms so they end up in the environment
            old_to_new_atom_map={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7},
            old_to_new_core_atom_map={}
        )

def test_cmap_torsions_incompatible_error():
    """Test that an error is raised if the CMAP torsions differ between the end states using a dummy system.
    In this case there is an extra cmap torsion in the new system not present in the old system."""
    old_system, old_topology, old_positions = _make_system_with_cmap([4], num_atoms=12)
    new_system, new_topology, new_positions = _make_system_with_cmap([4], num_atoms=12, mapped_torsions=[
        # change the mapped atoms from the default
        (0, 4, 5, 6, 7, 8, 9, 10, 11)
    ])
    with pytest.raises(RuntimeError, match="Incompatible CMAPTorsionForce term found between end states for atoms "):
        _ = HybridTopologyFactory(
            old_system=old_system,
            old_topology=old_topology,
            old_positions=old_positions,
            new_system=new_system,
            new_topology=new_topology,
            new_positions=new_positions,
            # map all atoms so they end up in the environment
            old_to_new_atom_map={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11},
            old_to_new_core_atom_map={}
        )

def test_cmap_map_index_incompatible_error():
    """Test that an error is raised if the CMAP map indices differ between the end states using a dummy system.
    In this case the map index for the single cmap torsion differs between the old and new systems."""
    old_system, old_topology, old_positions = _make_system_with_cmap([4, 5])
    new_system, new_topology, new_positions = _make_system_with_cmap([4, 5], mapped_torsions=[
        # change the map index from the default
        (1, 0, 1, 2, 3, 4, 5, 6, 7)
    ])
    # modify one of the torsions in the new system to make them incompatible
    with pytest.raises(RuntimeError, match="Incompatible CMAPTorsionForce map index found between end states for atoms "):
        _ = HybridTopologyFactory(
            old_system=old_system,
            old_topology=old_topology,
            old_positions=old_positions,
            new_system=new_system,
            new_topology=new_topology,
            new_positions=new_positions,
            # map all atoms so they end up in the environment
            old_to_new_atom_map={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7},
            old_to_new_core_atom_map={}
        )

def test_cmap_in_alchemical_region_error():
    """Test that an error is raised if a CMAP torsion is in the alchemical region."""
    old_system, old_topology, old_positions = _make_system_with_cmap([4])
    new_system, new_topology, new_positions = _make_system_with_cmap([4])
    with pytest.raises(RuntimeError, match="Incompatible CMAPTorsionForce term found in alchemical region for old system atoms"):
        _ = HybridTopologyFactory(
            old_system=old_system,
            old_topology=old_topology,
            old_positions=old_positions,
            new_system=new_system,
            new_topology=new_topology,
            new_positions=new_positions,
            # map all atoms so that one of the cmap atoms is in the alchemical core region
            old_to_new_atom_map={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7},
            old_to_new_core_atom_map={4: 4}  # atom 4 is part of the cmap torsion
        )