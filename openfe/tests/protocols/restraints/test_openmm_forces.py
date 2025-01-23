# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import pytest
import numpy as np
import openmm
from openfe.protocols.restraint_utils.openmm.omm_forces import (
    get_boresch_energy_function,
    get_periodic_boresch_energy_function,
    get_custom_compound_bond_force,
    add_force_in_separate_group,
)


@pytest.mark.parametrize('param', ['foo', 'bar'])
def test_boresch_energy_function(param):
    """
    Base regression test for the energy function
    """
    fn = get_boresch_energy_function(param)
    assert fn == (
        f"{param} * E; "
        "E = (K_r/2)*(distance(p3,p4) - r_aA0)^2 "
        "+ (K_thetaA/2)*(angle(p2,p3,p4)-theta_A0)^2 + (K_thetaB/2)*(angle(p3,p4,p5)-theta_B0)^2 "
        "+ (K_phiA/2)*dphi_A^2 + (K_phiB/2)*dphi_B^2 + (K_phiC/2)*dphi_C^2; "
        "dphi_A = dA - floor(dA/(2.0*pi)+0.5)*(2.0*pi); dA = dihedral(p1,p2,p3,p4) - phi_A0; "
        "dphi_B = dB - floor(dB/(2.0*pi)+0.5)*(2.0*pi); dB = dihedral(p2,p3,p4,p5) - phi_B0; "
        "dphi_C = dC - floor(dC/(2.0*pi)+0.5)*(2.0*pi); dC = dihedral(p3,p4,p5,p6) - phi_C0; "
        f"pi = {np.pi}; "
    )


@pytest.mark.parametrize('param', ['foo', 'bar'])
def test_periodic_boresch_energy_function(param):
    """
    Base regression test for the energy function
    """
    fn = get_periodic_boresch_energy_function(param)
    assert fn == (
        f"{param} * E; "
        "E = (K_r/2)*(distance(p3,p4) - r_aA0)^2 "
        "+ (K_thetaA/2)*(angle(p2,p3,p4)-theta_A0)^2 + (K_thetaB/2)*(angle(p3,p4,p5)-theta_B0)^2 "
        "+ (K_phiA/2)*uphi_A + (K_phiB/2)*uphi_B + (K_phiC/2)*uphi_C; "
        "uphi_A = (1-cos(dA)); dA = dihedral(p1,p2,p3,p4) - phi_A0; "
        "uphi_B = (1-cos(dB)); dB = dihedral(p2,p3,p4,p5) - phi_B0; "
        "uphi_C = (1-cos(dC)); dC = dihedral(p3,p4,p5,p6) - phi_C0; "
        f"pi = {np.pi}; "
    )


@pytest.mark.parametrize('num_atoms', [6, 20])
def test_custom_compound_force(num_atoms):
    fn = get_boresch_energy_function('lambda_restraints')
    force = get_custom_compound_bond_force(fn, num_atoms)

    # Check we have the right object
    assert isinstance(force, openmm.CustomCompoundBondForce)

    # Check the energy function
    assert force.getEnergyFunction() == fn

    # Check the number of particles
    assert force.getNumParticlesPerBond() == num_atoms


@pytest.mark.parametrize('groups, expected', [
    [[0, 1, 2, 3, 4], 5],
    [[1, 2, 3, 4, 5], 0],
])
def test_add_force_in_separate_group(groups, expected):
    # Create an empty system
    system = openmm.System()

    # Create some forces with some force groups
    base_forces = [
        openmm.NonbondedForce(),
        openmm.HarmonicBondForce(),
        openmm.HarmonicAngleForce(),
        openmm.PeriodicTorsionForce(),
        openmm.CMMotionRemover(),
    ]

    for force, group in zip(base_forces, groups):
        force.setForceGroup(group)

    [system.addForce(force) for force in base_forces]

    # Get your CustomCompoundBondForce
    fn = get_boresch_energy_function('lambda_restraints')
    new_force = get_custom_compound_bond_force(fn, 6)
    # new_force.setForceGroup(5)
    # system.addForce(new_force)
    add_force_in_separate_group(system=system, force=new_force)

    # Loop through and check that we go assigned the expected force group
    for force in system.getForces():
        if isinstance(force, openmm.CustomCompoundBondForce):
            assert force.getForceGroup() == expected


def test_add_too_many_force_groups():
    # Create a system
    system = openmm.System()

    # Fill it upu with 32 forces with different groups
    for i in range(32):
        f = openmm.HarmonicBondForce()
        f.setForceGroup(i)
        system.addForce(f)

    # Now try to add another force
    with pytest.raises(ValueError, match="No available force group"):
        add_force_in_separate_group(
            system=system, force=openmm.HarmonicBondForce()
        )