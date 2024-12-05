# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Custom OpenMM Forces

TODO
----
* Add relevant duecredit entries.
"""
import numpy as np
import openmm


def get_boresch_energy_function(
    control_parameter: str,
    K_r: float, r_aA0: float,
    K_thetaA: float, theta_A0: float,
    K_thetaB: float, theta_B0: float,
    K_phiA: float, phi_A0: float,
    K_phiB: float, phi_B0: float,
    K_phiC: float, phi_C0: float
) -> str:
    energy_function = (
        f"{control_parameter} * E; "
        "E = (K_r/2)*(distance(p3,p4) - r_aA0)^2 "
        "+ (K_thetaA/2)*(angle(p2,p3,p4)-theta_A0)^2 + (K_thetaB/2)*(angle(p3,p4,p5)-theta_B0)^2 "
        "+ (K_phiA/2)*dphi_A^2 + (K_phiB/2)*dphi_B^2 + (K_phiC/2)*dphi_C^2; "
        "dphi_A = dA - floor(dA/(2*pi)+0.5)*(2*pi); dA = dihedral(p1,p2,p3,p4) - phi_A0; "
        "dphi_B = dB - floor(dB/(2*pi)+0.5)*(2*pi); dB = dihedral(p2,p3,p4,p5) - phi_B0; "
        "dphi_C = dC - floor(dC/(2*pi)+0.5)*(2*pi); dC = dihedral(p3,p4,p5,p6) - phi_C0; "
        f"pi = {np.pi}; "
        f"K_r = {K_r}; "
        f"r_aA0 = {r_aA0}; "
        f"K_thetaA = {K_thetaA}; "
        f"theta_A0 = {theta_A0}; "
        f"K_thetaB = {K_thetaB}; "
        f"theta_B0 = {theta_B0}; "
        f"K_phiA = {K_phiA}; "
        f"phi_A0 = {phi_A0}; "
        f"K_phiB = {K_phiB}; "
        f"phi_B0 = {phi_B0}; "
        f"K_phiC = {K_phiC}; "
        f"phi_C0 = {phi_C0}; "
    )
    return energy_function


def get_periodic_boresch_energy_function(
    control_parameter: str,
    K_r: float, r_aA0: float,
    K_thetaA: float, theta_A0: float,
    K_thetaB: float, theta_B0: float,
    K_phiA: float, phi_A0: float,
    K_phiB: float, phi_B0: float,
    K_phiC: float, phi_C0: float
) -> str:
    energy_function = (
        f"{control_parameter} * E; "
        "E = (K_r/2)*(distance(p3,p4) - r_aA0)^2 "
        "+ (K_thetaA/2)*(angle(p2,p3,p4)-theta_A0)^2 + (K_thetaB/2)*(angle(p3,p4,p5)-theta_B0)^2 "
        "+ (K_phiA/2)*uphi_A + (K_phiB/2)*uphi_B + (K_phiC/2)*uphi_C; "
        "uphi_A = (1-cos(dA)); dA = dihedral(p1,p2,p3,p4) - phi_A0; "
        "uphi_B = (1-cos(dB)); dB = dihedral(p2,p3,p4,p5) - phi_B0; "
        "uphi_C = (1-cos(dC)); dC = dihedral(p3,p4,p5,p6) - phi_C0; "
        f"pi = {np.pi}; "
        f"K_r = {K_r}; "
        f"r_aA0 = {r_aA0}; "
        f"K_thetaA = {K_thetaA}; "
        f"theta_A0 = {theta_A0}; "
        f"K_thetaB = {K_thetaB}; "
        f"theta_B0 = {theta_B0}; "
        f"K_phiA = {K_phiA}; "
        f"phi_A0 = {phi_A0}; "
        f"K_phiB = {K_phiB}; "
        f"phi_B0 = {phi_B0}; "
        f"K_phiC = {K_phiC}; "
        f"phi_C0 = {phi_C0}; "
    )
    return energy_function


def get_custom_compound_bond_force(
    n_particles: int = 6, energy_function: str = BORESCH_ENERGY_FUNCTION
):
    """
    Return an OpenMM CustomCompoundForce

    TODO
    ----
    Change this to a direct subclass like openmmtools.force.

    Acknowledgements
    ----------------
    Boresch-like energy functions are reproduced from `Yank <https://github.com/choderalab/yank>`_
    """
    return openmm.CustomCompoundBondForce(n_particles, energy_function)


def add_force_in_separate_group(
    system: openmm.System,
    force: openmm.Force,
):
    """
    Add force to a System in a separate force group.

    Parameters
    ----------
    system : openmm.System
      System to add the Force to.
    force : openmm.Force
      The Force to add to the System.

    Raises
    ------
    ValueError
      If all 32 force groups are occupied.


    TODO
    ----
    Unlike the original Yank implementation, we assume that
    all 32 force groups will not be filled. Should this be an issue
    we can consider just separating it from NonbondedForce.

    Acknowledgements
    ----------------
    Mostly reproduced from `Yank <https://github.com/choderalab/yank>`_.
    """
    available_force_groups = set(range(32))
    for force in system.getForces():
        available_force_groups.discard(force.getForceGroup())

    force.setForceGroup(min(available_force_groups))
    system.addForce(force)
