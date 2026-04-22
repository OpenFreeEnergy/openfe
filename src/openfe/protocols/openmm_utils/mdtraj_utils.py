# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import mdtraj as mdt
import numpy as np
import openmm
from openmm import unit as omm_unit


def mdtraj_from_openmm(
    omm_topology: openmm.app.Topology,
    omm_positions: openmm.unit.Quantity,
):
    """
    Get an mdtraj object from an OpenMM topology and positions.

    Parameters
    ----------
    omm_topology : openmm.app.Topology
      The OpenMM topology
    omm_positions : openmm.unit.Quantity
      The OpenMM positions

    Returns
    -------
    mdtraj_trajectory : md.Trajectory
    """
    mdtraj_topology = mdt.Topology.from_openmm(omm_topology)
    positions_in_mdtraj_format = omm_positions.value_in_unit(omm_unit.nanometers)

    box = omm_topology.getPeriodicBoxVectors()
    if box is not None:
        x, y, z = [np.array(b._value) for b in box]
        lx = np.linalg.norm(x)
        ly = np.linalg.norm(y)
        lz = np.linalg.norm(z)
        # angle between y and z
        alpha = np.arccos(np.dot(y, z) / (ly * lz))
        # angle between x and z
        beta = np.arccos(np.dot(x, z) / (lx * lz))
        # angle between x and y
        gamma = np.arccos(np.dot(x, y) / (lx * ly))

        unitcell_lengths = np.array([lx, ly, lz])
        unitcell_angles = np.array([np.rad2deg(alpha), np.rad2deg(beta), np.rad2deg(gamma)])
    else:
        unitcell_lengths = None
        unitcell_angles = None

    mdtraj_trajectory = mdt.Trajectory(
        positions_in_mdtraj_format,
        mdtraj_topology,
        unitcell_lengths=unitcell_lengths,
        unitcell_angles=unitcell_angles,
    )

    return mdtraj_trajectory
