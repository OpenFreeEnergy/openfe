# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Reusable utility methods to validate input settings to OpenMM-based alchemical
Protocols.
"""
import warnings
from typing import Dict, List, Tuple
from openff.units import unit
from gufe import (
    Component, ChemicalSystem, SolventComponent, ProteinComponent
)
from openfe.protocols.openmm_utils.omm_settings import OpenMMSolvationSettings


def validate_openmm_solvation_settings(
    settings: OpenMMSolvationSettings
) -> None:
    """
    Checks that the OpenMMSolvation settings are correct.

    Raises
    ------
    ValueError
      If more than one of ``solvent_padding``, ``number_of_solvent_molecules``,
      ``box_vectors``, or ``box_size`` are defined.
      If ``box_shape`` is defined alongside either ``box_vectors``,
      or ``box_size``.
    """
    unique_attributes = (
        settings.solvent_padding, settings.number_of_solvent_molecules,
        settings.box_vectors, settings.box_size,
    )
    if len([x for x in unique_attributes if x is not None]) > 1:
        errmsg = ("Only one of solvent_padding, number_of_solvent_molecules, "
                  "box_vectors, and box_size can be defined in the solvation "
                  "settings.")
        raise ValueError(errmsg)

    if settings.box_shape is not None:
        if settings.box_size is not None or settings.box_vectors is not None:
            errmsg = ("box_shape cannot be defined alongside either box_size "
                      "or box_vectors in the solvation settings.")
            raise ValueError(errmsg)


def validate_timestep(hmass: float, timestep: unit.Quantity):
    """
    Check that the input timestep is suitable for the given hydrogen
    mass.

    Parameters
    ----------
    hmass : float
      The target hydrogen mass (assumed units of amu).
    timestep : unit.Quantity
      The integration time step.


    Raises
    ------
    ValueError
      If the hydrogen mass is less than 3 amu and the timestep is
      greater than 2 fs.
    """
    if hmass < 3.0:
        if timestep > 2.0 * unit.femtoseconds:
            errmsg = f"timestep {timestep} too large for hydrogen mass {hmass}"
            raise ValueError(errmsg)


def get_simsteps(sim_length: unit.Quantity,
                 timestep: unit.Quantity, mc_steps: int) -> int:
    """
    Gets and validates the number of simulation steps.

    Parameters
    ----------
    sim_length : unit.Quantity
      Simulation length.
    timestep : unit.Quantity
      Integration timestep.
    mc_steps : int
      Number of integration timesteps between MCMC moves.

    Returns
    -------
    sim_steps : int
      The number of simulation timesteps.
    """

    sim_time = round(sim_length.to('attosecond').m)
    ts = round(timestep.to('attosecond').m)

    sim_steps, mod = divmod(sim_time, ts)
    if mod != 0:
        raise ValueError("Simulation time not divisible by timestep")

    if (sim_steps % mc_steps) != 0:
        errmsg =  (f"Simulation time {sim_time/1000000} ps should contain a "
                   "number of steps divisible by the number of integrator "
                   f"timesteps between MC moves {mc_steps}")
        raise ValueError(errmsg)

    return sim_steps
