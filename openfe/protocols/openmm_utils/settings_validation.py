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


def get_simsteps(equil_length: unit.Quantity, prod_length: unit.Quantity,
                 timestep: unit.Quantity, mc_steps: int) -> Tuple[int, int]:
    """
    Gets and validates the number of equilibration and production steps.

    Parameters
    ----------
    equil_length : unit.Quantity
      Simulation equilibration length.
    prod_length : unit.Quantity
      Simulation production length.
    timestep : unit.Quantity
      Integration timestep.
    mc_steps : int
      Number of integration timesteps between MCMC moves.

    Returns
    -------
    equil_steps : int
      The number of equilibration timesteps.
    prod_steps : int
      The number of production timesteps.
    """

    equil_time = round(equil_length.to('attosecond').m)
    prod_time = round(prod_length.to('attosecond').m)
    ts = round(timestep.to('attosecond').m)

    equil_steps, mod = divmod(equil_time / ts)
    if mod != 0:
        raise ValueError("Equilibration time not divisible by timestep")
    prod_steps, mod = divmod(prod_time / ts)
    if mod != 0:
        raise ValueError("Production time not divisible by timestep")

    for var in [("Equilibration", equil_steps, equil_time),
                ("Production", prod_steps, prod_time)]:
        if (var[1] % mc_steps) != 0:
            errmsg =  (f"{var[0]} time {var[2]/1000000} ps should contain a "
                       "number of steps divisible by the number of integrator "
                       f"timesteps between MC moves {mc_steps}")
            raise ValueError(errmsg)

    return equil_steps, prod_steps
