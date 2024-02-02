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


def convert_steps_per_iteration(
        simulation_settings,
        integrator_settings,
):
    """Convert time per iteration to steps

    Parameters
    ----------
        simulation_settings: MultiStateSimulationSettings
    integrator_settings: IntegratorSettings

    Returns
    -------
    steps_per_iteration : int
      suitable for input to Integrator
    """
    # TODO: Check this is correct
    tpi_fs = simulation_settings.time_per_iteration.to(unit.femtosecond).m
    ts_fs = integrator_settings.timestep.to(unit.femtosecond).m
    steps_per_iteration = int(round(tpi_fs / ts_fs))

    return steps_per_iteration


def convert_real_time_analysis_iterations(
        simulation_settings,
):
    """Convert time units in Settings to various other units

    Interally openmmtools uses various quantities with units of time,
    steps, and iterations.

    Our Settings objects instead have things defined in time (fs or ps).

    This function generates suitable inputs for the openmmtools objects

    Parameters
    ----------
    simulation_settings: MultiStateSimulationSettings

    Returns
    -------
    real_time_analysis_iterations : int
      suitable for input to online_analysis_interval
    real_time_analysis_minimum_iterations : int
      suitable for input to real_time_analysis_minimum_iterations
    """
    # TODO: Check this is correct
    tpi_fs = simulation_settings.time_per_iteration.to(unit.femtosecond).m

    # convert real_time_analysis time to interval
    # rta_its must be number of MCMC iterations
    # i.e. rta_fs / tpi_fs -> number of iterations
    rta_fs = simulation_settings.real_time_analysis_interval.to(unit.femtosecond).m
    rta_its = round(int(rta_fs / tpi_fs))

    # convert RTA_minimum_time to iterations
    rta_min_fs = simulation_settings.real_time_analysis_minimum_time.to(unit.femtosecond).m
    rta_min_its = round(int(rta_min_fs / tpi_fs))

    return rta_its, rta_min_its


def convert_target_error(
        thermo_settings,
        simulation_settings,
):
    """Convert kcal/mol target error to kT units

    Parameters
    ----------
    thermo_settings: ThermoSettings
    simulation_settings: MultiStateSimulationSettings

    Returns
    -------
    early_termination_target_error : float
      in units of kT, suitable for input as "online_analysis_target_error" in a
      Sampler
    """
    temp = thermo_settings.temperature
    if simulation_settings.early_termination_target_error:
        # TODO: Check conversions here
        kB = 0.001987204 * unit.kilocalorie_per_mole / unit.kelvin
        kT = temp * kB
        early_termination_target_error = kT / simulation_settings.early_termination_target_error
    else:
        early_termination_target_error = 0.0

    return early_termination_target_error
