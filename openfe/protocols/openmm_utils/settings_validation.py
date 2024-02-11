# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Reusable utility methods to validate input settings to OpenMM-based alchemical
Protocols.
"""
from openff.units import unit
from typing import Optional
from .omm_settings import (
    IntegratorSettings,
    MultiStateSimulationSettings,
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
        simulation_settings: MultiStateSimulationSettings,
        integrator_settings: IntegratorSettings,
) -> int:
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
    tpi_fs = round(simulation_settings.time_per_iteration.to(unit.attosecond).m)
    ts_fs = round(integrator_settings.timestep.to(unit.attosecond).m)
    steps_per_iteration, rem = divmod(tpi_fs, ts_fs)

    if rem:
        raise ValueError(f"time_per_iteration ({simulation_settings.time_per_iteration}) "
                         f"not divisible by timestep ({integrator_settings.timestep})")

    return steps_per_iteration


def convert_time_to_iterations(
    time: unit.Quantity,
    time_per_iteration: unit.Quantity,
    check_remainder: bool = True,
) -> tuple[int, int]:
    """
    Convert a set amount of time to a number of iterations.

    This method allows one to get the number of MC iterations as used
    in OpenMMTools' MultiStateSampler and MultiStatereporter.


    Parameters
    ---------
    time: unit.Quantity
      The time to convert in a number of MC iterations.
    time_per_iteration : unit.Quantity
      The amount of time which each iteration takes.
    check_remainder : bool
      If true, raises an error if the remainder is not zero.

    Returns
    -------
    iterations : int
      The number of iterations covered by the input time.
    remainder : int
      The remainder of the input time and time_per_iteration division.

    Raises
    ------
    ValueError
      If ``check_remainder`` is true and the the time does not exactly
      divide by the time per iteration.
    """
    time_ats = round(time.to(unit.attosecond).m)
    tpi_ats = round(time_per_iteration.to(unit.attosecond).m)

    iterations, remainder = divmod(time_ats, tpi_ats)

    if check_remainder and remainder:
        errmsg = "Input time does not divide exactly by the time per iteration"
        raise ValueError(errmsg)

    return iterations, remainder


def convert_real_time_analysis_iterations(
    simulation_settings: MultiStateSimulationSettings,
) -> tuple[Optional[int], Optional[int]]:
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
    real_time_analysis_iterations : Optional[int]
      suitable for input to online_analysis_interval
    real_time_analysis_minimum_iterations : Optional[int]
      suitable for input to real_time_analysis_minimum_iterations
    """
    if simulation_settings.real_time_analysis_interval is None:
        # option to turn off real time analysis
        return None, None

    rta_its, rem = convert_time_to_iterations(
        simulation_settings.real_time_analysis_interval,
        simulation_settings.time_per_iteration,
        check_remainder=False,
    )

    if rem:
        raise ValueError(f"real_time_analysis_interval ({simulation_settings.real_time_analysis_interval}) "
                         f"is not divisible by time_per_iteration ({simulation_settings.time_per_iteration})")

    # convert RTA_minimum_time to iterations
    rta_min_its, rem = convert_time_to_iterations(
        simulation_settings.real_time_analysis_minimum_time,
        simulation_settings.time_per_iteration,
        check_remainder=False,
    )

    if rem:
        raise ValueError(f"real_time_analysis_minimum_time ({simulation_settings.real_time_analysis_minimum_time}) "
                         f"is not divisible by time_per_iteration ({simulation_settings.time_per_iteration})")

    return rta_its, rta_min_its


def convert_target_error_from_kcal_per_mole_to_kT(
    temperature,
    target_error,
) -> float:
    """Convert kcal/mol target error to kT units

    If target_error is 0.0, returns 0.0

    Parameters
    ----------
    temperature : unit.Quantity
      temperature in K
    target_error : unit.Quantity
      error in kcal/mol

    Returns
    -------
    early_termination_target_error : float
      in units of kT, suitable for input as "online_analysis_target_error" in a
      Sampler
    """
    if target_error:
        kB = 0.001987204 * unit.kilocalorie_per_mole / unit.kelvin
        kT = temperature * kB
        early_termination_target_error = target_error / kT
    else:
        return 0.0

    return early_termination_target_error.m
