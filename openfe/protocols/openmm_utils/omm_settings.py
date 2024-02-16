# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""Equilibrium Free Energy Protocols input settings.

This module implements base settings necessary to run
free energy calculations using OpenMM +/- Tools, such
as :mod:`openfe.protocols.openmm_rfe.equil_rfe_methods.py`
and :mod`openfe.protocols.openmm_afe.equil_afe_methods.py`
"""
from __future__ import annotations

from typing import Optional, Literal
from openff.units import unit
from openff.models.types import FloatQuantity

from gufe.settings import (
    Settings,
    SettingsBaseModel,
    OpenMMSystemGeneratorFFSettings,
    ThermoSettings,
)


try:
    from pydantic.v1 import validator
except ImportError:
    from pydantic import validator  # type: ignore[assignment]


class BaseSolvationSettings(SettingsBaseModel):
    """
    Base class for SolvationSettings objects
    """
    class Config:
        arbitrary_types_allowed = True


class OpenMMSolvationSettings(BaseSolvationSettings):
    """Settings for controlling how a system is solvated using OpenMM tooling

    Note
    ----
    No solvation will happen if a SolventComponent is not passed.

    """
    solvent_model: Literal['tip3p', 'spce', 'tip4pew', 'tip5p'] = 'tip3p'
    """
    Force field water model to use.
    Allowed values are; `tip3p`, `spce`, `tip4pew`, and `tip5p`.
    """

    solvent_padding: FloatQuantity['nanometer'] = 1.2 * unit.nanometer
    """Minimum distance from any solute atoms to the solvent box edge."""

    @validator('solvent_padding')
    def is_positive_distance(cls, v):
        # these are time units, not simulation steps
        if not v.is_compatible_with(unit.nanometer):
            raise ValueError("solvent_padding must be in distance units "
                             "(i.e. nanometers)")
        if v < 0:
            errmsg = "solvent_padding must be a positive value"
            raise ValueError(errmsg)
        return v


class BasePartialChargeSettings(SettingsBaseModel):
    """
    Base class for partial charge assignment.
    """
    class Config:
        arbitrary_types_allowed = True


class OpenFFPartialChargeSettings(BasePartialChargeSettings):
    """
    Empty placeholder class, will implement OpenFF partial charge assignment.
    """


class OpenMMEngineSettings(SettingsBaseModel):
    """OpenMM MD engine settings"""

    """
    TODO
    ----
    * In the future make precision and deterministic forces user defined too.
    """

    compute_platform: Optional[str] = None
    """
    OpenMM compute platform to perform MD integration with. If None, will
    choose fastest available platform. Default None.
    """


class IntegratorSettings(SettingsBaseModel):
    """Settings for the LangevinSplittingDynamicsMove integrator"""

    class Config:
        arbitrary_types_allowed = True

    timestep: FloatQuantity['femtosecond'] = 4 * unit.femtosecond
    """Size of the simulation timestep. Default 4 * unit.femtosecond."""
    langevin_collision_rate: FloatQuantity['1/picosecond'] = 1.0 / unit.picosecond
    """Collision frequency. Default 1.0 / unit.pisecond."""
    reassign_velocities = False
    """
    If True, velocities are reassigned from the Maxwell-Boltzmann
    distribution at the beginning of move. Default False.
    """
    n_restart_attempts = 20
    """
    Number of attempts to restart from Context if there are NaNs in the
    energies after integration. Default 20.
    """
    constraint_tolerance = 1e-06
    """Tolerance for the constraint solver. Default 1e-6."""
    barostat_frequency = 25 * unit.timestep  # todo: IntQuantity
    """
    Frequency at which volume scaling changes should be attempted.
    Default 25 * unit.timestep.
    """
    remove_com: bool = False
    """
    Whether or not to remove the center of mass motion. Default False.
    """

    @validator('langevin_collision_rate', 'n_restart_attempts')
    def must_be_positive_or_zero(cls, v):
        if v < 0:
            errmsg = ("langevin_collision_rate, and n_restart_attempts must be"
                      f" zero or positive values, got {v}.")
            raise ValueError(errmsg)
        return v

    @validator('timestep', 'constraint_tolerance')
    def must_be_positive(cls, v):
        if v <= 0:
            errmsg = ("timestep, and constraint_tolerance "
                      f"must be positive values, got {v}.")
            raise ValueError(errmsg)
        return v

    @validator('timestep')
    def is_time(cls, v):
        # these are time units, not simulation steps
        if not v.is_compatible_with(unit.picosecond):
            raise ValueError("timestep must be in time units "
                             "(i.e. picoseconds)")
        return v

    @validator('langevin_collision_rate')
    def must_be_inverse_time(cls, v):
        if not v.is_compatible_with(1 / unit.picosecond):
            raise ValueError("langevin collision_rate must be in inverse time "
                             "(i.e. 1/picoseconds)")
        return v


class OutputSettings(SettingsBaseModel):
    """
    Settings for simulation output settings,
    writing to disk, etc...
    """
    class Config:
        arbitrary_types_allowed = True

    # reporter settings
    output_filename = 'simulation.nc'
    """Path to the trajectory storage file. Default 'simulation.nc'."""
    output_structure = 'hybrid_system.pdb'
    """
    Path of the output hybrid topology structure file. This is used
    to visualise and further manipulate the system.
    Default 'hybrid_system.pdb'.
    """
    output_indices = 'not water'
    """
    Selection string for which part of the system to write coordinates for.
    Default 'not water'.
    """
    checkpoint_interval: FloatQuantity['picosecond'] = 1 * unit.picosecond
    """
    Frequency to write the checkpoint file. Default 1 * unit.picosecond.
    """
    checkpoint_storage_filename = 'checkpoint.chk'
    """
    Separate filename for the checkpoint file. Note, this should
    not be a full path, just a filename. Default 'checkpoint.chk'.
    """
    forcefield_cache: Optional[str] = 'db.json'
    """
    Filename for caching small molecule residue templates so they can be
    later reused.
    """

    @validator('checkpoint_interval')
    def must_be_positive(cls, v):
        if v <= 0:
            errmsg = f"Checkpoint intervals must be positive, got {v}."
            raise ValueError(errmsg)
        return v


class SimulationSettings(SettingsBaseModel):
    """
    Settings for simulation control, including lengths, etc...
    """
    class Config:
        arbitrary_types_allowed = True

    minimization_steps = 5000
    """Number of minimization steps to perform. Default 5000."""
    equilibration_length: FloatQuantity['nanosecond']
    """
    Length of the equilibration phase in units of time. The total number of
    steps from this equilibration length
    (i.e. ``equilibration_length`` / :class:`IntegratorSettings.timestep`)
    must be a multiple of the value defined for
    :class:`AlchemicalSamplerSettings.steps_per_iteration`.
    """
    production_length: FloatQuantity['nanosecond']
    """
    Length of the production phase in units of time. The total number of
    steps from this production length (i.e.
    ``production_length`` / :class:`IntegratorSettings.timestep`) must be
    a multiple of the value defined for :class:`IntegratorSettings.nsteps`.
    """

    @validator('equilibration_length', 'production_length')
    def is_time(cls, v):
        # these are time units, not simulation steps
        if not v.is_compatible_with(unit.picosecond):
            raise ValueError("Durations must be in time units")
        return v

    @validator('minimization_steps', 'equilibration_length',
               'production_length')
    def must_be_positive(cls, v):
        if v <= 0:
            errmsg = ("Minimization steps, and MD lengths must be positive, "
                      f"got {v}")
            raise ValueError(errmsg)
        return v


class MultiStateSimulationSettings(SimulationSettings):
    """
    Settings for simulation control for multistate simulations,
    including simulation length and details of the alchemical sampler.
    """

    """
    TODO
    ----
    * It'd be great if we could pass in the sampler object rather than using
      strings to define which one we want.
    * Make n_replicas optional such that: If `None` or greater than the number
      of lambda windows set in :class:`AlchemicalSettings`, this will default
      to the number of lambda windows. If less than the number of lambda
      windows, the replica lambda states will be picked at equidistant
      intervals along the lambda schedule.
    """

    class Config:
        arbitrary_types_allowed = True

    sampler_method = "repex"
    """
    Alchemical sampling method, must be one of;
    `repex` (Hamiltonian Replica Exchange),
    `sams` (Self-Adjusted Mixture Sampling),
    or `independent` (independently sampled lambda windows).
    Default `repex`.
    """
    time_per_iteration: FloatQuantity['picosecond'] = 1 * unit.picosecond
    # todo: Add validators in the protocol
    """
    Simulation time between each MCMC move attempt. Default 1 * unit.picosecond.
    """
    real_time_analysis_interval: Optional[FloatQuantity['picosecond']] = 250 * unit.picosecond
    # todo: Add validators in the protocol
    """
    Time interval at which to perform an analysis of the free energies.

    At each interval, real time analysis data (e.g. current free energy
    estimate and timing data) will be written to a yaml file named
    ``<OutputSettings.output_filename>_real_time_analysis.yaml``. The
    current error in the estimate will also be assessed and if it drops
    below ``MultiStateSimulationSettings.early_termination_target_error``
    the simulation will be terminated.

    If ``None``, no real time analysis will be performed and the yaml
    file will not be written.

    Must be a multiple of ``OutputSettings.checkpoint_interval``

    Default `250`.
    
    """
    early_termination_target_error: Optional[FloatQuantity['kcal/mol']] = 0.0 * unit.kilocalorie_per_mole
    # todo: have default ``None`` or ``0.0 * unit.kilocalorie_per_mole``
    #  (later would give an example of unit).
    """
    Target error for the real time analysis measured in kcal/mol. Once the MBAR 
    error of the free energy is at or below this value, the simulation will be 
    considered complete. 
    A suggested value of 0.12 * `unit.kilocalorie_per_mole` has
    shown to be effective in both hydration and binding free energy benchmarks.
    Default ``None``, i.e. no early termination will occur.
    """
    real_time_analysis_minimum_time: FloatQuantity['picosecond'] = 500 * unit.picosecond
    # todo: Add validators in the protocol
    """
    Simulation time which must pass before real time analysis is
    carried out. 
    
    Default 500 * unit.picosecond.
    """

    sams_flatness_criteria = 'logZ-flatness'
    """
    SAMS only. Method for assessing when to switch to asymptomatically
    optimal scheme.
    One of ['logZ-flatness', 'minimum-visits', 'histogram-flatness'].
    Default 'logZ-flatness'.
    """
    sams_gamma0 = 1.0
    """SAMS only. Initial weight adaptation rate. Default 1.0."""
    n_replicas = 11
    """Number of replicas to use. Default 11."""

    @validator('sams_flatness_criteria')
    def supported_flatness(cls, v):
        supported = [
            'logz-flatness', 'minimum-visits', 'histogram-flatness'
        ]
        if v.lower() not in supported:
            errmsg = ("Only the following sams_flatness_criteria are "
                      f"supported: {supported}")
            raise ValueError(errmsg)
        return v

    @validator('sampler_method')
    def supported_sampler(cls, v):
        supported = ['repex', 'sams', 'independent']
        if v.lower() not in supported:
            errmsg = ("Only the following sampler_method values are "
                      f"supported: {supported}")
            raise ValueError(errmsg)
        return v

    @validator('n_replicas', 'time_per_iteration')
    def must_be_positive(cls, v):
        if v <= 0:
            errmsg = "n_replicas and steps_per_iteration must be positive " \
                     f"values, got {v}."
            raise ValueError(errmsg)
        return v

    @validator('early_termination_target_error',
               'real_time_analysis_minimum_time', 'sams_gamma0',
               'n_replicas')
    def must_be_zero_or_positive(cls, v):
        if v < 0:
            errmsg = ("Early termination target error, minimum iteration and"
                      f" SAMS gamma0 must be 0 or positive values, got {v}.")
            raise ValueError(errmsg)
        return v


class MDSimulationSettings(SimulationSettings):
    """
    Settings for simulation control for plain MD simulations
    """
    class Config:
        arbitrary_types_allowed = True

    equilibration_length_nvt: unit.Quantity
    """
    Length of the equilibration phase in the NVT ensemble in units of time. 
    The total number of steps from this equilibration length
    (i.e. ``equilibration_length_nvt`` / :class:`IntegratorSettings.timestep`).
    """


class MDOutputSettings(OutputSettings):
    """ Settings for simulation output settings for plain MD simulations."""
    class Config:
        arbitrary_types_allowed = True

    # reporter settings
    production_trajectory_filename = 'simulation.xtc'
    """Path to the storage file for analysis. Default 'simulation.xtc'."""
    trajectory_write_interval: FloatQuantity['picosecond'] = 20 * unit.picosecond
    """
    Frequency to write the xtc file. Default 5000 * unit.timestep.
    """
    preminimized_structure = 'system.pdb'
    """Path to the pdb file of the full pre-minimized system. 
    Default 'system.pdb'."""
    minimized_structure = 'minimized.pdb'
    """Path to the pdb file of the system after minimization. 
    Only the specified atom subset is saved. Default 'minimized.pdb'."""
    equil_nvt_structure = 'equil_nvt.pdb'
    """Path to the pdb file of the system after NVT equilibration. 
    Only the specified atom subset is saved. Default 'equil_nvt.pdb'."""
    equil_npt_structure = 'equil_npt.pdb'
    """Path to the pdb file of the system after NPT equilibration. 
    Only the specified atom subset is saved. Default 'equil_npt.pdb'."""
    log_output = 'simulation.log'
    """
    Filename for writing the log of the MD simulation, including timesteps,
    energies, density, etc.
    """
