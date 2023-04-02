# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

"""Settings class for equilibrium AFE Protocols using OpenMM + OpenMMTools

This module implements the necessary settings necessary to run absolute free
energies using :class:`openfe.protocols.openmm_abfe.equil_abfe_methods.py`


TODO
----
* Add support for restraints
* Improve this docstring by adding an example use case.

"""
from typing import Optional
from pydantic import validator
from openff.units import unit
from gufe import settings


class SystemSettings(settings.SettingsBaseModel):
    """Settings describing the simulation system settings."""

    class Config:
        arbitrary_types_allowed = True

    nonbonded_method = 'PME'
    """
    Method for treating nonbonded interactions, currently only PME and
    NoCutoff are allowed. Default PME.
    """
    nonbonded_cutoff = 1.0 * unit.nanometer
    """
    Cutoff value for short range nonbonded interactions.
    Default 1.0 * unit.nanometer.
    """

    @validator('nonbonded_method')
    def allowed_nonbonded(cls, v):
        if v.lower() not in ['pme', 'nocutoff']:
            errmsg = ("Only PME and NoCutoff are allowed nonbonded_methods")
            raise ValueError(errmsg)
        return v

    @validator('nonbonded_cutoff')
    def is_positive_distance(cls, v):
        # these are time units, not simulation steps
        if not v.is_compatible_with(unit.nanometer):
            raise ValueError("nonbonded_cutoff must be in distance units "
                             "(i.e. nanometers)")
        if v < 0:
            errmsg = "nonbonded_cutoff must be a positive value"
            raise ValueError(errmsg)
        return v


class SolventSettings(settings.SettingsBaseModel):
    """Settings for solvating the system

    NOTE
    ----
    * No solvation will happen if a SolventComponent is not passed.
    """
    solvent_model = 'tip3p'
    """
    Force field water model to use.
    Allowed values are; `tip3p`, `spce`, `tip4pew`, and `tip5p`.
    """
    class Config:
        arbitrary_types_allowed = True

    solvent_padding = 1.2 * unit.nanometer

    @validator('solvent_model')
    def allowed_solvent(cls, v):
        allowed_models = ['tip3p', 'spce', 'tip4pew', 'tip5p']
        if v.lower() not in allowed_models:
            errmsg = (
                f"Only {allowed_models} are allowed solvent_model values"
            )
            raise ValueError(errmsg)
        return v

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


class AlchemicalSettings(settings.SettingsBaseModel):
    """Settings for the alchemical protocol

    These settings describe the lambda schedule and the creation of the
    hybrid system.
    """

    lambda_elec_windows = 12
    """Number of lambda electrostatic alchemical steps, default 12"""
    lambda_vdw_windows = 12
    """Number of lambda vdw alchemical steps, default 12"""

    @validator('lambda_elec_windows', 'lambda_vdw_windows')
    def must_be_positive(cls, v):
        if v <= 0:
            errmsg = ("Number of lambda steps must be positive ")
            raise ValueError(errmsg)
        return v


class OpenMMEngineSettings(settings.SettingsBaseModel):
    """OpenMM MD engine settings

    TODO
    ----
    * In the future make precision and deterministic forces user defined too.
    """

    compute_platform: Optional[str] = None
    """
    OpenMM compute platform to perform MD integration with. If None, will
    choose fastest available platform. Default None.
    """


class AlchemicalSamplerSettings(settings.SettingsBaseModel):
    """Settings for the Equilibrium Alchemical sampler, currently supporting
    either MultistateSampler, SAMSSampler or ReplicaExchangeSampler.


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
    online_analysis_interval: Optional[int] = None
    """
    Interval at which to perform an analysis of the free energies.
    At each interval the free energy is estimate and the simulation is
    considered complete if the free energy estimate is below
    ``online_analysis_target_error``. If set, will write a yaml file with
    real time analysis data. Default `None`.
    """
    online_analysis_target_error = 0.005 * unit.boltzmann_constant * unit.kelvin
    """
    Target error for the online analysis measured in kT. Once the free energy
    is at or below this value, the simulation will be considered complete.
    """
    online_analysis_minimum_iterations = 1000
    """
    Number of iterations which must pass before online analysis is
    carried out. Default 50.
    """
    n_repeats: int = 3
    """
    Number of independent repeats to run.  Default 3
    """
    flatness_criteria = 'logZ-flatness'
    """
    SAMS only. Method for assessing when to switch to asymptomatically
    optimal scheme.

    One of ['logZ-flatness', 'minimum-visits', 'histogram-flatness'].

    Default 'logZ-flatness'.
    """
    gamma0 = 1.0
    """SAMS only. Initial weight adaptation rate. Default 1.0."""
    n_replicas = 24
    """Number of replicas to use. Default 24."""

    @validator('flatness_criteria')
    def supported_flatness(cls, v):
        supported = [
            'logz-flatness', 'minimum-visits', 'histogram-flatness'
        ]
        if v.lower() not in supported:
            errmsg = ("Only the following flatness_criteria are "
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

    @validator('n_repeats', 'n_replicas')
    def must_be_positive(cls, v):
        if v <= 0:
            errmsg = "n_repeats and n_replicas must be positive values"
            raise ValueError(errmsg)
        return v

    @validator('online_analysis_target_error', 'n_repeats',
               'online_analysis_minimum_iterations', 'gamma0', 'n_replicas')
    def must_be_zero_or_positive(cls, v):
        if v < 0:
            errmsg = ("Online analysis target error, minimum iteration "
                      "and SAMS gamm0 must be 0 or positive values.")
            raise ValueError(errmsg)
        return v


class IntegratorSettings(settings.SettingsBaseModel):
    """Settings for the LangevinSplittingDynamicsMove integrator"""

    class Config:
        arbitrary_types_allowed = True

    timestep = 2 * unit.femtosecond
    """Size of the simulation timestep. Default 2 * unit.femtosecond."""
    collision_rate = 1 / unit.picosecond
    """Collision frequency. Default 1 / unit.pisecond."""
    n_steps = 250 * unit.timestep
    """
    Number of integration timesteps between each time the MCMC move
    is applied. Default 1000.
    """
    reassign_velocities = True
    """
    If True, velocities are reassigned from the Maxwell-Boltzmann
    distribution at the beginning of move. Default False.
    """
    splitting = "V R O R V"
    """
    Sequence of "R", "V", "O" substeps to be carried out at each timestep.
    Default "V R O R V".
    """
    n_restart_attempts = 20
    """
    Number of attempts to restart from Context if there are NaNs in the
    energies after integration. Default 20.
    """
    constraint_tolerance = 1e-06
    """Tolerance for the constraint solver. Default 1e-6."""
    barostat_frequency = 25 * unit.timestep
    """
    Frequency at which volume scaling changes should be attempted.
    Default 25 * unit.timestep.
    """

    @validator('collision_rate', 'n_restart_attempts')
    def must_be_positive_or_zero(cls, v):
        if v < 0:
            errmsg = ("collision_rate, and n_restart_attempts must be "
                      "zero or positive values")
            raise ValueError(errmsg)
        return v

    @validator('timestep', 'n_steps', 'constraint_tolerance')
    def must_be_positive(cls, v):
        if v <= 0:
            errmsg = ("timestep, n_steps, constraint_tolerance "
                      "must be positive values")
            raise ValueError(errmsg)
        return v

    @validator('timestep')
    def is_time(cls, v):
        # these are time units, not simulation steps
        if not v.is_compatible_with(unit.picosecond):
            raise ValueError("timestep must be in time units "
                             "(i.e. picoseconds)")
        return v

    @validator('collision_rate')
    def must_be_inverse_time(cls, v):
        if not v.is_compatible_with(1 / unit.picosecond):
            raise ValueError("collision_rate must be in inverse time "
                             "(i.e. 1/picoseconds)")
        return v


class SimulationSettings(settings.SettingsBaseModel):
    """
    Settings for simulation control, including lengths,
    writing to disk, etc...
    """
    class Config:
        arbitrary_types_allowed = True

    minimization_steps = 5000
    """Number of minimization steps to perform. Default 10000."""
    equilibration_length: unit.Quantity
    """
    Length of the equilibration phase in units of time. The total number of
    steps from this equilibration length
    (i.e. ``equilibration_length`` / :class:`IntegratorSettings.timestep`)
    must be a multiple of the value defined for
    :class:`IntegratorSettings.n_steps`.
    """
    production_length: unit.Quantity
    """
    Length of the production phase in units of time. The total number of
    steps from this production length (i.e.
    ``production_length`` / :class:`IntegratorSettings.timestep`) must be
    a multiple of the value defined for :class:`IntegratorSettings.nsteps`.
    """

    # reporter settings
    output_filename = 'abfe.nc'
    """Path to the storage file for analysis. Default 'rbfe.nc'."""
    output_indices = 'all'
    """
    Selection string for which part of the system to write coordinates for.
    Default 'all'.
    """
    checkpoint_interval = 100 * unit.timestep
    """
    Frequency to write the checkpoint file. Default 50 * unit.timestep.
    """
    checkpoint_storage = 'abfe_checkpoint.nc'
    """
    Separate filename for the checkpoint file. Note, this should
    not be a full path, just a filename. Default 'rbfe_checkpoint.nc'.
    """
    forcefield_cache: Optional[str] = None
    """
    Filename for caching small molecule residue templates so they can be
    later reused.
    """

    @validator('equilibration_length', 'production_length')
    def is_time(cls, v):
        # these are time units, not simulation steps
        if not v.is_compatible_with(unit.picosecond):
            raise ValueError("Durations must be in time units")
        return v

    @validator('minimization_steps', 'equilibration_length',
               'production_length', 'checkpoint_interval')
    def must_be_positive(cls, v):
        if v <= 0:
            errmsg = ("Minimization steps, MD lengths, and checkpoint "
                      "intervals must be positive")
            raise ValueError(errmsg)
        return v


class AbsoluteTransformSettings(settings.Settings):
    class Config:
        arbitrary_types_allowed = True

    # Things for creating the systems
    system_settings: SystemSettings
    solvent_settings: SolventSettings

    # Alchemical settings
    alchemical_settings: AlchemicalSettings
    alchemsampler_settings: AlchemicalSamplerSettings

    # MD Engine things
    engine_settings: OpenMMEngineSettings

    # Sampling State defining things
    integrator_settings: IntegratorSettings

    # Simulation run settings
    simulation_settings: SimulationSettings
