# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""Equilibrium Relative Free Energy Protocol input settings.

This module implements the necessary settings necessary to run absolute free
energies using :class:`openfe.protocols.openmm_rfe.equil_rfe_methods.py`

"""
from __future__ import annotations

import abc
from typing import Optional, Union
from pydantic import Extra, validator, BaseModel, PositiveFloat, Field
from openff.units import unit
import os


if os.environ.get('SPHINX', False):  #pragma: no cover
    class SettingsBaseModel(BaseModel):
        class Config:
            extra = Extra.forbid
            validate_assignment = True
            arbitrary_types_allowed = True
            smart_union = True


    class ThermoSettings(SettingsBaseModel):
        """Settings for thermodynamic parameters.

        .. note::
           No checking is done to ensure a valid thermodynamic ensemble is
           possible.
        """

        temperature = 298.15 * unit.kelvin
        """Simulation temperature, default units kelvin"""

        pressure = 1.0 * unit.bar
        """Simulation pressure, default units standard atmosphere (atm)"""

        ph: Union[PositiveFloat, None] = Field(None, description="Simulation pH")
        redox_potential: Optional[float] = Field(
            None, description="Simulation redox potential"
        )


    class BaseForceFieldSettings(SettingsBaseModel, abc.ABC):
        """Base class for ForceFieldSettings objects"""
        ...


    class OpenMMSystemGeneratorFFSettings(SettingsBaseModel):
        """Parameters to set up the force field with OpenMM ForceFields

        .. note::
           Right now we just basically just grab what we need for the
           :class:`openmmforcefields.system_generators.SystemGenerator`
           signature. See the `OpenMMForceField SystemGenerator documentation`_
           for more details.


        .. _`OpenMMForceField SystemGenerator documentation`:
           https://github.com/openmm/openmmforcefields#automating-force-field-management-with-systemgenerator
        """
        constraints: Optional[str] = 'hbonds'
        """Constraints to be applied to system.
           One of 'hbonds', 'allbonds', 'hangles' or None, default 'hbonds'"""

        rigid_water: bool = True
        remove_com: bool = False
        hydrogen_mass: float = 3.0
        """Mass to be repartitioned to hydrogens from neighbouring
           heavy atoms (in amu), default 3.0"""

        forcefields: list[str] = [
            "amber/ff14SB.xml",  # ff14SB protein force field
            "amber/tip3p_standard.xml",  # TIP3P and recommended monovalent ion parameters
            "amber/tip3p_HFE_multivalent.xml",  # for divalent ions
            "amber/phosaa10.xml",  # Handles THE TPO
        ]
        """List of force field paths for all components except :class:`SmallMoleculeComponent` """

        small_molecule_forcefield: str = "openff-2.0.0"  # other default ideas 'openff-2.0.0', 'gaff-2.11', 'espaloma-0.2.0'
        """Name of the force field to be used for :class:`SmallMoleculeComponent` """

        @validator('constraints')
        def constraint_check(cls, v):
            allowed = {'hbonds', 'hangles', 'allbonds'}

            if not (v is None or v.lower() in allowed):
                raise ValueError(f"Bad constraints value, use one of {allowed}")

            return v


    class Settings(SettingsBaseModel):
        """
        Container for all settings needed by a protocol

        This represents the minimal surface that all settings objects will have.

        Protocols can subclass this to extend this to cater for their additional settings.
        """
        forcefield_settings: BaseForceFieldSettings
        thermo_settings: ThermoSettings

        @classmethod
        def get_defaults(cls):
            return cls(
                forcefield_settings=OpenMMSystemGeneratorFFSettings(),
                thermo_settings=ThermoSettings(temperature=300 * unit.kelvin),
            )
else:
    from gufe.settings import Settings, SettingsBaseModel  # type: ignore


class SystemSettings(SettingsBaseModel):
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


class SolvationSettings(SettingsBaseModel):
    """Settings for solvating the system

    Note
    ----
    No solvation will happen if a SolventComponent is not passed.

    """
    class Config:
        arbitrary_types_allowed = True

    solvent_model = 'tip3p'
    """
    Force field water model to use.
    Allowed values are; `tip3p`, `spce`, `tip4pew`, and `tip5p`.
    """

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


class AlchemicalSettings(SettingsBaseModel):
    """Settings for the alchemical protocol

    This describes the lambda schedule and the creation of the
    hybrid system.
    """
    # Lambda settings
    lambda_functions = 'default'
    """
    Key of which switching functions to use for alchemical mutation.
    Default 'default'.
    """
    lambda_windows = 11
    """Number of lambda windows to calculate. Default 11."""
    unsampled_endstates = False
    """
    Whether to have extra unsampled endstate windows for long range
    correction. Default False.
    """

    # alchemical settings
    use_dispersion_correction = False
    """
    Whether to use dispersion correction in the hybrid topology state.
    Default False.
    """
    softcore_LJ_v2 = True
    """
    Whether to use the LJ softcore function as defined by
    Gapsys et al. JCTC 2012 Default True.
    """
    softcore_electrostatics = True
    """Whether to use softcore electrostatics. Default True."""
    softcore_alpha = 0.85
    """Softcore alpha parameter. Default 0.85"""
    softcore_electrostatics_alpha = 0.3
    """Softcore alpha parameter for electrostatics. Default 0.3"""
    softcore_sigma_Q = 1.0
    """
    Softcore sigma parameter for softcore electrostatics. Default 1.0.
    """
    interpolate_old_and_new_14s = False
    """
    Whether to turn off interactions for new exceptions (not just 1,4s)
    at lambda 0 and old exceptions at lambda 1. If False they are present
    in the nonbonded force. Default False.
    """
    flatten_torsions = False
    """
    Whether to scale torsion terms involving unique atoms, such that at the
    endstate the torsion term is turned off/on depending on the state the
    unique atoms belong to. Default False.
    """


class AlchemicalSamplerSettings(SettingsBaseModel):
    """Settings for the Equilibrium Alchemical sampler, currently supporting
    either MultistateSampler, SAMSSampler or ReplicaExchangeSampler.

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
    online_analysis_interval: Optional[int] = 200
    """
    MCMC steps (i.e. ``IntegratorSettings.n_steps``) interval at which
    to perform an analysis of the free energies.

    At each interval, real time analysis data (e.g. current free energy
    estimate and timing data) will be written to a yaml file named 
    ``<SimulationSettings.output_name>_real_time_analysis.yaml``. The
    current error in the estimate will also be assed and if it drops
    below ``AlchemicalSamplerSettings.online_analysis_target_error``
    the simulation will be terminated.

    If ``None``, no real time analysis will be performed and the yaml
    file will not be written.

    Default `200`.
    """
    online_analysis_target_error = 0.0 * unit.boltzmann_constant * unit.kelvin
    """
    Target error for the online analysis measured in kT. Once the free energy
    is at or below this value, the simulation will be considered complete. A
    suggested value of 0.2 * `unit.boltzmann_constant` * `unit.kelvin` has
    shown to be effective in both hydration and binding free energy benchmarks.
    Default 0.0 * `unit.boltzmann_constant` * `unit.kelvin`, i.e. no early
    termination will occur.
    """
    online_analysis_minimum_iterations = 500
    """
    Number of iterations which must pass before online analysis is
    carried out. Default 500.
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
    n_replicas = 11
    """Number of replicas to use. Default 11."""

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

    timestep = 4 * unit.femtosecond
    """Size of the simulation timestep. Default 4 * unit.femtosecond."""
    collision_rate = 1.0 / unit.picosecond
    """Collision frequency. Default 1.0 / unit.pisecond."""
    n_steps = 250 * unit.timestep
    """
    Number of integration timesteps between each time the MCMC move
    is applied. Default 250 * unit.timestep.
    """
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


class SimulationSettings(SettingsBaseModel):
    """
    Settings for simulation control, including lengths,
    writing to disk, etc...
    """
    class Config:
        arbitrary_types_allowed = True

    minimization_steps = 5000
    """Number of minimization steps to perform. Default 5000."""
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
    output_filename = 'simulation.nc'
    """Path to the storage file for analysis. Default 'simulation.nc'."""
    output_indices = 'all'
    """
    Selection string for which part of the system to write coordinates for.
    Default 'all'.
    """
    checkpoint_interval = 250 * unit.timestep
    """
    Frequency to write the checkpoint file. Default 250 * unit.timestep.
    """
    checkpoint_storage = 'checkpoint.nc'
    """
    Separate filename for the checkpoint file. Note, this should
    not be a full path, just a filename. Default 'checkpoint.nc'.
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


class RelativeHybridTopologyProtocolSettings(Settings):
    class Config:
        arbitrary_types_allowed = True

    # Things for creating the systems
    system_settings: SystemSettings
    solvation_settings: SolvationSettings

    # Alchemical settings
    alchemical_settings: AlchemicalSettings
    alchemical_sampler_settings: AlchemicalSamplerSettings

    # MD Engine things
    engine_settings: OpenMMEngineSettings

    # Sampling State defining things
    integrator_settings: IntegratorSettings

    # Simulation run settings
    simulation_settings: SimulationSettings
