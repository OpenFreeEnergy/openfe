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
import os

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


class SystemSettings(SettingsBaseModel):
    """Settings describing the simulation system settings."""

    class Config:
        arbitrary_types_allowed = True

    nonbonded_method = 'PME'
    """
    Method for treating nonbonded interactions, currently only PME and
    NoCutoff are allowed. Default PME.
    """
    nonbonded_cutoff: FloatQuantity['nanometer'] = 1.0 * unit.nanometer
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


class BaseSolvationSettings(SettingsBaseModel):
    """
<<<<<<< HEAD
    Base class for SolvationSettings objects.
    """
    class Config:
        arbitrary_types_allowed = True


class OpenMMSolvationSettings(BaseSolvationSettings):
    """Settings for controlling how a system is solvated using OpenMM tooling.

    Defining the number of waters
    -----------------------------

    The number of waters is controlled by either:
      a) defining a solvent padding (``solvent_padding``) in combination
         with a box shape
      b) defining the number of solvent molecules
         (``number_of_solvent_molecules``)
         alongside the box shape (``box_shape``)
      c) defining the box directly either through the box vectors
         (``box_vectors``) or rectangular box lengths (``box_size``)

    When using ``solvent_padding``, ``box_vectors``, or ``box_size``,
    the exact number of waters added is determined automatically by OpenMM
    through :meth:`openmm.app.Modeller.addSolvent` internal heuristics.
    Briefly, the necessary volume required by a single water is estimated
    and then the defined target cell is packed with waters avoiding clashes
    with existing solutes and box edges.


    Defining the periodic cell size
    -------------------------------

    The periodic cell size is defined by one, and only one, of the following:
      * ``solvent_padding`` in combination with ``box_shape``,
      * ``number_of_solvent_molecules`` in combination with ``box_shape``,
      * ``box_vectors``,
      * ``box_size``

    When using ``number_of_solvent_molecules``, the size of the cell is
    defined by :meth:`openmm.app.Modeller.addSolvent` internal heuristics,
    automatically selecting a padding value that is large enough to contain
    the number of waters based on a geometric estimate of the volume required
    by each water molecule.


    Defining the pertiodic cell shape
    ---------------------------------

    The periodic cell shape is defined by one, and only one, of the following:
      * ``box_shape``,
      * ``box_vectors``,
      * ``box_size``

    Default settings will create a cubic box, although more space efficient
    shapes (e.g. ``dodecahedrons``) are recommended to improve simulation
    performance.


    Notes
    -----
    * The number of water molecules added will be affected by the number of
      ions defined in SolventComponent. For example, the value of
      ``number_of_solvent_molecules`` is the sum of the number of counterions
      added and the number of water molecules added.
    * Solvent addition does not account for any pre-existing waters explicitly
      defined in the :class:`openfe.ChemicalSystem`. Any waters will be added
      in addition to those pre-existing waters.
    * No solvation will happen if a SolventComponent is not passed.


    See Also
    --------
    :mod:`openmm.app.Modeller`
=======
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
    Force field water model to use when solvating and defining the model
    properties (e.g. adding virtual site particles).

    Allowed values are; `tip3p`, `spce`, `tip4pew`, and `tip5p`.
    """
    solvent_padding: Optional[FloatQuantity['nanometer'] = 1.2 * unit.nanometer]
    """
    Minimum distance from any solute bounding sphere to the edge of the box.

    Note
    ----
    * Cannot be defined alongside ``number_of_solvent_molecules``,
      ``box_size``, or ``box_vectors``.
    """
    box_shape: Literal['cube', 'dodecahedron', 'octahedron'] = 'cube'
    """
    The shape of the periodic box to create.

    Notes
    -----
    * Must be one of `cube`, `dodecahedron`, or `octahedron`.
    * Cannot be defined alongside ``box_vectors`` or ``box_size``.
    """
    number_of_solvent_molecules: Optional[int] = None
    """
    The number of solvent molecules (water + ions) to add.

    Note
    ----
    * Cannot be defined alongside ``solvent_padding``, ``box_size``,
      or ``box_vectors``.
    """
    box_vectors: Optional[ArrayQuantity['nanometer']] = None
    """
    `OpenMM reduced form box vectors <http://docs.openmm.org/latest/userguide/theory/05_other_features.html#periodic-boundary-conditions>`.

    Notes
    -----
    * Cannot be defined alongside ``solvent_padding``,
      ``number_of_solvent_molecules``, or ``box_size``.

    See Also
    --------
    :mod:`openff.interchange.components.interchange`
    :mod:`openff.interchange.components._packmol`
    """
    box_size: Optional[FloatQuantity['nanometer']] = None
    """
    X, Y, and Z lengths of the unit cell for a rectangular box.

    Notes
    -----
    * Cannot be defined alongside ``solvent_padding``,
      ``number_of_solvent_molecules``, or ``box_vectors``.
    """

    @validator('box_vectors')
    def supported_vectors(cls, v):
        if v is not None:
             from openff.interchange.components._packmol import _box_vectors_are_in_reduced_form
        if v.lower() not in supported:
            errmsg = ("Only the following sampler_method values are "
                      f"supported: {supported}")
            raise ValueError(errmsg)
        return v


=======
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
>>>>>>> solvation-prep


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
    online_analysis_interval: Optional[int] = 250
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

    Must be a multiple of ``SimulationSettings.checkpoint_interval``

    Default `250`.
    """
    online_analysis_target_error: FloatQuantity = 0.0 * unit.boltzmann_constant * unit.kelvin
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

    timestep: FloatQuantity['femtosecond'] = 4 * unit.femtosecond
    """Size of the simulation timestep. Default 4 * unit.femtosecond."""
    collision_rate: FloatQuantity['1/picosecond'] = 1.0 / unit.picosecond
    """Collision frequency. Default 1.0 / unit.pisecond."""
    n_steps = 250 * unit.timestep  # todo: IntQuantity
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
    barostat_frequency = 25 * unit.timestep  # todo: IntQuantity
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
    equilibration_length: FloatQuantity['nanosecond']
    """
    Length of the equilibration phase in units of time. The total number of
    steps from this equilibration length
    (i.e. ``equilibration_length`` / :class:`IntegratorSettings.timestep`)
    must be a multiple of the value defined for
    :class:`IntegratorSettings.n_steps`.
    """
    production_length: FloatQuantity['nanosecond']
    """
    Length of the production phase in units of time. The total number of
    steps from this production length (i.e.
    ``production_length`` / :class:`IntegratorSettings.timestep`) must be
    a multiple of the value defined for :class:`IntegratorSettings.nsteps`.
    """

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
    checkpoint_interval = 250 * unit.timestep  # todo: Needs IntQuantity
    """
    Frequency to write the checkpoint file. Default 250 * unit.timestep.
    """
    checkpoint_storage = 'checkpoint.nc'
    """
    Separate filename for the checkpoint file. Note, this should
    not be a full path, just a filename. Default 'checkpoint.nc'.
    """
    forcefield_cache: Optional[str] = 'db.json'
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


class SimulationSettingsMD(SimulationSettings):
    """
    Settings for simulation control for plain MD simulations, including
    writing outputs
    """
    class Config:
        arbitrary_types_allowed = True

    equilibration_length_nvt: unit.Quantity
    """
    Length of the equilibration phase in the NVT ensemble in units of time. 
    The total number of steps from this equilibration length
    (i.e. ``equilibration_length_nvt`` / :class:`IntegratorSettings.timestep`)
    must be a multiple of the value defined for
    :class:`IntegratorSettings.n_steps`.
    """
    # reporter settings
    production_trajectory_filename = 'simulation.xtc'
    """Path to the storage file for analysis. Default 'simulation.xtc'."""
    trajectory_write_interval = 5000 * unit.timestep
    """
    Frequency to write the xtc file. Default 5000 * unit.timestep.
    """
    preminimized_structure = 'system.pdb'
    """Path to the pdb file of the full pre-minimized system. Default 'system.pdb'."""
    minimized_structure = 'minimized.pdb'
    """Path to the pdb file of the system after minimization. 
    Only the specified atom subset is saved. Default 'minimized.pdb'."""
    equil_NVT_structure = 'equil_NVT.pdb'
    """Path to the pdb file of the system after NVT equilibration. 
    Only the specified atom subset is saved. Default 'equil_NVT.pdb'."""
    equil_NPT_structure = 'equil_NPT.pdb'
    """Path to the pdb file of the system after NPT equilibration. 
    Only the specified atom subset is saved. Default 'equil_NPT.pdb'."""
    checkpoint_storage_filename = 'checkpoint.chk'
    """
    Separate filename for the checkpoint file. Note, this should
    not be a full path, just a filename. Default 'checkpoint.chk'.
    """
    log_output = 'simulation.log'
    """
    Filename for writing the log of the MD simulation, including timesteps,
    energies, density, etc.
    """
