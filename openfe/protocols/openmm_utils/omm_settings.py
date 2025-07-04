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
from gufe.vendor.openff.models.types import FloatQuantity, ArrayQuantity
from openff.interchange.components._packmol import _box_vectors_are_in_reduced_form

from gufe.settings import (
    Settings,
    SettingsBaseModel,
    OpenMMSystemGeneratorFFSettings,
    ThermoSettings,
)


from pydantic.v1 import validator


class BaseSolvationSettings(SettingsBaseModel):
    """
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


    Defining the periodic cell shape
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
    Base class for SolvationSettings objects
    """
    solvent_model: Literal['tip3p', 'spce', 'tip4pew', 'tip5p'] = 'tip3p'
    """
    Force field water model to use when solvating and defining the model
    properties (e.g. adding virtual site particles).

    Allowed values are; `tip3p`, `spce`, `tip4pew`, and `tip5p`.
    """
    solvent_padding: Optional[FloatQuantity['nanometer']] = 1.2 * unit.nanometer
    """
    Minimum distance from any solute bounding sphere to the edge of the box.

    Note
    ----
    * Cannot be defined alongside ``number_of_solvent_molecules``,
      ``box_size``, or ``box_vectors``.
    """
    box_shape: Optional[Literal['cube', 'dodecahedron', 'octahedron']] = 'cube'
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
    box_size: Optional[ArrayQuantity['nanometer']] = None
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
            if not _box_vectors_are_in_reduced_form(v):
                errmsg = f"box_vectors: {v} are not in OpenMM reduced form"
                raise ValueError(errmsg)
        return v

    @validator('solvent_padding')
    def is_positive_distance(cls, v):
        # these are time units, not simulation steps
        if v is None:
            return v

        if not v.is_compatible_with(unit.nanometer):
            raise ValueError("solvent_padding must be in distance units "
                             "(i.e. nanometers)")
        if v < 0:
            errmsg = "solvent_padding must be a positive value"
            raise ValueError(errmsg)

        return v

    @validator('number_of_solvent_molecules')
    def positive_solvent_number(cls, v):
        if v is None:
            return v

        if v <= 0:
            errmsg = f"number_of_solvent molecules: {v} must be positive"
            raise ValueError(errmsg)

        return v

    @validator('box_size')
    def box_size_properties(cls, v):
        if v is None:
            return v

        if v.shape != (3, ):
            errmsg = (f"box_size must be a 1-D array of length 3 "
                      f"got {v} with shape {v.shape}")
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
    Settings for controlling partial charge assignment using the OpenFF tooling
    """
    partial_charge_method: Literal['am1bcc', 'am1bccelf10', 'nagl', 'espaloma'] = 'am1bcc'
    """
    Selection of method for partial charge generation.

    Description of options
    ----------------------
    ``am1bcc``:
      Generate partial charges using the AM1-BCC approach, as detailed
      by Araz Jalkalian et al. J. Comp. Chem. 2000.
      AM1-BCC charges are either assigned using AmberTools (via SQM)
      if ``off_toolkit_backend`` is set to ``ambertools`, or
      using the OpenEye Toolkit (via Quacpac) if ``off_toolkit_backend``
      is set to ``openeye``. A maximum of one conformer is allowed.

    ``am1bccelf10``:
      Assign AM1-BCC partialk charges using the `ELF10 method
      <https://docs.eyesopen.com/toolkits/python/quacpactk/molchargetheory.html#elf-conformer-selection>`_
      This is only currently possible via the OpenEye toolkit
      if setting ``off_toolkit_backend`` to ``openeye``.
      We recommend setting ``number_of_conformers`` to at least `500`.

    ``nagl``:
      Assign partial charges using the `OpenFF NAGL ML method
      <https://github.com/openforcefield/openff-nagl>`_
      All ``off_toolkit_backend`` options are supported.
      A maximum of one conformer is allowed.

    ``espaloma``:
      Assign partial charges using the `Espaloma Charge method
      <https://github.com/choderalab/espaloma_charge>`_
      Only ``ambertools`` and ``rdkit`` `off_toolkit_backend`` options
      are supported. A maximum of one conformer is allowed.

    """
    off_toolkit_backend: Literal['ambertools', 'openeye', 'rdkit'] = 'ambertools'
    """
    The OpenFF toolkit registry backend to use for partial charge generation.


    OpenFF backend selection options
    --------------------------------

    ``ambertools``:
      This limits partial charge generation to using a mixture of AmberTools
      and RDKit.

    ``openeye``:
      This limits partial charge generation to using the OpenEye toolkit.
      This cannot be used with ``espaloma`` as the ``partial_charge_method``

    ``rdkit``:
      This limits partial charge generation to using the RDKit toolkit.
      Note that this alone cannot be used for conventional am1bcc partial
      charge generation, but is usually used in combination with
      the ``nagl`` or ``espaloma`` ``partial_charge_method`` selections.

    """
    number_of_conformers: Optional[int] = None
    """
    Number of conformers to generate as part of the partial charge assignment.

    If ``None`` (default), the existing conformer of the input
    SmallMoleculeComponent will be used.

    Values greater than 1 or ``None`` are only allowed when calculating
    partial charges through ``am1bccelf10``. See ``partial_charge_method``'s
    ``Description of options`` documentation.
    """
    nagl_model: Optional[str] = None
    """
    The `NAGL <https://github.com/openforcefield/openff-nagl>`_ model to use
    for partial charge assignment.

    If ``None`` (default) and ``partial_charge_method`` is set to ``nagl``,
    the latest available production am1bcc charge model will be used.
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
    OpenMM compute platform to perform MD integration with. If ``None``, will
    choose fastest available platform. Default ``None``.
    """
    gpu_device_index: Optional[list[int]] = None
    """
    List of integer indices for the GPU device to select when
    ``compute_platform`` is either set to ``CUDA`` or ``OpenCL``.

    If ``None``, the default OpenMM GPU selection behaviour is used.

    See the `OpenMM platform properties documentation <http://docs.openmm.org/latest/userguide/library/04_platform_specifics.html>`_
    for more details.

    Default ``None``.
    """


class IntegratorSettings(SettingsBaseModel):
    """Settings for the LangevinDynamicsMove integrator

    Note
    ----
    For some Protocols, an MC "move" (e.g. replica exchange swap) is applied
    at a given frequency. In most Protocols the move frequency is defined in
    `MultiStateSimulationSettings.time_per_iteration`.
    """

    class Config:
        arbitrary_types_allowed = True

    timestep: FloatQuantity['femtosecond'] = 4.0 * unit.femtosecond
    """Size of the simulation timestep. Default 4.0 * unit.femtosecond."""
    langevin_collision_rate: FloatQuantity['1/picosecond'] = 1.0 / unit.picosecond
    """Collision frequency. Default 1.0 / unit.pisecond."""
    reassign_velocities = False
    """
    If ``True``, velocities are reassigned from the Maxwell-Boltzmann
    distribution at the beginning of each MC move. Default ``False``.
    """
    n_restart_attempts = 20
    """
    Number of attempts to restart from Context if there are NaNs in the
    energies after integration. Default 20.
    """
    constraint_tolerance = 1e-06
    """Tolerance for the constraint solver. Default 1e-6."""
    barostat_frequency: FloatQuantity['timestep'] = 25.0 * unit.timestep  # todo: IntQuantity
    """
    Frequency at which volume scaling changes should be attempted.
    Note: The barostat frequency is ignored for gas-phase simulations.
    Default 25 * unit.timestep.
    """
    remove_com: bool = False
    """
    Whether or not to remove the center of mass motion. Default ``False``.
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
    output_indices = 'not water'
    """
    Selection string for which part of the system to write coordinates for.
    Default 'not water'.
    """
    checkpoint_interval: FloatQuantity['nanosecond'] = 1.0 * unit.nanosecond
    """
    Frequency to write the checkpoint file. Default 1 * unit.nanosecond.
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


class MultiStateOutputSettings(OutputSettings):
    """
    Settings for MultiState simulation output settings,
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
    positions_write_frequency: Optional[FloatQuantity['picosecond']] = 100.0 * unit.picosecond
    """
    Frequency at which positions are written to the simulation trajectory
    storage file (defined by ``output_filename``).

    If ``None``, no positions will be written to the trajectory.

    Unless set to ``None``, must be divisible by
    ``MultiStateSimulationSettings.time_per_iteration``.
    """
    velocities_write_frequency: Optional[FloatQuantity['picosecond']] = None
    """
    Frequency at which velocities are written to the simulation
    trajectory storage file (defined by ``output_filename``).

    If ``None`` (default), no velocities will be written to the trajectory.

    Unless set to ``None``, must be divisible by
    ``MultiStateSimulationSettings.time_per_iteration``.
    """


    @validator('positions_write_frequency', 'velocities_write_frequency')
    def must_be_positive(cls, v):
        if v is not None and v < 0:
            errmsg = ("Position_write_frequency and velocities_write_frequency"
                      f" must be positive (or None), got {v}.")
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
    Length of the equilibration phase in units of time.
    Must be divisible by the :class:`IntegratorSettings.timestep`.
    """
    production_length: FloatQuantity['nanosecond']
    """
    Length of the production phase in units of time.
    Must be divisible by the :class:`IntegratorSettings.timestep`.
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
    * Make n_replicas optional such that: If ``None`` or greater than the
      number of lambda windows set in :class:`LambdaSettings`, this will
      default to the number of lambda windows. If less than the number of
      lambda windows, the replica lambda states will be picked at equidistant
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
    time_per_iteration: FloatQuantity['picosecond'] = 1.0 * unit.picosecond
    # todo: Add validators in the protocol
    """
    Simulation time between each MCMC move attempt. Default 1 * unit.picosecond.
    """
    real_time_analysis_interval: Optional[FloatQuantity['picosecond']] = 250.0 * unit.picosecond
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

    Default `250`.

    """
    early_termination_target_error: Optional[FloatQuantity['kilocalorie_per_mole']] = 0.0 * unit.kilocalorie_per_mole
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
    real_time_analysis_minimum_time: FloatQuantity['picosecond'] = 500.0 * unit.picosecond
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

    equilibration_length_nvt: Optional[FloatQuantity['nanosecond']]
    """
    Length of the equilibration phase in the NVT ensemble in units of time.
    The total number of steps from this equilibration length
    (i.e. ``equilibration_length_nvt`` / :class:`IntegratorSettings.timestep`).
    If None, no NVT equilibration will be performed.
    """


class MDOutputSettings(OutputSettings):
    """ Settings for simulation output settings for plain MD simulations."""
    class Config:
        arbitrary_types_allowed = True

    # reporter settings
    production_trajectory_filename: Optional[str] = 'simulation.xtc'
    """Path to the storage file for analysis. Default 'simulation.xtc'."""
    trajectory_write_interval: FloatQuantity['picosecond'] = 20.0 * unit.picosecond
    """
    Frequency to write the xtc file. Default 5000 * unit.timestep.
    """
    preminimized_structure: Optional[str] = 'system.pdb'
    """Path to the pdb file of the full pre-minimized system.
    Default 'system.pdb'."""
    minimized_structure: Optional[str] = 'minimized.pdb'
    """Path to the pdb file of the system after minimization.
    Only the specified atom subset is saved. Default 'minimized.pdb'."""
    equil_nvt_structure: Optional[str] = 'equil_nvt.pdb'
    """Path to the pdb file of the system after NVT equilibration.
    Only the specified atom subset is saved. Default 'equil_nvt.pdb'."""
    equil_npt_structure: Optional[str] = 'equil_npt.pdb'
    """Path to the pdb file of the system after NPT equilibration.
    Only the specified atom subset is saved. Default 'equil_npt.pdb'."""
    log_output: Optional[str] = 'simulation.log'
    """
    Filename for writing the log of the MD simulation, including timesteps,
    energies, density, etc.
    """
