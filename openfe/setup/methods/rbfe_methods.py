# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""Settings for Equilibrium RBFE methods using OpenMM in a Perses-like manner.

This module implements....

"""
from __future__ import annotations

from openfe.setup import LigandAtomMapping, LigandMolecule
from openfe.setup.methods import FEMethod
from typing import Dict, List, Union
from pydantic import BaseModel, validator
from openff.units import unit

# define a timestep
# this isn't convertible to time (e.g ps) and is used to not confuse these two
# definitions of "duration" within a simulation
unit.define('timestep = [timestep] = _ = timesteps')


class SystemSettings(BaseModel):
    """Settings describing the simulation system settings.

    Attributes
    ----------
    nonbonded_method : str
        Which nonbonded electrostatic method to use, currently only PME
        is supported.
    nonbonded_cutoff : float * unit.nanometer
        Cutoff value for short range interactions.
        Default 1.0 * unit.nanometer.
    constraints : str
        Which bonds and angles should be constrained. Default None.
    rigid_water : bool
        Whether to apply rigid constraints to water molecules. Default True.
    hydrogen_mass : float
        How much mass to repartition to hydrogen. Default None, no
        repartitioning will occur.
    """
    class Config:
        arbitrary_types_allowed = True

    nonbonded_method = 'PME'
    nonbonded_cutoff = 1.0 * unit.nanometer
    constraints = Union[str, None] # Usually use HBonds
    rigid_water = True
    remove_com = True # Probably want False here
    hydrogen_mass = Union[float, None]


class TopologySettings(BaseModel):
    """Settings for creating Topologies for each component

    Attributes
    ----------
    forcefield : dictionary of list of strings
      A mapping of each components name to the xml forcefield to apply
    solvent_model : str
      The water model to use. Note, the relevant force field file should
      also be included in ``forcefield``. Default 'tip3p'.

    TODO
    ----
    * We can probably just detect the solvent model from the force field
      defn. In that case we wouldn't have to have ``solvent_model`` here.
    """
    # mapping of component name to forcefield path(s)
    forcefield: Dict[str, Union[List[str, ...], str]]
    solvent_model = 'tip3p'


class AlchemicalSettings(BaseModel):
    """Settings for the alchemical protocol

    This describes the lambda schedule and the creation of the
    hybrid system.

    Attributes
    ----------
    lambda_functions : str, default 'default'
      Key of which switching functions to use for alchemical mutation.
      Default 'default'.
    lambda_windows : int
      Number of lambda windows to calculate. Default 11.
    unsample_endstate : bool
      Whether to have extra unsampled endstate windows for long range
      correction. Default False.
    use_dispersion_correction: bool
      Whether to use dispersion correction in the hybrid topology state. Default False.
    softcore_LJ_v2 : bool
      Whether to use the LJ softcore function as defined by
      Gapsys et al. JCTC 2012 Default True.
    softcore_alpha : float
      Softcore alpha parameter. Default 0.85
    softcore_electrostatics : bool
      Whether to use softcore electrostatics. Default True.
    sofcore_electorstatics_alpha : float
      Softcore alpha parameter for electrostatics. Default 0.3
    softcore_sigma_Q : float
      Softcore sigma parameter for softcore electrostatics. Default 1.0.
    interpolate_old_and_new_14s : bool
      Whether to turn off interactions for new exceptions (not just 1,4s)
      at lambda 0 and old exceptions at lambda 1. If False they are present
      in the nonbonded force. Default False.
    flatten_torsions : bool
      Whether to scale torsion terms involving unique atoms, such that at the
      endstate the torsion term is turned off/on depending on the state the
      unique atoms belong to.
    """
    # Lambda settings
    lambda_functions = 'default'
    lambda_windows = 11
    unsampled_endstates = False

    # alchemical settings
    use_dispersion_correction = False
    softcore_LJ_v2 = True
    softcore_electrostatics = True
    softcore_alpha = 0.85
    softcore_electrostatics_alpha = 0.3
    softcore_sigma_Q = 1.0
    interpolate_old_and_new_14s = False
    flatten_torsions = False


class OpenMMEngineSettings(BaseModel):
    """OpenMM MD engine settings

    Attributes
    ----------
    compute_platform : str
      Which compute platform to perform the simulation on. If 'fastest', the
      fastest compute platform available will be chosen. Default 'fastest'.
     
    TODO
    ----
    * In the future make precision and deterministic forces user defined too.
    """
    compute_platform = 'fastest'


class EquilibriumSamplerSettings(BaseModel):
    pass


class BarostatSettings(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    pressure = 1 * unit.bar
    frequency = 50 * unit.timestep

    @validator('pressure')
    def must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Pressure and temperature must be positive")
        return v

    @validator('pressure')
    def is_pressure(cls, v):
        if not v.is_compatible_with(unit.bar):
            raise ValueError("Must be pressure value, e.g. use unit.bar")
        return v


class IntegratorSettings(BaseModel):
    """Settings for the LangevinSplittingDynamicsMove integrator

    Attributes
    ----------
    timestep : float * unit.femtosecond
      Size of the simulation timestep. Default 2 * unit.femtosecond.
    temperature : float * unit.kelvin
      Target simulation temperature. Default 298.15 * unit.kelvin.
    collision_rate : float / unit.picosecond
      Collision frequency. Default 1 / unit.pisecond.
    n_steps : int * unit.timestep
      Number of integration timesteps each time the MCMC move is applied.
      Default 1000.
    reassign_velocities : bool
      If True, velocities are reassigned from the Maxwell-Boltzmann
      distribution at the beginning of move. Default False.
    splitting : str
      Sequence of "R", "V", "O" substeps to be carried out at each
      timestep. Default "V R O R V".
    n_restart_attempts : int
      Number of attempts to restart from Context if there are NaNs in the
      energies after integration. Default 20.
    constraint_tolerance : float
      Tolerance for the constraint solver. Default 1e-6.
    """
    class Config:
        arbitrary_types_allowed = True

    timestep = 2 * unit.femtosecond
    temperature = 298.15 * unit.kelvin
    collision_rate = 1 / unit.picosecond
    n_steps = 1000 * unit.timestep
    reassign_velocities = True
    splitting = "V R O R V"
    n_restart_attempts = 20
    constraint_tolerance = 1e-06

    @validator('timestep', 'temperature', 'collision_rate', 'n_steps',
               'n_restart_attemps', 'constraint_tolerance')
    def must_be_positive(cls, v):
        if v <= 0:
            errmsg = ("timestep, temperature, collision_rate, n_steps, "
                      "n_restart_atttempts, constraint_tolerance must be "
                      "positive")
            raise ValueError(errmsg)

    @validator('temperature')
    def is_temperature(cls, v):
        if not v.is_compatible_with(unit.kelvin):
            raise ValueError("Must be temperature value, e.g. use unit.kelvin")

    @validator('timestep', 'collision_rate')
    def is_time(cls, v):
        # these are time units, not simulation steps
        if not v.is_compatible_with(unit.picosecond):
            errmsg = "timestep and collision_rate must be in time units"
            raise ValueError(errmsg)
        return v


class SimulationSettings(BaseModel):
    """Settings for simulation control, including lengths, writing to disk,
       etc...

    Attributes
    ----------
    minimization_steps : int
      Number of minimization steps to perform. Default 10000.
    equilibration_length : float * unit.picosecond
      Length of the equilibration phase in units of time.
    production_length : float * unit.picosecond
      Length of the production phase in units of time.
    output_filename : str
      Path to the storage file for analysis. Default 'rbfe.nc'.
    checkpoint_interval : int * unit.timestep
      Frequency to write the checkpoint file. Default 50 * unit.timestep
    checkpoint_storage : str
      Optional separate filename for the checkpoint file. Note, this should
      not be a full path, just a filename. If None, the checkpoint will be
      written to the same file as output_filename. Default None.
    """
    class Config:
        arbitrary_types_allowed = True

    minimization_steps = 10000
    equilibration_length: unit.Quantity
    production_length: unit.Quantity

    # reporter settings
    output_filename = 'rbfe.nc'
    checkpoint_interval = 50 * unit.timestep
    checkpoint_storage = Union[str, None]

    @validator('equilibration', 'production')
    def is_time(cls, v):
        # these are time units, not simulation steps
        if not v.is_compatible_with(unit.picosecond):
            raise ValueError("Durations must be in time units")
        return v

    @validator('minimization', 'equilibration', 'production',
               'checkpoint_interval')
    def must_be_positive(cls, v):
        if v <= 0:
            errmsg = ("Minimization steps, MD lengths, and checkpoint "
                      "intervals must be positive")
            raise ValueError(errmsg)
        return v


class RelativeLigandTransformSettings(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    forcefield_settings: ForcefieldSettings
    state_point: StatePoint

    lambda_protocol: LambdaProtocolSettings

    # solvent model?
    solvent_padding = 1.2 * unit.nanometer

    barostat: MonteCarloBarostatSettings
    integrator: MCMCLangevinSplittingDynamicsMoveSettings

    hybrid_topology_factory_settings: HybridTopologyFactorySettings
    simulation_length: SimulationLengthSettings


class RelativeLigandTransformResults:
    """Dict-like container for the output of a LigandLigandTransform"""
    pass


class RelativeLigandTransform(FEMethod):
    """Calculates the free energy of an alchemical ligand swap in solvent

    """
    _SETTINGS_CLASS = LigandLigandTransformSettings

    def __init__(self,
                 ligandA: LigandMolecule,
                 ligandB: LigandMolecule,
                 ligandmapping: LigandAtomMapping,
                 settings: Union[Dict, LigandLigandTransformSettings] = None,
                 ):
        """
        Parameters
        ----------
        ligandA, ligandB : LigandMolecule
          the two ligand LigandMolecules to transform between.  The
          transformation will go from ligandA to ligandB.
        ligandmapping : AtomMapping
          the mapping of atoms between the
        settings : dict
          the settings for the Method.

        The default settings for this method can be accessed via the
        get_default_settings method,
        """
        self._ligandA = ligandA
        self._ligandB = ligandB
        self._mapping = ligandmapping
        self._settings = self.__class__.get_default_settings()
        if settings is not None:
            self._settings.update(settings)
        # TODO: Prepare the workload

    def to_xml(self) -> str:
        raise NotImplementedError()

    @classmethod
    def from_xml(cls, xml: str):
        raise NotImplementedError()

    def run(self) -> bool:
        if self.is_complete():
            return True
        # TODO: Execute the workload
        return False

    def is_complete(self) -> bool:
        return False

    def get_results(self) -> LigandLigandTransformResults:
        """Return payload created by this workload

        Raises
        ------
        ValueError
          if the results do not exist yet
        """
        if not self.is_complete():
            raise ValueError("Results have not been generated")
        return LigandLigandTransformResults()
