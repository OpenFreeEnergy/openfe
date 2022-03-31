# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""Contains RBFE methods

This module implements.....

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


class ForcefieldSettings(BaseModel):
    """Settings describing the forcefield to use for each component

    Attributes
    ----------
    forcefield
      a mapping of each components name to the xml forcefield to apply
    nonbonded_electrostatics
      which nonbonded electrostatics method to use, currently only 'PME'
      allowed
    nonbonded_cutoff
      range of nonbonded forces
    constraints
      which bonds to apply constraints to
    rigid_water
      whether apply rigid constraints to water molecules, default True
    """
    class Config:
        arbitrary_types_allowed = True

    # mapping of component name to forcefield path(s)
    forcefield: Dict[str, Union[List[str, ...], str]]
    nonbonded_electostatics = 'PME'
    nonbonded_cutoff = 0.9 * unit.nanometer
    constraints = 'HBonds'
    rigid_water = True


class StatePoint(BaseModel):
    """Description of the temperature and pressure to model"""
    class Config:
        arbitrary_types_allowed = True

    temperature = 298.15 * unit.kelvin
    pressure = 1 * unit.bar

    @validator('pressure', 'temperature')
    def must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Pressure and temperature must be positive")
        return v

    @validator('pressure')
    def is_pressure(cls, v):
        if not v.is_compatible_with(unit.bar):
            raise ValueError("Must be pressure value, e.g. use unit.bar")
        return v

    @validator('temperature')
    def is_temperature(cls, v):
        if not v.is_compatible_with(unit.kelvin):
            raise ValueError("Must be temperature value, e.g. use unit.kelvin")


class LambdaProtocolSettings(BaseModel):
    """Settings for the lambda protocol

    This describes a fixed number of windows

    Attributes
    ----------
    functions : str, default 'default'
      key of which switching functions to use for alchemical mutation
    windows : int, default 11
      number of lambda windows to calculate
    sample_endstate : bool, default False
      whether to sample the endstates ???
    """
    functions = 'default'
    windows = 11
    sample_endstates = False


class MonteCarloBarostatSettings(BaseModel):
    """Settings for the OpenMM Monte-Carlo Barostat

    The temperature and pressure value is taken from the StatePoint variable

    Attributes
    ----------
    frequency : unit.timestep
      the number of timesteps between attempts of changing the pressure
    """
    class Config:
        extra = 'forbid'
        arbitrary_types_allowed = True

    frequency = 50 * unit.timestep


class MCMCLangevinSplittingDynamicsMoveSettings(BaseModel):
    """Settings for the integrator

    Attributes
    ----------
    timestep
      size of timestep
    collision_rate

    n_steps

    reassign_velocities

    n_restart_attempts

    constraint_tolerance
    """
    class Config:
        arbitrary_types_allowed = True

    timestep = 0.02 * unit.femtosecond
    collision_rate = 1 / unit.picosecond
    n_steps = 2500 * unit.timestep
    reassign_velocities = True
    n_restart_attempts = 20
    constraint_tolerance = 1e-06


class HybridTopologyFactorySettings(BaseModel):
    use_dispersion_correction = False
    softcore_alpha = 0.5
    softcore_LJ_v2 = True
    softcore_LJ_v2_alpha = 0.85
    softcore_electrostatics = True
    softcore_electrostatics_alpha = 0.3
    softcore_sigma_Q = 1.0
    interpolate_old_and_new_14s = False
    flatten_torsions = False


class SimulationLengthSettings(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    minimization = 1000
    equilibration = 5 * unit.picosecond
    production: unit.Quantity

    # reporter settings
    output_filename = 'rbfe.nc'
    checkpoint_interval = 10 * unit.timestep

    @validator('equilibration', 'production')
    def is_time(cls, v):
        # these are time units, not simulation steps
        if not v.is_compatible_with(unit.picosecond):
            raise ValueError("Durations must be in time units")
        return v


class LigandLigandTransformSettings(BaseModel):
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


class LigandLigandTransformResults:
    """Dict-like container for the output of a LigandLigandTransform"""
    pass


class LigandLigandTransform(FEMethod):
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
