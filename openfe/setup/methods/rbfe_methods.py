# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""Contains RBFE methods

This module implements.....

"""
from __future__ import annotations

from openfe.setup import LigandAtomMapping, LigandMolecule
from openfe.setup.methods import FEMethod
from typing import Dict, Union
from pydantic import BaseModel, validator
from openff.units import unit


class ForcefieldSettings(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    # mapping of component name to forcefield path
    forcefield: Dict[str, str]
    nonbonded_electostatics = 'PME'
    nonbonded_cutoff = 0.9 * unit.nanometer
    constraints = 'HBonds'
    rigid_water = True


class StatePoint(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    temperature = 298.15 * unit.kelvin
    pressure = 1 * unit.bar

    # TODO: Validate that the units are correct (i.e. a pressure unit for pressure)
    @validator('pressure', 'temperature')
    def must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Must be positive")
        return v


class LambdaProtocol(BaseModel):
    functions = 'default'
    windows = 11
    sample_endstates = False


class MonteCarloBarostat(BaseModel):
    class Config:
        extra = 'forbid'

    frequency = 50 * unit.timestep


class MCMCLangevinSplittingDynamicsMove(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    timestep = 0.02 * unit.femtosecond
    collision_rate = 1 / unit.picosecond
    n_steps = 2500
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


class ReporterSettings(BaseModel):
    output_filename = 'rbfe.nc'
    checkpoint_interval = 10


class SimulationLength(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    minimization = 1000
    equilibration = 5 * unit.picosecond
    production: unit.Quantity


class LigandLigandTransformSettings(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    forcefield_settings: ForcefieldSettings
    state_point: StatePoint

    lambda_protocol: LambdaProtocol

    # solvent model?
    solvent_padding = 1.2 * unit.nanometer

    barostat: MonteCarloBarostat
    integrator: MCMCLangevinSplittingDynamicsMove

    hybrid_topology_factory_settings: HybridTopologyFactorySettings
    reporter_settings: ReporterSettings
    simulation_length: SimulationLength


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
