# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""Equilibrium RBFE methods using OpenMM in a Perses-like manner.

This module implements the necessary methodology toolking to run calculate a
ligand relative free energy transformation using OpenMM tools and one of the
following methods:
    - Hamiltonian Replica Exchange
    - Self-adjusted mixture sampling
    - Independent window sampling

TODO
----
* Improve this docstring by adding an example use case.

"""
from __future__ import annotations

import os
import logging

from collections import defaultdict
import gufe
from gufe import settings
import json
import numpy as np
import openmm
from openff.units import unit
from openff.units.openmm import to_openmm, ensure_quantity
from openmmtools import multistate
from pydantic import validator
from typing import Dict, List, Union, Optional
from openmm import app
from openmm import unit as omm_unit
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
import pathlib
from typing import Any, Iterable
import openmmtools
import uuid

from gufe import (
    ChemicalSystem, LigandAtomMapping,
)

from . import _rbfe_utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SystemSettings(settings.SettingsBaseModel):
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
    constraints: Union[str, None] = 'HBonds'  # Usually use HBonds
    rigid_water = True
    remove_com = True  # Probably want False here
    hydrogen_mass: Union[float, None] = None


class TopologySettings(settings.SettingsBaseModel):
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
    forcefield: Dict[str, Union[List[str], str]]
    solvent_model = 'tip3p'


class AlchemicalSettings(settings.SettingsBaseModel):
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
      Whether to use dispersion correction in the hybrid topology state.
      Default False.
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
    atom_overlap_tolerance : float
      Maximum allowed deviation along any dimension (x,y,z) in mapped atoms
      between the positions of state A and B. Default 0.5.
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
    atom_overlap_tolerance = 0.5


class OpenMMEngineSettings(settings.SettingsBaseModel):
    """OpenMM MD engine settings

    Attributes
    ----------
    compute_platform : str, optional
      Which compute platform to perform the simulation on. If None, the
      fastest compute platform available will be chosen. Default None.

    TODO
    ----
    * In the future make precision and deterministic forces user defined too.
    """
    compute_platform: Optional[str] = None


class SamplerSettings(settings.SettingsBaseModel):
    """Settings for the Equilibrium sampler, currently supporting either
    HybridSAMSSampler or HybridRepexSampler.

    Attributes
    ----------
    sampler_method : str
      Sampler method to use, currently supports:
          - repex (hamiltonian replica exchange)
          - sams (self-adjusted mixture sampling)
          - independent (independent lambda sampling)
      Default repex.
    online_analysis_interval : int
      The interval at which to perform online analysis of the free energy.
      At each interval the free energy is estimate and the simulation is
      considered complete if the free energy estimate is below
      ``online_analysis_target_error``. Default `None`.
    online_analysis_target_error : float * unit.boltzmann_constant * unit.kelvin
      Target error for the online analysis measured in kT.
      Once the free energy is at or below this value, the simulation will be
      considered complete.
    online_analysis_minimum_iterations : float
      Set number of iterations which must pass before online analysis is
      carried out. Default 50.
    n_repeats : int
      number of independent repeats to run.  Default 3
    flatness_criteria : str
      SAMS only. Method for assessing when to switch to asymptomatically
      optimal scheme.
      One of ['logZ-flatness', 'minimum-visits', 'histogram-flatness'].
      Default 'logZ-flatness'.
    gamma0 : float
      SAMS only. Initial weight adaptation rate. Default 1.0.
    n_replicas : int
      Number of replicas to use. Default 11.

    TODO
    ----
    * Work out how this fits within the context of independent window FEPs.
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
    online_analysis_interval: Optional[int] = None
    online_analysis_target_error = 0.2 * unit.boltzmann_constant * unit.kelvin
    online_analysis_minimum_iterations = 50
    n_repeats: int = 3
    flatness_criteria = 'logZ-flatness'
    gamma0 = 1.0
    n_replicas = 11

    @validator('online_analysis_target_error',
               'online_analysis_minimum_iterations', 'gamma0')
    def must_be_positive(cls, v):
        if v < 0:
            errmsg = ("Online analysis target error, minimum iteration "
                      "and SAMS gamm0 must be 0 or positive values")
            raise ValueError(errmsg)
        return v


class BarostatSettings(settings.SettingsBaseModel):
    """Settings for the OpenMM Monte Carlo barostat series

    Attributes
    ----------
    frequency : int * unit.timestep
      Frequency at which volume scaling changes should be attempted.
      Default 25 * unit.timestep.

    Notes
    -----
    * The temperature is defined under IntegratorSettings

    TODO
    ----
    * Add support for anisotropic and membrane barostats.
    """
    class Config:
        arbitrary_types_allowed = True

    frequency = 25 * unit.timestep


class IntegratorSettings(settings.SettingsBaseModel):
    """Settings for the LangevinSplittingDynamicsMove integrator

    Attributes
    ----------
    timestep : float * unit.femtosecond
      Size of the simulation timestep. Default 2 * unit.femtosecond.
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
    collision_rate = 1.0 / unit.picosecond
    n_steps = 1000 * unit.timestep
    reassign_velocities = True
    splitting = "V R O R V"
    n_restart_attempts = 20
    constraint_tolerance = 1e-06

    @validator('timestep', 'collision_rate', 'n_steps',
               'n_restart_attempts', 'constraint_tolerance')
    def must_be_positive(cls, v):
        if v <= 0:
            errmsg = ("timestep, temperature, collision_rate, n_steps, "
                      "n_restart_atttempts, constraint_tolerance must be "
                      "positive")
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
    """Settings for simulation control, including lengths, writing to disk,
       etc...

    Attributes
    ----------
    minimization_steps : int
      Number of minimization steps to perform. Default 10000.
    equilibration_length : float * unit.picosecond
      Length of the equilibration phase in units of time. The total number of
      steps from this equilibration length (i.e.
      ``equilibration_length`` / :class:`IntegratorSettings.timestep`) must be
      a multiple of the value defined for :class:`IntegratorSettings.n_steps`.
    production_length : float * unit.picosecond
      Length of the production phase in units of time. The total number of
      steps from this production length (i.e.
      ``production_length`` / :class:`IntegratorSettings.timestep`) must be
      a multiple of the value defined for :class:`IntegratorSettings.nsteps`.
    output_filename : str
      Path to the storage file for analysis. Default 'rbfe.nc'.
    output_indices : str
      Selection string for which part of the system to write coordinates for.
      Default 'all'.
    checkpoint_interval : int * unit.timestep
      Frequency to write the checkpoint file. Default 50 * unit.timestep
    checkpoint_storage : str
      Separate filename for the checkpoint file. Note, this should
      not be a full path, just a filename. Default 'rbfe_checkpoint.nc'
    """
    class Config:
        arbitrary_types_allowed = True

    minimization_steps = 10000
    equilibration_length: unit.Quantity
    production_length: unit.Quantity

    # reporter settings
    output_filename = 'rbfe.nc'
    output_indices = 'all'
    checkpoint_interval = 50 * unit.timestep
    checkpoint_storage = 'rbfe_checkpoint.nc'

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


class RelativeLigandProtocolSettings(settings.Settings):
    class Config:
        arbitrary_types_allowed = True

    # Things for creating the systems
    system_settings: SystemSettings
    topology_settings: TopologySettings

    # Alchemical settings
    alchemical_settings: AlchemicalSettings

    # MD Engine things
    engine_settings = OpenMMEngineSettings()

    # Sampling State defining things
    integrator_settings: IntegratorSettings
    barostat_settings: BarostatSettings
    sampler_settings: SamplerSettings

    # Simulation run settings
    simulation_settings: SimulationSettings

    # solvent model?
    solvent_padding = 1.2 * unit.nanometer
