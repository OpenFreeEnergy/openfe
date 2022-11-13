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
import json
import numpy as np
import openmm
from openff.units import unit
from openff.units.openmm import to_openmm, ensure_quantity
from openmmtools import multistate
from pydantic import BaseModel, validator
from typing import Dict, List, Union, Optional
from openmm import app
from openmm import unit as omm_unit
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
import pathlib
from typing import Any, Iterable
import openmmtools
import uuid

from openfe.setup import (
    ChemicalSystem, LigandAtomMapping,
)

from . import _rbfe_utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    constraints: Union[str, None] = 'HBonds'  # Usually use HBonds
    rigid_water = True
    remove_com = True  # Probably want False here
    hydrogen_mass: Union[float, None] = None


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
    forcefield: Dict[str, Union[List[str], str]]
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


class OpenMMEngineSettings(BaseModel):
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


class SamplerSettings(BaseModel):
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


class BarostatSettings(BaseModel):
    """Settings for the OpenMM Monte Carlo barostat series

    Attributes
    ----------
    pressure : float * unit.bar
      Target pressure acting on the system. Default 1 * unit.bar.
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

    pressure = 1 * unit.bar
    frequency = 25 * unit.timestep

    @validator('pressure')
    def must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Pressure must be positive")
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
               'n_restart_attempts', 'constraint_tolerance')
    def must_be_positive(cls, v):
        if v <= 0:
            errmsg = ("timestep, temperature, collision_rate, n_steps, "
                      "n_restart_atttempts, constraint_tolerance must be "
                      "positive")
            raise ValueError(errmsg)
        return v

    @validator('temperature')
    def is_temperature(cls, v):
        if not v.is_compatible_with(unit.kelvin):
            raise ValueError("Must be temperature value, e.g. use unit.kelvin")
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


class SimulationSettings(BaseModel):
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


class RelativeLigandTransformSettings(BaseModel):
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

    def _gufe_tokenize(self):
        return serialise_pydantic(self)


def serialise_pydantic(settings: RelativeLigandTransformSettings):
    def serialise_unit(thing):
        # this gets called when a thing can't get jsonified by pydantic
        # for now only unit.Quantity fall foul of this requirement
        if not isinstance(thing, unit.Quantity):
            raise TypeError
        return '__Quantity__' + str(thing)
    return settings.json(encoder=serialise_unit)


def deserialise_pydantic(raw: str) -> RelativeLigandTransformSettings:
    def undo_mash(d):
        for k, v in d.items():
            if isinstance(v, str) and v.startswith('__Quantity__'):
                d[k] = unit.Quantity(v[12:])  # 12==strlen ^^
            elif isinstance(v, dict):
                d[k] = undo_mash(v)
        return d

    dct = json.loads(raw)
    dct = undo_mash(dct)

    return RelativeLigandTransformSettings(**dct)


def _get_resname(off_mol) -> str:
    # behaviour changed between 0.10 and 0.11
    omm_top = off_mol.to_topology().to_openmm()
    names = [r.name for r in omm_top.residues()]
    if len(names) > 1:
        raise ValueError("We assume single residue")
    return names[0]


class RelativeLigandTransformResult(gufe.ProtocolResult):
    """Dict-like container for the output of a RelativeLigandTransform"""
    def __init__(self, **data):
        super().__init__(**data)
        # TODO: Detect when we have extensions and stitch these together?
        if any(len(files['nc_paths']) > 2 for files in self.data['nc_files']):
            raise NotImplementedError("Can't stitch together results yet")

        self._analyzers = []
        for f in self.data['nc_files']:
            nc = f['nc_paths'][0]
            chk = f['checkpoint_paths'][0]
            reporter = multistate.MultiStateReporter(
                           storage=nc,
                           checkpoint_storage=chk)
            analyzer = multistate.MultiStateSamplerAnalyzer(reporter)

            self._analyzers.append(analyzer)

    def get_estimate(self):
        """Free energy difference of this transformation

        Returns
        -------
        dG : unit.Quantity
          The free energy difference between the first and last states. This is
          a Quantity defined with units.

        TODO
        ----
        * Check this holds up completely for SAMS.
        """
        dGs = []
        #weights = []

        for analyzer in self._analyzers:
            # this returns:
            # (matrix of) estimated free energy difference
            # (matrix of) estimated statistical uncertainty (one S.D.)
            dG, _ = analyzer.get_free_energy()
            dG = (dG[0, -1] * analyzer.kT).in_units_of(
                omm_unit.kilocalories_per_mole)

            # hack to get simulation length in uncorrelated samples
            #weight = analyzer._get_equilibration_data()[2]

            dGs.append(dG)
            #weights.append(weight)
            
        avg_val = np.average([i.value_in_unit(dGs[0].unit) for i in dGs])

        return avg_val * dGs[0].unit

    def get_uncertainty(self):
        """The uncertainty/error in the dG value"""
        dGs = []

        for analyzer in self._analyzers:
            # this returns:
            # (matrix of) estimated free energy difference
            # (matrix of) estimated statistical uncertainty (one S.D.)
            dG, _ = analyzer.get_free_energy()
            dG = (dG[0, -1] * analyzer.kT).in_units_of(
                omm_unit.kilocalories_per_mole)

            dGs.append(dG)
        
        std_val = np.std([i.value_in_unit(dGs[0].unit) for i in dGs])

        return std_val * dGs[0].unit

    def get_rate_of_convergence(self):
        raise NotImplementedError


class RelativeLigandTransform(gufe.Protocol):
    result_cls = RelativeLigandTransformResult

    def __init__(self, settings: RelativeLigandTransformSettings):
        super().__init__(settings)

    def _to_dict(self):
        return {'settings': serialise_pydantic(self.settings)}

    @classmethod
    def _from_dict(cls, dct: Dict):
        return cls(settings=deserialise_pydantic(dct['settings']))

    @classmethod
    def _default_settings(cls) -> RelativeLigandTransformSettings:
        """A dictionary of initial settings for this creating this Protocol

        These settings are intended as a suitable starting point for creating
        an instance of this protocol.  It is recommended, however that care is
        taken to inspect and customize these before performing a Protocol.

        Returns
        -------
        RelativeLigandTransformSettings
          a set of default settings
        """
        return RelativeLigandTransformSettings(
            system_settings=SystemSettings(
                constraints='HBonds'
            ),
            topology_settings=TopologySettings(
                forcefield={'protein': 'amber99sb.xml',
                            'ligand': 'openff-2.0.0.offxml',
                            'solvent': 'tip3p.xml'},
            ),
            alchemical_settings=AlchemicalSettings(),
            sampler_settings=SamplerSettings(),
            barostat_settings=BarostatSettings(),
            integrator_settings=IntegratorSettings(),
            simulation_settings=SimulationSettings(
                equilibration_length=2.0 * unit.nanosecond,
                production_length=5.0 * unit.nanosecond,
            )
        )

    def _create(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: Optional[dict[str, gufe.ComponentMapping]] = None,
        extend_from: Optional[gufe.ProtocolDAGResult] = None,
    ) -> list[gufe.ProtocolUnit]:
        # TODO: Extensions?
        if mapping is None:
            raise ValueError("`mapping` is required for this Protocol")
        if 'ligand' not in mapping:
            raise ValueError("'ligand' must be specified in `mapping` dict")
        if extend_from:
            raise NotImplementedError("Can't extend simulations yet")

        # Checks on the inputs!
        # 1) check that both states have solvent and ligand
        for state, label in [(stateA, 'A'), (stateB, 'B')]:
            if 'solvent' not in state.components:
                nonbond = self.settings.system_settings.nonbonded_method
                if nonbond != 'nocutoff':
                    errmsg = f"{nonbond} cannot be used for vacuum transform"
                    raise ValueError(errmsg)
            if 'ligand' not in state.components:
                raise ValueError(f"Missing ligand in state {label}")
        # 1b) check xnor have Protein component
        nproteins = sum(1 for state in (stateA, stateB) if 'protein' in state)
        if nproteins == 1:  # only one state has a protein defined
            raise ValueError("Only one state had a protein component")
        elif nproteins == 2:
            if stateA['protein'] != stateB['protein']:
                raise ValueError("Proteins in each state aren't compatible")

        # 2) check that both states have same solvent
        # TODO: defined box compatibility check
        #       probably lives as a ChemicalSystem.box_is_compatible_with(other)
        if 'solvent' in stateA.components:
            if not stateA['solvent'] == stateB['solvent']:
                raise ValueError("Solvents aren't identical between states")
        # check that the mapping refers to the two ligand components
        ligandmapping : LigandAtomMapping = mapping['ligand']
        if stateA['ligand'] != ligandmapping.componentA:
            raise ValueError("Ligand in state A doesn't match mapping")
        if stateB['ligand'] != ligandmapping.componentB:
            raise ValueError("Ligand in state B doesn't match mapping")
        # 3) check that the mapping doesn't involve element changes
        # this is currently a requirement of the method
        molA = ligandmapping.componentA.to_rdkit()
        molB = ligandmapping.componentB.to_rdkit()
        for i, j in ligandmapping.componentA_to_componentB.items():
            atomA = molA.GetAtomWithIdx(i)
            atomB = molB.GetAtomWithIdx(j)
            if atomA.GetAtomicNum() != atomB.GetAtomicNum():
                raise ValueError(
                    f"Element change in mapping between atoms "
                    f"Ligand A: {i} (element {atomA.GetAtomicNum()} and "
                    f"Ligand B: {j} (element {atomB.GetAtomicNum()}")

        # actually create and return Units
        Aname = stateA['ligand'].name
        Bname = stateB['ligand'].name
        # our DAG has no dependencies, so just list units
        units = [RelativeLigandTransformUnit(
            stateA=stateA, stateB=stateB, ligandmapping=ligandmapping,
            settings=self.settings,
            generation=0, repeat_id=i,
            name=f'{Aname} {Bname} repeat {i} generation 0')
            for i in range(self.settings.sampler_settings.n_repeats)]

        return units

    def _gather(
        self, protocol_dag_results: Iterable[gufe.ProtocolDAGResult]
    ) -> Dict[str, Any]:
        # result units will have a repeat_id and generation
        # first group according to repeat_id
        repeats = defaultdict(list)
        for d in protocol_dag_results:
            for pu in d.protocol_unit_results:
                rep = pu.outputs['repeat_id']
                gen = pu.outputs['generation']

                repeats[rep].append((
                    gen, pu.outputs['nc'],
                    pu.outputs['last_checkpoint']))
        data = []
        for rep in sorted(repeats.items()):
            # then sort within a repeat according to generation
            nc_paths = [ncpath for gen, ncpath, nc_check in sorted(rep[1])]
            chk_files = [nc_check for gen, ncpath, nc_check in sorted(rep[1])]
            data.append({'nc_paths': nc_paths,
                         'checkpoint_paths': chk_files})

        return {
            'nc_files': data,
        }


class RelativeLigandTransformUnit(gufe.ProtocolUnit):
    """Calculates the relative free energy of an alchemical ligand transformation.

    """
    _stateA: ChemicalSystem
    _stateB: ChemicalSystem
    _mapping: LigandAtomMapping
    _settings: RelativeLigandTransformSettings
    generation: int
    repeat_id: int
    name: str

    def __init__(self, *,
                 stateA: ChemicalSystem,
                 stateB: ChemicalSystem,
                 ligandmapping: LigandAtomMapping,
                 settings: RelativeLigandTransformSettings,
                 name: Optional[str] = None,
                 generation: int = 0,
                 repeat_id: int = 0,
                 ):
        """
        Parameters
        ----------
        stateA, stateB : ChemicalSystem
          the two ligand SmallMoleculeComponents to transform between.  The
          transformation will go from ligandA to ligandB.
        ligandmapping : LigandAtomMapping
          the mapping of atoms between the two ligand components
        settings : RelativeLigandTransformSettings
          the settings for the Method.  This can be constructed using the
          get_default_settings classmethod to give a starting point that
          can be updated to suit.
        name : str, optional
          human-readable identifier for this Unit
        repeat_id : int, optional
          identifier for which repeat (aka replica/clone) this Unit is,
          default 0
        generation : int, optional
          counter for how many times this repeat has been extended, default 0

        Notes
        -----
        The mapping used must not involve any elemental changes.  A check for
        this is done on class creation.
        """
        super().__init__(
            name=name,
            stateA=stateA,
            stateB=stateB,
            mapping=ligandmapping,
            settings=settings,
        )
        self.repeat_id = repeat_id
        self.generation = generation

    def _to_dict(self):
        return {
            'inputs': self.inputs,
            'generation': self.generation,
            'repeat_id': self.repeat_id,
            'name': self.name,
        }

    @classmethod
    def _from_dict(cls, dct: Dict):
        dct['_settings'] = deserialise_pydantic(dct['_settings'])

        inps = dct.pop('inputs')

        return cls(
            **inps,
            **dct
        )

    def run(self, dry=False, verbose=True, basepath=None) -> dict[str, Any]:
        """Run the relative free energy calculation.

        Parameters
        ----------
        dry : bool
          Do a dry run of the calculation, creating all necessary hybrid
          system components (topology, system, sampler, etc...) but without
          running the simulation.
        verbose : bool
          Verbose output of the simulation progress. Output is provided via
          INFO level logging.
        basepath : Pathlike, optional
          Where to run the calculation, defaults to current working directory

        Returns
        -------
        dict
          Outputs created in the basepath directory or the debug objects
          (i.e. sampler) if ``dry==True``.

        Raises
        ------
        error
          Exception if anything failed
        """
        if verbose:
            logger.info("creating hybrid system")
        if basepath is None:
            # use cwd
            basepath = pathlib.Path('.')

        # 0. General setup and settings dependency resolution step

        # a. check equilibration and production are divisible by n_steps
        settings = self._inputs['settings']
        stateA = self._inputs['stateA']
        stateB = self._inputs['stateB']
        mapping = self._inputs['mapping']

        sim_settings = settings.simulation_settings
        timestep = settings.integrator_settings.timestep
        mc_steps = settings.integrator_settings.n_steps.m

        equil_time = sim_settings.equilibration_length.to('femtosecond')
        equil_steps = round(equil_time / timestep)

        # mypy gets the return type of round wrong, it's a Quantity
        if (equil_steps.m % mc_steps) != 0:  # type: ignore
            errmsg = (f"Equilibration time {equil_time} should contain a "
                      "number of steps divisible by the number of integrator "
                      f"timesteps between MC moves {mc_steps}")
            raise ValueError(errmsg)

        prod_time = sim_settings.production_length.to('femtosecond')
        prod_steps = round(prod_time / timestep)

        if (prod_steps.m % mc_steps) != 0:  # type: ignore
            errmsg = (f"Production time {prod_time} should contain a "
                      "number of steps divisible by the number of integrator "
                      f"timesteps between MC moves {mc_steps}")
            raise ValueError(errmsg)

        # b. get the openff objects for the ligands
        stateA_openff_ligand = stateA['ligand'].to_openff()
        stateB_openff_ligand = stateB['ligand'].to_openff()

        #  1. Get smirnoff template generators
        smirnoff_stateA = SMIRNOFFTemplateGenerator(
            forcefield=settings.topology_settings.forcefield['ligand'],
            molecules=[stateA_openff_ligand],
        )

        smirnoff_stateB = SMIRNOFFTemplateGenerator(
            forcefield=settings.topology_settings.forcefield['ligand'],
            molecules=[stateB_openff_ligand],
        )

        # 2. Create forece fields and register them
        #  a. state A
        omm_forcefield_stateA = app.ForceField(
            *[ff for (comp, ff) in settings.topology_settings.forcefield.items()
              if not comp == 'ligand']
        )

        omm_forcefield_stateA.registerTemplateGenerator(
                smirnoff_stateA.generator)

        #  b. state B
        omm_forcefield_stateB = app.ForceField(
            *[ff for (comp, ff) in settings.topology_settings.forcefield.items()
              if not comp == 'ligand']
        )

        omm_forcefield_stateB.registerTemplateGenerator(
                smirnoff_stateB.generator)

        # 3. Model state A
        stateA_ligand_topology = stateA_openff_ligand.to_topology().to_openmm()
        if 'protein' in stateA.components:
            pdbfile: gufe.ProteinComponent = stateA['protein']
            stateA_modeller = app.Modeller(pdbfile.to_openmm_topology(),
                                           pdbfile.to_openmm_positions())
            stateA_modeller.add(
                stateA_ligand_topology,
                ensure_quantity(stateA_openff_ligand.conformers[0], 'openmm'),
            )
        else:
            stateA_modeller = app.Modeller(
                stateA_ligand_topology,
                ensure_quantity(stateA_openff_ligand.conformers[0], 'openmm'),
            )
        # make note of which chain id(s) the ligand is,
        # we'll need this to swap it out later
        stateA_ligand_nchains = stateA_ligand_topology.getNumChains()
        stateA_ligand_chain_id = stateA_modeller.topology.getNumChains()

        # 4. Solvate the complex in a `concentration` mM cubic water box with
        # `solvent_padding` from the solute to the edges of the box
        if 'solvent' in stateA.components:
            conc = stateA['solvent'].ion_concentration
            pos = stateA['solvent'].positive_ion
            neg = stateA['solvent'].negative_ion

            stateA_modeller.addSolvent(
                omm_forcefield_stateA,
                model=settings.topology_settings.solvent_model,
                padding=to_openmm(settings.solvent_padding),
                positiveIon=pos, negativeIon=neg,
                ionicStrength=to_openmm(conc),
            )

        # 5.  Create OpenMM system + topology + initial positions for "A" system
        #  a. Get nonbond method
        nonbonded_method = {
            'pme': app.PME,
            'nocutoff': app.NoCutoff,
            'cutoffnonperiodic': app.CutoffNonPeriodic,
            'cutoffperiodic': app.CutoffPeriodic,
            'ewald': app.Ewald
        }[settings.system_settings.nonbonded_method.lower()]

        #  b. Get the constraint method
        constraints = {
            'hbonds': app.HBonds,
            'none': None,
            'allbonds': app.AllBonds,
            'hangles': app.HAngles
            # vvv can be None so string it
        }[str(settings.system_settings.constraints).lower()]

        #  c. create the stateA System
        stateA_system = omm_forcefield_stateA.createSystem(
            stateA_modeller.topology,
            nonbondedMethod=nonbonded_method,
            nonbondedCutoff=to_openmm(settings.system_settings.nonbonded_cutoff),
            constraints=constraints,
            rigidWater=settings.system_settings.rigid_water,
            hydrogenMass=settings.system_settings.hydrogen_mass,
            removeCMMotion=settings.system_settings.remove_com,
        )

        #  d. create stateA topology
        stateA_topology = stateA_modeller.getTopology()

        #  e. get stateA positions
        stateA_positions = stateA_modeller.getPositions()
        ## canonicalize positions (tuples to np.array)
        stateA_positions = omm_unit.Quantity(
            value=np.array([list(pos) for pos in stateA_positions.value_in_unit_system(openmm.unit.md_unit_system)]),
            unit = openmm.unit.nanometers
        )

        # 6.  Create OpenMM system + topology + positions for "B" system
        #  a. stateB topology from stateA (replace out the ligands)
        stateB_topology = _rbfe_utils.topologyhelpers.combined_topology(
            stateA_topology,
            stateB_openff_ligand.to_topology().to_openmm(),
            # as we kept track as we added, we can slice the ligand out
            # this isn't at the end because of solvents
            exclude_chains=list(stateA_topology.chains())[stateA_ligand_chain_id - stateA_ligand_nchains:stateA_ligand_chain_id]
        )

        #  b. Create the system
        stateB_system = omm_forcefield_stateB.createSystem(
            stateB_topology,
            nonbondedMethod=nonbonded_method,
            nonbondedCutoff=to_openmm(settings.system_settings.nonbonded_cutoff),
            constraints=constraints,
            rigidWater=settings.system_settings.rigid_water,
            hydrogenMass=settings.system_settings.hydrogen_mass,
            removeCMMotion=settings.system_settings.remove_com,
        )

        #  c. Define correspondence mappings between the two systems
        ligand_mappings = _rbfe_utils.topologyhelpers.get_system_mappings(
            mapping.componentA_to_componentB,
            stateA_system, stateA_topology, _get_resname(stateA_openff_ligand),
            stateB_system, stateB_topology, _get_resname(stateB_openff_ligand),
            # These are non-optional settings for this method
            fix_constraints=True,
        )

        #  d. Finally get the positions
        stateB_positions = _rbfe_utils.topologyhelpers.set_and_check_new_positions(
            ligand_mappings, stateA_topology, stateB_topology,
            old_positions=ensure_quantity(stateA_positions, 'openmm'),
            insert_positions=ensure_quantity(stateB_openff_ligand.conformers[0], 'openmm'),
            tolerance=settings.alchemical_settings.atom_overlap_tolerance
        )

        # 7. Create the hybrid topology
        #  a. Get alchemical settings
        alchem_settings = settings.alchemical_settings

        #  b. Create the hybrid topology factory
        hybrid_factory = _rbfe_utils.relative.HybridTopologyFactory(
            stateA_system, stateA_positions, stateA_topology,
            stateB_system, stateB_positions, stateB_topology,
            old_to_new_atom_map=ligand_mappings['old_to_new_atom_map'],
            old_to_new_core_atom_map=ligand_mappings['old_to_new_core_atom_map'],
            use_dispersion_correction=alchem_settings.use_dispersion_correction,
            softcore_alpha=alchem_settings.softcore_alpha,
            softcore_LJ_v2=alchem_settings.softcore_LJ_v2,
            softcore_LJ_v2_alpha=alchem_settings.softcore_alpha,
            softcore_electrostatics=alchem_settings.softcore_electrostatics,
            softcore_electrostatics_alpha=alchem_settings.softcore_electrostatics_alpha,
            softcore_sigma_Q=alchem_settings.softcore_sigma_Q,
            interpolate_old_and_new_14s=alchem_settings.interpolate_old_and_new_14s,
            flatten_torsions=alchem_settings.flatten_torsions,
        )

        #  c. Add a barostat to the hybrid system
        if 'solvent' in stateA.components:
            hybrid_factory.hybrid_system.addForce(
                openmm.MonteCarloBarostat(
                    settings.barostat_settings.pressure.to(unit.bar).m,
                    settings.integrator_settings.temperature.m,
                    settings.barostat_settings.frequency.m,
                )
            )

        # 8. Create lambda schedule
        # TODO - this should be exposed to users, maybe we should offer the
        # ability to print the schedule directly in settings?
        lambdas = _rbfe_utils.lambdaprotocol.LambdaProtocol(
            functions=alchem_settings.lambda_functions,
            windows=alchem_settings.lambda_windows
        )

        # PR #125 temporarily pin lambda schedule spacing to n_replicas
        n_replicas = settings.sampler_settings.n_replicas
        if n_replicas != len(lambdas.lambda_schedule):
            errmsg = (f"Number of replicas {n_replicas} "
                      f"does not equal the number of lambda windows "
                      f"{len(lambdas.lambda_schedule)}")
            raise ValueError(errmsg)

        # 9. Create the multistate reporter
        # Get the sub selection of the system to print coords for
        selection_indices = hybrid_factory.hybrid_topology.select(
                settings.simulation_settings.output_indices
        )

        #  a. Create the multistate reporter
        reporter = multistate.MultiStateReporter(
            storage=basepath / settings.simulation_settings.output_filename,
            analysis_particle_indices=selection_indices,
            checkpoint_interval=settings.simulation_settings.checkpoint_interval.m,
            checkpoint_storage=basepath / settings.simulation_settings.checkpoint_storage,
        )

        # 10. Get platform and context caches
        platform = _rbfe_utils.compute.get_openmm_platform(
            settings.engine_settings.compute_platform
        )

        #  a. Create context caches (energy + sampler)
        #     Note: these needs to exist on the compute node
        energy_context_cache = openmmtools.cache.ContextCache(
            capacity=None, time_to_live=None, platform=platform,
        )

        sampler_context_cache = openmmtools.cache.ContextCache(
            capacity=None, time_to_live=None, platform=platform,
        )

        # 11. Set the integrator
        #  a. get integrator settings
        integrator_settings = settings.integrator_settings

        #  b. create langevin integrator
        integrator = openmmtools.mcmc.LangevinSplittingDynamicsMove(
            timestep=to_openmm(integrator_settings.timestep),
            collision_rate=to_openmm(integrator_settings.collision_rate),
            n_steps=integrator_settings.n_steps.m,
            reassign_velocities=integrator_settings.reassign_velocities,
            n_restart_attempts=integrator_settings.n_restart_attempts,
            constraint_tolerance=integrator_settings.constraint_tolerance,
            splitting=integrator_settings.splitting
        )

        # 12. Create sampler
        sampler_settings = settings.sampler_settings
        
        if sampler_settings.sampler_method.lower() == "repex":
            sampler = _rbfe_utils.multistate.HybridRepexSampler(
                mcmc_moves=integrator,
                hybrid_factory=hybrid_factory,
                online_analysis_interval=sampler_settings.online_analysis_interval,
                online_analysis_target_error=sampler_settings.online_analysis_target_error.m,
                online_analysis_minimum_iterations=sampler_settings.online_analysis_minimum_iterations
            )
        elif sampler_settings.sampler_method.lower() == "sams":
            sampler = _rbfe_utils.multistate.HybridSAMSSampler(
                mcmc_moves=integrator,
                hybrid_factory=hybrid_factory,
                online_analysis_interval=sampler_settings.online_analysis_interval,
                online_analysis_minimum_iterations=sampler_settings.online_analysis_minimum_iterations,
                flatness_criteria=sampler_settings.flatness_criteria,
                gamma0=sampler_settings.gamma0,
            )
        elif sampler_settings.sampler_method.lower() == 'independent':
            sampler = _rbfe_utils.multistate.HybridMultiStateSampler(
                mcmc_moves=integrator,
                hybrid_factory=hybrid_factory,
                online_analysis_interval=sampler_settings.online_analysis_interval,
                online_analysis_target_error=sampler_settings.online_analysis_target_error.m,
                online_analysis_minimum_iterations=sampler_settings.online_analysis_minimum_iterations
            )

        else:
            raise AttributeError(f"Unknown sampler {sampler_settings.sampler_method}")

        sampler.setup(
            n_replicas=sampler_settings.n_replicas,
            reporter=reporter,
            platform=platform,
            lambda_protocol=lambdas,
            temperature=to_openmm(settings.integrator_settings.temperature),
            endstates=alchem_settings.unsampled_endstates,
        )

        sampler.energy_context_cache = energy_context_cache
        sampler.sampler_context_cache = sampler_context_cache

        if not dry:
            # minimize
            if verbose:
                logger.info("minimizing systems")

            sampler.minimize(max_iterations=settings.simulation_settings.minimization_steps)

            # equilibrate
            if verbose:
                logger.info("equilibrating systems")

            sampler.equilibrate(int(equil_steps.m / mc_steps))  # type: ignore

            # production
            if verbose:
                logger.info("running production phase")

            sampler.extend(int(prod_steps.m / mc_steps))  # type: ignore
            
            # close reporter when you're done
            reporter.close()

            nc = basepath / settings.simulation_settings.output_filename
            chk = basepath / settings.simulation_settings.checkpoint_storage
            return {
                'nc': nc,
                'last_checkpoint': chk,
            }
        else:
            # close reporter when you're done, prevent file handle clashes
            reporter.close()

            # clean up the reporter file
            fns = [basepath / settings.simulation_settings.output_filename,
                   basepath / settings.simulation_settings.checkpoint_storage]
            for fn in fns:
                os.remove(fn)
            return {'debug': {'sampler': sampler}}

    def _execute(
        self, ctx: gufe.Context, **kwargs,
    ) -> dict[str, Any]:
        # create directory for *this* unit within the context of the *DAG*
        # stops output files mashing into each other within a DAG
        myid = uuid.uuid4()
        mypath = pathlib.Path(os.path.join(ctx.shared, str(myid)))
        mypath.mkdir(parents=True, exist_ok=False)

        outputs = self.run(basepath=mypath)

        return {
            'repeat_id': self.repeat_id,
            'generation': self.generation,
            **outputs
        }
