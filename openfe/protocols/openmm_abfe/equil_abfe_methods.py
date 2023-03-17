# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""Equilibrium ABFE methods using OpenMM + OpenMMTools

This module implements the necessary methodology toolking to run calculate an
absolute free energy transformation using OpenMM tools and one of the
following methods:
    - Hamiltonian Replica Exchange
    - Self-adjusted mixture sampling
    - Independent window sampling

Acknowledgements
----------------
* Originally based on a script from hydration.py in
  [espaloma](https://github.com/choderalab/espaloma_charge)

TODO
----
* Add support for restraints
* Improve this docstring by adding an example use case.

"""
from __future__ import annotations

import os
import logging

from collections import defaultdict
import gufe
from gufe.components import Component
from gufe.protocols import ProtocolDAG, ProtocolDAGResult
import json
import numpy as np
import openmm
from openff.units import unit
from openff.units.openmm import to_openmm, ensure_quantity
from openmmtools import multistate
from openmmtools.states import (ThermodynamicState, SamplerState,
                                create_thermodynamic_state_protocol,)
from openmmtools.alchemy import (AlchemicalRegion, AbsoluteAlchemicalFactory,
                                 AlchemicalState,)
from pydantic import BaseModel, validator
from typing import Dict, List, Union, Optional
from openmm import app
from openmm import unit as omm_unit
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
import pathlib
from typing import Any, Iterable
import openmmtools
import uuid
import mdtraj as mdt

from openfe.setup import (
    ChemicalSystem, LigandAtomMapping,
)
from openfe.protocols.openmm_rbfe._rbfe_utils import compute


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
    lambda_functions : str
      Key of which switching functions to use for alchemical mutation.
      Currently only default is supported. Default 'default'.
    lambda_windows : int
      Number of lambda windows to calculate. Default 24.
    """
    # Lambda settings
    lambda_functions = 'default'
    lambda_windows = 24


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
    SAMSSampler or ReplicaExchangeSampler.

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
      Number of replicas to use. Default 24.

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
    n_replicas = 24

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
    output_filename = 'abfe.nc'
    output_indices = 'all'
    checkpoint_interval = 50 * unit.timestep
    checkpoint_storage = 'abfe_checkpoint.nc'

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


class AbsoluteTransformSettings(BaseModel):
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


def serialise_pydantic(settings: AbsoluteTransformSettings):
    def serialise_unit(thing):
        # this gets called when a thing can't get jsonified by pydantic
        # for now only unit.Quantity fall foul of this requirement
        if not isinstance(thing, unit.Quantity):
            raise TypeError
        return '__Quantity__' + str(thing)
    return settings.json(encoder=serialise_unit)


def deserialise_pydantic(raw: str) -> AbsoluteTransformSettings:
    def undo_mash(d):
        for k, v in d.items():
            if isinstance(v, str) and v.startswith('__Quantity__'):
                d[k] = unit.Quantity(v[12:])  # 12==strlen ^^
            elif isinstance(v, dict):
                d[k] = undo_mash(v)
        return d

    dct = json.loads(raw)
    dct = undo_mash(dct)

    return AbsoluteTransformSettings(**dct)


def _get_resname(off_mol) -> str:
    # behaviour changed between 0.10 and 0.11
    omm_top = off_mol.to_topology().to_openmm()
    names = [r.name for r in omm_top.residues()]
    if len(names) > 1:
        raise ValueError("We assume single residue")
    return names[0]


class AbsoluteTransformResult(gufe.ProtocolResult):
    """Dict-like container for the output of a AbsoluteTransform"""
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

        for analyzer in self._analyzers:
            # this returns:
            # (matrix of) estimated free energy difference
            # (matrix of) estimated statistical uncertainty (one S.D.)
            dG, _ = analyzer.get_free_energy()
            dG = (dG[0, -1] * analyzer.kT).in_units_of(
                omm_unit.kilocalories_per_mole)

            dGs.append(dG)
            
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


class AbsoluteTransform(gufe.Protocol):
    result_cls = AbsoluteTransformResult

    def __init__(self, settings: AbsoluteTransformSettings):
        super().__init__(settings)

    def _to_dict(self):
        return {'settings': serialise_pydantic(self.settings)}

    @classmethod
    def _from_dict(cls, dct: Dict):
        return cls(settings=deserialise_pydantic(dct['settings']))

    @classmethod
    def _default_settings(cls) -> AbsoluteTransformSettings:
        """A dictionary of initial settings for this creating this Protocol

        These settings are intended as a suitable starting point for creating
        an instance of this protocol.  It is recommended, however that care is
        taken to inspect and customize these before performing a Protocol.

        Returns
        -------
        AbsoluteTransformSettings
          a set of default settings
        """
        return AbsoluteTransformSettings(
            system_settings=SystemSettings(
                constraints='HBonds',
                hydrogen_mass = 3.0,
            ),
            topology_settings=TopologySettings(
                forcefield = {
                    'protein': 'amber/ff14SB.xml',
                    'solvent': 'amber/tip3p_standard.xml',  # TIP3P and recommended monovalent ion parameters
                    'ions': 'amber/tip3p_HFE_multivalent.xml',  # for divalent ions
                    'tpo': 'amber/phosaa10.xml',  # HANDLES THE TPO
                    'ligand': 'openff-2.0.0.offxml',
                }
            ),
            alchemical_settings=AlchemicalSettings(),
            sampler_settings=SamplerSettings(),
            barostat_settings=BarostatSettings(),
            integrator_settings=IntegratorSettings(
                timestep = 4.0 * unit.femtosecond,
                n_steps = 250 * unit.timestep,
            ),
            simulation_settings=SimulationSettings(
                equilibration_length=2.0 * unit.nanosecond,
                production_length=5.0 * unit.nanosecond,
            )
        )

    def _get_alchemical_components(
            stateA, stateB) -> dict[str, List(Component)]:
        """
        Checks equality of ChemicalSystem components across both states and
        identify which components do not match.

        Parameters
        ----------
        stateA : ChemicalSystem
          The chemical system of end state A.
        stateB : ChemicalSystem
          The chemical system of end staate B.

        Returns
        -------
        alchemical_components : Dictionary
            Dictionary containing a list of alchemical components for each
            state.
        """
        matched_keys = {}
        alchemical_components = {'stateA': [], 'stateB': {}}

        for keyA, valA in stateA.components.items():
            for keyB, valB in stateB.component.items():
                if valA.to_dict() == valB.to_dict():
                    matched_keys[keyA] = keyB
                    break

        for state in ['A', 'B']:
            for 

    def _create(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: Optional[dict[str, gufe.ComponentMapping]] = None
        extend_from: Optional[gufe.ProtocolDAGResult] = None,
    ) -> list[gufe.ProtocolUnit]:
        if extend_from:
            raise NotImplementedError("Can't extend simulations yet")

        # Checks on the inputs!
        # 1) check that both states have solvent and ligand
        if 'solvent' not in stateA.components:
            nonbond = self.settings.system_settings.nonbonded_method
            if nonbond != 'nocutoff':
                errmsg = f"{nonbond} cannot be used for vacuum transform"
                raise ValueError(errmsg)
        if 'ligand' not in chem_system.components:
                raise ValueError(f"Missing ligand in system")

        # actually create and return Units
        ligand_name = chem_system['ligand'].name
        # our DAG has no dependencies, so just list units
        units = [AbsoluteTransformUnit(
            chem_system=chem_system,
            settings=self.settings,
            generation=0, repeat_id=i,
            name=f'{ligand_name} repeat {i} generation 0')
            for i in range(self.settings.sampler_settings.n_repeats)]

        return units

    def _gather(
        self, protocol_dag_results: Iterable[gufe.ProtocolDAGResult]
    ) -> Dict[str, Any]:
        # result units will have a repeat_id and generation
        # first group according to repeat_id
        repeats = defaultdict(list)
        for d in protocol_dag_results:
            pu: gufe.ProtocolUnitResult
            for pu in d.protocol_unit_results:
                if not pu.ok():
                    continue
                rep = pu.outputs['repeat_id']
                gen = pu.outputs['generation']

                repeats[rep].append((
                    gen, pu.outputs['nc'],
                    pu.outputs['last_checkpoint']))

        data = []
        for replicate_id, replicate_data in sorted(repeats.items()):
            # then sort within a repeat according to generation
            nc_paths = [ncpath for gen, ncpath, nc_check in sorted(replicate_data)]
            chk_files = [nc_check for gen, ncpath, nc_check in sorted(replicate_data)]
            data.append({'nc_paths': nc_paths,
                         'checkpoint_paths': chk_files})

        return {
            'nc_files': data,
        }


class AbsoluteTransformUnit(gufe.ProtocolUnit):
    """Calculates the absolute free energy of an alchemical ligand transformation.

    """
    _chem_system: ChemicalSystem
    _settings: AbsoluteTransformSettings
    generation: int
    repeat_id: int
    name: str

    def __init__(self, *,
                 chem_system: ChemicalSystem,
                 settings: AbsoluteTransformSettings,
                 name: Optional[str] = None,
                 generation: int = 0,
                 repeat_id: int = 0,
                 ):
        """
        Parameters
        ----------
        chem_system : ChemicalSystem
          the ChemicalSystem containing a SmallMoleculeComponent being
          alchemically removed from it.
        settings : AbsoluteTransformSettings
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

        """
        super().__init__(
            name=name,
            chem_system=chem_system,
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
        """Run the absolute free energy calculation.

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
        chem_system = self._inputs['chem_system']

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

        # b. get the openff object for the ligand
        openff_ligand = chem_system['ligand'].to_openff()

        #  1. Get smirnoff template generators
        smirnoff_system = SMIRNOFFTemplateGenerator(
            forcefield=settings.topology_settings.forcefield['ligand'],
            molecules=[openff_ligand],
        )

        # 2. Create forece fields and register them
        omm_forcefield = app.ForceField(
            *[ff for (comp, ff) in settings.topology_settings.forcefield.items()
              if not comp == 'ligand']
        )

        omm_forcefield.registerTemplateGenerator(
                smirnoff_system.generator)

        # 3. Model state A
        ligand_topology = openff_ligand.to_topology().to_openmm()
        if 'protein' in chem_system.components:
            pdbfile: gufe.ProteinComponent = chem_system['protein']
            system_modeller = app.Modeller(pdbfile.to_openmm_topology(),
                                           pdbfile.to_openmm_positions())
            system_modeller.add(
                ligand_topology,
                ensure_quantity(openff_ligand.conformers[0], 'openmm'),
            )
        else:
            system_modeller = app.Modeller(
                ligand_topology,
                ensure_quantity(openff_ligand.conformers[0], 'openmm'),
            )
        # make note of which chain id(s) the ligand is,
        # we'll need this to chemically modify it later
        ligand_nchains = ligand_topology.getNumChains()
        ligand_chain_id = system_modeller.topology.getNumChains()

        # 4. Solvate the complex in a `concentration` mM cubic water box with
        # `solvent_padding` from the solute to the edges of the box
        if 'solvent' in chem_system.components:
            conc = chem_system['solvent'].ion_concentration
            pos = chem_system['solvent'].positive_ion
            neg = chem_system['solvent'].negative_ion

            system_modeller.addSolvent(
                omm_forcefield,
                model=settings.topology_settings.solvent_model,
                padding=to_openmm(settings.solvent_padding),
                positiveIon=pos, negativeIon=neg,
                ionicStrength=to_openmm(conc),
            )

        # 5.  Create OpenMM system + topology + initial positions
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

        #  c. create the System
        omm_system = omm_forcefield.createSystem(
            system_modeller.topology,
            nonbondedMethod=nonbonded_method,
            nonbondedCutoff=to_openmm(settings.system_settings.nonbonded_cutoff),
            constraints=constraints,
            rigidWater=settings.system_settings.rigid_water,
            hydrogenMass=settings.system_settings.hydrogen_mass,
            removeCMMotion=settings.system_settings.remove_com,
        )

        #  d. create stateA topology
        system_topology = system_modeller.getTopology()

        #  e. get ligand indices
        lig_chain = list(system_topology.chains())[ligand_chain_id - ligand_nchains:ligand_chain_id]
        assert len(lig_chain) == 1
        lig_atoms = list(lig_chain[0].atoms())
        lig_indices = [at.index for at in lig_atoms]

        #  f. get stateA positions
        system_positions = system_modeller.getPositions()
        ## canonicalize positions (tuples to np.array)
        system_positions = omm_unit.Quantity(
            value=np.array([list(pos) for pos in system_positions.value_in_unit_system(openmm.unit.md_unit_system)]),
            unit = openmm.unit.nanometers
        )

        # 6. Create the alchemical system
        #  a. Get alchemical settings
        alchem_settings = settings.alchemical_settings

        #  b. add a barostat if necessary
        if 'solvent' in chem_system.components:
            omm_system.addForce(
                openmm.MonteCarloBarostat(
                    settings.barostat_settings.pressure.to(unit.bar).m,
                    settings.integrator_settings.temperature.m,
                    settings.barostat_settings.frequency.m,
                )
            )

        #  c. Define the thermodynamic state
        ## Note: we should be able to remove the if around the barostat here..
        ## TODO: check that the barostat settings are preseved
        if 'solvent' in chem_system.components:
            thermostate = ThermodynamicState(
                system=omm_system,
                temperature=to_openmm(settings.integrator_settings.temperature),
                pressure=to_openmm(settings.barostat_settings.pressure)
            )
        else:
            thermostate = ThermodynamicState(
                system=omm_system,
                temperature=to_openmm(settings.integrator_settings.temperature),
            )

        # pre-minimize system for a few steps to avoid GPU overflow
        integrator = openmm.VerletIntegrator(0.001)
        context = openmm.Context(
                omm_system, integrator, 
                openmm.Platform.getPlatformByName('CPU'),
        )
        context.setPositions(system_positions)
        openmm.LocalEnergyMinimizer.minimize(
                context, maxIterations=100
        )
        positions = context.getState(getPositions=True).getPositions(asNumpy=True)
        del context, integrator

        # 8. Create alchemical system
        ## TODO add support for all the variants here
        ## TODO: check that adding indices this way works
        alchemical_region = AlchemicalRegion(
                alchemical_atoms=lig_indices
        )
        alchemical_factory = AbsoluteAlchemicalFactory()
        alchemical_system = alchemical_factory.create_alchemical_system(
                omm_system, alchemical_region
        )

        # 9. Create lambda schedule
        ## TODO: do this properly using LambdaProtocol
        ## TODO: double check we definitely don't need to define
        ##       temperature & pressure (pressure sure that's the case)
        lambdas = dict()
        n_elec = int(alchem_settings.lambda_windows / 2)
        n_vdw = alchem_settings.lambda_windows - n_elec + 1
        lambdas['lambda_electrostatics'] = np.concatenate(
                [np.linspace(1, 0, n_elec), np.linspace(0, 0, n_vdw)[1:]]
        )
        lambdas['lambda_sterics'] = np.concatenate(
                [np.linspace(1, 1, n_elec), np.linspace(1, 0, n_vdw)[1:]]
        )

        ## Check that the lambda schedule matches n_replicas
        #### TODO: undo for SAMS
        n_replicas = settings.sampler_settings.n_replicas

        if n_replicas != (len(lambdas['lambda_sterics'])):
            errmsg = (f"Number of replicas {n_replicas} "
                      "does not equal the number of lambda windows ")
            raise ValueError(errmsg)

        # 10. Create compound states
        alchemical_state = AlchemicalState.from_system(alchemical_system)
        constants = dict()
        constants['temperature'] = to_openmm(settings.integrator_settings.temperature)
        if 'solvent' in chem_system.components:
            constants['pressure'] = to_openmm(settings.barostat_settings.pressure)
        cmp_states = create_thermodynamic_state_protocol(
                alchemical_system,
                protocol=lambdas,
                constants=constants,
                composable_states=[alchemical_state],
        )

        # 11. Create the sampler states
        # Fill up a list of sampler states all with the same starting state
        sampler_state = SamplerState(positions=positions)
        if omm_system.usesPeriodicBoundaryConditions():
            box = omm_system.getDefaultPeriodicBoxVectors()
            sampler_state.box_vectors = box

        sampler_states = [sampler_state for _ in cmp_states]
        

        # 9. Create the multistate reporter
        # Get the sub selection of the system to print coords for
        ## TODO: check this actually works
        mdt_top = mdt.Topology.from_openmm(system_topology)
        selection_indices = mdt_top.select(
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
        platform = compute.get_openmm_platform(
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
            sampler = multistate.ReplicaExchangeSampler(
                mcmc_moves=integrator,
                online_analysis_interval=sampler_settings.online_analysis_interval,
                online_analysis_target_error=sampler_settings.online_analysis_target_error.m,
                online_analysis_minimum_iterations=sampler_settings.online_analysis_minimum_iterations
            )
        elif sampler_settings.sampler_method.lower() == "sams":
            sampler = multistate.SAMSSampler(
                mcmc_moves=integrator,
                online_analysis_interval=sampler_settings.online_analysis_interval,
                online_analysis_minimum_iterations=sampler_settings.online_analysis_minimum_iterations,
                flatness_criteria=sampler_settings.flatness_criteria,
                gamma0=sampler_settings.gamma0,
            )
        elif sampler_settings.sampler_method.lower() == 'independent':
            sampler = multistate.MultiStateSampler(
                mcmc_moves=integrator,
                online_analysis_interval=sampler_settings.online_analysis_interval,
                online_analysis_target_error=sampler_settings.online_analysis_target_error.m,
                online_analysis_minimum_iterations=sampler_settings.online_analysis_minimum_iterations
            )
        else:
            raise AttributeError(f"Unknown sampler {sampler_settings.sampler_method}")

        sampler.create(
                thermodynamic_states=cmp_states,
                sampler_states=sampler_states,
                storage=reporter
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
