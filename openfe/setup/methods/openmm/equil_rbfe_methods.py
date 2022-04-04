# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""Equilibrium RBFE methods using OpenMM in a Perses-like manner.

This module implements the necessary methodology toolking to run calculate a
ligand relative free energy transformation using OpenMM tools and either
Hamiltonian Replica Exchange or SAMS.

TODO
----
* Improve this docstring by adding an example use case.

"""
from __future__ import annotations

import os

import openmm
from openff.units import unit
import openmmtools
from openmmtools import multistate
from pydantic import BaseModel, validator
from typing import Dict, List, Union
from openmm import app
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
import openmmtools

from openfe.setup import (
    ChemicalSystem, LigandAtomMapping, SmallMoleculeComponent, SolventComponent,
)
from openfe.setup.methods import FEMethod
from openfe.setup import _rbfe_utils

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
    forcefield: Dict[str, Union[list[str], str]]
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


class SamplerSettings(BaseModel):
    """Settings for the Equilibrium sampler, currently supporting either
    HybridSAMSSampler or HybridRepexSampler.

    Attributes
    ----------
    sampler_method : str
      Sampler method to use, currently supports repex (hamiltonian replica
      exchange) and sams (self-adjusted mixture sampling). Default repex.
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
    flatness_criteria : str
      SAMS only. Method for assessing when to switch to asymptomatically
      optimal scheme.
      One of ['logZ-flatness', 'minimum-visits', 'histogram-flatness'].
      Default 'logZ-flatness'.
    gamma0 : float
      SAMS only. Initial weight adaptation rate. Default 0.0.

    TODO
    ----
    * Work out how this fits within the context of independent window FEPs.
    * It'd be great if we could pass in the sampler object rather than using
      strings to define which one we want.
    """
    class Config:
        arbitrary_types_allowed = True

    sampler_method = "repex"
    online_analysis_interval = Union[int, None]
    online_analysis_target_error = 0.2 * unit.boltzmann_constant * unit.kelvin
    online_analysis_minimum_iterations = 50
    flatness_criteria = 'logZ-flatness'
    gamma0 = 0.0

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


class RelativeLigandTransformResults:
    """Dict-like container for the output of a RelativeLigandTransform"""
    def __init__(self, settings: RelativeLigandTransformSettings):
        self._parent_settings = settings
        fn = self._parent_settings.simulation_settings.output_filename
        self._reporter = multistate.MultiStateReporter(fn)
        self._analyzer = multistate.MultiStateSamplerAnalyzer(self._reporter)

    def dG(self):
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
        dG, _ = self._analyzer.get_free_energy()
        dG = (dG[0, -1] * self._analyzer.kT).in_units_of(
            unit.kilocalories_per_mol)

        return dG

    def dG_error(self):
        """The uncertainty/error in the dG value"""
        _, error = self._analyzer.get_free_energy()
        error = (error[0, -1] * self._analyzer.kT).in_units_of(
            unit.kilocalories_per_mol)

        return error


class RelativeLigandTransform(FEMethod):
    """Calculates the relative free energy of an alchemical ligand transformation.

    """
    _SETTINGS_CLASS = RelativeLigandTransformSettings

    def __init__(self,
                 stateA: ChemicalState,
                 stateB: ChemicalState,
                 ligandmapping: LigandAtomMapping,
                 settings: RelativeLigandTransformSettings,
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

        """
        self._stateA = stateA
        self._stateB = stateB
        self._mapping = ligandmapping
        self._settings = settings

        # Checks on the inputs!
        # check that both states have solvent and ligand
        for state, label in [(stateA, 'A'), (stateB, 'B')]:
            if 'solvent' not in state.components:
                raise ValueError(f"Missing solvent in state {label}")
            if 'ligand' not in state.components:
                raise ValueError(f"Missing ligand in state {label}")
        # check that both states have same solvent
        # TODO: this is not always true if state A or B have a defined box etc... logic to be expanded
        if not stateA.components['solvent'] == stateB.components['solvent']:
            raise ValueError("Solvents aren't identical between states")
        # check that the mapping refers to the two ligand components
        if stateA.components['ligand'] != ligandmapping.molA:
            raise ValueError("Ligand in state A doesn't match mapping")
        if stateB.components['ligand'] != ligandmapping.molB:
            raise ValueError("Ligand in state B doesn't match mapping")

    @classmethod
    def get_default_settings(cls) -> RelativeLigandTransformSettings:
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

    def to_dict(self) -> dict:
        """Serialize to dict representation"""
        return {
            'stateA': self._stateA.to_dict(),
            'stateB': self._stateB.to_dict(),
            'mapping': self._mapping.to_dict(),
            'settings': dict(self._settings),
        }

    @classmethod
    def from_dict(cls, d: dict):
        """Deserialize from a dict representation"""
        return cls(
            stateA=ChemicalState.from_dict(d['stateA']),
            stateB=ChemicalState.from_dict(d['stateB']),
            mapping=LigandAtomMapping.from_dict(d['mapping']),
            settings=dict(**d['settings']),
        )

    def run(self) -> bool:
        stateA_openff_ligand = self._stateA.components['ligand'].to_openff()
        stateB_openff_ligand = self._stateB.components['ligand'].to_openff()

        ## Get smirnoff template generators
        smirnoff_stateA = SMIRNOFFTemplateGenerator(
            forcefield=self._settings.topology_settings.forcefield['ligand'],
            molecules=[stateA_openff_ligand],
        )

        smirnoff_stateB = SMIRNOFFTemplateGenerator(
            forcefield=self._settings.topology_settings.forcefield['ligand'],
            molecules=[stateB_openff_ligand],
        )

        ## Create forece fields and register them
        # state A
        omm_forcefield_stateA = app.ForceField(
            *[ff for (comp, ff) in self._settings.topology_settings.forcefield.items()
              if not comp == 'ligand']
        )

        omm_forcefield_stateA.registerTemplateGenerator(
                smirnoff_stateA.generator)

        # state B
        omm_forcefield_stateB = app.ForceField(
            *[ff for (comp, ff) in self._settings.topology_settings.forcefield.items()
              if not comp == 'ligand']
        )

        omm_forcefield_stateB.registerTemplateGenerator(
                smirnoff_stateB.generator)


        ##  Model state A
        # todo - add logic for PDBFile once it's been fixed upstream
        # if 'protein' in self._stateA.components:
        #     pdbfile = self._stateA.components['protein'].to_openmm_PDBFile()
        #     stateA_modeller = app.Modeller(pdbfile.topology,
        #                                    pdbfile.positions)
        #     stateA_modeller.add(
        #         stateA_openff_ligand.to_topology().to_openmm(),
        #         stateA_openff_ligand.conformers[0],
        #     )
        # else:
        # --- add below --
        stateA_modeller = app.Modeller(
            stateA_openff_ligand.to_topology().to_openmm(),
            stateA_openff_ligand.conformers[0],
        )

        # Solvate the complex in a `concentration` mM cubic water box with `solvent_padding` from the
        # solute to the edges of the box
        stateA_modeller.addSolvent(
            omm_forcefield_stateA,
            model=self._settings.topology_settings.solvent_model,
            padding=self._settings.solvent_padding,
            ionicStrength=self._stateA.components['solvent'].concentration,
        )

        ## Create OpenMM system + topology + initial positions for "A" system
        # Get nonbond method
        nonbonded_method = {
            # TODO: Other methods here
            'PME': app.PME,
            'NoCutoff': app.NoCutoff,
            'CutoffNonPeriodic': app.CutoffNonPeriodic,
            'CutoffPeriodic': app.CutoffPeriodic,
            'Ewald': app.Ewald
        }[self._settings.system_settings.nonbonded_method]

        constraints = {
            # TODO: Other constraints here
            'HBonds': app.HBonds,
        }[self._settings.system_settings.constraints]

        stateA_system = omm_forcefield_stateA.createSystem(
            stateA_modeller.topology,
            nonbondedMethod=nonbonded_method,
            nonbondedCutoff=self._settings.system_settings.nonbonded_cutoff,
            constraints=constraints,
            rigidWater=self._settings.system_settings.rigid_water,
        )

        stateA_topology = stateA_modeller.getTopology()

        # TODO: Algorithm this magic number
        # We center the system by adding (14, 14, 14) [box is ~ 28.6 A per side]
        stateA_positions = stateA_modeller.getPositions() + np.array([14, 14, 14]) * unit.angstrom

        # Remove the ligand from state A and replace with state B ligand
        stateB_topology = _rbfe_utils.append_new_topology_item(
            stateA_topology,
            stateB_openff_ligand.to_topology().to_openmm(),
            exclude_residue_name=stateA_openff_ligand.name,
        )

        stateB_system = omm_forcefield_stateB.createSystem(
            stateB_topology,
            nonbondedMethod=nonbonded_method,
            nonbondedCutoff=self._settings.system_settings.nonbonded_cutoff,
            constraints=constraints,
            rigidWater=self._settings.system_settings.rigid_water,
        )

        # Define mappings between the two systems
        ligand_mappings = _rbfe_utils.topologyhelpers.get_system_mappings(
            self._mapping.molA_to_molB,
            stateA_system, stateA_topology, stateA_openff_ligand.name,
            stateB_system, stateB_topology, stateB_openff_ligand.name,
            # TODO: Are these settings?
            fix_constraints=True,
            remove_element_changes=True,
        )

        stateB_positions = _rbfe_utils.topologyhelpers.set_and_check_new_positions(
            ligand_mappings, stateA_topology, stateB_topology,
            insert_positions=stateB_openff_ligand.conformers[0],
            # TODO: Remove this magic number
            shift_insert=np.array([14, 14, 14]),
        )

        alchem_settings = self._settings.alchemical_settings
        hybrid_factory = _rbfe_utils.relative.HybridTopologyFactory(
            stateA_system, stateA_positions, stateA_topology,
            stateB_system, stateB_positions, stateB_topology,
            old_to_new_atom_map=solvent_mappings['old_to_new_atom_map'],
            old_to_new_core_atom_map=solvent_mappings['old_to_new_core_atom_map'],
            use_dispersion_correction=alchem_settings.use_dispersion_correction,
            softcore_alpha=alchem_settings.softcore_alpha,
            softcore_LJ_v2=alchem_settings.softcore_LJ_v2,
            # TODO: Is this setting missing?
            softcore_LJ_v2_alpha=alchem_settings.softcore_alpha,
            softcore_electrostatics=alchem_settings.softcore_electrostatics,
            softcore_electrostatics_alpha=alchem_settings.softcore_electrostatics_alpha,
            softcore_sigma_Q=alchem_settings.softcore_sigma_Q,
            interpolate_old_and_new_14s=alchem_settings.interpolate_old_and_new_14s,
            flatten_torsions=alchem_settings.flatten_torsions,
        )
        hybrid_factory.addForce(
            openmm.MonteCarloBarostat(
                self._settings.barostat_settings.pressure.to(unit.bar).m,
                self._settings.integrator_settings.temperature.m,
                self._settings.barostat_settings.frequency.m,
            )
        )

        lambdas = _rbfe_utils.lambdaprotocol.LambdaProtocol(
            functions=alchem_settings.lambda_functions,
            windows=alchem_settings.lambda_windows
        )

        selection_indices = hybrid_factory.hybrid_topology.select('all')
        reporter = multistate.MultiStateReporter(
            self._settings.simulation_settings.output_filename,
            analysis_particle_indices=selection_indices,
            checkpoint_interval=self._settings.simulation_settings.checkpoint_interval.m,
        )

        # Get platform and context caches
        platform = _rbfe_utils.compute.get_openmm_platform(
            self._settings.engine_settings.compute_platform
        )
        energy_context_cache = openmmtools.cache.ContextCache(
            capacity=None, time_to_live=None, platform=platform,
        )
        sampler_context_cache = openmmtools.cache.ContextCache(
            capacity=None, time_to_live=None, platform=platform,
        )

        int_settings = self._settings.integrator_settings
        integrator = openmmtools.mcmc.LangevinSplittingDynamicsMove(
            timestep=int_settings.timestep,
            collision_rate=int_settings.collision_rate,
            n_steps=int_settings.n_steps.m,
            reassign_velocities=int_settings.reassign_velocities,
            n_restart_attempts=int_settings.n_restart_attempts,
            constraint_tolerance=int_settings.constraint_tolerance,
        )

        sampler = _rbfe_utils.multistate.HybridRepexSampler(
            mcmc_moves=integrator,
            hybrid_factory=hybrid_factory,
        )
        sampler.setup(
            reporter=reporter,
            platform=platform,
            lambda_protocol=lambdas,
            temperature=self._settings.integrator_settings.temperature,
            endstates=alchem_settings.unsampled_endstates,
        )
        sampler.energy_context_cache = energy_context_cache
        sampler.sampler_context_cache = sampler_context_cache

        return True

    def is_complete(self) -> bool:
        results_file = self._settings.simulation_settings.output_filename

        # TODO: Can improve upon this by checking expected length of the
        #       nc archive?
        return os.path.exists(results_file)

    def get_results(self) -> RelativeLigandTransformResults:
        """Return payload created by this workload

        Raises
        ------
        ValueError
          if the results do not exist yet
        """
        if not self.is_complete():
            raise ValueError("Results have not been generated")
        return RelativeLigandTransformResults(self._settings)
