# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""OpenMM Equilibrium AFE Protocol base classes
===============================================

Base classes for the equilibrium OpenMM absolute free energy ProtocolUnits.

Thist mostly implements BaseAbsoluteUnit whose methods can be
overriden to define different types of alchemical transformations.

TODO
----
* Add in all the AlchemicalFactory and AlchemicalRegion kwargs
  as settings.
* Allow for a more flexible setting of Lambda regions.
"""
from __future__ import annotations

import abc
import os
import logging

import gufe
from gufe.components import Component
import numpy as np
import numpy.typing as npt
import openmm
from openff.units import unit
from openff.units.openmm import from_openmm, to_openmm, ensure_quantity
from openff.toolkit.topology import Molecule as OFFMolecule
from openmmtools import multistate
from openmmtools.states import (SamplerState,
                                ThermodynamicState,
                                create_thermodynamic_state_protocol,)
from openmmtools.alchemy import (AlchemicalRegion, AbsoluteAlchemicalFactory,
                                 AlchemicalState,)
from typing import Dict, List, Optional
from openmm import app
from openmm import unit as omm_unit
from openmmforcefields.generators import SystemGenerator
import pathlib
from typing import Any
import openmmtools
import mdtraj as mdt

from gufe import (
    settings, ChemicalSystem, SmallMoleculeComponent,
    ProteinComponent, SolventComponent
)
from openfe.protocols.openmm_utils.omm_settings import (
    SettingsBaseModel,
)
from openfe.protocols.openmm_afe.equil_afe_settings import (
    SolvationSettings,
    AlchemicalSamplerSettings, OpenMMEngineSettings,
    IntegratorSettings, SimulationSettings,
)
from openfe.protocols.openmm_rfe._rfe_utils import compute
from ..openmm_utils import (
    settings_validation, system_creation,
    multistate_analysis
)

logger = logging.getLogger(__name__)


class BaseAbsoluteUnit(gufe.ProtocolUnit):
    """
    Base class for ligand absolute free energy transformations.
    """
    def __init__(self, *,
                 stateA: ChemicalSystem,
                 stateB: ChemicalSystem,
                 settings: settings.Settings,
                 alchemical_components: dict[str, list[Component]],
                 generation: int = 0,
                 repeat_id: int = 0,
                 name: Optional[str] = None,):
        """
        Parameters
        ----------
        stateA : ChemicalSystem
          ChemicalSystem containing the components defining the state at
          lambda 0.
        stateB : ChemicalSystem
          ChemicalSystem containing the components defining the state at
          lambda 1.
        settings : gufe.settings.Setings
          Settings for the Absolute Tranformation Protocol. This can be
          constructed by calling the
          :class:`AbsoluteTransformProtocol.get_default_settings` method
          to get a default set of settings.
        alchemical_components : dict[str, Component]
          the alchemical components for each state in this Unit
        name : str, optional
          Human-readable identifier for this Unit
        repeat_id : int, optional
          Identifier for which repeat (aka replica/clone) this Unit is,
          default 0
        generation : int, optional
          Generation counter which keeps track of how many times this repeat
          has been extended, default 0.
        """
        super().__init__(
            name=name,
            stateA=stateA,
            stateB=stateB,
            settings=settings,
            alchemical_components=alchemical_components,
            repeat_id=repeat_id,
            generation=generation,
        )

    @staticmethod
    def _get_alchemical_indices(omm_top: openmm.Topology,
                                comp_resids: dict[Component, npt.NDArray],
                                alchem_comps: dict[str, list[Component]]
                                ) -> list[int]:
        """
        Get a list of atom indices for all the alchemical species

        Parameters
        ----------
        omm_top : openmm.Topology
          Topology of OpenMM System.
        comp_resids : dict[Component, npt.NDArray]
          A dictionary of residues for each component in the System.
        alchem_comps : dict[str, list[Component]]
          A dictionary of alchemical components for each end state.

        Return
        ------
        atom_ids : list[int]
          A list of atom indices for the alchemical species
        """

        # concatenate a list of residue indexes for all alchemical components
        residxs = np.concatenate(
            [comp_resids[key] for key in alchem_comps['stateA']]
        )

        # get the alchemicical atom ids
        atom_ids = []

        for r in omm_top.residues():
            if r.index in residxs:
                atom_ids.extend([at.index for at in r.atoms()])

        return atom_ids

    @staticmethod
    def _pre_minimize(system: openmm.System,
                      positions: omm_unit.Quantity) -> npt.NDArray:
        """
        Short CPU minization of System to avoid GPU NaNs

        Parameters
        ----------
        system : openmm.System
          An OpenMM System to minimize.
        positionns : openmm.unit.Quantity
          Initial positions for the system.

        Returns
        -------
        minimized_positions : npt.NDArray
          Minimized positions
        """
        integrator = openmm.VerletIntegrator(0.001)
        context = openmm.Context(
            system, integrator,
            openmm.Platform.getPlatformByName('CPU'),
        )
        context.setPositions(positions)
        # Do a quick 100 steps minimization, usually avoids NaNs
        openmm.LocalEnergyMinimizer.minimize(
            context, maxIterations=100
        )
        state = context.getState(getPositions=True)
        minimized_positions = state.getPositions(asNumpy=True)
        return minimized_positions

    def _prepare(
        self, verbose: bool,
        scratch_basepath: Optional[pathlib.Path],
        shared_basepath: Optional[pathlib.Path],
    ):
        """
        Set basepaths and do some initial logging.

        Parameters
        ----------
        verbose : bool
          Verbose output of the simulation progress. Output is provided via
          INFO level logging.
        basepath : Optional[pathlib.Path]
          Optional base path to write files to.
        """
        self.verbose = verbose

        if self.verbose:
            self.logger.info("setting up alchemical system")

        # set basepaths
        def _set_optional_path(basepath):
            if basepath is None:
                return pathlib.Path('.')
            return basepath

        self.scratch_basepath = _set_optional_path(scratch_basepath)
        self.shared_basepath = _set_optional_path(shared_basepath)

    @abc.abstractmethod
    def _get_components(self) -> tuple[dict[str, list[Component]],
                                       Optional[gufe.SolventComponent],
                                       Optional[gufe.ProteinComponent],
                                       dict[SmallMoleculeComponent, OFFMolecule]]:
        """
        Get the relevant components to create the alchemical system with.

        Note
        ----
        Must be implemented in the child class.
        """
        ...

    @abc.abstractmethod
    def _handle_settings(self):
        """
        Get a dictionary with the following entries:
          * forcefield_settings : OpenMMSystemGeneratorFFSettings
          * thermo_settings : ThermoSettings
          * system_settings : SystemSettings
          * solvation_settings : SolvationSettings
          * alchemical_settings : AlchemicalSettings
          * sampler_settings : AlchemicalSamplerSettings
          * engine_settings : OpenMMEngineSettings
          * integrator_settings : IntegratorSettings
          * simulation_settings : SimulationSettings

        Settings may change depending on what type of simulation you are
        running. Cherry pick them and return them to be available later on.

        This method should also add various validation checks as necessary.

        Note
        ----
        Must be implemented in the child class.
        """
        ...

    def _get_system_generator(
        self, settings: dict[str, SettingsBaseModel],
        solvent_comp: Optional[SolventComponent]
    ) -> SystemGenerator:
        """
        Get a system generator through the system creation
        utilities

        Parameters
        ----------
        settings : dict[str, SettingsBaseModel]
          A dictionary of settings object for the unit.
        solvent_comp : Optional[SolventComponent]
          The solvent component of this system, if there is one.

        Returns
        -------
        system_generator : openmmforcefields.generator.SystemGenerator
          System Generator to parameterise this unit.
        """
        ffcache = settings['simulation_settings'].forcefield_cache
        if ffcache is not None:
            ffcache = self.shared_basepath / ffcache

        system_generator = system_creation.get_system_generator(
            forcefield_settings=settings['forcefield_settings'],
            thermo_settings=settings['thermo_settings'],
            system_settings=settings['system_settings'],
            cache=ffcache,
            has_solvent=solvent_comp is not None,
        )
        return system_generator

    def _get_modeller(
        self,
        protein_component: Optional[ProteinComponent],
        solvent_component: Optional[SolventComponent],
        smc_components: dict[SmallMoleculeComponent, OFFMolecule],
        system_generator: SystemGenerator,
        solvation_settings: SolvationSettings
    ) -> tuple[app.Modeller, dict[Component, npt.NDArray]]:
        """
        Get an OpenMM Modeller object and a list of residue indices
        for each component in the system.

        Parameters
        ----------
        protein_component : Optional[ProteinComponent]
          Protein Component, if it exists.
        solvent_component : Optional[ProteinCompoinent]
          Solvent Component, if it exists.
        smc_components : list[openff.toolkit.topology.Molecule]
          List of openff Molecules to add.
        system_generator : openmmforcefields.generator.SystemGenerator
          System Generator to parameterise this unit.
        solvation_settings : SolvationSettings
          Settings detailing how to solvate the system.

        Returns
        -------
        system_modeller : app.Modeller
          OpenMM Modeller object generated from ProteinComponent and
          OpenFF Molecules.
        comp_resids : dict[Component, npt.NDArray]
          Dictionary of residue indices for each component in system.
        """
        if self.verbose:
            self.logger.info("Parameterizing molecules")

        # force the creation of parameters for the small molecules
        # this is necessary because we need to have the FF generated ahead
        # of solvating the system.
        # Note by default this is cached to ctx.shared/db.json which should
        # reduce some of the costs.
        for mol in smc_components.values():
            # don't do this if we have user charges
            if not (mol.partial_charges is not None and np.any(mol.partial_charges)):
                try:
                    # try and follow official spec method
                    mol.assign_partial_charges('am1bcc')
                except ValueError:  # this is what a confgen failure yields
                    # but fallback to using existing conformer
                    mol.assign_partial_charges('am1bcc',
                                               use_conformers=mol.conformers)

            system_generator.create_system(
                mol.to_topology().to_openmm(), molecules=[mol]
            )

        # get OpenMM modeller + dictionary of resids for each component
        system_modeller, comp_resids = system_creation.get_omm_modeller(
            protein_comp=protein_component,
            solvent_comp=solvent_component,
            small_mols=smc_components,
            omm_forcefield=system_generator.forcefield,
            solvent_settings=solvation_settings,
        )

        return system_modeller, comp_resids

    def _get_omm_objects(
        self,
        system_modeller: app.Modeller,
        system_generator: SystemGenerator,
        smc_components: list[OFFMolecule],
    ) -> tuple[app.Topology, openmm.unit.Quantity, openmm.System]:
        """
        Get the OpenMM Topology, Positions and System of the
        parameterised system.

        Parameters
        ----------
        system_modeller : app.Modeller
          OpenMM Modeller object representing the system to be
          parametrized.
        system_generator : SystemGenerator
          SystemGenerator object to create a System with.
        smc_components : list
          A list of openff Molecules to add to the system.

        Returns
        -------
        topology : app.Topology
          Topology object describing the parameterized system
        system : openmm.System
          An OpenMM System of the alchemical system.
        positionns : openmm.unit.Quantity
          Positions of the system.
        """
        topology = system_modeller.getTopology()
        # roundtrip positions to remove vec3 issues
        positions = to_openmm(from_openmm(system_modeller.getPositions()))
        system = system_generator.create_system(
            system_modeller.topology,
            molecules=smc_components,
        )
        return topology, system, positions

    def _get_lambda_schedule(
        self, settings: dict[str, SettingsBaseModel]
    ) -> dict[str, npt.NDArray]:
        """
        Create the lambda schedule

        Parameters
        ----------
        settings : dict[str, SettingsBaseModel]
          Settings for the unit.

        Returns
        -------
        lambdas : dict[str, npt.NDArray]

        TODO
        ----
        * Augment this by using something akin to the RFE protocol's
          LambdaProtocol
        """
        lambdas = dict()
        n_elec = settings['alchemical_settings'].lambda_elec_windows
        n_vdw = settings['alchemical_settings'].lambda_vdw_windows + 1
        lambdas['lambda_electrostatics'] = np.concatenate(
                [np.linspace(1, 0, n_elec), np.linspace(0, 0, n_vdw)[1:]]
        )
        lambdas['lambda_sterics'] = np.concatenate(
                [np.linspace(1, 1, n_elec), np.linspace(1, 0, n_vdw)[1:]]
        )

        n_replicas = settings['sampler_settings'].n_replicas

        if n_replicas != (len(lambdas['lambda_sterics'])):
            errmsg = (f"Number of replicas {n_replicas} "
                      "does not equal the number of lambda windows ")
            raise ValueError(errmsg)

        return lambdas

    def _add_restraints(self, system, topology, settings):
        """
        Placeholder method to add restraints if necessary
        """
        return

    def _get_alchemical_system(
        self,
        topology: app.Topology,
        system: openmm.System,
        comp_resids: dict[Component, npt.NDArray],
        alchem_comps: dict[str, list[Component]]
    ) -> tuple[AbsoluteAlchemicalFactory, openmm.System, list[int]]:
        """
        Get an alchemically modified system and its associated factory

        Parameters
        ----------
        topology : openmm.Topology
          Topology of OpenMM System.
        system : openmm.System
          System to alchemically modify.
        comp_resids : dict[str, npt.NDArray]
          A dictionary of residues for each component in the System.
        alchem_comps : dict[str, list[Component]]
          A dictionary of alchemical components for each end state.


        Returns
        -------
        alchemical_factory : AbsoluteAlchemicalFactory
          Factory for creating an alchemically modified system.
        alchemical_system : openmm.System
          Alchemically modified system
        alchemical_indices : list[int]
          A list of atom indices for the alchemically modified
          species in the system.

        TODO
        ----
        * Add support for all alchemical factory options
        """
        alchemical_indices = self._get_alchemical_indices(
            topology, comp_resids, alchem_comps
        )

        alchemical_region = AlchemicalRegion(
            alchemical_atoms=alchemical_indices,
        )

        alchemical_factory = AbsoluteAlchemicalFactory()
        alchemical_system = alchemical_factory.create_alchemical_system(
            system, alchemical_region
        )

        return alchemical_factory, alchemical_system, alchemical_indices

    def _get_states(
        self,
        alchemical_system: openmm.System,
        positions: openmm.unit.Quantity,
        settings: dict[str, SettingsBaseModel],
        lambdas: dict[str, npt.NDArray],
        solvent_comp: Optional[SolventComponent],
    ) -> tuple[list[SamplerState], list[ThermodynamicState]]:
        """
        Get a list of sampler and thermodynmic states from an
        input alchemical system.

        Parameters
        ----------
        alchemical_system : openmm.System
          Alchemical system to get states for.
        positions : openmm.unit.Quantity
          Positions of the alchemical system.
        settings : dict[str, SettingsBaseModel]
          A dictionary of settings for the protocol unit.
        lambdas : dict[str, npt.NDArray]
          A dictionary of lambda scales.
        solvent_comp : Optional[SolventComponent]
          The solvent component of the system, if there is one.

        Returns
        -------
        sampler_states : list[SamplerState]
          A list of SamplerStates for each replica in the system.
        cmp_states : list[ThermodynamicState]
          A list of ThermodynamicState for each replica in the system.
        """
        alchemical_state = AlchemicalState.from_system(alchemical_system)
        # Set up the system constants
        temperature = settings['thermo_settings'].temperature
        pressure = settings['thermo_settings'].pressure
        constants = dict()
        constants['temperature'] = ensure_quantity(temperature, 'openmm')
        if solvent_comp is not None:
            constants['pressure'] = ensure_quantity(pressure, 'openmm')

        cmp_states = create_thermodynamic_state_protocol(
            alchemical_system, protocol=lambdas,
            constants=constants, composable_states=[alchemical_state],
        )

        sampler_state = SamplerState(positions=positions)
        if alchemical_system.usesPeriodicBoundaryConditions():
            box = alchemical_system.getDefaultPeriodicBoxVectors()
            sampler_state.box_vectors = box

        sampler_states = [sampler_state for _ in cmp_states]

        return sampler_states, cmp_states

    def _get_reporter(
        self,
        topology: app.Topology,
        positions: openmm.unit.Quantity,
        simulation_settings: SimulationSettings,
    ) -> multistate.MultiStateReporter:
        """
        Get a MultistateReporter for the simulation you are running.

        Parameters
        ----------
        topology : app.Topology
          A Topology of the system being created.
        simulation_settings : SimulationSettings
          Settings for the simulation.

        Returns
        -------
        reporter : multistate.MultiStateReporter
          The reporter for the simulation.
        """
        mdt_top = mdt.Topology.from_openmm(topology)

        selection_indices = mdt_top.select(
                simulation_settings.output_indices
        )

        nc = self.shared_basepath / simulation_settings.output_filename
        chk = self.shared_basepath / simulation_settings.checkpoint_storage

        reporter = multistate.MultiStateReporter(
            storage=nc,
            analysis_particle_indices=selection_indices,
            checkpoint_interval=simulation_settings.checkpoint_interval.m,
            checkpoint_storage=chk,
        )

        # Write out the structure's PDB whilst we're here
        if len(selection_indices) > 0:
            traj = mdt.Trajectory(
                positions[selection_indices, :],
                mdt_top.subset(selection_indices),
            )
            traj.save_pdb(
                self.shared_basepath / simulation_settings.output_structure
            )

        return reporter

    def _get_ctx_caches(
        self,
        engine_settings: OpenMMEngineSettings
    ) -> tuple[openmmtools.cache.ContextCache, openmmtools.cache.ContextCache]:
        """
        Set the context caches based on the chosen platform

        Parameters
        ----------
        engine_settings : OpenMMEngineSettings,

        Returns
        -------
        energy_context_cache : openmmtools.cache.ContextCache
          The energy state context cache.
        sampler_context_cache : openmmtools.cache.ContextCache
          The sampler state context cache.
        """
        platform = compute.get_openmm_platform(
            engine_settings.compute_platform,
        )

        energy_context_cache = openmmtools.cache.ContextCache(
            capacity=None, time_to_live=None, platform=platform,
        )

        sampler_context_cache = openmmtools.cache.ContextCache(
            capacity=None, time_to_live=None, platform=platform,
        )

        return energy_context_cache, sampler_context_cache

    def _get_integrator(
        self,
        integrator_settings: IntegratorSettings
    ) -> openmmtools.mcmc.LangevinDynamicsMove:
        """
        Return a LangevinDynamicsMove integrator

        Parameters
        ----------
        integrator_settings : IntegratorSettings

        Returns
        -------
        integrator : openmmtools.mcmc.LangevinDynamicsMove
          A configured integrator object.
        """
        integrator = openmmtools.mcmc.LangevinDynamicsMove(
            timestep=to_openmm(integrator_settings.timestep),
            collision_rate=to_openmm(integrator_settings.collision_rate),
            n_steps=integrator_settings.n_steps.m,
            reassign_velocities=integrator_settings.reassign_velocities,
            n_restart_attempts=integrator_settings.n_restart_attempts,
            constraint_tolerance=integrator_settings.constraint_tolerance,
        )

        return integrator

    def _get_sampler(
        self,
        integrator: openmmtools.mcmc.LangevinDynamicsMove,
        reporter: openmmtools.multistate.MultiStateReporter,
        sampler_settings: AlchemicalSamplerSettings,
        cmp_states: list[ThermodynamicState],
        sampler_states: list[SamplerState],
        energy_context_cache: openmmtools.cache.ContextCache,
        sampler_context_cache: openmmtools.cache.ContextCache
    ) -> multistate.MultiStateSampler:
        """
        Get a sampler based on the equilibrium sampling method requested.

        Parameters
        ----------
        integrator : openmmtools.mcmc.LangevinDynamicsMove
          The simulation integrator.
        reporter : openmmtools.multistate.MultiStateReporter
          The reporter to hook up to the sampler.
        sampler_settings : AlchemicalSamplerSettings
          Settings for the alchemical sampler.
        cmp_states : list[ThermodynamicState]
          A list of thermodynamic states to sample.
        sampler_states : list[SamplerState]
          A list of sampler states.
        energy_context_cache : openmmtools.cache.ContextCache
          Context cache for the energy states.
        sampler_context_cache : openmmtool.cache.ContextCache
          Context cache for the sampler states.

        Returns
        -------
        sampler : multistate.MultistateSampler
          A sampler configured for the chosen sampling method.
        """

        # Select the right sampler
        # Note: doesn't need else, settings already validates choices
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

        sampler.create(
            thermodynamic_states=cmp_states,
            sampler_states=sampler_states,
            storage=reporter
        )

        sampler.energy_context_cache = energy_context_cache
        sampler.sampler_context_cache = sampler_context_cache

        return sampler

    def _run_simulation(
        self,
        sampler: multistate.MultiStateSampler,
        reporter: multistate.MultiStateReporter,
        settings: dict[str, SettingsBaseModel],
        dry: bool
    ):
        """
        Run the simulation.

        Parameters
        ----------
        sampler : multistate.MultiStateSampler
          The sampler associated with the simulation to run.
        reporter : multistate.MultiStateReporter
          The reporter associated with the sampler.
        settings : dict[str, SettingsBaseModel]
          The dictionary of settings for the protocol.
        dry : bool
          Whether or not to dry run the simulation

        Returns
        -------
        unit_results_dict : Optional[dict]
          A dictionary containing all the free energy results,
          if not a dry run.
        """
        # Get the relevant simulation steps
        mc_steps = settings['integrator_settings'].n_steps.m

        equil_steps, prod_steps = settings_validation.get_simsteps(
            equil_length=settings['simulation_settings'].equilibration_length,
            prod_length=settings['simulation_settings'].production_length,
            timestep=settings['integrator_settings'].timestep,
            mc_steps=mc_steps,
        )

        if not dry:  # pragma: no-cover
            # minimize
            if self.verbose:
                self.logger.info("minimizing systems")

            sampler.minimize(
                max_iterations=settings['simulation_settings'].minimization_steps
            )

            # equilibrate
            if self.verbose:
                self.logger.info("equilibrating systems")

            sampler.equilibrate(int(equil_steps / mc_steps))  # type: ignore

            # production
            if self.verbose:
                self.logger.info("running production phase")

            sampler.extend(int(prod_steps / mc_steps))  # type: ignore

            if self.verbose:
                self.logger.info("production phase complete")

            if self.verbose:
                self.logger.info("post-simulation result analysis")

            analyzer = multistate_analysis.MultistateEquilFEAnalysis(
                reporter,
                sampling_method=settings['sampler_settings'].sampler_method.lower(),
                result_units=unit.kilocalorie_per_mole
            )
            analyzer.plot(filepath=self.shared_basepath, filename_prefix="")
            analyzer.close()

            return analyzer.unit_results_dict

        else:
            # close reporter when you're done, prevent file handle clashes
            reporter.close()

            # clean up the reporter file
            fns = [self.shared_basepath / settings['simulation_settings'].output_filename,
                   self.shared_basepath / settings['simulation_settings'].checkpoint_storage]
            for fn in fns:
                os.remove(fn)

            return None

    def run(self, dry=False, verbose=True,
            scratch_basepath=None, shared_basepath=None) -> Dict[str, Any]:
        """Run the absolute free energy calculation.

        Parameters
        ----------
        dry : bool
          Do a dry run of the calculation, creating all necessary alchemical
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

        Attributes
        ----------
        solvent : Optional[SolventComponent]
          SolventComponent to be applied to the system
        protein : Optional[ProteinComponent]
          ProteinComponent for the system
        openff_mols : List[openff.Molecule]
          List of OpenFF Molecule objects for each SmallMoleculeComponent in
          the stateA ChemicalSystem
        """
        # 0. Generaly preparation tasks
        self._prepare(verbose, scratch_basepath, shared_basepath)

        # 1. Get components
        alchem_comps, solv_comp, prot_comp, smc_comps = self._get_components()

        # 2. Get settings
        settings = self._handle_settings()

        # 3. Get system generator
        system_generator = self._get_system_generator(settings, solv_comp)

        # 4. Get modeller
        system_modeller, comp_resids = self._get_modeller(
            prot_comp, solv_comp, smc_comps, system_generator,
            settings['solvation_settings']
        )

        # 5. Get OpenMM topology, positions and system
        omm_topology, omm_system, positions = self._get_omm_objects(
            system_modeller, system_generator, list(smc_comps.values())
        )

        # 6. Pre-minimize System (Test + Avoid NaNs)
        positions = self._pre_minimize(omm_system, positions)

        # 7. Get lambdas
        lambdas = self._get_lambda_schedule(settings)

        # 8. Add restraints
        self._add_restraints(omm_system, omm_topology, settings)

        # 9. Get alchemical system
        alchem_factory, alchem_system, alchem_indices = self._get_alchemical_system(
            omm_topology, omm_system, comp_resids, alchem_comps
        )

        # 10. Get compound and sampler states
        sampler_states, cmp_states = self._get_states(
            alchem_system, positions, settings,
            lambdas, solv_comp
        )

        # 11. Create the multistate reporter & create PDB
        reporter = self._get_reporter(
            omm_topology, positions,
            settings['simulation_settings'],
        )

        # Wrap in try/finally to avoid memory leak issues
        try:
            # 12. Get context caches
            energy_ctx_cache, sampler_ctx_cache = self._get_ctx_caches(
                settings['engine_settings']
            )

            # 13. Get integrator
            integrator = self._get_integrator(settings['integrator_settings'])

            # 14. Get sampler
            sampler = self._get_sampler(
                integrator, reporter, settings['sampler_settings'],
                cmp_states, sampler_states,
                energy_ctx_cache, sampler_ctx_cache
            )

            # 15. Run simulation
            unit_result_dict = self._run_simulation(
                sampler, reporter, settings, dry
            )

        finally:
            # close reporter when you're done to prevent file handle clashes
            reporter.close()

            # clear GPU context
            # Note: use cache.empty() when openmmtools #690 is resolved
            for context in list(energy_ctx_cache._lru._data.keys()):
                del energy_ctx_cache._lru._data[context]
            for context in list(sampler_ctx_cache._lru._data.keys()):
                del sampler_ctx_cache._lru._data[context]
            # cautiously clear out the global context cache too
            for context in list(
                    openmmtools.cache.global_context_cache._lru._data.keys()):
                del openmmtools.cache.global_context_cache._lru._data[context]

            del sampler_ctx_cache, energy_ctx_cache

            # Keep these around in a dry run so we can inspect things
            if not dry:
                del integrator, sampler

        if not dry:
            nc = self.shared_basepath / settings['simulation_settings'].output_filename
            chk = self.shared_basepath / settings['simulation_settings'].checkpoint_storage
            return {
                'nc': nc,
                'last_checkpoint': chk,
                **unit_result_dict,
            }
        else:
            return {'debug': {'sampler': sampler}}
