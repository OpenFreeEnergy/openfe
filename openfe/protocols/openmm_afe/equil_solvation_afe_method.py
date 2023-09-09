# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""OpenMM Equilibrium Solvation AFE Protocol --- :mod:`openfe.protocols.openmm_afe.equil_solvation_afe_methods`
===============================================================================================================

This module implements the necessary methodology tooling to run calculate an
absolute solvation free energy using OpenMM tools and one of the following
alchemical sampling methods:

* Hamiltonian Replica Exchange
* Self-adjusted mixture sampling
* Independent window sampling

Current limitations
-------------------
* Disapearing molecules are only allowed in state A. Support for
  appearing molecules will be added in due course.
* Only small molecules are allowed to act as alchemical molecules.
  Alchemically changing protein or solvent components would induce
  perturbations which are too large to be handled by this Protocol.


Acknowledgements
----------------
* Originally based on hydration.py in
  `espaloma <https://github.com/choderalab/espaloma_charge>`_


TODO
----
* Add in all the AlchemicalFactory and AlchemicalRegion kwargs
  as settings.
* Allow for a more flexible setting of Lambda regions.
* Add support for restraints.
* Improve this docstring by adding an example use case.

"""
from __future__ import annotations

import os
import logging

from collections import defaultdict
import gufe
from gufe.components import Component
import numpy as np
import numpy.typing as npt
import openmm
from openff.toolkit import Molecule as OFFMol
from openff.units import unit
from openff.units.openmm import from_openmm, to_openmm, ensure_quantity
from openmmtools import multistate
from openmmtools.states import (SamplerState,
                                create_thermodynamic_state_protocol,)
from openmmtools.alchemy import (AlchemicalRegion, AbsoluteAlchemicalFactory,
                                 AlchemicalState,)
from typing import Dict, List, Optional, Tuple
from openmm import app
from openmm import unit as omm_unit
from openmmforcefields.generators import SystemGenerator
import pathlib
from typing import Any, Iterable
import openmmtools
import uuid
import mdtraj as mdt

from gufe import (
    settings, ChemicalSystem, SmallMoleculeComponent,
    ProteinComponent, SolventComponent
)
from openfe.protocols.openmm_afe.equil_afe_settings import (
    AbsoluteTransformSettings, SystemSettings,
    SolvationSettings, AlchemicalSettings,
    AlchemicalSamplerSettings, OpenMMEngineSettings,
    IntegratorSettings, SimulationSettings,
)
from openfe.protocols.openmm_rfe._rfe_utils import compute
from ..openmm_utils import (
    system_validation, settings_validation, system_creation
)


logger = logging.getLogger(__name__)


class AbsoluteTransformProtocolResult(gufe.ProtocolResult):
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

    def get_rate_of_convergence(self):  # pragma: no-cover
        raise NotImplementedError


class AbsoluteSolvationProtocol(gufe.Protocol):
    result_cls = AbsoluteTransformProtocolResult
    _settings: AbsoluteTransformSettings

    @classmethod
    def _default_settings(cls):
        """A dictionary of initial settings for this creating this Protocol

        These settings are intended as a suitable starting point for creating
        an instance of this protocol.  It is recommended, however that care is
        taken to inspect and customize these before performing a Protocol.

        Returns
        -------
        Settings
          a set of default settings
        """
        return AbsoluteTransformSettings(
            forcefield_settings=settings.OpenMMSystemGeneratorFFSettings(),
            thermo_settings=settings.ThermoSettings(
                temperature=298.15 * unit.kelvin,
                pressure=1 * unit.bar,
            ),
            solvent_system_settings=SystemSettings(),
            vacuum_system_settings=SystemSettings(nonbonded_method='nocutoff'),
            alchemical_settings=AlchemicalSettings(),
            alchemsampler_settings=AlchemicalSamplerSettings(),
            solvation_settings=SolvationSettings(),
            engine_settings=OpenMMEngineSettings(),
            integrator_settings=IntegratorSettings(),
            solvent_simulation_settings=SimulationSettings(
                equilibration_length=1.0 * unit.nanosecond,
                production_length=10.0 * unit.nanosecond,
                output_filename='solvent.nc',
                checkpoint_storage='solvent_checkpoint.nc',
            ),
            vacuum_simulation_settings=SimulationSettings(
                equilibration_length=0.5 * unit.nanosecond,
                production_length=2.0 * unit.nanosecond,
                output_filename='vacuum.nc',
                checkpoint_storage='vacuum_checkpoint.nc'
            ),
        )

    @staticmethod
    def _validate_solvent_endstates(
        stateA: ChemicalSystem, stateB: ChemicalSystem,
    ) -> None:
        """
        A solvent transformation is defined (in terms of gufe components)
        as starting from a ligand in solvent and ending up just in solvent.

        Parameters
        ----------
        stateA : ChemicalSystem
          The chemical system of end state A
        stateB : ChemicalSystem
          The chemical system of end state B

        Raises
        ------
        ValueError
          If stateB contains anything else but a SolventComponent.
          If stateA contains a ProteinComponent
        """
        if ((len(stateB) != 1) or
            (not isinstance(stateB.values()[0], SolventComponent))):
            errmsg = "Only a single SolventComponent is allowed in stateB"
            raise ValueError(errmsg)

        for comp in stateA.values():
            if isinstance(comp, ProteinComponent):
                errmsg = ("Protein components are not allow for "
                          "absolute solvation free energies")
                raise ValueError(errmsg)

    @staticmethod
    def _validate_alchemical_components(
        alchemical_components: dict[str, list[Component]]
    ) -> None:
        """
        Checks that the ChemicalSystem alchemical components are correct.

        Parameters
        ----------
        alchemical_components : Dict[str, list[Component]]
          Dictionary containing the alchemical components for
          stateA and stateB.

        Raises
        ------
        ValueError
          If there are alchemical components in state B.
          If there are non SmallMoleculeComponent alchemical species.
          If there are more than one alchemical species.

        Notes
        -----
        * Currently doesn't support alchemical components in state B.
        * Currently doesn't support alchemical components which are not
          SmallMoleculeComponents.
        * Currently doesn't support more than one alchemical component
          being desolvated.
        """

        # Crash out if there are any alchemical components in state B for now
        if len(alchemical_components['stateB']) > 0:
            errmsg = ("Components appearing in state B are not "
                      "currently supported")
            raise ValueError(errmsg)
        
        if len(alchemical_components['stateA']) > 1:
            errmsg = ("More than one alchemical components is not supported "
                      "for absolute solvation free energies")

        # Crash out if any of the alchemical components are not
        # SmallMoleculeComponent
        for comp in alchemical_components['stateA']:
            if not isinstance(comp, SmallMoleculeComponent):
                errmsg = ("Non SmallMoleculeComponent alchemical species "
                          "are not currently supported")
                raise ValueError(errmsg)

    def _create(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: Optional[Dict[str, gufe.ComponentMapping]] = None,
        extends: Optional[gufe.ProtocolDAGResult] = None,
    ) -> list[gufe.ProtocolUnit]:
        # TODO: extensions
        if extends:  # pragma: no-cover
            raise NotImplementedError("Can't extend simulations yet")

        # Validate components and get alchemical components
        self._validate_solvation_endstates(stateA, stateB)
        alchem_comps = system_validation.get_alchemical_components(
            stateA, stateB,
        )
        self._validate_alchemical_components(alchem_comps)

        # Check nonbond & solvent compatibility
        solv_nonbonded_method = self.settings.solvent_system_settings.nonbonded_method
        vac_nonbonded_method = self.settings.vacuum_system_settings.nonbonded_method
        # Use the more complete system validation solvent checks
        system_validation.validate_solvent(stateA, solv_nonbonded_method)
        # Gas phase is always gas phase
        assert vac_nonbonded_method.lower() != 'pme'

        # Get the name of the alchemical species
        alchname = alchem_comps['stateA'][0].name

        # Create list units for vacuum and solvent transforms

        solvent_units = [
            AbsoluteSolventTransformUnit(
                stateA=stateA, stateB=stateB,
                settings=self.settings,
                alchemical_components=alchemical_comps,
                generation=0, repeat_id=i,
                name=(f"Absolute Solvation, {alchname} solvent leg: "
                      f"repeat {i} generation 0"),
            )
            for i in range(self.settings.alchemsampler_settings.n_repeats)
        ]

        vacuum_units = [
            AbsoluteVacuumTransformUnit(
                # These don't really reflect the actual transform
                # Should these be overriden to be ChemicalSystem{smc} -> ChemicalSystem{} ?
                stateA=stateA, stateB=stateB,
                settings=self.settings,
                alchemical_components=alchemical_comps,
                generation=0, repeat_id=i,
                name=(f"Absolute Solvation, {alchname} solvent leg: "
                      f"repeat {i} generation 0"),
            )
            for i in range(self.settings.alchemsampler_settings.n_repeats)
        ]

        return solvent_units + vacuum_units

    # TODO: update to match new unit list
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
        for rep_id, rep_data in sorted(repeats.items()):
            # then sort within a repeat according to generation
            nc_paths = [
                ncpath for gen, ncpath, nc_check in sorted(rep_data)
            ]
            chk_files = [
                nc_check for gen, ncpath, nc_check in sorted(rep_data)
            ]
            data.append({'nc_paths': nc_paths,
                         'checkpoint_paths': chk_files})

        return {
            'nc_files': data,
        }


class BaseAbsoluteTransformUnit(gufe.ProtocolUnit):
    """
    Base class for ligand absolute free energy transformations.
    """
    def __init__(self, *,
                 stateA: ChemicalSystem,
                 stateB: ChemicalSystem,
                 settings: settings.Settings,
                 alchemical_components: Dict[str, List[str]],
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
                                comp_resids: Dict[str, npt.NDArray],
                                alchem_comps: Dict[str, List[str]]
                                ) -> List[int]:
        """
        Get a list of atom indices for all the alchemical species

        Parameters
        ----------
        omm_top : openmm.Topology
          Topology of OpenMM System.
        comp_resids : Dict[str, npt.NDArray]
          A dictionary of residues for each component in the System.
        alchem_comps : Dict[str, List[str]]
          A dictionary of alchemical components for each end state.

        Return
        ------
        atom_ids : List[int]
          A list of atom indices for the alchemical species
        """

        # concatenate a list of residue indexes for all alchemical components
        residxs = np.concatenate(
            [comp_resids[key] for key in alchem_comps['stateA']]
        )

        # get the alchemicical residues from the topology
        alchres = [
            r for r in omm_top.residues() if r.index in residxs
        ]

        atom_ids = []

        for res in alchres:
            atom_ids.extend([at.index for at in res.atoms()])

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
            logger.info("setting up alchemical system")

        # set basepaths
        def _set_optional_path(basepath):
            if basepath is None:
                return pathlib.Path('.')
            return basepath
        
        self.scratch_basepath = _set_optional_path(scratch_basepath)
        self.shared_basepath = _set_optional_path(shared_basepath)

    def _get_components(self):
        """
        Get the relevant components to create the alchemical system with.

        Note
        ----
        Must be implemented in child class.

        Returns
        -------
        alchem_comps : ..
        solv_comp : ..
        prot_comp : ..
        smc_comps : ..

        To move:
        stateA = self._inputs['stateA']
        alchem_comps = self._inputs['alchemical_components']
        # Get the relevant solvent & protein components & openff molecules
        solvent_comp, protein_comp, off_mols = self._parse_components(stateA)
        """
        raise NotImplementedError
    
    def _get_settings(self):
        """
        Settings may change depending on what type of simulation you are
        running. Cherry pick them and return them to be available later on.

        Base case would be:
        protocol_settings: RelativeHybridTopologyProtocolSettings = self._inputs['settings']

        Also add:
        # a. Validation checks
        settings_validation.validate_timestep(
            settings.forcefield_settings.hydrogen_mass,
            settings.integrator_settings.timestep
        )
        """
        raise NotImplementedError
    
    def _get_system_generator(self, settings, solvent_comp):
        """
        Get a system generator through the system creation
        utilities

        Parameters
        ----------
        settings :
        solv_comp :
        """
        ffcache = settings.simulation_settings.forcefield_cache
        if ffcache is not None:
            ffcache = self.shared_basepath / ffcache

        system_generator = system_creation.get_system_generator(
            forcefield_settings=settings.forcefield_settings,
            thermo_settings=settings.thermo_settings,
            cache=ffcache,
            has_solvent=solvent_comp is not None,
        )
        return system_generator
    
    def _get_modeller(self, protein_component, solvent_component,
                      smc_components, system_generator, settings):
        """
        # force the creation of parameters for the small molecules
        # this is cached and shouldn't incur further cost
        for mol in off_mols.values():
            system_generator.create_system(mol.to_topology().to_openmm(),
                                           molecules=[mol])

        # b. Get OpenMM Modller + a dictionary of resids for each component
        system_modeller, comp_resids = self._get_omm_modeller(
            protein_comp, solvent_comp, off_mols, system_generator.forcefield,
            settings.solvent_settings
        )

        """
        if self.verbose:
            logger.info("Parameterizing molecules")

        # force the creation of parameters for the small molecules
        # this is necessary because we need to have the FF generated ahead
        # of solvating the system.
        # Note by default this is cached to ctx.shared/db.json which should
        # reduce some of the costs.
        for comp in smc_components:
            offmol = comp.to_openff()
            system_generator.create_system(
                offmol.to_topology().to_openmm(), molecules=[offmol]
            )

        # get OpenMM modeller + dictionary of resids for each component
        system_modeller, comp_resids = system_creation.get_omm_modeller(
            protein_comp=protein_component,
            solvent_comp=solvent_component,
            small_mols=smc_components,
            omm_forcefield=system_generator.forcefield,
            solvent_settings=settings.solvation_settings,
        )

        return system_modeller, comp_resids

    def _get_omm_objects(self, system_modeller, system_generator,
                         smc_components):
        """
        system_topology = system_modeller.getTopology()

        # roundtrip via off_units to canocalize
        positions = to_openmm(from_openmm(system_modeller.getPositions()))

        omm_system = system_generator.create_system(
            system_modeller.topology,
            molecules=list(off_mols.values())
        )
        """
        topology = system_modeller.getTopology()
        # roundtrip positions to remove vec3 issues
        positions = to_openmm(from_openmm(system_modeller.getPositions()))
        system = system_generator.create_system(
            system_modeller.topology,
            molecules=[s.to_openff() for s in smc_components]
        )
        return topology, positions, system

    def _get_lambda_schedule(self, settings):
        """
        Create the lambda schedule
          TODO: do this properly using LambdaProtocol
          TODO: double check we definitely don't need to define
                temperature & pressure (pressure sure that's the case)
        """
        lambdas = dict()
        n_elec = settings.alchemical_settings.lambda_elec_windows
        n_vdw = settings.alchemical_settings.lambda_vdw_windows + 1
        lambdas['lambda_electrostatics'] = np.concatenate(
                [np.linspace(1, 0, n_elec), np.linspace(0, 0, n_vdw)[1:]]
        )
        lambdas['lambda_sterics'] = np.concatenate(
                [np.linspace(1, 1, n_elec), np.linspace(1, 0, n_vdw)[1:]]
        )

        n_replicas = settings.alchemical_sampler_settings.n_replicas

        if n_replicas != (len(lambdas['lambda_sterics'])):
            errmsg = (f"Number of replicas {n_replicas} "
                      "does not equal the number of lambda windows ")
            raise ValueError(errmsg)
        
        return lambdas

    def _get_alchemical_system(self, topology, system, comp_resids,
                               alchem_comps):
        """
        # TODO: add support for all the variants here
        # TODO: check that adding indices this way works
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

    def _get_states(self, alchemical_system, positions, settings, lambdas, solvent_comp):
        """
        """
        alchemical_state = AlchemicalState.from_system(alchemical_system)
        # Set up the system constants
        temperature = settings.thermo_settings.temperature
        pressure = settings.thermo_settings.pressure
        constants = dict()
        constants['temperature'] = ensure_quantity(temperature, 'openmm')
        if solvent_comp is not None:
            constants['pressure'] = ensure_quantity(pressure, 'openmm')
        
        cmp_states = create_thermodynamic_state_protocol(
            alchemical_system, protocol=lambdas,
            consatnts=constants, composable_states=[alchemical_state],
        )

        sampler_state = SamplerState(positions=positions)
        if alchemical_system.usesPeriodicBoundaryConditions():
            box = alchemical_system.getDefaultPeriodicBoxVectors()
            sampler_state.box_vectors = box

        sampler_states = [sampler_state for _ in cmp_states]

        return sampler_states, cmp_states

    def _get_reporter(self, ...):
        """
        # a. Get the sub selection of the system to print coords for
        mdt_top = mdt.Topology.from_openmm(system_topology)
        selection_indices = mdt_top.select(
                sim_settings.output_indices
        )

        # b. Create the multistate reporter
        reporter = multistate.MultiStateReporter(
            storage=basepath / sim_settings.output_filename,
            analysis_particle_indices=selection_indices,
            checkpoint_interval=sim_settings.checkpoint_interval.m,
            checkpoint_storage=basepath / sim_settings.checkpoint_storage,
        )
        """
        mdt_top = mdt.Topology.from_openmm(system_topology)
        selection_indices = mdt_top.select(
                sim_settings.output_indices
        )
        sim_settings = settings.simulation_settings
        reporter = multistate.MultiStateReporter(
            storage=self.shared_basepathbasepath / sim_settings.output_filename,
            analysis_particle_indices=selection_indices,
            checkpoint_interval=sim_settings.checkpoint_interval.m,
            checkpoint_storage=basepath / sim_settings.checkpoint_storage,
        )

    def _get_ctx_caches(self, ...):
        """
        # 7. Get platform and context caches
        platform = compute.get_openmm_platform(
            settings.engine_settings.compute_platform
        )

        # a. Create context caches (energy + sampler)
        #    Note: these needs to exist on the compute node
        energy_context_cache = openmmtools.cache.ContextCache(
            capacity=None, time_to_live=None, platform=platform,
        )

        sampler_context_cache = openmmtools.cache.ContextCache(
            capacity=None, time_to_live=None, platform=platform,
        )
        """
        ...

    def _get_integrator(self, integrator_settings):
        """
        # 8. Set the integrator
        # a. get integrator settings
        integrator_settings = settings.integrator_settings

        # b. create langevin integrator
        integrator = openmmtools.mcmc.LangevinSplittingDynamicsMove(
            timestep=to_openmm(integrator_settings.timestep),
            collision_rate=to_openmm(integrator_settings.collision_rate),
            n_steps=integrator_settings.n_steps.m,
            reassign_velocities=integrator_settings.reassign_velocities,
            n_restart_attempts=integrator_settings.n_restart_attempts,
            constraint_tolerance=integrator_settings.constraint_tolerance,
            splitting=integrator_settings.splitting
        )
        """
        ...

    def _get_sampler(self, ...):
        """
        # 9. Create sampler
        sampler_settings = settings.alchemsampler_settings

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
        """
        ...

    def _run_simulation(self, ...):
        """
        if not dry:  # pragma: no-cover
            # minimize
            if verbose:
                logger.info("minimizing systems")

            sampler.minimize(
                max_iterations=sim_settings.minimization_steps
            )

            # equilibrate
            if verbose:
                logger.info("equilibrating systems")

            sampler.equilibrate(int(equil_steps / mc_steps))  # type: ignore

            # production
            if verbose:
                logger.info("running production phase")

            sampler.extend(int(prod_steps / mc_steps))  # type: ignore

            # close reporter when you're done
            reporter.close()

            nc = basepath / sim_settings.output_filename
            chk = basepath / sim_settings.checkpoint_storage
            return {
                'nc': nc,
                'last_checkpoint': chk,
            }
        else:
            # close reporter when you're done, prevent file handle clashes
            reporter.close()

            # clean up the reporter file
            fns = [basepath / sim_settings.output_filename,
                   basepath / sim_settings.checkpoint_storage]
            for fn in fns:
                os.remove(fn)
            return {'debug': {'sampler': sampler}}
        """
        ...

    def run(self, dry=False, verbose=True, basepath=None) -> Dict[str, Any]:
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
        self._prepare(verbose, basepath)

        # 1. Get components
        alchem_comps, solv_comp, prot_comp, smc_comps = self._get_components()

        # 2. Get settings
        settings = self._handle_settings()

        # 3. Get system generator
        system_generator = self._get_system_generator(settings, solv_comp)

        # 4. Get modeller
        system_modeller, comp_resids = self._get_modeller(
            prot_comp, solv_comp, smc_comps, system_generator,
            settings
        )

        # 5. Get OpenMM topology, positions and system
        omm_topology, omm_system, positions = self._get_omm_objects(
            system_generator, system_modeller, smc_comps
        )

        # Probably will need to handle restraints somewhere here

        # 6. Pre-minimize System (Test + Avoid NaNs)
        positions = self._pre_minimize(omm_system, positions)

        # 7. Get lambdas
        lambdas = self._get_lambda_schedule(settings)

        # 8. Get alchemical system
        alchem_system, alchem_factory = self._get_alchemical_system(
            omm_topology, comp_resids, alchem_comps
        )

        # 9. Get compound and sampler states
        cmp_states, sampler_states = self._get_states(
            alchem_system, solvent_comp, settings
        )

        # 10. Create the multistate reporter & create PDB
        reporter = self._get_reporter(
                omm_topology, settings.simulation_setttings
        )

        # 11. Get context caches
        energy_ctx_cache, sampler_ctx_cache = self._get_ctx_caches(
                settings.engine_settings
        )

        # 12. Get integrator
        integrator = self._get_integrator(settings.integrator_settings)

        # 13. Get sampler
        sampler = self._get_sampler(
            integrator, settings.sampler_settings, cmp_states, sampler_states,
            reporter, energy_ctx_cache, sampler_ctx_cache
        )

        # 14. Run simulation
        self._run_simulation(
            dry, verbose, sampler, reporter, settings.simulation_settings
        )

    def _execute(
        self, ctx: gufe.Context, **kwargs,
    ) -> Dict[str, Any]:
        # create directory for *this* unit within the context of the *DAG*
        # stops output files mashing into each other within a DAG
        myid = uuid.uuid4()
        mypath = pathlib.Path(os.path.join(ctx.shared, str(myid)))
        mypath.mkdir(parents=True, exist_ok=False)

        outputs = self.run(basepath=mypath)

        return {
            'repeat_id': self._inputs['repeat_id'],
            'generation': self._inputs['generation'],
            **outputs
        }
