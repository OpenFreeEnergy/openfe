# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

"""OpenMM MD Protocol --- :mod:`openfe.protocols.openmm_md.plain_md_methods`
===========================================================================================

This module implements the necessary methodology tools to run an MD
simulation using OpenMM tools.

"""
from __future__ import annotations

import logging

from collections import defaultdict
import gufe
import openmm
from openff.units import unit
from openff.units.openmm import from_openmm, to_openmm
import openmm.unit as omm_unit
from typing import Optional
from openmm import app
import pathlib
from typing import Any, Iterable
import uuid
import time
import numpy as np
import mdtraj
from mdtraj.reporters import XTCReporter
from openfe.utils import without_oechem_backend, log_system_probe
from gufe import (
    settings, ChemicalSystem, SmallMoleculeComponent,
    ProteinComponent, SolventComponent
)
from openfe.protocols.openmm_md.plain_md_settings import (
    PlainMDProtocolSettings, SystemSettings,
    SolvationSettings, OpenMMEngineSettings,
    IntegratorSettings, SimulationSettingsMD,
    RepeatSettings
)
from openff.toolkit.topology import Molecule as OFFMolecule

from openfe.protocols.openmm_rfe._rfe_utils import compute
from openfe.protocols.openmm_utils import (
    system_validation, settings_validation, system_creation
)

logger = logging.getLogger(__name__)


class PlainMDProtocolResult(gufe.ProtocolResult):
    """Dict-like container for the output of a PlainMDProtocol
    outputs filenames for the pdb file and trajectory"""
    def __init__(self, **data):
        super().__init__(**data)
        # data is mapping of str(repeat_id): list[protocolunitresults]
        if any(len(pur_list) > 2 for pur_list in self.data.values()):
            raise NotImplementedError("Can't stitch together results yet")

    def get_estimate(self):
        """Since no results as output --> returns None

        Returns
        -------
        None
        """

        return None

    def get_uncertainty(self):
        """Since no results as output --> returns None"""

        return None

    def get_traj_filename(self):
        """String of trajectory file name"""
        traj = [pus[0].outputs['nc'] for pus in self.data.values()]

        return traj

    def get_pdb_filename(self):
        """String of pdb file name"""
        pdbs = [pus[0].outputs['system_pdb'] for pus in self.data.values()]

        return pdbs


class PlainMDProtocol(gufe.Protocol):
    result_cls = PlainMDProtocolResult
    _settings: PlainMDProtocolSettings

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
        return PlainMDProtocolSettings(
            forcefield_settings=settings.OpenMMSystemGeneratorFFSettings(),
            thermo_settings=settings.ThermoSettings(
                temperature=298.15 * unit.kelvin,
                pressure=1 * unit.bar,
            ),
            system_settings=SystemSettings(),
            solvation_settings=SolvationSettings(),
            engine_settings=OpenMMEngineSettings(),
            integrator_settings=IntegratorSettings(),
            simulation_settings=SimulationSettingsMD(
                equilibration_length_nvt=0.1 * unit.nanosecond,
                equilibration_length=1.0 * unit.nanosecond,
                production_length=5.0 * unit.nanosecond,
            ),
            repeat_settings=RepeatSettings(),
        )

    def _create(
            self,
            stateA: ChemicalSystem,
            stateB: ChemicalSystem,
            mapping: Optional[dict[str, gufe.ComponentMapping]] = None,
            extends: Optional[gufe.ProtocolDAGResult] = None,
    ) -> list[gufe.ProtocolUnit]:
        # TODO: Extensions?
        if extends:
            raise NotImplementedError("Can't extend simulations yet")

        # Validate solvent component
        nonbond = self._settings.system_settings.nonbonded_method
        system_validation.validate_solvent(stateA, nonbond)

        # Validate protein component
        system_validation.validate_protein(stateA)

        # actually create and return Units
        # TODO: Deal with multiple ProteinComponents
        solvent_comp, protein_comp, small_mols = system_validation.get_components(stateA)

        system_name = "Solvent MD" if solvent_comp is not None else "Vacuum MD"

        for comp in [protein_comp] + small_mols:
            if comp is not None:
                comp_type = comp.__class__.__name__
                if len(comp.name) == 0:
                    comp_name = 'NoName'
                else:
                    comp_name = comp.name
                system_name += f" {comp_type}:{comp_name}"

        # our DAG has no dependencies, so just list units
        n_repeats = self._settings.repeat_settings.n_repeats
        units = [PlainMDProtocolUnit(
            stateA=stateA, stateB=stateB,
            settings=self._settings,
            generation=0, repeat_id=int(uuid.uuid4()),
            name=f'{system_name} repeat {i} generation 0')
            for i in range(n_repeats)]

        return units

    def _gather(
            self, protocol_dag_results: Iterable[gufe.ProtocolDAGResult]
    ) -> dict[str, Any]:
        # result units will have a repeat_id and generations within this
        # repeat_id
        # first group according to repeat_id
        unsorted_repeats = defaultdict(list)
        for d in protocol_dag_results:
            pu: gufe.ProtocolUnitResult
            for pu in d.protocol_unit_results:
                if not pu.ok():
                    continue

                unsorted_repeats[pu.outputs['repeat_id']].append(pu)

        # then sort by generation within each repeat_id list
        repeats: dict[str, list[gufe.ProtocolUnitResult]] = {}
        for k, v in unsorted_repeats.items():
            repeats[str(k)] = sorted(v, key=lambda x: x.outputs['generation'])

        # returns a dict of repeat_id: sorted list of ProtocolUnitResult
        return repeats


class PlainMDProtocolUnit(gufe.ProtocolUnit):
    """
    Base class for plain MD simulations (NonTransformation).
    """

    def __init__(self, *,
                 stateA: ChemicalSystem,
                 stateB: ChemicalSystem,
                 settings: PlainMDProtocolSettings,
                 generation: int,
                 repeat_id: int,
                 name: Optional[str] = None,
                 ):
        """
        Parameters
        ----------
        stateA, stateB : ChemicalSystem
          the two ligand SmallMoleculeComponents to transform between.  The
          transformation will go from ligandA to ligandB.
        settings : settings.Settings
          the settings for the Method.  This can be constructed using the
          get_default_settings classmethod to give a starting point that
          can be updated to suit.
        repeat_id : int
          identifier for which repeat (aka replica/clone) this Unit is
        generation : int
          counter for how many times this repeat has been extended
        name : str, optional
          human-readable identifier for this Unit

        Notes
        -----
        The mapping used must not involve any elemental changes.  A check for
        this is done on class creation.
        """
        super().__init__(
            name=name,
            stateA=stateA,
            stateB=stateB,
            settings=settings,
            repeat_id=repeat_id,
            generation=generation
        )

    @staticmethod
    def _run_MD(simulation: openmm.app.Simulation,
                positions: omm_unit.Quantity,
                settings: PlainMDProtocolSettings,
                equil_steps_nvt: int,
                equil_steps_npt: int,
                prod_steps: int,
                verbose=True,
                shared_basepath=None) -> np.NDArray:

        """
        Energy minimization, Equilibration and Production MD to be reused
        in multiple protocols

        Parameters
        ----------
        simulation : openmm.app.Simulation
          An OpenMM simulation to simulate.
        positions : openmm.unit.Quantity
          Initial positions for the system.
        settings : PlainMDProtocolSettings
          Settings for Plain MD protocol
        Returns
        -------

        """
        if shared_basepath is None:
            shared_basepath = pathlib.Path('.')
        simulation.context.setPositions(positions)
        # minimize
        if verbose:
            logger.info("minimizing systems")

        simulation.minimizeEnergy(
            maxIterations=settings.simulation_settings.minimization_steps
        )

        # Get the sub selection of the system to save coords for
        selection_indices = mdtraj.Topology.from_openmm(
            simulation.topology).select(settings.simulation_settings.output_indices)

        positions = to_openmm(from_openmm(
            simulation.context.getState(getPositions=True,
                                        enforcePeriodicBox=False
                                        ).getPositions()))
        # Store subset of atoms, specified in input, as PDB file
        mdtraj_top = mdtraj.Topology.from_openmm(simulation.topology)
        traj = mdtraj.Trajectory(
            positions[selection_indices, :],
            mdtraj_top.subset(selection_indices),
        )

        traj.save_pdb(
            shared_basepath / settings.simulation_settings.minimized_structure
        )
        # equilibrate
        # NVT equilibration

        if verbose:
            logger.info("Running NVT equilibration")

        # Set barostat frequency to zero for NVT
        for x in simulation.context.getSystem().getForces():
            if x.getName() == 'MonteCarloBarostat':
                x.setFrequency(0)

        simulation.context.setVelocitiesToTemperature(
            to_openmm(settings.thermo_settings.temperature))

        t0 = time.time()
        simulation.step(equil_steps_nvt)
        t1 = time.time()
        if verbose:
            logger.info(
                f"Completed NVT equilibration in {t1 - t0} seconds")

        # Save last frame NVT equilibration
        positions = to_openmm(
            from_openmm(simulation.context.getState(
                getPositions=True, enforcePeriodicBox=False
            ).getPositions()))

        traj = mdtraj.Trajectory(
            positions[selection_indices, :],
            mdtraj_top.subset(selection_indices),
        )
        traj.save_pdb(
            shared_basepath / settings.simulation_settings.equil_NVT_structure
        )

        # NPT equilibration
        if verbose:
            logger.info("Running NPT equilibration")
        simulation.context.setVelocitiesToTemperature(
            to_openmm(settings.thermo_settings.temperature))

        # Enable the barostat for NPT
        for x in simulation.context.getSystem().getForces():
            if x.getName() == 'MonteCarloBarostat':
                x.setFrequency(settings.integrator_settings.barostat_frequency.m)

        t0 = time.time()
        simulation.step(equil_steps_npt)
        t1 = time.time()
        if verbose:
            logger.info(
                f"Completed NPT equilibration in {t1 - t0} seconds")

        # Save last frame NPT equilibration
        positions = to_openmm(
            from_openmm(simulation.context.getState(
                getPositions=True, enforcePeriodicBox=False
            ).getPositions()))

        traj = mdtraj.Trajectory(
            positions[selection_indices, :],
            mdtraj_top.subset(selection_indices),
        )
        traj.save_pdb(
            shared_basepath / settings.simulation_settings.equil_NPT_structure
        )

        # production
        if verbose:
            logger.info("running production phase")

        # Setup the reporters
        simulation.reporters.append(XTCReporter(
            file=str(shared_basepath / settings.simulation_settings.output_filename),
            reportInterval=settings.simulation_settings.trajectory_interval.m,
            atomSubset=selection_indices))
        simulation.reporters.append(openmm.app.CheckpointReporter(
            file=str(shared_basepath / settings.simulation_settings.checkpoint_storage),
            reportInterval=settings.simulation_settings.checkpoint_interval.m))
        simulation.reporters.append(openmm.app.StateDataReporter(
            str(shared_basepath / settings.simulation_settings.log_output),
            settings.simulation_settings.checkpoint_interval.m,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
            density=True,
            speed=True,
        ))
        t0 = time.time()
        simulation.step(prod_steps)
        t1 = time.time()
        if verbose:
            logger.info(f"Completed simulation in {t1 - t0} seconds")

        return

    def run(self, *, dry=False, verbose=True,
            scratch_basepath=None,
            shared_basepath=None) -> dict[str, Any]:
        """Run the MD simulation.

        Parameters
        ----------
        dry : bool
          Do a dry run of the calculation, creating all necessary hybrid
          system components (topology, system, sampler, etc...) but without
          running the simulation.
        verbose : bool
          Verbose output of the simulation progress. Output is provided via
          INFO level logging.
        scratch_basepath: Pathlike, optional
          Where to store temporary files, defaults to current working directory
        shared_basepath : Pathlike, optional
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
            self.logger.info("Creating system")
        if shared_basepath is None:
            # use cwd
            shared_basepath = pathlib.Path('.')

        # 0. General setup and settings dependency resolution step

        # Extract relevant settings
        protocol_settings: PlainMDProtocolSettings = self._inputs[
            'settings']
        stateA = self._inputs['stateA']

        forcefield_settings: settings.OpenMMSystemGeneratorFFSettings = \
            protocol_settings.forcefield_settings
        thermo_settings: settings.ThermoSettings = \
            protocol_settings.thermo_settings
        system_settings: SystemSettings = protocol_settings.system_settings
        solvation_settings: SolvationSettings = \
            protocol_settings.solvation_settings
        sim_settings: SimulationSettingsMD = \
            protocol_settings.simulation_settings
        timestep = protocol_settings.integrator_settings.timestep
        mc_steps = protocol_settings.integrator_settings.n_steps.m
        integrator_settings = protocol_settings.integrator_settings

        # is the timestep good for the mass?
        settings_validation.validate_timestep(
            forcefield_settings.hydrogen_mass, timestep
        )

        equil_steps_nvt = settings_validation.get_simsteps(
            sim_length=sim_settings.equilibration_length_nvt,
            timestep=timestep, mc_steps=mc_steps
        )
        equil_steps_npt = settings_validation.get_simsteps(
            sim_length=sim_settings.equilibration_length,
            timestep=timestep, mc_steps=mc_steps
        )
        prod_steps = settings_validation.get_simsteps(
            sim_length=sim_settings.production_length,
            timestep=timestep, mc_steps=mc_steps
        )

        solvent_comp, protein_comp, small_mols = system_validation.get_components(stateA)

        # 1. Create stateA system
        # a. get a system generator
        if sim_settings.forcefield_cache is not None:
            ffcache = shared_basepath / sim_settings.forcefield_cache
        else:
            ffcache = None

        system_generator = system_creation.get_system_generator(
            forcefield_settings=forcefield_settings,
            thermo_settings=thermo_settings,
            system_settings=system_settings,
            cache=ffcache,
            has_solvent=solvent_comp is not None,
        )

        smc_components: dict[SmallMoleculeComponent, OFFMolecule]

        smc_components = {i: i.to_openff() for i in small_mols}
        for mol in smc_components.values():
            # don't do this if we have user charges
            if not (mol.partial_charges is not None and np.any(
                    mol.partial_charges)):
                # due to issues with partial charge generation in ambertools
                # we default to using the input conformer for charge generation
                mol.assign_partial_charges(
                    'am1bcc', use_conformers=mol.conformers
                )

            system_generator.create_system(
                mol.to_topology().to_openmm(), molecules=[mol]
            )

        # c. get OpenMM Modeller + a dictionary of resids for each component
        stateA_modeller, comp_resids = system_creation.get_omm_modeller(
            protein_comp=protein_comp,
            solvent_comp=solvent_comp,
            small_mols=smc_components,
            omm_forcefield=system_generator.forcefield,
            solvent_settings=solvation_settings,
        )

        # d. get topology & positions
        # Note: roundtrip positions to remove vec3 issues
        stateA_topology = stateA_modeller.getTopology()
        stateA_positions = to_openmm(
            from_openmm(stateA_modeller.getPositions())
        )

        # e. create the stateA System
        stateA_system = system_generator.create_system(
            stateA_topology,
            molecules=[s.to_openff() for s in small_mols],
        )

        # f. Save pdb of entire system
        with open(shared_basepath / sim_settings.output_structure, "w") as f:
            openmm.app.PDBFile.writeFile(
                stateA_topology, stateA_positions, file=f, keepIds=True
            )

        # 10. Get platform
        platform = compute.get_openmm_platform(
            protocol_settings.engine_settings.compute_platform
        )

        # 11. Set the integrator
        integrator = openmm.LangevinMiddleIntegrator(
            to_openmm(thermo_settings.temperature),
            to_openmm(integrator_settings.collision_rate),
            to_openmm(integrator_settings.timestep),
        )

        simulation = openmm.app.Simulation(
            stateA_modeller.topology,
            stateA_system,
            integrator,
            platform=platform
        )

        try:

            if not dry:  # pragma: no-cover
                self._run_MD(simulation,
                             stateA_positions,
                             protocol_settings,
                             equil_steps_nvt,
                             equil_steps_npt,
                             prod_steps,
                             shared_basepath=shared_basepath,
                             )

        finally:

            if not dry:
                del integrator, simulation

        if not dry:  # pragma: no-cover
            return {
                'system_pdb': shared_basepath / sim_settings.output_filename,
                'minimized_pdb': shared_basepath / sim_settings.minimized_structure,
                'nvt_equil_pdb': shared_basepath / sim_settings.equil_NVT_structure,
                'npt_equil_pdb': shared_basepath / sim_settings.equil_NPT_structure,
                'nc': shared_basepath / sim_settings.output_filename,
                'last_checkpoint': shared_basepath / sim_settings.checkpoint_storage,
            }
        else:
            return {'debug': {'system': stateA_system}}

    def _execute(
            self, ctx: gufe.Context, **kwargs,
    ) -> dict[str, Any]:
        log_system_probe(logging.INFO, paths=[ctx.scratch])

        with without_oechem_backend():
            outputs = self.run(scratch_basepath=ctx.scratch,
                               shared_basepath=ctx.shared)

        return {
            'repeat_id': self._inputs['repeat_id'],
            'generation': self._inputs['generation'],
            **outputs
        }
