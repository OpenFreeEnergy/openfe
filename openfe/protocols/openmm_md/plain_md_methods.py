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
import numpy.typing as npt
import openmm
from openff.units import unit
from openff.units.openmm import from_openmm, to_openmm
from typing import Optional
from openmm import app
from openmm import unit as omm_unit
import pathlib
from typing import Any, Iterable
import uuid
import time
import mdtraj
from mdtraj.reporters import XTCReporter

from gufe import (
    settings, ChemicalSystem,
)
from openfe.protocols.openmm_md.plain_md_settings import (
    PlainMDProtocolSettings, SystemSettings,
    SolvationSettings, OpenMMEngineSettings,
    IntegratorSettings, SimulationSettingsMD,
    RepeatSettings
)

from openfe.protocols.openmm_rfe._rfe_utils import compute
from openfe.protocols.openmm_utils import (
    system_validation, settings_validation, system_creation
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PlainMDProtocolResult(gufe.ProtocolResult):
    """EMPTY, Dict-like container for the output of a PlainMDProtocol"""
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
        nonbond = self.settings.system_settings.nonbonded_method
        system_validation.validate_solvent(stateA, nonbond)

        # Validate protein component
        system_validation.validate_protein(stateA)

        # actually create and return Units
        solvent_comp, protein_comp, small_mols = \
            system_validation.get_components(stateA)
        lig_name = small_mols[0].name
        # our DAG has no dependencies, so just list units
        n_repeats = self.settings.repeat_settings.n_repeats
        units = [PlainMDProtocolUnit(
            stateA=stateA, stateB=stateB,
            settings=self.settings,
            generation=0, repeat_id=int(uuid.uuid4()),
            name=f'{lig_name} repeat {i} generation 0')
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
    Base class for plain MD simulations (NonTransformation.
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
    def _pre_minimize(system: openmm.System,
                      positions: omm_unit.Quantity) -> npt.NDArray:
        """
        Short CPU minization of System to avoid GPU NaNs

        Parameters
        ----------
        system : openmm.System
          An OpenMM System to minimize.
        positions : openmm.unit.Quantity
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
            logger.info("creating system")
        if scratch_basepath is None:
            scratch_basepath = pathlib.Path('.')
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
        equil_steps, prod_steps = settings_validation.get_simsteps(
            equil_length=sim_settings.equilibration_length,
            prod_length=sim_settings.production_length,
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

        # b. force the creation of parameters
        # This is necessary because we need to have the FF generated ahead of
        # solvating the system.
        # Note: by default this is cached to ctx.shared/db.json so shouldn't
        # incur too large a cost
        for comp in small_mols:
            offmol = comp.to_openff()
            system_generator.create_system(offmol.to_topology().to_openmm(),
                                           molecules=[offmol])

        # c. get OpenMM Modeller + a dictionary of resids for each component
        stateA_modeller, comp_resids = system_creation.get_omm_modeller(
            protein_comp=protein_comp,
            solvent_comp=solvent_comp,
            small_mols=small_mols,
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

                simulation.context.setPositions(stateA_positions)

                # minimize
                if verbose:
                    logger.info("minimizing systems")

                simulation.minimizeEnergy(
                    maxIterations=sim_settings.minimization_steps
                )
                min_pdb_file = "minimized.pdb"
                positions = simulation.context.getState(
                    getPositions=True, enforcePeriodicBox=False
                ).getPositions()
                with open(shared_basepath / min_pdb_file, "w") as f:
                    openmm.app.PDBFile.writeFile(
                        simulation.topology, positions, file=f, keepIds=True
                    )
                # equilibrate
                if verbose:
                    logger.info("equilibrating systems")
                simulation.context.setVelocitiesToTemperature(
                    to_openmm(thermo_settings.temperature))
                simulation.step(equil_steps)

                # production
                if verbose:
                    logger.info("running production phase")

                # Get the sub selection of the system to print coords for
                selection_indices = mdtraj.Topology.from_openmm(
                    stateA_topology).select(sim_settings.output_indices)

                # Setup the reporters
                simulation.reporters.append(XTCReporter(
                    file=str(shared_basepath / sim_settings.output_filename),
                    reportInterval=sim_settings.checkpoint_interval.m,
                    atomSubset=selection_indices))
                simulation.reporters.append(openmm.app.CheckpointReporter(
                    file=str(shared_basepath / sim_settings.checkpoint_storage),
                    reportInterval=sim_settings.checkpoint_interval.m))
                simulation.reporters.append(openmm.app.StateDataReporter(
                    str(shared_basepath / sim_settings.log_output),
                    sim_settings.checkpoint_interval.m,
                    step=True,
                    time=True,
                    potentialEnergy=True,
                    kineticEnergy=True,
                    totalEnergy=True,
                    temperature=True,
                    volume=True,
                    density=True,
                ))
                t0 = time.time()
                simulation.step(prod_steps)
                t1 = time.time()
                logger.info(f"Completed simulation in {t1 - t0} seconds")

                nc = shared_basepath / sim_settings.output_filename
                chk = shared_basepath / sim_settings.checkpoint_storage

        finally:

            if not dry:
                del integrator, simulation

        if not dry:  # pragma: no-cover
            return {
                'nc': nc,
                'last_checkpoint': chk,
            }
        else:
            return {'debug': {'system': stateA_system}}

    def _execute(
            self, ctx: gufe.Context, **kwargs,
    ) -> dict[str, Any]:
        outputs = self.run(scratch_basepath=ctx.scratch,
                           shared_basepath=ctx.shared)

        return {
            'repeat_id': self._inputs['repeat_id'],
            'generation': self._inputs['generation'],
            **outputs
        }