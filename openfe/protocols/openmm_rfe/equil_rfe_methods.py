# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""Equilibrium Relative Free Energy methods using OpenMM in a
Perses-like manner.

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
import numpy as np
import openmm
from openff.units import unit
from openff.units.openmm import to_openmm, ensure_quantity
from openmmtools import multistate
from typing import Optional
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

from .equil_rfe_settings import (
    RelativeHybridTopologyProtocolSettings, SystemSettings,
    SolvationSettings, AlchemicalSettings,
    AlchemicalSamplerSettings, OpenMMEngineSettings,
    IntegratorSettings, SimulationSettings
)
from . import _rfe_utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _get_resname(off_mol) -> str:
    # behaviour changed between 0.10 and 0.11
    omm_top = off_mol.to_topology().to_openmm()
    names = [r.name for r in omm_top.residues()]
    if len(names) > 1:
        raise ValueError("We assume single residue")
    return names[0]


class RelativeHybridTopologyProtocolResult(gufe.ProtocolResult):
    """Dict-like container for the output of a RelativeHybridTopologyProtocol"""
    def __init__(self, **data):
        super().__init__(**data)
        # data is mapping of str(repeat_id): list[protocolunitresults]
        # TODO: Detect when we have extensions and stitch these together?
        if any(len(pur_list) > 2 for pur_list in self.data.values()):
            raise NotImplementedError("Can't stitch together results yet")

    def get_estimate(self):
        """Average free energy difference of this transformation

        Returns
        -------
        dG : unit.Quantity
          The free energy difference between the first and last states. This is
          a Quantity defined with units.
        """
        # TODO: Check this holds up completely for SAMS.
        dGs = [pus[0].outputs['unit_estimate'] for pus in self.data.values()]
        u = dGs[0].u
        # convert all values to units of the first value, then take average of magnitude
        # this would avoid a screwy case where each value was in different units
        vals = [dG.to(u).m for dG in dGs]

        return np.average(vals) * u

    def get_uncertainty(self):
        """The uncertainty/error in the dG value: The std of the estimates of each independent repeat"""
        dGs = [pus[0].outputs['unit_estimate'] for pus in self.data.values()]
        u = dGs[0].u
        # convert all values to units of the first value, then take average of magnitude
        # this would avoid a screwy case where each value was in different units
        vals = [dG.to(u).m for dG in dGs]

        return np.std(vals) * u

    def get_rate_of_convergence(self):
        raise NotImplementedError


class RelativeHybridTopologyProtocol(gufe.Protocol):
    result_cls = RelativeHybridTopologyProtocolResult
    _settings: RelativeHybridTopologyProtocolSettings

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
        return RelativeHybridTopologyProtocolSettings(
            forcefield_settings=settings.OpenMMSystemGeneratorFFSettings(),
            thermo_settings=settings.ThermoSettings(
                temperature=298.15 * unit.kelvin,
                pressure=1 * unit.bar,
            ),
            system_settings=SystemSettings(),
            solvation_settings=SolvationSettings(),
            alchemical_settings=AlchemicalSettings(),
            alchemical_sampler_settings=AlchemicalSamplerSettings(),
            engine_settings=OpenMMEngineSettings(),
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
        extends: Optional[gufe.ProtocolDAGResult] = None,
    ) -> list[gufe.ProtocolUnit]:
        # TODO: Extensions?
        if mapping is None:
            raise ValueError("`mapping` is required for this Protocol")
        if 'ligand' not in mapping:
            raise ValueError("'ligand' must be specified in `mapping` dict")
        if extends:
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
        ligandmapping: LigandAtomMapping = mapping['ligand']
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
        n_repeats = self.settings.alchemical_sampler_settings.n_repeats
        units = [RelativeHybridTopologyProtocolUnit(
            stateA=stateA, stateB=stateB, ligandmapping=ligandmapping,
            settings=self.settings,
            generation=0, repeat_id=int(uuid.uuid4()),
            name=f'{Aname} {Bname} repeat {i} generation 0')
            for i in range(n_repeats)]

        return units

    def _gather(
        self, protocol_dag_results: Iterable[gufe.ProtocolDAGResult]
    ) -> dict[str, Any]:
        # result units will have a repeat_id and generations within this repeat_id
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


class RelativeHybridTopologyProtocolUnit(gufe.ProtocolUnit):
    """
    Calculates the relative free energy of an alchemical ligand transformation.
    """

    def __init__(self, *,
                 stateA: ChemicalSystem,
                 stateB: ChemicalSystem,
                 ligandmapping: LigandAtomMapping,
                 settings: settings.RelativeHybridTopologyProtocolSettings,
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
        ligandmapping : LigandAtomMapping
          the mapping of atoms between the two ligand components
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
            ligandmapping=ligandmapping,
            settings=settings,
            repeat_id=repeat_id,
            generation=generation
        )

    def run(self, *, dry=False, verbose=True,
            scratch_basepath=None,
            shared_basepath=None) -> dict[str, Any]:
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
            logger.info("creating hybrid system")
        if scratch_basepath is None:
            scratch_basepath = pathlib.Path('.')
        if shared_basepath is None:
            # use cwd
            shared_basepath = pathlib.Path('.')

        # 0. General setup and settings dependency resolution step

        # a. check timestep correctness + that
        # equilibration & production are divisible by n_steps
        protocol_settings: RelativeHybridTopologyProtocolSettings = self._inputs['settings']
        stateA = self._inputs['stateA']
        stateB = self._inputs['stateB']
        mapping = self._inputs['ligandmapping']

        forcefield_settings: settings.OpenMMSystemGeneratorFFSettings = protocol_settings.forcefield_settings
        thermo_settings: settings.ThermoSettings = protocol_settings.thermo_settings
        alchem_settings: AlchemicalSettings = protocol_settings.alchemical_settings
        system_settings: SystemSettings = protocol_settings.system_settings
        solvation_settings: SolvationSettings = protocol_settings.solvation_settings
        sampler_settings: AlchemicalSamplerSettings = protocol_settings.alchemical_sampler_settings
        sim_settings: SimulationSettings = protocol_settings.simulation_settings
        timestep = protocol_settings.integrator_settings.timestep
        mc_steps = protocol_settings.integrator_settings.n_steps.m

        # is the timestep good for the mass?
        if forcefield_settings.hydrogen_mass < 3.0:
            if timestep > 2.0 * unit.femtoseconds:
                errmsg = (f"timestep {timestep} too large for "
                          "hydrogen mass {forcefield_settings.hydrogen_mass}")
                raise ValueError(errmsg)

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

        # temporary hack, will fix in next PR
        ligand_ff = f"{forcefield_settings.small_molecule_forcefield}.offxml"

        #  1. Get smirnoff template generators
        smirnoff_stateA = SMIRNOFFTemplateGenerator(
            forcefield=ligand_ff,
            molecules=[stateA_openff_ligand],
        )

        smirnoff_stateB = SMIRNOFFTemplateGenerator(
            forcefield=ligand_ff,
            molecules=[stateB_openff_ligand],
        )

        # 2. Create forece fields and register them
        #  a. state A
        omm_forcefield_stateA = app.ForceField(
            *[ff for ff in forcefield_settings.forcefields]
        )

        omm_forcefield_stateA.registerTemplateGenerator(
                smirnoff_stateA.generator)

        #  b. state B
        omm_forcefield_stateB = app.ForceField(
            *[ff for ff in forcefield_settings.forcefields]
        )

        omm_forcefield_stateB.registerTemplateGenerator(
                smirnoff_stateB.generator)

        # 3. Model state A
        # Note: protein dry run tests are part of the slow tests and don't show
        # up in coverage reports
        stateA_ligand_topology = stateA_openff_ligand.to_topology().to_openmm()
        if 'protein' in stateA.components:  # pragma: no-cover
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
                model=solvation_settings.solvent_model,
                padding=to_openmm(solvation_settings.solvent_padding),
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
        }[system_settings.nonbonded_method.lower()]

        #  b. Get the constraint method
        constraints = {
            'hbonds': app.HBonds,
            'none': None,
            'allbonds': app.AllBonds,
            'hangles': app.HAngles
            # vvv can be None so string it
        }[str(forcefield_settings.constraints).lower()]

        #  c. create the stateA System
        stateA_system = omm_forcefield_stateA.createSystem(
            stateA_modeller.topology,
            nonbondedMethod=nonbonded_method,
            nonbondedCutoff=to_openmm(system_settings.nonbonded_cutoff),
            constraints=constraints,
            rigidWater=forcefield_settings.rigid_water,
            hydrogenMass=forcefield_settings.hydrogen_mass * omm_unit.amu,
            removeCMMotion=forcefield_settings.remove_com,
        )

        #  d. create stateA topology
        stateA_topology = stateA_modeller.getTopology()

        #  e. get stateA positions
        stateA_positions = stateA_modeller.getPositions()
        # TODO: fix this using the same approach as the AFE Protocol
        # canonicalize positions (tuples to np.array)
        stateA_positions = omm_unit.Quantity(
            value=np.array([list(pos) for pos in stateA_positions.value_in_unit_system(openmm.unit.md_unit_system)]),
            unit=openmm.unit.nanometers
        )

        # 6.  Create OpenMM system + topology + positions for "B" system
        #  a. stateB topology from stateA (replace out the ligands)
        stateB_topology = _rfe_utils.topologyhelpers.combined_topology(
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
            nonbondedCutoff=to_openmm(system_settings.nonbonded_cutoff),
            constraints=constraints,
            rigidWater=forcefield_settings.rigid_water,
            hydrogenMass=forcefield_settings.hydrogen_mass * omm_unit.amu,
            removeCMMotion=forcefield_settings.remove_com,
        )

        #  c. Define correspondence mappings between the two systems
        ligand_mappings = _rfe_utils.topologyhelpers.get_system_mappings(
            mapping.componentA_to_componentB,
            stateA_system, stateA_topology, _get_resname(stateA_openff_ligand),
            stateB_system, stateB_topology, _get_resname(stateB_openff_ligand),
            # These are non-optional settings for this method
            fix_constraints=True,
        )

        #  d. Finally get the positions
        stateB_positions = _rfe_utils.topologyhelpers.set_and_check_new_positions(
            ligand_mappings, stateA_topology, stateB_topology,
            old_positions=ensure_quantity(stateA_positions, 'openmm'),
            insert_positions=ensure_quantity(stateB_openff_ligand.conformers[0], 'openmm'),
        )

        # 7. Create the hybrid topology
        hybrid_factory = _rfe_utils.relative.HybridTopologyFactory(
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
                    protocol_settings.thermo_settings.pressure.to(unit.bar).m,
                    protocol_settings.thermo_settings.temperature.m,
                    protocol_settings.integrator_settings.barostat_frequency.m,
                )
            )

        # 8. Create lambda schedule
        # TODO - this should be exposed to users, maybe we should offer the
        # ability to print the schedule directly in settings?
        lambdas = _rfe_utils.lambdaprotocol.LambdaProtocol(
            functions=alchem_settings.lambda_functions,
            windows=alchem_settings.lambda_windows
        )

        # PR #125 temporarily pin lambda schedule spacing to n_replicas
        n_replicas = sampler_settings.n_replicas
        if n_replicas != len(lambdas.lambda_schedule):
            errmsg = (f"Number of replicas {n_replicas} "
                      f"does not equal the number of lambda windows "
                      f"{len(lambdas.lambda_schedule)}")
            raise ValueError(errmsg)

        # 9. Create the multistate reporter
        # Get the sub selection of the system to print coords for
        selection_indices = hybrid_factory.hybrid_topology.select(
                sim_settings.output_indices
        )

        #  a. Create the multistate reporter
        reporter = multistate.MultiStateReporter(
            storage=shared_basepath / sim_settings.output_filename,
            analysis_particle_indices=selection_indices,
            checkpoint_interval=sim_settings.checkpoint_interval.m,
            checkpoint_storage=shared_basepath / sim_settings.checkpoint_storage,
        )

        # 10. Get platform and context caches
        platform = _rfe_utils.compute.get_openmm_platform(
            protocol_settings.engine_settings.compute_platform
        )

        # 11. Set the integrator
        #  a. get integrator settings
        integrator_settings = protocol_settings.integrator_settings

        #  a. Create context caches (energy + sampler)
        #     Note: these needs to exist on the compute node
        energy_context_cache = openmmtools.cache.ContextCache(
            capacity=None, time_to_live=None, platform=platform,
        )

        sampler_context_cache = openmmtools.cache.ContextCache(
            capacity=None, time_to_live=None, platform=platform,
        )

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
        if sampler_settings.sampler_method.lower() == "repex":
            sampler = _rfe_utils.multistate.HybridRepexSampler(
                mcmc_moves=integrator,
                hybrid_factory=hybrid_factory,
                online_analysis_interval=sampler_settings.online_analysis_interval,
                online_analysis_target_error=sampler_settings.online_analysis_target_error.m,
                online_analysis_minimum_iterations=sampler_settings.online_analysis_minimum_iterations
            )
        elif sampler_settings.sampler_method.lower() == "sams":
            sampler = _rfe_utils.multistate.HybridSAMSSampler(
                mcmc_moves=integrator,
                hybrid_factory=hybrid_factory,
                online_analysis_interval=sampler_settings.online_analysis_interval,
                online_analysis_minimum_iterations=sampler_settings.online_analysis_minimum_iterations,
                flatness_criteria=sampler_settings.flatness_criteria,
                gamma0=sampler_settings.gamma0,
            )
        elif sampler_settings.sampler_method.lower() == 'independent':
            sampler = _rfe_utils.multistate.HybridMultiStateSampler(
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
            temperature=to_openmm(protocol_settings.thermo_settings.temperature),
            endstates=alchem_settings.unsampled_endstates,
        )

        try:
            sampler.energy_context_cache = energy_context_cache
            sampler.sampler_context_cache = sampler_context_cache

            if not dry:  # pragma: no-cover
                # minimize
                if verbose:
                    logger.info("minimizing systems")

                sampler.minimize(max_iterations=sim_settings.minimization_steps)

                # equilibrate
                if verbose:
                    logger.info("equilibrating systems")

                sampler.equilibrate(int(equil_steps.m / mc_steps))  # type: ignore

                # production
                if verbose:
                    logger.info("running production phase")

                sampler.extend(int(prod_steps.m / mc_steps))  # type: ignore

                # calculate estimate of results from this individual unit
                ana = multistate.MultiStateSamplerAnalyzer(reporter)
                est, _ = ana.get_free_energy()
                est = (est[0, -1] * ana.kT).in_units_of(omm_unit.kilocalories_per_mole)
                est = ensure_quantity(est, 'openff')

                nc = shared_basepath / sim_settings.output_filename
                chk = shared_basepath / sim_settings.checkpoint_storage
            else:
                # clean up the reporter file
                fns = [shared_basepath / sim_settings.output_filename,
                       shared_basepath / sim_settings.checkpoint_storage]
                for fn in fns:
                    os.remove(fn)
        finally:
            # close reporter when you're done, prevent file handle clashes
            reporter.close()

            # clear GPU contexts
            #energy_context_cache.empty()
            #sampler_context_cache.empty()
            # TODO: remove once upstream solution in place: https://github.com/choderalab/openmmtools/pull/690
            # replace with above
            for context in list(energy_context_cache._lru._data.keys()):
                del self._lru._data[context]
            for context in list(sampler_context_cache._lru._data.keys()):
                del self._lru._data[context]

            del sampler_context_cache, energy_context_cache
            del integrator, sampler

        if not dry:  # pragma: no-cover
            return {
                'nc': nc,
                'last_checkpoint': chk,
                'unit_estimate': est,
            }
        else:
            return {'debug': {'sampler': sampler}}

    def _execute(
        self, ctx: gufe.Context, **kwargs,
    ) -> dict[str, Any]:
        outputs = self.run(scratch_basepath=ctx.scratch, shared_basepath=ctx.shared)

        return {
            'repeat_id': self._inputs['repeat_id'],
            'generation': self._inputs['generation'],
            **outputs
        }
