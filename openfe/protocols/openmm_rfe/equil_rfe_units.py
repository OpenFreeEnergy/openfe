# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""Equilibrium Relative Free Energy methods using OpenMM and OpenMMTools in a
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

Acknowledgements
----------------
This Protocol is based on, and leverages components originating from
the Perses toolkit (https://github.com/choderalab/perses).
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import subprocess
import warnings
from itertools import chain
from typing import Any, Optional

import gufe
import matplotlib.pyplot as plt
import mdtraj
import numpy as np
import openmmtools
from gufe import (
    ChemicalSystem,
    LigandAtomMapping,
    SmallMoleculeComponent,
    SolventComponent,
    settings,
)
from openff.toolkit.topology import Molecule as OFFMolecule
from openff.units import unit as offunit
from openff.units.openmm import ensure_quantity, from_openmm, to_openmm
from openmmtools import multistate

from openfe.protocols.openmm_utils.omm_settings import (
    BasePartialChargeSettings,
)

from ...analysis import plotting
from ...utils import log_system_probe, without_oechem_backend
from ..openmm_utils import (
    charge_generation,
    multistate_analysis,
    omm_compute,
    settings_validation,
    system_creation,
    system_validation,
)
from . import _rfe_utils
from .equil_rfe_settings import (
    AlchemicalSettings,
    IntegratorSettings,
    LambdaSettings,
    MultiStateOutputSettings,
    MultiStateSimulationSettings,
    OpenFFPartialChargeSettings,
    OpenMMSolvationSettings,
    RelativeHybridTopologyProtocolSettings,
)

logger = logging.getLogger(__name__)


class HybridTopProtocolSetupUnit(gufe.ProtocolUnit):

    @staticmethod
    def _assign_partial_charges(
        partial_charge_settings: OpenFFPartialChargeSettings,
        smc_components: dict[SmallMoleculeComponent, OFFMolecule],
    ) -> None:
        """
        Assign partial charges to SMCs.

        Parameters
        ----------
        charge_settings : OpenFFPartialChargeSettings
          Settings for controlling how the partial charges are assigned.
        smc_components : dict[SmallMoleculeComponent, openff.toolkit.Molecule]
          Dictionary of OpenFF Molecules to add, keyed by
          SmallMoleculeComponent.
        """
        for mol in smc_components.values():
            charge_generation.assign_offmol_partial_charges(
                offmol=mol,
                overwrite=False,
                method=partial_charge_settings.partial_charge_method,
                toolkit_backend=partial_charge_settings.off_toolkit_backend,
                generate_n_conformers=partial_charge_settings.number_of_conformers,
                nagl_model=partial_charge_settings.nagl_model,
            )

    def _prepare(
        self,
        verbose,
        scratch_basepath: pathlib.Path | None,
        shared_basepath: pathlib.Path | None,      
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
            self.logger.info("Setting up the hybrid topology simulation")

        # set basepaths
        def _set_optional_path(basepath):
            if basepath is None:
                return pathlib.Path(".")
            return basepath

        self.scratch_basepath = _set_optional_path(scratch_basepath)
        self.shared_basepath = _set_optional_path(shared_basepath)

    def _get_components(self):
        """
        Get the components from the ChemicalSystem inputs.

        Returns
        -------
        alchem_comps : dict[str, Component]
            Dictionary of alchemical components.
        solv_comp : SolventComponent
            The solvent component.
        protein_comp : ProteinComponent
            The protein component.
        small_mols : list[SmallMoleculeComponent: OFFMolecule]
            List of small molecule components.
        """
        stateA = self._inputs["stateA"]
        stateB = self._inputs["stateB"]
        alchem_comps = system_validation.get_alchemical_components(stateA, stateB)

        solvent_comp, protein_comp, smcs_A = system_validation.get_components(stateA)
        _, _, smcs_B = system_validation.get_components(stateB)

        small_mols = {
            m: m.to_openff()
            for m in set(smcs_A).union(set(smcs_B))
        }

        return alchem_comps, solvent_comp, protein_comp, small_mols

    def _get_settings(self) -> dict[str, SettingsBaseModel]:
        """
        Get the protocol settings from the inputs.

        Returns
        -------
        protocol_settings : RelativeHybridTopologyProtocolSettings
            The protocol settings.
        """
        settings: RelativeHybridTopologyProtocolSettings = self._inputs["protocol"].settings

        protocol_settings: dict[str, SettingsBaseModel] = {}
        protocol_settings["forcefield_settings"] = settings.forcefield_settings
        protocol_settings["thermo_settings"] = settings.thermo_settings
        protocol_settings["alchemical_settings"] = settings.alchemical_settings
        protocol_settings["lambda_settings"] = settings.lambda_settings
        protocol_settings["charge_settings"] = settings.partial_charge_settings
        protocol_settings["solvation_settings"] = settings.solvation_settings
        protocol_settings["simulation_settings"] = settings.simulation_settings
        protocol_settings["output_settings"] = settings.output_settings
        protocol_settings["integrator_settings"] = settings.integrator_settings
        protocol_settings["engine_settings"] = settings.engine_settings
        return protocol_settings
    
    @staticmethod
    def _get_system_generator(
        shared_basepath: pathlib.Path,
        settings: dict[str, SettingsBaseModel],
        solvent_comp: SolventComponent | None,
    ) -> SystemGenerator:
        """
        Get an OpenMM SystemGenerator.

        Parameters
        ----------
        settings : dict[str, SettingsBaseModel]
          A dictionary of protocol settings.
        solvent_comp : SolventComponent | None
            The solvent component of the system, if any.
        
        Returns
        -------
        system_generator : openmmtools.SystemGenerator
          The SystemGenerator for the protocol.
        """
        ffcache = settings["output_settings"].forcefield_cache
        if ffcache is not None:
            ffcache = shared_basepath / ffcache

        # Block out oechem backend in system_generator calls to avoid
        # any issues with smiles roundtripping between rdkit and oechem
        with without_oechem_backend():
            system_generator = system_creation.get_system_generator(
                forcefield_settings=settings["forcefield_settings"],
                integrator_settings=settings["integrator_settings"],
                thermo_settings=settings["thermo_settings"],
                cache=ffcache,
                has_solvent=solvent_comp is not None,
            )
        
        return system_generator
    
    @staticmethod
    def _create_stateA_system(
        protein_component: ProteinComponent | None,
        solvent_component: SolventComponent | None,
        small_mols_stateA: dict[SmallMoleculeComponent, OFFMolecule],
        system_generator: SystemGenerator,
        solvation_settings: OpenMMSolvationSettings,
    ):
        stateA_modeller, comp_resids = system_creation.get_omm_modeller(
            protein_comp=protein_component,
            solvent_comp=solvent_component,
            small_mols=small_mols_stateA,
            omm_forcefield=system_generator.forcefield,
            solvent_settings=solvation_settings,
        )

        stateA_topology = stateA_modeller.getTopology()
        # Note: roundtrip positions to remove vec3 issues
        stateA_positions = to_openmm(from_openmm(stateA_modeller.getPositions()))

        with without_oechem_backend():
            stateA_system = system_generator.create_system(
                stateA_modeller.topology,
                molecules=list(small_mols_stateA.values()),
            )

        return stateA_system, stateA_topology, stateA_positions, comp_resids
    
    @staticmethod
    def _create_stateB_system(
        small_mols_stateB: dict[SmallMoleculeComponent, OFFMolecule],
        mapping: LigandAtomMapping,
        stateA_topology: app.Topology,
        exclude_resids: np.ndarray,
        system_generator: SystemGenerator,
    ):
        stateB_topology, stateB_alchem_resids = _rfe_utils.topologyhelpers.combined_topology(
            topology1=stateA_topology,
            topology2=small_mols_stateB[mapping.componentB].to_topology().to_openmm(),
            exclude_resids=exclude_resids,
        )

        with without_oechem_backend():
            stateB_system = system_generator.create_system(
                stateB_topology,
                molecules=list(small_mols_stateB.values()),
            )
        
        return stateB_system, stateB_topology, stateB_alchem_resids
    
    @staticmethod
    def _handle_alchemical_waters(
        stateA_topology: app.Topology,
        stateA_positions: npt.NDArray,
        stateB_topology: app.Topology,
        stateB_system: openmm.System,
        charge_difference: int,
        system_mappings: dict[str, dict[int, int]],
        alchemical_settings: AlchemicalSettings,
        solvent_component: SolventComponent | None,
    ):
        if charge_difference == 0:
            return

        alchem_water_resids = _rfe_utils.topologyhelpers.get_alchemical_waters(
            stateA_topology,
            stateA_positions,
            charge_difference,
            alchemical_settings.explicit_charge_correction_cutoff,
        )

        _rfe_utils.topologyhelpers.handle_alchemical_waters(
            alchem_water_resids,
            stateB_topology,
            stateB_system,
            system_mappings,
            charge_difference,
            solvent_component,
        )
    
    def _get_omm_objects(
        self,
        stateA,
        stateB,
        mapping,
        settings: dict[str, SettingsBaseModel],
        protein_component: ProteinComponent | None,
        solvent_component: SolventComponent | None,
        small_mols: dict[SmallMoleculeComponent, OFFMolecule],
    ):
        if self.verbose:
            self.logger.info("Parameterizing system")

        # Get the system generator and register the templates
        system_generator = self._get_system_generator(
            shared_basepath=self.shared_basepath,
            settings=settings,
            solvent_comp=solvent_component
        )

        system_generator.add_molecules(
            molecules=list(small_mols.values())
        )

        # State A system creation
        small_mols_stateA = {
            smc: offmol
            for smc, offmol in small_mols.items()
            if stateA.contains(smc)
        }

        stateA_system, stateA_topology, stateA_positions, comp_resids = self._create_stateA_system(
            protein_component=protein_component,
            solvent_component=solvent_component,
            small_mols_stateA=small_mols_stateA,
            system_generator=system_generator,
            solvation_settings=settings["solvation_settings"],
        )

        # State B system creation
        small_mols_stateB = {
            smc: offmol
            for smc, offmol in small_mols.items()
            if stateB.contains(smc)
        }

        stateB_system, stateB_topology, stateB_alchem_resids = self._create_stateB_system(
            small_mols_stateB=small_mols_stateB,
            mapping=mapping,
            stateA_topology=stateA_topology,
            exclude_resids = comp_resids[mapping.componentA],
            system_generator=system_generator,
        )

        # Get the mapping between the two systems
        system_mappings = _rfe_utils.topologyhelpers.get_system_mappings(
            old_to_new_atom_map=mapping.componentA_to_componentB,
            old_system=stateA_system,
            old_topology=stateA_topology,
            old_resids=comp_resids[mapping.componentA],
            new_system=stateB_system,
            new_topology=stateB_topology,
            new_resids=stateB_alchem_resids,
            # These are non-optional settings for this method
            fix_constraints=True,
        )

        # Handle alchemical waters if needed
        if settings["alchemical_settings"].explicit_charge_correction:
            self._handle_alchemical_waters(
                stateA_topology=stateA_topology,
                stateA_positions=stateA_positions,
                stateB_topology=stateB_topology,
                stateB_system=stateB_system,
                charge_difference=mapping.get_alchemical_charge_difference(),
                system_mappings=system_mappings,
                alchemical_settings=settings["alchemical_settings"],
                solvent_component=solvent_component,
            )
        
        stateB_positions = _rfe_utils.topologyhelpers.set_and_check_new_positions(
            system_mappings,
            stateA_topology,
            stateB_topology,
            old_positions=ensure_quantity(stateA_positions, "openmm"),
            insert_positions=ensure_quantity(
                small_mols[mapping.componentB].conformers[0], "openmm"
            ),
        )

        return (
            stateA_system, stateA_topology, stateA_positions,
            stateB_system, stateB_topology, stateB_positions,
            system_mappings
        )

    @staticmethod
    def _get_alchemical_system(
        stateA_system,
        stateA_positions,
        stateA_topology,
        stateB_system,
        stateB_positions,
        stateB_topology,
        system_mappings,
        alchemical_settings: AlchemicalSettings,
    ):
        if alchemical_settings.softcore_LJ.lower() == "gapsys":
            softcore_LJ_v2 = True
        elif alchemical_settings.softcore_LJ.lower() == "beutler":
            softcore_LJ_v2 = False

        hybrid_factory = _rfe_utils.relative.HybridTopologyFactory(
            stateA_system,
            stateA_positions,
            stateA_topology,
            stateB_system,
            stateB_positions,
            stateB_topology,
            old_to_new_atom_map=system_mappings["old_to_new_atom_map"],
            old_to_new_core_atom_map=system_mappings["old_to_new_core_atom_map"],
            use_dispersion_correction=alchemical_settings.use_dispersion_correction,
            softcore_alpha=alchemical_settings.softcore_alpha,
            softcore_LJ_v2=softcore_LJ_v2,
            softcore_LJ_v2_alpha=alchemical_settings.softcore_alpha,
            interpolate_old_and_new_14s=alchemical_settings.turn_off_core_unique_exceptions,
        )

        return hybrid_factory, hybrid_factory.hybrid_system

    def run(
        self, *, dry=False, verbose=True, scratch_basepath=None, shared_basepath=None
    ) -> dict[str, Any]:
        """Set up a Hybrid Topology system.

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
        # set up logging and basepaths
        self._prepare(
            verbose=verbose,
            scratch_basepath=scratch_basepath,
            shared_basepath=shared_basepath,
        )

        # Get the components
        mapping = self._inputs["ligandmapping"]
        stateA = self._inputs["stateA"]
        stateB = self._inputs["stateB"]
        alchem_comps, solvent_comp, protein_comp, off_small_mols = self._get_components()

        # Get the settings
        settings = self._get_settings()

        # Get the OpenMM objects
        (
            stateA_system, stateA_topology, stateA_positions,
            stateB_system, stateB_topology, stateB_positions,
            ligand_mappings
        ) = self._get_omm_objects(
            stateA=stateA,
            stateB=stateB,
            mapping=mapping,
            settings=settings,
            protein_component=protein_comp,
            solvent_component=solvent_comp,
            small_mols=off_small_mols,
        )
        
        # Get the alchemical factory & system
        hybrid_factory, hybrid_system = self._get_alchemical_system(
            stateA_system,
            stateA_positions,
            stateA_topology,
            stateB_system,
            stateB_positions,
            stateB_topology,
            ligand_mappings,
            alchemical_settings=settings["alchemical_settings"],
        )

        # Verify alchemical system
        if hybrid_factory.has_virtual_sites:
            if not settings["integrator_settings"].reassign_velocities:
                errmsg = (
                    "Simulations with virtual sites without velocity "
                    "reassignments are unstable in openmmtools"
                )
                raise ValueError(errmsg)

        # Get the selection indices for the system
        selection_indices = hybrid_factory.hybrid_topology.select(
            settings["output_settings"].output_indices
        )

        # Write out a PDB containing the subsampled hybrid state
        bfactors = np.zeros_like(selection_indices, dtype=float)  # environment
        bfactors[np.in1d(selection_indices, list(hybrid_factory._atom_classes['unique_old_atoms']))] = 0.25  # lig A
        bfactors[np.in1d(selection_indices, list(hybrid_factory._atom_classes['core_atoms']))] = 0.50  # core
        bfactors[np.in1d(selection_indices, list(hybrid_factory._atom_classes['unique_new_atoms']))] = 0.75  # lig B

        if len(selection_indices) > 0:
            traj = mdtraj.Trajectory(
                hybrid_factory.hybrid_positions[selection_indices, :],
                hybrid_factory.hybrid_topology.subset(selection_indices),
            ).save_pdb(
                shared_basepath / settings["output_settings"].output_structure,
                bfactors=bfactors,
            )

        # Serialize the hybrid system
        system_outfile = self.shared_basepath / "hybrid_system.xml.bz2"
        serialize(hybrid_system, system_outfile)

        # Serialize the positions
        positions_outfile = self.shared_basepath / "hybrid_positions.npz"
        npy_positions_nm = from_openmm(hybrid_factory.hybrid_positions).to("nanometer").m
        np.savez(positions_outfile, npy_positions_nm)


        unit_result_dict = {
            "system": system_outfile,
            "positions": positions_outfile,
            "pdb_structure": shared_basepath / settings["output_settings"].output_structure
            "selection_indices": selection_indices,
        }

        # If this is a dry run, we return the objects directly
        if dry:
            unit_result_dict |= {
                "hybrid_factory": hybrid_factory,
                "hybrid_system": hybrid_system,
            }

        return unit_result_dict

    def _execute(
        self,
        ctx: gufe.Context,
        **inputs,
    ) -> dict[str, Any]:
        log_system_probe(logging.INFO, paths=[ctx.scratch])

        outputs = self.run(scratch_basepath=ctx.scratch, shared_basepath=ctx.shared)

        return {
            "repeat_id": self._inputs["repeat_id"],
            "generation": self._inputs["generation"],
            **outputs,
        }


class HybridTopProtocolSimulationUnit(gufe.ProtocolUnit):
    def _prepare(
        self,
        verbose,
        scratch_basepath: pathlib.Path | None,
        shared_basepath: pathlib.Path | None,      
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
            self.logger.info("Setting up the hybrid topology simulation")

        # set basepaths
        def _set_optional_path(basepath):
            if basepath is None:
                return pathlib.Path(".")
            return basepath

        self.scratch_basepath = _set_optional_path(scratch_basepath)
        self.shared_basepath = _set_optional_path(shared_basepath)

    def _get_settings(self) -> dict[str, SettingsBaseModel]:
        """
        Get the protocol settings from the inputs.

        Returns
        -------
        protocol_settings : RelativeHybridTopologyProtocolSettings
            The protocol settings.
        """
        settings: RelativeHybridTopologyProtocolSettings = self._inputs["protocol"].settings

        protocol_settings: dict[str, SettingsBaseModel] = {}
        protocol_settings["forcefield_settings"] = settings.forcefield_settings
        protocol_settings["thermo_settings"] = settings.thermo_settings
        protocol_settings["alchemical_settings"] = settings.alchemical_settings
        protocol_settings["lambda_settings"] = settings.lambda_settings
        protocol_settings["charge_settings"] = settings.partial_charge_settings
        protocol_settings["solvation_settings"] = settings.solvation_settings
        protocol_settings["simulation_settings"] = settings.simulation_settings
        protocol_settings["output_settings"] = settings.output_settings
        protocol_settings["integrator_settings"] = settings.integrator_settings
        protocol_settings["engine_settings"] = settings.engine_settings
        return protocol_settings
    
    def _get_reporter(
        self,
        selection_indices: np.ndarray,
        output_settings: MultiStateOutputSettings,
        simulation_settings: MultiStateSimulationSettings,
    ):
        nc = self.shared_basepath / output_settings.output_filename
        chk = output_settings.checkpoint_storage_filename

        if output_settings.positions_write_frequency is not None:
            pos_interval = settings_validation.divmod_time_and_check(
                numerator=output_settings.positions_write_frequency,
                denominator=simulation_settings.time_per_iteration,
                numerator_name="output settings' position_write_frequency",
                denominator_name="simulation settings' time_per_iteration",
            )
        else:
            pos_interval = 0

        if output_settings.velocities_write_frequency is not None:
            vel_interval = settings_validation.divmod_time_and_check(
                numerator=output_settings.velocities_write_frequency,
                denominator=sampler_settings.time_per_iteration,
                numerator_name="output settings' velocity_write_frequency",
                denominator_name="sampler settings' time_per_iteration",
            )
        else:
            vel_interval = 0

        chk_intervals = settings_validation.convert_checkpoint_interval_to_iterations(
            checkpoint_interval=output_settings.checkpoint_interval,
            time_per_iteration=simulation_settings.time_per_iteration,
        )

        return multistate.MultiStateReporter(
            storage=nc,
            analysis_particle_indices=selection_indices,
            checkpoint_interval=chk_intervals,
            checkpoint_storage=chk,
            position_interval=pos_interval,
            velocity_interval=vel_interval,
        )

    @staticmethod
    def _get_sampler(
        system: openmm.System,
        positions: openmm.Quantity,
        lambdas: _rfe_utils.lambdaprotocol.LambdaProtocol,
        integrator: openmmtools.mcmc.MCMCMove,
        reporter: multistate.MultiStateReporter,
        simulation_settings: MultiStateSimulationSettings,
        thermo_settings: ThermodynamicSettings,
        alchem_settings: AlchemicalSettings,
        platform: openmm.Platform,
        dry: bool,
    ):

        rta_its, rta_min_its = settings_validation.convert_real_time_analysis_iterations(
            simulation_settings=simulation_settings,
        )
    
        # convert early_termination_target_error from kcal/mol to kT
        early_termination_target_error = (
            settings_validation.convert_target_error_from_kcal_per_mole_to_kT(
                thermo_settings.temperature,
                simulation_settings.early_termination_target_error,
            )
        )

        if simulation_settings.sampler_method.lower() == "repex":
            sampler = _rfe_utils.multistate.HybridRepexSampler(
                mcmc_moves=integrator,
                hybrid_system=system,
                hybrid_positions=positions,
                online_analysis_interval=rta_its,
                online_analysis_target_error=early_termination_target_error,
                online_analysis_minimum_iterations=rta_min_its,
            )

        elif simulation_settings.sampler_method.lower() == "sams":
            sampler = _rfe_utils.multistate.HybridSAMSSampler(
                mcmc_moves=integrator,
                hybrid_system=system,
                hybrid_positions=positions,
                online_analysis_interval=rta_its,
                online_analysis_minimum_iterations=rta_min_its,
                flatness_criteria=simulation_settings.sams_flatness_criteria,
                gamma0=simulation_settings.sams_gamma0,
            )

        elif simulation_settings.sampler_method.lower() == "independent":
            sampler = _rfe_utils.multistate.HybridMultiStateSampler(
                mcmc_moves=integrator,
                hybrid_system=system,
                hybrid_positions=positions,
                online_analysis_interval=rta_its,
                online_analysis_target_error=early_termination_target_error,
                online_analysis_minimum_iterations=rta_min_its,
            )

        else:
            raise AttributeError(f"Unknown sampler {simulation_settings.sampler_method}")

        sampler.setup(
            n_replicas=simulation_settings.n_replicas,
            reporter=reporter,
            lambda_protocol=lambdas,
            temperature=to_openmm(thermo_settings.temperature),
            endstates=alchem_settings.endstate_dispersion_correction,
            minimization_platform=platform.getName(),
            # Set minimization steps to None when running in dry mode
            # otherwise do a very small one to avoid NaNs
            minimization_steps=100 if not dry else None,
        )

        sampler.energy_context_cache = energy_context_cache
        sampler.sampler_context_cache = sampler_context_cache

        return sampler

    def _get_ctx_caches(
        self,
        platform: openmm.Platform,
    ) -> tuple[openmmtools.cache.ContextCache, openmmtools.cache.ContextCache]:
        """
        Set the context caches based on the chosen platform

        Parameters
        ----------
        platform: openmm.Platform
          The OpenMM compute platform.

        Returns
        -------
        energy_context_cache : openmmtools.cache.ContextCache
          The energy state context cache.
        sampler_context_cache : openmmtools.cache.ContextCache
          The sampler state context cache.
        """
        energy_context_cache = openmmtools.cache.ContextCache(
            capacity=None,
            time_to_live=None,
            platform=platform,
        )

        sampler_context_cache = openmmtools.cache.ContextCache(
            capacity=None,
            time_to_live=None,
            platform=platform,
        )

        return energy_context_cache, sampler_context_cache

    def run(self, *, dry=False, verbose=True, scratch_basepath=None, shared_basepath=None):

        # Get relevant outputs from setup
        system = deserialize(self._inputs["setup_results"]["system"])
        positions = to_openmm(
            deserialize(self._inputs["setup_results"]["positions"]) * offunit.nm
        )
        selection_indices = self._inputs["setup_results"]["selection_indices"]

        # Get the settings
        settings = self._get_settings()

        # Get the lambda schedule
        lambdas = _rfe_utils.lambdaprotocol.LambdaProtocol(
            functions=lambda_settings.lambda_functions,
            windows=lambda_settings.lambda_windows
        )

        # Define simulation steps
        steps_per_iteration = settings_validation.convert_steps_per_iteration(
            simulation_settings=settings["simulation_settings"],
            integrator_settings=settings["integrator_settings"],
        )

        equilibration_steps = settings_validation.get_simsteps(
            sim_length=settings["simulation_settings"].equilibration_length,
            timestep=settings["integrator_settings"].timestep,
            mc_steps=steps_per_iteration,
        )

        production_steps = settings_validation.get_simsteps(
            sim_length=settings["simulation_settings"].production_length,
            timestep=settings["integrator_settings"].timestep,
            mc_steps=steps_per_iteration,
        )

        try:
            # Get the reporter
            reporter = self._get_reporter(
                selection_indices=selection_indices,
                output_settings=settings["output_settings"],
                simulation_settings=settings["simulation_settings"],
            )

            # Get the compute platform
            # restrict to a single CPU if running vacuum
            restrict_cpu = settings["forcefield_settings"].nonbonded_method.lower() == "nocutoff"
            platform = omm_compute.get_openmm_platform(
                platform_name=settings["engine_settings"].compute_platform,
                gpu_device_index=settings["engine_settings"].gpu_device_index,
                restrict_cpu_count=restrict_cpu,
            )

            # Get the integrator
            integrator = openmmtools.mcmc.LangevinDynamicsMove(
                timestep=to_openmm(settings["integrator_settings"].timestep),
                collision_rate=to_openmm(settings["integrator_settings"].langevin_collision_rate),
                n_steps=steps_per_iteration,
                reassign_velocities=settings["integrator_settings"].reassign_velocities,
                n_restart_attempts=settings["integrator_settings"].n_restart_attempts,
                constraint_tolerance=settings["integrator_settings"].constraint_tolerance,
            )
            # Create context caches
            energy_context_cache, sampler_context_cache = self._get_ctx_caches(platform)

            sampler = self._get_sampler(
                system=system,
                positions=positions,
                lambdas=lambdas,
                integrator=integrator,
                reporter=reporter,
                simulation_settings=settings["simulation_settings"],
                thermo_settings=settings["thermo_settings"],
                alchem_settings=settings["alchemical_settings"],
                platform=platform,
                dry=dry,
                energy_context_cache=energy_context_cache,
                sampler_context_cache=sampler_context_cache,
            )
    
            if not dry:  # pragma: no-cover
                # minimize
                if verbose:
                    self.logger.info("Running minimization")

                sampler.minimize(max_iterations=sampler_settings.minimization_steps)

                # equilibrate
                if verbose:
                    self.logger.info("Running equilibration phase")

                sampler.equilibrate(int(equil_steps / steps_per_iteration))

                # production
                if verbose:
                    self.logger.info("Running production phase")

                sampler.extend(int(prod_steps / steps_per_iteration))

                self.logger.info("Production phase complete")
            else:
                # clean up the reporter file
                fns = [
                    shared_basepath / output_settings.output_filename,
                    shared_basepath / output_settings.checkpoint_storage_filename,
                ]
                for fn in fns:
                    os.remove(fn)

        finally:
            # close reporter when you're done, prevent
            # file handle clashes
            reporter.close()

            # clear GPU contexts
            # TODO: use cache.empty() calls when openmmtools #690 is resolved
            # replace with above
            for context in list(energy_context_cache._lru._data.keys()):
                del energy_context_cache._lru._data[context]
            for context in list(sampler_context_cache._lru._data.keys()):
                del sampler_context_cache._lru._data[context]

            # cautiously clear out the global context cache too
            for context in list(openmmtools.cache.global_context_cache._lru._data.keys()):
                del openmmtools.cache.global_context_cache._lru._data[context]

            del sampler_context_cache, energy_context_cache

            if not dry:
                del integrator, sampler

        if not dry:  # pragma: no-cover
            return {
                "nc": nc,
                "last_checkpoint": chk
            }
        else:
            return {
                "debug": {
                    "sampler": sampler
                }
            }

    def _execute(
        self,
        ctx: gufe.Context,
        **inputs,
    ) -> dict[str, Any]:
        log_system_probe(logging.INFO, paths=[ctx.scratch])

        outputs = self.run(
            dry=False,
            scratch_basepath=ctx.scratch,
            shared_basepath=ctx.shared,
        )

        return {
            "repeat_id": self._inputs["repeat_id"],
            "generation": self._inputs["generation"],
            **outputs,
        }
    

class HybridTopProtocolAnalysisUnit(gufe.ProtocolUnit):
   def _prepare(
        self,
        verbose,
        scratch_basepath: pathlib.Path | None,
        shared_basepath: pathlib.Path | None,      
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
            self.logger.info("Setting up the hybrid topology simulation")

        # set basepaths
        def _set_optional_path(basepath):
            if basepath is None:
                return pathlib.Path(".")
            return basepath

        self.scratch_basepath = _set_optional_path(scratch_basepath)
        self.shared_basepath = _set_optional_path(shared_basepath)

    @staticmethod
    def structural_analysis(scratch, shared) -> dict:
        # don't put energy analysis in here, it uses the open file reporter
        # whereas structural stuff requires that the file handle is closed
        # TODO: we should just make openfe_analysis write an npz instead!
        analysis_out = scratch / "structural_analysis.json"

        ret = subprocess.run(
            [
                "openfe_analysis",  # CLI entry point
                "RFE_analysis",  # CLI option
                str(shared),  # Where the simulation.nc fille
                str(analysis_out),  # Where the analysis json file is written
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if ret.returncode:
            return {"structural_analysis_error": ret.stderr}

        with open(analysis_out, "rb") as f:
            data = json.load(f)

        savedir = pathlib.Path(shared)
        if d := data["protein_2D_RMSD"]:
            fig = plotting.plot_2D_rmsd(d)
            fig.savefig(savedir / "protein_2D_RMSD.png")
            plt.close(fig)
            f2 = plotting.plot_ligand_COM_drift(data["time(ps)"], data["ligand_wander"])
            f2.savefig(savedir / "ligand_COM_drift.png")
            plt.close(f2)

        f3 = plotting.plot_ligand_RMSD(data["time(ps)"], data["ligand_RMSD"])
        f3.savefig(savedir / "ligand_RMSD.png")
        plt.close(f3)

        # Save to numpy compressed format (~ 6x more space efficient than JSON)
        np.savez_compressed(
            shared / "structural_analysis.npz",
            protein_RMSD=np.asarray(data["protein_RMSD"], dtype=np.float32),
            ligand_RMSD=np.asarray(data["ligand_RMSD"], dtype=np.float32),
            ligand_COM_drift=np.asarray(data["ligand_wander"], dtype=np.float32),
            protein_2D_RMSD=np.asarray(data["protein_2D_RMSD"], dtype=np.float32),
            time_ps=np.asarray(data["time(ps)"], dtype=np.float32),
        )

        return {"structural_analysis": shared / "structural_analysis.npz"}

    def run(self, *, dry=False, verbose=True, scratch_basepath=None, shared_basepath=None):
        # set up logging and basepaths
        trajectory = self._inputs["simulation_results"]["nc"]
        checkpoint = self._inputs["simulation_results"]["last_checkpoint"]

        self._prepare(
            verbose=verbose,
            scratch_basepath=scratch_basepath,
            shared_basepath=shared_basepath,
        )

        # Get energies
        try:
            reporter = multistate.MultiStateReporter(
                storage=trajectory,
                checkpoint_storage=checkpoint,
            )
            
            analyzer = multistate_analysis.MultistateEquilFEAnalysis(
                reporter,
                sampling_method=self._inputs["protocol"].settings.simulation_settings.sampler_method.lower(),
                result_units=offunit.kilocalorie_per_mole,
            )
            analyzer.plot(filepath=self.shared_basepath, filename_prefix="")
            analyzer.close()

            # analyzer.unit_results_dict
        finally:
            reporter.close()

        # Get structural analysis -- todo: switch this away from the CLI
        structural_analysis_outputs = self.structural_analysis(
            scratch=self.scratch_basepath,
            shared=self.shared_basepath,
        )

        return analyzer.unit_results_dict | structural_analysis_outputs

    def _execute(
        self,
        ctx: gufe.Context,
        **inputs,
    ) -> dict[str, Any]:
        log_system_probe(logging.INFO, paths=[ctx.scratch])

        outputs = self.run(
            dry=False,
            scratch_basepath=ctx.scratch,
            shared_basepath=ctx.shared,
        )

        return {
            "repeat_id": self._inputs["repeat_id"],
            "generation": self._inputs["generation"],
            **outputs,
        }
