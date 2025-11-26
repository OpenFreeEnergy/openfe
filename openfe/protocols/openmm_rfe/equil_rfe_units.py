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


def _get_alchemical_charge_difference(
    mapping: LigandAtomMapping,
    nonbonded_method: str,
    explicit_charge_correction: bool,
    solvent_component: SolventComponent,
) -> int:
    """
    Checks and returns the difference in formal charge between state A and B.

    Raises
    ------
    ValueError
      * If an explicit charge correction is attempted and the
        nonbonded method is not PME.
      * If the absolute charge difference is greater than one
        and an explicit charge correction is attempted.
    UserWarning
      If there is any charge difference.

    Parameters
    ----------
    mapping : dict[str, ComponentMapping]
      Dictionary of mappings between transforming components.
    nonbonded_method : str
      The OpenMM nonbonded method used for the simulation.
    explicit_charge_correction : bool
      Whether or not to use an explicit charge correction.
    solvent_component : openfe.SolventComponent
      The SolventComponent of the simulation.

    Returns
    -------
    int
      The formal charge difference between states A and B.
      This is defined as sum(charge state A) - sum(charge state B)
    """

    difference = mapping.get_alchemical_charge_difference()

    if abs(difference) > 0:
        if explicit_charge_correction:
            if nonbonded_method.lower() != "pme":
                errmsg = "Explicit charge correction when not using PME is not currently supported."
                raise ValueError(errmsg)
            if abs(difference) > 1:
                errmsg = (
                    f"A charge difference of {difference} is observed "
                    "between the end states and an explicit charge  "
                    "correction has been requested. Unfortunately "
                    "only absolute differences of 1 are supported."
                )
                raise ValueError(errmsg)

            ion = {-1: solvent_component.positive_ion, 1: solvent_component.negative_ion}[
                difference
            ]
            wmsg = (
                f"A charge difference of {difference} is observed "
                "between the end states. This will be addressed by "
                f"transforming a water into a {ion} ion"
            )
            logger.warning(wmsg)
            warnings.warn(wmsg)
        else:
            wmsg = (
                f"A charge difference of {difference} is observed "
                "between the end states. No charge correction has "
                "been requested, please account for this in your "
                "final results."
            )
            logger.warning(wmsg)
            warnings.warn(wmsg)

    return difference

class RHTProtocolSetupUnit(gufe.Protocol):

    @staticmethod
    def _assign_partial_charges(
        charge_settings: OpenFFPartialChargeSettings,
        off_small_mols: dict[str, list[tuple[SmallMoleculeComponent, OFFMolecule]]],
    ) -> None:
        """
        Assign partial charges to SMCs.

        Parameters
        ----------
        charge_settings : OpenFFPartialChargeSettings
          Settings for controlling how the partial charges are assigned.
        off_small_mols : dict[str, list[tuple[SmallMoleculeComponent, OFFMolecule]]]
          Dictionary of dictionary of OpenFF Molecules to add, keyed by
          state and SmallMoleculeComponent.
        """
        for smc, mol in chain(
            off_small_mols["stateA"], off_small_mols["stateB"], off_small_mols["both"]
        ):
            charge_generation.assign_offmol_partial_charges(
                offmol=mol,
                overwrite=False,
                method=charge_settings.partial_charge_method,
                toolkit_backend=charge_settings.off_toolkit_backend,
                generate_n_conformers=charge_settings.number_of_conformers,
                nagl_model=charge_settings.nagl_model,
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
        
        """

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

        # Get the settings

        # Get the OpenMM objects

        # 0. General setup and settings dependency resolution step

        # Extract relevant settings
        protocol_settings: RelativeHybridTopologyProtocolSettings = self._inputs[
            "protocol"
        ].settings
        stateA = self._inputs["stateA"]
        stateB = self._inputs["stateB"]
        mapping = self._inputs["ligandmapping"]

        forcefield_settings: settings.OpenMMSystemGeneratorFFSettings = (
            protocol_settings.forcefield_settings
        )
        thermo_settings: settings.ThermoSettings = protocol_settings.thermo_settings
        alchem_settings: AlchemicalSettings = protocol_settings.alchemical_settings
        lambda_settings: LambdaSettings = protocol_settings.lambda_settings
        charge_settings: BasePartialChargeSettings = protocol_settings.partial_charge_settings
        solvation_settings: OpenMMSolvationSettings = protocol_settings.solvation_settings
        sampler_settings: MultiStateSimulationSettings = protocol_settings.simulation_settings
        output_settings: MultiStateOutputSettings = protocol_settings.output_settings
        integrator_settings: IntegratorSettings = protocol_settings.integrator_settings

        # TODO: Also validate various conversions?
        # Convert various time based inputs to steps/iterations
        steps_per_iteration = settings_validation.convert_steps_per_iteration(
            simulation_settings=sampler_settings,
            integrator_settings=integrator_settings,
        )

        equil_steps = settings_validation.get_simsteps(
            sim_length=sampler_settings.equilibration_length,
            timestep=integrator_settings.timestep,
            mc_steps=steps_per_iteration,
        )
        prod_steps = settings_validation.get_simsteps(
            sim_length=sampler_settings.production_length,
            timestep=integrator_settings.timestep,
            mc_steps=steps_per_iteration,
        )

        solvent_comp, protein_comp, small_mols = system_validation.get_components(stateA)

        # Get the change difference between the end states
        # and check if the charge correction used is appropriate
        charge_difference = _get_alchemical_charge_difference(
            mapping,
            forcefield_settings.nonbonded_method,
            alchem_settings.explicit_charge_correction,
            solvent_comp,
        )

        # 1. Create stateA system
        self.logger.info("Parameterizing molecules")

        # a. create offmol dictionaries and assign partial charges
        # workaround for conformer generation failures
        # see openfe issue #576
        # calculate partial charges manually if not already given
        # convert to OpenFF here,
        # and keep the molecule around to maintain the partial charges
        off_small_mols: dict[str, list[tuple[SmallMoleculeComponent, OFFMolecule]]]
        off_small_mols = {
            "stateA": [(mapping.componentA, mapping.componentA.to_openff())],
            "stateB": [(mapping.componentB, mapping.componentB.to_openff())],
            "both": [
                (m, m.to_openff())
                for m in small_mols
                if (m != mapping.componentA and m != mapping.componentB)
            ],
        }

        self._assign_partial_charges(charge_settings, off_small_mols)

        # b. get a system generator
        if output_settings.forcefield_cache is not None:
            ffcache = shared_basepath / output_settings.forcefield_cache
        else:
            ffcache = None

        # Block out oechem backend in system_generator calls to avoid
        # any issues with smiles roundtripping between rdkit and oechem
        with without_oechem_backend():
            system_generator = system_creation.get_system_generator(
                forcefield_settings=forcefield_settings,
                integrator_settings=integrator_settings,
                thermo_settings=thermo_settings,
                cache=ffcache,
                has_solvent=solvent_comp is not None,
            )

            # c. force the creation of parameters
            # This is necessary because we need to have the FF templates
            # registered ahead of solvating the system.
            for smc, mol in chain(
                off_small_mols["stateA"], off_small_mols["stateB"], off_small_mols["both"]
            ):
                system_generator.create_system(mol.to_topology().to_openmm(), molecules=[mol])

            # c. get OpenMM Modeller + a dictionary of resids for each component
            stateA_modeller, comp_resids = system_creation.get_omm_modeller(
                protein_comp=protein_comp,
                solvent_comp=solvent_comp,
                small_mols=dict(chain(off_small_mols["stateA"], off_small_mols["both"])),
                omm_forcefield=system_generator.forcefield,
                solvent_settings=solvation_settings,
            )

        # d. get topology & positions
        # Note: roundtrip positions to remove vec3 issues
        stateA_topology = stateA_modeller.getTopology()
        stateA_positions = to_openmm(from_openmm(stateA_modeller.getPositions()))

        # e. create the stateA System
        # Block out oechem backend in system_generator calls to avoid
        # any issues with smiles roundtripping between rdkit and oechem
        with without_oechem_backend():
            stateA_system = system_generator.create_system(
                stateA_modeller.topology,
                molecules=[m for _, m in chain(off_small_mols["stateA"], off_small_mols["both"])],
            )

        # 2. Get stateB system
        # a. get the topology
        stateB_topology, stateB_alchem_resids = _rfe_utils.topologyhelpers.combined_topology(
            stateA_topology,
            # zeroth item (there's only one) then get the OFF representation
            off_small_mols["stateB"][0][1].to_topology().to_openmm(),
            exclude_resids=comp_resids[mapping.componentA],
        )

        # b. get a list of small molecules for stateB
        # Block out oechem backend in system_generator calls to avoid
        # any issues with smiles roundtripping between rdkit and oechem
        with without_oechem_backend():
            stateB_system = system_generator.create_system(
                stateB_topology,
                molecules=[m for _, m in chain(off_small_mols["stateB"], off_small_mols["both"])],
            )

        #  c. Define correspondence mappings between the two systems
        ligand_mappings = _rfe_utils.topologyhelpers.get_system_mappings(
            mapping.componentA_to_componentB,
            stateA_system,
            stateA_topology,
            comp_resids[mapping.componentA],
            stateB_system,
            stateB_topology,
            stateB_alchem_resids,
            # These are non-optional settings for this method
            fix_constraints=True,
        )

        # d. if a charge correction is necessary, select alchemical waters
        #    and transform them
        if alchem_settings.explicit_charge_correction:
            alchem_water_resids = _rfe_utils.topologyhelpers.get_alchemical_waters(
                stateA_topology,
                stateA_positions,
                charge_difference,
                alchem_settings.explicit_charge_correction_cutoff,
            )
            _rfe_utils.topologyhelpers.handle_alchemical_waters(
                alchem_water_resids,
                stateB_topology,
                stateB_system,
                ligand_mappings,
                charge_difference,
                solvent_comp,
            )

        #  e. Finally get the positions
        stateB_positions = _rfe_utils.topologyhelpers.set_and_check_new_positions(
            ligand_mappings,
            stateA_topology,
            stateB_topology,
            old_positions=ensure_quantity(stateA_positions, "openmm"),
            insert_positions=ensure_quantity(
                off_small_mols["stateB"][0][1].conformers[0], "openmm"
            ),
        )

        # 3. Create the hybrid topology
        # a. Get softcore potential settings
        if alchem_settings.softcore_LJ.lower() == "gapsys":
            softcore_LJ_v2 = True
        elif alchem_settings.softcore_LJ.lower() == "beutler":
            softcore_LJ_v2 = False
        # b. Get hybrid topology factory
        hybrid_factory = _rfe_utils.relative.HybridTopologyFactory(
            stateA_system,
            stateA_positions,
            stateA_topology,
            stateB_system,
            stateB_positions,
            stateB_topology,
            old_to_new_atom_map=ligand_mappings["old_to_new_atom_map"],
            old_to_new_core_atom_map=ligand_mappings["old_to_new_core_atom_map"],
            use_dispersion_correction=alchem_settings.use_dispersion_correction,
            softcore_alpha=alchem_settings.softcore_alpha,
            softcore_LJ_v2=softcore_LJ_v2,
            softcore_LJ_v2_alpha=alchem_settings.softcore_alpha,
            interpolate_old_and_new_14s=alchem_settings.turn_off_core_unique_exceptions,
        )

        # 4. Create lambda schedule
        # TODO - this should be exposed to users, maybe we should offer the
        # ability to print the schedule directly in settings?
        # fmt: off
        lambdas = _rfe_utils.lambdaprotocol.LambdaProtocol(
            functions=lambda_settings.lambda_functions,
            windows=lambda_settings.lambda_windows
        )
        # fmt: on
        # PR #125 temporarily pin lambda schedule spacing to n_replicas
        n_replicas = sampler_settings.n_replicas
        if n_replicas != len(lambdas.lambda_schedule):
            errmsg = (
                f"Number of replicas {n_replicas} "
                f"does not equal the number of lambda windows "
                f"{len(lambdas.lambda_schedule)}"
            )
            raise ValueError(errmsg)

        # 9. Create the multistate reporter
        # Get the sub selection of the system to print coords for
        selection_indices = hybrid_factory.hybrid_topology.select(output_settings.output_indices)

        #  a. Create the multistate reporter
        # convert checkpoint_interval from time to iterations
        chk_intervals = settings_validation.convert_checkpoint_interval_to_iterations(
            checkpoint_interval=output_settings.checkpoint_interval,
            time_per_iteration=sampler_settings.time_per_iteration,
        )

        nc = shared_basepath / output_settings.output_filename
        chk = output_settings.checkpoint_storage_filename

        if output_settings.positions_write_frequency is not None:
            pos_interval = settings_validation.divmod_time_and_check(
                numerator=output_settings.positions_write_frequency,
                denominator=sampler_settings.time_per_iteration,
                numerator_name="output settings' position_write_frequency",
                denominator_name="sampler settings' time_per_iteration",
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

        reporter = multistate.MultiStateReporter(
            storage=nc,
            analysis_particle_indices=selection_indices,
            checkpoint_interval=chk_intervals,
            checkpoint_storage=chk,
            position_interval=pos_interval,
            velocity_interval=vel_interval,
        )

        #  b. Write out a PDB containing the subsampled hybrid state
        # fmt: off
        bfactors = np.zeros_like(selection_indices, dtype=float)  # solvent
        bfactors[np.in1d(selection_indices, list(hybrid_factory._atom_classes['unique_old_atoms']))] = 0.25  # lig A
        bfactors[np.in1d(selection_indices, list(hybrid_factory._atom_classes['core_atoms']))] = 0.50  # core
        bfactors[np.in1d(selection_indices, list(hybrid_factory._atom_classes['unique_new_atoms']))] = 0.75  # lig B
        # bfactors[np.in1d(selection_indices, protein)] = 1.0  # prot+cofactor
        if len(selection_indices) > 0:
            traj = mdtraj.Trajectory(
                hybrid_factory.hybrid_positions[selection_indices, :],
                hybrid_factory.hybrid_topology.subset(selection_indices),
            ).save_pdb(
                shared_basepath / output_settings.output_structure,
                bfactors=bfactors,
            )
        # fmt: on

        # 10. Get compute platform
        # restrict to a single CPU if running vacuum
        restrict_cpu = forcefield_settings.nonbonded_method.lower() == "nocutoff"
        platform = omm_compute.get_openmm_platform(
            platform_name=protocol_settings.engine_settings.compute_platform,
            gpu_device_index=protocol_settings.engine_settings.gpu_device_index,
            restrict_cpu_count=restrict_cpu,
        )

        # 11. Set the integrator
        # a. Validate integrator settings for current system
        # Virtual sites sanity check - ensure we restart velocities when
        # there are virtual sites in the system
        if hybrid_factory.has_virtual_sites:
            if not integrator_settings.reassign_velocities:
                errmsg = (
                    "Simulations with virtual sites without velocity "
                    "reassignments are unstable in openmmtools"
                )
                raise ValueError(errmsg)

        #  b. create langevin integrator
        integrator = openmmtools.mcmc.LangevinDynamicsMove(
            timestep=to_openmm(integrator_settings.timestep),
            collision_rate=to_openmm(integrator_settings.langevin_collision_rate),
            n_steps=steps_per_iteration,
            reassign_velocities=integrator_settings.reassign_velocities,
            n_restart_attempts=integrator_settings.n_restart_attempts,
            constraint_tolerance=integrator_settings.constraint_tolerance,
        )

        # 12. Create sampler
        self.logger.info("Creating and setting up the sampler")
        rta_its, rta_min_its = settings_validation.convert_real_time_analysis_iterations(
            simulation_settings=sampler_settings,
        )
        # convert early_termination_target_error from kcal/mol to kT
        early_termination_target_error = (
            settings_validation.convert_target_error_from_kcal_per_mole_to_kT(
                thermo_settings.temperature,
                sampler_settings.early_termination_target_error,
            )
        )

        if sampler_settings.sampler_method.lower() == "repex":
            sampler = _rfe_utils.multistate.HybridRepexSampler(
                mcmc_moves=integrator,
                hybrid_factory=hybrid_factory,
                online_analysis_interval=rta_its,
                online_analysis_target_error=early_termination_target_error,
                online_analysis_minimum_iterations=rta_min_its,
            )
        elif sampler_settings.sampler_method.lower() == "sams":
            sampler = _rfe_utils.multistate.HybridSAMSSampler(
                mcmc_moves=integrator,
                hybrid_factory=hybrid_factory,
                online_analysis_interval=rta_its,
                online_analysis_minimum_iterations=rta_min_its,
                flatness_criteria=sampler_settings.sams_flatness_criteria,
                gamma0=sampler_settings.sams_gamma0,
            )
        elif sampler_settings.sampler_method.lower() == "independent":
            sampler = _rfe_utils.multistate.HybridMultiStateSampler(
                mcmc_moves=integrator,
                hybrid_factory=hybrid_factory,
                online_analysis_interval=rta_its,
                online_analysis_target_error=early_termination_target_error,
                online_analysis_minimum_iterations=rta_min_its,
            )

        else:
            raise AttributeError(f"Unknown sampler {sampler_settings.sampler_method}")

        sampler.setup(
            n_replicas=sampler_settings.n_replicas,
            reporter=reporter,
            lambda_protocol=lambdas,
            temperature=to_openmm(thermo_settings.temperature),
            endstates=alchem_settings.endstate_dispersion_correction,
            minimization_platform=platform.getName(),
            # Set minimization steps to None when running in dry mode
            # otherwise do a very small one to avoid NaNs
            minimization_steps=100 if not dry else None,
        )

        try:
            # Create context caches (energy + sampler)
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

            sampler.energy_context_cache = energy_context_cache
            sampler.sampler_context_cache = sampler_context_cache

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

                self.logger.info("Post-simulation analysis of results")
                # calculate relevant analyses of the free energies & sampling
                # First close & reload the reporter to avoid netcdf clashes
                analyzer = multistate_analysis.MultistateEquilFEAnalysis(
                    reporter,
                    sampling_method=sampler_settings.sampler_method.lower(),
                    result_units=unit.kilocalorie_per_mole,
                )
                analyzer.plot(filepath=shared_basepath, filename_prefix="")
                analyzer.close()

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
            return {"nc": nc, "last_checkpoint": chk, **analyzer.unit_results_dict}
        else:
            return {"debug": {"sampler": sampler}}

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

    def _execute(
        self,
        ctx: gufe.Context,
        # protocol: gufe.Protocol, ?
        **kwargs,
    ) -> dict[str, Any]:
        log_system_probe(logging.INFO, paths=[ctx.scratch])

        outputs = self.run(scratch_basepath=ctx.scratch, shared_basepath=ctx.shared)

        return {
            "repeat_id": self._inputs["repeat_id"],
            "generation": self._inputs["generation"],
            **outputs,
        }


class RHTProtocolSimulationUnit(gufe.Protocol):
    def _execute(
        self,
        ctx: gufe.Context,
        *,
        # protocol: gufe.Protocol, ?
        setup_results: gufe.ProtocolUnitResult,
        **kwargs,
    ) -> dict[str, Any]:
        # Should we be doing this so often?
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
            **struct_analysis_outputs,
        }
