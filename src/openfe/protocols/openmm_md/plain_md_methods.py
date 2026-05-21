# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

"""OpenMM MD Protocol --- :mod:`openfe.protocols.openmm_md.plain_md_methods`
===========================================================================================

This module implements the necessary methodology tools to run an MD
simulation using OpenMM tools.

"""

from __future__ import annotations

import logging
import pathlib
import time
import uuid
import warnings
from collections import defaultdict
from typing import Any, Iterable, Optional

import gufe
import mdtraj
import numpy as np
import openmm
import openmm.unit as omm_unit
from gufe import (
    BaseSolventComponent,
    ChemicalSystem,
    SmallMoleculeComponent,
    SolvatedPDBComponent,
    settings,
)
from gufe.protocols.errors import ProtocolUnitExecutionError
from gufe.settings.typing import KelvinQuantity
from mdtraj.reporters import XTCReporter
from openff.toolkit.topology import Molecule as OFFMolecule
from openff.units import Quantity, unit
from openff.units.openmm import from_openmm, to_openmm
from openmm import MonteCarloBarostat, MonteCarloMembraneBarostat

import openfe
from openfe.protocols.openmm_md.plain_md_settings import (
    IntegratorSettings,
    MDOutputSettings,
    MDSimulationSettings,
    OpenFFPartialChargeSettings,
    OpenMMEngineSettings,
    OpenMMSolvationSettings,
    PlainMDProtocolSettings,
)
from openfe.protocols.openmm_utils import (
    charge_generation,
    omm_compute,
    serialization,
    settings_validation,
    system_creation,
    system_validation,
)
from openfe.protocols.openmm_utils.omm_settings import (
    BasePartialChargeSettings,
    FemtosecondQuantity,
)
from openfe.utils import log_system_probe, without_oechem_backend

logger = logging.getLogger(__name__)


class PlainMDProtocolResult(gufe.ProtocolResult):
    """
    Dict-like container for the output of a PlainMDProtocol.

    Provides access to simulation outputs including the pre-minimized
    system PDB and production trajectory files.
    """

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

    def get_traj_filename(self) -> list[pathlib.Path]:
        """
        Get a list of trajectory paths

        Returns
        -------
        traj : list[pathlib.Path]
          list of paths (pathlib.Path) to the simulation trajectory
        """
        traj = [pus[0].outputs["nc"] for pus in self.data.values()]

        return traj

    def get_pdb_filename(self) -> list[pathlib.Path]:
        """
        Get a list of paths to the pdb files of the pre-minimized system.

        Returns
        -------
        pdbs : list[pathlib.Path]
          list of paths (pathlib.Path) to the pdb files
        """
        pdbs = [pus[0].outputs["system_pdb"] for pus in self.data.values()]

        return pdbs


class PlainMDProtocol(gufe.Protocol):
    """
    Protocol for running Molecular Dynamics simulations using OpenMM.

    See Also
    --------
    :mod:`openfe.protocols`
    :class:`openfe.protocols.openmm_md.PlainMDProtocolSettings`
    :class:`openfe.protocols.openmm_md.PlainMDProtocolUnit`
    :class:`openfe.protocols.openmm_md.PlainMDProtocolResult`
    """

    result_cls = PlainMDProtocolResult
    _settings_cls = PlainMDProtocolSettings
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
            partial_charge_settings=OpenFFPartialChargeSettings(),
            solvation_settings=OpenMMSolvationSettings(),
            engine_settings=OpenMMEngineSettings(),
            integrator_settings=IntegratorSettings(),
            simulation_settings=MDSimulationSettings(
                equilibration_length_nvt=0.1 * unit.nanosecond,
                equilibration_length=1.0 * unit.nanosecond,
                production_length=5.0 * unit.nanosecond,
            ),
            output_settings=MDOutputSettings(checkpoint_storage_filename="checkpoint.xml"),
            protocol_repeats=1,
        )

    def _validate(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: Optional[dict[str, gufe.ComponentMapping]] = None,
        extends: Optional[gufe.ProtocolDAGResult] = None,
    ):
        # Check we're not extending
        if extends is not None:
            # This technically should be NotImplementedError
            # but gufe.Protocol.validate calls `_validate` wrapped around an
            # except for NotImplementedError, so we can't raise it here
            raise ValueError("Can't extend simulations yet")

        # Check we're not using a mapping, since we're not doing anything with it
        if mapping is not None:
            wmsg = "A mapping was passed but is not used by this Protocol."
            warnings.warn(wmsg)

        # check that stateA and stateB are the same
        if stateA is not stateB:
            errmsg = "The two end states do not match."
            raise ValueError(errmsg)

        # Validate the ChemicalSystem
        system_validation.validate_chemical_system(stateA)

        # Validate solvent component if present
        nonbond = self.settings.forcefield_settings.nonbonded_method
        system_validation.validate_solvent(stateA, nonbond)

        # Validate the BaseSolventComponents
        base_solvent = stateA.get_components_of_type(BaseSolventComponent)
        if len(base_solvent) > 1:
            errmsg = "Multiple BaseSolventComponents found, only one is supported."
            raise ValueError(errmsg)

        # Validate protein component if present
        system_validation.validate_protein(stateA)

        # Validate the barostat used in combination with the protein component
        system_validation.validate_barostat(stateA, self.settings.integrator_settings.barostat)

        # Validate solvation settings
        settings_validation.validate_openmm_solvation_settings(self.settings.solvation_settings)

        # is the timestep good for the mass?
        settings_validation.validate_timestep(
            self.settings.forcefield_settings.hydrogen_mass,
            self.settings.integrator_settings.timestep,
        )

    def _create(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: Optional[dict[str, gufe.ComponentMapping]] = None,
        extends: Optional[gufe.ProtocolDAGResult] = None,
    ) -> list[gufe.ProtocolUnit]:
        # validate the inputs
        self.validate(stateA=stateA, stateB=stateB, mapping=mapping, extends=extends)

        # actually create and return Units
        # TODO: Deal with multiple ProteinComponents
        solvent_comp, protein_comp, small_mols = system_validation.get_components(stateA)

        system_name = "Solvent MD" if stateA.contains(BaseSolventComponent) else "Vacuum MD"

        for comp in [protein_comp] + small_mols:
            if comp is not None:
                comp_type = comp.__class__.__name__
                if len(comp.name) == 0:
                    comp_name = "NoName"
                else:
                    comp_name = comp.name
                system_name += f" {comp_type}:{comp_name}"

        # make the DAG from the setup and simulation units
        n_repeats = self.settings.protocol_repeats
        units = []
        for i in range(n_repeats):
            repeat_id = int(uuid.uuid4())

            setup = PlainMDSetupUnit(
                protocol=self,
                stateA=stateA,
                generation=0,
                repeat_id=repeat_id,
                name=f"MD Setup: {system_name} repeat {i} generation 0",
            )
            sim = PlainMDSimulationUnit(
                protocol=self,
                stateA=stateA,
                generation=0,
                repeat_id=repeat_id,
                setup_results=setup,
                name=f"MD Simulation: {system_name} repeat {i} generation 0",
            )

            units.extend([setup, sim])

        return units

    def _gather(self, protocol_dag_results: Iterable[gufe.ProtocolDAGResult]) -> dict[str, Any]:
        # result units will have a repeat_id and generations within this
        # repeat_id
        # first group according to repeat_id
        unsorted_repeats = defaultdict(list)
        for d in protocol_dag_results:
            pu: gufe.ProtocolUnitResult
            for pu in d.protocol_unit_results:
                # Only keep the simulation units which are ok
                if ("Simulation" not in pu.name) or (not pu.ok()):
                    continue

                unsorted_repeats[pu.outputs["repeat_id"]].append(pu)

        # then sort by generation within each repeat_id list
        repeats: dict[str, list[gufe.ProtocolUnitResult]] = {}
        for k, v in unsorted_repeats.items():
            repeats[str(k)] = sorted(v, key=lambda x: x.outputs["generation"])

        # returns a dict of repeat_id: sorted list of ProtocolUnitResult
        return repeats


class PlainMDUnitMixin:
    def _prepare(
        self,
        verbose: bool,
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
        scratch_basepath : pathlib.Path | None
          Optional base path to write scratch files to.
        shared_basepath : pathlib.Path | None
          Optional base path to write shared files to.
        """
        self.verbose = verbose

        # set basepaths
        def _set_optional_path(basepath):
            if basepath is None:
                return pathlib.Path(".")
            return basepath

        self.scratch_basepath = _set_optional_path(scratch_basepath)
        self.shared_basepath = _set_optional_path(shared_basepath)


class PlainMDSetupUnit(PlainMDUnitMixin, gufe.ProtocolUnit):
    """
    Protocol setup unit for plain MD simulations which handles charging, system building and solvation.
    """

    @staticmethod
    def _assign_partial_charges(
        charge_settings: OpenFFPartialChargeSettings,
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
                method=charge_settings.partial_charge_method,
                toolkit_backend=charge_settings.off_toolkit_backend,
                generate_n_conformers=charge_settings.number_of_conformers,
                nagl_model=charge_settings.nagl_model,
            )

    def run(
        self,
        *,
        dry: bool = False,
        verbose: bool = True,
        scratch_basepath: pathlib.Path | None = None,
        shared_basepath: pathlib.Path | None = None,
    ) -> dict[str, Any]:
        """Setup a plain MD system.

        Parameters
        ----------
        dry : bool
          Do a dry run of the calculation, creating all necessary hybrid
          system components (topology, system, sampler, etc...) but without
          running the simulation.
        verbose : bool
          Verbose output of the simulation progress. Output is provided via
          INFO level logging.
        scratch_basepath: pathlib.Path | None
          Where to store temporary files, defaults to current working directory
        shared_basepath : pathlib.Path | None
          Where to run the calculation, defaults to current working directory

        Returns
        -------
        dict
          Outputs created by the setup unit or the debug objects
          (e.g. HybridTopologyFactory) if ``dry==True``.

        Raises
        ------
        error
          Exception if anything failed
        """
        # Prepare paths and set verbosity
        self._prepare(verbose, scratch_basepath, shared_basepath)

        if verbose:
            self.logger.info("Creating system")

        # 0. General setup and settings dependency resolution step
        # Extract relevant settings
        protocol_settings: PlainMDProtocolSettings = self._inputs["protocol"].settings
        stateA = self._inputs["stateA"]

        forcefield_settings: settings.OpenMMSystemGeneratorFFSettings = (
            protocol_settings.forcefield_settings
        )
        thermo_settings: settings.ThermoSettings = protocol_settings.thermo_settings
        solvation_settings: OpenMMSolvationSettings = protocol_settings.solvation_settings
        charge_settings: BasePartialChargeSettings = protocol_settings.partial_charge_settings
        sim_settings: MDSimulationSettings = protocol_settings.simulation_settings
        output_settings: MDOutputSettings = protocol_settings.output_settings
        integrator_settings: IntegratorSettings = protocol_settings.integrator_settings
        timestep = integrator_settings.timestep

        # is the timestep good for the mass?
        settings_validation.validate_timestep(forcefield_settings.hydrogen_mass, timestep)

        # do step validation early and pass through the units
        if sim_settings.equilibration_length_nvt is not None:
            equil_steps_nvt = settings_validation.get_simsteps(
                sim_length=sim_settings.equilibration_length_nvt,
                timestep=timestep,
                mc_steps=1,
            )
        else:
            equil_steps_nvt = None

        equil_steps_npt = settings_validation.get_simsteps(
            sim_length=sim_settings.equilibration_length,
            timestep=timestep,
            mc_steps=1,
        )
        prod_steps = settings_validation.get_simsteps(
            sim_length=sim_settings.production_length,
            timestep=timestep,
            mc_steps=1,
        )

        solvent_comp, protein_comp, small_mols = system_validation.get_components(stateA)
        if isinstance(protein_comp, SolvatedPDBComponent):
            solvent_comp = protein_comp

        # 1. Create stateA system
        # Create a dictionary of OFFMol for each SMC for bookkeeping
        smc_components: dict[SmallMoleculeComponent, OFFMolecule] = {
            i: i.to_openff() for i in small_mols
        }

        # a. assign partial charges to smcs
        self._assign_partial_charges(charge_settings, smc_components)

        # b. get a system generator
        if output_settings.forcefield_cache is not None:
            ffcache = self.shared_basepath / output_settings.forcefield_cache
        else:
            ffcache = None

        # Note: we block out the oechem backend for all systemgenerator
        # linked operations to avoid any smiles operations that can
        # go wrong when doing rdkit->OEchem roundtripping
        with without_oechem_backend():
            system_generator = system_creation.get_system_generator(
                forcefield_settings=forcefield_settings,
                integrator_settings=integrator_settings,
                thermo_settings=thermo_settings,
                cache=ffcache,
                has_solvent=solvent_comp is not None,
            )

            # Force creation of smc templates so we can solvate later
            for mol in smc_components.values():
                system_generator.create_system(mol.to_topology().to_openmm(), molecules=[mol])

            # c. get OpenMM Modeller + a resids dictionary for each component
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
            stateA_positions = to_openmm(from_openmm(stateA_modeller.getPositions()))

            # e. create the stateA System
            stateA_system = system_generator.create_system(
                stateA_topology,
                molecules=[s.to_openff() for s in small_mols],
            )

        # f. Save pdb of entire system topology to file, this is always needed for restarts
        with open(self.shared_basepath / output_settings.preminimized_structure, "w") as f:
            openmm.app.PDBFile.writeFile(stateA_topology, stateA_positions, file=f, keepIds=True)

        # g. Save the system and positions to file
        system_outfile = self.shared_basepath / "system.xml.bz2"
        serialization.serialize(stateA_system, system_outfile)
        positions_outfile = self.shared_basepath / "input_positions.npy"
        np.save(positions_outfile, stateA_positions.value_in_unit(omm_unit.nanometers))

        unit_results_dict = {
            "system": system_outfile,
            # save the positions to higher precision
            "positions": positions_outfile,
            "system_pdb": self.shared_basepath / output_settings.preminimized_structure,
            "equil_steps_nvt": equil_steps_nvt,
            "equil_steps_npt": equil_steps_npt,
            "prod_steps": prod_steps,
        }
        if dry:
            # add non serialised stuff for testing
            debug_info = {
                "system": stateA_system,
                "positions": stateA_positions,
                "topology": stateA_topology,
            }
            unit_results_dict["debug"] = debug_info

        return unit_results_dict

    def _execute(
        self,
        ctx: gufe.Context,
        **kwargs,
    ) -> dict[str, Any]:
        log_system_probe(logging.INFO, paths=[ctx.scratch])

        outputs = self.run(scratch_basepath=ctx.scratch, shared_basepath=ctx.shared)

        return {
            "repeat_id": self._inputs["repeat_id"],
            "generation": self._inputs["generation"],
            # track some version restart info to check compatibility
            "openmm_version": openmm.__version__,
            "openfe_version": openfe.__version__,
            "gufe_version": gufe.__version__,
            **outputs,
        }


class PlainMDSimulationUnit(PlainMDUnitMixin, gufe.ProtocolUnit):
    """
    Protocol unit for plain MD simulation equilibration and production runs (NonTransformation).
    """

    @staticmethod
    def _check_restart(output_settings: MDOutputSettings, shared_path: pathlib.Path):
        """
        Check if we are doing a restart.

        Parameters
        ----------
        output_settings : MDOutputSettings
          The simulation output settings
        shared_path : pathlib.Path
          The shared directory where we should be looking for existing files.

        Notes
        -----
        For now this just checks if the checkpoint state file is present in the
        shared directory but in the future this may expand depending on
        how warehouse works.
        """
        checkpoint = shared_path / output_settings.checkpoint_storage_filename

        return checkpoint.is_file()

    @staticmethod
    def _verify_execution_environment(
        setup_outputs: dict[str, Any],
    ) -> None:
        """
        Check that the Python environment hasn't changed based on the
        relevant Python library versions stored in the setup outputs.
        """
        try:
            if (
                (gufe.__version__ != setup_outputs["gufe_version"])
                or (openfe.__version__ != setup_outputs["openfe_version"])
                or (openmm.__version__ != setup_outputs["openmm_version"])
            ):
                errmsg = "Python environment has changed, cannot continue Protocol execution."
                raise ProtocolUnitExecutionError(errmsg)
        except KeyError:
            errmsg = "Missing environment information from setup outputs."
            raise ProtocolUnitExecutionError(errmsg)

    @staticmethod
    def _save_pdb_subset(
        simulation: openmm.app.Simulation,
        output_settings: MDOutputSettings,
        file_name: pathlib.Path,
    ):
        # get the positions
        positions = to_openmm(
            from_openmm(
                simulation.context.getState(
                    getPositions=True, enforcePeriodicBox=False
                ).getPositions()
            )
        )
        # get the subset from the output settings
        mdtraj_top = mdtraj.Topology.from_openmm(simulation.topology)
        selection_indices = mdtraj_top.select(output_settings.output_indices)
        traj = mdtraj.Trajectory(
            positions[selection_indices, :],
            mdtraj_top.subset(selection_indices),
        )
        traj.save_pdb(file_name)

    @staticmethod
    def _run_dynamics(
        simulation: openmm.app.Simulation,
        steps: int,
        temperature: KelvinQuantity,
        barostat_frequency: Quantity,
        output_settings: MDOutputSettings,
        verbose: bool = True,
        output_path: None | pathlib.Path = None,
        reinitialize_velocities: bool = True,
    ):
        """
        Worker method to set the temperature, barostat and run dynamics and save final structure output.
        """
        # only set the velocities to temperature if we are not restarting this section
        if reinitialize_velocities:
            # set the velocities to temperature
            simulation.context.setVelocitiesToTemperature(to_openmm(temperature))

        # Setup the barostat
        for x in simulation.context.getSystem().getForces():
            if isinstance(x, (MonteCarloBarostat, MonteCarloMembraneBarostat)):
                x.setFrequency(barostat_frequency.m)

        # run the simulation
        t0 = time.time()
        simulation.step(steps)
        t1 = time.time()
        if verbose:
            logger.info(f"Completed dynamics in {t1 - t0} seconds")

        # save the final frame if a file path is passed
        if output_path is not None:
            PlainMDSimulationUnit._save_pdb_subset(
                simulation,
                output_settings,
                output_path,
            )

    @staticmethod
    def _get_remaining_steps(
        current_step_count: int,
        equil_steps_nvt: int,
        equil_steps_npt: int,
        prod_steps: int,
    ) -> tuple[int, int, int, bool]:
        """
        Work out the remaining steps for each phase of the simulation based on the current step count,
        and determine if production has already started.

        Returns
        -------
        equil_steps_nvt : int
            The number of nvt steps left to run
        equil_steps_npt : int
            The number of npt steps left to run
        prod_steps : int
            The number of production steps left to run
        production_started : bool
            Whether the production phase has already started or not
        """
        nvt_end = equil_steps_nvt
        npt_end = equil_steps_nvt + equil_steps_npt
        prod_end = equil_steps_nvt + equil_steps_npt + prod_steps

        if npt_end < current_step_count <= prod_end:
            # In the production phase
            return 0, 0, prod_end - current_step_count, True

        elif nvt_end < current_step_count <= npt_end:
            # In the NPT equilibration phase
            return 0, npt_end - current_step_count, prod_steps, False

        else:
            # In the NVT equilibration phase
            return nvt_end - current_step_count, equil_steps_npt, prod_steps, False

    @staticmethod
    def _run_MD(
        simulation: openmm.app.Simulation,
        positions: omm_unit.Quantity,
        simulation_settings: MDSimulationSettings,
        output_settings: MDOutputSettings,
        temperature: KelvinQuantity,
        barostat_frequency: Quantity,
        timestep: FemtosecondQuantity,
        equil_steps_nvt: int | None,
        equil_steps_npt: int,
        prod_steps: int,
        verbose: bool = True,
        shared_basepath: pathlib.Path | None = None,
        restart: bool = False,
    ) -> None:
        """
        Energy minimization, Equilibration and Production MD to be reused
        in multiple protocols

        Parameters
        ----------
        simulation : openmm.app.Simulation
          An OpenMM simulation to simulate.
        positions : openmm.unit.Quantity
          Initial positions for the system.
        simulation_settings : SimulationSettingsMD
          Settings for MD simulation
        output_settings: OutputSettingsMD
          Settings for output of MD simulation
        temperature: KelvinQuantity
          temperature setting
        barostat_frequency: openff.units.Quantity
          Frequency for the barostat
        timestep: FemtosecondQuantity
          Simulation integration timestep
        equil_steps_nvt: Optional[int]
          number of steps for NVT equilibration
          if None, no NVT equilibration will be performed
        equil_steps_npt: int
          number of steps for NPT equilibration
        prod_steps: int
          number of steps for the production run
        verbose: bool
          Verbose output of the simulation progress. Output is provided via
          INFO level logging.
        shared_basepath : Pathlike, optional
          Where to run the calculation, defaults to current working directory
        restart: bool, optional, default=False
          Whether we are restarting from a previous simulation or not, the checkpoint file should be
          present in the shared directories.

        """
        if shared_basepath is None:
            shared_basepath = pathlib.Path(".")

        # get the checkpointing interval for states and positions
        checkpoint_interval = settings_validation.get_simsteps(
            sim_length=output_settings.checkpoint_interval,
            timestep=timestep,
            mc_steps=1,
        )

        # as nvt steps can be None set to 0 in this case
        equil_steps_nvt = equil_steps_nvt or 0

        # track if production has already been started
        production_started = False
        # track if we need to reinitialize velocities for a phase
        # on a fresh run, reinitialize velocities for the first phase.
        # on a restart, preserve the checkpoint velocities for the phase being restarted.
        reinitialize_velocities = not restart
        # if restarting skip setup and minimization as they should be completed by the time the checkpoint reporter is used
        if restart:
            if verbose:
                logger.info("Restarting simulation from checkpoint state")
            simulation.loadState(str(shared_basepath / output_settings.checkpoint_storage_filename))

            # workout the number of steps to run in each phase based on the current simulation step count
            current_step_count = simulation.context.getStepCount()
            equil_steps_nvt, equil_steps_npt, prod_steps, production_started = (
                PlainMDSimulationUnit._get_remaining_steps(
                    current_step_count=current_step_count,
                    equil_steps_nvt=equil_steps_nvt,
                    equil_steps_npt=equil_steps_npt,
                    prod_steps=prod_steps,
                )
            )

        else:
            # this is the non restart case and requires minimization before moving on
            simulation.context.setPositions(positions)
            # minimize
            if verbose:
                logger.info("Minimizing systems")

            simulation.minimizeEnergy(maxIterations=simulation_settings.minimization_steps)

            if output_settings.minimized_structure:
                PlainMDSimulationUnit._save_pdb_subset(
                    simulation,
                    output_settings,
                    shared_basepath / output_settings.minimized_structure,
                )

        # add the checkpoint reporter so we can recover during the equilibration / production phases
        if output_settings.checkpoint_storage_filename:
            simulation.reporters.append(
                openmm.app.CheckpointReporter(
                    file=str(shared_basepath / output_settings.checkpoint_storage_filename),
                    reportInterval=checkpoint_interval,
                    writeState=True,  # writes portable XML via simulation.saveState()
                )
            )

        # equilibrate
        # NVT equilibration
        if equil_steps_nvt > 0:
            if verbose:
                logger.info(f"Running NVT equilibration for {equil_steps_nvt} steps")
            # setup the output path if we have one for the nvt equilibration
            if output_settings.equil_nvt_structure is not None:
                output_path = shared_basepath / output_settings.equil_nvt_structure
            else:
                output_path = None
            PlainMDSimulationUnit._run_dynamics(
                simulation=simulation,
                steps=equil_steps_nvt,
                temperature=temperature,
                barostat_frequency=0 * unit.timestep,  # turn off the barostat for this stage
                output_settings=output_settings,
                verbose=verbose,
                output_path=output_path,
                reinitialize_velocities=reinitialize_velocities,
            )
            # if we have run this stage we then need to reinitialize velocities in the next stages
            reinitialize_velocities = True

        # NPT equilibration
        if equil_steps_npt > 0:
            if verbose:
                logger.info(f"Running NPT equilibration for {equil_steps_npt} steps")
            # setup the output path if we have one for the npt equilibration
            if output_settings.equil_npt_structure is not None:
                output_path = shared_basepath / output_settings.equil_npt_structure
            else:
                output_path = None

            PlainMDSimulationUnit._run_dynamics(
                simulation=simulation,
                steps=equil_steps_npt,
                temperature=temperature,
                barostat_frequency=barostat_frequency,
                output_settings=output_settings,
                verbose=verbose,
                output_path=output_path,
                reinitialize_velocities=reinitialize_velocities,
            )
            # the production stage can use these same velocities
            reinitialize_velocities = False

        # production
        if verbose:
            logger.info(f"Running production phase for {prod_steps} steps")

        # Setup the reporters
        write_interval = settings_validation.divmod_time_and_check(
            output_settings.trajectory_write_interval,
            timestep,
            "trajectory_write_interval",
            "timestep",
        )

        if output_settings.production_trajectory_filename:
            # Get the sub selection of the system to save coords for
            selection_indices = mdtraj.Topology.from_openmm(simulation.topology).select(
                output_settings.output_indices
            )
            xtc_reporter = XTCReporter(
                file=str(shared_basepath / output_settings.production_trajectory_filename),
                reportInterval=write_interval,
                atomSubset=selection_indices,
                # append to the trajectory if restarting and we have run the production stage before
                append=production_started,
            )
            simulation.reporters.append(xtc_reporter)

        if output_settings.log_output:
            simulation.reporters.append(
                openmm.app.StateDataReporter(
                    str(shared_basepath / output_settings.log_output),
                    checkpoint_interval,
                    step=True,
                    time=True,
                    potentialEnergy=True,
                    kineticEnergy=True,
                    totalEnergy=True,
                    temperature=True,
                    volume=True,
                    density=True,
                    speed=True,
                    append=production_started,
                )
            )

        PlainMDSimulationUnit._run_dynamics(
            simulation=simulation,
            steps=prod_steps,
            temperature=temperature,
            barostat_frequency=barostat_frequency,
            output_settings=output_settings,
            verbose=verbose,
            output_path=None,  # the trajectory is saved for the production run so don't save again
            reinitialize_velocities=reinitialize_velocities,
        )

    def run(
        self,
        *,
        system: openmm.System,
        positions: openmm.unit.Quantity,
        topology: openmm.app.Topology,
        equil_steps_nvt: int | None,
        equil_steps_npt: int,
        prod_steps: int,
        dry: bool = False,
        verbose: bool = True,
        scratch_basepath: pathlib.Path | None = None,
        shared_basepath: pathlib.Path | None = None,
    ) -> dict[str, Any]:
        """Run the MD simulation.

        Parameters
        ----------
        system : openmm.System
          The System to simulate.
        positions : openmm.unit.Quantity
          The positions of the System.
        topology: openmm.app.Topology
            The topology of the System.
        equil_steps_nvt : int
            The number of nvt equilibration steps.
        equil_steps_npt : int
            The number of npt equilibration steps.
        prod_steps : int
            The number of production steps.
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
        # Prepare paths and set verbosity
        self._prepare(verbose, scratch_basepath, shared_basepath)

        # Extract relevant settings
        protocol_settings: PlainMDProtocolSettings = self._inputs["protocol"].settings

        forcefield_settings: settings.OpenMMSystemGeneratorFFSettings = (
            protocol_settings.forcefield_settings
        )
        thermo_settings: settings.ThermoSettings = protocol_settings.thermo_settings
        sim_settings: MDSimulationSettings = protocol_settings.simulation_settings
        output_settings: MDOutputSettings = protocol_settings.output_settings
        timestep = protocol_settings.integrator_settings.timestep
        integrator_settings = protocol_settings.integrator_settings

        # Get platform
        restrict_cpu = forcefield_settings.nonbonded_method.lower() == "nocutoff"
        platform = omm_compute.get_openmm_platform(
            platform_name=protocol_settings.engine_settings.compute_platform,
            gpu_device_index=protocol_settings.engine_settings.gpu_device_index,
            restrict_cpu_count=restrict_cpu,
        )

        # Set the integrator
        integrator = openmm.LangevinMiddleIntegrator(
            to_openmm(thermo_settings.temperature),
            to_openmm(integrator_settings.langevin_collision_rate),
            to_openmm(timestep),
        )
        # Build the simulation
        simulation = openmm.app.Simulation(
            topology,
            system,
            integrator,
            platform,
        )

        try:
            if not dry:  # pragma: no-cover
                # check for a restart
                restart = self._check_restart(output_settings, self.shared_basepath)
                # start the simulation
                self._run_MD(
                    simulation,
                    positions,
                    sim_settings,
                    output_settings,
                    thermo_settings.temperature,
                    integrator_settings.barostat_frequency,
                    timestep,
                    equil_steps_nvt,
                    equil_steps_npt,
                    prod_steps,
                    shared_basepath=self.shared_basepath,
                    restart=restart,
                    verbose=self.verbose,
                )

        finally:
            if not dry:
                del integrator, simulation

        if not dry:  # pragma: no-cover
            output = {
                "system_pdb": self.shared_basepath / output_settings.preminimized_structure,
                "minimized_pdb": self.shared_basepath / output_settings.minimized_structure,
                "nc": self.shared_basepath / output_settings.production_trajectory_filename,
                "last_checkpoint": self.shared_basepath
                / output_settings.checkpoint_storage_filename,
            }
            # The checkpoint file can not exist if frequency > sim length
            if not output["last_checkpoint"].exists():
                output["last_checkpoint"] = None

            # The NVT PDB can be omitted if we don't run the simulation
            # Note: we could also just check the file exist
            if (
                output_settings.equil_nvt_structure
                and sim_settings.equilibration_length_nvt is not None
            ):
                output["nvt_equil_pdb"] = self.shared_basepath / output_settings.equil_nvt_structure
            else:
                output["nvt_equil_pdb"] = None

            if output_settings.equil_npt_structure:
                output["npt_equil_pdb"] = self.shared_basepath / output_settings.equil_npt_structure
            else:
                output["npt_equil_pdb"] = None

            return output
        else:
            return {"debug": {"system": system}}

    def _execute(
        self,
        ctx: gufe.Context,
        setup_results,
        **kwargs,
    ) -> dict[str, Any]:
        log_system_probe(logging.INFO, paths=[ctx.scratch])
        # Ensure that the environment hasn't changed
        self._verify_execution_environment(setup_results.outputs)

        # Get the relevant inputs for running the unit
        system = serialization.deserialize(setup_results.outputs["system"])
        positions = (
            np.load(setup_results.outputs["positions"]) * omm_unit.nanometers
        )  # convert to openmm units
        topology = openmm.app.PDBFile(str(setup_results.outputs["system_pdb"])).getTopology()
        equil_steps_nvt = setup_results.outputs["equil_steps_nvt"]
        equil_steps_npt = setup_results.outputs["equil_steps_npt"]
        prod_steps = setup_results.outputs["prod_steps"]

        outputs = self.run(
            system=system,
            positions=positions,
            topology=topology,
            equil_steps_nvt=equil_steps_nvt,
            equil_steps_npt=equil_steps_npt,
            prod_steps=prod_steps,
            scratch_basepath=ctx.scratch,
            shared_basepath=ctx.shared,
        )

        return {
            "repeat_id": self._inputs["repeat_id"],
            "generation": self._inputs["generation"],
            **outputs,
        }
