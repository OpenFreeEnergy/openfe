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
from openfe.protocols.openmm_utils.base_equilibration import (
    BaseEquilibrationUnit, EquilibrationEntry, EquilibrationInputs,
)
from openfe.protocols.openmm_utils.offmolecule_utils import (
    _get_offmol_metadata,
    _get_unique_name,
    _get_used_offmol_property,
    _next_available_number,
    _set_offmol_metadata,
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

        # make the DAG from setup, equilibration, and production units
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
            equil = PlainMDEquilibrationUnit(
                protocol=self,
                stateA=stateA,
                generation=0,
                repeat_id=repeat_id,
                setup_results=setup,
                name=f"MD Equilibration: {system_name} repeat {i} generation 0",
            )
            sim = PlainMDSimulationUnit(
                protocol=self,
                stateA=stateA,
                generation=0,
                repeat_id=repeat_id,
                setup_results=setup,
                equilibration_results=equil,
                name=f"MD Simulation: {system_name} repeat {i} generation 0",
            )

            units.extend([setup, equil, sim])

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

    @classmethod
    def _run_dynamics(
        cls,
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
            simulation.context.setVelocitiesToTemperature(
                to_openmm(temperature))

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
            cls._save_pdb_subset(
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
    def _create_simulation(
        *,
        system: openmm.System,
        topology: openmm.app.Topology,
        protocol_settings: PlainMDProtocolSettings,
    ) -> tuple[openmm.app.Simulation, openmm.Integrator]:
        """Create a PlainMD OpenMM Simulation."""

        forcefield_settings = protocol_settings.forcefield_settings
        thermo_settings = protocol_settings.thermo_settings
        integrator_settings = protocol_settings.integrator_settings
        engine_settings = protocol_settings.engine_settings

        restrict_cpu = (
                forcefield_settings.nonbonded_method.lower() == "nocutoff"
        )
        platform = omm_compute.get_openmm_platform(
            platform_name=engine_settings.compute_platform,
            gpu_device_index=engine_settings.gpu_device_index,
            restrict_cpu_count=restrict_cpu,
        )

        integrator = openmm.LangevinMiddleIntegrator(
            to_openmm(thermo_settings.temperature),
            to_openmm(integrator_settings.langevin_collision_rate),
            to_openmm(integrator_settings.timestep),
        )

        simulation = openmm.app.Simulation(
            topology,
            system,
            integrator,
            platform,
        )

        return simulation, integrator

    @staticmethod
    def _add_checkpoint_reporter(
        simulation: openmm.app.Simulation,
        checkpoint_path: pathlib.Path | None,
        checkpoint_interval: int,
    ) -> None:
        """Attach a portable XML State checkpoint reporter."""

        if checkpoint_path is not None:
            simulation.reporters.append(
                openmm.app.CheckpointReporter(
                    file=str(checkpoint_path),
                    reportInterval=checkpoint_interval,
                    writeState=True,
                )
            )

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

        Legacy all-in-one path: still used by the ABFE and SepTop
        pre-equilibration. Remove once those protocols use BaseEquilibrationUnit.

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
            simulation.loadState(
                str(shared_basepath / output_settings.checkpoint_storage_filename))

            # workout the number of steps to run in each phase based on the current simulation step count
            current_step_count = simulation.context.getStepCount()
            equil_steps_nvt, equil_steps_npt, prod_steps, production_started = (
                PlainMDUnitMixin._get_remaining_steps(
                    # was PlainMDSimulationUnit
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

            simulation.minimizeEnergy(
                maxIterations=simulation_settings.minimization_steps)

            if output_settings.minimized_structure:
                PlainMDUnitMixin._save_pdb_subset(  # was PlainMDSimulationUnit
                    simulation,
                    output_settings,
                    shared_basepath / output_settings.minimized_structure,
                )

        # add the checkpoint reporter so we can recover during the equilibration / production phases
        if output_settings.checkpoint_storage_filename:
            simulation.reporters.append(
                openmm.app.CheckpointReporter(
                    file=str(
                        shared_basepath / output_settings.checkpoint_storage_filename),
                    reportInterval=checkpoint_interval,
                    writeState=True,
                    # writes portable XML via simulation.saveState()
                )
            )

        # equilibrate
        # NVT equilibration
        if equil_steps_nvt > 0:
            if verbose:
                logger.info(
                    f"Running NVT equilibration for {equil_steps_nvt} steps")
            # setup the output path if we have one for the nvt equilibration
            if output_settings.equil_nvt_structure is not None:
                output_path = shared_basepath / output_settings.equil_nvt_structure
            else:
                output_path = None
            PlainMDUnitMixin._run_dynamics(  # was PlainMDSimulationUnit
                simulation=simulation,
                steps=equil_steps_nvt,
                temperature=temperature,
                barostat_frequency=0 * unit.timestep,
                # turn off the barostat for this stage
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
                logger.info(
                    f"Running NPT equilibration for {equil_steps_npt} steps")
            # setup the output path if we have one for the npt equilibration
            if output_settings.equil_npt_structure is not None:
                output_path = shared_basepath / output_settings.equil_npt_structure
            else:
                output_path = None

            PlainMDUnitMixin._run_dynamics(  # was PlainMDSimulationUnit
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
            selection_indices = mdtraj.Topology.from_openmm(
                simulation.topology).select(
                output_settings.output_indices
            )
            xtc_reporter = XTCReporter(
                file=str(
                    shared_basepath / output_settings.production_trajectory_filename),
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

        PlainMDUnitMixin._run_dynamics(  # was PlainMDSimulationUnit
            simulation=simulation,
            steps=prod_steps,
            temperature=temperature,
            barostat_frequency=barostat_frequency,
            output_settings=output_settings,
            verbose=verbose,
            output_path=None,
            # the trajectory is saved for the production run so don't save again
            reinitialize_velocities=reinitialize_velocities,
        )


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

        # Set the solvent component to be the protein component if the latter
        # is already solvated.
        if isinstance(protein_comp, SolvatedPDBComponent):
            solvent_comp = protein_comp

        # 1. Create stateA system
        # Create a dictionary of OFFMol for each SMC for bookkeeping
        smc_components: dict[SmallMoleculeComponent, OFFMolecule] = {
            i: i.to_openff() for i in small_mols
        }

        # a. assign residue names / numbers to the small molecules.
        # First gather all the unique names & resnums
        used_names = _get_used_offmol_property(list(smc_components.values()), "residue_name")
        used_resnums = {
            int(i)
            for i in _get_used_offmol_property(list(smc_components.values()), "residue_number")
        }

        # We assign everything LIG unless LIG is already a name
        resname = _get_unique_name("LIG", "LG", used_names)

        # Now we assign all the residue names & numbers
        for offmol in smc_components.values():
            name = _get_offmol_metadata(offmol, "residue_name")
            num = _get_offmol_metadata(offmol, "residue_number")
            if name is None:
                _set_offmol_metadata(offmol, "residue_name", resname)
            if num is None:
                resnum = _next_available_number(used_resnums)
                _set_offmol_metadata(offmol, "residue_number", str(resnum))
                used_resnums.add(resnum)

        # b. assign partial charges to smcs
        self._assign_partial_charges(charge_settings, smc_components)

        # c. get a system generator
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

            # d. get OpenMM Modeller + a resids dictionary for each component
            stateA_modeller, comp_resids = system_creation.get_omm_modeller(
                protein_comp=protein_comp,
                solvent_comp=solvent_comp,
                small_mols=smc_components,
                omm_forcefield=system_generator.forcefield,
                solvent_settings=solvation_settings,
            )

            # e. get topology & positions
            # Note: roundtrip positions to remove vec3 issues
            stateA_topology = stateA_modeller.getTopology()
            stateA_positions = to_openmm(from_openmm(stateA_modeller.getPositions()))

            # f. create the stateA System
            stateA_system = system_generator.create_system(
                stateA_topology,
                molecules=[s.to_openff() for s in small_mols],
            )

        # g. Save pdb of entire system topology to file, this is always needed for restarts
        with open(self.shared_basepath / output_settings.preminimized_structure, "w") as f:
            openmm.app.PDBFile.writeFile(stateA_topology, stateA_positions, file=f, keepIds=True)

        # h. Save the system and positions to file
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


class PlainMDEquilibrationUnit(PlainMDUnitMixin, BaseEquilibrationUnit):
    """Run PlainMD's existing minimization, optional NVT, and NPT stages.

    ``system_pdb`` from PlainMD setup describes the entire System. A subset
    PDB produced for visualization cannot construct an equivalent Simulation.
    The common base does not define the physics: future alchemical protocols
    must construct their own Systems, parameters, and equilibration schedules.
    """

    def _execute(
        self,
        ctx: gufe.Context,
        setup_results: gufe.ProtocolUnitResult,
        **kwargs,
    ) -> dict[str, Any]:
        log_system_probe(logging.INFO, paths=[ctx.scratch])
        setup = setup_results.outputs

        # Ensure that the environment hasn't changed
        self._verify_execution_environment(setup)

        # PlainMD setup serializes the full System and topology. Its NumPy
        # coordinates are stored in nanometers, so restore the units here.
        system = serialization.deserialize(setup["system"])
        positions = np.load(setup["positions"]) * omm_unit.nanometers
        topology = openmm.app.PDBFile(str(setup["system_pdb"])).getTopology()

        if len(positions) != system.getNumParticles():
            raise ValueError("Setup positions do not match System particle count")
        if topology.getNumAtoms() != system.getNumParticles():
            raise ValueError("Setup topology must describe the entire OpenMM System")

        # Setup already validated the step counts; do not calculate them again.
        outputs = self.run(
            equilibration_inputs=EquilibrationInputs(
                states=[None],
                systems=system,
                positions=positions,
                topologies=topology,
            ),
            equil_steps_nvt=setup["equil_steps_nvt"] or 0,
            equil_steps_npt=setup["equil_steps_npt"],
            scratch_basepath=ctx.scratch,
            shared_basepath=ctx.shared,
        )
        return {
            "repeat_id": self._inputs["repeat_id"],
            "generation": self._inputs["generation"],
            **outputs,
        }

    @staticmethod
    def _get_remaining_equilibration_steps(
            *, current_step_count: int, equil_steps_nvt: int, equil_steps_npt: int
    ) -> tuple[int, int]:
        """Return NVT and NPT steps left from an equilibration checkpoint.

        Step counts are absolute across the two dynamics phases; minimization
        does not increment the OpenMM step count. Checkpoints from production
        are not valid equilibration checkpoints.
        """
        nvt_end = equil_steps_nvt
        npt_end = nvt_end + equil_steps_npt
        if current_step_count < 0 or current_step_count > npt_end:
            raise ValueError(
                "Checkpoint step count is outside the equilibration interval: "
                f"step={current_step_count}, equilibration_end_step={npt_end}"
            )
        nvt_remaining = max(0, nvt_end - current_step_count)
        npt_remaining = npt_end - max(nvt_end, current_step_count)
        return nvt_remaining, npt_remaining

    def _equilibrate_state(
        self,
        *,
        state_input: EquilibrationEntry,
        output_dir: pathlib.Path,
        scratch_dir: pathlib.Path,
        dry: bool,
        verbose: bool,
        equil_steps_nvt: int,
        equil_steps_npt: int,
    ) -> dict[str, Any]:
        """Minimize and equilibrate, resuming NVT or NPT when possible.

        A portable checkpoint written during equilibration skips minimization
        and preserves velocities within the resumed phase. Starting NPT after
        NVT reinitializes velocities as in the original PlainMD implementation.
        The production unit manages its own checkpoint separately.
        """
        self._prepare(verbose, scratch_dir, output_dir)

        protocol_settings: PlainMDProtocolSettings = self._inputs["protocol"].settings
        thermo_settings = protocol_settings.thermo_settings
        sim_settings = protocol_settings.simulation_settings
        output_settings = protocol_settings.output_settings
        integrator_settings = protocol_settings.integrator_settings

        simulation, integrator = self._create_simulation(
            system=state_input.system,
            topology=state_input.topology,
            protocol_settings=protocol_settings,
        )
        try:
            if dry:
                return {"debug": {"system": state_input.system}}

            checkpoint_name = output_settings.checkpoint_storage_filename
            checkpoint_path = output_dir / checkpoint_name if checkpoint_name else None
            restart = checkpoint_path is not None and checkpoint_path.is_file()
            if restart:
                if verbose:
                    logger.info("Restarting simulation from checkpoint state")
                simulation.loadState(str(checkpoint_path))
            else:
                simulation.context.setPositions(state_input.positions)
                if verbose:
                    logger.info("Minimizing systems")
                simulation.minimizeEnergy(
                    maxIterations=sim_settings.minimization_steps)
                if output_settings.minimized_structure:
                    self._save_pdb_subset(
                        simulation,
                        output_settings,
                        output_dir / output_settings.minimized_structure,
                    )

            current_step_count = simulation.context.getStepCount()
            nvt_remaining, npt_remaining = self._get_remaining_equilibration_steps(
                current_step_count=current_step_count,
                equil_steps_nvt=equil_steps_nvt,
                equil_steps_npt=equil_steps_npt,
            )

            # Checkpointing starts after minimization and covers NVT/NPT.
            if checkpoint_path is not None:
                checkpoint_interval = settings_validation.get_simsteps(
                    sim_length=output_settings.checkpoint_interval,
                    timestep=integrator_settings.timestep,
                    mc_steps=1,
                )

                self._add_checkpoint_reporter(
                    simulation=simulation,
                    checkpoint_path=checkpoint_path,
                    checkpoint_interval=checkpoint_interval,
                )

            nvt_path = (
                output_dir / output_settings.equil_nvt_structure
                if output_settings.equil_nvt_structure else None
            )
            if nvt_remaining > 0:
                if verbose:
                    logger.info(
                        f"Running NVT equilibration for {nvt_remaining} steps")
                self._run_dynamics(
                    simulation=simulation,
                    steps=nvt_remaining,
                    temperature=thermo_settings.temperature,
                    barostat_frequency=0 * unit.timestep,
                    output_settings=output_settings,
                    verbose=verbose,
                    output_path=nvt_path,
                    # Retain checkpoint velocities when resuming inside NVT.
                    reinitialize_velocities=not restart,
                )

            npt_path = (
                output_dir / output_settings.equil_npt_structure
                if output_settings.equil_npt_structure else None
            )
            if npt_remaining > 0:
                if verbose:
                    logger.info(
                        f"Running NPT equilibration for {npt_remaining} steps")
                self._run_dynamics(
                    simulation=simulation,
                    steps=npt_remaining,
                    temperature=thermo_settings.temperature,
                    barostat_frequency=integrator_settings.barostat_frequency,
                    output_settings=output_settings,
                    verbose=verbose,
                    output_path=npt_path,
                    # Within NPT, preserve saved velocities. At the NVT/NPT
                    # boundary, initialize a new NPT velocity distribution.
                    reinitialize_velocities=(
                            not restart or current_step_count <= equil_steps_nvt
                    ),
                )

            state_path = output_dir / "equilibrated_state.xml"
            # This State is suitable for PlainMD's compatible production System.
            # Future restrained/alchemical Systems may need to strip or map
            # Context parameters before loading the State into another System.
            simulation.saveState(str(state_path))

            minimized_path = (
                output_dir / output_settings.minimized_structure
                if output_settings.minimized_structure else None
            )
            return {
                "equilibrated_state": state_path,
                "equilibration_end_step": simulation.context.getStepCount(),
                "reinitialize_velocities_for_production": equil_steps_npt == 0,
                "minimized_pdb": (
                    minimized_path if minimized_path is not None and minimized_path.is_file()
                    else None
                ),
                "nvt_equil_pdb": nvt_path if nvt_path is not None and nvt_path.is_file() else None,
                "npt_equil_pdb": npt_path if npt_path is not None and npt_path.is_file() else None,
            }
        finally:
            if not dry:
                del integrator, simulation


class PlainMDSimulationUnit(PlainMDUnitMixin, gufe.ProtocolUnit):
    """
    Run production MD from the State written by PlainMDEquilibrationUnit.

    Setup supplies the original serialized System and full topology. Equilibration
    supplies a portable XML State with coordinates, box, velocities, simulation
    time/step count, and Context/integrator parameters. Both stages must use
    compatible OpenMM Systems for ``Simulation.loadState`` to succeed.

    Checkpoint files in this unit's shared directory cover *production only*.
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
    def _get_remaining_production_steps(
            *, current_step_count: int, equilibration_end_step: int,
            prod_steps: int
    ) -> int:
        """Steps left after loading either the equilibration State or a checkpoint.

        OpenMM's State retains the absolute step count, so production starts at
        ``equilibration_end_step`` rather than zero.
        """
        completed = current_step_count - equilibration_end_step
        if completed < 0 or completed > prod_steps:
            raise ValueError(
                "Loaded State step count is outside the production interval: "
                f"step={current_step_count}, "
                f"equilibration_end_step={equilibration_end_step}, "
                f"prod_steps={prod_steps}"
            )
        return prod_steps - completed

    def run(
        self,
        *,
        system: openmm.System,
        topology: openmm.app.Topology,
        equilibrated_state: pathlib.Path,
        equilibration_end_step: int,
        prod_steps: int,
        reinitialize_velocities_for_production: bool,
        dry: bool = False,
        verbose: bool = True,
        scratch_basepath: pathlib.Path | None = None,
        shared_basepath: pathlib.Path | None = None,
    ) -> dict[str, Any]:
        """Run only production MD, continuing from the equilibrated OpenMM State.

        On restart, load the production checkpoint instead, preserving its
        velocities and appending existing trajectory and log files. The setup
        System must be loaded again (rather than reusing equilibration's System).

        Parameters
        ----------
        system : openmm.System
          The System to simulate.
        topology: openmm.app.Topology
            The topology of the System.
        equilibrated_state : pathlib.Path
            Path to the XML State produced by PlainMDEquilibrationUnit.
        equilibration_end_step : int
            OpenMM step count at the end of equilibration.
        prod_steps : int
            Total requested production steps.
        reinitialize_velocities_for_production : bool
            Whether to initialize velocities when starting fresh production.
            Ignored when resuming from a production checkpoint.
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
        thermo_settings: settings.ThermoSettings = protocol_settings.thermo_settings
        output_settings: MDOutputSettings = protocol_settings.output_settings
        timestep = protocol_settings.integrator_settings.timestep
        integrator_settings = protocol_settings.integrator_settings

        simulation, integrator = self._create_simulation(
            system=system,
            topology=topology,
            protocol_settings=protocol_settings,
        )

        try:
            if dry:
                return {"debug": {"system": system}}
            # check for a restart
            restart = self._check_restart(output_settings, self.shared_basepath)
            if restart:
                if verbose:
                    logger.info("Restarting simulation from checkpoint state")
                state_path = self.shared_basepath / output_settings.checkpoint_storage_filename
            else:
                state_path = pathlib.Path(equilibrated_state)

            # PlainMD equilibration and production use compatible Systems.
            simulation.loadState(str(state_path))

            current_step_count = simulation.context.getStepCount()
            remaining_steps = self._get_remaining_production_steps(
                current_step_count=current_step_count,
                equilibration_end_step=equilibration_end_step,
                prod_steps=prod_steps,
            )
            production_started = restart and current_step_count > equilibration_end_step

            checkpoint_interval = settings_validation.get_simsteps(
                sim_length=output_settings.checkpoint_interval,
                timestep=timestep,
                mc_steps=1,
            )
            checkpoint_name = output_settings.checkpoint_storage_filename
            checkpoint_path = (
                self.shared_basepath / checkpoint_name
                if checkpoint_name else None
            )

            self._add_checkpoint_reporter(
                simulation=simulation,
                checkpoint_path=checkpoint_path,
                checkpoint_interval=checkpoint_interval,
            )

            if output_settings.production_trajectory_filename:
                selection_indices = mdtraj.Topology.from_openmm(
                    simulation.topology).select(
                    output_settings.output_indices
                )
                trajectory_path = self.shared_basepath / output_settings.production_trajectory_filename
                simulation.reporters.append(
                    XTCReporter(
                        file=str(trajectory_path),
                        reportInterval=settings_validation.divmod_time_and_check(
                            output_settings.trajectory_write_interval,
                            timestep,
                            "trajectory_write_interval",
                            "timestep",
                        ),
                        atomSubset=selection_indices,
                        append=production_started and trajectory_path.is_file(),
                    )
                )

            if output_settings.log_output:
                log_path = self.shared_basepath / output_settings.log_output
                simulation.reporters.append(
                    openmm.app.StateDataReporter(
                        str(log_path),
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
                        append=production_started and log_path.is_file(),
                    )
                )

            if verbose:
                logger.info(
                    f"Running production phase for {remaining_steps} steps")
            if remaining_steps:
                self._run_dynamics(
                    simulation=simulation,
                    steps=remaining_steps,
                    temperature=thermo_settings.temperature,
                    barostat_frequency=integrator_settings.barostat_frequency,
                    output_settings=output_settings,
                    verbose=verbose,
                    output_path=None,
                    # Preserve the checkpoint's velocities on restart. A fresh
                    # handoff inherits NPT velocities, but if NPT was skipped
                    # PlainMD traditionally initializes production velocities.
                    reinitialize_velocities=(
                            not restart and reinitialize_velocities_for_production
                    ),
                )
        finally:
            if not dry:
                del integrator, simulation

        checkpoint_name = output_settings.checkpoint_storage_filename
        checkpoint_path = (
            self.shared_basepath / checkpoint_name if checkpoint_name else None
        )
        return {
            "nc": (
                self.shared_basepath / output_settings.production_trajectory_filename
                if output_settings.production_trajectory_filename else None
            ),
            "last_checkpoint": (
                checkpoint_path if checkpoint_path is not None and checkpoint_path.is_file()
                else None
            ),
        }

    def _execute(
        self,
        ctx: gufe.Context,
        setup_results: gufe.ProtocolUnitResult,
        equilibration_results: gufe.ProtocolUnitResult,
        **kwargs,
    ) -> dict[str, Any]:
        log_system_probe(logging.INFO, paths=[ctx.scratch])
        setup = setup_results.outputs
        # Ensure that the environment hasn't changed
        self._verify_execution_environment(setup)

        equilibrated_states = equilibration_results.outputs["equilibrated_states"]
        if len(equilibrated_states) != 1 or equilibrated_states[0]["index"] != 0:
            raise ValueError(
                "PlainMD production expects one equilibrated state (index 0)")
        equilibrated = equilibrated_states[0]

        # Deserialize a fresh, unmodified setup System: the equilibration unit
        # owns and releases a separate OpenMM Context.
        system = serialization.deserialize(setup["system"])
        topology = openmm.app.PDBFile(str(setup["system_pdb"])).getTopology()
        outputs = self.run(
            system=system,
            topology=topology,
            equilibrated_state=equilibrated["equilibrated_state"],
            equilibration_end_step=equilibrated["equilibration_end_step"],
            prod_steps=setup["prod_steps"],
            reinitialize_velocities_for_production=equilibrated[
                "reinitialize_velocities_for_production"
            ],
            scratch_basepath=ctx.scratch,
            shared_basepath=ctx.shared,
        )
        return {
            "repeat_id": self._inputs["repeat_id"],
            "generation": self._inputs["generation"],
            "system_pdb": setup["system_pdb"],
            # Preserve the former simulation-unit output keys for existing
            # consumers, even though equilibration now writes these files.
            "minimized_pdb": equilibrated.get("minimized_pdb"),
            "nvt_equil_pdb": equilibrated.get("nvt_equil_pdb"),
            "npt_equil_pdb": equilibrated.get("npt_equil_pdb"),
            **outputs,
        }
