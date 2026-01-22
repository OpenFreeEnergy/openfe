# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""OpenMM AFE Protocol base classes
===================================

Base classes for the OpenMM absolute free energy ProtocolUnits.

Thist mostly implements BaseAbsoluteUnit whose methods can be
overriden to define different types of alchemical transformations.

TODO
----
* Add in all the AlchemicalFactory and AlchemicalRegion kwargs
  as settings.
* Allow for a more flexible setting of Lambda regions.
"""

import abc
import copy
import logging
import os
import pathlib
from typing import Any

import gufe
import mdtraj as mdt
import numpy as np
import numpy.typing as npt
import openmm
import openmmtools
from gufe import (
    ProteinComponent,
    SmallMoleculeComponent,
    SolventComponent,
)
from gufe.components import Component
from openff.toolkit.topology import Molecule as OFFMolecule
from openff.units import Quantity
from openff.units import unit as offunit
from openff.units.openmm import ensure_quantity, from_openmm, to_openmm
from openmm import app
from openmm import unit as ommunit
from openmmforcefields.generators import SystemGenerator
from openmmtools import multistate
from openmmtools.alchemy import (
    AbsoluteAlchemicalFactory,
    AlchemicalRegion,
    AlchemicalState,
)
from openmmtools.states import (
    GlobalParameterState,
    SamplerState,
    ThermodynamicState,
    create_thermodynamic_state_protocol,
)

from openfe.protocols.openmm_afe.equil_afe_settings import (
    AlchemicalSettings,
    BaseSolvationSettings,
    IntegratorSettings,
    MultiStateOutputSettings,
    MultiStateSimulationSettings,
    OpenFFPartialChargeSettings,
    OpenMMEngineSettings,
    OpenMMSystemGeneratorFFSettings,
    ThermoSettings,
)
from openfe.protocols.openmm_md.plain_md_methods import PlainMDProtocolUnit
from openfe.protocols.openmm_utils import (
    charge_generation,
    multistate_analysis,
    omm_compute,
    settings_validation,
    system_creation,
)
from openfe.protocols.openmm_utils.omm_settings import (
    SettingsBaseModel,
)
from openfe.protocols.openmm_utils.serialization import (
    deserialize,
    make_vec3_box,
    serialize,
)
from openfe.protocols.restraint_utils import geometry
from openfe.protocols.restraint_utils.openmm import omm_restraints
from openfe.utils import log_system_probe, without_oechem_backend

logger = logging.getLogger(__name__)


class AbsoluteUnitMixin:
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

        if self.verbose:
            self.logger.info("setting up alchemical system")  # type: ignore[attr-defined]

        # set basepaths
        def _set_optional_path(basepath):
            if basepath is None:
                return pathlib.Path(".")
            return basepath

        self.scratch_basepath = _set_optional_path(scratch_basepath)
        self.shared_basepath = _set_optional_path(shared_basepath)

    @abc.abstractmethod
    def _get_settings(self) -> dict[str, SettingsBaseModel]:
        """
        Get a dictionary with the following entries:
          * forcefield_settings : OpenMMSystemGeneratorFFSettings
          * thermo_settings : ThermoSettings
          * solvation_settings : BaseSolvationSettings
          * alchemical_settings : AlchemicalSettings
          * lambda_settings : LambdaSettings
          * engine_settings : OpenMMEngineSettings
          * integrator_settings : IntegratorSettings
          * equil_simulation_settings : MDSimulationSettings
          * equil_output_settings : MDOutputSettings
          * simulation_settings : MultiStateSimulationSettings
          * output_settings : MultiStateOutputSettings

        Settings may change depending on what type of simulation you are
        running. Cherry pick them and return them to be available later on.

        This method should also add various validation checks as necessary.

        Note
        ----
        Must be implemented in the child class.
        """
        ...


class BaseAbsoluteSetupUnit(gufe.ProtocolUnit, AbsoluteUnitMixin):
    """
    Base class for setting up an absolute free energy transformations.
    """

    @abc.abstractmethod
    def _get_components(
        self,
    ) -> tuple[
        dict[str, list[Component]],
        gufe.SolventComponent | None,
        gufe.ProteinComponent | None,
        dict[SmallMoleculeComponent, OFFMolecule],
    ]:
        """
        Get the relevant components to create the alchemical system with.

        Note
        ----
        Must be implemented in the child class.
        """
        ...

    @staticmethod
    def _get_alchemical_indices(
        omm_top: openmm.app.Topology,
        comp_resids: dict[Component, npt.NDArray],
        alchem_comps: dict[str, list[Component]],
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
        residxs = np.concatenate([comp_resids[key] for key in alchem_comps["stateA"]])

        # get the alchemicical atom ids
        atom_ids = []

        for r in omm_top.residues():
            if r.index in residxs:
                atom_ids.extend([at.index for at in r.atoms()])

        return atom_ids

    def _pre_equilibrate(
        self,
        system: openmm.System,
        topology: openmm.app.Topology,
        positions: ommunit.Quantity,
        settings: dict[str, SettingsBaseModel],
        dry: bool,
    ) -> tuple[ommunit.Quantity, ommunit.Quantity]:
        """
        Run a non-alchemical equilibration to get a stable system.

        Parameters
        ----------
        system : openmm.System
          The OpenMM System to equilibrate.
        topology : openmm.app.Topology
          OpenMM Topology of the System.
        positions : openmm.unit.Quantity
          Initial positions for the system.
        settings : dict[str, SettingsBaseModel]
          A dictionary of settings objects. Expects the
          following entries:
          * `forcefield_settings`
          * `engine_settings`
          * `thermo_settings`
          * `integrator_settings`
          * `equil_simulation_settings`
          * `equil_output_settings`
        dry: bool
          Whether or not this is a dry run.

        Returns
        -------
        equilibrated_positions : npt.NDArray
          Equilibrated system positions
        box : openmm.unit.Quantity
          Box vectors of the equilibrated system.
        """
        # Prep the simulation object
        # Restrict CPU count if running vacuum simulation
        restrict_cpu = settings["forcefield_settings"].nonbonded_method.lower() == "nocutoff"
        platform = omm_compute.get_openmm_platform(
            platform_name=settings["engine_settings"].compute_platform,
            gpu_device_index=settings["engine_settings"].gpu_device_index,
            restrict_cpu_count=restrict_cpu,
        )

        integrator = openmm.LangevinMiddleIntegrator(
            to_openmm(settings["thermo_settings"].temperature),
            to_openmm(settings["integrator_settings"].langevin_collision_rate),
            to_openmm(settings["integrator_settings"].timestep),
        )

        simulation = openmm.app.Simulation(
            topology=topology,
            system=system,
            integrator=integrator,
            platform=platform,
        )

        # Get the necessary number of steps
        if settings["equil_simulation_settings"].equilibration_length_nvt is not None:
            equil_steps_nvt = settings_validation.get_simsteps(
                sim_length=settings["equil_simulation_settings"].equilibration_length_nvt,
                timestep=settings["integrator_settings"].timestep,
                mc_steps=1,
            )
        else:
            equil_steps_nvt = None

        equil_steps_npt = settings_validation.get_simsteps(
            sim_length=settings["equil_simulation_settings"].equilibration_length,
            timestep=settings["integrator_settings"].timestep,
            mc_steps=1,
        )

        prod_steps_npt = settings_validation.get_simsteps(
            sim_length=settings["equil_simulation_settings"].production_length,
            timestep=settings["integrator_settings"].timestep,
            mc_steps=1,
        )

        if self.verbose:
            self.logger.info("running non-alchemical equilibration MD")

        # Don't do anything if we're doing a dry run
        if dry:
            return positions, system.getDefaultPeriodicBoxVectors()

        # Use the _run_MD method from the PlainMDProtocolUnit
        # Should in-place modify the simulation
        PlainMDProtocolUnit._run_MD(
            simulation=simulation,
            positions=positions,
            simulation_settings=settings["equil_simulation_settings"],
            output_settings=settings["equil_output_settings"],
            temperature=settings["thermo_settings"].temperature,
            barostat_frequency=settings["integrator_settings"].barostat_frequency,
            timestep=settings["integrator_settings"].timestep,
            equil_steps_nvt=equil_steps_nvt,
            equil_steps_npt=equil_steps_npt,
            prod_steps=prod_steps_npt,
            verbose=self.verbose,
            shared_basepath=self.shared_basepath,
        )

        # TODO: if we still see crashes, see if using enforcePeriodicBox is necessary
        # on newer tests, these were not necessary.
        state = simulation.context.getState(getPositions=True)
        equilibrated_positions = state.getPositions(asNumpy=True)
        box = state.getPeriodicBoxVectors()

        # cautiously delete out contexts & integrator
        del simulation.context, integrator

        return equilibrated_positions, box

    @staticmethod
    def _assign_partial_charges(
        partial_charge_settings: OpenFFPartialChargeSettings,
        small_mols: dict[SmallMoleculeComponent, OFFMolecule],
    ) -> None:
        """
        Assign partial charges to the OpenFF Molecules associated with
        all the SmallMoleculeComponents in the transformation.

        Parameters
        ----------
        charge_settings : OpenFFPartialChargeSettings
          Settings for controlling how the partial charges are assigned.
        small_mols : dict[SmallMoleculeComponent, openff.toolkit.Molecule]
          Dictionary of OpenFF Molecules to add, keyed by their
          associated SmallMoleculeComponent.
        """
        for mol in small_mols.values():
            charge_generation.assign_offmol_partial_charges(
                offmol=mol,
                overwrite=False,
                method=partial_charge_settings.partial_charge_method,
                toolkit_backend=partial_charge_settings.off_toolkit_backend,
                generate_n_conformers=partial_charge_settings.number_of_conformers,
                nagl_model=partial_charge_settings.nagl_model,
            )

    @staticmethod
    def _get_system_generator(
        settings: dict[str, SettingsBaseModel],
        solvent_component: SolventComponent | None,
        openff_molecules: list[OFFMolecule],
        ffcache: pathlib.Path | None,
    ) -> SystemGenerator:
        """
        Get a system generator through the system creation
        utilities

        Parameters
        ----------
        settings : dict[str, SettingsBaseModel]
          A dictionary of settings object for the unit.
        solvent_comp : SolventComponent | None
          The solvent component of this system, if there is one.
        openff_molecules : list[openff.toolkit.Molecule] | None
          A list of OpenFF Molecules to generate templates for, if any.
        ffcache : pathlib.Path | None
          Path to the force field parameter cache.

        Returns
        -------
        system_generator : openmmforcefields.generator.SystemGenerator
          System Generator to parameterise this unit.
        """
        system_generator = system_creation.get_system_generator(
            forcefield_settings=settings["forcefield_settings"],
            integrator_settings=settings["integrator_settings"],
            thermo_settings=settings["thermo_settings"],
            cache=ffcache,
            has_solvent=solvent_component is not None,
        )

        # Handle openff Molecule templates
        # TODO: revisit this once the SystemGenerator update happens
        if openff_molecules is None:
            return system_generator

        # Register all the templates, pass unique molecules to avoid clashes
        system_generator.add_molecules(list(set(openff_molecules)))

        return system_generator

    @staticmethod
    def _get_modeller(
        protein_component: ProteinComponent | None,
        solvent_component: SolventComponent | None,
        small_mols: dict[SmallMoleculeComponent, OFFMolecule],
        system_generator: SystemGenerator,
        solvation_settings: BaseSolvationSettings,
    ) -> tuple[app.Modeller, dict[Component, npt.NDArray]]:
        """
        Get an OpenMM Modeller object and a list of residue indices
        for each component in the system.

        Parameters
        ----------
        protein_component : ProteinComponent | None
          Protein Component, if it exists.
        solvent_component : SolventComponent | None
          Solvent Component, if it exists.
        small_mols : dict[SmallMoleculeComponent, openff.toolkit.Molecule]
          Dictionary of OpenFF Molecules to add, keyed by
          SmallMoleculeComponent.
        system_generator : openmmforcefields.generator.SystemGenerator
          System Generator to parameterise this unit.
        solvation_settings : BaseSolvationSettings
          Settings detailing how to solvate the system.

        Returns
        -------
        system_modeller : app.Modeller
          OpenMM Modeller object generated from ProteinComponent and
          OpenFF Molecules.
        comp_resids : dict[Component, npt.NDArray]
          Dictionary of residue indices for each component in system.
        """
        # get OpenMM modeller + dictionary of resids for each component
        system_modeller, comp_resids = system_creation.get_omm_modeller(
            protein_comp=protein_component,
            solvent_comp=solvent_component,
            small_mols=small_mols,
            omm_forcefield=system_generator.forcefield,
            solvent_settings=solvation_settings,
        )

        return system_modeller, comp_resids

    def _get_omm_objects(
        self,
        settings: dict[str, SettingsBaseModel],
        protein_component: ProteinComponent | None,
        solvent_component: SolventComponent | None,
        small_mols: dict[SmallMoleculeComponent, OFFMolecule],
    ) -> tuple[
        app.Topology,
        openmm.System,
        openmm.unit.Quantity,
        dict[Component, npt.NDArray],
    ]:
        """
        Get the OpenMM Topology, Positions and System of the
        parameterised system.

        Parameters
        ----------
        settings : dict[str, SettingsBaseModel]
          Protocol settings
        protein_component : ProteinComponent | None
          Protein component for the system.
        solvent_component : SolventComponent | None
          Solvent component for the system.
        small_mols : dict[str, openff.toolkit.Molecule]
          Dictionary of SmallMoleculeComponents and OpenFF Molecules
          defining the ligands to be added to the system

        Returns
        -------
        topology : app.Topology
          OpenMM Topology object describing the parameterized system.
        system : openmm.System
          A non-alchemical OpenMM System of the simulated system.
        positions : openmm.unit.Quantity
          Positions of the system.
        comp_resids : dict[Component, npt.NDArray]
          A dictionary of the residues for each component in the System.
        """
        if self.verbose:
            self.logger.info("Parameterizing system")

        with without_oechem_backend():
            system_generator = self._get_system_generator(
                settings=settings,
                solvent_component=solvent_component,
                openff_molecules=list(small_mols.values()),
                ffcache=self.shared_basepath / settings["output_settings"].forcefield_cache,
            )

            modeller, comp_resids = self._get_modeller(
                protein_component=protein_component,
                solvent_component=solvent_component,
                small_mols=small_mols,
                system_generator=system_generator,
                solvation_settings=settings["solvation_settings"],
            )

            system = system_generator.create_system(
                topology=modeller.topology,
                molecules=list(small_mols.values()),
            )

        topology = modeller.getTopology()
        # roundtrip positions to remove vec3 issues
        positions = to_openmm(from_openmm(modeller.getPositions()))

        return topology, system, positions, comp_resids

    def _add_restraints(
        self,
        system: openmm.System,
        topology: GlobalParameterState,
        positions: openmm.unit.Quantity,
        alchem_comps: dict[str, list[Component]],
        comp_resids: dict[Component, npt.NDArray],
        settings: dict[str, SettingsBaseModel],
    ) -> tuple[
        Quantity | None,
        openmm.System | None,
        geometry.BaseRestraintGeometry | None,
    ]:
        """
        Placeholder method to add restraints if necessary
        """
        return None, system, None

    def _get_alchemical_system(
        self,
        topology: app.Topology,
        system: openmm.System,
        comp_resids: dict[Component, npt.NDArray],
        alchem_comps: dict[str, list[Component]],
        alchemical_settings: AlchemicalSettings,
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
        alchemical_settings : AlchemicalSettings
          Settings controlling how the alchemical system is built.

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
        alchemical_indices = self._get_alchemical_indices(topology, comp_resids, alchem_comps)

        alchemical_region = AlchemicalRegion(
            alchemical_atoms=alchemical_indices,
            softcore_alpha=alchemical_settings.softcore_alpha,
            annihilate_electrostatics=True,
            annihilate_sterics=alchemical_settings.annihilate_sterics,
            softcore_a=alchemical_settings.softcore_a,
            softcore_b=alchemical_settings.softcore_b,
            softcore_c=alchemical_settings.softcore_c,
            softcore_beta=0.0,
            softcore_d=1.0,
            softcore_e=1.0,
            softcore_f=2.0,
        )

        alchemical_factory = AbsoluteAlchemicalFactory(
            consistent_exceptions=False,
            switch_width=1.0 * ommunit.angstroms,
            alchemical_pme_treatment="exact",
            alchemical_rf_treatment="switched",
            disable_alchemical_dispersion_correction=alchemical_settings.disable_alchemical_dispersion_correction,
            split_alchemical_forces=True,
        )
        alchemical_system = alchemical_factory.create_alchemical_system(system, alchemical_region)

        return alchemical_factory, alchemical_system, alchemical_indices

    @staticmethod
    def _subsample_topology(
        topology: openmm.app.Topology,
        positions: openmm.unit.Quantity,
        output_selection: str,
        output_file: pathlib.Path,
    ) -> npt.NDArray:
        """
        Subsample the system based on user-selected output selection
        and write the subsampled topology to a PDB file.

        Parameters
        ----------
        topology : openmm.app.Topology
          The system topology to subsample.
        positions : openmm.unit.Quantity
          The system positions.
        output_selection : str
          An MDTraj selection string to subsample the topology with.
        output_file : pathlib.Path
          Path to the file to write the PDB to.

        Returns
        -------
        selection_indices : npt.NDArray
          The indices of the subselected system.
        """
        mdt_top = mdt.Topology.from_openmm(topology)
        selection_indices = mdt_top.select(output_selection)

        # Write out the subselected structure to PDB if not empty
        if len(selection_indices) > 0:
            traj = mdt.Trajectory(
                positions[selection_indices, :],
                mdt_top.subset(selection_indices),
            )
            traj.save_pdb(output_file)

        return selection_indices

    def run(
        self,
        dry: bool = False,
        verbose: bool = True,
        scratch_basepath: pathlib.Path | None = None,
        shared_basepath: pathlib.Path | None = None,
    ) -> dict[str, Any]:
        """Run the setup phase of an absolute free energy calculation.

        Parameters
        ----------
        dry : bool
          Do a dry run of the calculation, creating all necessary alchemical
          system components (topology, system, etc...) but without
          running the simulation, default False
        verbose : bool
          Verbose output of the simulation progress. Output is provided via
          INFO level logging, default True
        scratch_basepath : pathlib.Path | None
          Path to the scratch (temporary) directory space. Defaults to the
          current working directory if ``None``.
        shared_basepath : pathlib.Path | None
          Path to the shared (persistent) directory space. Defaults to the
          current working directory if ``None``.

        Returns
        -------
        dict
          Outputs created in the basepath directory or the debug objects
          (i.e. sampler) if ``dry==True``.
        """
        # General preparation tasks
        self._prepare(verbose, scratch_basepath, shared_basepath)

        # Get components
        alchem_comps, solv_comp, prot_comp, small_mols = self._get_components()

        # Get settings
        settings = self._get_settings()

        # Assign partial charges now to avoid any discrepancies later
        self._assign_partial_charges(settings["charge_settings"], small_mols)

        # Get OpenMM topology, positions, system, and comp_resids
        omm_topology, omm_system, positions, comp_resids = self._get_omm_objects(
            settings=settings,
            protein_component=prot_comp,
            solvent_component=solv_comp,
            small_mols=small_mols,
        )

        # Pre-equilbrate System (Test + Avoid NaNs + get stable system)
        positions, box_vectors = self._pre_equilibrate(
            omm_system, omm_topology, positions, settings, dry
        )

        # Add restraints
        # Note: when no restraint is applied, restrained_omm_system == omm_system
        (
            standard_state_corr,
            restrained_omm_system,
            restraint_geometry,
        ) = self._add_restraints(
            omm_system,
            omm_topology,
            positions,
            alchem_comps,
            comp_resids,
            settings,
        )

        # Get alchemical system
        alchem_factory, alchem_system, alchem_indices = self._get_alchemical_system(
            topology=omm_topology,
            system=restrained_omm_system,
            comp_resids=comp_resids,
            alchem_comps=alchem_comps,
            alchemical_settings=settings["alchemical_settings"],
        )

        # Subselect system based on user inputs & write initial PDB
        selection_indices = self._subsample_topology(
            topology=omm_topology,
            positions=positions,
            output_selection=settings["output_settings"].output_indices,
            output_file=self.shared_basepath / settings["output_settings"].output_structure,
        )

        # Serialize relevant outputs
        system_outfile = self.shared_basepath / "alchemical_system.xml.bz2"
        serialize(alchem_system, system_outfile)

        positions_outfile = self.shared_basepath / "system_positions.npy"
        npy_positions = from_openmm(positions).to("nanometer").m
        np.save(positions_outfile, npy_positions)

        # Set the PDB file name
        if len(selection_indices) > 0:
            pdb_structure = self.shared_basepath / settings["output_settings"].output_structure
        else:
            pdb_structure = None

        unit_results_dict = {
            "system": system_outfile,
            "positions": positions_outfile,
            "pdb_structure": pdb_structure,
            "selection_indices": selection_indices,
            "box_vectors": from_openmm(box_vectors),
        }

        if standard_state_corr is not None:
            unit_results_dict["standard_state_correction"] = standard_state_corr.to(
                "kilocalorie_per_mole"
            )
        else:
            unit_results_dict["standard_state_correction"] = 0 * offunit.kilocalorie_per_mole

        if restraint_geometry is not None:
            unit_results_dict["restraint_geometry"] = restraint_geometry.model_dump()
        else:
            unit_results_dict["restraint_geometry"] = None

        if dry:
            unit_results_dict |= {
                "standard_system": omm_system,
                "restrained_system": restrained_omm_system,
                "alchem_system": alchem_system,
                "alchem_indices": alchem_indices,
                "alchem_factory": alchem_factory,
                "debug_positions": positions,
            }
        return unit_results_dict

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
            "simtype": self.simtype,
            **outputs,
        }


class BaseAbsoluteMultiStateSimulationUnit(gufe.ProtocolUnit, AbsoluteUnitMixin):
    @staticmethod
    def _check_restart(output_settings: SettingsBaseModel, shared_path: pathlib.Path):
        """
        Check if we are doing a restart.

        Parameters
        ----------
        output_settings : SettingsBaseModel
          The simulation output settings
        shared_path : pathlib.Path
          The shared directory where we should be looking for existing files.

        Notes
        -----
        For now this just checks if the netcdf files are present in the
        shared directory but in the future this may expand depending on
        how warehouse works.
        """
        trajectory = shared_path / output_settings.output_filename
        checkpoint = shared_path / output_settings.checkpoint_storage_filename

        if trajectory.is_file() and checkpoint.is_file():
            return True

        return False

    @abc.abstractmethod
    def _get_components(
        self,
    ) -> tuple[
        dict[str, list[Component]],
        gufe.SolventComponent | None,
        gufe.ProteinComponent | None,
        dict[SmallMoleculeComponent, OFFMolecule],
    ]:
        """
        Get the relevant components to create the alchemical system with.

        Note
        ----
        Must be implemented in the child class.
        """
        ...

    def _get_lambda_schedule(
        self, settings: dict[str, SettingsBaseModel]
    ) -> dict[str, list[float]]:
        """
        Create the lambda schedule

        Parameters
        ----------
        settings : dict[str, SettingsBaseModel]
          Settings for the unit.

        Returns
        -------
        lambdas : dict[str, list[float]]

        TODO
        ----
        * Augment this by using something akin to the RFE protocol's
          LambdaProtocol
        """
        lambdas = dict()

        lambda_elec = settings["lambda_settings"].lambda_elec
        lambda_vdw = settings["lambda_settings"].lambda_vdw
        lambda_rest = settings["lambda_settings"].lambda_restraints

        # Reverse lambda schedule for vdw, end elec,
        # since in AbsoluteAlchemicalFactory 1 means fully
        # interacting (which would be non-interacting for us)
        lambdas["lambda_electrostatics"] = [1 - x for x in lambda_elec]
        lambdas["lambda_sterics"] = [1 - x for x in lambda_vdw]
        lambdas["lambda_restraints"] = [x for x in lambda_rest]

        return lambdas

    def _get_states(
        self,
        alchemical_system: openmm.System,
        positions: openmm.unit.Quantity,
        box_vectors: openmm.unit.Quantity,
        thermodynamic_settings: ThermoSettings,
        lambdas: dict[str, list[float]],
        solvent_component: SolventComponent | None,
        alchemically_restrained: bool,
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
        box_vectors :  openmm.unit.Quantity
          Box vectors of the alchemical system.
        thermodynamic_settings : ThermoSettings
          Settings controlling the thermodynamic parameters.
        lambdas : dict[str, list[float]]
          A dictionary of lambda scales.
        solvent_component : SolventComponent | None
          The solvent component of the system, if there is one.
        alchemically_restrained : bool
          Whether or not the system requires a control parameter
          for any alchemical restraints.

        Returns
        -------
        sampler_states : list[SamplerState]
          A list of SamplerStates for each replica in the system.
        cmp_states : list[ThermodynamicState]
          A list of ThermodynamicState for each replica in the system.
        """
        # Fetch an alchemical state
        alchemical_state = AlchemicalState.from_system(alchemical_system)

        # Set up the system constants
        temperature = thermodynamic_settings.temperature
        pressure = thermodynamic_settings.pressure
        constants = dict()
        constants["temperature"] = ensure_quantity(temperature, "openmm")

        if solvent_component is not None:
            constants["pressure"] = ensure_quantity(pressure, "openmm")

        # Get the thermodynamic parameter protocol
        param_protocol = copy.deepcopy(lambdas)

        # Get the composable states
        if alchemically_restrained:
            restraint_state = omm_restraints.RestraintParameterState(lambda_restraints=1.0)
            composable_states = [alchemical_state, restraint_state]
        else:
            composable_states = [alchemical_state]

            # In this case we also don't have a restraint being controlled
            # so we drop it from the protocol
            param_protocol.pop("lambda_restraints", None)

        cmp_states = create_thermodynamic_state_protocol(
            alchemical_system,
            protocol=param_protocol,
            constants=constants,
            composable_states=composable_states,
        )

        sampler_state = SamplerState(positions=positions)
        if alchemical_system.usesPeriodicBoundaryConditions():
            sampler_state.box_vectors = box_vectors

        sampler_states = [sampler_state for _ in cmp_states]

        return sampler_states, cmp_states

    @staticmethod
    def _get_integrator(
        integrator_settings: IntegratorSettings,
        simulation_settings: MultiStateSimulationSettings,
        system: openmm.System,
    ) -> openmmtools.mcmc.LangevinDynamicsMove:
        """
        Return a LangevinDynamicsMove integrator

        Parameters
        ----------
        integrator_settings : IntegratorSettings
          Settings controlling the Langevin integrator
        simulation_settings : MultiStateSimulationSettings
          Settings controlling the simulation.
        system : openmm.System
          The OpenMM System.

        Returns
        -------
        integrator : openmmtools.mcmc.LangevinDynamicsMove
          A configured integrator object.

        Raises
        ------
        ValueError
          If there are virtual sites in the system, but
          velocities are not being reassigned after every MCMC move.
        """
        steps_per_iteration = settings_validation.convert_steps_per_iteration(
            simulation_settings, integrator_settings
        )

        integrator = openmmtools.mcmc.LangevinDynamicsMove(
            timestep=to_openmm(integrator_settings.timestep),
            collision_rate=to_openmm(integrator_settings.langevin_collision_rate),
            n_steps=steps_per_iteration,
            reassign_velocities=integrator_settings.reassign_velocities,
            n_restart_attempts=integrator_settings.n_restart_attempts,
            constraint_tolerance=integrator_settings.constraint_tolerance,
        )

        # Validate for known issue when dealing with virtual sites
        # and mutltistate simulations
        if not integrator_settings.reassign_velocities:
            for particle_idx in range(system.getNumParticles()):
                if system.isVirtualSite(particle_idx):
                    errmsg = (
                        "Simulations with virtual sites without velocity "
                        "reassignments are unstable with MCMC integrators. "
                        "You can set `reassign_velocities` to ``True`` in the "
                        "`integrator_settings` to avoid this issue."
                    )
                    raise ValueError(errmsg)

        return integrator

    @staticmethod
    def _get_reporter(
        storage_path: pathlib.Path,
        selection_indices: npt.NDArray,
        simulation_settings: MultiStateSimulationSettings,
        output_settings: MultiStateOutputSettings,
    ) -> multistate.MultiStateReporter:
        """
        Get a MultistateReporter for the simulation you are running.

        Parameters
        ----------
        storage_path : pathlib.Path
          Path to the directory where files should be written.
        selection_indices : npt.NDArray
          Array of system particle indices to subsample the system by.
        simulation_settings : MultiStateSimulationSettings
          Multistate simulation control settings, specifically containing
          the amount of time per state sampling iteration.
        output_settings: MultiStateOutputSettings
          Output settings for the simulations

        Returns
        -------
        reporter : multistate.MultiStateReporter
          The reporter for the simulation.
        """
        nc = storage_path / output_settings.output_filename
        # The checkpoint file in openmmtools is taken as a file relative
        # to the location of the nc file, so you only want the filename
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
                denominator=simulation_settings.time_per_iteration,
                numerator_name="output settings' velocity_write_frequency",
                denominator_name="simulation settings' time_per_iteration",
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
        integrator: openmmtools.mcmc.LangevinDynamicsMove,
        reporter: openmmtools.multistate.MultiStateReporter,
        simulation_settings: MultiStateSimulationSettings,
        thermodynamic_settings: ThermoSettings,
        compound_states: list[ThermodynamicState],
        sampler_states: list[SamplerState],
        platform: openmm.Platform,
    ) -> multistate.MultiStateSampler:
        """
        Get a sampler based on the equilibrium sampling method requested.

        Parameters
        ----------
        integrator : openmmtools.mcmc.LangevinDynamicsMove
          The simulation integrator.
        reporter : openmmtools.multistate.MultiStateReporter
          The reporter to hook up to the sampler.
        simulation_settings : MultiStateSimulationSettings
          Settings for the alchemical sampler.
        thermodynamic_settings : ThermoSettings
          Thermodynamic settings
        compound_states : list[ThermodynamicState]
          A list of thermodynamic states to sample.
        sampler_states : list[SamplerState]
          A list of sampler states.
        platform : openmm.Platform
          The compute platform to use.

        Returns
        -------
        sampler : multistate.MultistateSampler
          A sampler configured for the chosen sampling method.
        """
        rta_its, rta_min_its = settings_validation.convert_real_time_analysis_iterations(
            simulation_settings=simulation_settings,
        )
        et_target_err = settings_validation.convert_target_error_from_kcal_per_mole_to_kT(
            thermodynamic_settings.temperature,
            simulation_settings.early_termination_target_error,
        )

        # Select the right sampler
        # Note: doesn't need else, settings already validates choices
        if simulation_settings.sampler_method.lower() == "repex":
            sampler = multistate.ReplicaExchangeSampler(
                mcmc_moves=integrator,
                online_analysis_interval=rta_its,
                online_analysis_target_error=et_target_err,
                online_analysis_minimum_iterations=rta_min_its,
            )
        elif simulation_settings.sampler_method.lower() == "sams":
            sampler = multistate.SAMSSampler(
                mcmc_moves=integrator,
                online_analysis_interval=rta_its,
                online_analysis_minimum_iterations=rta_min_its,
                flatness_criteria=simulation_settings.sams_flatness_criteria,
                gamma0=simulation_settings.sams_gamma0,
            )
        elif simulation_settings.sampler_method.lower() == "independent":
            sampler = multistate.MultiStateSampler(
                mcmc_moves=integrator,
                online_analysis_interval=rta_its,
                online_analysis_target_error=et_target_err,
                online_analysis_minimum_iterations=rta_min_its,
            )

        sampler.create(
            thermodynamic_states=compound_states,
            sampler_states=sampler_states,
            storage=reporter,
        )

        sampler.energy_context_cache = openmmtools.cache.ContextCache(
            capacity=None,
            time_to_live=None,
            platform=platform,
        )

        sampler.sampler_context_cache = openmmtools.cache.ContextCache(
            capacity=None,
            time_to_live=None,
            platform=platform,
        )

        return sampler

    def _run_simulation(
        self,
        sampler: multistate.MultiStateSampler,
        reporter: multistate.MultiStateReporter,
        settings: dict[str, SettingsBaseModel],
        dry: bool,
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
        """
        # Get the relevant simulation steps
        mc_steps = settings_validation.convert_steps_per_iteration(
            simulation_settings=settings["simulation_settings"],
            integrator_settings=settings["integrator_settings"],
        )

        equil_steps = settings_validation.get_simsteps(
            sim_length=settings["simulation_settings"].equilibration_length,
            timestep=settings["integrator_settings"].timestep,
            mc_steps=mc_steps,
        )
        prod_steps = settings_validation.get_simsteps(
            sim_length=settings["simulation_settings"].production_length,
            timestep=settings["integrator_settings"].timestep,
            mc_steps=mc_steps,
        )

        if not dry:  # pragma: no-cover
            # minimize
            if self.verbose:
                self.logger.info("minimizing systems")

            sampler.minimize(max_iterations=settings["simulation_settings"].minimization_steps)

            # equilibrate
            if self.verbose:
                self.logger.info("equilibrating systems")

            sampler.equilibrate(int(equil_steps / mc_steps))

            # production
            if self.verbose:
                self.logger.info("running production phase")
            sampler.extend(int(prod_steps / mc_steps))

            if self.verbose:
                self.logger.info("production phase complete")

        else:
            # close reporter when you're done, prevent file handle clashes
            reporter.close()

            # clean up the reporter file
            fns = [
                self.shared_basepath / settings["output_settings"].output_filename,
                self.shared_basepath / settings["output_settings"].checkpoint_storage_filename,
            ]
            for fn in fns:
                os.remove(fn)

    def run(
        self,
        *,
        system: openmm.System,
        positions: openmm.unit.Quantity,
        box_vectors: Quantity,
        selection_indices: npt.NDArray,
        alchemical_restraints: bool,
        dry: bool = False,
        verbose: bool = True,
        scratch_basepath: pathlib.Path | None = None,
        shared_basepath: pathlib.Path | None = None,
    ) -> dict[str, Any]:
        """
        Run the free energy calculation using a multistate sampler.

        Parameters
        ----------
        system : openmm.System
          The System to simulate.
        positions : openmm.unit.Quantity
          The positions of the System.
        box_vectors : openff.units.Quantity
          The box vectors of the System.
        selection_indices : npt.NDArray
          Indices of the System particles to write to file.
        alchemical_restraints: bool,
          Whether or not the system has alchemical restraints.
        dry: bool
          Do a dry run of the calculation, creating all the necessary
          components, but without running the simulation.
        verbose : bool
          Verbose output of the simulation progress. Output is provided at
          the INFO logging level.
        scratch_basepath : pathlib.Path | None
          Where to store temporary files, defaults to the current working
          directory if ``None``.
        shared_basepath : pathlib.Path | None
          Where to store calculation outputs, defaults to the current working
          directory if ``None``.

        Returns
        -------
        dict
          Outputs created by the unit, including the debug objects
          (i.e. sampler) if ``dry==True``
        """
        # Prepare paths & verbosity
        self._prepare(verbose, scratch_basepath, shared_basepath)

        # Get the settings
        settings = self._get_settings()

        # Get the components
        alchem_comps, solv_comp, prot_comp, small_mols = self._get_components()

        # Get the lambda schedule
        lambdas = self._get_lambda_schedule(settings)

        # Get the compute platform
        restrict_cpu = settings["forcefield_settings"].nonbonded_method.lower() == "nocutoff"
        platform = omm_compute.get_openmm_platform(
            platform_name=settings["engine_settings"].compute_platform,
            gpu_device_index=settings["engine_settings"].gpu_device_index,
            restrict_cpu_count=restrict_cpu,
        )

        # Get compound and sampler states
        sampler_states, cmp_states = self._get_states(
            alchemical_system=system,
            positions=positions,
            # convert the box vectors to vec3 from openff
            box_vectors=make_vec3_box(box_vectors),
            thermodynamic_settings=settings["thermo_settings"],
            lambdas=lambdas,
            solvent_component=solv_comp,
            alchemically_restrained=alchemical_restraints,
        )

        # Get the integrator
        integrator = self._get_integrator(
            integrator_settings=settings["integrator_settings"],
            simulation_settings=settings["simulation_settings"],
            system=system,
        )

        try:
            # Create or get the multistate reporter
            reporter = self._get_reporter(
                storage_path=self.shared_basepath,
                selection_indices=selection_indices,
                simulation_settings=settings["simulation_settings"],
                output_settings=settings["output_settings"],
            )

            # Get sampler
            sampler = self._get_sampler(
                integrator=integrator,
                reporter=reporter,
                simulation_settings=settings["simulation_settings"],
                thermodynamic_settings=settings["thermo_settings"],
                compound_states=cmp_states,
                sampler_states=sampler_states,
                platform=platform,
            )

            # Run simulation
            self._run_simulation(
                sampler=sampler,
                reporter=reporter,
                settings=settings,
                dry=dry,
            )

        finally:
            # close reporter when you're done to prevent file handle clashes
            reporter.close()

            # clear GPU context
            # Note: use cache.empty() when openmmtools #690 is resolved
            for context in list(sampler.energy_context_cache._lru._data.keys()):
                del sampler.energy_context_cache._lru._data[context]
            for context in list(sampler.sampler_context_cache._lru._data.keys()):
                del sampler.sampler_context_cache._lru._data[context]
            # cautiously clear out the global context cache too
            for context in list(openmmtools.cache.global_context_cache._lru._data.keys()):
                del openmmtools.cache.global_context_cache._lru._data[context]

            del sampler.sampler_context_cache, sampler.energy_context_cache

            # Keep these around in a dry run so we can inspect things
            if not dry:
                del integrator, sampler

        if not dry:
            nc = self.shared_basepath / settings["output_settings"].output_filename
            chk = self.shared_basepath / settings["output_settings"].checkpoint_storage_filename
            return {
                "trajectory": nc,
                "checkpoint": chk,
            }
        else:
            return {
                "sampler": sampler,
                "integrator": integrator,
            }

    def _execute(
        self,
        ctx: gufe.Context,
        *,
        setup_results,
        **inputs,
    ) -> dict[str, Any]:
        log_system_probe(logging.INFO, paths=[ctx.scratch])

        system = deserialize(setup_results.outputs["system"])
        positions = to_openmm(np.load(setup_results.outputs["positions"]) * offunit.nanometer)
        selection_indices = setup_results.outputs["selection_indices"]
        box_vectors = setup_results.outputs["box_vectors"]

        if setup_results.outputs["restraint_geometry"] is not None:
            alchemical_restraints = True
        else:
            alchemical_restraints = False

        outputs = self.run(
            system=system,
            positions=positions,
            box_vectors=box_vectors,
            selection_indices=selection_indices,
            alchemical_restraints=alchemical_restraints,
            scratch_basepath=ctx.scratch,
            shared_basepath=ctx.shared,
        )

        return {
            "repeat_id": self._inputs["repeat_id"],
            "generation": self._inputs["generation"],
            "simtype": self.simtype,
            **outputs,
        }


class BaseAbsoluteMultiStateAnalysisUnit(gufe.ProtocolUnit, AbsoluteUnitMixin):
    @staticmethod
    def _analyze_multistate_energies(
        trajectory: pathlib.Path,
        checkpoint: pathlib.Path,
        sampler_method: str,
        output_directory: pathlib.Path,
        dry: bool,
    ):
        """
        Analyze multistate energies and generate plots.

        Parameters
        ----------
        trajectory : pathlib.Path
          Path to the NetCDF trajectory file.
        checkpoint : pathlib.Path
          The name of the checkpoint file. Note this is
          relative in path to the trajectory file.
        sampler_method : str
          The multistate sampler method used.
        output_directory : pathlib.Path
          The path to where plots will be written.
        dry : bool
          Whether or not we are running a dry run.
        """
        reporter = multistate.MultiStateReporter(
            storage=trajectory,
            # Note: openmmtools only wants the name of the checkpoint
            # file, it assumes it to be in the same place as the trajectory
            checkpoint_storage=checkpoint.name,
            open_mode="r",
        )

        analyzer = multistate_analysis.MultistateEquilFEAnalysis(
            reporter=reporter,
            sampling_method=sampler_method,
            result_units=offunit.kilocalorie_per_mole,
        )

        # Only create plots when not doing a dry run
        if not dry:
            analyzer.plot(filepath=output_directory, filename_prefix="")

        analyzer.close()
        reporter.close()
        return analyzer.unit_results_dict

    def run(
        self,
        *,
        trajectory: pathlib.Path,
        checkpoint: pathlib.Path,
        dry: bool = False,
        verbose: bool = True,
        scratch_basepath: pathlib.Path | None = None,
        shared_basepath: pathlib.Path | None = None,
    ) -> dict[str, Any]:
        """Analyze the multistate simulation.

        Parameters
        ----------
        trajectory : pathlib.Path
          Path to the MultiStateReporter generated NetCDF file.
        checkpoint : pathlib.Path
          Path to the checkpoint file generated by MultiStateReporter.
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
          Outputs created in the basepath directory or the debug objects
          (i.e. sampler) if ``dry==True``.
        """
        # Prepare paths & verbosity
        self._prepare(verbose, scratch_basepath, shared_basepath)

        # Get the settings
        settings = self._get_settings()

        # Energies analysis
        if verbose:
            self.logger.info("Analyzing energies")

        energy_analysis = self._analyze_multistate_energies(
            trajectory=trajectory,
            checkpoint=checkpoint,
            sampler_method=settings["simulation_settings"].sampler_method.lower(),
            output_directory=self.shared_basepath,
            dry=dry,
        )

        return energy_analysis

    def _execute(
        self,
        ctx: gufe.Context,
        *,
        setup_results,
        simulation_results,
        **inputs,
    ) -> dict[str, Any]:
        log_system_probe(logging.INFO, paths=[ctx.scratch])

        pdb_file = setup_results.outputs["pdb_structure"]
        selection_indices = setup_results.outputs["selection_indices"]
        restraint_geometry = setup_results.outputs["restraint_geometry"]
        standard_state_corr = setup_results.outputs["standard_state_correction"]
        trajectory = simulation_results.outputs["trajectory"]
        checkpoint = simulation_results.outputs["checkpoint"]

        outputs = self.run(
            trajectory=trajectory,
            checkpoint=checkpoint,
            scratch_basepath=ctx.scratch,
            shared_basepath=ctx.shared,
        )

        return {
            "repeat_id": self._inputs["repeat_id"],
            "generation": self._inputs["generation"],
            "simtype": self.simtype,
            # We re-include things here also to make
            # life easier when gathering results.
            "pdb_structure": pdb_file,
            "trajectory": trajectory,
            "checkpoint": checkpoint,
            "selection_indices": selection_indices,
            "restraint_geometry": restraint_geometry,
            "standard_state_correction": standard_state_corr,
            **outputs,
        }
