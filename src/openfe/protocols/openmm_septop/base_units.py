# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""OpenMM Equilibrium SepTop Protocol base classes
==================================================

Base classes for the equilibrium OpenMM SepTop free energy ProtocolUnits.

This mostly implements BaseSepTopUnit whose methods can be
overridden to define different types of alchemical transformations.

TODO
----
* Add in all the AlchemicalFactory and AlchemicalRegion kwargs
  as settings.
"""

import abc
import logging
import pathlib
from typing import Any, Literal, Optional

import gufe
import numpy.typing as npt
import openmm
import openmmtools
from gufe import (
    ChemicalSystem,
    ProteinComponent,
    SmallMoleculeComponent,
    SolventComponent,
)
from gufe.components import Component
from openff.toolkit.topology import Molecule as OFFMolecule
from openff.units import unit as offunit
from openff.units.openmm import ensure_quantity, from_openmm, to_openmm
from openmm import unit as omm_unit
from openmmforcefields.generators import SystemGenerator
from openmmtools import multistate
from openmmtools.alchemy import AbsoluteAlchemicalFactory, AlchemicalRegion
from openmmtools.states import (
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
from openfe.protocols.openmm_utils import omm_compute
from openfe.protocols.openmm_utils.omm_settings import SettingsBaseModel
from openfe.protocols.openmm_utils.serialization import deserialize
from openfe.utils import log_system_probe, without_oechem_backend

from ..openmm_utils import (
    charge_generation,
    multistate_analysis,
    settings_validation,
    system_creation,
)
from ..openmm_utils.mdtraj_utils import mdtraj_from_openmm
from .utils import SepTopParameterState

logger = logging.getLogger(__name__)


def _pre_equilibrate(
    system: openmm.System,
    topology: openmm.app.Topology,
    positions: omm_unit.Quantity,
    settings: dict[str, SettingsBaseModel],
    endstate: Literal["A", "B", "AB"],
    dry: bool,
    shared_basepath: pathlib.Path,
    platform: openmm.Platform,
    verbose: bool,
    logger,
) -> tuple[omm_unit.Quantity, omm_unit.Quantity]:
    """
    Run a non-alchemical equilibration to get a stable system.

    Parameters
    ----------
    system : openmm.System
      An OpenMM System to equilibrate.
    topology : openmm.app.Topology
      OpenMM Topology of the System.
    positions : openmm.unit.Quantity
      Initial positions for the system.
    settings : dict[str, SettingsBaseModel]
      A dictionary of settings objects. Expects the
      following entries:
      * `engine_settings`
      * `thermo_settings`
      * `integrator_settings`
      * `equil_simulation_settings`
      * `equil_output_settings`
    endstate: Literal['A', 'B', 'AB']
      The endstate that is pre-equilibrated,can be 'A', 'B' or 'AB'.
    dry: bool
      Whether or not this is a dry run.
    shared_basepath: pathlib.Path
      The Path to the shared storage.
    verbose: bool
      Whether to print extra information
    logger: logging.getLogger
      Name of the logger

    Returns
    -------
    equilibrated_positions : npt.NDArray
      Equilibrated system positions
    box : openmm.unit.Quantity
      Box vectors of the equilibrated system.
    """
    # Prep the simulation object
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

    if verbose:
        logger.info("running non-alchemical equilibration MD")

    # Don't do anything if we're doing a dry run
    if dry:
        box = system.getDefaultPeriodicBoxVectors()
        return positions, to_openmm(from_openmm(box))

    # TODO: Refactor this part to live outside the method call
    # We have to modify the output settings to have different output
    # names for the files from the two end states
    unfrozen_outsettings = settings["equil_output_settings"].unfrozen_copy()

    if endstate == "A" or endstate == "B" or endstate == "AB":
        if unfrozen_outsettings.production_trajectory_filename:
            unfrozen_outsettings.production_trajectory_filename = (
                unfrozen_outsettings.production_trajectory_filename + f"_state{endstate}.xtc"
            )
        if unfrozen_outsettings.preminimized_structure:
            unfrozen_outsettings.preminimized_structure = (
                unfrozen_outsettings.preminimized_structure + f"_state{endstate}.pdb"
            )
        if unfrozen_outsettings.minimized_structure:
            unfrozen_outsettings.minimized_structure = (
                unfrozen_outsettings.minimized_structure + f"_state{endstate}.pdb"
            )
        if unfrozen_outsettings.equil_nvt_structure:
            unfrozen_outsettings.equil_nvt_structure = (
                unfrozen_outsettings.equil_nvt_structure + f"_state{endstate}.pdb"
            )
        if unfrozen_outsettings.equil_npt_structure:
            unfrozen_outsettings.equil_npt_structure = (
                unfrozen_outsettings.equil_npt_structure + f"_state{endstate}.pdb"
            )
        if unfrozen_outsettings.log_output:
            unfrozen_outsettings.log_output = (
                unfrozen_outsettings.log_output + f"_state{endstate}.log"
            )
    else:
        errmsg = f"Only 'A', 'B', and 'AB' are accepted as endstates. Got {endstate}"
        raise ValueError(errmsg)

    # Use the _run_MD method from the PlainMDProtocolUnit
    # Should in-place modify the simulation
    PlainMDProtocolUnit._run_MD(
        simulation=simulation,
        positions=positions,
        simulation_settings=settings["equil_simulation_settings"],
        output_settings=unfrozen_outsettings,
        temperature=settings["thermo_settings"].temperature,
        barostat_frequency=settings["integrator_settings"].barostat_frequency,
        timestep=settings["integrator_settings"].timestep,
        equil_steps_nvt=equil_steps_nvt,
        equil_steps_npt=equil_steps_npt,
        prod_steps=prod_steps_npt,
        verbose=verbose,
        shared_basepath=shared_basepath,
    )
    state = simulation.context.getState(
        getPositions=True,
    )
    equilibrated_positions = state.getPositions(asNumpy=True)
    box = state.getPeriodicBoxVectors()

    # cautiously delete out contexts & integrator
    del simulation.context, integrator

    return equilibrated_positions, to_openmm(from_openmm(box))


class SepTopUnitMixin:
    """
    Mixin for SepTop ProtocolUnits, defining some of the common methods.
    """

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


class BaseSepTopSetupUnit(gufe.ProtocolUnit, SepTopUnitMixin):
    """
    Base class for the setup of ligand SepTop RBFE free energy transformations.
    """

    def _get_alchemical_system(
        self,
        system: openmm.System,
        alchem_indices_A: list[int],
        alchem_indices_B: list[int],
        alchemical_settings: AlchemicalSettings,
    ) -> tuple[AbsoluteAlchemicalFactory, openmm.System]:
        """
        Get an alchemically modified system and its associated factory

        Parameters
        ----------
        system : openmm.System
          System to alchemically modify.
        alchem_indices_A : list[int]
          A list of atom indices for the alchemically modified
          ligand A in the system.
        alchem_indices_B : list[int]
          A list of atom indices for the alchemically modified
          ligand B in the system.
        alchemical_settings : AlchemicalSettings
          Settings controlling how the alchemical system will be built.

        Returns
        -------
        alchemical_factory : AbsoluteAlchemicalFactory
          Factory for creating an alchemically modified system.
        alchemical_system : openmm.System
          Alchemically modified system
        """

        alchemical_factory = AbsoluteAlchemicalFactory(
            consistent_exceptions=False,
            switch_width=1.0 * offunit.angstroms,
            alchemical_pme_treatment="exact",
            alchemical_rf_treatment="switched",
            disable_alchemical_dispersion_correction=alchemical_settings.disable_alchemical_dispersion_correction,
            split_alchemical_forces=True,
        )

        # Alchemical Region for ligand A
        alchemical_region_A = AlchemicalRegion(
            alchemical_atoms=alchem_indices_A,
            name="A",
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

        # Alchemical Region for ligand B
        alchemical_region_B = AlchemicalRegion(
            alchemical_atoms=alchem_indices_B,
            name="B",
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

        alchemical_system = alchemical_factory.create_alchemical_system(
            system, [alchemical_region_A, alchemical_region_B]
        )

        return alchemical_factory, alchemical_system

    @abc.abstractmethod
    def _get_components(
        self,
    ) -> tuple[
        dict[str, list[Component]],
        Optional[gufe.SolventComponent],
        Optional[gufe.ProteinComponent],
        dict[SmallMoleculeComponent, OFFMolecule],
    ]:
        """
        Get the relevant components to create the alchemical system with.

        Note
        ----
        Must be implemented in the child class.
        """
        ...

    def _get_system_generator(
        self,
        settings: dict[str, SettingsBaseModel],
        solvent_comp: Optional[SolventComponent],
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
        ffcache = settings["output_settings"].forcefield_cache
        if ffcache is not None:
            ffcache = self.shared_basepath / ffcache

        # Block out oechem backend to avoid any issues with
        # smiles roundtripping between rdkit and oechem
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
    def _assign_partial_charges(
        partial_charge_settings: OpenFFPartialChargeSettings,
        smc_components: dict[SmallMoleculeComponent, OFFMolecule],
    ) -> None:
        """
        Assign partial charges to OFFMolecules inplace.

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

    def _get_modeller(
        self,
        protein_component: Optional[ProteinComponent],
        solvent_component: SolventComponent,
        smc_components: dict[SmallMoleculeComponent, OFFMolecule],
        system_generator: SystemGenerator,
        solvation_settings: BaseSolvationSettings,
    ) -> tuple[openmm.app.Modeller, dict[Component, npt.NDArray]]:
        """
        Get an OpenMM Modeller object and a list of residue indices
        for each component in the system.

        Parameters
        ----------
        protein_component : Optional[ProteinComponent]
          Protein Component, if it exists.
        solvent_component : SolventComponent
          Solvent Component.
        smc_components : dict[SmallMoleculeComponent, openff.toolkit.Molecule]
          Dictionary of OpenFF Molecules to add, keyed by
          SmallMoleculeComponent.
        system_generator : openmmforcefields.generator.SystemGenerator
          System Generator to parameterise this unit.
        partial_charge_settings : BasePartialChargeSettings
          Settings detailing how to assign partial charges to the
          SMCs of the system.
        solvation_settings : BaseSolvationSettings
          Settings detailing how to solvate the system.

        Returns
        -------
        system_modeller : openmm.app.Modeller
          OpenMM Modeller object generated from ProteinComponent and
          OpenFF Molecules.
        comp_resids : dict[Component, npt.NDArray]
          Dictionary of residue indices for each component in system.
        """
        if self.verbose:
            self.logger.info("Parameterizing molecules")

        # TODO: guard the following from non-RDKit backends
        # force the creation of parameters for the small molecules
        # this is necessary because we need to have the FF generated ahead
        # of solvating the system.
        # Block out oechem backend to avoid any issues with
        # smiles roundtripping between rdkit and oechem
        with without_oechem_backend():
            for mol in smc_components.values():
                system_generator.create_system(mol.to_topology().to_openmm(), molecules=[mol])

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
        system_modeller: openmm.app.Modeller,
        system_generator: SystemGenerator,
        smc_components: list[OFFMolecule],
    ) -> tuple[openmm.app.Topology, openmm.unit.Quantity, openmm.System]:
        """
        Get the OpenMM Topology, Positions and System of the
        parameterised system.

        Parameters
        ----------
        system_modeller : openmm.app.Modeller
          OpenMM Modeller object representing the system to be
          parametrized.
        system_generator : SystemGenerator
          The SystemGenerator object to create a System with.
        smc_components : list[openff.toolkit.Molecule]
          A list of openff Molecules to add to the system.

        Returns
        -------
        topology : openmm.app.Topology
          Topology object describing the parameterized system
        system : openmm.System
          An OpenMM System of the alchemical system.
        positions : openmm.unit.Quantity
          Positions of the system.
        """
        topology = system_modeller.getTopology()
        # roundtrip positions to remove vec3 issues
        positions = to_openmm(from_openmm(system_modeller.getPositions()))

        # Block out oechem backend to avoid any issues with
        # smiles roundtripping between rdkit and oechem
        with without_oechem_backend():
            system = system_generator.create_system(
                system_modeller.topology,
                molecules=smc_components,
            )
        return topology, system, positions

    @staticmethod
    def _get_atom_indices(
        omm_topology: openmm.app.Topology,
        comp_resids: dict[Component, npt.NDArray],
    ) -> dict[Component, list]:
        """
        Get all the atom indices for each component in the system, based on
        the dictionary of residue indices for each component.

        Parameters
        ----------
        omm_topology: openmm.app.Topology
          OpenMM Topology object with the full system.
        comp_resids: dict[Component, npt.NDArray]
          Dictionary of the components in the topology with their residue indices.

        Returns
        -------
        comp_atomids: dict[Component, list]
          A dictionary of atom indices for each component in the System.
        """
        comp_atomids = {}
        for key, values in comp_resids.items():
            atom_indices = []
            for residue in omm_topology.residues():
                if residue.index in values:
                    atom_indices.extend([atom.index for atom in residue.atoms()])
            comp_atomids[key] = atom_indices
        return comp_atomids

    @staticmethod
    def get_smc_comps(
        alchem_comps: dict[str, list[Component]],
        smc_comps: dict[SmallMoleculeComponent, OFFMolecule],
    ) -> tuple[
        dict[SmallMoleculeComponent, OFFMolecule],
        dict[SmallMoleculeComponent, OFFMolecule],
        dict[SmallMoleculeComponent, OFFMolecule],
    ]:
        # Get smcs for the different states and the common smcs
        smc_off_A = {m: m.to_openff() for m in alchem_comps["stateA"]}
        smc_off_B = {m: m.to_openff() for m in alchem_comps["stateB"]}
        # Common smcs could e.g. be cofactors
        smc_off_both = {
            m: m.to_openff()
            for m in smc_comps
            if (m not in alchem_comps["stateA"] and m not in alchem_comps["stateB"])
        }
        smc_comps_A = smc_off_A | smc_off_both
        smc_comps_B = smc_off_B | smc_off_both
        smc_comps_AB = smc_off_A | smc_off_B | smc_off_both

        return smc_comps_A, smc_comps_B, smc_comps_AB

    def get_system(
        self,
        solv_comp: SolventComponent,
        prot_comp: ProteinComponent,
        smc_comp: dict[SmallMoleculeComponent, OFFMolecule],
        settings: dict[str, SettingsBaseModel],
    ):
        """
        Creates an OpenMM system, topology, positions, modeller and also
        residue IDs of the different components

        Parameters
        ----------
        solv_comp: SolventComponent
        prot_comp: Optional[ProteinComponent]
        smc_comp: dict[SmallMoleculeComponent,OFFMolecule]
        settings: dict[str, SettingsBaseModel]
          A dictionary of settings object for the unit.

        Returns
        -------
        omm_system: openmm.app.System
        omm_topology: openmm.app.Topology
        positions: openmm.unit.Quantity
        system_modeller: openmm.app.Modeller
        comp_resids: dict[Component, npt.NDArray]
          A dictionary of residues for each component in the System.
        """
        # Get system generator
        system_generator = self._get_system_generator(settings, solv_comp)

        # Get modeller
        system_modeller, comp_resids = self._get_modeller(
            prot_comp,
            solv_comp,
            smc_comp,
            system_generator,
            settings["solvation_settings"],
        )

        # Get OpenMM topology, positions and system
        omm_topology, omm_system, positions = self._get_omm_objects(
            system_modeller, system_generator, list(smc_comp.values())
        )

        return omm_system, omm_topology, positions, system_modeller, comp_resids

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
        traj = mdtraj_from_openmm(topology, positions)

        selection_indices = traj.topology.select(output_selection)

        # Write out the subselected structure to PDB if not empty
        if len(selection_indices) > 0:
            sub_traj = traj.atom_slice(selection_indices)
            sub_traj.save_pdb(output_file)

        return selection_indices

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
            "simtype": self.simtype,
            **outputs,
        }


class BaseSepTopRunUnit(gufe.ProtocolUnit, SepTopUnitMixin):
    """
    Base class for running ligand SepTop RBFE free energy transformations.
    """

    @abc.abstractmethod
    def _get_components(
        self,
    ) -> tuple[
        dict[str, list[Component]],
        Optional[gufe.SolventComponent],
        Optional[gufe.ProteinComponent],
        dict[SmallMoleculeComponent, OFFMolecule],
    ]:
        """
        Get the relevant components to create the alchemical system with.

        Note
        ----
        Must be implemented in the child class.
        """
        ...

    @abc.abstractmethod
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

        Note
        ----
        Must be implemented in the child class.
        """
        ...

    def _get_states(
        self,
        alchemical_system: openmm.System,
        positions: openmm.unit.Quantity,
        box_vectors: Optional[openmm.unit.Quantity],
        settings: dict[str, SettingsBaseModel],
        lambdas: dict[str, list[float]],
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
        box_vectors : Optional[openmm.unit.Quantity]
          Box vectors of the alchemical system.
        settings : dict[str, SettingsBaseModel]
          A dictionary of settings for the protocol unit.
        lambdas : dict[str, list[float]]
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
        alchemical_state = SepTopParameterState.from_system(alchemical_system)

        # Set up the system constants
        temperature = settings["thermo_settings"].temperature
        pressure = settings["thermo_settings"].pressure
        constants = dict()
        constants["temperature"] = ensure_quantity(temperature, "openmm")
        if solvent_comp is not None:
            constants["pressure"] = ensure_quantity(pressure, "openmm")

        cmp_states = create_thermodynamic_state_protocol(
            alchemical_system,
            protocol=lambdas,
            constants=constants,
            composable_states=[alchemical_state],
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
          Settings controlling the Langevin integrator.
        simulation_settings : MultiStateSimulationSettings
          Settings controlling the simulation.
        system: openmm.System
          The OpenMM System being simulated.

        Returns
        -------
        integrator : openmmtools.mcmc.LangevinDynamicsMove
          A configured integrator object.
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
        # Define the trajectory & checkpoint files
        nc = storage_path / output_settings.output_filename
        # The checkpoint file in openmmtools is taken as the file relative
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
        _SAMPLERS = {
            "repex": multistate.ReplicaExchangeSampler,
            "sams": multistate.SAMSSampler,
            "independent": multistate.MultiStateSampler,
        }

        sampler_method = simulation_settings.sampler_method.lower()
        try:
            sampler_class = _SAMPLERS[sampler_method]
        except KeyError:
            errmsg = f"Unknown sampler {sampler_method}"
            raise AttributeError(errmsg)

        # Get the real time analysis values to use
        rta_its, rta_min_its = settings_validation.convert_real_time_analysis_iterations(
            simulation_settings=simulation_settings,
        )

        # Get the number of production iterations to run for
        steps_per_iteration = integrator.n_steps
        timestep = from_openmm(integrator.timestep)
        number_of_iterations = int(
            settings_validation.get_simsteps(
                sim_length=simulation_settings.production_length,
                timestep=timestep,
                mc_steps=steps_per_iteration,
            )
            / steps_per_iteration
        )

        # convert early_termination_target_error from kcal/mol to kT
        early_termination_target_error = (
            settings_validation.convert_target_error_from_kcal_per_mole_to_kT(
                thermodynamic_settings.temperature,
                simulation_settings.early_termination_target_error,
            )
        )

        sampler_kwargs = {
            "mcmc_moves": integrator,
            "online_analysis_interval": rta_its,
            "online_analysis_target_error": early_termination_target_error,
            "online_analysis_minimum_iterations": rta_min_its,
            "number_of_iterations": number_of_iterations,
        }

        if sampler_method == "sams":
            sampler_kwargs |= {
                "flatness_criteria": simulation_settings.sams_flatness_criteria,
                "gamma0": simulation_settings.sams_gamma0,
            }

        if sampler_method == "repex":
            sampler_kwargs |= {
                "replica_mixing_scheme": "swap-all",
            }

        sampler = sampler_class(**sampler_kwargs)

        sampler.create(
            thermodynamic_states=compound_states,
            sampler_states=sampler_states,
            storage=reporter,
        )

        # Get and set the context caches
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

        Returns
        -------
        unit_results_dict : Optional[dict]
          A dictionary containing all the free energy results,
          if not a dry run.
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
                fn.unlink()

    def run(
        self,
        system: openmm.System,
        pdb_file: openmm.app.pdbfile.PDBFile,
        selection_indices: npt.NDArray,
        dry: bool = False,
        verbose: bool = True,
        scratch_basepath: pathlib.Path | None = None,
        shared_basepath: pathlib.Path | None = None,
    ) -> dict[str, Any]:
        """
        Run the simulation part of the SepTop protocol.

        Parameters
        ----------
        system: openmm.System
          System used for the SepTop calculation.
        pdb_file: openmm.app.pdbfile.PDBFile
          OpenMM PDBFile object representing the SepTop System.
        selection_indices: npt.NDArray
          The indices of the particles to output in the trajectory.
        dry: bool
          Do a dry run of the calculation, creating all necessary alchemical
          system components (topology, system, sampler, etc...) but without
          running the simulation, default False
        verbose: bool
          Verbose output of the simulation progress. Output is provided via
          INFO level logging, default True
        scratch_basepath : pathlib.Path | None
          Path to the scratch (temporary) directory space.
        shared_basepath : pathlib.Path | None
          Path to the shared (persistent) directory space.

        Returns
        -------
        dict: dict[str, Any]
          Dictionary of the outputs created in the basepath directory
          (e.g. path to the simulation .nc file, checkpoint file)
          or the sampler if ``dry==True``.
        """
        # 0. General preparation tasks
        self._prepare(verbose, scratch_basepath, shared_basepath)

        if self.verbose:
            self.logger.info("Running the SepTop simulation.")

        # Get settings, components, and positions
        settings = self._get_settings()
        alchem_comps, solv_comp, prot_comp, smc_comps = self._get_components()
        positions = pdb_file.getPositions(asNumpy=True)

        # Get the compute platform
        platform = omm_compute.get_openmm_platform(
            platform_name=settings["engine_settings"].compute_platform,
            gpu_device_index=settings["engine_settings"].gpu_device_index,
            restrict_cpu_count=False,
        )

        # Check that the restraints are correctly applied by running a short equilibration
        equil_positions, box_AB = _pre_equilibrate(
            system=system,
            topology=pdb_file.topology,
            positions=positions,
            settings=settings,
            endstate="AB",
            dry=dry,
            shared_basepath=self.shared_basepath,
            platform=platform,
            verbose=self.verbose,
            logger=self.logger,
        )

        # Get the lambda schedule
        lambdas = self._get_lambda_schedule(settings)

        # Get compound and sampler states
        sampler_states, cmp_states = self._get_states(
            alchemical_system=system,
            positions=equil_positions,
            box_vectors=box_AB,
            settings=settings,
            lambdas=lambdas,
            solvent_comp=solv_comp,
        )

        # Get the integrator
        integrator = self._get_integrator(
            integrator_settings=settings["integrator_settings"],
            simulation_settings=settings["simulation_settings"],
            system=system,
        )

        # Wrap in try/finally to avoid memory leak issues
        try:
            # Get the reporter
            reporter = self._get_reporter(
                storage_path=self.shared_basepath,
                selection_indices=selection_indices,
                simulation_settings=settings["simulation_settings"],
                output_settings=settings["output_settings"],
            )

            # Get the sampler
            sampler = self._get_sampler(
                integrator=integrator,
                reporter=reporter,
                simulation_settings=settings["simulation_settings"],
                thermodynamic_settings=settings["thermo_settings"],
                compound_states=cmp_states,
                sampler_states=sampler_states,
                platform=platform,
            )

            # 8. Run simulation
            self._run_simulation(
                sampler,
                reporter,
                settings,
                dry,
            )

        finally:
            # Have to wrap this in a try/except, because we might
            # be in a situatino where the reporter and sampler weren't created
            try:
                # Order is reporter, contexts, sampler, integrator
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
                    # At this point we know the sampler exists, so we del the integrator
                    # first since it's associated with the sampler
                    del integrator, sampler
            except UnboundLocalError:
                pass

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
                "equil_positions": equil_positions,
            }

    def _execute(
        self,
        ctx: gufe.Context,
        *,
        setup,
        **kwargs,
    ) -> dict[str, Any]:
        log_system_probe(logging.INFO, paths=[ctx.scratch])

        system = deserialize(setup.outputs["system"])
        pdb_file = openmm.app.pdbfile.PDBFile(str(setup.outputs["topology"]))
        selection_indices = setup.outputs["selection_indices"]

        outputs = self.run(
            system=system,
            pdb_file=pdb_file,
            selection_indices=selection_indices,
            scratch_basepath=ctx.scratch,
            shared_basepath=ctx.shared,
        )

        return {
            "repeat_id": self._inputs["repeat_id"],
            "generation": self._inputs["generation"],
            "simtype": self.simtype,
            **outputs,
        }


class BaseSepTopAnalysisUnit(gufe.ProtocolUnit, SepTopUnitMixin):
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

        if self.verbose:
            self.logger.info("Starting simulation analysis unit")

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
        setup,
        simulation,
        **inputs,
    ) -> dict[str, Any]:
        log_system_probe(logging.INFO, paths=[ctx.scratch])

        trajectory = simulation.outputs["trajectory"]
        checkpoint = simulation.outputs["checkpoint"]

        outputs = self.run(
            trajectory=trajectory,
            checkpoint=checkpoint,
            scratch_basepath=ctx.scratch,
            shared_basepath=ctx.shared,
        )

        # We re-include things here to make life easier when gathering results
        if self.simtype == "complex":
            previous_outputs = {
                "standard_state_correction_A": setup.outputs["standard_state_correction_A"],
                "standard_state_correction_B": setup.outputs["standard_state_correction_B"],
                "restraint_geometry_A": setup.outputs["restraint_geometry_A"],
                "restraint_geometry_B": setup.outputs["restraint_geometry_B"],
            }
        else:
            previous_outputs = {
                "standard_state_correction": setup.outputs["standard_state_correction"]
            }

        previous_outputs["subsampled_pdb_structure"] = setup.outputs["subsampled_pdb_structure"]
        previous_outputs["selection_indices"] = setup.outputs["selection_indices"]
        previous_outputs["trajectory"] = trajectory
        previous_outputs["checkpoint"] = checkpoint

        return {
            "repeat_id": self._inputs["repeat_id"],
            "generation": self._inputs["generation"],
            "simtype": self.simtype,
            **outputs,
            **previous_outputs,
        }
