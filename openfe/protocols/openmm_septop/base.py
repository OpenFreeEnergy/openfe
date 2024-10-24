# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""OpenMM Equilibrium SepTop Protocol base classes
==================================================

Base classes for the equilibrium OpenMM SepTop free energy ProtocolUnits.

Thist mostly implements BaseSepTopUnit whose methods can be
overriden to define different types of alchemical transformations.

TODO
----
* Add in all the AlchemicalFactory and AlchemicalRegion kwargs
  as settings.
* Allow for a more flexible setting of Lambda regions.
"""
from __future__ import annotations

import abc
import os
import logging

import gufe
from gufe.components import Component
import numpy as np
import numpy.typing as npt
import openmm
from openff.units import unit
from openff.units.openmm import from_openmm, to_openmm, ensure_quantity
from openff.toolkit.topology import Molecule as OFFMolecule
from openmmtools import multistate
from openmmtools.states import (SamplerState,
                                ThermodynamicState,
                                create_thermodynamic_state_protocol,)
from openmmtools.alchemy import (AlchemicalRegion, AbsoluteAlchemicalFactory,
                                 AlchemicalState,)
from typing import Optional
from openmm import app
from openmm import unit as omm_unit
from openmmforcefields.generators import SystemGenerator
import pathlib
from typing import Any
import openmmtools
import mdtraj as mdt

from gufe import (
    settings, ChemicalSystem, SmallMoleculeComponent,
    ProteinComponent, SolventComponent
)
from openfe.protocols.openmm_utils.omm_settings import (
    SettingsBaseModel,
)
from openfe.protocols.openmm_utils.omm_settings import (
    BasePartialChargeSettings,
)
from openfe.protocols.openmm_afe.equil_afe_settings import (
    BaseSolvationSettings,
    MultiStateSimulationSettings, OpenMMEngineSettings,
    IntegratorSettings, LambdaSettings, MultiStateOutputSettings,
    ThermoSettings, OpenFFPartialChargeSettings,
)
from openfe.protocols.openmm_rfe._rfe_utils import compute
from openfe.protocols.openmm_md.plain_md_methods import PlainMDProtocolUnit
from ..openmm_utils import (
    settings_validation, system_creation,
    multistate_analysis, charge_generation
)
from openfe.utils import without_oechem_backend


logger = logging.getLogger(__name__)


class BaseSepTopSetupUnit(gufe.ProtocolUnit):
    """
    Base class for the setup of ligand SepTop RBFE free energy transformations.
    """
    def __init__(self, *,
                 protocol: gufe.Protocol,
                 stateA: ChemicalSystem,
                 stateB: ChemicalSystem,
                 alchemical_components: dict[str, list[Component]],
                 generation: int = 0,
                 repeat_id: int = 0,
                 name: Optional[str] = None,):
        """
        Parameters
        ----------
        protocol : gufe.Protocol
          protocol used to create this Unit. Contains key information such
          as the settings.
        stateA : ChemicalSystem
          ChemicalSystem containing the components defining the state at
          lambda 0.
        stateB : ChemicalSystem
          ChemicalSystem containing the components defining the state at
          lambda 1.
        alchemical_components : dict[str, Component]
          the alchemical components for each state in this Unit
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
            protocol=protocol,
            stateA=stateA,
            stateB=stateB,
            alchemical_components=alchemical_components,
            repeat_id=repeat_id,
            generation=generation,
        )

    @staticmethod
    def _get_alchemical_indices(omm_top: openmm.Topology,
                                comp_resids: dict[Component, npt.NDArray],
                                alchem_comps: dict[str, list[Component]]
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
        residxs = np.concatenate(
            [comp_resids[key] for key in alchem_comps['stateA']]
        )

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
        positions: omm_unit.Quantity,
        settings: dict[str, SettingsBaseModel],
        dry: bool
    ) -> omm_unit.Quantity:
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
        dry: bool
          Whether or not this is a dry run.

        Returns
        -------
        equilibrated_positions : npt.NDArray
          Equilibrated system positions
        """
        # Prep the simulation object
        platform = compute.get_openmm_platform(
            settings['engine_settings'].compute_platform
        )

        integrator = openmm.LangevinMiddleIntegrator(
            to_openmm(settings['thermo_settings'].temperature),
            to_openmm(settings['integrator_settings'].langevin_collision_rate),
            to_openmm(settings['integrator_settings'].timestep),
        )

        simulation = openmm.app.Simulation(
            topology=topology,
            system=system,
            integrator=integrator,
            platform=platform,
        )

        # Get the necessary number of steps
        if settings['equil_simulation_settings'].equilibration_length_nvt is not None:
            equil_steps_nvt = settings_validation.get_simsteps(
                sim_length=settings[
                    'equil_simulation_settings'].equilibration_length_nvt,
                timestep=settings['integrator_settings'].timestep,
                mc_steps=1,
            )
        else:
            equil_steps_nvt = None

        equil_steps_npt = settings_validation.get_simsteps(
            sim_length=settings['equil_simulation_settings'].equilibration_length,
            timestep=settings['integrator_settings'].timestep,
            mc_steps=1,
        )

        prod_steps_npt = settings_validation.get_simsteps(
            sim_length=settings['equil_simulation_settings'].production_length,
            timestep=settings['integrator_settings'].timestep,
            mc_steps=1,
        )

        if self.verbose:
            logger.info("running non-alchemical equilibration MD")

        # Don't do anything if we're doing a dry run
        if dry:
            return positions

        # Use the _run_MD method from the PlainMDProtocolUnit
        # Should in-place modify the simulation
        PlainMDProtocolUnit._run_MD(
            simulation=simulation,
            positions=positions,
            simulation_settings=settings['equil_simulation_settings'],
            output_settings=settings['equil_output_settings'],
            temperature=settings['thermo_settings'].temperature,
            barostat_frequency=settings['integrator_settings'].barostat_frequency,
            timestep=settings['integrator_settings'].timestep,
            equil_steps_nvt=equil_steps_nvt,
            equil_steps_npt=equil_steps_npt,
            prod_steps=prod_steps_npt,
            verbose=self.verbose,
            shared_basepath=self.shared_basepath,
        )

        state = simulation.context.getState(getPositions=True)
        equilibrated_positions = state.getPositions(asNumpy=True)

        # cautiously delete out contexts & integrator
        del simulation.context, integrator

        return equilibrated_positions

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
            self.logger.info("setting up alchemical system")

        # set basepaths
        def _set_optional_path(basepath):
            if basepath is None:
                return pathlib.Path('.')
            return basepath

        self.scratch_basepath = _set_optional_path(scratch_basepath)
        self.shared_basepath = _set_optional_path(shared_basepath)

    @abc.abstractmethod
    def _get_components(self) -> tuple[dict[str, list[Component]],
                                       Optional[gufe.SolventComponent],
                                       Optional[gufe.ProteinComponent],
                                       dict[SmallMoleculeComponent, OFFMolecule]]:
        """
        Get the relevant components to create the alchemical system with.

        Note
        ----
        Must be implemented in the child class.
        """
        ...

    @abc.abstractmethod
    def _handle_settings(self):
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

    def _get_system_generator(
        self, settings: dict[str, SettingsBaseModel],
        solvent_comp: Optional[SolventComponent]
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
        ffcache = settings['output_settings'].forcefield_cache
        if ffcache is not None:
            ffcache = self.shared_basepath / ffcache

        # Block out oechem backend to avoid any issues with
        # smiles roundtripping between rdkit and oechem
        with without_oechem_backend():
            system_generator = system_creation.get_system_generator(
                forcefield_settings=settings['forcefield_settings'],
                integrator_settings=settings['integrator_settings'],
                thermo_settings=settings['thermo_settings'],
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

    def _get_modeller(
        self,
        protein_component: Optional[ProteinComponent],
        solvent_component: Optional[SolventComponent],
        smc_components: dict[SmallMoleculeComponent, OFFMolecule],
        system_generator: SystemGenerator,
        partial_charge_settings: BasePartialChargeSettings,
        solvation_settings: BaseSolvationSettings
    ) -> tuple[app.Modeller, dict[Component, npt.NDArray]]:
        """
        Get an OpenMM Modeller object and a list of residue indices
        for each component in the system.

        Parameters
        ----------
        protein_component : Optional[ProteinComponent]
          Protein Component, if it exists.
        solvent_component : Optional[ProteinCompoinent]
          Solvent Component, if it exists.
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
        system_modeller : app.Modeller
          OpenMM Modeller object generated from ProteinComponent and
          OpenFF Molecules.
        comp_resids : dict[Component, npt.NDArray]
          Dictionary of residue indices for each component in system.
        """
        if self.verbose:
            self.logger.info("Parameterizing molecules")

        # Assign partial charges to smcs
        self._assign_partial_charges(partial_charge_settings, smc_components)

        # TODO: guard the following from non-RDKit backends
        # force the creation of parameters for the small molecules
        # this is necessary because we need to have the FF generated ahead
        # of solvating the system.
        # Block out oechem backend to avoid any issues with
        # smiles roundtripping between rdkit and oechem
        with without_oechem_backend():
            for mol in smc_components.values():
                system_generator.create_system(
                    mol.to_topology().to_openmm(), molecules=[mol]
                )

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
        system_modeller: app.Modeller,
        system_generator: SystemGenerator,
        smc_components: list[OFFMolecule],
    ) -> tuple[app.Topology, openmm.unit.Quantity, openmm.System]:
        """
        Get the OpenMM Topology, Positions and System of the
        parameterised system.

        Parameters
        ----------
        system_modeller : app.Modeller
          OpenMM Modeller object representing the system to be
          parametrized.
        system_generator : SystemGenerator
          SystemGenerator object to create a System with.
        smc_components : list[openff.toolkit.Molecule]
          A list of openff Molecules to add to the system.

        Returns
        -------
        topology : app.Topology
          Topology object describing the parameterized system
        system : openmm.System
          An OpenMM System of the alchemical system.
        positionns : openmm.unit.Quantity
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

    def _get_lambda_schedule(
        self, settings: dict[str, SettingsBaseModel]
    ) -> dict[str, npt.NDArray]:
        """
        Create the lambda schedule

        Parameters
        ----------
        settings : dict[str, SettingsBaseModel]
          Settings for the unit.

        Returns
        -------
        lambdas : dict[str, npt.NDArray]

        TODO
        ----
        * Augment this by using something akin to the RFE protocol's
          LambdaProtocol
        """
        lambdas = dict()

        lambda_elec = settings['lambda_settings'].lambda_elec
        lambda_vdw = settings['lambda_settings'].lambda_vdw

        # Reverse lambda schedule since in AbsoluteAlchemicalFactory 1
        # means fully interacting, not stateB
        lambda_elec = [1-x for x in lambda_elec]
        lambda_vdw = [1-x for x in lambda_vdw]
        lambdas['lambda_electrostatics'] = lambda_elec
        lambdas['lambda_sterics'] = lambda_vdw

        return lambdas

    def _add_restraints(self, system, topology, settings):
        """
        Placeholder method to add restraints if necessary
        """
        return

    def _get_alchemical_system(
        self,
        topology: app.Topology,
        system: openmm.System,
        comp_resids: dict[Component, npt.NDArray],
        alchem_comps: dict[str, list[Component]]
    ):
        """
        Empty placeholder for getting
        an alchemically modified system and its associated factory
        """

        return

    def _get_states(
        self,
        alchemical_system: openmm.System,
        positions: openmm.unit.Quantity,
        settings: dict[str, SettingsBaseModel],
        lambdas: dict[str, npt.NDArray],
        solvent_comp: Optional[SolventComponent],
    ) -> tuple[list[SamplerState], list[ThermodynamicState]]:
        """
        Empty placeholder for getting
        a list of sampler and thermodynmic states from an
        input alchemical system.
        """

        return

    def run(self, dry=False, verbose=True,
            scratch_basepath=None, shared_basepath=None) -> dict[str, Any]:
        """
        Run the SepTop free energy calculation.

        Parameters
        ----------
        dry : bool
          Do a dry run of the calculation, creating all necessary alchemical
          system components (topology, system, sampler, etc...) but without
          running the simulation, default False
        verbose : bool
          Verbose output of the simulation progress. Output is provided via
          INFO level logging, default True
        scratch_basepath : pathlib.Path
          Path to the scratch (temporary) directory space.
        shared_basepath : pathlib.Path
          Path to the shared (persistent) directory space.

        Returns
        -------
        dict
          Outputs created in the basepath directory or the debug objects
          (i.e. sampler) if ``dry==True``.
        """
        # 0. Generaly preparation tasks
        self._prepare(verbose, scratch_basepath, shared_basepath)

        # 1. Get components
        alchem_comps, solv_comp, prot_comp, smc_comps = self._get_components()

        # 2. Get settings
        settings = self._handle_settings()

        # 3. Get system generator
        system_generator = self._get_system_generator(settings, solv_comp)

        # 4. Get modeller
        system_modeller, comp_resids = self._get_modeller(
            prot_comp, solv_comp, smc_comps, system_generator,
            settings['charge_settings'],
            settings['solvation_settings'],
        )

        # 5. Get OpenMM topology, positions and system
        omm_topology, omm_system, positions = self._get_omm_objects(
            system_modeller, system_generator, list(smc_comps.values())
        )

        # 6. Pre-equilbrate System (Test + Avoid NaNs + get stable system)
        positions = self._pre_equilibrate(
            omm_system, omm_topology, positions, settings, dry
        )

        # 7. Get lambdas
        lambdas = self._get_lambda_schedule(settings)

        # # 8. Add restraints
        # self._add_restraints(omm_system, omm_topology, settings)
        #
        # # 9. Get alchemical system
        # alchem_factory, alchem_system, alchem_indices = self._get_alchemical_system(
        #     omm_topology, omm_system, comp_resids, alchem_comps
        # )
        #
        # # 10. Get compound and sampler states
        # sampler_states, cmp_states = self._get_states(
        #     alchem_system, positions, settings,
        #     lambdas, solv_comp
        # )
        #
        #
        # eventually save the serialized alchemical systems to disc to be
        # picked up by the run unit


class BaseSepTopRunUnit(gufe.ProtocolUnit):
    """
    Empty place holder
    Base class for running ligand SepTop RBFE free energy transformations.
    """
