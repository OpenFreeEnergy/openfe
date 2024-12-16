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
import copy
import logging
import simtk
import itertools
import gufe
from gufe.components import Component
import numpy as np
import numpy.typing as npt
import openmm
import simtk.unit as omm_units
from openff.units import unit
from openff.units.openmm import from_openmm, to_openmm, ensure_quantity
from openff.toolkit.topology import Molecule as OFFMolecule
from openmmtools import multistate
from openmmtools.states import (SamplerState,
                                ThermodynamicState,
                                create_thermodynamic_state_protocol, )
from typing import Optional
from openmm import app
from openmm import unit as omm_unit
from openmmforcefields.generators import SystemGenerator
import pathlib
from typing import Any
import openmmtools
import mdtraj as mdt

from gufe import (
    ChemicalSystem, SmallMoleculeComponent,
    ProteinComponent, SolventComponent
)
from openfe.protocols.openmm_utils.omm_settings import (
    SettingsBaseModel,
)
from openfe.protocols.openmm_utils.omm_settings import (
    BasePartialChargeSettings,
)
from openfe.utils import log_system_probe

from openfe.protocols.openmm_afe.equil_afe_settings import (
    BaseSolvationSettings,
    MultiStateSimulationSettings, OpenMMEngineSettings,
    IntegratorSettings, LambdaSettings, MultiStateOutputSettings,
    ThermoSettings, OpenFFPartialChargeSettings,
)
from openfe.protocols.openmm_septop.equil_septop_settings import SepTopSettings
from openfe.protocols.openmm_utils import omm_compute
from openfe.protocols.openmm_md.plain_md_methods import PlainMDProtocolUnit
from ..openmm_utils import (
    settings_validation, system_creation,
    multistate_analysis, charge_generation
)
from openfe.utils import without_oechem_backend

from .femto_restraints import select_ligand_idxs
from .utils import serialize, deserialize, SepTopParameterState
from openmmtools.alchemy import (
    AbsoluteAlchemicalFactory,
    AlchemicalRegion,
)

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
                 name: Optional[str] = None, ):
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
    def _get_alchemical_indices(
            omm_top: openmm.app.Topology,
            comp_resids: dict[Component, npt.NDArray],
            alchem_comps: dict[str, list[Component]]
    ) -> list[int]:
        """
        Get a list of atom indices for all the alchemical species

        Parameters
        ----------
        omm_top : openmm.app.Topology
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
        platform = omm_compute.get_openmm_platform(
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
            sim_length=settings[
                'equil_simulation_settings'].equilibration_length,
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
            barostat_frequency=settings[
                'integrator_settings'].barostat_frequency,
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
                                       dict[
                                           SmallMoleculeComponent,
                                           OFFMolecule]]:
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
            self,
            settings: dict[str, SettingsBaseModel],
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
            solvent_component: SolventComponent,
            smc_components: dict[SmallMoleculeComponent, OFFMolecule],
            system_generator: SystemGenerator,
            solvation_settings: BaseSolvationSettings
    ) -> tuple[app.Modeller, dict[Component, npt.NDArray]]:
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
        system_modeller : app.Modeller
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

    @staticmethod
    def _get_atom_indices(
            omm_topology: app.Topology,
            comp_resids: dict[Component, npt.NDArray],
    ):
        comp_atomids = {}
        for key, values in comp_resids.items():
            atom_indices = []
            for residue in omm_topology.residues():
                if residue.index in values:
                    atom_indices.extend([atom.index for atom in residue.atoms()])
            comp_atomids[key] = atom_indices
        return comp_atomids

    @staticmethod
    def _update_positions(
            omm_topology_A: openmm.app.Topology,
            omm_topology_B: openmm.app.Topology,
            positions_A: simtk.unit.Quantity,
            positions_B: simtk.unit.Quantity,
            atom_indices_A: Optional[list],
            atom_indices_B: Optional[list],
    ) -> simtk.unit.Quantity:
        """
        Get new positions for the stateB after equilibration.

        Note
        ----
        Must be implemented in the child class.
        In the complex phase, this is achieved by aligning the proteins,
        in the solvent phase, the ligand B are offset from ligand A
        """
        ...

    @staticmethod
    def _set_positions(
            off_molecule: OFFMolecule,
            positions: unit.Quantity,
    ) -> OFFMolecule:
        """
        Updates the positions of an OFFMolecule.Topology
        """
        off_topology = off_molecule.to_topology()
        off_topology.clear_positions()
        off_topology.set_positions(positions)
        return off_molecule

    @staticmethod
    def _add_restraints(
            system: openmm.System,
            positions: simtk.unit.Quantity,
            topology: Optional[openmm.app.Topology],
            ligand_1: Optional[OFFMolecule],
            ligand_2: Optional[OFFMolecule],
            settings: Optional[dict[str, SettingsBaseModel]],
            ligand_1_ref_idxs: tuple[int, int, int],  # indices from the ligand topology
            ligand_2_ref_idxs: tuple[int, int, int],  # indices from the ligand topology
            ligand_1_idxs: tuple[int, int, int],  # indices from the full topology
            ligand_2_idxs: tuple[int, int, int],  # indices from the full topology
    ) -> openmm.System:
        """
        Get new positions for the stateB after equilibration.

        Note
        ----
        Must be implemented in the child class.
        In the complex phase, this is achieved by aligning the proteins,
        in the solvent phase, the ligand B are offset from ligand A
        """
        ...

    @staticmethod
    def get_smc_comps(
            alchem_comps: dict[str, list[Component]],
            smc_comps: dict[SmallMoleculeComponent,OFFMolecule],
    ) -> tuple[dict[SmallMoleculeComponent,OFFMolecule],
               dict[SmallMoleculeComponent,OFFMolecule],
               dict[SmallMoleculeComponent,OFFMolecule],
               dict[SmallMoleculeComponent,OFFMolecule]]:
        # 6. Get smcs for the different states and the common smcs
        smc_off_A = {m: m.to_openff() for m in alchem_comps['stateA']}
        smc_off_B = {m: m.to_openff() for m in alchem_comps['stateB']}
        smc_off_both = {m: m.to_openff() for m in smc_comps
                        if (m not in alchem_comps["stateA"] and m not in
                            alchem_comps["stateB"])}
        smc_comps_A = smc_off_A | smc_off_both
        smc_comps_B = smc_off_B | smc_off_both
        smc_comps_AB = smc_off_A | smc_off_B | smc_off_both

        return smc_comps_A, smc_comps_B, smc_comps_AB, smc_off_B

    def get_system(
            self,
            solv_comp: SolventComponent,
            prot_comp: ProteinComponent,
            smc_comp: dict[SmallMoleculeComponent,OFFMolecule],
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
        omm_system: app.System
        omm_topology: app.Topology
        positions: simtk.unit.Quantity
        system_modeller: app.Modeller
        comp_resids: dict[Component, npt.NDArray]
          A dictionary of residues for each component in the System.
        """
        # 5. Get system generator
        system_generator = self._get_system_generator(settings, solv_comp)

        # 8. Get modeller for stateA, stateB, and stateAB
        system_modeller, comp_resids = self._get_modeller(
            prot_comp, solv_comp, smc_comp,
            system_generator, settings['solvation_settings'],
        )

        # 5. Get OpenMM topology, positions and system
        omm_topology, omm_system, positions = self._get_omm_objects(
            system_modeller, system_generator, list(smc_comp.values())
        )

        return omm_system, omm_topology, positions, system_modeller, comp_resids

    def get_system_AB(
            self,
            solv_comp: SolventComponent,
            system_modeller_A: app.Modeller,
            smc_comps_AB: dict[SmallMoleculeComponent,OFFMolecule],
            smc_off_B: dict[SmallMoleculeComponent,OFFMolecule],
            settings: dict[str, SettingsBaseModel],
            shared_basepath: pathlib.Path,
    ):
        """
        Creates an OpenMM system, topology, positions, and modeller.

        Parameters
        ----------
        solv_comp: SolventComponent
        system_modeller_A: app.Modeller
        smc_comps_AB: dict[SmallMoleculeComponent,OFFMolecule]
        smc_off_B: dict[SmallMoleculeComponent,OFFMolecule]
        settings: dict[str, SettingsBaseModel]
          A dictionary of settings object for the unit.
        shared_basepath: pathlib.Path

        Returns
        -------
        omm_system_AB: app.System
        omm_topology_AB: app.Topology
        positions_AB: simtk.unit.Quantity
        system_modeller_AB: app.Modeller
        """
        # 5. Get system generator
        system_generator = self._get_system_generator(settings, solv_comp)

        # Get modeller B only ligand B
        modeller_ligandB, comp_resids_ligB = self._get_modeller(
            None, None, smc_off_B,
            system_generator, settings['solvation_settings'],
        )

        # Take the modeller from system A --> every water/ion should be in
        # the same location
        system_modeller_AB = copy.copy(system_modeller_A)
        system_modeller_AB.add(modeller_ligandB.topology,
                               modeller_ligandB.positions)

        omm_topology_AB, omm_system_AB, positions_AB = self._get_omm_objects(
            system_modeller_AB, system_generator, list(smc_comps_AB.values())
        )
        simtk.openmm.app.pdbfile.PDBFile.writeFile(omm_topology_AB,
                                                   positions_AB,
                                                   open(
                                                       shared_basepath / 'outputAB.pdb',
                                                       'w'))

        return omm_system_AB, omm_topology_AB, positions_AB, system_modeller_AB

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
        # 0. General preparation tasks
        self._prepare(verbose, scratch_basepath, shared_basepath)

        # 1. Get components
        self.logger.info("Creating and setting up the OpenMM systems")
        alchem_comps, solv_comp, prot_comp, smc_comps = self._get_components()
        smc_comps_A, smc_comps_B, smc_comps_AB, smc_off_B = self.get_smc_comps(
            alchem_comps, smc_comps)

        # 2. Get settings
        settings = self._handle_settings()

        # 7. Assign partial charges here to only do it once for smcs in stateA
        # and stateB (hence only charge e.g. cofactors ones)
        self._assign_partial_charges(settings['charge_settings'], smc_comps_AB)

        # # Get the OpenMM systems
        omm_system_A, omm_topology_A, positions_A, modeller_A, comp_resids_A = self.get_system(
            solv_comp,
            prot_comp,
            smc_comps_A,
            settings,
        )

        omm_system_B, omm_topology_B, positions_B, modeller_B, comp_resids_B = self.get_system(
            solv_comp,
            prot_comp,
            smc_comps_B,
            settings,
        )

        omm_system_AB, omm_topology_AB, positions_AB, modeller_AB = self.get_system_AB(
            solv_comp,
            modeller_A,
            smc_comps_AB,
            smc_off_B,
            settings,
            self.shared_basepath
        )

        # We assume that modeller.add will always put the ligand B towards
        # the end of the residues
        resids_A = list(itertools.chain(*comp_resids_A.values()))
        resids_AB = [r.index for r in modeller_AB.topology.residues()]
        diff_resids = list(set(resids_AB) - set(resids_A))
        comp_resids_AB = comp_resids_A | {
            alchem_comps["stateB"][0]: np.array(diff_resids)}

        # 6. Pre-equilbrate System (Test + Avoid NaNs + get stable system)
        self.logger.info("Pre-equilibrating the systems")
        equ_positions_A = self._pre_equilibrate(
            omm_system_A, omm_topology_A, positions_A, settings, dry
        )
        equ_positions_B = self._pre_equilibrate(
            omm_system_B, omm_topology_B, positions_B, settings, dry
        )

        simtk.openmm.app.pdbfile.PDBFile.writeFile(
            omm_topology_A, equ_positions_A, open(self.shared_basepath / 'outputA_equ.pdb', 'w'))
        simtk.openmm.app.pdbfile.PDBFile.writeFile(
            omm_topology_B, equ_positions_B, open(self.shared_basepath / 'outputB_equ.pdb', 'w'))

        # 7. Get all the right atom indices for alignments
        comp_atomids_A = self._get_atom_indices(omm_topology_A, comp_resids_A)
        all_atom_ids_A = list(itertools.chain(*comp_atomids_A.values()))
        comp_atomids_B = self._get_atom_indices(omm_topology_B, comp_resids_B)

        # Get the system A atom indices of ligand A
        atom_indices_A = comp_atomids_A[alchem_comps['stateA'][0]]
        # Get the system B atom indices of ligand B
        atom_indices_B = comp_atomids_B[alchem_comps['stateB'][0]]

        # 8. Update the positions of system B:
        #    - complex: Align protein
        #    - solvent: Offset ligand B with respect to ligand A
        updated_positions_B = self._update_positions(
            omm_topology_A, omm_topology_B, equ_positions_A, equ_positions_B,
            atom_indices_A, atom_indices_B)
        simtk.openmm.app.pdbfile.PDBFile.writeFile(omm_topology_B,
                                                   updated_positions_B,
                                                   open(self.shared_basepath / 'outputB_new.pdb',
                                                        'w'))

        # Get atom indices for ligand A and ligand B and the solvent in the
        # system AB
        comp_atomids_AB = self._get_atom_indices(omm_topology_AB, comp_resids_AB)
        atom_indices_AB_B = comp_atomids_AB[alchem_comps['stateB'][0]]
        atom_indices_AB_A = comp_atomids_AB[alchem_comps['stateA'][0]]

        # Update positions from AB system
        positions_AB[all_atom_ids_A[0]:all_atom_ids_A[-1] + 1, :] = equ_positions_A
        positions_AB[atom_indices_AB_B[0]:atom_indices_AB_B[-1] + 1,
        :] = updated_positions_B[atom_indices_B[0]:atom_indices_B[-1] + 1]

        # 9. Create the alchemical system
        self.logger.info("Creating the alchemical system and applying restraints")

        factory = AbsoluteAlchemicalFactory(consistent_exceptions=False)
        alchemical_region_A = AlchemicalRegion(
            alchemical_atoms=atom_indices_AB_A, name='ligandA')
        alchemical_region_B = AlchemicalRegion(
            alchemical_atoms=atom_indices_AB_B, name='ligandB')
        alchemical_system = factory.create_alchemical_system(
            omm_system_AB, [alchemical_region_A, alchemical_region_B])
        omm_system_AB = alchemical_system

        # 10. Apply Restraints
        off_A = alchem_comps["stateA"][0].to_openff()
        lig_A_pos = positions_AB[atom_indices_AB_A[0]:atom_indices_AB_A[-1]+1, :] / omm_units.nanometers * unit.nanometer
        self._set_positions(off_A, lig_A_pos)
        off_B = alchem_comps["stateB"][0].to_openff()
        lig_B_pos = positions_AB[
                    atom_indices_AB_B[0]:atom_indices_AB_B[-1] + 1,
                    :] / omm_units.nanometers * unit.nanometer
        self._set_positions(off_B, lig_B_pos)

        ligand_A_ref_inxs, ligand_B_ref_inxs = select_ligand_idxs(off_A, off_B)

        ligand_A_inxs = tuple([atom_indices_AB_A[inx] for inx in ligand_A_ref_inxs])
        ligand_B_inxs = tuple([atom_indices_AB_B[inx] for inx in ligand_B_ref_inxs])
        print(ligand_A_inxs)
        print(ligand_B_inxs)

        system = self._add_restraints(
            omm_system_AB, positions_AB, omm_topology_AB,
            off_A, off_B,
            settings,
            ligand_A_ref_inxs, ligand_B_ref_inxs,
            ligand_A_inxs, ligand_B_inxs)
        # # Check that the restraints are correctly applied by running a short equilibration
        equ_positions_restraints = self._pre_equilibrate(
            system, omm_topology_AB, positions_AB, settings, dry
        )
        topology_file = self.shared_basepath / 'topology.pdb'
        simtk.openmm.app.pdbfile.PDBFile.writeFile(omm_topology_AB,
                                                   equ_positions_restraints,
                                                   open(topology_file,
                                                        'w'))

        # ToDo: also apply REST

        system_outfile = self.shared_basepath / "system.xml.bz2"

        # Serialize system, state and integrator
        serialize(system, system_outfile)

        return {
            "system": system_outfile,
            "topology": topology_file,
        }


class BaseSepTopRunUnit(gufe.ProtocolUnit):
    """
    Base class for running ligand SepTop RBFE free energy transformations.
    """

    @abc.abstractmethod
    def _get_components(self) -> tuple[dict[str, list[Component]],
                                       Optional[gufe.SolventComponent],
                                       Optional[gufe.ProteinComponent],
                                       dict[
                                           SmallMoleculeComponent,
                                           OFFMolecule]]:
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

        Note
        ----
        Must be implemented in the child class.
        """
        ...


    def _get_states(
            self,
            alchemical_system: openmm.System,
            positions: openmm.unit.Quantity,
            settings: dict[str, SettingsBaseModel],
            lambdas: dict[str, npt.NDArray],
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
        settings : dict[str, SettingsBaseModel]
          A dictionary of settings for the protocol unit.
        lambdas : dict[str, npt.NDArray]
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
        # alchemical_state = AlchemicalState.from_system(alchemical_system)
        alchemical_state = SepTopParameterState(
            lambda_sterics_A=1.0,
            lambda_electrostatics_A=1.0,
            lambda_restraints_A=1.0,
            lambda_sterics_B=1.0,
            lambda_electrostatics_B=1.0,
            lambda_restraints_B=1.0,
        )
        # Set up the system constants
        temperature = settings['thermo_settings'].temperature
        pressure = settings['thermo_settings'].pressure
        constants = dict()
        constants['temperature'] = ensure_quantity(temperature, 'openmm')
        if solvent_comp is not None:
            constants['pressure'] = ensure_quantity(pressure, 'openmm')

        cmp_states = create_thermodynamic_state_protocol(
            alchemical_system, protocol=lambdas,
            constants=constants,
            composable_states=[alchemical_state],
        )

        sampler_state = SamplerState(positions=positions)
        if alchemical_system.usesPeriodicBoundaryConditions():
            box = alchemical_system.getDefaultPeriodicBoxVectors()
            sampler_state.box_vectors = box

        sampler_states = [sampler_state for _ in cmp_states]
        # potentials = [state.getPotentialEnergy() for state in sampler_states]
        # print(potentials)

        return sampler_states, cmp_states

    def _get_reporter(
        self,
        topology: app.Topology,
        positions: openmm.unit.Quantity,
        simulation_settings: MultiStateSimulationSettings,
        output_settings: MultiStateOutputSettings,
        shared_basepath: pathlib.Path,
    ) -> multistate.MultiStateReporter:
        """
        Get a MultistateReporter for the simulation you are running.

        Parameters
        ----------
        topology : app.Topology
          A Topology of the system being created.
        positions : openmm.unit.Quantity
          Positions of the pre-alchemical simulation system.
        simulation_settings : MultiStateSimulationSettings
          Multistate simulation control settings, specifically containing
          the amount of time per state sampling iteration.
        output_settings: MultiStateOutputSettings
          Output settings for the simulations
        shared_basepath: pathlib.Path

        Returns
        -------
        reporter : multistate.MultiStateReporter
          The reporter for the simulation.
        """
        mdt_top = mdt.Topology.from_openmm(topology)

        selection_indices = mdt_top.select(
                output_settings.output_indices
        )

        nc = shared_basepath / output_settings.output_filename
        chk = output_settings.checkpoint_storage_filename
        chk_intervals = settings_validation.convert_checkpoint_interval_to_iterations(
            checkpoint_interval=output_settings.checkpoint_interval,
            time_per_iteration=simulation_settings.time_per_iteration,
        )

        reporter = multistate.MultiStateReporter(
            storage=nc,
            analysis_particle_indices=selection_indices,
            checkpoint_interval=chk_intervals,
            checkpoint_storage=chk,
        )

        # Write out the structure's PDB whilst we're here
        if len(selection_indices) > 0:
            traj = mdt.Trajectory(
                positions[selection_indices, :],
                mdt_top.subset(selection_indices),
            )
            traj.save_pdb(
                shared_basepath / output_settings.output_structure
            )

        return reporter


    def _get_ctx_caches(
        self,
        engine_settings: OpenMMEngineSettings
    ) -> tuple[openmmtools.cache.ContextCache, openmmtools.cache.ContextCache]:
        """
        Set the context caches based on the chosen platform

        Parameters
        ----------
        engine_settings : OpenMMEngineSettings,

        Returns
        -------
        energy_context_cache : openmmtools.cache.ContextCache
          The energy state context cache.
        sampler_context_cache : openmmtools.cache.ContextCache
          The sampler state context cache.
        """
        platform = omm_compute.get_openmm_platform(
            engine_settings.compute_platform,
        )

        energy_context_cache = openmmtools.cache.ContextCache(
            capacity=None, time_to_live=None, platform=platform,
        )

        sampler_context_cache = openmmtools.cache.ContextCache(
            capacity=None, time_to_live=None, platform=platform,
        )

        return energy_context_cache, sampler_context_cache

    @staticmethod
    def _get_integrator(
            integrator_settings: IntegratorSettings,
            simulation_settings: MultiStateSimulationSettings
    ) -> openmmtools.mcmc.LangevinDynamicsMove:
        """
        Return a LangevinDynamicsMove integrator

        Parameters
        ----------
        integrator_settings : IntegratorSettings
        simulation_settings : MultiStateSimulationSettings

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
            collision_rate=to_openmm(
                integrator_settings.langevin_collision_rate),
            n_steps=steps_per_iteration,
            reassign_velocities=integrator_settings.reassign_velocities,
            n_restart_attempts=integrator_settings.n_restart_attempts,
            constraint_tolerance=integrator_settings.constraint_tolerance,
        )

        return integrator

    @staticmethod
    def _get_sampler(
            integrator: openmmtools.mcmc.LangevinDynamicsMove,
            reporter: openmmtools.multistate.MultiStateReporter,
            simulation_settings: MultiStateSimulationSettings,
            thermo_settings: ThermoSettings,
            cmp_states: list[ThermodynamicState],
            sampler_states: list[SamplerState],
            energy_context_cache: openmmtools.cache.ContextCache,
            sampler_context_cache: openmmtools.cache.ContextCache
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
        thermo_settings : ThermoSettings
          Thermodynamic settings
        cmp_states : list[ThermodynamicState]
          A list of thermodynamic states to sample.
        sampler_states : list[SamplerState]
          A list of sampler states.
        energy_context_cache : openmmtools.cache.ContextCache
          Context cache for the energy states.
        sampler_context_cache : openmmtool.cache.ContextCache
          Context cache for the sampler states.

        Returns
        -------
        sampler : multistate.MultistateSampler
          A sampler configured for the chosen sampling method.
        """
        rta_its, rta_min_its = settings_validation.convert_real_time_analysis_iterations(
            simulation_settings=simulation_settings,
        )
        et_target_err = settings_validation.convert_target_error_from_kcal_per_mole_to_kT(
            thermo_settings.temperature,
            simulation_settings.early_termination_target_error,
        )

        # Select the right sampler
        # Note: doesn't need else, settings already validates choices
        if simulation_settings.sampler_method.lower() == "repex":
            sampler = multistate.ReplicaExchangeSampler(
                mcmc_moves=integrator,
                online_analysis_interval=rta_its,
                online_analysis_target_error=et_target_err,
                online_analysis_minimum_iterations=rta_min_its
            )
        elif simulation_settings.sampler_method.lower() == "sams":
            sampler = multistate.SAMSSampler(
                mcmc_moves=integrator,
                online_analysis_interval=rta_its,
                online_analysis_minimum_iterations=rta_min_its,
                flatness_criteria=simulation_settings.sams_flatness_criteria,
                gamma0=simulation_settings.sams_gamma0,
            )
        elif simulation_settings.sampler_method.lower() == 'independent':
            sampler = multistate.MultiStateSampler(
                mcmc_moves=integrator,
                online_analysis_interval=rta_its,
                online_analysis_target_error=et_target_err,
                online_analysis_minimum_iterations=rta_min_its,
            )

        sampler.create(
            thermodynamic_states=cmp_states,
            sampler_states=sampler_states,
            storage=reporter
        )

        sampler.energy_context_cache = energy_context_cache
        sampler.sampler_context_cache = sampler_context_cache

        return sampler

    def _run_simulation(
            self,
            sampler: multistate.MultiStateSampler,
            reporter: multistate.MultiStateReporter,
            settings: dict[str, SettingsBaseModel],
            dry: bool,
            verbose: bool,
            shared_basepath: pathlib.Path,
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
            simulation_settings=settings['simulation_settings'],
            integrator_settings=settings['integrator_settings'],
        )

        equil_steps = settings_validation.get_simsteps(
            sim_length=settings['simulation_settings'].equilibration_length,
            timestep=settings['integrator_settings'].timestep,
            mc_steps=mc_steps,
        )
        prod_steps = settings_validation.get_simsteps(
            sim_length=settings['simulation_settings'].production_length,
            timestep=settings['integrator_settings'].timestep,
            mc_steps=mc_steps,
        )

        if not dry:  # pragma: no-cover
            # minimize
            if verbose:
                self.logger.info("minimizing systems")
            sampler.minimize(
                max_iterations=settings[
                    'simulation_settings'].minimization_steps
            )
            # equilibrate
            if verbose:
                self.logger.info("equilibrating systems")

            sampler.equilibrate(int(equil_steps / mc_steps))  # type: ignore

            # production
            if verbose:
                self.logger.info("running production phase")
            sampler.extend(int(prod_steps / mc_steps))  # type: ignore

            if verbose:
                self.logger.info("production phase complete")

            if verbose:
                self.logger.info("post-simulation result analysis")

            analyzer = multistate_analysis.MultistateEquilFEAnalysis(
                reporter,
                sampling_method=settings[
                    'simulation_settings'].sampler_method.lower(),
                result_units=unit.kilocalorie_per_mole
            )
            analyzer.plot(filepath=shared_basepath, filename_prefix="")
            analyzer.close()

            return analyzer.unit_results_dict

        else:
            # close reporter when you're done, prevent file handle clashes
            reporter.close()

            # clean up the reporter file
            fns = [shared_basepath / settings[
                'output_settings'].output_filename,
                   shared_basepath / settings[
                       'output_settings'].checkpoint_storage_filename]
            for fn in fns:
                os.remove(fn)

            return None

    def _execute(
            self, ctx: gufe.Context, *, setup, verbose=True, **kwargs,
                 ) -> dict[str, Any]:
        """
        Execute the simulation part of the SepTop protocol.

        Parameters
        ----------
        ctx : gufe.protocols.protocolunit.Context
            The gufe context for the unit.
        setup : gufe.protocols.ProtocolUnit
            The SetupUnit
        verbose: bool

        Returns
        -------
        dict : dict[str, str]
            Dictionary with paths to ...
        """
        log_system_probe(logging.INFO, paths=[ctx.scratch])
        dry = False
        if ctx.shared is None:
            # use cwd
            shared_basepath = pathlib.Path(".")
        else:
            shared_basepath = ctx.shared

        settings = self._handle_settings()
        alchem_comps, solv_comp, prot_comp, smc_comps = self._get_components()
        if prot_comp:
            phase = "complex"
        else:
            phase = "solvent"

        serialized_system = setup.outputs["system"]
        serialized_topology = setup.outputs["topology"]

        system = deserialize(serialized_system)
        pdb = simtk.openmm.app.pdbfile.PDBFile(str(serialized_topology))
        positions = pdb.getPositions(asNumpy=True)
        lambdas = self._get_lambda_schedule(settings)

        # 10. Get compound and sampler states
        sampler_states, cmp_states = self._get_states(
            system, positions, settings,
            lambdas, solv_comp
        )

        # 11. Create the multistate reporter & create PDB
        reporter = self._get_reporter(
            pdb.topology, positions,
            settings['simulation_settings'],
            settings['output_settings'],
            shared_basepath,
        )

        # Wrap in try/finally to avoid memory leak issues
        try:
            # 12. Get context caches
            energy_ctx_cache, sampler_ctx_cache = self._get_ctx_caches(
                settings['engine_settings']
            )

            # 13. Get integrator
            integrator = self._get_integrator(
                settings['integrator_settings'],
                settings['simulation_settings'],
            )

            # 14. Get sampler
            sampler = self._get_sampler(
                integrator, reporter, settings['simulation_settings'],
                settings['thermo_settings'],
                cmp_states, sampler_states,
                energy_ctx_cache, sampler_ctx_cache
            )

            # 15. Run simulation
            unit_result_dict = self._run_simulation(
                sampler, reporter, settings, dry, verbose, shared_basepath
            )

        finally:
            # close reporter when you're done to prevent file handle clashes
            reporter.close()

            # clear GPU context
            # Note: use cache.empty() when openmmtools #690 is resolved
            for context in list(energy_ctx_cache._lru._data.keys()):
                del energy_ctx_cache._lru._data[context]
            for context in list(sampler_ctx_cache._lru._data.keys()):
                del sampler_ctx_cache._lru._data[context]
            # cautiously clear out the global context cache too
            for context in list(
                    openmmtools.cache.global_context_cache._lru._data.keys()):
                del openmmtools.cache.global_context_cache._lru._data[context]

            del sampler_ctx_cache, energy_ctx_cache

            # Keep these around in a dry run so we can inspect things
            if not dry:
                del integrator, sampler

        if not dry:
            nc = shared_basepath / settings[
                'output_settings'].output_filename
            chk = settings['output_settings'].checkpoint_storage_filename
            return {
                'repeat_id': self._inputs['repeat_id'],
                'generation': self._inputs['generation'],
                'simtype': phase,
                'nc': nc,
                'last_checkpoint': chk,
                **unit_result_dict,
            }
        else:
            return {
                'repeat_id': self._inputs['repeat_id'],
                'generation': self._inputs['generation'],
                'debug': {'sampler': sampler}}
