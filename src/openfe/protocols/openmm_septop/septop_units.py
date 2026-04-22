# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
r"""OpenMM Equilibrium SepTop RBFE Protocol Units
================================================

This module implements the :class:`gufe.ProtocolUnit`\s for the
Separated Topologies RBFE protocol.
"""

from __future__ import annotations

import copy
import itertools
import logging
import pathlib
from typing import Any

import MDAnalysis as mda
import mdtraj as md
import numpy as np
import openmm
import openmm.unit
import openmm.unit as omm_units
from gufe import (
    SmallMoleculeComponent,
    SolvatedPDBComponent,
    SolventComponent,
)
from gufe.settings import SettingsBaseModel
from MDAnalysis.coordinates.memory import MemoryReader
from openff.toolkit.topology import Molecule as OFFMolecule
from openff.units import Quantity
from openff.units.openmm import from_openmm, to_openmm
from openmmtools.states import ThermodynamicState
from rdkit import Chem

from openfe.protocols.openmm_utils import omm_compute
from openfe.protocols.openmm_utils.serialization import serialize
from openfe.protocols.restraint_utils import geometry
from openfe.protocols.restraint_utils.geometry.boresch import BoreschRestraintGeometry
from openfe.protocols.restraint_utils.openmm import omm_restraints
from openfe.protocols.restraint_utils.openmm.omm_restraints import (
    BoreschRestraint,
    add_force_in_separate_group,
)

from ..openmm_utils import (
    settings_validation,
    system_validation,
)
from ..openmm_utils.mdtraj_utils import mdtraj_from_openmm
from ..restraint_utils.settings import (
    BoreschRestraintSettings,
    DistanceRestraintSettings,
)
from .base_units import (
    BaseSepTopAnalysisUnit,
    BaseSepTopRunUnit,
    BaseSepTopSetupUnit,
    _pre_equilibrate,
)

logger = logging.getLogger(__name__)


class SepTopComplexMixin:
    """
    A mixin to get the components and the settings for the Complex Units.
    """

    def _get_components(self):
        """
        Get the relevant components for a complex transformation.

        Returns
        -------
        alchem_comps : dict[str, Component]
          A list of alchemical components
        solv_comp : SolventComponent
          The SolventComponent of the system
        prot_comp : ProteinComponent | None
          The protein component of the system, if it exists.
        small_mols : dict[SmallMoleculeComponent: OFFMolecule]
          SmallMoleculeComponents to add to the system.
        """
        stateA = self._inputs["stateA"]
        alchem_comps = self._inputs["alchemical_components"]

        solv_comp, prot_comp, small_mols = system_validation.get_components(stateA)
        small_mols = {m: m.to_openff() for m in small_mols}
        # Also get alchemical smc from state B
        small_mols_B = {m: m.to_openff() for m in alchem_comps["stateB"]}
        small_mols = small_mols | small_mols_B

        # If there is a SolvatedPDBComponent, we set the solv_comp in the
        # complex to that, as the SolventComponent is only used in the solvent leg
        if isinstance(prot_comp, SolvatedPDBComponent):
            solv_comp = prot_comp

        return alchem_comps, solv_comp, prot_comp, small_mols

    def _get_settings(self) -> dict[str, SettingsBaseModel]:
        """
        Extract the relevant settings for a complex transformation.

        Returns
        -------
        settings : dict[str, SettingsBaseModel]
          A dictionary with the following entries:
            * forcefield_settings : OpenMMSystemGeneratorFFSettings
            * thermo_settings : ThermoSettings
            * charge_settings : OpenFFPartialChargeSettings
            * solvation_settings : OpenMMSolvationSettings
            * alchemical_settings : AlchemicalSettings
            * lambda_settings : LambdaSettings
            * engine_settings : OpenMMEngineSettings
            * integrator_settings : IntegratorSettings
            * equil_simulation_settings : MDSimulationSettings
            * equil_output_settings : SepTopEquilOutputSettings
            * simulation_settings : SimulationSettings
            * output_settings: MultiStateOutputSettings
            * restraint_settings: BoreschRestraintSettings
        """
        prot_settings = self._inputs["protocol"].settings  # type: ignore

        settings = {
            "forcefield_settings": prot_settings.forcefield_settings,
            "thermo_settings": prot_settings.thermo_settings,
            "charge_settings": prot_settings.partial_charge_settings,
            "solvation_settings": prot_settings.complex_solvation_settings,
            "alchemical_settings": prot_settings.alchemical_settings,
            "lambda_settings": prot_settings.complex_lambda_settings,
            "engine_settings": prot_settings.engine_settings,
            "integrator_settings": prot_settings.complex_integrator_settings,
            "equil_simulation_settings": prot_settings.complex_equil_simulation_settings,
            "equil_output_settings": prot_settings.complex_equil_output_settings,
            "simulation_settings": prot_settings.complex_simulation_settings,
            "output_settings": prot_settings.complex_output_settings,
            "restraint_settings": prot_settings.complex_restraint_settings,
        }

        settings_validation.validate_timestep(
            settings["forcefield_settings"].hydrogen_mass,
            settings["integrator_settings"].timestep,
        )

        return settings


class SepTopSolventMixin:
    """
    A mixin to get the components and the settings for the Solvent Units.
    """

    def _get_components(self):
        """
        Get the relevant components for a solvent transformation.

        Note
        -----
        The solvent portion of the transformation is the transformation of one
        ligand into the other in the solvent. The only thing that
        should be present is the alchemical species in state A and state B
        and the SolventComponent.

        Returns
        -------
        alchem_comps : dict[str, Component]
          A list of alchemical components
        solv_comp : SolventComponent
          The SolventComponent of the system
        prot_comp : ProteinComponent | None
          The protein component of the system, if it exists.
        small_mols : dict[SmallMoleculeComponent: OFFMolecule]
          SmallMoleculeComponents to add to the system.
        """
        stateA = self._inputs["stateA"]
        alchem_comps = self._inputs["alchemical_components"]

        small_mols_A = {m: m.to_openff() for m in alchem_comps["stateA"]}
        small_mols_B = {m: m.to_openff() for m in alchem_comps["stateB"]}
        small_mols = small_mols_A | small_mols_B

        solv_comp, _, _ = system_validation.get_components(stateA)

        return alchem_comps, solv_comp, None, small_mols

    def _get_settings(self) -> dict[str, SettingsBaseModel]:
        """
        Extract the relevant settings for a solvent transformation.

        Returns
        -------
        settings : dict[str, SettingsBaseModel]
          A dictionary with the following entries:
            * forcefield_settings : OpenMMSystemGeneratorFFSettings
            * thermo_settings : ThermoSettings
            * charge_settings : OpenFFPartialChargeSettings
            * solvation_settings : OpenMMSolvationSettings
            * alchemical_settings : AlchemicalSettings
            * lambda_settings : LambdaSettings
            * engine_settings : OpenMMEngineSettings
            * integrator_settings : IntegratorSettings
            * equil_simulation_settings : MDSimulationSettings
            * equil_output_settings : SepTopEquilOutputSettings
            * simulation_settings : MultiStateSimulationSettings
            * output_settings: MultiStateOutputSettings
            * restraint_settings: BaseRestraintsSettings
        """
        prot_settings = self._inputs["protocol"].settings  # type: ignore

        settings = {
            "forcefield_settings": prot_settings.forcefield_settings,
            "thermo_settings": prot_settings.thermo_settings,
            "charge_settings": prot_settings.partial_charge_settings,
            "solvation_settings": prot_settings.solvent_solvation_settings,
            "alchemical_settings": prot_settings.alchemical_settings,
            "lambda_settings": prot_settings.solvent_lambda_settings,
            "engine_settings": prot_settings.engine_settings,
            "integrator_settings": prot_settings.solvent_integrator_settings,
            "equil_simulation_settings": prot_settings.solvent_equil_simulation_settings,
            "equil_output_settings": prot_settings.solvent_equil_output_settings,
            "simulation_settings": prot_settings.solvent_simulation_settings,
            "output_settings": prot_settings.solvent_output_settings,
            "restraint_settings": prot_settings.solvent_restraint_settings,
        }

        settings_validation.validate_timestep(
            settings["forcefield_settings"].hydrogen_mass,
            settings["integrator_settings"].timestep,
        )

        return settings


class SepTopComplexSetupUnit(SepTopComplexMixin, BaseSepTopSetupUnit):
    """
    Protocol Unit for the complex phase of a SepTop free energy calculation
    """

    simtype = "complex"

    def get_system_AB(
        self,
        solv_comp: SolventComponent,
        system_modeller_A: openmm.app.Modeller,
        smc_comps_AB: dict[SmallMoleculeComponent, OFFMolecule],
        smc_off_B: dict[SmallMoleculeComponent, OFFMolecule],
        settings: dict[str, SettingsBaseModel],
    ):
        """
        Creates an OpenMM system, topology, positions, and modeller for a
        complex system that contains a protein and two ligands. This takes
        the modeller of complex A (solvated protein-ligand A complex) and
        inserts ligand B into that complex.

        Parameters
        ----------
        solv_comp: SolventComponent
          The SolventComponent
        system_modeller_A: openmm.app.Modeller
        smc_comps_AB: dict[SmallMoleculeComponent,OFFMolecule]
          The dictionary of all SmallMoleculeComponents in the system.
        smc_off_B: dict[SmallMoleculeComponent,OFFMolecule]
          The dictionary of the SmallMoleculeComponent and OFF Molecule of
          ligand B
        settings: dict[str, SettingsBaseModel]
          A dictionary of settings objects for the unit.

        Returns
        -------
        omm_system_AB: openmm.System
        omm_topology_AB: openmm.app.Topology
        positions_AB: openmm.unit.Quantity
        system_modeller_AB: openmm.app.Modeller
        """
        # Get system generator
        system_generator = self._get_system_generator(settings, solv_comp)

        # Get modeller B only ligand B
        modeller_ligandB, comp_resids_ligB = self._get_modeller(
            None,
            None,
            smc_off_B,
            system_generator,
            settings["solvation_settings"],
        )

        # Take the modeller from system A --> every water/ion should be in
        # the same location
        system_modeller_AB = copy.copy(system_modeller_A)
        system_modeller_AB.add(modeller_ligandB.topology, modeller_ligandB.positions)

        omm_topology_AB, omm_system_AB, positions_AB = self._get_omm_objects(
            system_modeller_AB, system_generator, list(smc_comps_AB.values())
        )

        return omm_system_AB, omm_topology_AB, positions_AB, system_modeller_AB

    @staticmethod
    def _get_selection_atom_indices(
        traj: md.Trajectory,
        selection: str = "backbone",
    ):
        """
        Get the atom indices of a MDTraj object, given a selection string.
        Parameters
        ----------
        traj: md.Trajectory
          The Mdtraj trajectory for which to get the atom indices.
        selection: str
          The selection string. Default: 'backbone'

        Returns
        -------
        indices: list
          The list of atom indices that satisfy the selection string.

        Raises
        ------
        ValueError
          If less than three atom indices are found for the selection string.
        """
        indices = traj.topology.select(selection)
        if len(indices) < 3:
            errmsg = (
                f"Less than 3 ({len(indices)} backbone atoms were found For "
                "complex A. No alignment of structures is possible."
                "Currently only proteins are supported as hosts."
            )
            raise ValueError(errmsg)
        return indices

    @staticmethod
    def _update_positions(
        omm_topology_A: openmm.app.Topology,
        omm_topology_B: openmm.app.Topology,
        positions_A: openmm.unit.Quantity,
        positions_B: openmm.unit.Quantity,
    ) -> openmm.unit.Quantity:
        """
        Aligns the protein from complex B onto the protein from complex A and
        updates the positions of complex B.

        Parameters
        ----------
        omm_topology_A: openmm.app.Topology
          OpenMM topology from complex A
        omm_topology_B: openmm.app.Topology
          OpenMM topology from complex B
        positions_A: openmm.unit.Quantity
          Positions of the system in state A
        positions_B: openmm.unit.Quantity
          Positions of the system in state B

        Returns
        -------
        updated_positions_B: openmm.unit.Quantity
          Updated positions of the complex B
        """
        mdtraj_complex_A = mdtraj_from_openmm(omm_topology_A, positions_A)
        mdtraj_complex_B = mdtraj_from_openmm(omm_topology_B, positions_B)
        alignment_indices = SepTopComplexSetupUnit._get_selection_atom_indices(mdtraj_complex_A)
        imaged_complex_B = mdtraj_complex_B.image_molecules()
        imaged_complex_B.superpose(
            mdtraj_complex_A,
            atom_indices=alignment_indices,
        )
        # Extract updated system positions.
        updated_positions_B = imaged_complex_B.openmm_positions(-1)

        return updated_positions_B

    @staticmethod
    def _get_mda_universe(
        topology: openmm.app.Topology,
        positions: openmm.unit.Quantity,
        trajectory: pathlib.Path | None,
        settings: dict[str, SettingsBaseModel],
    ) -> mda.Universe:
        """
        Helper method to get a Universe from an openmm Topology,
        and either an input trajectory or a set of positions.

        Parameters
        ----------
        topology : openmm.app.Topology
          An OpenMM Topology that defines the System.
        positions: openmm.unit.Quantity
          The System's current positions.
          Used if a trajectory file is None or is not a file.
        trajectory: pathlib.Path
          A Path to a trajectory file to read positions from.
        settings: dict
          The settings dictionary

        Returns
        -------
        mda.Universe
          An MDAnalysis Universe of the System.
        """

        # If the trajectory file doesn't exist, then we use positions
        write_int = settings["equil_output_settings"].trajectory_write_interval
        prod_length = settings["equil_simulation_settings"].production_length
        if trajectory is not None and trajectory.is_file() and write_int <= prod_length:
            return mda.Universe(
                topology,
                trajectory,
                topology_format="OPENMMTOPOLOGY",
            )
        else:
            # Positions is an openmm Quantity in nm we need
            # to convert to angstroms
            return mda.Universe(
                topology,
                np.array(positions._value) * 10,
                topology_format="OPENMMTOPOLOGY",
                trajectory_format=MemoryReader,
            )

    @staticmethod
    def _get_boresch_restraint(
        universe: mda.Universe,
        guest_rdmol: Chem.Mol,
        guest_atom_ids: list[int],
        host_atom_ids: list[int],
        temperature: Quantity,
        settings: BoreschRestraintSettings,
    ) -> tuple[BoreschRestraintGeometry, BoreschRestraint]:
        """
        Get a Boresch-like restraint Geometry and OpenMM restraint force
        supplier.

        Parameters
        ----------
        universe : mda.Universe
          An MDAnalysis Universe defining the system to get the restraint for.
        guest_rdmol : Chem.Mol
          An RDKit Molecule defining the guest molecule in the system.
        guest_atom_ids: list[int]
          A list of atom indices defining the guest molecule in the universe.
        host_atom_ids : list[int]
          A list of atom indices defining the host molecules in the universe.
        temperature : unit.Quantity
          The temperature of the simulation where the restraint will be added.
        settings : BoreschRestraintSettings
          Settings on how the Boresch-like restraint should be defined.

        Returns
        -------
        geom : BoreschRestraintGeometry
          A class defining the Boresch-like restraint.
        restraint : BoreschRestraint
          A factory class for generating Boresch restraints in OpenMM.
        """
        frc_const = min(settings.K_thetaA, settings.K_thetaB)

        geom = geometry.boresch.find_boresch_restraint(
            universe=universe,
            guest_rdmol=guest_rdmol,
            guest_idxs=guest_atom_ids,
            host_idxs=host_atom_ids,
            host_selection=settings.host_selection,
            anchor_finding_strategy=settings.anchor_finding_strategy,
            dssp_filter=settings.dssp_filter,
            rmsf_cutoff=settings.rmsf_cutoff,
            host_min_distance=settings.host_min_distance,
            host_max_distance=settings.host_max_distance,
            angle_force_constant=frc_const,
            temperature=temperature,
        )

        restraint = omm_restraints.BoreschRestraint(settings)
        return geom, restraint

    def _add_restraints(
        self,
        system: openmm.System,
        topology_A: openmm.app.Topology,
        topology_B: openmm.app.Topology,
        positions_A: openmm.unit.Quantity,
        positions_B: openmm.unit.Quantity,
        mol_A: SmallMoleculeComponent,
        mol_B: SmallMoleculeComponent,
        ligand_A_inxs: list[int],
        ligand_B_inxs: list[int],
        ligand_B_inxs_B: list[int],
        protein_inxs: list[int],
        settings: dict[str, SettingsBaseModel],
    ) -> tuple[
        Quantity,
        Quantity,
        openmm.System,
        geometry.HostGuestRestraintGeometry,
        geometry.HostGuestRestraintGeometry,
    ]:
        """
        Adds Boresch restraints to the system.

        Parameters
        ----------
        system: openmm.System
          The OpenMM system where the restraints will be applied to.
        topology_A: openmm.app.Topology
          The OpenMM topology that defines the system A
        topology_B: openmm.app.Topology
          The OpenMM topology that defines the system B
        positions_A: openmm.unit.Quantity
          Positions of the system A. This could be a single set of positions,
          or a full trajectory.
        positions_B: openmm.unit.Quantity
          Positions of the system B. This could be a single set of positions,
          or a full trajectory.
        mol_A: SmallMoleculeComponent
          The SmallMoleculeComponent of ligand A
        mol_B: SmallMoleculeComponent
          The SmallMoleculeComponent of ligand B
        ligand_A_inxs: list[int]
          Atom indices of ligand A in the complex A
        ligand_B_inxs: list[int]
          Atom indices of ligand B in the complex B
        ligand_B_inxs_B: list[int]
          Atom indices of ligand B in the full system (AB)
        protein_inxs: list[int]
          Atom indices from the protein atoms
        settings: dict[str, SettingsBaseModel]
          The settings dict

        Returns
        -------
        correction_A: unit.Quantity
          The standard state correction for the restraint for ligand A.
        correction_B: unit.Quantity
          The standard state correction for the restraint for ligand B.
        restrained_system: openmm.System
          The OpenMM system with the added restraints forces
        rest_geom_A: geometry.HostGuestRestraintGeometry
          The restraint Geometry object for ligand A.
        rest_geom_B: geometry.HostGuestRestraintGeometry
          The restraint Geometry object for ligand B.
        """
        # Get the MDA Universe for the restraints selection
        # We try to pass the equilibration production file path through
        # In some cases (debugging / dry runs) this won't be available
        # so we'll default to using input positions.
        out_traj = (
            self.shared_basepath / settings["equil_output_settings"].production_trajectory_filename
        )
        u_A = self._get_mda_universe(
            topology_A,
            positions_A,
            pathlib.Path(f"{out_traj}_stateA.xtc"),
            settings,
        )
        u_B = self._get_mda_universe(
            topology_B,
            positions_B,
            pathlib.Path(f"{out_traj}_stateB.xtc"),
            settings,
        )
        rdmol_A = mol_A.to_rdkit()
        rdmol_B = mol_B.to_rdkit()
        Chem.SanitizeMol(rdmol_A)
        Chem.SanitizeMol(rdmol_B)

        rest_geom_A, restraint_A = self._get_boresch_restraint(
            u_A,
            rdmol_A,
            ligand_A_inxs,
            protein_inxs,
            settings["thermo_settings"].temperature,
            settings["restraint_settings"],
        )

        rest_geom_B, restraint_B = self._get_boresch_restraint(
            u_B,
            rdmol_B,
            ligand_B_inxs_B,
            protein_inxs,
            settings["thermo_settings"].temperature,
            settings["restraint_settings"],
        )
        # We have to update the indices for ligand B to match the AB complex
        new_boresch_B_indices = [ligand_B_inxs_B.index(i) for i in rest_geom_B.guest_atoms]
        rest_geom_B.guest_atoms = [ligand_B_inxs[i] for i in new_boresch_B_indices]

        if self.verbose:
            self.logger.info(
                f"restraint geometry is: ligand A: {rest_geom_A}and ligand B: {rest_geom_B}."
            )

        # We need a temporary thermodynamic state to add the restraint
        # & get the correction
        thermodynamic_state = ThermodynamicState(
            system,
            temperature=to_openmm(settings["thermo_settings"].temperature),
            pressure=to_openmm(settings["thermo_settings"].pressure),
        )

        # Add the force to the thermodynamic state
        restraint_A.add_force(
            thermodynamic_state,
            rest_geom_A,
            controlling_parameter_name="lambda_restraints_A",
        )
        restraint_B.add_force(
            thermodynamic_state,
            rest_geom_B,
            controlling_parameter_name="lambda_restraints_B",
        )
        # Get the standard state correction as a unit.Quantity
        correction_A = restraint_A.get_standard_state_correction(
            thermodynamic_state,
            rest_geom_A,
        )
        correction_B = restraint_B.get_standard_state_correction(
            thermodynamic_state,
            rest_geom_B,
        )
        # Multiply the correction for ligand B by -1 as for this ligands,
        # Boresch restraint has to be turned on in the analytical corr.
        correction_B = -correction_B  # type: ignore[operator]

        # Get the system
        # Note:  you have to remove the thermostat, otherwise you end up
        # with an Andersen thermostat by default!
        restrained_system = thermodynamic_state.get_system(remove_thermostat=True)

        return (
            correction_A,
            correction_B,
            restrained_system,
            rest_geom_A,
            rest_geom_B,
        )

    def run(
        self,
        dry=False,
        verbose=True,
        scratch_basepath=None,
        shared_basepath=None,
    ) -> dict[str, Any]:
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

        self.logger.info("Setting up SepTop complex system.")

        # 1. Get components
        self.logger.info("Creating and setting up the OpenMM systems")
        alchem_comps, solv_comp, prot_comp, smc_comps = self._get_components()
        smc_comps_A, smc_comps_B, smc_comps_AB = self.get_smc_comps(alchem_comps, smc_comps)

        # 3. Get settings
        settings = self._get_settings()

        # 4. Assign partial charges
        self._assign_partial_charges(settings["charge_settings"], smc_comps_AB)

        # 5. Get the OpenMM systems
        omm_system_A, omm_topology_A, positions_A, modeller_A, comp_resids_A = (
            self.get_system(
                solv_comp,
                prot_comp,
                smc_comps_A,
                settings,
            )
        )  # fmt: skip

        omm_system_B, omm_topology_B, positions_B, modeller_B, comp_resids_B = (
            self.get_system(
                solv_comp,
                prot_comp,
                smc_comps_B,
                settings,
            )
        )  # fmt: skip

        smc_B_unique_keys = smc_comps_B.keys() - smc_comps_A.keys()
        smc_comp_B_unique = {key: smc_comps_B[key] for key in smc_B_unique_keys}
        omm_system_AB, omm_topology_AB, positions_AB, modeller_AB = self.get_system_AB(
            solv_comp,
            modeller_A,
            smc_comps_AB,
            smc_comp_B_unique,
            settings,
        )

        # Get the comp_resids of the AB system
        resids_A = list(itertools.chain(*comp_resids_A.values()))
        resids_AB = [r.index for r in modeller_AB.topology.residues()]
        diff_resids = list(set(resids_AB) - set(resids_A))
        comp_resids_AB = comp_resids_A | {alchem_comps["stateB"][0]: np.array(diff_resids)}

        # 6. Pre-equilbrate System (for restraint selection)
        platform = omm_compute.get_openmm_platform(
            platform_name=settings["engine_settings"].compute_platform,
            gpu_device_index=settings["engine_settings"].gpu_device_index,
            restrict_cpu_count=False,
        )

        self.logger.info("Pre-equilibrating the systems")

        equil_positions_A, box_A = _pre_equilibrate(
            system=omm_system_A,
            topology=omm_topology_A,
            positions=positions_A,
            settings=settings,
            endstate="A",
            dry=dry,
            shared_basepath=self.shared_basepath,
            platform=platform,
            verbose=self.verbose,
            logger=self.logger,
        )

        equil_positions_B, box_B = _pre_equilibrate(
            system=omm_system_B,
            topology=omm_topology_B,
            positions=positions_B,
            settings=settings,
            endstate="B",
            dry=dry,
            shared_basepath=self.shared_basepath,
            platform=platform,
            verbose=self.verbose,
            logger=self.logger,
        )

        # 7. Get all the right atom indices for alignments
        comp_atomids_A = self._get_atom_indices(omm_topology_A, comp_resids_A)
        all_atom_ids_A = list(itertools.chain(*comp_atomids_A.values()))
        comp_atomids_B = self._get_atom_indices(omm_topology_B, comp_resids_B)

        # Get the atom indices of ligand B in system B
        atom_indices_B = comp_atomids_B[alchem_comps["stateB"][0]]

        # 8. Update the positions of system B: Align protein
        updated_positions_B = self._update_positions(
            omm_topology_A,
            omm_topology_B,
            equil_positions_A,
            equil_positions_B,
        )

        # Get atom indices for ligand A and ligand B and the solvent in the
        # system AB
        comp_atomids_AB = self._get_atom_indices(omm_topology_AB, comp_resids_AB)
        atom_indices_AB_B = comp_atomids_AB[alchem_comps["stateB"][0]]
        atom_indices_AB_A = comp_atomids_AB[alchem_comps["stateA"][0]]

        # Update positions from AB system
        positions_AB[all_atom_ids_A[0] : all_atom_ids_A[-1] + 1, :] = equil_positions_A
        positions_AB[atom_indices_AB_B[0] : atom_indices_AB_B[-1] + 1, :] = updated_positions_B[
            atom_indices_B[0] : atom_indices_B[-1] + 1
        ]

        # 9. Create the alchemical system
        self.logger.info("Creating the alchemical system and applying restraints")

        alchemical_factory, alchemical_system = self._get_alchemical_system(
            omm_system_AB,
            atom_indices_AB_A,
            atom_indices_AB_B,
            settings["alchemical_settings"],
        )

        # 10. Apply Restraints
        corr_A, corr_B, system, restraint_geom_A, restraint_geom_B = self._add_restraints(
            alchemical_system,
            omm_topology_A,
            omm_topology_B,
            equil_positions_A,
            equil_positions_B,
            alchem_comps["stateA"][0],
            alchem_comps["stateB"][0],
            atom_indices_AB_A,
            atom_indices_AB_B,
            atom_indices_B,
            comp_atomids_AB[prot_comp],
            settings,
        )

        equil_positions_AB, box_AB = _pre_equilibrate(
            system=system,
            topology=omm_topology_AB,
            positions=positions_AB,
            settings=settings,
            endstate="AB",
            dry=dry,
            platform=platform,
            shared_basepath=self.shared_basepath,
            verbose=self.verbose,
            logger=self.logger,
        )

        # Update box vectors
        omm_topology_AB.setPeriodicBoxVectors(box_AB)

        # Subselect system based on user inputs & write initial subsampled PDB
        sub_pdb_structure = self.shared_basepath / settings["output_settings"].output_structure
        selection_indices = self._subsample_topology(
            topology=omm_topology_AB,
            positions=positions_AB,
            output_selection=settings["output_settings"].output_indices,
            output_file=self.shared_basepath / settings["output_settings"].output_structure,
        )
        # The subsampled PDB may not have been written if selection_indices == 0
        # Issue #1942 - maybe move this to the method?
        if len(selection_indices) == 0:
            sub_pdb_structure = None

        # Serialize the system and PDB topology
        system_outfile = self.shared_basepath / "system.xml.bz2"
        serialize(system, system_outfile)

        topology_file = self.shared_basepath / "topology.pdb"
        openmm.app.pdbfile.PDBFile.writeFile(
            omm_topology_AB,
            equil_positions_AB,
            open(topology_file, "w"),
        )

        if not dry:
            return {
                "system": system_outfile,
                "topology": topology_file,
                "standard_state_correction_A": corr_A.to("kilocalorie_per_mole"),
                "standard_state_correction_B": corr_B.to("kilocalorie_per_mole"),
                "restraint_geometry_A": restraint_geom_A.model_dump(),
                "restraint_geometry_B": restraint_geom_B.model_dump(),
                "selection_indices": selection_indices,
                "subsampled_pdb_structure": sub_pdb_structure,
            }
        else:
            return {
                # Add in various objects we can use to test the system
                "system": system_outfile,
                "topology": topology_file,
                "system_A": omm_system_A,
                "system_B": omm_system_B,
                "system_AB": omm_system_AB,
                "alchem_restrained_system": system,
                "alchem_system": alchemical_system,
                "alchem_factory": alchemical_factory,
                "positions": equil_positions_AB,
                "selection_indices": selection_indices,
                "subsampled_pdb_structure": sub_pdb_structure,
            }


class SepTopSolventSetupUnit(SepTopSolventMixin, BaseSepTopSetupUnit):
    """
    Protocol Unit for the solvent phase of a relative SepTop free energy
    """

    simtype = "solvent"

    @staticmethod
    def _update_positions(
        mol_A: SmallMoleculeComponent,
        mol_B: SmallMoleculeComponent,
    ) -> SmallMoleculeComponent:
        """
        Computes the amount to offset the second ligand by in the solution
        phase during RBFE calculations and applies the offset to the ligand,
        returning the SmallMoleculeComponent with the updated positions.

        Parameters
        ----------
        mol_A: SmallMoleculeComponent
          The SmallMoleculeComponent of ligand A
        mol_B: SmallMoleculeComponent
          The SmallMoleculeComponent of ligand B
        Returns
        -------
        updated_mol_B: SmallMoleculeComponent
          The SmallMoleculeComponent of ligand B after updating its positions
          to be a certain distance away from ligand A
        """

        # Convert SmallMolecule to Rdkit Molecule
        rdmol_A = mol_A.to_rdkit()
        rdmol_B = mol_B.to_rdkit()
        # Offset ligand B from ligand A in the solvent
        pos_ligandA = rdmol_A.GetConformers()[0].GetPositions()
        pos_ligandB = rdmol_B.GetConformers()[0].GetPositions()

        ligand_1_radius = np.linalg.norm(pos_ligandA - pos_ligandA.mean(axis=0), axis=1).max()
        ligand_2_radius = np.linalg.norm(pos_ligandB - pos_ligandB.mean(axis=0), axis=1).max()
        ligand_distance = (ligand_1_radius + ligand_2_radius) * 1.5

        ligand_offset = pos_ligandA.mean(0) - pos_ligandB.mean(0)
        ligand_offset[0] += ligand_distance

        # Offset the ligandB.
        pos_ligandB += ligand_offset

        # Extract updated system positions.
        rdmol_B.GetConformers()[0].SetPositions(pos_ligandB)

        updated_mol_B = SmallMoleculeComponent(rdmol_B)

        return updated_mol_B

    def _add_restraints(
        self,
        system: openmm.System,
        ligand_1: Chem.rdchem.Mol,
        ligand_2: Chem.rdchem.Mol,
        ligand_1_inxs: list[int],
        ligand_2_inxs: list[int],
        settings: dict[str, SettingsBaseModel],
        positions_AB: openmm.unit.Quantity,
    ) -> tuple[
        Quantity,
        openmm.System,
    ]:
        """
        Apply the distance restraint between the ligands.

        Parameters
        ----------
        system: openmm.System
          The OpenMM system where the restraints will be applied to.
        ligand_1: Chem.rdchem.Mol
          The RDKit Molecule of ligand A
        ligand_2: Chem.rdchem.Mol
          The RDKit Molecule of ligand B
        ligand_1_idxs: list[int]
          Atom indices from the ligand A in the system.
        ligand_2_idxs: list[int]
          Atom indices from the ligand B in the system.
        settings: dict[str, SettingsBaseModel]
          The settings dict
        positions_AB: openmm.unit.Quantity
          The positions of the OpenMM system

        Returns
        -------
        correction: unit.Quantity
          Standard state correction for the harmonic distance restraint.
        system: openmm.System
          The OpenMM system with the added restraints forces
        """

        if isinstance(settings["restraint_settings"], DistanceRestraintSettings):
            rest_geom = geometry.harmonic.get_molecule_centers_restraint(
                molA_rdmol=ligand_1,
                molB_rdmol=ligand_2,
                molA_idxs=ligand_1_inxs,
                molB_idxs=ligand_2_inxs,
            )

        else:
            # TODO turn this into a direction for different restraint types supported?
            raise NotImplementedError("Other restraint types are not yet available")

        if self.verbose:
            self.logger.info(f"restraint geometry is: {rest_geom}")

        distance = np.linalg.norm(
            positions_AB[rest_geom.guest_atoms[0]] - positions_AB[rest_geom.host_atoms[0]]
        )

        k_distance = to_openmm(settings["restraint_settings"].spring_constant)

        force = openmm.HarmonicBondForce()
        force.addBond(
            rest_geom.guest_atoms[0],
            rest_geom.host_atoms[0],
            distance * openmm.unit.nanometers,
            k_distance,
        )
        force.setName("alignment_restraint")
        # Add force to a separate force group
        add_force_in_separate_group(system, force)

        # No correction necessary as only a single harmonic bond is applied between the ligands
        correction = (
            from_openmm(
                openmm.unit.MOLAR_GAS_CONSTANT_R
                * to_openmm(settings["thermo_settings"].temperature)
            )
            * 0.0
        )

        return correction, system

    def run(
        self, dry=False, verbose=True, scratch_basepath=None, shared_basepath=None
    ) -> dict[str, Any]:
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

        self.logger.info("Setting up SepTop solvent system.")

        # 1. Get components
        self.logger.info("Creating and setting up the OpenMM systems")
        alchem_comps, solv_comp, prot_comp, smc_comps = self._get_components()
        smc_comps_A, smc_comps_B, smc_comps_AB = self.get_smc_comps(alchem_comps, smc_comps)

        # 2. Get settings
        settings = self._get_settings()

        # 3. Assign partial charges
        self._assign_partial_charges(settings["charge_settings"], smc_comps_AB)

        # 4. Update the positions of ligand B:
        #    - solvent: Offset ligand B with respect to ligand A
        smc_B = self._update_positions(
            alchem_comps["stateA"][0],
            alchem_comps["stateB"][0],
        )
        smc_off_B = {smc_B: smc_B.to_openff()}

        # 5. Get the OpenMM systems
        omm_system_AB, omm_topology_AB, positions_AB, modeller_AB, comp_resids_AB = (
            self.get_system(
                solv_comp,
                prot_comp,
                smc_comps_A | smc_off_B,
                settings,
            )
        )  # fmt: skip

        # 6. Get atom indices for ligand A and ligand B and the solvent in the
        # system AB
        comp_atomids_AB = self._get_atom_indices(omm_topology_AB, comp_resids_AB)
        atom_indices_AB_A = comp_atomids_AB[alchem_comps["stateA"][0]]
        atom_indices_AB_B = comp_atomids_AB[smc_B]

        # 7. Create the alchemical system
        self.logger.info("Creating the alchemical system and applying restraints")

        alchemical_factory, alchemical_system = self._get_alchemical_system(
            omm_system_AB,
            atom_indices_AB_A,
            atom_indices_AB_B,
            settings["alchemical_settings"],
        )

        # 8. Apply Restraints
        rdmol_A = alchem_comps["stateA"][0].to_rdkit()
        rdmol_B = smc_B.to_rdkit()
        Chem.SanitizeMol(rdmol_A)
        Chem.SanitizeMol(rdmol_B)

        corr, system = self._add_restraints(
            alchemical_system,
            rdmol_A,
            rdmol_B,
            atom_indices_AB_A,
            atom_indices_AB_B,
            settings,
            positions_AB,
        )

        # Write the full system PDB
        topology_file = self.shared_basepath / "topology.pdb"
        openmm.app.pdbfile.PDBFile.writeFile(
            omm_topology_AB, positions_AB, open(topology_file, "w")
        )

        # Subselect system based on user inputs & write initial subsampled PDB
        sub_pdb_structure = self.shared_basepath / settings["output_settings"].output_structure
        selection_indices = self._subsample_topology(
            topology=omm_topology_AB,
            positions=positions_AB,
            output_selection=settings["output_settings"].output_indices,
            output_file=self.shared_basepath / settings["output_settings"].output_structure,
        )
        # The subsampled PDB may not have been written if selection_indices == 0
        # Issue #1942 - maybe move this to the method?
        if len(selection_indices) == 0:
            sub_pdb_structure = None

        # Serialize the system
        system_outfile = self.shared_basepath / "system.xml.bz2"
        serialize(system, system_outfile)

        if not dry:
            return {
                "system": system_outfile,
                "topology": topology_file,
                "standard_state_correction": corr.to("kilocalorie_per_mole"),
                "selection_indices": selection_indices,
                "subsampled_pdb_structure": sub_pdb_structure,
            }
        else:
            return {
                # Add in various objects we can used to test the system
                "system": system_outfile,
                "topology": topology_file,
                "system_AB": omm_system_AB,
                "alchem_restrained_system": system,
                "alchem_system": alchemical_system,
                "alchem_factory": alchemical_factory,
                "positions": positions_AB,
                "selection_indices": selection_indices,
                "subsampled_pdb_structure": sub_pdb_structure,
            }


class SepTopSolventRunUnit(SepTopSolventMixin, BaseSepTopRunUnit):
    """
    Protocol Unit for the solvent phase of a relative SepTop free energy
    """

    simtype = "solvent"

    def _get_lambda_schedule(
        self, settings: dict[str, SettingsBaseModel]
    ) -> dict[str, list[float]]:
        lambdas = dict()

        lambda_elec_A = settings["lambda_settings"].lambda_elec_A
        lambda_vdw_A = settings["lambda_settings"].lambda_vdw_A
        lambda_elec_B = settings["lambda_settings"].lambda_elec_B
        lambda_vdw_B = settings["lambda_settings"].lambda_vdw_B

        # Reverse lambda schedule since in AbsoluteAlchemicalFactory 1
        # means fully interacting, not stateB
        lambda_elec_A = [1 - x for x in lambda_elec_A]
        lambda_vdw_A = [1 - x for x in lambda_vdw_A]
        lambda_elec_B = [1 - x for x in lambda_elec_B]
        lambda_vdw_B = [1 - x for x in lambda_vdw_B]
        # # Set lambda restraint for the solvent to 1
        # lambda_restraints = len(lambda_elec_A) * [1]

        lambdas["lambda_electrostatics_A"] = lambda_elec_A
        lambdas["lambda_sterics_A"] = lambda_vdw_A
        lambdas["lambda_electrostatics_B"] = lambda_elec_B
        lambdas["lambda_sterics_B"] = lambda_vdw_B
        # lambdas['lambda_restraints'] = lambda_restraints

        return lambdas


class SepTopComplexRunUnit(SepTopComplexMixin, BaseSepTopRunUnit):
    """
    Protocol Unit for the complex phase of a relative SepTop free energy
    """

    simtype = "complex"

    def _get_lambda_schedule(
        self, settings: dict[str, SettingsBaseModel]
    ) -> dict[str, list[float]]:
        lambdas = dict()

        lambda_elec_A = settings["lambda_settings"].lambda_elec_A
        lambda_vdw_A = settings["lambda_settings"].lambda_vdw_A
        lambda_elec_B = settings["lambda_settings"].lambda_elec_B
        lambda_vdw_B = settings["lambda_settings"].lambda_vdw_B
        lambda_restraints_A = settings["lambda_settings"].lambda_restraints_A
        lambda_restraints_B = settings["lambda_settings"].lambda_restraints_B

        # Reverse lambda schedule since in AbsoluteAlchemicalFactory 1
        # means fully interacting, not stateB
        lambda_elec_A = [1 - x for x in lambda_elec_A]
        lambda_vdw_A = [1 - x for x in lambda_vdw_A]
        lambda_elec_B = [1 - x for x in lambda_elec_B]
        lambda_vdw_B = [1 - x for x in lambda_vdw_B]

        lambdas["lambda_electrostatics_A"] = lambda_elec_A
        lambdas["lambda_sterics_A"] = lambda_vdw_A
        lambdas["lambda_electrostatics_B"] = lambda_elec_B
        lambdas["lambda_sterics_B"] = lambda_vdw_B
        lambdas["lambda_restraints_A"] = lambda_restraints_A
        lambdas["lambda_restraints_B"] = lambda_restraints_B

        return lambdas


class SepTopSolventAnalysisUnit(SepTopSolventMixin, BaseSepTopAnalysisUnit):
    """
    Protocol Unit for the analysis of the solvent phase of a relative SepTop free energy
    """

    simtype = "solvent"


class SepTopComplexAnalysisUnit(SepTopComplexMixin, BaseSepTopAnalysisUnit):
    """
    Protocol Unit for the analysis of the complex phase of a relative SepTop free energy
    """

    simtype = "complex"
