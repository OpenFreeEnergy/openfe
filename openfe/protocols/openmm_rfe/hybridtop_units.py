# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
ProtocolUnits for Hybrid Topology methods using OpenMM and OpenMMTools in a
Perses-like manner.

Acknowledgements
----------------
These ProtocolUnits are based on, and leverage components originating from
the Perses toolkit (https://github.com/choderalab/perses).
"""

import json
import logging
import os
import pathlib
import subprocess
from itertools import chain
from typing import Any

import matplotlib.pyplot as plt
import mdtraj
import numpy as np
import numpy.typing as npt
import openmm
import openmmtools
from openmmforcefields.generators import SystemGenerator

import gufe
from gufe.settings import (
    SettingsBaseModel,
    ThermoSettings,
)
from gufe import (
    ChemicalSystem,
    LigandAtomMapping,
    Component,
    SolventComponent,
    ProteinComponent,
    SmallMoleculeComponent,
)
from openff.toolkit.topology import Molecule as OFFMolecule
from openff.units import unit as offunit
from openff.units import Quantity
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
from ..openmm_utils.serialization import (
    serialize,
    deserialize,
)
from . import _rfe_utils
from ._rfe_utils.relative import HybridTopologyFactory
from .equil_rfe_settings import (
    AlchemicalSettings,
    IntegratorSettings,
    LambdaSettings,
    MultiStateOutputSettings,
    MultiStateSimulationSettings,
    OpenFFPartialChargeSettings,
    OpenMMSolvationSettings,
    OpenMMEngineSettings,
    RelativeHybridTopologyProtocolSettings,
)

logger = logging.getLogger(__name__)


class HybridTopologyUnitMixin:
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
          Verbose output of the simulation progress. Output is provided at the
          INFO level logging.
        scratch_basepath : pathlib.Path | None
          Optional scratch base path to write scratch files to.
        shared_basepath : pathlib.Path | None
          Optional shared base path to write shared files to.
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
    def _get_settings(
        settings: RelativeHybridTopologyProtocolSettings
    ) -> dict[str, SettingsBaseModel]:
        """
        Get a dictionary of Protocol settings.

        Returns
        -------
        protocol_settings : dict[str, SettingsBaseModel]

        Notes
        -----
        We return a dict so that we can duck type behaviour between phases.
        For example subclasses may contain both `solvent` and `complex`
        settings, using this approach we can extract the relevant entry
        to the same key and pass it to other methods in a seamless manner.
        """
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


class HybridTopologySetupUnit(gufe.ProtocolUnit, HybridTopologyUnitMixin):
    """
    Calculates the relative free energy of an alchemical ligand transformation.
    """
    @staticmethod
    def _get_components(
        stateA: ChemicalSystem,
        stateB: ChemicalSystem
        ) -> tuple[
            SolventComponent,
            ProteinComponent,
            dict[SmallMoleculeComponent, OFFMolecule]
        ]:
        """
        Get the components from the ChemicalSystem inputs.

        Parameters
        ----------
        stateA : ChemicalSystem
          ChemicalSystem defining the state A components.
        stateB : CHemicalSystem
          ChemicalSystem defining the state B components.

        Returns
        -------
        solv_comp : SolventComponent
            The solvent component.
        protein_comp : ProteinComponent
            The protein component.
        small_mols : dict[SmallMoleculeComponent, openff.toolkit.Molecule]
            Dictionary of small molecule components paired
            with their OpenFF Molecule.
        """
        solvent_comp, protein_comp, smcs_A = system_validation.get_components(stateA)
        _, _, smcs_B = system_validation.get_components(stateB)

        small_mols = {
            m: m.to_openff()
            for m in set(smcs_A).union(set(smcs_B))
        }

        return solvent_comp, protein_comp, small_mols

    @staticmethod
    def _assign_partial_charges(
        charge_settings: OpenFFPartialChargeSettings,
        small_mols: dict[SmallMoleculeComponent, OFFMolecule],
    ) -> None:
        """
        Assign partial charges to the OpenFF Molecules associated with all
        the SmallMoleculeComponents in the transformation.

        Parameters
        ----------
        charge_settings : OpenFFPartialChargeSettings
          Settings for controlling how the partial charges are assigned.
        small_mols : dict[SmallMoleculeComponent, openff.toolkit.Molecule]
          Dictionary of OpenFF Molecules to add, keyed by
          their associated SmallMoleculeComponent.
        """
        for smc, mol in small_mols.items():
            charge_generation.assign_offmol_partial_charges(
                offmol=mol,
                overwrite=False,
                method=charge_settings.partial_charge_method,
                toolkit_backend=charge_settings.off_toolkit_backend,
                generate_n_conformers=charge_settings.number_of_conformers,
                nagl_model=charge_settings.nagl_model,
            )

    @staticmethod
    def _get_system_generator(
        settings: dict[str, SettingsBaseModel],
        solvent_component: SolventComponent | None,
        openff_molecules: list[OFFMolecule] | None,
        ffcache: pathlib.Path | None
    ) -> SystemGenerator:
        """
        Get an OpenMM SystemGenerator.

        Parameters
        ----------
        settings : dict[str, SettingsBaseModel]
          A dictionary of protocol settings.
        solvent_component : SolventComponent | None
          The solvent component of the system, if any.
        openff_molecules : list[openff.toolkit.Molecule] | None 
          A list of openff molecules to generate templates for, if any.
        ffcache : pathlib.Path | None
          Path to the force field parameter cache.
        
        Returns
        -------
        system_generator : openmmtools.SystemGenerator
          The SystemGenerator for the protocol.
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
        # and we start loading the whole protein into OpenFF Topologies
        if openff_molecules is None:
            return system_generator
        
        # First deduplicate isomoprhic molecules
        unique_offmols = []
        for mol in openff_molecules:
            unique = all(
                [
                    not mol.is_isomorphic_with(umol)
                    for umol in unique_offmols
                ]
            )
            if unique:
                unique_offmols.append(mol)

        # register all the templates
        system_generator.add_molecules(unique_offmols)
        
        return system_generator

    @staticmethod
    def _create_stateA_system(
        small_mols: dict[SmallMoleculeComponent, OFFMolecule],
        protein_component: ProteinComponent | None,
        solvent_component: SolventComponent | None,
        system_generator: SystemGenerator,
        solvation_settings: OpenMMSolvationSettings,
    ) -> tuple[
        openmm.System,
        openmm.app.Topology,
        openmm.unit.Quantity,
        dict[Component, npt.NDArray]
    ]:
        """
        Create an OpenMM System for state A.

        Parameters
        ----------
        small_mols : dict[SmallMoleculeComponent, openff.toolkit.Molecule]
          A list of small molecules to include in the System.
        protein_component : ProteinComponent | None
          Optionally, the protein component to include in the System.
        solvent_component : SolventComponent | None
          Optionally, the solvent component to include in the System.
        system_generator : SystemGenerator
          The SystemGenerator object ot use to construct the System.
        solvation_settings : OpenMMSolvationSettings
          Settings defining how to build the System.

        Returns
        -------
        system : openmm.System
          The System that defines state A.
        topology : openmm.app.Topology
          The Topology defining the returned System.
        positions : openmm.unit.Quantity
          The positions of the particles in the System.
        comp_residues : dict[Component, npt.NDArray]
          A dictionary defining which residues in the System
          belong to which ChemicalSystem Component.
        """
        modeller, comp_resids = system_creation.get_omm_modeller(
            protein_comp=protein_component,
            solvent_comp=solvent_component,
            small_mols=small_mols,
            omm_forcefield=system_generator.forcefield,
            solvent_settings=solvation_settings,
        )

        topology = modeller.getTopology()
        # Note: roundtrip positions to remove vec3 issues
        positions = to_openmm(from_openmm(modeller.getPositions()))

        system = system_generator.create_system(
            modeller.topology,
            molecules=list(small_mols.values()),
        )

        return system, topology, positions, comp_resids

    @staticmethod
    def _create_stateB_system(
        small_mols: dict[SmallMoleculeComponent, OFFMolecule],
        mapping: LigandAtomMapping,
        stateA_topology: openmm.app.Topology,
        exclude_resids: npt.NDArray,
        system_generator: SystemGenerator,
    ) -> tuple[openmm.System, openmm.app.Topology, npt.NDArray]:
        """
        Create the state B System from the state A Topology.

        Parameters
        ----------
        small_mols : dict[SmallMoleculeComponent, openff.toolkit.Molecule]
          Dictionary of OpenFF Molecules keyed by SmallMoleculeComponent
          to be present in system B.
        mapping : LigandAtomMapping
          LigandAtomMapping defining the correspondance betwee state A
          and B's alchemical ligand.
        stateA_topology : openmm.app.Topology
          The OpenMM topology for state A.
        exclude_resids : npt.NDArray
          A list of residues to exclude from state A when building state B.
        system_generator : SystemGenerator
          The SystemGenerator to use to build System B.

        Returns
        -------
        system : openmm.System
          The state B System.
        topology : openmm.app.Topology
          The OpenMM Topology associated with the state B System.
        alchem_resids : npt.NDArray
          The residue indices of the state B alchemical species.
        """
        topology, alchem_resids = _rfe_utils.topologyhelpers.combined_topology(
            topology1=stateA_topology,
            topology2=small_mols[mapping.componentB].to_topology().to_openmm(),
            exclude_resids=exclude_resids,
        )

        system = system_generator.create_system(
            topology,
            molecules=list(small_mols.values()),
        )

        return system, topology, alchem_resids

    @staticmethod
    def _handle_net_charge(
        stateA_topology: openmm.app.Topology,
        stateA_positions: openmm.unit.Quantity,
        stateB_topology: openmm.app.Topology,
        stateB_system: openmm.System,
        charge_difference: int,
        system_mappings: dict[str, dict[int, int]],
        distance_cutoff: Quantity,
        solvent_component: SolventComponent | None,
    ) -> None:
        """
        Handle system net charge by adding an alchemical water.

        Parameters
        ----------
        stateA_topology : openmm.app.Topology
        stateA_positions : openmm.unit.Quantity
        stateB_topology : openmm.app.Topology
        stateB_system : openmm.System
        charge_difference : int
        system_mappings : dict[str, dict[int, int]]
        distance_cutoff : Quantity
        solvent_component : SolventComponent | None
        """
        # Base case, return if no net charge
        if charge_difference == 0:
            return

        # Get the residue ids for waters to turn alchemical
        alchem_water_resids = _rfe_utils.topologyhelpers.get_alchemical_waters(
            topology=stateA_topology,
            positions=stateA_positions,
            charge_difference=charge_difference,
            distance_cutoff=distance_cutoff,
        )

        # In-place modify state B alchemical waters to ions
        _rfe_utils.topologyhelpers.handle_alchemical_waters(
            water_resids=alchem_water_resids,
            topology=stateB_topology,
            system=stateB_system,
            system_mapping=system_mappings,
            charge_difference=charge_difference,
            solvent_component=solvent_component,
        )

    def _get_omm_objects(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: LigandAtomMapping,
        settings: dict[str, SettingsBaseModel],
        protein_component: ProteinComponent | None,
        solvent_component: SolventComponent | None,
        small_mols: dict[SmallMoleculeComponent, OFFMolecule]
    ) -> tuple[
        openmm.System,
        openmm.app.Topology,
        openmm.unit.Quantity,
        openmm.System,
        openmm.app.Topology,
        openmm.unit.Quantity,
        dict[str, dict[int, int]],
    ]:
        """
        Get OpenMM objects for both end states A and B.

        Parameters
        ----------
        stateA : ChemicalSystem
          ChemicalSystem defining end state A.
        stateB : ChmiecalSysstem
          ChemicalSystem defining end state B.
        mapping : LigandAtomMapping
          The mapping for alchemical components between state A and B.
        settings : dict[str, SettingsBaseModel]
          Settings for the transformation.
        protein_component : ProteinComponent | None
          The common ProteinComponent between the end states, if there is is one.
        solvent_component : SolventComponent | None
          The common SolventComponent between the end states, if there is one.
        small_mols : dict[SmallMoleculeCOmponent, openff.toolkit.Molecule]
          The small molecules for both end states.

        Returns
        -------
        stateA_system : openmm.System
          OpenMM System for state A.
        stateA_topology : openmm.app.Topology
          OpenMM Topology for the state A System.
        stateA_positions : openmm.unit.Quantity
          Positions of partials for state A System.
        stateB_system : openmm.System
          OpenMM System for state B.
        stateB_topology : openmm.app.Topology
          OpenMM Topology for the state B System.
        stateB_positions : openmm.unit.Quantity
          Positions of partials for state B System.
        system_mapping : dict[str, dict[int, int]]
          Dictionary of mappings defining the correspondance between
          the two state Systems.
        """
        if self.verbose:
            self.logger.info("Parameterizing systems")

        def _filter_mols(smols, state):
            return {
                smc: offmol
                for smc, offmol in smols.items()
                if state.contains(smc)
            }

        states_inputs = {
            'A': {'state': stateA, 'mols': _filter_mols(small_mols, stateA)},
            'B': {'state': stateB, 'mols': _filter_mols(small_mols, stateB)},
        }

        # Everything involving systemgenerator handling has a risk of
        # oechem <-> rdkit smiles conversion clashes, cautiously ban it.
        with without_oechem_backend():
            # Get the system generators with all the templates registered
            for state in ['A', 'B']:
                ffcache = settings["output_settings"].forcefield_cache
                if ffcache is not None:
                    ffcache = self.shared_basepath / (f"{state}_" + ffcache)

                states_inputs[state]['generator'] = self._get_system_generator(
                    settings=settings,
                    solvent_component=solvent_component,
                    openff_molecules=list(states_inputs[state]['mols'].values()),
                    ffcache=ffcache,
                )

            (
                stateA_system, stateA_topology, stateA_positions,
                comp_resids
            ) = self._create_stateA_system(
                small_mols=states_inputs['A']['mols'],
                protein_component=protein_component,
                solvent_component=solvent_component,
                system_generator=states_inputs['A']['generator'],
                solvation_settings=settings["solvation_settings"]
            )

            (
                stateB_system, stateB_topology, stateB_alchem_resids
            ) = self._create_stateB_system(
                small_mols=states_inputs['B']['mols'],
                mapping=mapping,
                stateA_topology=stateA_topology,
                exclude_resids=comp_resids[mapping.componentA],
                system_generator=states_inputs['B']['generator'],
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

        # Net charge: add alchemical water if needed
        # Must be done here as we in-place modify the particles of state B.
        if settings["alchemical_settings"].explicit_charge_correction:
            self._handle_net_charge(
                stateA_topology=stateA_topology,
                stateA_positions=stateA_positions,
                stateB_topology=stateB_topology,
                stateB_system=stateB_system,
                charge_difference=mapping.get_alchemical_charge_difference(),
                system_mappings=system_mappings,
                distance_cutoff=settings["alchemical_settings"].explicit_charge_correction_cutoff,
                solvent_component=solvent_component,
            )

        # Finally get the state B positions
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
        stateA_system: openmm.System,
        stateA_positions: openmm.unit.Quantity,
        stateA_topology: openmm.app.Topology,
        stateB_system: openmm.System,
        stateB_positions: openmm.unit.Quantity,
        stateB_topology: openmm.app.Topology,
        system_mappings: dict[str, dict[int, int]],
        alchemical_settings: AlchemicalSettings,
    ):
        """
        Get the hybrid topology alchemical system.

        Parameters
        ----------
        stateA_system : openmm.System
          State A OpenMM System
        stateA_positions : openmm.unit.Quantity
          Positions of state A System
        stateA_topology : openmm.app.Topology
          Topology of state A System
        stateB_system : openmm.System
          State B OpenMM System
        stateB_positions : openmm.unit.Quantity
          Positions of state B System
        stateB_topology : openmm.app.Topology
          Topology of state B System
        system_mappings : dict[str, dict[int, int]]
          Mapping of corresponding atoms between the two Systems.
        alchemical_settings : AlchemicalSettings
          The alchemical settings defining how the alchemical system
          will be built.

        Returns
        -------
        hybrid_factory : HybridTopologyFactory
          The factory creating the hybrid system.
        hybrid_system : openmm.System
          The hybrid System.
        """
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

    def _subsample_topology(
        self,
        hybrid_topology: openmm.app.Topology,
        hybrid_positions: openmm.unit.Quantity,
        output_selection: str,
        output_filename: str,
        atom_classes: dict[str, set[int]],
    ) -> npt.NDArray:
        """
        Subsample the hybrid topology based on user-selected output selection
        and write the subsampled topology to a PDB file.

        Parameters
        ----------
        hybrid_topology : openmm.app.Topology
          The hybrid system topology to subsample.
        hybrid_positions : openmm.unit.Quantity
          The hybrid system positions.
        output_selection : str
          An MDTraj selection string to subsample the topology with.
        output_filename : str
          The name of the file to write the PDB to.
        atom_classes : dict[str, set[int]]
          A dictionary defining what atoms belong to the different
          components of the hybrid system.

        Returns
        -------
        selection_indices : npt.NDArray
          The indices of the subselected system.

        TODO
        ----
        Modify this to also store the full system.
        """
        selection_indices = hybrid_topology.select(output_selection)

        # Write out a PDB containing the subsampled hybrid state
        # We use bfactors as a hack to label different states
        # bfactor of 0 is environment atoms
        # bfactor of 0.25 is unique old atoms
        # bfactor of 0.5 is core atoms
        # bfactor of 0.75 is unique new atoms
        bfactors = np.zeros_like(selection_indices, dtype=float)
        bfactors[np.isin(selection_indices, list(atom_classes['unique_old_atoms']))] = 0.25
        bfactors[np.isin(selection_indices, list(atom_classes['core_atoms']))] = 0.50
        bfactors[np.isin(selection_indices, list(atom_classes['unique_new_atoms']))] = 0.75

        if len(selection_indices) > 0:
            traj = mdtraj.Trajectory(
                hybrid_positions[selection_indices, :],
                hybrid_topology.subset(selection_indices),
            ).save_pdb(
                self.shared_basepath / output_filename,
                bfactors=bfactors,
            )

        return selection_indices

    def run(
        self,
        *,
        dry: bool = False,
        verbose: bool = True,
        scratch_basepath: pathlib.Path | None = None,
        shared_basepath: pathlib.Path | None = None
    ) -> dict[str, Any]:
        """Setup a hybrid topology system.

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
        # Prepare paths & verbosity
        self._prepare(verbose, scratch_basepath, shared_basepath)

        # Get settings
        settings = self._get_settings(self._inputs["protocol"].settings)

        # Get components
        stateA = self._inputs["stateA"]
        stateB = self._inputs["stateB"]
        mapping = self._inputs["ligandmapping"]
        alchem_comps = self._inputs["alchemical_components"]
        solvent_comp, protein_comp, small_mols = self._get_components(
            stateA, stateB
        )

        # Assign partial charges now to avoid any discrepancies later
        self._assign_partial_charges(settings["charge_settings"], small_mols)

        (
            stateA_system, stateA_topology, stateA_positions,
            stateB_system, stateB_topology, stateB_positions,
            system_mappings
        ) = self._get_omm_objects(
            stateA=stateA,
            stateB=stateB,
            mapping=mapping,
            settings=settings,
            protein_component=protein_comp,
            solvent_component=solvent_comp,
            small_mols=small_mols
        )

        # Get the hybrid factory & system
        hybrid_factory, hybrid_system = self._get_alchemical_system(
            stateA_system=stateA_system,
            stateA_positions=stateA_positions,
            stateA_topology=stateA_topology,
            stateB_system=stateB_system,
            stateB_positions=stateB_positions,
            stateB_topology=stateB_topology,
            system_mappings=system_mappings,
            alchemical_settings=settings["alchemical_settings"],
        )

        # Subselect system based on user inputs & write initial PDB
        selection_indices = self._subsample_topology(
            hybrid_topology=hybrid_factory.hybrid_topology,
            hybrid_positions=hybrid_factory.hybrid_positions,
            output_selection=settings["output_settings"].output_indices,
            output_filename=settings["output_settings"].output_structure,
            atom_classes=hybrid_factory._atom_classes,
        )

        # Serialize things
        # OpenMM System
        system_outfile = self.shared_basepath / "hybrid_system.xml.bz2"
        serialize(hybrid_system, system_outfile)

        # Positions
        positions_outfile = self.shared_basepath / "hybrid_positions.npy"
        npy_positions = from_openmm(hybrid_factory.hybrid_positions).to("nanometer").m
        np.save(positions_outfile, npy_positions)

        unit_results_dict = {
            "system": system_outfile,
            "positions": positions_outfile,
            "pdb_structure": self.shared_basepath / settings["output_settings"].output_structure,
            "selection_indices": selection_indices,
        }

        if dry:
            unit_results_dict |= {
                # Adding unserialized objects so we can directly use them
                # to chain units in tests
                "hybrid_factory": hybrid_factory,
                "hybrid_system": hybrid_system,
                "hybrid_positions": hybrid_factory.hybrid_positions,
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
            **outputs,
        }


class HybridTopologyMultiStateSimulationUnit(gufe.ProtocolUnit, HybridTopologyUnitMixin):

    @staticmethod
    def _check_restart(
        settings: dict[str, SettingsBaseModel],
        shared_path: pathlib.Path
    ):
        """
        Check if we are doing a restart.

        Parameters
        ----------
        settings : dict[str, SettingsBaseModel]
          The settings for this transformation
        shared_path : pathlib.Path
          The shared directory where we should be looking for existing files.

        Notes
        -----
        For now this just checks if the netcdf files are present in the
        shared directory but in the future this may expand depending on
        how warehouse works.
        """
        trajectory = shared_path / settings["output_settings"].output_filename
        checkpoint = shared_path / settings["output_settings"].checkpoint_storage_filename

        if trajectory.is_file() and checkpoint.is_file():
            return True
        
        return False

    @staticmethod
    def _get_integrator(
        integrator_settings: IntegratorSettings,
        simulation_settings: MultiStateSimulationSettings,
        system: openmm.System
    ) -> openmmtools.mcmc.LangevinDynamicsMove:
        """
        Get and validate the integrator

        Parameters
        ----------
        integrator_settings : IntegratorSettings
          Settings controlling the Langevin integrator.
        simulation_settings : MultiStateSimulationSettings
          Settings controlling the simulation.
        system : openmm.System
          The OpenMM System.

        Returns
        -------
        integrator : openmmtools.mcmc.LangevinDynamicsMove
          The LangevinDynamicsMove integrator.

        Raises
        ------
        ValueError
          If there are virtual sites in the system, but velocities
          are not being reassigned after every MCMC move.
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
                        "reassignments are unstable with MCMC integrators."
                    )
                    raise ValueError(errmsg)

        return integrator

    @staticmethod
    def _get_reporter(
        storage_path: pathlib.Path,
        selection_indices: npt.NDArray,
        output_settings: MultiStateOutputSettings,
        simulation_settings: MultiStateSimulationSettings,
    ) -> multistate.MultiStateReporter:
        """
        Get the multistate reporter.

        Parameters
        ----------
        storage_path : pathlib.Path
          Path to the directory where files should be written.
        selection_indices : npt.NDArray
          The set of system indices to report positions & velocities for.
        output_settings : MultiStateOutputSettings
          Settings defining how outputs should be written.
        simulation_settings : MultiStateSimulationSettings
          Settings defining out the simulation should be run.

        Notes
        -----
        All this does is create the reporter, it works for both
        new reporters and if we are doing a restart.
        """
        # Define the trajectory & checkpoint files
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
        positions: openmm.unit.Quantity,
        lambdas: _rfe_utils.lambdaprotocol.LambdaProtocol,
        integrator: openmmtools.mcmc.MCMCMove,
        reporter: multistate.MultiStateReporter,
        simulation_settings: MultiStateSimulationSettings,
        thermo_settings: ThermoSettings,
        alchem_settings: AlchemicalSettings,
        platform: openmm.Platform,
        restart: bool,
        dry: bool,
    ) -> multistate.MultiStateSampler:
        """
        Get the MultiStateSampler.

        Parameters
        ----------
        system : openmm.System
          The OpenMM System to simulate.
        positions : openmm.unit.Quantity
          The positions of the OpenMM System.
        lambdas : LambdaProtocol
          The lambda protocol to sample along.
        integrator : openmmtools.mcmc.MCMCMove
          The integrator to use.
        reporter : multistate.MultiStateReporter
          The reporter to attach to the sampler.
        simulation_settings : MultiStateSimulationSettings
          The simulation control settings.
        thermo_settings : ThermoSettings
          The thermodynamic control settings.
        alchem_settings : AlchemicalSettings
          The alchemical transformation settings.
        platform : openmm.Platform
          The compute platform to use.
        restart : bool
          ``True`` if we are doing a simulation restart.
        dry : bool
          Whether or not this is a dry run.

        Returns
        -------
        sampler : multistate.MultiStateSampler
          The requested sampler.
        """
        _SAMPLERS = {
            "repex" : _rfe_utils.multistate.HybridRepexSampler,
            "sams": _rfe_utils.multistate.HybridSAMSSampler,
            "independent": _rfe_utils.multistate.HybridMultiStateSampler,
        }

        sampler_method = simulation_settings.sampler_method.lower()

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
            ) / steps_per_iteration
        )

        # convert early_termination_target_error from kcal/mol to kT
        early_termination_target_error = (
            settings_validation.convert_target_error_from_kcal_per_mole_to_kT(
                thermo_settings.temperature,
                simulation_settings.early_termination_target_error,
            )
        )

        sampler_kwargs = {
            "mcmc_moves": integrator,
            "hybrid_system": system,
            "hybrid_positions": positions,
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
                "replica_mixing_scheme": "swap-all"
            }

        # Restarting doesn't need any setup, we just rebuild from storage.
        if restart:
            sampler = _SAMPLERS[sampler_method].from_storage(reporter)
        else:
            sampler = _SAMPLERS[sampler_method](**sampler_kwargs)

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
        simulation_settings : MultiStateSimulationSettings,
        integrator_settings : IntegratorSettings,
        output_settings : MultiStateOutputSettings,
        dry: bool,
    ):
        """
        Run the simulation.

        Parameters
        ----------
        sampler : multistate.MultiStateSampler.
          The sampler associated with the simulation to run.
        reporter : multistate.MultiStateReporter
          The reporter associated with the sampler.
        simulation_settings : MultiStateSimulationSettings
          Simulation control settings.
        integrator_settings : IntegratorSettings
          Integrator control settings.
        output_settings : MultiStateOutputSettings
          Simulation output control settings.
        dry : bool
          Whether or not to dry run the simulation.
        """
        # Get the relevant simulation steps
        mc_steps = settings_validation.convert_steps_per_iteration(
            simulation_settings=simulation_settings,
            integrator_settings=integrator_settings,
        )

        equil_steps = settings_validation.get_simsteps(
            sim_length=simulation_settings.equilibration_length,
            timestep=integrator_settings.timestep,
            mc_steps=mc_steps,
        )
        prod_steps = settings_validation.get_simsteps(
            sim_length=simulation_settings.production_length,
            timestep=integrator_settings.timestep,
            mc_steps=mc_steps,
        )

        if not dry:  # pragma: no-cover
            # No productions steps have been taken, so start from scratch
            if sampler._iteration == 0:
                # minimize
                if self.verbose:
                    self.logger.info("minimizing systems")

                sampler.minimize(max_iterations=simulation_settings.minimization_steps)

                # equilibrate
                if self.verbose:
                    self.logger.info("equilibrating systems")

                sampler.equilibrate(int(equil_steps / mc_steps))

            # At this point we are ready for production
            if self.verbose:
                self.logger.info("running production phase")

            # We use `run` so that we're limited by the number of iterations
            # we passed when we built the sampler.
            # TODO: I'm being extra prudent by passing in n_iterations here - remove?
            sampler.run(n_iterations=int(prod_steps / mc_steps)-sampler._iteration)

            if self.verbose:
                self.logger.info("production phase complete")
        else:
            # We ran a dry simulation
            # close reporter when you're done, prevent file handle clashes
            reporter.close()

            # TODO: review this is likely no longer necessary
            # clean up the reporter file
            fns = [
                self.shared_basepath / output_settings.output_filename,
                self.shared_basepath / output_settings.checkpoint_storage_filename,
            ]
            for fn in fns:
                os.remove(fn)

    def run(
        self,
        *,
        system: openmm.System,
        positions: openmm.unit.Quantity,
        selection_indices: npt.NDArray,
        dry: bool = False,
        verbose: bool = True,
        scratch_basepath: pathlib.Path | None = None,
        shared_basepath: pathlib.Path | None = None
    ) -> dict[str, Any]:
        """Run the free energy calculation using a multistate sampler.

        Parameters
        ----------
        system : openmm.System
          The System to simulate.
        positions : openmm.unit.Quantity
          The positions of the System.
        selection_indices : npt.NDArray
          Indices of the System particles to write to file.
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

        Raises
        ------
        error
          Exception if anything failed
        """
        # Prepare paths & verbosity
        self._prepare(verbose, scratch_basepath, shared_basepath)

        # Get the settings
        settings = self._get_settings(self._inputs["protocol"].settings)

        # Check for a restart
        self.restart = self._check_restart(
            settings=settings,
            shared_path=self.shared_basepath
        )

        # Get the lambda schedule
        # TODO - this should be better exposed to users
        lambdas = _rfe_utils.lambdaprotocol.LambdaProtocol(
            functions=settings["lambda_settings"].lambda_functions,
            windows=settings["lambda_settings"].lambda_windows
        )

        # Get the compute platform
        restrict_cpu = settings["forcefield_settings"].nonbonded_method.lower() == "nocutoff"
        platform = omm_compute.get_openmm_platform(
            platform_name=settings["engine_settings"].compute_platform,
            gpu_device_index=settings["engine_settings"].gpu_device_index,
            restrict_cpu_count=restrict_cpu,
        )


        try:
            # Get the integrator
            integrator = self._get_integrator(
                integrator_settings=settings["integrator_settings"],
                simulation_settings=settings["simulation_settings"],
                system=system
            )

            # Get the reporter
            reporter = self._get_reporter(
                storage_path=self.shared_basepath,
                selection_indices=selection_indices,
                output_settings=settings["output_settings"],
                simulation_settings=settings["simulation_settings"],
            )

            # Get the sampler
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
                restart=self.restart,
                dry=dry
            )

            # Run the simulation
            self._run_simulation(
                sampler=sampler,
                reporter=reporter,
                simulation_settings=settings["simulation_settings"],
                integrator_settings=settings["integrator_settings"],
                output_settings=settings["output_settings"],
                dry=dry,
            )
        finally:
            # close reporter when you're done, prevent
            # file handle clashes
            reporter.close()

            # clear GPU contexts
            # TODO: use cache.empty() calls when openmmtools #690 is resolved
            # replace with above
            for context in list(sampler.energy_context_cache._lru._data.keys()):
                del sampler.energy_context_cache._lru._data[context]
            for context in list(sampler.sampler_context_cache._lru._data.keys()):
                del sampler.sampler_context_cache._lru._data[context]
            # cautiously clear out the global context cache too
            for context in list(openmmtools.cache.global_context_cache._lru._data.keys()):
                del openmmtools.cache.global_context_cache._lru._data[context]

            del sampler.sampler_context_cache, sampler.energy_context_cache

            if not dry:
                del integrator, sampler

        if not dry:  # pragma: no-cover
            return {
                "nc": self.shared_basepath / settings["output_settings"].output_filename,
                "checkpoint": self.shared_basepath / settings["output_settings"].checkpoint_storage_filename,
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

        # Get the relevant inputs
        system = deserialize(setup_results.outputs["system"])
        positions = to_openmm(np.load(setup_results.outputs["positions"]) * offunit.nm)
        selection_indices = setup_results.outputs["selection_indices"]

        # Run the unit
        outputs = self.run(
            system=system,
            positions=positions,
            selection_indices=selection_indices,
            scratch_basepath=ctx.scratch,
            shared_basepath=ctx.shared
        )

        return {
            "repeat_id": self._inputs["repeat_id"],
            "generation": self._inputs["generation"],
            **outputs,
        }


class HybridTopologyMultiStateAnalysisUnit(gufe.ProtocolUnit, HybridTopologyUnitMixin):

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

    @staticmethod
    def _structural_analysis(
        pdb_file: pathlib.Path,
        trj_file: pathlib.Path,
        output_directory : pathlib.Path,
        dry: bool,
    ) -> dict[str, str | pathlib.Path]:
        """
        Run structural analysis using ``openfe-analysis``.

        Parameters
        ----------
        pdb_file : pathlib.Path
          Path to the PDB file.
        trj_filen : pathlib.Path
          Path to the trajectory file.
        output_directory : pathlib.Path
          The output directory where plots and the data NPZ file
          will be stored.
        dry : bool
          Whether or not we are running a dry run.

        Returns
        -------
        dict[str, str]
          Dictionary containing either the path to the NPZ
          file with the structural data, or the analysis error.

        Notes
        -----
        Don't put energy analysis here as it uses the MultiStateReporter,
        the structural analysis requires the file handle to be closed.
        """
        from openfe_analysis import rmsd

        try:
            data = rmsd.gather_rms_data(pdb_file, trj_file)
        # TODO: eventually change this to more specific exception types
        except Exception as e:
            return {"structural_analysis_error": str(e)}

        # Generate relevant plots if not a dry run
        if not dry:
            if d := data["protein_2D_RMSD"]:
                fig = plotting.plot_2D_rmsd(d)
                fig.savefig(output_directory / "protein_2D_RMSD.png")
                plt.close(fig)
                f2 = plotting.plot_ligand_COM_drift(
                    data["time(ps)"],
                    data["ligand_wander"]
                )
                f2.savefig(output_directory / "ligand_COM_drift.png")
                plt.close(f2)
    
            f3 = plotting.plot_ligand_RMSD(data["time(ps)"], data["ligand_RMSD"])
            f3.savefig(output_directory / "ligand_RMSD.png")
            plt.close(f3)

        # Write out an NPZ with all the relevant analysis data
        npz_file = output_directory / "structural_analysis.npz"
        np.savez_compressed(
            npz_file,
            protein_RMSD=np.asarray(data["protein_RMSD"], dtype=np.float32),
            ligand_RMSD=np.asarray(data["ligand_RMSD"], dtype=np.float32),
            ligand_COM_drift=np.asarray(data["ligand_wander"], dtype=np.float32),
            protein_2D_RMSD=np.asarray(data["protein_2D_RMSD"], dtype=np.float32),
            time_ps=np.asarray(data["time(ps)"], dtype=np.float32),
        )

        return {"structural_analysis": npz_file}

    def run(
        self,
        *,
        pdb_file: pathlib.Path,
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
        pdb_file : pathlib.Path
          Path to the PDB file representing the subsampled structure.
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

        Raises
        ------
        error
          Exception if anything failed
        """
        # Prepare paths & verbosity
        self._prepare(verbose, scratch_basepath, shared_basepath)

        # Get the settings
        settings = self._get_settings(self._inputs["protocol"].settings)

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

        # Structural analysis
        if verbose:
            self.logger.info("Analyzing structural outputs")

        structural_analysis = self._structural_analysis(
            pdb_file=pdb_file,
            trj_file=trajectory,
            output_directory=self.shared_basepath,
            dry=dry,
        )

        # Return relevant things
        outputs = energy_analysis | structural_analysis
        return outputs

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
        trajectory = simulation_results.outputs["nc"]
        checkpoint = simulation_results.outputs["checkpoint"]

        outputs = self.run(
            pdb_file=pdb_file,
            trajectory=trajectory,
            checkpoint=checkpoint,
            scratch_basepath=ctx.scratch,
            shared_basepath=ctx.shared
        )

        return {
            "repeat_id": self._inputs["repeat_id"],
            "generation": self._inputs["generation"],
            # We include various other outputs here to make
            # things easier when gathering.
            "pdb_structure": pdb_file,
            "trajectory": trajectory,
            "checkpoint": checkpoint,
            "selection_indices": selection_indices,
            **outputs,
        }
