# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""ABFE Protocol Units --- :mod:`openfe.protocols.openmm_afe.abfe_units`
========================================================================
This module defines the ProtocolUnits for the
:class:`AbsoluteBindingProtocol`.
"""

import logging
import pathlib

import MDAnalysis as mda
import numpy as np
import numpy.typing as npt
from gufe import (
    SolventComponent,
)
from gufe.components import Component
from openff.units import Quantity
from openff.units.openmm import to_openmm
from openmm import System
from openmm import unit as ommunit
from openmm.app import Topology as omm_topology
from openmmtools.states import ThermodynamicState
from rdkit import Chem

from openfe.protocols.openmm_afe.equil_afe_settings import (
    BoreschRestraintSettings,
    SettingsBaseModel,
)
from openfe.protocols.openmm_utils import system_validation
from openfe.protocols.restraint_utils import geometry
from openfe.protocols.restraint_utils.geometry.boresch import BoreschRestraintGeometry
from openfe.protocols.restraint_utils.openmm import omm_restraints
from openfe.protocols.restraint_utils.openmm.omm_restraints import BoreschRestraint

from .base_afe_units import (
    BaseAbsoluteMultiStateAnalysisUnit,
    BaseAbsoluteMultiStateSimulationUnit,
    BaseAbsoluteSetupUnit,
)

logger = logging.getLogger(__name__)


class ComplexComponentsMixin:
    def _get_components(self):
        """
        Get the relevant components for a complex transformation.

        Returns
        -------
        alchem_comps : dict[str, Component]
          A dict of alchemical components
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
        off_comps = {m: m.to_openff() for m in small_mols}

        # We don't need to check that solv_comp is not None, otherwise
        # an error will have been raised when calling `validate_solvent`
        # in the Protocol's `_create`.
        # Similarly we don't need to check prot_comp
        return alchem_comps, solv_comp, prot_comp, off_comps


class ComplexSettingsMixin:
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
            * equil_output_settings : ABFEPreEquilOutputSettings
            * simulation_settings : SimulationSettings
            * output_settings: MultiStateOutputSettings
            * restraint_settings: BaseRestraintSettings
        """
        prot_settings = self._inputs["protocol"].settings

        settings = {}
        settings["forcefield_settings"] = prot_settings.forcefield_settings
        settings["thermo_settings"] = prot_settings.thermo_settings
        settings["charge_settings"] = prot_settings.partial_charge_settings
        settings["solvation_settings"] = prot_settings.complex_solvation_settings
        settings["alchemical_settings"] = prot_settings.alchemical_settings
        settings["lambda_settings"] = prot_settings.complex_lambda_settings
        settings["engine_settings"] = prot_settings.engine_settings
        settings["integrator_settings"] = prot_settings.integrator_settings
        settings["equil_simulation_settings"] = prot_settings.complex_equil_simulation_settings
        settings["equil_output_settings"] = prot_settings.complex_equil_output_settings
        settings["simulation_settings"] = prot_settings.complex_simulation_settings
        settings["output_settings"] = prot_settings.complex_output_settings
        settings["restraint_settings"] = prot_settings.restraint_settings

        return settings


class ABFEComplexSetupUnit(ComplexComponentsMixin, ComplexSettingsMixin, BaseAbsoluteSetupUnit):
    """
    Protocol Unit for the complex phase of an absolute binding free energy
    """

    simtype = "complex"

    @staticmethod
    def _get_mda_universe(
        topology: omm_topology,
        positions: ommunit.Quantity,
        trajectory: pathlib.Path | None,
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

        Returns
        -------
        mda.Universe
          An MDAnalysis Universe of the System.
        """
        from MDAnalysis.coordinates.memory import MemoryReader

        # If the trajectory file doesn't exist, then we use positions
        if trajectory is not None and trajectory.is_file():
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
    def _get_idxs_from_residxs(
        topology: omm_topology,
        residxs: list[int],
    ) -> list[int]:
        """
        Helper method to get the a list of atom indices which belong to a list
        of residues.

        Parameters
        ----------
        topology : openmm.app.Topology
          An OpenMM Topology that defines the System.
        residxs : list[int]
          A list of residue numbers who's atoms we should get atom indices.

        Returns
        -------
        atom_ids : list[int]
          A list of atom indices.

        TODO
        ----
        * Check how this works when we deal with virtual sites.
        """
        atom_ids = []

        for r in topology.residues():
            if r.index in residxs:
                atom_ids.extend([at.index for at in r.atoms()])

        return atom_ids

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
        temperature : openff.units.Quantity
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
        # Take the minimum of the two possible force constants to check against
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
        system: System,
        topology: omm_topology,
        positions: ommunit.Quantity,
        alchem_comps: dict[str, list[Component]],
        comp_resids: dict[Component, npt.NDArray],
        settings: dict[str, SettingsBaseModel],
    ) -> tuple[
        Quantity,
        System,
        geometry.HostGuestRestraintGeometry,
    ]:
        """
        Find and add restraints to the OpenMM System.

        Notes
        -----
        Currently, only Boresch-like restraints are supported.

        Parameters
        ----------
        system : openmm.System
          The System to add the restraint to.
        topology : openmm.app.Topology
          An OpenMM Topology that defines the System.
        positions: openmm.unit.Quantity
          The System's current positions.
          Used if a trajectory file isn't found.
        alchem_comps: dict[str, list[Component]]
          A dictionary with a list of alchemical components
          in both state A and B.
        comp_resids: dict[Component, npt.NDArray]
          A dictionary keyed by each Component in the System
          which contains arrays with the residue indices that is contained
          by that Component.
        settings : dict[str, SettingsBaseModel]
          A dictionary of settings that defines how to find and set
          the restraint.

        Returns
        -------
        correction : openff.units.Quantity
          The standard state correction for the restraint.
        system : openmm.System
          A copy of the System with the restraint added.
        rest_geom : geometry.HostGuestRestraintGeometry
          The restraint Geometry object.
        """
        if self.verbose:
            self.logger.info("Generating restraints")

        # Get the guest rdmol
        guest_rdmol = alchem_comps["stateA"][0].to_rdkit()

        # sanitize the rdmol if possible - warn if you can't
        err = Chem.SanitizeMol(guest_rdmol, catchErrors=True)

        if err:
            msg = "restraint generation: could not sanitize ligand rdmol"
            logger.warning(msg)

        # Get the guest idxs
        # concatenate a list of residue indexes for all alchemical components
        residxs = np.concatenate([comp_resids[key] for key in alchem_comps["stateA"]])

        # get the alchemicical atom ids
        guest_atom_ids = self._get_idxs_from_residxs(topology, residxs)

        # Now get the host idxs
        # We assume this is everything but the alchemical component
        # and the solvent.
        solv_comps = [c for c in comp_resids if isinstance(c, SolventComponent)]
        exclude_comps = [alchem_comps["stateA"]] + solv_comps
        residxs = np.concatenate([v for i, v in comp_resids.items() if i not in exclude_comps])

        host_atom_ids = self._get_idxs_from_residxs(topology, residxs)

        # Finally create an MDAnalysis Universe
        # We try to pass the equilibration production file path through
        # In some cases (debugging / dry runs) this won't be available
        # so we'll default to using input positions.
        univ = self._get_mda_universe(
            topology,
            positions,
            self.shared_basepath / settings["equil_output_settings"].production_trajectory_filename,
        )

        if isinstance(settings["restraint_settings"], BoreschRestraintSettings):
            rest_geom, restraint = self._get_boresch_restraint(
                univ,
                guest_rdmol,
                guest_atom_ids,
                host_atom_ids,
                settings["thermo_settings"].temperature,
                settings["restraint_settings"],
            )
        else:
            # TODO turn this into a direction for different restraint types supported?
            raise NotImplementedError("Other restraint types are not yet available")

        if self.verbose:
            self.logger.info(f"restraint geometry is: {rest_geom}")

        # We need a temporary thermodynamic state to add the restraint
        # & get the correction
        thermodynamic_state = ThermodynamicState(
            system,
            temperature=to_openmm(settings["thermo_settings"].temperature),
            pressure=to_openmm(settings["thermo_settings"].pressure),
        )

        # Add the force to the thermodynamic state
        restraint.add_force(
            thermodynamic_state,
            rest_geom,
            controlling_parameter_name="lambda_restraints",
        )
        # Get the standard state correction as a unit.Quantity
        correction = restraint.get_standard_state_correction(
            thermodynamic_state,
            rest_geom,
        )

        return (
            correction,
            # Remove the thermostat, otherwise you'll get an
            # Andersen thermostat by default!
            thermodynamic_state.get_system(remove_thermostat=True),
            rest_geom,
        )


class ABFEComplexSimUnit(
    ComplexComponentsMixin, ComplexSettingsMixin, BaseAbsoluteMultiStateSimulationUnit
):
    """
    ProtocolUnit for the vacuum simulation phase of an absolute hydration free energy
    """

    simtype = "vacuum"


class ABFEComplexAnalysisUnit(ComplexSettingsMixin, BaseAbsoluteMultiStateAnalysisUnit):
    """
    ProtocolUnit for the vacuum analysis phase of an absolute hydration free energy
    """

    simtype = "vacuum"


class SolventComponentsMixin:
    def _get_components(self):
        """
        Get the relevant components for a solvent transformation.

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
        off_comps = {m: m.to_openff() for m in alchem_comps["stateA"]}

        # We don't need to check that solv_comp is not None, otherwise
        # an error will have been raised when calling `validate_solvent`
        # in the Protocol's `_create`.
        # Similarly we don't need to check prot_comp just return None
        return alchem_comps, solv_comp, None, off_comps


class SolventSettingsMixin:
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
            * equil_output_settings : ABFEPreEquilOutputSettings
            * simulation_settings : MultiStateSimulationSettings
            * output_settings: MultiStateOutputSettings
        """
        prot_settings = self._inputs["protocol"].settings

        settings = {}
        settings["forcefield_settings"] = prot_settings.forcefield_settings
        settings["thermo_settings"] = prot_settings.thermo_settings
        settings["charge_settings"] = prot_settings.partial_charge_settings
        settings["solvation_settings"] = prot_settings.solvent_solvation_settings
        settings["alchemical_settings"] = prot_settings.alchemical_settings
        settings["lambda_settings"] = prot_settings.solvent_lambda_settings
        settings["engine_settings"] = prot_settings.engine_settings
        settings["integrator_settings"] = prot_settings.integrator_settings
        settings["equil_simulation_settings"] = prot_settings.solvent_equil_simulation_settings
        settings["equil_output_settings"] = prot_settings.solvent_equil_output_settings
        settings["simulation_settings"] = prot_settings.solvent_simulation_settings
        settings["output_settings"] = prot_settings.solvent_output_settings

        return settings


class ABFESolventSetupUnit(SolventComponentsMixin, SolventSettingsMixin, BaseAbsoluteSetupUnit):
    """
    ProtocolUnit for the solvent setup phase of an absolute binding free energy
    """

    simtype = "solvent"


class ABFESolventSimUnit(
    SolventComponentsMixin, SolventSettingsMixin, BaseAbsoluteMultiStateSimulationUnit
):
    """
    ProtocolUnit for the solvent simulation phase of an absolute binding free energy
    """

    simtype = "solvent"


class ABFESolventAnalysisUnit(SolventSettingsMixin, BaseAbsoluteMultiStateAnalysisUnit):
    """
    ProtocolUnit for the solvent analysis phase of an absolute binding free energy
    """

    simtype = "solvent"
