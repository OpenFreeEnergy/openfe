# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
AHFE Protocol Units --- :mod:`openfe.protocols.openmm_afe.ahfe_units`
=====================================================================

This module defines the ProtocolUnits for the
:class:`AbsoluteSolvationProtocol`.
"""

import logging

from openfe.protocols.openmm_afe.equil_afe_settings import (
    SettingsBaseModel,
)

from ..openmm_utils import system_validation
from .base_afe_units import (
    BaseAbsoluteMultiStateAnalysisUnit,
    BaseAbsoluteMultiStateSimulationUnit,
    BaseAbsoluteSetupUnit,
)

logger = logging.getLogger(__name__)


class VacuumComponentsMixin:
    def _get_components(self):
        """
        Get the relevant components for a vacuum transformation.

        Returns
        -------
        alchem_comps : dict[str, list[Component]]
          A list of alchemical components
        solv_comp : None
          For the gas phase transformation, None will always be returned
          for the solvent component of the chemical system.
        prot_comp : Optional[ProteinComponent]
          The protein component of the system, if it exists.
        small_mols : dict[Component, OpenFF Molecule]
          The openff Molecules to add to the system. This
          is equivalent to the alchemical components in stateA (since
          we only allow for disappearing ligands).
        """
        stateA = self._inputs["stateA"]
        alchem_comps = self._inputs["alchemical_components"]

        off_comps = {m: m.to_openff() for m in alchem_comps["stateA"]}

        _, prot_comp, _ = system_validation.get_components(stateA)

        # Notes:
        # 1. Our input state will contain a solvent, we ``None`` that out
        # since this is the gas phase unit.
        # 2. Our small molecules will always just be the alchemical components
        # (of stateA since we enforce only one disappearing ligand)
        return alchem_comps, None, prot_comp, off_comps


class VacuumSettingsMixin:
    def _get_settings(self) -> dict[str, SettingsBaseModel]:
        """
        Extract the relevant settings for a vacuum transformation.

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
            * equil_output_settings : MDOutputSettings
            * simulation_settings : SimulationSettings
            * output_settings: MultiStateOutputSettings
        """
        prot_settings = self._inputs["protocol"].settings  # type: ignore[attr-defined]

        settings = {}
        settings["forcefield_settings"] = prot_settings.vacuum_forcefield_settings
        settings["thermo_settings"] = prot_settings.thermo_settings
        settings["charge_settings"] = prot_settings.partial_charge_settings
        settings["solvation_settings"] = prot_settings.solvation_settings
        settings["alchemical_settings"] = prot_settings.alchemical_settings
        settings["lambda_settings"] = prot_settings.lambda_settings
        settings["engine_settings"] = prot_settings.vacuum_engine_settings
        settings["integrator_settings"] = prot_settings.integrator_settings
        settings["equil_simulation_settings"] = prot_settings.vacuum_equil_simulation_settings
        settings["equil_output_settings"] = prot_settings.vacuum_equil_output_settings
        settings["simulation_settings"] = prot_settings.vacuum_simulation_settings
        settings["output_settings"] = prot_settings.vacuum_output_settings

        return settings


class AHFEVacuumSetupUnit(VacuumComponentsMixin, VacuumSettingsMixin, BaseAbsoluteSetupUnit):
    """
    Setup unit for the vacuum phase of absolute hydration free energy
    transformations.
    """

    simtype = "vacuum"


class AHFEVacuumSimUnit(
    VacuumComponentsMixin, VacuumSettingsMixin, BaseAbsoluteMultiStateSimulationUnit
):
    """
    Multi-state simulation (e.g. multi replica methods like Hamiltonian
    replica exchange) unit for the vacuum phase of absolute hydration
    free energy transformations.
    """

    simtype = "vacuum"


class AHFEVacuumAnalysisUnit(VacuumSettingsMixin, BaseAbsoluteMultiStateAnalysisUnit):
    """
    Analysis unit for multi-state simulations with the vacuum phase
    of absolute hydration free energy transformations.
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
        prot_comp : Optional[ProteinComponent]
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
        # Similarly we don't need to check prot_comp since that's also
        # disallowed on create
        return alchem_comps, solv_comp, prot_comp, off_comps


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
            * equil_output_settings : MDOutputSettings
            * simulation_settings : MultiStateSimulationSettings
            * output_settings: MultiStateOutputSettings
        """
        prot_settings = self._inputs["protocol"].settings  # type: ignore[attr-defined]

        settings = {}
        settings["forcefield_settings"] = prot_settings.solvent_forcefield_settings
        settings["thermo_settings"] = prot_settings.thermo_settings
        settings["charge_settings"] = prot_settings.partial_charge_settings
        settings["solvation_settings"] = prot_settings.solvation_settings
        settings["alchemical_settings"] = prot_settings.alchemical_settings
        settings["lambda_settings"] = prot_settings.lambda_settings
        settings["engine_settings"] = prot_settings.solvent_engine_settings
        settings["integrator_settings"] = prot_settings.integrator_settings
        settings["equil_simulation_settings"] = prot_settings.solvent_equil_simulation_settings
        settings["equil_output_settings"] = prot_settings.solvent_equil_output_settings
        settings["simulation_settings"] = prot_settings.solvent_simulation_settings
        settings["output_settings"] = prot_settings.solvent_output_settings

        return settings


class AHFESolventSetupUnit(SolventComponentsMixin, SolventSettingsMixin, BaseAbsoluteSetupUnit):
    """
    Setup unit for the solvent phase of absolute hydration free energy
    transformations.
    """

    simtype = "solvent"


class AHFESolventSimUnit(
    SolventComponentsMixin, SolventSettingsMixin, BaseAbsoluteMultiStateSimulationUnit
):
    """
    Multi-state simulation (e.g. multi replica methods like Hamiltonian
    replica exchange) unit for the solvent phase of absolute hydration
    free energy transformations.
    """

    simtype = "solvent"


class AHFESolventAnalysisUnit(SolventSettingsMixin, BaseAbsoluteMultiStateAnalysisUnit):
    """
    Analysis unit for multi-state simulations with the solvent phase
    of absolute hydration free energy transformations.
    """

    simtype = "solvent"
