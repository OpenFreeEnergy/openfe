# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""Equilibrium Relative Free Energy Protocol input settings.

This module implements the necessary settings necessary to run relative free
energies using :class:`openfe.protocols.openmm_rfe.equil_rfe_methods.py`

"""

from __future__ import annotations

from typing import Literal

from gufe.settings import (
    OpenMMSystemGeneratorFFSettings,
    Settings,
    SettingsBaseModel,
    ThermoSettings,
)
from gufe.settings.typing import NanometerQuantity
from openff.units import unit
from pydantic import ConfigDict, field_validator

from openfe.protocols.openmm_utils.omm_settings import (
    IntegratorSettings,
    MultiStateOutputSettings,
    MultiStateSimulationSettings,
    OpenFFPartialChargeSettings,
    OpenMMEngineSettings,
    OpenMMSolvationSettings,
)


class LambdaSettings(SettingsBaseModel):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    """Lambda schedule settings.

    Settings controlling the lambda schedule, these include the switching
    function type, and the number of windows.
    """
    lambda_functions: str = "default"
    """
    Key of which switching functions to use for alchemical mutation.
    Default 'default'.
    """
    lambda_windows: int = 11
    """Number of lambda windows to calculate. Default 11."""


class AlchemicalSettings(SettingsBaseModel):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    """Settings for the alchemical protocol

    This describes the creation of the hybrid system.
    """

    endstate_dispersion_correction: bool = False
    """
    Whether to have extra unsampled endstate windows for long range
    correction. Default False.
    """

    # alchemical settings
    use_dispersion_correction: bool = False
    """
    Whether to use dispersion correction in the hybrid topology state.
    Default False.
    """
    softcore_LJ: Literal["gapsys", "beutler"]
    """
    Whether to use the LJ softcore function as defined by Gapsys et al.
    JCTC 2012, or the one by Beutler et al. Chem. Phys. Lett. 1994.
    Default 'gapsys'.
    """
    softcore_alpha: float = 0.85
    """Softcore alpha parameter. Default 0.85"""
    turn_off_core_unique_exceptions: bool = True
    """
    Whether to turn off interactions for new exceptions (not just 1,4s)
    at lambda 0 and old exceptions at lambda 1 between unique atoms and core
    atoms. If False they are present in the nonbonded force. Default True.
    """
    explicit_charge_correction: bool = False
    """
    Whether to explicitly account for a charge difference during the
    alchemical transformation by transforming a water to a counterion
    of the opposite charge of the formal charge difference.

    Please note that this feature is currently in beta and poorly tested.

    Absolute charge changes greater than 1 are
    currently not supported.

    Default False.
    """
    explicit_charge_correction_cutoff: NanometerQuantity = 0.8 * unit.nanometer
    """
    The minimum distance from the system solutes from which an
    alchemical water can be chosen. Default 0.8 * unit.nanometer.
    """


class BaseHTopProtocolSettings(SettingsBaseModel):
    """
    Base configuration object for ``HTopProtocol`` and its subclasses.

    See Also
    --------
    openfe.protocols.openmm_rfe.HTopProtocol
    """

    protocol_repeats: int
    """
    The number of completely independent repeats of the entire sampling
    process. The mean of the repeats defines the final estimate of the ΔΔG
    difference, while the variance between repeats is used as the uncertainty.
    """

    @field_validator("protocol_repeats")
    def must_be_positive(cls, v):
        if v <= 0:
            errmsg = f"protocol_repeats must be a positive value, got {v}."
            raise ValueError(errmsg)
        return v

    thermo_settings: ThermoSettings
    """Settings for thermodynamic parameters."""

    partial_charge_settings: OpenFFPartialChargeSettings
    """Settings for assigning partial charges to small molecules."""

    # Alchemical settings
    alchemical_settings: AlchemicalSettings
    """
    Alchemical protocol settings including soft core scaling.
    """


class RelativeHybridTopologyProtocolSettings(BaseHTopProtocolSettings):
    forcefield_settings: OpenMMSystemGeneratorFFSettings
    """Parameters to set up the force field with OpenMM Force Fields."""

    solvation_settings: OpenMMSolvationSettings
    """Settings for solvating the system."""

    lambda_settings: LambdaSettings
    """
    Lambda protocol settings including lambda windows and lambda functions.
    """

    simulation_settings: MultiStateSimulationSettings
    """
    Settings for alchemical sampler.
    """

    # MD Engine things
    engine_settings: OpenMMEngineSettings
    """Settings specific to the OpenMM engine such as the compute platform."""

    # Sampling State defining things
    integrator_settings: IntegratorSettings
    """Settings for the integrator such as timestep and barostat settings."""

    output_settings: MultiStateOutputSettings
    """
    Simulation output control settings.
    """


class RBFEHTopProtocolSettings(BaseHTopProtocolSettings):
    """
    Configuration object for ``RBFEHTopProtocol``.

    See Also
    --------
    openfe.protocols.openmm_rfe.RBFEHTopProtocol
    """

    # Force field settings - only need one
    forcefield_settings: OpenMMSystemGeneratorFFSettings
    """Parameters to control assigning force field parameters using OpenMMForceFields."""

    # Lambda schedule settings
    solvent_lambda_settings: LambdaSettings
    """
    Lambda protocol settings defining the lambda schedule, including
    the number of lambda windows and scaling function for the solvent leg.
    """
    complex_lambda_settings: LambdaSettings
    """
    Lambda protocol settings defining the lambda schedule, including
    the number of lambda windows and scaling function for the complex leg.
    """

    # Things for creating the systems
    solvent_solvation_settings: OpenMMSolvationSettings
    """Settings for solvating the solvent leg system."""
    complex_solvation_settings: OpenMMSolvationSettings
    """Settings for solvating the complex leg system."""

    # Simulation control settings
    solvent_simulation_settings: MultiStateSimulationSettings
    """
    Settings for controlling the solvent leg alchemical simulation.
    """
    complex_simulation_settings: MultiStateSimulationSettings
    """
    Settings for controlling the complex leg alchemical simulation.
    """

    # MD Engine things
    engine_settings: OpenMMEngineSettings
    """Settings specific to the OpenMM MD engine such as what compute platform to use."""

    # Integrator control
    solvent_integrator_settings: IntegratorSettings
    """Settings for the solvent leg integrator such as timestep and barostat settings."""
    complex_integrator_settings: IntegratorSettings
    """Settings for the complex leg integrator such as timestep and barostat settings."""

    # Output control
    solvent_output_settings: MultiStateOutputSettings
    """
    Solvent leg simulation output (e.g. filenames) control settings.
    """
    complex_output_settings: MultiStateOutputSettings
    """
    Complex leg simulation output (e.g. filenames) control settings.
    """


class RHFEHTopProtocolSettings(BaseHTopProtocolSettings):
    """
    Configuration object for ``RHFEHTopProtocol``.

    See Also
    --------
    openfe.protocols.openmm_rfe.RHFEHTopProtocol
    """

    solvent_forcefield_settings: OpenMMSystemGeneratorFFSettings
    """Parameters to control assigning force field parameters using OpenMMForceFields for the solvent leg."""
    vacuum_forcefield_settings: OpenMMSystemGeneratorFFSettings
    """
    Parameters to control assigning the force field using OpenMMForceFields for the vacuum leg.
    Must use a ``nonbonded_method`` of ``nocutoff``.
    """

    solvation_settings: OpenMMSolvationSettings
    """Settings for solvating the solvent leg system. Ignored by the vacuum leg."""

    solvent_lambda_settings: LambdaSettings
    """
    Lambda protocol settings defining the lambda schedule, including
    the number of lambda windows and scaling function for the solvent leg.
    """
    vacuum_lambda_settings: LambdaSettings
    """
    Lambda protocol settings defining the lambda schedule, including
    the number of lambda windows and scaling function for the vacuum leg.
    """

    solvent_simulation_settings: MultiStateSimulationSettings
    """
    Settings for controlling the solvent leg alchemical simulation.
    """
    vacuum_simulation_settings: MultiStateSimulationSettings
    """
    Settings for controlling the vacuum leg alchemical simulation.
    """

    # Engine settings control hardware usage
    solvent_engine_settings: OpenMMEngineSettings
    """Settings specific to the OpenMM MD engine for the solvent leg, such as what compute platform to use."""
    vacuum_engine_settings: OpenMMEngineSettings
    """Settings specific to the OpenMM MD engine for the vacuum leg, such as what compute platform to use."""

    solvent_integrator_settings: IntegratorSettings
    """Settings for the solvent leg integrator such as timestep and barostat settings."""
    vacuum_integrator_settings: IntegratorSettings
    """Settings for the vacuum leg integrator such as timestep settings."""

    solvent_output_settings: MultiStateOutputSettings
    """
    Solvent leg simulation output (e.g. filenames) control settings.
    """
    vacuum_output_settings: MultiStateOutputSettings
    """
    Vacuum leg simulation output (e.g. filenames) control settings.
    """
