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


class RelativeHybridTopologyProtocolSettings(Settings):
    protocol_repeats: int
    """
    The number of completely independent repeats of the entire sampling
    process. The mean of the repeats defines the final estimate of FE
    difference, while the variance between repeats is used as the uncertainty.
    """

    @field_validator("protocol_repeats")
    def must_be_positive(cls, v):
        if v <= 0:
            errmsg = f"protocol_repeats must be a positive value, got {v}."
            raise ValueError(errmsg)
        return v

    # Inherited things

    forcefield_settings: OpenMMSystemGeneratorFFSettings
    """Parameters to set up the force field with OpenMM Force Fields."""
    thermo_settings: ThermoSettings
    """Settings for thermodynamic parameters."""

    # Things for creating the systems
    solvation_settings: OpenMMSolvationSettings
    """Settings for solvating the system."""
    partial_charge_settings: OpenFFPartialChargeSettings
    """Settings for assigning partial charges to small molecules."""

    # Alchemical settings
    lambda_settings: LambdaSettings
    """
    Lambda protocol settings including lambda windows and lambda functions.
    """
    alchemical_settings: AlchemicalSettings
    """
    Alchemical protocol settings including soft core scaling.
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


class RBFEHTopProtocolSettings(Settings):
    """
    Configuration object for ``RBFEHTopProtocol``.

    See Also
    --------
    openfe.protocols.openmm_rfe.RBFEHTopProtocol
    """

    protocol_repeats: int
    """
    The number of completely independent repeats of the entire sampling
    process. The mean of the repeats defines the final estimate of FE
    difference, while the variance between repeats is used as the uncertainty.
    """

    @field_validator("protocol_repeats")
    def must_be_positive(cls, v):
        if v <= 0:
            errmsg = f"protocol_repeats must be a positive value, got {v}."
            raise ValueError(errmsg)
        return v

    # Inherited things
    forcefield_settings: OpenMMSystemGeneratorFFSettings
    """Parameters to set up the force field with OpenMM Force Fields."""
    thermo_settings: ThermoSettings
    """Settings for thermodynamic parameters."""
    partial_charge_settings: OpenFFPartialChargeSettings
    """Settings for assigning partial charges to small molecules."""

    # Alchemical settings
    alchemical_settings: AlchemicalSettings
    """
    Alchemical protocol settings including soft core scaling.
    """

    # Things for creating the systems
    solvent_solvation_settings: OpenMMSolvationSettings
    """Settings for solvating the solvent leg system."""
    complex_solvation_settings: OpenMMSolvationSettings
    """Settings for solvating the complex leg system."""

    # Lambda schedule settings
    solvent_lambda_settings: LambdaSettings
    """
    Lambda protocol settings including lambda windows and lambda functions
    for the solvent leg.
    """
    complex_lambda_settings: LambdaSettings
    """
    Lambda protocol settings including lambda windows and lambda functions
    for the complex leg.
    """

    # Simulation settings
    solvent_simulation_settings: MultiStateSimulationSettings
    """
    Settings for the solvent leg alchemical sampler.
    """
    complex_simulation_settings: MultiStateSimulationSettings
    """
    Settings for the complex leg alchemical sampler.
    """

    # MD Engine things
    engine_settings: OpenMMEngineSettings
    """Settings specific to the OpenMM engine such as the compute platform."""

    # Sampling State defining things
    solvent_integrator_settings: IntegratorSettings
    """Settings for the solvent leg integrator such as timestep and barostat settings."""
    complex_integrator_settings: IntegratorSettings
    """Settings for the complex leg integrator such as timestep and barostat settings."""

    solvent_output_settings: MultiStateOutputSettings
    """
    Solvent leg simulation output control settings.
    """
    complex_output_settings: MultiStateOutputSettings
    """
    Complex leg simulation output control settings.
    """


class RHFEHTopProtocolSettings(SettingsBaseModel):
    """
    Configuration object for ``RHFEHTopProtocol``.

    Note
    ----
    This subclasses ``SettingsBaseModel`` rather than ``Settings`` since the
    solvent and vacuum legs require different ``forcefield_settings``
    (the vacuum leg must use a ``nocutoff`` nonbonded method), so there is no
    single shared ``forcefield_settings`` field to satisfy ``Settings``.

    See Also
    --------
    openfe.protocols.openmm_rfe.RHFEHTopProtocol
    """

    protocol_repeats: int
    """
    The number of completely independent repeats of the entire sampling
    process. The mean of the repeats defines the final estimate of FE
    difference, while the variance between repeats is used as the uncertainty.
    """

    @field_validator("protocol_repeats")
    def must_be_positive(cls, v):
        if v <= 0:
            errmsg = f"protocol_repeats must be a positive value, got {v}."
            raise ValueError(errmsg)
        return v

    # Inherited things
    solvent_forcefield_settings: OpenMMSystemGeneratorFFSettings
    """Parameters to set up the force field with OpenMM Force Fields for the solvent leg."""
    vacuum_forcefield_settings: OpenMMSystemGeneratorFFSettings
    """
    Parameters to set up the force field with OpenMM Force Fields for the
    vacuum leg. Must use a ``nonbonded_method`` of ``nocutoff``.
    """
    thermo_settings: ThermoSettings
    """Settings for thermodynamic parameters."""
    partial_charge_settings: OpenFFPartialChargeSettings
    """Settings for assigning partial charges to small molecules."""

    # Alchemical settings
    alchemical_settings: AlchemicalSettings
    """
    Alchemical protocol settings including soft core scaling.
    """

    # Things for creating the systems
    solvation_settings: OpenMMSolvationSettings
    """Settings for solvating the solvent leg system. Ignored by the vacuum leg."""

    # Lambda schedule settings
    solvent_lambda_settings: LambdaSettings
    """
    Lambda protocol settings including lambda windows and lambda functions
    for the solvent leg.
    """
    vacuum_lambda_settings: LambdaSettings
    """
    Lambda protocol settings including lambda windows and lambda functions
    for the vacuum leg.
    """

    # Simulation settings
    solvent_simulation_settings: MultiStateSimulationSettings
    """
    Settings for the solvent leg alchemical sampler.
    """
    vacuum_simulation_settings: MultiStateSimulationSettings
    """
    Settings for the vacuum leg alchemical sampler.
    """

    # MD Engine things
    solvent_engine_settings: OpenMMEngineSettings
    """Settings specific to the OpenMM engine for the solvent leg, such as the compute platform."""
    vacuum_engine_settings: OpenMMEngineSettings
    """Settings specific to the OpenMM engine for the vacuum leg, such as the compute platform."""

    # Sampling State defining things
    solvent_integrator_settings: IntegratorSettings
    """Settings for the solvent leg integrator such as timestep and barostat settings."""
    vacuum_integrator_settings: IntegratorSettings
    """Settings for the vacuum leg integrator such as timestep settings."""

    solvent_output_settings: MultiStateOutputSettings
    """
    Solvent leg simulation output control settings.
    """
    vacuum_output_settings: MultiStateOutputSettings
    """
    Vacuum leg simulation output control settings.
    """
