# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""Equilibrium Relative Free Energy Protocol input settings.

This module implements the necessary settings necessary to run absolute free
energies using :class:`openfe.protocols.openmm_rfe.equil_rfe_methods.py`

"""
from __future__ import annotations

from typing import Optional
from pydantic import validator
from openff.units import unit
import os

from gufe.settings import (
    Settings,
    SettingsBaseModel,
    OpenMMSystemGeneratorFFSettings,
    ThermoSettings,
)
from openfe.protocols.openmm_utils.omm_settings import (
    SystemSettings, SolvationSettings, AlchemicalSamplerSettings,
    OpenMMEngineSettings, IntegratorSettings, SimulationSettings
)


class AlchemicalSettings(SettingsBaseModel):
    class Config:
        extra = 'ignore'

    """Settings for the alchemical protocol

    This describes the lambda schedule and the creation of the
    hybrid system.
    """
    # Lambda settings
    lambda_functions = 'default'
    """
    Key of which switching functions to use for alchemical mutation.
    Default 'default'.
    """
    lambda_windows = 11
    """Number of lambda windows to calculate. Default 11."""
    unsampled_endstates = False
    """
    Whether to have extra unsampled endstate windows for long range
    correction. Default False.
    """

    # alchemical settings
    use_dispersion_correction = False
    """
    Whether to use dispersion correction in the hybrid topology state.
    Default False.
    """
    softcore_LJ_v2 = True
    """
    Whether to use the LJ softcore function as defined by
    Gapsys et al. JCTC 2012 Default True.
    """
    softcore_alpha = 0.85
    """Softcore alpha parameter. Default 0.85"""
    interpolate_old_and_new_14s = False
    """
    Whether to turn off interactions for new exceptions (not just 1,4s)
    at lambda 0 and old exceptions at lambda 1. If False they are present
    in the nonbonded force. Default False.
    """
    flatten_torsions = False
    """
    Whether to scale torsion terms involving unique atoms, such that at the
    endstate the torsion term is turned off/on depending on the state the
    unique atoms belong to. Default False.
    """


class RelativeHybridTopologyProtocolSettings(Settings):
    class Config:
        arbitrary_types_allowed = True

    # Inherited things

    forcefield_settings: OpenMMSystemGeneratorFFSettings
    """Parameters to set up the force field with OpenMM Force Fields."""
    thermo_settings: ThermoSettings
    """Settings for thermodynamic parameters."""

    # Things for creating the systems
    system_settings: SystemSettings
    """Simulation system settings including the long-range non-bonded method."""
    solvation_settings: SolvationSettings
    """Settings for solvating the system."""

    # Alchemical settings
    alchemical_settings: AlchemicalSettings
    """
    Alchemical protocol settings including lambda windows and soft core scaling.
    """
    alchemical_sampler_settings: AlchemicalSamplerSettings
    """
    Settings for sampling alchemical space, including the number of repeats.
    """

    # MD Engine things
    engine_settings: OpenMMEngineSettings
    """Settings specific to the OpenMM engine such as the compute platform."""

    # Sampling State defining things
    integrator_settings: IntegratorSettings
    """Settings for the integrator such as timestep and barostat settings."""

    # Simulation run settings
    simulation_settings: SimulationSettings
    """
    Simulation control settings, including simulation lengths and record-keeping.
    """
