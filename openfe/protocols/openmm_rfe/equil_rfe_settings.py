# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""Equilibrium Relative Free Energy Protocol input settings.

This module implements the necessary settings necessary to run relative free
energies using :class:`openfe.protocols.openmm_rfe.equil_rfe_methods.py`

"""
from __future__ import annotations

import abc
from typing import Optional, Union
from pydantic import Extra, validator, BaseModel, PositiveFloat, Field
from openff.units import unit
from openfe.protocols.openmm_utils.omm_settings import (
    Settings, SettingsBaseModel, ThermoSettings,
    OpenMMSystemGeneratorFFSettings, SystemSettings,
    SolvationSettings, AlchemicalSamplerSettings,
    OpenMMEngineSettings, IntegratorSettings,
    SimulationSettings
)


class AlchemicalSettings(SettingsBaseModel):
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
    softcore_electrostatics = True
    """Whether to use softcore electrostatics. Default True."""
    softcore_alpha = 0.85
    """Softcore alpha parameter. Default 0.85"""
    softcore_electrostatics_alpha = 0.3
    """Softcore alpha parameter for electrostatics. Default 0.3"""
    softcore_sigma_Q = 1.0
    """
    Softcore sigma parameter for softcore electrostatics. Default 1.0.
    """
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

    # Things for creating the systems
    system_settings: SystemSettings
    solvation_settings: SolvationSettings

    # Alchemical settings
    alchemical_settings: AlchemicalSettings
    alchemical_sampler_settings: AlchemicalSamplerSettings

    # MD Engine things
    engine_settings: OpenMMEngineSettings

    # Sampling State defining things
    integrator_settings: IntegratorSettings

    # Simulation run settings
    simulation_settings: SimulationSettings
