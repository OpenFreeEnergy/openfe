"""
Settings models for Protocols.

Protocols often permit extensive configuration that would be cumbersome to
configure in an ``__init__`` method and which may want to be independently
recorded and shared. Settings models are Pydantic models that support these use
cases. Each protocol should have an associated settings object; for a protocol
called ``FooProtocol``, the corresponding settings object is called
``FooProtocolSettings``.

Settings are generally broken up into multiple levels to avoid overwhelming
users with information and to allow settings models to be shared across
multiple protocols. Top level protocol settings models should inherit
from :class:`.Settings`, whereas all other settings models should inherit from
:class:`.SettingsBaseModel`.

"""

from .openmm_utils.omm_settings import (
    Settings,
    SettingsBaseModel,
    SystemSettings,
    SolvationSettings,
    AlchemicalSamplerSettings,
    OpenMMEngineSettings,
    IntegratorSettings,
    SimulationSettings,
    ThermoSettings,
    OpenMMSystemGeneratorFFSettings,
)

from .openmm_rfe.equil_rfe_settings import AlchemicalSettings

__all__ = [
    "Settings",
    "SettingsBaseModel",
    "SystemSettings",
    "SolvationSettings",
    "AlchemicalSamplerSettings",
    "OpenMMEngineSettings",
    "IntegratorSettings",
    "SimulationSettings",
    "ThermoSettings",
    "OpenMMSystemGeneratorFFSettings",
    "AlchemicalSettings",

]
