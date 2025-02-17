# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Settings for adding restraints.
"""
from typing import Optional, Literal
from openff.units import unit
from openff.models.types import FloatQuantity, ArrayQuantity

from gufe.settings import (
    SettingsBaseModel,
)


from pydantic.v1 import validator


class BaseRestraintSettings(SettingsBaseModel):
    """
    Base class for RestraintSettings objects.
    """
    class Config:
        arbitrary_types_allowed = True
