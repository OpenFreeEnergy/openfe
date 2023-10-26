# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""Pydantic models for the definition of advanced CLI options

"""

from pydantic import BaseModel, ConfigDict
from typing import Any, Optional
import yaml
import warnings


class MapperSelection(BaseModel):
    model_config = ConfigDict(extra='allow')

    method: str = 'LomapAtomMapper'
    settings: Optional[dict[str, Any]] = None


class NetworkSelection(BaseModel):
    model_config = ConfigDict(extra='allow')

    method: str = 'generate_minimal_spanning_network'
    settings: Optional[dict[str, Any]] = None


class CliOptions(BaseModel):
    model_config = ConfigDict(extra='allow')

    mapper: Optional[MapperSelection] = None
    network: Optional[NetworkSelection] = None


def parse_yaml_planner_options(contents: str) -> CliOptions:
    """Parse and minimally validate a user provided yaml

    Parameters
    ----------
    contents : str
      raw yaml formatted input to parse

    Returns
    -------
    options : CliOptions
      will have keys for mapper and network topology choices

    Raises
    ------
    ValueError
      for any malformed inputs
    """
    raw = yaml.safe_load(contents)

    if False:
        # todo: warnings about extra fields we don't expect?
        expected = {'mapper', 'network'}
        for field in raw:
            if field in expected:
                continue
            warnings.warn(f"Ignoring unexpected section: '{field}'")

    return CliOptions(**raw)
