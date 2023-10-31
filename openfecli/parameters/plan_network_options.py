# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""Pydantic models for the definition of advanced CLI options

"""
import click
from pydantic.v1 import BaseModel  # , ConfigDict
from plugcli.params import Option
from typing import Any, Optional
import yaml
import warnings


class MapperSelection(BaseModel):
    # model_config = ConfigDict(extra='allow', str_to_lower=True)
    class Config:
        extra = 'allow'
        anystr_lower = True

    method: str = 'LomapAtomMapper'
    settings: dict[str, Any] = {}


class NetworkSelection(BaseModel):
    # model_config = ConfigDict(extra='allow', str_to_lower=True)
    class Config:
        extra = 'allow'
        anystr_lower = True

    method: str = 'generate_minimal_spanning_network'
    settings: dict[str, Any] = {}


class CliOptions(BaseModel):
    # model_config = ConfigDict(extra='allow')
    class Config:
        extra = 'allow'

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


def load_yaml_planner_options(path: str, context) -> dict:
    """Load cli options from yaml file path and resolve these to objects

    Parameters
    ----------
    path : str
      path to the yaml file
    context
      unused

    Returns
    -------
    options : dict
      dict optionally containing 'mapper' and 'network' keys:
      'mapper' key holds a AtomMapper object.
      'network' key holds a curried network planner function, whose signature
      matches generate_minimum_spanning_network.
    """
    from openfe.setup.ligand_network_planning import (
        generate_radial_network,
        generate_minimal_spanning_network,
        generate_maximal_network,
        generate_minimal_redundant_network,
    )
    from openfe.setup import (
        LomapAtomMapper,
    )
    from functools import partial

    with open(path, 'r') as f:
        raw = f.read()

    opt = parse_yaml_planner_options(raw)

    choices = {}

    if opt.mapper:
        mapper_choices = {
            'lomap': LomapAtomMapper,
            'lomapatommapper': LomapAtomMapper,
        }

        try:
            cls = mapper_choices[opt.mapper.method]
        except KeyError:
            raise KeyError(f"Bad mapper choice: '{opt.mapper.method}'")

        choices['mapper'] = cls(**opt.mapper.settings)
    if opt.network:
        network_choices = {
            'generate_radial_network': generate_radial_network,
            'radial': generate_radial_network,
            'generate_minimal_spanning_network': generate_minimal_spanning_network,
            'mst': generate_minimal_spanning_network,
            'generate_minimal_redundant_network': generate_minimal_redundant_network,
            'generate_maximal_network': generate_maximal_network,
        }

        try:
            func = network_choices[opt.network.method]
        except KeyError:
            raise KeyError(f"Bad network algorithm choice: '{opt.network.method}'")

        choices['network'] = partial(func, **opt.network.settings)

    return choices


YAML_OPTIONS = Option(
    '-s', "--settings", "yaml_settings",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to planning settings yaml file.",
    getter=load_yaml_planner_options,
)
