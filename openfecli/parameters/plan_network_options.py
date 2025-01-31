# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""Pydantic models for the definition of advanced CLI options

"""
import click
from collections import namedtuple
try:
    # todo; once we're fully v2, we can use ConfigDict not nested class
    from pydantic.v1 import BaseModel  # , ConfigDict
except ImportError:
    from pydantic import BaseModel
from plugcli.params import Option
from typing import Any, Optional
import yaml
import warnings


PlanNetworkOptions = namedtuple('PlanNetworkOptions',
                                ['mapper', 'scorer',
                                 'ligand_network_planner', 'solvent'])


class MapperSelection(BaseModel):
    # model_config = ConfigDict(extra='allow', str_to_lower=True)
    class Config:
        extra = 'allow'
        anystr_lower = True

    method: Optional[str] = None
    settings: dict[str, Any] = {}


class NetworkSelection(BaseModel):
    # model_config = ConfigDict(extra='allow', str_to_lower=True)
    class Config:
        extra = 'allow'
        anystr_lower = True

    method: Optional[str] = None
    settings: dict[str, Any] = {}


class CliYaml(BaseModel):
    # model_config = ConfigDict(extra='allow')
    class Config:
        extra = 'allow'

    mapper: Optional[MapperSelection] = None
    network: Optional[NetworkSelection] = None


def parse_yaml_planner_options(contents: str) -> CliYaml:
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

    expected_fields = {'mapper', 'network'}
    present_fields = set(raw.keys())
    usable_fields = present_fields.intersection(expected_fields)
    ignored_fields = present_fields.difference(expected_fields)

    for field in ignored_fields:
        warnings.warn(f"Ignoring unexpected section: '{field}'")

    filtered = {k:raw[k] for k in usable_fields}

    return CliYaml(**filtered)


def load_yaml_planner_options(path: Optional[str], context) -> PlanNetworkOptions:
    """Load cli options from yaml file path and resolve these to objects

    Parameters
    ----------
    path : str
      path to the yaml file
    context
      unused

    Returns
    -------
    PlanNetworkOptions : namedtuple
      a namedtuple with fields 'mapper', 'scorer', 'network_planning_algorithm',
      and 'solvent' fields.
      these fields each hold appropriate objects ready for use
    """
    from gufe import SolventComponent
    from openfe.setup.ligand_network_planning import (
        generate_radial_network,
        generate_minimal_spanning_network,
        generate_maximal_network,
        generate_minimal_redundant_network,
        generate_lomap_network,
    )
    from openfe.setup import (
        LomapAtomMapper,
        KartografAtomMapper,
    )
    from openfe.setup.atom_mapping.lomap_scorers import (
        default_lomap_score,
    )
    from functools import partial

    if path is not None:
        with open(path, 'r') as f:
            raw = f.read()

        # convert raw yaml to normalised pydantic model
        opt = parse_yaml_planner_options(raw)
    else:
        opt = None

    # convert normalised inputs to objects
    if opt and opt.mapper:
        mapper_choices = {
            'lomap': LomapAtomMapper,
            'lomapatommapper': LomapAtomMapper,
            'kartograf': KartografAtomMapper,
            'kartografatommapper': KartografAtomMapper,
        }

        try:
            cls = mapper_choices[opt.mapper.method]
        except KeyError:
            raise KeyError(f"Bad mapper choice: '{opt.mapper.method}'")
        mapper_obj = cls(**opt.mapper.settings)
    else:
        mapper_obj = LomapAtomMapper(
            time=20,
            threed=True,
            max3d=1.0,
            element_change=True,
            shift=False
        )

    # todo: choice of scorer goes here
    mapping_scorer = default_lomap_score

    if opt and opt.network:
        network_choices = {
            'generate_radial_network': generate_radial_network,
            'radial': generate_radial_network,
            'generate_minimal_spanning_network': generate_minimal_spanning_network,
            'mst': generate_minimal_spanning_network,
            'generate_minimal_redundant_network': generate_minimal_redundant_network,
            'generate_maximal_network': generate_maximal_network,
            'generate_lomap_network': generate_lomap_network,
        }

        try:
            func = network_choices[opt.network.method]
        except KeyError:
            raise KeyError(f"Bad network algorithm choice: '{opt.network.method}'")

        ligand_network_planner = partial(func, **opt.network.settings)
    else:
        ligand_network_planner = generate_minimal_spanning_network

    # todo: choice of solvent goes here
    solvent = SolventComponent()

    return PlanNetworkOptions(
        mapper_obj,
        mapping_scorer,
        ligand_network_planner,
        solvent,
    )


_yaml_help = """\
Path to a YAML file specifying the atom mapper (`mapper:`) and/or network planning algorithm (`network:`) to use.

Supported atom mapper choices are:
    - `LomapAtomMapper`
    - `KartografAtomMapper`

Supported network planning algorithms include (but are not limited to):
    - `generate_minimal_spanning_tree`
    - `generate_minimal_redundant_network`
    - `generate_radial_network`
    - `generate_lomap_network`

The `settings:` allows for passing in any keyword arguments of the method's corresponding Python API.

For example:
::

  mapper:
    method: LomapAtomMapper
    settings:
      element_change: false

  network:
    method: generate_minimal_redundant_network
    settings:
      mst_num: 3
"""

YAML_OPTIONS = Option(
    '-s', "--settings", "yaml_settings",
    type=click.Path(exists=True, dir_okay=False),
    help=_yaml_help,
    getter=load_yaml_planner_options,
)
