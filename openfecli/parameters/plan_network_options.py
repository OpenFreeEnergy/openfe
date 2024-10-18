# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""Pydantic models for the definition of advanced CLI options

"""
import click
import difflib
from collections import namedtuple
from gufe.settings import SettingsBaseModel
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
                                 'ligand_network_planner', 'solvent', 'protocol'])


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


class SolventSelection(BaseModel):
    class Config:
        extra = 'allow'
        anystr_lower = True

    method: Optional[str] = None
    settings: dict[str, Any] = {}


class ProtocolSelection(BaseModel):
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
    solvent: Optional[SolventSelection] = None
    protocol: Optional[ProtocolSelection] = None


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

    if False:
        # todo: warnings about extra fields we don't expect?
        expected = {'mapper', 'network'}
        for field in raw:
            if field in expected:
                continue
            warnings.warn(f"Ignoring unexpected section: '{field}'")

    return CliYaml(**raw)


def nearest_match(a: str, possible: list[str]) -> str:
    """figure out what *a* might have been meant from *possible*"""
    # todo: this is using a standard library approach, others are possible
    return max(
        possible,
        key=lambda x: difflib.SequenceMatcher(a=a, b=x).ratio()
    )


def apply_onto(settings: SettingsBaseModel, options: dict) -> None:
    """recursively apply things from options onto settings"""
    # this is pydantic v1, v2 has different name for this
    fields = list(settings.__fields__)

    for k, v in options.items():
        # print(f"doing k='{k}' v='{v}' on {settings.__class__}")
        if k not in fields:
            guess = nearest_match(k, fields)
            raise ValueError(f"Unknown field '{k}', "
                             f"did you mean '{guess}'?")

        thing = getattr(settings, k)
        if isinstance(thing, SettingsBaseModel):
            if not isinstance(v, dict):
                raise ValueError(f"must set sub-settings '{k}' to dict, "
                                 f"got: '{v}'")
            apply_onto(thing, v)
        else:
            # print(f'-> setting {k} to {v}')
            setattr(settings, k, v)


def resolve_protocol_choices(options: Optional[ProtocolSelection]):
    """Turn Protocol section into a fully formed Protocol

    Returns
    -------
    Optional[Protocol]

    Raises
    ------
    ValueError
      if an unsupported method name is input
    """
    if not options:
        return None

    # issue #644, make this selection not static
    allowed = {'RelativeHybridTopologyProtocol',
               # 'AbsoluteSolvationProtocol',
               # 'PlainMDProtocol',
               }
    if options.method.lower() == 'relativehybridtopologyprotocol':
        from openfe.protocols import openmm_rfe
        protocol = openmm_rfe.RelativeHybridTopologyProtocol
    # This wouldn't be reachable from any plan command, so leave out
    #elif options.method.lower() == 'absolutesolvationprotocol':
    #    from openfe.protocols import openmm_afe
    #    protocol = openmm_afe.AbsoluteSolvationProtocol
    #elif options.method.lower() == 'plainmdprotocol':
    #    from openfe.protocols import openmm_md
    #    protocol = openmm_md.PlainMDProtocol
    else:
        raise ValueError(f"Unsupported protocol method '{options.method}'. "
                         f"Supported methods are {','.join(allowed)}")

    settings = protocol.default_settings()
    # work through the fields in yaml input and apply these onto settings
    if options.settings:
        apply_onto(settings, options.settings)

    return protocol(settings)


def load_yaml_planner_options_from_cliyaml(opt: CliYaml) -> PlanNetworkOptions:
    from gufe import SolventComponent
    from openfe.setup.ligand_network_planning import (
        generate_radial_network,
        generate_minimal_spanning_network,
        generate_maximal_network,
        generate_minimal_redundant_network,
    )
    from openfe.setup import (
        LomapAtomMapper,
        KartografAtomMapper,
    )
    from openfe.setup.atom_mapping.lomap_scorers import (
        default_lomap_score,
    )
    from functools import partial

    # convert normalised inputs to objects
    if opt.mapper:
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

    if opt.network:
        network_choices = {
            'generate_radial_network': generate_radial_network,
            'generate_minimal_spanning_network': generate_minimal_spanning_network,
            'generate_minimal_redundant_network': generate_minimal_redundant_network,
            'generate_maximal_network': generate_maximal_network,
        }

        try:
            func = network_choices[opt.network.method]
        except KeyError:
            raise ValueError(f"Bad network algorithm choice: '{opt.network.method}'. "
                             f"Available options are {', '.join(network_choices.keys())}")

        ligand_network_planner = partial(func, **opt.network.settings)
    else:
        ligand_network_planner = generate_minimal_spanning_network

    # todo: choice of solvent goes here
    solvent = SolventComponent()

    if opt.protocol:
        protocol = resolve_protocol_choices(opt.protocol)
    else:
        protocol = None

    return PlanNetworkOptions(
        mapper=mapper_obj,
        scorer=mapping_scorer,
        ligand_network_planner=ligand_network_planner,
        solvent=solvent,
        protocol=protocol,
    )


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
    if path is not None:
        with open(path, 'r') as f:
            raw = f.read()

        # convert raw yaml to normalised pydantic model
        opt = parse_yaml_planner_options(raw)
    else:
        opt = CliYaml()

    return load_yaml_planner_options_from_cliyaml(opt)


_yaml_help = """\
Path to planning settings yaml file

Currently it can contain sections for customising the atom mapper, network
planning algorithm, and protocol. These are addressed using a ``mapper:``,
``network:`` or ``protocol:`` key in the yaml file.  The algorithm to be used
for these sections is then specified by the ``method:`` key. Finally, a
``settings:`` key can be given to customise the algorithm, with allowable
options corresponding to the keyword arguments of the Python API for these
algorithms.

For choosing mappers, either the ``LomapAtomMapper`` or ``KartografAtomMapper``
are allowed choices. For the network planning algorithm either the
``generate_minimal_spanning_tree``, ``generate_radial_network`` or
``generate_minimal_redundant_network`` options are allowed.

For example, this is a valid settings yaml file to specify that
the Lomap atom mapper should be used forbidding element changes,
while the generate_minimal_redundant_network function used to plan the network
::

  mapper:
    method: LomapAtomMapper
    settings:
      element_change: false

  network:
    method: generate_minimal_redundant_network
    settings:
      mst_num: 3

The Settings of a Protocol can also be customised in this settings yaml file.
To do this, the nested variable names from the Python API are directly converted
to the nested yaml format. 
"""


YAML_OPTIONS = Option(
    '-s', "--settings", "yaml_settings",
    type=click.Path(exists=True, dir_okay=False),
    help=_yaml_help,
    getter=load_yaml_planner_options,
)
