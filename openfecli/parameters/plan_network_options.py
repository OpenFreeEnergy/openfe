# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""Pydantic models for the definition of advanced CLI options

"""
import click
from collections import namedtuple
from pydantic import BaseModel, ConfigDict
from plugcli.params import Option
from typing import Any, Optional
import yaml
import warnings


PlanNetworkOptions = namedtuple(
    'PlanNetworkOptions',
    [
        'mapper',
        'scorer',
        'ligand_network_planner',
        'solvent',
        'partial_charge',
    ]
)


class MapperSelection(BaseModel):
    model_config = ConfigDict(extra='allow', str_to_lower=True)

    method: Optional[str] = None
    settings: dict[str, Any] = {}


class NetworkSelection(BaseModel):
    model_config = ConfigDict(extra='allow', str_to_lower=True)

    method: Optional[str] = None
    settings: dict[str, Any] = {}


class PartialChargeSelection(BaseModel):
    model_config = ConfigDict(extra='allow', str_to_lower=True)

    method: Optional[str] = 'am1bcc'
    settings: dict[str, Any] = {}


class CliYaml(BaseModel):
    model_config = ConfigDict(extra='allow')

    mapper: Optional[MapperSelection] = None
    network: Optional[NetworkSelection] = None
    partial_charge: Optional[PartialChargeSelection] = None


def parse_yaml_planner_options(contents: str) -> CliYaml:
    """Parse and minimally validate a user provided yaml

    Parameters
    ----------
    contents : str
      raw yaml formatted input to parse

    Returns
    -------
    options : CliOptions
      will have keys for mapper, network topology, and partial charge choices

    Raises
    ------
    ValueError
      for any malformed inputs
    """
    raw = yaml.safe_load(contents)

    expected_fields = {'mapper', 'network', 'partial_charge'}
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
    from openfe.protocols.openmm_utils.omm_settings import (
        OpenFFPartialChargeSettings
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
        mapper_obj = KartografAtomMapper(
            atom_max_distance=0.95,
            atom_map_hydrogens=True,
            # Non-default setting, as we remove these later anyway when correcting for constraints
            map_hydrogens_on_hydrogens_only=True,
            map_exact_ring_matches_only=True,
            # Current default, but should be changed in future Kartograf releases
            allow_partial_fused_rings=True,
            allow_bond_breaks=False,
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

    # We default to am1bcc on ambertools
    partial_charge_settings = OpenFFPartialChargeSettings()
    if opt and opt.partial_charge:
        partial_charge_settings.partial_charge_method = opt.partial_charge.method
        for setting in opt.partial_charge.settings:
            setattr(
                partial_charge_settings,
                setting,
                opt.partial_charge.settings[setting]
            )

    # todo: choice of solvent goes here
    solvent = SolventComponent()

    return PlanNetworkOptions(
        mapper_obj,
        mapping_scorer,
        ligand_network_planner,
        solvent,
        partial_charge_settings,
    )
# TODO: do we want this in the docs anywhere?
DEFAULT_YAML="""
    mapper: KartografAtomMapper
        settings:
            atom_max_distance: 0.95
            atom_map_hydrogens: true
            map_hydrogens_on_hydrogens_only: true
            map_exact_ring_matches_only: true
            allow_partial_fused_rings: true
            allow_bond_breaks: false

    network:
        method: generate_minimal_spanning_network

    partial_charge:
        method: am1bcc
        settings:
            off_toolkit_backend: ambertools
            number_of_conformers: None
            nagl_model: None
"""

_yaml_help = """
Path to a YAML file specifying the atom mapper (``mapper``), network planning algorithm (``network``),
and/or partial charge method (``partial_charge``) to use.

\b
Supported atom mapper choices are:
    - ``KartografAtomMapper`` (default as of v1.7.0)
    - ``LomapAtomMapper``
\b
Supported network planning algorithms include (but are not limited to):
    - ``generate_minimal_spanning_network`` (default)
    - ``generate_minimal_redundant_network``
    - ``generate_radial_network``
    - ``generate_lomap_network``
\b
Supported partial charge method choices are:
    - ``am1bcc`` (default)
    - ``am1bccelf10`` (only possible if ``off_toolkit_backend`` is ``openeye``)
    - ``nagl`` (must have openff-nagl installed)
    - ``espaloma`` (must have espaloma_charge installed)

``settings:`` allows for passing in any keyword arguments of the method's corresponding Python API.

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

  partial_charge:
    method: am1bcc
    settings:
      off_toolkit_backend: ambertools

"""

YAML_OPTIONS = Option(
    '-s', "--settings", "yaml_settings",
    type=click.Path(exists=True, dir_okay=False),
    help=_yaml_help,
    getter=load_yaml_planner_options,
)
