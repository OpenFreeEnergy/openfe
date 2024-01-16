# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from openfecli.parameters import plan_network_options
import pytest


@pytest.fixture
def full_yaml():
    return """\
mapper:
  method: LomapAtomMapper
  settings:
    timeout: 120.0

network:
  method: generate_radial_network
"""


@pytest.fixture
def partial_mapper_yaml():
    return """\
mapper:
  method: KartografAtomMapper
  settings:
    timeout: 120.0
"""

@pytest.fixture
def partial_network_yaml():
    return """\
network:
  method: generate_radial_network
  settings:
    scorer: default_lomap_scorer
"""


def test_loading_full_yaml(full_yaml):
    d = plan_network_options.parse_yaml_planner_options(full_yaml)

    assert d
    assert d.mapper
    assert d.mapper.method == 'LomapAtomMapper'.lower()
    assert d.mapper.settings['timeout'] == 120
    assert d.network
    assert d.network.method == 'generate_radial_network'


def test_loading_mapper_yaml(partial_mapper_yaml):
    d = plan_network_options.parse_yaml_planner_options(partial_mapper_yaml)

    assert d
    assert d.mapper
    assert d.mapper.method == 'KartografAtomMapper'.lower()
    assert d.network is None


def test_loading_network_yaml(partial_network_yaml):
    d = plan_network_options.parse_yaml_planner_options(partial_network_yaml)

    assert d
    assert d.mapper is None
    assert d.network
    assert d.network.method == 'generate_radial_network'
    assert d.network.settings['scorer'] == 'default_lomap_scorer'
