# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from openfecli.parameters import plan_network_options
import pytest
from openff.units import unit

from openfe.protocols import openmm_rfe


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


@pytest.fixture
def protocol_settings_yaml():
    return """\
protocol:
  method: RelativeHybridTopologyProtocol
  settings:
    protocol_repeats: 2
    simulation_settings:
      production_length: 7.5 ns
      equilibration_length: 2200 ps
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


def test_parsing_protocol_yaml(protocol_settings_yaml):
    d = plan_network_options.parse_yaml_planner_options(protocol_settings_yaml)

    assert d
    assert d.protocol.method == 'relativehybridtopologyprotocol'
    assert d.protocol.settings['protocol_repeats'] == 2
    assert d.protocol.settings['simulation_settings']['production_length'] == '7.5 ns'


def test_resolving_protocol_yaml(protocol_settings_yaml):
    cliyaml = plan_network_options.parse_yaml_planner_options(protocol_settings_yaml)

    pno = plan_network_options.load_yaml_planner_options_from_cliyaml(cliyaml)

    prot = pno.protocol
    assert isinstance(prot, openmm_rfe.RelativeHybridTopologyProtocol)
    assert prot.settings.protocol_repeats == 2
    assert prot.settings.simulation_settings.production_length.m == pytest.approx(7.5)
    assert prot.settings.simulation_settings.production_length.u == unit.nanosecond
    assert prot.settings.simulation_settings.equilibration_length.m == pytest.approx(2.2)
    assert prot.settings.simulation_settings.equilibration_length.u == unit.nanosecond


def test_nearest_match():
    # check that misspelt options are given a likely correction
    # here production -> production_length

    yaml = """\
protocol:
  method: RelativeHybridTopologyProtocol
  settings:
    simulation_settings:
      production: 5 ns
      equilibration_length: 1 ns
"""
    cliyaml = plan_network_options.parse_yaml_planner_options(yaml)

    with pytest.raises(ValueError, match="did you mean 'production_length'"):
        plan_network_options.load_yaml_planner_options_from_cliyaml(cliyaml)


def test_bad_network_option():
    yaml = """\
network:
  method: generate_starmap
  settings:
    central_ligand: 0
"""
    cliyaml = plan_network_options.parse_yaml_planner_options(yaml)

    with pytest.raises(ValueError, match="Bad network algorithm choice: 'generate_starmap'. Available options are"):
        plan_network_options.load_yaml_planner_options_from_cliyaml(cliyaml)


def test_bad_protocol_option():
    yaml = """\
protocol:
  method: wizardry 
"""
    cliyaml = plan_network_options.parse_yaml_planner_options(yaml)

    with pytest.raises(ValueError, match="Unsupported protocol method 'wizardry'. Supported methods are"):
        plan_network_options.load_yaml_planner_options_from_cliyaml(cliyaml)


def test_bad_protocol_settings_input():
    # input to modifying settings object must be dict
    # i.e. can't set `simulation_settings = 4'
    yaml = """\
protocol:
  method: RelativeHybridTopologyProtocol
  settings:
    simulation_settings: 24 
"""
    cliyaml = plan_network_options.parse_yaml_planner_options(yaml)

    with pytest.raises(ValueError, match="must set sub-settings 'simulation_settings' to dict"):
        plan_network_options.load_yaml_planner_options_from_cliyaml(cliyaml)
