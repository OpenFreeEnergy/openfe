# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from openfe.protocols import openmm_rfe
from gufe.tests.test_tokenization import GufeTokenizableTestsMixin
import pytest

"""
todo:
- RelativeLigandProtocolResult
- RelativeLigandProtocol
- RelativeLigandProtocolUnit
"""

@pytest.fixture
def protocol():
    return openmm_rfe.RelativeLigandProtocol(openmm_rfe.RelativeLigandProtocol.default_settings())


@pytest.fixture
def protocol_unit(protocol, benzene_system, toluene_system, benzene_to_toluene_mapping):
    pus = protocol.create(
        stateA=benzene_system, stateB=toluene_system,
        mapping={'ligand': benzene_to_toluene_mapping},
    )
    return list(pus.protocol_units)[0]


@pytest.mark.skip
class TestRelativeLigandProtocolResult(GufeTokenizableTestsMixin):
    cls = openmm_rfe.RelativeLigandProtocolResult
    repr = ""
    key = ""

    @pytest.fixture()
    def instance(self):
        pass


class TestRelativeLigandProtocol(GufeTokenizableTestsMixin):
    cls = openmm_rfe.RelativeLigandProtocol
    key = "RelativeLigandProtocol-b2e04d34a8dbc3fc67a2ae27cd07c813"
    repr = "<RelativeLigandProtocol-b2e04d34a8dbc3fc67a2ae27cd07c813>"

    @pytest.fixture()
    def instance(self, protocol):
        return protocol


class TestRelativeLigandProtocolUnit(GufeTokenizableTestsMixin):
    cls = openmm_rfe.RelativeLigandProtocolUnit
    repr = "RelativeLigandProtocolUnit(benzene toluene repeat 2 generation 0)"
    key = None

    @pytest.fixture()
    def instance(self, protocol_unit):
        return protocol_unit

    def test_key_stable(self):
        pytest.skip()
