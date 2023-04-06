# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from openfe.protocols import openmm_rbfe
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
    return openmm_rbfe.RelativeLigandProtocol(openmm_rbfe.RelativeLigandProtocol.default_settings())


@pytest.fixture
def protocol_unit(protocol, benzene_system, toluene_system, benzene_to_toluene_mapping):
    pus = protocol.create(
        stateA=benzene_system, stateB=toluene_system,
        mapping={'ligand': benzene_to_toluene_mapping},
    )
    return list(pus.protocol_units)[0]


@pytest.mark.skip
class TestRelativeLigandProtocolResult(GufeTokenizableTestsMixin):
    cls = openmm_rbfe.RelativeLigandProtocolResult
    repr = ""
    key = ""

    @pytest.fixture()
    def instance(self):
        pass


class TestRelativeLigandProtocol(GufeTokenizableTestsMixin):
    cls = openmm_rbfe.RelativeLigandProtocol
    key = "RelativeLigandProtocol-1c8eb6aa916199f6404a5c05e5274a46"
    repr = "<RelativeLigandProtocol-1c8eb6aa916199f6404a5c05e5274a46>"

    @pytest.fixture()
    def instance(self, protocol):
        return protocol


class TestRelativeLigandProtocolUnit(GufeTokenizableTestsMixin):
    cls = openmm_rbfe.RelativeLigandProtocolUnit
    repr = "RelativeLigandProtocolUnit(benzene toluene repeat 2 generation 0)"
    key = None

    @pytest.fixture()
    def instance(self, protocol_unit):
        return protocol_unit

    def test_key_stable(self):
        pytest.skip()
