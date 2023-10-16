# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import openfe
from openfe.protocols import openmm_afe
from gufe.tests.test_tokenization import GufeTokenizableTestsMixin
import pytest

"""
todo:
- AbsoluteSolvationProtocolResult
- AbsoluteSolvationProtocol
- AbsoluteSolvationProtocolUnit
"""

@pytest.fixture
def protocol():
    return openmm_afe.AbsoluteSolvationProtocol(
               openmm_afe.AbsoluteSolvationProtocol.default_settings()
           )


@pytest.fixture
def protocol_unit(protocol, benzene_system):
    pus = protocol.create(
        stateA=benzene_system, stateB=openfe.SolventComponent(),
        mapping=None,
    )
    return list(pus.protocol_units)[0]


class TestAbsoluteSolvationProtocol(GufeTokenizableTestsMixin):
    cls = openmm_afe.AbsoluteSolvationProtocol
    key = "RelativeHybridTopologyProtocol-8fe3eb4c318673db7d57c1bffca602df"
    repr = f"<{key}>"

    @pytest.fixture()
    def instance(self, protocol):
        return protocol


class TestAbsoluteSolventProtocolUnit(GufeTokenizableTestsMixin):
    cls = openmm_afe.AbsoluteSolvationProtocolUnit
    repr = "RelativeHybridTopologyProtocolUnit(benzene to toluene repeat 2 generation 0)"
    key = None

    @pytest.fixture()
    def instance(self, protocol_unit):
        return protocol_unit

    def test_key_stable(self):
        pytest.skip()
