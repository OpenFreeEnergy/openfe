# This ccode is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import json
import pytest
import gufe
from gufe.tests.test_tokenization import GufeTokenizableTestsMixin
from openfe.protocols import openmm_md


@pytest.fixture
def protocol():
    return openmm_md.PlainMDProtocol(openmm_md.PlainMDProtocol.default_settings())


@pytest.fixture
def protocol_unit(protocol, benzene_system):
    pus = protocol.create(
        stateA=benzene_system,
        stateB=benzene_system,
        mapping=None,
    )
    return list(pus.protocol_units)[0]


@pytest.fixture
def protocol_result(md_json):
    d = json.loads(md_json, cls=gufe.tokenization.JSON_HANDLER.decoder)
    pr = gufe.ProtocolResult.from_dict(d["protocol_result"])
    return pr


class TestPlainMDProtocol(GufeTokenizableTestsMixin):
    cls = openmm_md.PlainMDProtocol
    key = None
    repr = "PlainMDProtocol-"

    @pytest.fixture()
    def instance(self, protocol):
        return protocol

    def test_repr(self, instance):
        """
        Overwrites the base `test_repr` call to do a bit more.
        """
        assert isinstance(repr(instance), str)
        assert self.repr in repr(instance)


class TestPlainMDProtocolUnit(GufeTokenizableTestsMixin):
    cls = openmm_md.PlainMDProtocolUnit
    repr = "PlainMDProtocolUnit("
    key = None

    @pytest.fixture
    def instance(self, protocol_unit):
        return protocol_unit

    def test_repr(self, instance):
        """
        Overwrites the base `test_repr` call to do a bit more.
        """
        assert isinstance(repr(instance), str)
        assert self.repr in repr(instance)


class TestPlainMDProtocolResult(GufeTokenizableTestsMixin):
    cls = openmm_md.PlainMDProtocolResult
    key = None
    repr = "PlainMDProtocolResult-"

    @pytest.fixture()
    def instance(self, protocol_result):
        return protocol_result

    def test_repr(self, instance):
        """
        Overwrites the base `test_repr` call to do a bit more.
        """
        assert isinstance(repr(instance), str)
        assert self.repr in repr(instance)
