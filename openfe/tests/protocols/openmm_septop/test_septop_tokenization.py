# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import json
from openfe.protocols import openmm_septop
import gufe
from gufe.tests.test_tokenization import GufeTokenizableTestsMixin
import pytest


@pytest.fixture
def protocol():
    return openmm_septop.SepTopProtocol(openmm_septop.SepTopProtocol.default_settings())


@pytest.fixture
def protocol_units(protocol, benzene_complex_system, toluene_complex_system):
    pus = protocol.create(
        stateA=benzene_complex_system,
        stateB=toluene_complex_system,
        mapping=None,
    )
    return list(pus.protocol_units)


@pytest.fixture
def solvent_setup_protocol_unit(protocol_units):
    for pu in protocol_units:
        if isinstance(pu, openmm_septop.SepTopSolventSetupUnit):
            return pu


@pytest.fixture
def solvent_run_protocol_unit(protocol_units):
    for pu in protocol_units:
        if isinstance(pu, openmm_septop.SepTopSolventRunUnit):
            return pu


@pytest.fixture
def complex_setup_protocol_unit(protocol_units):
    for pu in protocol_units:
        if isinstance(pu, openmm_septop.SepTopComplexSetupUnit):
            return pu


@pytest.fixture
def complex_run_protocol_unit(protocol_units):
    for pu in protocol_units:
        if isinstance(pu, openmm_septop.SepTopComplexRunUnit):
            return pu


@pytest.fixture
def protocol_result(septop_json):
    d = json.loads(
        septop_json,
        cls=gufe.tokenization.JSON_HANDLER.decoder,
    )
    pr = openmm_septop.SepTopProtocolResult.from_dict(d["protocol_result"])
    return pr


class TestSepTopProtocol(GufeTokenizableTestsMixin):
    cls = openmm_septop.SepTopProtocol
    key = None
    repr = "<SepTopProtocol-"

    @pytest.fixture()
    def instance(self, protocol):
        return protocol

    def test_repr(self, instance):
        """
        Overwrites the base `test_repr` call.
        """
        assert isinstance(repr(instance), str)
        assert self.repr in repr(instance)


class TestSepTopSolventSetupUnit(GufeTokenizableTestsMixin):
    cls = openmm_septop.SepTopSolventSetupUnit
    repr = (
        "SepTopSolventSetupUnit(SepTop RBFE Setup, transformation benzene to toluene, solvent leg"
    )
    key = None

    @pytest.fixture()
    def instance(self, solvent_setup_protocol_unit):
        return solvent_setup_protocol_unit

    def test_repr(self, instance):
        """
        Overwrites the base `test_repr` call.
        """
        assert isinstance(repr(instance), str)
        assert self.repr in repr(instance)


class TestSepTopSolventRunUnit(GufeTokenizableTestsMixin):
    cls = openmm_septop.SepTopSolventRunUnit
    repr = "SepTopSolventRunUnit(SepTop RBFE Run, transformation benzene to toluene, solvent leg"
    key = None

    @pytest.fixture()
    def instance(self, solvent_run_protocol_unit):
        return solvent_run_protocol_unit

    def test_repr(self, instance):
        """
        Overwrites the base `test_repr` call.
        """
        assert isinstance(repr(instance), str)
        assert self.repr in repr(instance)


class TestSepTopComplexSetupUnit(GufeTokenizableTestsMixin):
    cls = openmm_septop.SepTopComplexSetupUnit
    repr = (
        "SepTopComplexSetupUnit(SepTop RBFE Setup, transformation benzene to toluene, complex leg"
    )
    key = None

    @pytest.fixture()
    def instance(self, complex_setup_protocol_unit):
        return complex_setup_protocol_unit

    def test_repr(self, instance):
        """
        Overwrites the base `test_repr` call.
        """
        assert isinstance(repr(instance), str)
        assert self.repr in repr(instance)


class TestSepTopComplexRunUnit(GufeTokenizableTestsMixin):
    cls = openmm_septop.SepTopComplexRunUnit
    repr = "SepTopComplexRunUnit(SepTop RBFE Run, transformation benzene to toluene, complex leg"
    key = None

    @pytest.fixture()
    def instance(self, complex_run_protocol_unit):
        return complex_run_protocol_unit

    def test_repr(self, instance):
        """
        Overwrites the base `test_repr` call.
        """
        assert isinstance(repr(instance), str)
        assert self.repr in repr(instance)


class TestSepTopProtocolResult(GufeTokenizableTestsMixin):
    cls = openmm_septop.SepTopProtocolResult
    key = None
    repr = "SepTopProtocolResult-"

    @pytest.fixture()
    def instance(self, protocol_result):
        return protocol_result

    def test_repr(self, instance):
        """
        Overwrites the base `test_repr` call.
        """
        assert isinstance(repr(instance), str)
        assert self.repr in repr(instance)
