# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import gzip

import openfe
import pytest
from gufe.tests.test_tokenization import GufeTokenizableTestsMixin
from openfe.protocols.openmm_afe import (
    AbsoluteBindingComplexUnit,
    AbsoluteBindingProtocol,
    AbsoluteBindingProtocolResult,
    AbsoluteBindingSolventUnit,
)


@pytest.fixture
def protocol():
    return AbsoluteBindingProtocol(AbsoluteBindingProtocol.default_settings())


@pytest.fixture
def protocol_units(protocol, benzene_complex_system, T4_protein_component):
    stateA = benzene_complex_system
    stateB = openfe.ChemicalSystem(
        {"protein": T4_protein_component, "solvent": openfe.SolventComponent()}
    )
    pus = protocol.create(
        stateA=stateA,
        stateB=stateB,
        mapping=None,
    )
    return list(pus.protocol_units)


@pytest.fixture
def solvent_protocol_unit(protocol_units):
    for pu in protocol_units:
        if isinstance(pu, AbsoluteBindingSolventUnit):
            return pu


@pytest.fixture
def complex_protocol_unit(protocol_units):
    for pu in protocol_units:
        if isinstance(pu, AbsoluteBindingComplexUnit):
            return pu


@pytest.fixture
def protocol_result(abfe_transformation_json_path):
    with gzip.open(abfe_transformation_json_path) as f:
        pr = AbsoluteBindingProtocolResult.from_json(f)
    return pr


class TestAbsoluteBindingProtocol(GufeTokenizableTestsMixin):
    cls = AbsoluteBindingProtocol
    key = None
    repr = "AbsoluteBindingProtocol-"

    @pytest.fixture()
    def instance(self, protocol):
        return protocol

    def test_repr(self, instance):
        """
        Overwrites the base `test_repr` call.
        """
        assert isinstance(repr(instance), str)
        assert self.repr in repr(instance)


class TestAbsoluteBindingSolventUnit(GufeTokenizableTestsMixin):
    cls = AbsoluteBindingSolventUnit
    repr = "AbsoluteBindingSolventUnit(Absolute Binding, benzene solvent leg"
    key = None

    @pytest.fixture()
    def instance(self, solvent_protocol_unit):
        return solvent_protocol_unit

    def test_repr(self, instance):
        """
        Overwrites the base `test_repr` call.
        """
        assert isinstance(repr(instance), str)
        assert self.repr in repr(instance)


class TestAbsoluteBindingComplexUnit(GufeTokenizableTestsMixin):
    cls = AbsoluteBindingComplexUnit
    repr = "AbsoluteBindingComplexUnit(Absolute Binding, benzene complex leg"
    key = None

    @pytest.fixture()
    def instance(self, complex_protocol_unit):
        return complex_protocol_unit

    def test_repr(self, instance):
        """
        Overwrites the base `test_repr` call.
        """
        assert isinstance(repr(instance), str)
        assert self.repr in repr(instance)


class TestAbsoluteBindingProtocolResult(GufeTokenizableTestsMixin):
    cls = AbsoluteBindingProtocolResult
    key = None
    repr = "AbsoluteBindingProtocolResult-"

    @pytest.fixture()
    def instance(self, protocol_result):
        return protocol_result

    def test_repr(self, instance):
        """
        Overwrites the base `test_repr` call.
        """
        assert isinstance(repr(instance), str)
        assert self.repr in repr(instance)
