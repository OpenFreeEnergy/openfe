# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import json
import openfe
from openfe.protocols import openmm_afe
import gufe
from gufe.tests.test_tokenization import GufeTokenizableTestsMixin
import pytest


@pytest.fixture
def protocol():
    return openmm_afe.AbsoluteBindingProtocol(
               openmm_afe.AbsoluteBindingProtocol.default_settings()
           )


@pytest.fixture
def protocol_units(protocol, benzene_complex_system, T4_protein_component):
    stateA = benzene_complex_system
    stateB = openfe.ChemicalSystem(
        {
            'protein': T4_protein_component,
            'solvent': openfe.SolventComponent()
        }
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
        if isinstance(pu, openmm_afe.AbsoluteBindingSolventUnit):
            return pu


@pytest.fixture
def complex_protocol_unit(protocol_units):
    for pu in protocol_units:
        if isinstance(pu, openmm_afe.AbsoluteBindingComplexUnit):
            return pu


# @pytest.fixture
# def protocol_result(afe_solv_transformation_json):
#     d = json.loads(afe_solv_transformation_json,
#                    cls=gufe.tokenization.JSON_HANDLER.decoder)
#     pr = openmm_afe.AbsoluteSolvationProtocolResult.from_dict(d['protocol_result'])
#     return pr


class TestAbsoluteBindingProtocol(GufeTokenizableTestsMixin):
    cls = openmm_afe.AbsoluteBindingProtocol
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
    cls = openmm_afe.AbsoluteBindingSolventUnit
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
    cls = openmm_afe.AbsoluteBindingComplexUnit
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


# class TestAbsoluteSolvationProtocolResult(GufeTokenizableTestsMixin):
#     cls = openmm_afe.AbsoluteSolvationProtocolResult
#     key = None
#     repr = "AbsoluteSolvationProtocolResult-"
# 
#     @pytest.fixture()
#     def instance(self, protocol_result):
#         return protocol_result
# 
#     def test_repr(self, instance):
#         """
#         Overwrites the base `test_repr` call.
#         """
#         assert isinstance(repr(instance), str)
#         assert self.repr in repr(instance)
