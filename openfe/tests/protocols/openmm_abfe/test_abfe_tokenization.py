# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import gzip

import pytest
from ..conftest import ModGufeTokenizableTestsMixin

import openfe
from openfe.protocols.openmm_afe import (
    AbsoluteBindingProtocol,
    AbsoluteBindingProtocolResult,
    ABFEComplexSetupUnit,
    ABFEComplexSimUnit,
    ABFEComplexAnalysisUnit,
    ABFESolventSetupUnit,
    ABFESolventSimUnit,
    ABFESolventAnalysisUnit,
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


def _filter_units(pus, classtype):
    for pu in pus:
        if isinstance(pu, classtype):
            return pu


@pytest.fixture
def complex_protocol_setup_unit(protocol_units):
    return _filter_units(protocol_units, ABFEComplexSetupUnit)


@pytest.fixture
def complex_protocol_sim_unit(protocol_units):
    return _filter_units(protocol_units, ABFEComplexSimUnit)


@pytest.fixture
def complex_protocol_analysis_unit(protocol_units):
    return _filter_units(protocol_units, ABFEComplexAnalysisUnit)


@pytest.fixture
def solvent_protocol_setup_unit(protocol_units):
    return _filter_units(protocol_units, ABFESolventSetupUnit)


@pytest.fixture
def solvent_protocol_sim_unit(protocol_units):
    return _filter_units(protocol_units, ABFESolventSimUnit)


@pytest.fixture
def solvent_protocol_analysis_unit(protocol_units):
    return _filter_units(protocol_units, ABFESolventAnalysisUnit)


@pytest.fixture
def protocol_result(abfe_transformation_json_path):
    with gzip.open(abfe_transformation_json_path) as f:
        pr = AbsoluteBindingProtocolResult.from_json(f)
    return pr


class TestAbsoluteBindingProtocol(ModGufeTokenizableTestsMixin):
    cls = AbsoluteBindingProtocol
    key = None
    repr = "AbsoluteBindingProtocol-"

    @pytest.fixture()
    def instance(self, protocol):
        return protocol


class TestABFESolventSetupUnit(ModGufeTokenizableTestsMixin):
    cls = ABFESolventSetupUnit
    repr = "ABFESolventSetupUnit(ABFE Setup: benzene solvent leg"
    key = None

    @pytest.fixture()
    def instance(self, solvent_protocol_setup_unit):
        return solvent_protocol_setup_unit


class TestABFESolventSimUnit(ModGufeTokenizableTestsMixin):
    cls = ABFESolventSimUnit
    repr = "ABFESolventSimUnit(ABFE Simulation: benzene solvent leg"
    key = None

    @pytest.fixture()
    def instance(self, solvent_protocol_sim_unit):
        return solvent_protocol_sim_unit


class TestABFESolventAnalysisUnit(ModGufeTokenizableTestsMixin):
    cls = ABFESolventAnalysisUnit
    repr = "ABFESolventAnalysisUnit(ABFE Analysis: benzene solvent leg"
    key = None

    @pytest.fixture()
    def instance(self, solvent_protocol_analysis_unit):
        return solvent_protocol_analysis_unit


class TestABFEComplexSetupUnit(ModGufeTokenizableTestsMixin):
    cls = ABFEComplexSetupUnit
    repr = "ABFEComplexSetupUnit(ABFE Setup: benzene complex leg"
    key = None

    @pytest.fixture()
    def instance(self, complex_protocol_setup_unit):
        return complex_protocol_setup_unit


class TestABFEComplexSimUnit(ModGufeTokenizableTestsMixin):
    cls = ABFEComplexSimUnit
    repr = "ABFEComplexSimUnit(ABFE Simulation: benzene complex leg"
    key = None

    @pytest.fixture()
    def instance(self, complex_protocol_sim_unit):
        return complex_protocol_sim_unit


class TestABFEComplexAnalysisUnit(ModGufeTokenizableTestsMixin):
    cls = ABFEComplexAnalysisUnit
    repr = "ABFEComplexAnalysisUnit(ABFE Analysis: benzene complex leg"
    key = None

    @pytest.fixture()
    def instance(self, complex_protocol_analysis_unit):
        return complex_protocol_analysis_unit


class TestAbsoluteBindingProtocolResult(ModGufeTokenizableTestsMixin):
    cls = AbsoluteBindingProtocolResult
    key = None
    repr = "AbsoluteBindingProtocolResult-"

    @pytest.fixture()
    def instance(self, protocol_result):
        return protocol_result
