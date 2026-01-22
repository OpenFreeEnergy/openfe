# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import json

import gufe
import pytest

import openfe
from openfe.protocols import openmm_afe
from openfe.protocols.openmm_afe import (
    AHFESolventAnalysisUnit,
    AHFESolventSetupUnit,
    AHFESolventSimUnit,
    AHFEVacuumAnalysisUnit,
    AHFEVacuumSetupUnit,
    AHFEVacuumSimUnit,
)

from ..conftest import ModGufeTokenizableTestsMixin


@pytest.fixture
def protocol():
    return openmm_afe.AbsoluteSolvationProtocol(
        openmm_afe.AbsoluteSolvationProtocol.default_settings()
    )


@pytest.fixture
def protocol_units(protocol, benzene_system):
    pus = protocol.create(
        stateA=benzene_system,
        stateB=openfe.ChemicalSystem({"solvent": openfe.SolventComponent()}),
        mapping=None,
    )
    return list(pus.protocol_units)


def _filter_units(pus, classtype):
    for pu in pus:
        if isinstance(pu, classtype):
            return pu


@pytest.fixture
def solvent_protocol_setup_unit(protocol_units):
    return _filter_units(protocol_units, AHFESolventSetupUnit)


@pytest.fixture
def solvent_protocol_sim_unit(protocol_units):
    return _filter_units(protocol_units, AHFESolventSimUnit)


@pytest.fixture
def solvent_protocol_analysis_unit(protocol_units):
    return _filter_units(protocol_units, AHFESolventAnalysisUnit)


@pytest.fixture
def vacuum_protocol_setup_unit(protocol_units):
    return _filter_units(protocol_units, AHFEVacuumSetupUnit)


@pytest.fixture
def vacuum_protocol_sim_unit(protocol_units):
    return _filter_units(protocol_units, AHFEVacuumSimUnit)


@pytest.fixture
def vacuum_protocol_analysis_unit(protocol_units):
    return _filter_units(protocol_units, AHFEVacuumAnalysisUnit)


@pytest.fixture
def protocol_result(afe_solv_transformation_json):
    d = json.loads(afe_solv_transformation_json, cls=gufe.tokenization.JSON_HANDLER.decoder)
    pr = openmm_afe.AbsoluteSolvationProtocolResult.from_dict(d["protocol_result"])
    return pr


class TestAbsoluteSolvationProtocol(ModGufeTokenizableTestsMixin):
    cls = openmm_afe.AbsoluteSolvationProtocol
    key = None
    repr = "AbsoluteSolvationProtocol-"

    @pytest.fixture()
    def instance(self, protocol):
        return protocol


class TestAHFESolventSetupUnit(ModGufeTokenizableTestsMixin):
    cls = AHFESolventSetupUnit
    repr = "AHFESolventSetupUnit(AHFE Setup: benzene solvent leg"
    key = None

    @pytest.fixture()
    def instance(self, solvent_protocol_setup_unit):
        return solvent_protocol_setup_unit


class TestAHFESolventSimUnit(ModGufeTokenizableTestsMixin):
    cls = AHFESolventSimUnit
    repr = "AHFESolventSimUnit(AHFE Simulation: benzene solvent leg"
    key = None

    @pytest.fixture()
    def instance(self, solvent_protocol_sim_unit):
        return solvent_protocol_sim_unit


class TestAHFESolventAnalysisUnit(ModGufeTokenizableTestsMixin):
    cls = AHFESolventAnalysisUnit
    repr = "AHFESolventAnalysisUnit(AHFE Analysis: benzene solvent leg"
    key = None

    @pytest.fixture()
    def instance(self, solvent_protocol_analysis_unit):
        return solvent_protocol_analysis_unit


class TestAHFEVacuumSetupUnit(ModGufeTokenizableTestsMixin):
    cls = AHFEVacuumSetupUnit
    repr = "AHFEVacuumSetupUnit(AHFE Setup: benzene vacuum leg"
    key = None

    @pytest.fixture()
    def instance(self, vacuum_protocol_setup_unit):
        return vacuum_protocol_setup_unit


class TestAHFEVacuumSimUnit(ModGufeTokenizableTestsMixin):
    cls = AHFEVacuumSimUnit
    repr = "AHFEVacuumSimUnit(AHFE Simulation: benzene vacuum leg"
    key = None

    @pytest.fixture()
    def instance(self, vacuum_protocol_sim_unit):
        return vacuum_protocol_sim_unit


class TestAHFEVacuumAnalysisUnit(ModGufeTokenizableTestsMixin):
    cls = AHFEVacuumAnalysisUnit
    repr = "AHFEVacuumAnalysisUnit(AHFE Analysis: benzene vacuum leg"
    key = None

    @pytest.fixture()
    def instance(self, vacuum_protocol_analysis_unit):
        return vacuum_protocol_analysis_unit


class TestAbsoluteSolvationProtocolResult(ModGufeTokenizableTestsMixin):
    cls = openmm_afe.AbsoluteSolvationProtocolResult
    key = None
    repr = "AbsoluteSolvationProtocolResult-"

    @pytest.fixture()
    def instance(self, protocol_result):
        return protocol_result
