# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import pytest
from gufe.tests.test_tokenization import GufeTokenizableTestsMixin
from openff.units import unit

from openfe.protocols import openmm_rfe
from openfe.protocols.openmm_rfe.hybridtop_units import (
    HybridTopologySetupUnit,
    HybridTopologyMultiStateSimulationUnit,
    HybridTopologyMultiStateAnalysisUnit,
)

"""
todo:
- RelativeHybridTopologyProtocolResult
- RelativeHybridTopologyProtocol
- RelativeHybridTopologyProtocolUnit
"""


@pytest.fixture
def rfe_protocol():
    return openmm_rfe.RelativeHybridTopologyProtocol(
        openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
    )


@pytest.fixture
def rfe_protocol_other_input_units():
    """Identical to rfe_protocol, but with `kcal / mol` as input unit instead of `kilocalorie_per_mole`."""
    new_settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
    new_settings.simulation_settings.early_termination_target_error = 0.0 * unit.kilocalorie/unit.mol  # fmt: skip
    return openmm_rfe.RelativeHybridTopologyProtocol(new_settings)


@pytest.fixture
def protocol_units(
    rfe_protocol,
    benzene_system,
    toluene_system,
    benzene_to_toluene_mapping
):
    pus = rfe_protocol.create(
        stateA=benzene_system,
        stateB=toluene_system,
        mapping=[benzene_to_toluene_mapping],
    )
    return list(pus.protocol_units)


@pytest.fixture
def protocol_setup_unit(
    protocol_units
):
    for pu in protocol_units:
        if isinstance(pu, HybridTopologySetupUnit):
            return pu


@pytest.fixture
def protocol_simulation_unit(
    protocol_units
):
    for pu in protocol_units:
        if isinstance(pu, HybridTopologyMultiStateSimulationUnit):
            return pu


@pytest.fixture
def protocol_analysis_unit(
    protocol_units
):
    for pu in protocol_units:
        if isinstance(pu, HybridTopologyMultiStateAnalysisUnit):
            return pu


@pytest.mark.skip
class TestRelativeHybridTopologyProtocolResult(GufeTokenizableTestsMixin):
    cls = openmm_rfe.RelativeHybridTopologyProtocolResult
    repr = ""
    key = ""

    @pytest.fixture()
    def instance(self):
        pass


class TestRelativeHybridTopologyProtocolOtherInputUnits(GufeTokenizableTestsMixin):
    cls = openmm_rfe.RelativeHybridTopologyProtocol
    key = None
    repr = "<RelativeHybridTopologyProtocol-"

    @pytest.fixture()
    def instance(self, rfe_protocol_other_input_units):
        return rfe_protocol_other_input_units

    def test_repr(self, instance):
        """
        Overwrites the base `test_repr` call.
        """
        assert isinstance(repr(instance), str)
        assert self.repr in repr(instance)


class TestRelativeHybridTopologyProtocol(GufeTokenizableTestsMixin):
    cls = openmm_rfe.RelativeHybridTopologyProtocol
    key = None
    repr = "<RelativeHybridTopologyProtocol-"

    @pytest.fixture()
    def instance(self, rfe_protocol):
        return rfe_protocol

    def test_repr(self, instance):
        """
        Overwrites the base `test_repr` call.
        """
        assert isinstance(repr(instance), str)
        assert self.repr in repr(instance)


class TestHybridTopologySetupUnit(GufeTokenizableTestsMixin):
    cls = openmm_rfe.HybridTopologySetupUnit
    repr = "HybridTopologySetupUnit(HybridTopology Setup:"
    key = None

    @pytest.fixture()
    def instance(self, protocol_setup_unit):
        return protocol_setup_unit

    def test_key_stable(self):
        pytest.skip()

    def test_repr(self, instance):
        """
        Overwrites the base `test_repr` call.
        """
        assert isinstance(repr(instance), str)
        assert self.repr in repr(instance)


class TestHybridTopologyMultiStateSimulationUnit(GufeTokenizableTestsMixin):
    cls = openmm_rfe.HybridTopologyMultiStateSimulationUnit
    repr = "HybridTopologyMultiStateSimulationUnit(HybridTopology Simulation:"
    key = None

    @pytest.fixture()
    def instance(self, protocol_simulation_unit):
        return protocol_simulation_unit

    def test_key_stable(self):
        pytest.skip()

    def test_repr(self, instance):
        """
        Overwrites the base `test_repr` call.
        """
        assert isinstance(repr(instance), str)
        assert self.repr in repr(instance)


class TestHybridTopologyMultiStateAnalysisUnit(GufeTokenizableTestsMixin):
    cls = openmm_rfe.HybridTopologyMultiStateAnalysisUnit
    repr = "HybridTopologyMultiStateAnalysisUnit(HybridTopology Analysis:"
    key = None

    @pytest.fixture()
    def instance(self, protocol_analysis_unit):
        return protocol_analysis_unit

    def test_key_stable(self):
        pytest.skip()

    def test_repr(self, instance):
        """
        Overwrites the base `test_repr` call.
        """
        assert isinstance(repr(instance), str)
        assert self.repr in repr(instance)
