# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from openfe.protocols import openmm_rfe
from gufe.tests.test_tokenization import GufeTokenizableTestsMixin
from openff.units import unit
import pytest

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
def rfe_protocol_other_units():
    """Identical to rfe_protocol, but with `kcal / mol` as input unit instead of `kilocalorie_per_mole`."""
    new_settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
    new_settings.simulation_settings.early_termination_target_error = 0.0 * unit.kilocalorie/unit.mol  # fmt: skip
    return openmm_rfe.RelativeHybridTopologyProtocol(new_settings)


@pytest.fixture
def protocol_unit(rfe_protocol, benzene_system, toluene_system, benzene_to_toluene_mapping):
    pus = rfe_protocol.create(
        stateA=benzene_system,
        stateB=toluene_system,
        mapping=[benzene_to_toluene_mapping],
    )
    return list(pus.protocol_units)[0]


@pytest.mark.skip
class TestRelativeHybridTopologyProtocolResult(GufeTokenizableTestsMixin):
    cls = openmm_rfe.RelativeHybridTopologyProtocolResult
    repr = ""
    key = ""

    @pytest.fixture()
    def instance(self):
        pass


class TestRelativeHybridTopologyProtocolOtherUnits(GufeTokenizableTestsMixin):
    cls = openmm_rfe.RelativeHybridTopologyProtocol
    key = None
    repr = "<RelativeHybridTopologyProtocol-"

    @pytest.fixture()
    def instance(self, rfe_protocol_other_units):
        return rfe_protocol_other_units

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


class TestRelativeHybridTopologyProtocolUnit(GufeTokenizableTestsMixin):
    cls = openmm_rfe.RelativeHybridTopologyProtocolUnit
    repr = "RelativeHybridTopologyProtocolUnit(benzene to toluene repeat"
    key = None

    @pytest.fixture()
    def instance(self, protocol_unit):
        return protocol_unit

    def test_key_stable(self):
        pytest.skip()

    def test_repr(self, instance):
        """
        Overwrites the base `test_repr` call.
        """
        assert isinstance(repr(instance), str)
        assert self.repr in repr(instance)
