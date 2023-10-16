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
def protocol_units(protocol, benzene_system):
    pus = protocol.create(
        stateA=benzene_system,
        stateB=openfe.ChemicalSystem({'solvent': openfe.SolventComponent()}),
        mapping=None,
    )
    return list(pus.protocol_units)


@pytest.fixture
def solvent_protocol_unit(protocol_units):
    for pu in protocol_units:
        if isinstance(pu, openmm_afe.AbsoluteSolventTransformUnit):
            return pu


@pytest.fixture
def vacuum_protocol_unit(protocol_units):
    for pu in protocol_units:
        if isinstance(pu, openmm_afe.AbsoluteVacuumTransformUnit):
            return pu

class TestAbsoluteSolvationProtocol(GufeTokenizableTestsMixin):
    cls = openmm_afe.AbsoluteSolvationProtocol
    key = "AbsoluteSolvationProtocol-fd22076bcea777207beb86ef7a6ded81"
    repr = f"<{key}>"

    @pytest.fixture()
    def instance(self, protocol):
        return protocol


class TestAbsoluteSolventTransformUnit(GufeTokenizableTestsMixin):
    cls = openmm_afe.AbsoluteSolventTransformUnit
    repr = "AbsoluteSolventTransformUnit(Absolute Solvation, benzene solvent leg: repeat 2 generation 0)"
    key = None

    @pytest.fixture()
    def instance(self, solvent_protocol_unit):
        return solvent_protocol_unit

    def test_key_stable(self):
        pytest.skip()


class TestAbsoluteVacuumTransformUnit(GufeTokenizableTestsMixin):
    cls = openmm_afe.AbsoluteVacuumTransformUnit
    repr = "AbsoluteVacuumTransformUnit(Absolute Solvation, benzene vacuum leg: repeat 2 generation 0)"
    key = None

    @pytest.fixture()
    def instance(self, vacuum_protocol_unit):
        return vacuum_protocol_unit

    def test_key_stable(self):
        pytest.skip()
