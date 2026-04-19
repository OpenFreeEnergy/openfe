# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import gufe
import pytest

from openfe.protocols.openmm_septop import SepTopProtocol


@pytest.fixture()
def protocol_dry_settings():
    # a set of settings for dry run tests
    s = SepTopProtocol.default_settings()
    s.engine_settings.compute_platform = None
    s.protocol_repeats = 1
    return s


@pytest.fixture
def benzene_toluene_dag(
    benzene_complex_system,
    toluene_complex_system,
    protocol_dry_settings,
):
    protocol = SepTopProtocol(settings=protocol_dry_settings)

    return protocol.create(
        stateA=benzene_complex_system,
        stateB=toluene_complex_system,
        mapping=None,
    )
