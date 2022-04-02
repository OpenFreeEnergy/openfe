# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from openfe.setup.methods import openmm


def test_create_default_settings():
    settings = openmm.RelativeLigandTransform.get_default_settings()

    assert settings
