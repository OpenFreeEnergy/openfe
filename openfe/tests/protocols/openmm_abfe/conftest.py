# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import gufe
import pytest
from openfe.protocols import openmm_afe


@pytest.fixture
def benzene_complex_dag(benzene_modifications, T4_protein_component):
    s = openmm_afe.AbsoluteBindingProtocol.default_settings()

    protocol = openmm_afe.AbsoluteBindingProtocol(
        settings=s,
    )

    stateA = gufe.ChemicalSystem(
        {
            "protein": T4_protein_component,
            "benzene": benzene_modifications["benzene"],
            "solvent": gufe.SolventComponent(),
        }
    )

    stateB = gufe.ChemicalSystem(
        {
            "protein": T4_protein_component,
            "solvent": gufe.SolventComponent(),
        }
    )

    return protocol.create(stateA=stateA, stateB=stateB, mapping=None)

