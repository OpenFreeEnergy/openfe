# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from importlib import resources
from rdkit import Chem
import gufe
import pytest
from openfe.protocols import openmm_afe


@pytest.fixture
def benzene_complex_dag(benzene_modifications, T4_protein_component):
    s = openmm_afe.AbsoluteBindingProtocol.default_settings()
    s.complex_output_settings.output_indices = "not water"
    s.solvent_output_settings.output_indices = "not water"

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


@pytest.fixture(scope="session")
def guests_OA():
    files = {}
    with resources.as_file(resources.files("openfe.tests.data.host_guest.OA")) as d:
        fn = str(d / "guests_charged.sdf")
        supp = Chem.SDMolSupplier(str(fn), removeHs=False)
        for rdmol in supp:
            files[rdmol.GetProp("_Name")] = gufe.SmallMoleculeComponent(rdmol)
    return files


@pytest.fixture(scope="session")
def host_OA():
    with resources.as_file(resources.files("openfe.tests.data.host_guest.OA")) as d:
        fn = str(d / "OA_charged.sdf")
        rdmol = [m for m in Chem.SDMolSupplier(str(fn), removeHs=False)][0]
    return gufe.SmallMoleculeComponent(rdmol)
