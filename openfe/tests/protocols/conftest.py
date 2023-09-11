# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import gzip
import pytest
from importlib import resources
from rdkit import Chem
from rdkit.Geometry import Point3D
import openfe
from openff.units import unit


@pytest.fixture
def benzene_vacuum_system(benzene_modifications):
    return openfe.ChemicalSystem(
        {'ligand': benzene_modifications['benzene']},
    )


@pytest.fixture
def benzene_system(benzene_modifications):
    return openfe.ChemicalSystem(
        {'ligand': benzene_modifications['benzene'],
         'solvent': openfe.SolventComponent(
             positive_ion='Na', negative_ion='Cl',
             ion_concentration=0.15 * unit.molar)
        },
    )


@pytest.fixture
def benzene_complex_system(benzene_modifications, T4_protein_component):
    return openfe.ChemicalSystem(
        {'ligand': benzene_modifications['benzene'],
         'solvent': openfe.SolventComponent(
             positive_ion='Na', negative_ion='Cl',
             ion_concentration=0.15 * unit.molar),
         'protein': T4_protein_component,}
    )


@pytest.fixture
def toluene_vacuum_system(benzene_modifications):
    return openfe.ChemicalSystem(
        {'ligand': benzene_modifications['toluene']},
    )


@pytest.fixture
def toluene_system(benzene_modifications):
    return openfe.ChemicalSystem(
        {'ligand': benzene_modifications['toluene'],
         'solvent': openfe.SolventComponent(
             positive_ion='Na', negative_ion='Cl',
             ion_concentration=0.15 * unit.molar),
        },
    )


@pytest.fixture
def toluene_complex_system(benzene_modifications, T4_protein_component):
    return openfe.ChemicalSystem(
        {'ligand': benzene_modifications['toluene'],
         'solvent': openfe.SolventComponent(
             positive_ion='Na', negative_ion='Cl',
             ion_concentration=0.15 * unit.molar),
         'protein': T4_protein_component,}
    )


@pytest.fixture
def benzene_to_toluene_mapping(benzene_modifications):
    mapper = openfe.setup.LomapAtomMapper(element_change=False)

    molA = benzene_modifications['benzene']
    molB = benzene_modifications['toluene']

    return next(mapper.suggest_mappings(molA, molB))


@pytest.fixture
def benzene_many_solv_system(benzene_modifications):

    rdmol_phenol = benzene_modifications['phenol'].to_rdkit()
    rdmol_benzo = benzene_modifications['benzonitrile'].to_rdkit()

    conf_phenol = rdmol_phenol.GetConformer()
    conf_benzo = rdmol_benzo.GetConformer()

    for atm in range(rdmol_phenol.GetNumAtoms()):
        x, y, z = conf_phenol.GetAtomPosition(atm)
        conf_phenol.SetAtomPosition(atm, Point3D(x+30, y, z))

    for atm in range(rdmol_benzo.GetNumAtoms()):
        x, y, z = conf_benzo.GetAtomPosition(atm)
        conf_benzo.SetAtomPosition(atm, Point3D(x, y+30, z))

    phenol = openfe.SmallMoleculeComponent.from_rdkit(
        rdmol_phenol, name='phenol'
    )

    benzo = openfe.SmallMoleculeComponent.from_rdkit(
        rdmol_benzo, name='benzonitrile'
    )

    return openfe.ChemicalSystem(
        {'whatligand': benzene_modifications['benzene'],
         "foo": phenol,
         "bar": benzo,
         "solvent": openfe.SolventComponent()},
    )


@pytest.fixture
def toluene_many_solv_system(benzene_modifications):

    rdmol_phenol = benzene_modifications['phenol'].to_rdkit()
    rdmol_benzo = benzene_modifications['benzonitrile'].to_rdkit()

    conf_phenol = rdmol_phenol.GetConformer()
    conf_benzo = rdmol_benzo.GetConformer()

    for atm in range(rdmol_phenol.GetNumAtoms()):
        x, y, z = conf_phenol.GetAtomPosition(atm)
        conf_phenol.SetAtomPosition(atm, Point3D(x+30, y, z))

    for atm in range(rdmol_benzo.GetNumAtoms()):
        x, y, z = conf_benzo.GetAtomPosition(atm)
        conf_benzo.SetAtomPosition(atm, Point3D(x, y+30, z))

    phenol = openfe.SmallMoleculeComponent.from_rdkit(
        rdmol_phenol, name='phenol'
    )

    benzo = openfe.SmallMoleculeComponent.from_rdkit(
        rdmol_benzo, name='benzonitrile'
    )
    return openfe.ChemicalSystem(
        {'whatligand': benzene_modifications['toluene'],
         "foo": phenol,
         "bar": benzo,
         "solvent": openfe.SolventComponent()},
    )


@pytest.fixture
def transformation_json() -> str:
    """string of a result of quickrun"""
    d = resources.files('openfe.tests.data.openmm_rfe')

    with gzip.open((d / 'vac_results.json.gz').as_posix(), 'r') as f:  # type: ignore
        return f.read().decode()  # type: ignore
