# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import gzip
import pytest
from importlib import resources
from rdkit import Chem
from rdkit.Geometry import Point3D
import openfe
from openff.units import unit
import pooch


@pytest.fixture
def benzene_vacuum_system(benzene_modifications):
    return openfe.ChemicalSystem(
        {'ligand': benzene_modifications['benzene']},
    )


@pytest.fixture(scope='session')
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


@pytest.fixture(scope='session')
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


@pytest.fixture(scope='session')
def benzene_to_toluene_mapping(benzene_modifications):
    mapper = openfe.setup.LomapAtomMapper(element_change=False)

    molA = benzene_modifications['benzene']
    molB = benzene_modifications['toluene']

    return next(mapper.suggest_mappings(molA, molB))


@pytest.fixture
def benzene_charges():
    files = {}
    with resources.files('openfe.tests.data.openmm_rfe') as d:
        fn = str(d / 'charged_benzenes.sdf')
        supp = Chem.SDMolSupplier(str(fn), removeHs=False)
        for rdmol in supp:
            files[rdmol.GetProp('_Name')] = openfe.SmallMoleculeComponent(rdmol)
    return files


@pytest.fixture
def benzene_to_benzoic_mapping(benzene_charges):
    mapper = openfe.setup.LomapAtomMapper(element_change=False)
    molA = benzene_charges['benzene']
    molB = benzene_charges['benzoic_acid']
    return next(mapper.suggest_mappings(molA, molB))


@pytest.fixture
def benzoic_to_benzene_mapping(benzene_charges):
    mapper = openfe.setup.LomapAtomMapper(element_change=False)
    molA = benzene_charges['benzoic_acid']
    molB = benzene_charges['benzene']
    return next(mapper.suggest_mappings(molA, molB))


@pytest.fixture
def benzene_to_aniline_mapping(benzene_charges):
    mapper = openfe.setup.LomapAtomMapper(element_change=False)
    molA = benzene_charges['benzene']
    molB = benzene_charges['aniline']
    return next(mapper.suggest_mappings(molA, molB))


@pytest.fixture
def aniline_to_benzene_mapping(benzene_charges):
    mapper = openfe.setup.LomapAtomMapper(element_change=False)
    molA = benzene_charges['aniline']
    molB = benzene_charges['benzene']
    return next(mapper.suggest_mappings(molA, molB))


@pytest.fixture
def aniline_to_benzoic_mapping(benzene_charges):
    mapper = openfe.setup.LomapAtomMapper(element_change=False)
    molA = benzene_charges['aniline']
    molB = benzene_charges['benzoic_acid']
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
def rfe_transformation_json() -> str:
    """string of a RFE results similar to quickrun

    generated with gen-serialized-results.py
    """
    d = resources.files('openfe.tests.data.openmm_rfe')

    with gzip.open((d / 'RHFEProtocol_json_results.gz').as_posix(), 'r') as f:  # type: ignore
        return f.read().decode()  # type: ignore


@pytest.fixture
def afe_solv_transformation_json() -> str:
    """
    string of a Absolute Solvation result (CN in water) generated by quickrun

    generated with gen-serialized-results.py
    """
    d = resources.files('openfe.tests.data.openmm_afe')
    fname = "AHFEProtocol_json_results.gz"
    
    with gzip.open((d / fname).as_posix(), 'r') as f:  # type: ignore
        return f.read().decode()  # type: ignore


@pytest.fixture
def md_json() -> str:
    """
    string of a MD result (TYK ligand lig_ejm_31  in water) generated by quickrun

    generated with gen-serialized-results.py
    """
    d = resources.files('openfe.tests.data.openmm_md')
    fname = "MDProtocol_json_results.gz"

    with gzip.open((d / fname).as_posix(), 'r') as f:  # type: ignore
        return f.read().decode()  # type: ignore


@pytest.fixture
def septop_json() -> str:
    """
    string of a SepTop result (BACE ligand lig_03 to lig_0) generated by quickrun

    generated with gen-serialized-results.py
    """
    d = resources.files('openfe.tests.data.openmm_septop')
    fname = "SepTopProtocol_json_results.gz"

    with gzip.open((d / fname).as_posix(), 'r') as f:  # type: ignore
        return f.read().decode()  # type: ignore


RFE_OUTPUT = pooch.create(
    path=pooch.os_cache("openfe_analysis"),
    base_url="doi:10.6084/m9.figshare.24101655",
    registry={
        "checkpoint.nc": "5af398cb14340fddf7492114998b244424b6c3f4514b2e07e4bd411484c08464",
        "db.json": "b671f9eb4daf9853f3e1645f9fd7c18150fd2a9bf17c18f23c5cf0c9fd5ca5b3",
        "hybrid_system.pdb": "07203679cb14b840b36e4320484df2360f45e323faadb02d6eacac244fddd517",
        "simulation.nc": "92361a0864d4359a75399470135f56642b72c605069a4c33dbc4be6f91f28b31",
        "simulation_real_time_analysis.yaml": "65706002f371fafba96037f29b054fd7e050e442915205df88567f48f5e5e1cf",
    }
)


@pytest.fixture
def simulation_nc():
    return RFE_OUTPUT.fetch("simulation.nc")
