# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import pytest

from openfe import setup
from openfe.setup.methods import openmm

@pytest.fixture
def benzene_system(benzene_modifications):
    return setup.ChemicalState(
        {'ligand': benzene_modifications['benzene'],
         'solvent': setup.SolventComponent(ions=('Na', 'Cl'))},
    )


@pytest.fixture
def toluene_system(benzene_modifications):
    return setup.ChemicalState(
        {'ligand': benzene_modifications['toluene'],
         'solvent': setup.SolventComponent(ions=('Na', 'Cl'))},
    )


@pytest.fixture
def benzene_to_toluene_mapping(benzene_modifications):
    mapper = setup.LomapAtomMapper()

    molA = benzene_modifications['benzene']
    molB = benzene_modifications['toluene']

    return next(mapper.suggest_mappings(molA, molB))


def test_create_default_settings():
    settings = openmm.RelativeLigandTransform.get_default_settings()

    assert settings


def test_create_protocol(benzene_system, toluene_system,
                         benzene_to_toluene_mapping):
    # this is roughly how it should be created
    protocol = openmm.RelativeLigandTransform(
        stateA=benzene_system,
        stateB=toluene_system,
        ligandmapping=benzene_to_toluene_mapping,
        settings=openmm.RelativeLigandTransform.get_default_settings(),
    )

    assert protocol


def test_missing_ligand(benzene_system, benzene_to_toluene_mapping):
    stateB = setup.ChemicalState({'solvent': setup.SolventComponent()})

    with pytest.raises(ValueError, match='Missing ligand in state B'):
        _ = openmm.RelativeLigandTransform(
            stateA=benzene_system,
            stateB=stateB,
            ligandmapping=benzene_to_toluene_mapping,
            settings=openmm.RelativeLigandTransform.get_default_settings(),
        )


def test_missing_solvent(benzene_system, benzene_modifications,
                         benzene_to_toluene_mapping):
    stateB = setup.ChemicalState({'ligand': benzene_modifications['toluene']})

    with pytest.raises(ValueError, match="Missing solvent in state B"):
        _ = openmm.RelativeLigandTransform(
            stateA=benzene_system,
            stateB=stateB,
            ligandmapping=benzene_to_toluene_mapping,
            settings=openmm.RelativeLigandTransform.get_default_settings(),
        )


def test_incompatible_solvent(benzene_system, benzene_modifications,
                              benzene_to_toluene_mapping):
    stateB = setup.ChemicalState(
        {'ligand': benzene_modifications['toluene'],
         'solvent': setup.SolventComponent(ions=('K', 'Cl'))},
    )

    with pytest.raises(ValueError, match="Solvents aren't identical"):
        _ = openmm.RelativeLigandTransform(
            stateA=benzene_system,
            stateB=stateB,
            ligandmapping=benzene_to_toluene_mapping,
            settings=openmm.RelativeLigandTransform.get_default_settings(),
        )
