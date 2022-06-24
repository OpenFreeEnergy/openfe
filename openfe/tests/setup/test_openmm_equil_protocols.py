# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import pytest
from openff.units import unit

from openfe import setup
from openfe.setup.methods import openmm

@pytest.fixture
def benzene_system(benzene_modifications):
    return setup.ChemicalSystem(
        {'ligand': benzene_modifications['benzene'],
         'solvent': setup.SolventComponent(
             positive_ion='Na', negative_ion='Cl',
             ion_concentration=0.15 * unit.molar)
        },
    )


@pytest.fixture
def benzene_complex_system(benzene_modifications, T4_protein_component):
    return setup.ChemicalSystem(
        {'ligand': benzene_modifications['benzene'],
         'solvent': setup.SolventComponent(
             positive_ion='Na', negative_ion='Cl',
             ion_concentration=0.15 * unit.molar),
         'protein': T4_protein_component,}
    )


@pytest.fixture
def toluene_system(benzene_modifications):
    return setup.ChemicalSystem(
        {'ligand': benzene_modifications['toluene'],
         'solvent': setup.SolventComponent(
             positive_ion='Na', negative_ion='Cl',
             ion_concentration=0.15 * unit.molar),
        },
    )


@pytest.fixture
def toluene_complex_system(benzene_modifications, T4_protein_component):
    return setup.ChemicalSystem(
        {'ligand': benzene_modifications['toluene'],
         'solvent': setup.SolventComponent(
             positive_ion='Na', negative_ion='Cl',
             ion_concentration=0.15 * unit.molar),
         'protein': T4_protein_component,}
    )


@pytest.fixture
def benzene_to_toluene_mapping(benzene_modifications):
    mapper = setup.atom_mapping.LomapAtomMapper()

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


def test_dry_run_ligand(benzene_system, toluene_system,
                        benzene_to_toluene_mapping, tmpdir):
    # this might be a bit time consuming
    protocol = openmm.RelativeLigandTransform(
            stateA=benzene_system,
            stateB=toluene_system,
            ligandmapping=benzene_to_toluene_mapping,
            settings=openmm.RelativeLigandTransform.get_default_settings(),
    )
    # Returns True if everything is OK
    with tmpdir.as_cwd():
        assert protocol.run(dry=True)


def test_dry_run_complex(benzene_complex_system, toluene_complex_system,
                         benzene_to_toluene_mapping, tmpdir):
    # this will be very time consuming
    protocol = openmm.RelativeLigandTransform(
            stateA=benzene_complex_system,
            stateB=toluene_complex_system,
            ligandmapping=benzene_to_toluene_mapping,
            settings=openmm.RelativeLigandTransform.get_default_settings(),
    )
    # Returns True if everything is OK
    with tmpdir.as_cwd():
        assert protocol.run(dry=True)


def test_missing_ligand(benzene_system, benzene_to_toluene_mapping):
    # state B doesn't have a ligand component
    stateB = setup.ChemicalSystem({'solvent': setup.SolventComponent()})

    with pytest.raises(ValueError, match='Missing ligand in state B'):
        _ = openmm.RelativeLigandTransform(
            stateA=benzene_system,
            stateB=stateB,
            ligandmapping=benzene_to_toluene_mapping,
            settings=openmm.RelativeLigandTransform.get_default_settings(),
        )


def test_missing_solvent(benzene_system, benzene_modifications,
                         benzene_to_toluene_mapping):
    # state B doesn't have a solvent component (i.e. its vacuum)
    stateB = setup.ChemicalSystem({'ligand': benzene_modifications['toluene']})

    with pytest.raises(ValueError, match="Missing solvent in state B"):
        _ = openmm.RelativeLigandTransform(
            stateA=benzene_system,
            stateB=stateB,
            ligandmapping=benzene_to_toluene_mapping,
            settings=openmm.RelativeLigandTransform.get_default_settings(),
        )


def test_incompatible_solvent(benzene_system, benzene_modifications,
                              benzene_to_toluene_mapping):
    # the solvents are different
    stateB = setup.ChemicalSystem(
        {'ligand': benzene_modifications['toluene'],
         'solvent': setup.SolventComponent(
             positive_ion='K', negative_ion='Cl')}
    )

    with pytest.raises(ValueError, match="Solvents aren't identical"):
        _ = openmm.RelativeLigandTransform(
            stateA=benzene_system,
            stateB=stateB,
            ligandmapping=benzene_to_toluene_mapping,
            settings=openmm.RelativeLigandTransform.get_default_settings(),
        )


def test_mapping_mismatch_A(benzene_system, toluene_system,
                            benzene_modifications):
    # the atom mapping doesn't refer to the ligands in the systems
    mapping = setup.atom_mapping.LigandAtomMapping(molA=benzene_system.components['ligand'],
                                      molB=benzene_modifications['phenol'],
                                      molA_to_molB=dict())

    with pytest.raises(ValueError,
                       match="Ligand in state B doesn't match mapping"):
        _ = openmm.RelativeLigandTransform(
            stateA=benzene_system,
            stateB=toluene_system,
            ligandmapping=mapping,
            settings=openmm.RelativeLigandTransform.get_default_settings(),
        )


def test_mapping_mismatch_B(benzene_system, toluene_system,
                            benzene_modifications):
    mapping = setup.atom_mapping.LigandAtomMapping(molA=benzene_modifications['phenol'],
                                      molB=toluene_system.components['ligand'],
                                      molA_to_molB=dict())

    with pytest.raises(ValueError,
                       match="Ligand in state A doesn't match mapping"):
        _ = openmm.RelativeLigandTransform(
            stateA=benzene_system,
            stateB=toluene_system,
            ligandmapping=mapping,
            settings=openmm.RelativeLigandTransform.get_default_settings(),
        )


def test_complex_mismatch(benzene_system, toluene_complex_system,
                          benzene_to_toluene_mapping):
    # only one complex
    with pytest.raises(ValueError):
        _ = openmm.RelativeLigandTransform(
            stateA=benzene_system,
            stateB=toluene_complex_system,
            ligandmapping=benzene_to_toluene_mapping,
            settings=openmm.RelativeLigandTransform.get_default_settings(),
        )


def test_protein_mismatch(benzene_complex_system, toluene_complex_system,
                          benzene_to_toluene_mapping):
    # hack one protein to be labelled differently
    prot = toluene_complex_system['protein']
    alt_prot = setup.ProteinComponent(prot._openmm_top, prot._openmm_pos,
                                      name='Mickey Mouse')
    alt_toluene_complex_system = setup.ChemicalSystem(
                 {'ligand': toluene_complex_system['ligand'],
                  'solvent': toluene_complex_system['solvent'],
                  'protein': alt_prot}
    )

    with pytest.raises(ValueError):
        _ = openmm.RelativeLigandTransform(
            stateA=benzene_complex_system,
            stateB=alt_toluene_complex_system,
            ligandmapping=benzene_to_toluene_mapping,
            settings=openmm.RelativeLigandTransform.get_default_settings(),
        )
