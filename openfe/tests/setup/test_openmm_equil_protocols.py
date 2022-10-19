# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import gufe
import pytest
from openff.units import unit
from openff.units.openmm import ensure_quantity

from openmm import app

from openfe import setup
from openfe.setup.methods import openmm
from openfe.setup import _rbfe_utils


@pytest.fixture
def benzene_vacuum_system(benzene_modifications):
    return setup.ChemicalSystem(
        {'ligand': benzene_modifications['benzene']},
    )


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
def toluene_vacuum_system(benzene_modifications):
    return setup.ChemicalSystem(
        {'ligand': benzene_modifications['toluene']},
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
    mapper = setup.LomapAtomMapper(element_change=False)

    molA = benzene_modifications['benzene']
    molB = benzene_modifications['toluene']

    return next(mapper.suggest_mappings(molA, molB))


def test_append_topology(benzene_complex_system, toluene_complex_system):
    mod = app.Modeller(
        benzene_complex_system['protein'].to_openmm_topology(),
        benzene_complex_system['protein'].to_openmm_positions(),
    )
    lig1 = benzene_complex_system['ligand'].to_openff()
    mod.add(
        lig1.to_topology().to_openmm(),
        ensure_quantity(lig1.conformers[0], 'openmm'),
    )

    top1 = mod.topology

    assert len(list(top1.atoms())) == 2625

    lig2 = toluene_complex_system['ligand'].to_openff()

    top2 = _rbfe_utils.topologyhelpers.combined_topology(
        top1, lig2.to_topology().to_openmm(),
        exclude_chains=list(top1.chains())[-1:],
    )

    assert len(list(top2.atoms())) == 2625 + 3  # added methyl


def test_create_default_settings():
    settings = openmm.RelativeLigandTransform.default_settings()

    assert settings


def test_create_default_protocol():
    # this is roughly how it should be created
    protocol = openmm.RelativeLigandTransform(
        settings=openmm.RelativeLigandTransform.default_settings(),
    )

    assert protocol


def test_serialize_protocol():
    protocol = openmm.RelativeLigandTransform(
        settings=openmm.RelativeLigandTransform.default_settings(),
    )

    ser = protocol.to_dict()

    ret = openmm.RelativeLigandTransform.from_dict(ser)

    assert protocol == ret


@pytest.mark.parametrize('method', [
    'repex', 'sams', 'independent', 'InDePeNdENT'
])
def test_dry_run_default_vacuum(benzene_vacuum_system, toluene_vacuum_system,
                                benzene_to_toluene_mapping, method, tmpdir):

    vac_settings = openmm.RelativeLigandTransform.default_settings()
    vac_settings.system_settings.nonbonded_method = 'nocutoff'
    vac_settings.sampler_settings.sampler_method = method
    vac_settings.sampler_settings.n_repeats = 1

    protocol = openmm.RelativeLigandTransform(
            settings=vac_settings,
    )

    # create DAG from protocol and take first (and only) work unit from within
    dag = protocol.create(
        stateA=benzene_vacuum_system,
        stateB=toluene_vacuum_system,
        mapping=benzene_to_toluene_mapping,
    )
    unit = list(dag.protocol_units)[0]

    with tmpdir.as_cwd():
        assert unit.run(dry=True) == {}


@pytest.mark.parametrize('method', ['repex', 'sams', 'independent'])
def test_dry_run_ligand(benzene_system, toluene_system,
                        benzene_to_toluene_mapping, method, tmpdir):
    # this might be a bit time consuming
    settings = openmm.RelativeLigandTransform.default_settings()
    settings.sampler_settings.sampler_method = method
    settings.sampler_settings.n_repeats = 1

    protocol = openmm.RelativeLigandTransform(
            settings=settings,
    )
    dag = protocol.create(
        stateA=benzene_system,
        stateB=toluene_system,
        mapping=benzene_to_toluene_mapping,
    )
    unit = list(dag.protocol_units)[0]
    # Returns True if everything is OK
    with tmpdir.as_cwd():
        assert unit.run(dry=True) == {}


@pytest.mark.parametrize('method', ['repex', 'sams', 'independent'])
def test_dry_run_complex(benzene_complex_system, toluene_complex_system,
                         benzene_to_toluene_mapping, method, tmpdir):
    # this will be very time consuming
    settings = openmm.RelativeLigandTransform.default_settings()
    settings.sampler_settings.sampler_method = method
    settings.sampler_settings.n_repeats = 1

    protocol = openmm.RelativeLigandTransform(
            settings=settings,
    )
    dag = protocol.create(
        stateA=benzene_complex_system,
        stateB=toluene_complex_system,
        mapping=benzene_to_toluene_mapping,
    )
    unit = list(dag.protocol_units)[0]
    # Returns True if everything is OK
    with tmpdir.as_cwd():
        assert unit.run(dry=True) == {}


def test_lambda_schedule_default():
    lambdas = _rbfe_utils.lambdaprotocol.LambdaProtocol(functions='default')
    assert len(lambdas.lambda_schedule) == 10


@pytest.mark.parametrize('windows', [11, 6, 9000])
def test_lambda_schedule(windows):
    lambdas = _rbfe_utils.lambdaprotocol.LambdaProtocol(
            functions='default', windows=windows)
    assert len(lambdas.lambda_schedule) == windows


def test_n_replicas_not_n_windows(benzene_vacuum_system,
                                  toluene_vacuum_system,
                                  benzene_to_toluene_mapping, tmpdir):
    # For PR #125 we pin such that the number of lambda windows
    # equals the numbers of replicas used - TODO: remove limitation
    settings = openmm.RelativeLigandTransform.default_settings()
    # default lambda windows is 11
    settings.sampler_settings.n_replicas = 13
    settings.system_settings.nonbonded_method = 'nocutoff'

    errmsg = ("Number of replicas 13 does not equal the number of "
              "lambda windows 11")

    with tmpdir.as_cwd():
        with pytest.raises(ValueError, match=errmsg):
            p = openmm.RelativeLigandTransform(
                    settings=settings,
            )
            dag = p.create(
                stateA=benzene_vacuum_system,
                stateB=toluene_vacuum_system,
                mapping=benzene_to_toluene_mapping,
            )
            unit = list(dag.protocol_units)[0]
            unit.run(dry=True)


def test_missing_ligand(benzene_system, benzene_to_toluene_mapping):
    # state B doesn't have a ligand component
    stateB = setup.ChemicalSystem({'solvent': setup.SolventComponent()})

    p = openmm.RelativeLigandTransform(
        settings=openmm.RelativeLigandTransform.default_settings(),
    )

    with pytest.raises(ValueError, match='Missing ligand in state B'):
        _ = p.create(
            stateA=benzene_system,
            stateB=stateB,
            mapping=benzene_to_toluene_mapping,
        )


def test_vaccuum_PME_error(benzene_system, benzene_modifications,
                           benzene_to_toluene_mapping):
    # state B doesn't have a solvent component (i.e. its vacuum)
    stateB = setup.ChemicalSystem({'ligand': benzene_modifications['toluene']})

    p = openmm.RelativeLigandTransform(
        settings=openmm.RelativeLigandTransform.default_settings(),
    )
    errmsg = "PME cannot be used for vacuum transform"
    with pytest.raises(ValueError, match=errmsg):
        _ = p.create(
            stateA=benzene_system,
            stateB=stateB,
            mapping=benzene_to_toluene_mapping,
        )


def test_incompatible_solvent(benzene_system, benzene_modifications,
                              benzene_to_toluene_mapping):
    # the solvents are different
    stateB = setup.ChemicalSystem(
        {'ligand': benzene_modifications['toluene'],
         'solvent': setup.SolventComponent(
             positive_ion='K', negative_ion='Cl')}
    )

    p = openmm.RelativeLigandTransform(
        settings=openmm.RelativeLigandTransform.default_settings(),
    )
    with pytest.raises(ValueError, match="Solvents aren't identical"):
        _ = p.create(
            stateA=benzene_system,
            stateB=stateB,
            mapping=benzene_to_toluene_mapping,
        )


def test_mapping_mismatch_A(benzene_system, toluene_system,
                            benzene_modifications):
    # the atom mapping doesn't refer to the ligands in the systems
    mapping = setup.LigandAtomMapping(molA=benzene_system.components['ligand'],
                                      molB=benzene_modifications['phenol'],
                                      molA_to_molB=dict())

    p = openmm.RelativeLigandTransform(
        settings=openmm.RelativeLigandTransform.default_settings(),
    )
    with pytest.raises(ValueError,
                       match="Ligand in state B doesn't match mapping"):
        _ = p.create(
            stateA=benzene_system,
            stateB=toluene_system,
            mapping=mapping,
        )


def test_mapping_mismatch_B(benzene_system, toluene_system,
                            benzene_modifications):
    mapping = setup.LigandAtomMapping(molA=benzene_modifications['phenol'],
                                      molB=toluene_system.components['ligand'],
                                      molA_to_molB=dict())

    p = openmm.RelativeLigandTransform(
        settings=openmm.RelativeLigandTransform.default_settings(),
    )
    with pytest.raises(ValueError,
                       match="Ligand in state A doesn't match mapping"):
        _ = p.create(
            stateA=benzene_system,
            stateB=toluene_system,
            mapping=mapping,
        )


def test_complex_mismatch(benzene_system, toluene_complex_system,
                          benzene_to_toluene_mapping):
    # only one complex
    p = openmm.RelativeLigandTransform(
        settings=openmm.RelativeLigandTransform.default_settings(),
    )
    with pytest.raises(ValueError):
        _ = p.create(
            stateA=benzene_system,
            stateB=toluene_complex_system,
            mapping=benzene_to_toluene_mapping,
        )


def test_protein_mismatch(benzene_complex_system, toluene_complex_system,
                          benzene_to_toluene_mapping):
    # hack one protein to be labelled differently
    prot = toluene_complex_system['protein']
    alt_prot = setup.ProteinComponent(prot.to_rdkit(),
                                      name='Mickey Mouse')
    alt_toluene_complex_system = setup.ChemicalSystem(
                 {'ligand': toluene_complex_system['ligand'],
                  'solvent': toluene_complex_system['solvent'],
                  'protein': alt_prot}
    )

    p = openmm.RelativeLigandTransform(
        settings=openmm.RelativeLigandTransform.default_settings(),
    )
    with pytest.raises(ValueError):
        _ = p.create(
            stateA=benzene_complex_system,
            stateB=alt_toluene_complex_system,
            mapping=benzene_to_toluene_mapping,
        )


def test_element_change_rejection(atom_mapping_basic_test_files):
    # check a mapping with element change gets rejected early
    l1 = atom_mapping_basic_test_files['2-methylnaphthalene']
    l2 = atom_mapping_basic_test_files['2-naftanol']

    mapper = setup.LomapAtomMapper()
    mapping = next(mapper.suggest_mappings(l1, l2))

    sys1 = setup.ChemicalSystem(
        {'ligand': l1, 'solvent': setup.SolventComponent()},
    )
    sys2 = setup.ChemicalSystem(
        {'ligand': l2, 'solvent': setup.SolventComponent()},
    )

    p = openmm.RelativeLigandTransform(
        settings=openmm.RelativeLigandTransform.default_settings(),
    )
    with pytest.raises(ValueError, match="Element change"):
        _ = p.create(
            stateA=sys1, stateB=sys2,
            mapping=mapping,
        )
