# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import gufe
import pytest
from unittest import mock
from openff.units import unit
from openff.units.openmm import ensure_quantity
from importlib import resources
import xml.etree.ElementTree as ET

from openmm import app, XmlSerializer
from openmmtools.multistate.multistatesampler import MultiStateSampler

from openfe import setup
from openfe.protocols import openmm_abfe
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from openff.units.openmm import ensure_quantity


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


def test_create_default_settings():
    settings = openmm_abfe.AbsoluteTransform.default_settings()

    assert settings


def test_create_default_protocol():
    # this is roughly how it should be created
    protocol = openmm_abfe.AbsoluteTransform(
        settings=openmm_abfe.AbsoluteTransform.default_settings(),
    )

    assert protocol


def test_serialize_protocol():
    protocol = openmm_abfe.AbsoluteTransform(
        settings=openmm_abfe.AbsoluteTransform.default_settings(),
    )

    ser = protocol.to_dict()

    ret = openmm_abfe.AbsoluteTransform.from_dict(ser)

    assert protocol == ret


@pytest.mark.parametrize('method', [
    'repex', 'sams', 'independent', 'InDePeNdENT'
])
def test_dry_run_default_vacuum(benzene_vacuum_system, method, tmpdir):
    vac_settings = openmm_abfe.AbsoluteTransform.default_settings()
    vac_settings.system_settings.nonbonded_method = 'nocutoff'
    vac_settings.sampler_settings.sampler_method = method
    vac_settings.sampler_settings.n_repeats = 1

    protocol = openmm_abfe.AbsoluteTransform(
            settings=vac_settings,
    )

    # create DAG from protocol and take first (and only) work unit from within
    dag = protocol.create(
        chem_system=benzene_vacuum_system,
    )
    unit = list(dag.protocol_units)[0]

    with tmpdir.as_cwd():
        assert isinstance(unit.run(dry=True)['debug']['sampler'],
                          MultiStateSampler)


#@pytest.mark.parametrize('method', ['repex', 'sams', 'independent'])
#def test_dry_run_ligand(benzene_system, toluene_system,
#                        benzene_to_toluene_mapping, method, tmpdir):
#    # this might be a bit time consuming
#    settings = openmm_rbfe.RelativeLigandTransform.default_settings()
#    settings.sampler_settings.sampler_method = method
#    settings.sampler_settings.n_repeats = 1
#
#    protocol = openmm_rbfe.RelativeLigandTransform(
#            settings=settings,
#    )
#    dag = protocol.create(
#        stateA=benzene_system,
#        stateB=toluene_system,
#        mapping={'ligand': benzene_to_toluene_mapping},
#    )
#    unit = list(dag.protocol_units)[0]
#
#    with tmpdir.as_cwd():
#        # Returns debug objects if everything is OK
#        assert isinstance(unit.run(dry=True)['debug']['sampler'],
#                          MultiStateSampler)
#
#
#@pytest.mark.parametrize('method', ['repex', 'sams', 'independent'])
#def test_dry_run_complex(benzene_complex_system, toluene_complex_system,
#                         benzene_to_toluene_mapping, method, tmpdir):
#    # this will be very time consuming
#    settings = openmm_rbfe.RelativeLigandTransform.default_settings()
#    settings.sampler_settings.sampler_method = method
#    settings.sampler_settings.n_repeats = 1
#
#    protocol = openmm_rbfe.RelativeLigandTransform(
#            settings=settings,
#    )
#    dag = protocol.create(
#        stateA=benzene_complex_system,
#        stateB=toluene_complex_system,
#        mapping={'ligand': benzene_to_toluene_mapping},
#    )
#    unit = list(dag.protocol_units)[0]
#
#    with tmpdir.as_cwd():
#        # Returns debug contents if everything is OK
#        assert isinstance(unit.run(dry=True)['debug']['sampler'],
#                          MultiStateSampler)
#
#
#def test_n_replicas_not_n_windows(benzene_vacuum_system,
#                                  toluene_vacuum_system,
#                                  benzene_to_toluene_mapping, tmpdir):
#    # For PR #125 we pin such that the number of lambda windows
#    # equals the numbers of replicas used - TODO: remove limitation
#    settings = openmm_rbfe.RelativeLigandTransform.default_settings()
#    # default lambda windows is 11
#    settings.sampler_settings.n_replicas = 13
#    settings.system_settings.nonbonded_method = 'nocutoff'
#
#    errmsg = ("Number of replicas 13 does not equal the number of "
#              "lambda windows 11")
#
#    with tmpdir.as_cwd():
#        with pytest.raises(ValueError, match=errmsg):
#            p = openmm_rbfe.RelativeLigandTransform(
#                    settings=settings,
#            )
#            dag = p.create(
#                stateA=benzene_vacuum_system,
#                stateB=toluene_vacuum_system,
#                mapping={'ligand': benzene_to_toluene_mapping},
#            )
#            unit = list(dag.protocol_units)[0]
#            unit.run(dry=True)
#
#
#def test_vaccuum_PME_error(benzene_system, benzene_modifications,
#                           benzene_to_toluene_mapping):
#    # state B doesn't have a solvent component (i.e. its vacuum)
#    stateB = setup.ChemicalSystem({'ligand': benzene_modifications['toluene']})
#
#    p = openmm_rbfe.RelativeLigandTransform(
#        settings=openmm_rbfe.RelativeLigandTransform.default_settings(),
#    )
#    errmsg = "PME cannot be used for vacuum transform"
#    with pytest.raises(ValueError, match=errmsg):
#        _ = p.create(
#            stateA=benzene_system,
#            stateB=stateB,
#            mapping={'ligand': benzene_to_toluene_mapping},
#        )
#
#
#@pytest.fixture
#def solvent_protocol_dag(benzene_system, toluene_system, benzene_to_toluene_mapping):
#    settings = openmm_rbfe.RelativeLigandTransform.default_settings()
#
#    protocol = openmm_rbfe.RelativeLigandTransform(
#        settings=settings,
#    )
#
#    return protocol.create(
#        stateA=benzene_system, stateB=toluene_system,
#        mapping={'ligand': benzene_to_toluene_mapping},
#    )
#
#
#def test_unit_tagging(solvent_protocol_dag, tmpdir):
#    # test that executing the Units includes correct generation and repeat info
#    units = solvent_protocol_dag.protocol_units
#    with mock.patch('openfe.protocols.openmm_rbfe.equil_rbfe_methods.RelativeLigandTransformUnit.run',
#                    return_value={'nc': 'file.nc', 'last_checkpoint': 'chk.nc'}):
#        results = []
#        for u in units:
#            ret = u.execute(shared=tmpdir)
#            results.append(ret)
#
#    repeats = set()
#    for ret in results:
#        assert isinstance(ret, gufe.ProtocolUnitResult)
#        assert ret.outputs['generation'] == 0
#        repeats.add(ret.outputs['repeat_id'])
#    assert repeats == {0, 1, 2}
#
#
#def test_gather(solvent_protocol_dag, tmpdir):
#    # check .gather behaves as expected
#    with mock.patch('openfe.protocols.openmm_rbfe.equil_rbfe_methods.RelativeLigandTransformUnit.run',
#                    return_value={'nc': 'file.nc', 'last_checkpoint': 'chk.nc'}):
#        dagres = gufe.protocols.execute_DAG(solvent_protocol_dag,
#                                            shared=tmpdir)
#
#    prot = openmm_rbfe.RelativeLigandTransform(
#        settings=openmm_rbfe.RelativeLigandTransform.default_settings()
#    )
#
#    with mock.patch('openfe.protocols.openmm_rbfe.equil_rbfe_methods.multistate') as m:
#        res = prot.gather([dagres])
#
#        # check we created the expected number of Reporters and Analyzers
#        assert m.MultiStateReporter.call_count == 3
#        m.MultiStateReporter.assert_called_with(
#            storage='file.nc', checkpoint_storage='chk.nc',
#        )
#        assert m.MultiStateSamplerAnalyzer.call_count == 3
#
#    assert isinstance(res, openmm_rbfe.RelativeLigandTransformResult)
