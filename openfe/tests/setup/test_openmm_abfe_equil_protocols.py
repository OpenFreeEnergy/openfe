# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import pytest
from openmmtools.multistate.multistatesampler import MultiStateSampler
from openfe import ChemicalSystem, SolventComponent
from openfe.protocols import openmm_afe


@pytest.fixture
def vacuum_system():
    return ChemicalSystem({})


@pytest.fixture
def benzene_vacuum_system(benzene_modifications):
    return ChemicalSystem(
        {'ligand': benzene_modifications['benzene']},
    )


@pytest.fixture
def solvent_system():
    return ChemicalSystem(
        {'solvent': SolventComponent(), }
    )


@pytest.fixture
def benzene_solvent_system(benzene_modifications):
    return ChemicalSystem(
        {'ligand': benzene_modifications['benzene'],
         'solvent': SolventComponent(),
         },
    )


@pytest.fixture
def protein_system(benzene_modifications, T4_protein_component):
    return ChemicalSystem(
        {'solvent': SolventComponent(),
         'protein': T4_protein_component, }
    )


@pytest.fixture
def benzene_complex_system(benzene_modifications, T4_protein_component):
    return ChemicalSystem(
        {'ligand': benzene_modifications['benzene'],
         'solvent': SolventComponent(),
         'protein': T4_protein_component, }
    )


def test_create_default_settings():
    settings = openmm_afe.AbsoluteTransformProtocol.default_settings()
    assert settings


def test_create_default_protocol():
    # this is roughly how it should be created
    protocol = openmm_afe.AbsoluteTransformProtocol(
        settings=openmm_afe.AbsoluteTransformProtocol.default_settings(),
    )
    assert protocol


def test_serialize_protocol():
    protocol = openmm_afe.AbsoluteTransformProtocol(
        settings=openmm_afe.AbsoluteTransformProtocol.default_settings(),
    )

    ser = protocol.to_dict()
    ret = openmm_afe.AbsoluteTransformProtocol.from_dict(ser)
    assert protocol == ret


@pytest.mark.parametrize('method', [
    'repex', 'sams', 'independent', 'InDePeNdENT'
])
def test_dry_run_default_vacuum(benzene_vacuum_system, vacuum_system,
                                method, tmpdir):
    vac_settings = openmm_afe.AbsoluteTransformProtocol.default_settings()
    vac_settings.system_settings.nonbonded_method = 'nocutoff'
    vac_settings.alchemsampler_settings.sampler_method = method
    vac_settings.alchemsampler_settings.n_repeats = 1

    protocol = openmm_afe.AbsoluteTransformProtocol(
            settings=vac_settings,
    )

    # create DAG from protocol and take first (and only) work unit from within
    dag = protocol.create(
        stateA=benzene_vacuum_system,
        stateB=vacuum_system,
        mapping=None,
    )
    unit = list(dag.protocol_units)[0]

    with tmpdir.as_cwd():
        sampler = unit.run(dry=True)['debug']['sampler']
        assert isinstance(sampler, MultiStateSampler)
        assert not sampler.is_periodic


@pytest.mark.parametrize('method', ['repex', 'sams', 'independent'])
def test_dry_run_solvent(benzene_solvent_system, solvent_system, method, tmpdir):
    # this might be a bit time consuming
    settings = openmm_afe.AbsoluteTransformProtocol.default_settings()
    settings.alchemsampler_settings.sampler_method = method
    settings.alchemsampler_settings.n_repeats = 1

    protocol = openmm_afe.AbsoluteTransformProtocol(
            settings=settings,
    )
    dag = protocol.create(
        stateA=benzene_solvent_system,
        stateB=solvent_system,
        mapping=None,
    )
    unit = list(dag.protocol_units)[0]

    with tmpdir.as_cwd():
        sampler = unit.run(dry=True)['debug']['sampler']
        assert isinstance(sampler, MultiStateSampler)
        assert sampler.is_periodic


@pytest.mark.parametrize('method', ['repex', 'sams', 'independent'])
def test_dry_run_complex(benzene_complex_system, protein_system,
                         method, tmpdir):
    # this will be very time consuming
    settings = openmm_afe.AbsoluteTransformProtocol.default_settings()
    settings.alchemsampler_settings.sampler_method = method
    settings.alchemsampler_settings.n_repeats = 1

    protocol = openmm_afe.AbsoluteTransformProtocol(
            settings=settings,
    )
    dag = protocol.create(
        stateA=benzene_complex_system,
        stateB=protein_system,
        mapping=None,
    )
    unit = list(dag.protocol_units)[0]

    with tmpdir.as_cwd():
        sampler = unit.run(dry=True)['debug']['sampler']
        assert isinstance(sampler, MultiStateSampler)
        assert sampler.is_periodic


#def test_n_replicas_not_n_windows(benzene_vacuum_system, tmpdir):
#    # For PR #125 we pin such that the number of lambda windows
#    # equals the numbers of replicas used - TODO: remove limitation
#    settings = openmm_afe.AbsoluteTransform.default_settings()
#    # default lambda windows is 24
#    settings.sampler_settings.n_replicas = 13
#    settings.system_settings.nonbonded_method = 'nocutoff'
#
#    errmsg = "Number of replicas 13 does not equal"
#
#    with tmpdir.as_cwd():
#        with pytest.raises(ValueError, match=errmsg):
#            p = openmm_afe.AbsoluteTransform(
#                    settings=settings,
#            )
#            dag = p.create(
#                chem_system=benzene_vacuum_system,
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
