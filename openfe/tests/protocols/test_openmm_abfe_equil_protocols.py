# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import pytest
from unittest import mock
from openmmtools.multistate.multistatesampler import MultiStateSampler
from openff.units import unit as offunit
import gufe
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


@pytest.fixture
def vacuum_protocol_dag(benzene_vacuum_system, vacuum_system):
    settings = openmm_afe.AbsoluteTransformProtocol.default_settings()
    protocol = openmm_afe.AbsoluteTransformProtocol(settings=settings)
    return protocol.create(stateA=benzene_vacuum_system, stateB=vacuum_system,
                           mapping=None)


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


def test_get_alchemical_components(benzene_modifications,
                                   T4_protein_component):
    stateA = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'toluene': benzene_modifications['toluene'],
        'protein': T4_protein_component,
        'solvent': SolventComponent(neutralize=False)
    })

    stateB = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'phenol': benzene_modifications['phenol'],
        'solvent': SolventComponent(),
    })

    func = openmm_afe.AbsoluteTransformProtocol._get_alchemical_components

    comps = func(stateA, stateB)

    assert len(comps['stateA']) == 3
    assert len(comps['stateB']) == 2
    
    for i in ['toluene', 'protein', 'solvent']:
        assert i in comps['stateA']

    for i in ['phenol', 'solvent']:
        assert i in comps['stateB']


def test_validate_alchem_comps_stateB():

    func = openmm_afe.AbsoluteTransformProtocol._validate_alchemical_components

    stateA = ChemicalSystem({})
    alchem_comps = {'stateA': [], 'stateB': ['foo', 'bar']}
    with pytest.raises(ValueError, match="Components appearing in state B"):
        func(stateA, alchem_comps)


@pytest.mark.parametrize('key', ['protein', 'solvent'])
def test_validate_alchem_comps_non_small(key, T4_protein_component):

    func = openmm_afe.AbsoluteTransformProtocol._validate_alchemical_components

    stateA = ChemicalSystem({
        'protein': T4_protein_component,
        'solvent': SolventComponent(),
    })

    alchem_comps = {'stateA': [key,], 'stateB': []}
    with pytest.raises(ValueError, match='Non SmallMoleculeComponent'):
        func(stateA, alchem_comps)


def test_validate_solvent_vacuum():

    state = ChemicalSystem({'solvent': SolventComponent()})

    func = openmm_afe.AbsoluteTransformProtocol._validate_solvent

    with pytest.raises(ValueError, match="cannot be used for solvent"):
        func(state, 'NoCutoff')


def test_validate_solvent_double_solvent():

    state = ChemicalSystem({
        'solvent': SolventComponent(),
        'solvent-two': SolventComponent(neutralize=False)
    })

    func = openmm_afe.AbsoluteTransformProtocol._validate_solvent

    with pytest.raises(ValueError, match="only one is supported"):
        func(state, 'pme')


def test_parse_components_expected(T4_protein_component,
                                   benzene_modifications):

    func = openmm_afe.AbsoluteTransformUnit._parse_components

    chem = ChemicalSystem({
        'protein': T4_protein_component,
        'benzene': benzene_modifications['benzene'],
        'solvent': SolventComponent(),
        'toluene': benzene_modifications['toluene'],
        'phenol': benzene_modifications['phenol'],
    })

    solvent_comp, protein_comp, off_small_mols = func(chem)

    assert len(off_small_mols.keys()) == 3
    assert protein_comp == T4_protein_component
    assert solvent_comp == SolventComponent()

    for i in ['benzene', 'toluene', 'phenol']:
        off_small_mols[i] == benzene_modifications[i].to_openff()


def test_parse_components_multi_protein(T4_protein_component):

    func = openmm_afe.AbsoluteTransformUnit._parse_components

    chem = ChemicalSystem({
        'protein': T4_protein_component,
        'protien2': T4_protein_component,  # should this even be allowed?
    })

    with pytest.raises(ValueError, match="Multiple proteins"):
        func(chem)


def test_simstep_return():

    func = openmm_afe.AbsoluteTransformUnit._get_sim_steps

    steps = func(time=250000 * offunit.femtoseconds,
                 timestep=4 * offunit.femtoseconds,
                 mc_steps=250)

    # check the number of steps for a 250 ps simulations
    assert steps == 62500


def test_simstep_undivisible_mcsteps():

    func = openmm_afe.AbsoluteTransformUnit._get_sim_steps

    with pytest.raises(ValueError, match="divisible by the number"):
        func(time=780 * offunit.femtoseconds,
             timestep=4 * offunit.femtoseconds,
             mc_steps=100)


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


def test_nreplicas_lambda_mismatch(benzene_vacuum_system,
                                   vacuum_system, tmpdir):
    """
    For now, should trigger failure if there are not as many replicas
    as there are summed lambda windows.
    """
    settings = openmm_afe.AbsoluteTransformProtocol.default_settings()
    settings.alchemsampler_settings.n_replicas = 12
    settings.alchemical_settings.lambda_elec_windows = 12
    settings.alchemical_settings.lambda_vdw_windows = 12

    protocol = openmm_afe.AbsoluteTransformProtocol(
            settings=settings,
    )

    dag = protocol.create(
            stateA=benzene_vacuum_system,
            stateB=vacuum_system,
            mapping=None,
    )
    unit = list(dag.protocol_units)[0]

    with tmpdir.as_cwd():
        errmsg = ("Number of replicas 12 does not equal the "
                  "number of lambda windows")
        with pytest.raises(ValueError, match=errmsg):
            unit.run(dry=True)


def test_unit_tagging(vacuum_protocol_dag, tmpdir):
    units = vacuum_protocol_dag.protocol_units
    with mock.patch('openfe.protocols.openmm_afe.equil_afe_methods.AbsoluteTransformUnit.run',
                    return_value={'nc': 'file.nc', 'last_checkpoint': 'chck.nc'}):
        results = []
        for u in units:
            ret = u.execute(shared=tmpdir)
            results.append(ret)

    repeats = set()
    for ret in results:
        assert isinstance(ret, gufe.ProtocolUnitResult)
        assert ret.outputs['generation'] == 0
        repeats.add(ret.outputs['repeat_id'])
    assert repeats == {0, 1, 2}


def test_gather(vacuum_protocol_dag, tmpdir):
    base_import = 'openfe.protocols.openmm_afe.equil_afe_methods.'
    with mock.patch(f"{base_import}AbsoluteTransformUnit.run",
                    return_value={'nc': 'file.nc', 'last_checkpoint': 'chck.nc'}):
        dagres = gufe.protocols.execute_DAG(vacuum_protocol_dag, shared=tmpdir)

    prot = openmm_afe.AbsoluteTransformProtocol(
            settings=openmm_afe.AbsoluteTransformProtocol.default_settings(),
    )

    with mock.patch(f"{base_import}multistate") as m:
        res = prot.gather([dagres])

        # check we created the expected number of Reporters and Analyzers
        assert m.MultiStateReporter.call_count == 3
        m.MultiStateReporter.assert_called_with(
                storage='file.nc', checkpoint_storage='chck.nc',
        )
        assert m.MultiStateSamplerAnalyzer.call_count == 3

    assert isinstance(res, openmm_afe.AbsoluteTransformProtocolResult)

