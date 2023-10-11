# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import pytest
from unittest import mock
from openmmtools.multistate.multistatesampler import MultiStateSampler
from openff.units import unit as offunit
import mdtraj as mdt
import gufe
from openfe import ChemicalSystem, SolventComponent
from openfe.protocols import openmm_afe
from openfe.protocols.openmm_afe import (
    AbsoluteSolventTransformUnit, AbsoluteVacuumTransformUnit
)
from openfe.protocols.openmm_utils import system_validation


@pytest.fixture()
def default_settings():
    return openmm_afe.AbsoluteSolvationProtocol.default_settings()


def test_create_default_settings():
    settings = openmm_afe.AbsoluteSolvationProtocol.default_settings()
    assert settings


@pytest.mark.parametrize('val', [
    {'elec': 0, 'vdw': 5},
    {'elec': -2, 'vdw': 5},
    {'elec': 5, 'vdw': -2},
    {'elec': 5, 'vdw': 0},
])
def test_incorrect_window_settings(val, default_settings):
    errmsg = "lambda steps must be positive"
    alchem_settings = default_settings.alchemical_settings
    with pytest.raises(ValueError, match=errmsg):
        alchem_settings.lambda_elec_windows = val['elec']
        alchem_settings.lambda_vdw_windows = val['vdw']


def test_create_default_protocol(default_settings):
    # this is roughly how it should be created
    protocol = openmm_afe.AbsoluteSolvationProtocol(
        settings=default_settings,
    )
    assert protocol


def test_serialize_protocol(default_settings):
    protocol = openmm_afe.AbsoluteSolvationProtocol(
        settings=default_settings,
    )

    ser = protocol.to_dict()
    ret = openmm_afe.AbsoluteSolvationProtocol.from_dict(ser)
    assert protocol == ret


def test_validate_solvent_endstates_protcomp(
    benzene_modifications,T4_protein_component
):
    stateA = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'protein': T4_protein_component,
        'solvent': SolventComponent()
    })

    stateB = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'phenol': benzene_modifications['phenol'],
        'solvent': SolventComponent(),
    })

    func = openmm_afe.AbsoluteSolvationProtocol._validate_solvent_endstates

    with pytest.raises(ValueError, match="Protein components are not allowed"):
        comps = func(stateA, stateB)


def test_validate_solvent_endstates_nosolvcomp_stateA(
    benzene_modifications, T4_protein_component
):
    stateA = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
    })

    stateB = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'phenol': benzene_modifications['phenol'],
        'solvent': SolventComponent(),
    })

    func = openmm_afe.AbsoluteSolvationProtocol._validate_solvent_endstates

    with pytest.raises(
        ValueError, match="No SolventComponent found in stateA"
    ):
        comps = func(stateA, stateB)


def test_validate_solvent_endstates_nosolvcomp_stateB(
    benzene_modifications, T4_protein_component
):
    stateA = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'solvent': SolventComponent(),
    })

    stateB = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'phenol': benzene_modifications['phenol'],
    })

    func = openmm_afe.AbsoluteSolvationProtocol._validate_solvent_endstates

    with pytest.raises(
        ValueError, match="No SolventComponent found in stateB"
    ):
        comps = func(stateA, stateB)

def test_validate_alchem_comps_appearingB(benzene_modifications):
    stateA = ChemicalSystem({
        'solvent': SolventComponent()
    })

    stateB = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'solvent': SolventComponent()
    })

    alchem_comps = system_validation.get_alchemical_components(stateA, stateB)

    func = openmm_afe.AbsoluteSolvationProtocol._validate_alchemical_components

    with pytest.raises(ValueError, match='Components appearing in state B'):
        func(alchem_comps)


def test_validate_alchem_comps_multi(benzene_modifications):
    stateA = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'toluene': benzene_modifications['toluene'],
        'solvent': SolventComponent()
    })

    stateB = ChemicalSystem({
        'solvent': SolventComponent()
    })

    alchem_comps = system_validation.get_alchemical_components(stateA, stateB)

    assert len(alchem_comps['stateA']) == 2

    func = openmm_afe.AbsoluteSolvationProtocol._validate_alchemical_components

    with pytest.raises(ValueError, match='More than one alchemical'):
        func(alchem_comps)


def test_validate_alchem_nonsmc(benzene_modifications):
    stateA = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'solvent': SolventComponent()
    })

    stateB = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
    })

    alchem_comps = system_validation.get_alchemical_components(stateA, stateB)

    func = openmm_afe.AbsoluteSolvationProtocol._validate_alchemical_components

    with pytest.raises(ValueError, match='Non SmallMoleculeComponent'):
        func(alchem_comps)


def test_vac_bad_nonbonded(benzene_modifications):
    settings = openmm_afe.AbsoluteSolvationProtocol.default_settings()
    settings.vacuum_system_settings.nonbonded_method = 'pme'
    protocol = openmm_afe.AbsoluteSolvationProtocol(settings=settings)


    stateA = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'solvent': SolventComponent()
    })

    stateB = ChemicalSystem({
        'solvent': SolventComponent(),
    })


    with pytest.raises(ValueError, match='Only the nocutoff'):
        protocol.create(stateA=stateA, stateB=stateB, mapping=None)


@pytest.mark.parametrize('method', [
    'repex', 'sams', 'independent', 'InDePeNdENT'
])
def test_dry_run_vac_benzene(benzene_modifications,
                             method, tmpdir):
    s = openmm_afe.AbsoluteSolvationProtocol.default_settings()
    s.alchemsampler_settings.n_repeats = 1
    s.alchemsampler_settings.sampler_method = method

    protocol = openmm_afe.AbsoluteSolvationProtocol(
            settings=s,
    )

    stateA = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'solvent': SolventComponent()
    })

    stateB = ChemicalSystem({
        'solvent': SolventComponent(),
    })

    # create DAG from protocol and take first (and only) work unit from within
    dag = protocol.create(
        stateA=stateA,
        stateB=stateB,
        mapping=None,
    )
    prot_units = list(dag.protocol_units)
    
    assert len(prot_units) == 2

    vac_unit = [u for u in prot_units
                if isinstance(u, AbsoluteVacuumTransformUnit)]
    sol_unit = [u for u in prot_units
                if isinstance(u, AbsoluteSolventTransformUnit)]

    assert len(vac_unit) == 1
    assert len(sol_unit) == 1

    with tmpdir.as_cwd():
        vac_sampler = vac_unit[0].run(dry=True)['debug']['sampler']
        assert not vac_sampler.is_periodic


def test_dry_run_solv_benzene(benzene_modifications, tmpdir):
    s = openmm_afe.AbsoluteSolvationProtocol.default_settings()
    s.alchemsampler_settings.n_repeats = 1
    s.solvent_simulation_settings.output_indices = "resname UNK"

    protocol = openmm_afe.AbsoluteSolvationProtocol(
            settings=s,
    )

    stateA = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'solvent': SolventComponent()
    })

    stateB = ChemicalSystem({
        'solvent': SolventComponent(),
    })

    # create DAG from protocol and take first (and only) work unit from within
    dag = protocol.create(
        stateA=stateA,
        stateB=stateB,
        mapping=None,
    )
    prot_units = list(dag.protocol_units)

    assert len(prot_units) == 2

    vac_unit = [u for u in prot_units
                if isinstance(u, AbsoluteVacuumTransformUnit)]
    sol_unit = [u for u in prot_units
                if isinstance(u, AbsoluteSolventTransformUnit)]

    assert len(vac_unit) == 1
    assert len(sol_unit) == 1

    with tmpdir.as_cwd():
        sol_sampler = sol_unit[0].run(dry=True)['debug']['sampler']
        assert sol_sampler.is_periodic

        pdb = mdt.load_pdb('hybrid_system.pdb')
        assert pdb.n_atoms == 12


def test_dry_run_solv_benzene_tip4p(benzene_modifications, tmpdir):
    s = openmm_afe.AbsoluteSolvationProtocol.default_settings()
    s.alchemsampler_settings.n_repeats = 1
    s.forcefield_settings.forcefields = [
        "amber/ff14SB.xml",    # ff14SB protein force field
        "amber/tip4pew_standard.xml", # FF we are testsing with the fun VS
        "amber/phosaa10.xml",  # Handles THE TPO
    ]
    s.solvation_settings.solvent_model = 'tip4pew'
    s.integrator_settings.reassign_velocities = True

    protocol = openmm_afe.AbsoluteSolvationProtocol(
            settings=s,
    )

    stateA = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'solvent': SolventComponent()
    })

    stateB = ChemicalSystem({
        'solvent': SolventComponent(),
    })

    # create DAG from protocol and take first (and only) work unit from within
    dag = protocol.create(
        stateA=stateA,
        stateB=stateB,
        mapping=None,
    )
    prot_units = list(dag.protocol_units)

    sol_unit = [u for u in prot_units
                if isinstance(u, AbsoluteSolventTransformUnit)]

    with tmpdir.as_cwd():
        sol_sampler = sol_unit[0].run(dry=True)['debug']['sampler']
        assert sol_sampler.is_periodic


def test_nreplicas_lambda_mismatch(benzene_modifications, tmpdir):
    s = openmm_afe.AbsoluteSolvationProtocol.default_settings()
    s.alchemsampler_settings.n_repeats = 1
    s.alchemsampler_settings.n_replicas = 12

    protocol = openmm_afe.AbsoluteSolvationProtocol(
            settings=s,
    )

    stateA = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'solvent': SolventComponent()
    })

    stateB = ChemicalSystem({
        'solvent': SolventComponent(),
    })

    # create DAG from protocol and take first (and only) work unit from within
    dag = protocol.create(
        stateA=stateA,
        stateB=stateB,
        mapping=None,
    )
    prot_units = list(dag.protocol_units)

    with tmpdir.as_cwd():
        errmsg = "Number of replicas 12"
        with pytest.raises(ValueError, match=errmsg):
            prot_units[0].run(dry=True)


def test_high_timestep(benzene_modifications, tmpdir):
    s = openmm_afe.AbsoluteSolvationProtocol.default_settings()
    s.alchemsampler_settings.n_repeats = 1
    s.forcefield_settings.hydrogen_mass = 1.0

    protocol = openmm_afe.AbsoluteSolvationProtocol(
            settings=s,
    )

    stateA = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'solvent': SolventComponent()
    })

    stateB = ChemicalSystem({
        'solvent': SolventComponent(),
    })

    # create DAG from protocol and take first (and only) work unit from within
    dag = protocol.create(
        stateA=stateA,
        stateB=stateB,
        mapping=None,
    )
    prot_units = list(dag.protocol_units)

    with tmpdir.as_cwd():
        errmsg = "too large for hydrogen mass"
        with pytest.raises(ValueError, match=errmsg):
            prot_units[0].run(dry=True)


@pytest.fixture
def benzene_solvation_dag(benzene_modifications):
    s = openmm_afe.AbsoluteSolvationProtocol.default_settings()

    protocol = openmm_afe.AbsoluteSolvationProtocol(
            settings=s,
    )

    stateA = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'solvent': SolventComponent()
    })

    stateB = ChemicalSystem({
        'solvent': SolventComponent(),
    })

    return protocol.create(stateA=stateA, stateB=stateB, mapping=None)


def test_unit_tagging(benzene_solvation_dag, tmpdir):
    # test that executing the units includes correct gen and repeat info

    dag_units = benzene_solvation_dag.protocol_units

    with (
        mock.patch('openfe.protocols.openmm_afe.equil_solvation_afe_method.AbsoluteSolventTransformUnit.run',
                   return_value={'nc': 'file.nc', 'last_checkpoint': 'chck.nc'}),
        mock.patch('openfe.protocols.openmm_afe.equil_solvation_afe_method.AbsoluteVacuumTransformUnit.run',
                   return_value={'nc': 'file.nc', 'last_checkpoint': 'chck.nc'}),
    ):
        results = []
        for u in dag_units:
            ret = u.execute(context=gufe.Context(tmpdir, tmpdir))
            results.append(ret)

    solv_repeats = set()
    vac_repeats = set()
    for ret in results:
        assert isinstance(ret, gufe.ProtocolUnitResult)
        assert ret.outputs['generation'] == 0
        if ret.outputs['simtype'] == 'vacuum':
            vac_repeats.add(ret.outputs['repeat_id'])
        else:
            solv_repeats.add(ret.outputs['repeat_id'])
    assert vac_repeats == {0, 1, 2}
    assert solv_repeats == {0, 1, 2}


def test_gather(benzene_solvation_dag, tmpdir):
    # check that .gather behaves as expected
    with (
        mock.patch('openfe.protocols.openmm_afe.equil_solvation_afe_method.AbsoluteSolventTransformUnit.run',
                   return_value={'nc': 'file.nc', 'last_checkpoint': 'chck.nc'}),
        mock.patch('openfe.protocols.openmm_afe.equil_solvation_afe_method.AbsoluteVacuumTransformUnit.run',
                   return_value={'nc': 'file.nc', 'last_checkpoint': 'chck.nc'}),
    ):
        dagres = gufe.protocols.execute_DAG(benzene_solvation_dag,
                                            shared_basedir=tmpdir,
                                            scratch_basedir=tmpdir,
                                            keep_shared=True)

    protocol = openmm_afe.AbsoluteSolvationProtocol(
        settings=openmm_afe.AbsoluteSolvationProtocol.default_settings(),
    )

    res = protocol.gather([dagres])

    assert isinstance(res, openmm_afe.AbsoluteSolvationProtocolResult)
