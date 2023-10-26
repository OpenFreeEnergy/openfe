# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import itertools
import json
import pytest
from unittest import mock
from openmm import NonbondedForce, CustomNonbondedForce
from openmmtools.multistate.multistatesampler import MultiStateSampler
from openff.units import unit as offunit
from openff.units.openmm import ensure_quantity
import mdtraj as mdt
import numpy as np
import gufe
import openfe
from openfe import ChemicalSystem, SolventComponent
from openfe.protocols import openmm_afe
from openfe.protocols.openmm_afe import (
    AbsoluteSolvationSolventUnit,
    AbsoluteSolvationVacuumUnit,
    AbsoluteSolvationProtocol,
)
from openfe.protocols.openmm_utils import system_validation


@pytest.fixture()
def default_settings():
    return AbsoluteSolvationProtocol.default_settings()


def test_create_default_settings():
    settings = AbsoluteSolvationProtocol.default_settings()
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
    protocol = AbsoluteSolvationProtocol(
        settings=default_settings,
    )
    assert protocol


def test_serialize_protocol(default_settings):
    protocol = AbsoluteSolvationProtocol(
        settings=default_settings,
    )

    ser = protocol.to_dict()
    ret = AbsoluteSolvationProtocol.from_dict(ser)
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

    with pytest.raises(ValueError, match="Protein components are not allowed"):
        comps = AbsoluteSolvationProtocol._validate_solvent_endstates(stateA, stateB)


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

    with pytest.raises(
        ValueError, match="No SolventComponent found in stateA"
    ):
        comps = AbsoluteSolvationProtocol._validate_solvent_endstates(stateA, stateB)


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

    with pytest.raises(
        ValueError, match="No SolventComponent found in stateB"
    ):
        comps = AbsoluteSolvationProtocol._validate_solvent_endstates(stateA, stateB)

def test_validate_alchem_comps_appearingB(benzene_modifications):
    stateA = ChemicalSystem({
        'solvent': SolventComponent()
    })

    stateB = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'solvent': SolventComponent()
    })

    alchem_comps = system_validation.get_alchemical_components(stateA, stateB)

    with pytest.raises(ValueError, match='Components appearing in state B'):
        AbsoluteSolvationProtocol._validate_alchemical_components(alchem_comps)


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

    with pytest.raises(ValueError, match='More than one alchemical'):
        AbsoluteSolvationProtocol._validate_alchemical_components(alchem_comps)


def test_validate_alchem_nonsmc(benzene_modifications):
    stateA = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'solvent': SolventComponent()
    })

    stateB = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
    })

    alchem_comps = system_validation.get_alchemical_components(stateA, stateB)

    with pytest.raises(ValueError, match='Non SmallMoleculeComponent'):
        AbsoluteSolvationProtocol._validate_alchemical_components(alchem_comps)


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

    # Create DAG from protocol, get the vacuum and solvent units
    # and eventually dry run the first vacuum unit
    dag = protocol.create(
        stateA=stateA,
        stateB=stateB,
        mapping=None,
    )
    prot_units = list(dag.protocol_units)
    
    assert len(prot_units) == 2

    vac_unit = [u for u in prot_units
                if isinstance(u, AbsoluteSolvationVacuumUnit)]
    sol_unit = [u for u in prot_units
                if isinstance(u, AbsoluteSolvationSolventUnit)]

    assert len(vac_unit) == 1
    assert len(sol_unit) == 1

    with tmpdir.as_cwd():
        vac_sampler = vac_unit[0].run(dry=True)['debug']['sampler']
        assert not vac_sampler.is_periodic


def test_confgen_fail_AFE(benzene_modifications,  tmpdir):
    # check system parametrisation works even if confgen fails
    s = openmm_afe.AbsoluteSolvationProtocol.default_settings()
    s.alchemsampler_settings.n_repeats = 1

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

    # Create DAG from protocol, get the vacuum and solvent units
    # and eventually dry run the first vacuum unit
    dag = protocol.create(
        stateA=stateA,
        stateB=stateB,
        mapping=None,
    )
    prot_units = list(dag.protocol_units)
    vac_unit = [u for u in prot_units
                if isinstance(u, AbsoluteSolvationVacuumUnit)]

    with tmpdir.as_cwd():
        with mock.patch('rdkit.Chem.AllChem.EmbedMultipleConfs', return_value=0):
            vac_sampler = vac_unit[0].run(dry=True)['debug']['sampler']

            assert vac_sampler


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

    # Create DAG from protocol, get the vacuum and solvent units
    # and eventually dry run the first solvent unit
    dag = protocol.create(
        stateA=stateA,
        stateB=stateB,
        mapping=None,
    )
    prot_units = list(dag.protocol_units)

    assert len(prot_units) == 2

    vac_unit = [u for u in prot_units
                if isinstance(u, AbsoluteSolvationVacuumUnit)]
    sol_unit = [u for u in prot_units
                if isinstance(u, AbsoluteSolvationSolventUnit)]

    assert len(vac_unit) == 1
    assert len(sol_unit) == 1

    with tmpdir.as_cwd():
        sol_sampler = sol_unit[0].run(dry=True)['debug']['sampler']
        assert sol_sampler.is_periodic

        pdb = mdt.load_pdb('hybrid_system.pdb')
        assert pdb.n_atoms == 12


def test_dry_run_benzene_packmol(benzene_modifications, tmpdir):
    s = openmm_afe.AbsoluteSolvationProtocol.default_settings()
    s.alchemsampler_settings.n_repeats = 1
    s.solvent_simulation_settings.output_indices = "not resname SOL"
    s.solvation_settings.backend = 'packmol'

    protocol = openmm_afe.AbsoluteSolvationProtocol(
            settings=s,
    )

    solvent = SolventComponent(
        smiles='CCC',
        neutralize=False, ion_concentration=0*offunit.molar
    )

    stateA = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'solvent': solvent,
    })

    stateB = ChemicalSystem({'solvent': solvent,})

    # Create DAG from protocol, get the vacuum and solvent units
    # and eventually dry run the first solvent unit
    dag = protocol.create(
        stateA=stateA,
        stateB=stateB,
        mapping=None,
    )
    prot_units = list(dag.protocol_units)

    assert len(prot_units) == 2

    vac_unit = [u for u in prot_units
                if isinstance(u, AbsoluteSolvationVacuumUnit)]
    sol_unit = [u for u in prot_units
                if isinstance(u, AbsoluteSolvationSolventUnit)]

    assert len(vac_unit) == 1
    assert len(sol_unit) == 1

    with tmpdir.as_cwd():
        sol_sampler = sol_unit[0].run(dry=True)['debug']['sampler']
        assert sol_sampler.is_periodic

        pdb = mdt.load_pdb('hybrid_system.pdb')
        assert pdb.n_atoms == 12

    with tmpdir.as_cwd():
        vac_sampler = vac_unit[0].run(dry=True)['debug']['sampler']
        assert not vac_sampler.is_periodic

        pdb = mdt.load_pdb('hybrid_system.pdb')
        assert pdb.n_atoms == 12


def test_dry_run_solv_benzene_tip4p(benzene_modifications, tmpdir):
    s = AbsoluteSolvationProtocol.default_settings()
    s.alchemsampler_settings.n_repeats = 1
    s.forcefield_settings.forcefields = [
        "amber/ff14SB.xml",    # ff14SB protein force field
        "amber/tip4pew_standard.xml", # FF we are testsing with the fun VS
        "amber/phosaa10.xml",  # Handles THE TPO
    ]
    s.solvation_settings.solvent_model = 'tip4pew'
    s.integrator_settings.reassign_velocities = True

    protocol = AbsoluteSolvationProtocol(
            settings=s,
    )

    stateA = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'solvent': SolventComponent()
    })

    stateB = ChemicalSystem({
        'solvent': SolventComponent(),
    })

    # Create DAG from protocol, get the vacuum and solvent units
    # and eventually dry run the first solvent unit
    dag = protocol.create(
        stateA=stateA,
        stateB=stateB,
        mapping=None,
    )
    prot_units = list(dag.protocol_units)

    sol_unit = [u for u in prot_units
                if isinstance(u, AbsoluteSolvationSolventUnit)]

    with tmpdir.as_cwd():
        sol_sampler = sol_unit[0].run(dry=True)['debug']['sampler']
        assert sol_sampler.is_periodic


@pytest.mark.parametrize('backend', ['openmm', 'packmol'])
def test_dry_run_solv_user_charges_benzene(
    benzene_modifications, backend, tmpdir
):
    """
    Create a test system with fictitious user supplied charges and
    ensure that they are properly passed through to the constructed
    alchemical system.
    """
    s = openmm_afe.AbsoluteSolvationProtocol.default_settings()
    s.alchemsampler_settings.n_repeats = 1
    s.solvation_settings.backend = backend

    protocol = openmm_afe.AbsoluteSolvationProtocol(
            settings=s,
    )

    def assign_fictitious_charges(offmol):
        """
        Get a random array of fake partial charges for your offmol.
        """
        rand_arr = np.random.randint(1, 10, size=offmol.n_atoms) / 100
        rand_arr[-1] = -sum(rand_arr[:-1])
        return rand_arr * offunit.elementary_charge

    benzene_offmol = benzene_modifications['benzene'].to_openff()
    offmol_pchgs = assign_fictitious_charges(benzene_offmol)
    benzene_offmol.partial_charges = offmol_pchgs
    benzene_smc = openfe.SmallMoleculeComponent.from_openff(benzene_offmol)

    # check propchgs
    prop_chgs = benzene_smc.to_dict()['molprops']['atom.dprop.PartialCharge']
    prop_chgs = np.array(prop_chgs.split(), dtype=float)
    np.testing.assert_allclose(prop_chgs, offmol_pchgs)

    if backend == 'packmol':
        solvent = SolventComponent(
            smiles="CCC", neutralize=False,
            ion_concentration=0*offunit.molar
        )
    else:
        solvent = SolventComponent()

    # Create ChemicalSystems
    stateA = ChemicalSystem({
        'benzene': benzene_smc,
        'solvent': solvent,
    })

    stateB = ChemicalSystem({
        'solvent': solvent,
    })

    # Create DAG from protocol, get the vacuum and solvent units
    # and eventually dry run the first solvent unit
    dag = protocol.create(stateA=stateA, stateB=stateB, mapping=None,)
    prot_units = list(dag.protocol_units)

    vac_unit = [u for u in prot_units
                if isinstance(u, AbsoluteSolvationVacuumUnit)][0]
    sol_unit = [u for u in prot_units
                if isinstance(u, AbsoluteSolvationSolventUnit)][0]

    # check sol_unit charges
    with tmpdir.as_cwd():
        sampler = sol_unit.run(dry=True)['debug']['sampler']
        system = sampler._thermodynamic_states[0].system
        nonbond = [f for f in system.getForces()
                   if isinstance(f, NonbondedForce)]

        assert len(nonbond) == 1

        # loop through the 12 benzene atoms
        # partial charge is stored in the offset
        for i in range(12):
            offsets = nonbond[0].getParticleParameterOffset(i)
            c = ensure_quantity(offsets[2], 'openff')
            assert pytest.approx(c) == prop_chgs[i]

    # check vac_unit charges
    with tmpdir.as_cwd():
        sampler = vac_unit.run(dry=True)['debug']['sampler']
        system = sampler._thermodynamic_states[0].system
        nonbond = [f for f in system.getForces()
                   if isinstance(f, CustomNonbondedForce)]
        assert len(nonbond) == 4

        custom_elec = [
            n for n in nonbond if
            n.getGlobalParameterName(0) == 'lambda_electrostatics'][0]

        # loop through the 12 benzene atoms
        for i in range(12):
            c, s = custom_elec.getParticleParameters(i)
            c = ensure_quantity(c, 'openff')
            assert pytest.approx(c) == prop_chgs[i]


def test_nreplicas_lambda_mismatch(benzene_modifications, tmpdir):
    s = AbsoluteSolvationProtocol.default_settings()
    s.alchemsampler_settings.n_repeats = 1
    s.alchemsampler_settings.n_replicas = 12

    protocol = AbsoluteSolvationProtocol(
            settings=s,
    )

    stateA = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'solvent': SolventComponent()
    })

    stateB = ChemicalSystem({
        'solvent': SolventComponent(),
    })

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
    s = AbsoluteSolvationProtocol.default_settings()
    s.alchemsampler_settings.n_repeats = 1
    s.forcefield_settings.hydrogen_mass = 1.0

    protocol = AbsoluteSolvationProtocol(
            settings=s,
    )

    stateA = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'solvent': SolventComponent()
    })

    stateB = ChemicalSystem({
        'solvent': SolventComponent(),
    })

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
    s = AbsoluteSolvationProtocol.default_settings()

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
        mock.patch('openfe.protocols.openmm_afe.equil_solvation_afe_method.AbsoluteSolvationSolventUnit.run',
                   return_value={'nc': 'file.nc', 'last_checkpoint': 'chck.nc'}),
        mock.patch('openfe.protocols.openmm_afe.equil_solvation_afe_method.AbsoluteSolvationVacuumUnit.run',
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
        mock.patch('openfe.protocols.openmm_afe.equil_solvation_afe_method.AbsoluteSolvationSolventUnit.run',
                   return_value={'nc': 'file.nc', 'last_checkpoint': 'chck.nc'}),
        mock.patch('openfe.protocols.openmm_afe.equil_solvation_afe_method.AbsoluteSolvationVacuumUnit.run',
                   return_value={'nc': 'file.nc', 'last_checkpoint': 'chck.nc'}),
    ):
        dagres = gufe.protocols.execute_DAG(benzene_solvation_dag,
                                            shared_basedir=tmpdir,
                                            scratch_basedir=tmpdir,
                                            keep_shared=True)

    protocol = AbsoluteSolvationProtocol(
        settings=AbsoluteSolvationProtocol.default_settings(),
    )

    res = protocol.gather([dagres])

    assert isinstance(res, openmm_afe.AbsoluteSolvationProtocolResult)


class TestProtocolResult:
    @pytest.fixture()
    def protocolresult(self, afe_solv_transformation_json):
        d = json.loads(afe_solv_transformation_json,
                       cls=gufe.tokenization.JSON_HANDLER.decoder)

        pr = openfe.ProtocolResult.from_dict(d['protocol_result'])

        return pr

    def test_reload_protocol_result(self, afe_solv_transformation_json):
        d = json.loads(afe_solv_transformation_json,
                       cls=gufe.tokenization.JSON_HANDLER.decoder)

        pr = openmm_afe.AbsoluteSolvationProtocolResult.from_dict(d['protocol_result'])

        assert pr

    def test_get_estimate(self, protocolresult):
        est = protocolresult.get_estimate()

        assert est
        assert est.m == pytest.approx(-3.00208997)
        assert isinstance(est, offunit.Quantity)
        assert est.is_compatible_with(offunit.kilojoule_per_mole)

    def test_get_uncertainty(self, protocolresult):
        est = protocolresult.get_uncertainty()

        assert est
        assert est.m == pytest.approx(0.1577349)
        assert isinstance(est, offunit.Quantity)
        assert est.is_compatible_with(offunit.kilojoule_per_mole)

    def test_get_individual(self, protocolresult):
        inds = protocolresult.get_individual_estimates()

        assert isinstance(inds, dict)
        assert isinstance(inds['solvent'], list)
        assert isinstance(inds['vacuum'], list)
        assert len(inds['solvent']) == len(inds['vacuum']) == 3
        for e, u in itertools.chain(inds['solvent'], inds['vacuum']):
            assert e.is_compatible_with(offunit.kilojoule_per_mole)
            assert u.is_compatible_with(offunit.kilojoule_per_mole)

    @pytest.mark.parametrize('key', ['solvent', 'vacuum'])
    def test_get_forwards_etc(self, key, protocolresult):
        far = protocolresult.get_forward_and_reverse_energy_analysis()

        assert isinstance(far, dict)
        assert isinstance(far[key], list)
        far1 = far[key][0]
        assert isinstance(far1, dict)

        for k in ['fractions', 'forward_DGs', 'forward_dDGs',
                  'reverse_DGs', 'reverse_dDGs']:
            assert k in far1

            if k == 'fractions':
                assert isinstance(far1[k], np.ndarray)

    @pytest.mark.parametrize('key', ['solvent', 'vacuum'])
    def test_get_overlap_matrices(self, key, protocolresult):
        ovp = protocolresult.get_overlap_matrices()

        assert isinstance(ovp, dict)
        assert isinstance(ovp[key], list)
        assert len(ovp[key]) == 3

        ovp1 = ovp[key][0]
        assert isinstance(ovp1['matrix'], np.ndarray)
        assert ovp1['matrix'].shape == (15, 15)

    @pytest.mark.parametrize('key', ['solvent', 'vacuum'])
    def test_get_replica_transition_statistics(self, key, protocolresult):
        rpx = protocolresult.get_replica_transition_statistics()

        assert isinstance(rpx, dict)
        assert isinstance(rpx[key], list)
        assert len(rpx[key]) == 3
        rpx1 = rpx[key][0]
        assert 'eigenvalues' in rpx1
        assert 'matrix' in rpx1
        assert rpx1['eigenvalues'].shape == (15,)
        assert rpx1['matrix'].shape == (15, 15)

    @pytest.mark.parametrize('key', ['solvent', 'vacuum'])
    def test_get_replica_states(self, key, protocolresult):
        rep = protocolresult.get_replica_states()

        assert isinstance(rep, dict)
        assert isinstance(rep[key], list)
        assert len(rep[key]) == 3
        assert rep[key][0].shape == (251, 15)

    @pytest.mark.parametrize('key', ['solvent', 'vacuum'])
    def test_equilibration_iterations(self, key, protocolresult):
        eq = protocolresult.equilibration_iterations()

        assert isinstance(eq, dict)
        assert isinstance(eq[key], list)
        assert len(eq[key]) == 3
        assert all(isinstance(v, float) for v in eq[key])

    @pytest.mark.parametrize('key', ['solvent', 'vacuum'])
    def test_production_iterations(self, key, protocolresult):
        prod = protocolresult.production_iterations()

        assert isinstance(prod, dict)
        assert isinstance(prod[key], list)
        assert len(prod[key]) == 3
        assert all(isinstance(v, float) for v in prod[key])
