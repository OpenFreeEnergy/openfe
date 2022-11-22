# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import gufe
import pytest
from unittest import mock
from openff.units import unit
from openff.units.openmm import ensure_quantity

from openmm import app
from openmmtools.multistate.multistatesampler import MultiStateSampler

from openfe import setup
from openfe.protocols import openmm_rbfe
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

    top2 = openmm_rbfe._rbfe_utils.topologyhelpers.combined_topology(
        top1, lig2.to_topology().to_openmm(),
        exclude_chains=list(top1.chains())[-1:],
    )

    assert len(list(top2.atoms())) == 2625 + 3  # added methyl


def test_create_default_settings():
    settings = openmm_rbfe.RelativeLigandTransform.default_settings()

    assert settings


def test_create_default_protocol():
    # this is roughly how it should be created
    protocol = openmm_rbfe.RelativeLigandTransform(
        settings=openmm_rbfe.RelativeLigandTransform.default_settings(),
    )

    assert protocol


def test_serialize_protocol():
    protocol = openmm_rbfe.RelativeLigandTransform(
        settings=openmm_rbfe.RelativeLigandTransform.default_settings(),
    )

    ser = protocol.to_dict()

    ret = openmm_rbfe.RelativeLigandTransform.from_dict(ser)

    assert protocol == ret


@pytest.mark.parametrize('method', [
    'repex', 'sams', 'independent', 'InDePeNdENT'
])
def test_dry_run_default_vacuum(benzene_vacuum_system, toluene_vacuum_system,
                                benzene_to_toluene_mapping, method, tmpdir):

    vac_settings = openmm_rbfe.RelativeLigandTransform.default_settings()
    vac_settings.system_settings.nonbonded_method = 'nocutoff'
    vac_settings.sampler_settings.sampler_method = method
    vac_settings.sampler_settings.n_repeats = 1

    protocol = openmm_rbfe.RelativeLigandTransform(
            settings=vac_settings,
    )

    # create DAG from protocol and take first (and only) work unit from within
    dag = protocol.create(
        stateA=benzene_vacuum_system,
        stateB=toluene_vacuum_system,
        mapping={'ligand': benzene_to_toluene_mapping},
    )
    unit = list(dag.protocol_units)[0]

    with tmpdir.as_cwd():
        assert isinstance(unit.run(dry=True)['debug']['sampler'],
                          MultiStateSampler)


@pytest.mark.parametrize('method', ['repex', 'sams', 'independent'])
def test_dry_run_ligand(benzene_system, toluene_system,
                        benzene_to_toluene_mapping, method, tmpdir):
    # this might be a bit time consuming
    settings = openmm_rbfe.RelativeLigandTransform.default_settings()
    settings.sampler_settings.sampler_method = method
    settings.sampler_settings.n_repeats = 1

    protocol = openmm_rbfe.RelativeLigandTransform(
            settings=settings,
    )
    dag = protocol.create(
        stateA=benzene_system,
        stateB=toluene_system,
        mapping={'ligand': benzene_to_toluene_mapping},
    )
    unit = list(dag.protocol_units)[0]

    with tmpdir.as_cwd():
        # Returns debug objects if everything is OK
        assert isinstance(unit.run(dry=True)['debug']['sampler'],
                          MultiStateSampler)


@pytest.mark.parametrize('method', ['repex', 'sams', 'independent'])
def test_dry_run_complex(benzene_complex_system, toluene_complex_system,
                         benzene_to_toluene_mapping, method, tmpdir):
    # this will be very time consuming
    settings = openmm_rbfe.RelativeLigandTransform.default_settings()
    settings.sampler_settings.sampler_method = method
    settings.sampler_settings.n_repeats = 1

    protocol = openmm_rbfe.RelativeLigandTransform(
            settings=settings,
    )
    dag = protocol.create(
        stateA=benzene_complex_system,
        stateB=toluene_complex_system,
        mapping={'ligand': benzene_to_toluene_mapping},
    )
    unit = list(dag.protocol_units)[0]

    with tmpdir.as_cwd():
        # Returns debug contents if everything is OK
        assert isinstance(unit.run(dry=True)['debug']['sampler'],
                          MultiStateSampler)


def test_lambda_schedule_default():
    lambdas = openmm_rbfe._rbfe_utils.lambdaprotocol.LambdaProtocol(functions='default')
    assert len(lambdas.lambda_schedule) == 10


@pytest.mark.parametrize('windows', [11, 6, 9000])
def test_lambda_schedule(windows):
    lambdas = openmm_rbfe._rbfe_utils.lambdaprotocol.LambdaProtocol(
            functions='default', windows=windows)
    assert len(lambdas.lambda_schedule) == windows


def test_n_replicas_not_n_windows(benzene_vacuum_system,
                                  toluene_vacuum_system,
                                  benzene_to_toluene_mapping, tmpdir):
    # For PR #125 we pin such that the number of lambda windows
    # equals the numbers of replicas used - TODO: remove limitation
    settings = openmm_rbfe.RelativeLigandTransform.default_settings()
    # default lambda windows is 11
    settings.sampler_settings.n_replicas = 13
    settings.system_settings.nonbonded_method = 'nocutoff'

    errmsg = ("Number of replicas 13 does not equal the number of "
              "lambda windows 11")

    with tmpdir.as_cwd():
        with pytest.raises(ValueError, match=errmsg):
            p = openmm_rbfe.RelativeLigandTransform(
                    settings=settings,
            )
            dag = p.create(
                stateA=benzene_vacuum_system,
                stateB=toluene_vacuum_system,
                mapping={'ligand': benzene_to_toluene_mapping},
            )
            unit = list(dag.protocol_units)[0]
            unit.run(dry=True)


def test_missing_ligand(benzene_system, benzene_to_toluene_mapping):
    # state B doesn't have a ligand component
    stateB = setup.ChemicalSystem({'solvent': setup.SolventComponent()})

    p = openmm_rbfe.RelativeLigandTransform(
        settings=openmm_rbfe.RelativeLigandTransform.default_settings(),
    )

    with pytest.raises(ValueError, match='Missing ligand in state B'):
        _ = p.create(
            stateA=benzene_system,
            stateB=stateB,
            mapping={'ligand': benzene_to_toluene_mapping},
        )


def test_vaccuum_PME_error(benzene_system, benzene_modifications,
                           benzene_to_toluene_mapping):
    # state B doesn't have a solvent component (i.e. its vacuum)
    stateB = setup.ChemicalSystem({'ligand': benzene_modifications['toluene']})

    p = openmm_rbfe.RelativeLigandTransform(
        settings=openmm_rbfe.RelativeLigandTransform.default_settings(),
    )
    errmsg = "PME cannot be used for vacuum transform"
    with pytest.raises(ValueError, match=errmsg):
        _ = p.create(
            stateA=benzene_system,
            stateB=stateB,
            mapping={'ligand': benzene_to_toluene_mapping},
        )


def test_incompatible_solvent(benzene_system, benzene_modifications,
                              benzene_to_toluene_mapping):
    # the solvents are different
    stateB = setup.ChemicalSystem(
        {'ligand': benzene_modifications['toluene'],
         'solvent': setup.SolventComponent(
             positive_ion='K', negative_ion='Cl')}
    )

    p = openmm_rbfe.RelativeLigandTransform(
        settings=openmm_rbfe.RelativeLigandTransform.default_settings(),
    )
    with pytest.raises(ValueError, match="Solvents aren't identical"):
        _ = p.create(
            stateA=benzene_system,
            stateB=stateB,
            mapping={'ligand': benzene_to_toluene_mapping},
        )


def test_mapping_mismatch_A(benzene_system, toluene_system,
                            benzene_modifications):
    # the atom mapping doesn't refer to the ligands in the systems
    mapping = setup.LigandAtomMapping(
        componentA=benzene_system.components['ligand'],
        componentB=benzene_modifications['phenol'],
        componentA_to_componentB=dict())

    p = openmm_rbfe.RelativeLigandTransform(
        settings=openmm_rbfe.RelativeLigandTransform.default_settings(),
    )
    with pytest.raises(ValueError,
                       match="Ligand in state B doesn't match mapping"):
        _ = p.create(
            stateA=benzene_system,
            stateB=toluene_system,
            mapping={'ligand': mapping},
        )


def test_mapping_mismatch_B(benzene_system, toluene_system,
                            benzene_modifications):
    mapping = setup.LigandAtomMapping(
        componentA=benzene_modifications['phenol'],
        componentB=toluene_system.components['ligand'],
        componentA_to_componentB=dict())

    p = openmm_rbfe.RelativeLigandTransform(
        settings=openmm_rbfe.RelativeLigandTransform.default_settings(),
    )
    with pytest.raises(ValueError,
                       match="Ligand in state A doesn't match mapping"):
        _ = p.create(
            stateA=benzene_system,
            stateB=toluene_system,
            mapping={'ligand': mapping},
        )


def test_complex_mismatch(benzene_system, toluene_complex_system,
                          benzene_to_toluene_mapping):
    # only one complex
    p = openmm_rbfe.RelativeLigandTransform(
        settings=openmm_rbfe.RelativeLigandTransform.default_settings(),
    )
    with pytest.raises(ValueError):
        _ = p.create(
            stateA=benzene_system,
            stateB=toluene_complex_system,
            mapping={'ligand': benzene_to_toluene_mapping},
        )


def test_badly_specified_mapping(benzene_system, toluene_system,
                                 benzene_to_toluene_mapping):
    # mapping dict requires 'ligand' key
    p = openmm_rbfe.RelativeLigandTransform(
        settings=openmm_rbfe.RelativeLigandTransform.default_settings(),
    )
    with pytest.raises(ValueError):
        _ = p.create(
            stateA=benzene_system,
            stateB=toluene_system,
            mapping={'solvent': benzene_to_toluene_mapping}
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

    p = openmm_rbfe.RelativeLigandTransform(
        settings=openmm_rbfe.RelativeLigandTransform.default_settings(),
    )
    with pytest.raises(ValueError):
        _ = p.create(
            stateA=benzene_complex_system,
            stateB=alt_toluene_complex_system,
            mapping={'ligand': benzene_to_toluene_mapping},
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

    p = openmm_rbfe.RelativeLigandTransform(
        settings=openmm_rbfe.RelativeLigandTransform.default_settings(),
    )
    with pytest.raises(ValueError, match="Element change"):
        _ = p.create(
            stateA=sys1, stateB=sys2,
            mapping={'ligand': mapping},
        )


@pytest.fixture
def solvent_protocol_dag(benzene_system, toluene_system, benzene_to_toluene_mapping):
    settings = openmm_rbfe.RelativeLigandTransform.default_settings()

    protocol = openmm_rbfe.RelativeLigandTransform(
        settings=settings,
    )

    return protocol.create(
        stateA=benzene_system, stateB=toluene_system,
        mapping={'ligand': benzene_to_toluene_mapping},
    )


def test_unit_tagging(solvent_protocol_dag, tmpdir):
    # test that executing the Units includes correct generation and repeat info
    units = solvent_protocol_dag.protocol_units
    with mock.patch('openfe.protocols.openmm_rbfe.equil_rbfe_methods.RelativeLigandTransformUnit.run',
                    return_value={'nc': 'file.nc', 'last_checkpoint': 'chk.nc'}):
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


def test_gather(solvent_protocol_dag):
    # check .gather behaves as expected
    with mock.patch('openfe.protocols.openmm_rbfe.equil_rbfe_methods.RelativeLigandTransformUnit.run',
                    return_value={'nc': 'file.nc', 'last_checkpoint': 'chk.nc'}):
        dagres = gufe.protocols.execute(solvent_protocol_dag)

    prot = openmm_rbfe.RelativeLigandTransform(
        settings=openmm_rbfe.RelativeLigandTransform.default_settings()
    )

    with mock.patch('openfe.protocols.openmm_rbfe.equil_rbfe_methods.multistate') as m:
        res = prot.gather([dagres])

        # check we created the expected number of Reporters and Analyzers
        assert m.MultiStateReporter.call_count == 3
        m.MultiStateReporter.assert_called_with(
            storage='file.nc', checkpoint_storage='chk.nc',
        )
        assert m.MultiStateSamplerAnalyzer.call_count == 3

    assert isinstance(res, openmm_rbfe.RelativeLigandTransformResult)


class TestConstraintRemoval:
    @staticmethod
    def make_systems(ligA: setup.SmallMoleculeComponent,
                     ligB: setup.SmallMoleculeComponent):
        """Make vacuum system for each, return Topology and System for each"""
        omm_forcefield_A = app.ForceField('tip3p.xml')
        smirnoff_A = SMIRNOFFTemplateGenerator(
            forcefield='openff-2.0.0.offxml',
            molecules=[ligA.to_openff()],
        )
        omm_forcefield_A.registerTemplateGenerator(smirnoff_A.generator)

        omm_forcefield_B = app.ForceField('tip3p.xml')
        smirnoff_B = SMIRNOFFTemplateGenerator(
            forcefield='openff-2.0.0.offxml',
            molecules=[ligB.to_openff()],
        )
        omm_forcefield_B.registerTemplateGenerator(smirnoff_B.generator)

        stateA_modeller = app.Modeller(
            ligA.to_openff().to_topology().to_openmm(),
            ensure_quantity(ligA.to_openff().conformers[0], 'openmm')
        )
        stateA_topology = stateA_modeller.getTopology()
        stateA_system = omm_forcefield_A.createSystem(
            stateA_topology,
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=ensure_quantity(1.1 * unit.nm, 'openmm'),
            constraints=app.HBonds,  # constraints,
            rigidWater=True,
            hydrogenMass=None,
            removeCMMotion=True,
        )

        stateB_topology = openmm_rbfe._rbfe_utils.topologyhelpers.combined_topology(
            stateA_topology,
            ligB.to_openff().to_topology().to_openmm(),
            exclude_chains=list(stateA_topology.chains())
        )
        # since we're doing a swap of the only molecule, this is equivalent:
        # stateB_topology = app.Modeller(
        #    sysB['ligand'].to_openff().to_topology().to_openmm(),
        #    ensure_quantity(sysB['ligand'].to_openff().conformers[0], 'openmm')
        # )

        stateB_system = omm_forcefield_B.createSystem(
            stateB_topology,
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=ensure_quantity(1.1 * unit.nm, 'openmm'),
            constraints=app.HBonds,  # constraints,
            rigidWater=True,
            hydrogenMass=None,
            removeCMMotion=True,
        )

        return stateA_topology, stateA_system, stateB_topology, stateB_system

    def test_remove_constraints_lengthchange(self, benzene_modifications):
        # check that mappings are correctly corrected to avoid changes in
        # constraint length
        # use a phenol->toluene transform to test
        ligA = benzene_modifications['phenol']
        ligB = benzene_modifications['toluene']

        mapping = setup.LigandAtomMapping(
            componentA=ligA,
            componentB=ligB,
            # this is default lomap
            # importantly the H in -OH maps to one of the -CH3
            # this constraint will change length
            componentA_to_componentB={0: 4, 1: 5, 2: 6, 3: 7, 4: 8, 5: 9, 6: 10,
                                      7: 11, 8: 12, 9: 13, 10: 1, 11: 14, 12: 2},
        )

        stateA_topology, stateA_system, stateB_topology, stateB_system = self.make_systems(ligA, ligB)

        # this normally requires global indices, however as ligandA/B is only thing
        # in system, this mapping is still correct
        ret = openmm_rbfe._rbfe_utils.topologyhelpers._remove_constraints(
            mapping.componentA_to_componentB,
            stateA_system, stateA_topology,
            stateB_system, stateB_topology,
        )

        # all of this just to check that an entry was removed from the mapping
        # sanity check
        assert 10 in mapping.componentA_to_componentB
        # the removed constraint
        assert 10 not in ret
        # but only one constraint should be removed
        assert len(ret) == len(mapping.componentA_to_componentB) - 1

    def test_constraint_to_harmonic(self, benzene_modifications):
        ligA = benzene_modifications['benzene']
        ligB = benzene_modifications['toluene']

        # this maps a -H to a -C, so the constraint on -H turns into a C-C bond
        # H constraint is A(4, 10) and C-C is B(8, 2)
        mapping = setup.LigandAtomMapping(
            componentA=ligA, componentB=ligB,
            componentA_to_componentB={0: 4, 1: 5, 2: 6, 3: 7, 4: 8, 5: 9,
                                      6: 10, 7: 11, 8: 12, 9: 13, 10: 2, 11: 14}
        )

        stateA_topology, stateA_system, stateB_topology, stateB_system = self.make_systems(ligA, ligB)

        ret = openmm_rbfe._rbfe_utils.topologyhelpers._remove_constraints(
            mapping.componentA_to_componentB,
            stateA_system, stateA_topology,
            stateB_system, stateB_topology,
        )

        assert 10 not in ret
        assert len(ret) == len(mapping.componentA_to_componentB) - 1

    def test_constraint_to_harmonic_reversed(self, benzene_modifications):
        # same as previous test, but ligands are swapped
        # this follows a slightly different code path
        ligA = benzene_modifications['toluene']
        ligB = benzene_modifications['benzene']

        mapping = {0: 4, 1: 5, 2: 6, 3: 7, 4: 8, 5: 9,
                   6: 10, 7: 11, 8: 12, 9: 13, 10: 2, 11: 14}
        mapping = setup.LigandAtomMapping(
            componentA=ligA, componentB=ligB,
            componentA_to_componentB={v: k for k, v in mapping.items()}
        )

        stateA_topology, stateA_system, stateB_topology, stateB_system = self.make_systems(ligA, ligB)

        ret = openmm_rbfe._rbfe_utils.topologyhelpers._remove_constraints(
            mapping.componentA_to_componentB,
            stateA_system, stateA_topology,
            stateB_system, stateB_topology,
        )

        assert 2 not in ret
        assert len(ret) == len(mapping.componentA_to_componentB) - 1

    @pytest.mark.parametrize('reverse', [False, True])
    def test_constraint_to_harmonic_nitrile(self, benzene_modifications,
                                            reverse):
        # same as previous test, but ligands are swapped
        # this follows a slightly different code path
        ligA = benzene_modifications['toluene']
        ligB = benzene_modifications['benzonitrile']

        if reverse:
            ligA, ligB = ligB, ligA

        mapping = {0: 0, 2: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8,
                   11: 9, 12: 10, 13: 11, 14: 12}
        if reverse:
            mapping = {v: k for k, v in mapping.items()}
        mapping = setup.LigandAtomMapping(
            componentA=ligA, componentB=ligB,
            componentA_to_componentB=mapping,
        )

        stateA_topology, stateA_system, stateB_topology, stateB_system = self.make_systems(ligA, ligB)

        ret = openmm_rbfe._rbfe_utils.topologyhelpers._remove_constraints(
            mapping.componentA_to_componentB,
            stateA_system, stateA_topology,
            stateB_system, stateB_topology,
        )

        assert 0 not in ret
        assert len(ret) == len(mapping.componentA_to_componentB) - 1