# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import os
from io import StringIO
import copy
import numpy as np
import gufe
from gufe.tests.test_tokenization import GufeTokenizableTestsMixin
import json
import pytest
from unittest import mock
from openff.units.openmm import to_openmm, from_openmm
from openff.units import unit
from importlib import resources
import xml.etree.ElementTree as ET

from openmm import (
    app, XmlSerializer, MonteCarloBarostat,
    NonbondedForce, CustomNonbondedForce
)
from openmm import unit as omm_unit
from openmmtools.multistate.multistatesampler import MultiStateSampler
import pathlib
from rdkit import Chem
from rdkit.Geometry import Point3D
import mdtraj as mdt

import openfe
from openfe import setup
from openfe.protocols import openmm_rfe
from openfe.protocols.openmm_rfe.equil_rfe_methods import (
        _validate_alchemical_components, _get_alchemical_charge_difference
)
from openfe.protocols.openmm_rfe._rfe_utils import topologyhelpers
from openfe.protocols.openmm_utils import system_creation
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from openff.units.openmm import ensure_quantity


def test_compute_platform_warn():
    with pytest.warns(UserWarning, match="Non-GPU platform selected: CPU"):
        openmm_rfe._rfe_utils.compute.get_openmm_platform('CPU')


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
    assert len(list(top1.bonds())) == 2645

    lig2 = toluene_complex_system['ligand'].to_openff()

    top2, appended_resids = openmm_rfe._rfe_utils.topologyhelpers.combined_topology(
        top1, lig2.to_topology().to_openmm(),
        exclude_resids=np.asarray(list(top1.residues())[-1].index),
    )

    assert len(list(top2.atoms())) == 2625 + 3  # added methyl
    assert len(list(top2.bonds())) == 2645 + 4 - 1 # add methyl bonds, minus hydrogen
    assert appended_resids[0] == len(list(top1.residues())) - 1


def test_append_topology_no_exclude(benzene_complex_system,
                                    toluene_complex_system):
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
    assert len(list(top1.bonds())) == 2645

    lig2 = toluene_complex_system['ligand'].to_openff()

    top2, appended_resids = openmm_rfe._rfe_utils.topologyhelpers.combined_topology(
        top1, lig2.to_topology().to_openmm(),
        exclude_resids=None,
    )

    assert len(list(top2.atoms())) == 2625 + 15  # added toluene
    assert len(list(top2.bonds())) == 2645 + 15 # 15 bonds in toluene
    assert appended_resids[0] == len(list(top1.residues()))


def test_create_default_settings():
    settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()

    assert settings


def test_create_default_protocol():
    # this is roughly how it should be created
    protocol = openmm_rfe.RelativeHybridTopologyProtocol(
        settings=openmm_rfe.RelativeHybridTopologyProtocol.default_settings(),
    )

    assert protocol


def test_serialize_protocol():
    protocol = openmm_rfe.RelativeHybridTopologyProtocol(
        settings=openmm_rfe.RelativeHybridTopologyProtocol.default_settings(),
    )

    ser = protocol.to_dict()

    ret = openmm_rfe.RelativeHybridTopologyProtocol.from_dict(ser)

    assert protocol == ret


def test_create_independent_repeat_ids(benzene_system, toluene_system, benzene_to_toluene_mapping):
    # if we create two dags each with 3 repeats, they should give 6 repeat_ids
    # this allows multiple DAGs in flight for one Transformation that don't clash on gather
    settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()

    protocol = openmm_rfe.RelativeHybridTopologyProtocol(
            settings=settings,
    )
    dag1 = protocol.create(
        stateA=benzene_system,
        stateB=toluene_system,
        mapping={'ligand': benzene_to_toluene_mapping},
    )
    dag2 = protocol.create(
        stateA=benzene_system,
        stateB=toluene_system,
        mapping={'ligand': benzene_to_toluene_mapping},
    )

    repeat_ids = set()
    u: openmm_rfe.RelativeHybridTopologyProtocolUnit
    for u in dag1.protocol_units:
        repeat_ids.add(u.inputs['repeat_id'])
    for u in dag2.protocol_units:
        repeat_ids.add(u.inputs['repeat_id'])

    assert len(repeat_ids) == 6


@pytest.mark.parametrize('mapping', [
    None, {'A': 'Foo', 'B': 'bar'},
])
def test_validate_alchemical_components_wrong_mappings(mapping):
    with pytest.raises(ValueError, match="A single LigandAtomMapping"):
        _validate_alchemical_components(
            {'stateA': [], 'stateB': []}, mapping
        )


def test_validate_alchemical_components_missing_alchem_comp(
        benzene_to_toluene_mapping):
    alchem_comps = {'stateA': [openfe.SolventComponent(),], 'stateB': []}
    with pytest.raises(ValueError, match="Unmapped alchemical component"):
        _validate_alchemical_components(
            alchem_comps, {'ligand': benzene_to_toluene_mapping},
        )


@pytest.mark.parametrize('method', [
    'repex', 'sams', 'independent', 'InDePeNdENT'
])
def test_dry_run_default_vacuum(benzene_vacuum_system, toluene_vacuum_system,
                                benzene_to_toluene_mapping, method, tmpdir):

    vac_settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
    vac_settings.system_settings.nonbonded_method = 'nocutoff'
    vac_settings.alchemical_sampler_settings.sampler_method = method
    vac_settings.alchemical_sampler_settings.n_repeats = 1

    protocol = openmm_rfe.RelativeHybridTopologyProtocol(
            settings=vac_settings,
    )

    # create DAG from protocol and take first (and only) work unit from within
    dag = protocol.create(
        stateA=benzene_vacuum_system,
        stateB=toluene_vacuum_system,
        mapping={'ligand': benzene_to_toluene_mapping},
    )
    dag_unit = list(dag.protocol_units)[0]

    with tmpdir.as_cwd():
        sampler = dag_unit.run(dry=True)['debug']['sampler']
        assert isinstance(sampler, MultiStateSampler)
        assert not sampler.is_periodic
        assert sampler._thermodynamic_states[0].barostat is None

        # Check hybrid OMM and MDTtraj Topologies
        htf = sampler._hybrid_factory
        # 16 atoms:
        # 11 common atoms, 1 extra hydrogen in benzene, 4 extra in toluene
        # 12 bonds in benzene + 4 extra toluene bonds
        assert len(list(htf.hybrid_topology.atoms)) == 16
        assert len(list(htf.omm_hybrid_topology.atoms())) == 16
        assert len(list(htf.hybrid_topology.bonds)) == 16
        assert len(list(htf.omm_hybrid_topology.bonds())) == 16

        # smoke test - can convert back the mdtraj topology
        ret_top = mdt.Topology.to_openmm(htf.hybrid_topology)
        assert len(list(ret_top.atoms())) == 16
        assert len(list(ret_top.bonds())) == 16

        # check that our PDB has the right number of atoms
        pdb = mdt.load_pdb('hybrid_system.pdb')
        assert pdb.n_atoms == 16


def test_dry_run_gaff_vacuum(benzene_vacuum_system, toluene_vacuum_system,
                             benzene_to_toluene_mapping, tmpdir):
    vac_settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
    vac_settings.system_settings.nonbonded_method = 'nocutoff'
    vac_settings.forcefield_settings.small_molecule_forcefield = 'gaff-2.11'

    protocol = openmm_rfe.RelativeHybridTopologyProtocol(
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
        sampler = unit.run(dry=True)['debug']['sampler']


@pytest.mark.slow
def test_dry_many_molecules_solvent(
    benzene_many_solv_system, toluene_many_solv_system,
    benzene_to_toluene_mapping, tmpdir
):
    """
    A basic test flushing "will it work if you pass multiple molecules"
    """
    settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()

    protocol = openmm_rfe.RelativeHybridTopologyProtocol(
            settings=settings,
    )

    # create DAG from protocol and take first (and only) work unit from within
    dag = protocol.create(
        stateA=benzene_many_solv_system,
        stateB=toluene_many_solv_system,
        mapping={'spicyligand': benzene_to_toluene_mapping},
    )
    unit = list(dag.protocol_units)[0]

    with tmpdir.as_cwd():
        sampler = unit.run(dry=True)['debug']['sampler']


BENZ = """\
benzene
  PyMOL2.5          3D                             0

 12 12  0  0  0  0  0  0  0  0999 V2000
    1.4045   -0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.7022    1.2164    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7023    1.2164    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.4045   -0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7023   -1.2164    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.7023   -1.2164    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.5079   -0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.2540    2.1720    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2540    2.1720    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.5079   -0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2540   -2.1719    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.2540   -2.1720    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0  0  0  0
  1  6  1  0  0  0  0
  1  7  1  0  0  0  0
  2  3  1  0  0  0  0
  2  8  1  0  0  0  0
  3  4  2  0  0  0  0
  3  9  1  0  0  0  0
  4  5  1  0  0  0  0
  4 10  1  0  0  0  0
  5  6  2  0  0  0  0
  5 11  1  0  0  0  0
  6 12  1  0  0  0  0
M  END
$$$$
"""


PYRIDINE = """\
pyridine
  PyMOL2.5          3D                             0

 11 11  0  0  0  0  0  0  0  0999 V2000
    1.4045   -0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7023    1.2164    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.4045   -0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7023   -1.2164    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.7023   -1.2164    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.4940   -0.0325    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.2473   -2.1604    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2473   -2.1604    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.4945   -0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2753    2.1437    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.7525    1.3034    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
  1  5  1  0  0  0  0
  1  6  1  0  0  0  0
  1 11  2  0  0  0  0
  2  3  2  0  0  0  0
  2 10  1  0  0  0  0
  3  4  1  0  0  0  0
  3  9  1  0  0  0  0
  4  5  2  0  0  0  0
  4  8  1  0  0  0  0
  5  7  1  0  0  0  0
  2 11  1  0  0  0  0
M  END
$$$$
"""


def test_dry_core_element_change(tmpdir):

    benz = openfe.SmallMoleculeComponent(Chem.MolFromMolBlock(BENZ, removeHs=False))
    pyr = openfe.SmallMoleculeComponent(Chem.MolFromMolBlock(PYRIDINE, removeHs=False))

    mapping = openfe.LigandAtomMapping(
        benz, pyr,
        {0: 0, 1: 10, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 9, 9: 8, 10: 7, 11: 6}
    )

    settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
    settings.system_settings.nonbonded_method = 'nocutoff'

    protocol = openmm_rfe.RelativeHybridTopologyProtocol(
            settings=settings,
    )

    dag = protocol.create(
        stateA=openfe.ChemicalSystem({'ligand': benz,}),
        stateB=openfe.ChemicalSystem({'ligand': pyr,}),
        mapping={'whatamapping': mapping},
    )

    dag_unit = list(dag.protocol_units)[0]

    with tmpdir.as_cwd():
        sampler = dag_unit.run(dry=True)['debug']['sampler']
        system = sampler._hybrid_factory.hybrid_system
        assert system.getNumParticles() == 12
        # Average mass between nitrogen and carbon
        assert system.getParticleMass(1) == 12.0127235 * omm_unit.amu

        # Get out the CustomNonbondedForce
        cnf = [f for f in system.getForces()
               if f.__class__.__name__ == 'CustomNonbondedForce'][0]
        # there should be no new unique atoms
        assert cnf.getInteractionGroupParameters(6) == [(), ()]
        # there should be one old unique atom (spare hydrogen from the benzene)
        assert cnf.getInteractionGroupParameters(7) == [(7,), (7,)]


@pytest.mark.parametrize('method', ['repex', 'sams', 'independent'])
def test_dry_run_ligand(benzene_system, toluene_system,
                        benzene_to_toluene_mapping, method, tmpdir):
    # this might be a bit time consuming
    settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
    settings.alchemical_sampler_settings.sampler_method = method
    settings.alchemical_sampler_settings.n_repeats = 1
    settings.simulation_settings.output_indices = 'resname UNK'

    protocol = openmm_rfe.RelativeHybridTopologyProtocol(
            settings=settings,
    )
    dag = protocol.create(
        stateA=benzene_system,
        stateB=toluene_system,
        mapping={'ligand': benzene_to_toluene_mapping},
    )
    dag_unit = list(dag.protocol_units)[0]

    with tmpdir.as_cwd():
        sampler = dag_unit.run(dry=True)['debug']['sampler']
        assert isinstance(sampler, MultiStateSampler)
        assert sampler.is_periodic
        assert isinstance(sampler._thermodynamic_states[0].barostat,
                          MonteCarloBarostat)
        assert sampler._thermodynamic_states[1].pressure == 1 * omm_unit.bar

        # Check we have the right number of atoms in the PDB
        pdb = mdt.load_pdb('hybrid_system.pdb')
        assert pdb.n_atoms == 16


def test_confgen_mocked_fail(benzene_system, toluene_system,
                             benzene_to_toluene_mapping, tmpdir):
    """
    Check that even if conformer generation fails, we can still perform a sim
    """
    settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
    settings.alchemical_sampler_settings.n_repeats = 1

    protocol = openmm_rfe.RelativeHybridTopologyProtocol(settings=settings)

    dag = protocol.create(stateA=benzene_system, stateB=toluene_system,
                          mapping={'ligand': benzene_to_toluene_mapping})
    dag_unit = list(dag.protocol_units)[0]

    with tmpdir.as_cwd():
        with mock.patch('rdkit.Chem.AllChem.EmbedMultipleConfs', return_value=0):
            sampler = dag_unit.run(dry=True)

            assert sampler


@pytest.fixture(scope='session')
def tip4p_hybrid_factory(
    benzene_system, toluene_system,
    benzene_to_toluene_mapping, tmp_path_factory
):
    """
    Hybrid system with virtual sites in the environment (waters)
    """
    settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
    settings.forcefield_settings.forcefields = [
        "amber/ff14SB.xml",    # ff14SB protein force field
        "amber/tip4pew_standard.xml", # FF we are testsing with the fun VS
        "amber/phosaa10.xml",  # Handles THE TPO
    ]
    settings.system_settings.nonbonded_cutoff = 0.9 * unit.nanometer
    settings.solvation_settings.solvent_model = 'tip4pew'
    settings.integrator_settings.reassign_velocities = True

    protocol = openmm_rfe.RelativeHybridTopologyProtocol(
            settings=settings,
    )
    dag = protocol.create(
        stateA=benzene_system,
        stateB=toluene_system,
        mapping={'ligand': benzene_to_toluene_mapping},
    )
    dag_unit = list(dag.protocol_units)[0]

    shared_temp = tmp_path_factory.mktemp("tip4p_shared")
    scratch_temp = tmp_path_factory.mktemp("tip4p_scratch")

    dag_unit_result = dag_unit.run(
            dry=True,
            scratch_basepath=scratch_temp,
            shared_basepath=shared_temp,
    )

    return dag_unit_result['debug']['sampler']._factory


def test_tip4p_particle_count(tip4p_hybrid_factory):
    """
    Check that the total number of particles in the system
    are as expected.
    """

    htf = tip4p_hybrid_factory

    old_particle_count = htf._old_system.getNumParticles()
    unique_new_count = len(htf._unique_new_atoms)
    hybrid_particle_count = htf.hybrid_system.getNumParticles()

    assert old_particle_count + unique_new_count == hybrid_particle_count


def test_tip4p_num_waters(tip4p_hybrid_factory):
    """
    Check that the number of virtual sites is equal the number of waters.
    """

    htf = tip4p_hybrid_factory

    # Test 2
    num_waters = len(
        [r for r in htf._old_topology.residues() if r.name =='HOH']
    )
    virtual_sites = [
        ix for ix in range(htf.hybrid_system.getNumParticles()) if
        htf.hybrid_system.isVirtualSite(ix)
    ]
    assert num_waters == len(virtual_sites)


def test_tip4p_check_vsite_parameters(tip4p_hybrid_factory):
    """
    Check that the virtual site parameters are those expected
    as defined by the tip4p-ew parameters in openmmforcefields
    """

    htf = tip4p_hybrid_factory

    virtual_sites = [
        ix for ix in range(htf.hybrid_system.getNumParticles()) if
        htf.hybrid_system.isVirtualSite(ix)
    ]

    # get the standard and custom nonbonded forces - one of each
    nonbond = [f for f in htf.hybrid_system.getForces()
               if isinstance(f, NonbondedForce)][0]

    cust_nonbond = [f for f in htf.hybrid_system.getForces()
                    if isinstance(f, CustomNonbondedForce)][0]

    # loop through every virtual site and check that they have the
    # expected tip4p parameters
    for entry in virtual_sites:
        vs = htf.hybrid_system.getVirtualSite(entry)
        vs_mass = htf.hybrid_system.getParticleMass(entry)
        assert ensure_quantity(vs_mass, 'openff').m == pytest.approx(0)
        vs_weights = [vs.getWeight(ix) for ix in range(vs.getNumParticles())]
        np.testing.assert_allclose(
            vs_weights, [0.786646558, 0.106676721, 0.106676721]
        )
        c, s, e = nonbond.getParticleParameters(entry)
        assert ensure_quantity(c, 'openff').m == pytest.approx(-1.04844)
        assert ensure_quantity(s, 'openff').m == 1
        assert ensure_quantity(e, 'openff').m == 0

        s1, e1, s2, e2, i, j = cust_nonbond.getParticleParameters(entry)

        assert i == j == 0
        assert s1 == s2 == 1
        assert e1 == e2 == 0


@pytest.mark.slow
@pytest.mark.parametrize('cutoff',
    [1.0 * unit.nanometer,
     12.0 * unit.angstrom,
     0.9 * unit.nanometer]
)
def test_dry_run_ligand_system_cutoff(
    cutoff, benzene_system, toluene_system, benzene_to_toluene_mapping, tmpdir
):
    """
    Test that the right nonbonded cutoff is propagated to the hybrid system.
    """
    settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
    settings.solvation_settings.solvent_padding = 1.5 * unit.nanometer
    settings.system_settings.nonbonded_cutoff = cutoff

    protocol = openmm_rfe.RelativeHybridTopologyProtocol(
            settings=settings,
    )
    dag = protocol.create(
        stateA=benzene_system,
        stateB=toluene_system,
        mapping={'ligand': benzene_to_toluene_mapping},
    )
    dag_unit = list(dag.protocol_units)[0]

    with tmpdir.as_cwd():
        sampler = dag_unit.run(dry=True)['debug']['sampler']
        hs = sampler._factory.hybrid_system

        nbfs = [f for f in hs.getForces() if
                isinstance(f, CustomNonbondedForce) or
                isinstance(f, NonbondedForce)]

        for f in nbfs:
            f_cutoff = from_openmm(f.getCutoffDistance())
            assert f_cutoff == cutoff


@pytest.mark.flaky(reruns=3)  # bad minimisation can happen
def test_dry_run_user_charges(benzene_modifications, tmpdir):
    """
    Create a hybrid system with a set of fictitious user supplied charges
    and ensure that they are properly passed through to the constructed
    hybrid topology.
    """
    vac_settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
    vac_settings.system_settings.nonbonded_method = 'nocutoff'
    vac_settings.alchemical_sampler_settings.n_repeats = 1

    protocol = openmm_rfe.RelativeHybridTopologyProtocol(
            settings=vac_settings,
    )

    def assign_fictitious_charges(offmol):
        """
        Get a random array of fake partial charges (ints because why not)
        that sums up to 0. Note that OpenFF will complain if you try to
        create a molecule that has a total charge that is different from
        the expected formal charge, hence we enforce a zero charge here.
        """
        rand_arr = np.random.randint(1, 10, size=offmol.n_atoms) / 100
        rand_arr[-1] = -sum(rand_arr[:-1])
        return rand_arr * unit.elementary_charge

    def check_propchgs(smc, charge_array):
        """
        Check that the partial charges we assigned to our offmol from which
        the smc was constructed are present and the right ones.
        """
        prop_chgs = smc.to_dict()['molprops']['atom.dprop.PartialCharge']
        prop_chgs = np.array(prop_chgs.split(), dtype=float)
        np.testing.assert_allclose(prop_chgs, charge_array.m)

    # Create new smc with overriden charges
    benzene_offmol = benzene_modifications['benzene'].to_openff()
    toluene_offmol = benzene_modifications['toluene'].to_openff()
    benzene_rand_chg = assign_fictitious_charges(benzene_offmol)
    toluene_rand_chg = assign_fictitious_charges(toluene_offmol)
    benzene_offmol.partial_charges = benzene_rand_chg
    toluene_offmol.partial_charges = toluene_rand_chg
    benzene_smc = openfe.SmallMoleculeComponent.from_openff(benzene_offmol)
    toluene_smc = openfe.SmallMoleculeComponent.from_openff(toluene_offmol)

    # Check that the new smcs have the new overriden charges
    check_propchgs(benzene_smc, benzene_rand_chg)
    check_propchgs(toluene_smc, toluene_rand_chg)

    # Create new mapping
    mapper = openfe.setup.LomapAtomMapper(element_change=False)
    mapping = next(mapper.suggest_mappings(benzene_smc, toluene_smc))

    # create DAG from protocol and take first (and only) work unit from within
    dag = protocol.create(
        stateA=openfe.ChemicalSystem({'l': benzene_smc,}),
        stateB=openfe.ChemicalSystem({'l': toluene_smc,}),
        mapping={'ligand': mapping},
    )
    dag_unit = list(dag.protocol_units)[0]

    with tmpdir.as_cwd():
        sampler = dag_unit.run(dry=True)['debug']['sampler']
        htf = sampler._factory
        hybrid_system = htf.hybrid_system

        # get the standard nonbonded force
        nonbond = [f for f in hybrid_system.getForces()
                   if isinstance(f, NonbondedForce)]
        assert len(nonbond) == 1

        # get the particle parameter offsets
        c_offsets = {}
        for i in range(nonbond[0].getNumParticleParameterOffsets()):
            offset = nonbond[0].getParticleParameterOffset(i)
            c_offsets[offset[1]] = ensure_quantity(offset[2], 'openff')

        # Here is a bit of exposition on what we're doing
        # HTF creates two sets of nonbonded forces, a standard one (for the
        # PME) and a custom one (for sterics).
        # Here we specifically check charges, so we only concentrate on the
        # standard NonbondedForce.
        # The way the NonbondedForce is constructed is as follows:
        # - unique old atoms:
        #  * The particle charge is set to the input molA particle charge
        #  * The chargeScale offset is set to the negative value of the molA
        #    particle charge (such that by scaling you effectively zero out
        #    the charge.
        # - unique new atoms:
        #  * The particle charge is set to zero (doesn't exist in the starting
        #    end state).
        #  * The chargeScale offset is set to the value of the molB particle
        #    charge (such that by scaling you effectively go from 0 to molB
        #    charge).
        # - core atoms:
        #  * The particle charge is set to the input molA particle charge
        #    (i.e. we start from a system that has molA charges).
        #  * The particle charge offset is set to the difference between
        #    the molB particle charge and the molA particle charge (i.e.
        #    we scale by that difference to get to the value of the molB
        #    particle charge).
        for i in range(hybrid_system.getNumParticles()):
            c, s, e = nonbond[0].getParticleParameters(i)
            # get the particle charge (c)
            c = ensure_quantity(c, 'openff')
            # particle charge (c) is equal to molA particle charge
            # offset (c_offsets) is equal to -(molA particle charge)
            if i in htf._atom_classes['unique_old_atoms']:
                idx = htf._hybrid_to_old_map[i]
                np.testing.assert_allclose(c, benzene_rand_chg[idx])
                np.testing.assert_allclose(c_offsets[i], -benzene_rand_chg[idx])
            # particle charge (c) is equal to 0
            # offset (c_offsets) is equal to molB particle charge
            elif i in htf._atom_classes['unique_new_atoms']:
                idx = htf._hybrid_to_new_map[i]
                np.testing.assert_allclose(c, 0 * unit.elementary_charge)
                np.testing.assert_allclose(c_offsets[i], toluene_rand_chg[idx])
            # particle charge (c) is equal to molA particle charge
            # offset (c_offsets) is equal to difference between molB and molA
            elif i in htf._atom_classes['core_atoms']:
                old_i = htf._hybrid_to_old_map[i]
                new_i = htf._hybrid_to_new_map[i]
                c_exp = toluene_rand_chg[new_i] - benzene_rand_chg[old_i]
                np.testing.assert_allclose(c, benzene_rand_chg[old_i])
                np.testing.assert_allclose(c_offsets[i], c_exp)


def test_virtual_sites_no_reassign(benzene_system, toluene_system,
                                   benzene_to_toluene_mapping, tmpdir):
    """
    Because of some as-of-yet not fully identified issue, not reassigning
    velocities will cause systems to NaN.
    See https://github.com/choderalab/openmmtools/issues/695
    """
    settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
    settings.forcefield_settings.forcefields = [
        "amber/ff14SB.xml",    # ff14SB protein force field
        "amber/tip4pew_standard.xml", # FF we are testsing with the fun VS
        "amber/phosaa10.xml",  # Handles THE TPO
    ]
    settings.system_settings.nonbonded_cutoff = 0.9 * unit.nanometer
    settings.solvation_settings.solvent_model = 'tip4pew'
    settings.integrator_settings.reassign_velocities = False

    protocol = openmm_rfe.RelativeHybridTopologyProtocol(
            settings=settings,
    )
    dag = protocol.create(
        stateA=benzene_system,
        stateB=toluene_system,
        mapping={'ligand': benzene_to_toluene_mapping},
    )
    dag_unit = list(dag.protocol_units)[0]

    with tmpdir.as_cwd():
        errmsg = "Simulations with virtual sites without velocity"
        with pytest.raises(ValueError, match=errmsg):
            dag_unit.run(dry=True)


@pytest.mark.slow
@pytest.mark.parametrize('method', ['repex', 'sams', 'independent'])
def test_dry_run_complex(benzene_complex_system, toluene_complex_system,
                         benzene_to_toluene_mapping, method, tmpdir):
    # this will be very time consuming
    settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
    settings.alchemical_sampler_settings.sampler_method = method
    settings.alchemical_sampler_settings.n_repeats = 1
    settings.simulation_settings.output_indices = 'protein or resname  UNK'

    protocol = openmm_rfe.RelativeHybridTopologyProtocol(
            settings=settings,
    )
    dag = protocol.create(
        stateA=benzene_complex_system,
        stateB=toluene_complex_system,
        mapping={'ligand': benzene_to_toluene_mapping},
    )
    dag_unit = list(dag.protocol_units)[0]

    with tmpdir.as_cwd():
        sampler = dag_unit.run(dry=True)['debug']['sampler']
        assert isinstance(sampler, MultiStateSampler)
        assert sampler.is_periodic
        assert isinstance(sampler._thermodynamic_states[0].barostat,
                          MonteCarloBarostat)
        assert sampler._thermodynamic_states[1].pressure == 1 * omm_unit.bar

        # Check we have the right number of atoms in the PDB
        pdb = mdt.load_pdb('hybrid_system.pdb')
        assert pdb.n_atoms == 2629


def test_lambda_schedule_default():
    lambdas = openmm_rfe._rfe_utils.lambdaprotocol.LambdaProtocol(functions='default')
    assert len(lambdas.lambda_schedule) == 10


@pytest.mark.parametrize('windows', [11, 6, 9000])
def test_lambda_schedule(windows):
    lambdas = openmm_rfe._rfe_utils.lambdaprotocol.LambdaProtocol(
            functions='default', windows=windows)
    assert len(lambdas.lambda_schedule) == windows


def test_hightimestep(benzene_vacuum_system,
                      toluene_vacuum_system,
                      benzene_to_toluene_mapping, tmpdir):
    settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
    settings.forcefield_settings.hydrogen_mass = 1.0
    settings.system_settings.nonbonded_method = 'nocutoff'

    p = openmm_rfe.RelativeHybridTopologyProtocol(
            settings=settings,
    )

    dag = p.create(
        stateA=benzene_vacuum_system,
        stateB=toluene_vacuum_system,
        mapping={'ligand': benzene_to_toluene_mapping},
    )
    dag_unit = list(dag.protocol_units)[0]

    errmsg = "too large for hydrogen mass"
    with tmpdir.as_cwd():
        with pytest.raises(ValueError, match=errmsg):
            dag_unit.run(dry=True)


def test_n_replicas_not_n_windows(benzene_vacuum_system,
                                  toluene_vacuum_system,
                                  benzene_to_toluene_mapping, tmpdir):
    # For PR #125 we pin such that the number of lambda windows
    # equals the numbers of replicas used - TODO: remove limitation
    settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
    # default lambda windows is 11
    settings.alchemical_sampler_settings.n_replicas = 13
    settings.system_settings.nonbonded_method = 'nocutoff'

    errmsg = ("Number of replicas 13 does not equal the number of "
              "lambda windows 11")

    with tmpdir.as_cwd():
        with pytest.raises(ValueError, match=errmsg):
            p = openmm_rfe.RelativeHybridTopologyProtocol(
                    settings=settings,
            )
            dag = p.create(
                stateA=benzene_vacuum_system,
                stateB=toluene_vacuum_system,
                mapping={'ligand': benzene_to_toluene_mapping},
            )
            dag_unit = list(dag.protocol_units)[0]
            dag_unit.run(dry=True)


def test_missing_ligand(benzene_system, benzene_to_toluene_mapping):
    # state B doesn't have a ligand component
    stateB = openfe.ChemicalSystem({'solvent': openfe.SolventComponent()})

    p = openmm_rfe.RelativeHybridTopologyProtocol(
        settings=openmm_rfe.RelativeHybridTopologyProtocol.default_settings(),
    )

    match_str = "missing alchemical components in stateB"
    with pytest.raises(ValueError, match=match_str):
        _ = p.create(
            stateA=benzene_system,
            stateB=stateB,
            mapping={'ligand': benzene_to_toluene_mapping},
        )


def test_vaccuum_PME_error(benzene_vacuum_system, benzene_modifications,
                           benzene_to_toluene_mapping):
    # state B doesn't have a solvent component (i.e. its vacuum)
    stateB = openfe.ChemicalSystem({'ligand': benzene_modifications['toluene']})

    p = openmm_rfe.RelativeHybridTopologyProtocol(
        settings=openmm_rfe.RelativeHybridTopologyProtocol.default_settings(),
    )
    errmsg = "PME cannot be used for vacuum transform"
    with pytest.raises(ValueError, match=errmsg):
        _ = p.create(
            stateA=benzene_vacuum_system,
            stateB=stateB,
            mapping={'ligand': benzene_to_toluene_mapping},
        )


def test_incompatible_solvent(benzene_system, benzene_modifications,
                              benzene_to_toluene_mapping):
    # the solvents are different
    stateB = openfe.ChemicalSystem(
        {'ligand': benzene_modifications['toluene'],
         'solvent': openfe.SolventComponent(
             positive_ion='K', negative_ion='Cl')}
    )

    p = openmm_rfe.RelativeHybridTopologyProtocol(
        settings=openmm_rfe.RelativeHybridTopologyProtocol.default_settings(),
    )
    # We don't have a way to map non-ligand components so for now it
    # just triggers that it's not a mapped component
    errmsg = "missing alchemical components in stateA"
    with pytest.raises(ValueError, match=errmsg):
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

    p = openmm_rfe.RelativeHybridTopologyProtocol(
        settings=openmm_rfe.RelativeHybridTopologyProtocol.default_settings(),
    )
    errmsg = (r"Unmapped alchemical component "
              r"SmallMoleculeComponent\(name=toluene\)")
    with pytest.raises(ValueError, match=errmsg):
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

    p = openmm_rfe.RelativeHybridTopologyProtocol(
        settings=openmm_rfe.RelativeHybridTopologyProtocol.default_settings(),
    )
    errmsg = (r"Unmapped alchemical component "
              r"SmallMoleculeComponent\(name=benzene\)")
    with pytest.raises(ValueError, match=errmsg):
        _ = p.create(
            stateA=benzene_system,
            stateB=toluene_system,
            mapping={'ligand': mapping},
        )


def test_complex_mismatch(benzene_system, toluene_complex_system,
                          benzene_to_toluene_mapping):
    # only one complex
    p = openmm_rfe.RelativeHybridTopologyProtocol(
        settings=openmm_rfe.RelativeHybridTopologyProtocol.default_settings(),
    )
    with pytest.raises(ValueError):
        _ = p.create(
            stateA=benzene_system,
            stateB=toluene_complex_system,
            mapping={'ligand': benzene_to_toluene_mapping},
        )


def test_too_many_specified_mappings(benzene_system, toluene_system,
                                 benzene_to_toluene_mapping):
    # mapping dict requires 'ligand' key
    p = openmm_rfe.RelativeHybridTopologyProtocol(
        settings=openmm_rfe.RelativeHybridTopologyProtocol.default_settings(),
    )
    errmsg = "A single LigandAtomMapping is expected for this Protocol"
    with pytest.raises(ValueError, match=errmsg):
        _ = p.create(
            stateA=benzene_system,
            stateB=toluene_system,
            mapping={'solvent': benzene_to_toluene_mapping,
                     'ligand': benzene_to_toluene_mapping,}
        )


def test_protein_mismatch(benzene_complex_system, toluene_complex_system,
                          benzene_to_toluene_mapping):
    # hack one protein to be labelled differently
    prot = toluene_complex_system['protein']
    alt_prot = openfe.ProteinComponent(prot.to_rdkit(),
                                       name='Mickey Mouse')
    alt_toluene_complex_system = openfe.ChemicalSystem(
                 {'ligand': toluene_complex_system['ligand'],
                  'solvent': toluene_complex_system['solvent'],
                  'protein': alt_prot}
    )

    p = openmm_rfe.RelativeHybridTopologyProtocol(
        settings=openmm_rfe.RelativeHybridTopologyProtocol.default_settings(),
    )
    with pytest.raises(ValueError):
        _ = p.create(
            stateA=benzene_complex_system,
            stateB=alt_toluene_complex_system,
            mapping={'ligand': benzene_to_toluene_mapping},
        )


def test_element_change_warning(atom_mapping_basic_test_files):
    # check a mapping with element change gets rejected early
    l1 = atom_mapping_basic_test_files['2-methylnaphthalene']
    l2 = atom_mapping_basic_test_files['2-naftanol']

    mapper = setup.LomapAtomMapper()
    mapping = next(mapper.suggest_mappings(l1, l2))

    sys1 = openfe.ChemicalSystem(
        {'ligand': l1, 'solvent': openfe.SolventComponent()},
    )
    sys2 = openfe.ChemicalSystem(
        {'ligand': l2, 'solvent': openfe.SolventComponent()},
    )

    p = openmm_rfe.RelativeHybridTopologyProtocol(
        settings=openmm_rfe.RelativeHybridTopologyProtocol.default_settings(),
    )
    with pytest.warns(UserWarning, match="Element change"):
        _ = p.create(
            stateA=sys1, stateB=sys2,
            mapping={'ligand': mapping},
        )


def test_ligand_overlap_warning(benzene_vacuum_system, toluene_vacuum_system,
                                benzene_to_toluene_mapping, tmpdir):
    vac_settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
    vac_settings.system_settings.nonbonded_method = 'nocutoff'

    protocol = openmm_rfe.RelativeHybridTopologyProtocol(
            settings=vac_settings,
    )

    # update atom positions
    sysA = benzene_vacuum_system
    rdmol = benzene_vacuum_system['ligand'].to_rdkit()
    conf = rdmol.GetConformer()

    for atm in range(rdmol.GetNumAtoms()):
        x, y, z = conf.GetAtomPosition(atm)
        conf.SetAtomPosition(atm, Point3D(x+3, y, z))

    new_ligand = openfe.SmallMoleculeComponent.from_rdkit(
        rdmol, name=benzene_vacuum_system['ligand'].name
    )
    components = dict(benzene_vacuum_system.components)
    components['ligand'] = new_ligand
    sysA = openfe.ChemicalSystem(components)

    mapping = benzene_to_toluene_mapping.copy_with_replacements(
        componentA=new_ligand
    )

    # Specifically check that the first pair throws a warning
    with pytest.warns(UserWarning, match='0 : 4 deviates'):
        dag = protocol.create(
            stateA=sysA, stateB=toluene_vacuum_system,
            mapping={'ligand': mapping},
            )
        dag_unit = list(dag.protocol_units)[0]
        with tmpdir.as_cwd():
            dag_unit.run(dry=True)


@pytest.fixture
def solvent_protocol_dag(benzene_system, toluene_system, benzene_to_toluene_mapping):
    settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()

    protocol = openmm_rfe.RelativeHybridTopologyProtocol(
        settings=settings,
    )

    return protocol.create(
        stateA=benzene_system, stateB=toluene_system,
        mapping={'ligand': benzene_to_toluene_mapping},
    )


def test_unit_tagging(solvent_protocol_dag, tmpdir):
    # test that executing the Units includes correct generation and repeat info
    dag_units = solvent_protocol_dag.protocol_units
    with mock.patch('openfe.protocols.openmm_rfe.equil_rfe_methods.RelativeHybridTopologyProtocolUnit.run',
                    return_value={'nc': 'file.nc', 'last_checkpoint': 'chk.nc'}):
        results = []
        for u in dag_units:
            ret = u.execute(context=gufe.Context(tmpdir, tmpdir))
            results.append(ret)

    repeats = set()
    for ret in results:
        assert isinstance(ret, gufe.ProtocolUnitResult)
        assert ret.outputs['generation'] == 0
        repeats.add(ret.outputs['repeat_id'])
    # repeats are random ints, so check we got 3 individual numbers
    assert len(repeats) == 3


def test_gather(solvent_protocol_dag, tmpdir):
    # check .gather behaves as expected
    with mock.patch('openfe.protocols.openmm_rfe.equil_rfe_methods.RelativeHybridTopologyProtocolUnit.run',
                    return_value={'nc': 'file.nc', 'last_checkpoint': 'chk.nc'}):
        dagres = gufe.protocols.execute_DAG(solvent_protocol_dag,
                                            shared_basedir=tmpdir,
                                            scratch_basedir=tmpdir,
                                            keep_shared=True)

    prot = openmm_rfe.RelativeHybridTopologyProtocol(
        settings=openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
    )

    res = prot.gather([dagres])

    assert isinstance(res, openmm_rfe.RelativeHybridTopologyProtocolResult)


class TestConstraintRemoval:
    @staticmethod
    def make_systems(ligA: openfe.SmallMoleculeComponent,
                     ligB: openfe.SmallMoleculeComponent,
                     constraints):
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
            constraints=constraints,
            rigidWater=True,
            hydrogenMass=None,
            removeCMMotion=True,
        )

        stateB_topology, _ = openmm_rfe._rfe_utils.topologyhelpers.combined_topology(
            stateA_topology,
            ligB.to_openff().to_topology().to_openmm(),
            exclude_resids=np.array([r.index for r in list(stateA_topology.residues())])
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
            constraints=constraints,
            rigidWater=True,
            hydrogenMass=None,
            removeCMMotion=True,
        )

        return stateA_topology, stateA_system, stateB_topology, stateB_system

    @pytest.mark.parametrize('reverse', [False, True])
    def test_remove_constraints_lengthchange(self, benzene_modifications,
                                             reverse):
        # check that mappings are correctly corrected to avoid changes in
        # constraint length
        # use a phenol->toluene transform to test
        ligA = benzene_modifications['phenol']
        ligB = benzene_modifications['toluene']

        mapping = {0: 4, 1: 5, 2: 6, 3: 7, 4: 8, 5: 9, 6: 10,
                   7: 11, 8: 12, 9: 13, 10: 1, 11: 14, 12: 2}

        expected = 10  # this should get removed from mapping

        if reverse:
            ligA, ligB = ligB, ligA
            expected = mapping[expected]
            mapping = {v: k for k, v in mapping.items()}

        mapping = setup.LigandAtomMapping(
            componentA=ligA,
            componentB=ligB,
            # this is default lomap
            # importantly the H in -OH maps to one of the -CH3
            # this constraint will change length
            componentA_to_componentB=mapping,
        )

        stateA_topology, stateA_system, stateB_topology, stateB_system = self.make_systems(
            ligA, ligB, constraints=app.HBonds)

        # this normally requires global indices, however as ligandA/B is only thing
        # in system, this mapping is still correct
        ret = openmm_rfe._rfe_utils.topologyhelpers._remove_constraints(
            mapping.componentA_to_componentB,
            stateA_system, stateA_topology,
            stateB_system, stateB_topology,
        )

        # all of this just to check that an entry was removed from the mapping
        # the removed constraint
        assert expected not in ret
        # but only one constraint should be removed
        assert len(ret) == len(mapping.componentA_to_componentB) - 1

    @pytest.mark.parametrize('reverse', [False, True])
    def test_constraint_to_harmonic(self, benzene_modifications, reverse):
        ligA = benzene_modifications['benzene']
        ligB = benzene_modifications['toluene']
        expected = 10
        mapping = {0: 4, 1: 5, 2: 6, 3: 7, 4: 8, 5: 9,
                   6: 10, 7: 11, 8: 12, 9: 13, 10: 2, 11: 14}
        if reverse:
            ligA, ligB = ligB, ligA
            expected = mapping[expected]
            mapping = {v: k for k, v in mapping.items()}

        # this maps a -H to a -C, so the constraint on -H turns into a C-C bond
        # H constraint is A(4, 10) and C-C is B(8, 2)
        mapping = setup.LigandAtomMapping(
            componentA=ligA, componentB=ligB,
            componentA_to_componentB=mapping
        )

        stateA_topology, stateA_system, stateB_topology, stateB_system = self.make_systems(
            ligA, ligB, constraints=app.HBonds)

        ret = openmm_rfe._rfe_utils.topologyhelpers._remove_constraints(
            mapping.componentA_to_componentB,
            stateA_system, stateA_topology,
            stateB_system, stateB_topology,
        )

        assert expected not in ret
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

        stateA_topology, stateA_system, stateB_topology, stateB_system = self.make_systems(
            ligA, ligB, constraints=app.HBonds)

        ret = openmm_rfe._rfe_utils.topologyhelpers._remove_constraints(
            mapping.componentA_to_componentB,
            stateA_system, stateA_topology,
            stateB_system, stateB_topology,
        )

        assert 0 not in ret
        assert len(ret) == len(mapping.componentA_to_componentB) - 1

    @pytest.mark.parametrize('reverse', [False, True])
    def test_non_H_constraint_fail(self, benzene_modifications, reverse):
        # here we specify app.AllBonds constraints
        # in this transform, the C-C[#N] to C-C[=O] constraint changes length
        # indices A(8, 2) to B(6, 1)
        # there's no Hydrogen involved so we can't trivially figure out the
        # best atom to remove from mapping
        # (but it would be 2 [& 1] in this case..)
        ligA = benzene_modifications['toluene']
        ligB = benzene_modifications['benzonitrile']

        mapping = {0: 0, 2: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8,
                   11: 9, 12: 10, 13: 11, 14: 12}

        if reverse:
            ligA, ligB = ligB, ligA
            mapping = {v: k for k, v in mapping.items()}

        mapping = setup.LigandAtomMapping(
            componentA=ligA, componentB=ligB,
            componentA_to_componentB=mapping,
        )

        stateA_topology, stateA_system, stateB_topology, stateB_system = self.make_systems(
            ligA, ligB, constraints=app.AllBonds)

        with pytest.raises(ValueError, match='resolve constraint') as e:
            _ = openmm_rfe._rfe_utils.topologyhelpers._remove_constraints(
                mapping.componentA_to_componentB,
                stateA_system, stateA_topology,
                stateB_system, stateB_topology,
            )
        if not reverse:
            assert 'A: 2-8 B: 1-6' in str(e)
        else:
            assert 'A: 1-6 B: 2-8' in str(e)


@pytest.fixture(scope='session')
def tyk2_xml(tmp_path_factory):
    with resources.files('openfe.tests.data.openmm_rfe') as d:
        fn1 = str(d / 'ligand_23.sdf')
        fn2 = str(d / 'ligand_55.sdf')
    lig23 = openfe.SmallMoleculeComponent.from_sdf_file(fn1)
    lig55 = openfe.SmallMoleculeComponent.from_sdf_file(fn2)

    mapping = setup.LigandAtomMapping(
        componentA=lig23, componentB=lig55,
        # perses mapper output
        componentA_to_componentB={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
                                  7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12,
                                  13: 13, 14: 14, 15: 15, 16: 16, 17: 17,
                                  18: 18, 23: 19, 26: 20, 27: 21, 28: 22,
                                  29: 23, 30: 24, 31: 25, 32: 26, 33: 27}
    )

    settings: openmm_rfe.RelativeHybridTopologyProtocolSettings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
    settings.forcefield_settings.small_molecule_forcefield = 'openff-2.0.0'
    settings.system_settings.nonbonded_method = 'nocutoff'
    settings.forcefield_settings.hydrogen_mass = 3.0
    settings.alchemical_sampler_settings.n_repeats = 1

    protocol = openmm_rfe.RelativeHybridTopologyProtocol(settings)

    dag = protocol.create(
        stateA=openfe.ChemicalSystem({'ligand': lig23}),
        stateB=openfe.ChemicalSystem({'ligand': lig55}),
        mapping={'ligand': mapping},
    )
    pu = list(dag.protocol_units)[0]

    tmp = tmp_path_factory.mktemp('xml_reg')

    dryrun = pu.run(dry=True, shared_basepath=tmp)

    system = dryrun['debug']['sampler']._hybrid_factory.hybrid_system

    return ET.fromstring(XmlSerializer.serialize(system))


@pytest.fixture(scope='session')
def tyk2_reference_xml():
    with resources.files('openfe.tests.data.openmm_rfe') as d:
        f = d / 'reference.xml'
        with open(f, 'r') as i:
            xmldata = i.read()
    return ET.fromstring(xmldata)


@pytest.mark.slow
class TestTyk2XmlRegression:
    """Generates Hybrid system XML and performs regression test"""
    @staticmethod
    def test_particles(tyk2_xml, tyk2_reference_xml):
        # < Particle mass = "10.018727" / >
        particles = tyk2_xml.find('Particles')
        assert particles

        ref_particles = tyk2_reference_xml.find('Particles')

        for a, b in zip(particles, ref_particles):
            assert float(a.get('mass')) == pytest.approx(float(b.get('mass')))

    @staticmethod
    def test_constraints(tyk2_xml, tyk2_reference_xml):
        # <Constraint d=".1085358495916" p1="12" p2="31"/>
        constraints = tyk2_xml.find('Constraints')
        assert constraints

        ref_constraints = tyk2_reference_xml.find('Constraints')
        for a, b in zip(constraints, ref_constraints):
            assert a.get('p1') == b.get('p1')
            assert a.get('p2') == b.get('p2')
            assert float(a.get('d')) == pytest.approx(float(b.get('d')))


class TestProtocolResult:
    @pytest.fixture()
    def protocolresult(self, rfe_transformation_json):
        d = json.loads(rfe_transformation_json,
                       cls=gufe.tokenization.JSON_HANDLER.decoder)

        pr = openfe.ProtocolResult.from_dict(d['protocol_result'])

        return pr

    def test_reload_protocol_result(self, rfe_transformation_json):
        d = json.loads(rfe_transformation_json,
                       cls=gufe.tokenization.JSON_HANDLER.decoder)

        pr = openmm_rfe.RelativeHybridTopologyProtocolResult.from_dict(d['protocol_result'])

        assert pr

    def test_get_estimate(self, protocolresult):
        est = protocolresult.get_estimate()

        assert est
        assert est.m == pytest.approx(16.896889)
        assert isinstance(est, unit.Quantity)
        assert est.is_compatible_with(unit.kilojoule_per_mole)

    def test_get_uncertainty(self, protocolresult):
        est = protocolresult.get_uncertainty()

        assert est
        assert est.m == pytest.approx(0.1102014)
        assert isinstance(est, unit.Quantity)
        assert est.is_compatible_with(unit.kilojoule_per_mole)

    def test_get_individual(self, protocolresult):
        inds = protocolresult.get_individual_estimates()

        assert isinstance(inds, list)
        assert len(inds) == 3
        for e, u in inds:
            assert e.is_compatible_with(unit.kilojoule_per_mole)
            assert u.is_compatible_with(unit.kilojoule_per_mole)

    def test_get_forwards_etc(self, protocolresult):
        far = protocolresult.get_forward_and_reverse_energy_analysis()

        assert isinstance(far, list)
        far1 = far[0]
        assert isinstance(far1, dict)
        for k in ['fractions', 'forward_DGs', 'forward_dDGs',
                  'reverse_DGs', 'reverse_dDGs']:
            assert k in far1

            if k == 'fractions':
                assert isinstance(far1[k], np.ndarray)
            else:
                assert isinstance(far1[k], unit.Quantity)
                assert far1[k].is_compatible_with(unit.kilojoule_per_mole)

    def test_get_overlap_matrices(self, protocolresult):
        ovp = protocolresult.get_overlap_matrices()

        assert isinstance(ovp, list)
        assert len(ovp) == 3

        ovp1 = ovp[0]
        assert isinstance(ovp1['matrix'], np.ndarray)
        assert ovp1['matrix'].shape == (11,11)

    def test_get_replica_transition_statistics(self, protocolresult):
        rpx = protocolresult.get_replica_transition_statistics()

        assert isinstance(rpx, list)
        assert len(rpx) == 3
        rpx1 = rpx[0]
        assert 'eigenvalues' in rpx1
        assert 'matrix' in rpx1
        assert rpx1['eigenvalues'].shape == (11,)
        assert rpx1['matrix'].shape == (11, 11)

    def test_equilibration_iterations(self, protocolresult):
        eq = protocolresult.equilibration_iterations()

        assert isinstance(eq, list)
        assert len(eq) == 3
        assert all(isinstance(v, float) for v in eq)

    def test_production_iterations(self, protocolresult):
        prod = protocolresult.production_iterations()

        assert isinstance(prod, list)
        assert len(prod) == 3
        assert all(isinstance(v, float) for v in prod)

    def test_filenotfound_replica_states(self, protocolresult):
        errmsg = "File could not be found"

        with pytest.raises(ValueError, match=errmsg):
            protocolresult.get_replica_states()

@pytest.mark.parametrize('mapping_name,result', [
    ["benzene_to_toluene_mapping", 0],
    ["benzene_to_benzoic_mapping", 1],
    ["benzene_to_aniline_mapping", -1],
    ["aniline_to_benzene_mapping", 1],
])
def test_get_charge_difference(mapping_name, result, request):
    mapping = request.getfixturevalue(mapping_name)
    if result != 0:
        ion = 'Na\+' if result == -1 else 'Cl\-'
        wmsg = (f"A charge difference of {result} is observed "
                "between the end states. This will be addressed by "
                f"transforming a water into a {ion} ion")
        with pytest.warns(UserWarning, match=wmsg):
            val = _get_alchemical_charge_difference(
                mapping, 'pme', True, openfe.SolventComponent()
            )
            assert result == pytest.approx(result)
    else:
        val = _get_alchemical_charge_difference(
            mapping, 'pme', True, openfe.SolventComponent()
        )
        assert result == pytest.approx(result)


def test_get_charge_difference_no_pme(benzene_to_benzoic_mapping):
    errmsg = "Explicit charge correction when not using PME"
    with pytest.raises(ValueError, match=errmsg):
        _get_alchemical_charge_difference(
            benzene_to_benzoic_mapping,
            'nocutoff', True, openfe.SolventComponent(),
        )


def test_get_charge_difference_no_corr(benzene_to_benzoic_mapping):
    wmsg = ("A charge difference of 1 is observed between the end states. "
            "No charge correction has been requested")
    with pytest.warns(UserWarning, match=wmsg):
        _get_alchemical_charge_difference(
            benzene_to_benzoic_mapping,
            'pme', False, openfe.SolventComponent(),
        )


def test_greater_than_one_charge_difference_error(aniline_to_benzoic_mapping):
    errmsg = "A charge difference of 2"
    with pytest.raises(ValueError, match=errmsg):
        _get_alchemical_charge_difference(
            aniline_to_benzoic_mapping,
            'pme', True, openfe.SolventComponent(),
        )


@pytest.fixture(scope='session')
def benzene_solvent_openmm_system(benzene_modifications):
    smc = benzene_modifications['benzene']
    offmol = smc.to_openff()
    settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()

    system_generator = system_creation.get_system_generator(
        forcefield_settings=settings.forcefield_settings,
        thermo_settings=settings.thermo_settings,
        integrator_settings=settings.integrator_settings,
        system_settings=settings.system_settings,
        cache=None,
        has_solvent=True,
    )

    system_generator.create_system(
        offmol.to_topology().to_openmm(),
        molecules=[offmol],
    )

    modeller, _ = system_creation.get_omm_modeller(
        protein_comp=None,
        solvent_comp=openfe.SolventComponent(),
        small_mols={smc: offmol},
        omm_forcefield=system_generator.forcefield,
        solvent_settings=settings.solvation_settings,
    )

    topology = modeller.getTopology()
    positions = to_openmm(from_openmm(modeller.getPositions()))
    system = system_generator.create_system(
        topology,
        molecules=[offmol]
    )

    return system, topology, positions


@pytest.fixture(scope='session')
def benzene_tip4p_solvent_openmm_system(benzene_modifications):
    smc = benzene_modifications['benzene']
    offmol = smc.to_openff()
    settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
    settings.forcefield_settings.forcefields = [
        'amber/ff14SB.xml', 'amber/tip4pew_standard.xml', 'amber/phosaa10.xml'
    ]
    settings.solvation_settings.solvent_model = 'tip4pew'

    system_generator = system_creation.get_system_generator(
        forcefield_settings=settings.forcefield_settings,
        thermo_settings=settings.thermo_settings,
        integrator_settings=settings.integrator_settings,
        system_settings=settings.system_settings,
        cache=None,
        has_solvent=True,
    )

    system_generator.create_system(
        offmol.to_topology().to_openmm(),
        molecules=[offmol],
    )

    modeller, _ = system_creation.get_omm_modeller(
        protein_comp=None,
        solvent_comp=openfe.SolventComponent(),
        small_mols={smc: offmol},
        omm_forcefield=system_generator.forcefield,
        solvent_settings=settings.solvation_settings,
    )

    topology = modeller.getTopology()
    positions = to_openmm(from_openmm(modeller.getPositions()))
    system = system_generator.create_system(
        topology,
        molecules=[offmol]
    )

    return system, topology, positions


@pytest.fixture
def benzene_self_system_mapping(benzene_solvent_openmm_system):
    """
    A fictitious mapping of benzene to benzene where there is no
    alchemical transformation (this technically doesn't work in practice
    because the RFE protocol expects an alchemical component).
    """
    system, topology, positions = benzene_solvent_openmm_system

    res = [r for r in topology.residues()]
    benzene_res = [r for r in res if r.name == 'UNK'][0]
    benzene_ids = [a.index for a in benzene_res.atoms()]
    env_ids = [a.index for a in topology.atoms() if a.index not in benzene_ids]
    all_ids = [a.index for a in topology.atoms()]

    system_mapping = {
        'new_to_old_atom_map': {i: i for i in all_ids},
        'old_to_new_atom_map': {i: i for i in all_ids},
        'new_to_old_core_atom_map': {i: i for i in benzene_ids},
        'old_to_new_core_atom_map': {i: i for i in benzene_ids},
        'old_to_new_env_atom_map': {i: i for i in env_ids},
        'new_to_old_env_atom_map': {i: i for i in env_ids},
        'old_mol_indices': benzene_ids,
        'new_mol_indices': benzene_ids,
    }

    return system_mapping


@pytest.mark.parametrize('ion, water', [
    ['NA', 'SOL'],
    ['NX', 'WAT'],
])
def test_get_ion_water_parameters_unknownresname(
    ion, water, benzene_solvent_openmm_system
):
    system, topology, positions = benzene_solvent_openmm_system

    errmsg = "Error encountered when attempting to explicitly handle"

    with pytest.raises(ValueError, match=errmsg):
        topologyhelpers._get_ion_and_water_parameters(
            topology, system,
            ion_resname=ion, water_resname=water
        )


def test_get_alchemical_waters_no_waters(
    benzene_solvent_openmm_system,
):
    system, topology, positions = benzene_solvent_openmm_system

    errmsg = "There are no waters"

    with pytest.raises(ValueError, match=errmsg):
        topologyhelpers.get_alchemical_waters(
            topology, positions, charge_difference=1,
            distance_cutoff=2.0 * unit.nanometer
        )


def test_handle_alchemwats_incorrect_count(
    benzene_solvent_openmm_system,
):
    """
    Check that an error is thrown when charge_difference != len(water_resids)
    """
    system, topology, positions = benzene_solvent_openmm_system

    errmsg = "There should be as many alchemical water residues:"

    with pytest.raises(ValueError, match=errmsg):
        topologyhelpers.handle_alchemical_waters(
            water_resids=[1, 2, 3],
            topology=topology,
            system=system,
            system_mapping={},
            charge_difference=1,
            solvent_component=openfe.SolventComponent(),
        )


def test_handle_alchemwats_too_many_nbf(
    benzene_solvent_openmm_system,
):
    """
    Check that an error is thrown when there are multiple NonbondedForces
    """
    system, topology, positions = benzene_solvent_openmm_system

    new_system = copy.deepcopy(system)
    new_system.addForce(NonbondedForce())

    errmsg = "Too many NonbondedForce forces"

    with pytest.raises(ValueError, match=errmsg):
        topologyhelpers.handle_alchemical_waters(
            water_resids=[1,],
            topology=topology,
            system=new_system,
            system_mapping={},
            charge_difference=1,
            solvent_component=openfe.SolventComponent(),
        )


def test_handle_alchemwats_vsite_water(
    benzene_tip4p_solvent_openmm_system,
):
    """
    Check that an error is thrown when trying to use a 4 site
    water as an alchemical species
    """
    system, topology, positions = benzene_tip4p_solvent_openmm_system

    errmsg = "Non 3-site waters"

    with pytest.raises(ValueError, match=errmsg):
        topologyhelpers.handle_alchemical_waters(
            water_resids=[1,],
            topology=topology,
            system=system,
            system_mapping={},
            charge_difference=1,
            solvent_component=openfe.SolventComponent(),
        )


def test_handle_alchemwats_incorrect_atom(
    benzene_solvent_openmm_system,
    benzene_self_system_mapping,
):
    """
    Check that an error is thrown when charge_difference != len(water_resids)
    """
    system, topology, positions = benzene_solvent_openmm_system

    # modify the charge of hydrogen atom 25
    new_system = copy.deepcopy(system)  # protect the session scoped object
    nbf = [i for i in new_system.getForces() if isinstance(i, NonbondedForce)][0]
    c, s, e = nbf.getParticleParameters(25)
    nbf.setParticleParameters(25, 1 * omm_unit.elementary_charge, s, e)

    errmsg = "modifying an atom that doesn't match"

    with pytest.raises(ValueError, match=errmsg):
        topologyhelpers.handle_alchemical_waters(
            water_resids=[5,],
            topology=topology,
            system=new_system,
            system_mapping=benzene_self_system_mapping,
            charge_difference=1,
            solvent_component=openfe.SolventComponent(),
        )


def test_handle_alchemical_wats(
    benzene_solvent_openmm_system,
    benzene_self_system_mapping,
):
    system, topology, positions = benzene_solvent_openmm_system

    n_env = len(benzene_self_system_mapping['old_to_new_env_atom_map'])
    n_core = len(benzene_self_system_mapping['old_to_new_core_atom_map'])

    topologyhelpers.handle_alchemical_waters(
        water_resids=[5,],
        topology=topology,
        system=system,
        system_mapping=benzene_self_system_mapping,
        charge_difference=1,
        solvent_component=openfe.SolventComponent(),
    )

    # check the mappings
    old_new_env = benzene_self_system_mapping['old_to_new_env_atom_map']
    old_new_core = benzene_self_system_mapping['old_to_new_core_atom_map']
    assert len(old_new_env) == n_env - 3
    assert old_new_env == benzene_self_system_mapping['new_to_old_env_atom_map']
    assert len(old_new_core) == n_core + 3
    assert old_new_core == benzene_self_system_mapping['new_to_old_core_atom_map']
    expected_old_new_core = {i: i for i in range(12)} | {24: 24, 25: 25, 26: 26}
    assert old_new_core == expected_old_new_core

    # system parameters checks
    nbf = [i for i in system.getForces() if isinstance(i, NonbondedForce)][0]
    # check the oxygen parameters
    i_chg, i_sig, i_eps, o_chg, h_chg = topologyhelpers._get_ion_and_water_parameters(
        topology, system, 'NA', 'HOH',
    )

    charge, sigma, epsilon = nbf.getParticleParameters(24)
    assert charge == 1.0 * omm_unit.elementary_charge == i_chg
    assert sigma == i_sig
    assert epsilon == i_eps

    # check the hydrogen parameters
    for i in [25, 26]:
        charge, _, _ = nbf.getParticleParameters(i)
        assert charge == 0.0 * omm_unit.elementary_charge


def _assert_total_charge(system, atom_classes, chgA, chgB):
    nonbond = [
        f for f in system.getForces() if isinstance(f, NonbondedForce)
    ]

    offsets = {}
    for i in range(nonbond[0].getNumParticleParameterOffsets()):
        offset = nonbond[0].getParticleParameterOffset(i)
        assert len(offset) == 5
        offsets[offset[1]] = ensure_quantity(offset[2], 'openff')

    stateA_charges = np.zeros(system.getNumParticles())
    stateB_charges = np.zeros(system.getNumParticles())

    for i in range(system.getNumParticles()):
        # get the particle charge (c) and the chargeScale offset (c_offset)
        c, s, e = nonbond[0].getParticleParameters(i)
        c = ensure_quantity(c, 'openff')

        # particle charge (c) is equal to molA particle charge
        # offset (c_offset) is equal to -(molA particle charge)
        if i in atom_classes['unique_old_atoms']:
            stateA_charges[i] = c.m
        # particle charge (c) is equal to 0
        # offset (c_offset) is equal to molB particle charge
        elif i in atom_classes['unique_new_atoms']:
            stateB_charges[i] = offsets[i].m
        # particle charge (c) is equal to molA particle charge
        # offset (c_offset) is equal to difference between molB and molA
        elif i in atom_classes['core_atoms']:
            stateA_charges[i] = c.m
            stateB_charges[i] = c.m + offsets[i].m
        # an environment atom
        else:
            assert i in atom_classes['environment_atoms']
            stateA_charges[i] = c.m
            stateB_charges[i] = c.m

    assert chgA == pytest.approx(np.sum(stateA_charges))
    assert chgB == pytest.approx(np.sum(stateB_charges))


def test_dry_run_alchemwater_solvent(benzene_to_benzoic_mapping, tmpdir):
    stateA_system = openfe.ChemicalSystem(
        {'ligand': benzene_to_benzoic_mapping.componentA,
         'solvent': openfe.SolventComponent()}
    )
    stateB_system = openfe.ChemicalSystem(
        {'ligand': benzene_to_benzoic_mapping.componentB,
         'solvent': openfe.SolventComponent()}
    )
    solv_settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
    solv_settings.alchemical_settings.explicit_charge_correction = True
    protocol = openmm_rfe.RelativeHybridTopologyProtocol(
            settings=solv_settings,
    )

    # create DAG from protocol and take first (and only) work unit from within
    dag = protocol.create(
        stateA=stateA_system,
        stateB=stateB_system,
        mapping={'ligand': benzene_to_benzoic_mapping},
    )
    unit = list(dag.protocol_units)[0]

    with tmpdir.as_cwd():
        sampler = unit.run(dry=True)['debug']['sampler']
        htf = sampler._factory
        _assert_total_charge(htf.hybrid_system,
                             htf._atom_classes, 0, 0)

        assert len(htf._atom_classes['core_atoms']) == 14
        assert len(htf._atom_classes['unique_new_atoms']) == 3
        assert len(htf._atom_classes['unique_old_atoms']) == 1


@pytest.mark.slow
@pytest.mark.parametrize('mapping_name,chgA,chgB,correction,core_atoms,new_uniq,old_uniq', [
    ['benzene_to_aniline_mapping', 0, 1, False, 11, 4, 1],
    ['aniline_to_benzene_mapping', 0, 0, True, 14, 1, 4],
    ['aniline_to_benzene_mapping', 0, -1, False, 11, 1, 4],
    ['benzene_to_benzoic_mapping', 0, 0, True, 14, 3, 1],
    ['benzene_to_benzoic_mapping', 0, -1, False, 11, 3, 1],
    ['benzoic_to_benzene_mapping', 0, 0, True, 14, 1, 3],
    ['benzoic_to_benzene_mapping', 0, 1, False, 11, 1, 3],
])
def test_dry_run_complex_alchemwater_totcharge(
    mapping_name, chgA, chgB, correction, core_atoms,
    new_uniq, old_uniq, tmpdir, request, T4_protein_component,
):

    mapping = request.getfixturevalue(mapping_name)
    stateA_system = openfe.ChemicalSystem(
        {'ligand': mapping.componentA,
         'solvent': openfe.SolventComponent(),
         'protein': T4_protein_component}
    )
    stateB_system = openfe.ChemicalSystem(
        {'ligand': mapping.componentB,
         'solvent': openfe.SolventComponent(),
         'protein': T4_protein_component}
    )

    solv_settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
    solv_settings.alchemical_settings.explicit_charge_correction = correction

    protocol = openmm_rfe.RelativeHybridTopologyProtocol(
            settings=solv_settings,
    )

    # create DAG from protocol and take first (and only) work unit from within
    dag = protocol.create(
        stateA=stateA_system,
        stateB=stateB_system,
        mapping={'ligand': mapping},
    )
    unit = list(dag.protocol_units)[0]

    with tmpdir.as_cwd():
        sampler = unit.run(dry=True)['debug']['sampler']
        htf = sampler._factory
        _assert_total_charge(htf.hybrid_system,
                             htf._atom_classes, chgA, chgB)

        assert len(htf._atom_classes['core_atoms']) == core_atoms
        assert len(htf._atom_classes['unique_new_atoms']) == new_uniq
        assert len(htf._atom_classes['unique_old_atoms']) == old_uniq
