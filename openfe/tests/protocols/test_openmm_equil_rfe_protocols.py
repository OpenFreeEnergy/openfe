# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import os
from io import StringIO
import numpy as np
import gufe
from gufe.tests.test_tokenization import GufeTokenizableTestsMixin
from gufe.protocols import execute_DAG
import pytest
from unittest import mock
from openff.units import unit
from importlib import resources
import xml.etree.ElementTree as ET

from openmm import app, Platform, XmlSerializer, MonteCarloBarostat
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
        _validate_alchemical_components,
)
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
        # 12 bonds in benzene + 4 extra tolunee bonds
        assert len(list(htf.hybrid_topology.atoms)) == 16
        assert len(list(htf.omm_hybrid_topology.atoms())) == 16
        assert len(list(htf.hybrid_topology.bonds)) == 16
        assert len(list(htf.omm_hybrid_topology.bonds())) == 16

        # smoke test - can convert back the mdtraj topology
        ret_top = mdt.Topology.to_openmm(htf.hybrid_topology)
        assert len(list(ret_top.atoms())) == 16
        assert len(list(ret_top.bonds())) == 16


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


def test_dry_run_ligand_tip4p(benzene_system, toluene_system,
                              benzene_to_toluene_mapping, tmpdir):
    """
    Test that we can create a system with virtual sites in the
    environment (waters)
    """
    settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
    settings.forcefield_settings.forcefields = [
        "amber/ff14SB.xml",    # ff14SB protein force field
        "amber/tip4pew_standard.xml", # FF we are testsing with the fun VS
        "amber/phosaa10.xml",  # Handles THE TPO
    ]
    settings.solvation_settings.solvent_padding = 1.0 * unit.nanometer
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

    with tmpdir.as_cwd():
        sampler = dag_unit.run(dry=True)['debug']['sampler']
        assert isinstance(sampler, MultiStateSampler)
        assert sampler._factory.hybrid_system


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
    settings.solvation_settings.solvent_padding = 1.0 * unit.nanometer
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
    with resources.path('openfe.tests.data.openmm_rfe', 'ligand_23.sdf') as f:
        lig23 = openfe.SmallMoleculeComponent.from_sdf_file(str(f))
    with resources.path('openfe.tests.data.openmm_rfe', 'ligand_55.sdf') as f:
        lig55 = openfe.SmallMoleculeComponent.from_sdf_file(str(f))

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
    with resources.path('openfe.tests.data.openmm_rfe', 'reference.xml') as f:
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


@pytest.fixture
def available_platforms() -> set[str]:
    return {Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())}


@pytest.fixture
def set_openmm_threads_1():
    # for vacuum sims, we want to limit threads to one
    # this fixture sets OPENMM_CPU_THREADS='1' for a single test, then reverts to previously held value
    previous: str | None = os.environ.get('OPENMM_CPU_THREADS')

    try:
        os.environ['OPENMM_CPU_THREADS'] = '1'
        yield
    finally:
        if previous is None:
            del os.environ['OPENMM_CPU_THREADS']
        else:
            os.environ['OPENMM_CPU_THREADS'] = previous


@pytest.mark.slow
@pytest.mark.flaky(reruns=3)  # pytest-rerunfailures; we can get bad minimisation
@pytest.mark.parametrize('platform', ['CPU', 'CUDA'])
def test_openmm_run_engine(benzene_vacuum_system, platform,
                           available_platforms, benzene_modifications,
                           set_openmm_threads_1, tmpdir):
    if platform not in available_platforms:
        pytest.skip(f"OpenMM Platform: {platform} not available")
    # this test actually runs MD
    # if this passes, you're 99% likely to have a good time
    # these settings are a small self to self sim, that has enough eq that it doesn't occasionally crash
    s = openfe.protocols.openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
    s.simulation_settings.equilibration_length = 0.1 * unit.picosecond
    s.simulation_settings.production_length = 0.1 * unit.picosecond
    s.integrator_settings.n_steps = 5 * unit.timestep
    s.system_settings.nonbonded_method = 'nocutoff'
    s.alchemical_sampler_settings.n_repeats = 1
    s.engine_settings.compute_platform = platform
    s.simulation_settings.checkpoint_interval = 5 * unit.timestep

    p = openmm_rfe.RelativeHybridTopologyProtocol(s)

    b = benzene_vacuum_system['ligand']
    
    # make a copy with a different name
    rdmol = benzene_modifications['benzene'].to_rdkit()
    b_alt = openfe.SmallMoleculeComponent.from_rdkit(rdmol, name='alt')
    benzene_vacuum_alt_system = openfe.ChemicalSystem({
        'ligand': b_alt
    })

    m = openfe.LigandAtomMapping(componentA=b, componentB=b_alt,
                                 componentA_to_componentB={i: i for i in range(12)})
    dag = p.create(stateA=benzene_vacuum_system, stateB=benzene_vacuum_alt_system,
                   mapping={'ligand': m})

    cwd = pathlib.Path(str(tmpdir))
    r = execute_DAG(dag, shared_basedir=cwd, scratch_basedir=cwd,
                    keep_shared=True)

    assert r.ok()
    for pur in r.protocol_unit_results:
        unit_shared = tmpdir / f"shared_{pur.source_key}_attempt_0"
        assert unit_shared.exists()
        assert pathlib.Path(unit_shared).is_dir()
        checkpoint = pur.outputs['last_checkpoint']
        assert checkpoint == unit_shared / "checkpoint.nc"
        assert checkpoint.exists()
        nc = pur.outputs['nc']
        assert nc == unit_shared / "simulation.nc"
        assert nc.exists()
