# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import logging
import copy
import json
import sys
import xml.etree.ElementTree as ET
from importlib import resources
from math import sqrt
from pathlib import Path
from unittest import mock

import gufe
import mdtraj as mdt
import numpy as np
import pytest
from kartograf import KartografAtomMapper
from kartograf.atom_aligner import align_mol_shape
from numpy.testing import assert_allclose
from openff.toolkit import Molecule
from openff.units import unit
from openff.units.openmm import ensure_quantity, from_openmm, to_openmm
from openmm import CustomNonbondedForce, MonteCarloBarostat, NonbondedForce, XmlSerializer, app
from openmm import unit as omm_unit
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from openmmtools.multistate.multistatesampler import MultiStateSampler
from rdkit import Chem
from rdkit.Geometry import Point3D

import openfe
from openfe import setup
from openfe.protocols import openmm_rfe
from openfe.protocols.openmm_rfe._rfe_utils import topologyhelpers
from openfe.protocols.openmm_utils import omm_compute, system_creation
from openfe.protocols.openmm_utils.charge_generation import (
    HAS_ESPALOMA_CHARGE,
    HAS_NAGL,
    HAS_OPENEYE,
)


@pytest.fixture()
def vac_settings():
    settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
    settings.forcefield_settings.nonbonded_method = "nocutoff"
    settings.engine_settings.compute_platform = None
    settings.protocol_repeats = 1
    return settings


@pytest.fixture()
def solv_settings():
    settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
    settings.engine_settings.compute_platform = None
    settings.protocol_repeats = 1
    return settings


def test_invalid_protocol_repeats():
    settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()
    with pytest.raises(ValueError, match="must be a positive value"):
        settings.protocol_repeats = -1


@pytest.mark.parametrize('state', ['A', 'B'])
def test_endstate_two_alchemcomp_stateA(state, benzene_modifications):
    first_state = openfe.ChemicalSystem({
        'ligandA': benzene_modifications['benzene'],
        'ligandB': benzene_modifications['toluene'],
        'solvent': openfe.SolventComponent(),
    })
    other_state = openfe.ChemicalSystem({
        'ligandC': benzene_modifications['phenol'],
        'solvent': openfe.SolventComponent(),
    })

    if state == 'A':
        args = (first_state, other_state)
    else:
        args = (other_state, first_state)

    with pytest.raises(ValueError, match="Only one alchemical component"):
        openmm_rfe.RelativeHybridTopologyProtocol._validate_endstates(
            *args
        )

@pytest.mark.parametrize('state', ['A', 'B'])
def test_endstates_not_smc(state, benzene_modifications):
    first_state = openfe.ChemicalSystem({
        'ligand': benzene_modifications['benzene'],
        'foo': openfe.SolventComponent(),
    })
    other_state = openfe.ChemicalSystem({
        'ligand': benzene_modifications['benzene'],
        'foo': benzene_modifications['toluene'],
    })

    if state == 'A':
        args = (first_state, other_state)
    else:
        args = (other_state, first_state)

    errmsg = "only SmallMoleculeComponents transformations"
    with pytest.raises(ValueError, match=errmsg):
        openmm_rfe.RelativeHybridTopologyProtocol._validate_endstates(
            *args
        )


def test_validate_mapping_none_mapping():
    errmsg = "A single LigandAtomMapping is expected"
    with pytest.raises(ValueError, match=errmsg):
        openmm_rfe.RelativeHybridTopologyProtocol._validate_mapping(None, None)


def test_validate_mapping_multi_mapping(benzene_to_toluene_mapping):
    errmsg = "A single LigandAtomMapping is expected"
    with pytest.raises(ValueError, match=errmsg):
        openmm_rfe.RelativeHybridTopologyProtocol._validate_mapping(
            [benzene_to_toluene_mapping] * 2,
            None
        )


@pytest.mark.parametrize('state', ['A', 'B'])
def test_validate_mapping_alchem_not_in(state, benzene_to_toluene_mapping):
    errmsg = f"not in alchemical components of state{state}"

    if state == "A":
        alchem_comps = {"stateA": [], "stateB": [benzene_to_toluene_mapping.componentB]}
    else:
        alchem_comps = {"stateA": [benzene_to_toluene_mapping.componentA], "stateB": []}

    with pytest.raises(ValueError, match=errmsg):
        openmm_rfe.RelativeHybridTopologyProtocol._validate_mapping(
            [benzene_to_toluene_mapping],
            alchem_comps,
        )


def test_element_change_warning(atom_mapping_basic_test_files):
    # check a mapping with element change gets rejected early
    l1 = atom_mapping_basic_test_files["2-methylnaphthalene"]
    l2 = atom_mapping_basic_test_files["2-naftanol"]

    # We use the 'old' lomap defaults because the
    # basic test files inputs we use aren't fully aligned
    mapper = setup.LomapAtomMapper(
        time=20, threed=True, max3d=1000.0, element_change=True, seed="", shift=True
    )

    mapping = next(mapper.suggest_mappings(l1, l2))

    alchem_comps = {"stateA": [l1], "stateB": [l2]}

    with pytest.warns(UserWarning, match="Element change"):
        openmm_rfe.RelativeHybridTopologyProtocol._validate_mapping(
            [mapping],
            alchem_comps,
        )


def test_charge_difference_no_corr(benzene_to_benzoic_mapping):
    wmsg = (
        "A charge difference of 1 is observed between the end states. "
        "No charge correction has been requested"
    )

    with pytest.warns(UserWarning, match=wmsg):
        openmm_rfe.RelativeHybridTopologyProtocol._validate_charge_difference(
            benzene_to_benzoic_mapping,
            "pme",
            False,
            openfe.SolventComponent(),
        )


def test_charge_difference_no_solvent(benzene_to_benzoic_mapping):
    errmsg = "Cannot use eplicit charge correction without solvent"

    with pytest.raises(ValueError, match=errmsg):
        openmm_rfe.RelativeHybridTopologyProtocol._validate_charge_difference(
            benzene_to_benzoic_mapping,
            "pme",
            True,
            None,
        )


def test_charge_difference_no_pme(benzene_to_benzoic_mapping):
    errmsg = "Explicit charge correction when not using PME"

    with pytest.raises(ValueError, match=errmsg):
        openmm_rfe.RelativeHybridTopologyProtocol._validate_charge_difference(
            benzene_to_benzoic_mapping,
            "nocutoff",
            True,
            openfe.SolventComponent(),
        )


def test_greater_than_one_charge_difference_error(aniline_to_benzoic_mapping):
    errmsg = "A charge difference of 2"
    with pytest.raises(ValueError, match=errmsg):
        openmm_rfe.RelativeHybridTopologyProtocol._validate_charge_difference(
            aniline_to_benzoic_mapping,
            "pme",
            True,
            openfe.SolventComponent(),
        )


@pytest.mark.parametrize(
    "mapping_name,result",
    [
        ["benzene_to_toluene_mapping", 0],
        ["benzene_to_benzoic_mapping", 1],
        ["benzene_to_aniline_mapping", -1],
        ["aniline_to_benzene_mapping", 1],
    ],
)
def test_get_charge_difference(mapping_name, result, request, caplog):
    mapping = request.getfixturevalue(mapping_name)
    caplog.set_level(logging.INFO)
    
    ion = r"Na+" if result == -1 else r"Cl-"
    msg = (
        f"A charge difference of {result} is observed "
        "between the end states. This will be addressed by "
        f"transforming a water into a {ion} ion"
    )
    
    openmm_rfe.RelativeHybridTopologyProtocol._validate_charge_difference(
        mapping,
        "pme",
        True,
        openfe.SolventComponent()
    )

    if result != 0:
        assert msg in caplog.text
    else:
        assert msg not in caplog.text


def test_hightimestep(
    benzene_vacuum_system,
    toluene_vacuum_system,
    benzene_to_toluene_mapping,
    vac_settings,
    tmpdir,
):
    vac_settings.forcefield_settings.hydrogen_mass = 1.0

    p = openmm_rfe.RelativeHybridTopologyProtocol(
        settings=vac_settings,
    )

    dag = p.create(
        stateA=benzene_vacuum_system,
        stateB=toluene_vacuum_system,
        mapping=benzene_to_toluene_mapping,
    )
    dag_unit = list(dag.protocol_units)[0]

    errmsg = "too large for hydrogen mass"
    with tmpdir.as_cwd():
        with pytest.raises(ValueError, match=errmsg):
            dag_unit.run(dry=True)


def test_n_replicas_not_n_windows(
    benzene_vacuum_system,
    toluene_vacuum_system,
    benzene_to_toluene_mapping,
    vac_settings,
    tmpdir,
):
    # For PR #125 we pin such that the number of lambda windows
    # equals the numbers of replicas used - TODO: remove limitation
    # default lambda windows is 11
    vac_settings.simulation_settings.n_replicas = 13

    errmsg = "Number of replicas 13 does not equal the number of lambda windows 11"

    with tmpdir.as_cwd():
        with pytest.raises(ValueError, match=errmsg):
            p = openmm_rfe.RelativeHybridTopologyProtocol(
                settings=vac_settings,
            )
            dag = p.create(
                stateA=benzene_vacuum_system,
                stateB=toluene_vacuum_system,
                mapping=benzene_to_toluene_mapping,
            )
            dag_unit = list(dag.protocol_units)[0]
            dag_unit.run(dry=True)


def test_vaccuum_PME_error(
    benzene_vacuum_system, benzene_modifications, benzene_to_toluene_mapping
):
    # state B doesn't have a solvent component (i.e. its vacuum)
    stateB = openfe.ChemicalSystem({"ligand": benzene_modifications["toluene"]})

    p = openmm_rfe.RelativeHybridTopologyProtocol(
        settings=openmm_rfe.RelativeHybridTopologyProtocol.default_settings(),
    )
    errmsg = "PME cannot be used for vacuum transform"
    with pytest.raises(ValueError, match=errmsg):
        _ = p.create(
            stateA=benzene_vacuum_system,
            stateB=stateB,
            mapping=benzene_to_toluene_mapping,
        )


def test_get_alchemical_waters_no_waters(
    benzene_solvent_openmm_system,
):
    system, topology, positions = benzene_solvent_openmm_system

    errmsg = "There are no waters"

    with pytest.raises(ValueError, match=errmsg):
        topologyhelpers.get_alchemical_waters(
            topology, positions, charge_difference=1, distance_cutoff=3.0 * unit.nanometer
        )
