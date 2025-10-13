# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import itertools
import json
from math import sqrt
import sys
import pytest
from unittest import mock
from openmm import NonbondedForce, CustomNonbondedForce
from openmmtools.multistate.multistatesampler import MultiStateSampler
from openff.units import unit as offunit
from openff.units.openmm import ensure_quantity, from_openmm
import mdtraj as mdt
import numpy as np
from numpy.testing import assert_allclose
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
from openfe.protocols.openmm_utils.charge_generation import (
    HAS_NAGL, HAS_OPENEYE, HAS_ESPALOMA_CHARGE
)


@pytest.fixture()
def default_settings():
    return AbsoluteSolvationProtocol.default_settings()


@pytest.mark.parametrize('val', [
    {'elec': [0.0, 1.0], 'vdw': [1.0, 1.0], 'restraints': [0.0, 0.0]},
])
def test_validate_lambda_schedule_naked_charge(val, default_settings):
    errmsg = ("There are states along this lambda schedule "
              "where there are atoms with charges but no LJ "
              f"interactions: lambda 0: "
              f"elec {val['elec'][0]} vdW {val['vdw'][0]}")
    default_settings.lambda_settings.lambda_elec = val['elec']
    default_settings.lambda_settings.lambda_vdw = val['vdw']
    default_settings.lambda_settings.lambda_restraints = val['restraints']
    default_settings.vacuum_simulation_settings.n_replicas = 2
    default_settings.solvent_simulation_settings.n_replicas = 2
    with pytest.raises(ValueError, match=errmsg):
        AbsoluteSolvationProtocol._validate_lambda_schedule(
            default_settings.lambda_settings,
            default_settings.vacuum_simulation_settings,
        )
    with pytest.raises(ValueError, match=errmsg):
        AbsoluteSolvationProtocol._validate_lambda_schedule(
            default_settings.lambda_settings,
            default_settings.solvent_simulation_settings,
        )


@pytest.mark.parametrize('val', [
    {'elec': [1.0, 1.0], 'vdw': [0.0, 1.0], 'restraints': [0.0, 0.0]},
])
def test_validate_lambda_schedule_nreplicas(val, default_settings):
    default_settings.lambda_settings.lambda_elec = val['elec']
    default_settings.lambda_settings.lambda_vdw = val['vdw']
    default_settings.lambda_settings.lambda_restraints = val['restraints']
    n_replicas = 3
    default_settings.vacuum_simulation_settings.n_replicas = n_replicas
    errmsg = (f"Number of replicas {n_replicas} does not equal the"
              f" number of lambda windows {len(val['vdw'])}")
    with pytest.raises(ValueError, match=errmsg):
        AbsoluteSolvationProtocol._validate_lambda_schedule(
            default_settings.lambda_settings,
            default_settings.vacuum_simulation_settings,
        )


@pytest.mark.parametrize('val', [
    {'elec': [1.0, 1.0, 1.0], 'vdw': [0.0, 1.0], 'restraints': [0.0, 0.0]},
])
def test_validate_lambda_schedule_nwindows(val, default_settings):
    default_settings.lambda_settings.lambda_elec = val['elec']
    default_settings.lambda_settings.lambda_vdw = val['vdw']
    default_settings.lambda_settings.lambda_restraints = val['restraints']
    n_replicas = 3
    default_settings.vacuum_simulation_settings.n_replicas = n_replicas
    errmsg = (
        "Components elec, vdw, and restraints must have equal amount"
        f" of lambda windows. Got {len(val['elec'])} elec lambda"
        f" windows, {len(val['vdw'])} vdw lambda windows, and"
        f"{len(val['restraints'])} restraints lambda windows.")
    with pytest.raises(ValueError, match=errmsg):
        AbsoluteSolvationProtocol._validate_lambda_schedule(
            default_settings.lambda_settings,
            default_settings.vacuum_simulation_settings,
        )


@pytest.mark.parametrize('val', [
    {'elec': [1.0, 1.0], 'vdw': [1.0, 1.0], 'restraints': [0.0, 1.0]},
])
def test_validate_lambda_schedule_nonzero_restraints(val, default_settings):
    wmsg = ("Non-zero restraint lambdas applied. The absolute "
            "solvation protocol doesn't apply restraints, "
            "therefore restraints won't be applied.")
    default_settings.lambda_settings.lambda_elec = val['elec']
    default_settings.lambda_settings.lambda_vdw = val['vdw']
    default_settings.lambda_settings.lambda_restraints = val['restraints']
    default_settings.vacuum_simulation_settings.n_replicas = 2
    with pytest.warns(UserWarning, match=wmsg):
        AbsoluteSolvationProtocol._validate_lambda_schedule(
            default_settings.lambda_settings,
            default_settings.vacuum_simulation_settings,
        )


def test_validate_solvent_endstates_protcomp(
    benzene_modifications, T4_protein_component
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
        AbsoluteSolvationProtocol._validate_solvent_endstates(stateA, stateB)


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
        AbsoluteSolvationProtocol._validate_solvent_endstates(stateA, stateB)


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
        AbsoluteSolvationProtocol._validate_solvent_endstates(stateA, stateB)


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
    settings.vacuum_forcefield_settings.nonbonded_method = 'pme'
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
