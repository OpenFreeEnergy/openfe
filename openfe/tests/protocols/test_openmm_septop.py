# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import itertools
import json
import sys

import openmmtools.alchemy
import pytest
import importlib
from unittest import mock
from openmm import NonbondedForce, CustomNonbondedForce
from openmmtools.multistate.multistatesampler import MultiStateSampler
from openff.units import unit as offunit
from openff.units.openmm import ensure_quantity, from_openmm
import mdtraj as mdt
import numpy as np
from numpy.testing import assert_allclose
from openff.units import unit
import gufe
import openfe
import simtk
from openfe import ChemicalSystem, SolventComponent
from openfe.protocols.openmm_septop import (
    SepTopSolventSetupUnit,
    SepTopComplexSetupUnit,
    SepTopProtocol,
    femto_restraints,
)
from openfe.protocols.openmm_septop.femto_utils import compute_energy, is_close
from openfe.protocols.openmm_septop.utils import deserialize
from openfe.protocols.openmm_septop.equil_septop_method import _check_alchemical_charge_difference
from openmmtools.states import (SamplerState,
                                ThermodynamicState,
                                create_thermodynamic_state_protocol, )

from openfe.protocols.openmm_utils import system_validation
from openfe.protocols.openmm_utils.charge_generation import (
    HAS_NAGL, HAS_OPENEYE, HAS_ESPALOMA
)
from openfe.protocols.openmm_septop.alchemy_copy import AlchemicalState


@pytest.fixture()
def default_settings():
    return SepTopProtocol.default_settings()


def test_create_default_settings():
    settings = SepTopProtocol.default_settings()
    assert settings


@pytest.mark.parametrize('val', [
    {'elec': [0.0, -1], 'vdw': [0.0, 1.0], 'restraints': [0.0, 1.0]},
    {'elec': [0.0, 1.5], 'vdw': [0.0, 1.5], 'restraints': [-0.1, 1.0]}
])
def test_incorrect_window_settings(val, default_settings):
    errmsg = "Lambda windows must be between 0 and 1."
    lambda_settings = default_settings.lambda_settings
    with pytest.raises(ValueError, match=errmsg):
        lambda_settings.lambda_elec_ligandA = val['elec']
        lambda_settings.lambda_vdw_ligandA = val['vdw']
        lambda_settings.lambda_restraints_ligandA = val['restraints']


@pytest.mark.parametrize('val', [
    {'elec': [0.0, 0.1, 0.0], 'vdw': [0.0, 1.0, 1.0], 'restraints': [0.0, 1.0, 1.0]},
])
def test_monotonic_lambda_windows(val, default_settings):
    errmsg = "The lambda schedule is not monotonic."
    lambda_settings = default_settings.lambda_settings

    with pytest.raises(ValueError, match=errmsg):
        lambda_settings.lambda_elec_ligandA = val['elec']
        lambda_settings.lambda_vdw_ligandA = val['vdw']
        lambda_settings.lambda_restraints_ligandA = val['restraints']


@pytest.mark.parametrize('val', [
    {'elec': [1.0, 1.0], 'vdw': [0.0, 1.0], 'restraints': [0.0, 0.0]},
])
def test_validate_lambda_schedule_nreplicas(val, default_settings):
    default_settings.lambda_settings.lambda_elec_ligandA = val['elec']
    default_settings.lambda_settings.lambda_vdw_ligandA = val['vdw']
    default_settings.lambda_settings.lambda_restraints_ligandA = val['restraints']
    default_settings.lambda_settings.lambda_elec_ligandB = val['elec']
    default_settings.lambda_settings.lambda_vdw_ligandB = val['vdw']
    default_settings.lambda_settings.lambda_restraints_ligandB = val[
        'restraints']
    n_replicas = 3
    default_settings.complex_simulation_settings.n_replicas = n_replicas
    errmsg = (f"Number of replicas {n_replicas} does not equal the"
              f" number of lambda windows {len(val['vdw'])}")
    with pytest.raises(ValueError, match=errmsg):
        SepTopProtocol._validate_lambda_schedule(
            default_settings.lambda_settings,
            default_settings.complex_simulation_settings,
        )


@pytest.mark.parametrize('val', [
    {'elec': [1.0, 1.0, 1.0], 'vdw': [0.0, 1.0], 'restraints': [0.0, 0.0]},
])
def test_validate_lambda_schedule_nwindows(val, default_settings):
    default_settings.lambda_settings.lambda_elec_ligandA = val['elec']
    default_settings.lambda_settings.lambda_vdw_ligandA = val['vdw']
    default_settings.lambda_settings.lambda_restraints_ligandA = val['restraints']
    n_replicas = 3
    default_settings.complex_simulation_settings.n_replicas = n_replicas
    errmsg = (
        "Components elec, vdw, and restraints must have equal amount of lambda "
        "windows. Got 3 and 19 elec lambda windows")
    with pytest.raises(ValueError, match=errmsg):
        SepTopProtocol._validate_lambda_schedule(
            default_settings.lambda_settings,
            default_settings.complex_simulation_settings,
        )


@pytest.mark.parametrize('val', [
    {'elec': [1.0, 0.5], 'vdw': [1.0, 1.0], 'restraints': [0.0, 0.0]},
])
def test_validate_lambda_schedule_nakedcharge(val, default_settings):
    default_settings.lambda_settings.lambda_elec_ligandA = val['elec']
    default_settings.lambda_settings.lambda_vdw_ligandA = val['vdw']
    default_settings.lambda_settings.lambda_restraints_ligandA = val[
        'restraints']
    default_settings.lambda_settings.lambda_elec_ligandB = val['elec']
    default_settings.lambda_settings.lambda_vdw_ligandB = val['vdw']
    default_settings.lambda_settings.lambda_restraints_ligandB = val[
        'restraints']
    n_replicas = 2
    default_settings.complex_simulation_settings.n_replicas = n_replicas
    default_settings.solvent_simulation_settings.n_replicas = n_replicas
    errmsg = (
        "There are states along this lambda schedule "
        "where there are atoms with charges but no LJ "
        "interactions: Ligand A: l")
    with pytest.raises(ValueError, match=errmsg):
        SepTopProtocol._validate_lambda_schedule(
            default_settings.lambda_settings,
            default_settings.complex_simulation_settings,
        )
    with pytest.raises(ValueError, match=errmsg):
        SepTopProtocol._validate_lambda_schedule(
            default_settings.lambda_settings,
            default_settings.solvent_simulation_settings,
        )


def test_create_default_protocol(default_settings):
    # this is roughly how it should be created
    protocol = SepTopProtocol(
        settings=default_settings,
    )
    assert protocol


def test_serialize_protocol(default_settings):
    protocol = SepTopProtocol(
        settings=default_settings,
    )

    ser = protocol.to_dict()
    ret = SepTopProtocol.from_dict(ser)
    assert protocol == ret


def test_create_independent_repeat_ids(
        benzene_complex_system, toluene_complex_system,
):
    # if we create two dags each with 3 repeats, they should give 6 repeat_ids
    # this allows multiple DAGs in flight for one Transformation that don't clash on gather
    settings = SepTopProtocol.default_settings()
    # Default protocol is 1 repeat, change to 3 repeats
    settings.protocol_repeats = 3
    protocol = SepTopProtocol(
            settings=settings,
    )

    dag1 = protocol.create(
        stateA=benzene_complex_system,
        stateB=toluene_complex_system,
        mapping=None,
    )
    dag2 = protocol.create(
        stateA=benzene_complex_system,
        stateB=toluene_complex_system,
        mapping=None,
    )
    # print([u for u in dag1.protocol_units])
    repeat_ids = set()
    for u in dag1.protocol_units:
        repeat_ids.add(u.inputs['repeat_id'])
    for u in dag2.protocol_units:
        repeat_ids.add(u.inputs['repeat_id'])

    # There are 4 units per repeat per DAG: 4 * 3 * 2 = 24
    assert len(repeat_ids) == 24


def test_check_alchem_charge_diff(charged_benzene_modifications):
    errmsg = "A charge difference of 1"
    with pytest.raises(ValueError, match=errmsg):
        _check_alchemical_charge_difference(
            charged_benzene_modifications["benzene"],
            charged_benzene_modifications["benzoic_acid"],
        )


def test_charge_error_create(
        charged_benzene_modifications, T4_protein_component,
):
    # if we create two dags each with 3 repeats, they should give 6 repeat_ids
    # this allows multiple DAGs in flight for one Transformation that don't clash on gather
    settings = SepTopProtocol.default_settings()
    # Default protocol is 1 repeat, change to 3 repeats
    settings.protocol_repeats = 3
    protocol = SepTopProtocol(
            settings=settings,
    )
    stateA = ChemicalSystem({
        'benzene': charged_benzene_modifications['benzene'],
        'protein': T4_protein_component,
        'solvent': SolventComponent()
    })

    stateB = ChemicalSystem({
        'benzoic': charged_benzene_modifications['benzoic_acid'],
        'protein': T4_protein_component,
        'solvent': SolventComponent(),
    })
    errmsg = "A charge difference of 1"
    with pytest.raises(ValueError, match=errmsg):
        protocol.create(
            stateA=stateA,
            stateB=stateB,
            mapping=None,
        )


def test_validate_complex_endstates_protcomp_stateA(
    benzene_modifications, T4_protein_component,
):
    stateA = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'solvent': SolventComponent()
    })

    stateB = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'protein': T4_protein_component,
        'solvent': SolventComponent(),
    })

    with pytest.raises(ValueError, match="No ProteinComponent found in stateA"):
        SepTopProtocol._validate_complex_endstates(stateA, stateB)


def test_validate_complex_endstates_protcomp_stateB(
    benzene_modifications, T4_protein_component,
):
    stateA = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'protein': T4_protein_component,
        'solvent': SolventComponent(),
    })

    stateB = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'solvent': SolventComponent(),
    })

    with pytest.raises(ValueError, match="No ProteinComponent found in stateB"):
        SepTopProtocol._validate_complex_endstates(stateA, stateB)



def test_validate_complex_endstates_nosolvcomp_stateA(
    benzene_modifications, T4_protein_component,
):
    stateA = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'protein': T4_protein_component,
    })

    stateB = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'protein': T4_protein_component,
        'solvent': SolventComponent(),
    })

    with pytest.raises(
        ValueError, match="No SolventComponent found in stateA"
    ):
        SepTopProtocol._validate_complex_endstates(stateA, stateB)


def test_validate_complex_endstates_nosolvcomp_stateB(
    benzene_modifications, T4_protein_component,
):
    stateA = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'protein': T4_protein_component,
        'solvent': SolventComponent(),
    })

    stateB = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'protein': T4_protein_component,
    })

    with pytest.raises(
        ValueError, match="No SolventComponent found in stateB"
    ):
        SepTopProtocol._validate_complex_endstates(stateA, stateB)


def test_validate_alchem_comps_missingA(
    benzene_modifications, T4_protein_component,
):
    stateA = ChemicalSystem({
        'protein': T4_protein_component,
        'solvent': SolventComponent(),
    })

    stateB = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'protein': T4_protein_component,
        'solvent': SolventComponent(),
    })

    alchem_comps = system_validation.get_alchemical_components(stateA, stateB)

    with pytest.raises(ValueError, match='one alchemical components must be present in stateA.'):
        SepTopProtocol._validate_alchemical_components(alchem_comps)


def test_validate_alchem_comps_missingB(
    benzene_modifications, T4_protein_component,
):
    stateA = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'protein': T4_protein_component,
        'solvent': SolventComponent(),
    })

    stateB = ChemicalSystem({
        'protein': T4_protein_component,
        'solvent': SolventComponent(),
    })

    alchem_comps = system_validation.get_alchemical_components(stateA, stateB)

    with pytest.raises(ValueError, match='one alchemical components must be present in stateB.'):
        SepTopProtocol._validate_alchemical_components(alchem_comps)


def test_validate_alchem_comps_toomanyA(
    benzene_modifications, T4_protein_component,
):
    stateA = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'toluene': benzene_modifications['toluene'],
        'protein': T4_protein_component,
        'solvent': SolventComponent(),
    })

    stateB = ChemicalSystem({
        'phenol': benzene_modifications['phenol'],
        'protein': T4_protein_component,
        'solvent': SolventComponent(),
    })

    alchem_comps = system_validation.get_alchemical_components(stateA, stateB)

    assert len(alchem_comps['stateA']) == 2

    assert len(alchem_comps['stateB']) == 1

    with pytest.raises(ValueError, match='Found 2 alchemical components in stateA'):
        SepTopProtocol._validate_alchemical_components(alchem_comps)


def test_validate_alchem_nonsmc(
    benzene_modifications, T4_protein_component,
):
    stateA = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'solvent': SolventComponent()
    })

    stateB = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'protein': T4_protein_component
    })

    alchem_comps = system_validation.get_alchemical_components(stateA, stateB)

    with pytest.raises(ValueError, match='Non SmallMoleculeComponent'):
        SepTopProtocol._validate_alchemical_components(alchem_comps)
