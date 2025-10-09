# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import itertools
import json
import sys
from math import sqrt
from unittest import mock

import gufe
import mdtraj as mdt
import numpy as np
import openfe
import pytest
from numpy.testing import assert_allclose
from openfe import ChemicalSystem, SolventComponent
from openfe.protocols import openmm_afe
from openfe.protocols.openmm_afe import (
    AbsoluteBindingComplexUnit,
    AbsoluteBindingProtocol,
    AbsoluteBindingSolventUnit,
)
from openfe.protocols.openmm_utils import system_validation
from openff.units import unit as offunit
from openff.units.openmm import ensure_quantity, from_openmm
from openmm import CustomNonbondedForce, NonbondedForce
from openmmtools.multistate.multistatesampler import MultiStateSampler


@pytest.fixture()
def default_settings():
    return AbsoluteBindingProtocol.default_settings()


def test_create_default_settings():
    settings = AbsoluteBindingProtocol.default_settings()
    assert settings


def test_negative_repeats_settings(default_settings):
    with pytest.raises(ValueError, match="protocol_repeats must be a positive"):
        default_settings.protocol_repeats = -1


@pytest.mark.parametrize(
    "val",
    [
        {"elec": [0.0, -1], "vdw": [0.0, 1.0], "restraints": [0.0, 1.0]},
        {"elec": [0.0, 1.5], "vdw": [0.0, 1.5], "restraints": [-0.1, 1.0]},
    ],
)
def test_incorrect_window_settings(val, default_settings):
    errmsg = "Lambda windows must be between 0 and 1."
    lambda_settings = default_settings.complex_lambda_settings
    with pytest.raises(ValueError, match=errmsg):
        lambda_settings.lambda_elec = val["elec"]
        lambda_settings.lambda_vdw = val["vdw"]
        lambda_settings.lambda_restraints = val["restraints"]


@pytest.mark.parametrize(
    "val",
    [
        {
            "elec": [0.0, 0.1, 0.0],
            "vdw": [0.0, 1.0, 1.0],
            "restraints": [0.0, 1.0, 1.0],
        },
    ],
)
def test_monotonic_lambda_windows(val, default_settings):
    errmsg = "The lambda schedule is not monotonic."
    lambda_settings = default_settings.complex_lambda_settings

    with pytest.raises(ValueError, match=errmsg):
        lambda_settings.lambda_elec = val["elec"]
        lambda_settings.lambda_vdw = val["vdw"]
        lambda_settings.lambda_restraints = val["restraints"]


@pytest.mark.parametrize(
    "val",
    [
        {"elec": [0.0, 1.0], "vdw": [1.0, 1.0], "restraints": [0.0, 0.0]},
    ],
)
def test_validate_lambda_schedule_naked_charge(val, default_settings):
    errmsg = (
        "There are states along this lambda schedule "
        "where there are atoms with charges but no LJ "
        f"interactions: lambda 0: "
        f"elec {val['elec'][0]} vdW {val['vdw'][0]}"
    )
    default_settings.complex_lambda_settings.lambda_elec = val["elec"]
    default_settings.complex_lambda_settings.lambda_vdw = val["vdw"]
    default_settings.complex_lambda_settings.lambda_restraints = val["restraints"]
    default_settings.solvent_lambda_settings.lambda_elec = val["elec"]
    default_settings.solvent_lambda_settings.lambda_vdw = val["vdw"]
    default_settings.solvent_lambda_settings.lambda_restraints = val["restraints"]
    default_settings.complex_simulation_settings.n_replicas = 2
    default_settings.solvent_simulation_settings.n_replicas = 2
    with pytest.raises(ValueError, match=errmsg):
        AbsoluteBindingProtocol._validate_lambda_schedule(
            default_settings.complex_lambda_settings,
            default_settings.complex_simulation_settings,
        )
    with pytest.raises(ValueError, match=errmsg):
        AbsoluteBindingProtocol._validate_lambda_schedule(
            default_settings.solvent_lambda_settings,
            default_settings.solvent_simulation_settings,
        )


@pytest.mark.parametrize(
    "val",
    [
        {"elec": [1.0, 1.0], "vdw": [0.0, 1.0], "restraints": [0.0, 0.0]},
    ],
)
def test_validate_lambda_schedule_nreplicas(val, default_settings):
    default_settings.complex_lambda_settings.lambda_elec = val["elec"]
    default_settings.complex_lambda_settings.lambda_vdw = val["vdw"]
    default_settings.complex_lambda_settings.lambda_restraints = val["restraints"]
    n_replicas = 3
    default_settings.complex_simulation_settings.n_replicas = n_replicas
    errmsg = (
        f"Number of replicas {n_replicas} does not equal the"
        f" number of lambda windows {len(val['vdw'])}"
    )
    with pytest.raises(ValueError, match=errmsg):
        AbsoluteBindingProtocol._validate_lambda_schedule(
            default_settings.complex_lambda_settings,
            default_settings.complex_simulation_settings,
        )


@pytest.mark.parametrize(
    "val",
    [
        {"elec": [1.0, 1.0, 1.0], "vdw": [0.0, 1.0], "restraints": [0.0, 0.0]},
    ],
)
def test_validate_lambda_schedule_nwindows(val, default_settings):
    default_settings.complex_lambda_settings.lambda_elec = val["elec"]
    default_settings.complex_lambda_settings.lambda_vdw = val["vdw"]
    default_settings.complex_lambda_settings.lambda_restraints = val["restraints"]
    n_replicas = 3
    default_settings.complex_simulation_settings.n_replicas = n_replicas
    errmsg = (
        "Components elec, vdw, and restraints must have equal amount"
        f" of lambda windows. Got {len(val['elec'])} elec lambda"
        f" windows, {len(val['vdw'])} vdw lambda windows, and"
        f"{len(val['restraints'])} restraints lambda windows."
    )
    with pytest.raises(ValueError, match=errmsg):
        AbsoluteBindingProtocol._validate_lambda_schedule(
            default_settings.complex_lambda_settings,
            default_settings.complex_simulation_settings,
        )


def test_validate_no_protcomp(
    benzene_modifications,
):
    stateA = ChemicalSystem(
        {"benzene": benzene_modifications["benzene"], "solvent": SolventComponent()}
    )

    stateB = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "solvent": SolventComponent(),
        }
    )

    errmsg = "No ProteinComponent found"
    with pytest.raises(ValueError, match=errmsg):
        AbsoluteBindingProtocol._validate_endstates(stateA, stateB)


def test_validate_endstates_nosolvcomp_stateA(
    benzene_modifications, T4_protein_component
):
    stateA = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "protein": T4_protein_component,
        }
    )

    stateB = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "protein": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )

    with pytest.raises(ValueError, match="No SolventComponent found"):
        AbsoluteBindingProtocol._validate_endstates(stateA, stateB)


def test_validate_endstates_multiple_uniqueA(
    benzene_modifications, T4_protein_component
):
    stateA = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "toluene": benzene_modifications["toluene"],
            "protein": T4_protein_component,
            "solvlent": SolventComponent(),
        }
    )

    stateB = ChemicalSystem(
        {
            "protein": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )

    with pytest.raises(ValueError, match="More than one unique"):
        AbsoluteBindingProtocol._validate_endstates(stateA, stateB)


def test_validate_solvent_endstates_protcomp_missing(
    benzene_modifications, T4_protein_component
):
    stateA = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "protein": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )

    stateB = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "phenol": benzene_modifications["phenol"],
            "solvent": SolventComponent(),
        }
    )

    errmsg = "Only dissapearing small molecule components"
    with pytest.raises(ValueError, match=errmsg):
        AbsoluteBindingProtocol._validate_endstates(stateA, stateB)


def test_validate_endstates_nosolvcomp_stateB(
    benzene_modifications, T4_protein_component
):
    stateA = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "protein": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )

    stateB = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "protein": T4_protein_component,
        }
    )

    with pytest.raises(ValueError, match="Only dissapearing small molecule components"):
        AbsoluteBindingProtocol._validate_endstates(stateA, stateB)


def test_validate_endstates_unique_stateB(benzene_modifications, T4_protein_component):
    stateA = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "protein": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )

    stateB = ChemicalSystem(
        {
            "toluene": benzene_modifications["toluene"],
            "protein": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )

    with pytest.raises(ValueError, match="Unique components are found in stateB"):
        AbsoluteBindingProtocol._validate_endstates(stateA, stateB)


def test_indices_not_all(
    benzene_modifications,
    T4_protein_component,
):
    s = openmm_afe.AbsoluteBindingProtocol.default_settings()
    s.protocol_repeats = 1
    s.complex_equil_output_settings.output_indices = "not water"

    protocol = openmm_afe.AbsoluteBindingProtocol(
        settings=s,
    )

    stateA = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "protein": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )

    stateB = ChemicalSystem(
        {
            "protein": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )

    # Create DAG from protocol, get the vacuum and solvent units
    # and eventually dry run the first solvent unit
    with pytest.raises(ValueError, match="need to output the full system"):
        dag = protocol.create(
            stateA=stateA,
            stateB=stateB,
            mapping=None,
        )
