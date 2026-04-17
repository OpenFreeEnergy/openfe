# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import itertools
import json
import math
import pathlib
from unittest import mock

import gufe
import mdtraj as md
import numpy as np
import openmm
import openmm.app
import openmm.unit
import pytest
from numpy.testing import assert_allclose
from openff.units import unit as offunit
from openff.units.openmm import ensure_quantity, from_openmm, to_openmm
from openmm import (
    CustomBondForce,
    CustomCompoundBondForce,
    CustomNonbondedForce,
    HarmonicAngleForce,
    HarmonicBondForce,
    MonteCarloBarostat,
    MonteCarloMembraneBarostat,
    NonbondedForce,
    PeriodicTorsionForce,
)
from openmmtools.alchemy import AbsoluteAlchemicalFactory, AlchemicalRegion
from openmmtools.multistate.multistatesampler import MultiStateSampler

import openfe.protocols.openmm_septop
from openfe import ChemicalSystem, SolventComponent
from openfe.protocols.openmm_septop import (
    SepTopComplexRunUnit,
    SepTopComplexSetupUnit,
    SepTopProtocol,
    SepTopProtocolResult,
    SepTopSolventRunUnit,
    SepTopSolventSetupUnit,
)
from openfe.protocols.openmm_septop.equil_septop_method import (
    _check_alchemical_charge_difference,
)
from openfe.protocols.openmm_septop.equil_septop_settings import SepTopSettings
from openfe.protocols.openmm_utils import system_validation
from openfe.protocols.openmm_utils.serialization import deserialize
from openfe.protocols.restraint_utils.geometry.boresch import BoreschRestraintGeometry
from openfe.tests.protocols.conftest import compute_energy
from openfe.tests.protocols.openmm_ahfe.test_ahfe_protocol import (
    _assert_num_forces,
    _verify_alchemical_sterics_force_parameters,
)


@pytest.fixture()
def default_settings():
    s = SepTopProtocol.default_settings()
    return s


@pytest.mark.parametrize(
    "val",
    [
        {
            "elec_A": [1.0, 1.0],
            "vdw_A": [0.0, 1.0],
            "restraints_A": [0.0, 0.0],
            "elec_B": [1.0, 1.0],
            "vdw_B": [1.0, 1.0],
            "restraints_B": [0.0, 0.0],
        },
    ],
)
def test_validate_lambda_schedule_nreplicas(val, default_settings):
    default_settings.complex_lambda_settings.lambda_elec_A = val["elec_A"]
    default_settings.complex_lambda_settings.lambda_vdw_A = val["vdw_A"]
    default_settings.complex_lambda_settings.lambda_restraints_A = val["restraints_A"]
    default_settings.complex_lambda_settings.lambda_elec_B = val["elec_B"]
    default_settings.complex_lambda_settings.lambda_vdw_B = val["vdw_B"]
    default_settings.complex_lambda_settings.lambda_restraints_B = val["restraints_B"]
    n_replicas = 3
    default_settings.complex_simulation_settings.n_replicas = n_replicas
    errmsg = (
        f"Number of replicas {n_replicas} does not equal the"
        f" number of lambda windows {len(val['vdw_A'])}"
    )
    with pytest.raises(ValueError, match=errmsg):
        SepTopProtocol._validate_lambda_schedule(
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
    default_settings.complex_lambda_settings.lambda_elec_A = val["elec"]
    default_settings.complex_lambda_settings.lambda_vdw_A = val["vdw"]
    default_settings.complex_lambda_settings.lambda_restraints_A = val["restraints"]
    n_replicas = 3
    default_settings.complex_simulation_settings.n_replicas = n_replicas
    errmsg = (
        "Components elec, vdw, and restraints must have equal amount of lambda "
        "windows. Got 3 and 19 elec lambda windows"
    )
    with pytest.raises(ValueError, match=errmsg):
        SepTopProtocol._validate_lambda_schedule(
            default_settings.complex_lambda_settings,
            default_settings.complex_simulation_settings,
        )


@pytest.mark.parametrize(
    "val",
    [
        {
            "elec_A": [0.0, 1.0],
            "vdw_A": [1.0, 1.0],
            "restraints_A": [0.0, 0.0],
            "elec_B": [1.0, 1.0],
            "vdw_B": [1.0, 1.0],
            "restraints_B": [0.0, 0.0],
        },
    ],
)
def test_validate_lambda_schedule_nakedcharge(val, default_settings):
    default_settings.complex_lambda_settings.lambda_elec_A = val["elec_A"]
    default_settings.complex_lambda_settings.lambda_vdw_A = val["vdw_A"]
    default_settings.complex_lambda_settings.lambda_restraints_A = val["restraints_A"]
    default_settings.complex_lambda_settings.lambda_elec_B = val["elec_B"]
    default_settings.complex_lambda_settings.lambda_vdw_B = val["vdw_B"]
    default_settings.complex_lambda_settings.lambda_restraints_B = val["restraints_B"]
    n_replicas = 2
    default_settings.complex_simulation_settings.n_replicas = n_replicas
    default_settings.solvent_simulation_settings.n_replicas = n_replicas
    errmsg = (
        "There are states along this lambda schedule "
        "where there are atoms with charges but no LJ "
        "interactions: State A: l"
    )
    with pytest.raises(ValueError, match=errmsg):
        SepTopProtocol._validate_lambda_schedule(
            default_settings.complex_lambda_settings,
            default_settings.complex_simulation_settings,
        )
    with pytest.raises(ValueError, match=errmsg):
        SepTopProtocol._validate_lambda_schedule(
            default_settings.complex_lambda_settings,
            default_settings.solvent_simulation_settings,
        )


def test_check_alchem_charge_diff(charged_benzene_modifications):
    errmsg = "A charge difference of 1"
    with pytest.raises(ValueError, match=errmsg):
        _check_alchemical_charge_difference(
            charged_benzene_modifications["benzene"],
            charged_benzene_modifications["benzoic_acid"],
        )


def test_charge_error_create(charged_benzene_modifications, T4_protein_component, default_settings):
    protocol = SepTopProtocol(
        settings=default_settings,
    )
    stateA = ChemicalSystem(
        {
            "benzene": charged_benzene_modifications["benzene"],
            "protein": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )

    stateB = ChemicalSystem(
        {
            "benzoic": charged_benzene_modifications["benzoic_acid"],
            "protein": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )
    errmsg = "A charge difference of 1"
    with pytest.raises(ValueError, match=errmsg):
        protocol.create(
            stateA=stateA,
            stateB=stateB,
            mapping=None,
        )


@pytest.mark.parametrize(
    "fail_endstate, system_A, system_B",
    [
        ("stateA", "benzene_system", "benzene_complex_system"),
        ("stateB", "benzene_complex_system", "benzene_system"),
    ],
)
def test_validate_endstates_protcomp(request, system_A, system_B, fail_endstate):
    with pytest.raises(ValueError, match="No ProteinComponent found"):
        SepTopProtocol._validate_endstates(
            request.getfixturevalue(system_A),
            request.getfixturevalue(system_B),
        )


@pytest.fixture
def T4L_benzene_vacuum(benzene_modifications, T4_protein_component):
    return openfe.ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "protein": T4_protein_component,
        }
    )


@pytest.mark.parametrize(
    "fail_endstate, system_A, system_B",
    [
        ("stateA", "T4L_benzene_vacuum", "benzene_complex_system"),
        ("stateB", "benzene_complex_system", "T4L_benzene_vacuum"),
    ],
)
def test_validate_endstates_nosolvcomp(
    request,
    system_A,
    system_B,
    fail_endstate,
):
    with pytest.raises(ValueError, match="No SolventComponent found"):
        SepTopProtocol._validate_endstates(
            request.getfixturevalue(system_A),
            request.getfixturevalue(system_B),
        )


@pytest.fixture
def T4L_system(T4_protein_component):
    return openfe.ChemicalSystem(
        {
            "solvent": openfe.SolventComponent(),
            "protein": T4_protein_component,
        }
    )


@pytest.mark.parametrize(
    "fail_endstate, system_A, system_B",
    [
        ("stateA", "T4L_system", "benzene_complex_system"),
        ("stateB", "benzene_complex_system", "T4L_system"),
    ],
)
def test_validate_alchem_comps_missing(
    request,
    system_A,
    system_B,
    fail_endstate,
):
    errmsg = (
        "Only one alchemical species is supported. "
        f"Number of unique components found in {fail_endstate}"
    )

    with pytest.raises(
        ValueError,
        match=errmsg,
    ):
        SepTopProtocol._validate_endstates(
            request.getfixturevalue(system_A),
            request.getfixturevalue(system_B),
        )


def test_validate_alchem_comps_toomanyA(
    benzene_modifications,
    T4_protein_component,
):
    stateA = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "toluene": benzene_modifications["toluene"],
            "protein": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )

    stateB = ChemicalSystem(
        {
            "phenol": benzene_modifications["phenol"],
            "protein": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )

    alchem_comps = system_validation.get_alchemical_components(stateA, stateB)

    assert len(alchem_comps["stateA"]) == 2

    assert len(alchem_comps["stateB"]) == 1

    errmsg = "Only one alchemical species is supported. Number of unique components found in stateA: 2."

    with pytest.raises(ValueError, match=errmsg):
        SepTopProtocol._validate_endstates(stateA, stateB)


def test_validate_alchem_nonsmc(
    benzene_modifications,
    T4_protein_component,
):
    stateA = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "protein": T4_protein_component,
            "solvent": SolventComponent(neutralize=False)
        }
    )

    stateB = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "protein": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )

    errmsg = "Only transforming SmallMoleculeComponents are supported by this Protocol."
    with pytest.raises(ValueError, match=errmsg):
        SepTopProtocol._validate_endstates(stateA, stateB)
