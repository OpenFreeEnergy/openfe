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


def compare_energies(alchemical_system, positions):

    alchemical_state = AlchemicalState.from_system(alchemical_system)

    from openfe.protocols.openmm_septop.alchemy_copy import AbsoluteAlchemicalFactory

    energy = AbsoluteAlchemicalFactory.get_energy_components(
        alchemical_system, alchemical_state, positions
    )
    na_A = 'alchemically modified NonbondedForce for non-alchemical/alchemical sterics for region A'
    na_B = 'alchemically modified NonbondedForce for non-alchemical/alchemical sterics for region B'
    nonbonded = 'unmodified NonbondedForce'

    # Lambda 0: LigandA sterics on, elec on, ligand B sterics off, elec off
    alchemical_state.lambda_sterics_A = 1
    alchemical_state.lambda_sterics_B = 0
    alchemical_state.lambda_electrostatics_A = 1
    alchemical_state.lambda_electrostatics_B = 0
    energy_0 = AbsoluteAlchemicalFactory.get_energy_components(
        alchemical_system, alchemical_state, positions
    )

    # Lambda 7: LigandA sterics on, elec on, ligand B sterics on, elec off
    alchemical_state.lambda_sterics_A = 1
    alchemical_state.lambda_sterics_B = 1
    alchemical_state.lambda_electrostatics_A = 1
    alchemical_state.lambda_electrostatics_B = 0
    energy_7 = AbsoluteAlchemicalFactory.get_energy_components(
        alchemical_system, alchemical_state, positions
    )

    # Lambda 8: LigandA sterics on, elec partially on,
    # ligand B sterics on, elec partially on
    alchemical_state.lambda_sterics_A = 1
    alchemical_state.lambda_sterics_B = 1
    alchemical_state.lambda_electrostatics_A = 0.75
    alchemical_state.lambda_electrostatics_B = 0.25
    energy_8 = AbsoluteAlchemicalFactory.get_energy_components(
        alchemical_system, alchemical_state, positions
    )

    # Lambda 12: LigandA sterics on, elec off, ligand B sterics on, elec on
    alchemical_state.lambda_sterics_A = 1
    alchemical_state.lambda_sterics_B = 1
    alchemical_state.lambda_electrostatics_A = 0
    alchemical_state.lambda_electrostatics_B = 1
    energy_12 = AbsoluteAlchemicalFactory.get_energy_components(
        alchemical_system, alchemical_state, positions
    )

    # Lambda 13: LigandA sterics partially on, elec off, ligand B sterics on, elec on
    alchemical_state.lambda_sterics_A = 0.857142857
    alchemical_state.lambda_sterics_B = 1
    alchemical_state.lambda_electrostatics_A = 0
    alchemical_state.lambda_electrostatics_B = 1
    energy_13 = AbsoluteAlchemicalFactory.get_energy_components(
        alchemical_system, alchemical_state, positions
    )

    return na_A, na_B, nonbonded, energy, energy_0, energy_7, energy_8, energy_12, energy_13


# @pytest.mark.integration  # takes too long to be a slow test ~ 4 mins locally
# @pytest.mark.flaky(reruns=3)  # pytest-rerunfailures; we can get bad minimisation
# @pytest.mark.parametrize('platform', ['CPU', 'CUDA'])
def test_lambda_energies(bace_ligands,  bace_protein_component, tmpdir):
    # check system parametrisation works even if confgen fails
    s = SepTopProtocol.default_settings()
    s.protocol_repeats = 1
    s.solvent_equil_simulation_settings.minimization_steps = 100
    s.solvent_equil_simulation_settings.equilibration_length_nvt = 10 * unit.picosecond
    s.solvent_equil_simulation_settings.equilibration_length = 10 * unit.picosecond
    s.solvent_equil_simulation_settings.production_length = 1 * unit.picosecond
    s.solvent_solvation_settings.box_shape = 'dodecahedron'
    s.solvent_solvation_settings.solvent_padding = 1.8 * unit.nanometer

    protocol = SepTopProtocol(
        settings=s,
    )

    stateA = ChemicalSystem({
        'lig_02': bace_ligands['lig_02'],
        'protein': bace_protein_component,
        'solvent': SolventComponent(),
    })

    stateB = ChemicalSystem({
        'lig_03': bace_ligands['lig_03'],
        'protein': bace_protein_component,
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
    solv_setup_unit = [u for u in prot_units
                       if isinstance(u, SepTopSolventSetupUnit)]

    with tmpdir.as_cwd():
        output = solv_setup_unit[0].run()
        system = output["system"]
        alchemical_system = deserialize(system)
        topology = output["topology"]
        pdb = simtk.openmm.app.pdbfile.PDBFile(str(topology))
        positions = pdb.getPositions(asNumpy=True)

        # Remove Harmonic restraint force solvent
        alchemical_system.removeForce(13)

        na_A, na_B, nonbonded, energy, energy_0, energy_7, energy_8, \
        energy_12, energy_13 = compare_energies(alchemical_system, positions)

        for key, value in energy.items():
            if key == na_A:
                assert is_close(value, energy_0[key])
                assert is_close(value, energy_7[key])
                assert is_close(value, energy_8[key])
                assert is_close(value, energy_12[key])
                assert not is_close(value, energy_13[key])
            elif key == na_B:
                assert not is_close(value, energy_0[key])
                assert energy_0[key].value_in_unit(
                    simtk.unit.kilojoule_per_mole) == 0
                assert is_close(value, energy_7[key])
                assert is_close(value, energy_8[key])
                assert is_close(value, energy_12[key])
                assert is_close(value, energy_13[key])
            elif key == nonbonded:
                assert not is_close(value, energy_0[key])
                assert is_close(energy_0[key], energy_7[key])
                assert not is_close(energy_0[key], energy_8[key])
                assert not is_close(energy_0[key], energy_12[key])
                assert not is_close(energy_0[key], energy_13[key])
            else:
                assert is_close(value, energy_0[key])
                assert is_close(value, energy_7[key])
                assert is_close(value, energy_8[key])
                assert is_close(value, energy_12[key])
                assert is_close(value, energy_13[key])
