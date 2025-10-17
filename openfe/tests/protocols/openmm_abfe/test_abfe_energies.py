# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from importlib import resources

import gufe
import numpy as np
import pytest
from openmmtools.alchemy import (
    AlchemicalRegion,
    AlchemicalState,
    AbsoluteAlchemicalFactory,
)
import openmm
from openmm import Platform
from openfe.protocols.openmm_utils.omm_settings import OpenMMSolvationSettings
from openfe.protocols import openmm_afe
from openfe.protocols.openmm_afe import (
    AbsoluteBindingComplexUnit,
)
from openmm import unit as ommunit
from openfe.protocols.openmm_septop.utils import deserialize
import collections
import copy


class AlchemStateRest(AlchemicalState):
    """
    A modified AlchemicalState for testing.

    Note: we don't need this in the main protocol since we use composable
    thermodynamic states.
    """

    lambda_restraints = AlchemicalState._LambdaParameter("lambda_restraints")


def get_alchemical_energy_components(
    alchemical_system, alchemical_state, positions, platform
):
    """Compute potential energy of the alchemical system by Force.

    This can be useful for debug and analysis.

    Parameters
    ----------
    alchemical_system : openmm.AlchemicalSystem
        An alchemically modified system.
    alchemical_state : AlchemicalState
        The alchemical state to set the Context to.
    positions : openmm.unit.Quantity of dimension (natoms, 3)
        Coordinates to use for energy test (units of distance).
    platform : openmm.Platform, optional
        The OpenMM platform to use to compute the energy. If None,
        OpenMM tries to select the fastest available.

    Returns
    -------
    energy_components : dict str: openmm.unit.Quantity
        A string label describing the role of the force associated to
        its contribution to the potential energy.

    """
    # Find and label all forces.
    force_labels = AbsoluteAlchemicalFactory._find_force_components(alchemical_system)
    assert len(force_labels) <= 32, (
        "The alchemical system has more than 32 force groups; "
        "can't compute individual force component energies."
    )

    # Create deep copy of alchemical system.
    system = copy.deepcopy(alchemical_system)

    # Separate all forces into separate force groups.
    for force_index, force in enumerate(system.getForces()):
        force.setForceGroup(force_index)

    assert len(force_labels) == len(system.getForces())

    # Create a Context in the given state.
    integrator = openmm.LangevinMiddleIntegrator(
        300 * openmm.unit.kelvin,
        1.0 / openmm.unit.picosecond,
        1.0 * openmm.unit.femtoseconds,
    )
    # integrator = openmm.VerletIntegrator(0.0 * ommunit.femtoseconds)
    context = openmm.Context(system, integrator, platform)
    context.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
    context.setPositions(np.round(positions, 3))
    context.setVelocitiesToTemperature(300, 42)
    alchemical_state.apply_to_context(context)

    # Get energy components
    energy_components = collections.OrderedDict()
    for force_label, force_index in force_labels.items():
        energy_components[force_label] = context.getState(
            getEnergy=True,
            groups={force_index},
        ).getPotentialEnergy()

    # Clean up
    del context, integrator
    return energy_components


@pytest.mark.slow
class TestT4EnergiesRegression:
    """
    Test:
      * Regression of a system energies against a previusly serialized one.
      * That the energies do what we think they should be doing.
    """

    @pytest.fixture()
    def t4_reference_system(self):
        with resources.as_file(resources.files("openfe.tests.data.openmm_afe")) as d:
            f = d / "T4_abfe_system.xml.bz2"
            system = deserialize(f)
        return system

    @pytest.fixture()
    def t4_validation_data(self, benzene_modifications, T4_protein_component, tmpdir):
        s = openmm_afe.AbsoluteBindingProtocol.default_settings()
        s.protocol_repeats = 1
        s.forcefield_settings.small_molecule_forcefield = "openff-2.2.1"
        s.complex_solvation_settings = OpenMMSolvationSettings(
            solvent_padding=None,
            number_of_solvent_molecules=1000,
            box_shape="dodecahedron",
        )

        protocol = openmm_afe.AbsoluteBindingProtocol(settings=s)

        stateA = gufe.ChemicalSystem(
            {
                "protein": T4_protein_component,
                "benzene": benzene_modifications["benzene"],
                "solvent": gufe.SolventComponent(),
            }
        )

        stateB = gufe.ChemicalSystem(
            {
                "protein": T4_protein_component,
                "solvent": gufe.SolventComponent(),
            }
        )

        dag = protocol.create(stateA=stateA, stateB=stateB, mapping=None)

        complex_units = [
            u for u in dag.protocol_units if isinstance(u, AbsoluteBindingComplexUnit)
        ]

        with tmpdir.as_cwd():
            data = complex_units[0].run(dry=True)["debug"]
            return data

    @staticmethod
    def get_energy_components(
        system,
        indices,
        positions,
        lambda_sterics,
        lambda_electrostatics,
        lambda_restraints,
    ):
        platform = Platform.getPlatformByName("Reference")
        alchemical_region = AlchemicalRegion(alchemical_atoms=indices)

        alchemical_state = AlchemStateRest.from_system(
            system, parameters_name_suffix=alchemical_region.name
        )

        alchemical_state.lambda_sterics = lambda_sterics
        alchemical_state.lambda_electrostatics = lambda_electrostatics
        alchemical_state.lambda_restraints = lambda_restraints

        return get_alchemical_energy_components(
            system, alchemical_state, positions, platform=platform
        )

    @pytest.mark.parametrize("lambda_val", [0, 1])
    def test_energies_regression(
        self, lambda_val, t4_reference_system, t4_validation_data
    ):

        energies_ref = self.get_energy_components(
            t4_reference_system,
            t4_validation_data["alchem_indices"],
            t4_validation_data["positions"],
            lambda_val,
            lambda_val,
            lambda_val,
        )

        energies_val = self.get_energy_components(
            t4_validation_data["alchem_system"],
            t4_validation_data["alchem_indices"],
            t4_validation_data["positions"],
            lambda_val,
            lambda_val,
            lambda_val,
        )

        # Check the keys match
        assert [k for k in energies_ref.keys()] == [k for k in energies_val.keys()]

        for key in energies_ref.keys():
            e_ref = energies_ref[key].value_in_unit(ommunit.kilojoule_per_mole)
            e_val = energies_val[key].value_in_unit(ommunit.kilojoule_per_mole)
            assert pytest.approx(e_ref, abs=1e-3) == e_val

    def test_lambda_scale(self, t4_validation_data):

        def assert_energies(actual, expected, nonbonded_lower: bool):
            assert [k for k in expected.keys()] == [k for k in actual.keys()]
            for key in expected.keys():
                e_ref = expected[key].value_in_unit(ommunit.kilojoule_per_mole)
                e_val = actual[key].value_in_unit(ommunit.kilojoule_per_mole)

                # Knowing exactly by how much the NBF has reduced is hard, so we
                # just check it's lower
                if nonbonded_lower and key == "unmodified NonbondedForce":
                    assert e_val < e_ref
                else:
                    assert e_val == pytest.approx(e_ref, abs=0.1)

        # lambda 1 on all
        energies = self.get_energy_components(
            t4_validation_data["alchem_system"],
            t4_validation_data["alchem_indices"],
            t4_validation_data["positions"],
            lambda_sterics=1.0,
            lambda_electrostatics=1.0,
            lambda_restraints=1.0,
        )

        # turn off restraints
        expected = copy.deepcopy(energies)
        expected["unmodified CustomCompoundBondForce"] = 0 * ommunit.kilojoule_per_mole

        energies = self.get_energy_components(
            t4_validation_data["alchem_system"],
            t4_validation_data["alchem_indices"],
            t4_validation_data["positions"],
            lambda_sterics=1.0,
            lambda_electrostatics=1.0,
            lambda_restraints=0.0,
        )

        assert_energies(energies, expected, nonbonded_lower=False)

        # turn off electrostatics

        energies = self.get_energy_components(
            t4_validation_data["alchem_system"],
            t4_validation_data["alchem_indices"],
            t4_validation_data["positions"],
            lambda_sterics=1.0,
            lambda_electrostatics=0.0,
            lambda_restraints=0.0,
        )

        # assert all but the NonbondedForce, that should just be lower
        assert_energies(energies, expected, nonbonded_lower=True)

        # turn off sterics
        expected = copy.deepcopy(energies)
        expected[
            "alchemically modified NonbondedForce for non-alchemical/alchemical sterics"
        ] = (0 * ommunit.kilojoule_per_mole)
        expected[
            "alchemically modified BondForce for non-alchemical/alchemical sterics exceptions"
        ] = (0 * ommunit.kilojoule_per_mole)

        energies = self.get_energy_components(
            t4_validation_data["alchem_system"],
            t4_validation_data["alchem_indices"],
            t4_validation_data["positions"],
            lambda_sterics=0.0,
            lambda_electrostatics=0.0,
            lambda_restraints=0.0,
        )

        # assert all but the NonbondedForce, that should just be lower
        assert_energies(energies, expected, nonbonded_lower=False)
