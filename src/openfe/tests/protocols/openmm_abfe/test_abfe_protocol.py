# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from math import sqrt
from unittest import mock

import gufe
import mdtraj as mdt
import numpy as np
import openmm
import pytest
from numpy.testing import assert_allclose
from openff.units import unit as offunit
from openff.units.openmm import from_openmm, to_openmm
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
from openmm import unit as omm_unit
from openmm import unit as ommunit
from openmmtools.alchemy import (
    AlchemicalRegion,
)
from openmmtools.multistate.multistatesampler import MultiStateSampler
from openmmtools.tests.test_alchemy import (
    check_interacting_energy_components,
    check_noninteracting_energy_components,
    compare_system_energies,
)

import openfe
from openfe import ChemicalSystem, SmallMoleculeComponent, SolventComponent
from openfe.protocols import openmm_afe
from openfe.protocols.openmm_afe import (
    AbsoluteBindingProtocol,
)
from openfe.protocols.openmm_afe.abfe_units import (
    ABFEComplexSetupUnit,
    ABFEComplexSimUnit,
    ABFESolventSetupUnit,
    ABFESolventSimUnit,
)

from .utils import UNIT_TYPES, _get_units


@pytest.fixture()
def default_settings():
    return AbsoluteBindingProtocol.default_settings()


@pytest.fixture(scope="module")
def benzene_wcharges(benzene_modifications):
    benz_off = benzene_modifications["benzene"].to_openff()
    benz_off.assign_partial_charges(partial_charge_method="gasteiger")
    return SmallMoleculeComponent.from_openff(benz_off)


def test_create_default_protocol(default_settings):
    # this is roughly how it should be created
    protocol = AbsoluteBindingProtocol(
        settings=default_settings,
    )
    assert protocol


def test_serialize_protocol(default_settings):
    protocol = AbsoluteBindingProtocol(
        settings=default_settings,
    )

    ser = protocol.to_dict()
    ret = AbsoluteBindingProtocol.from_dict(ser)
    assert protocol == ret


def test_repeat_units(benzene_modifications, T4_protein_component):
    protocol = openmm_afe.AbsoluteBindingProtocol(
        settings=openmm_afe.AbsoluteBindingProtocol.default_settings()
    )

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

    dag = protocol.create(
        stateA=stateA,
        stateB=stateB,
        mapping=None,
    )

    # 6 protocol unit, 3 per repeat
    pus = list(dag.protocol_units)
    assert len(pus) == 18

    # Check info for each repeat
    for phase in ["solvent", "complex"]:
        setup = _get_units(pus, UNIT_TYPES[phase]["setup"])
        sim = _get_units(pus, UNIT_TYPES[phase]["sim"])
        analysis = _get_units(pus, UNIT_TYPES[phase]["analysis"])

        # Should be 3 of each set
        assert len(setup) == len(sim) == len(analysis) == 3

        # Check that the dag chain is correct
        for analysis_pu in analysis:
            repeat_id = analysis_pu.inputs["repeat_id"]
            setup_pu = [s for s in setup if s.inputs["repeat_id"] == repeat_id][0]
            sim_pu = [s for s in sim if s.inputs["repeat_id"] == repeat_id][0]
            assert analysis_pu.inputs["setup_results"] == setup_pu
            assert analysis_pu.inputs["simulation_results"] == sim_pu
            assert sim_pu.inputs["setup_results"] == setup_pu


def test_create_independent_repeat_ids(benzene_modifications, T4_protein_component):
    s = openmm_afe.AbsoluteBindingProtocol.default_settings()

    protocol = openmm_afe.AbsoluteBindingProtocol(
        settings=s,
    )

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

    dags = []
    for i in range(2):
        dags.append(protocol.create(stateA=stateA, stateB=stateB, mapping=None))

    repeat_ids = set()

    for dag in dags:
        # 3 sets of 6 units
        assert len(list(dag.protocol_units)) == 18
        for u in dag.protocol_units:
            repeat_ids.add(u.inputs["repeat_id"])

    # squashed by repeat_id, that's 2 sets of 6
    assert len(repeat_ids) == 12


def test_mda_universe_error():
    """
    Test that we get an error if we pass no positions or trajectory
    when calling the mda Universe getter.
    """
    with pytest.raises(ValueError, match="No positions to create"):
        _ = openmm_afe.ABFEComplexSetupUnit._get_mda_universe(
            topology="foo", positions=None, trajectory=None
        )


class TestT4LysozymeDryRun:
    solvent = SolventComponent(ion_concentration=0 * offunit.molar)
    num_all_not_water = 2634
    num_complex_atoms = 2613
    # No ions
    num_ligand_atoms = 12

    barostat_by_phase = {
        "complex": MonteCarloBarostat,
        "solvent": MonteCarloBarostat,
    }

    @pytest.fixture(scope="class")
    def protocol(self, settings):
        return openmm_afe.AbsoluteBindingProtocol(
            settings=settings,
        )

    @pytest.fixture(scope="class")
    def settings(self):
        s = openmm_afe.AbsoluteBindingProtocol.default_settings()
        s.protocol_repeats = 1
        s.engine_settings.compute_platform = "cpu"
        s.complex_output_settings.output_indices = "not water"
        s.complex_solvation_settings.box_shape = "dodecahedron"
        s.complex_solvation_settings.solvent_padding = 0.9 * offunit.nanometer
        s.solvent_solvation_settings.box_shape = "cube"
        return s

    @pytest.fixture(scope="class")
    def dag(self, protocol, benzene_wcharges, T4_protein_component):
        stateA = ChemicalSystem(
            {
                "benzene": benzene_wcharges,
                "protein": T4_protein_component,
                "solvent": self.solvent,
            }
        )

        stateB = ChemicalSystem(
            {
                "protein": T4_protein_component,
                "solvent": self.solvent,
            }
        )

        return protocol.create(
            stateA=stateA,
            stateB=stateB,
            mapping=None,
        )

    @pytest.fixture(scope="class")
    def complex_setup_units(self, dag):
        return _get_units(dag.protocol_units, UNIT_TYPES["complex"]["setup"])

    @pytest.fixture(scope="class")
    def complex_sim_units(self, dag):
        return _get_units(dag.protocol_units, UNIT_TYPES["complex"]["sim"])

    @pytest.fixture(scope="class")
    def solvent_setup_units(self, dag):
        return _get_units(dag.protocol_units, UNIT_TYPES["solvent"]["setup"])

    @pytest.fixture(scope="class")
    def solvent_sim_units(self, dag):
        return _get_units(dag.protocol_units, UNIT_TYPES["solvent"]["sim"])

    def test_number_of_units(
        self,
        dag,
        complex_setup_units,
        complex_sim_units,
        solvent_setup_units,
        solvent_sim_units,
    ):
        assert len(list(dag.protocol_units)) == 6
        assert len(complex_setup_units) == len(complex_sim_units) == 1
        assert len(solvent_setup_units) == len(solvent_sim_units) == 1

    def _assert_force_num(self, system, forcetype, number):
        forces = [f for f in system.getForces() if isinstance(f, forcetype)]
        assert len(forces) == number

    def _assert_expected_alchemical_forces(self, system, phase: str, settings):
        """
        Assert the forces expected in the alchemical system.
        """
        barostat_type = self.barostat_by_phase[phase]
        self._assert_force_num(system, NonbondedForce, 1)
        self._assert_force_num(system, CustomNonbondedForce, 2)
        self._assert_force_num(system, CustomBondForce, 2)
        self._assert_force_num(system, HarmonicBondForce, 1)
        self._assert_force_num(system, HarmonicAngleForce, 1)
        self._assert_force_num(system, PeriodicTorsionForce, 1)
        self._assert_force_num(system, barostat_type, 1)

        if phase == "complex":
            self._assert_force_num(system, CustomCompoundBondForce, 1)
            assert len(system.getForces()) == 10
        else:
            assert len(system.getForces()) == 9

        # Check the nonbonded force has the right contents
        nonbond = [f for f in system.getForces() if isinstance(f, NonbondedForce)]
        assert len(nonbond) == 1
        assert nonbond[0].getNonbondedMethod() == NonbondedForce.PME
        assert (
            from_openmm(nonbond[0].getCutoffDistance())
            == settings.forcefield_settings.nonbonded_cutoff
        )

        # Check the barostat made it all the way through
        barostat = [f for f in system.getForces() if isinstance(f, barostat_type)]
        assert len(barostat) == 1
        assert barostat[0].getFrequency() == int(
            settings.complex_integrator_settings.barostat_frequency.m
        )
        assert barostat[0].getDefaultPressure() == to_openmm(settings.thermo_settings.pressure)
        assert barostat[0].getDefaultTemperature() == to_openmm(
            settings.thermo_settings.temperature
        )

    def _assert_expected_nonalchemical_forces(self, system, phase: str, settings):
        """
        Assert the forces expected in the non-alchemical system.
        """
        barostat_type = self.barostat_by_phase[phase]
        self._assert_force_num(system, NonbondedForce, 1)
        self._assert_force_num(system, HarmonicBondForce, 1)
        self._assert_force_num(system, HarmonicAngleForce, 1)
        self._assert_force_num(system, PeriodicTorsionForce, 1)
        self._assert_force_num(system, barostat_type, 1)

        assert len(system.getForces()) == 5

        # Check that the nonbonded force has the right contents
        nonbond = [f for f in system.getForces() if isinstance(f, NonbondedForce)]
        assert len(nonbond) == 1
        assert nonbond[0].getNonbondedMethod() == NonbondedForce.PME
        assert (
            from_openmm(nonbond[0].getCutoffDistance())
            == settings.forcefield_settings.nonbonded_cutoff
        )

        # Check the barostat made it all the way through
        barostat = [f for f in system.getForces() if isinstance(f, barostat_type)]
        assert len(barostat) == 1
        assert barostat[0].getFrequency() == int(
            settings.complex_integrator_settings.barostat_frequency.m
        )
        assert barostat[0].getDefaultPressure() == to_openmm(settings.thermo_settings.pressure)
        assert barostat[0].getDefaultTemperature() == to_openmm(
            settings.thermo_settings.temperature
        )

    def _verify_sampler(self, sampler, phase: str, settings):
        """
        Utility to verify the contents of the sampler.
        """
        assert sampler.is_periodic
        assert isinstance(sampler, MultiStateSampler)
        assert isinstance(sampler._thermodynamic_states[0].barostat, MonteCarloBarostat)
        assert sampler._thermodynamic_states[1].pressure == to_openmm(
            settings.thermo_settings.pressure
        )
        for state in sampler._thermodynamic_states:
            system = state.get_system(remove_thermostat=True)
            self._assert_expected_alchemical_forces(system, phase, settings)

    def _check_box_vectors(self, system):
        self._test_dodecahedron_vectors(system)

    @staticmethod
    def _test_dodecahedron_vectors(system):
        # dodecahedron has the following shape:
        # [width, 0, 0], [0, width, 0], [0.5, 0.5, 0.5 * sqrt(2)] * width

        vectors = system.getDefaultPeriodicBoxVectors()
        width = float(from_openmm(vectors)[0][0].to("nanometer").m)

        expected_vectors = [
            [width, 0, 0],
            [0, width, 0],
            [0.5 * width, 0.5 * width, 0.5 * sqrt(2) * width],
        ] * offunit.nanometer

        assert_allclose(
            expected_vectors,
            from_openmm(vectors),
        )

    @staticmethod
    def _test_cubic_vectors(system):
        # cube is an identity matrix
        vectors = system.getDefaultPeriodicBoxVectors()
        width = float(from_openmm(vectors)[0][0].to("nanometer").m)

        expected_vectors = [
            [width, 0, 0],
            [0, width, 0],
            [0, 0, width],
        ] * offunit.nanometer

        assert_allclose(
            expected_vectors,
            from_openmm(vectors),
        )

    @staticmethod
    def _test_energies(reference_system, alchemical_system, alchemical_regions, positions):
        compare_system_energies(
            reference_system=reference_system,
            alchemical_system=alchemical_system,
            alchemical_regions=alchemical_regions,
            positions=positions,
        )

        check_noninteracting_energy_components(
            reference_system=reference_system,
            alchemical_system=alchemical_system,
            alchemical_regions=alchemical_regions,
            positions=positions,
        )

        check_interacting_energy_components(
            reference_system=reference_system,
            alchemical_system=alchemical_system,
            alchemical_regions=alchemical_regions,
            positions=positions,
        )

    def test_complex_dry_run(self, complex_setup_units, complex_sim_units, settings, tmpdir):
        with tmpdir.as_cwd():
            setup_results = complex_setup_units[0].run(dry=True, verbose=True)
            sim_results = complex_sim_units[0].run(
                system=setup_results["alchem_system"],
                positions=setup_results["debug_positions"],
                selection_indices=setup_results["selection_indices"],
                box_vectors=setup_results["box_vectors"],
                alchemical_restraints=True,
                dry=True,
            )

            # Check the sampler
            self._verify_sampler(sim_results["sampler"], "complex", settings=settings)

            # Check the alchemical system
            self._assert_expected_alchemical_forces(
                setup_results["alchem_system"], "complex", settings=settings
            )
            self._check_box_vectors(setup_results["alchem_system"])

            # Check the alchemical indices
            expected_indices = [i + self.num_complex_atoms for i in range(self.num_ligand_atoms)]
            assert expected_indices == setup_results["alchem_indices"]

            # Check the non-alchemical system
            self._assert_expected_nonalchemical_forces(
                setup_results["standard_system"], "complex", settings=settings
            )
            self._check_box_vectors(setup_results["standard_system"])

            # Check the box vectors haven't changed (they shouldn't have because we didn't do MD)
            assert_allclose(
                from_openmm(setup_results["alchem_system"].getDefaultPeriodicBoxVectors()),
                from_openmm(setup_results["standard_system"].getDefaultPeriodicBoxVectors()),
            )

            # Check the PDB
            pdb = mdt.load_pdb(setup_results["pdb_structure"])
            assert pdb.n_atoms == self.num_all_not_water

            # Check energies
            alchem_region = AlchemicalRegion(alchemical_atoms=setup_results["alchem_indices"])
            self._test_energies(
                reference_system=setup_results["standard_system"],
                alchemical_system=setup_results["alchem_system"],
                alchemical_regions=alchem_region,
                positions=setup_results["debug_positions"],
            )

    def test_solvent_dry_run(self, solvent_setup_units, solvent_sim_units, settings, tmpdir):
        with tmpdir.as_cwd():
            setup_results = solvent_setup_units[0].run(dry=True, verbose=True)
            sim_results = solvent_sim_units[0].run(
                system=setup_results["alchem_system"],
                positions=setup_results["debug_positions"],
                selection_indices=setup_results["selection_indices"],
                box_vectors=setup_results["box_vectors"],
                alchemical_restraints=False,
                dry=True,
            )

            # Check the sampler
            self._verify_sampler(sim_results["sampler"], "solvent", settings=settings)

            # Check the alchemical system
            self._assert_expected_alchemical_forces(
                setup_results["alchem_system"], "solvent", settings=settings
            )
            self._test_cubic_vectors(setup_results["alchem_system"])

            # Check the alchemical indices
            expected_indices = [i for i in range(self.num_ligand_atoms)]
            assert expected_indices == setup_results["alchem_indices"]

            # Check the non-alchemical system
            self._assert_expected_nonalchemical_forces(
                setup_results["standard_system"], "solvent", settings=settings
            )
            self._test_cubic_vectors(setup_results["standard_system"])

            # Check the box vectors haven't changed (they shouldn't have because we didn't do MD)
            assert_allclose(
                from_openmm(setup_results["alchem_system"].getDefaultPeriodicBoxVectors()),
                from_openmm(setup_results["standard_system"].getDefaultPeriodicBoxVectors()),
            )

            # Check the PDB
            pdb = mdt.load_pdb(setup_results["pdb_structure"])
            assert pdb.n_atoms == self.num_ligand_atoms

            # Check energies
            alchem_region = AlchemicalRegion(alchemical_atoms=setup_results["alchem_indices"])

            self._test_energies(
                reference_system=setup_results["standard_system"],
                alchemical_system=setup_results["alchem_system"],
                alchemical_regions=alchem_region,
                positions=setup_results["debug_positions"],
            )


@pytest.mark.slow
class TestT4LysozymeTIP4PExtraSettingsDryRun(TestT4LysozymeDryRun):
    """
    TIP4P and a few extra settings to test out the dry run.
    """

    @pytest.fixture(scope="class")
    def settings(self):
        s = openmm_afe.AbsoluteBindingProtocol.default_settings()
        s.protocol_repeats = 1
        s.engine_settings.compute_platform = "cpu"
        s.complex_output_settings.output_indices = "not water"
        s.complex_solvation_settings.box_shape = "dodecahedron"
        s.complex_solvation_settings.solvent_padding = 0.9 * offunit.nanometer
        s.complex_solvation_settings.solvent_model = "tip4pew"
        s.solvent_solvation_settings.box_shape = "cube"
        s.solvent_solvation_settings.solvent_model = "tip4pew"
        s.forcefield_settings.nonbonded_cutoff = 0.8 * offunit.nanometer
        s.forcefield_settings.forcefields = [
            "amber/ff14SB.xml",  # ff14SB protein force field
            "amber/tip4pew_standard.xml",  # FF we are testsing with the fun VS
            "amber/phosaa10.xml",  # Handles THE TPO
        ]
        s.complex_integrator_settings.reassign_velocities = True
        s.solvent_integrator_settings.reassign_velocities = True
        s.complex_integrator_settings.barostat_frequency = 100.0 * offunit.timestep
        s.solvent_integrator_settings.barostat_frequency = 100.0 * offunit.timestep
        s.thermo_settings.pressure = 1.1 * offunit.bar
        return s


def test_user_charges(benzene_modifications, T4_protein_component, tmpdir):
    s = openmm_afe.AbsoluteBindingProtocol.default_settings()
    s.protocol_repeats = 1
    s.engine_settings.compute_platform = "cpu"
    s.complex_solvation_settings.box_shape = "dodecahedron"
    s.complex_solvation_settings.solvent_padding = 0.8 * offunit.nanometer
    s.forcefield_settings.nonbonded_cutoff = 0.7 * offunit.nanometer

    protocol = openmm_afe.AbsoluteBindingProtocol(settings=s)

    def assign_fictitious_charges(offmol):
        """
        Get a random array of fake partial charges for your offmol.
        """
        rand_arr = np.random.randint(1, 10, size=offmol.n_atoms) / 100
        rand_arr[-1] = -sum(rand_arr[:-1])
        return rand_arr * offunit.elementary_charge

    benzene_offmol = benzene_modifications["benzene"].to_openff()
    offmol_pchgs = assign_fictitious_charges(benzene_offmol)
    benzene_offmol.partial_charges = offmol_pchgs
    benzene_smc = openfe.SmallMoleculeComponent.from_openff(benzene_offmol)

    # check propchgs
    prop_chgs = benzene_smc.to_dict()["molprops"]["atom.dprop.PartialCharge"]
    prop_chgs = np.array(prop_chgs.split(), dtype=float)
    assert_allclose(prop_chgs, offmol_pchgs)

    stateA = gufe.ChemicalSystem(
        {
            "protein": T4_protein_component,
            "benzene": benzene_smc,
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

    complex_setup_units = _get_units(dag.protocol_units, UNIT_TYPES["complex"]["setup"])

    with tmpdir.as_cwd():
        results = complex_setup_units[0].run(dry=True)

        system_nbf = [
            f for f in results["standard_system"].getForces() if isinstance(f, NonbondedForce)
        ][0]
        alchem_system_nbf = [
            f
            for f in results["alchem_system"].getForces()
            if isinstance(f, NonbondedForce)
        ][0]  # fmt: skip

        for i in range(12):
            # add 2613 to account for the protein
            index = i + 2613

            c, s, e = system_nbf.getParticleParameters(index)
            assert pytest.approx(prop_chgs[i]) == c.value_in_unit(ommunit.elementary_charge)

            offsets = alchem_system_nbf.getParticleParameterOffset(i)
            assert pytest.approx(prop_chgs[i]) == offsets[2]


@pytest.mark.slow
class TestA2AMembraneDryRun(TestT4LysozymeDryRun):
    solvent = SolventComponent(ion_concentration=0 * offunit.molar)
    num_all_not_water = 16080
    num_complex_atoms = 39390
    # No ions
    num_ligand_atoms = 36

    barostat_by_phase = {
        "complex": MonteCarloMembraneBarostat,
        "solvent": MonteCarloBarostat,
    }

    @pytest.fixture(scope="class")
    def settings(self):
        s = openmm_afe.AbsoluteBindingProtocol.default_settings()
        s.protocol_repeats = 1
        s.engine_settings.compute_platform = "cpu"
        s.complex_output_settings.output_indices = "not water"
        s.solvent_solvation_settings.box_shape = "cube"
        s.complex_integrator_settings.barostat = "MonteCarloMembraneBarostat"
        s.forcefield_settings.forcefields = [
            "amber/ff14SB.xml",
            "amber/tip3p_standard.xml",
            "amber/tip3p_HFE_multivalent.xml",
            "amber/lipid17_merged.xml",
            "amber/phosaa10.xml",
        ]
        return s

    @pytest.fixture(scope="class")
    def dag(self, protocol, a2a_ligands, a2a_protein_membrane_component):
        stateA = ChemicalSystem(
            {
                "ligand": a2a_ligands[0],
                "protein": a2a_protein_membrane_component,
                "solvent": self.solvent,
            }
        )

        stateB = ChemicalSystem(
            {
                "protein": a2a_protein_membrane_component,
                "solvent": self.solvent,
            }
        )

        return protocol.create(
            stateA=stateA,
            stateB=stateB,
            mapping=None,
        )

    def _check_box_vectors(self, system):
        self._test_orthogonal_vectors(system)

    def _verify_sampler(self, sampler, phase: str, settings):
        """
        Utility to verify the contents of the sampler.
        """
        assert sampler.is_periodic
        assert isinstance(sampler, MultiStateSampler)
        barostat_type = self.barostat_by_phase[phase]
        assert isinstance(sampler._thermodynamic_states[0].barostat, barostat_type)
        assert sampler._thermodynamic_states[1].pressure == to_openmm(
            settings.thermo_settings.pressure
        )
        for state in sampler._thermodynamic_states:
            system = state.get_system(remove_thermostat=True)
            self._assert_expected_alchemical_forces(system, phase, settings)

    @staticmethod
    def _test_orthogonal_vectors(system):
        """Test that the system has an orthorhombic (rectangular) periodic box."""
        vectors = system.getDefaultPeriodicBoxVectors()
        vectors = from_openmm(vectors)  # convert to a Quantity array

        # Extract box lengths in nanometers
        width_x, width_y, width_z = [v[i].to("nanometer").m for i, v in enumerate(vectors)]

        # Expected orthogonal box (axis-aligned)
        expected_vectors = (
            np.array(
                [
                    [width_x, 0, 0],
                    [0, width_y, 0],
                    [0, 0, width_z],
                ]
            )
            * offunit.nanometer
        )

        assert_allclose(
            vectors, expected_vectors, atol=1e-5, err_msg=f"Box is not orthogonal:\n{vectors}"
        )
