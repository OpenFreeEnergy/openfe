# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from importlib import resources
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
    AbsoluteAlchemicalFactory,
    AlchemicalRegion,
    AlchemicalState,
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
    AbsoluteBindingComplexUnit,
    AbsoluteBindingProtocol,
    AbsoluteBindingSolventUnit,
)
from openfe.protocols.openmm_utils.omm_settings import OpenMMSolvationSettings


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


def test_unit_tagging(benzene_complex_dag, tmpdir):
    # test that executing the units includes correct gen and repeat info

    dag_units = benzene_complex_dag.protocol_units

    with (
        mock.patch(
            "openfe.protocols.openmm_afe.equil_binding_afe_method.AbsoluteBindingSolventUnit.run",
            return_value={"nc": "file.nc", "last_checkpoint": "chck.nc"},
        ),
        mock.patch(
            "openfe.protocols.openmm_afe.equil_binding_afe_method.AbsoluteBindingComplexUnit.run",
            return_value={"nc": "file.nc", "last_checkpoint": "chck.nc"},
        ),
    ):
        results = []
        for u in dag_units:
            ret = u.execute(context=gufe.Context(tmpdir, tmpdir))
            results.append(ret)

    solv_repeats = set()
    complex_repeats = set()
    for ret in results:
        assert isinstance(ret, gufe.ProtocolUnitResult)
        assert ret.outputs["generation"] == 0
        if ret.outputs["simtype"] == "complex":
            complex_repeats.add(ret.outputs["repeat_id"])
        else:
            solv_repeats.add(ret.outputs["repeat_id"])
    # Repeat ids are random ints so just check their lengths
    assert len(complex_repeats) == len(solv_repeats) == 3


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
        for u in dag.protocol_units:
            repeat_ids.add(u.inputs["repeat_id"])

    assert len(repeat_ids) == 12


class TestT4LysozymeDryRun:
    solvent = SolventComponent(ion_concentration=0 * offunit.molar)
    num_all_not_water = 2634
    num_complex_atoms = 2613
    # No ions
    num_ligand_atoms = 12

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
    def complex_units(self, dag):
        return [u for u in dag.protocol_units if isinstance(u, AbsoluteBindingComplexUnit)]

    @pytest.fixture(scope="class")
    def solvent_units(self, dag):
        return [u for u in dag.protocol_units if isinstance(u, AbsoluteBindingSolventUnit)]

    def test_number_of_units(self, dag, complex_units, solvent_units):
        assert len(list(dag.protocol_units)) == 2
        assert len(complex_units) == 1
        assert len(solvent_units) == 1

    def _get_barostat_type(self, complexed: bool):
        return MonteCarloBarostat

    def _assert_force_num(self, system, forcetype, number):
        forces = [f for f in system.getForces() if isinstance(f, forcetype)]
        assert len(forces) == number

    def _assert_expected_alchemical_forces(self, system, complexed: bool, settings):
        """
        Assert the forces expected in the alchemical system.
        """
        barostat_type = self._get_barostat_type(complexed)
        self._assert_force_num(system, NonbondedForce, 1)
        self._assert_force_num(system, CustomNonbondedForce, 2)
        self._assert_force_num(system, CustomBondForce, 2)
        self._assert_force_num(system, HarmonicBondForce, 1)
        self._assert_force_num(system, HarmonicAngleForce, 1)
        self._assert_force_num(system, PeriodicTorsionForce, 1)
        self._assert_force_num(system, barostat_type, 1)

        if complexed:
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

    def _assert_expected_nonalchemical_forces(self, system, complexed: bool, settings):
        """
        Assert the forces expected in the non-alchemical system.
        """
        barostat_type = self._get_barostat_type(complexed)
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

    def _verify_sampler(self, sampler, complexed: bool, settings):
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
            self._assert_expected_alchemical_forces(system, complexed, settings)

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

    def test_complex_dry_run(self, complex_units, settings, tmpdir):
        with tmpdir.as_cwd():
            data = complex_units[0].run(dry=True, verbose=True)["debug"]

            # Check the sampler
            self._verify_sampler(data["sampler"], complexed=True, settings=settings)

            # Check the alchemical system
            self._assert_expected_alchemical_forces(
                data["alchem_system"], complexed=True, settings=settings
            )
            self._check_box_vectors(data["alchem_system"])

            # Check the alchemical indices
            expected_indices = [i + self.num_complex_atoms for i in range(self.num_ligand_atoms)]
            assert expected_indices == data["alchem_indices"]

            # Check the non-alchemical system
            self._assert_expected_nonalchemical_forces(
                data["system"], complexed=True, settings=settings
            )
            self._check_box_vectors(data["system"])
            # Check the box vectors haven't changed (they shouldn't have because we didn't do MD)
            assert_allclose(
                from_openmm(data["alchem_system"].getDefaultPeriodicBoxVectors()),
                from_openmm(data["system"].getDefaultPeriodicBoxVectors()),
            )

            # Check the PDB
            pdb = mdt.load_pdb("alchemical_system.pdb")
            assert pdb.n_atoms == self.num_all_not_water

            # Check energies
            alchem_region = AlchemicalRegion(alchemical_atoms=data["alchem_indices"])
            self._test_energies(
                reference_system=data["system"],
                alchemical_system=data["alchem_system"],
                alchemical_regions=alchem_region,
                positions=data["positions"],
            )

    def test_solvent_dry_run(self, solvent_units, settings, tmpdir):
        with tmpdir.as_cwd():
            data = solvent_units[0].run(dry=True, verbose=True)["debug"]

            # Check the sampler
            self._verify_sampler(data["sampler"], complexed=False, settings=settings)

            # Check the alchemical system
            self._assert_expected_alchemical_forces(
                data["alchem_system"], complexed=False, settings=settings
            )
            self._test_cubic_vectors(data["alchem_system"])

            # Check the alchemical indices
            expected_indices = [i for i in range(self.num_ligand_atoms)]
            assert expected_indices == data["alchem_indices"]

            # Check the non-alchemical system
            self._assert_expected_nonalchemical_forces(
                data["system"], complexed=False, settings=settings
            )
            self._test_cubic_vectors(data["system"])
            # Check the box vectors haven't changed (they shouldn't have because we didn't do MD)
            assert_allclose(
                from_openmm(data["alchem_system"].getDefaultPeriodicBoxVectors()),
                from_openmm(data["system"].getDefaultPeriodicBoxVectors()),
            )

            # Check the PDB
            pdb = mdt.load_pdb("alchemical_system.pdb")
            assert pdb.n_atoms == self.num_ligand_atoms

            # Check energies
            alchem_region = AlchemicalRegion(alchemical_atoms=data["alchem_indices"])

            self._test_energies(
                reference_system=data["system"],
                alchemical_system=data["alchem_system"],
                alchemical_regions=alchem_region,
                positions=data["positions"],
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

    complex_units = [u for u in dag.protocol_units if isinstance(u, AbsoluteBindingComplexUnit)]

    with tmpdir.as_cwd():
        data = complex_units[0].run(dry=True)["debug"]

        system_nbf = [f for f in data["system"].getForces() if isinstance(f, NonbondedForce)][0]
        alchem_system_nbf = [
            f
            for f in data["alchem_system"].getForces()
            if isinstance(f, NonbondedForce)
        ][0]  # fmt: skip

        for i in range(12):
            # add 2613 to account for the protein
            index = i + 2613

            c, s, e = system_nbf.getParticleParameters(index)
            assert pytest.approx(prop_chgs[i]) == c.value_in_unit(ommunit.elementary_charge)

            offsets = alchem_system_nbf.getParticleParameterOffset(i)
            assert pytest.approx(prop_chgs[i]) == offsets[2]


class TestA2AMembraneDryRun(TestT4LysozymeDryRun):
    solvent = SolventComponent(ion_concentration=0 * offunit.molar)
    num_all_not_water = 22170
    num_complex_atoms = 64119
    # No ions
    num_ligand_atoms = 36

    @pytest.fixture(scope="class")
    def settings(self):
        s = openmm_afe.AbsoluteBindingProtocol.default_settings()
        s.protocol_repeats = 1
        s.engine_settings.compute_platform = "cpu"
        s.complex_output_settings.output_indices = "not water"
        s.complex_solvation_settings.box_shape = "dodecahedron"
        s.complex_solvation_settings.solvent_padding = 0.9 * offunit.nanometer
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

    def _get_barostat_type(self, complexed: bool):
        return MonteCarloMembraneBarostat if complexed else MonteCarloBarostat

    def _check_box_vectors(self, system):
        self._test_orthogonal_vectors(system)

    def _verify_sampler(self, sampler, complexed: bool, settings):
        """
        Utility to verify the contents of the sampler.
        """
        assert sampler.is_periodic
        assert isinstance(sampler, MultiStateSampler)
        if complexed:
            barostat_type = MonteCarloMembraneBarostat
        else:
            barostat_type = MonteCarloBarostat
        assert isinstance(sampler._thermodynamic_states[0].barostat, barostat_type)
        assert sampler._thermodynamic_states[1].pressure == to_openmm(
            settings.thermo_settings.pressure
        )
        for state in sampler._thermodynamic_states:
            system = state.get_system(remove_thermostat=True)
            self._assert_expected_alchemical_forces(system, complexed, settings)

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
