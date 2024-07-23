# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from importlib import resources
import copy
from pathlib import Path
import pytest
import sys
from pymbar.utils import ParameterError
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from openmm import app, MonteCarloBarostat, NonbondedForce
from openmm import unit as ommunit
from openmmtools import multistate
from openff.toolkit import Molecule as OFFMol
from openff.toolkit.utils.toolkits import RDKitToolkitWrapper
from openff.toolkit.utils.toolkit_registry import ToolkitRegistry
from openff.units import unit
from openff.units.openmm import ensure_quantity, from_openmm
from gufe.settings import OpenMMSystemGeneratorFFSettings, ThermoSettings
import openfe
from openfe.protocols.openmm_utils import (
    settings_validation, system_validation, system_creation,
    multistate_analysis, omm_settings, charge_generation
)
from openfe.protocols.openmm_utils.charge_generation import (
    HAS_NAGL, HAS_ESPALOMA, HAS_OPENEYE
)
from openfe.protocols.openmm_rfe.equil_rfe_settings import (
    IntegratorSettings,
    OpenMMSolvationSettings,
)
from unittest import mock


@pytest.mark.parametrize('padding, number_solv, box_vectors, box_size', [
    [1.2 * unit.nanometer, 20, 20 * np.identity(3) * unit.angstrom,
     [2, 2, 2] * unit.angstrom],
    [1.2 * unit.nanometer, None, None, [2, 2, 2] * unit.angstrom],
    [1.2 * unit.nanometer, None, 20 * np.identity(3) * unit.angstrom, None],
    [1.2 * unit.nanometer, 20, None, None],
])
def test_validate_ommsolvation_settings_unique_settings(
    padding, number_solv, box_vectors, box_size
):
    settings = OpenMMSolvationSettings(
        solvent_padding=padding,
        number_of_solvent_molecules=number_solv,
        box_vectors=box_vectors,
        box_size=box_size,
    )

    errmsg = "Only one of solvent_padding, number_of_solvent_molecules,"
    with pytest.raises(ValueError, match=errmsg):
        settings_validation.validate_openmm_solvation_settings(settings)


@pytest.mark.parametrize('box_vectors, box_size', [
    [20 * np.identity(3) * unit.angstrom, None],
    [None, [2, 2, 2] * unit.angstrom],
])
def test_validate_ommsolvation_settings_shape_conflicts(
    box_vectors, box_size,
):
    settings = OpenMMSolvationSettings(
        solvent_padding=None,
        box_vectors=box_vectors,
        box_size=box_size,
        box_shape='cube',
    )

    errmsg = "box_shape cannot be defined alongside either box_size"
    with pytest.raises(ValueError, match=errmsg):
        settings_validation.validate_openmm_solvation_settings(settings)


def test_validate_timestep():
    with pytest.raises(ValueError, match="too large for hydrogen mass"):
        settings_validation.validate_timestep(2.0, 4.0 * unit.femtoseconds)


@pytest.mark.parametrize('s,ts,mc,es', [
    [5 * unit.nanoseconds, 4 * unit.femtoseconds, 250, 1250000],
    [1 * unit.nanoseconds, 4 * unit.femtoseconds, 250, 250000],
    [1 * unit.picoseconds, 2 * unit.femtoseconds, 250, 500],
])
def test_get_simsteps(s, ts, mc, es):
    sim_steps = settings_validation.get_simsteps(s, ts, mc)

    assert sim_steps == es


def test_get_simsteps_indivisible_simtime():
    errmsg = "Simulation time not divisible by timestep"
    timelength = 1.003 * unit.picosecond
    with pytest.raises(ValueError, match=errmsg):
        settings_validation.get_simsteps(timelength, 2 * unit.femtoseconds, 100)


def test_mc_indivisible():
    errmsg = "Simulation time 1.0 ps should contain"
    timelength = 1 * unit.picoseconds
    with pytest.raises(ValueError, match=errmsg):
        settings_validation.get_simsteps(
                timelength, 2 * unit.femtoseconds, 1000)


def test_get_alchemical_components(benzene_modifications,
                                   T4_protein_component):

    stateA = openfe.ChemicalSystem({'A': benzene_modifications['benzene'],
                                    'B': benzene_modifications['toluene'],
                                    'P': T4_protein_component,
                                    'S': openfe.SolventComponent(smiles='C')})
    stateB = openfe.ChemicalSystem({'A': benzene_modifications['benzene'],
                                    'B': benzene_modifications['benzonitrile'],
                                    'P': T4_protein_component,
                                    'S': openfe.SolventComponent()})

    alchem_comps = system_validation.get_alchemical_components(stateA, stateB)

    assert len(alchem_comps['stateA']) == 2
    assert benzene_modifications['toluene'] in alchem_comps['stateA']
    assert openfe.SolventComponent(smiles='C') in alchem_comps['stateA']
    assert len(alchem_comps['stateB']) == 2
    assert benzene_modifications['benzonitrile'] in alchem_comps['stateB']
    assert openfe.SolventComponent() in alchem_comps['stateB']


def test_duplicate_chemical_components(benzene_modifications):
    stateA = openfe.ChemicalSystem({'A': benzene_modifications['toluene'],
                                    'B': benzene_modifications['toluene'], })
    stateB = openfe.ChemicalSystem({'A': benzene_modifications['toluene']})

    errmsg = "state A components B:"

    with pytest.raises(ValueError, match=errmsg):
        system_validation.get_alchemical_components(stateA, stateB)


def test_validate_solvent_nocutoff(benzene_modifications):

    state = openfe.ChemicalSystem({'A': benzene_modifications['toluene'],
                                   'S': openfe.SolventComponent()})

    with pytest.raises(ValueError, match="nocutoff cannot be used"):
        system_validation.validate_solvent(state, 'nocutoff')


def test_validate_solvent_multiple_solvent(benzene_modifications):

    state = openfe.ChemicalSystem({'A': benzene_modifications['toluene'],
                                   'S': openfe.SolventComponent(),
                                   'S2': openfe.SolventComponent()})

    with pytest.raises(ValueError, match="Multiple SolventComponent"):
        system_validation.validate_solvent(state, 'pme')


def test_not_water_solvent(benzene_modifications):

    state = openfe.ChemicalSystem({'A': benzene_modifications['toluene'],
                                   'S': openfe.SolventComponent(smiles='C')})

    with pytest.raises(ValueError, match="Non water solvent"):
        system_validation.validate_solvent(state, 'pme')


def test_multiple_proteins(T4_protein_component):

    state = openfe.ChemicalSystem({'A': T4_protein_component,
                                   'B': T4_protein_component})

    with pytest.raises(ValueError, match="Multiple ProteinComponent"):
        system_validation.validate_protein(state)


def test_get_components_gas(benzene_modifications):

    state = openfe.ChemicalSystem({'A': benzene_modifications['benzene'],
                                   'B': benzene_modifications['toluene'], })

    s, p, mols = system_validation.get_components(state)

    assert s is None
    assert p is None
    assert len(mols) == 2


def test_components_solvent(benzene_modifications):

    state = openfe.ChemicalSystem({'S': openfe.SolventComponent(),
                                   'A': benzene_modifications['benzene'],
                                   'B': benzene_modifications['toluene'], })

    s, p, mols = system_validation.get_components(state)

    assert s == openfe.SolventComponent()
    assert p is None
    assert len(mols) == 2


def test_components_complex(T4_protein_component, benzene_modifications):

    state = openfe.ChemicalSystem({'S': openfe.SolventComponent(),
                                   'A': benzene_modifications['benzene'],
                                   'B': benzene_modifications['toluene'],
                                   'P': T4_protein_component,})

    s, p, mols = system_validation.get_components(state)

    assert s == openfe.SolventComponent()
    assert p == T4_protein_component
    assert len(mols) == 2


@pytest.fixture(scope='module')
def get_settings():
    forcefield_settings = OpenMMSystemGeneratorFFSettings()
    integrator_settings = IntegratorSettings()
    thermo_settings = ThermoSettings(
        temperature=298.15 * unit.kelvin,
        pressure=1 * unit.bar,
    )

    return forcefield_settings, integrator_settings, thermo_settings


class TestFEAnalysis:

    # Note: class scope _will_ cause this to segfault - the reporter has to close
    @pytest.fixture(scope='function')
    def reporter(self):
        with resources.files('openfe.tests.data.openmm_rfe') as d:
            ncfile = str(d / 'vacuum_nocoord.nc')

        with resources.files('openfe.tests.data.openmm_rfe') as d:
            chkfile = str(d / 'vacuum_nocoord_checkpoint.nc')

        r = multistate.MultiStateReporter(
            storage=ncfile, checkpoint_storage=chkfile
        )
        try:
            yield r
        finally:
            r.close()
    
    @pytest.fixture()
    def analyzer(self, reporter):
        return multistate_analysis.MultistateEquilFEAnalysis(
            reporter, sampling_method='repex',
            result_units=unit.kilocalorie_per_mole,
        )
    
    def test_free_energies(self, analyzer):
        ret_dict = analyzer.unit_results_dict
        assert len(ret_dict.items()) == 7
        assert pytest.approx(ret_dict['unit_estimate'].m) == -47.9606
        assert pytest.approx(ret_dict['unit_estimate_error'].m) == 0.02396789
        # forward and reverse (since we do this ourselves)
        assert_allclose(
            ret_dict['forward_and_reverse_energies']['fractions'],
            np.array([0.08988764, 0.191011, 0.292135, 0.393258, 0.494382,
                      0.595506, 0.696629, 0.797753, 0.898876, 1.0]),
            rtol=1e-04,
        )
        assert_allclose(
            ret_dict['forward_and_reverse_energies']['forward_DGs'].m,
            np.array([-48.057326, -48.038367, -48.033994, -48.0228, -48.028532,
                      -48.025258, -48.006349, -47.986304, -47.972138, -47.960623]),
            rtol=1e-04,
        )
        assert_allclose(
            ret_dict['forward_and_reverse_energies']['forward_dDGs'].m,
            np.array([0.07471 , 0.052914, 0.041508, 0.036613, 0.032827, 0.030489,
                      0.028154, 0.026529, 0.025284, 0.023968]),
            rtol=1e-04,
        )
        assert_allclose(
            ret_dict['forward_and_reverse_energies']['reverse_DGs'].m,
            np.array([-47.823839, -47.833107, -47.845866, -47.858173, -47.883887,
                      -47.915963, -47.93319, -47.939125, -47.949016, -47.960623]),
            rtol=1e-04,
        )
        assert_allclose(
            ret_dict['forward_and_reverse_energies']['reverse_dDGs'].m,
            np.array([0.081209, 0.055975, 0.044693, 0.038691, 0.034603, 0.031894,
                      0.029417, 0.027082, 0.025316, 0.023968]),
            rtol=1e-04,
        )

    def test_plots(self, analyzer, tmpdir):
        with tmpdir.as_cwd():
            analyzer.plot(filepath=Path('.'), filename_prefix='')
            assert Path('forward_reverse_convergence.png').is_file()
            assert Path('mbar_overlap_matrix.png').is_file()
            assert Path('replica_exchange_matrix.png').is_file()
            assert Path('replica_state_timeseries.png').is_file()

    def test_plot_convergence_bad_units(self, analyzer):
        
        with pytest.raises(ValueError, match='Unknown plotting units'):
            openfe.analysis.plotting.plot_convergence(
                analyzer.forward_and_reverse_free_energies,
                unit.nanometer,
            )

    def test_analyze_unknown_method_warning_and_error(self, reporter):

        with pytest.warns(UserWarning, match='Unknown sampling method'):
            ana = multistate_analysis.MultistateEquilFEAnalysis(
                      reporter, sampling_method='replex',
                      result_units=unit.kilocalorie_per_mole,
                  )

        with pytest.raises(ValueError, match="Exchange matrix"):
            ana.replica_exchange_statistics


class TestSystemCreation:
    def test_system_generator_nosolv_nocache(self, get_settings):
        ffsets, intsets, thermosets = get_settings
        generator = system_creation.get_system_generator(
            ffsets, thermosets, intsets, None, False
        )
        assert generator.barostat is None
        assert generator.template_generator._cache is None
        assert not generator.postprocess_system

        forcefield_kwargs = {
            'constraints': app.HBonds,
            'rigidWater': True,
            'removeCMMotion': False,
            'hydrogenMass': 3.0 * ommunit.amu
        }
        assert generator.forcefield_kwargs == forcefield_kwargs
        periodic_kwargs = {
                'nonbondedMethod': app.PME,
                'nonbondedCutoff': 1.0 * ommunit.nanometer
        }
        nonperiodic_kwargs = {'nonbondedMethod': app.NoCutoff,}
        assert generator.nonperiodic_forcefield_kwargs == nonperiodic_kwargs
        assert generator.periodic_forcefield_kwargs == periodic_kwargs

    def test_system_generator_solv_cache(self, get_settings):
        ffsets, intsets, thermosets = get_settings

        thermosets.temperature = 320 * unit.kelvin
        thermosets.pressure = 1.25 * unit.bar
        intsets.barostat_frequency = 200 * unit.timestep
        generator = system_creation.get_system_generator(
            ffsets, thermosets, intsets, Path('./db.json'), True
        )

        # Check barostat conditions
        assert isinstance(generator.barostat, MonteCarloBarostat)

        pressure = ensure_quantity(
            generator.barostat.getDefaultPressure(), 'openff',
        )
        temperature = ensure_quantity(
            generator.barostat.getDefaultTemperature(), 'openff',
        )
        assert pressure.m == pytest.approx(1.25)
        assert pressure.units == unit.bar
        assert temperature.m == pytest.approx(320)
        assert temperature.units == unit.kelvin
        assert generator.barostat.getFrequency() == 200

        # Check cache file
        assert generator.template_generator._cache == 'db.json'

    def test_get_omm_modeller_complex(self, T4_protein_component,
                                      benzene_modifications,
                                      get_settings):
        ffsets, intsets, thermosets = get_settings
        generator = system_creation.get_system_generator(
            ffsets, thermosets, intsets, None, True
        )

        smc = benzene_modifications['toluene']
        mol = smc.to_openff()
        generator.create_system(mol.to_topology().to_openmm(),
                                molecules=[mol])

        model, comp_resids = system_creation.get_omm_modeller(
                T4_protein_component, openfe.SolventComponent(),
                {smc: mol},
                generator.forcefield,
                OpenMMSolvationSettings())

        resids = [r for r in model.topology.residues()]
        assert resids[163].name == 'NME'
        assert resids[164].name == 'UNK'
        assert resids[165].name == 'HOH'
        assert_equal(comp_resids[T4_protein_component], np.linspace(0, 163, 164))
        assert_equal(comp_resids[smc], np.array([164]))
        assert_equal(comp_resids[openfe.SolventComponent()],
                     np.linspace(165, len(resids)-1, len(resids)-165))

    @pytest.fixture(scope='module')
    def ligand_mol_and_generator(self, get_settings):
        # Create offmol
        offmol = OFFMol.from_smiles('[O-]C=O')
        offmol.generate_conformers()
        offmol.assign_partial_charges(partial_charge_method='am1bcc')
        smc = openfe.SmallMoleculeComponent.from_openff(offmol)

        ffsets, intsets, thermosets = get_settings
        generator = system_creation.get_system_generator(
            ffsets, thermosets, intsets, None, True
        )

        # Register offmol in generator
        generator.create_system(offmol.to_topology().to_openmm(),
                                molecules=[offmol])

        return (offmol, smc, generator)

    def test_get_omm_modeller_ligand_no_neutralize(
        self, ligand_mol_and_generator
    ):

        offmol, smc, generator = ligand_mol_and_generator

        model, comp_resids = system_creation.get_omm_modeller(
            None,
            openfe.SolventComponent(neutralize=False),
            {smc: offmol},
            generator.forcefield,
            OpenMMSolvationSettings(),
        )

        system = generator.create_system(
            model.topology,
            molecules=[offmol]
        )

        # Now let's check the total charge
        nonbonded = [f for f in system.getForces()
                     if isinstance(f, NonbondedForce)][0]

        charge = 0 * ommunit.elementary_charge

        for i in range(system.getNumParticles()):
            c, s, e = nonbonded.getParticleParameters(i)
            charge += c

        charge = ensure_quantity(charge, 'openff')

        assert pytest.approx(charge.m) == -1.0

    @pytest.mark.parametrize('n_expected, neutralize, shape',
        [[400, False, 'cube'], [399, True, 'dodecahedron'],
         [400, False, 'octahedron']]
    )
    def test_omm_modeller_ligand_n_solv(
        self, ligand_mol_and_generator, n_expected, neutralize, shape
    ):
        offmol, smc, generator = ligand_mol_and_generator

        solv_settings = OpenMMSolvationSettings(
            solvent_padding=None,
            number_of_solvent_molecules=400,
            box_vectors=None,
            box_size=None,
            box_shape=shape
        )

        model, comp_resids = system_creation.get_omm_modeller(
            None,
            openfe.SolventComponent(
                neutralize=neutralize,
                ion_concentration = 0 * unit.molar
            ),
            {smc: offmol},
            generator.forcefield,
            solv_settings,
        )

        waters = [r for r in model.topology.residues() if r.name == 'HOH']
        assert len(waters) == n_expected

    def test_omm_modeller_box_size(self, ligand_mol_and_generator):
        offmol, smc, generator = ligand_mol_and_generator

        solv_settings = OpenMMSolvationSettings(
            solvent_padding=None,
            number_of_solvent_molecules=None,
            box_vectors=None,
            box_size=[2, 2, 2]*unit.nanometer,
            box_shape=None
        )

        model, comp_resids = system_creation.get_omm_modeller(
            None,
            openfe.SolventComponent(),
            {smc: offmol},
            generator.forcefield,
            solv_settings
        )

        vectors = model.topology.getPeriodicBoxVectors()

        assert_allclose(
            from_openmm(vectors),
            [[2, 0, 0], [0, 2, 0], [0, 0, 2]] * unit.nanometer
        )

    def test_omm_modeller_box_vectors(self, ligand_mol_and_generator):
        offmol, smc, generator = ligand_mol_and_generator

        solv_settings = OpenMMSolvationSettings(
            solvent_padding=None,
            number_of_solvent_molecules=None,
            box_vectors=[
                [2, 0, 0], [0, 2, 0], [0, 0, 5]
            ] * unit.nanometer,
            box_size=None,
            box_shape=None,
        )

        model, comp_resids = system_creation.get_omm_modeller(
            None,
            openfe.SolventComponent(),
            {smc: offmol},
            generator.forcefield,
            solv_settings
        )

        vectors = model.topology.getPeriodicBoxVectors()

        assert_allclose(
            from_openmm(vectors),
            [[2, 0, 0], [0, 2, 0], [0, 0, 5]] * unit.nanometer
        )

def test_convert_steps_per_iteration():
    sim = omm_settings.MultiStateSimulationSettings(
        equilibration_length='10 ps',
        production_length='10 ps',
        time_per_iteration='1.0 ps',
    )
    inty = omm_settings.IntegratorSettings(
        timestep='4 fs'
    )

    spi = settings_validation.convert_steps_per_iteration(sim, inty)

    assert spi == 250


def test_convert_steps_per_iteration_failure():
    sim = omm_settings.MultiStateSimulationSettings(
        equilibration_length='10 ps',
        production_length='10 ps',
        time_per_iteration='1.0 ps',
    )
    inty = omm_settings.IntegratorSettings(
        timestep='3 fs'
    )

    with pytest.raises(ValueError, match="does not evenly divide"):
        settings_validation.convert_steps_per_iteration(sim, inty)


def test_convert_real_time_analysis_iterations():
    sim = omm_settings.MultiStateSimulationSettings(
        equilibration_length='10 ps',
        production_length='10 ps',
        time_per_iteration='1.0 ps',
        real_time_analysis_interval='250 ps',
        real_time_analysis_minimum_time='500 ps',
    )

    rta_its, rta_min_its = settings_validation.convert_real_time_analysis_iterations(sim)

    assert rta_its == 250, 500


def test_convert_real_time_analysis_iterations_interval_fail():
    # shouldn't like 250.5 ps / 1.0 ps
    sim = omm_settings.MultiStateSimulationSettings(
        equilibration_length='10 ps',
        production_length='10 ps',
        time_per_iteration='1.0 ps',
        real_time_analysis_interval='250.5 ps',
        real_time_analysis_minimum_time='500 ps',
    )

    with pytest.raises(ValueError, match='does not evenly divide'):
        settings_validation.convert_real_time_analysis_iterations(sim)


def test_convert_real_time_analysis_iterations_min_interval_fail():
    # shouldn't like 500.5 ps / 1 ps
    sim = omm_settings.MultiStateSimulationSettings(
        equilibration_length='10 ps',
        production_length='10 ps',
        time_per_iteration='1.0 ps',
        real_time_analysis_interval='250 ps',
        real_time_analysis_minimum_time='500.5 ps',
    )

    with pytest.raises(ValueError, match='does not evenly divide'):
        settings_validation.convert_real_time_analysis_iterations(sim)


def test_convert_real_time_analysis_iterations_None():
    sim = omm_settings.MultiStateSimulationSettings(
        equilibration_length='10 ps',
        production_length='10 ps',
        time_per_iteration='1.0 ps',
        real_time_analysis_interval=None,
        real_time_analysis_minimum_time='500 ps',
    )

    rta_its, rta_min_its = settings_validation.convert_real_time_analysis_iterations(sim)

    assert rta_its is None
    assert rta_min_its is None


def test_convert_target_error_from_kcal_per_mole_to_kT():
    kT = settings_validation.convert_target_error_from_kcal_per_mole_to_kT(
        temperature=298.15 * unit.kelvin,
        target_error=0.12 * unit.kilocalorie_per_mole,
    )

    assert kT == pytest.approx(0.20253681663365392)


def test_convert_target_error_from_kcal_per_mole_to_kT_zero():
    # special case, 0 input gives 0 output
    kT = settings_validation.convert_target_error_from_kcal_per_mole_to_kT(
        temperature=298.15 * unit.kelvin,
        target_error=0.0 * unit.kilocalorie_per_mole,
    )

    assert kT == 0.0


class TestOFFPartialCharge:
    @pytest.fixture(scope='function')
    def uncharged_mol(self, CN_molecule):
        return CN_molecule.to_openff()

    @pytest.mark.parametrize('overwrite', [True, False])
    def test_offmol_chg_gen_charged_overwrite(
        self, overwrite, uncharged_mol
    ):
        chg = [
            1 for _ in range(len(uncharged_mol.atoms))
        ] * unit.elementary_charge

        uncharged_mol.partial_charges = copy.deepcopy(chg)
    
        charge_generation.assign_offmol_partial_charges(
            uncharged_mol,
            overwrite=overwrite,
            method='am1bcc',
            toolkit_backend='ambertools',
            generate_n_conformers=None,
            nagl_model=None,
        )
    
        assert np.allclose(uncharged_mol.partial_charges, chg) != overwrite

    def test_unknown_method(self, uncharged_mol):
        with pytest.raises(ValueError, match="Unknown partial charge method"):
            charge_generation.assign_offmol_partial_charges(
                uncharged_mol,
                overwrite=False,
                method='foo',
                toolkit_backend='ambertools',
                generate_n_conformers=None,
                nagl_model=None,
            )

    @pytest.mark.parametrize('method, backend', [
        ['am1bcc', 'rdkit'],
        ['am1bccelf10', 'ambertools'],
        ['nagl', 'bar'],
        ['espaloma', 'openeye'],
    ])
    def test_incompatible_backend_am1bcc(
        self, method, backend, uncharged_mol
    ):
        with pytest.raises(ValueError, match='Selected toolkit_backend'):
            charge_generation.assign_offmol_partial_charges(
                uncharged_mol,
                overwrite=False,
                method=method,
                toolkit_backend=backend,
                generate_n_conformers=None,
                nagl_model=None
            )

    def test_no_conformers(self, uncharged_mol):
        uncharged_mol._conformers = None

        with pytest.raises(ValueError, match='No conformers'):
            charge_generation.assign_offmol_partial_charges(
                uncharged_mol,
                overwrite=False,
                method='am1bcc',
                toolkit_backend='ambertools',
                generate_n_conformers=None,
                nagl_model=None,
            )

    def test_too_many_existing_conformers(self, uncharged_mol):
        uncharged_mol.generate_conformers(
            n_conformers=2,
            rms_cutoff=0.001 * unit.angstrom,
            toolkit_registry=RDKitToolkitWrapper(),
        )

        with pytest.raises(ValueError, match="too many conformers"):
            charge_generation.assign_offmol_partial_charges(
                uncharged_mol,
                overwrite=False,
                method='am1bcc',
                toolkit_backend='ambertools',
                generate_n_conformers=None,
                nagl_model=None,
            )

    def test_too_many_requested_conformers(self, uncharged_mol):
        
        with pytest.raises(ValueError, match="5 conformers were requested"):
            charge_generation.assign_offmol_partial_charges(
                uncharged_mol,
                overwrite=False,
                method='am1bcc',
                toolkit_backend='ambertools',
                generate_n_conformers=5,
                nagl_model=None,
            )

    def test_am1bcc_no_conformer(self, uncharged_mol):

        uncharged_mol._conformers = None

        with pytest.raises(ValueError, match='at least one conformer'):
            charge_generation.assign_offmol_am1bcc_charges(
                uncharged_mol,
                partial_charge_method='am1bcc',
                toolkit_registry=ToolkitRegistry([RDKitToolkitWrapper()])
            )

    @pytest.mark.slow
    def test_am1bcc_conformer_nochange(self, eg5_ligands):

        lig = eg5_ligands[0].to_openff()

        conf = copy.deepcopy(lig.conformers)

        # Get charges without conf generation
        charge_generation.assign_offmol_partial_charges(
            lig,
            overwrite=False,
            method='am1bcc',
            toolkit_backend='ambertools',
            generate_n_conformers=None,
            nagl_model=None,
        )

        # check the conformation hasn't changed
        assert_allclose(conf, lig.conformers)

        # copy the charges to check that the conf gen will change things
        charges = copy.deepcopy(lig.partial_charges)

        # now with conformer generation
        charge_generation.assign_offmol_partial_charges(
            lig,
            overwrite=True,
            method='am1bcc',
            toolkit_backend='ambertools',
            generate_n_conformers=1,
            nagl_model=None
        )

        # conformer shouldn't have changed
        assert_allclose(conf, lig.conformers)

        # but the charges should have
        assert not np.allclose(charges, lig.partial_charges)

    @pytest.mark.skipif(not HAS_NAGL, reason='NAGL is not available')
    def test_no_production_nagl(self, uncharged_mol):
        
        with pytest.raises(ValueError, match='No production am1bcc NAGL'):
            charge_generation.assign_offmol_partial_charges(
                uncharged_mol,
                overwrite=False,
                method='nagl',
                toolkit_backend='rdkit',
                generate_n_conformers=None,
                nagl_model=None,
            )

    # Note: skipping nagl tests on macos/darwin due to known issues
    # see: https://github.com/openforcefield/openff-nagl/issues/78
    @pytest.mark.parametrize('method, backend, ref_key, confs', [
        ('am1bcc', 'ambertools', 'ambertools', None),
        pytest.param(
            'am1bcc', 'openeye', 'openeye', None,
            marks=pytest.mark.skipif(
                not HAS_OPENEYE, reason='needs oechem',
            ),
        ),
        pytest.param(
            'am1bccelf10', 'openeye', 'openeye', 500,
            marks=pytest.mark.skipif(
                not HAS_OPENEYE, reason='needs oechem',
            ),
        ),
        pytest.param(
            'nagl', 'rdkit', 'nagl', None,
            marks=pytest.mark.skipif(
                not HAS_NAGL or sys.platform.startswith('darwin'),
                reason='needs NAGL and/or on macos',
            ),
        ),
        pytest.param(
            'nagl', 'ambertools', 'nagl', None,
            marks=pytest.mark.skipif(
                not HAS_NAGL or sys.platform.startswith('darwin')
                , reason='needs NAGL and/or on macos',
            ),
        ),
        pytest.param(
            'nagl', 'openeye', 'nagl', None,
            marks=pytest.mark.skipif(
                not HAS_NAGL or not HAS_OPENEYE or sys.platform.startswith('darwin'),
                reason='needs NAGL and oechem and not on macos',
            ),
        ),
        pytest.param(
            'espaloma', 'rdkit', 'espaloma', None,
            marks=pytest.mark.skipif(
                not HAS_ESPALOMA, reason='needs espaloma',
            ),
        ),
        pytest.param(
            'espaloma', 'ambertools', 'espaloma', None,
            marks=pytest.mark.skipif(
                not HAS_ESPALOMA, reason='needs espaloma',
            ),
        ),
    ])
    def test_am1bcc_reference(
        self, uncharged_mol, method, backend, ref_key, confs,
        am1bcc_ref_charges,
    ):
        """
        Check partial charge generation using what would
        be intended default settings for a CN molecule
        """
        charge_generation.assign_offmol_partial_charges(
            uncharged_mol,
            overwrite=False,
            method=method,
            toolkit_backend=backend,
            generate_n_conformers=None,
            nagl_model="openff-gnn-am1bcc-0.1.0-rc.1.pt",
        )

        assert_allclose(
            am1bcc_ref_charges[ref_key],
            uncharged_mol.partial_charges,
            rtol=1e-4
        )

    def test_nagl_import_error(self, monkeypatch, uncharged_mol):
        monkeypatch.setattr(
            sys.modules['openfe.protocols.openmm_utils.charge_generation'],
            'HAS_NAGL',
            False
        )

        with pytest.raises(ImportError, match='NAGL toolkit is not available'):
            charge_generation.assign_offmol_partial_charges(
                uncharged_mol,
                overwrite=False,
                method='nagl',
                toolkit_backend='rdkit',
                generate_n_conformers=None,
                nagl_model=None
            )

    def test_espaloma_import_error(self, monkeypatch, uncharged_mol):
        monkeypatch.setattr(
            sys.modules['openfe.protocols.openmm_utils.charge_generation'],
            'HAS_ESPALOMA',
            False
        )

        with pytest.raises(ImportError, match='Espaloma'):
            charge_generation.assign_offmol_partial_charges(
                uncharged_mol,
                overwrite=False,
                method='espaloma',
                toolkit_backend='rdkit',
                generate_n_conformers=None,
                nagl_model=None,
            )

    def test_openeye_import_error(self, monkeypatch, uncharged_mol):
        monkeypatch.setattr(
            sys.modules['openfe.protocols.openmm_utils.charge_generation'],
            'HAS_OPENEYE',
            False
        )

        with pytest.raises(ImportError, match='OpenEye is not available'):
            charge_generation.assign_offmol_partial_charges(
                uncharged_mol,
                overwrite=False,
                method='am1bcc',
                toolkit_backend='openeye',
                generate_n_conformers=None,
                nagl_model=None,
            )


@pytest.mark.slow
@pytest.mark.download
def test_forward_backwards_failure(simulation_nc):
    rep = multistate.multistatereporter.MultiStateReporter(
        simulation_nc,
        open_mode='r'
    )
    ana = multistate_analysis.MultistateEquilFEAnalysis(
        rep,
        sampling_method='repex',
        result_units=unit.kilocalorie_per_mole,
    )

    with mock.patch('openfe.protocols.openmm_utils.multistate_analysis.MultistateEquilFEAnalysis._get_free_energy',
                    side_effect=ParameterError):
        ret = ana.get_forward_and_reverse_analysis()

    assert ret is None
