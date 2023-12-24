# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from importlib import resources
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from openmm import app, MonteCarloBarostat, NonbondedForce
from openmm import unit as ommunit
from openmmtools import multistate
from openff.toolkit import Molecule as OFFMol
from openff.units import unit
from openff.units.openmm import ensure_quantity
from gufe.settings import OpenMMSystemGeneratorFFSettings, ThermoSettings
import openfe
from openfe.protocols.openmm_utils import (
    settings_validation, system_validation, system_creation,
    multistate_analysis
)
from openfe.protocols.openmm_rfe.equil_rfe_settings import (
    SystemSettings, SolvationSettings, IntegratorSettings,
)


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
    thermo_settings = ThermoSettings(
        temperature=298.15 * unit.kelvin,
        pressure=1 * unit.bar,
    )
    system_settings = SystemSettings()
    integrator_settings = IntegratorSettings()

    return (forcefield_settings, thermo_settings, system_settings,
            integrator_settings)


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
        ffsets, thermosets, systemsets, intsets = get_settings
        generator = system_creation.get_system_generator(
            ffsets, thermosets, intsets, systemsets, None, False
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
        ffsets, thermosets, systemsets, intsets = get_settings

        thermosets.temperature = 320 * unit.kelvin
        thermosets.pressure = 1.25 * unit.bar
        intsets.barostat_frequency = 200 * unit.timestep
        generator = system_creation.get_system_generator(
            ffsets, thermosets, intsets, systemsets, Path('./db.json'), True
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
        ffsets, thermosets, systemsets, intsets = get_settings
        generator = system_creation.get_system_generator(
            ffsets, thermosets, intsets, systemsets, None, True
        )

        smc = benzene_modifications['toluene']
        mol = smc.to_openff()
        generator.create_system(mol.to_topology().to_openmm(),
                                molecules=[mol])

        model, comp_resids = system_creation.get_omm_modeller(
                T4_protein_component, openfe.SolventComponent(),
                {smc: mol},
                generator.forcefield,
                SolvationSettings())

        resids = [r for r in model.topology.residues()]
        assert resids[163].name == 'NME'
        assert resids[164].name == 'UNK'
        assert resids[165].name == 'HOH'
        assert_equal(comp_resids[T4_protein_component], np.linspace(0, 163, 164))
        assert_equal(comp_resids[smc], np.array([164]))
        assert_equal(comp_resids[openfe.SolventComponent()],
                     np.linspace(165, len(resids)-1, len(resids)-165))

    def test_get_omm_modeller_ligand_no_neutralize(self, get_settings):
        ffsets, thermosets, systemsets, intsets = get_settings
        generator = system_creation.get_system_generator(
            ffsets, thermosets, intsets, systemsets, None, True
        )

        offmol = OFFMol.from_smiles('[O-]C=O')
        offmol.generate_conformers()
        smc = openfe.SmallMoleculeComponent.from_openff(offmol)

        generator.create_system(offmol.to_topology().to_openmm(),
                                molecules=[offmol])
        model, comp_resids = system_creation.get_omm_modeller(
            None,
            openfe.SolventComponent(neutralize=False),
            {smc: offmol},
            generator.forcefield,
            SolvationSettings(),
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
