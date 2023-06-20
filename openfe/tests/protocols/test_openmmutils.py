# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import assert_equal
from openmm import app, MonteCarloBarostat
from openmm import unit as ommunit
from openff.units import unit
from gufe.settings import OpenMMSystemGeneratorFFSettings, ThermoSettings
import openfe
from openfe.protocols.openmm_utils import (
    settings_validation, system_validation, system_creation
)
from openfe.protocols.openmm_utils.utils import (
    subsample_omm_topology,
)
from openfe.protocols.openmm_rfe.equil_rfe_settings import (
    SystemSettings, SolvationSettings,
)


def test_validate_timestep():
    with pytest.raises(ValueError, match="too large for hydrogen mass"):
        settings_validation.validate_timestep(2.0, 4.0 * unit.femtoseconds)


@pytest.mark.parametrize('e,p,ts,mc,es,ps', [
    [1 * unit.nanoseconds, 5 * unit.nanoseconds, 4 * unit.femtoseconds,
     250, 250000, 1250000],
    [1 * unit.picoseconds, 1 * unit.picoseconds, 2 * unit.femtoseconds,
     250, 500, 500],
])
def test_get_simsteps(e, p, ts, mc, es, ps):
    equil_steps, prod_steps = settings_validation.get_simsteps(e, p, ts, mc)

    assert equil_steps == es
    assert prod_steps == ps


@pytest.mark.parametrize('nametype, timelengths', [
    ['Equilibration', [1.003 * unit.picoseconds, 1 * unit.picoseconds]],
    ['Production', [1 * unit.picoseconds, 1.003 * unit.picoseconds]],
])
def test_get_simsteps_indivisible_simtime(nametype, timelengths):
    errmsg = f"{nametype} time not divisible by timestep"
    with pytest.raises(ValueError, match=errmsg):
        settings_validation.get_simsteps(
                timelengths[0],
                timelengths[1],
                2 * unit.femtoseconds,
                100)


@pytest.mark.parametrize('nametype, timelengths', [
    ['Equilibration', [1 * unit.picoseconds, 10 * unit.picoseconds]],
    ['Production', [10 * unit.picoseconds,  1 * unit.picoseconds]],
])
def test_mc_indivisible(nametype, timelengths):
    errmsg = f"{nametype} time 1.0 ps should contain"
    with pytest.raises(ValueError, match=errmsg):
        settings_validation.get_simsteps(
                timelengths[0], timelengths[1],
                2 * unit.femtoseconds, 1000)


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
                                    'B': benzene_modifications['toluene'],})
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
                                   'B': benzene_modifications['toluene'],})

    s, p, mols = system_validation.get_components(state)

    assert s is None
    assert p is None
    assert len(mols) == 2


def test_components_solvent(benzene_modifications):

    state = openfe.ChemicalSystem({'S': openfe.SolventComponent(),
                                   'A': benzene_modifications['benzene'],
                                   'B': benzene_modifications['toluene'],})

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

    return forcefield_settings, thermo_settings, system_settings


@pytest.mark.parametrize(
    'indices,n_atoms,n_bonds,n_residues,n_chains,resname', [
        [tuple(2613 + i for i in range(15)), 15, 15, 1, 1, 'UNK'],
        [(6, 7, 23, 24, 37, 38, 2613), 7, 3, 4, 2, 'MET'],
    ],
)
def test_subsample_topology(T4_protein_component, benzene_modifications,
                            get_settings, indices, n_atoms, n_bonds,
                            n_residues, n_chains, resname):
    # Get settings
    ffsets, thermosets, systemsets = get_settings
    generator = system_creation.get_system_generator(
        ffsets, thermosets, systemsets, None, True
    )

    # Get generator and Modeller object
    mol = benzene_modifications['toluene'].to_openff()
    generator.create_system(mol.to_topology().to_openmm(),
                            molecules=[mol])

    model, comp_resids = system_creation.get_omm_modeller(
        T4_protein_component, openfe.SolventComponent(),
        [benzene_modifications['toluene'],],
        generator.forcefield,
        SolvationSettings()
    )

    # Subsample the Topology
    topology = model.getTopology()
    sub_top = subsample_omm_topology(topology, indices)

    # Checks
    assert len(list(sub_top.atoms())) == n_atoms
    assert len(list(sub_top.bonds())) == n_bonds
    assert len(list(sub_top.residues())) == n_residues
    assert len(list(sub_top.chains())) == n_chains
    assert list(sub_top.residues())[0].name == resname


class TestSystemCreation:
    @staticmethod
    def get_settings():
        forcefield_settings = OpenMMSystemGeneratorFFSettings()
        thermo_settings = ThermoSettings(
                temperature=298.15 * unit.kelvin,
                pressure=1 * unit.bar,
        )
        system_settings = SystemSettings()

        return forcefield_settings, thermo_settings, system_settings

    def test_system_generator_nosolv_nocache(self, get_settings):
        ffsets, thermosets, systemsets = get_settings
        generator = system_creation.get_system_generator(
                ffsets, thermosets, systemsets, None, False)
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
        ffsets, thermosets, systemsets = get_settings
        generator = system_creation.get_system_generator(
                ffsets, thermosets, systemsets, Path('./db.json'), True)
        assert isinstance(generator.barostat, MonteCarloBarostat)
        assert generator.template_generator._cache == 'db.json'

    def test_get_omm_modeller_complex(self, T4_protein_component,
                                      benzene_modifications,
                                      get_settings):
        ffsets, thermosets, systemsets = get_settings
        generator = system_creation.get_system_generator(
                ffsets, thermosets, systemsets, None, True)

        mol = benzene_modifications['toluene'].to_openff()
        generator.create_system(mol.to_topology().to_openmm(),
                                molecules=[mol])

        model, comp_resids = system_creation.get_omm_modeller(
                T4_protein_component, openfe.SolventComponent(),
                [benzene_modifications['toluene'],],
                generator.forcefield,
                SolvationSettings())

        resids = [r for r in model.topology.residues()]
        assert resids[163].name == 'NME'
        assert resids[164].name == 'UNK'
        assert resids[165].name == 'HOH'
        assert_equal(comp_resids[T4_protein_component], np.linspace(0, 163, 164))
        assert_equal(comp_resids[benzene_modifications['toluene']], np.array([164]))
        assert_equal(comp_resids[openfe.SolventComponent()],
                     np.linspace(165, len(resids)-1, len(resids)-165))
