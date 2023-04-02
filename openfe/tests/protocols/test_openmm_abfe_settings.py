# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import pytest
from openff.units import unit as offunit
from openfe.protocols import openmm_afe


@pytest.fixture()
def default_settings():
    return openmm_afe.AbsoluteTransformProtocol.default_settings()


def test_create_default_settings():
    settings = openmm_afe.AbsoluteTransformProtocol.default_settings()
    assert settings


@pytest.mark.parametrize('method, fail', [
    ['Pme', False],
    ['noCutoff', False],
    ['Ewald', True],
    ['CutoffNonPeriodic', True],
    ['CutoffPeriodic', True],
])
def test_systemsettings_nonbonded(method, fail, default_settings):
    if fail:
        with pytest.raises(ValueError, match="Only PME"):
            default_settings.system_settings.nonbonded_method = method
    else:
        default_settings.system_settings.nonbonded_method = method


@pytest.mark.parametrize('val, match', [
    [-1.0 * offunit.nanometer, 'must be a positive'],
    [2.5 * offunit.picoseconds, 'distance units'],
])
def test_systemsettings_cutoff_errors(val, match, default_settings):
    with pytest.raises(ValueError, match=match):
        default_settings.system_settings.nonbonded_cutoff = val


@pytest.mark.parametrize('val, fail', [
    ['TiP3p', False],
    ['SPCE', False],
    ['tip4pEw', False],
    ['Tip5p', False],
    ['opc', True],
    ['tips', True],
    ['tip3p-fb', True],
])
def test_solvent_model_setting(val, fail, default_settings):
    if fail:
        with pytest.raises(ValueError, match="allowed solvent_model"):
            default_settings.solvent_settings.solvent_model = val
    else:
        default_settings.solvent_settings.solvent_model = val


@pytest.mark.parametrize('val, match', [
    [-1.0 * offunit.nanometer, 'must be a positive'],
    [2.5 * offunit.picoseconds, 'distance units'],
])
def test_incorrect_padding(val, match, default_settings):
    with pytest.raises(ValueError, match=match):
        default_settings.solvent_settings.solvent_padding = val


@pytest.mark.parametrize('val', [
    {'elec': 0, 'vdw': 5},
    {'elec': -2, 'vdw': 5},
    {'elec': 5, 'vdw': -2},
    {'elec': 5, 'vdw': 0},
])
def test_incorrect_window_settings(val, default_settings):
    errmsg = "lambda steps must be positive"
    alchem_settings = default_settings.alchemical_settings
    with pytest.raises(ValueError, match=errmsg):
        alchem_settings.lambda_elec_windows = val['elec']
        alchem_settings.lambda_vdw_windows = val['vdw']


@pytest.mark.parametrize('val, fail', [
    ['LOGZ-FLATNESS', False],
    ['MiniMum-VisiTs', False],
    ['histogram-flatness', False],
    ['roundrobin', True],
    ['parsnips', True]
])
def test_supported_flatness_settings(val, fail, default_settings):
    if fail:
        with pytest.raises(ValueError, match="following flatness"):
            default_settings.alchemsampler_settings.flatness_criteria = val
    else:
        default_settings.alchemsampler_settings.flatness_criteria = val


@pytest.mark.parametrize('var, val', [
    ['online_analysis_target_error',
     -0.05 * offunit.boltzmann_constant * offunit.kelvin],
    ['n_repeats', -1],
    ['n_repeats', 0],
    ['online_analysis_minimum_iterations', -2],
    ['gamma0', -2],
    ['n_replicas', -2],
    ['n_replicas', 0]
])
def test_nonnegative_alchem_settings(var, val, default_settings):
    alchem_settings = default_settings.alchemsampler_settings
    with pytest.raises(ValueError, match="positive values"):
        setattr(alchem_settings, var, val)


@pytest.mark.parametrize('val, fail', [
    ['REPEX', False],
    ['SaMs', False],
    ['independent', False],
    ['noneq', True],
    ['AWH', True]
])
def test_supported_sampler(val, fail, default_settings):
    if fail:
        with pytest.raises(ValueError, match="sampler_method values"):
            default_settings.alchemsampler_settings.sampler_method = val
    else:
        default_settings.alchemsampler_settings.sampler_method = val


@pytest.mark.parametrize('var, val', [
    ['collision_rate', -1 / offunit.picosecond],
    ['n_restart_attempts', -2],
    ['timestep', 0 * offunit.femtosecond],
    ['timestep', -2 * offunit.femtosecond],
    ['n_steps', 0 * offunit.timestep],
    ['n_steps', -1 * offunit.timestep],
    ['constraint_tolerance', -2e-06],
    ['constraint_tolerance', 0]
])
def test_nonnegative_integrator_settings(var, val, default_settings):
    int_settings = default_settings.integrator_settings
    with pytest.raises(ValueError, match="positive values"):
        setattr(int_settings, var, val)


def test_timestep_is_not_time(default_settings):
    with pytest.raises(ValueError, match="time units"):
        default_settings.integrator_settings.timestep = 1 * offunit.nanometer


def test_collision_is_not_inverse_time(default_settings):
    with pytest.raises(ValueError, match="inverse time"):
        int_settings = default_settings.integrator_settings
        int_settings.collision_rate = 1 * offunit.picosecond


@pytest.mark.parametrize(
        'var', ['equilibration_length', 'production_length']
)
def test_sim_lengths_not_time(var, default_settings):
    settings = default_settings.simulation_settings
    with pytest.raises(ValueError, match="must be in time units"):
        setattr(settings, var, 1 * offunit.nanometer)


@pytest.mark.parametrize('var, val', [ 
    ['minimization_steps', -1],
    ['minimization_steps', 0],
    ['equilibration_length', -1 * offunit.picosecond],
    ['equilibration_length', 0 * offunit.picosecond],
    ['production_length', -1 * offunit.picosecond],
    ['production_length', 0 * offunit.picosecond],
    ['checkpoint_interval', -1 * offunit.timestep],
    ['checkpoint_interval', 0 * offunit.timestep],
])
def test_nonnegative_sim_settings(var, val, default_settings):
    settings = default_settings.simulation_settings
    with pytest.raises(ValueError, match="must be positive"):
        setattr(settings, var, val)

