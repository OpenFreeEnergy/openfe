# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import pytest
from openff.units import unit

from openfe.protocols.openmm_rfe import equil_rfe_settings
# afe settings currently have no FloatQuantity values
from openfe.protocols.openmm_utils import omm_settings


class TestOMMSettingsFromStrings:
    # checks that we can set Settings fields via strings
    def test_system_settings(self):
        s = omm_settings.OpenMMSystemGeneratorFFSettings()

        s.nonbonded_cutoff = '1.1 nm'

        assert s.nonbonded_cutoff == 1.1 * unit.nanometer

    def test_solvation_settings(self):
        s = omm_settings.OpenMMSolvationSettings()

        s.solvent_padding = '1.1 nm'

        assert s.solvent_padding == 1.1 * unit.nanometer

    def test_alchemical_sampler_settings(self):
        # todo: early_termination_target_error is in kT, how to pass this as string?
        pass

    def test_integator_settings(self):
        s = omm_settings.IntegratorSettings()

        s.timestep = '3 fs'

        assert s.timestep == 3.0 * unit.femtosecond

        s.langevin_collision_rate = '1.1 / ps'

        assert s.langevin_collision_rate == 1.1 / unit.picosecond

        # todo: nsteps, barostat frequency require IntQuantity

    def test_simulation_settings(self):
        s = omm_settings.SimulationSettings(
            equilibration_length=2.0 * unit.nanosecond,
            production_length=5.0 * unit.nanosecond,
        )

        s.equilibration_length = '2.5 ns'
        s.production_length = '10 ns'

        assert s.equilibration_length == 2.5 * unit.nanosecond
        assert s.production_length == 10.0 * unit.nanosecond

        # todo: checkpoint_interval IntQuantity


class TestEquilRFESettingsFromString:
    def test_alchemical_settings(self):
        s = equil_rfe_settings.AlchemicalSettings(softcore_LJ='gapsys')

        s.explicit_charge_correction_cutoff = '0.85 nm'

        assert s.explicit_charge_correction_cutoff == 0.85 * unit.nanometer


