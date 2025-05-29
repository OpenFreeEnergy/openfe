# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import pytest
import numpy as np
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


class TestEquilRFESettingsFromString:
    def test_alchemical_settings(self):
        s = equil_rfe_settings.AlchemicalSettings(softcore_LJ='gapsys')

        s.explicit_charge_correction_cutoff = '0.85 nm'

        assert s.explicit_charge_correction_cutoff == 0.85 * unit.nanometer


class TestOpenMMSolvationSettings:
    def test_unreduced_box_vectors(self):
        s = omm_settings.OpenMMSolvationSettings()

        # From interchange tests
        # rhombic dodecahedron with first and last rows swapped
        box_vectors = np.asarray(
            [
                [0.5, 0.5, np.sqrt(2.0) / 2.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
        )

        with pytest.raises(ValueError, match="not in OpenMM reduced form"):
            s.box_vectors = box_vectors

    @pytest.mark.parametrize('n_solv', [0, -1])
    def test_non_positive_solvent(self, n_solv):
        s = omm_settings.OpenMMSolvationSettings()

        with pytest.raises(ValueError, match="must be positive"):
            s.number_of_solvent_molecules=n_solv

    def test_box_size_properties_non_1d(self):
        s = omm_settings.OpenMMSolvationSettings()

        with pytest.raises(ValueError, match="must be a 1-D array"):
            s.box_size = np.array([[1, 2, 3], [1, 2, 3]]) * unit.angstrom
