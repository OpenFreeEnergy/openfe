# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import pytest

from openfe.protocols.restraint_utils.openmm.omm_restraints import (
    RestraintParameterState,
)


def test_parameter_state_default():
    param_state = RestraintParameterState()
    assert param_state.lambda_restraints is None


@pytest.mark.parametrize('suffix', [None, 'foo'])
@pytest.mark.parametrize('lambda_var', [0, 0.5, 1.0])
def test_parameter_state_suffix(suffix, lambda_var):
    param_state = RestraintParameterState(
        parameters_name_suffix=suffix, lambda_restraints=lambda_var 
    )

    if suffix is not None:
        param_name = f'lambda_restraints_{suffix}'
    else:
        param_name = 'lambda_restraints'

    assert getattr(param_state, param_name) == lambda_var
    assert len(param_state._parameters.keys()) == 1
    assert param_state._parameters[param_name] == lambda_var
    assert param_state._parameters_name_suffix == suffix
