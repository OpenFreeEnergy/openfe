# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from plugcli.params import MultiStrategyGetter, Option
from openfecli.parameters.utils import import_parameter


def _atommapper_from_openfe_setup(user_input, context):
    return import_parameter("openfe.setup." + user_input)


def _atommapper_from_qualname(user_input, context):
    return import_parameter(user_input)


get_atommapper = MultiStrategyGetter(
    strategies=[
        _atommapper_from_qualname,
        _atommapper_from_openfe_setup,
    ],
    error_message=("Unable to create atom mapper from user input "
                   "'{user_input}'. Please check spelling and "
                   "capitalization.")
)

MAPPER = Option(
    "--mapper",
    getter=get_atommapper,
    help=("Atom mapper; can either be a name in the openfe.setup namespace "
          "or a custom fully-qualified name.")
)
