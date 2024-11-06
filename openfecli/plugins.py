# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from plugcli.plugin_management import CommandPlugin


class OFECommandPlugin(CommandPlugin):
    def __init__(self, command, section, requires_ofe:tuple[int, ...]):
        """ Base class for openfecli commands.

            parameters
            ----------
            requires_ofe: tuple
                tuple representing the minimum allowed version of the underlying
                library. E.g. v3.1.2 is (3,1,2).
        """
        super().__init__(command=command,
                         section=section,
                         requires_lib=requires_ofe,
                         requires_cli=requires_ofe)
