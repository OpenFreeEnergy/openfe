from plugcli.plugin_management import CommandPlugin


class OFECommandPlugin(CommandPlugin):
    def __init__(self, command, section, requires_ofe):
        super().__init__(command=command,
                         section=section,
                         requires_lib=requires_ofe,
                         requires_cli=requires_ofe)


