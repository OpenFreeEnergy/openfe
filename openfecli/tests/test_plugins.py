import click
from openfecli.plugins import OFECommandPlugin


@click.command("fake")
def fake():
    pass  # -no-cov-  a fake placeholder click subcommand


class TestOFECommandPlugin:
    def setup_method(self):
        self.plugin = OFECommandPlugin(
            command=fake,
            section="Some Section",
            requires_ofe=(0, 0, 1)
        )

    def test_plugin_setup(self):
        assert self.plugin.command is fake
        assert isinstance(self.plugin.command, click.Command)
        assert self.plugin.section == "Some Section"
        assert self.plugin.requires_lib == self.plugin.requires_cli
        assert self.plugin.requires_lib == (0, 0, 1)
        assert self.plugin.requires_cli == (0, 0, 1)

