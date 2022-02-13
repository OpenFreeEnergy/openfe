import pytest

import click.testing

from openfecli.cli import OpenFECLI, main
from openfecli.plugins import OFECommandPlugin


class TestCLI:
    def setup(self):
        self.cli = OpenFECLI()

    def test_invoke(self):
        runner = click.testing.CliRunner()
        with runner.isolated_filesystem():
            # isolated_filesystem is overkill here, but good practice for
            # testing with CliRunner
            result = runner.invoke(main, ["-h"])
            assert result.exit_code == 0
            assert "Usage: openfe" in result.output

    def test_command_sections(self):
        # This test does not ensure the order of the sections, and does not
        # prevent other sections from being added later. It only ensures
        # that the main 4 sections continue to exist.
        included = ["Setup", "Simulation", "Orchestration", "Analysis"]
        for sec in included:
            assert sec in self.cli.COMMAND_SECTIONS

    def test_get_installed_plugins(self):
        # Test that we correctly load some plugins. This test only ensures
        # that some plugins are loaded; it currently does nothing to ensure
        # the identity of the specific plugins.
        plugins = self.cli.get_installed_plugins()
        for plugin in plugins:
            assert isinstance(plugin, OFECommandPlugin)

        assert len(plugins) > 0
