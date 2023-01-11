import pytest

import click.testing

import contextlib
import logging

from openfecli.cli import OpenFECLI, main
from openfecli.plugins import OFECommandPlugin

@click.command(
    'null-command',
    short_help="Do nothing (testing)"
)
def null_command():
    logger = logging.getLogger("null_command_logger")
    logger.info("Running null command")

PLUGIN = OFECommandPlugin(
    command=null_command,
    section="Analysis",
    requires_ofe=(0, 3),
)

@contextlib.contextmanager
def null_command_context(cli):
    PLUGIN.attach_metadata(location=__file__,
                           plugin_type="file")
    try:
        cli._register_plugin(PLUGIN)
        yield cli
    finally:
        cli._deregister_plugin(PLUGIN)



@pytest.fixture
def cli():
    return OpenFECLI()

class TestCLI:
    def test_invoke(self):
        runner = click.testing.CliRunner()
        with runner.isolated_filesystem():
            # isolated_filesystem is overkill here, but good practice for
            # testing with CliRunner
            result = runner.invoke(main, ["-h"])
            assert result.exit_code == 0
            assert "Usage: openfe" in result.output

    def test_command_sections(self, cli):
        # This test does not ensure the order of the sections, and does not
        # prevent other sections from being added later. It only ensures
        # that the main 4 sections continue to exist.
        included = ["Setup", "Simulation", "Orchestration", "Analysis"]
        for sec in included:
            assert sec in cli.COMMAND_SECTIONS

    def test_get_installed_plugins(self, cli):
        # Test that we correctly load some plugins. This test only ensures
        # that some plugins are loaded; it currently does nothing to ensure
        # the identity of the specific plugins.
        plugins = cli.get_installed_plugins()
        for plugin in plugins:
            assert isinstance(plugin, OFECommandPlugin)

        assert len(plugins) > 0


@pytest.mark.parametrize('with_log', [True, False])
def test_main_log(with_log):
    logged_text = "Running null command\n"
    logfile_text = "\n".join([
        "[loggers]", "keys=root", "",
        "[handlers]", "keys=std", "",
        "[formatters]", "keys=default", "",
        "[formatter_default]", "format=%(message)s", "",
        "[handler_std]", "class=StreamHandler", "level=NOTSET",
        "formatter=default", "args=(sys.stdout,)", ""
        "[logger_root]", "level=DEBUG", "handlers=std"
    ])
    runner = click.testing.CliRunner()
    invocation = ['null_command']
    if with_log:
        invocation = ['--log', 'logging.conf'] + invocation

    expected = logged_text if with_log else ""

    with runner.isolated_filesystem():
        with open("logging.conf", mode='w') as log_conf:
            log_conf.write(logfile_text)

        with null_command_context(main):
            result = runner.invoke(main, invocation)

    found = result.stdout_bytes
    assert found.decode('utf-8') == expected
