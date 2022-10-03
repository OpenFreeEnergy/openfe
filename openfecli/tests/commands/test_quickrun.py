import pytest
import click
import importlib
from click.testing import CliRunner

from openfecli.commands.quickrun import quickrun

def test_quickrun():
    runner = CliRunner()
    with importlib.resources.path('openfecli.tests.data',
                                  'transformation.json') as f:
        json_file = str(f)

    with runner.isolated_filesystem():
        result = runner.invoke(quickrun, [json_file])
        assert result.exit_code == 0
        assert "Here is the result" in result.output
        assert "Additional information" in result.output
