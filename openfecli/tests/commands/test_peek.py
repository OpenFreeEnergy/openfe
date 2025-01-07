import pytest
from importlib import resources
import pathlib
import json
from click.testing import CliRunner

from openfecli.commands.peek import peek
from gufe.tokenization import JSON_HANDLER


@pytest.fixture
def json_file():
    with resources.files('openfecli.tests.data') as d:
        json_file = str(d / 'transformation.json')

    return json_file

def test_peek(json_file):

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(peek, json_file)
        assert result.exit_code == 0
        assert "name" in result.output

      