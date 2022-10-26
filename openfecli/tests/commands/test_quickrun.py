import pytest
import click
import importlib
import pathlib
import json
from click.testing import CliRunner

from openfecli.commands.quickrun import quickrun
from gufe.tokenization import JSON_HANDLER


@pytest.fixture
def json_file():
    with importlib.resources.path('openfecli.tests.data',
                                  'transformation.json') as f:
        json_file = str(f)

    return json_file


@pytest.mark.parametrize('extra_args', [
    {},
    {'-d': 'foo_dir', '-o': 'foo.json'}
])
def test_quickrun(extra_args, json_file):
    extras = sum([list(kv) for kv in extra_args.items()], [])

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(quickrun, [json_file] + extras)
        assert result.exit_code == 0
        assert "Here is the result" in result.output
        assert "Additional information" in result.output

        if outfile := extra_args.get('-o'):
            assert pathlib.Path(outfile).exists()
            with open(outfile, mode='r') as outf:
                dct = json.load(outf, cls=JSON_HANDLER.decoder)

            assert set(dct) == {'estimate', 'uncertainty', 'result'}

        # TODO: need a protocol that drops files to actually do this!
        # if directory := extra_args.get('-d'):
        #     dirpath = pathlib.Path(directory)
        #     assert dirpath.exists()
        #     assert dirpath.is_dir()
        #     assert len(list(dirpath.iterdir())) > 0


def test_quickrun_output_file_exists(json_file):
    runner = CliRunner()
    with runner.isolated_filesystem():
        pathlib.Path('foo.json').touch()
        result = runner.invoke(quickrun, [json_file, '-o', 'foo.json'])
        assert result.exit_code == 2  # usage error
        assert "File 'foo.json' already exists." in result.output
