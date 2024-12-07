import pytest
import click
from importlib import resources
import pathlib
import json
from click.testing import CliRunner

from openfecli.commands.quickrun import quickrun
from gufe.tokenization import JSON_HANDLER


@pytest.fixture
def json_file():
    with resources.files('openfecli.tests.data') as d:
        json_file = str(d / 'transformation.json')

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

        if outfile := extra_args.get('-o'):
            assert pathlib.Path(outfile).exists()
            with open(outfile, mode='r') as outf:
                dct = json.load(outf, cls=JSON_HANDLER.decoder)

            assert set(dct) == {'estimate', 'uncertainty',
                                'protocol_result', 'unit_results'}

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

def test_quickrun_output_file_in_nonexistent_directory(json_file):
    """Should catch invalid filepaths up front."""
    runner = CliRunner()
    outfile = "not_dir/foo.json"
    result = runner.invoke(quickrun, [json_file, '-o', outfile])
    assert result.exit_code == 2
    assert "Cannot write" in result.output
 
def test_quickrun_unit_error():
    with resources.files('openfecli.tests.data') as d:
        json_file = str(d / 'bad_transformation.json')

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(quickrun, [json_file, '-o', 'foo.json'])
        assert result.exit_code == 1
        assert pathlib.Path("foo.json").exists()
        # TODO: I'm still not happy with this... failure result does not see
        # to be stored in JSON
        # not sure whether that means we should always be storing all
        # protocol dag results maybe?
