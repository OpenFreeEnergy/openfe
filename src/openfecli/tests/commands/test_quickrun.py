import json
import os
import pathlib
from importlib import resources
from unittest import mock

import pytest
from click.testing import CliRunner
from gufe import Transformation
from gufe.tokenization import JSON_HANDLER

from openfecli.commands.quickrun import _hash_quickrun_inputs, quickrun

from ..utils import assert_click_success

from ..utils import assert_click_success


@pytest.fixture
def json_file():
    with resources.as_file(resources.files("openfecli.tests.data")) as d:
        json_file = str(d / "transformation.json")

    return json_file


@pytest.mark.parametrize("extra_args", [{}, {"-d": "foo_dir", "-o": "foo.json"}])
def test_quickrun(extra_args, json_file):
    extras = sum([list(kv) for kv in extra_args.items()], [])

    runner = CliRunner()
    with runner.isolated_filesystem():
        # figure out what cached json should be
        trans = Transformation.from_json(json_file)
        work_dir = extra_args.get("-d", ".")
        outfile = pathlib.Path(extra_args.get("-o", f"{trans.key}_results.json"))
        hashed_key = _hash_quickrun_inputs(outfile, trans)

        # output json shouldn't be created before quickrun is executed
        assert not pathlib.Path(outfile).exists()
        result = runner.invoke(quickrun, [json_file] + extras)

        assert_click_success(result)
        assert "Here is the result" in result.output

        # cache should be deleted when job is complete
        assert not pathlib.Path(work_dir, "quickrun_cache", f"dag-cache-{hashed_key}.json").exists()

        # output json should exist with data when job is complete
        assert pathlib.Path(outfile).exists()
        with open(outfile, mode="r") as outf:
            dct = json.load(outf, cls=JSON_HANDLER.decoder)

        assert set(dct) == {"estimate", "uncertainty", "protocol_result", "unit_results"}
        # TODO: need a protocol that drops files to actually do this!
        # if directory := extra_args.get('-d'):
        #     dirpath = pathlib.Path(directory)
        #     assert dirpath.exists()
        #     assert dirpath.is_dir()
        #     assert len(list(dirpath.iterdir())) > 0


@pytest.mark.parametrize("extra_args", [{}, {"-d": "foo_dir", "-o": "foo.json"}])
def test_quickrun_interrupted(extra_args, json_file):
    """If quickrun starts but is unable to complete, the cached DAG should exist."""
    extras = sum([list(kv) for kv in extra_args.items()], [])

    runner = CliRunner()
    with runner.isolated_filesystem():
        # figure out what cached json should be
        trans = Transformation.from_json(json_file)
        work_dir = pathlib.Path(extra_args.get("-d", ".")).absolute()
        outfile = pathlib.Path(extra_args.get("-o", f"{trans.key}_results.json"))
        hashed_key = _hash_quickrun_inputs(outfile, trans)

        with mock.patch("gufe.protocols.protocoldag.execute_DAG", side_effect=RuntimeError):
            result = runner.invoke(quickrun, [json_file] + extras)

        assert "Here is the result" not in result.output
        assert pathlib.Path(work_dir, "quickrun_cache", f"dag-cache-{hashed_key}.json").exists()


def test_quickrun_output_file_exists(json_file):
    """Fail if the output file already exists."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        pathlib.Path("foo.json").touch()
        result = runner.invoke(quickrun, [json_file, "-o", "foo.json"])
        assert result.exit_code == 2  # usage error
        assert "is a file." in result.output


def test_quickrun_output_file_in_nonexistent_directory(json_file):
    """Should create the parent directory for output file if it doesn't exist."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        outfile = pathlib.Path("not_dir/foo.json")
        result = runner.invoke(quickrun, [json_file, "-o", outfile])
        assert_click_success(result)
        assert outfile.parent.is_dir()


def test_quickrun_dir_created_at_runtime(json_file):
    """It should be valid to have a new directory created by the -d flag."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        outdir = "not_dir"
        outfile = outdir + "foo.json"
        result = runner.invoke(quickrun, [json_file, "-d", outdir, "-o", outfile])
        assert_click_success(result)


def test_quickrun_unit_error():
    with resources.as_file(resources.files("openfecli.tests.data")) as d:
        json_file = str(d / "bad_transformation.json")

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(quickrun, [json_file, "-o", "foo.json"])
        assert result.exit_code == 1
        assert pathlib.Path("foo.json").exists()
        # TODO: I'm still not happy with this... failure result does not see
        # to be stored in JSON
        # not sure whether that means we should always be storing all
        # protocol dag results maybe?


def test_quickrun_existing_cache_error(json_file):
    """In the default case where resume=False, if the cache exists, quickrun should error out and not attempt to execute."""
    trans = Transformation.from_json(json_file)
    dag = trans.create()

    runner = CliRunner()
    with runner.isolated_filesystem():
        outfile = pathlib.Path(f"{trans.key}_results.json")
        hashed_key = _hash_quickrun_inputs(outfile, trans)
        pathlib.Path("quickrun_cache").mkdir()
        dag.to_json(pathlib.Path("quickrun_cache", f"dag-cache-{hashed_key}.json"))
        result = runner.invoke(quickrun, [json_file])
        assert result.exit_code == 1
        assert "Attempting to resume" not in result.output
        assert "Transformation has been started but is incomplete." in result.stderr


def test_quickrun_resume_from_cache(json_file):
    trans = Transformation.from_json(json_file)
    dag = trans.create()

    runner = CliRunner()
    with runner.isolated_filesystem():
        outfile = pathlib.Path(f"{trans.key}_results.json")
        hashed_key = _hash_quickrun_inputs(outfile, trans)
        pathlib.Path("quickrun_cache").mkdir()
        dag_cache = pathlib.Path("quickrun_cache", f"dag-cache-{hashed_key}.json")
        dag.to_json(dag_cache)
        result = runner.invoke(quickrun, [json_file, "--resume"])

        assert_click_success(result)
        assert f"resume execution using '{dag_cache.absolute()}" in result.output
        assert "Success" in result.output


def test_quickrun_resume_invalid_cache(json_file):
    """Fail if the output file doesn't load properly."""
    trans = Transformation.from_json(json_file)

    runner = CliRunner()
    with runner.isolated_filesystem():
        outfile = pathlib.Path(f"{trans.key}_results.json")
        hashed_key = _hash_quickrun_inputs(outfile, trans)
        pathlib.Path("quickrun_cache").mkdir()
        dag_cache = pathlib.Path("quickrun_cache", f"dag-cache-{hashed_key}.json")
        dag_cache.touch()
        result = runner.invoke(quickrun, [json_file, "--resume"])

        assert result.exit_code == 1
        assert f"resume execution using '{dag_cache.absolute()}" in result.output
        assert "Recovery failed" in result.stderr


def test_quickrun_resume_missing_cache(json_file):
    """If --resume is passed but there's no cache, just echo a message and start from scratch."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        # determine what the cache to be looked for should be named
        trans = Transformation.from_json(json_file)
        outfile = pathlib.Path(f"{trans.key}_results.json")
        hashed_key = _hash_quickrun_inputs(outfile, trans)
        dag_cache = pathlib.Path("quickrun_cache", f"dag-cache-{hashed_key}.json")

        result = runner.invoke(quickrun, [json_file, "--resume"])
        assert_click_success(result)
        assert (
            f"openfe quickrun was run with --resume, but no cached results found at {dag_cache.absolute()}"
            in result.output
        )
