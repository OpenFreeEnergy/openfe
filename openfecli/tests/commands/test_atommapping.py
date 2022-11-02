from unittest import mock

import pytest
import click
from click.testing import CliRunner

from openfe.setup import LigandAtomMapping, LomapAtomMapper

from openfecli.parameters import MOL
from openfecli.commands.atommapping import (
    atommapping, generate_mapping, atommapping_print_dict_main,
    atommapping_visualize_main
)


@pytest.fixture
def molA_args():
    return ["--mol", "Cc1ccccc1"]


@pytest.fixture
def molB_args():
    return ["--mol", "CC1CCCCC1"]


@pytest.fixture
def mapper_args():
    return ["--mapper", "LomapAtomMapper"]


@pytest.fixture
def mols(molA_args, molB_args):
    return MOL.get(molA_args[1]), MOL.get(molB_args[1])


def print_test(mapper, molA, molB):
    print(molA.smiles)
    print(molB.smiles)
    print(mapper.__class__.__name__)


def print_test_with_file(mapper, molA, molB, file, ext):
    print_test(mapper, molA, molB)
    print(file.name)
    print(ext)


@pytest.mark.parametrize('with_file', [True, False])
def test_atommapping(molA_args, molB_args, mapper_args, with_file):
    # Patch out the main function with a simple function to output
    # information about the objects we pass to the main; test the output of
    # that using tools from click. This tests the creation of objects from
    # user input on the command line.
    args = molA_args + molB_args + mapper_args
    expected_output = (f"{molA_args[1]}\n{molB_args[1]}\n"
                       f"{mapper_args[1]}\n")
    patch_base = "openfecli.commands.atommapping."
    if with_file:
        args += ["-o", "myfile.png"]
        expected_output += "myfile.png\npng\n"
        patch_loc = patch_base + "atommapping_visualize_main"
        patch_func = print_test_with_file
    else:
        patch_loc = patch_base + "atommapping_print_dict_main"
        patch_func = print_test

    runner = CliRunner()
    with mock.patch(patch_loc, patch_func):
        with runner.isolated_filesystem():
            result = runner.invoke(atommapping, args)
            assert result.exit_code == 0
            assert result.output == expected_output


def test_atommapping_bad_number_of_mols(molA_args, mapper_args):
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(atommapping, molA_args + mapper_args)
        assert result.exit_code == click.BadParameter.exit_code
        assert "exactly twice" in result.output


def test_atommapping_missing_mapper(molA_args, molB_args):
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(atommapping, molA_args + molB_args)
        assert result.exit_code == click.BadParameter.exit_code
        assert "Missing option '--mapper'" in result.output


@pytest.mark.parametrize('n_mappings', [0, 1, 2])
def test_generate_mapping(n_mappings, mols):
    molA, molB, = mols
    mappings = [
        LigandAtomMapping(molA, molB, {i: i for i in range(7)}),
        LigandAtomMapping(molA, molB, {i: (i + 1) % 7 for i in range(7)}),
    ]
    mapper = mock.Mock(
        suggest_mappings=mock.Mock(return_value=mappings[:n_mappings])
    )

    if n_mappings == 1:
        assert generate_mapping(mapper, molA, molB) == mappings[0]
    else:
        with pytest.raises(click.UsageError, match="exactly 1 mapping"):
            generate_mapping(mapper, molA, molB)


def test_atommapping_print_dict_main(capsys, mols):
    molA, molB = mols
    mapper = LomapAtomMapper
    mapping = LigandAtomMapping(molA, molB, {i: i for i in range(7)})
    with mock.patch('openfecli.commands.atommapping.generate_mapping',
                    mock.Mock(return_value=mapping)):
        atommapping_print_dict_main(mapper, molA, molB)
        captured = capsys.readouterr()
        assert captured.out == str(mapping.componentA_to_componentB) + "\n"


def test_atommapping_visualize_main(mols, tmpdir):
    molA, molB = mols
    mapper = LomapAtomMapper
    pytest.skip()  # TODO: probably with a smoke test


def test_atommapping_visualize_main_bad_extension(mols, tmpdir):
    molA, molB = mols
    mapper = LomapAtomMapper
    mapping = LigandAtomMapping(molA, molB, {i: i for i in range(7)})
    with mock.patch('openfecli.commands.atommapping.generate_mapping',
                    mock.Mock(return_value=mapping)):
        with open(tmpdir / "foo.bar", mode='w') as f:
            with pytest.raises(click.BadParameter,
                               match="Unknown file format"):
                atommapping_visualize_main(mapper, molA, molB, f, "bar")
