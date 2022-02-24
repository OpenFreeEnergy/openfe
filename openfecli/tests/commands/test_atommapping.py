from unittest import mock

import pytest
import click
from click.testing import CliRunner

from openfe.setup import AtomMapping, LomapAtomMapper

from openfecli.parameters import MOL
from openfecli.commands.atommapping import (
    atommapping, generate_mapping, atommapping_print_dict_main
)


@pytest.fixture
def mol1_args():
    return ["--mol", "Cc1ccccc1"]


@pytest.fixture
def mol2_args():
    return ["--mol", "CC1CCCCC1"]


@pytest.fixture
def mapper_args():
    return ["--mapper", "LomapAtomMapper"]


@pytest.fixture
def mols(mol1_args, mol2_args):
    return MOL.get(mol1_args[1]), MOL.get(mol2_args[1])


def print_test(mapper, mol1, mol2):
    print(mol1.smiles)
    print(mol2.smiles)
    print(mapper.__class__.__name__)


@mock.patch("openfecli.commands.atommapping.atommapping_print_dict_main",
            print_test)
def test_atommapping(mol1_args, mol2_args, mapper_args):
    # Patch out the main function with a simple function to output
    # information about the objects we pass to the main; test the output of
    # that using tools from click. This tests the creation of objects from
    # user input on the command line.
    runner = CliRunner()
    with runner.isolated_filesystem():
        args = mol1_args + mol2_args + mapper_args
        result = runner.invoke(atommapping, args)
        assert result.exit_code == 0
        assert result.output == (f"{mol1_args[1]}\n{mol2_args[1]}\n"
                                 f"{mapper_args[1]}\n")


def test_atommapping_bad_number_of_mols(mol1_args, mapper_args):
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(atommapping, mol1_args + mapper_args)
        assert result.exit_code == click.BadParameter.exit_code
        assert "exactly twice" in result.output


def test_atommapping_missing_mapper(mol1_args, mol2_args):
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(atommapping, mol1_args + mol2_args)
        assert result.exit_code == click.BadParameter.exit_code
        assert "Missing option '--mapper'" in result.output


@pytest.mark.parametrize('n_mappings', [0, 1, 2])
def test_generate_mapping(n_mappings, mols):
    mol1, mol2, = mols
    mappings = [
        AtomMapping(mol1, mol2, {i: i for i in range(7)}),
        AtomMapping(mol1, mol2, {i: (i + 1) % 7 for i in range(7)}),
    ]
    mapper = mock.Mock(
        suggest_mappings=mock.Mock(return_value=mappings[:n_mappings])
    )

    if n_mappings == 1:
        assert generate_mapping(mapper, mol1, mol2) == mappings[0]
    else:
        with pytest.raises(click.UsageError, match="exactly 1 mapping"):
            generate_mapping(mapper, mol1, mol2)


def test_atommapping_print_dict_main(capsys, mols):
    mol1, mol2 = mols
    mapper = LomapAtomMapper
    mapping = AtomMapping(mol1, mol2, {i: i for i in range(7)})
    with mock.patch('openfecli.commands.atommapping.generate_mapping',
                    mock.Mock(return_value=mapping)):
        atommapping_print_dict_main(mapper, mol1, mol2)
        captured = capsys.readouterr()
        assert captured.out == str(mapping.mol1_to_mol2) + "\n"
