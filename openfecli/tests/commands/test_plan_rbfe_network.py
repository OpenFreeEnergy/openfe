from unittest import mock

import pytest
import importlib
import os
from click.testing import CliRunner

from openfecli.commands.plan_rbfe_network import (
    plan_rbfe_network,
    plan_rbfe_network_main,
)


@pytest.fixture
def mol_dir_args():
    with importlib.resources.path(
        "openfe.tests.data.openmm_rfe", "__init__.py"
    ) as file_path:
        ofe_dir_path = os.path.dirname(file_path)

    return ["--molecules", ofe_dir_path]


@pytest.fixture
def protein_args():
    with importlib.resources.path(
        "openfe.tests.data", "181l_only.pdb"
    ) as file_path:
        return ["--protein", file_path]


def print_test_with_file(
    mapping_scorer,
    ligand_network_planner,
    small_molecules,
    solvent,
    protein,
):
    print(mapping_scorer)
    print(ligand_network_planner)
    print(small_molecules)
    print(solvent)
    print(protein)


def test_plan_rbfe_network_main():
    import os, glob
    from gufe import (
        ProteinComponent,
        SmallMoleculeComponent,
        SolventComponent,
    )
    from openfe.setup import (
        LomapAtomMapper,
        lomap_scorers,
        ligand_network_planning,
    )

    with importlib.resources.path(
        "openfe.tests.data.openmm_rfe", "__init__.py"
    ) as file_path:
        smallM_components = [
            SmallMoleculeComponent.from_sdf_file(f)
            for f in glob.glob(os.path.dirname(file_path) + "/*.sdf")
        ]
    with importlib.resources.path(
        "openfe.tests.data", "181l_only.pdb"
    ) as file_path:
        protein_compontent = ProteinComponent.from_pdb_file(
            os.path.dirname(file_path) + "/181l_only.pdb"
        )

    solvent_component = SolventComponent()
    alchemical_network, ligand_network = plan_rbfe_network_main(
        mapper=LomapAtomMapper(),
        mapping_scorer=lomap_scorers.default_lomap_score,
        ligand_network_planner=ligand_network_planning.generate_minimal_spanning_network,
        small_molecules=smallM_components,
        solvent=solvent_component,
        protein=protein_compontent,
        cofactors=[],
    )
    print(alchemical_network)


def test_plan_rbfe_network(mol_dir_args, protein_args):
    """
    smoke test
    """
    args = mol_dir_args + protein_args
    expected_output_always = [
        "RBFE-NETWORK PLANNER",
        "Protein: ProteinComponent(name=)",
        "Solvent: SolventComponent(name=O, Na+, Cl-)",
        "- tmp_network.json",
    ]
    # we can get these in either order: 22 first or 55 first
    expected_output_1 = [
        "Small Molecules: SmallMoleculeComponent(name=ligand_23) SmallMoleculeComponent(name=ligand_55)",
        "- easy_rbfe_ligand_23_complex_ligand_55_complex.json",
        "- easy_rbfe_ligand_23_solvent_ligand_55_solvent.json",
    ]
    expected_output_2 = [
        "Small Molecules: SmallMoleculeComponent(name=ligand_55) SmallMoleculeComponent(name=ligand_23)",
        "- easy_rbfe_ligand_55_complex_ligand_23_complex.json",
        "- easy_rbfe_ligand_55_solvent_ligand_23_solvent.json",
    ]

    patch_base = (
        "openfecli.commands.plan_rbfe_network."
    )
    args += ["-o", "tmp_network"]

    patch_loc = patch_base + "plan_rbfe_network"
    patch_func = print_test_with_file

    runner = CliRunner()
    with mock.patch(patch_loc, patch_func):
        with runner.isolated_filesystem():
            result = runner.invoke(plan_rbfe_network, args)
            print(result.output)
            assert result.exit_code == 0
            for line in expected_output_always:
                assert line in result.output

            for l1, l2 in zip(expected_output_1, expected_output_2):
                assert l1 in result.output or l2 in result.output


@pytest.fixture
def eg5_files():
    with importlib.resources.files('openfe.tests.data.eg5') as p:
        pdb_path = str(p.joinpath('eg5_protein.pdb'))
        lig_path = str(p.joinpath('eg5_ligands.sdf'))
        cof_path = str(p.joinpath('eg5_cofactor.sdf'))

        yield pdb_path, lig_path, cof_path


def test_plan_rbfe_network_cofactors(eg5_files):

    runner = CliRunner()

    args = [
        '-p', eg5_files[0],
        '-M', eg5_files[1],
        '-C', eg5_files[2],
    ]

    with runner.isolated_filesystem():
        result = runner.invoke(plan_rbfe_network, args)

        assert result.exit_code == 0


def test_plan_rbfe_radial(mol_dir_args, protein_args):
    runner = CliRunner()

    args = mol_dir_args + protein_args
    args += ['--radial']

    expected_1 = {
        '- easy_rbfe_ligand_55_complex_ligand_23_complex.json',
        '- easy_rbfe_ligand_55_solvent_ligand_23_solvent.json',
    }
    expected_2 = {
        '- easy_rbfe_ligand_23_complex_ligand_55_complex.json',
        '- easy_rbfe_ligand_23_solvent_ligand_55_solvent.json',
    }

    with mock.patch("openfecli.commands.plan_rbfe_network.plan_rbfe_network",
                    print_test_with_file):
        with runner.isolated_filesystem():
            result = runner.invoke(plan_rbfe_network, args)

            assert result.exit_code == 0

            output_lines = {l.strip() for l in result.output.split('\n')
                            if l.strip().startswith('- easy')}

            assert output_lines == expected_1 or output_lines == expected_2


def test_plan_rbfe_radial_withhub(mol_dir_args, protein_args):
    runner = CliRunner()

    args = mol_dir_args + protein_args
    args += ['--radial', '--radial_hub', 'ligand_55']

    expected_1 = {
        '- easy_rbfe_ligand_55_complex_ligand_23_complex.json',
        '- easy_rbfe_ligand_55_solvent_ligand_23_solvent.json',
    }
    expected_2 = {
        '- easy_rbfe_ligand_23_complex_ligand_55_complex.json',
        '- easy_rbfe_ligand_23_solvent_ligand_55_solvent.json',
    }

    with mock.patch("openfecli.commands.plan_rbfe_network.plan_rbfe_network",
                    print_test_with_file):
        with runner.isolated_filesystem():
            result = runner.invoke(plan_rbfe_network, args)

            assert result.exit_code == 0

            output_lines = {l.strip() for l in result.output.split('\n')
                            if l.strip().startswith('- easy')}

            assert output_lines == expected_1 or output_lines == expected_2


def test_plan_rbfe_radial_badname(mol_dir_args, protein_args):
    runner = CliRunner()

    args = mol_dir_args + protein_args
    args += ['--radial', '--radial_hub', 'bobbie']

    with mock.patch("openfecli.commands.plan_rbfe_network.plan_rbfe_network",
                    print_test_with_file):
        with runner.isolated_filesystem():
            result = runner.invoke(plan_rbfe_network, args)

            assert result.exit_code == 1
            assert 'ligand name bobbie not found' in result.exception.args[0]
