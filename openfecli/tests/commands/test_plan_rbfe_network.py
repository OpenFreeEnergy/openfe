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
    )
    print(alchemical_network)


def test_plan_rbfe_network(mol_dir_args, protein_args):
    """
    smoke test
    """
    args = mol_dir_args + protein_args
    expected_output = [
        "RBFE-NETWORK PLANNER",
        "Small Molecules: SmallMoleculeComponent(name=ligand_23) SmallMoleculeComponent(name=ligand_55)",
        "Protein: ProteinComponent(name=)",
        "Solvent: SolventComponent(name=O, Na+, Cl-)",
        "- tmp_network.json",
        "- easy_rbfe_ligand_23_complex_ligand_55_complex.json",
        "- easy_rbfe_ligand_23_solvent_ligand_55_solvent.json",
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
            assert all(
                [
                    expected_line in result.output
                    for expected_line in expected_output
                ]
            )
