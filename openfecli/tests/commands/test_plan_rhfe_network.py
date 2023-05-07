from unittest import mock

import pytest
import importlib
import os
from click.testing import CliRunner

from openfecli.commands.plan_rhfe_network import (
    plan_rhfe_network,
    plan_rhfe_network_main,
)


@pytest.fixture
def mol_dir_args():
    with importlib.resources.path(
        "openfe.tests.data.openmm_rfe", "__init__.py"
    ) as file_path:
        ofe_dir_path = os.path.dirname(file_path)

    return ["--molecules", ofe_dir_path]


@pytest.fixture
def mapper_args():
    return ["--mapper", "LomapAtomMapper"]


def print_test_with_file(
    mapper, mapping_scorer, ligand_network_planner, small_molecules, solvent
):
    print(mapper)
    print(mapping_scorer)
    print(ligand_network_planner)
    print(small_molecules)
    print(solvent)


@pytest.mark.parametrize('output_ligand_network', [True, False])
def test_plan_rhfe_network_main(output_ligand_network):
    import os, glob
    from gufe import SmallMoleculeComponent, SolventComponent
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

    solvent_component = SolventComponent()
    alchemical_network, ligand_network = plan_rhfe_network_main(
        mapper=LomapAtomMapper(),
        mapping_scorer=lomap_scorers.default_lomap_score,
        ligand_network_planner=ligand_network_planning.generate_minimal_spanning_network,
        small_molecules=smallM_components,
        solvent=solvent_component,
        output_ligand_network=output_ligand_network,
    )

    assert alchemical_network
    assert ligand_network


def test_plan_rhfe_network(mol_dir_args, mapper_args):
    """
    smoke test
    """
    args = mol_dir_args + mapper_args
    expected_output = [
        "RHFE-NETWORK PLANNER",
        "Small Molecules: SmallMoleculeComponent(name=ligand_23) SmallMoleculeComponent(name=ligand_55)",
        "Solvent: SolventComponent(name=O, Na+, Cl-)",
        "- tmp_network.json",
        "- easy_rhfe_ligand_23_vacuum_ligand_55_vacuum.json",
        "- easy_rhfe_ligand_23_solvent_ligand_55_solvent.json",
    ]

    patch_base = (
        "openfecli.commands.plan_rhfe_network."
    )
    args += ["-o", "tmp_network"]

    patch_loc = patch_base + "plan_rhfe_network"
    patch_func = print_test_with_file

    runner = CliRunner()
    with mock.patch(patch_loc, patch_func):
        with runner.isolated_filesystem():
            result = runner.invoke(plan_rhfe_network, args)
            print(result.output)
            assert result.exit_code == 0
            assert all(
                [
                    expected_line in result.output
                    for expected_line in expected_output
                ]
            )
