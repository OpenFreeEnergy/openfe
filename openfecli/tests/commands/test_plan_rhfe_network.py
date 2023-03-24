from unittest import mock

import pytest
import importlib
import os
import click
from click.testing import CliRunner

from openfecli.commands.plan_rhfe_network import (
    plan_rhfe_network, plan_rhfe_network_main
)


@pytest.fixture
def mol_dir_args():
    with importlib.resources.path("openfe.tests.data.openmm_rbfe","__init__.py") as file_path:
        ofe_dir_path = os.path.dirname(file_path)
    
    return ["--mol-dir", ofe_dir_path]


@pytest.fixture
def mapper_args():
    return ["--mapper", "LomapAtomMapper"]


def print_test_with_file(mapper, mapping_scorer, ligand_network_planner, small_molecules, solvent):
    print(mapper)
    print(mapping_scorer)
    print(ligand_network_planner)
    print(small_molecules)
    print(solvent)

def test_plan_rbfe_network_main():
    import os, glob
    from gufe import SmallMoleculeComponent, SolventComponent
    from openfe.setup import LomapAtomMapper, lomap_scorers, ligand_network_planning
    
    with importlib.resources.path("openfe.tests.data.openmm_rbfe","__init__.py") as file_path:
        smallM_components = [SmallMoleculeComponent.from_sdf_file(f) for f in glob.glob(os.path.dirname(file_path)+"/*.sdf")]
    solvent_component = SolventComponent()
    alchemical_network = plan_rhfe_network_main(mapper=LomapAtomMapper(), mapping_scorer=lomap_scorers.default_lomap_score,
                                                ligand_network_planner=ligand_network_planning.generate_minimal_spanning_network,
                                                small_molecules=smallM_components, solvent=solvent_component)
    print(alchemical_network)

def test_plan_rhfe_network(mol_dir_args, mapper_args):
    """
        smoke test
    """
    args = mol_dir_args + mapper_args
    expected_output = (f"{mol_dir_args[1]}\n"
                       f"{mapper_args[1]}\n")
    
    patch_base = "openfecli.commands.plan_rhfe_network."
    args += ["-o", "tmp_network"]

    patch_loc = patch_base + "plan_rhfe_network_main"
    patch_func = print_test_with_file

    runner = CliRunner()
    with mock.patch(patch_loc, patch_func):
        with runner.isolated_filesystem():
            result = runner.invoke(plan_rhfe_network, args)
            print(result.output)
            print(result)
            assert result.exit_code == 0
