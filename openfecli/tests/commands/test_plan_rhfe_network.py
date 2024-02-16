from unittest import mock

import pytest
from importlib import resources
import os
import shutil
from click.testing import CliRunner

from openfecli.commands.plan_rhfe_network import (
    plan_rhfe_network,
    plan_rhfe_network_main,
)


@pytest.fixture(scope="session")
def mol_dir_args(tmpdir_factory):
    ofe_dir_path = tmpdir_factory.mktemp("moldir")

    with resources.files("openfe.tests.data.openmm_rfe") as d:
        for f in ["ligand_23.sdf", "ligand_55.sdf"]:
            shutil.copyfile(d / f, ofe_dir_path / f)

    return ["--molecules", ofe_dir_path]


def print_test_with_file(mapping_scorer, ligand_network_planner, small_molecules, solvent):
    print(mapping_scorer)
    print(ligand_network_planner)
    print(small_molecules)
    print(solvent)


def test_plan_rhfe_network_main():
    import os
    from gufe import SmallMoleculeComponent, SolventComponent
    from openfe.setup import (
        LomapAtomMapper,
        lomap_scorers,
        ligand_network_planning,
    )

    with resources.files("openfe.tests.data.openmm_rfe") as d:
        smallM_components = [SmallMoleculeComponent.from_sdf_file(d / f) for f in ["ligand_23.sdf", "ligand_55.sdf"]]

    solvent_component = SolventComponent()
    alchemical_network, ligand_network = plan_rhfe_network_main(
        mapper=[LomapAtomMapper()],
        mapping_scorer=lomap_scorers.default_lomap_score,
        ligand_network_planner=ligand_network_planning.generate_minimal_spanning_network,
        small_molecules=smallM_components,
        solvent=solvent_component,
    )

    assert alchemical_network
    assert ligand_network


def test_plan_rhfe_network(mol_dir_args):
    """
    smoke test
    """
    args = mol_dir_args
    expected_output_always = [
        "RHFE-NETWORK PLANNER",
        "Solvent: SolventComponent(name=O, Na+, Cl-)",
        "- tmp_network.json",
    ]
    # we can get these in either order: 22 then 55 or 55 then 22
    expected_output_1 = [
        "Small Molecules: SmallMoleculeComponent(name=ligand_23) SmallMoleculeComponent(name=ligand_55)",
        "- easy_rhfe_ligand_23_vacuum_ligand_55_vacuum.json",
        "- easy_rhfe_ligand_23_solvent_ligand_55_solvent.json",
    ]
    expected_output_2 = [
        "Small Molecules: SmallMoleculeComponent(name=ligand_55) SmallMoleculeComponent(name=ligand_23)",
        "- easy_rhfe_ligand_55_vacuum_ligand_23_vacuum.json",
        "- easy_rhfe_ligand_55_solvent_ligand_23_solvent.json",
    ]

    patch_base = "openfecli.commands.plan_rhfe_network."
    args += ["-o", "tmp_network"]

    patch_loc = patch_base + "plan_rhfe_network"
    patch_func = print_test_with_file

    runner = CliRunner()
    with mock.patch(patch_loc, patch_func):
        with runner.isolated_filesystem():
            result = runner.invoke(plan_rhfe_network, args)
            print(result.output)
            assert result.exit_code == 0
            for line in expected_output_always:
                assert line in result.output

            for l1, l2 in zip(expected_output_1, expected_output_2):
                assert l1 in result.output or l2 in result.output


@pytest.fixture
def custom_yaml_settings():
    return """\
network:
  method: generate_minimal_redundant_network
  settings:
    mst_num: 2

mapper:
  method: LomapAtomMapper
  settings:
    time: 45
    element_change: True
"""


def test_custom_yaml_plan_rhfe_smoke_test(custom_yaml_settings, mol_dir_args, tmpdir):
    settings_path = tmpdir / "settings.yaml"
    with open(settings_path, "w") as f:
        f.write(custom_yaml_settings)

    assert settings_path.exists()

    args = mol_dir_args + ["-s", settings_path]

    runner = CliRunner()

    with runner.isolated_filesystem():
        result = runner.invoke(plan_rhfe_network, args)

        assert result.exit_code == 0
