from unittest import mock

import pytest
from importlib import resources
import shutil
import tarfile
from click.testing import CliRunner
from ..utils import assert_click_success

from openfe.protocols.openmm_utils.charge_generation import HAS_OPENEYE
from openfecli.commands.plan_rbfe_network import (
    plan_rbfe_network,
    plan_rbfe_network_main,
)

from gufe import AlchemicalNetwork, SmallMoleculeComponent
import json
import numpy as np
from openff.utilities import skip_if_missing

@pytest.fixture(scope='session')
def mol_dir_args(tmpdir_factory):
    ofe_dir_path = tmpdir_factory.mktemp('moldir')

    with resources.files('openfe.tests.data.openmm_rfe') as d:
        for f in ['ligand_23.sdf', 'ligand_55.sdf']:
            shutil.copyfile(d / f, ofe_dir_path / f)

    return ["--molecules", ofe_dir_path]

@pytest.fixture(scope="session")
def dummy_charge_dir_args(tmpdir_factory):
    ofe_dir_path = tmpdir_factory.mktemp('charge_moldir')

    with resources.files('openfe.tests.data.openmm_rfe') as d:
        for f in ['dummy_charge_ligand_23.sdf', 'dummy_charge_ligand_55.sdf']:
            shutil.copyfile(d / f, ofe_dir_path / f)

    return ["--molecules", ofe_dir_path]

@pytest.fixture(scope='session')
def cdk8_files(tmpdir_factory):
    cdk8_tmpdir = tmpdir_factory.mktemp("cdk8_data")

    with tarfile.open("../../../openfe/tests/data/cdk8.tar.gz") as f:
        f.extractall(cdk8_tmpdir, filter='tar')

    protein = str(cdk8_tmpdir/'cdk8'/'cdk8_protein.pdb')
    ligand = str(cdk8_tmpdir/'cdk8'/'cdk8_ligands.sdf')

    yield (protein, ligand)

@pytest.fixture
def protein_args():
    with resources.files("openfe.tests.data") as d:
        return ["--protein", str(d / "181l_only.pdb")]


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

def validate_charges(smc):
    """
    Validate that the SmallMoleculeComponent has partial charges assigned.
    """
    off_mol = smc.to_openff()
    assert off_mol.partial_charges is not None
    assert len(off_mol.partial_charges) == off_mol.n_atoms


@skip_if_missing("openff.nagl")
@skip_if_missing("openff.nagl_models")
def test_plan_rbfe_network_main():
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
    from openfe.protocols.openmm_utils.omm_settings import (
        OpenFFPartialChargeSettings
    )

    with resources.files("openfe.tests.data.openmm_rfe") as d:
        smallM_components = [
            SmallMoleculeComponent.from_sdf_file(d / f)
            for f in ['ligand_23.sdf', 'ligand_55.sdf']
        ]
    with resources.files("openfe.tests.data") as d:
        protein_component = ProteinComponent.from_pdb_file(
            str(d / "181l_only.pdb")
        )

    solvent_component = SolventComponent()
    alchemical_network, ligand_network = plan_rbfe_network_main(
        mapper=[LomapAtomMapper()],
        mapping_scorer=lomap_scorers.default_lomap_score,
        ligand_network_planner=ligand_network_planning.generate_minimal_spanning_network,
        small_molecules=smallM_components,
        solvent=solvent_component,
        protein=protein_component,
        cofactors=[],
        n_protocol_repeats=3,
        # use nagl to keep testing fast
        partial_charge_settings=OpenFFPartialChargeSettings(
            partial_charge_method="nagl",
            nagl_model="openff-gnn-am1bcc-0.1.0-rc.3.pt"
        ),
        processors=1,
        overwrite_charges=False
    )
    # check the ligands have charges assigned
    for node in alchemical_network.nodes:
        validate_charges(node.components["ligand"])


@pytest.fixture
def yaml_nagl_settings():
    return """\
partial_charge:
  method: nagl
  settings:
    nagl_model: openff-gnn-am1bcc-0.1.0-rc.3.pt
"""

@skip_if_missing("openff.nagl")
@skip_if_missing("openff.nagl_models")
def test_plan_rbfe_network(mol_dir_args, protein_args, tmpdir, yaml_nagl_settings):
    """
    smoke test
    """
    # use nagl charges for CI speed!
    settings_path = tmpdir / "settings.yaml"
    with open(settings_path, "w") as f:
        f.write(yaml_nagl_settings)

    args = mol_dir_args + protein_args
    expected_output_always = [
        "RBFE-NETWORK PLANNER",
        "Protein: ProteinComponent(name=)",
        "Solvent: SolventComponent(name=O, Na+, Cl-)",
        "- tmp_network.json",
        # make sure the partial charge settings are picked up
        "Partial Charge Generation: nagl",
        "assigning ligand partial charges -- this may be slow"
    ]
    # we can get these in either order: 22 first or 55 first
    expected_output_1 = [
        "Small Molecules: SmallMoleculeComponent(name=ligand_23) SmallMoleculeComponent(name=ligand_55)",
        "- rbfe_ligand_23_complex_ligand_55_complex.json",
        "- rbfe_ligand_23_solvent_ligand_55_solvent.json",
    ]
    expected_output_2 = [
        "Small Molecules: SmallMoleculeComponent(name=ligand_55) SmallMoleculeComponent(name=ligand_23)",
        "- rbfe_ligand_55_complex_ligand_23_complex.json",
        "- rbfe_ligand_55_solvent_ligand_23_solvent.json",
    ]

    patch_base = (
        "openfecli.commands.plan_rbfe_network."
    )
    args += ["-o", "tmp_network"]
    args += ["-s", settings_path]

    patch_loc = patch_base + "plan_rbfe_network"
    patch_func = print_test_with_file

    runner = CliRunner()
    with mock.patch(patch_loc, patch_func):
        with runner.isolated_filesystem():
            result = runner.invoke(plan_rbfe_network, args)
            assert result.exit_code == 0
            for line in expected_output_always:
                assert line in result.output

            for l1, l2 in zip(expected_output_1, expected_output_2):
                assert l1 in result.output or l2 in result.output

@pytest.mark.parametrize(['input_n_repeat', 'expected_n_repeat'], [([], 3), (["--n-protocol-repeats", "1"], 1)])
def test_plan_rbfe_network_n_repeats(mol_dir_args, protein_args, input_n_repeat, expected_n_repeat):
    runner = CliRunner()

    args = mol_dir_args + protein_args + input_n_repeat

    with runner.isolated_filesystem():
        result = runner.invoke(plan_rbfe_network, args)
        assert_click_success(result)

        # make sure the number of repeats is correct
        network = AlchemicalNetwork.from_json("alchemicalNetwork/alchemicalNetwork.json")
        for edge in network.edges:
            assert edge.protocol.settings.protocol_repeats == expected_n_repeat

@pytest.mark.parametrize("overwrite", [
    pytest.param(True, id="Overwrite"),
    pytest.param(False, id="No overwrite")
])
@skip_if_missing("openff.nagl")
@skip_if_missing("openff.nagl_models")
def test_plan_rbfe_network_charge_overwrite(dummy_charge_dir_args, protein_args, tmpdir, yaml_nagl_settings, overwrite):
    # make sure the dummy charges are overwritten when requested

    # use nagl charges for CI speed!
    settings_path = tmpdir / "settings.yaml"
    with open(settings_path, "w") as f:
        f.write(yaml_nagl_settings)

    args = dummy_charge_dir_args + protein_args + ["-s", settings_path]

    # get the input charges for the molecules to check they have been overwritten
    charges_by_name = {}
    for f in ['dummy_charge_ligand_23.sdf', 'dummy_charge_ligand_55.sdf']:
        smc = SmallMoleculeComponent.from_sdf_file(dummy_charge_dir_args[1] / f)
        charges_by_name[smc.name] = smc.to_openff().partial_charges.m

    if overwrite:
        args.append("--overwrite-charges")

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(plan_rbfe_network, args)

        assert result.exit_code == 0
        if overwrite:
            assert "Overwriting partial charges" in result.output

        network = AlchemicalNetwork.from_json("alchemicalNetwork/alchemicalNetwork.json")
        # make sure the ligands don't have dummy charges
        for node in network.nodes:
            off_mol = node.components["ligand"].to_openff()

            if overwrite:
                assert not np.allclose(off_mol.partial_charges.m, charges_by_name[off_mol.name])
            else:
                assert np.allclose(off_mol.partial_charges.m, charges_by_name[off_mol.name])


@pytest.fixture
def eg5_files():
    with resources.files('openfe.tests.data.eg5') as p:
        pdb_path = str(p.joinpath('eg5_protein.pdb'))
        lig_path = str(p.joinpath('eg5_ligands.sdf'))
        cof_path = str(p.joinpath('eg5_cofactor.sdf'))

        yield pdb_path, lig_path, cof_path


@pytest.mark.xfail(HAS_OPENEYE, reason="openff-nagl#177")
@skip_if_missing("openff.nagl")
@skip_if_missing("openff.nagl_models")
def test_plan_rbfe_network_cofactors(eg5_files, tmpdir, yaml_nagl_settings):
    # use nagl charges for CI speed!
    settings_path = tmpdir / "settings.yaml"
    with open(settings_path, "w") as f:
        f.write(yaml_nagl_settings)

    runner = CliRunner()

    args = [
        '-p', eg5_files[0],
        '-M', eg5_files[1],
        '-C', eg5_files[2],
        '-s', settings_path
    ]

    with runner.isolated_filesystem():
        result = runner.invoke(plan_rbfe_network, args)

        assert result.exit_code == 0
        # check charges are assigned
        assert "Partial Charge Generation: nagl" in result.output
        assert "assigning ligand partial charges -- this may be slow" in result.output
        assert "assigning cofactor partial charges -- this may be slow"

        # make sure the cofactor is in the transformations
        network = AlchemicalNetwork.from_json("alchemicalNetwork/alchemicalNetwork.json")
        for edge in network.edges:
            if "protein" in edge.stateA.components:
                assert "cofactor1" in edge.stateA.components
                assert "cofactor1" in edge.stateB.components
            else:
                assert "cofactor1" not in edge.stateA.components
                assert "cofactor1" not in edge.stateB.components
        # make sure the ligands and cofactors have charges
        for node in network.nodes:
            validate_charges(node.components["ligand"])
            if "cofactor1" in node.components:
                validate_charges(node.components["cofactor1"])

def test_plan_rbfe_network_charge_changes(cdk8_files):
    """
    Make sure the protocol settings are changed and a warning is printed when we plan a network
    with a net charge change.
    """

    runner = CliRunner()

    args = [
        '-p', cdk8_files[0],
        '-M', cdk8_files[1],
    ]

    with runner.isolated_filesystem():
        with pytest.warns(UserWarning, match="Charge changing transformation between ligands lig_40 and lig_41"):
            result = runner.invoke(plan_rbfe_network, args)

            assert result.exit_code == 0
            # load the transformations and check the settings
            network = AlchemicalNetwork.from_json("alchemicalNetwork/alchemicalNetwork.json")
            for edge in network.edges:
                settings = edge.protocol.settings
                # check the charged transform
                if edge.stateA.components["ligand"].name == "lig_40" and edge.stateB.components["ligand"].name == "lig_41":
                    assert settings.alchemical_settings.explicit_charge_correction is True
                    assert settings.simulation_settings.production_length.m == 20.0
                    assert settings.simulation_settings.n_replicas == 22
                    assert settings.lambda_settings.lambda_windows == 22
                else:
                    assert settings.alchemical_settings.explicit_charge_correction is False
                    assert settings.simulation_settings.production_length.m == 5.0
                    assert settings.simulation_settings.n_replicas == 11
                    assert settings.lambda_settings.lambda_windows == 11


@pytest.fixture
def lomap_yaml_settings():
    return """\
network:
  method: generate_lomap_network
  settings:
    max_path_length: 6

mapper:
  method: LomapAtomMapper
  settings:
    time: 45
    element_change: True

partial_charge:
  method: nagl
  settings:
    nagl_model: openff-gnn-am1bcc-0.1.0-rc.3.pt
"""

@skip_if_missing("openff.nagl")
@skip_if_missing("openff.nagl_models")
def test_lomap_yaml_plan_rbfe_smoke_test(lomap_yaml_settings, cdk8_files, tmpdir):
    protein, ligand = cdk8_files
    settings_path = tmpdir / "settings.yaml"
    with open(settings_path, "w") as f:
        f.write(lomap_yaml_settings)

    assert settings_path.exists()

    args = [
        '-p', protein,
        '-M', ligand,
        '-s', settings_path
    ]

    runner = CliRunner()

    with runner.isolated_filesystem():
        result = runner.invoke(plan_rbfe_network, args)

        assert result.exit_code == 0


@pytest.fixture
def custom_yaml_radial():
    return """\
network:
  method: generate_radial_network
  settings:
    central_ligand: lig_CHEMBL1078774

mapper:
  method: LomapAtomMapper
  settings:
    time: 45
    element_change: True

partial_charge:
  method: nagl
  settings:
    nagl_model: openff-gnn-am1bcc-0.1.0-rc.3.pt
"""

@pytest.mark.xfail(HAS_OPENEYE, reason="openff-nagl#177")
@skip_if_missing("openff.nagl")
@skip_if_missing("openff.nagl_models")
def test_custom_yaml_plan_radial_smoke_test(custom_yaml_radial, eg5_files, tmpdir):
    protein, ligand, cofactor = eg5_files
    settings_path = tmpdir / "settings.yaml"
    with open(settings_path, "w") as f:
        f.write(custom_yaml_radial)

    assert settings_path.exists()

    args = [
        '-p', protein,
        '-M', ligand,
        '-C', cofactor,
        '-s', settings_path,
    ]

    runner = CliRunner()

    with runner.isolated_filesystem():
        result = runner.invoke(plan_rbfe_network, args)

        assert result.exit_code == 0
