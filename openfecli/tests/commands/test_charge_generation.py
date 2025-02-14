import pytest
from click.testing import CliRunner

from gufe import SmallMoleculeComponent
from openfecli.commands.generate_partial_charges import charge_molecules
from openff.toolkit import Molecule
from openff.units import unit
import numpy as np
from openff.utilities.testing import skip_if_missing

@pytest.fixture
def yaml_nagl_settings():
    return """\
partial_charge:
  method: nagl
  settings:
    nagl_model: openff-gnn-am1bcc-0.1.0-rc.3.pt
"""

@pytest.fixture
def methane() -> Molecule:
    # ensure consistent atom ordering
    methane = Molecule.from_mapped_smiles("[C:1]([H:2])([H:3])([H:4])[H:5]")
    methane.generate_conformers(n_conformers=1)
    return methane

@pytest.fixture
def methane_with_charges(methane) -> Molecule:
    methane._partial_charges = [-1.0, 0.25, 0.25, 0.25, 0.25] * unit.elementary_charge
    return methane


def test_write_charges_to_input(methane, tmpdir):
    runner = CliRunner()
    mol_path = tmpdir / "methane.sdf"
    methane.to_file(str(mol_path), "sdf")

    with runner.isolated_filesystem():
        # check an error is raised if we try to overwrite the input
        with pytest.raises(FileExistsError, match="The output file "):
            _ = runner.invoke(
                charge_molecules,
                [
                    "-M",
                    mol_path,
                    "-o",
                    mol_path
                ], catch_exceptions=False
            )

def test_charge_molecules_default(methane, tmpdir):

    runner = CliRunner()
    mol_path = tmpdir / "methane.sdf"
    methane.to_file(str(mol_path), "sdf")
    output_file = str(tmpdir / "charged_methane.sdf")

    with runner.isolated_filesystem():
        # make sure the charges are picked up
        with pytest.warns(match="Partial charges have been provided, these will "):
            result = runner.invoke(
                charge_molecules,
                [
                    "-M",
                    mol_path,
                    "-o",
                    output_file
                ]
            )

        assert result.exit_code == 0
        assert "Partial Charge Generation: am1bcc" in result.output
        # make sure the charges have been saved
        methane = SmallMoleculeComponent.from_sdf_file(filename=output_file)
        off_methane = methane.to_openff()
        assert off_methane.partial_charges is not None
        assert len(off_methane.partial_charges) == 5

@pytest.mark.parametrize("overwrite, expected_charges", [
    pytest.param(False, [-1.0, 0.25, 0.25, 0.25, 0.25], id="Don't overwrite"),
    pytest.param(True, [-0.1084, 0.0271, 0.0271, 0.0271, 0.0271], id="Overwrite")
])
def test_charge_molecules_overwrite(overwrite, tmpdir, methane_with_charges, expected_charges):
    runner = CliRunner()
    mol_path = tmpdir / "methane.sdf"
    methane_with_charges.to_file(str(mol_path), "sdf")
    output_file = str(tmpdir / "charged_methane.sdf")

    args = [
        "-M",
        mol_path,
        "-o",
        output_file
    ]
    if overwrite:
        args.append("--overwrite-charges")

    with runner.isolated_filesystem():
        # make sure the charges are picked up
        with pytest.warns(match="Partial charges have been provided, these will "):
            result = runner.invoke(
                charge_molecules,
                args,
            )

        assert result.exit_code == 0
        assert "Partial Charge Generation: am1bcc" in result.output
        if overwrite:
            assert "Overwriting partial charges" in result.output

        # make sure the charges have not changed from the inputs
        methane = SmallMoleculeComponent.from_sdf_file(filename=output_file)
        off_methane = methane.to_openff()
        assert np.allclose(off_methane.partial_charges.m, expected_charges)


@pytest.mark.parametrize("ncores", [
    pytest.param(1, id="1"),
    pytest.param(2, id="2")
])
@skip_if_missing("openff.nagl")
@skip_if_missing("openff.nagl_models")
def test_charge_settings(methane, tmpdir, yaml_nagl_settings, ncores):
    runner = CliRunner()
    mol_path = tmpdir / "methane.sdf"
    methane.to_file(str(mol_path), "sdf")
    output_file = str(tmpdir / "charged_methane.sdf")

    # use nagl charges for CI speed!
    settings_path = tmpdir / "settings.yaml"
    with open(settings_path, "w") as f:
        f.write(yaml_nagl_settings)

    with runner.isolated_filesystem():
        # make sure the charges are picked up
        with pytest.warns(match="Partial charges have been provided, these will "):
            result = runner.invoke(
                charge_molecules,
                [
                    "-M",
                    mol_path,
                    "-o",
                    output_file,
                    "-s",
                    settings_path,
                    "-n",
                    ncores
                ]
            )

        assert result.exit_code == 0
        assert "Partial Charge Generation: nagl" in result.output
        # make sure the charges have been saved
        methane = SmallMoleculeComponent.from_sdf_file(filename=output_file)
        off_methane = methane.to_openff()
        assert off_methane.partial_charges is not None
        assert len(off_methane.partial_charges) == 5
