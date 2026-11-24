import json

import numpy as np
import pytest
from rdkit.Chem import SDMolSupplier

from openff.units import unit as offunit

from openfe.protocols.openmm_septop.septop_units import (
    SepTopComplexAnalysisUnit,
    SepTopSolventAnalysisUnit,
)
from openfe.protocols.restraint_utils.geometry.boresch import BoreschRestraintGeometry

BORESCH_KEYS = (
    "bond",
    "angle1",
    "angle2",
    "dihedral1",
    "dihedral2",
    "dihedral3",
)

@pytest.fixture(scope="session")
def septop_complex_data(septop_structural_results_dir):
    base = septop_structural_results_dir / "complex"
    return {
        "pdb": base / "alchemical_system.pdb",
        "nc": base / "complex.nc",
        "ligand_A_indices": np.load(base / "ligand_A_indices.npy").tolist(),
        "ligand_B_indices": np.load(base / "ligand_B_indices.npy").tolist(),
        "rdmol_A": SDMolSupplier(str(base / "ligand_A.sdf"), removeHs=False)[0],
        "rdmol_B": SDMolSupplier(str(base / "ligand_B.sdf"), removeHs=False)[0],
        "selection_indices": np.load(base / "selection_indices.npy"),
        "geometry_A": BoreschRestraintGeometry.model_validate(
            json.loads((base / "restraint_geometry_A.json").read_text())
        ),
        "geometry_B": BoreschRestraintGeometry.model_validate(
            json.loads((base / "restraint_geometry_B.json").read_text())
        ),
    }



@pytest.fixture(scope="session")
def septop_solvent_data(septop_structural_results_dir):
    base = septop_structural_results_dir / "solvent"
    return {
        "pdb": base / "alchemical_system.pdb",
        "nc": base / "solvent.nc",
        "ligand_A_indices": np.load(base / "ligand_A_indices.npy").tolist(),
        "ligand_B_indices": np.load(base / "ligand_B_indices.npy").tolist(),
        "rdmol_A": SDMolSupplier(str(base / "ligand_A.sdf"), removeHs=False)[0],
        "rdmol_B": SDMolSupplier(str(base / "ligand_B.sdf"), removeHs=False)[0],
    }


@pytest.fixture(scope="session")
def septop_complex_restraint_inputs(septop_complex_data):
    d = septop_complex_data
    indices = {}
    reference_values = {}
    for label in ("A", "B"):
        geometry = d[f"geometry_{label}"]
        indices[label] = (
            SepTopComplexAnalysisUnit._translate_to_subsampled_indices(
                [*geometry.host_atoms, *geometry.guest_atoms],
                d["selection_indices"],
            )
        )

        reference_values[f"ligand_{label}"] = (
            SepTopComplexAnalysisUnit._get_boresch_reference_values(geometry)
        )

    return indices, reference_values


@pytest.fixture(scope="session")
def septop_complex_structural_analysis_result(septop_complex_data, septop_complex_restraint_inputs, tmp_path_factory):
    d = septop_complex_data
    tmp = tmp_path_factory.mktemp("septop_complex_structural_analysis_result")
    unit = SepTopComplexAnalysisUnit()

    restraint_atom_indices, reference_values = septop_complex_restraint_inputs

    result = unit._structural_analysis(
        pdb_file=d["pdb"],
        trj_file=d["nc"],
        output_directory=tmp,
        ligand_A_indices=d["ligand_A_indices"],
        ligand_B_indices=d["ligand_B_indices"],
        rdmol_A=d["rdmol_A"],
        rdmol_B=d["rdmol_B"],
        protein_selection="protein and name CA",
        restraint_atom_indices_A=restraint_atom_indices["A"],
        restraint_atom_indices_B=restraint_atom_indices["B"],
        restraint_reference_values=reference_values,
        skip=None,
        dry=False,
    )
    return result, tmp


@pytest.fixture(scope="session")
def septop_solvent_structural_analysis_result(septop_solvent_data, tmp_path_factory):
    d = septop_solvent_data
    tmp = tmp_path_factory.mktemp("solvent_structural_analysis")
    unit = SepTopSolventAnalysisUnit()
    result = unit._structural_analysis(
        pdb_file=d["pdb"],
        trj_file=d["nc"],
        output_directory=tmp,
        ligand_A_indices=d["ligand_A_indices"],
        ligand_B_indices=d["ligand_B_indices"],
        rdmol_A=d["rdmol_A"],
        rdmol_B=d["rdmol_B"],
        protein_selection="protein and name CA",
        restraint_atom_indices_A=None,
        restraint_atom_indices_B=None,
        restraint_reference_values=None,
        skip=None,
        dry=False,
    )
    return result, tmp


class TestBoreschRestraintHelpers:
    @pytest.mark.parametrize(
        "full_system_indices, expected",
        [
            ([10, 30, 50], [0, 2, 4]),
            ([50, 10, 30], [4, 0, 2]),
            ([20, 40], [1, 3]),
            ([], []),
        ],
    )
    def test_translate_to_subsampled_indices(
        self, full_system_indices, expected
    ):
        selection_indices = np.array([10, 20, 30, 40, 50])

        result = (
            SepTopComplexAnalysisUnit._translate_to_subsampled_indices(
                full_system_indices,
                selection_indices,
            )
        )

        assert result == expected

    def test_translate_missing_restraint_indices(self):
        selection_indices = np.array([10, 20, 30, 40, 50])

        with pytest.raises(
            ValueError,
            match="Boresch restraint atoms are not in the trajectory output selection",
        ):
            SepTopComplexAnalysisUnit._translate_to_subsampled_indices(
                [10, 60, 30],
                selection_indices,
            )

    def test_get_boresch_reference_values(self):
        geometry = BoreschRestraintGeometry(
            host_atoms=[0, 1, 2],
            guest_atoms=[3, 4, 5],
            r_aA0=0.5 * offunit.nanometer,
            theta_A0=60.0 * offunit.degree,
            theta_B0=90.0 * offunit.degree,
            phi_A0=-90.0 * offunit.degree,
            phi_B0=120.0 * offunit.degree,
            phi_C0=-45.0 * offunit.degree,
        )

        reference_values = (
            SepTopComplexAnalysisUnit._get_boresch_reference_values(geometry)
        )

        expected = {
            "bond": 5.0,
            "angle1": np.pi / 3,
            "angle2": np.pi / 2,
            "dihedral1": -np.pi / 2,
            "dihedral2": 2 * np.pi / 3,
            "dihedral3": -np.pi / 4,
        }

        assert reference_values == pytest.approx(expected)


class TestComplexStructuralAnalysis:
    def test_npz_written(self, septop_complex_structural_analysis_result):
        result, _ = septop_complex_structural_analysis_result
        assert "structural_analysis" in result
        assert result["structural_analysis"].exists()

    def test_npz_keys_values_shape(self, septop_complex_structural_analysis_result):
        result, _ = septop_complex_structural_analysis_result
        rmsd_drift_keys = {
            "ligand_A_RMSD",
            "ligand_B_RMSD",
            "ligand_A_COM_drift",
            "ligand_B_COM_drift",
        }

        restraint_keys = {
            f"ligand_{ligand}_restraints_{key}"
            for ligand in ("A", "B")
            for key in BORESCH_KEYS
        }

        expected_keys = rmsd_drift_keys | restraint_keys | {
            "protein_2D_RMSD",
            "time_ps",
        }
        with np.load(result["structural_analysis"]) as npz:
            assert set(npz.files) == expected_keys

            n_lambda = 13
            n_frames = len(npz["time_ps"])

            # Time should start at zero
            assert npz["time_ps"][0] == pytest.approx(0.0, abs=1e-5)

            # Ligand RMSD and COM drift
            for key in rmsd_drift_keys:
                values = npz[key]
                assert values.shape == (n_lambda, n_frames)
                assert values[0, 0] == pytest.approx(0.0, abs=1e-5)
                assert np.all(values >= 0)
                assert np.all(np.isfinite(values))

            # Protein pairwise RMSD
            n_pairs = n_frames * (n_frames - 1) // 2
            values = npz["protein_2D_RMSD"]
            assert values.shape == (n_lambda, n_pairs)
            assert np.all(values >= 0)
            assert np.all(np.isfinite(values))

            # Boresch restraint coordinates
            for key in restraint_keys:
                values = npz[key]
                assert values.shape == (n_lambda, n_frames)
                assert np.all(np.isfinite(values))

    def test_plots_written(self, septop_complex_structural_analysis_result):
        _, tmp = septop_complex_structural_analysis_result
        expected_plots = {
            "ligand_A_RMSD.png",
            "ligand_B_RMSD.png",
            "ligand_A_COM_drift.png",
            "ligand_B_COM_drift.png",
            "protein_2D_RMSD.png",
            "ligand_A_restraints.png",
            "ligand_B_restraints.png",
        }
        written = {f.name for f in tmp.glob("*.png")}
        assert written == expected_plots

    def test_bad_trajectory_returns_error_dict(self, septop_complex_data, tmp_path):
        d = septop_complex_data
        result = SepTopComplexAnalysisUnit()._structural_analysis(
            pdb_file=d["pdb"],
            trj_file=tmp_path / "nonexistent.nc",
            output_directory=tmp_path,
            ligand_A_indices=d["ligand_A_indices"],
            ligand_B_indices=d["ligand_B_indices"],
            rdmol_A=d["rdmol_A"],
            rdmol_B=d["rdmol_B"],
            protein_selection="protein and name CA",
            restraint_atom_indices_A=None,
            restraint_atom_indices_B=None,
            restraint_reference_values=None,
            skip=None,
            dry=True,
        )

        assert "structural_analysis_error" in result
        assert "structural_analysis" not in result

    @pytest.mark.parametrize("skip, expected_n_frames", [(1, 6), (2, 3)])
    def test_skip_affects_output_length(
            self, septop_complex_data, septop_complex_restraint_inputs, tmp_path, skip, expected_n_frames
    ):
        d = septop_complex_data
        unit = SepTopComplexAnalysisUnit()

        restraint_indices, _ = septop_complex_restraint_inputs

        result = unit._structural_analysis(
            pdb_file=d["pdb"],
            trj_file=d["nc"],
            output_directory=tmp_path,
            ligand_A_indices=d["ligand_A_indices"],
            ligand_B_indices=d["ligand_B_indices"],
            rdmol_A=d["rdmol_A"],
            rdmol_B=d["rdmol_B"],
            protein_selection="protein and name CA",
            restraint_atom_indices_A=restraint_indices["A"],
            restraint_atom_indices_B=restraint_indices["B"],
            restraint_reference_values=None,
            skip=skip,
            dry=True,
        )

        with np.load(result["structural_analysis"]) as npz:
            assert len(npz["time_ps"]) == expected_n_frames

            for ligand in ("A", "B"):
                for key in BORESCH_KEYS:
                    values = npz[f"ligand_{ligand}_restraints_{key}"]
                    assert values.shape == (13, expected_n_frames)
        assert not list(tmp_path.glob("*.png"))

    @pytest.mark.parametrize("ligand", ["A", "B"])
    def test_single_ligand_restraint_analysis(
            self,
            septop_complex_data,
            septop_complex_restraint_inputs,
            tmp_path,
            ligand,
    ):
        d = septop_complex_data
        restraint_indices, _ = septop_complex_restraint_inputs

        result = SepTopComplexAnalysisUnit()._structural_analysis(
            pdb_file=d["pdb"],
            trj_file=d["nc"],
            output_directory=tmp_path,
            ligand_A_indices=d["ligand_A_indices"],
            ligand_B_indices=d["ligand_B_indices"],
            rdmol_A=d["rdmol_A"],
            rdmol_B=d["rdmol_B"],
            protein_selection="protein and name CA",
            restraint_atom_indices_A=(restraint_indices["A"] if ligand == "A" else None ),
            restraint_atom_indices_B=(restraint_indices["B"] if ligand == "B" else None),
            restraint_reference_values=None,
            skip=2,
            dry=True,
        )

        assert "structural_analysis" in result

        expected_restraint_keys = {f"ligand_{ligand}_restraints_{key}" for key in BORESCH_KEYS}

        with np.load(result["structural_analysis"]) as npz:
            restraint_keys = {
                key for key in npz.files if "_restraints_" in key
            }

            assert restraint_keys == expected_restraint_keys

            for key in restraint_keys:
                assert npz[key].shape == (13, 3)
                assert np.all(np.isfinite(npz[key]))

            # The other structural analyses should still be performed.
            assert "ligand_A_RMSD" in npz.files
            assert "ligand_B_RMSD" in npz.files
            assert "ligand_A_COM_drift" in npz.files
            assert "ligand_B_COM_drift" in npz.files
            assert "protein_2D_RMSD" in npz.files


class TestSolventStructuralAnalysis:
    def test_npz_written(self, septop_solvent_structural_analysis_result):
        result, _ = septop_solvent_structural_analysis_result
        assert "structural_analysis" in result
        assert result["structural_analysis"].exists()

    def test_npz_keys_values_shape(self, septop_solvent_structural_analysis_result):
        result, _ = septop_solvent_structural_analysis_result
        npz = np.load(result["structural_analysis"])
        expected_keys = {"ligand_A_RMSD", "ligand_B_RMSD", "time_ps"}
        assert set(npz.files) == expected_keys

        # First frame RMSD should be zero (reference frame)
        assert npz["ligand_A_RMSD"][0][0] == pytest.approx(0.0, abs=1e-5)
        assert npz["ligand_B_RMSD"][0][0] == pytest.approx(0.0, abs=1e-5)

        # Time should start at zero
        assert npz["time_ps"][0] == pytest.approx(0.0, abs=1e-5)

        # All RMSD values should be non-negative
        for state in npz["ligand_A_RMSD"]:
            assert np.all(state >= 0)
        for state in npz["ligand_B_RMSD"]:
            assert np.all(state >= 0)

        # Should be 13 lambda windows
        n_lambda = 13
        for key in expected_keys - {"time_ps"}:
            assert len(npz[key]) == n_lambda

    def test_plots_written(self, septop_solvent_structural_analysis_result):
        _, tmp = septop_solvent_structural_analysis_result
        expected_plots = {"ligand_A_RMSD.png", "ligand_B_RMSD.png"}
        written = {f.name for f in tmp.glob("*.png")}
        assert written == expected_plots

    def test_bad_trajectory_returns_error_dict(self, septop_solvent_data, tmp_path):
        d = septop_solvent_data
        result = SepTopSolventAnalysisUnit()._structural_analysis(
            pdb_file=d["pdb"],
            trj_file=tmp_path / "nonexistent.nc",
            output_directory=tmp_path,
            ligand_A_indices=d["ligand_A_indices"],
            ligand_B_indices=d["ligand_B_indices"],
            rdmol_A=d["rdmol_A"],
            rdmol_B=d["rdmol_B"],
            protein_selection="protein and name CA",
            restraint_atom_indices_A=None,
            restraint_atom_indices_B=None,
            restraint_reference_values=None,
            skip=None,
            dry=True,
        )

        assert "structural_analysis_error" in result
        assert "structural_analysis" not in result

    def test_empty_ligand_indices_warning_and_error(self, septop_solvent_data, tmp_path, caplog):
        import logging

        d = septop_solvent_data

        with caplog.at_level(logging.WARNING):
            result = SepTopSolventAnalysisUnit()._structural_analysis(
                pdb_file=d["pdb"],
                trj_file=tmp_path / "nonexistent.nc",  # won't be accessed
                output_directory=tmp_path,
                ligand_A_indices=[],
                ligand_B_indices=[],
                rdmol_A=d["rdmol_A"],
                rdmol_B=d["rdmol_B"],
                protein_selection="protein and name CA",
                restraint_atom_indices_A=None,
                restraint_atom_indices_B=None,
                restraint_reference_values=None,
                skip=None,
                dry=True,
            )

        assert "structural_analysis_error" in result
        assert "structural_analysis" not in result
        assert any("No ligand atoms found" in msg for msg in caplog.messages)
