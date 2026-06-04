import pathlib

import numpy as np
import pooch
import pytest
from rdkit.Chem import SDMolSupplier

from openfe.data._registry import POOCH_CACHE
from openfe.protocols.openmm_septop.base_units import BaseSepTopAnalysisUnit

pooch_septop_structural = pooch.create(
    path=POOCH_CACHE,
    base_url="https://zenodo.org/records/20507106/files/",
    registry={
        "septop_structural_results.zip": "md5:cffc193dacb5ad8d26c5467ae05b1a00",
    },
)


@pytest.fixture(scope="session")
def septop_structural_results_dir():
    pooch_septop_structural.fetch("septop_structural_results.zip", processor=pooch.Unzip())
    return pathlib.Path(
        POOCH_CACHE / "septop_structural_results.zip.unzip/septop_structural_results"
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
def complex_structural_analysis_result(septop_complex_data, tmp_path_factory):
    d = septop_complex_data
    tmp = tmp_path_factory.mktemp("complex_structural_analysis")
    result = BaseSepTopAnalysisUnit._structural_analysis(
        pdb_file=d["pdb"],
        trj_file=d["nc"],
        output_directory=tmp,
        dry=False,
        simtype="complex",
        ligand_A_indices=d["ligand_A_indices"],
        ligand_B_indices=d["ligand_B_indices"],
        rdmol_A=d["rdmol_A"],
        rdmol_B=d["rdmol_B"],
        protein_selection="protein and name CA",
        skip=None,
    )
    return result, tmp


@pytest.fixture(scope="session")
def solvent_structural_analysis_result(septop_solvent_data, tmp_path_factory):
    d = septop_solvent_data
    tmp = tmp_path_factory.mktemp("solvent_structural_analysis")
    result = BaseSepTopAnalysisUnit._structural_analysis(
        pdb_file=d["pdb"],
        trj_file=d["nc"],
        output_directory=tmp,
        dry=False,
        simtype="solvent",
        ligand_A_indices=d["ligand_A_indices"],
        ligand_B_indices=d["ligand_B_indices"],
        rdmol_A=d["rdmol_A"],
        rdmol_B=d["rdmol_B"],
        protein_selection="protein and name CA",
        skip=None,
    )
    return result, tmp


class TestComplexStructuralAnalysis:
    def test_npz_written(self, complex_structural_analysis_result):
        result, _ = complex_structural_analysis_result
        assert "structural_analysis" in result
        assert result["structural_analysis"].exists()

    def test_npz_keys_values_shape(self, complex_structural_analysis_result):
        result, _ = complex_structural_analysis_result
        npz = np.load(result["structural_analysis"])
        expected_keys = {
            "ligand_A_RMSD",
            "ligand_B_RMSD",
            "ligand_A_COM_drift",
            "ligand_B_COM_drift",
            "protein_2D_RMSD",
            "time_ps",
        }
        assert set(npz.files) == expected_keys

        # First frame RMSD should be zero (reference frame)
        assert npz["ligand_A_RMSD"][0][0] == pytest.approx(0.0, abs=1e-5)
        assert npz["ligand_B_RMSD"][0][0] == pytest.approx(0.0, abs=1e-5)
        assert npz["ligand_A_COM_drift"][0][0] == pytest.approx(0.0, abs=1e-5)
        assert npz["ligand_B_COM_drift"][0][0] == pytest.approx(0.0, abs=1e-5)

        # Time should start at zero
        assert npz["time_ps"][0] == pytest.approx(0.0, abs=1e-5)

        # All RMSD and COM drift values should be non-negative
        for state in npz["ligand_A_RMSD"]:
            assert np.all(state >= 0)
        for state in npz["ligand_B_RMSD"]:
            assert np.all(state >= 0)
        for state in npz["ligand_A_COM_drift"]:
            assert np.all(state >= 0)
        for state in npz["ligand_B_COM_drift"]:
            assert np.all(state >= 0)

        # Should be 13 lambda windows
        n_lambda = 13
        for key in expected_keys - {"time_ps"}:
            assert len(npz[key]) == n_lambda

    def test_plots_written(self, complex_structural_analysis_result):
        result, tmp = complex_structural_analysis_result
        expected_plots = {
            "ligand_A_RMSD.png",
            "ligand_B_RMSD.png",
            "ligand_A_COM_drift.png",
            "ligand_B_COM_drift.png",
            "protein_2D_RMSD.png",
        }
        written = {f.name for f in tmp.glob("*.png")}
        assert written == expected_plots

    def test_bad_trajectory_returns_error_dict(self, septop_complex_data, tmp_path):
        d = septop_complex_data
        result = BaseSepTopAnalysisUnit._structural_analysis(
            pdb_file=d["pdb"],
            trj_file=tmp_path / "nonexistent.nc",
            output_directory=tmp_path,
            dry=True,
            simtype="complex",
            ligand_A_indices=d["ligand_A_indices"],
            ligand_B_indices=d["ligand_B_indices"],
            rdmol_A=d["rdmol_A"],
            rdmol_B=d["rdmol_B"],
            protein_selection="protein and name CA",
            skip=None,
        )

        assert "structural_analysis_error" in result
        assert "structural_analysis" not in result


class TestSolventStructuralAnalysis:
    def test_npz_written(self, solvent_structural_analysis_result):
        result, _ = solvent_structural_analysis_result
        assert "structural_analysis" in result
        assert result["structural_analysis"].exists()

    def test_npz_keys_values_shape(self, solvent_structural_analysis_result):
        result, _ = solvent_structural_analysis_result
        npz = np.load(result["structural_analysis"])
        expected_keys = {"ligand_A_RMSD", "ligand_B_RMSD", "time_ps"}
        assert set(npz.files) == expected_keys

        # First frame RMSD should be zero (reference frame)
        assert npz["ligand_A_RMSD"][0][0] == pytest.approx(0.0, abs=1e-5)
        assert npz["ligand_B_RMSD"][0][0] == pytest.approx(0.0, abs=1e-5)

        # Time should start at zero
        assert npz["time_ps"][0] == pytest.approx(0.0, abs=1e-5)

        # All RMSD and COM drift values should be non-negative
        for state in npz["ligand_A_RMSD"]:
            assert np.all(state >= 0)
        for state in npz["ligand_B_RMSD"]:
            assert np.all(state >= 0)

        # Should be 13 lambda windows
        n_lambda = 13
        for key in expected_keys - {"time_ps"}:
            assert len(npz[key]) == n_lambda

    def test_plots_written(self, solvent_structural_analysis_result):
        result, tmp = solvent_structural_analysis_result
        expected_plots = {"ligand_A_RMSD.png", "ligand_B_RMSD.png"}
        written = {f.name for f in tmp.glob("*.png")}
        assert written == expected_plots

    def test_bad_trajectory_returns_error_dict(self, septop_solvent_data, tmp_path):
        d = septop_solvent_data
        result = BaseSepTopAnalysisUnit._structural_analysis(
            pdb_file=d["pdb"],
            trj_file=tmp_path / "nonexistent.nc",
            output_directory=tmp_path,
            dry=True,
            simtype="solvent",
            ligand_A_indices=d["ligand_A_indices"],
            ligand_B_indices=d["ligand_B_indices"],
            rdmol_A=d["rdmol_A"],
            rdmol_B=d["rdmol_B"],
            protein_selection="protein and name CA",
            skip=None,
        )

        assert "structural_analysis_error" in result
        assert "structural_analysis" not in result

    def test_empty_ligand_indices_warning_and_error(self, septop_solvent_data, tmp_path, caplog):
        import logging

        d = septop_solvent_data

        with caplog.at_level(logging.WARNING):
            result = BaseSepTopAnalysisUnit._structural_analysis(
                pdb_file=d["pdb"],
                trj_file=tmp_path / "nonexistent.nc",  # won't be accessed
                output_directory=tmp_path,
                dry=True,
                simtype="solvent",
                ligand_A_indices=[],
                ligand_B_indices=[],
                rdmol_A=d["rdmol_A"],
                rdmol_B=d["rdmol_B"],
                protein_selection="protein and name CA",
                skip=None,
            )

        assert "structural_analysis_error" in result
        assert "structural_analysis" not in result
        assert any("No ligand atoms found" in msg for msg in caplog.messages)
