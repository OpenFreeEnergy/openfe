# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import logging

import numpy as np
import pytest
from gufe import SmallMoleculeComponent

from openfe.protocols.openmm_afe.abfe_units import (
    ABFEComplexAnalysisUnit,
    ABFESolventAnalysisUnit,
)


@pytest.fixture(scope="session")
def abfe_ligand_smc(abfe_structural_results_dir):
    return SmallMoleculeComponent.from_sdf_file(abfe_structural_results_dir / "ligand.sdf")


@pytest.fixture(scope="session")
def abfe_complex_data(abfe_structural_results_dir, abfe_ligand_smc):
    base = abfe_structural_results_dir / "complex"
    return {
        "pdb": base / "alchemical_system.pdb",
        "nc": base / "complex.nc",
        "ligand_indices": np.load(base / "ligand_indices.npy").tolist(),
        "ligand_smcs": [abfe_ligand_smc],
    }


@pytest.fixture(scope="session")
def abfe_solvent_data(abfe_structural_results_dir, abfe_ligand_smc):
    base = abfe_structural_results_dir / "solvent"
    return {
        "pdb": base / "alchemical_system.pdb",
        "nc": base / "solvent.nc",
        "ligand_indices": np.load(base / "ligand_indices.npy").tolist(),
        "ligand_smcs": [abfe_ligand_smc],
    }


@pytest.fixture(scope="session")
def abfe_complex_structural_analysis_result(abfe_complex_data, tmp_path_factory):
    d = abfe_complex_data
    tmp = tmp_path_factory.mktemp("abfe_complex_structural_analysis")
    result = ABFEComplexAnalysisUnit()._structural_analysis(
        topology=d["pdb"],
        trajectory=d["nc"],
        output_directory=tmp,
        ligand_indices=d["ligand_indices"],
        ligand_smcs=d["ligand_smcs"],
        protein_selection="protein and name CA",
        skip=None,
        dry=False,
    )
    return result, tmp


@pytest.fixture(scope="session")
def abfe_solvent_structural_analysis_result(abfe_solvent_data, tmp_path_factory):
    d = abfe_solvent_data
    tmp = tmp_path_factory.mktemp("abfe_solvent_structural_analysis")
    result = ABFESolventAnalysisUnit()._structural_analysis(
        topology=d["pdb"],
        trajectory=d["nc"],
        output_directory=tmp,
        ligand_indices=d["ligand_indices"],
        ligand_smcs=d["ligand_smcs"],
        protein_selection="protein and name CA",
        skip=None,
        dry=False,
    )
    return result, tmp


@pytest.mark.slow
class TestComplexStructuralAnalysis:
    n_lambda = 30

    def test_npz_written(self, abfe_complex_structural_analysis_result):
        result, _ = abfe_complex_structural_analysis_result
        assert "structural_analysis" in result
        assert "structural_analysis_error" not in result
        assert result["structural_analysis"].exists()

    def test_npz_keys_values_shape(self, abfe_complex_structural_analysis_result):
        result, _ = abfe_complex_structural_analysis_result
        npz = np.load(result["structural_analysis"])
        expected_keys = {
            "ligand_RMSD",
            "ligand_COM_drift",
            "protein_2D_RMSD",
            "time_ps",
        }
        assert set(npz.files) == expected_keys

        # First frame RMSD should be zero (reference frame)
        assert npz["ligand_RMSD"][0][0] == pytest.approx(0.0, abs=1e-5)
        assert npz["ligand_COM_drift"][0][0] == pytest.approx(0.0, abs=1e-5)

        # Time should start at zero
        assert npz["time_ps"][0] == pytest.approx(0.0, abs=1e-5)

        # All RMSD and COM drift values should be non-negative
        assert np.all(npz["ligand_RMSD"] >= 0)
        assert np.all(npz["ligand_COM_drift"] >= 0)
        assert np.all(npz["protein_2D_RMSD"] >= 0)

        # One entry per lambda window
        for key in expected_keys - {"time_ps"}:
            assert len(npz[key]) == self.n_lambda

        # Per-state time series match the time array
        n_frames = len(npz["time_ps"])
        assert npz["ligand_RMSD"].shape == (self.n_lambda, n_frames)
        assert npz["ligand_COM_drift"].shape == (self.n_lambda, n_frames)
        assert npz["protein_2D_RMSD"].shape[0] == self.n_lambda

    def test_plots_written(self, abfe_complex_structural_analysis_result):
        _, tmp = abfe_complex_structural_analysis_result
        expected_plots = {
            "ligand_RMSD.png",
            "ligand_COM_drift.png",
            "protein_2D_RMSD.png",
        }
        written = {f.name for f in tmp.glob("*.png")}
        assert written == expected_plots

    def test_dry_no_plots(self, abfe_complex_data, tmp_path):
        d = abfe_complex_data
        result = ABFEComplexAnalysisUnit()._structural_analysis(
            topology=d["pdb"],
            trajectory=d["nc"],
            output_directory=tmp_path,
            ligand_indices=d["ligand_indices"],
            ligand_smcs=d["ligand_smcs"],
            protein_selection="protein and name CA",
            skip=10,
            dry=True,
        )

        assert result["structural_analysis"].exists()
        assert list(tmp_path.glob("*.png")) == []

    def test_bad_trajectory_returns_error_dict(self, abfe_complex_data, tmp_path):
        d = abfe_complex_data
        result = ABFEComplexAnalysisUnit()._structural_analysis(
            topology=d["pdb"],
            trajectory=tmp_path / "nonexistent.nc",
            output_directory=tmp_path,
            ligand_indices=d["ligand_indices"],
            ligand_smcs=d["ligand_smcs"],
            protein_selection="protein and name CA",
            skip=None,
            dry=True,
        )

        assert "structural_analysis_error" in result
        assert "structural_analysis" not in result

    def test_skip_affects_output_length(self, abfe_complex_data, tmp_path):
        d = abfe_complex_data

        def _get_n_frames(skip, outdir):
            outdir.mkdir()
            result = ABFEComplexAnalysisUnit()._structural_analysis(
                topology=d["pdb"],
                trajectory=d["nc"],
                output_directory=outdir,
                ligand_indices=d["ligand_indices"],
                ligand_smcs=d["ligand_smcs"],
                protein_selection="protein and name CA",
                skip=skip,
                dry=True,
            )
            return len(np.load(result["structural_analysis"])["time_ps"])

        n_frames_skip5 = _get_n_frames(5, tmp_path / "skip5")
        n_frames_skip10 = _get_n_frames(10, tmp_path / "skip10")

        assert n_frames_skip5 > n_frames_skip10
        assert n_frames_skip5 == 11
        assert n_frames_skip10 == 6


class TestSolventStructuralAnalysis:
    n_lambda = 14

    def test_npz_written(self, abfe_solvent_structural_analysis_result):
        result, _ = abfe_solvent_structural_analysis_result
        assert "structural_analysis" in result
        assert "structural_analysis_error" not in result
        assert result["structural_analysis"].exists()

    def test_npz_keys_values_shape(self, abfe_solvent_structural_analysis_result):
        result, _ = abfe_solvent_structural_analysis_result
        npz = np.load(result["structural_analysis"])
        expected_keys = {"ligand_RMSD", "time_ps"}
        assert set(npz.files) == expected_keys

        # First frame RMSD should be zero (reference frame)
        assert npz["ligand_RMSD"][0][0] == pytest.approx(0.0, abs=1e-5)

        # Time should start at zero
        assert npz["time_ps"][0] == pytest.approx(0.0, abs=1e-5)

        # All RMSD values should be non-negative
        assert np.all(npz["ligand_RMSD"] >= 0)

        # One entry per lambda window
        n_frames = len(npz["time_ps"])
        assert npz["ligand_RMSD"].shape == (self.n_lambda, n_frames)

    def test_plots_written(self, abfe_solvent_structural_analysis_result):
        _, tmp = abfe_solvent_structural_analysis_result
        written = {f.name for f in tmp.glob("*.png")}
        assert written == {"ligand_RMSD.png"}

    def test_bad_trajectory_returns_error_dict(self, abfe_solvent_data, tmp_path):
        d = abfe_solvent_data
        result = ABFESolventAnalysisUnit()._structural_analysis(
            topology=d["pdb"],
            trajectory=tmp_path / "nonexistent.nc",
            output_directory=tmp_path,
            ligand_indices=d["ligand_indices"],
            ligand_smcs=d["ligand_smcs"],
            protein_selection="protein and name CA",
            skip=None,
            dry=True,
        )

        assert "structural_analysis_error" in result
        assert "structural_analysis" not in result

    def test_no_ligand_atoms_warning_and_error(self, abfe_solvent_data, tmp_path, caplog):
        d = abfe_solvent_data

        with caplog.at_level(logging.WARNING):
            result = ABFESolventAnalysisUnit()._structural_analysis(
                topology=None,
                trajectory=tmp_path / "nonexistent.nc",  # won't be accessed
                output_directory=tmp_path,
                ligand_indices=d["ligand_indices"],
                ligand_smcs=d["ligand_smcs"],
                protein_selection="protein and name CA",
                skip=None,
                dry=True,
            )

        assert "structural_analysis_error" in result
        assert "structural_analysis" not in result
        assert any("No atoms found" in msg for msg in caplog.messages)

    def test_multiple_ligands_warning_and_error(self, abfe_solvent_data, tmp_path, caplog):
        d = abfe_solvent_data

        with caplog.at_level(logging.WARNING):
            result = ABFESolventAnalysisUnit()._structural_analysis(
                topology=d["pdb"],
                trajectory=d["nc"],
                output_directory=tmp_path,
                ligand_indices=d["ligand_indices"],
                ligand_smcs=d["ligand_smcs"] * 2,
                protein_selection="protein and name CA",
                skip=None,
                dry=True,
            )

        assert "structural_analysis_error" in result
        assert "structural_analysis" not in result
        assert any("single alchemical species" in msg for msg in caplog.messages)

    def test_ligand_indices_mismatch_warning_and_error(self, abfe_solvent_data, tmp_path, caplog):
        d = abfe_solvent_data

        with caplog.at_level(logging.WARNING):
            result = ABFESolventAnalysisUnit()._structural_analysis(
                topology=d["pdb"],
                trajectory=tmp_path / "nonexistent.nc",  # won't be accessed
                output_directory=tmp_path,
                ligand_indices=d["ligand_indices"][:-1],
                ligand_smcs=d["ligand_smcs"],
                protein_selection="protein and name CA",
                skip=None,
                dry=True,
            )

        assert "structural_analysis_error" in result
        assert "structural_analysis" not in result
        assert any("does not match the number of ligand atoms" in msg for msg in caplog.messages)
