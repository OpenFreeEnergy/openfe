# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import gzip
import itertools
import json
from pathlib import Path
from unittest import mock

import gufe
import numpy as np
import pytest
from openff.units import unit as offunit

import openfe
from openfe.protocols import openmm_afe
from openfe.protocols.restraint_utils.geometry.boresch import BoreschRestraintGeometry

from .utils import UNIT_TYPES, _get_units


@pytest.fixture()
def patcher():
    with (
        mock.patch(
            "openfe.protocols.openmm_afe.abfe_units.ABFESolventSetupUnit.run",
            return_value={
                "system": Path("system.xml.bz2"),
                "positions": Path("positions.npy"),
                "pdb_structure": Path("hybrid_system.pdb"),
                "selection_indices": np.zeros(100),
                "box_vectors": [np.zeros(3), np.zeros(3), np.zeros(3)] * offunit.nm,
                "standard_state_correction": 0 * offunit.kilocalorie_per_mole,
                "restraint_geometry": None,
            },
        ),
        mock.patch(
            "openfe.protocols.openmm_afe.abfe_units.ABFEComplexSetupUnit.run",
            return_value={
                "system": Path("system.xml.bz2"),
                "positions": Path("positions.npy"),
                "pdb_structure": Path("hybrid_system.pdb"),
                "selection_indices": np.zeros(100),
                "box_vectors": [np.zeros(3), np.zeros(3), np.zeros(3)] * offunit.nm,
                "standard_state_correction": 0 * offunit.kilocalorie_per_mole,
                "restraint_geometry": True,
            },
        ),
        mock.patch(
            "openfe.protocols.openmm_afe.base_afe_units.np.load",
            return_value=np.zeros(100),
        ),
        mock.patch(
            "openfe.protocols.openmm_afe.base_afe_units.deserialize",
            return_value="foo",
        ),
        mock.patch(
            "openfe.protocols.openmm_afe.abfe_units.ABFEComplexSimUnit.run",
            return_value={
                "trajectory": Path("file.nc"),
                "checkpoint": Path("chk.chk"),
            },
        ),
        mock.patch(
            "openfe.protocols.openmm_afe.abfe_units.ABFESolventSimUnit.run",
            return_value={
                "trajectory": Path("file.nc"),
                "checkpoint": Path("chk.chk"),
            },
        ),
        mock.patch(
            "openfe.protocols.openmm_afe.abfe_units.ABFEComplexAnalysisUnit.run",
            return_value={"foo": "bar"},
        ),
        mock.patch(
            "openfe.protocols.openmm_afe.abfe_units.ABFESolventAnalysisUnit.run",
            return_value={"foo": "bar"},
        ),
    ):
        yield


def test_gather(benzene_complex_dag, patcher, tmp_path):
    # check that .gather behaves as expected
    dagres = gufe.protocols.execute_DAG(
        benzene_complex_dag,
        shared_basedir=tmp_path,
        scratch_basedir=tmp_path,
        keep_shared=True,
    )

    protocol = openmm_afe.AbsoluteBindingProtocol(
        settings=openmm_afe.AbsoluteBindingProtocol.default_settings(),
    )

    res = protocol.gather([dagres])

    assert isinstance(res, openmm_afe.AbsoluteBindingProtocolResult)


def test_unit_tagging(benzene_complex_dag, patcher, tmpdir):
    # test that executing the units includes correct gen and repeat info

    dag_units = benzene_complex_dag.protocol_units

    for phase in ["solvent", "complex"]:
        setup_results = {}
        sim_results = {}
        analysis_results = {}

        setup_units = _get_units(dag_units, UNIT_TYPES[phase]["setup"])
        sim_units = _get_units(dag_units, UNIT_TYPES[phase]["sim"])
        a_units = _get_units(dag_units, UNIT_TYPES[phase]["analysis"])

        for u in setup_units:
            rid = u.inputs["repeat_id"]
            setup_results[rid] = u.execute(context=gufe.Context(tmpdir, tmpdir))

        for u in sim_units:
            rid = u.inputs["repeat_id"]
            sim_results[rid] = u.execute(
                context=gufe.Context(tmpdir, tmpdir),
                setup_results=setup_results[rid],
            )

        for u in a_units:
            rid = u.inputs["repeat_id"]
            analysis_results[rid] = u.execute(
                context=gufe.Context(tmpdir, tmpdir),
                setup_results=setup_results[rid],
                simulation_results=sim_results[rid],
            )

        for results in [setup_results, sim_results, analysis_results]:
            for ret in results.values():
                assert isinstance(ret, gufe.ProtocolUnitResult)
                assert ret.outputs["generation"] == 0

        assert len(setup_results) == len(sim_results) == len(analysis_results) == 3


class TestProtocolResult:
    @pytest.fixture()
    def protocolresult(self, abfe_transformation_json_path):
        with gzip.open(abfe_transformation_json_path) as f:
            pr = openfe.ProtocolResult.from_json(f)

        return pr

    def test_reload_protocol_result(self, afe_solv_transformation_json):
        d = json.loads(afe_solv_transformation_json, cls=gufe.tokenization.JSON_HANDLER.decoder)

        pr = openmm_afe.AbsoluteBindingProtocolResult.from_dict(d["protocol_result"])

        assert pr

    def test_get_estimate(self, protocolresult):
        est = protocolresult.get_estimate()

        assert est
        assert est.m == pytest.approx(-21.35, abs=0.01)
        assert isinstance(est, offunit.Quantity)
        assert est.is_compatible_with(offunit.kilojoule_per_mole)

    def test_get_uncertainty(self, protocolresult):
        est = protocolresult.get_uncertainty()

        assert est
        assert est.m == pytest.approx(1.04, abs=0.01)
        assert isinstance(est, offunit.Quantity)
        assert est.is_compatible_with(offunit.kilojoule_per_mole)

    def test_get_individual(self, protocolresult):
        inds = protocolresult.get_individual_estimates()

        assert isinstance(inds, dict)
        assert isinstance(inds["solvent"], list)
        assert isinstance(inds["complex"], list)
        assert len(inds["solvent"]) == len(inds["complex"]) == 3
        for e, u in itertools.chain(inds["solvent"], inds["complex"]):
            assert e.is_compatible_with(offunit.kilojoule_per_mole)
            assert u.is_compatible_with(offunit.kilojoule_per_mole)

    @pytest.mark.parametrize("key", ["solvent", "complex"])
    def test_get_forwards_etc(self, key, protocolresult):
        far = protocolresult.get_forward_and_reverse_energy_analysis()

        assert isinstance(far, dict)
        assert isinstance(far[key], list)

        for f in far[key]:
            if f is not None:
                assert isinstance(f, dict)

                for k in [
                    "fractions",
                    "forward_DGs",
                    "forward_dDGs",
                    "reverse_DGs",
                    "reverse_dDGs",
                ]:
                    assert k in f

                    if k == "fractions":
                        assert isinstance(f[k], np.ndarray)

    @pytest.mark.parametrize("key", ["solvent", "complex"])
    def test_get_frwd_reverse_none_return(self, key, protocolresult):
        # fetch the first result of type key
        data = [i for i in protocolresult.data[key].values()][0][0]
        # set the output to None
        data.outputs["forward_and_reverse_energies"] = None

        # now fetch the analysis results and expect a warning
        wmsg = f"were found in the forward and reverse dictionaries of the repeats of the {key}"
        with pytest.warns(UserWarning, match=wmsg):
            protocolresult.get_forward_and_reverse_energy_analysis()

    @pytest.mark.parametrize("key, n_rep", [("solvent", 14), ("complex", 30)])
    def test_get_overlap_matrices(self, key, n_rep, protocolresult):
        ovp = protocolresult.get_overlap_matrices()

        assert isinstance(ovp, dict)
        assert isinstance(ovp[key], list)
        assert len(ovp[key]) == 3

        ovp1 = ovp[key][0]
        assert isinstance(ovp1["matrix"], np.ndarray)
        assert ovp1["matrix"].shape == (n_rep, n_rep)

    @pytest.mark.parametrize("key, n_rep", [("solvent", 14), ("complex", 30)])
    def test_get_replica_transition_statistics(self, n_rep, key, protocolresult):
        rpx = protocolresult.get_replica_transition_statistics()

        assert isinstance(rpx, dict)
        assert isinstance(rpx[key], list)
        assert len(rpx[key]) == 3
        rpx1 = rpx[key][0]
        assert "eigenvalues" in rpx1
        assert "matrix" in rpx1
        assert rpx1["eigenvalues"].shape == (n_rep,)
        assert rpx1["matrix"].shape == (n_rep, n_rep)

    @pytest.mark.parametrize("key", ["solvent", "complex"])
    def test_equilibration_iterations(self, key, protocolresult):
        eq = protocolresult.equilibration_iterations()

        assert isinstance(eq, dict)
        assert isinstance(eq[key], list)
        assert len(eq[key]) == 3
        assert all(isinstance(v, float) for v in eq[key])

    @pytest.mark.parametrize("key", ["solvent", "complex"])
    def test_production_iterations(self, key, protocolresult):
        prod = protocolresult.production_iterations()

        assert isinstance(prod, dict)
        assert isinstance(prod[key], list)
        assert len(prod[key]) == 3
        assert all(isinstance(v, float) for v in prod[key])

    def test_filenotfound_replica_states(self, protocolresult):
        errmsg = "File could not be found"

        with pytest.raises(ValueError, match=errmsg):
            protocolresult.get_replica_states()

    def test_restraint_geometry(self, protocolresult):
        geom = protocolresult.restraint_geometries()
        assert isinstance(geom, list)
        assert len(geom) == 3
        assert isinstance(geom[0], BoreschRestraintGeometry)
        assert geom[0].guest_atoms == [1779, 1778, 1777]
        assert geom[0].host_atoms == [880, 865, 864]
        assert pytest.approx(geom[0].r_aA0, rel=1e-2) == 1.083558 * offunit.nanometer
        assert pytest.approx(geom[0].theta_A0, rel=1e-2) == 0.711876 * offunit.radian
        assert pytest.approx(geom[0].theta_B0, rel=1e-2) == 1.687366 * offunit.radian
        assert pytest.approx(geom[0].phi_A0, rel=1e-2) == -0.2164231 * offunit.radian
        assert pytest.approx(geom[0].phi_B0, rel=1e-2) == 1.892376 * offunit.radian
        assert pytest.approx(geom[0].phi_C0, rel=1e-2) == -0.522031870 * offunit.radian

    @pytest.mark.parametrize(
        "key, expected_size",
        [
            ["solvent", 41],
            ["complex", 1828],
        ],
    )
    def test_selection_indices(self, key, protocolresult, expected_size):
        indices = protocolresult.selection_indices()

        assert isinstance(indices, dict)
        assert isinstance(indices[key], list)
        for inds in indices[key]:
            assert isinstance(inds, np.ndarray)
            assert len(inds) == expected_size
