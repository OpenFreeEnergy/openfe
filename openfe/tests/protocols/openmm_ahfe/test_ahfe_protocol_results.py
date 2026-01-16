# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import itertools
import json
from pathlib import Path
from unittest import mock

import gufe
import numpy as np
import pytest
from openff.units import unit as offunit

import openfe
from openfe import ChemicalSystem, SolventComponent
from openfe.protocols import openmm_afe

from .utils import UNIT_TYPES, _get_units


@pytest.fixture()
def protocol_dry_settings():
    settings = openmm_afe.AbsoluteSolvationProtocol.default_settings()
    settings.vacuum_engine_settings.compute_platform = None
    settings.solvent_engine_settings.compute_platform = None
    settings.protocol_repeats = 1
    return settings


@pytest.fixture
def benzene_solvation_dag(benzene_system, protocol_dry_settings):
    protocol_dry_settings.protocol_repeats = 3
    protocol = openmm_afe.AbsoluteSolvationProtocol(settings=protocol_dry_settings)

    stateA = benzene_system

    stateB = ChemicalSystem({"solvent": SolventComponent()})

    return protocol.create(stateA=stateA, stateB=stateB, mapping=None)


def test_gather(benzene_solvation_dag, tmpdir):
    # check that .gather behaves as expected
    with (
        mock.patch(
            "openfe.protocols.openmm_afe.ahfe_units.AHFESolventSetupUnit.run",
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
            "openfe.protocols.openmm_afe.ahfe_units.AHFEVacuumSetupUnit.run",
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
            "openfe.protocols.openmm_afe.base_afe_units.np.load",
            return_value=np.zeros(100),
        ),
        mock.patch(
            "openfe.protocols.openmm_afe.base_afe_units.deserialize",
            return_value="foo",
        ),
        mock.patch(
            "openfe.protocols.openmm_afe.ahfe_units.AHFESolventSimUnit.run",
            return_value={
                "trajectory": Path("file.nc"),
                "checkpoint": Path("chk.chk"),
            },
        ),
        mock.patch(
            "openfe.protocols.openmm_afe.ahfe_units.AHFEVacuumSimUnit.run",
            return_value={
                "trajectory": Path("file.nc"),
                "checkpoint": Path("chk.chk"),
            },
        ),
        mock.patch(
            "openfe.protocols.openmm_afe.ahfe_units.AHFESolventAnalysisUnit.run",
            return_value={"foo": "bar"},
        ),
        mock.patch(
            "openfe.protocols.openmm_afe.ahfe_units.AHFEVacuumAnalysisUnit.run",
            return_value={"foo": "bar"},
        ),
    ):
        dagres = gufe.protocols.execute_DAG(
            benzene_solvation_dag,
            shared_basedir=tmpdir,
            scratch_basedir=tmpdir,
            keep_shared=True,
        )

    protocol = openmm_afe.AbsoluteSolvationProtocol(
        settings=openmm_afe.AbsoluteSolvationProtocol.default_settings(),
    )

    res = protocol.gather([dagres])

    assert isinstance(res, openmm_afe.AbsoluteSolvationProtocolResult)


def test_unit_tagging(benzene_solvation_dag, tmpdir):
    # test that executing the units includes correct gen and repeat info

    dag_units = benzene_solvation_dag.protocol_units

    with (
        mock.patch(
            "openfe.protocols.openmm_afe.ahfe_units.AHFESolventSetupUnit.run",
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
            "openfe.protocols.openmm_afe.ahfe_units.AHFEVacuumSetupUnit.run",
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
            "openfe.protocols.openmm_afe.base_afe_units.np.load",
            return_value=np.zeros(100),
        ),
        mock.patch(
            "openfe.protocols.openmm_afe.base_afe_units.deserialize",
            return_value="foo",
        ),
        mock.patch(
            "openfe.protocols.openmm_afe.ahfe_units.AHFESolventSimUnit.run",
            return_value={
                "trajectory": Path("file.nc"),
                "checkpoint": Path("chk.chk"),
            },
        ),
        mock.patch(
            "openfe.protocols.openmm_afe.ahfe_units.AHFEVacuumSimUnit.run",
            return_value={
                "trajectory": Path("file.nc"),
                "checkpoint": Path("chk.chk"),
            },
        ),
        mock.patch(
            "openfe.protocols.openmm_afe.ahfe_units.AHFESolventAnalysisUnit.run",
            return_value={"foo": "bar"},
        ),
        mock.patch(
            "openfe.protocols.openmm_afe.ahfe_units.AHFEVacuumAnalysisUnit.run",
            return_value={"foo": "bar"},
        ),
    ):
        for phase in ["solvent", "vacuum"]:
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
    def protocolresult(self, afe_solv_transformation_json):
        d = json.loads(afe_solv_transformation_json, cls=gufe.tokenization.JSON_HANDLER.decoder)

        pr = openfe.ProtocolResult.from_dict(d["protocol_result"])

        return pr

    def test_reload_protocol_result(self, afe_solv_transformation_json):
        d = json.loads(afe_solv_transformation_json, cls=gufe.tokenization.JSON_HANDLER.decoder)

        pr = openmm_afe.AbsoluteSolvationProtocolResult.from_dict(d["protocol_result"])

        assert pr

    def test_get_estimate(self, protocolresult):
        est = protocolresult.get_estimate()

        assert est
        assert est.m == pytest.approx(-2.47, abs=0.5)
        assert isinstance(est, offunit.Quantity)
        assert est.is_compatible_with(offunit.kilojoule_per_mole)

    def test_get_uncertainty(self, protocolresult):
        est = protocolresult.get_uncertainty()

        assert est
        assert est.m == pytest.approx(0.2, abs=0.2)
        assert isinstance(est, offunit.Quantity)
        assert est.is_compatible_with(offunit.kilojoule_per_mole)

    def test_get_individual(self, protocolresult):
        inds = protocolresult.get_individual_estimates()

        assert isinstance(inds, dict)
        assert isinstance(inds["solvent"], list)
        assert isinstance(inds["vacuum"], list)
        assert len(inds["solvent"]) == len(inds["vacuum"]) == 3
        for e, u in itertools.chain(inds["solvent"], inds["vacuum"]):
            assert e.is_compatible_with(offunit.kilojoule_per_mole)
            assert u.is_compatible_with(offunit.kilojoule_per_mole)

    @pytest.mark.parametrize("key", ["solvent", "vacuum"])
    def test_get_forwards_etc(self, key, protocolresult):
        far = protocolresult.get_forward_and_reverse_energy_analysis()

        assert isinstance(far, dict)
        assert isinstance(far[key], list)
        far1 = far[key][0]
        assert isinstance(far1, dict)

        for k in ["fractions", "forward_DGs", "forward_dDGs", "reverse_DGs", "reverse_dDGs"]:
            assert k in far1

            if k == "fractions":
                assert isinstance(far1[k], np.ndarray)

    @pytest.mark.parametrize("key", ["solvent", "vacuum"])
    def test_get_frwd_reverse_none_return(self, key, protocolresult):
        # fetch the first result of type key
        data = [i for i in protocolresult.data[key].values()][0][0]
        # set the output to None
        data.outputs["forward_and_reverse_energies"] = None

        # now fetch the analysis results and expect a warning
        wmsg = f"were found in the forward and reverse dictionaries of the repeats of the {key}"
        with pytest.warns(UserWarning, match=wmsg):
            protocolresult.get_forward_and_reverse_energy_analysis()

    @pytest.mark.parametrize("key", ["solvent", "vacuum"])
    def test_get_overlap_matrices(self, key, protocolresult):
        ovp = protocolresult.get_overlap_matrices()

        assert isinstance(ovp, dict)
        assert isinstance(ovp[key], list)
        assert len(ovp[key]) == 3

        ovp1 = ovp[key][0]
        assert isinstance(ovp1["matrix"], np.ndarray)
        assert ovp1["matrix"].shape == (14, 14)

    @pytest.mark.parametrize("key", ["solvent", "vacuum"])
    def test_get_replica_transition_statistics(self, key, protocolresult):
        rpx = protocolresult.get_replica_transition_statistics()

        assert isinstance(rpx, dict)
        assert isinstance(rpx[key], list)
        assert len(rpx[key]) == 3
        rpx1 = rpx[key][0]
        assert "eigenvalues" in rpx1
        assert "matrix" in rpx1
        assert rpx1["eigenvalues"].shape == (14,)
        assert rpx1["matrix"].shape == (14, 14)

    @pytest.mark.parametrize("key", ["solvent", "vacuum"])
    def test_equilibration_iterations(self, key, protocolresult):
        eq = protocolresult.equilibration_iterations()

        assert isinstance(eq, dict)
        assert isinstance(eq[key], list)
        assert len(eq[key]) == 3
        assert all(isinstance(v, float) for v in eq[key])

    @pytest.mark.parametrize("key", ["solvent", "vacuum"])
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
