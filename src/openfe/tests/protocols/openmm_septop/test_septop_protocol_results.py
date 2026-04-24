# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import itertools
import json
import math
import pathlib
from unittest import mock

import gufe
import mdtraj as md
import numpy as np
import openmm
import openmm.app
import openmm.unit
import pytest
from numpy.testing import assert_allclose
from openff.units import unit as offunit
from openff.units.openmm import ensure_quantity, from_openmm, to_openmm
from openmm import (
    CustomBondForce,
    CustomCompoundBondForce,
    CustomNonbondedForce,
    HarmonicAngleForce,
    HarmonicBondForce,
    MonteCarloBarostat,
    MonteCarloMembraneBarostat,
    NonbondedForce,
    PeriodicTorsionForce,
)
from openmmtools.alchemy import AbsoluteAlchemicalFactory, AlchemicalRegion
from openmmtools.multistate.multistatesampler import MultiStateSampler

import openfe.protocols.openmm_septop
from openfe import ChemicalSystem, SolventComponent
from openfe.protocols.openmm_septop import (
    SepTopComplexRunUnit,
    SepTopComplexSetupUnit,
    SepTopProtocol,
    SepTopProtocolResult,
    SepTopSolventRunUnit,
    SepTopSolventSetupUnit,
)
from openfe.protocols.openmm_septop.base_units import (
    BaseSepTopAnalysisUnit,
    BaseSepTopRunUnit,
    BaseSepTopSetupUnit,
)
from openfe.protocols.openmm_utils.serialization import deserialize
from openfe.protocols.restraint_utils.geometry.boresch import BoreschRestraintGeometry
from openfe.tests.protocols.conftest import compute_energy
from openfe.tests.protocols.openmm_ahfe.test_ahfe_protocol import (
    _assert_num_forces,
    _verify_alchemical_sterics_force_parameters,
)

from .utils import UNIT_TYPES, _get_units


@pytest.fixture
def patcher():
    base_path = "openfe.protocols.openmm_septop.base_units"
    protocol_path = "openfe.protocols.openmm_septop.equil_septop_method"
    with (
        mock.patch(
            f"{protocol_path}.SepTopComplexSetupUnit.run",
            return_value={
                "system": pathlib.Path("system.xml.bz2"),
                "topology": "topology.pdb",
                "standard_state_correction_A": 0 * offunit.kilocalorie_per_mole,
                "standard_state_correction_B": 0 * offunit.kilocalorie_per_mole,
                "restraint_geometry_A": None,
                "restraint_geometry_B": None,
                "selection_indices": np.array(
                    [
                        0,
                    ]
                ),
                "subsampled_pdb_structure": "subsampled.pdb",
            },
        ),
        mock.patch(
            f"{protocol_path}.SepTopSolventSetupUnit.run",
            return_value={
                "system": pathlib.Path("system.xml.bz2"),
                "topology": "topology.pdb",
                "standard_state_correction": 0 * offunit.kilocalorie_per_mole,
                "selection_indices": np.array(
                    [
                        0,
                    ]
                ),
                "subsampled_pdb_structure": "subsampled.pdb",
            },
        ),
        mock.patch(
            f"{protocol_path}.SepTopComplexRunUnit.run",
            return_value={
                "trajectory": "foo.nc",
                "checkpoint": "bar.nc",
            },
        ),
        mock.patch(
            f"{protocol_path}.SepTopSolventRunUnit.run",
            return_value={
                "trajectory": "foo.nc",
                "checkpoint": "bar.nc",
            },
        ),
        mock.patch(
            f"{protocol_path}.SepTopComplexAnalysisUnit.run",
            return_value={"foo": "bar"},
        ),
        mock.patch(
            f"{protocol_path}.SepTopSolventAnalysisUnit.run",
            return_value={"foo": "bar"},
        ),
        mock.patch(
            f"{base_path}.deserialize",
            return_value="foo",
        ),
        mock.patch(
            f"{base_path}.openmm.app.pdbfile.PDBFile",
            return_value="foo",
        ),
    ):
        yield


def test_unit_tagging(benzene_toluene_dag, patcher, tmp_path):
    # test that executing the units includes correct gen and repeat info
    dag_units = benzene_toluene_dag.protocol_units

    for phase in ["solvent", "complex"]:
        setup_results = {}
        sim_results = {}
        analysis_results = {}

        setup_units = _get_units(dag_units, UNIT_TYPES[phase]["setup"])
        sim_units = _get_units(dag_units, UNIT_TYPES[phase]["sim"])
        a_units = _get_units(dag_units, UNIT_TYPES[phase]["analysis"])

        for u in setup_units:
            rid = u.inputs["repeat_id"]
            setup_results[rid] = u.execute(context=gufe.Context(tmp_path, tmp_path))

        for u in sim_units:
            rid = u.inputs["repeat_id"]
            sim_results[rid] = u.execute(
                context=gufe.Context(tmp_path, tmp_path),
                setup=setup_results[rid],
            )

        for u in a_units:
            rid = u.inputs["repeat_id"]
            analysis_results[rid] = u.execute(
                context=gufe.Context(tmp_path, tmp_path),
                setup=setup_results[rid],
                simulation=sim_results[rid],
            )

        for results in [setup_results, sim_results, analysis_results]:
            for ret in results.values():
                assert isinstance(ret, gufe.ProtocolUnitResult)
                assert ret.outputs["generation"] == 0

        assert len(setup_results) == 1
        assert len(sim_results) == 1
        assert len(analysis_results) == 1


def test_gather(benzene_toluene_dag, patcher, tmp_path):
    # check that .gather behaves as expected
    dagres = gufe.protocols.execute_DAG(
        benzene_toluene_dag,
        shared_basedir=tmp_path,
        scratch_basedir=tmp_path,
        keep_shared=True,
    )

    protocol = SepTopProtocol(
        settings=SepTopProtocol.default_settings(),
    )

    res = protocol.gather([dagres])

    assert isinstance(res, openfe.protocols.openmm_septop.SepTopProtocolResult)


class TestProtocolResult:
    @pytest.fixture()
    def protocolresult(self, septop_json):
        d = json.loads(septop_json, cls=gufe.tokenization.JSON_HANDLER.decoder)

        pr = openfe.ProtocolResult.from_dict(d["protocol_result"])

        return pr

    def test_reload_protocol_result(self, septop_json):
        d = json.loads(septop_json, cls=gufe.tokenization.JSON_HANDLER.decoder)

        pr = SepTopProtocolResult.from_dict(d["protocol_result"])

        assert pr

    def test_get_estimate(self, protocolresult):
        est = protocolresult.get_estimate()

        assert est
        assert est.m == pytest.approx(1.6, abs=0.1)
        assert isinstance(est, offunit.Quantity)
        assert est.is_compatible_with(offunit.kilojoule_per_mole)

    def test_get_uncertainty(self, protocolresult):
        est = protocolresult.get_uncertainty()

        assert est.m == pytest.approx(0.0, abs=0.1)
        assert isinstance(est, offunit.Quantity)
        assert est.is_compatible_with(offunit.kilojoule_per_mole)

    def test_get_individual(self, protocolresult):
        inds = protocolresult.get_individual_estimates()

        assert isinstance(inds, dict)
        assert isinstance(inds["solvent"], list)
        assert isinstance(inds["complex"], list)
        assert len(inds["solvent"]) == len(inds["complex"]) == 1
        for e, u in itertools.chain(inds["solvent"], inds["complex"]):
            assert e.is_compatible_with(offunit.kilojoule_per_mole)
            assert u.is_compatible_with(offunit.kilojoule_per_mole)

    def test_get_forwards_etc(self, protocolresult):
        """
        Due to the short simulation times, we expect the frwd/reverse
        analysis of the solvent to be None.
        """
        wmsg = "were found in the forward and reverse dictionaries of the repeats of the solvent"
        with pytest.warns(UserWarning, match=wmsg):
            far = protocolresult.get_forward_and_reverse_energy_analysis()

        assert isinstance(far, dict)
        for key in ["solvent", "complex"]:
            assert isinstance(far[key], list)

        assert far["solvent"][0] is None

        complex_keys = list(far["complex"][0].keys())

        for key in ["fractions", "forward_DGs", "forward_dDGs", "reverse_DGs", "reverse_dDGs"]:
            assert key in complex_keys
            assert len(far["complex"][0][key]) == 10

    @pytest.mark.parametrize("key", ["solvent", "complex"])
    def test_get_overlap_matrices(self, key, protocolresult):
        ovp = protocolresult.get_overlap_matrices()

        assert isinstance(ovp, dict)
        assert isinstance(ovp[key], list)
        assert len(ovp[key]) == 1

        ovp1 = ovp[key][0]
        assert isinstance(ovp1["matrix"], np.ndarray)
        if key == "solvent":
            lambda_nr = 27
        else:
            lambda_nr = 19
        assert ovp1["matrix"].shape == (lambda_nr, lambda_nr)

    @pytest.mark.parametrize("key", ["solvent", "complex"])
    def test_get_replica_transition_statistics(self, key, protocolresult):
        rpx = protocolresult.get_replica_transition_statistics()
        if key == "solvent":
            lambda_nr = 27
        else:
            lambda_nr = 19
        assert isinstance(rpx, dict)
        assert isinstance(rpx[key], list)
        assert len(rpx[key]) == 1
        rpx1 = rpx[key][0]
        assert "eigenvalues" in rpx1
        assert "matrix" in rpx1

        assert rpx1["eigenvalues"].shape == (lambda_nr,)
        assert rpx1["matrix"].shape == (lambda_nr, lambda_nr)

    @pytest.mark.parametrize("key", ["solvent", "complex"])
    def test_equilibration_iterations(self, key, protocolresult):
        eq = protocolresult.equilibration_iterations()

        assert isinstance(eq, dict)
        assert isinstance(eq[key], list)
        assert len(eq[key]) == 1
        assert all(isinstance(v, float) for v in eq[key])

    @pytest.mark.parametrize("key", ["solvent", "complex"])
    def test_production_iterations(self, key, protocolresult):
        prod = protocolresult.production_iterations()

        assert isinstance(prod, dict)
        assert isinstance(prod[key], list)
        assert len(prod[key]) == 1
        assert all(isinstance(v, float) for v in prod[key])

    @pytest.mark.parametrize(
        "key, expected_size",
        [
            ["solvent", 87],
            ["complex", 1868],
        ],
    )
    def test_selection_indices(self, key, protocolresult, expected_size):
        indices = protocolresult.selection_indices()

        assert isinstance(indices, dict)
        assert isinstance(indices[key], list)
        for inds in indices[key]:
            assert isinstance(inds, np.ndarray)
            assert len(inds) == expected_size

    def test_filenotfound_replica_states(self, protocolresult):
        errmsg = "File could not be found"

        with pytest.raises(ValueError, match=errmsg):
            protocolresult.get_replica_states()

    def test_restraint_geometry(self, protocolresult):
        geom = protocolresult.restraint_geometries()
        assert isinstance(geom, tuple)
        assert len(geom) == 2
        assert isinstance(geom[0], list)
        assert isinstance(geom[0][0], BoreschRestraintGeometry)
        assert geom[0][0].guest_atoms == [1779, 1778, 1777]
        assert geom[0][0].host_atoms == [802, 801, 800]
        assert pytest.approx(geom[0][0].r_aA0, abs=0.01) == 0.75 * offunit.nanometer
        assert pytest.approx(geom[0][0].theta_A0, abs=0.01) == 1.95 * offunit.radian
        assert pytest.approx(geom[0][0].theta_B0, abs=0.01) == 1.33 * offunit.radian
        assert pytest.approx(geom[0][0].phi_A0, abs=0.01) == 1.01 * offunit.radian
        assert pytest.approx(geom[0][0].phi_B0, abs=0.01) == -1.24 * offunit.radian
        assert pytest.approx(geom[0][0].phi_C0, abs=0.01) == -1.08 * offunit.radian
