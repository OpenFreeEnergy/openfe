# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""OpenMM Equilibrium Binding AFE Protocol --- :mod:`openfe.protocols.openmm_afe.equil_binding_afe_method`
==========================================================================================================

This module implements the necessary methodology tooling to run calculate an
absolute binding free energy using OpenMM tools and one of the following
alchemical sampling methods:

* Hamiltonian Replica Exchange
* Self-adjusted mixture sampling
* Independent window sampling

Current limitations
-------------------
* Alchemical species with a net charge are not currently supported.
* Disapearing molecules are only allowed in state A.
* Only small molecules are allowed to act as alchemical molecules.

Acknowledgements
----------------
* This Protocol re-implements components from
  `Yank <https://github.com/choderalab/yank>`_.

"""
import itertools
import logging
import pathlib
import uuid
import warnings
from collections import defaultdict
from typing import Any, Iterable, Optional, Union

import gufe
import MDAnalysis as mda
import numpy as np
import numpy.typing as npt
from gufe import (
    ChemicalSystem,
    ProteinComponent,
    SmallMoleculeComponent,
    SolventComponent,
    settings,
)
from gufe.components import Component
from openfe.due import Doi, due
from openfe.protocols.openmm_afe.equil_afe_settings import (
    AbsoluteBindingSettings,
    AlchemicalSettings,
    BoreschRestraintSettings,
    DistanceRestraintSettings,
    FlatBottomRestraintSettings,
    IntegratorSettings,
    LambdaSettings,
    ABFEPreEquilOuputSettings,
    MDSimulationSettings,
    MultiStateOutputSettings,
    MultiStateSimulationSettings,
    OpenFFPartialChargeSettings,
    OpenMMEngineSettings,
    OpenMMSolvationSettings,
    SettingsBaseModel,
)
from openfe.protocols.openmm_utils import settings_validation, system_validation
from openfe.protocols.restraint_utils import geometry
from openfe.protocols.restraint_utils.geometry.boresch import BoreschRestraintGeometry
from openfe.protocols.restraint_utils.openmm import omm_restraints
from openfe.protocols.restraint_utils.openmm.omm_restraints import BoreschRestraint
from openfe.utils import log_system_probe
from openff.units import unit as offunit
from openff.units import Quantity
from openff.units.openmm import ensure_quantity, from_openmm, to_openmm
from openmm import System
from openmm import unit as ommunit
from openmm.app import Topology as omm_topology
from openmmtools import multistate
from openmmtools.states import GlobalParameterState, ThermodynamicState
from rdkit import Chem

from .base import BaseAbsoluteUnit

due.cite(
    Doi("10.5281/zenodo.596504"),
    description="Yank",
    path="openfe.protocols.openmm_afe.equil_binding_afe_method",
    cite_module=True,
)

due.cite(
    Doi("10.5281/zenodo.596622"),
    description="OpenMMTools",
    path="openfe.protocols.openmm_afe.equil_binding_afe_method",
    cite_module=True,
)

due.cite(
    Doi("10.1371/journal.pcbi.1005659"),
    description="OpenMM",
    path="openfe.protocols.openmm_afe.equil_binding_afe_method",
    cite_module=True,
)


logger = logging.getLogger(__name__)


class AbsoluteBindingProtocolResult(gufe.ProtocolResult):
    """Dict-like container for the output of a AbsoluteBindingProtocol"""

    def __init__(self, **data):
        super().__init__(**data)
        # TODO: Detect when we have extensions and stitch these together?
        if any(
            len(pur_list) > 2
            for pur_list in itertools.chain(
                self.data["solvent"].values(), self.data["complex"].values()
            )
        ):
            raise NotImplementedError("Can't stitch together results yet")

    def get_individual_estimates(
        self,
    ) -> dict[str, list[tuple[Quantity, Quantity]]]:
        """
        Get the individual estimate of the free energies.

        Returns
        -------
        dGs : dict[str, list[tuple[openff.units.Quantity, openff.units.Quantity]]]
          A dictionary, keyed `solvent`, `complex`, and 'standard_state'
          representing each portion of the thermodynamic cycle,
          with lists of tuples containing the individual free energy
          estimates and, for 'solvent' and 'complex', the associated MBAR
          uncertainties for each repeat of that simulation type.

        Notes
        -----
        * Standard state correction has no error and so will return a value
          of 0.
        """
        complex_dGs = []
        correction_dGs = []
        solv_dGs = []

        for pus in self.data["complex"].values():
            complex_dGs.append(
                (pus[0].outputs["unit_estimate"], pus[0].outputs["unit_estimate_error"])
            )
            correction_dGs.append(
                (
                    pus[0].outputs["standard_state_correction"],
                    0 * offunit.kilocalorie_per_mole,  # correction has no error
                )
            )

        for pus in self.data["solvent"].values():
            solv_dGs.append(
                (pus[0].outputs["unit_estimate"], pus[0].outputs["unit_estimate_error"])
            )

        return {
            "solvent": solv_dGs,
            "complex": complex_dGs,
            "standard_state": correction_dGs,
        }

    def get_estimate(self) -> Quantity:
        """Get the binding free energy estimate for this calculation.

        Returns
        -------
        dG : openff.units.Quantity
          The binding free energy. This is a Quantity defined with units.
        """

        def _get_average(estimates):
            # Get the unit value of the first value in the estimates
            u = estimates[0][0].u
            # Loop through estimates and get the free energy values
            # in the unit of the first estimate
            dGs = [i[0].to(u).m for i in estimates]

            return np.average(dGs) * u

        individual_estimates = self.get_individual_estimates()
        complex_dG = _get_average(individual_estimates["complex"])
        solv_dG = _get_average(individual_estimates["solvent"])
        standard_state_dG = _get_average(individual_estimates["standard_state"])

        return -(complex_dG + standard_state_dG) + solv_dG

    def get_uncertainty(self) -> Quantity:
        """Get the binding free energy error for this calculation.

        Returns
        -------
        err : openff.units.Quantity
          The standard deviation between estimates of the binding free
          energy. This is a Quantity defined with units.
        """

        def _get_stdev(estimates):
            # Get the unit value of the first value in the estimates
            u = estimates[0][0].u
            # Loop through estimates and get the free energy values
            # in the unit of the first estimate
            dGs = [i[0].to(u).m for i in estimates]

            return np.std(dGs) * u

        individual_estimates = self.get_individual_estimates()
        complex_err = _get_stdev(individual_estimates["complex"])
        solv_err = _get_stdev(individual_estimates["solvent"])
        standard_state_err = _get_stdev(individual_estimates["standard_state"])

        # return the combined error
        return np.sqrt(complex_err**2 + solv_err**2 + standard_state_err**2)

    def get_forward_and_reverse_energy_analysis(
        self,
    ) -> dict[str, list[Optional[dict[str, Union[npt.NDArray, Quantity]]]]]:
        """
        Get the reverse and forward analysis of the free energies.

        Returns
        -------
        forward_reverse : dict[str, list[Optional[dict[str, Union[npt.NDArray, openff.units.Quantity]]]]]
            A dictionary, keyed `solvent` and `complex` for each leg of the
            thermodynamic cycle which each contain a list of dictionaries
            containing the forward and reverse analysis of each repeat
            of that simulation type.

            The forward and reverse analysis dictionaries contain:
              - `fractions`: npt.NDArray
                  The fractions of data used for the estimates
              - `forward_DGs`, `reverse_DGs`: openff.units.Quantity
                  The forward and reverse estimates for each fraction of data
              - `forward_dDGs`, `reverse_dDGs`: openff.units.Quantity
                  The forward and reverse estimate uncertainty for each
                  fraction of data.

            If one of the cycle leg list entries is ``None``, this indicates
            that the analysis could not be carried out for that repeat. This
            is most likely caused by MBAR convergence issues when attempting to
            calculate free energies from too few samples.

        Raises
        ------
        UserWarning
          * If any of the forward and reverse dictionaries are ``None`` in a
            given thermodynamic cycle leg.
        """

        forward_reverse: dict[
            str, list[Optional[dict[str, Union[npt.NDArray, Quantity]]]]
        ] = {}

        for key in ["solvent", "complex"]:
            forward_reverse[key] = [
                pus[0].outputs["forward_and_reverse_energies"]
                for pus in self.data[key].values()
            ]

            if None in forward_reverse[key]:
                wmsg = (
                    "One or more ``None`` entries were found in the forward "
                    f"and reverse dictionaries of the repeats of the {key} "
                    "calculations. This is likely caused by an MBAR convergence "
                    "failure caused by too few independent samples when "
                    "calculating the free energies of the 10% timeseries slice."
                )
                warnings.warn(wmsg)

        return forward_reverse

    def get_overlap_matrices(self) -> dict[str, list[dict[str, npt.NDArray]]]:
        """
        Get a the MBAR overlap estimates for all legs of the simulation.

        Returns
        -------
        overlap_stats : dict[str, list[dict[str, npt.NDArray]]]
          A dictionary with keys `solvent` and `complex` for each
          leg of the thermodynamic cycle, which each containing a
          list of dictionaries with the MBAR overlap estimates of
          each repeat of that simulation type.

          The underlying MBAR dictionaries contain the following keys:
            * ``scalar``: One minus the largest nontrivial eigenvalue
            * ``eigenvalues``: The sorted (descending) eigenvalues of the
              overlap matrix
            * ``matrix``: Estimated overlap matrix of observing a sample from
              state i in state j
        """
        # Loop through and get the repeats and get the matrices
        overlap_stats: dict[str, list[dict[str, npt.NDArray]]] = {}

        for key in ["solvent", "complex"]:
            overlap_stats[key] = [
                pus[0].outputs["unit_mbar_overlap"] for pus in self.data[key].values()
            ]

        return overlap_stats

    def get_replica_transition_statistics(
        self,
    ) -> dict[str, list[dict[str, npt.NDArray]]]:
        """
        Get the replica exchange transition statistics for all
        legs of the simulation.

        Note
        ----
        This is currently only available in cases where a replica exchange
        simulation was run.

        Returns
        -------
        repex_stats : dict[str, list[dict[str, npt.NDArray]]]
          A dictionary with keys `solvent` and `complex` for each
          leg of the thermodynamic cycle, which each containing
          a list of dictionaries containing the replica transition
          statistics for each repeat of that simulation type.

          The replica transition statistics dictionaries contain the following:
            * ``eigenvalues``: The sorted (descending) eigenvalues of the
              lambda state transition matrix
            * ``matrix``: The transition matrix estimate of a replica switching
              from state i to state j.
        """
        repex_stats: dict[str, list[dict[str, npt.NDArray]]] = {}
        try:
            for key in ["solvent", "complex"]:
                repex_stats[key] = [
                    pus[0].outputs["replica_exchange_statistics"]
                    for pus in self.data[key].values()
                ]
        except KeyError:
            errmsg = (
                "Replica exchange statistics were not found, "
                "did you run a repex calculation?"
            )
            raise ValueError(errmsg)

        return repex_stats

    def get_replica_states(self) -> dict[str, list[npt.NDArray]]:
        """
        Get the timeseries of replica states for all simulation legs.

        Returns
        -------
        replica_states : dict[str, list[npt.NDArray]]
          Dictionary keyed `solvent` and `complex` for each leg of
          the thermodynamic cycle, with lists of replica states
          timeseries for each repeat of that simulation type.
        """
        replica_states: dict[str, list[npt.NDArray]] = {"solvent": [], "complex": []}

        def is_file(filename: str):
            p = pathlib.Path(filename)

            if not p.exists():
                errmsg = f"File could not be found {p}"
                raise ValueError(errmsg)

            return p

        def get_replica_state(nc, chk):
            nc = is_file(nc)
            dir_path = nc.parents[0]
            chk = is_file(dir_path / chk).name

            reporter = multistate.MultiStateReporter(
                storage=nc, checkpoint_storage=chk, open_mode="r"
            )

            retval = np.asarray(reporter.read_replica_thermodynamic_states())
            reporter.close()

            return retval

        for key in ["solvent", "complex"]:
            for pus in self.data[key].values():
                states = get_replica_state(
                    pus[0].outputs["nc"],
                    pus[0].outputs["last_checkpoint"],
                )
                replica_states[key].append(states)

        return replica_states

    def equilibration_iterations(self) -> dict[str, list[float]]:
        """
        Get the number of equilibration iterations for each simulation.

        Returns
        -------
        equilibration_lengths : dict[str, list[float]]
          Dictionary keyed `solvent` and `complex` for each leg
          of the thermodynamic cycle, with lists containing the
          number of equilibration iterations for each repeat
          of that simulation type.
        """
        equilibration_lengths: dict[str, list[float]] = {}

        for key in ["solvent", "complex"]:
            equilibration_lengths[key] = [
                pus[0].outputs["equilibration_iterations"]
                for pus in self.data[key].values()
            ]

        return equilibration_lengths

    def production_iterations(self) -> dict[str, list[float]]:
        """
        Get the number of production iterations for each simulation.
        Returns the number of uncorrelated production samples for each
        repeat of the calculation.

        Returns
        -------
        production_lengths : dict[str, list[float]]
          Dictionary keyed `solvent` and `complex` for each leg of the
          thermodynamic cycle, with lists with the number
          of production iterations for each repeat of that simulation
          type.
        """
        production_lengths: dict[str, list[float]] = {}

        for key in ["solvent", "complex"]:
            production_lengths[key] = [
                pus[0].outputs["production_iterations"]
                for pus in self.data[key].values()
            ]

        return production_lengths

    def restraint_geometries(self) -> list[BoreschRestraintGeometry]:
        """
        Get a list of the restraint geometries for the
        complex simulations. These define the atoms that have
        been restrained in the system.

        Returns
        -------
        geometries : list[dict[str, Any]]
          A list of dictionaries containing the details of the atoms
          in the system that are involved in the restraint.
        """
        geometries = [
            BoreschRestraintGeometry.model_validate(pus[0].outputs["restraint_geometry"])
            for pus in self.data["complex"].values()
        ]

        return geometries

    def selection_indices(self) -> dict[str, list[Optional[npt.NDArray]]]:
        """
        Get the system selection indices used to write PDB and
        trajectory files.

        Returns
        -------
        indices : dict[str, list[npt.NDArray]]
          A dictionary keyed as `complex` and `solvent` for each
          state, each containing a list of NDArrays containing the corresponding
          full system atom indices for each atom written in the production
          trajectory files for each replica.
        """
        indices: dict[str, list[Optional[npt.NDArray]]] = {}

        for key in ["complex", "solvent"]:
            indices[key] = []
            for pus in self.data[key].values():
                indices[key].append(pus[0].outputs["selection_indices"])

        return indices


class AbsoluteBindingProtocol(gufe.Protocol):
    """
    Absolute binding free energy calculations using OpenMM and OpenMMTools.

    See Also
    --------
    :mod:`openfe.protocols`
    :class:`openfe.protocols.openmm_afe.AbsoluteBindingSettings`
    :class:`openfe.protocols.openmm_afe.AbsoluteBindingProtocolResult`
    :class:`openfe.protocols.openmm_afe.AbsoluteBindingSolventUnit`
    :class:`openfe.protocols.openmm_afe.AbsoluteBindingComplexUnit`
    """

    result_cls = AbsoluteBindingProtocolResult
    _settings_cls = AbsoluteBindingSettings
    _settings: AbsoluteBindingSettings

    @classmethod
    def _default_settings(cls):
        """A dictionary of initial settings for this creating this Protocol

        These settings are intended as a suitable starting point for creating
        an instance of this protocol.  It is recommended, however that care is
        taken to inspect and customize these before performing a Protocol.

        Returns
        -------
        Settings
          a set of default settings
        """
        return AbsoluteBindingSettings(
            protocol_repeats=3,
            forcefield_settings=settings.OpenMMSystemGeneratorFFSettings(),
            thermo_settings=settings.ThermoSettings(
                temperature=298.15 * offunit.kelvin,
                pressure=1 * offunit.bar,
            ),
            alchemical_settings=AlchemicalSettings(),
            # fmt: off
            solvent_lambda_settings=LambdaSettings(
                lambda_elec=[
                    0.0, 0.25, 0.5, 0.75, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                ],
                lambda_vdw=[
                    0.0, 0.0, 0.0, 0.0, 0.0,
                    0.12, 0.24, 0.36, 0.48, 0.6, 0.7, 0.77, 0.85, 1.0
                ],
                lambda_restraints=[
                    0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                ],
            ),
            complex_lambda_settings=LambdaSettings(
                lambda_elec=[
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.00, 1.0, 1.00, 1.0, 1.00, 1.0, 1.00, 1.0
                ],
                lambda_vdw=[
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0
                ],
                lambda_restraints=[
                    0.0, 0.2, 0.4, 0.6, 0.8, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.00, 1.0, 1.00, 1.0, 1.00, 1.0, 1.00, 1.0
                ],
            ),
            # fmt: on
            partial_charge_settings=OpenFFPartialChargeSettings(),
            complex_solvation_settings=OpenMMSolvationSettings(),
            solvent_solvation_settings=OpenMMSolvationSettings(),
            engine_settings=OpenMMEngineSettings(),
            integrator_settings=IntegratorSettings(),
            restraint_settings=BoreschRestraintSettings(),
            solvent_equil_simulation_settings=MDSimulationSettings(
                equilibration_length_nvt=0.1 * offunit.nanosecond,
                equilibration_length=0.2 * offunit.nanosecond,
                production_length=0.5 * offunit.nanosecond,
            ),
            solvent_equil_output_settings=ABFEPreEquilOuputSettings(),
            solvent_simulation_settings=MultiStateSimulationSettings(
                n_replicas=14,
                equilibration_length=1.0 * offunit.nanosecond,
                production_length=10.0 * offunit.nanosecond,
            ),
            solvent_output_settings=MultiStateOutputSettings(
                output_structure="alchemical_system.pdb",
                output_filename="solvent.nc",
                checkpoint_storage_filename="solvent_checkpoint.nc",
            ),
            complex_equil_simulation_settings=MDSimulationSettings(
                equilibration_length_nvt=0.25 * offunit.nanosecond,
                equilibration_length=0.5 * offunit.nanosecond,
                production_length=5.0 * offunit.nanosecond,
            ),
            complex_equil_output_settings=ABFEPreEquilOuputSettings(),
            complex_simulation_settings=MultiStateSimulationSettings(
                n_replicas=30,
                equilibration_length=1 * offunit.nanosecond,
                production_length=10.0 * offunit.nanosecond,
            ),
            complex_output_settings=MultiStateOutputSettings(
                output_structure="alchemical_system.pdb",
                output_filename="complex.nc",
                checkpoint_storage_filename="complex_checkpoint.nc",
            ),
        )

    @staticmethod
    def _validate_endstates(
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
    ) -> None:
        """
        A binding transformation is defined (in terms of gufe components)
        as starting from one or more ligands with one protein and solvent,
        that then ends up in a state with one less ligand.

        Parameters
        ----------
        stateA : ChemicalSystem
          The chemical system of end state A
        stateB : ChemicalSystem
          The chemical system of end state B

        Raises
        ------
        ValueError
          If stateA & stateB do not contain a ProteinComponent.
          If stateA & stateB do not contain a SolventComponent.
          If stateA has more than one unique Component.
          If the stateA unique Component is not a SmallMoleculeComponent.
          If stateB contains any unique Components.
          If the alchemical species is charged.
        """
        if not (
            stateA.contains(ProteinComponent) and stateB.contains(ProteinComponent)
        ):
            errmsg = "No ProteinComponent found"
            raise ValueError(errmsg)

        if not (
            stateA.contains(SolventComponent) and stateB.contains(SolventComponent)
        ):
            errmsg = "No SolventComponent found"
            raise ValueError(errmsg)

        # Needs gufe 1.3
        diff = stateA.component_diff(stateB)
        if len(diff[0]) != 1:
            errmsg = (
                "Only one alchemical species is supported. "
                f"Number of unique components found in stateA: {len(diff[0])}."
            )
            raise ValueError(errmsg)

        if not isinstance(diff[0][0], SmallMoleculeComponent):
            errmsg = (
                "Only dissapearing small molecule components "
                "are supported by this protocol. "
                f"Found a {type(diff[0][0])}"
            )
            raise ValueError(errmsg)

        # Check that the state A unique isn't charged
        if diff[0][0].total_charge != 0:
            errmsg = (
                "Charged alchemical molecules are not currently "
                "supported for solvation free energies. "
                f"Molecule total charge: {diff[0][0].total_charge}."
            )
            raise ValueError(errmsg)

        # If there are any alchemical Components in state B
        if len(diff[1]) > 0:
            errmsg = "Components appearing in state B are not currently supported"
            raise ValueError(errmsg)

    @staticmethod
    def _validate_lambda_schedule(
        lambda_settings: LambdaSettings,
        simulation_settings: MultiStateSimulationSettings,
    ) -> None:
        """
        Checks that the lambda schedule is set up correctly.

        Parameters
        ----------
        lambda_settings : LambdaSettings
          the lambda schedule Settings
        simulation_settings : MultiStateSimulationSettings
          the settings for either the complex or solvent phase

        Raises
        ------
        ValueError
          If the number of lambda windows differs for electrostatics, sterics,
          and restraints.
          If the number of replicas does not match the number of lambda windows.
          If there are states with naked charges.
        """

        lambda_elec = lambda_settings.lambda_elec
        lambda_vdw = lambda_settings.lambda_vdw
        lambda_restraints = lambda_settings.lambda_restraints
        n_replicas = simulation_settings.n_replicas

        # Ensure that all lambda components have equal amount of windows
        lambda_components = [lambda_vdw, lambda_elec, lambda_restraints]
        it = iter(lambda_components)
        the_len = len(next(it))
        if not all(len(l) == the_len for l in it):
            errmsg = (
                "Components elec, vdw, and restraints must have equal amount"
                f" of lambda windows. Got {len(lambda_elec)} elec lambda"
                f" windows, {len(lambda_vdw)} vdw lambda windows, and"
                f"{len(lambda_restraints)} restraints lambda windows."
            )
            raise ValueError(errmsg)

        # Ensure that number of overall lambda windows matches number of lambda
        # windows for individual components
        if n_replicas != len(lambda_vdw):
            errmsg = (
                f"Number of replicas {n_replicas} does not equal the"
                f" number of lambda windows {len(lambda_vdw)}"
            )
            raise ValueError(errmsg)

        # Check if there are no lambda windows with naked charges
        for inx, lam in enumerate(lambda_elec):
            if lam < 1 and lambda_vdw[inx] == 1:
                errmsg = (
                    "There are states along this lambda schedule "
                    "where there are atoms with charges but no LJ "
                    f"interactions: lambda {inx}: "
                    f"elec {lam} vdW {lambda_vdw[inx]}"
                )
                raise ValueError(errmsg)

    def _validate(
        self,
        *,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: Optional[
            Union[gufe.ComponentMapping, list[gufe.ComponentMapping]]
        ] = None,
        extends: Optional[gufe.ProtocolDAGResult] = None,
    ):
        # Check we're not extending
        if extends is not None:
            # This technically should be NotImplementedError
            # but gufe.Protocol.validate calls `_validate` wrapped around an
            # except for NotImplementedError, so we can't raise it here
            raise ValueError("Can't extend simulations yet")

        # Check we're not using a mapping, since we're not doing anything with it
        if mapping is not None:
            wmsg = "A mapping was passed but is not used by this Protocol."
            warnings.warn(wmsg)

        # Validate the end states & alchemical components
        self._validate_endstates(stateA, stateB)

        # Validate the complex lambda schedule
        self._validate_lambda_schedule(
            self.settings.complex_lambda_settings,
            self.settings.complex_simulation_settings,
        )

        # If the complex restraints schedule is all zero, it might be bad
        # but we don't dissallow it.
        if all(
            [
                i == 0
                for i in self.settings.complex_lambda_settings.lambda_restraints
            ]
        ):
            wmsg = (
                "No restraints are being applied in the complex phase, "
                "this will likely lead to problematic results."
            )
            warnings.warn(wmsg)

        # Validate the solvent lambda schedule
        self._validate_lambda_schedule(
            self.settings.solvent_lambda_settings,
            self.settings.solvent_simulation_settings,
        )

        # If the solvent restraints schedule is all zero, it was likely
        # copied from the complex schedule. In this case we just ignore
        # the values and let the user know.
        # P.S. we don't need to change the settings at this point
        # the list gets popped out later in the SolventUnit, because we
        # don't have a restraint parameter state.

        if any(
            [
                i != 0
                for i in self.settings.solvent_lambda_settings.lambda_restraints
            ]
        ):
            wmsg = (
                "There is an attempt to add restraints in the solvent "
                "phase. This protocol does not apply restraints in the "
                "solvent phase. These restraint lambda values will be ignored."
            )
            warnings.warn(wmsg)

        # Check nonbond & solvent compatibility
        nonbonded_method = self.settings.forcefield_settings.nonbonded_method
        # Use the more complete system validation solvent checks
        system_validation.validate_solvent(stateA, nonbonded_method)

        # Validate solvation settings
        settings_validation.validate_openmm_solvation_settings(
            self.settings.solvent_solvation_settings
        )
        settings_validation.validate_openmm_solvation_settings(
            self.settings.complex_solvation_settings
        )

    def _create(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: Optional[
            Union[gufe.ComponentMapping, list[gufe.ComponentMapping]]
        ] = None,
        extends: Optional[gufe.ProtocolDAGResult] = None,
    ) -> list[gufe.ProtocolUnit]:
        # Validate inputs
        self.validate(stateA=stateA, stateB=stateB, mapping=mapping, extends=extends)

        # Get the alchemical components
        alchem_comps = system_validation.get_alchemical_components(
            stateA,
            stateB,
        )

        # Get the name of the alchemical species
        alchname = alchem_comps["stateA"][0].name

        # Create list units for complex and solvent transforms

        solvent_units = [
            AbsoluteBindingSolventUnit(
                protocol=self,
                stateA=stateA,
                stateB=stateB,
                alchemical_components=alchem_comps,
                generation=0,
                repeat_id=int(uuid.uuid4()),
                name=(
                    f"Absolute Binding, {alchname} solvent leg: "
                    f"repeat {i} generation 0"
                ),
            )
            for i in range(self.settings.protocol_repeats)
        ]

        complex_units = [
            AbsoluteBindingComplexUnit(
                protocol=self,
                stateA=stateA,
                stateB=stateB,
                alchemical_components=alchem_comps,
                generation=0,
                repeat_id=int(uuid.uuid4()),
                name=(
                    f"Absolute Binding, {alchname} complex leg: "
                    f"repeat {i} generation 0"
                ),
            )
            for i in range(self.settings.protocol_repeats)
        ]

        return solvent_units + complex_units

    def _gather(
        self, protocol_dag_results: Iterable[gufe.ProtocolDAGResult]
    ) -> dict[str, dict[str, Any]]:
        # result units will have a repeat_id and generation
        # first group according to repeat_id
        unsorted_solvent_repeats = defaultdict(list)
        unsorted_complex_repeats = defaultdict(list)
        for d in protocol_dag_results:
            pu: gufe.ProtocolUnitResult
            for pu in d.protocol_unit_results:
                if not pu.ok():
                    continue
                if pu.outputs["simtype"] == "solvent":
                    unsorted_solvent_repeats[pu.outputs["repeat_id"]].append(pu)
                else:
                    unsorted_complex_repeats[pu.outputs["repeat_id"]].append(pu)

        repeats: dict[str, dict[str, list[gufe.ProtocolUnitResult]]] = {
            "solvent": {},
            "complex": {},
        }
        for k, v in unsorted_solvent_repeats.items():
            repeats["solvent"][str(k)] = sorted(
                v, key=lambda x: x.outputs["generation"]
            )

        for k, v in unsorted_complex_repeats.items():
            repeats["complex"][str(k)] = sorted(
                v, key=lambda x: x.outputs["generation"]
            )
        return repeats


class AbsoluteBindingComplexUnit(BaseAbsoluteUnit):
    """
    Protocol Unit for the complex phase of an absolute binding free energy
    """

    def _get_components(self):
        """
        Get the relevant components for a complex transformation.

        Returns
        -------
        alchem_comps : dict[str, Component]
          A list of alchemical components
        solv_comp : SolventComponent
          The SolventComponent of the system
        prot_comp : Optional[ProteinComponent]
          The protein component of the system, if it exists.
        small_mols : dict[SmallMoleculeComponent: OFFMolecule]
          SmallMoleculeComponents to add to the system.
        """
        stateA = self._inputs["stateA"]
        alchem_comps = self._inputs["alchemical_components"]

        solv_comp, prot_comp, small_mols = system_validation.get_components(stateA)
        off_comps = {m: m.to_openff() for m in small_mols}

        # We don't need to check that solv_comp is not None, otherwise
        # an error will have been raised when calling `validate_solvent`
        # in the Protocol's `_create`.
        # Similarly we don't need to check prot_comp
        return alchem_comps, solv_comp, prot_comp, off_comps

    def _handle_settings(self) -> dict[str, SettingsBaseModel]:
        """
        Extract the relevant settings for a complex transformation.

        Returns
        -------
        settings : dict[str, SettingsBaseModel]
          A dictionary with the following entries:
            * forcefield_settings : OpenMMSystemGeneratorFFSettings
            * thermo_settings : ThermoSettings
            * charge_settings : OpenFFPartialChargeSettings
            * solvation_settings : OpenMMSolvationSettings
            * alchemical_settings : AlchemicalSettings
            * lambda_settings : LambdaSettings
            * engine_settings : OpenMMEngineSettings
            * integrator_settings : IntegratorSettings
            * equil_simulation_settings : MDSimulationSettings
            * equil_output_settings : ABFEPreEquilOuputSettings
            * simulation_settings : SimulationSettings
            * output_settings: MultiStateOutputSettings
            * restraint_settings: BaseRestraintSettings
        """
        prot_settings = self._inputs["protocol"].settings

        settings = {}
        settings["forcefield_settings"] = prot_settings.forcefield_settings
        settings["thermo_settings"] = prot_settings.thermo_settings
        settings["charge_settings"] = prot_settings.partial_charge_settings
        settings["solvation_settings"] = prot_settings.complex_solvation_settings
        settings["alchemical_settings"] = prot_settings.alchemical_settings
        settings["lambda_settings"] = prot_settings.complex_lambda_settings
        settings["engine_settings"] = prot_settings.engine_settings
        settings["integrator_settings"] = prot_settings.integrator_settings
        settings["equil_simulation_settings"] = (
            prot_settings.complex_equil_simulation_settings
        )
        settings["equil_output_settings"] = prot_settings.complex_equil_output_settings
        settings["simulation_settings"] = prot_settings.complex_simulation_settings
        settings["output_settings"] = prot_settings.complex_output_settings
        settings["restraint_settings"] = prot_settings.restraint_settings

        settings_validation.validate_timestep(
            settings["forcefield_settings"].hydrogen_mass,
            settings["integrator_settings"].timestep,
        )

        return settings

    @staticmethod
    def _get_mda_universe(
        topology: omm_topology,
        positions: ommunit.Quantity,
        trajectory: Optional[pathlib.Path],
    ) -> mda.Universe:
        """
        Helper method to get a Universe from an openmm Topology,
        and either an input trajectory or a set of positions.

        Parameters
        ----------
        topology : openmm.app.Topology
          An OpenMM Topology that defines the System.
        positions: openmm.unit.Quantity
          The System's current positions.
          Used if a trajectory file is None or is not a file.
        trajectory: pathlib.Path
          A Path to a trajectory file to read positions from.

        Returns
        -------
        mda.Universe
          An MDAnalysis Universe of the System.
        """
        from MDAnalysis.coordinates.memory import MemoryReader

        # If the trajectory file doesn't exist, then we use positions
        if trajectory is not None and trajectory.is_file():
            return mda.Universe(
                topology,
                trajectory,
                topology_format="OPENMMTOPOLOGY",
            )
        else:
            # Positions is an openmm Quantity in nm we need
            # to convert to angstroms
            return mda.Universe(
                topology,
                np.array(positions._value) * 10,
                topology_format="OPENMMTOPOLOGY",
                trajectory_format=MemoryReader,
            )

    @staticmethod
    def _get_idxs_from_residxs(
        topology: omm_topology,
        residxs: list[int],
    ) -> list[int]:
        """
        Helper method to get the a list of atom indices which belong to a list
        of residues.

        Parameters
        ----------
        topology : openmm.app.Topology
          An OpenMM Topology that defines the System.
        residxs : list[int]
          A list of residue numbers who's atoms we should get atom indices.

        Returns
        -------
        atom_ids : list[int]
          A list of atom indices.

        TODO
        ----
        * Check how this works when we deal with virtual sites.
        """
        atom_ids = []

        for r in topology.residues():
            if r.index in residxs:
                atom_ids.extend([at.index for at in r.atoms()])

        return atom_ids

    @staticmethod
    def _get_boresch_restraint(
        universe: mda.Universe,
        guest_rdmol: Chem.Mol,
        guest_atom_ids: list[int],
        host_atom_ids: list[int],
        temperature: Quantity,
        settings: BoreschRestraintSettings,
    ) -> tuple[BoreschRestraintGeometry, BoreschRestraint]:
        """
        Get a Boresch-like restraint Geometry and OpenMM restraint force
        supplier.

        Parameters
        ----------
        universe : mda.Universe
          An MDAnalysis Universe defining the system to get the restraint for.
        guest_rdmol : Chem.Mol
          An RDKit Molecule defining the guest molecule in the system.
        guest_atom_ids: list[int]
          A list of atom indices defining the guest molecule in the universe.
        host_atom_ids : list[int]
          A list of atom indices defining the host molecules in the universe.
        temperature : openff.units.Quantity
          The temperature of the simulation where the restraint will be added.
        settings : BoreschRestraintSettings
          Settings on how the Boresch-like restraint should be defined.

        Returns
        -------
        geom : BoreschRestraintGeometry
          A class defining the Boresch-like restraint.
        restraint : BoreschRestraint
          A factory class for generating Boresch restraints in OpenMM.
        """
        # Take the minimum of the two possible force constants to check against
        frc_const = min(settings.K_thetaA, settings.K_thetaB)

        geom = geometry.boresch.find_boresch_restraint(
            universe=universe,
            guest_rdmol=guest_rdmol,
            guest_idxs=guest_atom_ids,
            host_idxs=host_atom_ids,
            host_selection=settings.host_selection,
            anchor_finding_strategy=settings.anchor_finding_strategy,
            dssp_filter=settings.dssp_filter,
            rmsf_cutoff=settings.rmsf_cutoff,
            host_min_distance=settings.host_min_distance,
            host_max_distance=settings.host_max_distance,
            angle_force_constant=frc_const,
            temperature=temperature,
        )

        restraint = omm_restraints.BoreschRestraint(settings)
        return geom, restraint

    def _add_restraints(
        self,
        system: System,
        topology: omm_topology,
        positions: ommunit.Quantity,
        alchem_comps: dict[str, list[Component]],
        comp_resids: dict[Component, npt.NDArray],
        settings: dict[str, SettingsBaseModel],
    ) -> tuple[
        GlobalParameterState,
        Quantity,
        System,
        geometry.HostGuestRestraintGeometry,
    ]:
        """
        Find and add restraints to the OpenMM System.

        Notes
        -----
        Currently, only Boresch-like restraints are supported.

        Parameters
        ----------
        system : openmm.System
          The System to add the restraint to.
        topology : openmm.app.Topology
          An OpenMM Topology that defines the System.
        positions: openmm.unit.Quantity
          The System's current positions.
          Used if a trajectory file isn't found.
        alchem_comps: dict[str, list[Component]]
          A dictionary with a list of alchemical components
          in both state A and B.
        comp_resids: dict[Component, npt.NDArray]
          A dictionary keyed by each Component in the System
          which contains arrays with the residue indices that is contained
          by that Component.
        settings : dict[str, SettingsBaseModel]
          A dictionary of settings that defines how to find and set
          the restraint.

        Returns
        -------
        restraint_parameter_state : RestraintParameterState
          A RestraintParameterState object that defines the control
          parameter for the restraint.
        correction : openff.units.Quantity
          The standard state correction for the restraint.
        system : openmm.System
          A copy of the System with the restraint added.
        rest_geom : geometry.HostGuestRestraintGeometry
          The restraint Geometry object.
        """
        if self.verbose:
            self.logger.info("Generating restraints")

        # Get the guest rdmol
        guest_rdmol = alchem_comps["stateA"][0].to_rdkit()

        # sanitize the rdmol if possible - warn if you can't
        err = Chem.SanitizeMol(guest_rdmol, catchErrors=True)

        if err:
            msg = "restraint generation: could not sanitize ligand rdmol"
            logger.warning(msg)

        # Get the guest idxs
        # concatenate a list of residue indexes for all alchemical components
        residxs = np.concatenate([comp_resids[key] for key in alchem_comps["stateA"]])

        # get the alchemicical atom ids
        guest_atom_ids = self._get_idxs_from_residxs(topology, residxs)

        # Now get the host idxs
        # We assume this is everything but the alchemical component
        # and the solvent.
        solv_comps = [c for c in comp_resids if isinstance(c, SolventComponent)]
        exclude_comps = [alchem_comps["stateA"]] + solv_comps
        residxs = np.concatenate(
            [v for i, v in comp_resids.items() if i not in exclude_comps]
        )

        host_atom_ids = self._get_idxs_from_residxs(topology, residxs)

        # Finally create an MDAnalysis Universe
        # We try to pass the equilibration production file path through
        # In some cases (debugging / dry runs) this won't be available
        # so we'll default to using input positions.
        univ = self._get_mda_universe(
            topology,
            positions,
            self.shared_basepath
            / settings["equil_output_settings"].production_trajectory_filename,
        )

        if isinstance(settings["restraint_settings"], BoreschRestraintSettings):
            rest_geom, restraint = self._get_boresch_restraint(
                univ,
                guest_rdmol,
                guest_atom_ids,
                host_atom_ids,
                settings["thermo_settings"].temperature,
                settings["restraint_settings"],
            )
        else:
            # TODO turn this into a direction for different restraint types supported?
            raise NotImplementedError("Other restraint types are not yet available")

        if self.verbose:
            self.logger.info(f"restraint geometry is: {rest_geom}")

        # We need a temporary thermodynamic state to add the restraint
        # & get the correction
        thermodynamic_state = ThermodynamicState(
            system,
            temperature=to_openmm(settings["thermo_settings"].temperature),
            pressure=to_openmm(settings["thermo_settings"].pressure),
        )

        # Add the force to the thermodynamic state
        restraint.add_force(
            thermodynamic_state,
            rest_geom,
            controlling_parameter_name="lambda_restraints",
        )
        # Get the standard state correction as a unit.Quantity
        correction = restraint.get_standard_state_correction(
            thermodynamic_state,
            rest_geom,
        )

        # Get the GlobalParameterState for the restraint
        restraint_parameter_state = omm_restraints.RestraintParameterState(
            lambda_restraints=1.0
        )
        return (
            restraint_parameter_state,
            correction,
            thermodynamic_state.system,
            rest_geom,
        )

    def _execute(
        self,
        ctx: gufe.Context,
        **kwargs,
    ) -> dict[str, Any]:
        log_system_probe(logging.INFO, paths=[ctx.scratch])

        outputs = self.run(scratch_basepath=ctx.scratch, shared_basepath=ctx.shared)

        return {
            "repeat_id": self._inputs["repeat_id"],
            "generation": self._inputs["generation"],
            "simtype": "complex",
            **outputs,
        }


class AbsoluteBindingSolventUnit(BaseAbsoluteUnit):
    """
    Protocol Unit for the solvent phase of an absolute binding free energy
    """

    def _get_components(self):
        """
        Get the relevant components for a solvent transformation.

        Returns
        -------
        alchem_comps : dict[str, Component]
          A list of alchemical components
        solv_comp : SolventComponent
          The SolventComponent of the system
        prot_comp : Optional[ProteinComponent]
          The protein component of the system, if it exists.
        small_mols : dict[SmallMoleculeComponent: OFFMolecule]
          SmallMoleculeComponents to add to the system.
        """
        stateA = self._inputs["stateA"]
        alchem_comps = self._inputs["alchemical_components"]

        solv_comp, prot_comp, small_mols = system_validation.get_components(stateA)
        off_comps = {m: m.to_openff() for m in alchem_comps["stateA"]}

        # We don't need to check that solv_comp is not None, otherwise
        # an error will have been raised when calling `validate_solvent`
        # in the Protocol's `_create`.
        # Similarly we don't need to check prot_comp just return None
        return alchem_comps, solv_comp, None, off_comps

    def _handle_settings(self) -> dict[str, SettingsBaseModel]:
        """
        Extract the relevant settings for a solvent transformation.

        Returns
        -------
        settings : dict[str, SettingsBaseModel]
          A dictionary with the following entries:
            * forcefield_settings : OpenMMSystemGeneratorFFSettings
            * thermo_settings : ThermoSettings
            * charge_settings : OpenFFPartialChargeSettings
            * solvation_settings : OpenMMSolvationSettings
            * alchemical_settings : AlchemicalSettings
            * lambda_settings : LambdaSettings
            * engine_settings : OpenMMEngineSettings
            * integrator_settings : IntegratorSettings
            * equil_simulation_settings : MDSimulationSettings
            * equil_output_settings : ABFEPreEquilOuputSettings
            * simulation_settings : MultiStateSimulationSettings
            * output_settings: MultiStateOutputSettings
        """
        prot_settings = self._inputs["protocol"].settings

        settings = {}
        settings["forcefield_settings"] = prot_settings.forcefield_settings
        settings["thermo_settings"] = prot_settings.thermo_settings
        settings["charge_settings"] = prot_settings.partial_charge_settings
        settings["solvation_settings"] = prot_settings.solvent_solvation_settings
        settings["alchemical_settings"] = prot_settings.alchemical_settings
        settings["lambda_settings"] = prot_settings.solvent_lambda_settings
        settings["engine_settings"] = prot_settings.engine_settings
        settings["integrator_settings"] = prot_settings.integrator_settings
        settings["equil_simulation_settings"] = (
            prot_settings.solvent_equil_simulation_settings
        )
        settings["equil_output_settings"] = prot_settings.solvent_equil_output_settings
        settings["simulation_settings"] = prot_settings.solvent_simulation_settings
        settings["output_settings"] = prot_settings.solvent_output_settings

        settings_validation.validate_timestep(
            settings["forcefield_settings"].hydrogen_mass,
            settings["integrator_settings"].timestep,
        )

        return settings

    def _execute(
        self,
        ctx: gufe.Context,
        **kwargs,
    ) -> dict[str, Any]:
        log_system_probe(logging.INFO, paths=[ctx.scratch])

        outputs = self.run(scratch_basepath=ctx.scratch, shared_basepath=ctx.shared)

        return {
            "repeat_id": self._inputs["repeat_id"],
            "generation": self._inputs["generation"],
            "simtype": "solvent",
            **outputs,
        }
