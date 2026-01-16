# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""OpenMM Equilibrium Binding AFE Protocol --- :mod:`openfe.protocols.openmm_afe.equil_binding_afe_method`
==========================================================================================================

This module implements the necessary methodology tooling to calculate an
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
    BaseSolventComponent,
    ChemicalSystem,
    ProteinComponent,
    ProteinMembraneComponent,
    SmallMoleculeComponent,
    SolvatedPDBComponent,
    SolventComponent,
    settings,
)
from gufe.components import Component
from openff.units import Quantity
from openff.units import unit as offunit
from openff.units.openmm import to_openmm
from openmm import System
from openmm import unit as ommunit
from openmm.app import Topology as omm_topology
from openmmtools import multistate
from openmmtools.states import GlobalParameterState, ThermodynamicState
from rdkit import Chem

from openfe.due import Doi, due
from openfe.protocols.openmm_afe.equil_afe_settings import (
    ABFEPreEquilOutputSettings,
    AbsoluteBindingSettings,
    AlchemicalSettings,
    BoreschRestraintSettings,
    IntegratorSettings,
    LambdaSettings,
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
            "standard_state_correction": correction_dGs,
        }

    @staticmethod
    def _add_complex_standard_state_corr(
        complex_dG: list[tuple[Quantity, Quantity]],
        standard_state_dG: list[tuple[Quantity, Quantity]],
    ) -> list[tuple[Quantity, Quantity]]:
        """
        Helper method to combine the
        complex & standard state corrections legs.

        Parameters
        ----------
        complex_dG : list[tuple[openff.units.Quantity, openff.units.Quantity]]
          The individual estimates of the complex leg,
          where the first entry of each tuple is the dG estimate
          and the second entry is the MBAR error.
        standard_state_dG : list[tuple[Quantity, Quantity]]
          The individual standard state corrections for each corresponding
          complex leg. The first entry is the correction, the second
          is an empty error value of 0.

        Returns
        -------
        combined_dG : list[tuple[openff.units.Quantity,openff.units. Quantity]]
          A list of dG estimates & MBAR errors for the combined
          complex & standard state correction of each repeat.

        Notes
        -----
        We assume that both list of items are in the right order.
        """
        combined_dG: list[tuple[Quantity, Quantity]] = []
        for comp, corr in zip(complex_dG, standard_state_dG):
            # No need to convert unit types, since pint takes care of that
            # except that mypy hates it because pint isn't typed properly...
            # No need to add errors since there's just the one
            combined_dG.append((comp[0] + corr[0], comp[1]))  # type: ignore[operator]

        return combined_dG

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
        complex_dG = _get_average(
            self._add_complex_standard_state_corr(
                individual_estimates["complex"],
                individual_estimates["standard_state_correction"],
            )
        )
        solv_dG = _get_average(individual_estimates["solvent"])

        return -complex_dG + solv_dG

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

        complex_err = _get_stdev(
            self._add_complex_standard_state_corr(
                individual_estimates["complex"], individual_estimates["standard_state_correction"]
            )
        )
        solv_err = _get_stdev(individual_estimates["solvent"])

        # return the combined error
        return np.sqrt(complex_err**2 + solv_err**2)

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

        forward_reverse: dict[str, list[Optional[dict[str, Union[npt.NDArray, Quantity]]]]] = {}

        for key in ["solvent", "complex"]:
            forward_reverse[key] = [
                pus[0].outputs["forward_and_reverse_energies"] for pus in self.data[key].values()
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
                    pus[0].outputs["replica_exchange_statistics"] for pus in self.data[key].values()
                ]
        except KeyError:
            errmsg = "Replica exchange statistics were not found, did you run a repex calculation?"
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
                pus[0].outputs["equilibration_iterations"] for pus in self.data[key].values()
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
                pus[0].outputs["production_iterations"] for pus in self.data[key].values()
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
        # fmt: off
        return AbsoluteBindingSettings(
            protocol_repeats=3,
            forcefield_settings=settings.OpenMMSystemGeneratorFFSettings(),
            thermo_settings=settings.ThermoSettings(
                temperature=298.15 * offunit.kelvin,
                pressure=1 * offunit.bar,
            ),
            alchemical_settings=AlchemicalSettings(),
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
            partial_charge_settings=OpenFFPartialChargeSettings(),
            complex_solvation_settings=OpenMMSolvationSettings(
                solvent_padding=1.0 * offunit.nanometer,
            ),
            solvent_solvation_settings=OpenMMSolvationSettings(),
            engine_settings=OpenMMEngineSettings(),
            solvent_integrator_settings=IntegratorSettings(),
            complex_integrator_settings=IntegratorSettings(),
            restraint_settings=BoreschRestraintSettings(),
            solvent_equil_simulation_settings=MDSimulationSettings(
                equilibration_length_nvt=0.1 * offunit.nanosecond,
                equilibration_length=0.2 * offunit.nanosecond,
                production_length=0.5 * offunit.nanosecond,
            ),
            solvent_equil_output_settings=ABFEPreEquilOutputSettings(),
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
            complex_equil_output_settings=ABFEPreEquilOutputSettings(),
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
        # fmt: on

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
          If stateA & stateB do not contain a BaseSolventComponent.
          If stateA has more than one unique Component.
          If the stateA unique Component is not a SmallMoleculeComponent.
          If stateB contains any unique Components.
          If the alchemical species is charged.
        """
        if not (stateA.contains(ProteinComponent) and stateB.contains(ProteinComponent)):
            errmsg = "No ProteinComponent found"
            raise ValueError(errmsg)

        if not (stateA.contains(BaseSolventComponent) and stateB.contains(BaseSolventComponent)):
            errmsg = "No solvent found"
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
        if not all(len(lambda_comp) == the_len for lambda_comp in it):
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
        mapping: Optional[Union[gufe.ComponentMapping, list[gufe.ComponentMapping]]] = None,
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
        if all([i == 0.0 for i in self.settings.complex_lambda_settings.lambda_restraints]):
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

        # If the solvent restraints schedule is not all one, it was likely
        # copied from the complex schedule. In this case we just ignore
        # the values and let the user know.
        # P.S. we don't need to change the settings at this point
        # the list gets popped out later in the SolventUnit, because we
        # don't have a restraint parameter state.

        if any([i != 0.0 for i in self.settings.solvent_lambda_settings.lambda_restraints]):
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

        # Validate the barostat used in combination with the protein component
        system_validation.validate_protein_barostat(stateA, self.settings.complex_integrator_settings.barostat)

        # Validate solvation settings
        settings_validation.validate_openmm_solvation_settings(
            self.settings.solvent_solvation_settings
        )
        settings_validation.validate_openmm_solvation_settings(
            self.settings.complex_solvation_settings
        )

        # Validate integrator things
        settings_validation.validate_timestep(
            self.settings.forcefield_settings.hydrogen_mass,
            self.settings.complex_integrator_settings.timestep,
        )
        settings_validation.validate_timestep(
            self.settings.forcefield_settings.hydrogen_mass,
            self.settings.solvent_integrator_settings.timestep,
        )

    def _create(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: Optional[Union[gufe.ComponentMapping, list[gufe.ComponentMapping]]] = None,
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
                name=(f"Absolute Binding, {alchname} solvent leg: repeat {i} generation 0"),
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
                name=(f"Absolute Binding, {alchname} complex leg: repeat {i} generation 0"),
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
            repeats["solvent"][str(k)] = sorted(v, key=lambda x: x.outputs["generation"])

        for k, v in unsorted_complex_repeats.items():
            repeats["complex"][str(k)] = sorted(v, key=lambda x: x.outputs["generation"])
        return repeats
