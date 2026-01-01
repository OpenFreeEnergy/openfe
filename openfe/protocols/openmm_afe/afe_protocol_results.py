# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Result classes for the Absolute Free Energy Protocols
=====================================================

This module implements :class:`gufe.ProtocolResult` classes for the absolute
free energy Protocols.

Specifically it implements:
  * AbsoluteBindingProtocolResult
  * AbsoluteSolvationProtocolResult
"""
import itertools
import logging
import pathlib
import warnings
from typing import Optional, Union

import gufe
import numpy as np
import numpy.typing as npt
from openff.units import unit as offunit
from openff.units import Quantity
from openmmtools import multistate

from openfe.protocols.restraint_utils.geometry.boresch import (
    BoreschRestraintGeometry
)


logger = logging.getLogger(__name__)


class AbsoluteProtocolResultMixin:
    bound_state = "solvent"
    unbound_state = "vacuum"

    def __init__(self, **data):
        super().__init__(**data)
        # TODO: Detect when we have extensions and stitch these together?
        if any(
            len(pur_list) > 2
            for pur_list in itertools.chain(
                self.data[self.bound_state].values(), self.data[self.unbound_state].values()
            )
        ):
            raise NotImplementedError("Can't stitch together results yet")

    def get_forward_and_reverse_energy_analysis(
        self,
    ) -> dict[str, list[Optional[dict[str, Union[npt.NDArray, Quantity]]]]]:
        """
        Get the reverse and forward analysis of the free energies.

        Returns
        -------
        forward_reverse : dict[str, list[Optional[dict[str, Union[npt.NDArray, openff.units.Quantity]]]]]
            A dictionary, keyed for each leg of the thermodynamic cycle,
            either ``solvent`` and ``vaccuum` for a solvation free energy or
          ``solvent`` and ``complex`` for a binding free energy,
            with each containing a list of dictionaries containing the forward
            and reverse analysis of each repeat of that simulation type.

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

        for key in [self.bound_state, self.unbound_state]:
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
          A dictionary keyed for each leg of the thermodynamic cycle, either
          ``solvent`` and ``vaccuum` for a solvation free energy or
          ``solvent`` and ``complex`` for a binding free energy,
          with each containing a list of dictionaries with the MBAR overlap
          estimates of each repeat of that simulation type.

          The underlying MBAR dictionaries contain the following keys:
            * ``scalar``: One minus the largest nontrivial eigenvalue
            * ``eigenvalues``: The sorted (descending) eigenvalues of the
              overlap matrix
            * ``matrix``: Estimated overlap matrix of observing a sample from
              state i in state j
        """
        # Loop through and get the repeats and get the matrices
        overlap_stats: dict[str, list[dict[str, npt.NDArray]]] = {}

        for key in [self.bound_state, self.unbound_state]:
            overlap_stats[key] = [
                pus[0].outputs["unit_mbar_overlap"] for pus in self.data[key].values()
            ]

        return overlap_stats

    def get_replica_transition_statistics(self) -> dict[str, list[dict[str, npt.NDArray]]]:
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
          A dictionary with keys for each leg of the thermodynamic cycle, either
          ``solvent`` and ``vaccuum` for a solvation free energy or
          ``solvent`` and ``complex`` for a binding free energy,
          with each containing a list of dictionaries containing the replica
          transition statistics for each repeat of that simulation type.

          The replica transition statistics dictionaries contain the following:
            * ``eigenvalues``: The sorted (descending) eigenvalues of the
              lambda state transition matrix
            * ``matrix``: The transition matrix estimate of a replica switching
              from state i to state j.
        """
        repex_stats: dict[str, list[dict[str, npt.NDArray]]] = {}
        try:
            for key in [self.bound_state, self.unbound_state]:
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
          Dictionary keyed for each leg of the thermodynamic cycle, either
          `solvent` and `vacuum` for solvation free energies,
          or `complex` and `solvent` for binding free energies,
          with lists of replica states timeseries for each repeat of that
          simulation type.
        """
        replica_states: dict[str, list[npt.NDArray]] = {
            self.bound_state: [],
            self.unbound_state: []
        }

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

        for key in [self.bound_state, self.unbound_state]:
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
          Dictionary keyed for each leg of the thermodynamic cycle, either
         `solvent` and `vacuum` for solvation free energies,
          or `complex` and `solvent` for binding free energies,
          with lists containing the number of equilibration iterations for
          each repeat of that simulation type.
        """
        equilibration_lengths: dict[str, list[float]] = {}

        for key in [self.bound_state, self.unbound_state]:
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
          Dictionary keyed for each leg of the thermodynamic cycle, either
         `solvent` and `vacuum` for solvation free energies,
          or `complex` and `solvent` for binding free energies,
          with lists containing the number of equilibration iterations for
          each repeat of that simulation type.
        """
        production_lengths: dict[str, list[float]] = {}

        for key in [self.bound_state, self.unbound_state]:
            production_lengths[key] = [
                pus[0].outputs["production_iterations"] for pus in self.data[key].values()
            ]

        return production_lengths

    def selection_indices(self) -> dict[str, list[Optional[npt.NDArray]]]:
        """
        Get the system selection indices used to write PDB and
        trajectory files.

        Returns
        -------
        indices : dict[str, list[npt.NDArray]]
          A dictionary keyed for each state, either
         `solvent` and `vacuum` for solvation free energies,
          or `complex` and `solvent` for binding free energies,
          each containing a list of NDArrays containing the corresponding
          full system atom indices for each atom written in the production
          trajectory files for each replica.
        """
        indices: dict[str, list[Optional[npt.NDArray]]] = {}

        for key in [self.bound_state, self.unbound_state]:
            indices[key] = []
            for pus in self.data[key].values():
                indices[key].append(pus[0].outputs["selection_indices"])

        return indices


class AbsoluteSolvationProtocolResult(gufe.ProtocolResult, AbsoluteProtocolResultMixin):
    """Dict-like container for the output of a AbsoluteSolvationProtocol"""

    bound_state = "solvent"
    unbound_state = "vacuum"

    def get_individual_estimates(self) -> dict[str, list[tuple[Quantity, Quantity]]]:
        """
        Get the individual estimate of the free energies.

        Returns
        -------
        dGs : dict[str, list[tuple[openff.units.Quantity, openff.units.Quantity]]]
          A dictionary, keyed `solvent` and `vacuum` for each leg
          of the thermodynamic cycle, with lists of tuples containing
          the individual free energy estimates and associated MBAR
          uncertainties for each repeat of that simulation type.
        """
        vac_dGs = []
        solv_dGs = []

        for pus in self.data["vacuum"].values():
            vac_dGs.append((pus[0].outputs["unit_estimate"], pus[0].outputs["unit_estimate_error"]))

        for pus in self.data["solvent"].values():
            solv_dGs.append(
                (pus[0].outputs["unit_estimate"], pus[0].outputs["unit_estimate_error"])
            )

        return {"solvent": solv_dGs, "vacuum": vac_dGs}

    def get_estimate(self):
        """Get the solvation free energy estimate for this calculation.

        Returns
        -------
        dG : openff.units.Quantity
          The solvation free energy. This is a Quantity defined with units.
        """

        def _get_average(estimates):
            # Get the unit value of the first value in the estimates
            u = estimates[0][0].u
            # Loop through estimates and get the free energy values
            # in the unit of the first estimate
            dGs = [i[0].to(u).m for i in estimates]

            return np.average(dGs) * u

        individual_estimates = self.get_individual_estimates()
        vac_dG = _get_average(individual_estimates["vacuum"])
        solv_dG = _get_average(individual_estimates["solvent"])

        return vac_dG - solv_dG

    def get_uncertainty(self):
        """Get the solvation free energy error for this calculation.

        Returns
        -------
        err : openff.units.Quantity
          The standard deviation between estimates of the solvation free
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
        vac_err = _get_stdev(individual_estimates["vacuum"])
        solv_err = _get_stdev(individual_estimates["solvent"])

        # return the combined error
        return np.sqrt(vac_err**2 + solv_err**2)


class AbsoluteBindingProtocolResult(gufe.ProtocolResult, AbsoluteProtocolResultMixin):
    """Dict-like container for the output of a AbsoluteBindingProtocol"""

    bound_state = "complex"
    unbound_state = "solvent"

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
                individual_estimates["complex"],
                individual_estimates["standard_state_correction"],
            )
        )
        solv_err = _get_stdev(individual_estimates["solvent"])

        # return the combined error
        return np.sqrt(complex_err**2 + solv_err**2)

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
