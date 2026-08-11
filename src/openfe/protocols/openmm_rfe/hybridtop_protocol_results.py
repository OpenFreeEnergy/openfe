# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
ProtocolUnitResults for Hybrid Topology methods using
OpenMM and OpenMMTools in a Perses-like manner.
"""

import itertools
import logging
import pathlib
import warnings
from typing import Optional, Union

import gufe
import numpy as np
import numpy.typing as npt
from openff.units import Quantity
from openmmtools import multistate

logger = logging.getLogger(__name__)


class RelativeHybridTopologyProtocolResult(gufe.ProtocolResult):
    """
    Protocol results with the output of a RelativeHybridTopologyProtocol.
    """

    def __init__(self, **data):
        super().__init__(**data)
        # data is mapping of str(repeat_id): list[protocolunitresults]
        # TODO: Detect when we have extensions and stitch these together?
        if any(len(pur_list) > 2 for pur_list in self.data.values()):
            raise NotImplementedError("Can't stitch together results yet")

    @staticmethod
    def compute_mean_estimate(dGs: list[Quantity]) -> Quantity:
        u = dGs[0].u
        # convert all values to units of the first value, then take average of magnitude
        # this would avoid an edge case where each value was in different units
        vals = np.asarray([dG.to(u).m for dG in dGs])

        return np.average(vals) * u

    def get_estimate(self) -> Quantity:
        """Average free energy difference of this transformation

        Returns
        -------
        dG : openff.units.Quantity
          The free energy difference between the first and last states. This is
          a Quantity defined with units.
        """
        # TODO: Check this holds up completely for SAMS.
        dGs = [pus[0].outputs["unit_estimate"] for pus in self.data.values()]
        return self.compute_mean_estimate(dGs)

    @staticmethod
    def compute_uncertainty(dGs: list[Quantity]) -> Quantity:
        u = dGs[0].u
        # convert all values to units of the first value, then take average of magnitude
        # this would avoid a screwy case where each value was in different units
        vals = np.asarray([dG.to(u).m for dG in dGs])
        # use the unbiased sample standard deviation (ddof=1) as the repeats are sampled from the
        # (inaccessible) population of possible repeats.
        std = np.std(vals, ddof=1)
        if np.isnan(std):
            std = 0.0
        return std * u

    def get_uncertainty(self) -> Quantity:
        """The uncertainty/error in the dG value: The unbiased sample std of the estimates of
        each independent repeat
        """

        dGs = [pus[0].outputs["unit_estimate"] for pus in self.data.values()]
        return self.compute_uncertainty(dGs)

    def get_individual_estimates(self) -> list[tuple[Quantity, Quantity]]:
        """Return a list of tuples containing the individual free energy
        estimates and associated MBAR errors for each repeat.

        Returns
        -------
        dGs : list[tuple[openff.units.Quantity]]
          n_replicate simulation list of tuples containing the free energy
          estimates (first entry) and associated MBAR estimate errors
          (second entry).
        """
        dGs = [
            (pus[0].outputs["unit_estimate"], pus[0].outputs["unit_estimate_error"])
            for pus in self.data.values()
        ]
        return dGs

    def get_forward_and_reverse_energy_analysis(
        self,
    ) -> list[Optional[dict[str, Union[npt.NDArray, Quantity]]]]:
        """
        Get a list of forward and reverse analysis of the free energies
        for each repeat using uncorrelated production samples.

        The returned dicts have keys:
        'fractions' - the fraction of data used for this estimate
        'forward_DGs', 'reverse_DGs' - for each fraction of data, the estimate
        'forward_dDGs', 'reverse_dDGs' - for each estimate, the uncertainty

        The 'fractions' values are a numpy array, while the other arrays are
        Quantity arrays, with units attached. A fraction at which MBAR failed
        to converge is recorded as ``NaN`` in both directions.

        A list entry is ``None`` only when MBAR could not obtain an estimate
        from the full set of uncorrelated samples (the fraction 1.0 estimate,
        i.e. the reported free energy) for that repeat. If MBAR fails only at a
        lower fraction, that fraction is recorded as ``NaN`` and the remaining
        fractions are retained, so the entry is still a dictionary.


        Returns
        -------
        forward_reverse : list[Optional[dict[str, Union[npt.NDArray, openff.units.Quantity]]]]


        Raises
        ------
        UserWarning
          If any of the forward and reverse entries are ``None``.
        """
        forward_reverse = [
            pus[0].outputs["forward_and_reverse_energies"] for pus in self.data.values()
        ]

        if None in forward_reverse:
            wmsg = (
                "One or more ``None`` entries were found in the list of "
                "forward and reverse analyses. This indicates that MBAR could "
                "not obtain a free energy estimate from the full set of "
                "uncorrelated samples for that repeat."
            )
            warnings.warn(wmsg)

        return forward_reverse

    def get_overlap_matrices(self) -> list[dict[str, npt.NDArray]]:
        """
        Return a list of dictionary containing the MBAR overlap estimates
        calculated for each repeat.

        Returns
        -------
        overlap_stats : list[dict[str, npt.NDArray]]
          A list of dictionaries containing the following keys:
            * ``scalar``: One minus the largest nontrivial eigenvalue
            * ``eigenvalues``: The sorted (descending) eigenvalues of the
              overlap matrix
            * ``matrix``: Estimated overlap matrix of observing a sample from
              state i in state j
        """
        # Loop through and get the repeats and get the matrices
        overlap_stats = [pus[0].outputs["unit_mbar_overlap"] for pus in self.data.values()]

        return overlap_stats

    def get_replica_transition_statistics(self) -> list[dict[str, npt.NDArray]]:
        """The replica lambda state transition statistics for each repeat.

        Note
        ----
        This is currently only available in cases where a replica exchange
        simulation was run.

        Returns
        -------
        repex_stats : list[dict[str, npt.NDArray]]
          A list of dictionaries containing the following:
            * ``eigenvalues``: The sorted (descending) eigenvalues of the
              lambda state transition matrix
            * ``matrix``: The transition matrix estimate of a replica switching
              from state i to state j.
        """
        try:
            repex_stats = [
                pus[0].outputs["replica_exchange_statistics"] for pus in self.data.values()
            ]
        except KeyError:
            errmsg = "Replica exchange statistics were not found, did you run a repex calculation?"
            raise ValueError(errmsg)

        return repex_stats

    def get_replica_states(self) -> list[npt.NDArray]:
        """
        Returns the timeseries of replica states for each repeat.

        Returns
        -------
        replica_states : List[npt.NDArray]
          List of replica states for each repeat
        """

        def is_file(filename: str):
            p = pathlib.Path(filename)
            if not p.exists():
                errmsg = f"File could not be found {p}"
                raise ValueError(errmsg)
            return p

        replica_states = []

        for pus in self.data.values():
            nc = is_file(pus[0].outputs["trajectory"])
            dir_path = nc.parents[0]
            chk = is_file(pus[0].outputs["checkpoint"]).name
            reporter = multistate.MultiStateReporter(
                storage=nc, checkpoint_storage=chk, open_mode="r"
            )
            replica_states.append(np.asarray(reporter.read_replica_thermodynamic_states()))
            reporter.close()

        return replica_states

    def equilibration_iterations(self) -> list[float]:
        """
        Returns the number of equilibration iterations for each repeat
        of the calculation.

        Returns
        -------
        equilibration_lengths : list[float]
        """
        equilibration_lengths = [
            pus[0].outputs["equilibration_iterations"] for pus in self.data.values()
        ]

        return equilibration_lengths

    def production_iterations(self) -> list[float]:
        """
        Returns the number of uncorrelated production samples for each
        repeat of the calculation.

        Returns
        -------
        production_lengths : list[float]
        """
        production_lengths = [pus[0].outputs["production_iterations"] for pus in self.data.values()]

        return production_lengths


class HTopProtocolResultMixin:
    """
    Mixin providing the shared utilities for two-leg hybrid topology
    ProtocolResults (``RBFEHTopProtocolResult``, ``RHFEHTopProtocolResult``).

    Subclasses must define the class attributes ``env_state`` and
    ``ref_state``, naming the two legs of the thermodynamic cycle stored in
    ``self.data`` (e.g. ``env_state = "complex"``, ``ref_state = "solvent"``).
    """

    env_state: str
    ref_state: str

    def __init__(self, **data):
        super().__init__(**data)
        # data is a mapping of leg: str(repeat_id): list[protocolunitresults]
        # TODO: Detect when we have extensions and stitch these together?
        if any(
            len(pur_list) > 2
            for pur_list in itertools.chain(
                self.data[self.env_state].values(), self.data[self.ref_state].values()
            )
        ):
            raise NotImplementedError("Can't stitch together results yet")

    def get_individual_estimates(self) -> dict[str, list[tuple[Quantity, Quantity]]]:
        """
        Get the individual estimate of the free energies for both legs.

        Returns
        -------
        dGs : dict[str, list[tuple[openff.units.Quantity, openff.units.Quantity]]]
          A dictionary, keyed for each leg of the thermodynamic cycle, e.g.
          ``solvent`` and ``complex`` for a relaltive binding free energy or
          ``solvent`` and ``vacuum`` for a relative hydration free energy,
          with lists of tuples containing the individual free energy estimates
          and associated MBAR uncertainties for each repeat of that simulation type.
        """
        dGs = {}

        for state in [self.env_state, self.ref_state]:
            dGs[state] = [
                (pus[0].outputs["unit_estimate"], pus[0].outputs["unit_estimate_error"])
                for pus in self.data[state].values()
            ]

        return dGs

    @staticmethod
    def _get_average(estimates: list[tuple[Quantity, Quantity]]) -> Quantity:
        u = estimates[0][0].u
        dGs = [i[0].to(u).m for i in estimates]
        return np.average(dGs) * u

    @staticmethod
    def _get_stdev(estimates: list[tuple[Quantity, Quantity]]) -> Quantity:
        u = estimates[0][0].u
        dGs = [i[0].to(u).m for i in estimates]
        # use the unbiased sample standard deviation (ddof=1) as the repeats are sampled from the
        # (inaccessible) population of possible repeats.
        std = np.std(dGs, ddof=1)
        if np.isnan(std):
            std = 0.0
        return std * u

    def get_estimate(self) -> Quantity:
        """Get the relative free energy estimate for this calculation.

        Returns
        -------
        ddG : openff.units.Quantity
          The difference free energy. This is a Quantity defined
          with units.
        """
        individual_estimates = self.get_individual_estimates()
        env_dG = self._get_average(individual_estimates[self.env_state])
        ref_dG = self._get_average(individual_estimates[self.ref_state])

        return env_dG - ref_dG

    def get_uncertainty(self) -> Quantity:
        """Get the relative free energy error for this calculation.

        Returns
        -------
        err : openff.units.Quantity
          The unbiased standard deviation between estimates of the relative
          free energy. This is a Quantity defined with units.
        """
        individual_estimates = self.get_individual_estimates()
        env_err = self._get_stdev(individual_estimates[self.env_state])
        ref_err = self._get_stdev(individual_estimates[self.ref_state])

        return np.sqrt(env_err**2 + ref_err**2)

    def get_forward_and_reverse_energy_analysis(
        self,
    ) -> dict[str, list[Optional[dict[str, Union[npt.NDArray, Quantity]]]]]:
        """
        Get the reverse and forward analysis of the free energies for both
        legs of the thermodynamic cycle.

        Returns
        -------
        forward_reverse : dict[str, list[Optional[dict[str, Union[npt.NDArray, openff.units.Quantity]]]]]
            A dictionary, keyed by leg of the thermodynamic cycle, e.g. ``solvent``
            and ``vacuum`` for a relative hydration free energy or ``solvent`` and
            ``complex`` for a relative binding free energy, with each
            entry containing a list of dictionaries with the forward and
            reverse analysis of each repeat of that simulation type.

            The forward and reverse analysis dictionaries contain:
              - `fractions`: npt.NDArray
                  The fractions of data used for the estimates
              - `forward_DGs`, `reverse_DGs`: openff.units.Quantity
                  The forward and reverse estimates for each fraction of data.
                  A fraction at which MBAR failed to converge is recorded as
                  ``NaN`` in both directions.
              - `forward_dDGs`, `reverse_dDGs`: openff.units.Quantity
                  The forward and reverse estimate uncertainty for each
                  fraction of data (``NaN`` wherever the estimate is ``NaN``).

            A cycle leg list entry is ``None`` only when MBAR could not obtain
            an estimate from the *full* set of uncorrelated samples (the
            fraction 1.0 estimate, i.e. the reported free energy). If MBAR
            fails only at a lower fraction, that fraction is recorded as
            ``NaN`` (see ``forward_DGs`` above) and the remaining fractions
            are retained, so the entry is still a dictionary.

        Raises
        ------
        UserWarning
          * If any of the forward and reverse dictionaries are ``None`` in a
            given thermodynamic cycle leg.
        """
        forward_reverse: dict[str, list[Optional[dict[str, Union[npt.NDArray, Quantity]]]]] = {}

        for key in [self.env_state, self.ref_state]:
            forward_reverse[key] = [
                pus[0].outputs["forward_and_reverse_energies"]
                for pus in self.data[key].values()  # type: ignore[attr-defined]
            ]

            if None in forward_reverse[key]:
                wmsg = (
                    "One or more ``None`` entries were found in the forward "
                    f"and reverse dictionaries of the repeats of the {key} "
                    "calculations. This indicates that MBAR could not obtain a "
                    "free energy estimate from the full set of uncorrelated "
                    "samples for that repeat."
                )
                warnings.warn(wmsg)

        return forward_reverse

    def get_overlap_matrices(self) -> dict[str, list[dict[str, npt.NDArray]]]:
        """
        Get the MBAR overlap estimates for both legs of the simulation.

        Returns
        -------
        overlap_stats : dict[str, list[dict[str, npt.NDArray]]]
          A dictionary keyed by leg of the thermodynamic cycle, e.g. 
          ``solvent`` and ``vacuum`` for a relative hydration free energy
          or ``solvent`` and ``complex`` for a relative binding free energy,
          with each entry containing a list of dictionaries with the MBAR overlap
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

        for key in [self.env_state, self.ref_state]:
            overlap_stats[key] = [
                pus[0].outputs["unit_mbar_overlap"]
                for pus in self.data[key].values()  # type: ignore[attr-defined]
            ]

        return overlap_stats

    def get_replica_transition_statistics(self) -> dict[str, list[dict[str, npt.NDArray]]]:
        """
        Get the replica exchange transition statistics for both legs of the
        thermodynamic cycle.

        Note
        ----
        This is currently only available in cases where a replica exchange
        simulation was run.

        Returns
        -------
        repex_stats : dict[str, list[dict[str, npt.NDArray]]]
          A dictionary keyed by leg of the thermodynamic cycle, e.g.
          ``solvent`` and ``vacuum`` for a relative hydration free energy or ``solvent`` and
          ``complex`` for a relative binding free energy, with each
          entry containing a list of dictionaries with the replica
          transition statistics for each repeat of that simulation type.

          The replica transition statistics dictionaries contain the following:
            * ``eigenvalues``: The sorted (descending) eigenvalues of the
              lambda state transition matrix
            * ``matrix``: The transition matrix estimate of a replica switching
              from state i to state j.
        """
        repex_stats: dict[str, list[dict[str, npt.NDArray]]] = {}
        try:
            for key in [self.env_state, self.ref_state]:
                repex_stats[key] = [
                    pus[0].outputs["replica_exchange_statistics"]
                    for pus in self.data[key].values()  # type: ignore[attr-defined]
                ]
        except KeyError:
            errmsg = "Replica exchange statistics were not found, did you run a repex calculation?"
            raise ValueError(errmsg)

        return repex_stats

    def get_replica_states(self) -> dict[str, list[npt.NDArray]]:
        """
        Get the timeseries of replica states for both simulation legs.

        Returns
        -------
        replica_states : dict[str, list[npt.NDArray]]
          Dictionary keyed by leg of the thermodynamic cycle, e.g.
          ``solvent`` and ``vacuum`` for a relative hydration free energy or ``solvent`` and
          ``complex`` for a relative binding free energy, with lists of
          replica states timeseries for each repeat of that simulation type.
        """
        replica_states: dict[str, list[npt.NDArray]] = {
            self.env_state: [],
            self.ref_state: [],
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

        for key in [self.env_state, self.ref_state]:
            for pus in self.data[key].values():  # type: ignore[attr-defined]
                states = get_replica_state(
                    pus[0].outputs["trajectory"],
                    pus[0].outputs["checkpoint"],
                )
                replica_states[key].append(states)

        return replica_states

    def equilibration_iterations(self) -> dict[str, list[float]]:
        """
        Returns the number of equilibration iterations for each repeat of
        both legs of the calculation.

        Returns
        -------
        equilibration_lengths : dict[str, list[float]]
          Dictionary keyed for each leg of the thermodynamic cycle, e.g.
          ``solvent`` and ``vacuum`` for a relative hydration free energy or
          ``solvent`` and ``complex`` for a relative binding free energy,
          with lists of the number of equilibration iterations for each
          repeat of that simulation type.
        """
        equilibration_lengths: dict[str, list[float]] = {}

        for key in [self.env_state, self.ref_state]:
            equilibration_lengths[key] = [
                pus[0].outputs["equilibration_iterations"]
                for pus in self.data[key].values()  # type: ignore[attr-defined]
            ]

        return equilibration_lengths

    def production_iterations(self) -> dict[str, list[float]]:
        """
        Returns the number of uncorrelated production samples for each
        repeat of both legs of the calculation.

        Returns
        -------
        production_lengths : dict[str, list[float]]
          Dictionary keyed for each leg of the thermodynamic cycle, e.g.
          ``solvent`` and ``vacuum`` for a relative hydration free energy or
          ``solvent`` and ``complex`` for a relative binding free energy,
          with lists of the number of uncorrelated production samples for
          each repeat of that simulation type.
        """
        production_lengths: dict[str, list[float]] = {}

        for key in [self.env_state, self.ref_state]:
            production_lengths[key] = [
                pus[0].outputs["production_iterations"]
                for pus in self.data[key].values()  # type: ignore[attr-defined]
            ]

        return production_lengths

    def selection_indices(self) -> dict[str, list[Optional[npt.NDArray]]]:
        """
        Get the system selection indices used to write PDB and trajectory
        files, for both legs of the calculation.

        Returns
        -------
        indices : dict[str, list[Optional[npt.NDArray]]]
          A dictionary keyed by leg of the thermodynamic cycle, e.g. 
          ``solvent`` and ``vacuum`` for a relative hydration free energy or
          ``solvent`` and ``complex`` for a relative binding free energy,
          each containing a list of NDArrays with the corresponding full system
          atom indices for each atom written in the PDB or production trajectory
          files for each replica.
        """
        indices: dict[str, list[Optional[npt.NDArray]]] = {}

        for key in [self.env_state, self.ref_state]:
            indices[key] = [
                pus[0].outputs["selection_indices"]
                for pus in self.data[key].values()  # type: ignore[attr-defined]
            ]

        return indices


class RBFEHTopProtocolResult(gufe.ProtocolResult, HTopProtocolResultMixin):
    """
    Protocol results with the output of a ``RBFEHTopProtocol``.
    """

    env_state = "complex"
    ref_state = "solvent"


class RHFEHTopProtocolResult(gufe.ProtocolResult, HTopProtocolResultMixin):
    """
    Protocol results with the output of a ``RHFEHTopProtocol``.
    """

    env_state = "solvent"
    ref_state = "vacuum"