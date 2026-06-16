# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Reusable utility methods to analyze results from multistate calculations.
"""

import warnings
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from openff.units import Quantity, unit
from openff.units.openmm import from_openmm
from openmmtools import multistate

from openfe.analysis import plotting
from openfe.due import Doi, due

due.cite(
    Doi("10.5281/zenodo.596622"),
    description="OpenMMTools",
    path="openfe.protocols.openmm_utils.multistate_analysis",
    cite_module=True,
)


due.cite(
    Doi("10.1063/1.2978177"),
    description="MBAR paper",
    path="openfe.protocols.openmm_utils.multistate_analysis",
    cite_module=True,
)


due.cite(
    Doi("10.1021/ct0502864"),
    description="MBAR timeseries algorithms",
    path="openfe.protocols.openmm_utils.multistate_analysis",
    cite_module=True,
)


due.cite(
    Doi("10.1021/acs.jctc.5b00784"),
    description="Automatic equilibration detection method",
    path="openfe.protocols.openmm_utils.multistate_analysis",
    cite_module=True,
)


due.cite(
    Doi("10.5281/zenodo.596220"),
    description="pyMBAR zenodo",
    path="openfe.protocols.openmm_utils.multistate_analysis",
    cite_module=True,
)


class MultistateEquilFEAnalysis:
    """
    A class to generate and plot all necessary analyses for a free energy
    calculation using a :class:`openmmtools.MultiStateSampler`.

    Currently implemented analyses are:
      - Decorrelated MBAR analysis of free energies (and associated errors)
      - Number of equilibration & sampling steps
      - MBAR overlap matrices
      - Replica lambda traversal (exchange matrix and timeseries)
      - Forward and reverse analysis of free energies

    Parameters
    ----------
    reporter : openmmtools.MultiStateReporter
      Reporter for the MultiStateSampler
    sampling_method : str
      The sampling method. Expected values are `repex`, `sams`,
      and `independent`.
    result_units : openff.units.Quantity
      Units to report results in.
    forward_reverse_samples : int
      The number of samples to use in the forward and reverse analysis
      of the free energies. Default 10.
    """

    def __init__(
        self,
        reporter: multistate.MultiStateReporter,
        sampling_method: str,
        result_units: Quantity,
        forward_reverse_samples: int = 10,
    ):
        self.analyzer = multistate.MultiStateSamplerAnalyzer(reporter)
        self.units = result_units

        if sampling_method.lower() not in ["repex", "sams", "independent"]:
            wmsg = f"Unknown sampling method {sampling_method}"
            warnings.warn(wmsg)
        self.sampling_method = sampling_method.lower()

        # Do a first pass at the analysis
        self._analyze(forward_reverse_samples)

    def plot(self, filepath: Path, filename_prefix: str):
        """
        Plot out results from the free energy analyses.

        Specifically the following plots are generated:
          * The free energy overlap matrix
          * The replica exchange overlap matrix (if sampler_method is repex)
          * The timeseries of replica states over time
          * The forward and reverse estimate of the free energies

        Parameters
        ----------
        filepath : pathlib.Path
          The path to where files should be written.
        filename_prefix : str
          A prefix for the written filenames.
        """
        # MBAR overlap matrix
        ax = plotting.plot_lambda_transition_matrix(self.free_energy_overlaps["matrix"])
        ax.set_title("MBAR overlap matrix")
        ax.figure.savefig(  # type: ignore
            filepath / (filename_prefix + "mbar_overlap_matrix.png")
        )
        plt.close(ax.figure)  # type: ignore

        # Reverse and forward analysis
        if self.forward_and_reverse_free_energies is not None:
            ax = plotting.plot_convergence(self.forward_and_reverse_free_energies, self.units)
            ax.set_title("Forward and Reverse free energy convergence")
            ax.figure.savefig(  # type: ignore
                filepath / (filename_prefix + "forward_reverse_convergence.png")
            )
            plt.close(ax.figure)  # type: ignore

        # Replica state timeseries plot
        ax = plotting.plot_replica_timeseries(self.replica_states, self.equilibration_iterations)
        ax.set_title("Change in replica state over time")
        ax.figure.savefig(  # type: ignore
            filepath / (filename_prefix + "replica_state_timeseries.png")
        )
        plt.close(ax.figure)  # type: ignore

        # Replica exchange transition matrix
        if self.sampling_method == "repex":
            ax = plotting.plot_lambda_transition_matrix(self.replica_exchange_statistics["matrix"])
            ax.set_title("Replica exchange transition matrix")
            ax.figure.savefig(  # type: ignore
                filepath / (filename_prefix + "replica_exchange_matrix.png")
            )
            plt.close(ax.figure)  # type: ignore

    def _analyze(self, forward_reverse_samples: int):
        """
        Run the following analyses:
          * MBAR free energy difference between end states using
            post-equilibration decorrelated samples of the energies.
          * Forward and reverse fractional analysis of free energies over
            the equilibrated & decorrelated data points.
          * MBAR estimate of the overlap matrix across states.
          * Replica exchange transition matrix
            (if sampler_method is ``repex``)

        Parameters
        ----------
        forward_reverse_samples : int
          Number of samples to take in the forward and reverse analysis of
          the free energies.
        """
        # Do things that get badly cached later
        self._replica_states = self.analyzer.reporter.read_replica_thermodynamic_states()
        # convert full masked array to simple array
        # downcast to int32, we don't have more than 4 billion states thankfully
        self._replica_states = np.asarray(self._replica_states, dtype=np.int32)
        # float conversions to avoid having to deal with numpy dtype serialization
        self._equil_iters = float(self.analyzer.n_equilibration_iterations)
        self._prod_iters = float(self.analyzer._equilibration_data[2])

        # Gather estimate of free energy
        self._free_energy, self._free_energy_err = self.get_equil_free_energy()

        # forward and reverse analysis
        self._forward_reverse = self.get_forward_and_reverse_analysis(forward_reverse_samples)

        # Gather overlap matrix
        self._overlap_matrix = self.get_overlap_matrix()

        # Gather exchange transition matrix
        # Note we only generate these for replica exchange calculations
        # TODO: consider if this would also work for SAMS
        if self.sampling_method == "repex":
            self._exchange_matrix = self.get_exchanges()

    @staticmethod
    def _get_free_energy(
        analyzer: multistate.MultiStateSamplerAnalyzer,
        u_ln: npt.NDArray,
        N_l: npt.NDArray,
        bootstraps: int = 1000,
        return_units: Quantity = unit.kilocalorie_per_mole,
    ) -> tuple[Quantity, Quantity]:
        """
        Helper method to create an MBAR object and extract free energies
        between end states.

        Parameters
        ----------
        analyzer : multistate.MultiStateSamplerAnalyzer
          MultiStateSamplerAnalyzer to extract free energies from.
        u_ln : npt.NDArray
          A n_states x (n_sampled_states * n_iterations)
          array of energies (in kT).
        N_l : npt.NDArray
          An array containing the total number of samples drawn from each
          state.
        bootstraps : int
          How many bootstrap samples will be computed. If 0, no bootstraps
          will be computed and analytical errors will be returned.
        return_units : openff.units.Quantity
          The return units the results will be provided in.

        Returns
        -------
        DG : openff.units.Quantity
          The free energy difference between the end states.
        dDG : openff.units.Quantity
          The MBAR bootstrap (1000 iterations) error estimate for the free energy difference.

        TODO
        ----
        * Allow folks to pass in extra options for bootstrapping etc..
        * Add standard test against analyzer.get_free_energy()
        """
        # pymbar has some side effects when imported so we only import it right when we
        # need it
        from pymbar import MBAR

        mbar = MBAR(
            u_ln,
            N_l,
            solver_protocol="robust",
            n_bootstraps=bootstraps,
            bootstrap_solver_protocol="robust",
        )
        if bootstraps > 0:
            uncertainty_method = "bootstrap"
        else:
            uncertainty_method = None

        r = mbar.compute_free_energy_differences(
            compute_uncertainty=True,
            uncertainty_method=uncertainty_method,
        )

        DF_ij = r["Delta_f"]
        dDF_ij = r["dDelta_f"]

        DG = DF_ij[0, -1] * analyzer.kT
        dDG = dDF_ij[0, -1] * analyzer.kT

        return (from_openmm(DG).to(return_units), from_openmm(dDG).to(return_units))

    def get_equil_free_energy(self) -> tuple[Quantity, Quantity]:
        """
        Extract unbiased and uncorrelated estimates of the free energy
        and the associated error from a MultiStateSamplerAnalyzer object.

        Returns
        -------
        DG : openff.units.Quantity
          The free energy difference between the end states.
        dDG : openff.units.Quantity
          The MBAR error for the free energy difference estimate.
        """
        u_ln_decorr = self.analyzer._unbiased_decorrelated_u_ln
        N_l_decorr = self.analyzer._unbiased_decorrelated_N_l

        DG, dDG = self._get_free_energy(self.analyzer, u_ln_decorr, N_l_decorr, 1000, self.units)

        return DG, dDG

    def _get_fraction_free_energy(
        self,
        u_ln: npt.NDArray,
        N_l: npt.NDArray,
        samples: int,
        fraction: float,
    ) -> tuple[tuple[Quantity, Quantity], tuple[Quantity, Quantity]]:
        """
        Helper method to estimate the forward and reverse free energies for a
        fraction of the uncorrelated samples.

        Used by :meth:`get_forward_and_reverse_analysis` for each chunk. MBAR
        can fail to converge at low fractions of uncorrelated samples. While
        such a failure is directly caused by the estimator, it is more broadly
        caused by too few (effective) data points at that fraction, so it is
        not reasonable to trust either direction. Therefore, if MBAR fails for
        *either* the forward or reverse estimate, NaN is recorded for *both*
        directions (and a warning raised) and the analysis continues, so that
        estimates at higher fractions are still reported.

        Parameters
        ----------
        u_ln : npt.NDArray
          The full sub-sampled energy matrix. The forward estimate uses the
          first ``samples`` columns and the reverse estimate the last
          ``samples`` columns.
        N_l : npt.NDArray
          An array containing the number of samples drawn from each state.
        samples : int
          The number of samples (columns of ``u_ln``) to use for each estimate.
        fraction : float
          The fraction of uncorrelated samples this corresponds to. Only used
          for the warning message.

        Returns
        -------
        forward : tuple[openff.units.Quantity, openff.units.Quantity]
          The forward free energy difference and its MBAR error estimate, or
          NaN for both if MBAR failed to obtain an estimate.
        reverse : tuple[openff.units.Quantity, openff.units.Quantity]
          The reverse free energy difference and its MBAR error estimate, or
          NaN for both if MBAR failed to obtain an estimate.
        """
        # pymbar has some side effects from being imported, so we only want to
        # import it right when we need it
        from pymbar.utils import ParameterError

        try:
            forward = self._get_free_energy(
                self.analyzer,
                u_ln[:, :samples],
                N_l,
                0,
                self.units,
            )
            reverse = self._get_free_energy(
                self.analyzer,
                u_ln[:, -samples:],
                N_l,
                0,
                self.units,
            )
        except ParameterError:
            wmsg = (
                f"Could not obtain a free energy estimate at fraction "
                f"{fraction:.2f} of the uncorrelated samples; recording NaN "
                "for both the forward and reverse estimates."
            )
            warnings.warn(wmsg)
            forward = reverse = (np.nan * self.units, np.nan * self.units)  # type: ignore

        return forward, reverse

    def get_forward_and_reverse_analysis(
        self, num_samples: int = 10
    ) -> Optional[dict[str, Union[npt.NDArray, Quantity]]]:
        """
        Calculate free energies with a progressively larger
        fraction of the decorrelated timeseries data in both
        the forward and reverse direction.

        Parameters
        ----------
        num_samples : int
          The number data points to sample.

        Returns
        -------
        forward_reverse : Optional[dict[str, Union[npt.NDArray, openff.units.Quantity]]]
          If this analysis fails, returns None; otherwise returns a dictionary
          containing;
            * ``fractions``: fractions of sample used to calculate free energies
            * ``forward_DGs`` and `forward_dDGs`: the free energy estimates
              and errors along each sample fraction in the forward direction
            * ``reverse_DGs`` and `reverse_dDGs`: the free energy estimates
              and errors along each sample fraction in the reverse direction

        Notes
        -----
        * This method does not currently use bootstrap uncertainties due to
          issues with the solver when using low amounts of data points. All
          uncertainties are MBAR analytical errors.
        * If MBAR fails to obtain an estimate for a given sample fraction (a
          ``ParameterError``, typically at low fractions of uncorrelated
          samples), a NaN is recorded for that fraction in both the forward
          and reverse directions and the analysis continues, so that estimates
          at higher fractions are still returned.
        * The exception is the final fraction (1.0, the full set of
          uncorrelated samples): if MBAR fails there, the whole analysis is
          discarded and None is returned, since that estimate is the reported
          free energy and anchors the convergence plot.
        """
        u_ln = self.analyzer._unbiased_decorrelated_u_ln
        N_l = self.analyzer._unbiased_decorrelated_N_l
        n_states = len(N_l)

        # Check that the N_l is the same across all states
        if not np.all(N_l == N_l[0]):
            errmsg = f"The number of samples is not equivalent across all states {N_l}"
            raise ValueError(errmsg)

        # Get the chunks of N_l going from 10% to ~ 100%
        # Note: you always lose out a few data points but it's fine
        chunks = [max(int(N_l[0] / num_samples * i), 1) for i in range(1, num_samples + 1)]

        forward_DGs = []
        forward_dDGs = []
        reverse_DGs = []
        reverse_dDGs = []
        fractions = []

        for chunk in chunks:
            new_N_l = np.array([chunk for _ in range(n_states)])
            samples = chunk * n_states
            fraction = chunk / N_l[0]

            # If MBAR fails for either the forward or reverse estimate, both
            # are recorded as NaN (too few effective samples at this
            # fraction to trust either direction).
            (forward_DG, forward_dDG), (reverse_DG, reverse_dDG) = self._get_fraction_free_energy(
                u_ln, new_N_l, samples, fraction
            )
            forward_DGs.append(forward_DG)
            forward_dDGs.append(forward_dDG)
            reverse_DGs.append(reverse_DG)
            reverse_dDGs.append(reverse_dDG)

            fractions.append(fraction)

        forward_reverse = {
            "fractions": np.array(fractions),
            "forward_DGs": Quantity.from_list(forward_DGs),  # type: ignore
            "forward_dDGs": Quantity.from_list(forward_dDGs),  # type: ignore
            "reverse_DGs": Quantity.from_list(reverse_DGs),  # type: ignore
            "reverse_dDGs": Quantity.from_list(reverse_dDGs),  # type: ignore
        }

        # The final fraction (1.0) uses the full set of uncorrelated samples.
        # Its estimate is the reported free energy and anchors the error band in
        # the convergence plot, so if MBAR failed for it (NaN) the whole
        # analysis is not meaningful and we discard it by returning None.
        final_forward = forward_reverse["forward_DGs"][-1].m
        final_reverse = forward_reverse["reverse_DGs"][-1].m
        if np.isnan(final_forward) or np.isnan(final_reverse):
            wmsg = (
                "MBAR could not obtain a free energy estimate using the full set "
                "of uncorrelated samples (fraction 1.0); discarding the forward "
                "and reverse convergence analysis."
            )
            warnings.warn(wmsg)
            return None

        return forward_reverse

    def get_overlap_matrix(self) -> dict[str, npt.NDArray]:
        """
        Generate an overlap matrix across lambda states.

        Return
        ------
        overlap_matrix : dict[str, npt.NDArray]
          A dictionary containing the following keys:
            * ``scalar``: One minus the largest nontrivial eigenvalue
            * ``eigenvalues``: The sorted (descending) eigenvalues of the
              overlap matrix
            * ``matrix``: Estimated overlap matrix of observing a sample from
              state i in state j
        """
        return self.analyzer.mbar.compute_overlap()

    def get_exchanges(self) -> dict[str, npt.NDArray]:
        """
        Gather both the transition matrix (and relevant eigenvalues) between
        replicas.

        Return
        ------
        transition_matrix : dict[str, npt.NDArray]
          A dictionary containing the following:
            * ``eigenvalues``: The sorted (descending) eigenvalues of the
              lambda state transition matrix
            * ``matrix``: The transition matrix estimate of a replica switching
              from state i to state j.
        """
        # Get replica mixing statistics
        mixing_stats = self.analyzer.generate_mixing_statistics()
        transition_matrix = {
            "eigenvalues": mixing_stats.eigenvalues,
            "matrix": mixing_stats.transition_matrix,
        }
        return transition_matrix

    @property
    def replica_states(self):
        """
        Timeseries of states for each replica.
        """
        return self._replica_states

    @property
    def equilibration_iterations(self):
        """
        Number of iterations discarded as equilibration.
        """
        return self._equil_iters

    @property
    def production_iterations(self):
        """
        Number of production iterations from which energies are sampled.
        """
        return self._prod_iters

    @property
    def free_energy(self):
        """
        The free energy estimate from decorrelated unbiased samples
        """
        return self._free_energy

    @property
    def free_energy_error(self):
        """
        The MBAR estimate of the free energy estimate
        """
        return self._free_energy_err

    @property
    def forward_and_reverse_free_energies(self):
        """
        The dictionary forward and reverse analysis of the free energies
        using the number of samples defined at class initialization
        """
        return self._forward_reverse

    @property
    def free_energy_overlaps(self):
        """
        A dictionary containing the estimated overlap matrix and corresponding
        eigenvalues and scalars of the free energies.
        """
        return self._overlap_matrix

    @property
    def replica_exchange_statistics(self):
        """
        A dictionary containing the estimated replica exchange matrix
        and corresponding eigenvalues.
        """
        if hasattr(self, "_exchange_matrix"):
            return self._exchange_matrix
        else:
            errmsg = (
                "Exchange matrix was not generated, this is likely "
                f"{self.sampling_method} is not repex."
            )
            raise ValueError(errmsg)

    @property
    def unit_results_dict(self):
        results_dict = {
            "unit_estimate": self.free_energy,
            "unit_estimate_error": self.free_energy_error,
            "unit_mbar_overlap": self.free_energy_overlaps,
            "forward_and_reverse_energies": self.forward_and_reverse_free_energies,
            "production_iterations": self.production_iterations,
            "equilibration_iterations": self.equilibration_iterations,
        }

        if hasattr(self, "_exchange_matrix"):
            results_dict["replica_exchange_statistics"] = self.replica_exchange_statistics

        return results_dict

    def close(self):
        self.analyzer.clear()
