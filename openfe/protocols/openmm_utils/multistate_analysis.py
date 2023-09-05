# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Reusable utility methods to analyze results from multistate calculations.
"""
from pathlib import Path
from typing import Union, Optional
import warnings

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from openmmtools import multistate
from openff.units import unit, ensure_quantity


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
    result_units : unit.Quantity
      Units to report results in.
    forward_reverse_samples : int
      The number of samples to use in the foward and reverse analysis
      of the free energies. Default 10.
    """
    def __init__(self, reporter: multistate.MultiStateReporter,
                 sampling_method: str, result_units: unit.Quantity,
                 forward_reverse_samples: int = 10):
        self.analyzer = multistate.MultiStateSamplerAnalyzer(reporter)
        self.units = result_units

        if sampling_method.lower() not in ['repex', 'sams', 'independent']:
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
          * The foward and reverse estimate of the free energies

        Parameters
        ----------
        filepath : pathlib.Path
          The path to where files should be written.
        filename_prefix : str
          A prefix for the written filenames.
        """
        # MBAR overlap matrix
        ax = plot_lambda_transition_matrix(self.free_energy_overlaps['matrix'])
        ax.set_title('MBAR overlap matrix')
        ax.figure.savefig(
            filepath / (filename_prefix + 'mbar_overlap_matrix.png')
        )

        # Reverse and forward analysis
        ax = plot_convergence(
            self.forward_and_reverse_free_energies, self.units
        )
        ax.set_title('Forward and Reverse free energy convergence')
        ax.figure.savefig(
            filepath / (filename_prefix + 'forward_reverse_convergence.png')
        )

        # Replica state timeseries plot
        ax = plot_replica_timeseries(
            self.replica_states, self.equilibration_iterations
        )
        ax.set_title('Change in replica state over time')
        ax.figure.savefig(
            filepath / (filename_prefix + 'replica_state_timeseries.png')
        )

        # Replica exchange transition matrix
        if self.sampling_method == 'repex':
            ax = plot_lambda_transition_matrix(
                self.replica_exchange_statistics['matrix']
            )
            ax.set_title('Replica exchange transition matrix')
            ax.figure.savefig(
                filepath / (filename_prefix + 'replica_exchange_matrix.png')
            )

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
          Number of samples to take in the foward and reverse analysis of
          the free energies.
        """
        # Do things that get badly cached later
        self._replica_states = self.analyzer.reporter.read_replica_thermodynamic_states()
        self._equil_iters = self.analyzer.n_equilibration_iterations
        self._prod_iters = self.analyzer._equilibration_data[2]

        # Gather estimate of free energy
        self._free_energy, self._free_energy_err = self.get_equil_free_energy()

        # forward and reverse analysis
        self._forward_reverse = self.get_forward_and_reverse_analysis(
            forward_reverse_samples
        )

        # Gather overlap matrix
        self._overlap_matrix = self.get_overlap_matrix()

        # Gather exchange transition matrix
        # Note we only generate these for replica exchange calculations
        # TODO: consider if this would also work for SAMS
        if self.sampling_method == 'repex':
            self._exchange_matrix = self.get_exchanges()

    @staticmethod
    def _get_free_energy(
        analyzer: multistate.MultiStateSamplerAnalyzer,
        u_ln: npt.NDArray, N_l: npt.NDArray,
        return_units: unit.Quantity,
    ) -> tuple[unit.Quantity, unit.Quantity]:
        """
        Helper method to create an MBAR object and extract free energies
        between end states.

        Parameters
        ----------
        analyzer : multistate.MultiStateSamplerAnalyzer
          MultiStateSamplerAnalyzer to extract free eneriges from.
        u_ln : npt.NDArray
          A n_states x (n_sampled_states * n_iterations)
          array of energies (in kT).
        N_l : npt.NDArray
          An array containing the total number of samples drawn from each
          state.
        unit_type : unit.Quantity
          What units to return the free energies in.

        Returns
        -------
        DG : unit.Quantity
          The free energy difference between the end states.
        dDG : unit.Quantity
          The MBAR error for the free energy difference estimate.

        TODO
        ----
        * Allow folks to pass in extra options for bootstrapping etc..
        * Add standard test against analyzer.get_free_energy()
        """
        mbar = analyzer._create_mbar(u_ln, N_l)

        try:
            # pymbar 3
            DF_ij, dDF_ij = mbar.getFreeEnergyDifferences()
        except AttributeError:
            r = mbar.compute_free_energy_differences()
            DF_ij = r['Delta_f']
            dDF_ij = r['dDelta_f']

        DG = DF_ij[0, -1] * analyzer.kT
        dDG = dDF_ij[0, -1] * analyzer.kT

        return (ensure_quantity(DG, 'openff').to(return_units),
                ensure_quantity(dDG, 'openff').to(return_units))

    def get_equil_free_energy(self) -> tuple[unit.Quantity, unit.Quantity]:
        """
        Extract unbiased and uncorrelated estimates of the free energy
        and the associated error from a MultiStateSamplerAnalyzer object.

        Returns
        -------
        DG : unit.Quantity
          The free energy difference between the end states.
        dDG : unit.Quantity
          The MBAR error for the free energy difference estimate.
        """
        u_ln_decorr = self.analyzer._unbiased_decorrelated_u_ln
        N_l_decorr = self.analyzer._unbiased_decorrelated_N_l

        DG, dDG = self._get_free_energy(
            self.analyzer, u_ln_decorr, N_l_decorr, self.units
        )

        return DG, dDG

    def get_forward_and_reverse_analysis(
        self, num_samples: int = 10
    ) -> dict[str, Union[npt.NDArray, unit.Quantity]]:
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
        forward_reverse : dict[str, Union[npt.NDArray, unit.Quantity]]
          A dictionary containing;
            * ``fractions``: fractions of sample used to calculate free energies
            * ``forward_DGs`` and `forward_dDGs`: the free energy estimates
              and errors along each sample fraction in the forward direction
            * ``reverse_DGs`` and `reverse_dDGs`: the free energy estimates
              and errors along each sample fraction in the reverse direction
        """
        u_ln = self.analyzer._unbiased_decorrelated_u_ln
        N_l = self.analyzer._unbiased_decorrelated_N_l
        n_states = len(N_l)

        # Check that the N_l is the same across all states
        if not np.all(N_l == N_l[0]):
            errmsg = ("The number of samples is not equivalent across all "
                      f"states {N_l}")
            raise ValueError(errmsg)

        # Get the chunks of N_l going from 10% to ~ 100%
        # Note: you always lose out a few data points but it's fine
        chunks = [N_l[0] // num_samples * i
                  for i in range(1, num_samples + 1)]

        forward_DGs = []
        forward_dDGs = []
        reverse_DGs = []
        reverse_dDGs = []
        fractions = []

        for chunk in chunks:
            new_N_l = np.array([chunk for _ in range(n_states)])
            samples = chunk * n_states

            # Forward
            DG, dDG = self._get_free_energy(
                self.analyzer,
                u_ln[:, :samples], new_N_l,
                self.units,
            )
            forward_DGs.append(DG)
            forward_dDGs.append(dDG)

            # Reverse
            DG, dDG = self._get_free_energy(
                self.analyzer,
                u_ln[:, -samples:], new_N_l,
                self.units,
            )
            reverse_DGs.append(DG)
            reverse_dDGs.append(dDG)

            fractions.append(chunk / N_l[0])

        forward_reverse = {
            'fractions': np.array(fractions),
            'forward_DGs': unit.Quantity.from_list(forward_DGs),
            'forward_dDGs': unit.Quantity.from_list(forward_dDGs),
            'reverse_DGs': unit.Quantity.from_list(reverse_DGs),
            'reverse_dDGs': unit.Quantity.from_list(reverse_dDGs)
        }
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
        try:
            # pymbar 3
            overlap_matrix = self.analyzer.mbar.computeOverlap()
        except AttributeError:
            overlap_matrix = self.analyzer.mbar.compute_overlap()

        return overlap_matrix

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
            * ``matrix``: The transition matrix estimate of a replica switchin
              from state i to state j.
        """
        # Get replica mixing statistics
        mixing_stats = self.analyzer.generate_mixing_statistics()
        transition_matrix = {'eigenvalues': mixing_stats.eigenvalues,
                             'matrix': mixing_stats.transition_matrix}
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
        if hasattr(self, '_exchange_matrix'):
            return self._exchange_matrix
        else:
            errmsg = ("Exchange matrix was not generated, this is likely "
                      f"{self.sampling_method} is not repex.")
            raise ValueError(errmsg)

    @property
    def _unit_results_dict(self):
        results_dict = {
            'unit_estimate': self.free_energy,
            'unit_estimate_error': self.free_energy_error,
            '_unit_mbar_overlap': self.free_energy_overlaps,
            '_forward_and_reverse_energies': self.forward_and_reverse_free_energies,
            '_production_iterations': self.production_iterations,
            '_equilibration_iterations': self.equilibration_iterations,
            '_replica_states': self.replica_states}

        if hasattr(self, '_exchange_matrix'):
            results_dict['_replica_exchange_statistics'] = self.replica_exchange_statistics

        return results_dict

    def close(self):
        self.analyzer.clear()


def plot_lambda_transition_matrix(matrix: npt.NDArray) -> plt.Axes:
    """
    Plot out a transition matrix.

    Parameters
    ----------
    matrix : npt.NDArray
      A nstates by nstates matrix of transition estimates.

    Returns
    -------
    ax : matplotlib.pyplot.Axes
      An Axes object to plot.
    """
    num_states = len(matrix)
    fig, ax = plt.subplots(figsize=(num_states / 2, num_states / 2))
    ax.axis('off')
    for i in range(num_states):
        if i != 0:
            ax.axvline(x=i, ls="-", lw=0.5, color="k", alpha=0.25)
            ax.axhline(y=i, ls="-", lw=0.5, color="k", alpha=0.25)
        for j in range(num_states):
            val = matrix[i, j]
            val_str = "{:.2f}".format(val)[1:]
            rel_prob = val / matrix.max()

            # shade box
            ax.fill_between(
                [i, i+1], [num_states - j, num_states - j],
                [num_states - (j + 1), num_states - (j + 1)],
                color='k', alpha=rel_prob
            )
            # annotate box
            ax.annotate(
                val_str, xy=(i, j), xytext=(i+0.5, num_states - (j + 0.5)),
                size=8, va="center", ha="center",
                color=("k" if rel_prob < 0.5 else "w"),
            )

        # anotate axes
        base_settings = {
            'size': 10, 'va': 'center', 'ha': 'center', 'color': 'k',
            'family': 'sans-serif'
        }
        for i in range(num_states):
            ax.annotate(
                i, xy=(i + 0.5, 1), xytext=(i + 0.5, num_states + 0.5),
                **base_settings,
            )
            ax.annotate(
                i, xy=(-0.5, num_states - (num_states - 0.5)),
                xytext=(-0.5, num_states - (i + 0.5)),
                **base_settings,
            )

        ax.annotate(
            r"$\lambda$", xy=(-0.5, num_states - (num_states - 0.5)),
            xytext=(-0.5, num_states + 0.5),
            **base_settings,
        )

    # add border
    ax.plot([0, num_states], [0, 0], "k-", lw=2.0)
    ax.plot([num_states, num_states], [0, num_states], "k-", lw=2.0)
    ax.plot([0, num_states], [num_states, num_states], "k-", lw=2.0)
    ax.plot([0, 0], [0, num_states], "k-", lw=2.0)

    return ax


def plot_convergence(
    forward_and_reverse: dict[str, Union[npt.NDArray, unit.Quantity]],
    units: unit.Quantity
) -> plt.Axes:
    """
    Plot a Reverse and Forward convergence analysis of the
    free energies.

    Parameters
    ----------
    forward_and_reverse : dict[str, npt.NDArray]
      A dictionary containing the reverse and forward
      values of the free energies sampled along a given fraction
      of the sample size.
    units : unit.Quantity
      The units the free energies are provided in.

    Returns
    -------
    ax : matplotlib.pyplot.Axes
      An Axes object to plot.
    """
    known_units = {
        'kilojoule_per_mole': 'kj/mol',
        'kilojoules_per_mole': 'kj/mol',
        'kilocalorie_per_mole': 'kcal/mol',
        'kilocalories_per_mole': 'kcal/mol',
    }

    try:
        plt_units = known_units[str(units)]
    except KeyError:
        errmsg = (f"Unknown plotting units {units} passed, acceptable "
                  "values are kilojoule(s)_per_mole and "
                  "kilocalorie(s)_per_mole")
        raise ValueError(errmsg)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Old style alchemical analysis formatting
    plt.setp(ax.spines["bottom"], color="#D2B9D3", lw=3, zorder=-2)
    plt.setp(ax.spines["left"], color="#D2B9D3", lw=3, zorder=-2)

    for dire in ["top", "right"]:
        ax.spines[dire].set_color("none")

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    # Set the overall error bar to the final error for the reverse results
    overall_error = forward_and_reverse['reverse_dDGs'][-1].m
    final_value = forward_and_reverse['reverse_DGs'][-1].m
    ax.fill_between([0, 1],
                    final_value - overall_error,
                    final_value + overall_error,
                    color='#D2B9D3', zorder=1)

    ax.errorbar(
        forward_and_reverse['fractions'],
        [val.m
         for val in forward_and_reverse['forward_DGs']],
        yerr=[err.m
              for err in forward_and_reverse['forward_dDGs']],
        color="#736AFF", lw=3, zorder=2,
        marker="o", mfc="w", mew=2.5,
        mec="#736AFF", ms=8, label='Forward'
    )

    ax.errorbar(
        forward_and_reverse['fractions'],
        [val.m
         for val in forward_and_reverse['reverse_DGs']],
        yerr=[err.m
              for err in forward_and_reverse['reverse_dDGs']],
        color="#C11B17", lw=3, zorder=2,
        marker="o", mfc="w", mew=2.5,
        mec="#C11B17", ms=8, label='Reverse',
    )
    ax.legend(frameon=False)

    ax.set_ylabel(r'$\Delta G$' + f' ({plt_units})')
    ax.set_xlabel('Fraction of uncorrelated samples')

    return ax


def plot_replica_timeseries(
    state_timeseries: npt.NDArray,
    equilibration_iterations: Optional[int] = None,
) -> plt.Axes:
    """
    Plot a the state timeseries of a set of replicas.

    Parameters
    ----------
    state_timeseries : npt.NDArray
      A 2D n_iterattions by n_states array of the replica timeseries.
    equilibration_iterations : Optional[int]
      The number of iterations used up as equilibration time.

    Returns
    -------
    ax : matplotlib.pyplot.Axes
      An Axes object to plot.
    """
    num_states = len(state_timeseries.T)

    fig, ax = plt.subplots(figsize=(num_states, 4))
    iterations = [i for i in range(len(state_timeseries))]

    for i in range(num_states):
        ax.scatter(iterations, state_timeseries.T[i], label=f'replica {i}', s=8)

    ax.set_xlabel("Number of simulation iterations")
    ax.set_ylabel("Lambda state")
    ax.set_title("Change in replica lambda state over time")

    if equilibration_iterations is not None:
        ax.axvline(
            x=equilibration_iterations, color='grey',
            linestyle='--', label='equilibration limit'
        )

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return ax
