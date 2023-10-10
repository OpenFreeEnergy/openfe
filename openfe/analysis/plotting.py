# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy.typing as npt
from openff.units import unit
from typing import Optional, Union


def plot_lambda_transition_matrix(matrix: npt.NDArray) -> Axes:
    """
    Plot out a transition matrix.

    Parameters
    ----------
    matrix : npt.NDArray
      A nstates by nstates matrix of transition estimates.

    Returns
    -------
    ax : matplotlib.axes.Axes
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
) -> Axes:
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
    ax : matplotlib.axes.Axes
      An Axes object to plot.
    """
    known_units = {
        'kilojoule_per_mole': 'kJ/mol',
        'kilojoules_per_mole': 'kJ/mol',
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
) -> Axes:
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
    ax : matplotlib.axes.Axes
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
