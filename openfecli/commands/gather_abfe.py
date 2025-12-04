import os
import pathlib
import sys
from typing import List, Literal

import click
import numpy as np
import pandas as pd
from openff.units import unit

from openfecli import OFECommandPlugin
from openfecli.clicktypes import HyphenAwareChoice
from openfecli.commands.gather import (
    _collect_result_jsons,
    format_df_with_precision,
    load_json,
    rich_print_to_stdout,
)


def _get_name(result: dict) -> str:
    """Get the ligand name from a unit's results data.

    Parameters
    ----------
    result : dict
        A results dict.

    Returns
    -------
    str
        Ligand name corresponding to the results.
    """

    solvent_data = list(result["protocol_result"]["data"]["solvent"].values())[0][0]
    name = solvent_data["inputs"]["alchemical_components"]["stateA"][0]["molprops"]["ofe-name"]

    return str(name)


def _load_valid_result_json(fpath: os.PathLike | str) -> tuple[tuple | None, dict | None]:
    """Load the data from a results JSON into a dict.

    Parameters
    ----------
    fpath : os.PathLike | str
        The path to deserialized results.

    Returns
    -------
    dict | None
        A dict containing data from the results JSON,
        or None if the JSON file is invalid or missing.

    Raises
    ------
    ValueError
      If the JSON file contains an ``estimate`` or ``uncertainty`` key with the
      value ``None``.
      If
    """
    # TODO: replace this with gather's _load_valid_result

    # TODO: only load this once during collection, then pass namedtuple(fname, dict) into this function
    # for now though, it's not the bottleneck on performance
    result = load_json(fpath)
    try:
        names = _get_name(result)
    except (ValueError, IndexError):
        click.secho(f"{fpath}: Missing ligand names and/or simulation type. Skipping.",err=True, fg="yellow")  # fmt: skip
        return None, None
    if result["estimate"] is None:
        click.secho(f"{fpath}: No 'estimate' found, assuming to be a failed simulation.",err=True, fg="yellow")  # fmt: skip
        return names, None
    if result["uncertainty"] is None:
        click.secho(f"{fpath}: No 'uncertainty' found, assuming to be a failed simulation.",err=True, fg="yellow")  # fmt: skip
        return names, None
    if all("exception" in u for u in result["unit_results"].values()):
        click.secho(f"{fpath}: Exception found in all 'unit_results', assuming to be a failed simulation.",err=True, fg="yellow")  # fmt: skip
        return names, None
    return names, result


def _get_legs_from_result_jsons(
    result_fns: list[pathlib.Path],
) -> dict[str, dict[str, list]]:
    """
    Iterate over a list of result JSONs and populate a dict of dicts with all data needed
    for results processing.


    Parameters
    ----------
    result_fns : list[pathlib.Path]
        List of filepaths containing results formatted as JSON.
    report : Literal["dg", "raw"]
        Type of report to generate.

    Returns
    -------
    legs: dict[str, dict[str, list]]
        Data extracted from the given result JSONs, organized by the leg's ligand name and simulation type.
    """
    from collections import defaultdict

    dgs = defaultdict(lambda: defaultdict(list))

    for result_fn in result_fns:
        name, result = _load_valid_result_json(result_fn)
        if name is None:  # this means it couldn't find name and/or simtype
            continue

        dgs[name]["overall"].append([result["estimate"], result["uncertainty"]])
        proto_key = [k for k in result["unit_results"].keys() if k.startswith("ProtocolUnitResult")]
        for p in proto_key:
            if "unit_estimate" in result["unit_results"][p]["outputs"]:
                simtype = result["unit_results"][p]["outputs"]["simtype"]
                dg = result["unit_results"][p]["outputs"]["unit_estimate"]
                dg_error = result["unit_results"][p]["outputs"]["unit_estimate_error"]

                dgs[name][simtype].append([dg, dg_error])
            if "standard_state_correction" in result["unit_results"][p]["outputs"]:
                corr = result["unit_results"][p]["outputs"]["standard_state_correction"]
                dgs[name]["standard_state_correction"].append([corr, 0 * unit.kilocalorie_per_mole])
            else:
                continue
    return dgs


def _error_std(r):
    """
    Calculate the error of the estimate as the std of the repeats
    """
    return np.std([v[0].m for v in r["overall"]])


def _error_mbar(r):
    """
    Calculate the error of the estimate using the reported MBAR errors.

    This also takes into account that repeats may have been run for this edge by using the average MBAR error
    """
    complex_errors = np.array([x[1].m for x in r["complex"]])
    solvent_errors = np.array([x[1].m for x in r["solvent"]])
    return np.sqrt(np.mean(complex_errors**2) + np.mean(solvent_errors**2))


def _generate_dg(results_dict: dict[str, dict[str, list]], allow_partial: bool) -> pd.DataFrame:
    """Compute and write out DG values for the given results.

    Parameters
    ----------
    results_dict : dict[str, dict[str, list]]
        Dictionary of results created by ``_get_legs_from_result_jsons``.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with the dG results for each ligand.
    """

    # check the type of error which should be used based on the number of repeats
    n_repeats = {len(v["overall"]) for v in results_dict.values()}

    if 1 in n_repeats:
        error_func = _error_mbar
        unc_col_name = "MBAR uncertainty (kcal/mol)"
    else:
        error_func = _error_std
        unc_col_name = "std dev uncertainty (kcal/mol)"

    data = []
    for lig, results in sorted(results_dict.items()):
        dg = np.mean([v[0].m for v in results["overall"]])
        error = error_func(results)
        data.append((lig, dg, error))

    df = pd.DataFrame(
        data,
        columns=[
            "ligand",
            "DG (kcal/mol)",
            unc_col_name,
        ],
    )
    df_out = format_df_with_precision(df, "DG (kcal/mol)", unc_col_name, unc_prec=2)
    return df_out


def _generate_dg_raw(results_dict: dict[str, dict[str, list]], allow_partial: bool) -> pd.DataFrame:
    """
    Get all the transformation cycle legs found and their DG values.

    Parameters
    ----------
    results_dict : dict[str, dict[str, list]]
        Dictionary of results created by ``_get_legs_from_result_jsons``.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with the individual cycle leg dG results.
    """
    data = []
    for lig, results in sorted(results_dict.items()):
        for simtype, repeats in sorted(results.items()):
            if simtype != "overall":
                for repeat in repeats:
                    measurement, uncertainty = (repeat[0].m, repeat[1].m)
                    data.append((simtype, lig, measurement, uncertainty))

    df = pd.DataFrame(
        data,
        columns=[
            "leg",
            "ligand",
            "DG (kcal/mol)",
            "MBAR uncertainty (kcal/mol)",
        ],
    )
    df_out = format_df_with_precision(
        df, "DG (kcal/mol)", "MBAR uncertainty (kcal/mol)", unc_prec=2
    )
    return df_out


@click.command(
    "gather-abfe", short_help="Gather result JSONs for network of ABFE results into a TSV file."
)
@click.argument(
    "results",
    nargs=-1,  # accept any number of results
    type=click.Path(dir_okay=True, file_okay=True, path_type=pathlib.Path),
    required=True,
)
@click.option(
    "--report",
    type=HyphenAwareChoice(["dg", "raw"], case_sensitive=False),
    default="dg",
    show_default=True,
    help=(
        "What data to report. 'dg' computes the overall binding free energy of each ligand in the dataset (dG), and"
        "'raw' outputs the raw dG values for each individual leg in the ABFE transformation cycles."
    ),
)
@click.option(
    "output",
    "-o",
    type=click.File(mode="w"),
    default="-",
    help="Filepath at which to write the tsv report.",
)
@click.option(
    "--tsv",
    is_flag=True,
    default=False,
    help=(
        "Results that are output to stdout will be formatted as tab-separated, "
        "identical to the formatting used when writing to file."
        "By default, the output table will be formatted for human-readability."
    ),
)
@click.option(
    "--allow-partial",
    is_flag=True,
    default=False,
    help=(
        "Do not raise errors if results are missing parts for some edges. "
        "(Skip those edges and issue warning instead.)"
    ),
)
def gather_abfe(
    results: List[os.PathLike | str],
    output: os.PathLike | str,
    report: Literal["dg", "raw"],
    tsv: bool,
    allow_partial: bool,
):
    """
    WARNING! Gathering of ABFE results with ``openfe gather-abfe`` is an experimental feature
    and is subject to change in a future release of openfe!

    Gather simulation result JSON files from ABFE simulations and generate a report.

    RESULTS is the path(s) to JSON files or directories of JSON files containing ABFE protocol results as generated by ``openfe quickrun``.

    All directories will be walked recursively and any valid JSON results files will be gathered.
    Files must end in .json to be collected, and invalid files will be ignored.

    Each JSON contains the results of a separate leg from a Absolute Binding Free Energy calculation.
    See https://docs.openfree.energy/en/latest/tutorials/abfe_tutorial.html for details on running ABFE calculations.

    The results reported depends on ``--report`` flag:

    \b
    * ``--report=dg`` (default) reports the ligand, its absolute free energy, and
      the associated uncertainty as the maximum likelihood estimate obtained
      from DG replica averages and standard deviations.  These MLE estimates
      are centred around 0.0, and when plotted can be shifted to match
      experimental values.
    * ``--report=raw`` reports the raw results, which each repeat simulation given
      separately (i.e. no combining of redundant simulations is performed)

    The output is a table of **tab** separated values. By default, this
    outputs to stdout, use the -o option to specify an output filepath.
    """
    msg = "WARNING! Gathering of ABFE results with `openfe gather-abfe` is an experimental feature and is subject to change in a future release of openfe."
    click.secho(msg, err=True, fg="yellow")  # fmt: skip

    # find and filter result jsons
    result_fns = _collect_result_jsons(results)

    # pair legs of simulations together into dict of dicts
    legs = _get_legs_from_result_jsons(result_fns)

    if legs == {}:
        click.secho("No results JSON files found.", err=True)
        sys.exit(1)

    # compute report
    report_func = {
        "dg": _generate_dg,
        "raw": _generate_dg_raw,
    }[report.lower()]
    df = report_func(legs, allow_partial)

    # write output
    is_output_file = isinstance(output, click.utils.LazyFile)
    if is_output_file:
        click.echo(f"writing {report} output to '{output.name}'")
    if is_output_file or tsv:
        df.to_csv(output, sep="\t", lineterminator="\n", index=False)

    # TODO: we can add a --pretty flag if we want this to be optional/preserve backwards compatibility
    else:
        rich_print_to_stdout(df)


PLUGIN = OFECommandPlugin(
    command=gather_abfe,
    section="Quickrun Executor",
    requires_ofe=(0, 6),
)
