import os
import pathlib
import sys
from typing import List, Literal

import click
import numpy as np
import pandas as pd
from cinnabar import FEMap, Measurement
from openff.units import unit

from openfecli import OFECommandPlugin
from openfecli.clicktypes import HyphenAwareChoice
from openfecli.commands.gather import (
    _collect_result_jsons,
    format_df_with_precision,
    load_json,
    rich_print_to_stdout,
)


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
        names = _get_names(result)
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
    report : Literal["dg", "ddg", "raw"]
        Type of report to generate.

    Returns
    -------
    legs: dict[str, dict[str, list]]
        Data extracted from the given result JSONs, organized by the leg's ligand names and simulation type.
    """
    from collections import defaultdict

    ddgs = defaultdict(lambda: defaultdict(list))

    for result_fn in result_fns:
        names, result = _load_valid_result_json(result_fn)
        if names is None:  # this means it couldn't find names and/or simtype
            continue

        ddgs[names]["overall"].append([result["estimate"], result["uncertainty"]])
        proto_key = [
            k
            for k in result["unit_results"].keys()
            if k.startswith("ProtocolUnitResult")
        ]  # fmt: skip
        for p in proto_key:
            if "unit_estimate" in result["unit_results"][p]["outputs"]:
                simtype = result["unit_results"][p]["outputs"]["simtype"]
                dg = result["unit_results"][p]["outputs"]["unit_estimate"]
                dg_error = result["unit_results"][p]["outputs"]["unit_estimate_error"]

                ddgs[names][simtype].append([dg, dg_error])
            elif "standard_state_correction_A" in result["unit_results"][p]["outputs"]:
                corr_A = result["unit_results"][p]["outputs"]["standard_state_correction_A"]
                corr_B = result["unit_results"][p]["outputs"]["standard_state_correction_B"]
                ddgs[names]["standard_state_correction_A"].append(
                    [corr_A, 0 * unit.kilocalorie_per_mole]
                )
                ddgs[names]["standard_state_correction_B"].append(
                    [corr_B, 0 * unit.kilocalorie_per_mole]
                )
            else:
                continue

    return ddgs


def _get_names(result: dict) -> tuple[str, str]:
    """Get the ligand names from a unit's results data.

    Parameters
    ----------
    result : dict
        A results dict.

    Returns
    -------
    tuple[str, str]
        Ligand names corresponding to the results.
    """

    solvent_data = list(result["protocol_result"]["data"]["solvent"].values())[0][0]

    name_A = solvent_data["inputs"]["alchemical_components"]["stateA"][0]["molprops"]["ofe-name"]
    name_B = solvent_data["inputs"]["alchemical_components"]["stateB"][0]["molprops"]["ofe-name"]

    return str(name_A), str(name_B)


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


def _get_ddgs(
    results_dict: dict[str, dict[str, list]], allow_partial: bool = False
) -> pd.DataFrame:
    """Compute and write out DDG values for the given results.

    Parameters
    ----------
    results_dict : dict[str, dict[str, list]]
        Dictionary of results created by ``_get_legs_from_result_jsons``.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with the ddG results for each ligand pair.
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
    for ligpair, results in sorted(results_dict.items()):
        ddg = np.mean([v[0].m for v in results["overall"]])
        error = error_func(results)
        data.append((ligpair[0], ligpair[1], ddg, error))
    df = pd.DataFrame(
        data,
        columns=[
            "ligand_i",
            "ligand_j",
            "DDG(i->j) (kcal/mol)",
            unc_col_name,
        ],
    )

    return df


def _infer_unc_col_name(df: pd.DataFrame) -> str:
    """Return the full name of the first column name in df containing "uncertainty"."""

    unc_col_name = df.filter(regex="uncertainty").columns[0]
    return unc_col_name


def _generate_ddg(results_dict, allow_partial: bool = False) -> pd.DataFrame:
    df_ddgs = _get_ddgs(results_dict)
    unc_col_name = _infer_unc_col_name(df_ddgs)

    df_out = format_df_with_precision(df_ddgs, "DDG(i->j) (kcal/mol)", unc_col_name, unc_prec=2)
    return df_out


def _generate_dg_mle(
    results_dict: dict[str, dict[str, list]], allow_partial: bool = False
) -> pd.DataFrame:
    """Compute and write out MLE-derived DG values for the given results.

    Parameters
    ----------
    results_dict : dict[str, dict[str, list]]
        Dictionary of results created by ``_get_legs_from_result_jsons``.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with the dG results for each ligand pair.
    """

    DDGs = _get_ddgs(results_dict)

    fe_results = []
    for inx, row in DDGs.iterrows():
        ligA, ligB, DDGbind, bind_unc = row.tolist()
        m = Measurement(
            labelA=ligA,
            labelB=ligB,
            DG=DDGbind * unit.kilocalorie_per_mole,
            uncertainty=bind_unc * unit.kilocalorie_per_mole,
            computational=True,
        )
        fe_results.append(m)

    # Feed into the FEMap object
    femap = FEMap()

    for entry in fe_results:
        femap.add_measurement(entry)

    femap.generate_absolute_values()

    df = femap.get_absolute_dataframe()
    df = df.iloc[:, :3]
    unc_col_name = _infer_unc_col_name(DDGs)
    df.rename(
        {"label": "ligand", "uncertainty (kcal/mol)": unc_col_name}, axis="columns", inplace=True
    )
    df_out = format_df_with_precision(df, "DG (kcal/mol)", unc_col_name, unc_prec=2)
    return df_out


def _generate_raw(
    results_dict: dict[str, dict[str, list]], allow_partial: bool = True
) -> pd.DataFrame:
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
    for ligpair, results in sorted(results_dict.items()):
        for simtype, repeats in sorted(results.items()):
            if simtype != "overall":
                for repeat in repeats:
                    m, u = (repeat[0].m, repeat[1].m)
                    data.append((simtype, ligpair[0], ligpair[1], m, u))

    df = pd.DataFrame(
        data,
        columns=[
            "leg",
            "ligand_i",
            "ligand_j",
            "DG(i->j) (kcal/mol)",
            "MBAR uncertainty (kcal/mol)",
        ],
    )
    df_out = format_df_with_precision(
        df, "DG(i->j) (kcal/mol)", "MBAR uncertainty (kcal/mol)", unc_prec=2
    )

    return df_out


@click.command(
    "gather-septop",
    short_help="Gather result JSONs for a network of SepTop results into a TSV file.",
)
@click.argument(
    "results",
    nargs=-1,  # accept any number of results
    type=click.Path(dir_okay=True, file_okay=True, path_type=pathlib.Path),
    required=True,
)
@click.option(
    "--report",
    type=HyphenAwareChoice(["dg", "ddg", "raw"], case_sensitive=False),
    default="dg",
    show_default=True,
    help=(
        "What data to report. 'dg' gives the maximum-likelihood estimate derived "
        "absolute deltaG value,  'ddg' gives delta-delta-G, and 'raw' gives "
        "the raw result of the deltaG for a leg."
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
def gather_septop(
    results: List[os.PathLike | str],
    output: os.PathLike | str,
    report: Literal["dg", "ddg", "raw"],
    tsv: bool,
    allow_partial: bool,
):
    """
    Gathering of SepTop results with ``openfe gather-septop`` is an experimental feature
    and is subject to change in a future release of openfe!

    Gather simulation result JSON files from SepTop simulations and generate a report.

    RESULTS is the path(s) to JSON files or directories of JSON files containing SepTop protocol results as generated by ``openfe quickrun``.

    All directories will be walked recursively and any valid JSON results files will be gathered.
    Files must end in .json to be collected, and invalid files will be ignored.

    Each JSON contains the results of a separate leg from a Separated Topologies calculation.
    See https://docs.openfree.energy/en/latest/tutorials/septop_tutorial.html for details on running SepTop calculations.

    The results reported depends on ``--report`` flag:

    \b
    * ``--report=dg`` (default) reports the ligand, its absolute free energy, and
      the associated uncertainty as the maximum likelihood estimate obtained
      from DDG replica averages and standard deviations.  These MLE estimates
      are centred around 0.0, and when plotted can be shifted to match
      experimental values.
    * ``--report=ddg`` reports pairs of ligand_i and ligand_j, the calculated
      relative free energy DDG(i->j) = DG(j) - DG(i) and its uncertainty.
    * ``--report=raw`` reports the raw results, which each repeat simulation given
      separately (i.e. no combining of redundant simulations is performed)

    The output is a table of **tab** separated values. By default, this
    outputs to stdout, use the -o option to specify an output filepath.
    """

    msg = "WARNING! Gathering of SepTop results with `openfe gather-septop` is an experimental feature and is subject to change in a future release of openfe."
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
        "dg": _generate_dg_mle,
        "ddg": _generate_ddg,
        "raw": _generate_raw,
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
    command=gather_septop,
    section="Quickrun Executor",
    requires_ofe=(0, 6),
)
