import os
import pathlib
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
    try:
        nm = list(result["unit_results"].values())[0]["name"]

    except KeyError:
        raise ValueError("Failed to guess name")

    toks = nm.split("Binding, ")
    if "solvent" in toks[1]:
        name = toks[1].split(" solvent")[0]
    if "complex" in toks[1]:
        name = toks[1].split(" complex")[0]
    return name


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
    complex_errors = [x[1].m for x in r["complex"]]
    solvent_errors = [x[1].m for x in r["solvent"]]
    return np.sqrt(np.mean(complex_errors) ** 2 + np.mean(solvent_errors) ** 2)


def extract_results_dict(
    results_files: list[os.PathLike | str],
) -> dict[str, dict[str, list]]:
    """
    Get a dictionary of ABFE results from a list of directories.

    Parameters
    ----------
    results_files : list[ps.PathLike | str]
        A list of directors with ABFE result files to process.

    Returns
    -------
    sim_results : dict[str, dict[str, list]]
        Simulation results, organized by the leg's ligand names and simulation type.
    """
    # find and filter result jsons
    result_fns = _collect_result_jsons(results_files)
    # pair legs of simulations together into dict of dicts
    sim_results = _get_legs_from_result_jsons(result_fns)

    return sim_results


def generate_dg(results_dict: dict[str, dict[str, list]]) -> pd.DataFrame:
    """Compute and write out DG values for the given results.

    Parameters
    ----------
    results_dict : dict[str, dict[str, list]]
        Dictionary of results created by ``extract_results_dict``.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with the dG results for each ligand.
    """
    data = []
    # check the type of error which should be used based on the number of repeats
    repeats = {len(v["overall"]) for v in results_dict.values()}
    error_func = _error_mbar if 1 in repeats else _error_std
    for lig, results in sorted(results_dict.items()):
        dg = np.mean([v[0].m for v in results["overall"]])
        error = error_func(results)
        data.append((lig, dg, error))

    df = pd.DataFrame(
        data,
        columns=[
            "ligand",
            "DG (kcal/mol)",
            "uncertainty (kcal/mol)",
        ],
    )
    df_out = format_df_with_precision(df, "DG (kcal/mol)", "uncertainty (kcal/mol)", unc_prec=2)
    return df_out


def generate_dg_raw(results_dict: dict[str, dict[str, list]]) -> pd.DataFrame:
    """
    Get all the transformation cycle legs found and their DG values.

    Parameters
    ----------
    results_dict : dict[str, dict[str, list]]
        Dictionary of results created by ``extract_results_dict``.

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
            "uncertainty (kcal/mol)",
        ],
    )
    df_out = format_df_with_precision(df, "DG (kcal/mol)", "uncertainty (kcal/mol)", unc_prec=2)
    return df_out


@click.command("gather-abfe")
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
@click.option("output", "-o", type=click.File(mode="w"), default="-")
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
def gather_abfe(
    results: List[os.PathLike | str],
    output: os.PathLike | str,
    report: Literal["dg", "raw"],
    tsv: bool,
):
    sim_results = extract_results_dict(results)

    if report == "raw":
        df = generate_dg_raw(sim_results)
    elif report == "dg":
        df = generate_dg(sim_results)

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
