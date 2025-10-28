from cinnabar import Measurement, FEMap
import click
import numpy as np
from openff.units import unit
import os
import pathlib
import pandas as pd
from typing import Literal, List

from openfecli.commands.gather import (
    format_estimate_uncertainty,
    _collect_result_jsons,
    load_json,
)
from openfecli import OFECommandPlugin
from openfecli.clicktypes import HyphenAwareChoice


def _load_valid_result_json(
    fpath: os.PathLike | str,
) -> tuple[tuple | None, dict | None]:
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
        print(f"{fpath}: Missing ligand names. Skipping.")
        return None, None
    if result["estimate"] is None:
        errormsg = f"{fpath}: No 'estimate' found, assuming to be a failed simulation."
        raise ValueError(errormsg)

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
    # TODO: use gather's _get_names once it is improves (uses input.stateA/B.name)
    try:
        nm = list(result["unit_results"].values())[0]["name"]

    except KeyError:
        raise ValueError("Failed to guess names")

    toks = nm.split(",")
    toks = toks[1].split()
    return toks[1], toks[3]


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
    Get a dictionary of SepTop results from a list of directories.

    Parameters
    ----------
    results_files : list[ps.PathLike | str]
        A list of directors with SepTop result files to process.

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


def generate_ddg(results_dict: dict[str, dict[str, list]]) -> pd.DataFrame:
    """Compute and write out DDG values for the given results.

    Parameters
    ----------
    results_dict : dict[str, dict[str, list]]
        Dictionary of results created by ``extract_results_dict``.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with the ddG results for each ligand pair.
    """
    data = []
    # check the type of error which should be used based on the number of repeats
    repeats = {len(v["overall"]) for v in results_dict.values()}
    error_func = _error_mbar if 1 in repeats else _error_std
    for ligpair, results in sorted(results_dict.items()):
        ddg = np.mean([v[0].m for v in results["overall"]])
        error = error_func(results)
        m, u = format_estimate_uncertainty(ddg, error, unc_prec=2)
        data.append((ligpair[0], ligpair[1], m, u))

    df = pd.DataFrame(
        data,
        columns=[
            "ligand_i",
            "ligand_j",
            "DDG(i->j) (kcal/mol)",
            "uncertainty (kcal/mol)",
        ],
    )
    return df


def generate_dg_mle(results_dict: dict[str, dict[str, list]]) -> pd.DataFrame:
    """Compute and write out MLE-derived DG values for the given results.

    Parameters
    ----------
    results_dict : dict[str, dict[str, list]]
        Dictionary of results created by ``extract_results_dict``.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with the dG results for each ligand pair.
    """

    DDGs = generate_ddg(results_dict)
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
    df.rename({"label": "ligand"}, axis="columns", inplace=True)

    return df


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
    for ligpair, results in sorted(results_dict.items()):
        for simtype, repeats in sorted(results.items()):
            if simtype != "overall":
                for repeat in repeats:
                    m, u = format_estimate_uncertainty(repeat[0].m, repeat[1].m, unc_prec=2)
                    data.append((simtype, ligpair[0], ligpair[1], m, u))

    df = pd.DataFrame(
        data,
        columns=[
            "leg",
            "ligand_i",
            "ligand_j",
            "DG(i->j) (kcal/mol)",
            "uncertainty (kcal/mol)",
        ],
    )
    return df


@click.command("gather-septop")
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
        "What data to report. 'dg' gives maximum-likelihood estimate of "
        "absolute deltaG,  'ddg' gives delta-delta-G, and 'raw' gives "
        "the raw result of the deltaG for a leg."
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
def gather_septop(
    results: List[os.PathLike | str],
    output: os.PathLike | str,
    report: Literal["dg", "ddg", "raw"],
    tsv: bool,):

    ddgs = extract_results_dict(results)
    if report == "ddg":
        df_ddg = generate_ddg(ddgs)
        df_ddg.to_csv(output, sep="\t", lineterminator="\n", index=False)

    if report == "dg":
        df_dg = generate_dg_mle(ddgs)
        df_dg.to_csv("dg.tsv", sep="\t", lineterminator="\n", index=False)

    if report == "raw":
        df_raw = generate_dg_raw(ddgs)
        df_raw.to_csv(output, sep="\t", lineterminator="\n", index=False)


PLUGIN = OFECommandPlugin(
    command=gather_septop,
    section="Quickrun Executor",
    requires_ofe=(0, 6),
)


def main():
    gather_septop()
