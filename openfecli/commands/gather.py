# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import click
import os
import pathlib
import pandas as pd
import sys
from typing import Literal, List

from openfecli import OFECommandPlugin
from openfecli.clicktypes import HyphenAwareChoice

FAIL_STR = "Error" # string used to indicate a failed run in output tables.

def _get_column(val:float|int)->int:
    """Determine the index (where the 0th index is the decimal) at which the
    first non-zero value occurs in a full-precision string representation of a value.

    Parameters
    ----------
    val : float|int
        The raw value.

    Returns
    -------
    int
        Column index
    """
    import numpy as np
    if val == 0:
        return 0

    log10 = np.log10(val)

    if log10 >= 0.0:
        col = np.floor(log10 + 1)
    else:
        col = np.floor(log10)
    return int(col)


def format_estimate_uncertainty(
    est: float,
    unc: float,
    unc_prec: int = 1,
) -> tuple[str, str]:
    """Truncate raw estimate and uncertainty values to the appropriate uncertainty.

    Parameters
    ----------
    est : float
        Raw estimate value.
    unc : float
        Raw uncertainty value.
    unc_prec : int, optional
        Precision, by default 1

    Returns
    -------
    tuple[str, str]
        The truncated raw and uncertainty values.
    """

    import numpy as np
    # get the last column needed for uncertainty
    unc_col = _get_column(unc) - (unc_prec - 1)

    if unc_col < 0:
        est_str = f"{est:.{-unc_col}f}"
        unc_str = f"{unc:.{-unc_col}f}"
    else:
        est_str = f"{np.round(est, -unc_col + 1)}"
        unc_str = f"{np.round(unc, -unc_col + 1)}"

    return est_str, unc_str

def is_results_json(fpath:os.PathLike|str)->bool:
    """Sanity check that file is a result json before we try to deserialize"""
    return 'estimate' in open(fpath, 'r').read(20)

def load_json(fpath:os.PathLike|str)->dict:
    """Load a JSON file containing a gufe object.

    Parameters
    ----------
    fpath : os.PathLike | str
        The path to a gufe-serialized JSON.


    Returns
    -------
    dict
        A dict containing data from the results JSON.

    """
    # TODO: move this function to openfe/utils
    import json
    from gufe.tokenization import JSON_HANDLER

    return json.load(open(fpath, 'r'), cls=JSON_HANDLER.decoder)

def _get_names(result:dict) -> tuple[str, str]:
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
    try:
        nm = list(result['unit_results'].values())[0]['name']

    except KeyError:
        raise ValueError("Failed to guess names")

    # TODO: make this more robust by pulling names from inputs.state[A/B].name

    toks = nm.split()
    if toks[2] == 'repeat':
        return toks[0], toks[1]
    else:
        return toks[0], toks[2]

def _get_type(res:dict)->Literal['vacuum','solvent','complex']:
    """Determine the simulation type based on the component names."""
    # TODO: use component *types* instead here
    list_of_pur = list(res['protocol_result']['data'].values())[0]
    pur = list_of_pur[0]
    components = pur['inputs']['stateA']['components']

    if 'solvent' not in components:
        return 'vacuum'
    elif 'protein' in components:
        return 'complex'
    else:
        return 'solvent'

def _legacy_get_type(res_fn:os.PathLike|str)->Literal['vacuum','solvent','complex']:
    # TODO: Deprecate this when we no longer rely on key names in `_get_type()`

    if 'solvent' in res_fn:
        return 'solvent'
    elif 'vacuum' in res_fn:
        return 'vacuum'
    # TODO: if there is no identifier in the filename, do we really want to assume it's a complex?
    else:
        return 'complex'

def _get_result_id(
    result: dict, result_fn: os.PathLike | str
) -> tuple[tuple[str, str], Literal["vacuum", "solvent", "complex"]]:
    """Extract the name and simulation type from a results dict.

    Parameters
    ----------
    result : dict
        A result object
    result_fn : os.PathLike | str
        The path to deserialized results, only used if unable to extract from results dict.
        TODO: only take in ``result_fn`` for backwards compatibility, remove this in 2.0

    Returns
    -------
    tuple
        Identifying information (ligand names and simulation type) for the given results data.
    """
    ligA, ligB = _get_names(result)

    try:
        simtype = _get_type(result)
    except KeyError:
        simtype = _legacy_get_type(result_fn)

    return (ligA, ligB), simtype

def _load_valid_result_json(fpath:os.PathLike|str)->tuple[tuple|None, dict|None]:
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

    """

    # TODO: only load this once during collection, then pass namedtuple(fname, dict) into this function
    # for now though, it's not the bottleneck on performance
    result = load_json(fpath)
    try:
        result_id = _get_result_id(result, fpath)
    except (ValueError, IndexError):
        click.secho(f"{fpath}: Missing ligand names and/or simulation type. Skipping.", err=True, fg='yellow')
        return None, None
    if result['estimate'] is None:
        click.secho(f"{fpath}: No 'estimate' found, assuming to be a failed simulation.", err=True, fg='yellow')
        return result_id, None
    if result['uncertainty'] is None:
        click.secho(f"{fpath}: No 'uncertainty' found, assuming to be a failed simulation.", err=True, fg='yellow')
        return result_id, None
    if all('exception' in u for u in result['unit_results'].values()):
        click.secho(f"{fpath}: Exception found in all 'unit_results', assuming to be a failed simulation.", err=True, fg='yellow')
        return result_id, None

    return result_id, result

def _generate_bad_legs_error_message(bad_legs:list[tuple[set[str], tuple[str]]])->str:
    """Format output describing RBFE or RHFE legs that are missing runs.

    Parameters
    ----------
    bad_legs : list[set[str], tuple[str]]]
        A list of tuples of (leg_types, ligpair) pairs from failed edges/legs.

    Returns
    -------
    str
        An error message containing information on all failed legs.
    """
    msg = (
        "\nSome edge(s) are missing runs!\n"
        "The following edges were found but are missing one or more run types "
        "('solvent', 'complex', or 'vacuum') to complete the calculation:\n\n"
        "ligand_i\tligand_j\trun_type_found\n"
    )
    # TODO: format this better
    for ligA, ligB, leg_types in bad_legs:
        msg += f"{ligA}\t{ligB}\t{','.join(leg_types)}\n"

    return msg


def _get_ddgs(legs: dict, allow_partial=False) -> None:
    import numpy as np
    from openfe.protocols.openmm_rfe.equil_rfe_methods import (
        RelativeHybridTopologyProtocolResult as rfe_result,
    )
    # TODO: if there's a failed edge but other valid results in a leg, ddgs will be computed
    # only fails if there are no valid results
    DDGs = []
    bad_legs = []
    for ligpair, vals in sorted(legs.items()):
        leg_types = set(vals)
        # drop any leg types that have no values (these are failed runs)
        valid_leg_types = {k for k in vals if vals[k]}

        DDGbind = None
        DDGhyd = None
        bind_unc = None
        hyd_unc = None

        do_rbfe = (len(valid_leg_types & {'complex', 'solvent'}) == 2)
        do_rhfe = (len(valid_leg_types & {'vacuum', 'solvent'}) == 2)

        if do_rbfe:
            DG1_mag = rfe_result.compute_mean_estimate(vals['complex'])
            DG1_unc = rfe_result.compute_uncertainty(vals['complex'])
            DG2_mag = rfe_result.compute_mean_estimate(vals['solvent'])
            DG2_unc = rfe_result.compute_uncertainty(vals['solvent'])
            if not ((DG1_mag is None) or (DG2_mag is None)):
                # DDG(2,1)bind = DG(1->2)complex - DG(1->2)solvent
                DDGbind = (DG1_mag - DG2_mag).m
                bind_unc = np.sqrt(np.sum(np.square([DG1_unc.m, DG2_unc.m])))

        if do_rhfe:
            DG1_mag = rfe_result.compute_mean_estimate(vals['solvent'])
            DG1_unc = rfe_result.compute_uncertainty(vals['solvent'])
            DG2_mag = rfe_result.compute_mean_estimate(vals['vacuum'])
            DG2_unc = rfe_result.compute_uncertainty(vals['vacuum'])
            if not ((DG1_mag is None) or (DG2_mag is None)):
                DDGhyd = (DG1_mag - DG2_mag).m
                hyd_unc = np.sqrt(np.sum(np.square([DG1_unc.m, DG2_unc.m])))

        if not do_rbfe and not do_rhfe:
            bad_legs.append((*ligpair, leg_types))
            DDGs.append((*ligpair, None, None, None, None))
        else:
            DDGs.append((*ligpair, DDGbind, bind_unc, DDGhyd, hyd_unc))

    if bad_legs:
        err_msg = _generate_bad_legs_error_message(bad_legs)
        if allow_partial:
            click.secho(err_msg, err=True, fg='yellow')
        else:
            err_msg += (
                "\nYou can force partial gathering of results, without "
                "problematic edges, by using the --allow-partial flag of the gather "
                "command.\nNOTE: This may cause problems with predicting "
                "absolute free energies from the relative free energies."
                )
            click.secho(err_msg, err=True, fg='red')
            sys.exit(1)
    return DDGs


def _generate_ddg(legs:dict, allow_partial:bool) -> None:
    """Compute and write out DDG values for the given legs.

    Parameters
    ----------
    legs : dict
        Dict of legs to write out.
    allow_partial : bool
        If ``True``, no error will be thrown for incomplete or invalid results,
        and DDGs will be reported for whatever valid results are found.
    """
    DDGs = _get_ddgs(legs, allow_partial=allow_partial)
    data = []
    for ligA, ligB, DDGbind, bind_unc, DDGhyd, hyd_unc in DDGs:
        if DDGbind is not None:
            DDGbind, bind_unc = format_estimate_uncertainty(DDGbind, bind_unc)
            data.append((ligA, ligB, DDGbind, bind_unc))
        if DDGhyd is not None:
            DDGhyd, hyd_unc = format_estimate_uncertainty(DDGhyd, hyd_unc)
            data.append((ligA, ligB, DDGhyd, hyd_unc))
        elif DDGbind is None and DDGhyd is None:
            data.append((ligA, ligB, FAIL_STR, FAIL_STR))
    df = pd.DataFrame(data, columns=["ligand_i", "ligand_j", "DDG(i->j) (kcal/mol)", "uncertainty (kcal/mol)"])
    return df

def _generate_raw(legs:dict, allow_partial=True) -> None:
    """
    Write out all legs found and their DG values, or indicate that they have failed.

    Parameters
    ----------
    legs : dict
        Dict of legs to write out.
    allow_partial : bool, optional
        Unused for this function, since all results will be included.
    """
    data = []
    for ligpair, results in sorted(legs.items()):
        for simtype, repeats in sorted(results.items()):
            for repeat in repeats:
                for m, u in repeat:
                    if m is None:
                        m, u = FAIL_STR, FAIL_STR
                    else:
                        m, u = format_estimate_uncertainty(m.m, u.m)
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
    return df

def _check_legs_have_sufficient_repeats(legs):
    """Throw an error if all legs do not have 2 or more simulation repeat results"""

    for leg in legs.values():
        for run_type, sim_results in leg.items():
            if len(sim_results) < 2:
                msg='ERROR: Every edge must have at least two simulation repeats'
                click.secho(msg, err=True, fg='red')
                sys.exit(1)

def _generate_dg_mle(legs: dict, allow_partial: bool) -> None:
    """Compute and write out DG values for the given legs.

    Parameters
    ----------
    legs : dict
        Dict of legs to write out.
    allow_partial : bool
        If ``True``, no error will be thrown for incomplete or invalid results,
        and DGs will be reported for whatever valid results are found.
    """
    import networkx as nx
    import numpy as np
    from cinnabar.stats import mle

    _check_legs_have_sufficient_repeats(legs)

    DDGs = _get_ddgs(legs, allow_partial=allow_partial)
    MLEs = []
    expected_ligs = []

    # perform MLE
    g = nx.DiGraph()
    nm_to_idx = {}
    DDGbind_count = 0
    for ligA, ligB, DDGbind, bind_unc, _, _ in DDGs:
        for lig in (ligA, ligB):
            if lig not in expected_ligs:
                expected_ligs.append(lig)

        if DDGbind is None or DDGbind == FAIL_STR:
            continue
        DDGbind_count += 1

        # tl;dr this is super paranoid, but safer for now:
        # cinnabar seems to rely on the ordering of values within the graph
        # to correspond to the matrix that comes out from mle()
        # internally they also convert the ligand names to ints, which I think
        # has a side effect of giving the graph nodes a predictable order.
        # fwiw this code didn't affect ordering locally
        try:
            idA = nm_to_idx[ligA]
        except KeyError:
            idA = len(nm_to_idx)
            nm_to_idx[ligA] = idA
        try:
            idB = nm_to_idx[ligB]
        except KeyError:
            idB = len(nm_to_idx)
            nm_to_idx[ligB] = idB

        g.add_edge(
            idA, idB, calc_DDG=DDGbind, calc_dDDG=bind_unc,
        )

    if DDGbind_count > 2:
        if not nx.is_weakly_connected(g):
            # TODO: dump the network for debugging?
            # TODO: use largest connected component when possible
            msg = (
                "ERROR: The results network is disconnected due to failed or missing edges.\n"
                "Absolute free energies cannot be calculated in a disconnected network.\n"
                "Please either connect the network by addressing failed runs or adding edges.\n"
                "You can still compute relative free energies using the ``--report=ddg`` flag."
            )
            click.secho(msg, err=True, fg='red')
            sys.exit(1)
        idx_to_nm = {v: k for k, v in nm_to_idx.items()}
        f_i, df_i = mle(g, factor="calc_DDG")
        df_i = np.diagonal(df_i) ** 0.5

        for node, f, df in zip(g.nodes, f_i, df_i):
            ligname = idx_to_nm[node]
            MLEs.append((ligname, f, df))
    else:
        click.secho(
            f"The results network has {DDGbind_count} edge(s), but 3 or more edges are required to calculate DG values.",
            err=True,
            fg="red",
        )
        sys.exit(1)

    data = []
    for ligA, DG, unc_DG in MLEs:
        DG, unc_DG = format_estimate_uncertainty(DG, unc_DG)
        data.append({'ligand':ligA,  "DG(MLE) (kcal/mol)": DG, "uncertainty (kcal/mol)": unc_DG})
        expected_ligs.remove(ligA)

    for ligA in expected_ligs:
        data.append({'ligand':ligA,  "DG(MLE) (kcal/mol)": FAIL_STR, "uncertainty (kcal/mol)": FAIL_STR})

    df = pd.DataFrame(data)
    return df

def _collect_result_jsons(results: List[os.PathLike | str]) -> List[pathlib.Path]:
    """Recursively collects all results JSONs from the paths in ``results``,
    which can include directories and/or filepaths.
    """
    import glob

    def collect_jsons(results: List[os.PathLike]):
        all_jsons = []
        for p in results:
            if str(p).endswith("json"):
                all_jsons.append(p)
            elif p.is_dir():
                all_jsons.extend(glob.glob(f"{p}/**/*json", recursive=True))

        return all_jsons

    def is_results_json(fpath: os.PathLike | str) -> bool:
        """Sanity check that file is a result json before we try to deserialize"""
        return "estimate" in open(fpath, "r").read(20)

    results = sorted(results)  # ensures reproducible output order regardless of input order

    # 1) find all possible jsons
    json_fns = collect_jsons(results)

    # 2) filter only result jsons
    result_fns = filter(is_results_json, json_fns)
    return result_fns


def _get_legs_from_result_jsons(
    result_fns: list[pathlib.Path], report: Literal["dg", "ddg", "raw"]
) -> dict[tuple[str, str], dict[str, list]]:
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
    legs: dict[tuple[str,str],dict[str, list]]
        Data extracted from the given result JSONs, organized by the leg's ligand names and simulation type.
    """
    from collections import defaultdict

    legs = defaultdict(lambda: defaultdict(list))

    for result_fn in result_fns:
        result_info, result = _load_valid_result_json(result_fn)

        if result_info is None:  # this means it couldn't find names and/or simtype
            continue
        names, simtype = result_info
        if report.lower() == "raw":
            if result is None:
                parsed_raw_data =[(None, None)]
            else:
                parsed_raw_data = [(v[0]['outputs']['unit_estimate'],
                                    v[0]['outputs']['unit_estimate_error'])
                                    for v in result["protocol_result"]["data"].values()]
            legs[names][simtype].append(parsed_raw_data)
        else:
            if result is None:
                # we want the dict name/simtype entry to exist for error reporting, even if there's no valid data
                dGs = []
            else:
                dGs = [v[0]["outputs"]["unit_estimate"] for v in result["protocol_result"]["data"].values()]
            legs[names][simtype].extend(dGs)

    return legs


def rich_print_to_stdout(df: pd.DataFrame) -> None:
    """Use rich to pretty print a table to stdout."""

    from rich.console import Console
    from rich.table import Table
    from rich import box

    table = Table(box=box.MINIMAL_HEAVY_HEAD)

    for col in df.columns:
        table.add_column(col)

    for row_values in df.values:
        row = [str(val) for val in row_values]
        table.add_row(*row)

    console = Console()
    console.print(table)


@click.command(
    'gather',
    short_help="Gather result jsons for network of RFE results into a TSV file"
)
@click.argument('results',
                nargs=-1,  # accept any number of results
                type=click.Path(dir_okay=True, file_okay=True,
                                path_type=pathlib.Path),
                required=True)
@click.option(
    '--report',
    type=HyphenAwareChoice(['dg', 'ddg', 'raw'],
                           case_sensitive=False),
    default="dg", show_default=True,
    help=(
        "What data to report. 'dg' gives maximum-likelihood estimate of "
        "absolute deltaG,  'ddg' gives delta-delta-G, and 'raw' gives "
        "the raw result of the deltaG for a leg."
    )
)
@click.option('output', '-o',
              type=click.File(mode='w'),
              default='-')
@click.option(
    '--allow-partial', is_flag=True, default=False,
    help=(
        "Do not raise errors if results are missing parts for some edges. "
        "(Skip those edges and issue warning instead.)"
    )
)
def gather(results:List[os.PathLike|str],
           output:os.PathLike|str,
           report:Literal['dg','ddg','raw'],
           allow_partial:bool
           ):
    """Gather simulation result JSON files of relative calculations to a tsv file.

    This walks RESULTS recursively and finds all result JSON files from the
    quickrun command (these files must end in .json). Each of these contains
    the results of a separate leg from a relative free energy thermodynamic
    cycle.

    The results reported depend on ``--report`` flag:

    \b
    * 'dg' (default) reports the ligand, its absolute free energy, and
      the associated uncertainty as the maximum likelihood estimate obtained
      from DDG replica averages and standard deviations.  These MLE estimates
      are centred around 0.0, and when plotted can be shifted to match
      experimental values.
    * 'ddg' reports pairs of ligand_i and ligand_j, the calculated
      relative free energy DDG(i->j) = DG(j) - DG(i) and its uncertainty.
    * 'raw' reports the raw results, which each repeat simulation given
      separately (i.e. no combining of redundant simulations is performed)

    The output is a table of **tab** separated values. By default, this
    outputs to stdout, use the -o option to choose an output file.
    """
    # find and filter result jsons
    result_fns = _collect_result_jsons(results)

    # pair legs of simulations together into dict of dicts
    legs = _get_legs_from_result_jsons(result_fns, report)

    if legs == {}:
        click.secho('No results JSON files found.',err=True)
        sys.exit(1)

    # compute report
    report_func = {
        'dg': _generate_dg_mle,
        'ddg': _generate_ddg,
        'raw': _generate_raw,
    }[report.lower()]
    df = report_func(legs, allow_partial)

    # write output
    if isinstance(output, click.utils.LazyFile):
        click.echo(f"writing {report} output to '{output.name}'")
        df.to_csv(output, sep="\t", lineterminator="\n", index=False)

    # TODO: we can add a --pretty flag if we want this to be optional/preserve backwards compatibility
    else:
        rich_print_to_stdout(df)


PLUGIN = OFECommandPlugin(
    command=gather,
    section='Quickrun Executor',
    requires_ofe=(0, 6),
)

if __name__ == "__main__":
    gather()
