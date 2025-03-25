# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import click
import os
import pathlib
from typing import Callable, Literal
import warnings

from openfecli import OFECommandPlugin
from openfecli.clicktypes import HyphenAwareChoice


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

def load_valid_result_json(fpath:os.PathLike|str)->dict|None:
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


    try:
        result = load_json(fpath)
    except FileNotFoundError:
        click.echo(f"Warning: {fpath} does not exist. Skipping.", err=True)
        return None
    if "unit_results" not in result.keys():
        click.echo(f"{fpath}: No 'unit_results' found, assuming to be a failed simulation.", err=True)
        return None
    if result['estimate'] is None:
        click.echo(f"{fpath}: No 'estimate' found, assuming to be a failed simulation.", err=True)
        return None
    if result['uncertainty'] is None:
        click.echo(f"{fpath}: No 'uncertainty' found, assuming to be a failed simulation.", err=True)
        return None
    if all('exception' in u for u in result['unit_results'].values()):
        click.echo(f"{fpath}: Exception found in all 'unit_results', assuming to be a failed simulation.", err=True)
        return None

    return result

def get_names(result:dict) -> tuple[str, str]:
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

    nm = list(result['unit_results'].values())[0]['name']
    toks = nm.split()
    if toks[2] == 'repeat':
        return toks[0], toks[1]
    else:
        return toks[0], toks[2]


def get_type(res:dict)->Literal['vacuum','solvent','complex']:
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


def legacy_get_type(res_fn:os.PathLike|str)->Literal['vacuum','solvent','complex']:
    # TODO: Deprecate this when we no longer rely on key names in `get_type()`

    if 'solvent' in res_fn:
        return 'solvent'
    elif 'vacuum' in res_fn:
        return 'vacuum'
    else:
        return 'complex'

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
    msg="Some edge(s) are missing runs!\nBelow are the problematic edges and run types found:\n\n"
    for leg_types, ligpair in bad_legs:
        msg += f"{ligpair}: {','.join(leg_types)}\n"

    msg += (
        "\nYou can force partial gathering of results, without "
        "problematic edges, by using the --allow-partial flag of the gather "
        "command.\nNOTE: This may cause problems with predicting "
        "absolute free energies from the relative free energies."
    )
    return msg


def _parse_raw_units(results: dict) -> list[tuple]:
    # grab individual unit results from master results dict
    # returns list of (estimate, uncertainty) tuples
    list_of_pur = list(results['protocol_result']['data'].values())

    # could add to each tuple pu[0]["source_key"] for repeat ID
    return [(pu[0]['outputs']['unit_estimate'],
             pu[0]['outputs']['unit_estimate_error'])
            for pu in list_of_pur]

def _get_ddgs(legs:dict, error_on_missing=True):
    import numpy as np
    from openfe.protocols.openmm_rfe.equil_rfe_methods import RelativeHybridTopologyProtocolResult as rfe_result

    DDGs = []
    bad_legs = []
    for ligpair, vals in sorted(legs.items()):
        leg_types = set(vals)
        DDGbind = None
        DDGhyd = None
        bind_unc = None
        hyd_unc = None

        do_rbfe = (len(leg_types & {'complex', 'solvent'}) == 2)
        do_rhfe = (len(leg_types & {'vacuum', 'solvent'}) == 2)

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
            bad_legs.append((leg_types, ligpair))
            continue
        else:
            DDGs.append((*ligpair, DDGbind, bind_unc, DDGhyd, hyd_unc))

    if bad_legs:
        err_msg = _generate_bad_legs_error_message(bad_legs)
        if error_on_missing:
            raise RuntimeError(err_msg)
        else:
            warnings.warn(err_msg)

    return DDGs


def _write_ddg(legs:dict, writer:Callable, allow_partial:bool):
    DDGs = _get_ddgs(legs, error_on_missing=not allow_partial)
    writer.writerow(["ligand_i", "ligand_j", "DDG(i->j) (kcal/mol)",
                     "uncertainty (kcal/mol)"])
    for ligA, ligB, DDGbind, bind_unc, DDGhyd, hyd_unc in DDGs:
        if DDGbind is not None:
            DDGbind, bind_unc = format_estimate_uncertainty(DDGbind, bind_unc)
            writer.writerow([ligA, ligB, DDGbind, bind_unc])
        if DDGhyd is not None:
            DDGhyd, hyd_unc = format_estimate_uncertainty(DDGhyd, hyd_unc)
            writer.writerow([ligA, ligB, DDGhyd, hyd_unc])


def _write_raw(legs:dict, writer:Callable, allow_partial=True):
    writer.writerow(["leg", "ligand_i", "ligand_j",
                     "DG(i->j) (kcal/mol)", "MBAR uncertainty (kcal/mol)"])

    for ligpair, results in sorted(legs.items()):
        for simtype, repeats in sorted(results.items()):
            for repeat in repeats:
                for m, u in repeat:
                    if m is None:
                        m, u = 'NaN', 'NaN'
                    else:
                        m, u = format_estimate_uncertainty(m.m, u.m)
                    writer.writerow([simtype, *ligpair, m, u])


def _write_dg_raw(legs:dict, writer:Callable,  allow_partial):  # pragma: no-cover
    writer.writerow(["leg", "ligand_i", "ligand_j", "DG(i->j) (kcal/mol)",
                     "uncertainty (kcal/mol)"])
    for ligpair, vals in sorted(legs.items()):
        for simtype, (m, u) in sorted(vals.items()):
            if m is None:
                m, u = 'NaN', 'NaN'
            else:
                m, u = format_estimate_uncertainty(m.m, u.m)
            writer.writerow([simtype, *ligpair, m, u])


def _write_dg_mle(legs:dict, writer:Callable, allow_partial:bool):
    import networkx as nx
    import numpy as np
    from cinnabar.stats import mle
    DDGs = _get_ddgs(legs, error_on_missing=not allow_partial)
    MLEs = []
    # 4b) perform MLE
    g = nx.DiGraph()
    nm_to_idx = {}
    DDGbind_count = 0
    for ligA, ligB, DDGbind, bind_unc, DDGhyd, hyd_unc in DDGs:
        if DDGbind is None:
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
        idx_to_nm = {v: k for k, v in nm_to_idx.items()}

        f_i, df_i = mle(g, factor='calc_DDG')
        df_i = np.diagonal(df_i) ** 0.5

        for node, f, df in zip(g.nodes, f_i, df_i):
            ligname = idx_to_nm[node]
            MLEs.append((ligname, f, df))

    writer.writerow(["ligand", "DG(MLE) (kcal/mol)",
                     "uncertainty (kcal/mol)"])
    for ligA, DG, unc_DG in MLEs:
        DG, unc_DG = format_estimate_uncertainty(DG, unc_DG)
        writer.writerow([ligA, DG, unc_DG])


@click.command(
    'gather',
    short_help="Gather result jsons for network of RFE results into a TSV file"
)
@click.argument('rootdir',
                type=click.Path(dir_okay=True, file_okay=False,
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
def gather(rootdir:os.PathLike|str,
           output:os.PathLike|str,
           report:Literal['dg','ddg','raw'],
           allow_partial:bool
           ):
    """Gather simulation result jsons of relative calculations to a tsv file

    This walks ROOTDIR recursively and finds all result JSON files from the
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
    from collections import defaultdict
    import glob
    import csv

    # 1) find all possible jsons
    json_fns = glob.glob(str(rootdir) + '/**/*json', recursive=True)

    # 2) filter only result jsons
    result_fns = filter(is_results_json, json_fns)

    # 3) pair legs of simulations together into dict of dicts
    legs = defaultdict(lambda: defaultdict(list))

    for result_fn in result_fns:
        result = load_valid_result_json(result_fn)
        if result is None:
            continue

        try:
            names = get_names(result)
        except KeyError:
            raise ValueError("Failed to guess names")
        try:
            simtype = get_type(result)
        except KeyError:
            simtype = legacy_get_type(result_fn)

        if report.lower() == 'raw':
            legs[names][simtype].append(_parse_raw_units(result))
        else:
            dGs = [v[0]['outputs']['unit_estimate'] for v in result['protocol_result']['data'].values()]
            ## for jobs run in parallel, we need to compute these values
            legs[names][simtype].extend(dGs)

    writer = csv.writer(
        output,
        delimiter="\t",
        lineterminator="\n",  # to exactly reproduce previous, prefer "\r\n"
    )

    # 5a) write out MLE values
    # 5b) write out DDG values
    # 5c) write out each leg
    writing_func = {
        'dg': _write_dg_mle,
        'ddg': _write_ddg,
        #  'dg-raw': _write_dg_raw,
        'raw': _write_raw,
    }[report.lower()]
    writing_func(legs, writer, allow_partial)


PLUGIN = OFECommandPlugin(
    command=gather,
    section='Quickrun Executor',
    requires_ofe=(0, 6),
)

if __name__ == "__main__":
    gather()
