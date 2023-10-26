# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import click
from openfecli import OFECommandPlugin
from openfecli.clicktypes import HyphenAwareChoice
import pathlib


def _get_column(val):
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


def is_results_json(f):
    # sanity check on files before we try and deserialize
    return 'estimate' in open(f, 'r').read(20)


def load_results(f):
    # path to deserialized results
    import json
    from gufe.tokenization import JSON_HANDLER

    return json.load(open(f, 'r'), cls=JSON_HANDLER.decoder)


def get_names(result) -> tuple[str, str]:
    # Result to tuple of ligand names
    nm = list(result['unit_results'].values())[0]['name']
    toks = nm.split()
    if toks[2] == 'repeat':
        return toks[0], toks[1]
    else:
        return toks[0], toks[2]


def get_type(res):
    list_of_pur = list(res['protocol_result']['data'].values())[0]
    pur = list_of_pur[0]
    components = pur['inputs']['stateA']['components']

    if 'solvent' not in components:
        return 'vacuum'
    elif 'protein' in components:
        return 'complex'
    else:
        return 'solvent'


def legacy_get_type(res_fn):
    if 'solvent' in res_fn:
        return 'solvent'
    elif 'vacuum' in res_fn:
        return 'vacuum'
    else:
        return 'complex'


def _generate_bad_legs_error_message(set_vals, ligpair):
    expected_rbfe = {'complex', 'solvent'}
    expected_rhfe = {'solvent', 'vacuum'}
    maybe_rhfe = bool(set_vals & expected_rhfe)
    maybe_rbfe = bool(set_vals & expected_rbfe)
    if maybe_rhfe and not maybe_rbfe:
        msg = (
                "This appears to be an RHFE calculation, but we're "
                f"missing {expected_rhfe - set_vals} runs for the "
                f"edge with ligands {ligpair}."
            )
    elif maybe_rbfe and not maybe_rhfe:
        msg = (
            "This appears to be an RBFE calculation, but we're "
            f"missing {expected_rbfe - set_vals} runs for the "
            f"edge with ligands {ligpair}."
        )
    elif maybe_rbfe and maybe_rhfe:
        msg = (
            "Unable to determine whether this is an RBFE "
            f"or an RHFE calculation. Found legs {set_vals} "
            f"for ligands {ligpair}. Those ligands are missing one "
            f"of: {expected_rhfe | expected_rbfe - set_vals}."
        )
    else:
        # this should never happen
        msg = (
            "Something went very wrong while determining the type "
            f"of RFE calculation. For the ligand pair {ligpair}, "
            f"we found legs labelled {set_vals}. We expected either "
            f"{expected_rhfe} or {expected_rbfe}."
        )

    msg += (
        "\n\nYou can force partial gathering of results, without "
        "problematic edges, by using the --allow-partial flag of the gather "
        "command. Note that this may cause problems with predicting "
        "absolute free energies from the relative free energies."
    )
    return msg


def _get_ddgs(legs, error_on_missing=True):
    import numpy as np
    DDGs = []
    for ligpair, vals in sorted(legs.items()):
        set_vals = set(vals)
        DDGbind = None
        DDGhyd = None
        bind_unc = None
        hyd_unc = None

        if set_vals == {'complex', 'solvent'}:
            DG1_mag, DG1_unc = vals['complex']
            DG2_mag, DG2_unc = vals['solvent']
            if not ((DG1_mag is None) or (DG2_mag is None)):
                # DDG(2,1)bind = DG(1->2)complex - DG(1->2)solvent
                DDGbind = (DG1_mag - DG2_mag).m
                bind_unc = np.sqrt(np.sum(np.square([DG1_unc.m, DG2_unc.m])))
        elif set_vals == {'vacuum', 'solvent'}:
            DG1_mag, DG1_unc = vals['solvent']
            DG2_mag, DG2_unc = vals['vacuum']
            if not ((DG1_mag is None) or (DG2_mag is None)):
                DDGhyd = (DG1_mag - DG2_mag).m
                hyd_unc = np.sqrt(np.sum(np.square([DG1_unc.m, DG2_unc.m])))
        else:
            msg = _generate_bad_legs_error_message(set_vals, ligpair)
            if error_on_missing:
                raise RuntimeError(msg)
            else:
                warning.warn(msg)

        DDGs.append((*ligpair, DDGbind, bind_unc, DDGhyd, hyd_unc))

    return DDGs


def _write_ddg(legs, writer):
    DDGs = _get_ddgs(legs)
    writer.writerow(["ligand_i", "ligand_j", "DDG(i->j) (kcal/mol)",
                      "uncertainty (kcal/mol)"])
    for ligA, ligB, DDGbind, bind_unc, DDGhyd, hyd_unc in DDGs:
        name = f"{ligB}, {ligA}"
        if DDGbind is not None:
            DDGbind, bind_unc = format_estimate_uncertainty(DDGbind,
                                                            bind_unc)
            writer.writerow([ligA, ligB, DDGbind, bind_unc])
        if DDGhyd is not None:
            DDGhyd, hyd_unc = format_estimate_uncertainty(DDGbind,
                                                          bind_unc)
            writer.writerow([ligA, ligB, DDGhyd, hyd_unc])


def _write_dg_raw(legs, writer):
    writer.writerow(["leg", "ligand_i", "ligand_j", "DG(i->j) (kcal/mol)",
                     "uncertainty (kcal/mol)"])
    for ligpair, vals in sorted(legs.items()):
        name = ', '.join(ligpair)
        for simtype, (m, u) in sorted(vals.items()):
            if m is None:
                m, u = 'NaN', 'NaN'
            else:
                m, u = format_estimate_uncertainty(m.m, u.m)
            writer.writerow([simtype, *ligpair, m, u])


def _write_dg_mle(legs, writer):
    import networkx as nx
    import numpy as np
    from cinnabar.stats import mle
    DDGs = _get_ddgs(legs)
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
    type=HyphenAwareChoice(['dg', 'ddg', 'dg-raw'], case_sensitive=False),
    default="dg", show_default=True,
    help=(
        "What data to report. 'dg' gives maximum-likelihood estimate of "
        "absolute deltaG,  'ddg' gives delta-delta-G, and 'dg-raw' gives "
        "the raw result of the deltaG for a leg."
    )
)
@click.option('output', '-o',
              type=click.File(mode='w'),
              default='-')
def gather(rootdir, output, report):
    """Gather simulation result jsons of relative calculations to a tsv file

    This walks ROOTDIR recursively and finds all result JSON files from the
    quickrun command (these files must end in .json). Each of these contains
    the results of a separate leg from a relative free energy thermodynamic
    cycle.

    The results reported depend on ``--report`` flag:

    \b
    * 'dg' (default) reports the ligand, its absolute free energy, and
      the associated uncertainty as the maximum likelihood estimate obtained
      from DDG replica averages and standard deviations.
    * 'ddg' reports pairs of ligand_i and ligand_j, the calculated
      relative free energy DDG(i->j) = DG(j) - DG(i) and its uncertainty.
    * 'dg-raw' reports the raw results, giving the leg (vacuum, solvent, or
      complex), ligand_i, ligand_j, the raw DG(i->j) associated with it.

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
    legs = defaultdict(dict)

    for result_fn in result_fns:
        result = load_results(result_fn)
        if result is None:
            continue
        elif result['estimate'] is None or result['uncertainty'] is None:
            click.echo(f"WARNING: Calculations for {result_fn} did not finish succesfully!",
                       err=True)

        try:
            names = get_names(result)
        except KeyError:
            raise ValueError("Failed to guess names")
        try:
            simtype = get_type(result)
        except KeyError:
            simtype = legacy_get_type(result_fn)

        legs[names][simtype] = result['estimate'], result['uncertainty']

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
        'dg-raw': _write_dg_raw,
    }[report.lower()]
    writing_func(legs, writer)


PLUGIN = OFECommandPlugin(
    command=gather,
    section='Quickrun Executor',
    requires_ofe=(0, 6),
)

if __name__ == "__main__":
    gather()
