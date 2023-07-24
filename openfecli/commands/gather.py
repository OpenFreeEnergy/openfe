# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import click
from openfecli import OFECommandPlugin
import pathlib

from typing import Tuple

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
    unc_prec: int = 2,
) -> Tuple[str, str]:
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


def _get_ddgs(legs):
    import numpy as np
    DDGs = []
    for ligpair, vals in sorted(legs.items()):
        DDGbind = None
        DDGhyd = None
        bind_unc = None
        hyd_unc = None

        if 'complex' in vals and 'solvent' in vals:
            DG1_mag, DG1_unc = vals['complex']
            DG2_mag, DG2_unc = vals['solvent']
            if not ((DG1_mag is None) or (DG2_mag is None)):
                # DDG(2,1)bind = DG(1->2)complex - DG(1->2)solvent
                DDGbind = (DG1_mag - DG2_mag).m
                bind_unc = np.sqrt(np.sum(np.square([DG1_unc.m, DG2_unc.m])))
        elif 'solvent' in vals and 'vacuum' in vals:
            DG1_mag, DG1_unc = vals['solvent']
            DG2_mag, DG2_unc = vals['vacuum']
            if not ((DG1_mag is None) or (DG2_mag is None)):
                DDGhyd = (DG1_mag - DG2_mag).m
                hyd_unc = np.sqrt(np.sum(np.square([DG1_unc.m, DG2_unc.m])))
        else:  # -no-cov-
            raise RuntimeError(f"Unknown DDG type for {vals}")

        DDGs.append((*ligpair, DDGbind, bind_unc, DDGhyd, hyd_unc))

    return DDGs

def _write_ddg(legs, output):
    DDGs = _get_ddgs(legs)
    for ligA, ligB, DDGbind, bind_unc, DDGhyd, hyd_unc in DDGs:
        name = f"{ligB}, {ligA}"
        if DDGbind is not None:
            DDGbind, bind_unc = format_estimate_uncertainty(DDGbind,
                                                            bind_unc)
            # DDGbind, bind_unc = dp2(DDGbind), dp2(bind_unc)
            output.write(f'DDGbind({name})\tRBFE\t{ligA}\t{ligB}'
                         f'\t{DDGbind}\t{bind_unc}\n')
        if DDGhyd is not None:
            DDGhyd, hyd_unc = dp2(DDGhyd), dp2(hyd_unc)
            output.write(f'DDGhyd({name})\tRHFE\t{ligA}\t{ligB}\t'
                         f'{DDGhyd}\t{hyd_unc}\n')


    ...

def _write_raw_dg(legs, output):
    for ligpair, vals in sorted(legs.items()):
        name = ', '.join(ligpair)
        for simtype, (m, u) in sorted(vals.items()):
            if m is None:
                m, u = 'NaN', 'NaN'
            else:
                m, u = format_estimate_uncertainty(m.m, u.m)
            output.write(f'DG{simtype}({name})\t{simtype}\t{ligpair[0]}\t'
                         f'{ligpair[1]}\t{m}\t{u}\n')

def _write_dg_mle(legs, output):
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

    for ligA, DG, unc_DG in MLEs:
        DG, unc_DG = format_estimate_uncertainty(DG, unc_DG)
        output.write(f'DGbind({ligA})\tDG(MLE)\tZero\t{ligA}\t{DG}\t{unc_DG}\n')


def dp2(v: float) -> str:
    # turns 0.0012345 -> '0.0012', round() would get this wrong
    import numpy as np
    return np.format_float_positional(v, precision=2, trim='0',
                                      fractional=False)





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
    type=click.Choice(['dg', 'ddg', 'leg'], case_sensitive=False),
    default="dg", show_default=True,
    help=(
        "What data to report. 'dg' gives maximum-likelihood estimate of "
        "asbolute deltaG,  'ddg' gives delta-delta-G, and 'leg' gives the "
        "raw result of the deltaG for a leg."
    )
)
@click.option('output', '-o',
              type=click.File(mode='w'),
              default='-')
def gather(rootdir, output, report):
    """Gather simulation result jsons of relative calculations to a tsv file

    Will walk ROOTDIR recursively and find all results files ending in .json
    (i.e those produced by the quickrun command).  Each of these contains the
    results of a separate leg from a relative free energy thermodynamic cycle.

    Paired legs of simulations will be combined to give the DDG values between
    two ligands in the corresponding phase, producing either binding ('DDGbind')
    or hydration ('DDGhyd') relative free energies.  These will be reported as
    'DDGbind(B,A)' meaning DGbind(B) - DGbind(A), the difference in free energy
    of binding for ligand B relative to ligand A.

    Individual leg results will be also be written.  These are reported as
    either DGvacuum(A,B) DGsolvent(A,B) or DGcomplex(A,B) for the vacuum,
    solvent or complex free energy of transmuting ligand A to ligand B.

    \b
    Will produce a **tab** separated file with 6 columns:
    1) a description of the measurement, for example DDGhyd(A, B)
    2) the type of this measurement, either RBFE or RHFE
    3) the identifier of the first ligand
    4) the identifier of the second ligand
    5) the estimated value (in kcal/mol)
    6) the uncertainty on the value (also kcal/mol)

    By default, outputs to stdout, use -o option to choose file.
    """
    from collections import defaultdict
    import glob

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

    # 4a for each ligand pair, resolve legs
    DDGs = _get_ddgs(legs)

    output.write('measurement\ttype\tligand_i\tligand_j\testimate (kcal/mol)'
                 '\tuncertainty (kcal/mol)\n')

    # 5a) write out MLE values
    # 5b) write out DDG values
    # 5c) write out each leg
    writing_func = {
        'dg': _write_dg_mle,
        'ddg': _write_ddg,
        'leg': _write_raw_dg,
    }[report.lower()]
    writing_func(legs, output)


PLUGIN = OFECommandPlugin(
    command=gather,
    section='Quickrun Executor',
    requires_ofe=(0, 6),
)

if __name__ == "__main__":
    gather()
