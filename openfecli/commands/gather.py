# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import click
from openfecli import OFECommandPlugin
import pathlib
from openfecli.utils import write


def is_results_json(f):
    # sanity check on files before we try and deserialize
    return 'estimate' in open(f, 'r').read(20)


def load_results(f):
    # path to deserialized results
    import json
    from gufe.tokenization import JSON_HANDLER

    return json.load(open(f, 'r'), cls=JSON_HANDLER.decoder)


def get_names(result):
    # Result to list of ligand names
    return tuple(list(result['unit_results'].values())[0]['name'].split()[:2])


def get_type(f):
    if 'solvent' in f:
        return 'solvent'
    elif 'vacuum' in f:
        return 'vacuum'
    else:
        return 'complex'

@click.command(
    'gather',
    short_help="Gather DAG result jsons for network of RFE results into single TSV file"
)
@click.argument('rootdir',
                type=click.Path(dir_okay=True, file_okay=False,
                                         path_type=pathlib.Path),
                required=True)
@click.argument('output',
                type=click.File(mode='w'),
                required=True)
def gather(rootdir, output):
    """Gather DAG result jsons of relative calculations and write to single tsv file

    Will walk ROOTDIR recursively and find all results files ending in .json
    (i.e those produced by the quickrun command).

    Paired legs of simulations will be combined to give the DDG values between two ligands
    in the corresponding phase.

    Will produce a **tab** separated file with 3 columns.  Use output = '-' to stream to
    stdout.
    """
    from collections import defaultdict
    import glob
    import numpy as np

    # 1) find all possible jsons
    json_fns = glob.glob(str(rootdir) + '**/*json', recursive=True)

    # 2) filter only result jsons
    result_fns = filter(is_results_json, json_fns)

    # 3) pair legs of simulations together into dict of dicts
    legs = defaultdict(dict)

    for result_fn in result_fns:
        result = load_results(result_fn)
        if result is None:
            continue

        names = get_names(result)
        simtype = get_type(result_fn)

        legs[names][simtype] = result['estimate'], result['uncertainty']

    # 4 for each ligand pair, write out the DDG
    output.write('ligand pair\testimate (kcal/mol)\tuncertainty\n')
    for ligpair, vals in legs.items():
        if 'complex' in vals and 'solvent' in vals:
            DG1_mag, DG1_unc = vals['complex']
            DG2_mag, DG2_unc = vals['solvent']
        elif 'solvent' in vals and 'vacuum' in vals:
            DG1_mag, DG1_unc = vals['solvent']
            DG2_mag, DG2_unc = vals['vacuum']
        else:
            # mismatched legs?
            continue

        # either (DGsolvent - DGvacuum) OR (DGcomplex - DGsolvent)
        DDG = DG1_mag - DG2_mag
        unc = round(np.sqrt(np.sum(np.square([DG1_unc.m, DG2_unc.m]))), 4) * DG1_unc.u

        name = " ".join(ligpair)
        output.write(f'{name}\t{str(DDG.m)}\t+-{str(unc.m)}\n')


PLUGIN = OFECommandPlugin(
    command=gather,
    section='Simulation',
    requires_ofe=(0, 6),
)
