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
@click.option('output', '-o',
              type=click.File(mode='w'),
              default='-')
def gather(rootdir, output):
    """Gather simulation result jsons of relative calculations and write to single tsv file

    Will walk ROOTDIR recursively and find all results files ending in .json (i.e those produced by the quickrun
    command).  Each of these contains the results of a separate leg from a relative free energy thermodynamic cycle.

    Paired legs of simulations will be combined to give the DDG values between two ligands in the corresponding phase,
    producing either binding ('DDGbind') or hydration ('DDGhyd') relative free energies.  These will be reported as
    'DDGbind(B,A)' meaning DGbind(B) - DGbind(A), the difference in free energy of binding for ligand B relative to
    ligand A.

    Individual leg results will be also be written.  These are reported as either DGvacuum(A,B) DGsolvent(A,B) or
    DGcomplex(A,B) for the vacuum, solvent or complex free energy of transmuting ligand A to ligand B.

    \b
    Will produce a **tab** separated file with 3 columns:
    1) a description of the measurement, for example DDGhyd(A, B)
    2) the estimated value (in kcal/mol)
    3) the uncertainty on the value (also kcal/mol)

    By default, outputs to stdout, use -o option to choose file.
    """
    from collections import defaultdict
    import glob
    import numpy as np

    def dp2(v: float) -> str:
        # turns 0.0012345 -> '0.0012', round() would get this wrong
        return np.format_float_positional(v, precision=2, trim='0', fractional=False)

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

    # 4a for each ligand pair, write out the DDG
    output.write('measurement\ttype\tligand_i\tligand_j\testimate (kcal/mol)\tuncertainty (kcal/mol)\n')
    for ligpair, vals in legs.items():
        DDGbind = None
        DDGhyd = None
        bind_unc = None
        hyd_unc = None

        if 'complex' in vals and 'solvent' in vals:
            DG1_mag, DG1_unc = vals['complex']
            DG2_mag, DG2_unc = vals['solvent']
            # DDG(2,1)bind = DG(1->2)complex - DG(1->2)solvent
            DDGbind = dp2((DG1_mag - DG2_mag).m)
            bind_unc = dp2(np.sqrt(np.sum(np.square([DG1_unc.m, DG2_unc.m]))))
        if 'solvent' in vals and 'vacuum' in vals:
            DG1_mag, DG1_unc = vals['solvent']
            DG2_mag, DG2_unc = vals['vacuum']
            DDGhyd = dp2((DG1_mag - DG2_mag).m)
            hyd_unc = dp2(np.sqrt(np.sum(np.square([DG1_unc.m, DG2_unc.m]))))

        name = ", ".join(ligpair[::-1])
        if DDGbind is not None:
            output.write(f'DDGbind({name})\tRBFE\t{ligpair[-2]}\t{ligpair[-1]}\t{DDGbind}\t{bind_unc}\n')
        if DDGhyd is not None:
            output.write(f'DDGhyd({name})\tRHFE\t{ligpair[-2]}\t{ligpair[-1]}\t{DDGhyd}\t{hyd_unc}\n')

    # 4b write out each leg
    for ligpair, vals in legs.items():
        name = ', '.join(ligpair)
        for simtype, (m, u) in vals.items():
            m, u = dp2(m.m), dp2(u.m)
            output.write(f'DG{simtype}({name})\t{simtype}\t{ligpair[-2]}\t{ligpair[-1]}\t{m}\t{u}\n')


PLUGIN = OFECommandPlugin(
    command=gather,
    section='Simulation',
    requires_ofe=(0, 6),
)
