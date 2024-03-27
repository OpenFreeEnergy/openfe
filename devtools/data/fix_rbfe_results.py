"""A script to fix up rbfe_results.tar.gz

Useful if Settings are ever changed in a backwards-incompatible way

Will expect "rbfe_results.tar.gz" in this directory, will overwrite this file
"""
from gufe.tokenization import JSON_HANDLER
import glob
import json
from openfe.protocols import openmm_rfe
import os.path
import tarfile


def untar(fn):
    """extract tarfile called *fn*"""
    with tarfile.open(fn) as f:
        f.extractall()


def retar(loc, name):
    """create tar.gz called *name* of directory *loc*"""
    with tarfile.open(name, mode='w:gz') as f:
        f.add(loc, arcname=os.path.basename(loc))


def replace_settings(fn, new_settings):
    """replace settings instances in *fn* with *new_settings*"""
    with open(fn, 'r') as f:
        data = json.load(f)

    for k in data['protocol_result']['data']:
        data['protocol_result']['data'][k][0]['inputs']['settings'] = new_settings

    for k in data['unit_results']:
        data['unit_results'][k]['inputs']['settings'] = new_settings

    with open(fn, 'w') as f:
        json.dump(data, f, cls=JSON_HANDLER.encoder)


def fix_rbfe_results():
    untar('rbfe_results.tar.gz')

    # generate valid settings as defaults
    new_settings = openmm_rfe.RelativeHybridTopologyProtocol.default_settings()

    # walk over all result jsons
    for fn in glob.glob('./results/*json'):
        # replace instances of settings within with valid settings
        replace_settings(fn, new_settings)

    retar('results', 'rbfe_results.tar.gz')


if __name__ == '__main__':
    fix_rbfe_results()
