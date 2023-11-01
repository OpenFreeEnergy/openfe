import os
import importlib
import pytest

import openfe


pytest.importorskip('duecredit')


@pytest.mark.skipif((os.environ.get('DUECREDIT_ENABLE', 'yes').lower()
                     in ('no', '0', 'false')),
                     reason="duecredit is disabled")
class TestDuecredit:

    @pytest.mark.parametrize('module, dois', [
        ['openfe.protocols.openmm_afe.equil_solvation_afe_method',
         ['10.5281/zenodo.596504', '10.48550/arxiv.2302.06758',
          '10.5281/zenodo.596622', '10.1371/journal.pcbi.1005659']],
        ['openfe.protocols.openmm_rfe.equil_rfe_methods',
         ['10.5281/zenodo.1297683', '10.5281/zenodo.596622',
          '10.1371/journal.pcbi.1005659']],
    ])
    def test_duecredit_protocol_collection(self, module, dois):
        importlib.import_module(module)
        for doi in dois:
            assert openfe.due.due.citations[(module, doi)].cites_module

    def test_duecredit_active(self):
        assert openfe.due.due.active
