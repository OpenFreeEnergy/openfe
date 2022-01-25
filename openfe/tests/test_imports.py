import sys
import pytest

from openfe import setup, orchestration, simulation, analysis


@pytest.mark.parametrize('module',
        ['setup', 'orchestration', 'simulation', 'analysis'])
def test_imported(module):
    """Sample test, will always pass so long as import statement worked"""
    assert f"openfe.{module}" in sys.modules
