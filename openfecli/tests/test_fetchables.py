import pytest
from click.testing import CliRunner
from .conftest import HAS_INTERNET
import pathlib

from openfecli.fetching import FetchablePlugin

from openfecli.fetchables import (
    RBFE_TUTORIAL, RBFE_TUTORIAL_RESULTS, RBFE_SHOWCASE
)

def fetchable_test(fetchable):
    """Unit test to ensure that a given FetchablePlugin works"""
    assert isinstance(fetchable, FetchablePlugin)
    expected_paths = [pathlib.Path(f) for f in fetchable.filenames]
    runner = CliRunner()
    if fetchable.fetcher.REQUIRES_INTERNET and not HAS_INTERNET:  # -no-cov-
        pytest.skip("Internet seems to be unavailable")
    with runner.isolated_filesystem():
        result = runner.invoke(fetchable.command, ['-d' 'output-dir'])
        assert result.exit_code == 0
        for path in expected_paths:
            assert (pathlib.Path("output-dir") / path).exists()

@pytest.mark.flaky(reruns=3)  # in case of Too Many Request error
def test_rbfe_tutorial():
    fetchable_test(RBFE_TUTORIAL)

@pytest.mark.flaky(reruns=3)  # in case of Too Many Request error
def test_rbfe_tutorial_results():
    fetchable_test(RBFE_TUTORIAL_RESULTS)

@pytest.mark.flaky(reruns=3)  # in case of Too Many Request error
def test_rbfe_showcase():
    fetchable_test(RBFE_SHOWCASE)
