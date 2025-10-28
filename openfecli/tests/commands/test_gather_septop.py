import glob
from typing import Callable
from click.testing import CliRunner
import os
import pathlib
import pytest
import pooch

from ..utils import assert_click_success
from ..conftest import HAS_INTERNET

from unittest import mock

POOCH_CACHE = pooch.os_cache("openfe")
ZENODO_RBFE_DATA = pooch.create(
    path=POOCH_CACHE,
    base_url="doi:10.5281/zenodo.17019063",
    registry={"septop_results.zip": "md5:2355ecc80e03242a4c7fcbf20cb45487"},
    retry_if_failed=5,
)


@pytest.fixture
def septop_result_dir() -> pathlib.Path:
    ZENODO_RBFE_DATA.fetch(f"septop_results.zip", processor=pooch.Untar())
    result_dir = pathlib.Path(POOCH_CACHE) / f"septop_results.untar/septop_results/"

    return result_dir


def test_septop_gather(septop_result_dir, dataset):
    results = septop_result_dir(dataset)
