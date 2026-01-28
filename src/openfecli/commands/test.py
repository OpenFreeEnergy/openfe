import os
import pathlib
import sys

import click
import pooch
import pytest

from openfecli import OFECommandPlugin
from openfecli.utils import POOCH_CACHE, write


def retrieve_all_test_data(path):
    downloader = pooch.DOIDownloader(progressbar=True)

    zenodo_cmet_data = dict(
        base_url="doi:10.5281/zenodo.15200083/",
        fname="cmet_results.tar.gz",
        known_hash="md5:a4ca67a907f744c696b09660dc1eb8ec",
    )
    zenodo_rbfe_serial_data = dict(
        base_url="doi:10.5281/zenodo.15042470/",
        fname="rbfe_results_serial_repeats.tar.gz",
        known_hash="md5:2355ecc80e03242a4c7fcbf20cb45487",
    )
    zenodo_rbfe_parallel_data = dict(
        base_url="doi:10.5281/zenodo.15042470/",
        fname="rbfe_results_parallel_repeats.tar.gz",
        known_hash="md5:ff7313e14eb6f2940c6ffd50f2192181",
    )
    zenodo_abfe_data = dict(
        base_url="doi:10.5281/zenodo.17348229/",
        fname="abfe_results.zip",
        known_hash="md5:547f896e867cce61979d75b7e082f6ba",
    )
    zenodo_septop_data = dict(
        base_url="doi:10.5281/zenodo.17435569/",
        fname="septop_results.zip",
        known_hash="md5:2cfa18da59a20228f5c75a1de6ec879e",
    )

    def _infer_processor(fname: str):
        if fname.endswith("tar.gz"):
            return pooch.Untar()
        elif fname.endswith("zip"):
            return pooch.Unzip()
        else:
            return None

    for d in [
        zenodo_cmet_data,
        zenodo_rbfe_serial_data,
        zenodo_rbfe_parallel_data,
        zenodo_abfe_data,
        zenodo_septop_data,
    ]:
        pooch.retrieve(
            url=d["base_url"] + d["fname"],
            known_hash=d["known_hash"],
            fname=d["fname"],
            processor=_infer_processor(d["fname"]),
            downloader=downloader,
            path=path,
        )


@click.command("test", short_help="Run the OpenFE test suite")
@click.option('--long', is_flag=True, default=False, help="Run additional tests (takes much longer)")  # fmt: skip
@click.option(
    "--download-only",
    is_flag=True,
    default=False,
    help="Download data to the cache (this is helpful if internet is flaky).",
)
def test(long, download_only):
    """
    Run the OpenFE test suite. This first checks that OpenFE is correctly
    imported, and then runs the main test suite, which should take several
    minutes. If given the ``--long`` flag, this will include several tests
    that take significantly longer, but ensure that we're able to fully run
    in your environment.

    A successful run will include tests that pass, skip, or "xfail". In many
    terminals, these show as green or yellow. Warnings are not a concern.
    However, You should not see anything that fails or errors (red).
    """

    if download_only:
        retrieve_all_test_data(POOCH_CACHE)
        sys.exit(0)

    try:
        old_env = dict(os.environ)
        os.environ["OFE_SLOW_TESTS"] = str(long)

        write("Testing can import....")
        import openfe

        write("Running the main package tests")
        return_value = pytest.main(["-v", "--pyargs", "openfe", "--pyargs", "openfecli"])
    finally:
        os.environ.clear()
        os.environ.update(old_env)
        sys.exit(return_value)


PLUGIN = OFECommandPlugin(test, "Miscellaneous", requires_ofe=(0, 7, 5))
