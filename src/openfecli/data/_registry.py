"""Registry for all remotely-stored CLI test data."""

import pooch

POOCH_CACHE = pooch.os_cache("openfe")
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

zenodo_data_registry = [
    zenodo_cmet_data,
    zenodo_rbfe_serial_data,
    zenodo_rbfe_parallel_data,
    zenodo_abfe_data,
    zenodo_septop_data,
]
