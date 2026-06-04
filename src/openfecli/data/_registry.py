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
    base_url="doi:10.5281/zenodo.19498687/",
    fname="abfe_results.zip",
    known_hash="md5:44db4ce8195f4fe99989f8f57e0d7081",
)
zenodo_septop_data = dict(
    base_url="doi:10.5281/zenodo.19805681/",
    fname="septop_results.zip",
    known_hash="md5:5de5cac5acdf195a13b0f1ce016a8660",
)

zenodo_data_registry = [
    zenodo_cmet_data,
    zenodo_rbfe_serial_data,
    zenodo_rbfe_parallel_data,
    zenodo_abfe_data,
    zenodo_septop_data,
]
