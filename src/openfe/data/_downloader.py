import pooch

from ._registry import zenodo_data_registry


def retrieve_all_test_data(zenodo_registry: list[dict], path: str) -> None:
    """Helper function for pulling all test data up-front.

    Parameters
    ----------
    path : str
        path to store the data - usually a pooch.os_cache instance.

    """
    downloader = pooch.DOIDownloader(progressbar=True)

    def _infer_processor(fname: str):
        if fname.endswith("tar.gz"):
            return pooch.Untar()
        elif fname.endswith("zip"):
            return pooch.Unzip()
        else:
            return None

    for d in zenodo_registry:
        pooch.retrieve(
            url=d["base_url"] + d["fname"],
            known_hash=d["known_hash"],
            fname=d["fname"],
            processor=_infer_processor(d["fname"]),
            downloader=downloader,
            path=path,
        )
