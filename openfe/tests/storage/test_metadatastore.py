import json
import pathlib

import pytest
from gufe.storage.errors import ChangedExternalResourceError, MissingExternalResourceError
from gufe.storage.externalresource import FileStorage
from gufe.storage.externalresource.base import Metadata

from openfe.storage.metadatastore import JSONMetadataStore, PerFileJSONMetadataStore


@pytest.fixture
def json_metadata(tmpdir):
    metadata_dict = {"path/to/foo.txt": {"md5": "bar"}}
    external_store = FileStorage(str(tmpdir))
    with open(tmpdir / "metadata.json", mode="wb") as f:
        f.write(json.dumps(metadata_dict).encode("utf-8"))
    json_metadata = JSONMetadataStore(external_store)
    return json_metadata


@pytest.fixture
def per_file_metadata(tmp_path):
    metadata_dict = {"path": "path/to/foo.txt", "metadata": {"md5": "bar"}}
    external_store = FileStorage(str(tmp_path))
    metadata_loc = "metadata/path/to/foo.txt.json"
    metadata_path = tmp_path / pathlib.Path(metadata_loc)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, mode="wb") as f:
        f.write(json.dumps(metadata_dict).encode("utf-8"))

    per_file_metadata = PerFileJSONMetadataStore(external_store)
    return per_file_metadata


class MetadataTests:
    """Mixin with a few tests for any subclass of MetadataStore"""

    def test_store_metadata(self, metadata):
        raise NotImplementedError()

    def test_load_all_metadata(self):
        raise NotImplementedError("This should call " "self._test_load_all_metadata")

    def test_delete(self):
        raise NotImplementedError("This should call self._test_delete")

    def _test_load_all_metadata(self, metadata):
        expected = {"path/to/foo.txt": Metadata(md5="bar")}
        metadata._metadata_cache = {}
        loaded = metadata.load_all_metadata()
        assert loaded == expected

    def _test_delete(self, metadata):
        assert "path/to/foo.txt" in metadata
        assert len(metadata) == 1
        del metadata["path/to/foo.txt"]
        assert "path/to/foo.txt" not in metadata
        assert len(metadata) == 0

    def _test_iter(self, metadata):
        assert list(metadata) == ["path/to/foo.txt"]

    def _test_len(self, metadata):
        assert len(metadata) == 1

    def _test_getitem(self, metadata):
        assert metadata["path/to/foo.txt"] == Metadata(md5="bar")


class TestJSONMetadataStore(MetadataTests):
    def test_store_metadata(self, json_metadata):
        meta = Metadata(md5="other")
        json_metadata.store_metadata("path/to/other.txt", meta)
        base_path = json_metadata.external_store.root_dir
        metadata_json = base_path / "metadata.json"
        assert metadata_json.exists()
        with open(metadata_json, mode="r") as f:
            metadata_dict = json.load(f)

        metadata = {key: Metadata(**val) for key, val in metadata_dict.items()}

        assert metadata == json_metadata._metadata_cache
        assert json_metadata["path/to/other.txt"] == meta
        assert len(metadata) == 2

    def test_load_all_metadata(self, json_metadata):
        self._test_load_all_metadata(json_metadata)

    def test_load_all_metadata_nofile(self, tmpdir):
        json_metadata = JSONMetadataStore(FileStorage(str(tmpdir)))
        # implicitly called on init anyway
        assert json_metadata._metadata_cache == {}
        # but we also call explicitly
        assert json_metadata.load_all_metadata() == {}

    def test_delete(self, json_metadata):
        self._test_delete(json_metadata)

    def test_iter(self, json_metadata):
        self._test_iter(json_metadata)

    def test_len(self, json_metadata):
        self._test_len(json_metadata)

    def test_getitem(self, json_metadata):
        self._test_getitem(json_metadata)


class TestPerFileJSONMetadataStore(MetadataTests):
    def test_store_metadata(self, per_file_metadata):
        expected_loc = "metadata/path/to/other.txt.json"
        root = per_file_metadata.external_store.root_dir
        expected_path = root / expected_loc
        assert not expected_path.exists()
        meta = Metadata(md5="other")
        per_file_metadata.store_metadata("path/to/other.txt", meta)
        assert expected_path.exists()
        expected = {"path": "path/to/other.txt", "metadata": {"md5": "other"}}
        with open(expected_path, mode="r") as f:
            assert json.load(f) == expected

    def test_load_all_metadata(self, per_file_metadata):
        self._test_load_all_metadata(per_file_metadata)

    def test_delete(self, per_file_metadata):
        self._test_delete(per_file_metadata)
        # TODO: add additional test that the file is gone

    def test_iter(self, per_file_metadata):
        self._test_iter(per_file_metadata)

    def test_len(self, per_file_metadata):
        self._test_len(per_file_metadata)

    def test_getitem(self, per_file_metadata):
        self._test_getitem(per_file_metadata)

    def test_bad_metadata_contents(self, tmp_path):
        loc = tmp_path / "metadata/foo.txt.json"
        loc.parent.mkdir(parents=True, exist_ok=True)
        bad_dict = {"foo": "bar"}
        with open(loc, mode="wb") as f:
            f.write(json.dumps(bad_dict).encode("utf-8"))

        with pytest.raises(ChangedExternalResourceError, match="Bad metadata"):
            PerFileJSONMetadataStore(FileStorage(tmp_path))
