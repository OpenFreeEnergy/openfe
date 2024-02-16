import pytest
from unittest import mock

import pathlib

from openfe.storage.resultserver import ResultServer
from gufe.storage.externalresource.base import Metadata

from gufe.storage.externalresource import FileStorage
from openfe.storage.metadatastore import JSONMetadataStore
from gufe.storage.errors import MissingExternalResourceError, ChangedExternalResourceError


@pytest.fixture
def result_server(tmpdir):
    external = FileStorage(tmpdir)
    metadata = JSONMetadataStore(external)
    result_server = ResultServer(external, metadata)
    result_server.store_bytes("path/to/foo.txt", "foo".encode("utf-8"))
    return result_server


class TestResultServer:
    def test_store_bytes(self, result_server):
        # first check the thing stored during the fixture
        metadata_store = result_server.metadata_store
        foo_loc = "path/to/foo.txt"
        assert len(metadata_store) == 1
        assert foo_loc in metadata_store
        assert result_server.external_store.exists(foo_loc)

        # also explicitly test storing here
        mock_hash = mock.Mock(return_value=mock.Mock(hexdigest=mock.Mock(return_value="deadbeef")))
        bar_loc = "path/to/bar.txt"
        with mock.patch("hashlib.md5", mock_hash):
            result_server.store_bytes(bar_loc, "bar".encode("utf-8"))

        assert len(metadata_store) == 2
        assert bar_loc in metadata_store
        assert result_server.external_store.exists(bar_loc)
        assert metadata_store[bar_loc].to_dict() == {"md5": "deadbeef"}
        external = result_server.external_store
        with external.load_stream(bar_loc) as f:
            assert f.read().decode("utf-8") == "bar"

    def test_store_path(self, result_server, tmp_path):
        orig_file = tmp_path / ".hidden" / "bar.txt"
        orig_file.parent.mkdir(parents=True, exist_ok=True)
        with open(orig_file, mode="wb") as f:
            f.write("bar".encode("utf-8"))

        mock_hash = mock.Mock(return_value=mock.Mock(hexdigest=mock.Mock(return_value="deadc0de")))
        bar_loc = "path/to/bar.txt"

        assert len(result_server.metadata_store) == 1
        assert bar_loc not in result_server.metadata_store

        with mock.patch("hashlib.md5", mock_hash):
            result_server.store_path(bar_loc, orig_file)

        assert len(result_server.metadata_store) == 2
        assert bar_loc in result_server.metadata_store
        metadata_dict = result_server.metadata_store[bar_loc].to_dict()
        assert metadata_dict == {"md5": "deadc0de"}
        external = result_server.external_store
        with external.load_stream(bar_loc) as f:
            assert f.read().decode("utf-8") == "bar"

    def test_iter(self, result_server):
        assert list(result_server) == ["path/to/foo.txt"]

    def test_find_missing_files(self, result_server):
        meta = Metadata(md5="1badc0de")
        result_server.metadata_store.store_metadata("fake/file.txt", meta)

        assert result_server.find_missing_files() == ["fake/file.txt"]

    def test_load_stream(self, result_server):
        with result_server.load_stream("path/to/foo.txt") as f:
            contents = f.read()

        assert contents.decode("utf-8") == "foo"

    def test_delete(self, result_server, tmpdir):
        location = "path/to/foo.txt"
        path = tmpdir / pathlib.Path(location)
        assert path.exists()
        assert location in result_server.metadata_store
        result_server.delete(location)
        assert not path.exists()
        assert location not in result_server.metadata_store

    def test_load_stream_missing(self, result_server):
        with pytest.raises(MissingExternalResourceError, match="not found"):
            result_server.load_stream("path/does/not/exist.txt")

    def test_load_stream_error_bad_hash(self, result_server):
        meta = Metadata(md5="1badc0de")
        result_server.metadata_store.store_metadata("path/to/foo.txt", meta)
        with pytest.raises(ChangedExternalResourceError):
            result_server.load_stream("path/to/foo.txt")

    def test_load_stream_allow_bad_hash(self, result_server):
        meta = Metadata(md5="1badc0de")
        result_server.metadata_store.store_metadata("path/to/foo.txt", meta)
        with pytest.warns(UserWarning, match="Metadata mismatch"):
            file = result_server.load_stream("path/to/foo.txt", allow_changed=True)

        with file as f:
            assert f.read().decode("utf-8") == "foo"
