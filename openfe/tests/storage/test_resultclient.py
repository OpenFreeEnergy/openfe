import os
from unittest import mock

import pytest
from gufe.storage.externalresource import MemoryStorage
from gufe.tokenization import TOKENIZABLE_REGISTRY

from openfe.storage.resultclient import CloneResult, ExtensionResult, ResultClient, TransformationResult


@pytest.fixture
def result_client(tmpdir):
    external = MemoryStorage()
    result_client = ResultClient(external)

    # store one file with contents "foo"
    result_client.result_server.store_bytes("transformations/MAIN_TRANS/0/0/file.txt", "foo".encode("utf-8"))

    # create some empty files as well
    empty_files = [
        "transformations/MAIN_TRANS/0/0/other.txt",
        "transformations/MAIN_TRANS/0/1/file.txt",
        "transformations/MAIN_TRANS/1/0/file.txt",
        "transformations/OTHER_TRANS/0/0/file.txt",
        "other_dir/file.txt",
    ]

    for file in empty_files:
        result_client.result_server.store_bytes(file, b"")  # empty

    return result_client


def _make_mock_transformation(hash_str):
    return mock.Mock(
        # TODO: fill this in so that it mocks out the digest we use
    )


def test_load_file(result_client):
    file_handler = result_client / "MAIN_TRANS" / "0" / 0 / "file.txt"
    with file_handler as f:
        assert f.read().decode("utf-8") == "foo"


class _ResultContainerTest:
    @staticmethod
    def get_container(result_client):
        raise NotImplementedError()

    def _getitem_object(self, container):
        raise NotImplementedError()

    def test_iter(self, result_client):
        container = self.get_container(result_client)
        assert set(container) == set(self.expected_files)

    def _get_key(self, as_object, container):
        # TODO: this isn't working yet -- need an interface that allows me
        # to patch the hex digest that we'll be using
        if as_object:
            pytest.skip("Waiting on hex digest patching")
        obj = self._getitem_object(container)
        # next line uses some internal implementation
        key = obj if as_object else obj._path_component
        return key, obj

    @pytest.mark.parametrize("as_object", [True, False])
    def test_getitem(self, as_object, result_client):
        container = self.get_container(result_client)
        key, obj = self._get_key(as_object, container)
        assert container[key] == obj

    @pytest.mark.parametrize("as_object", [True, False])
    def test_div(self, as_object, result_client):
        container = self.get_container(result_client)
        key, obj = self._get_key(as_object, container)
        assert container / key == obj

    @pytest.mark.parametrize("load_with", ["div", "getitem"])
    def test_caching(self, result_client, load_with):
        # used to test caching regardless of how first loaded was loaded
        container = self.get_container(result_client)
        key, obj = self._get_key(False, container)

        if load_with == "div":
            loaded = container / key
        elif load_with == "getitem":
            loaded = container[key]
        else:  # -no-cov-
            raise RuntimeError(f"Bad input: can't load with '{load_with}'")

        assert loaded == obj
        assert loaded is not obj
        reloaded_div = container / key
        reloaded_getitem = container[key]

        assert loaded is reloaded_div
        assert reloaded_div is reloaded_getitem

    def test_load_stream(self, result_client):
        container = self.get_container(result_client)
        loc = "transformations/MAIN_TRANS/0/0/file.txt"
        with container.load_stream(loc) as f:
            assert f.read().decode("utf-8") == "foo"

    def test_load_bytes(self, result_client):
        container = self.get_container(result_client)
        loc = "transformations/MAIN_TRANS/0/0/file.txt"
        assert container.load_bytes(loc).decode("utf-8") == "foo"

    def test_path(self, result_client):
        container = self.get_container(result_client)
        assert container.path == self.expected_path

    def test_result_server(self, result_client):
        container = self.get_container(result_client)
        assert container.result_server == result_client.result_server


class TestResultClient(_ResultContainerTest):
    expected_files = [
        "transformations/MAIN_TRANS/0/0/file.txt",
        "transformations/MAIN_TRANS/0/0/other.txt",
        "transformations/MAIN_TRANS/0/1/file.txt",
        "transformations/MAIN_TRANS/1/0/file.txt",
        "transformations/OTHER_TRANS/0/0/file.txt",
    ]
    expected_path = "transformations"

    @staticmethod
    def get_container(result_client):
        return result_client

    def _getitem_object(self, container):
        return TransformationResult(parent=container, transformation=_make_mock_transformation("MAIN_TRANS"))

    def test_store_protocol_dag_result(self):
        pytest.skip("Not implemented yet")

    @staticmethod
    def _test_store_load_same_process(obj, store_func_name, load_func_name):
        store = MemoryStorage()
        client = ResultClient(store)
        store_func = getattr(client, store_func_name)
        load_func = getattr(client, load_func_name)
        assert store._data == {}
        store_func(obj)
        assert store._data != {}
        reloaded = load_func(obj.key)
        assert reloaded is obj

    @staticmethod
    def _test_store_load_different_process(obj, store_func_name, load_func_name):
        store = MemoryStorage()
        client = ResultClient(store)
        store_func = getattr(client, store_func_name)
        load_func = getattr(client, load_func_name)
        assert store._data == {}
        store_func(obj)
        assert store._data != {}
        # make it look like we have an empty cache, as if this was a
        # different process
        key = obj.key
        registry_dict = "gufe.tokenization.TOKENIZABLE_REGISTRY"
        with mock.patch.dict(registry_dict, {}, clear=True):
            reload = load_func(key)
            assert reload == obj
            assert reload is not obj

    @pytest.mark.parametrize(
        "fixture",
        [
            "absolute_transformation",
            "complex_equilibrium",
        ],
    )
    def test_store_load_transformation_same_process(self, request, fixture):
        transformation = request.getfixturevalue(fixture)
        self._test_store_load_same_process(transformation, "store_transformation", "load_transformation")

    @pytest.mark.parametrize(
        "fixture",
        [
            "absolute_transformation",
            "complex_equilibrium",
        ],
    )
    def test_store_load_transformation_different_process(self, request, fixture):
        transformation = request.getfixturevalue(fixture)
        self._test_store_load_different_process(transformation, "store_transformation", "load_transformation")

    @pytest.mark.parametrize("fixture", ["benzene_variants_star_map"])
    def test_store_load_network_same_process(self, request, fixture):
        network = request.getfixturevalue(fixture)
        self._test_store_load_same_process(network, "store_network", "load_network")

    @pytest.mark.parametrize("fixture", ["benzene_variants_star_map"])
    def test_store_load_network_different_process(self, request, fixture):
        network = request.getfixturevalue(fixture)
        self._test_store_load_different_process(network, "store_network", "load_network")

    def test_delete(self, result_client):
        file_to_delete = self.expected_files[0]
        storage = result_client.result_server.external_store
        assert storage.exists(file_to_delete)
        result_client.delete(file_to_delete)
        assert not storage.exists(file_to_delete)


class TestTransformationResults(_ResultContainerTest):
    expected_files = [
        "transformations/MAIN_TRANS/0/0/file.txt",
        "transformations/MAIN_TRANS/0/0/other.txt",
        "transformations/MAIN_TRANS/0/1/file.txt",
        "transformations/MAIN_TRANS/1/0/file.txt",
    ]
    expected_path = "transformations/MAIN_TRANS"

    @staticmethod
    def get_container(result_client):
        container = TransformationResult(
            parent=TestResultClient.get_container(result_client),
            transformation=_make_mock_transformation("MAIN_TRANS"),
        )
        container._path_component = "MAIN_TRANS"
        return container

    def _getitem_object(self, container):
        return CloneResult(parent=container, clone=0)


class TestCloneResults(_ResultContainerTest):
    expected_files = [
        "transformations/MAIN_TRANS/0/0/file.txt",
        "transformations/MAIN_TRANS/0/0/other.txt",
        "transformations/MAIN_TRANS/0/1/file.txt",
    ]
    expected_path = "transformations/MAIN_TRANS/0"

    @staticmethod
    def get_container(result_client):
        return CloneResult(parent=TestTransformationResults.get_container(result_client), clone=0)

    def _getitem_object(self, container):
        return ExtensionResult(parent=container, extension=0)


class TestExtensionResults(_ResultContainerTest):
    expected_files = [
        "transformations/MAIN_TRANS/0/0/file.txt",
        "transformations/MAIN_TRANS/0/0/other.txt",
    ]
    expected_path = "transformations/MAIN_TRANS/0/0"

    @staticmethod
    def get_container(result_client):
        return ExtensionResult(parent=TestCloneResults.get_container(result_client), extension=0)

    def _get_key(self, as_object, container):
        if self.as_object:  # -no-cov-
            raise RuntimeError("TestExtensionResults does not support " "as_object=True")
        path = "transformations/MAIN_TRANS/0/0/"
        fname = "file.txt"
        return fname, container.result_server.load_stream(path + fname)

    # things involving div and getitem need custom treatment
    def test_div(self, result_client):
        container = self.get_container(result_client)
        with container / "file.txt" as f:
            assert f.read().decode("utf-8") == "foo"

    def test_getitem(self, result_client):
        container = self.get_container(result_client)
        with container["file.txt"] as f:
            assert f.read().decode("utf-8") == "foo"

    def test_caching(self, result_client):
        # this one does not cache results; the cache should remain empty
        container = self.get_container(result_client)
        assert container._cache == {}
        from_div = container / "file.txt"
        assert container._cache == {}
        from_getitem = container["file.txt"]
        assert container._cache == {}
