import os
import tempfile
from pathlib import Path

import pytest
from unittest import mock

from gufe.storage.externalresource import MemoryStorage
from gufe.tokenization import GufeTokenizable
from openfe.storage.warehouse import (
    FileSystemWarehouse,
    WarehouseBaseClass,
    WarehouseStores,
)


class TestWarehouseBaseClass:
    def test_store_protocol_dag_result(self):
        pytest.skip("Not implemented yet")

    @staticmethod
    def _test_store_load_same_process(obj, store_func_name, load_func_name):
        store = MemoryStorage()
        stores = WarehouseStores(setup=store)
        client = WarehouseBaseClass(stores)
        store_func = getattr(client, store_func_name)
        load_func = getattr(client, load_func_name)
        assert store._data == {}
        store_func(obj)
        assert store._data != {}
        reloaded = load_func(obj.key)
        assert reloaded is obj

    @staticmethod
    def _test_store_load_different_process(obj: GufeTokenizable, store_func_name, load_func_name):
        store = MemoryStorage()
        stores = WarehouseStores(setup=store)
        client = WarehouseBaseClass(stores)
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
        ["absolute_transformation", "complex_equilibrium"],
    )
    def test_store_load_transformation_same_process(self, request, fixture):
        transformation = request.getfixturevalue(fixture)
        self._test_store_load_same_process(
            transformation,
            "store_setup_tokenizable",
            "load_setup_tokenizable",
        )

    @pytest.mark.parametrize(
        "fixture",
        ["absolute_transformation", "complex_equilibrium"],
    )
    def test_store_load_transformation_different_process(self, request, fixture):
        transformation = request.getfixturevalue(fixture)
        self._test_store_load_different_process(
            transformation,
            "store_setup_tokenizable",
            "load_setup_tokenizable",
        )

    #
    @pytest.mark.parametrize("fixture", ["benzene_variants_star_map"])
    def test_store_load_network_same_process(self, request, fixture):
        network = request.getfixturevalue(fixture)
        assert isinstance(network, GufeTokenizable)
        self._test_store_load_same_process(
            network, "store_setup_tokenizable", "load_setup_tokenizable"
        )

    #
    @pytest.mark.parametrize("fixture", ["benzene_variants_star_map"])
    def test_store_load_network_different_process(self, request, fixture):
        network = request.getfixturevalue(fixture)
        self._test_store_load_different_process(
            network, "store_setup_tokenizable", "load_setup_tokenizable"
        )

    #
    @pytest.mark.parametrize("fixture", ["benzene_variants_star_map"])
    def test_delete(self, request, fixture):
        store = MemoryStorage()
        stores = WarehouseStores(setup=store)
        client = WarehouseBaseClass(stores)

        network = request.getfixturevalue(fixture)
        assert store._data == {}
        client.store_setup_tokenizable(network)
        assert store._data != {}
        key = network.key
        loaded = client.load_setup_tokenizable(key)
        assert loaded is network
        assert client.setup_store.exists(key)
        client.delete("setup", key)
        assert not client.exists(key)


class TestFileSystemWarehouse:
    @staticmethod
    def _test_store_load_same_process(obj, store_func_name, load_func_name):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = FileSystemWarehouse(tmpdir)
            store_func = getattr(client, store_func_name)
            load_func = getattr(client, load_func_name)
            assert not any(Path(f"{tmpdir}").iterdir())
            store_func(obj)
            assert any(Path(f"{tmpdir}").iterdir())
            reloaded = load_func(obj.key)
            assert reloaded is obj

    @staticmethod
    def _test_store_load_different_process(obj: GufeTokenizable, store_func_name, load_func_name):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = FileSystemWarehouse(tmpdir)
            store_func = getattr(client, store_func_name)
            load_func = getattr(client, load_func_name)
            assert not any(Path(f"{tmpdir}").iterdir())
            store_func(obj)
            assert any(Path(f"{tmpdir}").iterdir())
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
        ["absolute_transformation", "complex_equilibrium"],
    )
    def test_store_load_transformation_same_process(self, request, fixture):
        transformation = request.getfixturevalue(fixture)
        self._test_store_load_same_process(
            transformation,
            "store_setup_tokenizable",
            "load_setup_tokenizable",
        )

    @pytest.mark.parametrize(
        "fixture",
        ["absolute_transformation", "complex_equilibrium"],
    )
    def test_store_load_transformation_different_process(self, request, fixture):
        transformation = request.getfixturevalue(fixture)
        self._test_store_load_different_process(
            transformation,
            "store_setup_tokenizable",
            "load_setup_tokenizable",
        )
