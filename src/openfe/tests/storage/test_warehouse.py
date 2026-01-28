import os
import tempfile
from pathlib import Path
from typing import Literal
from unittest import mock

import pytest
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
    def _test_store_load_same_process(
        obj, store_func_name, load_func_name, store_name: Literal["setup", "result"]
    ):
        setup_store = MemoryStorage()
        result_store = MemoryStorage()
        stores = WarehouseStores(setup=setup_store, result=result_store)
        client = WarehouseBaseClass(stores)
        store_func = getattr(client, store_func_name)
        load_func = getattr(client, load_func_name)
        assert setup_store._data == {}
        assert result_store._data == {}
        store_func(obj)
        store_under_test: MemoryStorage = stores[store_name]
        assert store_under_test._data != {}
        reloaded: GufeTokenizable = load_func(obj.key)
        assert reloaded is obj
        return reloaded, client

    @staticmethod
    def _test_store_load_different_process(
        obj: GufeTokenizable,
        store_func_name,
        load_func_name,
        store_name: Literal["setup", "result"],
    ):
        setup_store = MemoryStorage()
        result_store = MemoryStorage()
        stores = WarehouseStores(setup=setup_store, result=result_store)
        client = WarehouseBaseClass(stores)
        store_func = getattr(client, store_func_name)
        load_func = getattr(client, load_func_name)
        assert setup_store._data == {}
        assert result_store._data == {}
        store_func(obj)
        store_under_test: MemoryStorage = stores[store_name]
        assert store_under_test._data != {}
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
    @pytest.mark.parametrize("store", ["setup", "result"])
    def test_store_load_transformation_same_process(self, request, fixture, store):
        transformation = request.getfixturevalue(fixture)
        store_func_name = f"store_{store}_tokenizable"
        load_func_name = f"load_{store}_tokenizable"
        self._test_store_load_same_process(transformation, store_func_name, load_func_name, store)

    @pytest.mark.parametrize(
        "fixture",
        ["absolute_transformation", "complex_equilibrium"],
    )
    @pytest.mark.parametrize("store", ["setup", "result"])
    def test_store_load_transformation_different_process(self, request, fixture, store):
        transformation = request.getfixturevalue(fixture)
        store_func_name = f"store_{store}_tokenizable"
        load_func_name = f"load_{store}_tokenizable"
        self._test_store_load_different_process(
            transformation, store_func_name, load_func_name, store
        )

    #
    @pytest.mark.parametrize("fixture", ["benzene_variants_star_map"])
    @pytest.mark.parametrize("store", ["setup", "result"])
    def test_store_load_network_same_process(self, request, fixture, store):
        network = request.getfixturevalue(fixture)
        assert isinstance(network, GufeTokenizable)
        store_func_name = f"store_{store}_tokenizable"
        load_func_name = f"load_{store}_tokenizable"
        self._test_store_load_same_process(network, store_func_name, load_func_name, store)

    @pytest.mark.parametrize("fixture", ["benzene_variants_star_map"])
    @pytest.mark.parametrize("store", ["setup", "result"])
    def test_store_load_network_different_process(self, request, fixture, store):
        network = request.getfixturevalue(fixture)
        assert isinstance(network, GufeTokenizable)
        store_func_name = f"store_{store}_tokenizable"
        load_func_name = f"load_{store}_tokenizable"
        self._test_store_load_different_process(network, store_func_name, load_func_name, store)

    @pytest.mark.parametrize("fixture", ["benzene_variants_star_map"])
    @pytest.mark.parametrize("store", ["setup", "result"])
    def test_delete(self, request, fixture, store):
        network = request.getfixturevalue(fixture)
        store_func_name = f"store_{store}_tokenizable"
        load_func_name = f"load_{store}_tokenizable"
        obj, client = self._test_store_load_same_process(
            network, store_func_name, load_func_name, store
        )
        client.delete(store, obj.key)
        assert not client.exists(obj.key)


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
