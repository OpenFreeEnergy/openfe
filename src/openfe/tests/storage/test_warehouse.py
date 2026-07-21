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
    def _build_stores() -> WarehouseStores:
        return WarehouseStores(
            setup=MemoryStorage(),
            result=MemoryStorage(),
            shared=MemoryStorage(),
            tasks=MemoryStorage(),
        )

    @staticmethod
    def _get_protocol_unit(transformation):
        dag = transformation.create()
        return next(iter(dag.protocol_units))

    @staticmethod
    def _test_store_load_same_process(
        obj,
        store_func_name,
        load_func_name,
        store_name: Literal["setup", "result", "tasks"],
    ):
        stores = TestWarehouseBaseClass._build_stores()
        client = WarehouseBaseClass(stores)
        store_func = getattr(client, store_func_name)
        load_func = getattr(client, load_func_name)
        assert stores["setup"]._data == {}
        assert stores["result"]._data == {}
        assert stores["shared"]._data == {}
        assert stores["tasks"]._data == {}
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
        store_name: Literal["setup", "result", "tasks"],
    ):
        stores = TestWarehouseBaseClass._build_stores()
        client = WarehouseBaseClass(stores)
        store_func = getattr(client, store_func_name)
        load_func = getattr(client, load_func_name)
        assert stores["setup"]._data == {}
        assert stores["result"]._data == {}
        assert stores["shared"]._data == {}
        assert stores["tasks"]._data == {}
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

    def test_store_load_task_same_process(self, absolute_transformation):
        unit = self._get_protocol_unit(absolute_transformation)
        self._test_store_load_same_process(unit, "store_task", "load_task", "tasks")

    def test_store_load_task_different_process(self, absolute_transformation):
        unit = self._get_protocol_unit(absolute_transformation)
        self._test_store_load_different_process(unit, "store_task", "load_task", "tasks")

    def test_store_task_writes_to_tasks_store(self, absolute_transformation):
        unit = self._get_protocol_unit(absolute_transformation)
        stores = self._build_stores()
        client = WarehouseBaseClass(stores)
        client.store_task(unit)

        assert stores["tasks"]._data != {}
        assert stores["setup"]._data == {}
        assert stores["result"]._data == {}
        assert stores["shared"]._data == {}

    def test_exists_finds_task_key(self, absolute_transformation):
        unit = self._get_protocol_unit(absolute_transformation)
        stores = self._build_stores()
        client = WarehouseBaseClass(stores)

        client.store_task(unit)

        assert client.exists(unit.key)

    def test_load_task_returns_object(self, absolute_transformation):
        unit = self._get_protocol_unit(absolute_transformation)
        stores = self._build_stores()
        client = WarehouseBaseClass(stores)

        client.store_task(unit)
        loaded = client.load_task(unit.key)

        assert loaded is not None
        assert isinstance(loaded, GufeTokenizable)

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

    def test_filesystemwarehouse_has_shared_and_tasks_stores(self, absolute_transformation):
        unit = TestWarehouseBaseClass._get_protocol_unit(absolute_transformation)

        with tempfile.TemporaryDirectory() as tmpdir:
            client = FileSystemWarehouse(tmpdir)

            assert "shared" in client.stores
            assert "tasks" in client.stores

            client.stores["shared"].store_bytes("sentinel", b"shared-data")
            with client.stores["shared"].load_stream("sentinel") as f:
                assert f.read() == b"shared-data"

            client.store_task(unit)
            assert client.exists(unit.key)

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
