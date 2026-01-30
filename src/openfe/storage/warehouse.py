# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/gufe
import abc
import json
import re
from typing import Literal, TypedDict

from gufe.protocols.protocoldag import ProtocolDAG
from gufe.protocols.protocolunit import ProtocolUnit
from gufe.storage.externalresource import ExternalStorage, FileStorage
from gufe.tokenization import (
    JSON_HANDLER,
    GufeKey,
    GufeTokenizable,
    from_dict,
    get_all_gufe_objs,
    key_decode_dependencies,
)

GUFEKEY_JSON_REGEX = re.compile('":gufe-key:": "(?P<token>[A-Za-z0-9_]+-[0-9a-f]+)"')


class WarehouseStores(TypedDict):
    """Typed dictionary for accessing warehouse storage locations.

    Parameters
    ----------
    setup : ExternalStorage
        Storage location for setup-related objects and configurations.
    result : ExternalStorage
        Storage location for result-related object.

    Notes
    -----
    Additional stores for results and tasks may be added in future versions.
    """

    setup: ExternalStorage
    result: ExternalStorage
    shared: ExternalStorage
    tasks: ExternalStorage


class WarehouseBaseClass:
    """Base class for warehouse storage management.

    Provides functionality to store, load, and manage GufeTokenizable objects
    across different storage backends.

    Parameters
    ----------
    stores : WarehouseStores
        Typed dictionary containing the storage locations for different
        types of objects.

    Attributes
    ----------
    stores : WarehouseStores
        The storage locations managed by this warehouse instance.
    """

    def __init__(self, stores: WarehouseStores):
        self.stores = stores

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.stores == other.stores

    def __repr__(self):
        # probably should include repr of external store, too
        return f"{self.__class__.__name__}({self.stores})"

    def delete(self, store_name: Literal["setup", "result"], location: str):
        """Delete an object from a specific store.

        Parameters
        ----------
        store_name : Literal["setup"]
            Name of the store to delete from.
        location : str
            Location/path of the object to delete.

        Raises
        -------
        MissingExternalResourceError
            Thrown if the object you are trying to delete, can't delete from the store
        """
        store: ExternalStorage = self.stores[store_name]
        store.delete(location)

    def store_task(self, obj: ProtocolUnit):
        self._store_gufe_tokenizable("tasks", obj)

    def load_task(self, obj: GufeKey) -> ProtocolUnit:
        unit = self._load_gufe_tokenizable(obj)
        if not isinstance(unit, ProtocolUnit):
            raise ValueError("Unable to load ProtocolUnit")
        return unit

    def store_setup_tokenizable(self, obj: GufeTokenizable):
        """Store a GufeTokenizable object in the setup store.

        Parameters
        ----------
        obj : GufeTokenizable
            The object to store.
        """
        self._store_gufe_tokenizable("setup", obj)

    def load_setup_tokenizable(self, obj: GufeKey) -> GufeTokenizable:
        """Load a GufeTokenizable object from the setup store.

        Parameters
        ----------
        obj : GufeKey
            The key of the object to load.

        Returns
        -------
        GufeTokenizable
            The loaded object.
        """
        return self._load_gufe_tokenizable(gufe_key=obj)

    def store_result_tokenizable(self, obj: GufeTokenizable):
        """Store a GufeTokenizable object from the result store.

        Parameters
        ----------
        obj : GufeKey
            The key of the object to store.
        """
        return self._store_gufe_tokenizable("result", obj)

    def load_result_tokenizable(self, obj: GufeKey) -> GufeTokenizable:
        """Load a GufeTokenizable object from the result store.

        Parameters
        ----------
        obj : GufeKey
            The key of the object to load.

        Returns
        -------
        GufeTokenizable
            The loaded object.
        """
        return self._load_gufe_tokenizable(gufe_key=obj)

    def exists(self, key: GufeKey) -> bool:
        """Check if an object with the given key exists in any store that holds tokenizables.

        Parameters
        ----------
        key : GufeKey
            The key to check for existence.

        Returns
        -------
        bool
            True if the object exists, False otherwise.
        """
        return any(key in store for store in self.stores.values())

    def _get_store_for_key(self, key: GufeKey) -> ExternalStorage:
        """Function to find the store in which a gufe key is stored in.

        Parameters
        ----------
        key : GufeKey
            The key to locate.

        Returns
        -------
        ExternalStorage
            The store containing the key.

        Raises
        ------
        ValueError
            If the key is not found in any store.
        """
        print(key)
        for name in self.stores:
            if key in self.stores[name]:
                return self.stores[name]
        raise ValueError(f"GufeKey {key} is not stored")

    def _store_gufe_tokenizable(
        self,
        store_name: Literal["setup", "result", "tasks"],
        obj: GufeTokenizable,
        name: str | None = None,
    ):
        """Store a GufeTokenizable object with deduplication.

            Parameters
            ----------
            store_name : Literal["setup"]
                Name of the store to store the object in.
            obj : GufeTokenizable
                The object to store.

        Notes
        -----
        This function performs deduplication by checking if the object
        already exists in any store before storing.
        """
        # Try and get the key for the given store
        target: ExternalStorage = self.stores[store_name]
        # Get all of the sub-objects
        chain = obj.to_keyed_chain()
        for item in chain:
            gufe_key = GufeKey(item[0])
            keyed_dict = item[1]
            if not self.exists(gufe_key):
                data = json.dumps(keyed_dict, cls=JSON_HANDLER.encoder, sort_keys=True).encode(
                    "utf-8"
                )
                if name:
                    target.store_bytes(name, data)
                else:
                    target.store_bytes(gufe_key, data)

    def _load_gufe_tokenizable(self, gufe_key: GufeKey) -> GufeTokenizable:
        """Load a deduplicated object from a GufeKey.

        Parameters
        ----------
        gufe_key : GufeKey
            The key of the object to load.

        Returns
        -------
        GufeTokenizable
            The loaded object with all dependencies resolved.

        Notes
        -----
        Uses depth-first search to rebuild object hierarchy and ensure
        proper deduplication in memory.
        """
        registry = {}

        def recursive_build_object_cache(key: GufeKey) -> GufeTokenizable:
            """DFS to rebuild object hierarchy.

            Parameters
            ----------
            key : GufeKey
                The key of the object to build.

            Returns
            -------
            GufeTokenizable
                The reconstructed object.
            """
            # This implementation is a bit fragile, because ensuring that we
            # don't duplicate objects in memory depends on the fact that
            # `key_decode_dependencies` gets keyencoded objects from a cache
            # (they are cached on creation).
            store = self._get_store_for_key(key=key)

            with store.load_stream(key) as f:
                keyencoded_json = f.read().decode("utf-8")

            dct = json.loads(keyencoded_json, cls=JSON_HANDLER.decoder)
            # this implementation may seem strange, but it will be a
            # faster than traversing the dict
            key_encoded = set(GUFEKEY_JSON_REGEX.findall(keyencoded_json))

            # this approach takes the dct instead of the json str
            # found = []
            # modify_dependencies(dct, found.append, is_gufe_key_dict)
            # key_encoded = {d[":gufe-key:"] for d in found}

            for key in key_encoded:
                # obj = GufeTokenizable.from_dict(dct)
                recursive_build_object_cache(key)
                # obj = GufeTokenizable.from_json(content=keyencoded_json)

            if len(key_encoded) == 0:
                # fast path for objects that don't contain other gufe
                # objects (these tend to be larger dicts; avoid walking
                # them)
                obj = GufeTokenizable.from_dict(dct)
                # objects that contain other gufe objects need be walked to
                # replace everything
            else:
                obj = key_decode_dependencies(dct, registry)
            #
            registry[obj.key] = obj
            return obj

        return recursive_build_object_cache(gufe_key)

    @property
    def setup_store(self):
        """Get the setup store

        Returns
        -------
        ExternalStorage
            The setup storage location
        """
        return self.stores["setup"]

    @property
    def result_store(self):
        """Get the result store.

        Returns
        -------
        ExternalStorage
            The result storage location
        """
        return self.stores["result"]

    @property
    def shared_store(self):
        """Get the shared store.

        Returns
        -------
        ExternalStorage
            The shared storage location
        """
        return self.stores["shared"]


class FileSystemWarehouse(WarehouseBaseClass):
    """Warehouse implementation using local filesystem storage.

    Provides a file-based storage backend for GufeTokenizable objects
    organized in a directory structure.

    Parameters
    ----------
    root_dir : str, optional
        Root directory for the warehouse storage, by default "warehouse".

    Notes
    -----
    Creates a "setup" subdirectory within the root directory for storing
    setup-related objects. Future versions may include additional stores
    for results and other data types.
    """

    def __init__(self, root_dir: str = "warehouse"):
        self.root_dir = root_dir
        setup_store = FileStorage(f"{root_dir}/setup")
        result_store = FileStorage(f"{root_dir}/result")
        shared_store = FileStorage(f"{root_dir}/shared")
        tasks_store = FileStorage(f"{root_dir}/tasks")
        stores = WarehouseStores(
            setup=setup_store, result=result_store, shared=shared_store, tasks=tasks_store
        )
        super().__init__(stores)
