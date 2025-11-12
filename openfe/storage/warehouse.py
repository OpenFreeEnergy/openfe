# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/gufe
import abc
import json
import re
from typing import Any, NotRequired, TypedDict

from gufe.storage.externalresource import ExternalStorage
from gufe.tokenization import (
    JSON_HANDLER,
    GufeKey,
    GufeTokenizable,
    from_dict,
    get_all_gufe_objs,
    key_decode_dependencies,
)
from numpy import delete

GUFEKEY_JSON_REGEX = re.compile('":gufe-key:": "(?P<token>[A-Za-z0-9_]+-[0-9a-f]+)"')


class _DataContainer(abc.ABC):
    """
    Abstract class, represents all data under some level of the heirarchy.
    """

    def __init__(self, parent, path_component):
        self.parent = parent
        self._path_component = self._to_path_component(path_component)
        self._cache = {}

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.path == other.path

    @staticmethod
    def _to_path_component(item: Any) -> str:
        """Convert input (object or string) to path string"""
        if isinstance(item, str):
            return item

        # TODO: instead of str(hash(...)), this should return the digest
        # that is being introduced in another PR; Python hash is not stable
        # across sessions
        return str(hash(item))

    def __getitem__(self, item):
        # code for the case this is a file
        if item in self.external_storage:
            return self.external_storage.load_stream(item)

        # code for the case this is a "directory"
        hash_item = self._to_path_component(item)

        if hash_item not in self._cache:
            self._cache[hash_item] = self._load_next_level(item)

        return self._cache[hash_item]

    def __truediv__(self, item):
        return self[item]

    @abc.abstractmethod
    def _load_next_level(self, item):
        raise NotImplementedError()

    def __iter__(self):
        for loc in self.external_storage:
            if loc.startswith(self.path):
                yield loc

    def load_stream(self, location):
        return self.external_storage.load_stream(location)

    def load_bytes(self, location):
        with self.load_stream(location) as f:
            byte_data = f.read()

        return byte_data

    @property
    def path(self):
        return self.parent.path + "/" + self._path_component

    @property
    def external_storage(self) -> ExternalStorage:
        return self.parent.external_storage

    def __repr__(self):
        # probably should include repr of external store, too
        return f"{self.__class__.__name__}({self.path})"


class WarehouseStores(TypedDict):
    """This class serves to create a typesafe way of accessing the stores for
    the WarehouseBaseClass.
    """

    setup: ExternalStorage


class WarehouseBaseClass(_DataContainer):
    def __init__(self, stores: WarehouseStores):
        self.stores = stores
        super().__init__(parent=self, path_component=None)

    def delete(self, store_name: str, location: str):
        store: ExternalStorage = self.stores[store_name]
        return store.delete(location)

    # @staticmethod
    # def _gufe_key_to_storage_key(store: ExternalStorage, key: str):
    #     """Create the storage key from the gufe key.
    #
    #     Parameters
    #     ----------
    #     prefix : str
    #         the prefix defining which section of storage should be used for
    #         this (e.g., ``setup``, ...)
    #     key : str
    #         the GufeKey for a GufeTokenizable (technically, is likely to be
    #         passed as a :class:`.GufeKey`, which is a subclass of ``str``)
    #
    #     Returns
    #     -------
    #     str :
    #         storage key (string identifier used by storage to locate this
    #         object)
    #     """
    #     pref = prefix.split("/")  # remove this if we switch to tuples
    #     cls, token = key.split("-")
    #     tup = tuple(list(pref) + [cls, f"{token}.json"])
    #     # right now we're using strings, but we've talked about switching
    #     # that to tuples
    #     return "/".join(tup)

    def _store_gufe_tokenizable(self, store_name: str, obj: GufeTokenizable):
        """generic function for deduplicating/storing a GufeTokenizable"""

        # Try and get the key for the given store
        target = self.stores[store_name]
        # Get all of the sub-objects
        for o in get_all_gufe_objs(obj):
            key = o.key
            # Check if the key exists at all in any store
            if not self._key_exists(key):
                data = json.dumps(
                    o.to_keyed_dict(), cls=JSON_HANDLER.encoder, sort_keys=True
                ).encode("utf-8")
                target.store_bytes(key, data)

    def _key_exists(self, key: GufeKey) -> bool:
        return any(key in store for store in self.stores.values())

    def _load_gufe_tokenizable(self, store_name: str, gufe_key: GufeKey):
        """generic function to load deduplicated object from a key"""
        registry = {}

        def recursive_build_object_cache(gufe_key):
            """DFS to rebuild object heirarchy"""
            # This implementation is a bit fragile, because ensuring that we
            # don't duplicate objects in memory depends on the fact that
            # `key_decode_dependencies` gets keyencoded objects from a cache
            # (they are cached on creation).
            with self.load_stream(gufe_key) as f:
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
                # we're actually only doing this for the side effect of
                # generating the objects and adding them to the registry
                recursive_build_object_cache(key)

            if len(key_encoded) == 0:
                # fast path for objects that don't contain other gufe
                # objects (these tend to be larger dicts; avoid walking
                # them)
                obj = from_dict(dct)
            else:
                # objects that contain other gufe objects need be walked to
                # replace everything
                obj = key_decode_dependencies(dct, registry)

            registry[obj.key] = obj
            return obj

        return recursive_build_object_cache(gufe_key)

    # def _load_next_level(self, item):
    #     return TransformationResult(self, item)
    def _load_next_level(self, item):
        return super()._load_next_level(item)

    # override these two inherited properies since this is always the end of
    # the recursive chain
    @property
    def path(self):
        return "transformations"

    @property
    def external_storage(self):
        return self.setup_store
