# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/gufe
import abc
import json
import re
from typing import Any

from gufe.tokenization import JSON_HANDLER, from_dict, get_all_gufe_objs, key_decode_dependencies

from .metadatastore import JSONMetadataStore
from .resultserver import ResultServer

GUFEKEY_JSON_REGEX = re.compile('":gufe-key:": "(?P<token>[A-Za-z0-9_]+-[0-9a-f]+)"')


class _ResultContainer(abc.ABC):
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
        if item in self.result_server:
            return self.result_server.load_stream(item)

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
        for loc in self.result_server:
            if loc.startswith(self.path):
                yield loc

    def load_stream(self, location, *, allow_changed=False):
        return self.result_server.load_stream(location, allow_changed)

    def load_bytes(self, location, *, allow_changed=False):
        with self.load_stream(location, allow_changed=allow_changed) as f:
            byte_data = f.read()

        return byte_data

    @property
    def path(self):
        return self.parent.path + "/" + self._path_component

    @property
    def result_server(self):
        return self.parent.result_server

    def __repr__(self):
        # probably should include repr of external store, too
        return f"{self.__class__.__name__}({self.path})"


class ResultClient(_ResultContainer):
    def __init__(self, external_store):
        # default client is using JSONMetadataStore with the given external
        # result store; users could easily write a subblass that behaves
        # differently
        metadata_store = JSONMetadataStore(external_store)
        self._result_server = ResultServer(external_store, metadata_store)
        super().__init__(parent=self, path_component=None)

    def delete(self, location):
        self._result_server.delete(location)

    @staticmethod
    def _gufe_key_to_storage_key(prefix: str, key: str):
        """Create the storage key from the gufe key.

        Parameters
        ----------
        prefix : str
            the prefix defining which section of storage should be used for
            this (e.g., ``setup``, ...)
        key : str
            the GufeKey for a GufeTokenizable (technically, is likely to be
            passed as a :class:`.GufeKey`, which is a subclass of ``str``)

        Returns
        -------
        str :
            storage key (string identifier used by storage to locate this
            object)
        """
        pref = prefix.split("/")  # remove this if we switch to tuples
        cls, token = key.split("-")
        tup = tuple(list(pref) + [cls, f"{token}.json"])
        # right now we're using strings, but we've talked about switching
        # that to tuples
        return "/".join(tup)

    def _store_gufe_tokenizable(self, prefix, obj):
        """generic function for deduplicating/storing a GufeTokenizable"""
        for o in get_all_gufe_objs(obj):
            key = self._gufe_key_to_storage_key(prefix, o.key)

            # we trust that if we get the same key, it's the same object, so
            # we only store on keys that we don't already know
            if key not in self.result_server:
                data = json.dumps(o.to_keyed_dict(), cls=JSON_HANDLER.encoder, sort_keys=True).encode("utf-8")
                self.result_server.store_bytes(key, data)

    def store_transformation(self, transformation):
        """Store a :class:`.Transformation`.

        Parmeters
        ---------
        transformation: :class:`.Transformation`
            the transformation to store
        """
        self._store_gufe_tokenizable("setup", transformation)

    def store_network(self, network):
        """Store a :class:`.AlchemicalNetwork`.

        Parmeters
        ---------
        network: :class:`.AlchemicalNetwork`
            the network to store
        """
        self._store_gufe_tokenizable("setup", network)

    def _load_gufe_tokenizable(self, prefix, gufe_key):
        """generic function to load deduplicated object from a key"""
        registry = {}

        def recursive_build_object_cache(gufe_key):
            """DFS to rebuild object heirarchy"""
            # This implementation is a bit fragile, because ensuring that we
            # don't duplicate objects in memory depends on the fact that
            # `key_decode_dependencies` gets keyencoded objects from a cache
            # (they are cached on creation).
            storage_key = self._gufe_key_to_storage_key(prefix, gufe_key)
            with self.load_stream(storage_key) as f:
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

    def load_transformation(self, key: str):
        """Load a :class:`.Transformation` from its GufeKey

        Parameters
        ----------
        key: str
            the gufe key for this object

        Returns
        -------
        :class:`.Transformation`
            the desired transformation
        """
        return self._load_gufe_tokenizable("setup", key)

    def load_network(self, key: str):
        """Load a :class:`.AlchemicalNetwork` from its GufeKey

        Parameters
        ----------
        key: str
            the gufe key for this object

        Returns
        -------
        :class:`.AlchemicalNetwork`
            the desired network
        """
        return self._load_gufe_tokenizable("setup", key)

    def _load_next_level(self, transformation):
        return TransformationResult(self, transformation)

    # override these two inherited properies since this is always the end of
    # the recursive chain
    @property
    def path(self):
        return "transformations"

    @property
    def result_server(self):
        return self._result_server


class TransformationResult(_ResultContainer):
    def __init__(self, parent, transformation):
        super().__init__(parent, transformation)
        self.transformation = transformation

    def _load_next_level(self, clone):
        return CloneResult(self, clone)


class CloneResult(_ResultContainer):
    def __init__(self, parent, clone):
        super().__init__(parent, clone)
        self.clone = clone

    @staticmethod
    def _to_path_component(item):
        return str(item)

    def _load_next_level(self, extension):
        return ExtensionResult(self, extension)


class ExtensionResult(_ResultContainer):
    def __init__(self, parent, extension):
        super().__init__(parent, str(extension))
        self.extension = extension

    @staticmethod
    def _to_path_component(item):
        return str(item)

    def __getitem__(self, filename):
        # different here -- we don't cache the actual file objects
        return self._load_next_level(filename)

    def _load_next_level(self, filename):
        return self.result_server.load_stream(self.path + "/" + filename)
