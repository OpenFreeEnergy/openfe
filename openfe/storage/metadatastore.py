# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/gufe
import json
import abc
import collections

from typing import Tuple, Dict

from gufe.storage.externalresource.base import Metadata

from gufe.storage.errors import MissingExternalResourceError, ChangedExternalResourceError


class MetadataStore(collections.abc.Mapping):
    def __init__(self, external_store):
        self.external_store = external_store
        self._metadata_cache = self.load_all_metadata()

    @abc.abstractmethod
    def store_metadata(self, location: str, metadata: Metadata):
        raise NotImplementedError()

    @abc.abstractmethod
    def load_all_metadata(self) -> Dict[str, Metadata]:
        raise NotImplementedError()

    @abc.abstractmethod
    def __delitem__(self, location):
        raise NotImplementedError()

    def __getitem__(self, location):
        return self._metadata_cache[location]

    def __iter__(self):
        return iter(self._metadata_cache)

    def __len__(self):
        return len(self._metadata_cache)


class JSONMetadataStore(MetadataStore):
    # Using JSON for now because it is easy to write this class and doesn't
    # require any external dependencies. It is NOT the right way to go in
    # the long term. API will probably stay the same, though.
    def _dump_file(self):
        metadata_dict = {key: val.to_dict() for key, val in self._metadata_cache.items()}
        metadata_bytes = json.dumps(metadata_dict).encode("utf-8")
        self.external_store.store_bytes("metadata.json", metadata_bytes)

    def store_metadata(self, location: str, metadata: Metadata):
        self._metadata_cache[location] = metadata
        self._dump_file()

    def load_all_metadata(self):
        if not self.external_store.exists("metadata.json"):
            return {}

        with self.external_store.load_stream("metadata.json") as json_f:
            all_metadata_dict = json.loads(json_f.read().decode("utf-8"))

        all_metadata = {key: Metadata(**val) for key, val in all_metadata_dict.items()}

        return all_metadata

    def __delitem__(self, location):
        del self._metadata_cache[location]
        self._dump_file()


class PerFileJSONMetadataStore(MetadataStore):
    _metadata_prefix = "metadata/"

    def _metadata_path(self, location):
        return self._metadata_prefix + location + ".json"

    def store_metadata(self, location: str, metadata: Metadata):
        self._metadata_cache[location] = metadata
        path = self._metadata_path(location)
        dct = {
            "path": location,
            "metadata": metadata.to_dict(),
        }
        metadata_bytes = json.dumps(dct).encode("utf-8")
        self.external_store.store_bytes(path, metadata_bytes)

    def load_all_metadata(self):
        metadata_cache = {}
        prefix = self._metadata_prefix
        for location in self.external_store.iter_contents(prefix=prefix):
            if location.endswith(".json"):
                with self.external_store.load_stream(location) as f:
                    dct = json.loads(f.read().decode("utf-8"))

                if set(dct) != {"path", "metadata"}:
                    raise ChangedExternalResourceError("Bad metadata file: " f"'{location}'")
                metadata_cache[dct["path"]] = Metadata(**dct["metadata"])

        return metadata_cache

    def __delitem__(self, location):
        del self._metadata_cache[location]
        self.external_store.delete(self._metadata_path(location))
