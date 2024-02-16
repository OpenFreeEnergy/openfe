# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/gufe
import warnings
from typing import ClassVar

from gufe.storage.errors import ChangedExternalResourceError, MissingExternalResourceError


class ResultServer:
    """Class to manage communication between metadata and data storage.

    At this level, we provide an abstraction where client code no longer
    needs to be aware of the nature of the metadata, or even that it exists.
    """

    def __init__(self, external_store, metadata_store):
        self.external_store = external_store
        self.metadata_store = metadata_store

    def _store_metadata(self, location):
        metadata = self.external_store.get_metadata(location)
        self.metadata_store.store_metadata(location, metadata)

    def store_bytes(self, location, byte_data):
        self.external_store.store_bytes(location, byte_data)
        self._store_metadata(location)

    def store_path(self, location, path):
        self.external_store.store_path(location, path)
        self._store_metadata(location)

    def delete(self, location):
        del self.metadata_store[location]
        self.external_store.delete(location)

    def validate(self, location, allow_changed=False):
        try:
            metadata = self.metadata_store[location]
        except KeyError:
            raise MissingExternalResourceError(f"Metadata for '{location}' " "not found")

        if not self.external_store.get_metadata(location) == metadata:
            msg = f"Metadata mismatch for {location}: this object " "may have changed."
            if not allow_changed:
                raise ChangedExternalResourceError(msg + " To allow this, set ExternalStorage." "allow_changed = True")
            else:
                warnings.warn(msg)

    def __iter__(self):
        return iter(self.metadata_store)

    def find_missing_files(self):
        """Identify files listed in metadata but unavailable in storage"""
        return [f for f in self if not self.external_store.exists(f)]

    def load_stream(self, location, allow_changed=False):
        self.validate(location, allow_changed)
        return self.external_store.load_stream(location)
