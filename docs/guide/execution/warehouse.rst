Storage with ``openfe.Warehouse``
=================================

openfe's Warehouse is a Python object that defines the necessary interface for an execution engine to store and access data during execution.


Importantly, a Warehouse is simply any derived class of the abstract ``WarehouseBaseClass``, meaning that *where* the data is stored is completely up to the author, but *how* the data is accessed is defined by the `WarehouseBaseClass`.

You can think of the WarehouseBaseClass as a set of specifications that must be met by a Warehouse implementation, such that any openfe Protocol can then interact appropriately with its data.

For example, openfe Protocols require several types of storage - **scratch**, **setup**, and **result**. 

Naively, we could require that all three of these storage types be directories that can be accessed locally by the Protocol, but this significantly limits the ways in which the Protocol can be executed, e.g. a Protocol could not store **result** data on a remote machine or cloud storage.

Where a Warehouse stores its data is defined by a ``WarehouseStores`` object, which is a small TypedDict that requires ``'setup'`` and ``'result'``.

.. note ::
    ``scratch`` is not a key in WarehouseStores, since `scratch` *must* be locally-accessible file storage.

Take the example implementation, :ref:`FileSystemWarehouse`, which inherits from ``WarehouseBaseClass``, but makes it easy to construct a Warehouse given a root directory, which is uses to create ``WarehouseStores``.

.. code-block ::

    class FileSystemWarehouse(WarehouseBaseClass):
        """Warehouse implementation using local filesystem storage.

        Provides a file-based storage backend for GufeTokenizable objects
        organized in a directory structure.

        Parameters
        ----------
        root_dir : str, optional
            Root directory for the warehouse storage, by default "warehouse/".

        Notes
        -----
        Creates a "setup/" subdirectory within the root directory for storing
        setup-related objects. Future versions may include additional stores
        for results and other data types.
        """

        def __init__(self, root_dir: str = "warehouse"):
            setup_store = FileStorage(f"{root_dir}/setup")
            result_store = FileStorage(f"{root_dir}/result")
            stores = WarehouseStores(setup=setup_store, result=result_store)
            super().__init__(stores)

For more information about what types of storage are available, see  <gufe storage docs>.