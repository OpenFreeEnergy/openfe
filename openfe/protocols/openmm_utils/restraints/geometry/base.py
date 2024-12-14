# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Restraint Geometry classes

TODO
----
* Add relevant duecredit entries.
"""
import abc
from pydantic.v1 import BaseModel, validator


class BaseRestraintGeometry(BaseModel, abc.ABC):
    class Config:
        arbitrary_types_allowed = True


class HostGuestRestraintGeometry(BaseRestraintGeometry):
    """
    An ordered list of guest atoms to restrain.

    Note
    ----
    The order matters! It will be used to define the underlying
    force.
    """

    guest_atoms: list[int]
    """
    An ordered list of host atoms to restrain.

    Note
    ----
    The order matters! It will be used to define the underlying
    force.
    """
    host_atoms: list[int]

    @validator("guest_atoms", "host_atoms")
    def positive_idxs(cls, v):
        if any([i < 0 for i in v]):
            errmsg = "negative indices passed"
            raise ValueError(errmsg)
        return v
