
from typing import Annotated, TypeAlias
from gufe.settings.types import (
    AngstromQuantity,
    NanometerQuantity,
    NanometerArrayQuantity,
    PicosecondQuantity,
    KCalPerMolQuantity,
    KelvinQuantity,
    BoxQuantity,
    NanosecondQuantity,
    GufeQuantity,
    specify_quantity_units,
)


FemtosecondQuantity: TypeAlias = Annotated[GufeQuantity, specify_quantity_units("femtosecond")]
InversePicosecondQuantity: TypeAlias =  Annotated[GufeQuantity, specify_quantity_units("1/picosecond")]
RadiansQuantity:TypeAlias = Annotated[GufeQuantity, specify_quantity_units("radians")]
TimestepQuantity: TypeAlias =  Annotated[GufeQuantity, specify_quantity_units("timestep")]

