import os
import pathlib


def serialize(item, filename: pathlib.Path):
    """
    Serialize an OpenMM System, State, or Integrator.

    Parameters
    ----------
    item : System, State, or Integrator
        The thing to be serialized
    filename : str
        The filename to serialize to
    """
    from openmm import XmlSerializer

    # Create parent directory if it doesn't exist
    filename_basedir = filename.parent
    if not filename_basedir.exists():
        os.makedirs(filename_basedir)

    if filename.suffix == ".bz2":
        import bz2

        with bz2.open(filename, mode="wb") as outfile:
            serialized_thing = XmlSerializer.serialize(item)
            outfile.write(serialized_thing.encode())
    else:
        with open(filename, mode="w") as outfile:
            serialized_thing = XmlSerializer.serialize(item)
            outfile.write(serialized_thing)


def deserialize(filename: pathlib.Path):
    """
    Deserialize an OpenMM System, State, or Integrator.

    Parameters
    ----------
    item : System, State, or Integrator
        The thing to be serialized
    filename : str
        The filename to serialize to
    """
    from openmm import XmlSerializer

    # Create parent directory if it doesn't exist
    filename_basedir = filename.parent
    if not filename_basedir.exists():
        os.makedirs(filename_basedir)

    if filename.suffix == ".bz2":
        import bz2

        with bz2.open(filename, mode="rb") as infile:
            serialized_thing = infile.read().decode()
            item = XmlSerializer.deserialize(serialized_thing)
    else:
        with open(filename) as infile:
            serialized_thing = infile.read()
            item = XmlSerializer.deserialize(serialized_thing)

    return item


def make_vec3_box(dimensions: Quantity) -> openmm.Vec3:
    """
    Convert an OpenFF box dimensions Quantity back into Vec3 format.

    Parameters
    ----------
    dimensions : openff.units.Quantity
      United array to turn to Vec3 format.

    Returns
    -------
    openmm.Vec3
      The input array in Vec3 format.
    """
    from openmm import Vec3
    from openmm import unit as ommunit

    return [
        Vec3(
            float(row[0]), float(row[1]), float(row[2])
        ) * ommunit.nanometer
        for row in dimensions.m_as("nanometer")
    ]
