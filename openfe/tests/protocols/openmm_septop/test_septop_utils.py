# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import bz2
import gzip
import pathlib
from unittest import mock

from openfe.protocols.openmm_septop.utils import deserialize, serialize


def test_serialize_creates_parent_directory(tmp_path):
    filename = tmp_path / "file.xml"
    with mock.patch("openmm.XmlSerializer.serialize", return_value="<xml></xml>"):
        serialize(object(), filename)

    assert filename.exists()
    assert filename.read_text() == "<xml></xml>"


def test_serialize_xml(tmp_path):
    filename = tmp_path / "file.xml"
    with mock.patch("openmm.XmlSerializer.serialize", return_value="<data>"):
        serialize(object(), filename)

    with open(filename, "r") as f:
        assert f.read() == "<data>"


def test_serialize_gz(tmpdir):
    filename = pathlib.Path(tmpdir / "file.xml.gz")
    expected = "<gzip_data>"

    with mock.patch("openmm.XmlSerializer.serialize", return_value=expected):
        serialize(object(), filename)

    with gzip.open(filename, "rb") as f:
        read_back = f.read().decode()
    assert read_back == expected


def test_serialize_bz2(tmpdir):
    filename = pathlib.Path(tmpdir / "file.xml.bz2")
    expected = "<bz2_data>"

    with mock.patch("openmm.XmlSerializer.serialize", return_value=expected):
        serialize(object(), filename)

    with bz2.open(filename, "rb") as f:
        read_back = f.read().decode()
    assert read_back == expected


def test_deserialize_xml(tmpdir):
    filename = pathlib.Path(tmpdir / "file.xml")
    filename.write_text("<xml>things</xml>")

    with mock.patch("openmm.XmlSerializer.deserialize", return_value="DESERIALIZED") as deser:
        result = deserialize(filename)

    deser.assert_called_once_with("<xml>things</xml>")
    assert result == "DESERIALIZED"


def test_deserialize_gz(tmpdir):
    filename = pathlib.Path(tmpdir / "file.xml.gz")
    expected_serialized = "<xml>gz</xml>"
    with gzip.open(filename, "wb") as f:
        f.write(expected_serialized.encode())

    with mock.patch("openmm.XmlSerializer.deserialize", return_value="FROM_GZ") as deser:
        result = deserialize(filename)

    deser.assert_called_once_with(expected_serialized)
    assert result == "FROM_GZ"


def test_deserialize_bz2(tmpdir):
    filename = pathlib.Path(tmpdir / "file.xml.bz2")
    expected_serialized = "<xml>bz2</xml>"
    with bz2.open(filename, "wb") as f:
        f.write(expected_serialized.encode())

    with mock.patch("openmm.XmlSerializer.deserialize", return_value="FROM_BZ2") as deser:
        result = deserialize(filename)

    deser.assert_called_once_with(expected_serialized)
    assert result == "FROM_BZ2"
