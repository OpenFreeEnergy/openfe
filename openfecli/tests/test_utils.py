import os
import pytest

from openfecli.utils import import_thing


@pytest.mark.parametrize('import_string,expected', [
    ('os.path.exists', os.path.exists),
    ('os.getcwd', os.getcwd),
    ('os', os),
])
def test_import_thing(import_string, expected):
    assert import_thing(import_string) is expected


def test_import_thing_import_error():
    with pytest.raises(ImportError):
        import_thing('foo.bar')


def test_import_thing_attribute_error():
    with pytest.raises(AttributeError):
        import_thing('os.foo')
