import os
import pytest

from plugcli.params import NOT_PARSED
from openfecli.parameters.utils import import_parameter

@pytest.mark.parametrize('import_str,expected', [
    ('os.path.exists', os.path.exists),
    ('os.getcwd', os.getcwd),
    ('os.foo', NOT_PARSED),
    ('foo.bar', NOT_PARSED),
    ('foo', NOT_PARSED),
])
def test_import_parameter(import_str, expected):
    assert import_parameter(import_str) is expected
