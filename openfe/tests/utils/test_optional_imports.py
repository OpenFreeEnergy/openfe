# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from openfe.utils import requires_package
import pytest


@requires_package('no_such_package_hopefully')
def the_answer():
    return 42


def test_requires_decorator():
    with pytest.raises(ImportError):
        the_answer()
