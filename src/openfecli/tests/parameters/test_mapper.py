import click
import pytest

from openfe.setup import LomapAtomMapper
from openfecli.parameters.mapper import get_atommapper


@pytest.mark.parametrize(
    "user_input,expected",
    [
        ("LomapAtomMapper", LomapAtomMapper),
        ("openfe.setup.LomapAtomMapper", LomapAtomMapper),
    ],
)
def test_get_atommapper(user_input, expected):
    assert get_atommapper(user_input) is expected


def test_get_atommapper_error():
    with pytest.raises(click.BadParameter):
        get_atommapper("foo")
