import pytest
import click
from openfecli.clicktypes.hyphenchoice import HyphenAwareChoice


class TestHyphenAwareChoice:
    @pytest.mark.parametrize("value", ["foo_bar_baz", "foo_bar-baz", "foo-bar_baz", "foo-bar-baz"])
    def test_init(self, value):
        ch = HyphenAwareChoice([value])
        assert ch.choices == ["foo-bar-baz"]

    @pytest.mark.parametrize("value", ["foo_bar_baz", "foo_bar-baz", "foo-bar_baz", "foo-bar-baz"])
    def test_convert(self, value):
        ch = HyphenAwareChoice(["foo-bar-baz"])
        # counting on __call__ to get to convert()
        assert ch(value) == "foo-bar-baz"
