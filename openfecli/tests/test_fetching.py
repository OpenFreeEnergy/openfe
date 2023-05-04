import pytest

from .conftest import HAS_INTERNET

from openfecli.fetching import URLFetcher, PkgResourceFetcher
from openfecli.fetching import FetchablePlugin

class FetcherTester:
    @pytest.fixture
    def fetcher(self):
        raise NotImplementedError()

    def test_resources(self):
        raise NotImplementedError()

    def test_plugin(self, fetcher):
        # this is just a smoke test; individual plugins should test that
        # they work
        plugin = fetcher.plugin
        assert isinstance(plugin, FetchablePlugin)

    def test_call(self, fetcher, tmp_path):
        # Here we just check that the machinery works. Each plugin should
        # have a test to ensure that we're getting the right kind of file.
        paths = [tmp_path / filename for _, filename in fetcher.resources]
        for path in paths:
            assert not path.exists()

        fetcher(tmp_path)

        for path in paths:
            assert path.exists()


class TestURLFetcher(FetcherTester):
    @pytest.fixture
    def fetcher(self):
        return URLFetcher(
            resources=[("https://www.google.com/", "index.html")],
            short_name="google",
            short_help="The Goog",
            requires_ofe=(0, 7, 0),
            long_help="Google, an Alphabet company"
        )

    def test_resources(self, fetcher):
        expected = [("https://www.google.com/", "index.html")]
        assert list(fetcher.resources) == expected

    @pytest.mark.skipif(not HAS_INTERNET,
                        reason="Internet seems to be unavailable")
    def test_call(self, fetcher, tmp_path):
        super().test_call(fetcher, tmp_path)

    def test_without_trailing_slash(self, tmp_path):
        fetcher = URLFetcher(
            resources=[("https://www.google.com", "index.html")],
            short_name="goog2",
            short_help="more goog",
            requires_ofe=(0, 7, 0),
            long_help="What if you forget the trailing slash?"
        )

        self.test_call(fetcher, tmp_path)


class TestPkgResourceFetcher(FetcherTester):
    @pytest.fixture
    def fetcher(self):
        return PkgResourceFetcher(
            resources=[('openfecli.tests', 'test_fetching.py')],
            short_name="me",
            short_help="download this file",
            requires_ofe=(0, 7, 4),
            long_help="whoa, meta."
        )

    def test_resources(self, fetcher):
        expected = [('openfecli.tests', 'test_fetching.py')]
        assert list(fetcher.resources) == expected

