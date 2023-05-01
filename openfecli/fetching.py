import click
from plugcli.plugin_management import CommandPlugin

import urllib.request
import importlib.resources
import shutil

import pathlib

class _Fetcher:
    """Base class for fetchers. Defines the API and plugin creation.

    Parameters
    ----------
    resources: Iterable[Tuple[str, str]]
        resources to be downloaded, as (source, filename)
    short_name: str
        name of the command used after openfe fetch
    short_help: str
        short help shown in openfe fetch --help
    long_help: str
        help shown in openfe fetch short_name --help
    requires_ofe: Tuple
        minimum version of OpenFE required
    """
    REQUIRES_INTERNET = None
    def __init__(
        self,
        resources,
        short_name,
        short_help,
        requires_ofe,
        long_help=None
    ):
        self._resources = resources
        self.short_name = short_name
        self.short_help = short_help
        self.requires_ofe = requires_ofe
        self.long_help = long_help

    @property
    def resources(self):
        yield from self._resources

    def __call__(self, directory: pathlib.Path):
        raise NotImplementedError()

    @property
    def plugin(self):
        docs = self.long_help or ""
        docs += "\n\nThis will fetch the following files:\n\n"
        # if you're getting a problem with unpacking here, you probably
        # forgot to make resources a list of tuple of (base, filename)
        for _, filename in self.resources:
            docs += f"* {filename}\n"
        @click.command(
            self.short_name,
            short_help=self.short_help,
            help=docs,
        )
        @click.option(
            '-d', '--directory', default='.',
            help="output directory, defaults to current directory",
            type=click.Path(file_okay=False, dir_okay=True, writable=True),
        )
        def command(directory):
            directory.mkdir(parents=True, exist_ok=True)
            self(directory)

        if self.REQUIRES_INTERNET is True:
            section = "Requires Internet"
        elif self.REQUIRES_INTERNET is False:
            section = "Built-in"
        else:
            raise RuntimeError("Class must set boolean REQUIRES_INTERNET")

        return FetchablePlugin(
            command,
            section,
            requires_lib=self.requires_ofe,
            requires_cli=self.requires_ofe
        )

class URLFetcher(_Fetcher):
    """Fetcher for URLs.

    Resources should be (base, filename), e.g., ("https://google.com/",
    "index.html).
    """
    REQUIRES_INTERNET = True
    def __call__(self, dest_dir):
        for base, filename in self.resources:
            # let's just prevent one footgun here
            if not base.endswith('/'):
                base += "/"

            with urllib.request.urlopen(base + filename) as resp:
                contents = resp.read()

            with open(dest_dir / filename, mode='wb') as f:
                f.write(contents)


class PkgResourceFetcher(_Fetcher):
    """Fetcher for data included with the package

    Resources should be (package, filename), e.g., ("openfecli",
    "__init__.py").
    """
    REQUIRES_INTERNET = False
    def __call__(self, dest_dir):
        for package, filename in self.resources:
            ref = importlib.resources.files(package) / filename
            with importlib.resources.as_file(ref) as f:
                shutil.copyfile(ref, dest_dir / filename)


# should work, but don't want to write tests yet or deal with typing
# class MixedResourcesFetcher(_Fetcher):
#     @property
#     def REQUIRES_INTERNET(self):
#         return any([fetcher.REQUIRES_INTERNET
#                     for fetcher in self._resources])

#     @property
#     def resources(self):
#         for resource in self._resources:
#             yield from resource.resources

#     def __call__(self):
#         for fetcher in self._resources:
#             fetcher()


class FetchablePlugin(CommandPlugin):
    pass  # separate class for isinstance reasons


