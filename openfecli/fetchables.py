"""Plugins for the ``fetch`` command"""

from openfecli.fetching import URLFetcher, PkgResourceFetcher

_EXAMPLE_NB_BASE = ("https://raw.githubusercontent.com/"
                    "OpenFreeEnergy/ExampleNotebooks/master/")

RHFE_TUTORIAL = URLFetcher(
    resources=[
        (_EXAMPLE_NB_BASE + "easy_campaign/molecules/rhfe/",
         "benzenes_RHFE.sdf"),
        (_EXAMPLE_NB_BASE + "easy_campaign/", "cli-tutorial.md"),
        (_EXAMPLE_NB_BASE + "easy_campaign/", "rhfe-python-tutorial.ipynb"),
    ],
    short_name="rhfe-tutorial",
    short_help="CLI and Python tutorial on relative hydration free energies",
    requires_ofe=(0, 7, 0),
).plugin

RHFE_TUTORIAL_RESULTS = PkgResourceFetcher(
    resources=[
        ("openfecli.tests.data", "results.tar.gz"),
    ],
    short_name="rhfe-tutorial-results",
    short_help="Results package to follow-up the rhfe-tutorial",
    requires_ofe=(0, 7, 5),
).plugin
