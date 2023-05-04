"""Plugins for the ``fetch`` command"""

from openfecli.fetching import URLFetcher, PkgResourceFetcher

_EXAMPLE_NB_BASE = ("https://raw.githubusercontent.com/"
                    "OpenFreeEnergy/ExampleNotebooks/master/")

RHFE_TUTORIAL = URLFetcher(
    resources=[
        (_EXAMPLE_NB_BASE + "easy_campaign/molecules/rhfe/",
         "benzenes_RHFE.sdf")
    ],
    short_name="rhfe-cli-tutorial",
    short_help="CLI tutorial on relative hydration free energies",
    requires_ofe=(0, 7, 0),
).plugin
