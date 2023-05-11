"""Plugins for the ``fetch`` command"""

from openfecli.fetching import URLFetcher, PkgResourceFetcher

_EXAMPLE_NB_BASE = ("https://raw.githubusercontent.com/"
                    "OpenFreeEnergy/ExampleNotebooks/master/")

RBFE_TUTORIAL = URLFetcher(
    resources=[
        (_EXAMPLE_NB_BASE + "rbfe_tutorial/", "tyk2_ligands.sdf"),
        (_EXAMPLE_NB_BASE + "rbfe_tutorial/", "tyk2_protein.pdb"),
        (_EXAMPLE_NB_BASE + "rbfe_tutorial/", "cli_tutorial.md"),
        (_EXAMPLE_NB_BASE + "rbfe_tutorial/", "python_tutorial.ipynb"),
    ],
    short_name="rbfe-tutorial",
    short_help="CLI and Python tutorial on relative binding free energies",
    section="Tutorials",
    requires_ofe=(0, 7, 0),
).plugin

RBFE_TUTORIAL_RESULTS = PkgResourceFetcher(
    resources=[
        ("openfecli.tests.data", "rbfe_results.tar.gz"),
    ],
    short_name="rbfe-tutorial-results",
    short_help="Results package to follow-up the rbfe-tutorial",
    section="Tutorials",
    requires_ofe=(0, 7, 5),
).plugin
