# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from importlib.metadata import version
from packaging.version import parse
from pathlib import Path
from inspect import cleandoc

from git import Repo
import nbsphinx
import nbformat

sys.path.insert(0, os.path.abspath('../'))


os.environ["SPHINX"] = "True"

# -- Project information -----------------------------------------------------

project = "OpenFE"
copyright = "2022, The OpenFE Development Team"
author = "The OpenFE Development Team"
version = parse(version("openfe")).base_version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click.ext",
    "sphinxcontrib.autodoc_pydantic",
    "sphinx_toolbox.collapse",
    "sphinx.ext.autosectionlabel",
    "sphinx_design",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "docs._ext.sass",
    "myst_parser",
    "nbsphinx",
    "nbsphinx_link",
    "sphinx.ext.mathjax",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.9", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "scikit.learn": ("https://scikit-learn.org/stable", None),
    "openmm": ("http://docs.openmm.org/latest/api-python/", None),
    "rdkit": ("https://www.rdkit.org/docs", None),
    "openeye": ("https://docs.eyesopen.com/toolkits/python/", None),
    "mdtraj": ("https://www.mdtraj.org/1.9.5/", None),
    "openff.units": ("https://docs.openforcefield.org/projects/units/en/stable", None),
    "gufe": ("https://gufe.readthedocs.io/en/latest/", None),
}

autoclass_content = "both"
# Make sure labels are unique
# https://www.sphinx-doc.org/en/master/usage/extensions/autosectionlabel.html#confval-autosectionlabel_prefix_document
autosectionlabel_prefix_document = True

autodoc_pydantic_model_show_json = False

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "inherited-members": "GufeTokenizable,BaseModel",
    "undoc-members": True,
    "special-members": "__call__",
}
toc_object_entries_show_parents = "hide"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "**/Thumbs.db",
    "**/.DS_Store",
    "_ext",
    "_sass",
    "**/README.md",
    "ExampleNotebooks",
]

autodoc_mock_imports = [
    "matplotlib",
    "mdtraj",
    "openmmforcefields",
    "openmmtools",
    "pymbar",
]

# Extensions for the myst parser
myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
    "smartquotes",
    "replacements",
    "deflist",
    "attrs_inline",
]
myst_heading_anchors = 3

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "ofe_sphinx_theme"
html_theme_options = {
    "logo": {"text": "OpenFE Documentation"},
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/OpenFreeEnergy/openfe",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        }
    ],
    "accent_color": "DarkGoldenYellow",
    "navigation_with_keys": False,
}
html_logo = "_static/Squaredcircle.svg"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']


# replace macros
rst_prolog = """
.. |rdkit.mol| replace:: :class:`rdkit.Chem.rdchem.Mol`
"""

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
    "css/custom-api.css",
    "css/deflist-flowchart.css",
]

# custom-api.css is compiled from custom-api.scss
sass_src_dir = "_sass"
sass_out_dir = "_static/css"

# Clone or update ExampleNotebooks
example_notebooks_path = Path("ExampleNotebooks")
try:
    if example_notebooks_path.exists():
        repo = Repo(example_notebooks_path)
        repo.remote('origin').pull()
    else:
        repo = Repo.clone_from(
            "https://github.com/OpenFreeEnergy/ExampleNotebooks.git",
            to_path=example_notebooks_path,
        )
except Exception as e:
    from sphinx.util.logging import getLogger

    filename = e.__traceback__.tb_frame.f_code.co_filename
    lineno = e.__traceback__.tb_lineno
    getLogger('sphinx.ext.openfe_git').warning(
        f"Getting ExampleNotebooks failed in {filename} line {lineno}: {e}"
    )


# First, create links at top of notebook pages
# All notebooks are in ExampleNotebooks repo, so link to that
# Finally, add sphinx reference anchor in prolog so that we can make refs
nbsphinx_prolog = cleandoc(r"""
    {%- set gh_repo = "OpenFreeEnergy/ExampleNotebooks" -%}
    {%- set gh_branch = "main" -%}
    {%- set path = env.doc2path(env.docname, base=None) -%}
    {%- if path.endswith(".nblink") -%}
        {%- set path = env.metadata[env.docname]["nbsphinx-link-target"] -%}
    {%- endif -%}
    {%- if path.startswith("ExampleNotebooks/") -%}
        {%- set path = path.replace("ExampleNotebooks/", "", 1) -%}
    {%- endif -%}
    {%- set gh_url =
        "https://www.github.com/"
        ~ gh_repo
        ~ "/blob/"
        ~ gh_branch
        ~ "/"
        ~ path
    -%}
    {%- set dl_url =
        "https://raw.githubusercontent.com/"
        ~ gh_repo
        ~ "/"
        ~ gh_branch
        ~ "/"
        ~ path
    -%}
    {%- set colab_url =
        "http://colab.research.google.com/github/"
        ~ gh_repo
        ~ "/blob/"
        ~ gh_branch
        ~ "/"
        ~ path
    -%}

    .. container:: ofe-top-of-notebook

        .. button-link:: {{gh_url}}
            :color: primary
            :shadow:
            :outline:

            :octicon:`mark-github` View on GitHub

        .. button-link:: {{dl_url}}
            :color: primary
            :shadow:
            :outline:

            :octicon:`download` Download Notebook

        .. button-link:: {{colab_url}}
            :color: primary
            :shadow:
            :outline:

            :octicon:`rocket` Run in Colab

    .. _{{ env.doc2path(env.docname, base=None) }}:
""")
