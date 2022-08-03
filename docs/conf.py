# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sphinx_rtd_theme

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import timeeval


# -- Project information -----------------------------------------------------

project = 'TimeEval'
copyright = '2022, Sebastian Schmidl and Phillip Wenig'
author = 'Sebastian Schmidl and Phillip Wenig'
version = timeeval.__version__
release = timeeval.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'myst_parser',
    'sphinx_rtd_dark_mode',
    'sphinx.ext.mathjax'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_logo = '../timeeval.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    'css/custom.css'
]


# -- Napolean settings -------------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_preprocess_types = True

# -- myst_parser settings ----------------------------------------------------
myst_heading_anchors = 3

# -- intersphinx settings ----------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'dask': ('https://docs.dask.org/en/stable/', None),
    'numpy': ('https://numpy.org/doc/1.21/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/version/1.3/', None),
    'sklearn': ('https://scikit-learn.org/0.24/', None),
    'statsmodels': ('https://www.statsmodels.org/v0.12.0/', None),
}

# -- sphinx_rtd_dark_mode settings -------------------------------------------
default_dark_mode = False
