# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'cabana'
copyright = '2025, Gavin'
author = 'Gavin'
release = 'v1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
     'myst_parser',
     # 'recommonmark',
     'sphinx.ext.mathjax',
     'sphinx_markdown_tables',
 ]


templates_path = ['_templates']
exclude_patterns = []
myst_heading_anchors = 3   # allow [text](#heading) links within pages

myst_enable_extensions = [
    "dollarmath",  # Enables $...$ and $$...$$ syntax
    "amsmath",     # Enables LaTeX environments like \begin{align}
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
