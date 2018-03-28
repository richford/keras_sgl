from __future__ import absolute_import, division, print_function

from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = ("keras-sgl: a python library to perform regression with  "
               "sparse group lasso using the keras library.")
# Long description will go up on the pypi page
long_description = """
keras-sgl
========
keras-sgl is a python library designed to perform regression with sparse
group lasso regularization using the keras library.
"""

NAME = "keras_sgl"
MAINTAINER = "Adam Richie-Halford"
MAINTAINER_EMAIL = "richiehalford@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/richford/keras_sgl"
DOWNLOAD_URL = ""
LICENSE = ""
AUTHOR = "Adam Richie-Halford"
AUTHOR_EMAIL = "richiehalford@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'ksgl': [pjoin('data', '*', '*'), ]}
REQUIRES = [
    "numpy>=1.11.3", "keras>=2.0", "tensorflow",
    "sphinx>=1.4",
]
EXTRAS_REQUIRE = {}
ENTRY_POINTS = {}
