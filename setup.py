import io
import os
import sys
from shutil import rmtree
from setuptools import find_packages, setup, Command
import setuptools
# Package meta-data.
NAME = 'anomaly_detection_models'
DESCRIPTION = 'Models for anomaly detection; see e.g. https://arxiv.org/abs/2009.02205'
VERSION = '0.1.3'
AUTHOR = 'Luc Le Pottier'
EMAIL = 'luclepot@umich.edu'
URL = 'https://github.com/luclepot/anomaly_detection_models'
REQUIRES_PYTHON = '>=3.6.5'

# What packages are required for this module to be executed?
REQUIRED = [
    'matplotlib',
    'numpy',
    'scikit-learn>=0.21.2',
    'pickleshare>=0.7.5',
    'pandas>=0.25.1',
]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name=NAME,
    version=VERSION,
    install_requires=REQUIRED,
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    project_urls={
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # package_dir={"": "src"},
    py_modules=["anomaly_detection_models"],
    # scripts=['modes.py'],
    # packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
