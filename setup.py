from setuptools import setup, find_packages
import os

__version__ = "0.0.1"
NAME = "neuroinfer"

setup(
    name=NAME,
    version=__version__,
    packages=find_packages(),
    description="Neuroinfer",
)
