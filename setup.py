# setup.py for experimental projects.  Allows packaging for pip install /
# wheel build.
import os
from distutils import log
from setuptools import setup, find_packages

def read(fname):
    """Read file from the main directory of this package"""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="matching_exp",
    version="0.0.0a0",
    author="azadeh.mobasher",
    author_email="{azadeh.mobasher}@gmail.com",
    description=("Matching"),
    packages=find_packages(exclude=["tests"]),
    install_requires=["ortools==9.3.10497"],
	include_package_data=True,
    classifiers=[
		"Programming Language :: Python :: 3",
		"Topic :: Matching Control Treatment Cases",
		"Operating System :: Windows"],
)
