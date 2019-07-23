#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name='rasapi',
    version='1.0',
    description='Rasa HTTP API bindings',
    author='Armin Schaare',
    author_email='armin-schaare@hotmail.de',
    packages=find_packages(),
    install_requires=['requests>=2.22.0']
)
