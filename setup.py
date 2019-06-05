#!/usr/bin/env python

from distutils.core import setup

setup(
    name='rasapi',
    version='1.0',
    description='Rasa HTTP API bindings',
    author='Armin Schaare',
    author_email='armin-schaare@hotmail.de',
    packages=['rasapi'],
    requires=['requests>=2.22.0']
)
