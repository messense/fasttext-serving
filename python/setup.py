#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

readme = 'README.md'
with open(readme, encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='fasttext-serving',
    version='0.2.0',
    author='messense',
    author_email='messense@icloud.com',
    url='https://github.com/messense/fasttext-serving',
    packages=find_packages(exclude=('tests', 'tests.*')),
    description='fasttext-serving gRPC client',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=['grpcio', 'protobuf'],
    include_package_data=True,
    classifiers=[
        'Operating System :: MacOS',
        'Operating System :: POSIX',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: Implementation :: CPython',
    ]
)

