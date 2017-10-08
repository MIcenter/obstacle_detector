#!/usr/bin/python3

from setuptools import setup, find_packages
from os.path import join, dirname

setup(
    name='obastcle detector',
    version='0.1',
    description='Package for detecting obstacles in the plane of motion.',
    author='Ivan Deylid',
    license='MIT',
    author_email='ivanov.dale@gmail.com',
    url='https://www.github.com/sid1057/obstacle_detector',
    packages=find_packages(),
    long_description=open(join(dirname(__file__), 'README.md')).read(),
)
