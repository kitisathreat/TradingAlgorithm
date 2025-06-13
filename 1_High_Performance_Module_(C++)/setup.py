from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import setuptools

__version__ = '0.1.0'

# Get the absolute path to the directory containing this file
HERE = os.path.abspath(os.path.dirname(__file__))

class get_pybind_include(object):
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

ext_modules = [
    Extension(
        'decision_engine',
        ['bindings.cpp', 'engine.cpp'],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            HERE  # Include the current directory for engine.h
        ],
        language='c++',
        extra_compile_args=['/std:c++17'] if sys.platform == 'win32' else ['-std=c++17'],
    ),
]

setup(
    name='decision_engine',
    version=__version__,
    author='Trading Algorithm Team',
    author_email='your.email@example.com',
    description='High-performance C++ trading decision engine with Python bindings',
    long_description='',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.6.0'],
    setup_requires=['pybind11>=2.6.0'],
    zip_safe=False,
    python_requires=">=3.7",
) 