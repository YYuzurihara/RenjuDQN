"""Declares the `renju_dqn._rules_native` pybind11 extension.

`pyproject.toml` covers the rest of the package metadata (PEP 621); a plain
setuptools `ext_modules` list has no declarative pyproject.toml equivalent yet,
so it's wired up here. No cmake required: `Pybind11Extension` compiles the
extension directly via the configured C++ compiler (g++/clang).
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "renju_dqn._rules_native",
        ["native/rules_native.cpp"],
        cxx_std=17,
    ),
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
