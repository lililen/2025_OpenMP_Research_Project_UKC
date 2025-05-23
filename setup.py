from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

ext_modules = [
    Extension(
        "rotate",  # name of the .pyd module
        ["rotate.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["/O2", "/std:c++14", "/openmp:experimental"],
    ),
]

setup(
    name="rotate",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
