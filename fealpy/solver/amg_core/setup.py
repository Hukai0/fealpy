from setuptools import setup, Extension
import pybind11
import sys
import numpy as np
ext_modules = [
    Extension(
        "amg_core",
        ["rs.cpp"],
        include_dirs=[pybind11.get_include(), np.get_include()],  # Pybind11 & NumPy 头文件
        language="c++",
        extra_compile_args=["-O3", "-Wall", "-std=c++17", "-fPIC"]
    )
]

setup(
    name="amg_core",
    ext_modules=ext_modules,
)
