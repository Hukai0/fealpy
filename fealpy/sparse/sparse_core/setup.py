from setuptools import setup, Extension
import pybind11
import numpy as np

ext_modules = [
    Extension(
        "csr_utils",
        ["csr_utils_numpy.cpp", "csr_utils.cpp"],  # 你的 C++ 代码
        include_dirs=[pybind11.get_include(), np.get_include()],  # Pybind11 & NumPy 头文件
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],  # 使用C++17和优化选项
    )
]

setup(
    name="csr_utils",
    ext_modules=ext_modules,
)
