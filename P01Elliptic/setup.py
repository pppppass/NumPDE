from distutils.core import setup, Extension
import numpy

exts = Extension("exts", sources=["exts/algos.c", "exts/wrappers.c"], include_dirs=[numpy.get_include()], libraries=["mkl_rt"])

setup(ext_modules=[exts])
