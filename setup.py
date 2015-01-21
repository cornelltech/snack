from distutils.core import setup
from distutils.extension import Extension
import sys

from Cython.Build import cythonize
import numpy

setup(name='tste_sideinfo',
      ext_modules = cythonize(Extension('tste_sideinfo',["tste_sideinfo.pyx"],
                                        include_dirs = [numpy.get_include()],
                                        extra_compile_args = ['-fopenmp', '-O3', '-ffast-math', '-march=native'],
                                        extra_link_args = ['-fopenmp'],
                  )),
      description="t-STE with sideinfo",
)
