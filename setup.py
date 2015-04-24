from distutils.core import setup
from distutils.extension import Extension
import sys

from Cython.Build import cythonize
import numpy

setup(name='snack',
      ext_modules = cythonize(Extension(
          '_snack',[
              "_snack.pyx",
              "lib-bhtsne/tsne.cpp",
              "lib-bhtsne/sptree.cpp",
          ],
          include_dirs = [numpy.get_include(), "lib-bhtsne"],
          language="c++",
          extra_compile_args = [
              '-fopenmp',
              '-O3',
              '-ffast-math',
              '-march=native',
          ],
          extra_link_args = ['-fopenmp', '-lcblas'],
      )),
      description="SNaCK embedding: Stochastic Neighbor and Crowd Kernel",
)
