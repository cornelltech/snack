from distutils.core import setup
from distutils.extension import Extension
import sys
import os
import platform

from Cython.Build import cythonize
import numpy

# Disable this if you get 'no such instruction' errors
USE_SSE_AVX = True

compile_args = [
        '-fopenmp',
        '-O3',
        '-ffast-math',
]
include_dirs = [
    numpy.get_include(),
    "lib-bhtsne",
]
library_dirs = [
]

if USE_SSE_AVX:
    compile_args.append("-march=native")

if platform.system() == "Darwin":
    BLAS_INCLUDE = "/usr/local/opt/openblas/include"
    BLAS_LIB = "/usr/local/opt/openblas/lib"
    GCC_VERSION = "/usr/local/bin/gcc-5"
    print "On OSX, ensure you have the following dependencies:"
    print "- You must use a gcc version from homebrew that supports openmp"
    print "- You must install OpenBLAS from homebrew"
    if not os.path.exists(BLAS_INCLUDE+"/cblas.h"):
        print "Please install OpenBLAS with:"
        print "    $ brew install openblas"
        print "or edit setup.py."
        sys.exit(1)
    if not os.path.exists(GCC_VERSION):
        print "Please install GCC from homebrew wth:"
        print "    $ brew install gcc"
        print "Note that on OSX, /usr/local/bin/gcc is a link to CLang by default,"
        print "which will not work."
        sys.exit(1)
    os.environ["CC"] = GCC_VERSION
    os.environ["CXX"] = GCC_VERSION
    include_dirs.append(BLAS_INCLUDE)
    library_dirs.append(BLAS_LIB)
    if USE_SSE_AVX:
        compile_args.append("-Wa,-q")
        # from gcc man page: "-q: Use the clang(1) integrated
        # assembler instead of the GNU based system assembler."
        # The clang assembler knows about AVX instructions.
        # GNU assembler does not, for some reason.

snack_extension = Extension(
    '_snack', [
        "_snack.pyx",
        "lib-bhtsne/tsne.cpp",
        "lib-bhtsne/sptree.cpp",
    ],
    include_dirs = include_dirs,
    library_dirs = library_dirs,
    language="c++",
    extra_compile_args = compile_args,
    extra_link_args = ['-fopenmp', '-lcblas'],
)
setup(name='snack',
      ext_modules = cythonize(snack_extension),
      description="SNaCK embedding: Stochastic Neighbor and Crowd Kernel",
)
