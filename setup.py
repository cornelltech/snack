# (C) Michael Wilber, 2013-2015, UCSD and Cornell Tech.
# All rights reserved. Please see the 'LICENSE.txt' file for details.

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
link_args = ['-fopenmp']
blas_library_name = "-lblas" # added to link_args at the end
include_dirs = [
    numpy.get_include(),
    "lib-bhtsne",
]
library_dirs = [
]

if USE_SSE_AVX:
    compile_args.append("-march=native")

# OSX-specific tweaks:
if platform.system() == "Darwin":
    # On OSX, ensure you have the following dependencies:
    # - You must use a gcc version from homebrew that supports openmp
    # - You must install OpenBLAS from homebrew
    #   (not necessary if you link with Accelerate)

    # To use OpenBLAS:
    #BLAS_LIB = "/usr/local/opt/openblas/include"
    #BLAS_LIB = "/usr/local/opt/openblas/lib"
    # To use Apple's Accelerate framework for BLAS:
    BLAS_INCLUDE = "/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers"
    BLAS_LIB = "/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework"
    # adjust this for your gcc version:
    GCC_VERSION = "/usr/local/bin/gcc-5"
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


# Inside Conda, we should link against Conda-provided OpenBLAS.
if platform.system() == "Linux" and 'CONDA_BUILD' in os.environ:
    include_dirs.append(os.environ["PREFIX"]+"/include")
    library_dirs.append(os.environ["PREFIX"]+"/include")
    blas_library_name = "-lopenblas"

snack_extension = Extension(
    'snack._snack', [
        "snack/_snack.pyx",
        "lib-bhtsne/tsne.cpp",
        "lib-bhtsne/sptree.cpp",
    ],
    include_dirs = include_dirs,
    library_dirs = library_dirs,
    language="c++",
    extra_compile_args = compile_args,
    extra_link_args = link_args + [blas_library_name],
)
setup(name = 'snack',
      version = '0.0.3',
      packages = ['snack'],
      ext_modules = cythonize(snack_extension),
      description="Stochastic Neighbor and Crowd Kernel (SNaCK) embeddings: Quick and dirty visualization of large-scale datasets via concept embeddings",
      author='Michael Wilber',
      author_email='mwilber@mjwilber.org',
      url='http://vision.cornell.edu/se3/projects/concept-embeddings/',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'License :: OSI Approved :: zlib/libpng License',
          'Operating System :: MacOS',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Cython',
          'Programming Language :: Python :: 2.7',
          'Topic :: Scientific/Engineering :: Visualization',
      ],
      keywords='snack embedding tsne visualization triplets tste',

)
