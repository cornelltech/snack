Stochastic Neighbor and Crowd Kernel (SNaCK) embedding
======================================================
Quick and dirty visualization of large-scale datasets via concept embeddings

See

Installation
============
The following platforms are supported:
- Python 2.7 on Linux, x64
   - Binary packages available on Conda
   - Source packages available from Pip
- Python 2.7 on OSX
   - Binary packages for Yosemite available on Conda
   - Source packages available from Pip (Homebrew-GCC required)

Linux and Mac OS X: Install from Conda
--------------------------------------
Just run:
    $ conda install -c https://conda.anaconda.org/gcr snack

Linux: Install from source with Pip
-----------------------------------
Just run:
    $ pip install snack

You need to install Python 2.7, Numpy, and Cython. You also need a
working compiler, CBLAS, and the Python development headers, which are
installable from your distribution's package manager.

To install SNaCK and its dependencies on a clean Ubuntu Trusty x64
system, run:

    # sudo aptitude install \
      build-essential       \
      python-dev            \
      libblas3              \
      libblas-dev           \
      python-virtualenv
    $ virtualenv venv; source venv/bin/activate
    $ pip install numpy
    $ pip install cython
    $ pip install snack

OS X: Install from source with Pip and Homebrew
-----------------------------------------------
If you are on Mac OS X, you must install the real "not-clang" version
of gcc because it has OpenMP support. At the time of writing, clang
does not support OpenMP, and Apple has unhelpfully symlinked clang to
`/usr/bin/gcc`. This is not sufficient.

Using Apple-provided GCC is NOT supported. If `gcc-5 --version`
contains the string `clang` anywhere in its output, you do not have
the correct version of gcc.

Using Apple-provided Python is NOT supported.

The recommended installation method on OS X is with Homebrew:

    $ brew install gcc
    $ brew install python
    $ virtualenv venv; source venv/bin/activate
    $ pip install numpy
    $ pip install cython
    $ pip install snack

You may need to edit `setup.py` and change `GCC_VERSION` to point to
the correct version, if you are not using `/usr/local/bin/gcc-5`.
