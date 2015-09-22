Stochastic Neighbor and Crowd Kernel (SNaCK) embedding
======================================================
Quick and dirty visualization of large-scale datasets via concept embeddings


Installation
============
The following platforms are supported:
- Python 2.7 on Linux, using binary packages on Conda
- Python 2.7 on Linux, using Pip
- Python 2.7 on OSX, using binary packages on Conda
- Python 2.7 on OSX, using Conda and GCC from Homebrew

Linux, Mac OS X: Install from Conda
------------------------------
TODO.

Linux: Install from Pip
-----------------------
Just run:

    $ pip install snack

You need to install Python 2.7, Numpy, and Cython. You also need a
working compiler, CBLAS, and the Python development headers, which are
installable from your distribution's package manager.

To install SNaCK on a clean Ubuntu Trusty x64 system, run:

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

OS X: Install from Pip and Homebrew
----------------------------------
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


Just build without installing
-----------------------------

To simply build Snack without installing it, run:

    $ python setup.py build_ext --inplace

This builds `snack/_snack.so`. You can move the `snack` folder to your
project's directory and then `import snack`. This should work as long
as the `snack` folder is inside your current directory.

How to use
----------

Examples
--------

See also
--------
