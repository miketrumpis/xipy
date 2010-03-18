.. _installation:

====================
Download and Install
====================

This page lists the software that XIPY is built upon.

Dependencies
------------

Non-Python Requirements
```````````````````````
  `Qt4 <http://qt.nokia.com/>`_
    The UI widget library used for much of XIPY's visualization

  `gcc <http://gcc.gnu.org/>`_
    XIPY currently builds a threaded wrapper for the FFTW3 complex DFT 
    routines, and may take advantage of other C-extension code in the future. 
    Therefore, you must have a compiler to build from
    source.  XCode (OSX) and MinGW (Windows) both include gcc.  
    Furthermore, MinGW is included in the Enthought Python Distribution for Windows.

Python Level Packages
`````````````````````
  Python_ 2.4 or later
  
  NumPy_ 1.2 or later
    Numpy equips Python with N-dimensional data/numerical data objects

  SciPy_ 0.7 or later
    Scipy is a collection of high-level, optimized scientific computing libraries, many of which are Python wrappers for widely used numerical libraries originally written in C and Fortran

  NIPY_ (Neuroimaging in Python)
    XIPY uses many image data types and spatial mapping tools from the
    NIPY project codebase.

  Matplotlib_
    2D python plotting library.

  `PyQt4 <http://www.riverbankcomputing.co.uk/software/pyqt/intro>`_
    Python bindings for the C++ Qt4 libraries.
    

Strong Recommendations
``````````````````````

  iPython_
    Interactive python environment.

  `virtualenv <http://pypi.python.org/pypi/virtualenv>`_
    Utility for changing your shell environment to a custom setup within a session

.. _building_mac:

Building from Mac OS X >= 10.4
------------------------------
  
Binary Installs
```````````````
  * Apple Xcode Dev Tools (on OS X DVD)
  * Qt4 (LGPL version) 
    http://qt.nokia.com/downloads
  * Enthought Python Distribution (EPD, Academic version)
    http://enthought.com/products/getepd.php
  * Subversion ( Already installed on OS X 10.6 )
    http://subversion.tigris.org/getting.html
  * Bazaar version control 
    http://wiki.bazaar.canonical.com/Download

If you had to install subversion with the package above, then you have to add		something like this (bash shell syntax) to your shell environment profile
::
  # Subversion package
  PATH=$PATH:/opt/subversion/bin
  export PATH

Source Installs
```````````````
Unfortunately the two following libraries do not have pre-compiled builds for various systems, so they must be compiled from source (sequentially)
  * SIP http://www.riverbankcomputing.co.uk/software/sip/download
  * PyQt4 http://www.riverbankcomputing.co.uk/software/pyqt/download

For each tar.gz file, you can build and install them with the following commands:
::
  tar zxvf <xyz>.tar.gz
  cd <xyz>
  python configure.py
  make && make install

Now is a good point to get the NIPY code and install it. Make a directory for code (eg "~/trunks"), and do:

``bzr branch lp:nipy nipy-trunk``

You will need to build and install NIPY ... (don't have good recommendation yet)

Finally check out the XIPY code, and set it up
``svn co https://cirl.berkeley.edu/svn/bic/scratch/nm nutmeg-py``

python setup.py build_ext --inplace

.. include:: ../links.txt
