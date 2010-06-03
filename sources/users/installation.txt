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
    XIPY currently builds some `Cython <http://www.cython.org/>`_-produced C code.
    Therefore, you must have a compiler to build from source.  
    XCode (OSX) and MinGW (Windows) both include gcc.  
    Furthermore, MinGW is included in the Enthought Python Distribution for Windows.

  `VTK <http://vtk.org/>`_
    VTK (Visual Toolkit) is the underlying graphics technology of Mayavi, which in turn
    is the Python package that powers much of the XIPY visualization. As with many
    other libraries, the Enthought Python Distribution ships with compiled VTK binaries.

Python Level Packages
`````````````````````
  Python_ 2.4 or later
  
  NumPy_ 1.2 or later
    Numpy equips Python with N-dimensional data/numerical data objects

  SciPy_ 0.7 or later
    Scipy is a collection of high-level, optimized scientific computing libraries, many 
    of which are Python wrappers for widely used numerical libraries originally written 
    in C and Fortran

  `ETS <https://svn.enthought.com/enthought>`_
    Enthought Tool Suite is a collection of tools intended for scientific computing. 
    XIPY uses Traits, TraitsUI, and Mayavi/TVTK.

  NIPY_ (Neuroimaging in Python)
    XIPY uses many image data types and spatial mapping tools from the NIPY 
    project codebase.

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

Needed
******

  * Apple Xcode Dev Tools (on OS X DVD)
  * Qt4 (LGPL version) 
    http://qt.nokia.com/downloads
  * Git distributed version control
    http://git-scm.com/download

Strongly Recommended
********************

  * Enthought Python Distribution (EPD, Academic version)     
    http://enthought.com/products/getepd.php
    If you are newly installing Python, this is an extremely convenient package
    providing a full suite of Scientific Python tools, including Enthought's own
    tool suite (`ETS <https://svn.enthought.com/enthought>`__), and various
    pre-compiled libraries from Fortran and C/C++.


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

NIPY And Friends
````````````````

Now is a good point to get the NIPY code and install it. Make a directory for code (eg "~/trunks"), and do:

``git clone git://github.com/nipy/nipy.git``

You will need to build and install NIPY ... (don't have good recommendation yet)

``cd nipy && python setup.py install``

XIPY
````

Finally check out the XIPY code, and set it up::

  git clone git://github.com/miketrumpis/xipy.git
  cd xipy && python setup.py install

.. include:: ../links.txt
