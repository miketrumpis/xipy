''' Module to automate cython building '''

import os
from os.path import join as pjoin

from distutils.extension import Extension


# Thanks @ Eleftherios Garyfallidis
def make_cython_ext(modulename,
                    has_cython,
                    include_dirs=[],
                    extra_c_sources=[],
                    extra_compile_args=[]):
    ''' Create Cython extension builder from module names

    Returns extension for building and command class depending on
    whether you want to use Cython and ``.pyx`` files for building
    (`has_cython` == True) or the Cython-generated C files (`has_cython`
    == False).

    Assumes ``pyx`` or C file has the same path as that implied by
    modulename. 

    Parameters
    ----------
    modulename : string
       module name, relative to setup.py path, with python dot
       separators, e.g mypkg.mysubpkg.mymodule
    has_cython : bool
       True if we have cython, False otherwise
    include_dirs : (empty) sequence
       include directories
    extra_c_sources : (empty) sequence
       sequence of strings giving extra C source files
    extra_compile_args : (empty) sequence
       sequence of strings for compiler options (eg, optimization flags)

    Returns
    -------
    ext : extension object
    cmdclass : dict
       command class dictionary for setup.py

    Examples
    --------
    You will need Cython on your python path to run these tests. 
    
    >>> modulename = 'pkg.subpkg.mymodule'
    >>> ext, cmdclass = make_cython_ext(modulename, True, None,['test.c'])
    >>> ext.name == modulename
    True
    >>> pyx_src = os.path.join('pkg', 'subpkg', 'mymodule.pyx')
    >>> ext.sources == [pyx_src, 'test.c']
    True
    >>> import Cython.Distutils
    >>> cmdclass['build_ext'] == Cython.Distutils.build_ext
    True
    >>> ext, cmdclass = make_cython_ext(modulename, False, None, ['test.c'])
    >>> ext.name == modulename
    True
    >>> pyx_src = os.path.join('pkg', 'subpkg', 'mymodule.c')
    >>> ext.sources == [pyx_src, 'test.c']
    True
    >>> cmdclass
    {}
    '''
    if has_cython:
        src_ext = '.pyx'
    else:
        src_ext = '.c'
    pyx_src = pjoin(*modulename.split('.')) + src_ext
    sources = [pyx_src] + extra_c_sources
    ext = Extension(modulename, sources,
                    include_dirs = include_dirs,
                    extra_compile_args=extra_compile_args)
    if has_cython:
        from Cython.Distutils import build_ext
        cmdclass = {'build_ext': build_ext}
    else:
        cmdclass = {}
    return ext, cmdclass
        
# ----------------------------------------------------------------------------

# Thanks @ Matthew Brett

# Standard library imports
from os.path import join as pjoin, dirname
from distutils.dep_util import newer_group
from distutils.errors import DistutilsError

from numpy.distutils.misc_util import appendpath
from numpy.distutils import log


def generate_a_pyrex_source(self, base, ext_name, source, extension):
    ''' Monkey patch for numpy build_src.build_src method

    Uses Cython instead of Pyrex.

    Assumes Cython is present
    '''
    if self.inplace:
        target_dir = dirname(base)
    else:
        target_dir = appendpath(self.build_src, dirname(base))
    target_file = pjoin(target_dir, ext_name + '.c')
    depends = [source] + extension.depends
    if self.force or newer_group(depends, target_file, 'newer'):
        import Cython.Compiler.Main
        log.info("cythonc:> %s" % (target_file))
        self.mkpath(target_dir)
        options = Cython.Compiler.Main.CompilationOptions(
            defaults=Cython.Compiler.Main.default_options,
            include_path=extension.include_dirs,
            output_file=target_file)
        cython_result = Cython.Compiler.Main.compile(source,
                                                   options=options)
        if cython_result.num_errors != 0:
            raise DistutilsError("%d errors while compiling %r with Cython" \
                  % (cython_result.num_errors, source))
    return target_file


