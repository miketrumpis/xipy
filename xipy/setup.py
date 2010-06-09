from glob import glob
from os import path

try:
    import Cython
    has_cython = True
except ImportError:
    has_cython = False

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import system_info
    from numpy import get_include
    config = Configuration('xipy', parent_package, top_path)
    config.add_subpackage('overlay')
    config.add_subpackage('external')
    config.add_subpackage('slicing')
    config.add_subpackage('vis')
    config.add_subpackage('io')

    config.add_data_dir('resources')

    # if Cython is present, then try to build the pyx source
    if has_cython:
        src = ['_quick_utils.pyx']
    else:
        src = ['_quick_utils.c']
    config.add_extension('_quick_utils', src, include_dirs=[get_include()])

    
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
