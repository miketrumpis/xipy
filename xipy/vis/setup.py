from glob import glob
from os import path

try:
    import Cython
    has_cython = True
except ImportError:
    has_cython = False

from numpy.distutils import log

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy import get_include
    config = Configuration('vis', parent_package, top_path)
    config.add_subpackage('qt4_widgets')
    config.add_subpackage('mayavi_widgets')

    config.add_data_dir('tests')
    
    # if Cython is present, then try to build the pyx source
    if has_cython:
        src = ['_blend_pix.pyx']
    else:
        src = ['_blend_pix.c']
    config.add_extension('_blend_pix', src, include_dirs=[get_include()])
        
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
    
