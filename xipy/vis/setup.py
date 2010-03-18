from glob import glob
from os import path

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('vis', parent_package, top_path)
    config.add_subpackage('qt4_widgets')
    # should add tests directory when there is one!    
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
    
