from glob import glob
from os import path

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import system_info
    config = Configuration('xipy', parent_package, top_path)
    config.add_subpackage('overlay')
    config.add_subpackage('external')
    config.add_subpackage('slicing')
    config.add_subpackage('vis')
    config.add_subpackage('io')

    config.add_data_dir('resources')
    
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
