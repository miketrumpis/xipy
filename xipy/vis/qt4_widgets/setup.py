from glob import glob
from os import path

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('qt4_widgets', parent_package, top_path)

    # should add tests directory when there is one!
    qt4_dir = 'designer_layouts'
    qt4_files = [path.join(qt4_dir, 'ortho_viewer_with_blender.ui'),
                 path.join(qt4_dir, 'ortho_viewer_layout.ui'),
                 path.join(qt4_dir, 'mayavi_viewer_layout.ui')]
    plugin_files = glob(path.join(qt4_dir, 'plugin/python/*.py'))
    config.add_data_files('designer_layouts/*.ui')
    
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
    
