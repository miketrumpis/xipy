#!/usr/bin/env python
from PyQt4 import QtGui, QtCore

from xipy.vis.qt4_widgets.ortho_slices import MplQT4OrthoSlicesWidget
from xipy.vis.rgba_blending import BlendedImages
import xipy.vis.color_mapping as cm
from xipy.slicing import SAG, COR, AXI


import numpy as np

class BlendedOrtho(QtGui.QMainWindow):
    def __init__(self, img1, img2, **bimage_kws):
        QtGui.QMainWindow.__init__(self)
        self.ortho_figs_widget = MplQT4OrthoSlicesWidget(parent=self)
        self.setCentralWidget(self.ortho_figs_widget)
        self.image = BlendedImages(main=img1, over=img2,
                                   over_alpha = 0.5,
                                   **bimage_kws)
        self.initialize_plots()
        # connect variants of position states
        QtCore.QObject.connect(self.ortho_figs_widget,
                               QtCore.SIGNAL("xyz_state(int,int,int)"),
                               self.xyz_position_watcher)
        QtCore.QObject.connect(self.ortho_figs_widget,
                               QtCore.SIGNAL("xyz_state(int,int)"),
                               self.xyz_position_watcher)
        QtCore.QObject.connect(self.ortho_figs_widget,
                               QtCore.SIGNAL("xyz_state(int)"),
                               self.xyz_position_watcher)

    def initialize_plots(self):
        planes = self.image.cut_image((0,0,0))
        lims = self.image.bbox
        self.ortho_figs_widget.initialize_plots(
            planes, (0,0,0), lims,
            interpolation='nearest'
            )


    def xyz_position_watcher(self, *args):
        axes = args
        xyz_loc = self.ortho_figs_widget.active_voxel
        # just simply replot for now
        self.update_fig_data(xyz_loc, axes=axes)

    def update_fig_data(self, xyz_loc, axes=(SAG, COR, AXI)):
        planes = self.image.cut_image(xyz_loc, axes=axes)
        self.ortho_figs_widget.update_plot_data(
            planes, fig_labels=axes
            )

if __name__=='__main__':
    import sys
    import xipy.io as xio
    from test_two_images import make_option_parser
    op = make_option_parser()
    op.add_option('-t', '--vtk-order', dest='vtk_order',
                  action='store_true', default=False)

    (opts, args) = op.parse_args()
    
    images = []
    for arg in args:
        images.append(xio.load_spatial_image(arg))
    images = images + [None] * (2-len(images))
    
    if QtGui.QApplication.startingUp():
        app = QtGui.QApplication(sys.argv)
    else:
        app = QtGui.QApplication.instance()
    kws = dict(vtk_order=opts.vtk_order,
               main_spline_order=opts.main_spline_order,
               over_spline_order=opts.over_spline_order)
    win = BlendedOrtho(*images, **kws)
    win.show()
    sys.exit(app.exec_())
