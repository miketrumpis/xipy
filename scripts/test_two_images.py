#!/usr/bin/env python
from PyQt4 import QtGui, QtCore

from matplotlib.colors import Normalize

from xipy.vis.qt4_widgets.ortho_slices import MplQT4OrthoSlicesWidget
import xipy.colors.color_mapping as cm
from xipy.slicing import SAG, COR, AXI
from xipy.slicing.image_slicers import ResampledVolumeSlicer, \
     ResampledIndexVolumeSlicer

import numpy as np

class SimpleOrtho(QtGui.QMainWindow):
    def __init__(self, img1, img2):
        QtGui.QMainWindow.__init__(self)
        self.ortho_figs_widget = MplQT4OrthoSlicesWidget(parent=self)
        self.setCentralWidget(self.ortho_figs_widget)
        self.image = img1
        self._main_norm = None
        self.over_image = img2
        self._over_norm = None
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

    def _slice_and_map_planes(self, axes=(SAG, COR, AXI)):
        xyz_loc = self.ortho_figs_widget.active_voxel
        planes = self.image.cut_image(xyz_loc, axes=axes)
        o_planes = self.over_image.cut_image(xyz_loc, axes=axes)
        if not isinstance(self.image, ResampledIndexVolumeSlicer):
            if self._main_norm is None:
                self._main_norm = Normalize(
                    vmin=np.nanmin(self.image.image_arr),
                    vmax=np.nanmax(self.image.image_arr)
                    )
            planes = map(self._main_norm, planes)
            if self._over_norm is None:
                self._over_norm = Normalize(
                    vmin=np.nanmin(self.over_image.image_arr),
                    vmax=np.nanmax(self.over_image.image_arr)
                    )
            o_planes = map(self._over_norm, o_planes)

        planes = map(lambda x: cm.gray(x, bytes=True), planes)
        o_planes = map(lambda x: cm.jet(x, bytes=True, alpha=0.5), o_planes)

        return zip(planes, o_planes)
    
    def initialize_plots(self):
        planes = self._slice_and_map_planes()
        main_lims = self.image.bbox
        main_planes = [p for p,q in planes]
        self.ortho_figs_widget.initialize_plots(
            main_planes, (0,0,0), main_lims,
            interpolation='nearest'
            )

        over_planes = [q for p,q in planes]
        over_lims = self.over_image.bbox
        self.ortho_figs_widget.initialize_overlay_plots(
            over_planes, over_lims,
            interpolation='nearest',
            )

    def xyz_position_watcher(self, *args):
        axes = args
        xyz_loc = self.ortho_figs_widget.active_voxel
        # just simply replot for now
        self.update_fig_data(xyz_loc, axes=axes)

    def update_fig_data(self, xyz_loc, axes=(SAG, COR, AXI)):
        planes = self._slice_and_map_planes(axes=axes)
        self.ortho_figs_widget.update_plot_data(
            planes, fig_labels=axes
            )

def make_option_parser():
    import optparse
    usage = 'usage: %prog [options]'
    op = optparse.OptionParser(usage=usage)
    op.add_option('-m', '--main-spline-order', dest='main_spline_order',
                  type='int', default=0)
    op.add_option('-o', '--over-spline-order', dest='over_spline_order',
                  type='int', default=0)
    return op

if __name__=='__main__':
    import sys
    import xipy.volume_utils as vu
    import xipy.io as xio
    op = make_option_parser()
    op.add_option('-i', '--index-slicer', dest='use_index_slicer',
                  action='store_true', default=False)
    (opts, args) = op.parse_args()

    files = args

    cls = ResampledIndexVolumeSlicer if opts.use_index_slicer \
          else ResampledVolumeSlicer
    orders = (opts.main_spline_order, opts.over_spline_order)
    images = [cls(xio.load_spatial_image(i), order=o)
              for i, o in zip(files, orders)]
    if QtGui.QApplication.startingUp():
        app = QtGui.QApplication(sys.argv)
    else:
        app = QtGui.QApplication.instance()
    win = SimpleOrtho(*images)
    win.show()
    sys.exit(app.exec_())
