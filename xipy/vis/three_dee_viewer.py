import os, sys
# try loading this generated class from the uic module
from PyQt4 import uic
design_path = os.path.join(os.path.split(__file__)[0], 'qt4_widgets/designer_layouts/mayavi_viewer_layout.ui')
ui_layout_class, base_class = uic.loadUiType(design_path)

from PyQt4 import QtCore, QtGui

import nipy.core.api as ni_api

from xipy.vis.qt4_widgets import browse_files
from xipy.vis.qt4_widgets.xipy_window_app import XIPYWindowApp
from xipy.slicing import load_resampled_slicer, load_sampled_slicer
from xipy.io import load_spatial_image

class MayaviViewer(XIPYWindowApp):
    def __init__(self, parent=None, image=None):
        super(MayaviViewer, self).__init__(parent=parent,
                                           designer_layout=ui_layout_class())
        # mayavi_widget coming from layout 
        self.viewer = self.mayavi_widget.mr_vis
        self.plugin_launched.connect(self._add_panel)
        self.actionLoad_MR_File.triggered.connect(self.on_load_mr)

        if image is not None:
            try:
                self.update_image(image)
            except:
                pass
        

    def _add_panel(self, plugin_tool):
        self.mayavi_widget.add_toolbar(plugin_tool.func_man)

    def _update_plugin_args(self):
        # Don't connect any Qt level signal/slots -- this widget is
        # all hooked up with Traits notification
        self._plugin_args = ( (), (), (), self.image.bbox )
        self._plugin_kwargs['main_ref'] = self

    ########## ACTION/MENU HANDLERS ##########
    def on_load_mr(self, bool):
        fname = browse_files(self, dialog='Select Image File',
                             wildcard='Images (*.nii *.nii.gz *.hdr *.img)')
        if fname:
            self.update_image(fname)

    ########## IMAGE DATA UPDATES ##########
    def update_image(self, image, mode='world'):
        if type(image) != ni_api.Image:
            try:
                image = load_spatial_image(image)
            except RuntimeError:
                self.image = None
                self._image_loaded = False
                raise
        self._image_loaded = True
        self.viewer.blender.main = image
        self.image = self.viewer.blender.main
        self._update_plugin_args()

