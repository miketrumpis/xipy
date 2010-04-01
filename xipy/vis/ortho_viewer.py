import os, sys
# try loading this generated class from the uic module
from PyQt4 import uic
design_path = os.path.join(os.path.split(__file__)[0], 'qt4_widgets/designer_layouts/ortho_viewer_layout.ui')
ui_layout_class, base_class = uic.loadUiType(design_path)

from PyQt4 import QtCore, QtGui
import matplotlib as mpl
## import matplotlib.cm as cm
import xipy.vis.color_mapping as cm

import numpy as np

from xipy.slicing import SAG, COR, AXI, load_resampled_slicer, \
     load_sampled_slicer
from xipy.utils import with_attribute
from xipy.vis.qt4_widgets import browse_files
from xipy.vis.qt4_widgets.xipy_window_app import XIPYWindowApp
from xipy.vis import mayavi_widgets
from xipy.overlay import overlay_thresholding_function, \
     make_mpl_image_properties
from xipy.overlay.plugins import all_registered_plugins

interpolations = ['nearest', 'bilinear', 'sinc']
cmaps = cm.cmap_d.keys()
cmaps.sort()

class OrthoViewer(XIPYWindowApp):
    
    def __init__(self, image=None, fliplr=False, parent=None):
##         QtGui.QMainWindow.__init__(self)
        super(OrthoViewer, self).__init__(parent=parent,
                                          designer_layout=ui_layout_class())

        # if ANALYZE Images are known to be presented in right-handed
        # orientation, then you can forcibly ignore the sign of
        # R[0,0], in the index to world transform
        self._fliplr = fliplr
        
##         self.setupUi(self)

        # kick in the manual connections etc written below
        self.extra_setup_ui()
        self._image_loaded = False
        self._overlay_active = False

        try:
            self.update_image(image)
        except ValueError:
            print 'no image to load in __init__()'
            
        # Creates Mayavi 3D view
        self.create_mayavi_window()
        # do a little catch up with the informing
        if self._image_loaded:
            self.mayavi_widget.mr_vis.anat_image = self.image

##         self.create_plugin_options()
    
    def extra_setup_ui(self):
        # set up cmap options
        self.cmap_box.insertItems(0, cmaps)
        self.cmap_box.setCurrentIndex(cmaps.index('gray'))

        # set up interp options
        self.interp_box.insertItems(0, interpolations)
        self.interp_box.setCurrentIndex(interpolations.index('nearest'))

        # connect menu items
        self.actionLoad_MR_File.triggered.connect(self.on_load_mr)
        self.actionUnload_Overlay.triggered.connect(self.remove_overlay)

        # connect image space buttons
        self.worldspace_button.toggled.connect(self.change_to_world)
        self.voxspace_button.toggled.connect(self.change_to_vox)
        
        # connect variants of position states
        QtCore.QObject.connect(self.ortho_figs_widget,
                               QtCore.SIGNAL('xyz_state(int,int,int)'),
                               self.xyz_position_watcher)
        QtCore.QObject.connect(self.ortho_figs_widget,
                               QtCore.SIGNAL('xyz_state(int,int)'),
                               self.xyz_position_watcher)
        QtCore.QObject.connect(self.ortho_figs_widget,
                               QtCore.SIGNAL('xyz_state(int)'),
                               self.xyz_position_watcher)

        # connect the plugin launching event to notify the mayavi widget
##         # why not this way?? because of the object type??
##         QtCore.QObject.connect(self,
##                                QtCore.SIGNAL('plugin_launched(object)'),
##                                self._update_mayavi_viewer_panel)
        self.plugin_launched.connect(self._update_mayavi_viewer_panel)
        

    def create_mayavi_window(self):
        mayavi_widget = mayavi_widgets.MayaviWidget(main_ref=self)
        mayavi_widget.show()
        mayavi_widget.activateWindow()
        self.mwidget_toggle = mayavi_widget.toggle_view_action()
        self.menuView.addAction(self.mwidget_toggle)
        r = mayavi_widget.geometry()
        mayavi_widget.setGeometry(900,100,r.width(),r.height())
        self.mwidget_toggle.setChecked(False)
        self.mayavi_widget = mayavi_widget

    def _update_mayavi_viewer_panel(self, plugin):
        self.mayavi_widget.add_toolbar(plugin.func_man)

    @with_attribute('_image_loaded')
    def _update_plugin_params(self):
        # these are the in-place arguments for any plugin
        loc_methods = (self.ortho_figs_widget.update_location, )
        image_methods = (self.triggered_overlay_update, )
        im_props_methods = (self.change_overlay_props, )
        bbox = self.image.bbox
        self._plugin_args = (loc_methods, image_methods,
                             im_props_methods, bbox)

        # here are some keyword arguments
        self._plugin_kwargs['external_loc'] = self.ortho_figs_widget.xyz_state[float,float,float]
        self._plugin_kwargs['main_ref'] = self
        
    
    ########## ACTION/MENU HANDLERS ##########
    def on_load_mr(self, bool):
        fname = browse_files(self, dialog='Select Image File',
                             wildcard='Images (*.nii *.nii.gz *.hdr *.img)')
        if fname:
            self.update_image(fname)

    ########## IMAGE DATA UPDATES ##########
    def update_image(self, image, mode='world'):
        try:
            s_img = load_resampled_slicer(image, fliplr=self._fliplr)
        except ValueError:
            try:
                s_img = load_sampled_slicer(image, fliplr=self._fliplr)
            except ValueError:
                self.image = None
                self._image_loaded = False
                raise
        self.image = s_img
        self._image_loaded = True
        # need to update:
        # slider ranges
        # plugin params
        limits = self.image.bbox
        self.update_ranges(limits)
        self._update_plugin_params()
        planes = self.image.cut_image((0,0,0))
        interp = str(self.interp_box.currentText())
        cmap = cm.cmap_d[str(self.cmap_box.currentText())]
        self.ortho_figs_widget.initialize_plots(planes, (0,0,0), limits,
                                                interpolation=interp,
                                                cmap=cmap)
        if hasattr(self, 'mayavi_widget'):
            self.mayavi_widget.mr_vis.anat_image = self.image

##     @QtCore.pyqtSlot(QtCore.QObject, float, float, float)
    def triggered_overlay_update(self, func_man):
        self._overlay_thresholding = overlay_thresholding_function(
            func_man.threshold, positive=False
            )
        pdict = make_mpl_image_properties(func_man)
        self.update_overlay_slices(func_man.overlay, **pdict)

    def change_overlay_props(self, func_man):
        pdict = make_mpl_image_properties(func_man)
        self.ortho_figs_widget.set_over_props(**pdict)

    @with_attribute('_image_loaded')
    def update_overlay_slices(self, overlay, **kwargs):
        main_limits = self.image.bbox
        try:
            o_img = load_sampled_slicer(overlay, bbox=main_limits)
        except ValueError:
            try:
                o_img = load_resampled_slicer(overlay, bbox=main_limits)
            except ValueError:
                self._overlay_active = False
                self.over_img = None
                raise
        loc = self.ortho_figs_widget.active_voxel
##         fx = self._overlay_thresholding
##         planes = [np.ma.masked_where(fx(x), x, copy=False)
##                   for x in o_img.cut_image(loc)]
        planes = o_img.cut_image(loc)
        self.over_img = o_img
        self._overlay_active = True
        limits = self.over_img.bbox
        self.ortho_figs_widget.initialize_overlay_plots(
            planes, limits, **kwargs
            )
        self.check_max_extents()

    @with_attribute('_overlay_active')
    def remove_overlay(self, bool):
        print 'unloading MR overlays'
        self.ortho_figs_widget.unload_overlay_plots(draw=True)
        del self.over_img
        self._overlay_active = False
        if hasattr(self, 'timefreqwin'):
            self.timefreqwin.deactivate(strip_overlay=True)
        if hasattr(self, 'overlay_win'):
            self.overlay_win.deactivate(strip_overlay=True)

    def change_to_world(self, active):
        if active:
            self.change_image_space('world')
    def change_to_vox(self, active):
        if active:
            self.change_image_space('voxel')
        
    @with_attribute('_image_loaded')
    def change_image_space(self, mode):
##         self.image.switch_mode(mode)
        print 'this is broken'
        return
        self.update_image(self.image)
        if self._overlay_active:
            self.over_img.update_target_space(self.image)
            self.update_overlay_slices(self.over_img)
            
        # NEED TO TEST THIS
        #self.check_max_extents()
        limits = self.image.bbox
        self.update_ranges(limits)

    ########## FIGURE/PLOTTING UPDATES ##########
    @with_attribute('_image_loaded')
    def check_max_extents(self):
        limits = self.image.bbox
        self.update_ranges(limits)
        return limits

    def cut_to_location(self, loc):
        pass

##     def _new_vox(self, loc):
##         vsize = np.asarray(self.image.grid_spacing)
##         loc = np.asarray(loc)
##         dist = np.abs(loc - self.__saved_loc)
##         print 'main loc:', loc, self.__saved_loc, dist, vsize        
##         if (np.abs(loc - self.__saved_loc) > vsize).any():
##             self.__saved_loc = loc
##             return True
##         return False
    
##     def _new_overlay_slices(self, loc):
##         # could implement caching or movement thresholding someday
##         if not self._overlay_active:
##             return        
## ##         vsize = np.asarray(self.over_img.grid_spacing)
## ##         loc = np.asarray(loc)
## ##         dist = np.abs(loc-self.__saved_over_loc)
## ##         print 'overlay loc:', loc, self.__saved_over_loc, dist, vsize
## ##         if (dist > vsize).any():
## ##             self.__saved_over_loc = loc
## ##             return True
## ##         return False
    
    @QtCore.pyqtSlot(int, int, int)
    @QtCore.pyqtSlot(int, int)
    @QtCore.pyqtSlot(int)
    def xyz_position_watcher(self, *args):
        axes = args
        xyz_loc = self.ortho_figs_widget.active_voxel
##         if self._new_vox(xyz_loc):
        self.update_fig_data(xyz_loc, axes=axes)

    @with_attribute('_image_loaded')
    def update_fig_data(self, xyz_loc, axes=(SAG, COR, AXI)):
        planes = self.image.cut_image(xyz_loc, axes=axes)
        if self._overlay_active: # and self._new_overlay_vox(xyz_loc):
##             fx = self._overlay_thresholding
##             o_planes = [np.ma.masked_where(fx(x), x, copy=False)
##                         for x in self.over_img.cut_image(xyz_loc, axes=axes)]
            o_planes = self.over_img.cut_image(xyz_loc, axes=axes)
            planes = zip(planes, o_planes)
            self.ortho_figs_widget.update_plot_data(
                planes, fig_labels=axes
                )
        else:
            self.ortho_figs_widget.update_main_plot_data(
                planes, fig_labels=axes
                )

    @with_attribute('_image_loaded')
    def update_ranges(self, limits):
        sliders = [self.sag_slider, self.cor_slider, self.axi_slider]
        spinners = [self.sag_spinner, self.cor_spinner, self.axi_spinner]
        for slider, spinner, lim in zip(sliders, spinners, limits):
            slider.setMinimum(lim[0]); slider.setMaximum(lim[1])
            slider.setSliderPosition(0)
            spinner.setRange(lim[0], lim[1])
            spinner.setSingleStep(1)

    def _do_rand_data(self):
        rand_data = [np.random.randn(10,10) for x in [0,1,2]]
        interp = str(self.interp_box.currentText())
        cmap = cm.cmap_d[str(self.cmap_box.currentText())]
        plot_limits = [(-50,50)]*3
        self.ortho_figs_widget.initialize_plots(rand_data, (0,0,0),
                                                plot_limits,
                                                interpolation=interp,
                                                cmap=cmap)
        self.update_ranges(plot_limits)


def ortho_viewer(image=None, fliplr=False):
    win = OrthoViewer(image=image, fliplr=fliplr)
    return win

def view_mr(mr_image, fliplr=False):
    return ortho_viewer(image=mr_image, fliplr=fliplr)


if __name__=='__main__':
    from PyQt4 import QtGui
    
    if QtGui.QApplication.startingUp():
        app = QtGui.QApplication(sys.argv)
    else:
        app = QtGui.QApplication.instance()    
    if len(sys.argv) > 1:
        image = sys.argv[1]
    else:
        image = None
    win = OrthoViewer(image=image)
    if not image:
        win._do_rand_data()
    win.show()
    sys.exit(app.exec_())
