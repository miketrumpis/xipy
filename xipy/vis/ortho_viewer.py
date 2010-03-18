import os, sys
# try loading this generated class from the uic module
## from pyqt4.pyqt4_viewer import Ui_MainWindow
from PyQt4 import uic
design_path = os.path.join(os.path.split(__file__)[0], 'qt4_widgets/designer_layouts/ortho_viewer_layout.ui')
ui_layout_class, base_class = uic.loadUiType(design_path)

from PyQt4 import QtCore, QtGui
import matplotlib as mpl
import matplotlib.cm as cm

import numpy as np

from xipy.slicing import SAG, COR, AXI, load_resampled_slicer, \
     load_sampled_slicer
from xipy.utils import with_attribute
from xipy.vis import qt4_widgets as qw
from xipy.vis import mayavi_widgets
from xipy.overlay import overlay_thresholding_function
from xipy.overlay.plugins import all_registered_plugins

interpolations = ['nearest', 'bilinear', 'sinc']
cmaps = cm.cmap_d.keys()
cmaps.sort()

class OrthoViewer(QtGui.QMainWindow, ui_layout_class):

    __active_tools = []
    
    def __init__(self, image=None, fliplr=False, parent=None):
##         QtGui.QMainWindow.__init__(self)
        super(OrthoViewer, self).__init__(parent)

        # if ANALYZE Images are known to be presented in right-handed
        # orientation, then you can forcibly ignore the sign of
        # R[0,0], in the index to world transform
        self._fliplr = fliplr
        
        # kick in the Qt Designer generated layout and signal/slot connections
        self.setupUi(self)
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

        self.create_plugin_options()
    
    def extra_setup_ui(self):

        # set up cmap options
        self.cmap_box.insertItems(0, cmaps)
        self.cmap_box.setCurrentIndex(cmaps.index('gray'))
        #self.img_cmap = property(fget=self.cmap_box.currentText)
        self.o_cmap_box.insertItems(0, cmaps)
        self.o_cmap_box.setCurrentIndex(cmaps.index('jet'))
        #self.overlay_cmap = property(fget=self.o_cmap_box.currentText)
        # set up interp options
        self.interp_box.insertItems(0, interpolations)
        self.interp_box.setCurrentIndex(interpolations.index('nearest'))
        #self.img_interp = property(fget=self.interp_box.currentText)
        self.o_interp_box.insertItems(0, interpolations)
        self.o_interp_box.setCurrentIndex(interpolations.index('bilinear'))
        #self.overlay_interp = property(fget=self.o_interp_box.currentText)

        # connect menu items
        self.actionLoad_MR_File.triggered.connect(self.on_load_mr)
        self.actionUnload_Overlay.triggered.connect(self.remove_overlay)

        # connect image space buttons
        self.worldspace_button.toggled.connect(self.change_to_world)
        self.voxspace_button.toggled.connect(self.change_to_vox)
        
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

    def create_plugin_options(self):
        pitems = all_registered_plugins()
        for pname, pclass in pitems:
            action = QtGui.QAction(self)
            action.setObjectName('action'+pname.replace(' ', '_'))
            callback = lambda b: self._launch_plugin_tool(pclass)
            action.triggered.connect(callback)
            self.menuTools.addAction(action)
            action.setText(QtGui.QApplication.translate(
                'MainWindow', pname, None, QtGui.QApplication.UnicodeUTF8
                ))
    @with_attribute('_image_loaded')
    def _launch_plugin_tool(self, pclass):
        loc_methods = (self.ortho_figs_widget.update_location, )
        image_methods = (self.triggered_overlay_update, )
        active_tool = filter(lambda x: type(x)==pclass, self.__active_tools)
        if active_tool:
            print 'already launched this tool, so make it active (eventually)'
            tool = active_tool[0]
            print tool
            return
        print 'would launch class', pclass
        # lauch this in top-level mode
        tool = pclass(loc_methods, image_methods,
                      self.image.bbox, main_ref=self)
        tool.show()
        tool.activateWindow()
        toggle = tool.toggle_view_action()
        self.menuView.addAction(toggle)
    
    
##     def _launch_tf_window(self, beam=None):
##         if hasattr(self, 'timefreqwin'):
##             if beam:
##                 self.timefreqwin.beam_manager.update_beam(beam)
##                 self.tfwidget_toggle.setChecked(True)
##             return self.timefreqwin.beam_manager
##         # Time Frequency Plane and Beam Manager widget
##         image_connections = (self.triggered_overlay_update,
##                                 #self.mayavi_widget.tf_update
##                              )
##         print image_connections
##         loc_connections = (self.ortho_figs_widget.update_location,)
##         tf_connections = () # currently no tf signal callbacks
##         self.timefreqwin = qw.MplQT4TimeFreqWindow(
##             tf_connections,
##             image_connections,
##             loc_connections,
##             main_ref=self,
##             beam=beam
##             )
##         QtCore.QObject.connect(self.o_cmap_box,
##                                QtCore.SIGNAL('currentIndexChanged(QString)'),
##                                self.timefreqwin.setCmap)
##         QtCore.QObject.connect(self.o_alpha_slider,
##                                QtCore.SIGNAL('sliderMoved(int)'),
##                                self.timefreqwin.setAlpha)
## ##         QtCore.QObject.connect(self.animate_tf_button,
## ##                                QtCore.SIGNAL('clicked()'),
## ##                                self.initiate_tf_animate)
## ##         self.timefreqwin.tf_point.connect(self.triggered_overlay_update)
##         r = self.timefreqwin.geometry()
##         w = r.width(); h = r.height()
##         self.timefreqwin.setGeometry(50,100,w,h)

##         self.timefreqwin.show()
##         self.timefreqwin.activateWindow()
##         self.tfwidget_toggle = self.timefreqwin.toggle_view_action()
##         self.menuView.addAction(self.tfwidget_toggle)
##         return self.timefreqwin.beam_manager

##     def _launch_overlay_control_window(self, overlay=None):
##         if hasattr(self, 'overlay_win'):
##             if overlay:
##                 self.overlay_win.overlay_manager.update_overlay(overlay)
##                 self.owin_toggle.setChecked(True)
##             return self.overlay_win.overlay_manager
##         ovr_con = (self.triggered_overlay_update,)
## ##         if hasattr(self, 'mayavi_widget'):
## ##             ovr_con = ovr_con + (self.mayavi_widget.overlay_update,)
##         loc_con = (self.ortho_figs_widget.update_location,)
##         self.overlay_win = ImageOverlayControl(loc_con, ovr_con,
##                                                overlay=overlay,
##                                                main_ref=self)
##         QtCore.QObject.connect(self.o_cmap_box,
##                                QtCore.SIGNAL('currentIndexChanged(QString)'),
##                                self.overlay_win.cbar.change_cmap)
##         cmap = self.o_cmap_box.currentText()
##         self.overlay_win.cbar.change_cmap(cmap)
##         r = self.overlay_win.geometry()
##         self.overlay_win.setGeometry(50,100,r.width(),r.height())
##         self.overlay_win.show()
##         self.overlay_win.activateWindow()
##         self.owin_toggle = self.overlay_win.toggle_view_action()
##         self.menuView.addAction(self.owin_toggle)
##         return self.overlay_win.overlay_manager
## ##         self.owin_toggle.setChecked(True)
        

    ########## ACTION/MENU HANDLERS ##########
    def on_load_mr(self, bool):
        fname = qw.browse_files(self, dialog='Select Image File',
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
        limits = self.image.bbox
        self.update_ranges(limits)
        planes = self.image.cut_image((0,0,0))
        interp = str(self.interp_box.currentText())
        cmap = cm.cmap_d[str(self.cmap_box.currentText())]
        self.ortho_figs_widget.initialize_plots(planes, (0,0,0), limits,
                                                interpolation=interp,
                                                cmap=cmap)
        if hasattr(self, 'mayavi_widget'):
            self.mayavi_widget.mr_vis.anat_image = self.image

##     @QtCore.pyqtSlot(QtCore.QObject, float, float, float)
    def triggered_overlay_update(self, *args):
        # query obj for the image and the norm (and probably the cmap in future)
        obj = args[0]
        self._overlay_thresholding = overlay_thresholding_function(
            obj.threshold, positive=False
            )
        self.update_overlay_slices(obj.overlay,
                                   norm=mpl.colors.normalize(*obj.norm))

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
        fx = self._overlay_thresholding
        planes = [np.ma.masked_where(fx(x), x, copy=False)
                  for x in o_img.cut_image(loc)]        
        self.over_img = o_img
        self._overlay_active = True
        limits = self.over_img.bbox
        interp = str(self.o_interp_box.currentText())
        cmap = cm.cmap_d[str(self.o_cmap_box.currentText())]
        norm = kwargs.get('norm', None)
        self.ortho_figs_widget.initialize_overlay_plots(planes, limits,
                                                        interpolation=interp,
                                                        cmap=cmap,
                                                        **kwargs)
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
            fx = self._overlay_thresholding
            o_planes = [np.ma.masked_where(fx(x), x, copy=False)
                        for x in self.over_img.cut_image(xyz_loc, axes=axes)]
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
