from PyQt4 import QtGui, QtCore
from xipy.volume_utils import limits_to_extents
from xipy.utils import with_attribute
from xipy.slicing import SAG, COR, AXI, transverse_plane_lookup
from xipy.vis.qt4_widgets.auxiliary_window import TopLevelAuxiliaryWindow
from xipy.vis import BLITTING
import xipy.vis.single_slice_plot as ssp

import numpy as np

import matplotlib as mpl
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
## import matplotlib.cm as cm
import xipy.vis.color_mapping as cm

class MplQT4OrthoSlicesWidget(TopLevelAuxiliaryWindow):
    """This class is a Qt4 panel displaying three SliceFigures which cut
    into the standard orthogonal planes of a volumetric medical image. The
    voxel of intersection is manipulated by moving crosshairs belonging to
    each SliceFigure object, and whose positions are linked up by this class.

    This class can load/refresh base images and overlay images into the three
    SliceFigures, and emits PyQt4 signals when the intersecting voxel changes.
    """

    
    # there are 3 forms of updating the voxel position
    # 1) moving the axis of any single image dimension
    #    -- This requires an update to the plot whose plane is sliced by
    #       that axis, as well as the update of any object following the axis
    #
    # 2) moving the crosshairs in one plot
    #    -- This requires an update to the location status in any object
    #       following the axes of the transverse plane of the clicked plot,
    #       as well as updating the plot data that is sliced by the two axes
    #
    # 3) moving all three components of voxel of intersection
    #    -- This requires replotting all three planes, and signaling a
    #       change in xyz status.
    
    xyz_state = QtCore.pyqtSignal((int,),
                                  (int, int),
                                  (int, int, int),
                                  (float, float, float))
    x_state = QtCore.pyqtSignal((int,), (float,), (float, float, float))
    y_state = QtCore.pyqtSignal((int,), (float,), (float, float, float))
    z_state = QtCore.pyqtSignal((int,), (float,), (float, float, float))
    # this will allow a "read-only" ortho_figs.active_voxel inquiry
    active_voxel = property(lambda x: x._xyz_position, None)
    # this is to keep track of zooming to the full FOV
    _full_fov_lims = [(-1,1), (-1,1), (-1,1)]
    # keep track of when to re-save the background plots after a resize
    _was_resized = False
    
    def __init__(self, parent=None, **kwargs):
        figsize = kwargs.pop('figsize', (3,3))
        dpi = kwargs.pop('dpi', 100)
        if 'limits' in kwargs:
            self._full_fov_lims = kwargs.pop('limits')
        extents = limits_to_extents(self._full_fov_lims)
        QtGui.QWidget.__init__(self, parent)
        # set up the Sag, Cor, Axi SliceFigures
        self.horizontalLayout = QtGui.QHBoxLayout(self)
        self.horizontalLayout.setObjectName("FigureLayout")


        # set axial figure
        fig = Figure(figsize=figsize, dpi=dpi)
        fig.canvas = Canvas(fig)
        Canvas.setSizePolicy(fig.canvas, QtGui.QSizePolicy.Expanding,
                             QtGui.QSizePolicy.Expanding)
        Canvas.updateGeometry(fig.canvas)
        fig.canvas.setParent(self)
        self.axi_fig = ssp.SliceFigure(fig, extents[AXI], blit=BLITTING)
        self.axi_fig.ax.set_xlabel('left to right', fontsize=8)
        self.axi_fig.ax.set_ylabel('posterior to anterior', fontsize=8)
        self.horizontalLayout.addWidget(fig.canvas)
        # set coronal figure
        fig = Figure(figsize=figsize, dpi=dpi)
        fig.canvas = Canvas(fig)
        Canvas.setSizePolicy(fig.canvas, QtGui.QSizePolicy.Expanding,
                             QtGui.QSizePolicy.Expanding)
        Canvas.updateGeometry(fig.canvas)
        fig.canvas.setParent(self)
        self.cor_fig = ssp.SliceFigure(fig, extents[COR], blit=BLITTING)
        self.cor_fig.ax.set_xlabel('left to right', fontsize=8)
        self.cor_fig.ax.set_ylabel('inferior to superior', fontsize=8)
        self.horizontalLayout.addWidget(fig.canvas)        
        # set sagittal figure
        fig = Figure(figsize=figsize, dpi=dpi)
        fig.canvas = Canvas(fig)
        Canvas.setSizePolicy(fig.canvas, QtGui.QSizePolicy.Expanding,
                             QtGui.QSizePolicy.Expanding)
        Canvas.updateGeometry(fig.canvas)
        fig.canvas.setParent(self)
        self.sag_fig = ssp.SliceFigure(fig, extents[SAG], blit=BLITTING)
        self.sag_fig.ax.set_xlabel('posterior to anterior', fontsize=8)
        self.sag_fig.ax.set_ylabel('inferior to superior', fontsize=8)        
        self.horizontalLayout.addWidget(fig.canvas)
        
        # put down figures in x, y, z order
        self.figs = [self.sag_fig, self.cor_fig, self.axi_fig]
        self.canvas_lookup = dict( ((f.canvas, f) for f in self.figs) )
        self._mouse_dragging = False
        # this should be something like a signal too, which can emit status
        self._xyz_position = [0,0,0]
        self._connect_events()

        self.setParent(parent)
        QtGui.QWidget.setSizePolicy(self, QtGui.QSizePolicy.Expanding,
                                    QtGui.QSizePolicy.Expanding)
        QtGui.QWidget.updateGeometry(self)
        self._main_plotted = False
        self.main_plots = []
        self._over_plotted = False
        self.over_plots = []

    def _connect_events(self):
        for f in self.figs:
            f.canvas.mpl_connect('button_press_event',
                                 self.xhair_mousedn)
            f.canvas.mpl_connect('button_release_event',
                                 self.xhair_mouseup)
            f.canvas.mpl_connect('motion_notify_event',
                                 self.xhair_motion)

    def update_plot_data(self, data_list, fig_labels=[]):
        """Update the data in each plot in each figure.
        Parameters
        ----------
        data_list : list
            a nested list of ndarrays, one for each fig indicated in fig_labels
            (or length-3, if fig_labels is empty). Each nested list will contain
            as many arrays as the corresponding figure has images.
        fig_labels : list
            the labels (SAG, COR, and/or AXI) of the figures to update
        """
        if fig_labels:
            figs = [self.figs[idx] for idx in fig_labels]
        else:
            figs = self.figs
        print 'updating all data at figs:', fig_labels
        for fig, data in zip(figs, data_list):
            fig.set_data(data)

    @with_attribute('main_plots')
    def update_main_plot_data(self, data_list, fig_labels=[]):
        """Update the data in each main plot.
        Parameters
        ----------
        data_list : list
            a list of ndarrays, one for each fig indicated in fig_labels
            (or length-3, if fig_labels is empty)
        fig_labels : list
            the labels (SAG, COR, and/or AXI) of the figures to update
        """
        if fig_labels:
            imgs = [self.main_plots[idx] for idx in fig_labels]
        else:
            imgs = self.main_plots
        print 'updating main data at figs:', fig_labels
        for img, data in zip(imgs, data_list):
            img.set_data(data)

    @with_attribute('over_plots')
    def update_over_data(self, data_list, fig_labels=[]):
        """Update the data in each overlay plot.
        Parameters
        ----------
        data_list : list
            a list of ndarrays, one for each fig indicated in fig_labels
            (or length-3, if fig_labels is empty)
        fig_labels : list
            the labels (SAG, COR, and/or AXI) of the figures to update
        """
        if fig_labels:
            imgs = [self.over_plots[idx] for idx in fig_labels]
        else:
            imgs = self.over_plots
        print 'updating overlay data at figs:', fig_labels
        for img, data in zip(imgs, data_list):
            img.set_data(data)

    def initialize_plots(self, data_list, loc, ax_lims, **img_kw):
        """Initialize the main plots of each orthogonal plane with the
        images given in data_list.

        Parameters
        ----------
        data_list : list-like
            a list of images for the sagittal, coronal, and axial planes
        extents : list-like
            a list of extent pairs, ie: [(xmin,xmax), (ymin,ymax), (zmin,zmax)]
        loc : list-like
            the (x,y,z) position at the intersection of the planes
        img_kw : optional
            any AxesImage image properties
        """
        
        self.unload_main_plots(draw=False)
        self.unload_overlay_plots(draw=False)
        # these are the new extents for each plot (sag, cor, axial)
        self._full_fov_lims = ax_lims
        plot_extents = limits_to_extents(ax_lims)
        x,y,z = loc
        self._xyz_position[:] = x,y,z
        self.emit_states(SAG,COR,AXI)
        fig_locs = [ (x,y), (y,z), (x,z) ]
        for n, fig in enumerate(self.figs):
            fig.set_limits(plot_extents[n])
        self.main_plots = [f.spawn_image(p, loc=l, extent=e, **img_kw)
                           for  f, p, l, e in zip(self.figs, data_list,
                                                  fig_locs, plot_extents)]
        self.draw()
        
    def initialize_overlay_plots(self, data_list, ax_lims, **img_kw):
        self.unload_overlay_plots(draw=False)
        # these are the new extents for each plot (sag, cor, axial)
        plot_extents = limits_to_extents(ax_lims)
        self.over_plots = [f.spawn_image(p, extent=e, **img_kw)
                           for  f, p, e in zip(self.figs, data_list,
                                               plot_extents)]
    @QtCore.pyqtSlot()
    def unload_main_plots(self, draw=True):
        if self.main_plots:
            for fig, im_plot in zip(self.figs, self.main_plots):
                fig.pop_image(im_plot)
        if draw:
            self.draw()
        
    @QtCore.pyqtSlot()
    def unload_overlay_plots(self, draw=True):
        if self.over_plots:
            for fig, im_plot in zip(self.figs, self.over_plots):
                fig.pop_image(im_plot)
        if draw:
            self.draw()

    def xhair_mousedn(self, event):
        self._mouse_dragging = event.inaxes is not None
        #self.coord_event_handling(event)
    def xhair_motion(self, event):
        if not self._mouse_dragging:
            return
        self.coord_event_handling(event)
    def xhair_mouseup(self, event):
        if not self._mouse_dragging:
            return
        self.coord_event_handling(event)
        self._mouse_dragging = False

    @with_attribute('main_plots')
    def coord_event_handling(self, event, emitting=True):
        # 1) get two coords from the event canvas, and 3rd from the fixed axis
        # 2) determine which are the transverse images
        if event.xdata == None or event.ydata == None:
            # must have been a stray hit
            return
        if self._was_resized:
            for fig in self.figs:
                fig.draw(save=True)
            print 'saved plots'
            self._was_resized = False
        fig = self.canvas_lookup[event.canvas]
        fig_limits = fig.xlim + fig.ylim
        # be safe with the coordinates
        pu = np.clip(event.xdata, fig_limits[0], fig_limits[1])
        pv = np.clip(event.ydata, fig_limits[2], fig_limits[3])
        fig_idx = self.figs.index(fig)
        # these will be the indices in an xyz list that correspond to pu,pv,
        # and also to the indices of the figures/images to update
        ui, vi = transverse_plane_lookup(fig_idx)
        xyz = self._xyz_position[:]
        xyz[ui] = pu
        xyz[vi] = pv
        self._xyz_position = xyz[:]

        self._update_crosshairs()

        # THE REST SHOULD BE TRIGGERED ON SOME KIND OF XYZ SIGNAL
        if emitting:
            self.emit_states(ui, vi)
        
    def _update_crosshairs(self):
        xyz = self.active_voxel
        # now move the crosshairs on each plot
        self.figs[SAG].move_crosshairs(xyz[COR], xyz[AXI]) # sag gets yz
        self.figs[COR].move_crosshairs(xyz[SAG], xyz[AXI]) # cor get xz
        self.figs[AXI].move_crosshairs(xyz[SAG], xyz[COR]) # ax gets xy
        

    # for the SliceFigures, want to access:
    # PROPERTIES:
    # -- xlim, ylim (not wrapped here)
    # METHODS:
    # -- set_limits(limits)
    # -- move_crosshairs(x,y)
    # -- toggle_crosshairs_visible(mode=True)
    # -- spawn_image(sl_data, **img_kws)
    # -- pop_image(s_img)
    # -- set_data(slice_list)
    # -- draw(when=not now, save=False) (maybe?)
    #

    def set_limits(self, limits):
        """Set the axes limits on each SliceFigure.

        Paramters
        ---------
        limits : iterable
            the min/max limits (in mm units) for each SliceFigure in
            sagittal, coronal, axial order
        """
        extents = limits_to_extents(limits)
        for fig, lim in zip(self.figs, extents):
            fig.set_limits(lim)

    @QtCore.pyqtSlot(int)
    def zoom_slices(self, zoom_idx):
        zooms = [-1, 160, 80, 40, 20, 10]
        z = zooms[zoom_idx]
        if z > 0:
            xyz = np.array(self._xyz_position, 'd')
            dist = np.array( [z]*3, 'd')/2.0
            lims = np.array( (xyz - dist, xyz + dist) ).transpose().tolist()
        else:
            lims = self._full_fov_lims
        self.set_limits(lims)
        

    @QtCore.pyqtSlot(float, float, float)
    def update_location(self, *xyz_loc):
        self._xyz_position[:] = xyz_loc
        self._update_crosshairs()
        self.emit_states(SAG,COR,AXI)

    @QtCore.pyqtSlot(int)
    @QtCore.pyqtSlot(float)    
    def move_sag_position(self, x):
        _,y,z = self._xyz_position
        # change in the sagittal axis is treated as coming from the axial plane
        fig = self.figs[AXI]
        # this method should trigger an xyz state change
        self.coord_event_handling(FakeMPLEvent(fig.canvas,
                                               fig.ax,
                                               x, y), emitting=False)
        self.emit_states(SAG)

    @QtCore.pyqtSlot(int)
    @QtCore.pyqtSlot(float)    
    def move_cor_position(self, y):
        x,_,z = self._xyz_position
        # change in the coronal axis is treated as coming from the sag. plane
        fig = self.figs[SAG]
        # this method should trigger an xyz state change
        self.coord_event_handling(FakeMPLEvent(fig.canvas,
                                               fig.ax,
                                               y, z), emitting=False)
        self.emit_states(COR)
    
    @QtCore.pyqtSlot(int)
    @QtCore.pyqtSlot(float)    
    def move_axi_position(self, z):
        x,y,_ = self._xyz_position
        # change in the axial axis is treated as coming from the coronal plane
        fig = self.figs[COR]
        # this method should trigger an xyz state change
        self.coord_event_handling(FakeMPLEvent(fig.canvas,
                                               fig.ax,
                                               x, z), emitting=False)
        self.emit_states(AXI)

    @QtCore.pyqtSlot(bool, name='toggleCrosshairsVisible')
    def toggle_crosshairs_visible(self, mode):
        for fig in self.figs:
            fig.toggle_crosshairs_visible(mode=mode)

    @QtCore.pyqtSlot()
    def draw(self):
        for fig in self.figs:
            fig.draw()

    # for the SliceImage, want to access:
    # PROPERTIES:
    # -- cmap, interpolation, norm, alpha, extent

    # set any or all properties at once
    @with_attribute('main_plots')
    def set_main_props(self, **props):
        for p in self.main_plots:
            p.set_properties(**props)
    @with_attribute('over_plots')
    def set_over_props(self, **props):
        for p in self.over_plots:
            p.set_properties(**props)

    # ----- MAIN IMAGE PROPS/METHS -----
    @QtCore.pyqtSlot(str)
    @with_attribute('main_plots')
    def set_cmap(self, cmap):
        mpl_cmap = cm.cmap_d[str(cmap)]
        for p in self.main_plots:
            p.cmap = mpl_cmap
    @QtCore.pyqtSlot(str)
    @with_attribute('main_plots')
    def set_interp(self, interp):
        for p in self.main_plots:
            p.interp = str(interp)
    #@QtCore.pyqtSlot()
    @with_attribute('main_plots')
    def set_norm(self, norm):
        for p in self.main_plots:
            p.norm = norm
    @QtCore.pyqtSlot(int)
    @with_attribute('main_plots')
    def set_alpha(self, alpha):
        if type(alpha)==int and alpha>1:
            alpha /= 100.
        for p in self.main_plots:
            p.alpha = alpha
    #@QtCore.pyqtSlot()
    @with_attribute('main_plots')
    def set_extent(self, extent):
        for p in self.main_plots:
            p.extent = extent
    # ----- OVER IMAGE PROPS -----
    @QtCore.pyqtSlot(str)
    @with_attribute('over_plots')
    def set_cmapO(self, cmap):
        mpl_cmap = cm.cmap_d[str(cmap)]
        for p in self.over_plots:
            p.cmap = mpl_cmap
    @QtCore.pyqtSlot(str)
    @with_attribute('over_plots')
    def set_interpO(self, interp):
        for p in self.over_plots:
            p.interp = str(interp)
    #@QtCore.pyqtSlot()
    @with_attribute('over_plots')
    def set_normO(self, norm):
        for p in self.over_plots:
            p.norm = norm
    @QtCore.pyqtSlot(int)
    @with_attribute('over_plots')
    def set_alphaO(self, alpha):
        if type(alpha)==int and alpha>1:
            alpha /= 100.
        for p in self.over_plots:
            p.alpha = alpha
    #@QtCore.pyqtSlot()
    @with_attribute('over_plots')
    def set_extentO(self, extent):
        for p in self.over_plots:
            p.extent = extent

    def emit_states(self, *indices):
        # this is either called as emit_states(i) or emit_states(i,j)
        # by either a single axis update process, or a two-axes update process.
        # Technically, it could be called as (i,j,k), and then one of the
        # overloaded variants of xyz_state.emit will fail
##         print 'emitting states:', indices
        states = [self.x_state, self.y_state, self.z_state]
        for i in indices:
            s = states[i]
            s[int].emit(int(round(self._xyz_position[i])))
            s[float].emit(self._xyz_position[i])
        variant = (int,) * len(indices)
        try:
            self.xyz_state[variant].emit(*indices)
        except:
            pass
        self.xyz_state[(float,)*3].emit(*self.active_voxel)
        
    def sizeHint(self):
        w, h = self.figs[0].canvas.get_width_height()
        return QtCore.QSize(3*(w+10), 1.1*h)

    def minimumSizeHint(self):
        return QtCore.QSize(3*(100+10), 1.1*100)

    def resizeEvent(self, event):
        self._was_resized = True
        QtGui.QWidget.resizeEvent(self, event)


#===============================================================================
#   Example
#===============================================================================
if __name__ == '__main__':
    import sys
    from PyQt4 import QtGui, QtCore
    import numpy as np
    
    class ApplicationWindow(QtGui.QMainWindow):
        n_updates = 0

##         data = np.random.randint(low=0, high=255,
##                                  size=(20,160,160,4)).astype(np.uint8)
        data = np.random.randn(20,160,160)
        def __init__(self):
            QtGui.QMainWindow.__init__(self)
            self.ortho_widget = MplQT4OrthoSlicesWidget(parent=self)
            self.setCentralWidget(self.ortho_widget)
            self.plot()
            # connect variants of position states
            QtCore.QObject.connect(self.ortho_widget,
                                   QtCore.SIGNAL("xyz_state(int,int,int)"),
                                   self.xyz_position_watcher)
            QtCore.QObject.connect(self.ortho_widget,
                                   QtCore.SIGNAL("xyz_state(int,int)"),
                                   self.xyz_position_watcher)
            QtCore.QObject.connect(self.ortho_widget,
                                   QtCore.SIGNAL("xyz_state(int)"),
                                   self.xyz_position_watcher)


        def xyz_position_watcher(self, *args):
            axes = args
            xyz_loc = self.ortho_widget.active_voxel
            # just simply replot for now
            self.plot(axes=axes)
        
        def plot(self, axes=[]):
            if not axes:
                axes = [0,1,2]
            rand_indices = np.random.randint(low=0, high=19, size=len(axes))
            data = [ self.data[ri] for ri in rand_indices ]
            if not self.ortho_widget.main_plots:
                self.ortho_widget.initialize_plots(data,
                                                   (0,0,0),
                                                   [(-50,50)]*3,
                                                   interpolation='nearest')
                                                   
            else:
                self.ortho_widget.update_plot_data(data, fig_labels=axes)
                self.n_updates += 1


    if QtGui.QApplication.startingUp():
        app = QtGui.QApplication(sys.argv)
    else:
        app = QtGui.QApplication.instance()
    win = ApplicationWindow()
    win.show()
    sys.exit(app.exec_())
##     import cProfile, pstats
##     cProfile.runctx('app.exec_()', globals(), locals(), 'orthoslices.prof')
##     s = pstats.Stats('orthoslices.prof')
##     print "NUMBER OF UPDATES:", win.n_updates
##     s.strip_dirs().sort_stats('cumulative').print_stats()
    
        
