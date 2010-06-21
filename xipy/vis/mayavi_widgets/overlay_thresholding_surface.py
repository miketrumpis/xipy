# NumPy
import numpy as np

# NIPY
from nipy.core import api as ni_api

# Enthought library
from enthought.traits.api import HasTraits, Instance, on_trait_change, Array, \
     Bool, Range, Enum, Property, List, Tuple, DelegatesTo, TraitError
from enthought.traits.ui.api import View, Item, HGroup, VGroup, Group, \
     RangeEditor
from enthought.tvtk.api import tvtk
from enthought.mayavi.core.api import Source
from enthought.mayavi.sources.array_source import ArraySource
from enthought.mayavi.core.ui.api import MayaviScene, MlabSceneModel, \
     SceneEditor
from enthought.mayavi.modules.text import Text
from enthought.mayavi import mlab

# XIPY imports
from xipy.slicing.image_slicers import ResampledVolumeSlicer, \
     VolumeSlicerInterface
from xipy.overlay import OverlayInterface, ThresholdMap
from xipy.vis.qt4_widgets.auxiliary_window import TopLevelAuxiliaryWindow
from xipy.vis.mayavi_tools import ArraySourceRGBA, image_plane_widget_rgba
from xipy.vis.mayavi_tools import time_wrap as tw
from xipy.vis.mayavi_widgets import VisualComponent
from xipy.vis import rgba_blending
import xipy.vis.color_mapping as cm
import xipy.volume_utils as vu

class OverlayThresholdingSurfaceComponent(VisualComponent):
    """A class to take control of thresholding the overlay, and creating
    surfaces of unmasked regions
    """

    func_man = DelegatesTo('display')
    _func_thresh = DelegatesTo('display')
    func_scalars = DelegatesTo('display')
    notch_mode = Property
    thresh = Instance('enthought.mayavi.filters.threshold.Threshold')
    blender = DelegatesTo('display')

    # ----- This will eventually become this VisualComponent's UI widget -----
    show_tsurfs = DelegatesTo('display')
    # ------------------------------------------------------------------------

    @on_trait_change('show_tsurfs')
    def _show_thresh_surfaces(self):
        if not self.thresh:
            print 'no overlay thresholding available'
            self.trait_setq(show_tsurfs=False)
            return
        elif not hasattr(self, 'thresh_surf') or not self.thresh_surf:
            self.add_threshold_surf()
        self.thresh_surf.visible = self.show_tsurfs
        self.display.scene.render_window.render()
        
    def add_threshold_surf(self):
        if not self.thresh:
            return
        # BUT VMIN, VMAX IS NOW (0,255) SINCE WE'RE LOOKING AT INDICES
        mn, mx = self.func_man.norm
        colormap = self.func_man.cmap_option
        self.display._stop_scene()
        surf = mlab.pipeline.surface(
            self.surf_scalars, vmin=0, vmax=255,
            colormap=colormap, representation='wireframe',
            opacity=0.35, figure=self.display.scene.mayavi_scene
            )
        self.thresh_surf = surf
        self.display._start_scene()

    @on_trait_change('func_man')
    def _reset_all_arrays(self):
        self.func_scalars.children = []
        # flush previous arrays
        n_arr = self.func_scalars.image_data.point_data.number_of_arrays
        names = [self.func_scalars.image_data.point_data.get_array(i).name
                 for i in xrange(n_arr)]
        for n in names:
            self.func_scalars.image_data.point_data.remove_array(n)
        self.func_scalars.scalar_data = None
        self.thresh = None
        self.thresh_surf = None
        self.surf_scalars = None

##     @on_trait_change('func_man.overlay')
##     def _update_overlay_scalars(self):
##         overlay = self.func_man.overlay
##         if not overlay:
##             return
##         self.func_scalars.scalar_data = overlay.image_arr.transpose().copy()
##         self.func_scalars.image_data.point_data.get_array(0).name = 'overlay'
##         self.func_scalars.origin = overlay.coordmap.affine[:3,-1]
##         self.func_scalars.spacing = vu.voxel_size(overlay.coordmap.affine)
##         self.display._stop_scene()
##         self.func_scalars.update_image_data = True
##         self.display._start_scene()

##     @on_trait_change('blender.over')
    @on_trait_change('func_man.overlay_updated')
    def _update_overlay_scalars(self):
        print 'blender over changed'
##         overlay = self.func_man.overlay
        overlay = self.blender.over
        scalars = self.func_scalars
        if not overlay:
            return

##         s_arr = np.ma.filled(overlay.image_arr)
        # this is now the index array--scalars normalized between 0-255
        # MixedAlphaColormap.i_bad is the key for masking: look for that value
        s_arr = overlay.image_arr
        sctype = s_arr.dtype
        # this will be a negative mask, so threshold everything > 0.5
##         t_arr = np.ma.getmask(overlay.image_arr)
        i_bad = cm.MixedAlphaColormap.i_bad
        t_arr = np.zeros_like(s_arr)
        np.putmask(t_arr, s_arr==i_bad, 1)
        t_arr = np.ravel(t_arr)
        
##         if t_arr is np.ma.nomask:
##             t_arr = np.zeros(overlay.image_arr.shape, sctype).transpose().flatten()
##         else:
##             t_arr = t_arr.astype(sctype).transpose().flatten()
        

        scalars.scalar_data = s_arr
        scalars.scalar_name = 'overlay'
        scalars.origin = np.array(overlay.bbox)[:,0]
        scalars.spacing = overlay.grid_spacing
##         scalars.origin = overlay.coordmap.affine[:3,-1]
##         scalars.spacing = vu.voxel_size(overlay.coordmap.affine)
        scalars.update_image_data = True
        thresh_pts = scalars.image_data.point_data.get_array('threshold')
        if thresh_pts:
            thresh_pts.from_array(t_arr)
        else:
            n = scalars.image_data.point_data.add_array(t_arr)
            scalars.image_data.point_data.get_array(n).name = 'threshold'
            ts = mlab.pipeline.set_active_attribute(scalars,
                                                    point_scalars='threshold')
            self.thresh = mlab.pipeline.threshold(ts)
            self.thresh.filter_type = 'cells'
            self.surf_scalars = mlab.pipeline.set_active_attribute(
                self.thresh, point_scalars='overlay'
                )

        # try to threshold between (0, .5)..
        # if this fails, then let everything pass
        try:
            self.thresh.upper_threshold = 0.5
        except:
            pass
        try:
            self.thresh.lower_threshold = 0.0
        except:
            pass


##     def _get_notch_mode(self):
##         return self._func_thresh.thresh_mode=='mask between'

##     @on_trait_change('_func_thresh.map_scalars')
##     def _remap_threshold_scalars(self):
##         if not self._func_thresh.thresh_map_name:
##             print 'map_scalars changed, but threshold inactive'
##             return
##         if self.notch_mode:
##             self._map_threshold_mask()
##             return
##         stats_vol = self.func_man.map_stats_like_overlay()
##         s_arr = stats_vol.image_arr.transpose().ravel()
##         assert np.size(s_arr) == np.size(self.func_man.overlay.image_arr), \
##                'The size of the threshold mask does not match the size ' \
##                'of the overlay array'

##         pt_arr = self.func_scalars.image_data.point_data.get_array('threshold')
##         if pt_arr:
##             pt_arr.from_array(s_arr)
##         else:
##             n = self.func_scalars.image_data.point_data.add_array(s_arr)
##             self.func_scalars.image_data.point_data.get_array(n).name = 'threshold'
##             self.thresh = mlab.pipeline.threshold(
##                 mlab.pipeline.set_active_attribute(
##                     self.func_scalars, point_scalars='threshold'
##                     ),
##                 figure=self.display.scene.mayavi_scene                
##                 )
##             self._update_thresh_lims()

##     def _map_threshold_mask(self):
##         mask_vol = self.func_man.map_stats_like_overlay(map_mask=True,
##                                                         mask_type='positive')
##         m_arr = mask_vol.image_arr.transpose().ravel()
##         assert np.size(m_arr) == np.size(self.func_man.overlay.image_arr), \
##                'The size of the threshold mask does not match the size ' \
##                'of the overlay array'

##         pt_arr = self.func_scalars.image_data.point_data.get_array('threshold')
##         if pt_arr:
##             pt_arr.from_array(m_arr)
##         else:
##             n = self.func_scalars.image_data.point_data.add_array(m_arr)
##             self.func_scalars.image_data.point_data.get_array(n).name = 'threshold'
##             self.thresh = mlab.pipeline.threshold(
##                 mlab.pipeline.set_active_attribute(
##                     self.func_scalars, point_scalars='threshold'
##                     ),
##                 low=0.5,
##                 figure=self.display.scene.mayavi_scene
##                 )
            
##     @on_trait_change('_func_thresh.thresh_limits')
##     def _update_thresh_lims(self):
##         if not self.thresh:
##             return
##         if self.notch_mode:
##             # need to remap the whole array
##             self._map_threshold_mask()
##         lims = self._func_thresh.thresh_limits
##         try:
##             self.thresh.lower_threshold = lims[0]
##         except TraitError:
##             pass
##         try:
##             self.thresh.upper_threshold = lims[1]
##         except TraitError:
##             pass
