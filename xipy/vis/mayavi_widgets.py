import os
os.environ['ETS_TOOLKIT'] = 'qt4'

from PyQt4 import QtGui, QtCore

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

import numpy as np
from scipy import ndimage
## from matplotlib import cm
import xipy.vis.color_mapping as cm

from xipy.slicing.image_slicers import ResampledVolumeSlicer, \
     VolumeSlicerInterface
from xipy.overlay import OverlayInterface, ThresholdMap
from xipy.vis.qt4_widgets.auxiliary_window import TopLevelAuxiliaryWindow
from xipy.vis.mayavi_tools import ArraySourceRGBA, image_plane_widget_rgba
from xipy.vis.mayavi_tools import time_wrap as tw
from xipy.vis import rgba_blending
import xipy.volume_utils as vu
from nipy.core import api as ni_api

import time

import cProfile, pstats

P_THRESH = 0.05
CLUSTER_THRESH = 3

def three_plane_pt(n1, n2, n3, x1, x2, x3):
    nm = np.array((n1,n2,n3)).T
    dt = np.linalg.det(nm)
    n2x3 = np.cross(n2,n3)
    n3x1 = np.cross(n3,n1)
    n1x2 = np.cross(n1,n2)
    x = ( np.dot(x1,n1)*n2x3 + np.dot(x2,n2)*n3x1 + np.dot(x3,n3)*n1x2 )
    return x / dt
                                                       
class VisualComponent(HasTraits):
    # NOTE! IT IS VERY IMPORTANT TO SET THIS UP AS A CONCRETE
    # INSTANCE BEFORE CREATING ANY SUBCLASSES
##     display = Instance('xipy.vis.mayavi_widgets.OrthoView3D')
    pass
    
class OverlayBlendingComponent(VisualComponent):
    """A class to take control of blending overlay colors into the main
    display's blended image source
    """
    
    blender = DelegatesTo('display')
    func_man = DelegatesTo('display')

    # ----- This will eventually become this VisualComponent's UI widget -----
    show_func = DelegatesTo('display')
    alpha_compress = DelegatesTo('display')
    # ------------------------------------------------------------------------
    @on_trait_change('show_func')
    def _show_func(self):
        if not self.func_man or not self.func_man.overlay:
            print 'no functional manager to provide an overlay'
            self.trait_setq(show_func=False)
            return
        self.display.change_source_data()
    
    @on_trait_change('alpha_compress')
    def _alpha_scale(self):
        if not self.func_man:
            return
        self.blender.over_alpha = self.func_man.alpha(scale=self.alpha_compress)

    @on_trait_change('func_man.norm')
    def _set_blender_norm(self):
        print 'resetting scalar normalization from func_man.norm'
        self.blender.over_norm = self.func_man.norm
        
    @on_trait_change('func_man.threshold')
    def _set_threshold(self):
        print 'remapping alpha because of func_man.threshold',
        if not self.func_man.overlay:
            print 'but no func_man'
            return
        print ''
        self.blender.over_alpha = self.func_man.alpha(scale=self.alpha_compress)

    @on_trait_change('func_man.cmap_option')
    def _set_over_cmap(self):
        self.blender.over_cmap = self.func_man.colormap

    @on_trait_change('func_man.overlay_updated')
    def _update_colors_from_func_man(self):
        """ When a new overlay is signaled, update the overlay color bytes
        """
        if not self.func_man or not self.func_man.overlay:
            return
        overlay = self.func_man.overlay
        # this could potentially change scalar mapping properties too
        self.blender.trait_setq(
            over_cmap=self.func_man.colormap,
            over_norm=self.func_man.norm,
            over_alpha=self.func_man.alpha(scale=self.alpha_compress)
            )
            
        self.blender.over = overlay.raw_image

class OverlayThresholdingSurfaceComponent(VisualComponent):
    """A class to take control of thresholding the overlay, and creating
    surfaces of unmasked regions
    """

    func_man = DelegatesTo('display')
    _func_thresh = DelegatesTo('display')
    func_scalars = DelegatesTo('display')
    notch_mode = Property
    thresh = Instance('enthought.mayavi.filters.threshold.Threshold')

    # ----- This will eventually become this VisualComponent's UI widget -----
    show_tsurfs = DelegatesTo('display')
    # ------------------------------------------------------------------------

    @on_trait_change('show_tsurfs')
    def _show_thresh_surfaces(self):
        if not self.thresh:
            print 'no overlay thresholding available'
            self.trait_setq(show_tsurfs=False)
        elif not hasattr(self, 'thresh_surf') or not self.thresh_surf:
            self.add_threshold_surf()
        self.thresh_surf.visible = self.show_tsurfs
        self.display.scene.render_window.render()
        
    def add_threshold_surf(self):
        if not self.thresh:
            return
        mn, mx = self.func_man.norm
        colormap = self.func_man.cmap_option
        self.display._stop_scene()
        surf = mlab.pipeline.surface(
            self.surf_scalars, vmin=mn, vmax=mx,
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

    @on_trait_change('func_man.overlay')
    def _update_overlay_scalars(self):
        overlay = self.func_man.overlay
        scalars = self.func_scalars
        if not overlay:
            return

        s_arr = np.ma.filled(overlay.image_arr)
        sctype = s_arr.dtype
        # this will be a negative mask, so threshold everything > 0.5
        t_arr = np.ma.getmask(overlay.image_arr)
        if t_arr is False:
            t_arr = np.ones(overlay.image_arr.shape, sctype).transpose().flatten()
        else:
            t_arr = t_arr.astype(sctype).transpose().flatten()

        scalars.scalar_data = s_arr.copy()
##         scalars.image_data.point_data.get_array(0).name = 'overlay'
        scalars.scalar_name = 'overlay'
        scalars.origin = overlay.coordmap.affine[:3,-1]
        scalars.spacing = vu.voxel_size(overlay.coordmap.affine)
        self.display._stop_scene()
        scalars.update_image_data = True
        thresh_pts = scalars.image_data.point_data.get_array('threshold')
        if thresh_pts:
            thresh_pts.from_array(t_arr)
        else:
            n = scalars.image_data.point_data.add_array(t_arr)
            scalars.image_data.point_data.get_array(n).name = 'threshold'
            ts = mlab.pipeline.set_active_attribute(scalars,
                                                    point_scalars='threshold')
            self.thresh = mlab.pipeline.threshold(
                ts, low=0.0, up=0.5
                )
            self.thresh.auto_reset_lower = False
            self.thresh.auto_reset_upper = False
            self.thresh.filter_type = 'cells'
            self.surf_scalars = mlab.pipeline.set_active_attribute(
                self.thresh, point_scalars='overlay'
                )
            
        self.display._start_scene()


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
    
class CorticalSurfaceComponent(VisualComponent):

    anat_scalars = DelegatesTo('display')
    anat_image = DelegatesTo('display')
    poly_extractor = DelegatesTo('display')

    # ----- This will eventually become this VisualComponent's UI widget -----
    show_cortex = DelegatesTo('display')
    # ------------------------------------------------------------------------

    @on_trait_change('show_cortex')
    def _show_cortex(self):
        if not self.anat_image:
            self.trait_setq(show_cortex=False)
            return
        elif not hasattr(self, 'cortical_surf') or not self.cortical_surf:
            self.add_cortical_surf()
        self.cortical_surf.visible = self.show_cortex
    
    def add_cortical_surf(self):
        # lifted from Gael Varoquax
        self.display._stop_scene()
        # this seems to
        # 1) smooth a thresholding mask (>4800)
        # 2) fill the holes of wherever this smoothed field is > 0.5
        # 3) smooth the final, filled mask

        # XYZ: TRY TO USE BRAIN MASK EXTRACTION HERE INSTEAD OF THRESHOLD

        # OK.. WANT TO HAVE EXACTLY THE SAME SUPPORT FOR THE SURFACE IMAGE
        # AND THE IMAGE PLANE IMAGES..

##         print 'getting brain mask'
##         img_arr = self.anat_image.image_arr.filled(fill_value=0)
## ##         anat_support = vu.auto_brain_mask(img_arr)
## ##         print 'getting smoothed mask'
## ##         anat_blurred = ndimage.gaussian_filter(anat_support.astype('d'), 6)
## ##         anat_blurred = ( (anat_blurred > .5) | anat_support ).astype('d')
##         anat_blurred = ndimage.gaussian_filter(img_arr, 6)
        
        anat_blurred = ndimage.gaussian_filter(
            (ndimage.morphology.binary_fill_holes(
                ndimage.gaussian_filter(
                    (self.anat_image.image_arr > 4800).astype(np.float), 6)
                > 0.5
                )).astype(np.float),
            2).T.ravel()
        n = self.anat_scalars.image_data.point_data.add_array(
            anat_blurred.T.ravel()
            )
        self.anat_scalars.image_data.point_data.get_array(n).name = 'blurred'
        surf_name = self.anat_scalars.image_data.point_data.get_array(n-1).name
        self.anat_scalars.image_data.point_data.update()
        anat_blurred = mlab.pipeline.set_active_attribute(
            self.anat_scalars, point_scalars='blurred'
            )
        anat_blurred.update_pipeline()
        self._contour = mlab.pipeline.contour(anat_blurred)
        
        csurf = mlab.pipeline.set_active_attribute(
            mlab.pipeline.user_defined(self._contour,
                                       filter=self.poly_extractor),
            point_scalars='scalar'
            )
        self.cortical_surf = mlab.pipeline.surface(
            csurf,
            colormap='copper',
            opacity=1,
            #vmin=4800, vmax=5000)
            vmin=10, vmax=7230,
            figure=self.display.scene.mayavi_scene
            )
        self.cortical_surf.enable_contours = True
        self.cortical_surf.contour.filled_contours = True
        self.cortical_surf.contour.auto_contours = True
##         self.cortical_surf.contour.contours = [5000, 7227.8]
        self.cortical_surf.actor.property.backface_culling = True
        self.cortical_surf.actor.mapper.interpolate_scalars_before_mapping = True
        self.cortical_surf.actor.property.interpolation = 'flat'
        
##         # Add opacity variation to the colormap
##         cmap = self.cortical_surf.module_manager.scalar_lut_manager.lut.table.to_array()
##         cmap[128:, -1] = 0.7*255
##         cmap[:128, -1] = 0.9*255
##         self.cortical_surf.module_manager.scalar_lut_manager.lut.table = cmap
        self.display._start_scene()


class OrthoView3D(HasTraits):
    #---------------------------------------------------------------------------
    # Data and Figure
    #---------------------------------------------------------------------------
    scene = Instance(MlabSceneModel, ())

    anat_scalars = Instance(Source)
    func_scalars = Instance(Source)

    blender = Instance(rgba_blending.BlendedImage, (),
                       spline_order=0, transpose_inputs=True)
    blended_src = Instance(Source)
    blob_surf_src = Instance(Source)

    anat_image = Instance(VolumeSlicerInterface)

    #---------------------------------------------------------------------------
    # Functional Overlay Manager
    #---------------------------------------------------------------------------
    func_man = Instance(OverlayInterface, ())
    _func_thresh = Instance(ThresholdMap)

    #---------------------------------------------------------------------------
    # Scene Control Traits
    #---------------------------------------------------------------------------
    show_tsurfs = Bool(False)
    show_anat = Bool(False)
    show_func = Bool(False)
    show_cortex = Bool(False)
    alpha_compress = Range(low=0.0, high=4.0, value=1.0,
                           editor=RangeEditor(low=0.0, high=4.0,
                                              format='%1.2f', mode='slider'))


    #---------------------------------------------------------------------------
    # Other Traits
    #---------------------------------------------------------------------------
    #planes_function = Instance(tvtk.Planes, ())
    poly_extractor = Instance(tvtk.ExtractPolyDataGeometry, ())
    info = Instance(Text)
    
    _axis_index = dict(x=0, y=1, z=2)

    def __init__(self, parent=None, **traits):
        HasTraits.__init__(self, **traits)
        self.blender
        self.func_man
        # First, add self to the VisualComponent base class
        VisualComponent.add_class_trait('display', self)
        # -- In the future, I want to actually have the VisualComponent
        # classes "own" the traits, and have this class have a bunch
        # of DelegatesTo references to a list of them
        # .. can have each VisualComponent have an "exported_traits" list

        # -- In the future, each VisualComponent will have its own
        # GUI panel, which will be viewed in this window
        
        # set up components
        self.add_trait('overlay_image_helper',
                       OverlayBlendingComponent(display=self))
        self.add_trait('overlay_thresh_helper',
                       OverlayThresholdingSurfaceComponent(display=self))
        self.add_trait('cortical_surf_helper',
                       CorticalSurfaceComponent(display=self))
                        
        
        anat_alpha = np.ones(256)
        anat_alpha[:5] = 0
        self.blender.set(main_alpha=anat_alpha, trait_notify_change=False)
        self.__reposition_planes_after_interaction = False
        self.show_anat = True

    #---------------------------------------------------------------------------
    # Default values
    #---------------------------------------------------------------------------
    def _anat_scalars_default(self):
        s = ArraySource(transpose_input_array=False)
        return mlab.pipeline.add_dataset(s, figure=self.scene.mayavi_scene)
    def _func_scalars_default(self):
        s = ArraySource(transpose_input_array=True)
        return mlab.pipeline.add_dataset(s, figure=self.scene.mayavi_scene)
    def _blended_src_default(self):
        s = ArraySourceRGBA(transpose_input_array=False)
        return mlab.pipeline.add_dataset(s, figure=self.scene.mayavi_scene)
    def _blob_surf_src_default(self):
        return mlab.pipeline.scalar_field(np.zeros((2,2,2)),
                                          figure=self.scene.mayavi_scene)
    def _func_man_default(self):
        return OverlayInterface()
    def _info_default(self):
        info = mlab.text(.05,.05,'Welcome',width=0.4,
                         figure=self.scene.mayavi_scene)
        info.property.font_size = 10
        return info

    #---------------------------------------------------------------------------
    # Property Getters
    #---------------------------------------------------------------------------
    
    #---------------------------------------------------------------------------
    # Construct ImplicitPlaneWidget plots
    #---------------------------------------------------------------------------
    def make_ipw(self, axis_name):
        ipw = mlab.pipeline.image_plane_widget(
            self.blended_src,
            figure=self.scene.mayavi_scene,
            plane_orientation='%s_axes'%axis_name
            )
        ipw.use_lookup_table = False
        ipw.ipw.reslice_interpolate = 0 # hmmm?
        return ipw
    
    def add_plots_to_scene(self):
        self._stop_scene()
        for ax in ('x', 'y', 'z'):
            ipw = self.make_ipw(ax)

            # position the image plane widget at an unobtrusive cut
            dim = self._axis_index[ax]
            data_shape = self.blended_src.image_data.extent[::2]
            ipw.ipw.slice_position = data_shape[dim]/4
            # switch actions here
            ipw.ipw.middle_button_action = 2
            ipw.ipw.right_button_action = 0
            # name this widget and attribute it to self
            setattr(self, 'ipw_%s'%ax, ipw)
            ipw.ipw.add_observer(
                'StartInteractionEvent',
                getattr(self, '_%s_plane_interaction'%ax)
                )
            ipw.ipw.add_observer(
                'InteractionEvent',
                getattr(self, '_%s_plane_interaction'%ax)
                )
            ipw.ipw.add_observer(
                'EndInteractionEvent',
                self._handle_end_interaction
                )

        self._start_scene()

    def _ipw_x(self, axname):
        return getattr(self, 'ipw_%s'%axname, None)
    
    #---------------------------------------------------------------------------
    # Scene interaction callbacks 
    #---------------------------------------------------------------------------
    def _x_plane_interaction(self, widget, event):
        self._handle_plane_interaction('x', widget)
    def _y_plane_interaction(self, widget, event):
        self._handle_plane_interaction('y', widget)
    def _z_plane_interaction(self, widget, event):
        self._handle_plane_interaction('z', widget)
    def _handle_plane_interaction(self, ax, widget):
        if widget.GetCursorDataStatus():
            # In other words, if moving the crosshairs in a plane
            print 'listening for endinteraction'
            self.__reposition_planes_after_interaction = True
            return
        # otherwise, do normal interaction
        self._link_plane_points(ax, widget)

    def _link_plane_points(self, ax, widget):
        ax_idx = self._axis_index[ax]
        ipw = self._ipw_x(ax).ipw #getattr(self, 'ipw_%s'%ax).ipw
        planes_function = self.poly_extractor.implicit_function
        planes_function.points[ax_idx] = ipw.center
        planes_function.normals[ax_idx] = map(lambda x: -x, ipw.normal)

    def _handle_end_interaction(self, widget, event):
        if self.__reposition_planes_after_interaction:
            pos_ijk = widget.GetCurrentCursorPosition()
            pos = self.anat_image.coordmap(pos_ijk)[0]
            self._snap_to_position(pos)
            self.__reposition_planes_after_interaction = False
        else:
            pos = self._current_intersection()
        self._register_position(pos)

    def _current_intersection(self):
        nx = self.ipw_x.ipw.normal; cx = self.ipw_x.ipw.center
        ny = self.ipw_y.ipw.normal; cy = self.ipw_y.ipw.center
        nz = self.ipw_z.ipw.normal; cz = self.ipw_z.ipw.center
        return three_plane_pt(nx, ny, nz, cx, cy, cz)

    def _snap_to_position(self, pos):
        if self._ipw_x('x') is None:
            return
        print 'snapping to', pos
        self._stop_scene()
        anames = ('x', 'y', 'z')
        pd = dict(zip( anames, pos ))
        for ax in anames:
            ipw = self._ipw_x(ax).ipw
            ipw.plane_orientation='%s_axes'%ax
            ipw.slice_position = pd[ax]
        self._start_scene()

    def _register_position(self, pos):
        if self.func_man:
            self.func_man.world_position = pos
        if self.show_cortex:
            self.anat_scalars.update()        
    #---------------------------------------------------------------------------
    # Traits callbacks
    #---------------------------------------------------------------------------

    @on_trait_change('show_anat')
    def _show_anat(self):
        if not self.anat_image:
            print 'no anat array loaded yet'
            self.set(show_anat=False, trait_change_notify=False)
            return
        # this will assess the status of the image source and plotting
        self.change_source_data()

    @on_trait_change('func_man.world_position_updated')
    def _follow_functional_position(self):
        print 'heard that position updated'
        self._snap_to_position(self.func_man.world_position)
    
    @on_trait_change('func_man,func_man.overlay_updated')
    def _update_functional_info(self):
        """ Update misc. aspect of the display when a new overlay crops up
        """
        if not self.func_man or not self.func_man.overlay:
            return
        if self.func_man.description:
            self.info.text = self.func_man.description

        self._func_thresh = self.func_man.threshold
                
    @on_trait_change('anat_image')
    def _update_colors_from_anat_image(self):
        """ When a new image is loaded, update the anat color bytes
        """
        self.__blocking_draw = True
        self.blender.main = self.anat_image.raw_image
        # hmmm there is a problem here when re-loading images..
        # the appropriate rgba array may not be available or valid yet
        self.change_source_data(new_position=True)
        self.__blocking_draw = False

        # flush previous arrays
        n_arr = self.anat_scalars.image_data.point_data.number_of_arrays
        names = [self.anat_scalars.image_data.point_data.get_array(i).name
                 for i in xrange(n_arr)]
        for n in names:
            self.anat_scalars.image_data.point_data.remove_array(n)
        # add new array
        image_arr = self.anat_image.image_arr.transpose().copy()
        self.anat_scalars.scalar_data = image_arr
        self.anat_scalars.spacing = self.blender.img_spacing
        self.anat_scalars.origin = self.blender.img_origin
    
    #---------------------------------------------------------------------------
    # Scene update methods
    #---------------------------------------------------------------------------
    def _stop_scene(self):
        self._render_mode = self.scene.disable_render
        self.scene.disable_render = True
    def _start_scene(self):
        mode = getattr(self, '_render_mode', True)
        self.scene.disable_render = mode
        if mode:
            self.scene.render_window.render()

    @on_trait_change('blender.main_rgba,blender.over_rgba,blender.blended_rgba')
    def _monitor_sources(self, obj, name, new):
        if self.__blocking_draw:
            return
        print name, 'changed'
        if name == 'main_rgba' and self.show_anat:
            self.change_source_data()
        elif name == 'over_rgba' and self.show_func:
            self.change_source_data()
    
    def change_source_data(self, new_position=False):
        """ Create a pixel-blended array, whose contents depends on the
        current plotting conditions. Also check the status of the
        visibility of the plots.

        """
        # XYZ: SAY FOR NOW THAT THE ANAT IMAGE MUST BE AVAILABLE BEFORE
        # PLOTTING THE OVERLAY... THIS WILL SIDESTEP THE ISSUE OF UPDATING
        # THE IMAGE DATA PROPERTIES, WHICH APPEARS TO CAUSE A HUGE DELAY
        


        if self.show_func and self.show_anat:
            print 'will plot blended'
            img_data = self.blender.blended_rgba
        elif self.show_func:
            print 'will plot over plot'
            img_data = self.blender.over_rgba
        else:
            print 'will plot anatomical'
            img_data = self.blender.main_rgba

        # if a new position, update (even if invisibly)
        if new_position:
            # this will kick off the scalar_data_changed stuff
            self.blended_src.scalar_data = img_data.copy()
            self.blended_src.spacing = self.blender.img_spacing
            self.blended_src.origin = self.blender.img_origin
            self.blended_src.update_image_data = True #???
        
        if not self.show_func and not self.show_anat:
            self.toggle_planes_visible(False)
            return

        self._stop_scene()

        if not new_position:
            print 'changing data in-place'
            self.blended_src.scalar_data[:] = img_data

        #self.blended_src.update_image_data = True
        self.blended_src.update()
##         cProfile.runctx('self.blended_src.update()', globals(), locals(),
##                         'mayavi.prof')
##         s = pstats.Stats('mayavi.prof')
##         s.strip_dirs().sort_stats('cumulative').print_stats()

        if not hasattr(self, 'ipw_x'):
            t0 = time.time()
            print 'also adding plots to scene ',
            self.add_plots_to_scene()
            t = time.time()
            print 'done, %1.3f sec'%(t-t0)
        else:
            t0 = time.time()
            print 'also turning on plots ',
            self.toggle_planes_visible(True)
            t = time.time()
            print 'done, %1.3f sec'%(t-t0)

        t0 = time.time()
        print 'also starting scene ',
        self._start_scene()
##         # this seems hacky, but works
##         self.scene.render_window.render()
        t = time.time()
        print 'done, %1.3f sec'%(t-t0)

    def toggle_planes_visible(self, value):
        self._toggle_poly_extractor_mode(cut_mode=value)
        for ax in ('x', 'y', 'z'):
            ipw = self._ipw_x(ax) #getattr(self, 'ipw_%s'%ax, False)
            if ipw:
                ipw.visible = value

    def _toggle_poly_extractor_mode(self, cut_mode=True):
        if not self.poly_extractor.implicit_function:
            self.poly_extractor.implicit_function = tvtk.Planes()
        pfunc = self.poly_extractor.implicit_function
        if not hasattr(self, 'ipw_x'):
            cut_mode = False
        if cut_mode:
            pts = [ [] ] * 3
            normals = [ [] ] * 3
            for ax, index in self._axis_index.iteritems():
                ipw = self._ipw_x(ax).ipw #getattr(self, 'ipw_%s'%ax).ipw
                pts[index] = ipw.center
                normals[index] = map(lambda x: -x, ipw.normal)
            pfunc.points = pts
            pfunc.normals = normals
        else:
            # set up the poly extractor filter with an all-inclusive
            # Implicit Function
            scene_points = [0]*24
            try:
                self.scene.camera.get_frustum_planes(1, scene_points)
                pfunc.set_frustum_planes(scene_points)
            except:
                # guess scene isn't set up yet
                pass
        
        self.poly_extractor.extract_inside = not cut_mode
            
            
    #---------------------------------------------------------------------------
    # Scene activation callbacks 
    #---------------------------------------------------------------------------
    @on_trait_change('scene.activated')
    def display_scene3d(self):
        print 'making 3d scene'
        self.scene.mlab.view(100, 100)
        self.scene.scene.background = (0, 0, 0)
        # set up the poly extractor filter with an all-inclusive
        # Implicit Function
        self._toggle_poly_extractor_mode(cut_mode=False)
        # Keep the view always pointing up
        self.scene.scene.interactor.interactor_style = \
                                 tvtk.InteractorStyleTerrain()
        self.scene.picker.pointpicker.add_observer('EndPickEvent',
                                                   self.pick_callback)
        self.scene.picker.show_gui = False


    def pick_callback(self, picker_obj, evt):
        p = tvtk.to_tvtk(picker_obj)
        x, y, z = p.pick_position
        self._snap_to_position((x,y,z))
        self._register_position((x,y,z))
        
    #---------------------------------------------------------------------------
    # The layout of the dialog created
    #---------------------------------------------------------------------------
    view = View(
        VGroup(
            HGroup(
                Item('scene',
                     editor=SceneEditor(scene_class=MayaviScene),
                     height=450, width=300),
                show_labels=False,
                dock='vertical'
                ),
            HGroup(
                Item('show_anat', label='Show anatomical'),
                Item('show_func', label='Show functional'),
                Item('show_tsurfs', label='Show unmasked surfaces'),
                Item('show_cortex', label='Show cortex'),
                Item('alpha_compress',style='custom',label='Alpha compression')
                ),
##             HGroup(
##                 Item('pscore_map', label='P Score Map'),
##                 Item('show_psurfs', label='Show Significance Blobs'),
##                 Item('cluster_threshold', label='Min Cluster Size'),
##                 Item('sig_threshold', label='Significance Level',
##                      style='custom')
##                 )
            ),
        resizable=True,
        title='XIPY 3D Viewer Controls',
        )


class MayaviWidget(TopLevelAuxiliaryWindow):

    def __init__(self, parent=None, main_ref=None, functional_manager=None,
                 **traits):
        TopLevelAuxiliaryWindow.__init__(self,
                                         parent=parent,
                                         main_ref=main_ref)
        layout = QtGui.QVBoxLayout(self)
        layout.setMargin(0)
        layout.setSpacing(0)
        if functional_manager:
            traits['func_man'] = functional_manager
        self.mr_vis = OrthoView3D(**traits)
        layout.addWidget(self.mr_vis.edit_traits(parent=self,
                                                 kind='subpanel').control)
        self.func_widget = None
        self.layout_box = layout
        if functional_manager is not None:
            self.add_toolbar(functional_manager)
        self.setObjectName('3D Plot')

    def add_toolbar(self, functional_manager):
        self.mr_vis.func_man = functional_manager
##         if self.func_widget is not None:
##             print 'removing old func widget'
##             self.layout_box.removeWidget(self.func_widget)
##             self.func_widget.close()
##         else:
##             print 'not removing old func widget'
##         self.func_widget = functional_manager.make_panel(parent=self)
##         self.layout_box.addWidget(self.func_widget)
##         self.update()

if __name__=='__main__':
    from PyQt4 import QtCore, QtGui
    import sys
    if QtGui.QApplication.startingUp():
        app = QtGui.QApplication(sys.argv)
    else:
        app = QtGui.QApplication.instance() 

    win = MayaviWidget()
    
    win.show()
    app.exec_()
                        
    

    
